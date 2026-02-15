use std::fmt;

use uniswap_v3_math::tick_math::get_sqrt_ratio_at_tick;

use crate::pools::{
    FEE_PIPS, Slot0Result, normalize_market_name, sqrt_price_x96_to_price_outcome, u256_to_f64,
};

pub(super) const NEWTON_ITERS: usize = 8;
pub(super) const FEE_FACTOR: f64 = 1.0 - (FEE_PIPS as f64 / 1_000_000.0);
pub(super) const EPS: f64 = 1e-12;
pub(super) const DUST: f64 = 1e-18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SimBuildError {
    MissingPrediction { market_name: &'static str },
}

impl fmt::Display for SimBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPrediction { market_name } => write!(
                f,
                "missing prediction for market '{}'; all tradeable outcomes must have predictions",
                market_name
            ),
        }
    }
}

impl std::error::Error for SimBuildError {}

/// Mutable pool state for swap simulation during rebalancing.
/// All arithmetic is pure f64 — no U256 in the hot path.
#[derive(Clone)]
pub(super) struct PoolSim {
    pub(super) market_name: &'static str,
    pub(super) market_id: &'static str,
    #[allow(dead_code)]
    pub(super) outcome_token: &'static str,
    #[allow(dead_code)]
    pub(super) pool: &'static crate::markets::Pool,
    price: f64,                       // current outcome price (mutable via set_price)
    pub(super) buy_limit_price: f64,  // max outcome price reachable via buying (tick boundary)
    pub(super) sell_limit_price: f64, // min outcome price reachable via selling (tick boundary)
    pub(super) liquidity: u128,
    pub(super) l_eff: f64, // precomputed L / (1e18 × (1-fee))
    pub(super) prediction: f64,
}

impl PoolSim {
    pub(super) fn from_slot0(
        slot0: &Slot0Result,
        market: &'static crate::markets::MarketData,
        prediction: f64,
    ) -> Option<Self> {
        let pool = market.pool.as_ref()?;
        let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
        let zero_for_one_buy = pool.token0.eq_ignore_ascii_case(market.quote_token);
        let liquidity: u128 = pool.liquidity.parse().ok()?;
        if liquidity == 0 {
            return None;
        }
        let t0 = pool.ticks.first()?.tick_idx;
        let t1 = pool.ticks.get(1)?.tick_idx;
        let sqrt_lo = get_sqrt_ratio_at_tick(t0.min(t1)).ok()?;
        let sqrt_hi = get_sqrt_ratio_at_tick(t0.max(t1)).ok()?;

        let (sqrt_buy_limit, sqrt_sell_limit) = if zero_for_one_buy {
            (sqrt_lo, sqrt_hi)
        } else {
            (sqrt_hi, sqrt_lo)
        };

        // Convert all U256 prices to f64 once at construction
        let price = u256_to_f64(sqrt_price_x96_to_price_outcome(
            slot0.sqrt_price_x96,
            is_token1_outcome,
        )?);
        let buy_limit_price = u256_to_f64(sqrt_price_x96_to_price_outcome(
            sqrt_buy_limit,
            is_token1_outcome,
        )?);
        let sell_limit_price = u256_to_f64(sqrt_price_x96_to_price_outcome(
            sqrt_sell_limit,
            is_token1_outcome,
        )?);

        Some(PoolSim {
            market_name: market.name,
            market_id: market.market_id,
            outcome_token: market.outcome_token,
            pool,
            price,
            buy_limit_price,
            sell_limit_price,
            liquidity,
            l_eff: (liquidity as f64) / (1e18 * FEE_FACTOR),
            prediction,
        })
    }

    /// Current outcome price.
    pub(super) fn price(&self) -> f64 {
        self.price
    }

    /// Set price with tick-boundary assertion (fires in debug builds only).
    pub(super) fn set_price(&mut self, new_price: f64) {
        let tol = 1e-9 * (1.0 + new_price.abs().max(self.buy_limit_price.abs()));
        debug_assert!(
            new_price >= self.sell_limit_price - tol && new_price <= self.buy_limit_price + tol,
            "price {} outside tick bounds [{}, {}] for {}",
            new_price,
            self.sell_limit_price,
            self.buy_limit_price,
            self.market_name
        );
        self.price = new_price;
    }

    /// Max tokens sellable before hitting the tick boundary.
    pub(super) fn max_sell_tokens(&self) -> f64 {
        let k = self.kappa();
        if k <= 0.0 {
            return 0.0;
        }
        let lp = self.sell_limit_price;
        if lp <= 0.0 || lp >= self.price {
            return 0.0;
        }
        ((self.price / lp).sqrt() - 1.0) / k
    }

    /// Price sensitivity for selling: κ = (1-fee) × √price × 1e18 / L.
    pub(super) fn kappa(&self) -> f64 {
        if self.price <= 0.0 || self.liquidity == 0 {
            return 0.0;
        }
        FEE_FACTOR * self.price.sqrt() * 1e18 / (self.liquidity as f64)
    }

    /// Price sensitivity for buying: λ = √price × 1e18 / L.
    pub(super) fn lambda(&self) -> f64 {
        if self.price <= 0.0 || self.liquidity == 0 {
            return 0.0;
        }
        self.price.sqrt() * 1e18 / (self.liquidity as f64)
    }

    /// Max tokens buyable before hitting the tick boundary.
    pub(super) fn max_buy_tokens(&self) -> f64 {
        let lam = self.lambda();
        if lam <= 0.0 {
            return 0.0;
        }
        let blp = self.buy_limit_price;
        if blp <= self.price {
            return 0.0;
        }
        (1.0 - (self.price / blp).sqrt()) / lam
    }

    /// Buy exact amount of outcome tokens.
    /// Returns (tokens_actually_bought, quote_cost, new_price).
    /// Price after buying m: P(m) = P₀ / (1 - mλ)²
    /// Cost: m × P₀ / ((1-fee) × (1 - mλ))
    pub(super) fn buy_exact(&self, amount: f64) -> Option<(f64, f64, f64)> {
        if amount <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let lam = self.lambda();
        let max_buy = self.max_buy_tokens();
        let actual = amount.min(max_buy);
        if actual <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let d = 1.0 - actual * lam;
        if d <= 0.0 {
            return None;
        }
        let new_price = self.price / (d * d);
        let cost = actual * self.price / (FEE_FACTOR * d);
        Some((actual, cost, new_price))
    }

    /// Effective liquidity: L_eff = L / (1e18 × (1-fee)).
    pub(super) fn l_eff(&self) -> f64 {
        self.l_eff
    }

    /// Cost to move pool price to target_price via direct buy.
    /// Returns (quote_cost, outcome_received, new_price).
    pub(super) fn cost_to_price(&self, target_price: f64) -> Option<(f64, f64, f64)> {
        let clamped = target_price.min(self.buy_limit_price);
        if clamped <= self.price {
            return Some((0.0, 0.0, self.price));
        }
        let cost = self.l_eff * (clamped.sqrt() - self.price.sqrt());
        let l_raw = self.liquidity as f64 / 1e18;
        let amount = l_raw * (1.0 / self.price.sqrt() - 1.0 / clamped.sqrt());
        Some((cost, amount, clamped))
    }

    /// Sell outcome tokens until price drops to target.
    /// Returns (tokens_sold, quote_received, new_price).
    pub(super) fn sell_to_price(&self, target_price: f64) -> Option<(f64, f64, f64)> {
        let clamped = target_price.max(self.sell_limit_price);
        if clamped >= self.price {
            return Some((0.0, 0.0, self.price));
        }
        let k = self.kappa();
        if k <= 0.0 {
            return None;
        }
        let tokens = ((self.price / clamped).sqrt() - 1.0) / k;
        let d = 1.0 + tokens * k;
        let proceeds = self.price * tokens * FEE_FACTOR / d;
        Some((tokens, proceeds, clamped))
    }

    /// Sell exact amount of outcome tokens.
    /// Returns (tokens_actually_sold, quote_received, new_price).
    pub(super) fn sell_exact(&self, amount: f64) -> Option<(f64, f64, f64)> {
        if amount <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let k = self.kappa();
        let max_sell = self.max_sell_tokens();
        let actual = amount.min(max_sell);
        if actual <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let d = 1.0 + actual * k;
        let new_price = self.price / (d * d);
        let proceeds = self.price * actual * FEE_FACTOR / d;
        Some((actual, proceeds, new_price))
    }
}

/// Build PoolSim entries from slot0 results, matching with predictions.
/// Returns an error if any outcome has no matching prediction.
pub(super) fn build_sims(
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &std::collections::HashMap<String, f64>,
) -> Result<Vec<PoolSim>, SimBuildError> {
    let preds = predictions;
    let mut sims = Vec::with_capacity(slot0_results.len());
    let mut dropped_markets: Vec<&'static str> = Vec::new();

    for (slot0, market) in slot0_results {
        let key = normalize_market_name(market.name);
        let prediction = preds
            .get(&key)
            .copied()
            .ok_or(SimBuildError::MissingPrediction {
                market_name: market.name,
            })?;

        match PoolSim::from_slot0(slot0, market, prediction) {
            Some(sim) => sims.push(sim),
            None => dropped_markets.push(market.name),
        }
    }

    if !dropped_markets.is_empty() {
        let preview = dropped_markets
            .iter()
            .take(5)
            .copied()
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = if dropped_markets.len() > 5 {
            ", ..."
        } else {
            ""
        };
        tracing::info!(
            dropped_count = dropped_markets.len(),
            markets = %format!("{}{}", preview, suffix),
            "dropped outcomes from simulation due to invalid pool state"
        );
    }

    Ok(sims)
}

/// Compute alt price for outcome at `idx`: 1 - sum(other outcome prices).
/// O(1) when price_sum is precomputed.
pub(super) fn alt_price(sims: &[PoolSim], idx: usize, price_sum: f64) -> f64 {
    1.0 - (price_sum - sims[idx].price)
}

/// Profitability = (prediction - best_price) / best_price
pub(super) fn profitability(prediction: f64, best_price: f64) -> f64 {
    let effective_price = best_price.max(EPS);
    (prediction - effective_price) / effective_price
}

pub(super) fn sanitize_nonnegative_finite(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Target price for a given target profitability: prediction / (1 + target_prof)
pub(super) fn target_price_for_prof(prediction: f64, target_prof: f64) -> f64 {
    prediction / (1.0 + target_prof)
}

/// Which acquisition route is cheaper for an outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Route {
    Direct,
    Mint,
}
