use std::collections::HashMap;

use alloy_primitives::{I256, U256};
use uniswap_v3_math::{swap_math::compute_swap_step, tick_math::get_sqrt_ratio_at_tick};

use crate::markets::{MarketData, Pool};
use crate::predictions::PREDICTIONS_L1;

use super::pricing::{
    normalize_market_name, prediction_to_sqrt_price_x96, sqrt_price_x96_to_price_outcome,
    u256_to_f64,
};
use super::rpc::Slot0Result;
use super::swap::FEE_PIPS;

/// Liquidity depth at tick boundary and breakeven point.
#[derive(Debug, Clone, Default)]
pub struct DepthResult {
    pub outcome_at_tick: f64,
    pub cost_at_tick: f64,
    pub outcome_at_breakeven: f64,
    pub cost_at_breakeven: f64,
}

/// Computes liquidity depth: max outcome tokens and costs at tick boundary and breakeven.
fn compute_depth(
    pool: &Pool,
    sqrt_price_x96: U256,
    quote_token: &str,
    is_token1_outcome: bool,
    prediction: f64,
) -> DepthResult {
    let zero_for_one = pool.token0.eq_ignore_ascii_case(quote_token);
    let liquidity: u128 = match pool.liquidity.parse() {
        Ok(v) if v > 0 => v,
        _ => return DepthResult::default(),
    };

    let (tick_0, tick_1) = match (pool.ticks.first(), pool.ticks.get(1)) {
        (Some(t0), Some(t1)) => (t0, t1),
        _ => return DepthResult::default(),
    };
    let target_tick = if zero_for_one {
        tick_0.tick_idx.min(tick_1.tick_idx)
    } else {
        tick_0.tick_idx.max(tick_1.tick_idx)
    };
    let sqrt_price_tick = match get_sqrt_ratio_at_tick(target_tick) {
        Ok(v) => v,
        Err(_) => return DepthResult::default(),
    };

    // I256::MAX as exact-input: compute_swap_step caps at the target price
    let large_amount = I256::MAX;

    // Max at tick boundary
    let (tick_out, tick_cost) = match compute_swap_step(
        sqrt_price_x96,
        sqrt_price_tick,
        liquidity,
        large_amount,
        FEE_PIPS,
    ) {
        Ok((_, amount_in, amount_out, fee_amount)) => {
            (u256_to_f64(amount_out), u256_to_f64(amount_in + fee_amount))
        }
        Err(_) => (0.0, 0.0),
    };

    // Breakeven: buy until price reaches prediction
    let (breakeven_out, breakeven_cost) =
        match prediction_to_sqrt_price_x96(prediction, is_token1_outcome) {
            Some(sqrt_price_breakeven) => {
                // Clamp to tick boundary (don't go past available liquidity)
                let target = if zero_for_one {
                    sqrt_price_breakeven.max(sqrt_price_tick)
                } else {
                    sqrt_price_breakeven.min(sqrt_price_tick)
                };

                // Verify target is in the right direction from current price
                let valid = if zero_for_one {
                    target <= sqrt_price_x96
                } else {
                    target >= sqrt_price_x96
                };

                if !valid {
                    (0.0, 0.0)
                } else {
                    match compute_swap_step(
                        sqrt_price_x96,
                        target,
                        liquidity,
                        large_amount,
                        FEE_PIPS,
                    ) {
                        Ok((_, amount_in, amount_out, fee_amount)) => {
                            (u256_to_f64(amount_out), u256_to_f64(amount_in + fee_amount))
                        }
                        Err(_) => (0.0, 0.0),
                    }
                }
            }
            None => (0.0, 0.0),
        };

    DepthResult {
        outcome_at_tick: tick_out,
        cost_at_tick: tick_cost,
        outcome_at_breakeven: breakeven_out,
        cost_at_breakeven: breakeven_cost,
    }
}

/// Entry in the profitability result: how much the prediction exceeds the market price.
#[derive(Debug, Clone)]
pub struct ProfitabilityEntry {
    pub market_name: &'static str,
    pub prediction: f64,
    pub market_price: f64,
    pub diff: f64,
    pub has_liquidity: bool,
    pub depth: DepthResult,
}

/// Returns the difference between predictions (PREDICTIONS_L1) and current
/// outcome prices for each matched market. Matches by market name (case-insensitive).
///
/// `slot0_results` should come from `fetch_all_slot0`.
pub fn profitability_simple(
    slot0_results: &[(Slot0Result, &MarketData)],
) -> Vec<ProfitabilityEntry> {
    let mut entries = Vec::new();

    // Build slot0 lookup by normalized market name
    let slot0_by_name: HashMap<String, &(Slot0Result, &MarketData)> = slot0_results
        .iter()
        .map(|pair| (normalize_market_name(pair.1.name), pair))
        .collect();

    for prediction in PREDICTIONS_L1.iter() {
        let pred_name = normalize_market_name(prediction.market);

        let Some(&(slot0, market)) = slot0_by_name.get(&pred_name) else {
            continue;
        };

        let pool = match market.pool.as_ref() {
            Some(p) => p,
            None => continue,
        };

        let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);

        let price = match sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome) {
            Some(p) => p,
            None => continue,
        };

        let price_f64 = u256_to_f64(price);
        let diff = (prediction.prediction - price_f64) / price_f64;

        let has_liquidity = pool
            .liquidity
            .parse::<u128>()
            .is_ok_and(|liquidity| liquidity > 0);

        let depth = if has_liquidity && diff > 0.0 {
            compute_depth(
                pool,
                slot0.sqrt_price_x96,
                market.quote_token,
                is_token1_outcome,
                prediction.prediction,
            )
        } else {
            DepthResult::default()
        };

        entries.push(ProfitabilityEntry {
            market_name: market.name,
            prediction: prediction.prediction,
            market_price: price_f64,
            diff,
            has_liquidity,
            depth,
        });
    }

    entries.sort_by(|a, b| b.diff.total_cmp(&a.diff));
    entries
}

/// Calculates the implied long price for an outcome token by summing the prices
/// of all other outcome tokens. This represents the cost of selling all tokens
/// except the target, effectively going long on it.
pub fn price_long_simple_alt(outcome_token: &str, prices: &[(f64, &str)]) -> f64 {
    let others_sum: f64 = prices
        .iter()
        .filter(|(_, token)| !token.eq_ignore_ascii_case(outcome_token))
        .map(|(price, _)| price)
        .sum();
    1.0 - others_sum
}
