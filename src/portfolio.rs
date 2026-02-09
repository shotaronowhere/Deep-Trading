use std::collections::HashMap;

use alloy_primitives::{I256, U256};
use uniswap_v3_math::{swap_math::compute_swap_step, tick_math::get_sqrt_ratio_at_tick};

use crate::pools::{
    normalize_market_name, prediction_map, prediction_to_sqrt_price_x96,
    sqrt_price_x96_to_price_outcome, u256_to_f64, FEE_PIPS, Slot0Result,
};

const BINARY_SEARCH_ITERS: usize = 50;

/// A rebalancing action to execute.
#[derive(Debug, Clone)]
pub enum Action {
    /// Mint complete sets across both L1 contracts.
    /// Contract 1 mint produces "other repos" token used as collateral for contract 2 mint.
    Mint {
        contract_1: &'static str,
        contract_2: &'static str,
        amount: f64,
        target_market: &'static str,
    },
    /// Buy outcome tokens directly from the pool.
    Buy {
        market_name: &'static str,
        amount: f64,
        cost: f64,
    },
    /// Sell outcome tokens back to the pool for sUSD.
    Sell {
        market_name: &'static str,
        amount: f64,
        proceeds: f64,
    },
}

/// Mutable pool state for swap simulation during rebalancing.
struct PoolSim {
    market_name: &'static str,
    #[allow(dead_code)]
    market_id: &'static str,
    #[allow(dead_code)]
    outcome_token: &'static str,
    #[allow(dead_code)]
    pool: &'static crate::markets::Pool,
    sqrt_price_x96: U256,
    is_token1_outcome: bool,
    zero_for_one_buy: bool,
    liquidity: u128,
    sqrt_price_buy_limit: U256,
    sqrt_price_sell_limit: U256,
    prediction: f64,
}

impl PoolSim {
    fn from_slot0(
        slot0: &Slot0Result,
        market: &'static crate::markets::MarketData,
        prediction: f64,
    ) -> Option<Self> {
        let pool = market.pool.as_ref()?;
        let is_token1_outcome =
            pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
        let zero_for_one_buy =
            pool.token0.to_lowercase() == market.quote_token.to_lowercase();
        let liquidity: u128 = pool.liquidity.parse().ok()?;
        if liquidity == 0 {
            return None;
        }
        let t0 = pool.ticks.get(0)?.tick_idx;
        let t1 = pool.ticks.get(1)?.tick_idx;
        let sqrt_lo = get_sqrt_ratio_at_tick(t0.min(t1)).ok()?;
        let sqrt_hi = get_sqrt_ratio_at_tick(t0.max(t1)).ok()?;

        // Buy direction: zero_for_one → price decreases → target = lower
        //                !zero_for_one → price increases → target = upper
        // Sell is opposite direction
        let (sqrt_price_buy_limit, sqrt_price_sell_limit) = if zero_for_one_buy {
            (sqrt_lo, sqrt_hi)
        } else {
            (sqrt_hi, sqrt_lo)
        };

        Some(PoolSim {
            market_name: market.name,
            market_id: market.market_id,
            outcome_token: market.outcome_token,
            pool,
            sqrt_price_x96: slot0.sqrt_price_x96,
            is_token1_outcome,
            zero_for_one_buy,
            liquidity,
            sqrt_price_buy_limit,
            sqrt_price_sell_limit,
            prediction,
        })
    }

    /// Current outcome price as f64.
    fn price(&self) -> f64 {
        sqrt_price_x96_to_price_outcome(self.sqrt_price_x96, self.is_token1_outcome)
            .map(|p| u256_to_f64(p))
            .unwrap_or(0.0)
    }

    /// Price at a hypothetical sqrt_price.
    fn price_at(&self, sqrt_price: U256) -> f64 {
        sqrt_price_x96_to_price_outcome(sqrt_price, self.is_token1_outcome)
            .map(|p| u256_to_f64(p))
            .unwrap_or(0.0)
    }

    /// Cost to move pool price to target_price via direct buy.
    /// Returns (quote_cost, outcome_received, new_sqrt_price).
    fn cost_to_price(&self, target_price: f64) -> Option<(f64, f64, U256)> {
        let target_sqrt =
            prediction_to_sqrt_price_x96(target_price, self.is_token1_outcome)?;
        // Clamp to tick boundary in buy direction
        let clamped = if self.zero_for_one_buy {
            target_sqrt.max(self.sqrt_price_buy_limit)
        } else {
            target_sqrt.min(self.sqrt_price_buy_limit)
        };
        // Verify target is in the buy direction from current price
        let valid = if self.zero_for_one_buy {
            clamped <= self.sqrt_price_x96
        } else {
            clamped >= self.sqrt_price_x96
        };
        if !valid {
            return Some((0.0, 0.0, self.sqrt_price_x96));
        }

        let (sqrt_next, amount_in, amount_out, fee) = compute_swap_step(
            self.sqrt_price_x96,
            clamped,
            self.liquidity,
            I256::MAX,
            FEE_PIPS,
        )
        .ok()?;
        Some((
            u256_to_f64(amount_in + fee),
            u256_to_f64(amount_out),
            sqrt_next,
        ))
    }

    /// Sell outcome tokens until price drops to target.
    /// Returns (tokens_sold, quote_received, new_sqrt_price).
    fn sell_to_price(&self, target_price: f64) -> Option<(f64, f64, U256)> {
        let target_sqrt =
            prediction_to_sqrt_price_x96(target_price, self.is_token1_outcome)?;
        // Clamp to tick boundary in sell direction
        let zero_for_one_sell = !self.zero_for_one_buy;
        let clamped = if zero_for_one_sell {
            target_sqrt.max(self.sqrt_price_sell_limit)
        } else {
            target_sqrt.min(self.sqrt_price_sell_limit)
        };
        // Verify target is in the sell direction from current price
        let valid = if zero_for_one_sell {
            clamped <= self.sqrt_price_x96
        } else {
            clamped >= self.sqrt_price_x96
        };
        if !valid {
            return Some((0.0, 0.0, self.sqrt_price_x96));
        }

        let (sqrt_next, amount_in, amount_out, fee) = compute_swap_step(
            self.sqrt_price_x96,
            clamped,
            self.liquidity,
            I256::MAX,
            FEE_PIPS,
        )
        .ok()?;
        // amount_in = outcome tokens consumed, amount_out = quote received
        Some((
            u256_to_f64(amount_in + fee),
            u256_to_f64(amount_out),
            sqrt_next,
        ))
    }

    /// Sell exact amount of outcome tokens.
    /// Returns (tokens_actually_sold, quote_received, new_sqrt_price).
    fn sell_exact(&self, amount: f64) -> Option<(f64, f64, U256)> {
        if amount <= 0.0 {
            return Some((0.0, 0.0, self.sqrt_price_x96));
        }
        let sell_amount = I256::try_from(U256::from((amount * 1e18) as u128)).ok()?;
        let (sqrt_next, amount_in, amount_out, fee) = compute_swap_step(
            self.sqrt_price_x96,
            self.sqrt_price_sell_limit,
            self.liquidity,
            sell_amount,
            FEE_PIPS,
        )
        .ok()?;
        Some((
            u256_to_f64(amount_in + fee),
            u256_to_f64(amount_out),
            sqrt_next,
        ))
    }
}

/// Build PoolSim entries from slot0 results, matching with predictions.
fn build_sims(slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)]) -> Vec<PoolSim> {
    let preds = prediction_map();
    slot0_results
        .iter()
        .filter_map(|(slot0, market)| {
            let key = normalize_market_name(market.name);
            let &pred = preds.get(&key)?;
            PoolSim::from_slot0(slot0, market, pred)
        })
        .collect()
}

/// Compute alt price for outcome at `idx`: 1 - sum(other outcome prices).
fn alt_price(sims: &[PoolSim], idx: usize) -> f64 {
    let others_sum: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, s)| s.price())
        .sum();
    1.0 - others_sum
}

/// Compute alt price after hypothetically selling `amount` of each non-target outcome.
/// Returns the marginal alt cost at that mint quantity.
fn alt_price_after_mint(sims: &[PoolSim], target_idx: usize, mint_amount: f64, skip: &[usize]) -> f64 {
    let others_sum: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx)
        .map(|(i, sim)| {
            if mint_amount <= 0.0 || skip.contains(&i) {
                return sim.price();
            }
            // Price after selling mint_amount tokens from this pool
            match sim.sell_exact(mint_amount) {
                Some((_, _, new_sqrt)) => sim.price_at(new_sqrt),
                None => sim.price(),
            }
        })
        .sum();
    1.0 - others_sum
}

/// Profitability = (prediction - best_price) / best_price
fn profitability(prediction: f64, best_price: f64) -> f64 {
    if best_price <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (prediction - best_price) / best_price
}

/// Target price for a given target profitability: prediction / (1 + target_prof)
fn target_price_for_prof(prediction: f64, target_prof: f64) -> f64 {
    prediction / (1.0 + target_prof)
}

/// Which acquisition route is cheaper for an outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Route {
    Direct,
    Mint,
}

/// Determine best route and its price for an outcome.
/// `mint_available`: true when all tradeable outcomes have liquid pools in sims.
/// `missing_pred_sum`: sum of predictions for outcomes not in sims (basket bonus for partial mint).
fn best_route(sims: &[PoolSim], idx: usize, mint_available: bool, missing_pred_sum: f64) -> (Route, f64) {
    let direct = sims[idx].price();
    let alt = alt_price(sims, idx);
    if mint_available {
        if alt > 0.0 && alt < direct {
            return (Route::Mint, alt);
        }
    } else if missing_pred_sum > 0.0 {
        // Partial mint: unsellable tokens have expected value = their predictions.
        // Effective cost = alt_price - basket_bonus.
        let adjusted = alt - missing_pred_sum;
        if adjusted > 0.0 && adjusted < direct {
            return (Route::Mint, adjusted);
        }
    }
    (Route::Direct, direct)
}

/// For the direct route, compute cost to bring an outcome's profitability to `target_prof`.
/// Returns (cost, outcome_amount, new_sqrt_price).
fn direct_cost_to_prof(sim: &PoolSim, target_prof: f64) -> Option<(f64, f64, U256)> {
    let tp = target_price_for_prof(sim.prediction, target_prof);
    sim.cost_to_price(tp)
}

/// For the mint route, compute cost to bring an outcome's profitability to `target_prof`.
/// Binary searches for the mint amount where marginal alt cost = target price.
/// `missing_pred_sum`: expected value of unsellable outcomes (basket bonus per unit minted).
/// Returns (net_cost, mint_amount).
fn mint_cost_to_prof(
    sims: &[PoolSim],
    target_idx: usize,
    target_prof: f64,
    missing_pred_sum: f64,
    skip: &[usize],
) -> Option<(f64, f64)> {
    let tp = target_price_for_prof(sims[target_idx].prediction, target_prof);
    // Adjusted target: the alt price (before basket bonus) that corresponds to target_prof
    let tp_adjusted = tp + missing_pred_sum;
    let current_alt = alt_price(sims, target_idx);
    if current_alt >= tp_adjusted {
        return Some((0.0, 0.0));
    }

    // Binary search: find mint_amount where marginal alt cost = adjusted target price
    let mut lo = 0.0_f64;
    let mut hi = 1e24_f64;

    for _ in 0..BINARY_SEARCH_ITERS {
        let mid = (lo + hi) / 2.0;
        let mac = alt_price_after_mint(sims, target_idx, mid, skip);
        if mac < tp_adjusted {
            lo = mid; // alt price too low, need more minting
        } else {
            hi = mid; // alt price at/above target, need less minting
        }
    }

    let amount = lo;
    if amount < 1e-18 {
        return Some((0.0, 0.0));
    }

    // Net cost = mint_cost - sell_proceeds (actual cash outflow).
    // Basket bonus is already captured by tp_adjusted reducing the mint amount.
    let total_proceeds: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx && !skip.contains(i))
        .filter_map(|(_, sim)| sim.sell_exact(amount).map(|(_, proceeds, _)| proceeds))
        .sum();
    let net_cost = amount - total_proceeds;

    Some((net_cost, amount))
}

/// Compute cost to reach target_prof for a given outcome using the best route.
/// Returns (cost, Route, amount, Option<new_sqrt_for_direct>).
fn cost_to_prof(
    sims: &[PoolSim],
    idx: usize,
    target_prof: f64,
    mint_available: bool,
    missing_pred_sum: f64,
    skip: &[usize],
) -> Option<(f64, Route, f64, Option<U256>)> {
    let direct = direct_cost_to_prof(&sims[idx], target_prof);
    let mint = if mint_available || missing_pred_sum > 0.0 {
        mint_cost_to_prof(sims, idx, target_prof, missing_pred_sum, skip)
    } else {
        None
    };

    match (direct, mint) {
        (Some((dc, da, ds)), Some((mc, ma))) => {
            if mc <= dc {
                Some((mc.max(0.0), Route::Mint, ma, None))
            } else {
                Some((dc, Route::Direct, da, Some(ds)))
            }
        }
        (Some((dc, da, ds)), None) => Some((dc, Route::Direct, da, Some(ds))),
        (None, Some((mc, ma))) => Some((mc, Route::Mint, ma, None)),
        (None, None) => None,
    }
}

fn lookup_balance(balances: &HashMap<&str, f64>, market_name: &str) -> f64 {
    if let Some(&v) = balances.get(market_name) {
        return v;
    }
    let lower = market_name.to_lowercase();
    for (&k, &v) in balances {
        if k.to_lowercase() == lower {
            return v;
        }
    }
    0.0
}

/// Emit mint actions: mint on both contracts, sell all non-target outcomes.
/// Contract addresses are derived from the distinct market_ids in MARKETS_L1.
fn emit_mint_actions(
    sims: &mut [PoolSim],
    target_idx: usize,
    amount: f64,
    actions: &mut Vec<Action>,
    skip: &[usize],
) -> f64 {
    // Derive contract addresses from MARKETS_L1 (not sims, which may be partial).
    let mut contracts: Vec<&'static str> = crate::markets::MARKETS_L1
        .iter()
        .map(|m| m.market_id)
        .collect();
    contracts.sort();
    contracts.dedup();
    actions.push(Action::Mint {
        contract_1: contracts[0],
        contract_2: if contracts.len() > 1 { contracts[1] } else { contracts[0] },
        amount,
        target_market: sims[target_idx].market_name,
    });

    // Sell all other outcomes across both contracts, update pool states
    let mut total_proceeds = 0.0;
    for i in 0..sims.len() {
        if i == target_idx || skip.contains(&i) {
            continue;
        }
        if let Some((sold, proceeds, new_sqrt)) = sims[i].sell_exact(amount) {
            if sold > 0.0 {
                total_proceeds += proceeds;
                sims[i].sqrt_price_x96 = new_sqrt;
                actions.push(Action::Sell {
                    market_name: sims[i].market_name,
                    amount: sold,
                    proceeds,
                });
            }
        }
    }
    total_proceeds
}

/// Computes optimal rebalancing trades for L1 markets.
///
/// 1. Sell overpriced holdings (price > prediction) via swap simulation
/// 2. Waterfall allocation: deploy capital to highest profitability outcomes,
///    equalizing profitability progressively
/// 3. Post-allocation liquidation: sell held outcomes less profitable than
///    the last bought outcome, reallocate via waterfall
pub fn rebalance(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<Action> {
    let mut actions: Vec<Action> = Vec::new();
    let mut budget = susds_balance;
    let mut sims = build_sims(slot0_results);

    if sims.is_empty() {
        return actions;
    }

    // Mint route requires all tradeable outcomes to have liquid pools.
    // sims only contains outcomes with non-zero liquidity and matching predictions.
    let mint_available = sims.len() == slot0_results.len();

    // Sum of predictions for outcomes not in sims (basket bonus for partial mint).
    let missing_pred_sum: f64 = if mint_available {
        0.0
    } else {
        let sim_names: std::collections::HashSet<String> = sims
            .iter()
            .map(|s| normalize_market_name(s.market_name))
            .collect();
        let preds = prediction_map();
        preds
            .iter()
            .filter(|(name, _)| !sim_names.contains(name.as_str()))
            .map(|(_, &pred)| pred)
            .sum()
    };

    // Track holdings changes during simulation
    let mut sim_balances: HashMap<&str, f64> = HashMap::new();
    for sim in &sims {
        sim_balances.insert(sim.market_name, lookup_balance(balances, sim.market_name));
    }

    // ── Phase 1: Sell overpriced holdings ──
    for i in 0..sims.len() {
        let price = sims[i].price();
        if price <= sims[i].prediction {
            continue;
        }
        let held = *sim_balances.get(sims[i].market_name).unwrap_or(&0.0);
        if held <= 0.0 {
            continue;
        }

        // Sell until price = prediction, or sell all if holdings insufficient
        let (tokens_needed, proceeds_to_pred, sqrt_at_pred) = sims[i]
            .sell_to_price(sims[i].prediction)
            .unwrap_or((0.0, 0.0, sims[i].sqrt_price_x96));

        if tokens_needed > 0.0 && tokens_needed <= held {
            sims[i].sqrt_price_x96 = sqrt_at_pred;
            budget += proceeds_to_pred;
            *sim_balances.get_mut(sims[i].market_name).unwrap() -= tokens_needed;
            actions.push(Action::Sell {
                market_name: sims[i].market_name,
                amount: tokens_needed,
                proceeds: proceeds_to_pred,
            });
        } else if held > 0.0 {
            let (sold, proceeds, new_sqrt) = sims[i]
                .sell_exact(held)
                .unwrap_or((0.0, 0.0, sims[i].sqrt_price_x96));
            if sold > 0.0 {
                sims[i].sqrt_price_x96 = new_sqrt;
                budget += proceeds;
                *sim_balances.get_mut(sims[i].market_name).unwrap() -= sold;
                actions.push(Action::Sell {
                    market_name: sims[i].market_name,
                    amount: sold,
                    proceeds,
                });
            }
        }
    }

    // ── Phase 2: Waterfall allocation ──
    let actions_before = actions.len();
    let last_bought_prof = waterfall(&mut sims, &mut budget, &mut actions, mint_available, missing_pred_sum);

    // Update sim_balances with positions acquired during waterfall.
    // Mint gives `amount` of every outcome; subsequent Sells reduce non-targets.
    // Partial sells leave residual holdings (amount - sold) that must be tracked.
    for action in &actions[actions_before..] {
        match action {
            Action::Buy { market_name, amount, .. } => {
                *sim_balances.entry(market_name).or_insert(0.0) += amount;
            }
            Action::Mint { amount, .. } => {
                for sim in sims.iter() {
                    *sim_balances.entry(sim.market_name).or_insert(0.0) += amount;
                }
            }
            Action::Sell { market_name, amount, .. } => {
                *sim_balances.entry(market_name).or_insert(0.0) -= amount;
            }
        }
    }

    // ── Phase 3: Post-allocation liquidation ──
    // If we bought something, check if any held outcomes are less profitable
    // than the last asset we bought. Sell lowest-profitability holdings first
    // and reallocate upward via waterfall.
    if last_bought_prof > 0.0 {
        // Collect held outcomes with profitability below last_bought_prof
        let mut liquidation_candidates: Vec<(usize, f64)> = Vec::new();
        for (i, sim) in sims.iter().enumerate() {
            let held = *sim_balances.get(sim.market_name).unwrap_or(&0.0);
            if held <= 0.0 {
                continue;
            }
            let (_, best_p) = best_route(&sims, i, mint_available, missing_pred_sum);
            let prof = profitability(sim.prediction, best_p);
            if prof < last_bought_prof {
                liquidation_candidates.push((i, prof));
            }
        }

        // Sort by profitability ascending (sell least profitable first)
        liquidation_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

        for (idx, _) in liquidation_candidates {
            let held = *sim_balances.get(sims[idx].market_name).unwrap_or(&0.0);
            if held <= 0.0 {
                continue;
            }
            // Sell only enough to raise profitability to last_bought_prof.
            // Target price = prediction / (1 + last_bought_prof).
            let target_price = target_price_for_prof(sims[idx].prediction, last_bought_prof);
            let (tokens_needed, proceeds_to_target, sqrt_at_target) = sims[idx]
                .sell_to_price(target_price)
                .unwrap_or((0.0, 0.0, sims[idx].sqrt_price_x96));
            let sell_amount = tokens_needed.min(held);
            if sell_amount > 0.0 {
                let (sold, proceeds, new_sqrt) = if (sell_amount - held).abs() < 1e-18 {
                    // Selling all holdings
                    sims[idx]
                        .sell_exact(held)
                        .unwrap_or((0.0, 0.0, sims[idx].sqrt_price_x96))
                } else if (sell_amount - tokens_needed).abs() < 1e-18 {
                    // Selling exactly the amount needed to reach target price
                    (tokens_needed, proceeds_to_target, sqrt_at_target)
                } else {
                    sims[idx]
                        .sell_exact(sell_amount)
                        .unwrap_or((0.0, 0.0, sims[idx].sqrt_price_x96))
                };
                if sold > 0.0 {
                    sims[idx].sqrt_price_x96 = new_sqrt;
                    budget += proceeds;
                    *sim_balances.get_mut(sims[idx].market_name).unwrap() -= sold;
                    actions.push(Action::Sell {
                        market_name: sims[idx].market_name,
                        amount: sold,
                        proceeds,
                    });
                }
            }
        }

        // Reallocate recovered capital via waterfall
        if budget > 0.0 {
            waterfall(&mut sims, &mut budget, &mut actions, mint_available, missing_pred_sum);
        }
    }

    actions
}

/// Waterfall allocation: deploy capital to the highest profitability outcome.
/// As capital is deployed, profitability drops until it matches the next outcome.
/// Then deploy to both, then three, etc.
///
/// Returns the profitability level of the last bought outcome (for post-liquidation).
fn waterfall(
    sims: &mut [PoolSim],
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
    missing_pred_sum: f64,
) -> f64 {
    if *budget <= 0.0 {
        return 0.0;
    }

    // Build sorted index of profitable outcomes
    let mut ranked: Vec<(usize, f64)> = sims
        .iter()
        .enumerate()
        .filter_map(|(i, sim)| {
            let (_, best_p) = best_route(sims, i, mint_available, missing_pred_sum);
            let prof = profitability(sim.prediction, best_p);
            if prof > 0.0 {
                Some((i, prof))
            } else {
                None
            }
        })
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    if ranked.is_empty() {
        return 0.0;
    }

    let mut active: Vec<usize> = vec![ranked[0].0];
    let mut next_rank = 1;
    let mut current_prof = ranked[0].1;
    let mut last_prof = 0.0;

    loop {
        if *budget <= 1e-12 || current_prof <= 0.0 {
            break;
        }

        // Target: next outcome's profitability, or 0 if none
        let target_prof = if next_rank < ranked.len() {
            ranked[next_rank].1.max(0.0)
        } else {
            0.0
        };

        // Compute cost for each active outcome to reach target_prof
        let mut step_costs: Vec<(f64, Route, f64, Option<U256>)> = Vec::new();
        let mut total_cost = 0.0;
        let mut any_failed = false;

        for &idx in &active {
            match cost_to_prof(sims, idx, target_prof, mint_available, missing_pred_sum, &active) {
                Some((cost, route, amount, new_sqrt)) => {
                    total_cost += cost;
                    step_costs.push((cost, route, amount, new_sqrt));
                }
                None => {
                    any_failed = true;
                    break;
                }
            }
        }

        if any_failed {
            // Remove outcomes that can't be computed, try again
            let snap = active.clone();
            active.retain(|&idx| cost_to_prof(sims, idx, target_prof, mint_available, missing_pred_sum, &snap).is_some());
            if active.is_empty() {
                break;
            }
            continue;
        }

        if total_cost <= *budget && total_cost > 0.0 {
            // Can afford to bring all active outcomes to target_prof: execute
            // Recompute each cost right before executing, since mint actions mutate other pools.
            // Guard against budget overspend from recomputed costs exceeding the estimate.
            let mut any_skipped = false;
            for &idx in &active {
                if let Some((cost, route, amount, new_sqrt)) =
                    cost_to_prof(sims, idx, target_prof, mint_available, missing_pred_sum, &active)
                {
                    if cost < 0.0 || cost > *budget {
                        any_skipped = true;
                        continue;
                    }
                    execute_buy(sims, idx, cost, amount, route, new_sqrt, budget, actions, &active);
                }
            }

            if any_skipped {
                // Not all outcomes reached target_prof; don't advance profitability level
                last_prof = current_prof;
                break;
            }

            current_prof = target_prof;
            last_prof = target_prof;

            // Add next outcome to active set
            if next_rank < ranked.len() {
                active.push(ranked[next_rank].0);
                next_rank += 1;
            } else {
                break; // No more outcomes, all reached prof 0
            }
        } else if total_cost > 0.0 {
            // Can't afford full step. Binary search for achievable profitability.
            let achievable = binary_search_prof(sims, &active, current_prof, target_prof, *budget, mint_available, missing_pred_sum);

            // Compute costs at achievable level
            for &idx in &active {
                if let Some((cost, route, amount, new_sqrt)) =
                    cost_to_prof(sims, idx, achievable, mint_available, missing_pred_sum, &active)
                {
                    if cost < 0.0 || cost > *budget {
                        continue;
                    }
                    execute_buy(sims, idx, cost, amount, route, new_sqrt, budget, actions, &active);
                }
            }
            last_prof = achievable;
            break;
        } else {
            // Zero cost means we're already at target, add next
            current_prof = target_prof;
            if next_rank < ranked.len() {
                active.push(ranked[next_rank].0);
                next_rank += 1;
            } else {
                break;
            }
        }
    }

    last_prof
}

/// Execute a buy via the chosen route, updating state.
fn execute_buy(
    sims: &mut [PoolSim],
    idx: usize,
    cost: f64,
    amount: f64,
    route: Route,
    new_sqrt: Option<U256>,
    budget: &mut f64,
    actions: &mut Vec<Action>,
    skip: &[usize],
) {
    match route {
        Route::Direct => {
            if let Some(ns) = new_sqrt {
                sims[idx].sqrt_price_x96 = ns;
            }
            *budget -= cost;
            actions.push(Action::Buy {
                market_name: sims[idx].market_name,
                amount,
                cost,
            });
        }
        Route::Mint => {
            // Mint `amount` complete sets, sell all others
            let upfront = amount; // 1 sUSD per set
            let proceeds = emit_mint_actions(sims, idx, amount, actions, skip);
            let net_cost = upfront - proceeds;
            *budget -= net_cost;
        }
    }
}

/// Binary search for the lowest profitability level affordable with the budget.
fn binary_search_prof(
    sims: &[PoolSim],
    active: &[usize],
    prof_hi: f64,
    prof_lo: f64,
    budget: f64,
    mint_available: bool,
    missing_pred_sum: f64,
) -> f64 {
    let mut lo = prof_lo; // most buying (expensive)
    let mut hi = prof_hi; // least buying (cheap, ~0 cost)

    for _ in 0..BINARY_SEARCH_ITERS {
        let mid = (lo + hi) / 2.0;
        let total: f64 = active
            .iter()
            .filter_map(|&idx| cost_to_prof(sims, idx, mid, mint_available, missing_pred_sum, active).map(|(c, _, _, _)| c))
            .sum();

        if total <= budget {
            hi = mid; // can afford, try lower prof (more buying)
        } else {
            lo = mid; // too expensive
        }
    }

    hi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markets::{MarketData, Pool, Tick};
    use crate::pools::Slot0Result;
    use alloy::primitives::Address;

    /// Mock pool: token0=quote(sUSD), token1=outcome → is_token1_outcome=true.
    /// Uses positive tick range matching real L1 pools with this ordering.
    /// Price range: ~[0.0001, 0.2] for outcome token.
    fn mock_pool() -> Pool {
        Pool {
            token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0", // quote (sUSD)
            token1: "0x1111111111111111111111111111111111111111", // outcome
            pool_id: "0x0000000000000000000000000000000000000001",
            liquidity: "1000000000000000000000", // 1000 tokens
            ticks: &[
                Tick {
                    tick_idx: 16095,
                    liquidity_net: 1000000000000000000000,
                },
                Tick {
                    tick_idx: 92108,
                    liquidity_net: -1000000000000000000000,
                },
            ],
        }
    }

    /// Leak a pool into a static reference for tests
    fn leak_pool(pool: Pool) -> &'static Pool {
        Box::leak(Box::new(pool))
    }

    fn leak_market(market: MarketData) -> &'static MarketData {
        Box::leak(Box::new(market))
    }

    /// Create slot0 + market data pair for testing.
    /// `price_fraction` sets the outcome price approximately (0.0 to 1.0).
    fn mock_slot0_market(
        name: &'static str,
        outcome_token: &'static str,
        price_fraction: f64,
    ) -> (Slot0Result, &'static MarketData) {
        let mut p = mock_pool();
        p.token1 = outcome_token;
        let pool = leak_pool(p);

        // token0 is quote (sUSD), token1 is outcome
        // is_token1_outcome = true
        // outcome price = inv_price = 2^192 / sqrtPriceX96^2
        // To get outcome price = price_fraction:
        // sqrtPriceX96 = sqrt(2^192 / (price_fraction * 1e18)) * sqrt(1e18)
        // Simpler: use prediction_to_sqrt_price_x96
        let sqrt_price = prediction_to_sqrt_price_x96(price_fraction, true)
            .unwrap_or(U256::from(1u128 << 96));

        let market = leak_market(MarketData {
            name,
            market_id: crate::markets::MARKETS_L1[0].market_id,
            outcome_token,
            pool: Some(*pool),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        });

        let slot0 = Slot0Result {
            pool_id: Address::ZERO,
            sqrt_price_x96: sqrt_price,
            tick: 0,
            observation_index: 0,
            observation_cardinality: 0,
            observation_cardinality_next: 0,
            fee_protocol: 0,
            unlocked: true,
        };

        (slot0, market)
    }

    #[test]
    fn test_pool_sim_price_roundtrip() {
        // Create a sim and verify the price roughly matches what we set
        let (slot0, market) = mock_slot0_market(
            "test_market",
            "0x1111111111111111111111111111111111111111",
            0.3,
        );
        let sim = PoolSim::from_slot0(&slot0, market, 0.5).unwrap();
        let price = sim.price();
        assert!(
            (price - 0.3).abs() < 0.01,
            "price {} should be ~0.3",
            price
        );
    }

    #[test]
    fn test_sell_overpriced() {
        // Outcome priced at 0.5, prediction is 0.3 → overpriced, should sell
        let (slot0, market) = mock_slot0_market(
            "overpriced",
            "0x1111111111111111111111111111111111111111",
            0.5,
        );
        let slot0_results = vec![(slot0, market)];
        let mut balances = HashMap::new();
        balances.insert("overpriced", 100.0);

        // Temporarily add a matching prediction
        // Since we can't add to PREDICTIONS_L1 (static), we test via PoolSim directly
        let mut sims = vec![PoolSim::from_slot0(&slot0_results[0].0, slot0_results[0].1, 0.3).unwrap()];

        // Simulate sell phase
        let sim = &mut sims[0];
        let price = sim.price();
        assert!(price > 0.3, "price {} should be > 0.3", price);

        let (tokens_needed, proceeds, new_sqrt) = sim.sell_to_price(0.3).unwrap();
        assert!(tokens_needed > 0.0, "should need to sell some tokens");
        assert!(proceeds > 0.0, "should receive proceeds");

        // Check price after sell
        let price_after = sim.price_at(new_sqrt);
        assert!(
            (price_after - 0.3).abs() < 0.01,
            "price after sell {} should be ~0.3",
            price_after
        );
    }

    #[test]
    fn test_cost_to_price() {
        // Price at 0.01, want to buy until price = 0.1 (within pool range ~[0.0001, 0.2])
        let (slot0, market) = mock_slot0_market(
            "cheap",
            "0x1111111111111111111111111111111111111111",
            0.01,
        );
        let sim = PoolSim::from_slot0(&slot0, market, 0.5).unwrap();

        let (cost, amount, new_sqrt) = sim.cost_to_price(0.1).unwrap();
        assert!(cost > 0.0, "should cost something to move price");
        assert!(amount > 0.0, "should receive outcome tokens");

        // Price should increase toward target, clamped to tick boundary
        let price_after = sim.price_at(new_sqrt);
        assert!(
            price_after > 0.01,
            "price after buy {} should be > initial 0.01",
            price_after
        );
    }

    #[test]
    fn test_waterfall_equalizes() {
        // Two outcomes: A at price 0.05 pred 0.1, B at price 0.05 pred 0.08
        // profitability(A) = (0.1 - 0.05)/0.05 = 1.0
        // profitability(B) = (0.08 - 0.05)/0.05 = 0.6
        // Waterfall should first buy A until prof(A) = 0.6, then buy both

        let (slot0_a, market_a) = mock_slot0_market(
            "A",
            "0x1111111111111111111111111111111111111111",
            0.05,
        );
        let (slot0_b, market_b) = mock_slot0_market(
            "B",
            "0x2222222222222222222222222222222222222222",
            0.05,
        );

        let mut sims = vec![
            PoolSim::from_slot0(&slot0_a, market_a, 0.10).unwrap(),
            PoolSim::from_slot0(&slot0_b, market_b, 0.08).unwrap(),
        ];

        let mut budget = 1000.0;
        let mut actions = Vec::new();

        waterfall(&mut sims, &mut budget, &mut actions, false, 0.0);

        // Should have buy actions
        let buys: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, Action::Buy { .. }))
            .collect();
        assert!(!buys.is_empty(), "should have buy actions");

        // A should be bought first (higher profitability)
        if let Action::Buy { market_name, .. } = &buys[0] {
            assert_eq!(*market_name, "A", "A should be bought first");
        }
    }

    #[test]
    fn test_no_action_when_all_overpriced_no_holdings() {
        // Price > prediction but no holdings → nothing to do
        let (slot0, market) = mock_slot0_market(
            "overpriced",
            "0x1111111111111111111111111111111111111111",
            0.5,
        );
        let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.3).unwrap()];
        let mut budget = 100.0;
        let mut actions = Vec::new();

        waterfall(&mut sims, &mut budget, &mut actions, false, 0.0);

        assert!(actions.is_empty(), "no actions when everything overpriced");
        assert!(
            (budget - 100.0).abs() < 1e-6,
            "budget should be unchanged"
        );
    }

    #[test]
    fn test_budget_exhaustion() {
        let (slot0, market) = mock_slot0_market(
            "cheap",
            "0x1111111111111111111111111111111111111111",
            0.01,
        );
        let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.5).unwrap()];

        let mut budget = 0.001; // tiny budget
        let mut actions = Vec::new();

        waterfall(&mut sims, &mut budget, &mut actions, false, 0.0);

        // Should have a buy but budget should be nearly exhausted
        let buys: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, Action::Buy { .. }))
            .collect();
        if !buys.is_empty() {
            assert!(budget < 0.001, "budget should be reduced");
        }
    }

    #[test]
    fn test_mint_route_actions() {
        // Directly test emit_mint_actions and execute_buy with Route::Mint.
        // 3 outcomes: target=M1, non-targets M2,M3 will be sold.
        let tokens = [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
        ];
        let names = ["M1", "M2", "M3"];

        let slot0_results: Vec<_> = tokens
            .iter()
            .zip(names.iter())
            .map(|(tok, name)| mock_slot0_market(name, tok, 0.05))
            .collect();

        let mut sims: Vec<_> = slot0_results
            .iter()
            .map(|(s, m)| PoolSim::from_slot0(s, m, 0.3).unwrap())
            .collect();

        // Test emit_mint_actions directly
        let mint_amount = 10.0;
        let mut actions = Vec::new();
        let proceeds = emit_mint_actions(&mut sims, 0, mint_amount, &mut actions, &[]);

        // First action: Mint with target_market = M1
        assert!(
            matches!(&actions[0], Action::Mint { target_market: "M1", .. }),
            "first action should be Mint targeting M1"
        );
        if let Action::Mint { amount, .. } = &actions[0] {
            assert!((*amount - 10.0).abs() < 1e-12, "mint amount should be 10.0");
        }

        // Sells for non-target outcomes
        let sells: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                Action::Sell { market_name, amount, .. } => Some((*market_name, *amount)),
                _ => None,
            })
            .collect();
        assert_eq!(sells.len(), 2, "should sell 2 non-target outcomes");
        for (name, amt) in &sells {
            assert!(*name == "M2" || *name == "M3");
            assert!(*amt > 0.0);
        }
        assert!(proceeds > 0.0, "selling non-targets should yield proceeds");

        // Test execute_buy with Route::Mint updates budget correctly
        let mut sims2: Vec<_> = slot0_results
            .iter()
            .map(|(s, m)| PoolSim::from_slot0(s, m, 0.3).unwrap())
            .collect();
        let mut budget = 100.0;
        let mut actions2 = Vec::new();
        execute_buy(&mut sims2, 0, 5.0, 10.0, Route::Mint, None, &mut budget, &mut actions2, &[]);
        assert!(budget < 100.0, "budget should decrease after mint");

        // Test sim_balances tracking: Mint adds to all, Sell subtracts
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        for action in &actions2 {
            match action {
                Action::Buy { market_name, amount, .. } => {
                    *sim_balances.entry(market_name).or_insert(0.0) += amount;
                }
                Action::Mint { amount, .. } => {
                    for sim in sims2.iter() {
                        *sim_balances.entry(sim.market_name).or_insert(0.0) += amount;
                    }
                }
                Action::Sell { market_name, amount, .. } => {
                    *sim_balances.entry(market_name).or_insert(0.0) -= amount;
                }
            }
        }
        // Target M1 should hold the full mint amount
        let m1_bal = *sim_balances.get("M1").unwrap_or(&0.0);
        assert!(
            (m1_bal - 10.0).abs() < 1e-12,
            "M1 should hold 10.0 minted tokens, got {}", m1_bal
        );
        // Non-targets should have residual >= 0 (mint - sold)
        for name in &["M2", "M3"] {
            let bal = *sim_balances.get(name).unwrap_or(&0.0);
            assert!(bal >= -1e-12, "{} balance should be >= 0 (residual), got {}", name, bal);
        }
    }

    #[tokio::test]
    async fn test_rebalance_integration() {
        // Integration test with real pool data
        dotenvy::dotenv().ok();
        let rpc_url = match std::env::var("RPC") {
            Ok(url) => url,
            Err(_) => return, // skip if no RPC
        };
        let provider =
            alloy::providers::ProviderBuilder::new().connect_http(rpc_url.parse().unwrap());

        let slot0_results = crate::pools::fetch_all_slot0(provider).await.unwrap();

        let actions = rebalance(&HashMap::new(), 100.0, &slot0_results);

        println!("Rebalance actions ({}):", actions.len());
        for action in &actions {
            match action {
                Action::Mint {
                    contract_1,
                    contract_2,
                    amount,
                    target_market,
                } => {
                    println!("  MINT {} sets for {} (c1={}, c2={})", amount, target_market, contract_1, contract_2)
                }
                Action::Buy {
                    market_name,
                    amount,
                    cost,
                } => println!("  BUY {} {} (cost: {:.6})", amount, market_name, cost),
                Action::Sell {
                    market_name,
                    amount,
                    proceeds,
                } => println!(
                    "  SELL {} {} (proceeds: {:.6})",
                    amount, market_name, proceeds
                ),
            }
        }
    }
}
