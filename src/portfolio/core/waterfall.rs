use std::collections::{HashMap, HashSet};

use super::Action;
use super::planning::{
    PlannedRoute, active_skip_indices, cost_for_route, plan_active_routes_with_scratch,
    plan_is_budget_feasible, solve_prof,
};
use super::sim::{EPS, PoolSim, Route, alt_price, profitability};
use super::trading::ExecutionState;

/// Find the highest-profitability (outcome, route) pair not already in the active set.
/// Scans current pool state each call, so mint perturbations are reflected immediately.
///
/// `remaining_budget`, `gas_direct_susd`, `gas_mint_susd`: gas-aware minimum-trade filter.
/// An outcome is only admitted if `remaining_budget × prof >= gas_cost`, i.e. the maximum
/// possible profit from the remaining budget covers the gas cost. Pass `0.0` to disable.
pub(super) fn best_non_active(
    sims: &[PoolSim],
    active_set: &HashSet<(usize, Route)>,
    mint_available: bool,
    price_sum: f64,
    remaining_budget: f64,
    gas_direct_susd: f64,
    gas_mint_susd: f64,
) -> Option<(usize, Route, f64)> {
    let mut best: Option<(usize, Route, f64)> = None;
    for (i, sim) in sims.iter().enumerate() {
        if !active_set.contains(&(i, Route::Direct)) {
            let prof = profitability(sim.prediction, sim.price());
            if prof > 0.0
                && remaining_budget * prof >= gas_direct_susd
                && best.is_none_or(|b| prof > b.2)
            {
                best = Some((i, Route::Direct, prof));
            }
        }
        if mint_available && !active_set.contains(&(i, Route::Mint)) {
            let mp = alt_price(sims, i, price_sum);
            let prof = profitability(sim.prediction, mp);
            if prof > 0.0
                && remaining_budget * prof >= gas_mint_susd
                && best.is_none_or(|b| prof > b.2)
            {
                best = Some((i, Route::Mint, prof));
            }
        }
    }
    best
}

pub(super) const MAX_WATERFALL_ITERS: usize = 1000;
const MAX_STALLED_CONTINUES: usize = 4;

fn iteration_made_progress(
    prev_prof: f64,
    next_prof: f64,
    prev_budget: f64,
    next_budget: f64,
) -> bool {
    let prof_tol = EPS * (1.0 + prev_prof.abs().max(next_prof.abs()));
    if (next_prof - prev_prof).abs() > prof_tol {
        return true;
    }

    let budget_tol = EPS * (1.0 + prev_budget.abs().max(next_budget.abs()));
    (next_budget - prev_budget).abs() > budget_tol
}

fn realized_step_profitability(
    sims: &[PoolSim],
    step: &PlannedRoute,
    price_sum: f64,
) -> Option<f64> {
    let prof = match step.route {
        Route::Direct => profitability(sims[step.idx].prediction, sims[step.idx].price()),
        Route::Mint => profitability(
            sims[step.idx].prediction,
            alt_price(sims, step.idx, price_sum),
        ),
    };
    prof.is_finite().then_some(prof)
}

/// Waterfall allocation: deploy capital to the highest profitability outcome.
/// As capital is deployed, profitability drops until it matches the next outcome.
/// Then deploy to both, then three, etc.
///
/// Returns the profitability level of the last bought outcome (for post-liquidation).
pub(super) fn waterfall(
    sims: &mut [PoolSim],
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
    gas_direct_susd: f64,
    gas_mint_susd: f64,
) -> f64 {
    if *budget <= 0.0 {
        return 0.0;
    }

    // Precompute price_sum; maintained incrementally after executions.
    let mut price_sum: f64 = sims.iter().map(|s| s.price()).sum();
    let mut active_set: HashSet<(usize, Route)> = HashSet::new();

    // Seed active set with the highest-profitability entry.
    let first = match best_non_active(sims, &active_set, mint_available, price_sum, *budget, gas_direct_susd, gas_mint_susd) {
        Some(entry) if entry.2 > 0.0 => entry,
        _ => return 0.0,
    };

    let mut active: Vec<(usize, Route)> = vec![(first.0, first.1)];
    active_set.insert((first.0, first.1));
    let mut current_prof = first.2;
    let mut last_prof = 0.0;
    let mut waterfall_balances: HashMap<&str, f64> = HashMap::new();
    let mut planning_sim_state: Vec<PoolSim> = Vec::with_capacity(sims.len());
    let mut stalled_continues = 0usize;

    for _iter in 0..MAX_WATERFALL_ITERS {
        if *budget <= EPS || current_prof <= 0.0 {
            break;
        }
        let iter_start_prof = current_prof;
        let iter_start_budget = *budget;

        // Dynamically find the next best entry from current pool state.
        // If a mint perturbed prices and pushed an entry above current_prof,
        // absorb it into active immediately (no cost step needed).
        loop {
            match best_non_active(sims, &active_set, mint_available, price_sum, *budget, gas_direct_susd, gas_mint_susd) {
                Some((idx, route, prof)) if prof > current_prof => {
                    active.push((idx, route));
                    active_set.insert((idx, route));
                }
                _ => break,
            }
        }

        let next = best_non_active(sims, &active_set, mint_available, price_sum, *budget, gas_direct_susd, gas_mint_susd);
        let target_prof = match next {
            Some((_, _, p)) if p > 0.0 => p,
            _ => 0.0,
        };

        // Prune entries that can't reach target_prof (e.g. tick boundary hit).
        // Re-derive skip after each removal so remaining entries see the correct set.
        loop {
            let skip = active_skip_indices(&active);
            let before = active.len();
            active.retain(|&(idx, route)| {
                cost_for_route(sims, idx, route, target_prof, &skip, price_sum).is_some()
            });
            if active.len() == before || active.is_empty() {
                break;
            }
            active_set = active.iter().copied().collect();
        }
        if active.is_empty() {
            break;
        }

        let skip = active_skip_indices(&active);
        let full_plan = match plan_active_routes_with_scratch(
            sims,
            &active,
            target_prof,
            &skip,
            &mut planning_sim_state,
            Some(current_prof),
        ) {
            Some(plan) => plan,
            None => {
                last_prof = current_prof;
                break;
            }
        };
        let full_plan_hits_active_boundary = full_plan
            .last()
            .is_some_and(|step| step.active_set_boundary_hit);

        if plan_is_budget_feasible(&full_plan, *budget) {
            let executed = {
                waterfall_balances.clear();
                let mut exec = ExecutionState::new(sims, budget, actions, &mut waterfall_balances);
                exec.execute_planned_routes(&full_plan, &skip)
            };
            if !executed {
                price_sum = sims.iter().map(|s| s.price()).sum();
                last_prof = full_plan
                    .last()
                    .and_then(|step| realized_step_profitability(sims, step, price_sum))
                    .unwrap_or(current_prof);
                break;
            }

            // Refresh price_sum after executions
            price_sum = sims.iter().map(|s| s.price()).sum();
            if full_plan_hits_active_boundary {
                let realized_prof = full_plan
                    .last()
                    .and_then(|step| realized_step_profitability(sims, step, price_sum))
                    .unwrap_or(current_prof);
                last_prof = realized_prof;
                current_prof = realized_prof;
                if !iteration_made_progress(
                    iter_start_prof,
                    current_prof,
                    iter_start_budget,
                    *budget,
                ) {
                    stalled_continues += 1;
                    if stalled_continues >= MAX_STALLED_CONTINUES {
                        break;
                    }
                } else {
                    stalled_continues = 0;
                }
                continue;
            }
            current_prof = target_prof;
            last_prof = target_prof;
            stalled_continues = 0;

            // Re-query best entry from post-execution state
            match best_non_active(sims, &active_set, mint_available, price_sum, *budget, gas_direct_susd, gas_mint_susd) {
                Some((idx, route, prof)) if prof > 0.0 => {
                    active.push((idx, route));
                    active_set.insert((idx, route));
                }
                _ => break,
            }
        } else {
            // Can't afford full step. Solve for lowest feasible profitability in [target_prof, current_prof].
            let mut achievable =
                solve_prof(sims, &active, current_prof, target_prof, *budget, &skip);
            let mut execution_plan = plan_active_routes_with_scratch(
                sims,
                &active,
                achievable,
                &skip,
                &mut planning_sim_state,
                Some(current_prof),
            );

            // Numerical guard: tighten toward current_prof until feasible.
            if execution_plan
                .as_ref()
                .map(|p| !plan_is_budget_feasible(p, *budget))
                .unwrap_or(true)
            {
                let mut lo = achievable;
                let mut hi = current_prof;
                let mut best: Option<(f64, Vec<PlannedRoute>)> = None;
                for _ in 0..32 {
                    let mid = 0.5 * (lo + hi);
                    if let Some(plan) = plan_active_routes_with_scratch(
                        sims,
                        &active,
                        mid,
                        &skip,
                        &mut planning_sim_state,
                        Some(current_prof),
                    ) {
                        if plan_is_budget_feasible(&plan, *budget) {
                            best = Some((mid, plan));
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    } else {
                        lo = mid;
                    }
                    if (hi - lo).abs() <= EPS * (1.0 + hi.abs()) {
                        break;
                    }
                }
                if let Some((best_prof, plan)) = best {
                    achievable = best_prof;
                    execution_plan = Some(plan);
                }
            }

            let Some(execution_plan) = execution_plan else {
                last_prof = current_prof;
                break;
            };
            if !plan_is_budget_feasible(&execution_plan, *budget) {
                last_prof = current_prof;
                break;
            }
            let execution_plan_hits_active_boundary = execution_plan
                .last()
                .is_some_and(|step| step.active_set_boundary_hit);
            let executed = {
                waterfall_balances.clear();
                let mut exec = ExecutionState::new(sims, budget, actions, &mut waterfall_balances);
                exec.execute_planned_routes(&execution_plan, &skip)
            };
            if !executed {
                price_sum = sims.iter().map(|s| s.price()).sum();
                last_prof = execution_plan
                    .last()
                    .and_then(|step| realized_step_profitability(sims, step, price_sum))
                    .unwrap_or(current_prof);
                break;
            }
            price_sum = sims.iter().map(|s| s.price()).sum();
            if execution_plan_hits_active_boundary {
                let realized_prof = execution_plan
                    .last()
                    .and_then(|step| realized_step_profitability(sims, step, price_sum))
                    .unwrap_or(current_prof);
                last_prof = realized_prof;
                current_prof = realized_prof;
                if !iteration_made_progress(
                    iter_start_prof,
                    current_prof,
                    iter_start_budget,
                    *budget,
                ) {
                    stalled_continues += 1;
                    if stalled_continues >= MAX_STALLED_CONTINUES {
                        break;
                    }
                } else {
                    stalled_continues = 0;
                }
                continue;
            }
            last_prof = achievable;
            current_prof = achievable;
            if !iteration_made_progress(iter_start_prof, current_prof, iter_start_budget, *budget) {
                stalled_continues += 1;
                if stalled_continues >= MAX_STALLED_CONTINUES {
                    break;
                }
            } else {
                stalled_continues = 0;
            }
            continue;
        }
    }

    last_prof
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markets::{MarketData, Pool, Tick, MARKETS_L1};
    use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};
    use alloy::primitives::{Address, U256};

    /// Minimal PoolSim for waterfall tests: price ~= price_frac, prediction = pred.
    /// Uses Box::leak to produce 'static references (test process memory, not freed).
    fn make_sim(name: &'static str, token: &'static str, price_frac: f64, pred: f64) -> PoolSim {
        let liq_str: &'static str =
            Box::leak("1000000000000000000000".to_string().into_boxed_str());
        let ticks: &'static [Tick] = Box::leak(Box::new([
            Tick { tick_idx: 1, liquidity_net: 1_000_000_000_000_000_000_000 },
            Tick { tick_idx: 92108, liquidity_net: -1_000_000_000_000_000_000_000 },
        ]));
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
            token1: token,
            pool_id: "0x0000000000000000000000000000000000000001",
            liquidity: liq_str,
            ticks,
        }));
        let sqrt = prediction_to_sqrt_price_x96(price_frac, true)
            .unwrap_or(U256::from(1u128 << 96));
        let market: &'static MarketData = Box::leak(Box::new(MarketData {
            name,
            market_id: MARKETS_L1[0].market_id,
            outcome_token: token,
            pool: Some(*pool),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        }));
        let slot0 = Slot0Result {
            pool_id: Address::ZERO,
            sqrt_price_x96: sqrt,
            tick: 0,
            observation_index: 0,
            observation_cardinality: 0,
            observation_cardinality_next: 0,
            fee_protocol: 0,
            unlocked: true,
        };
        PoolSim::from_slot0(&slot0, market, pred).unwrap()
    }

    #[test]
    fn waterfall_skips_outcome_when_budget_below_break_even() {
        // profitability ≈ (0.0101 - 0.01) / 0.01 = 1% = 0.01
        // gas_direct = $0.50, break-even min budget = 0.50 / 0.01 = $50
        // budget = $1 < $50 → outcome must be skipped → zero actions
        let mut sims =
            vec![make_sim("m1", "0x1111111111111111111111111111111111111111", 0.01, 0.0101)];
        let mut budget = 1.0_f64;
        let mut actions = vec![];

        let last_prof = waterfall(
            &mut sims,
            &mut budget,
            &mut actions,
            false,
            0.50, // gas_direct_susd: $0.50
            2.00, // gas_mint_susd: $2.00
        );
        assert!(
            actions.is_empty(),
            "budget $1 at 1% profitability cannot cover $0.50 gas; got {} actions, last_prof={last_prof}",
            actions.len()
        );
        assert!(
            (budget - 1.0).abs() < 1e-9,
            "budget must be unchanged when all trades skipped; got {budget}"
        );
    }

    #[test]
    fn waterfall_executes_when_budget_above_break_even() {
        // Same outcome, budget = $100; break-even = $50; $100 > $50 → should trade
        let mut sims =
            vec![make_sim("m2", "0x2222222222222222222222222222222222222222", 0.01, 0.0101)];
        let mut budget = 100.0_f64;
        let mut actions = vec![];

        let _last_prof =
            waterfall(&mut sims, &mut budget, &mut actions, false, 0.50, 2.00);
        assert!(
            !actions.is_empty(),
            "budget $100 at 1% profitability must exceed $0.50 gas break-even; got 0 actions"
        );
    }

    #[test]
    fn waterfall_with_zero_gas_thresholds_behaves_as_before() {
        // gas_direct=0, gas_mint=0 → no filtering, same as old signature
        let mut sims =
            vec![make_sim("m3", "0x3333333333333333333333333333333333333333", 0.01, 0.0101)];
        let mut budget = 1.0_f64;
        let mut actions = vec![];

        let _last_prof =
            waterfall(&mut sims, &mut budget, &mut actions, false, 0.0, 0.0);
        assert!(
            !actions.is_empty(),
            "with zero gas thresholds, any positive profitability should produce actions"
        );
    }
}
