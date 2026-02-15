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
pub(super) fn best_non_active(
    sims: &[PoolSim],
    active_set: &HashSet<(usize, Route)>,
    mint_available: bool,
    price_sum: f64,
) -> Option<(usize, Route, f64)> {
    let mut best: Option<(usize, Route, f64)> = None;
    for (i, sim) in sims.iter().enumerate() {
        if !active_set.contains(&(i, Route::Direct)) {
            let prof = profitability(sim.prediction, sim.price());
            if prof > 0.0 && best.is_none_or(|b| prof > b.2) {
                best = Some((i, Route::Direct, prof));
            }
        }
        if mint_available && !active_set.contains(&(i, Route::Mint)) {
            let mp = alt_price(sims, i, price_sum);
            let prof = profitability(sim.prediction, mp);
            if prof > 0.0 && best.is_none_or(|b| prof > b.2) {
                best = Some((i, Route::Mint, prof));
            }
        }
    }
    best
}

pub(super) const MAX_WATERFALL_ITERS: usize = 1000;

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
) -> f64 {
    if *budget <= 0.0 {
        return 0.0;
    }

    // Precompute price_sum; maintained incrementally after executions.
    let mut price_sum: f64 = sims.iter().map(|s| s.price()).sum();
    let mut active_set: HashSet<(usize, Route)> = HashSet::new();

    // Seed active set with the highest-profitability entry.
    let first = match best_non_active(sims, &active_set, mint_available, price_sum) {
        Some(entry) if entry.2 > 0.0 => entry,
        _ => return 0.0,
    };

    let mut active: Vec<(usize, Route)> = vec![(first.0, first.1)];
    active_set.insert((first.0, first.1));
    let mut current_prof = first.2;
    let mut last_prof = 0.0;
    let mut waterfall_balances: HashMap<&str, f64> = HashMap::new();
    let mut planning_sim_state: Vec<PoolSim> = Vec::with_capacity(sims.len());

    for _iter in 0..MAX_WATERFALL_ITERS {
        if *budget <= EPS || current_prof <= 0.0 {
            break;
        }

        // Dynamically find the next best entry from current pool state.
        // If a mint perturbed prices and pushed an entry above current_prof,
        // absorb it into active immediately (no cost step needed).
        loop {
            match best_non_active(sims, &active_set, mint_available, price_sum) {
                Some((idx, route, prof)) if prof > current_prof => {
                    active.push((idx, route));
                    active_set.insert((idx, route));
                }
                _ => break,
            }
        }

        let next = best_non_active(sims, &active_set, mint_available, price_sum);
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
        ) {
            Some(plan) => plan,
            None => {
                last_prof = current_prof;
                break;
            }
        };

        if plan_is_budget_feasible(&full_plan, *budget) {
            let executed = {
                waterfall_balances.clear();
                let mut exec = ExecutionState::new(sims, budget, actions, &mut waterfall_balances);
                exec.execute_planned_routes(&full_plan, &skip)
            };
            if !executed {
                last_prof = current_prof;
                break;
            }

            // Refresh price_sum after executions
            price_sum = sims.iter().map(|s| s.price()).sum();
            current_prof = target_prof;
            last_prof = target_prof;

            // Re-query best entry from post-execution state
            match best_non_active(sims, &active_set, mint_available, price_sum) {
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
            let executed = {
                waterfall_balances.clear();
                let mut exec = ExecutionState::new(sims, budget, actions, &mut waterfall_balances);
                exec.execute_planned_routes(&execution_plan, &skip)
            };
            if !executed {
                last_prof = current_prof;
                break;
            }
            last_prof = achievable;
            break;
        }
    }

    last_prof
}
