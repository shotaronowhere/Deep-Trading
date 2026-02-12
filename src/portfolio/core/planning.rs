use std::collections::HashSet;

use super::sim::{DUST, EPS, PoolSim, Route, target_price_for_prof};
use super::solver::mint_cost_to_prof;

const SOLVE_PROF_ITERS: usize = 64;

/// For the direct route, compute cost to bring an outcome's profitability to `target_prof`.
/// Returns (cost, outcome_amount, new_price).
fn direct_cost_to_prof(sim: &PoolSim, target_prof: f64) -> Option<(f64, f64, f64)> {
    let tp = target_price_for_prof(sim.prediction, target_prof);
    sim.cost_to_price(tp)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RouteEstimate {
    pub(super) cash_cost: f64,
    pub(super) amount: f64,
    pub(super) new_price: Option<f64>,
}

/// Compute cost to reach target_prof for a given outcome via a specific route.
/// cash_cost = actual sUSD spent; value_cost = cash_cost minus expected value of unsold tokens.
pub(super) fn cost_for_route(
    sims: &[PoolSim],
    idx: usize,
    route: Route,
    target_prof: f64,
    skip: &HashSet<usize>,
    price_sum: f64,
) -> Option<RouteEstimate> {
    match route {
        Route::Direct => {
            direct_cost_to_prof(&sims[idx], target_prof).map(|(cost, amount, new_price)| {
                RouteEstimate {
                    cash_cost: cost,
                    amount,
                    new_price: Some(new_price),
                }
            })
        }
        Route::Mint => mint_cost_to_prof(sims, idx, target_prof, skip, price_sum).map(
            |(cash, _value, amount, _dcost)| RouteEstimate {
                cash_cost: cash,
                amount,
                new_price: None,
            },
        ),
    }
}

/// Extract deduplicated outcome indices from active (outcome, route) pairs.
pub(super) fn active_skip_indices(active: &[(usize, Route)]) -> HashSet<usize> {
    active.iter().map(|(idx, _)| *idx).collect()
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PlannedRoute {
    pub(super) idx: usize,
    pub(super) route: Route,
    pub(super) cost: f64,
    pub(super) amount: f64,
    pub(super) new_price: Option<f64>,
}

fn execution_order(active: &[(usize, Route)]) -> impl Iterator<Item = (usize, Route)> + '_ {
    active
        .iter()
        .copied()
        .filter(|(_, route)| *route == Route::Mint)
        .chain(
            active
                .iter()
                .copied()
                .filter(|(_, route)| *route == Route::Direct),
        )
}

pub(super) fn plan_active_routes(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    target_prof: f64,
    skip: &HashSet<usize>,
) -> Option<Vec<PlannedRoute>> {
    let mut sim_state: Vec<PoolSim> = Vec::with_capacity(sims.len());
    plan_active_routes_with_scratch(sims, active, target_prof, skip, &mut sim_state)
}

pub(super) fn plan_active_routes_with_scratch(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    target_prof: f64,
    skip: &HashSet<usize>,
    sim_state: &mut Vec<PoolSim>,
) -> Option<Vec<PlannedRoute>> {
    if sim_state.len() != sims.len() {
        sim_state.clear();
        sim_state.extend_from_slice(sims);
    } else {
        sim_state.clone_from_slice(sims);
    }

    let mut price_sum: f64 = sim_state.iter().map(|s| s.price).sum();
    let mut plan: Vec<PlannedRoute> = Vec::with_capacity(active.len());

    for (idx, route) in execution_order(active) {
        let estimate = cost_for_route(sim_state, idx, route, target_prof, skip, price_sum)?;
        let direct_cost = estimate.cash_cost;
        let amount = estimate.amount;
        let new_price = estimate.new_price;

        let (actual_cost, applied_new_price) = match route {
            Route::Direct => {
                let np = new_price?;
                sim_state[idx].price = np;
                (direct_cost, Some(np))
            }
            Route::Mint => {
                let mut proceeds = 0.0_f64;
                if amount > DUST {
                    for (i, sim) in sim_state.iter_mut().enumerate() {
                        if i == idx || skip.contains(&i) {
                            continue;
                        }
                        if let Some((sold, leg_proceeds, new_leg_price)) = sim.sell_exact(amount)
                            && sold > DUST
                        {
                            proceeds += leg_proceeds;
                            sim.price = new_leg_price;
                        }
                    }
                }
                (amount - proceeds, None)
            }
        };

        if !actual_cost.is_finite() {
            return None;
        }
        plan.push(PlannedRoute {
            idx,
            route,
            cost: actual_cost,
            amount,
            new_price: applied_new_price,
        });

        price_sum = sim_state.iter().map(|s| s.price).sum();
    }

    Some(plan)
}

pub(super) fn plan_is_budget_feasible(plan: &[PlannedRoute], budget: f64) -> bool {
    let mut running_budget = budget;
    for step in plan {
        if step.cost > running_budget + EPS {
            return false;
        }
        running_budget -= step.cost;
        if !running_budget.is_finite() {
            return false;
        }
    }
    true
}

/// Find the lowest profitability level affordable with the available budget.
/// Uses closed-form for all-direct, and simulation-backed bisection for mixed routes.
pub(super) fn solve_prof(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    prof_hi: f64,
    prof_lo: f64,
    budget: f64,
    skip: &HashSet<usize>,
) -> f64 {
    let all_direct = active.iter().all(|&(_, route)| route == Route::Direct);

    if all_direct {
        // Closed form: π = (A/B)² - 1
        let mut a_sum = 0.0_f64;
        let mut b_sum = 0.0_f64;
        for &(idx, _) in active {
            let l = sims[idx].l_eff();
            let p = sims[idx].price();
            a_sum += l * sims[idx].prediction.sqrt();
            b_sum += l * p.sqrt();
        }
        let b = budget + b_sum;
        if b <= 0.0 {
            return prof_hi;
        }
        let ratio = a_sum / b;
        let prof = ratio * ratio - 1.0;
        return prof.clamp(prof_lo, prof_hi);
    }

    let mut sim_state: Vec<PoolSim> = Vec::with_capacity(sims.len());
    let mut affordable = |prof: f64| -> bool {
        plan_active_routes_with_scratch(sims, active, prof, skip, &mut sim_state)
            .map(|plan| plan_is_budget_feasible(&plan, budget))
            .unwrap_or(false)
    };

    if affordable(prof_lo) {
        return prof_lo;
    }
    if !affordable(prof_hi) {
        return prof_hi;
    }

    let mut lo = prof_lo;
    let mut hi = prof_hi;
    for _ in 0..SOLVE_PROF_ITERS {
        let mid = 0.5 * (lo + hi);
        if affordable(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo).abs() <= EPS * (1.0 + hi.abs()) {
            break;
        }
    }

    hi.clamp(prof_lo, prof_hi)
}
