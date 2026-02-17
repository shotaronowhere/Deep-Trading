use std::collections::HashSet;

use super::sim::{DUST, EPS, PoolSim, Route, alt_price, profitability, target_price_for_prof};
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
    pub(super) active_set_boundary_hit: bool,
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

fn profitability_level_reached(prof: f64, target_prof: f64) -> bool {
    let tol = EPS * (1.0 + prof.abs() + target_prof.abs());
    prof + tol >= target_prof
}

fn mint_amount_to_reach_profitability(sim: &PoolSim, target_prof: f64) -> Option<f64> {
    let target_price = target_price_for_prof(sim.prediction, target_prof);
    let (amount, _proceeds, new_price) = sim.sell_to_price(target_price)?;
    let prof_after = profitability(sim.prediction, new_price);
    profitability_level_reached(prof_after, target_prof).then_some(amount.max(0.0))
}

fn mint_active_vs_best_non_active_gap(
    sims: &[PoolSim],
    target_idx: usize,
    skip: &HashSet<usize>,
    candidate_indices: &[usize],
    target_prof: f64,
    amount: f64,
    sim_state: &mut Vec<PoolSim>,
) -> Option<f64> {
    if candidate_indices.is_empty() {
        return None;
    }

    if sim_state.len() != sims.len() {
        sim_state.clear();
        sim_state.extend_from_slice(sims);
    } else {
        sim_state.clone_from_slice(sims);
    }

    if amount > DUST {
        for (i, sim) in sim_state.iter_mut().enumerate() {
            if i == target_idx || skip.contains(&i) {
                continue;
            }
            if let Some((sold, _leg_proceeds, new_leg_price)) = sim.sell_exact(amount)
                && sold > DUST
            {
                sim.set_price(new_leg_price);
            }
        }
    }

    let price_sum: f64 = sim_state.iter().map(|s| s.price()).sum();
    let active_prof = profitability(
        sim_state[target_idx].prediction,
        alt_price(sim_state, target_idx, price_sum),
    );
    if !active_prof.is_finite() {
        return None;
    }

    let mut best_non_active_prof = f64::NEG_INFINITY;
    for &i in candidate_indices {
        let prof = profitability(sim_state[i].prediction, sim_state[i].price());
        if prof > best_non_active_prof {
            best_non_active_prof = prof;
        }
    }
    if !best_non_active_prof.is_finite() {
        return None;
    }

    let crossover_gap = best_non_active_prof - active_prof;
    let target_level_gap = best_non_active_prof - target_prof;
    Some(crossover_gap.min(target_level_gap))
}

fn mint_active_set_boundary_amount(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    target_idx: usize,
    target_prof: f64,
    planned_amount: f64,
    skip: &HashSet<usize>,
    active_prof_cap: f64,
) -> Option<f64> {
    let mut direct_active = vec![false; sims.len()];
    for &(idx, route) in active {
        if route == Route::Direct {
            direct_active[idx] = true;
        }
    }

    let mut candidate_indices: Vec<usize> = Vec::new();
    let mut boundary_amount: Option<f64> = None;
    for (i, sim) in sims.iter().enumerate() {
        if i == target_idx || skip.contains(&i) || direct_active[i] {
            continue;
        }
        candidate_indices.push(i);
        let current_prof = profitability(sim.prediction, sim.price());
        if !profitability_level_reached(current_prof, active_prof_cap)
            && let Some(amount_to_cap) = mint_amount_to_reach_profitability(sim, active_prof_cap)
            && amount_to_cap > DUST
            && amount_to_cap + EPS < planned_amount
        {
            boundary_amount = Some(boundary_amount.map_or(amount_to_cap, |b| b.min(amount_to_cap)));
        }
    }

    let allow_crossover_split = active.len() == 1 && active[0] == (target_idx, Route::Mint);
    if allow_crossover_split && planned_amount > DUST && !candidate_indices.is_empty() {
        let mut sim_state: Vec<PoolSim> = Vec::with_capacity(sims.len());
        if let Some(gap_at_end) = mint_active_vs_best_non_active_gap(
            sims,
            target_idx,
            skip,
            &candidate_indices,
            target_prof,
            planned_amount,
            &mut sim_state,
        ) {
            let gap_tol = EPS * (1.0 + gap_at_end.abs() + active_prof_cap.abs());
            if gap_at_end + gap_tol >= 0.0 {
                let mut lo = 0.0_f64;
                let mut hi = planned_amount;
                let mut converged = true;
                for _ in 0..SOLVE_PROF_ITERS {
                    let mid = 0.5 * (lo + hi);
                    let Some(gap_mid) = mint_active_vs_best_non_active_gap(
                        sims,
                        target_idx,
                        skip,
                        &candidate_indices,
                        target_prof,
                        mid,
                        &mut sim_state,
                    ) else {
                        converged = false;
                        break;
                    };
                    let mid_tol = EPS * (1.0 + gap_mid.abs() + active_prof_cap.abs());
                    if gap_mid + mid_tol >= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                    if (hi - lo).abs() <= EPS * (1.0 + hi.abs()) {
                        break;
                    }
                }
                if converged && hi > DUST && hi + EPS < planned_amount {
                    boundary_amount = Some(boundary_amount.map_or(hi, |b| b.min(hi)));
                }
            }
        }
    }

    boundary_amount
}

#[cfg(test)]
pub(super) fn plan_active_routes(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    target_prof: f64,
    skip: &HashSet<usize>,
) -> Option<Vec<PlannedRoute>> {
    let mut sim_state: Vec<PoolSim> = Vec::with_capacity(sims.len());
    plan_active_routes_with_scratch(sims, active, target_prof, skip, &mut sim_state, None)
}

pub(super) fn plan_active_routes_with_scratch(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    target_prof: f64,
    skip: &HashSet<usize>,
    sim_state: &mut Vec<PoolSim>,
    active_prof_cap: Option<f64>,
) -> Option<Vec<PlannedRoute>> {
    if sim_state.len() != sims.len() {
        sim_state.clear();
        sim_state.extend_from_slice(sims);
    } else {
        sim_state.clone_from_slice(sims);
    }

    let mut price_sum: f64 = sim_state.iter().map(|s| s.price()).sum();
    let mut plan: Vec<PlannedRoute> = Vec::with_capacity(active.len());

    for (idx, route) in execution_order(active) {
        let estimate = cost_for_route(sim_state, idx, route, target_prof, skip, price_sum)?;
        let mut amount = estimate.amount;
        let new_price = estimate.new_price;
        let mut active_set_boundary_hit = false;

        let (actual_cost, applied_new_price) = match route {
            Route::Direct => {
                let np = new_price?;
                sim_state[idx].set_price(np);
                (estimate.cash_cost, Some(np))
            }
            Route::Mint => {
                if let Some(cap_prof) = active_prof_cap
                    && let Some(boundary_amount) = mint_active_set_boundary_amount(
                        sim_state,
                        active,
                        idx,
                        target_prof,
                        amount,
                        skip,
                        cap_prof,
                    )
                {
                    amount = boundary_amount;
                    active_set_boundary_hit = true;
                }
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
                            sim.set_price(new_leg_price);
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
            active_set_boundary_hit,
        });

        price_sum = sim_state.iter().map(|s| s.price()).sum();
        if active_set_boundary_hit {
            // Split the current step at the first join boundary. Waterfall will
            // promote newly-equalized outcomes, refresh skip/ranking, then continue.
            break;
        }
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
        plan_active_routes_with_scratch(sims, active, prof, skip, &mut sim_state, None)
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
