use super::bundle::{BundleDirectEstimate, BundleMintEstimate};
use super::sim::{DUST, EPS, PoolSim, target_price_for_prof};
#[cfg(test)]
use super::sim::{FEE_FACTOR, NEWTON_ITERS, alt_price};
#[cfg(test)]
use std::collections::HashSet;

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct MintLegParam {
    price: f64,
    kappa: f64,
    cap: f64,
    prediction: f64,
}

pub(super) fn direct_bundle_cost_to_prof(
    sims: &[PoolSim],
    bundle_members: &[usize],
    target_prof: f64,
) -> Option<BundleDirectEstimate> {
    let mut cash_cost = 0.0_f64;
    let mut member_plans = Vec::with_capacity(bundle_members.len());
    for &idx in bundle_members {
        let target_price = target_price_for_prof(sims[idx].prediction, target_prof);
        let (cost, amount, new_price) = sims[idx].cost_to_price(target_price)?;
        if !cost.is_finite() || !amount.is_finite() || !new_price.is_finite() {
            return None;
        }
        cash_cost += cost;
        member_plans.push((idx, amount, cost, new_price));
    }
    cash_cost.is_finite().then_some(BundleDirectEstimate {
        cash_cost,
        member_plans,
    })
}

fn mint_bundle_boundary_amount(
    sims: &[PoolSim],
    bundle_members: &[usize],
    target_prof: f64,
) -> Option<f64> {
    let mut boundary_amount: Option<f64> = None;
    for (idx, sim) in sims.iter().enumerate() {
        if bundle_members.contains(&idx) {
            continue;
        }
        let frontier_price = target_price_for_prof(sim.prediction, target_prof);
        let tol = EPS * (1.0 + sim.price().abs().max(frontier_price.abs()));
        if sim.price() <= frontier_price + tol {
            continue;
        }
        let (sold, _, _) = sim.sell_to_price(frontier_price)?;
        if sold <= DUST {
            continue;
        }
        boundary_amount = Some(boundary_amount.map_or(sold, |current| current.min(sold)));
    }
    boundary_amount
}

pub(super) fn mint_bundle_cost_for_amount(
    sims: &[PoolSim],
    bundle_members: &[usize],
    target_prof: f64,
    mint_amount: f64,
) -> Option<BundleMintEstimate> {
    if mint_amount <= DUST {
        return Some(BundleMintEstimate {
            cash_cost: 0.0,
            mint_amount: 0.0,
            sell_leg_plans: Vec::new(),
        });
    }

    let mut sell_leg_plans = Vec::new();
    let mut proceeds = 0.0_f64;
    for (idx, sim) in sims.iter().enumerate() {
        if bundle_members.contains(&idx) {
            continue;
        }
        let frontier_price = target_price_for_prof(sim.prediction, target_prof);
        let tol = EPS * (1.0 + sim.price().abs().max(frontier_price.abs()));
        if sim.price() <= frontier_price + tol {
            continue;
        }
        let (sold, leg_proceeds, new_price) = sim.sell_exact(mint_amount)?;
        if sold <= DUST {
            continue;
        }
        let frontier_tol = EPS * (1.0 + new_price.abs().max(frontier_price.abs()));
        if new_price + frontier_tol < frontier_price {
            return None;
        }
        if !leg_proceeds.is_finite() || !new_price.is_finite() {
            return None;
        }
        proceeds += leg_proceeds;
        sell_leg_plans.push((idx, sold, leg_proceeds, new_price));
    }

    if sell_leg_plans.is_empty() {
        return None;
    }

    let cash_cost = mint_amount - proceeds;
    cash_cost.is_finite().then_some(BundleMintEstimate {
        cash_cost,
        mint_amount,
        sell_leg_plans,
    })
}

pub(super) fn mint_bundle_cost_to_prof(
    sims: &[PoolSim],
    bundle_members: &[usize],
    target_prof: f64,
) -> Option<BundleMintEstimate> {
    let mint_amount = mint_bundle_boundary_amount(sims, bundle_members, target_prof)?;
    mint_bundle_cost_for_amount(sims, bundle_members, target_prof, mint_amount)
}

#[cfg(test)]
fn mint_rhs_for_target(
    sims: &[PoolSim],
    target_idx: usize,
    target_price: f64,
    skip: &HashSet<usize>,
) -> Option<f64> {
    let skip_price_sum: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx && skip.contains(i))
        .map(|(_, s)| s.price())
        .sum();
    let rhs = (1.0 - target_price) - skip_price_sum;
    (rhs > 0.0).then_some(rhs)
}

#[cfg(test)]
fn build_mint_leg_params(
    sims: &[PoolSim],
    target_idx: usize,
    skip: &HashSet<usize>,
) -> Vec<MintLegParam> {
    sims.iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx && !skip.contains(i))
        .map(|(_, sim)| MintLegParam {
            price: sim.price(),
            kappa: sim.kappa(),
            cap: sim.max_sell_tokens(),
            prediction: sim.prediction,
        })
        .collect()
}

#[cfg(test)]
fn mint_reachability_bounds(params: &[MintLegParam]) -> (f64, f64, f64, f64) {
    let g0: f64 = params.iter().map(|p| p.price).sum();
    let m_cap: f64 = params.iter().map(|p| p.cap).fold(f64::INFINITY, f64::min);
    if !m_cap.is_finite() || m_cap <= DUST {
        let g_tol = EPS * (1.0 + g0.abs());
        return (g0, g0, g_tol, 0.0);
    }
    let g_cap: f64 = params
        .iter()
        .map(|p| {
            let d = 1.0 + m_cap * p.kappa;
            p.price / (d * d)
        })
        .sum();
    let g_tol = EPS * (1.0 + g0.abs() + g_cap.abs());
    (g0, g_cap, g_tol, m_cap)
}

#[cfg(test)]
fn solve_mint_amount_newton(
    params: &[MintLegParam],
    target_price: f64,
    current_alt: f64,
    rhs: f64,
    g_cap: f64,
    g_tol: f64,
    m_upper: f64,
) -> f64 {
    // Warm start: first Newton step from m=0 gives m = (g(0) - rhs) / (-g'(0))
    // = (tp - current_alt) / (2 × Σ Pⱼκⱼ), saving one iteration.
    let sum_pk: f64 = params.iter().map(|p| p.price * p.kappa).sum();
    let mut m = if sum_pk > 1e-30 {
        ((target_price - current_alt) / (2.0 * sum_pk)).max(0.0)
    } else {
        0.0
    };
    if rhs <= g_cap + g_tol || m > m_upper {
        m = m_upper;
    }

    for _ in 0..NEWTON_ITERS {
        let mut g = 0.0_f64;
        let mut gp = 0.0_f64;
        for p in params {
            let me = m.min(p.cap);
            let d = 1.0 + me * p.kappa;
            let d2 = d * d;
            g += p.price / d2;
            if m < p.cap {
                let d3 = d2 * d;
                gp += -2.0 * p.price * p.kappa / d3;
            }
        }
        if gp.abs() < 1e-30 {
            break;
        }

        let step = (g - rhs) / gp;
        m -= step;
        if m < 0.0 {
            m = 0.0;
        } else if m > m_upper {
            m = m_upper;
        }
        if step.abs() < EPS {
            break;
        }
    }

    m
}

#[cfg(test)]
fn mint_cost_from_amount(
    params: &[MintLegParam],
    mint_amount: f64,
    target_price: f64,
    target_prof: f64,
) -> (f64, f64, f64) {
    // Net cost and analytical derivative in one pass.
    // d(cash_cost)/dπ = d(cash_cost)/dm × dm/dπ
    // dm/dπ = P_target / ((1+π) × g'(m))  [implicit function theorem]
    // d(cash_cost)/dm = 1 - (1-f) × Σⱼ∈uncapped Pⱼ(m)
    let mut sum_marginal = 0.0_f64;
    let mut gp_final = 0.0_f64;
    let mut unsold_value = 0.0_f64;
    let total_proceeds: f64 = params
        .iter()
        .map(|p| {
            let me = mint_amount.min(p.cap);
            let d = 1.0 + me * p.kappa;
            if mint_amount < p.cap {
                let d2 = d * d;
                let d3 = d2 * d;
                sum_marginal += p.price / d2;
                gp_final += -2.0 * p.price * p.kappa / d3;
            } else {
                unsold_value += p.prediction * (mint_amount - me);
            }
            p.price * me * FEE_FACTOR / d
        })
        .sum();
    let cash_cost = mint_amount - total_proceeds;
    let value_cost = cash_cost - unsold_value;

    let d_cost_d_pi = if gp_final.abs() > 1e-30 {
        let dm_d_pi = target_price / ((1.0 + target_prof) * gp_final);
        let d_cost_d_m = 1.0 - FEE_FACTOR * sum_marginal;
        d_cost_d_m * dm_d_pi
    } else {
        0.0
    };

    (cash_cost, value_cost, d_cost_d_pi)
}

#[cfg(test)]
/// For the mint route, compute cost to bring an outcome's profitability to `target_prof`.
/// Uses Newton's method to find mint amount where alt price = target price.
/// Returns (cash_cost, value_cost, mint_amount, d_cash_cost_d_pi).
/// cash_cost = actual sUSD spent; value_cost = cash_cost minus expected value of unsold tokens.
pub(super) fn mint_cost_to_prof(
    sims: &[PoolSim],
    target_idx: usize,
    target_prof: f64,
    skip: &HashSet<usize>,
    price_sum: f64,
) -> Option<(f64, f64, f64, f64)> {
    let tp = target_price_for_prof(sims[target_idx].prediction, target_prof);
    let current_alt = alt_price(sims, target_idx, price_sum);
    if current_alt >= tp {
        return Some((0.0, 0.0, 0.0, 0.0));
    }

    // Newton's method: solve g(m) = rhs where g(m) = Σⱼ pⱼ/(1+m×κⱼ)² (non-skip pools only).
    // Correct rhs accounts for skip pool prices frozen at P⁰_j:
    // rhs = (1 - tp) - Σ_{j∈skip, j≠target} P⁰_j
    let rhs = mint_rhs_for_target(sims, target_idx, tp, skip)?;

    let params = build_mint_leg_params(sims, target_idx, skip);
    if params.is_empty() {
        return None;
    }

    // Reachability guard:
    // g(m) decreases from g0 (m=0) to g_cap (all legs at cap). If rhs is below g_cap,
    // target alt price is unreachable with available liquidity. Clamp to g_cap so
    // we return the executable saturated mint amount instead of dropping the route.
    let (_g0, g_cap, g_tol, m_upper) = mint_reachability_bounds(&params);
    if m_upper <= DUST {
        return Some((0.0, 0.0, 0.0, 0.0));
    }
    let effective_rhs = if rhs < g_cap - g_tol { g_cap } else { rhs };

    let m = solve_mint_amount_newton(
        &params,
        tp,
        current_alt,
        effective_rhs,
        g_cap,
        g_tol,
        m_upper,
    );

    if m < DUST {
        return Some((0.0, 0.0, 0.0, 0.0));
    }

    let (cash_cost, value_cost, d_cost_d_pi) = mint_cost_from_amount(&params, m, tp, target_prof);

    Some((cash_cost, value_cost, m, d_cost_d_pi))
}
