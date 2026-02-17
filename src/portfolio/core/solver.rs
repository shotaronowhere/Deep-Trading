use std::collections::HashSet;

use super::sim::{DUST, EPS, FEE_FACTOR, NEWTON_ITERS, PoolSim, alt_price, target_price_for_prof};

#[derive(Debug, Clone, Copy)]
struct MintLegParam {
    price: f64,
    kappa: f64,
    cap: f64,
    prediction: f64,
}

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

fn mint_reachability_bounds(params: &[MintLegParam]) -> (f64, f64, f64) {
    let g0: f64 = params.iter().map(|p| p.price).sum();
    let g_cap: f64 = params
        .iter()
        .map(|p| {
            let d = 1.0 + p.cap * p.kappa;
            p.price / (d * d)
        })
        .sum();
    let g_tol = EPS * (1.0 + g0.abs() + g_cap.abs());
    (g0, g_cap, g_tol)
}

fn solve_mint_amount_newton(
    params: &[MintLegParam],
    target_price: f64,
    current_alt: f64,
    rhs: f64,
    g_cap: f64,
    g_tol: f64,
) -> f64 {
    // Warm start: first Newton step from m=0 gives m = (g(0) - rhs) / (-g'(0))
    // = (tp - current_alt) / (2 × Σ Pⱼκⱼ), saving one iteration.
    let sum_pk: f64 = params.iter().map(|p| p.price * p.kappa).sum();
    let m_upper: f64 = params.iter().map(|p| p.cap).fold(0.0, f64::max);
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
    let (_g0, g_cap, g_tol) = mint_reachability_bounds(&params);
    let effective_rhs = if rhs < g_cap - g_tol { g_cap } else { rhs };

    let m = solve_mint_amount_newton(&params, tp, current_alt, effective_rhs, g_cap, g_tol);

    if m < DUST {
        return Some((0.0, 0.0, 0.0, 0.0));
    }

    let (cash_cost, value_cost, d_cost_d_pi) = mint_cost_from_amount(&params, m, tp, target_prof);

    Some((cash_cost, value_cost, m, d_cost_d_pi))
}
