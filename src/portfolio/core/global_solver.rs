use std::collections::HashMap;

use crate::markets::MarketData;
use crate::pools::Slot0Result;

use super::Action;
use super::diagnostics::replay_actions_to_portfolio_state;
use super::merge::action_contract_pair;
use super::sim::{DUST, EPS, FEE_FACTOR, PoolSim, build_sims, profitability};
use super::trading::{ExecutionState, portfolio_expected_value};
use super::waterfall::waterfall;
use super::types::{BalanceMap, apply_actions_to_sim_balances, lookup_balance};

const DOMAIN_EPS: f64 = 1e-12;
const DEFAULT_BARRIER_SHIFT: f64 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalCandidateInvalidReason {
    ProjectionUnavailable,
    ReplayCashMismatch,
    ReplayHoldingsMismatch,
    ProjectedGradientTooLarge,
    NonFiniteSolveState,
    MintWithoutSellFlow,
    BudgetEpsilonViolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalOptimizer {
    DiagonalProjectedNewton,
    LbfgsbProjected,
    DualDecompositionPrototype,
    DualRouterV1,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalSolveConfig {
    pub optimizer: GlobalOptimizer,
    pub max_iters: usize,
    pub pg_tol: f64,
    pub armijo_c1: f64,
    pub backtrack_beta: f64,
    pub max_line_search_trials: usize,
    pub line_search_rescue_trials: usize,
    pub line_search_rescue_min_decrease: f64,
    pub hess_floor: f64,
    pub lbfgs_history: usize,
    pub lbfgs_min_curvature: f64,
    pub active_set_eps: f64,
    pub barrier_mu_cash: f64,
    pub barrier_mu_hold: f64,
    pub barrier_shift: f64,
    pub theta_l2_reg: f64,
    pub dual_outer_iters: usize,
    pub dual_lambda_step: f64,
    pub dual_lambda_decay: f64,
    pub dual_theta_tolerance: f64,
    pub dual_router_max_iters: usize,
    pub dual_router_pg_tol: f64,
    pub dual_router_lbfgs_history: usize,
    pub dual_router_primal_restore_iters: usize,
    pub dual_router_primal_residual_tol: f64,
    pub dual_router_price_floor: f64,
    pub buy_sell_churn_reg: f64,
    pub enable_route_refinement: bool,
    pub solver_budget_eps: f64,
    pub zero_trade_band_eps: f64,
}

impl Default for GlobalSolveConfig {
    fn default() -> Self {
        Self {
            optimizer: GlobalOptimizer::LbfgsbProjected,
            max_iters: 1024,
            pg_tol: 1e-5,
            armijo_c1: 1e-4,
            backtrack_beta: 0.5,
            max_line_search_trials: 64,
            line_search_rescue_trials: 16,
            line_search_rescue_min_decrease: 1e-12,
            hess_floor: 1e-8,
            lbfgs_history: 15,
            lbfgs_min_curvature: 1e-12,
            active_set_eps: 1e-9,
            barrier_mu_cash: 1e-8,
            barrier_mu_hold: 1e-8,
            barrier_shift: DEFAULT_BARRIER_SHIFT,
            theta_l2_reg: 1e-9,
            dual_outer_iters: 8,
            dual_lambda_step: 1e-3,
            dual_lambda_decay: 0.7,
            dual_theta_tolerance: 1e-6,
            dual_router_max_iters: 24,
            dual_router_pg_tol: 1e-5,
            dual_router_lbfgs_history: 10,
            dual_router_primal_restore_iters: 8,
            dual_router_primal_residual_tol: 1e-8,
            dual_router_price_floor: 1e-6,
            buy_sell_churn_reg: 1e-10,
            enable_route_refinement: true,
            solver_budget_eps: 1e-8,
            zero_trade_band_eps: 1e-8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalSolveResult {
    pub optimizer: GlobalOptimizer,
    pub direct_buys: Vec<f64>,
    pub direct_sells: Vec<f64>,
    pub net_complete_set: f64,
    pub dual_residual_norm: f64,
    pub primal_restore_iters: usize,
    pub projected_grad_norm: f64,
    pub coupled_residual: f64,
    pub converged: bool,
    pub outer_iters: usize,
    pub line_search_trials: usize,
    pub line_search_accepts: usize,
    pub line_search_invalid_evals: usize,
    pub line_search_rescue_attempts: usize,
    pub line_search_rescue_accepts: usize,
    pub feasibility_repairs: usize,
    pub feasibility_hold_clamps: usize,
    pub feasibility_cash_scales: usize,
    pub active_dims: usize,
    pub curvature_skips: usize,
}

#[derive(Debug, Clone)]
pub(super) struct GlobalCandidatePlan {
    pub(super) actions: Vec<Action>,
    pub(super) solve: GlobalSolveResult,
    pub(super) replay_cash_delta: f64,
    pub(super) replay_holdings_delta: f64,
    pub(super) candidate_valid: bool,
    pub(super) invalid_reason: Option<GlobalCandidateInvalidReason>,
}

#[derive(Debug, Clone)]
pub(super) struct BoxBounds {
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
}

#[derive(Debug, Clone)]
struct EvalState {
    f: f64,
    grad: Vec<f64>,
    hdiag: Vec<f64>,
    cash: f64,
    holds: Vec<f64>,
    cost_sum: f64,
}

#[derive(Debug, Clone)]
struct SolveTrace {
    optimizer: GlobalOptimizer,
    x: Vec<f64>,
    eval: EvalState,
    projected_grad_norm: f64,
    coupled_residual: f64,
    converged: bool,
    objective_trace: Vec<f64>,
    outer_iters: usize,
    line_search_trials: usize,
    line_search_accepts: usize,
    line_search_invalid_evals: usize,
    line_search_rescue_attempts: usize,
    line_search_rescue_accepts: usize,
    feasibility_repairs: usize,
    feasibility_hold_clamps: usize,
    feasibility_cash_scales: usize,
    active_dims: usize,
    curvature_skips: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct FeasibilityRepairStats {
    repair_passes: usize,
    hold_clamps: usize,
    cash_scalings: usize,
}

#[derive(Debug, Clone)]
struct LbfgsHistoryPair {
    s: Vec<f64>,
    y: Vec<f64>,
    rho: f64,
}

#[derive(Debug, Default, Clone)]
struct LbfgsState {
    pairs: Vec<LbfgsHistoryPair>,
    curvature_skips: usize,
}

impl FeasibilityRepairStats {
    fn merge(&mut self, other: Self) {
        self.repair_passes += other.repair_passes;
        self.hold_clamps += other.hold_clamps;
        self.cash_scalings += other.cash_scalings;
    }
}

impl LbfgsState {
    fn clear(&mut self) {
        self.pairs.clear();
    }

    fn update(&mut self, s: Vec<f64>, y: Vec<f64>, cfg: GlobalSolveConfig) {
        if cfg.lbfgs_history == 0 {
            self.clear();
            return;
        }

        let s_dot_y = dot(&s, &y);
        let s_norm = s.iter().map(|v| v * v).sum::<f64>().sqrt();
        let y_norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let curvature_floor = cfg.lbfgs_min_curvature * s_norm * y_norm;

        if !s_dot_y.is_finite()
            || !s_norm.is_finite()
            || !y_norm.is_finite()
            || s_dot_y <= curvature_floor
            || s_dot_y <= 0.0
        {
            self.curvature_skips += 1;
            return;
        }

        let rho = 1.0 / s_dot_y;
        self.pairs.push(LbfgsHistoryPair { s, y, rho });
        if self.pairs.len() > cfg.lbfgs_history {
            let overflow = self.pairs.len() - cfg.lbfgs_history;
            self.pairs.drain(0..overflow);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dot_masked(a: &[f64], b: &[f64], free: &[bool]) -> f64 {
    a.iter()
        .zip(b.iter())
        .zip(free.iter())
        .map(|((&ai, &bi), &is_free)| if is_free { ai * bi } else { 0.0 })
        .sum()
}

fn project_box(x: &mut [f64], bounds: &BoxBounds) {
    for ((xi, lo), hi) in x
        .iter_mut()
        .zip(bounds.lower.iter())
        .zip(bounds.upper.iter())
    {
        *xi = xi.clamp(*lo, *hi);
    }
}

fn projected_grad_component(x: f64, g: f64, lo: f64, hi: f64) -> f64 {
    if (hi - lo).abs() <= EPS {
        0.0
    } else if (x - lo).abs() <= EPS * (1.0 + lo.abs()) {
        g.min(0.0)
    } else if (hi - x).abs() <= EPS * (1.0 + hi.abs()) {
        g.max(0.0)
    } else {
        g
    }
}

fn projected_grad_norm(x: &[f64], g: &[f64], bounds: &BoxBounds) -> f64 {
    x.iter()
        .zip(g.iter())
        .zip(bounds.lower.iter().zip(bounds.upper.iter()))
        .map(|((&xj, &gj), (&lo, &hi))| projected_grad_component(xj, gj, lo, hi).abs())
        .fold(0.0, f64::max)
}

fn build_free_mask(x: &[f64], grad: &[f64], bounds: &BoxBounds, active_set_eps: f64) -> Vec<bool> {
    x.iter()
        .zip(grad.iter())
        .zip(bounds.lower.iter().zip(bounds.upper.iter()))
        .map(|((&xj, &gj), (&lo, &hi))| {
            if (hi - lo).abs() <= EPS {
                return false;
            }
            let eps = active_set_eps.max(0.0);
            let lo_tol = eps * (1.0 + lo.abs());
            let hi_tol = eps * (1.0 + hi.abs());
            let near_lower = xj <= lo + lo_tol;
            let near_upper = xj >= hi - hi_tol;
            let pinned_lower = near_lower && gj > eps;
            let pinned_upper = near_upper && gj < -eps;
            !(pinned_lower || pinned_upper)
        })
        .collect()
}

fn scaled_negative_gradient_direction(
    grad: &[f64],
    hdiag: &[f64],
    free_mask: &[bool],
    cfg: GlobalSolveConfig,
) -> Vec<f64> {
    grad.iter()
        .zip(hdiag.iter())
        .zip(free_mask.iter())
        .map(|((&g, &h), &is_free)| {
            if is_free {
                -g / h.max(cfg.hess_floor)
            } else {
                0.0
            }
        })
        .collect()
}

fn compute_lbfgs_direction(
    grad: &[f64],
    hdiag: &[f64],
    free_mask: &[bool],
    lbfgs: &LbfgsState,
    cfg: GlobalSolveConfig,
) -> Vec<f64> {
    let mut q = vec![0.0_f64; grad.len()];
    for i in 0..grad.len() {
        if free_mask.get(i).copied().unwrap_or(false) {
            q[i] = grad[i];
        }
    }

    let mut alphas: Vec<f64> = Vec::with_capacity(lbfgs.pairs.len());
    for pair in lbfgs.pairs.iter().rev() {
        let alpha = pair.rho * dot_masked(&pair.s, &q, free_mask);
        alphas.push(alpha);
        for i in 0..q.len() {
            if free_mask[i] {
                q[i] -= alpha * pair.y[i];
            }
        }
    }

    let mut r = vec![0.0_f64; grad.len()];
    for i in 0..r.len() {
        if free_mask[i] {
            r[i] = q[i] / hdiag[i].max(cfg.hess_floor);
        }
    }

    for (pair, alpha) in lbfgs.pairs.iter().zip(alphas.into_iter().rev()) {
        let beta = pair.rho * dot_masked(&pair.y, &r, free_mask);
        for i in 0..r.len() {
            if free_mask[i] {
                r[i] += pair.s[i] * (alpha - beta);
            }
        }
    }

    for ri in &mut r {
        *ri = -*ri;
    }
    r
}

fn within_execution_tol(target: f64, actual: f64, eps: f64) -> bool {
    (target - actual).abs() <= eps * (1.0 + target.abs().max(actual.abs())) + 1e-12
}

fn buy_cost_and_derivatives(sim: &PoolSim, buy: f64) -> Option<(f64, f64, f64)> {
    let price = sim.price();
    if !price.is_finite() || price <= 0.0 {
        return None;
    }
    if buy < -DOMAIN_EPS {
        return None;
    }

    let lambda = sim.lambda();
    if !lambda.is_finite() || lambda < 0.0 {
        return None;
    }
    let d = 1.0 - buy.max(0.0) * lambda;
    if !d.is_finite() || d <= 1e-15 {
        return None;
    }

    let cost = buy.max(0.0) * price / (FEE_FACTOR * d);
    let d1 = price / (FEE_FACTOR * d * d);
    let d2 = 2.0 * price * lambda / (FEE_FACTOR * d * d * d);

    (cost.is_finite() && d1.is_finite() && d2.is_finite()).then_some((cost, d1, d2))
}

fn sell_proceeds_and_derivatives(sim: &PoolSim, sell: f64) -> Option<(f64, f64, f64)> {
    let price = sim.price();
    if !price.is_finite() || price <= 0.0 {
        return None;
    }
    if sell < -DOMAIN_EPS {
        return None;
    }

    let kappa = sim.kappa();
    if !kappa.is_finite() || kappa < 0.0 {
        return None;
    }
    let d = 1.0 + sell.max(0.0) * kappa;
    if !d.is_finite() || d <= 1e-15 {
        return None;
    }

    let proceeds = price * sell.max(0.0) * FEE_FACTOR / d;
    let d1 = price * FEE_FACTOR / (d * d);
    let d2 = -2.0 * price * FEE_FACTOR * kappa / (d * d * d);

    (proceeds.is_finite() && d1.is_finite() && d2.is_finite()).then_some((proceeds, d1, d2))
}

fn evaluate_objective(
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    pred_sum: f64,
    x: &[f64],
    cfg: GlobalSolveConfig,
) -> Option<EvalState> {
    let n = sims.len();
    if x.len() != 2 * n + 1 || holdings0.len() != n {
        return None;
    }

    let theta = x[2 * n];
    let barrier_shift = cfg.barrier_shift.max(DOMAIN_EPS);

    let mut holds = Vec::with_capacity(n);
    let mut buy_cost_sum = 0.0_f64;
    let mut sell_proceeds_sum = 0.0_f64;
    let mut holdings_ev = 0.0_f64;
    let mut sum_inv_hold = 0.0_f64;
    let mut sum_inv_hold_sq = 0.0_f64;
    let mut buy_d1 = vec![0.0_f64; n];
    let mut buy_d2 = vec![0.0_f64; n];
    let mut sell_d1 = vec![0.0_f64; n];
    let mut sell_d2 = vec![0.0_f64; n];

    for i in 0..n {
        let buy = x[i].max(0.0);
        let sell = x[n + i].max(0.0);
        let hold = holdings0[i] + buy - sell + theta;
        if !hold.is_finite() || hold < -DOMAIN_EPS {
            return None;
        }

        let (buy_cost, d1_buy, d2_buy) = buy_cost_and_derivatives(&sims[i], buy)?;
        let (sell_proceeds, d1_sell, d2_sell) = sell_proceeds_and_derivatives(&sims[i], sell)?;
        let hold_bar = hold.max(0.0) + barrier_shift;

        holds.push(hold);
        buy_cost_sum += buy_cost;
        sell_proceeds_sum += sell_proceeds;
        holdings_ev += hold * sims[i].prediction;
        sum_inv_hold += 1.0 / hold_bar;
        sum_inv_hold_sq += 1.0 / (hold_bar * hold_bar);
        buy_d1[i] = d1_buy;
        buy_d2[i] = d2_buy;
        sell_d1[i] = d1_sell;
        sell_d2[i] = d2_sell;
    }

    if !buy_cost_sum.is_finite() || !sell_proceeds_sum.is_finite() || !holdings_ev.is_finite() {
        return None;
    }

    let cash = cash0 - theta - buy_cost_sum + sell_proceeds_sum;
    if !cash.is_finite() {
        return None;
    }
    let cash_margin = cash - cfg.solver_budget_eps;
    if cash_margin < -DOMAIN_EPS {
        return None;
    }
    let cash_bar = cash_margin.max(0.0) + barrier_shift;

    let ev = cash + holdings_ev - cfg.theta_l2_reg * theta * theta;
    let mut f = -ev - cfg.barrier_mu_cash * cash_bar.ln();
    for hold in &holds {
        let hold_bar = hold.max(0.0) + barrier_shift;
        f -= cfg.barrier_mu_hold * hold_bar.ln();
    }
    if !f.is_finite() {
        return None;
    }

    let mut grad = vec![0.0_f64; 2 * n + 1];
    let mut hdiag = vec![cfg.hess_floor; 2 * n + 1];

    for i in 0..n {
        let hold = holds[i];
        let pred = sims[i].prediction;
        let hold_bar = hold.max(0.0) + barrier_shift;

        let d1_buy = buy_d1[i];
        let d2_buy = buy_d2[i];
        let g_buy = d1_buy - pred + (cfg.barrier_mu_cash * d1_buy / cash_bar)
            - (cfg.barrier_mu_hold / hold_bar);
        let h_buy = d2_buy * (1.0 + cfg.barrier_mu_cash / cash_bar)
            + (cfg.barrier_mu_cash * d1_buy * d1_buy / (cash_bar * cash_bar))
            + (cfg.barrier_mu_hold / (hold_bar * hold_bar));
        grad[i] = g_buy;
        hdiag[i] = h_buy.max(cfg.hess_floor);

        let d1_sell = sell_d1[i];
        let d2_sell = sell_d2[i];
        let g_sell = pred - d1_sell - (cfg.barrier_mu_cash * d1_sell / cash_bar)
            + (cfg.barrier_mu_hold / hold_bar);
        let h_sell = (-d2_sell) * (1.0 + cfg.barrier_mu_cash / cash_bar)
            + (cfg.barrier_mu_cash * d1_sell * d1_sell / (cash_bar * cash_bar))
            + (cfg.barrier_mu_hold / (hold_bar * hold_bar));
        grad[n + i] = g_sell;
        hdiag[n + i] = h_sell.max(cfg.hess_floor);
    }

    let g_theta =
        1.0 - pred_sum + 2.0 * cfg.theta_l2_reg * theta + (cfg.barrier_mu_cash / cash_bar)
            - (cfg.barrier_mu_hold * sum_inv_hold);
    let h_theta = 2.0 * cfg.theta_l2_reg
        + (cfg.barrier_mu_cash / (cash_bar * cash_bar))
        + (cfg.barrier_mu_hold * sum_inv_hold_sq);
    grad[2 * n] = g_theta;
    hdiag[2 * n] = h_theta.max(cfg.hess_floor);

    if grad.iter().any(|g| !g.is_finite()) || hdiag.iter().any(|h| !h.is_finite()) {
        return None;
    }

    Some(EvalState {
        f,
        grad,
        hdiag,
        cash,
        holds,
        cost_sum: buy_cost_sum - sell_proceeds_sum,
    })
}

fn build_bounds(
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    allow_complete_set: bool,
) -> BoxBounds {
    let n = sims.len();
    let mut lower = vec![0.0_f64; 2 * n + 1];
    let mut upper = vec![0.0_f64; 2 * n + 1];
    let mut buy_caps = vec![0.0_f64; n];
    let mut sell_caps = vec![0.0_f64; n];
    let mut depth_budget = 0.0_f64;

    for i in 0..n {
        let buy_cap = sims[i].max_buy_tokens().max(0.0);
        let sell_cap = sims[i].max_sell_tokens().max(0.0);
        buy_caps[i] = buy_cap;
        sell_caps[i] = sell_cap;

        let local_budget = buy_cap + sell_cap + holdings0[i].max(0.0);
        if local_budget.is_finite() {
            depth_budget += local_budget;
        }
    }

    let mut theta_pos_cap = if allow_complete_set {
        let cap = cash0.max(0.0) + depth_budget.max(0.0);
        if cap.is_finite() { cap } else { 0.0 }
    } else {
        0.0
    };
    if !theta_pos_cap.is_finite() {
        theta_pos_cap = 0.0;
    }
    theta_pos_cap = theta_pos_cap.max(0.0);

    let mut theta_neg_cap = 0.0_f64;
    if allow_complete_set {
        for i in 0..n {
            let candidate = holdings0[i].max(0.0) + buy_caps[i] + theta_pos_cap;
            if i == 0 {
                theta_neg_cap = candidate;
            } else {
                theta_neg_cap = theta_neg_cap.min(candidate);
            }
        }
        if !theta_neg_cap.is_finite() {
            theta_neg_cap = 0.0;
        }
        theta_neg_cap = theta_neg_cap.max(0.0);
    }

    for i in 0..n {
        lower[i] = 0.0;
        upper[i] = buy_caps[i];

        lower[n + i] = 0.0;
        let inventory_cap = holdings0[i].max(0.0) + theta_pos_cap;
        upper[n + i] = sell_caps[i].min(inventory_cap.max(0.0));
    }

    lower[2 * n] = -theta_neg_cap;
    upper[2 * n] = theta_pos_cap;

    BoxBounds { lower, upper }
}

fn interior_start(
    cash0: f64,
    n: usize,
    bounds: &BoxBounds,
    warm_start: Option<&[f64]>,
) -> Option<Vec<f64>> {
    if cash0 < -DOMAIN_EPS || !cash0.is_finite() {
        return None;
    }

    let mut x = if let Some(seed) = warm_start {
        if seed.len() != 2 * n + 1 {
            return None;
        }
        seed.to_vec()
    } else {
        vec![0.0_f64; 2 * n + 1]
    };
    project_box(&mut x, bounds);
    Some(x)
}

fn apply_zero_trade_clamp(
    x: &[f64],
    grad: &mut [f64],
    bounds: &BoxBounds,
    cfg: GlobalSolveConfig,
    n: usize,
) {
    for i in 0..n {
        let buy_idx = i;
        let sell_idx = n + i;

        let buy_at_zero = x[buy_idx] <= bounds.lower[buy_idx] + cfg.zero_trade_band_eps;
        let sell_at_zero = x[sell_idx] <= bounds.lower[sell_idx] + cfg.zero_trade_band_eps;

        if buy_at_zero && grad[buy_idx].abs() <= cfg.zero_trade_band_eps {
            grad[buy_idx] = 0.0;
        }
        if sell_at_zero && grad[sell_idx].abs() <= cfg.zero_trade_band_eps {
            grad[sell_idx] = 0.0;
        }

        if buy_at_zero
            && sell_at_zero
            && grad[buy_idx] >= -cfg.zero_trade_band_eps
            && grad[sell_idx] >= -cfg.zero_trade_band_eps
        {
            grad[buy_idx] = 0.0;
            grad[sell_idx] = 0.0;
        }
    }
}

fn compute_cash_state(sims: &[PoolSim], cash0: f64, x: &[f64]) -> Option<f64> {
    let n = sims.len();
    if x.len() != 2 * n + 1 {
        return None;
    }

    let theta = x[2 * n];
    let mut buy_cost_sum = 0.0_f64;
    let mut sell_proceeds_sum = 0.0_f64;

    for i in 0..n {
        let buy = x[i].max(0.0);
        let sell = x[n + i].max(0.0);
        let (buy_cost, _, _) = buy_cost_and_derivatives(&sims[i], buy)?;
        let (sell_proceeds, _, _) = sell_proceeds_and_derivatives(&sims[i], sell)?;
        buy_cost_sum += buy_cost;
        sell_proceeds_sum += sell_proceeds;
    }

    let cash = cash0 - theta - buy_cost_sum + sell_proceeds_sum;
    cash.is_finite().then_some(cash)
}

fn coupled_feasibility_residual(
    x: &[f64],
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    cfg: GlobalSolveConfig,
) -> f64 {
    let n = sims.len();
    if x.len() != 2 * n + 1 || holdings0.len() != n {
        return f64::INFINITY;
    }

    let theta = x[2 * n];
    let mut max_violation = 0.0_f64;

    for i in 0..n {
        let buy = x[i].max(0.0);
        let sell = x[n + i].max(0.0);
        let hold = holdings0[i] + buy - sell + theta;
        if !hold.is_finite() {
            return f64::INFINITY;
        }
        max_violation = max_violation.max((-hold).max(0.0));

        let sell_cap = (holdings0[i] + buy + theta).max(0.0);
        let sell_violation = (sell - sell_cap).max(0.0);
        if !sell_violation.is_finite() {
            return f64::INFINITY;
        }
        max_violation = max_violation.max(sell_violation);
    }

    let Some(cash) = compute_cash_state(sims, cash0, x) else {
        return f64::INFINITY;
    };
    let cash_violation = (cfg.solver_budget_eps - cash).max(0.0);
    if !cash_violation.is_finite() {
        return f64::INFINITY;
    }
    max_violation.max(cash_violation)
}

fn restore_coupled_feasibility(
    x: &mut [f64],
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    cfg: GlobalSolveConfig,
) -> FeasibilityRepairStats {
    let n = sims.len();
    let mut stats = FeasibilityRepairStats::default();
    if x.len() != 2 * n + 1 || holdings0.len() != n {
        return stats;
    }

    let cash_target = (10.0 * cfg.solver_budget_eps).max(10.0 * DOMAIN_EPS);
    for _ in 0..4 {
        let mut changed = false;
        let theta = x[2 * n];

        for i in 0..n {
            let buy = x[i].max(0.0);
            let max_sell = (holdings0[i] + buy + theta).max(0.0);
            let sell_violation = x[n + i] - max_sell;
            if sell_violation > 0.0 {
                x[n + i] = max_sell;
                let sell_tol = cfg.solver_budget_eps * (1.0 + x[n + i].abs().max(max_sell.abs()));
                if sell_violation > sell_tol {
                    stats.hold_clamps += 1;
                }
                changed = true;
            }
        }

        let Some(cash_now) = compute_cash_state(sims, cash0, x) else {
            if changed {
                stats.repair_passes += 1;
            }
            break;
        };
        if cash_now + DOMAIN_EPS >= cash_target {
            if changed {
                stats.repair_passes += 1;
            }
            break;
        }

        if (0..n).all(|i| x[i].max(0.0) <= DUST) {
            if changed {
                stats.repair_passes += 1;
            }
            break;
        }

        let mut cash_now = cash_now;
        let mut ranked_buys: Vec<(usize, f64)> = (0..n)
            .filter_map(|i| {
                let buy = x[i].max(0.0);
                if buy <= DUST {
                    return None;
                }
                let edge = buy_cost_and_derivatives(&sims[i], buy)
                    .map(|(_, d1, _)| sims[i].prediction - d1)
                    .unwrap_or(f64::NEG_INFINITY);
                Some((i, edge))
            })
            .collect();
        ranked_buys.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));

        for (idx, _) in ranked_buys {
            if cash_now + DOMAIN_EPS >= cash_target {
                break;
            }

            let buy = x[idx].max(0.0);
            if buy <= DUST {
                continue;
            }

            let mut x_zero = x.to_vec();
            x_zero[idx] = 0.0;
            let Some(cash_zero) = compute_cash_state(sims, cash0, &x_zero) else {
                continue;
            };

            if cash_zero + DOMAIN_EPS < cash_target {
                x[idx] = 0.0;
                cash_now = cash_zero;
                stats.cash_scalings += 1;
                changed = true;
                continue;
            }

            let mut lo = 0.0_f64;
            let mut hi = 1.0_f64;
            let mut best = 0.0_f64;
            for _ in 0..48 {
                let mid = 0.5 * (lo + hi);
                let mut x_trial = x.to_vec();
                x_trial[idx] = buy * mid;

                match compute_cash_state(sims, cash0, &x_trial) {
                    Some(cash_mid) if cash_mid + DOMAIN_EPS >= cash_target => {
                        best = mid;
                        lo = mid;
                    }
                    _ => {
                        hi = mid;
                    }
                }

                if (hi - lo).abs() <= 1e-10 {
                    break;
                }
            }

            if best < 1.0 - 1e-12 {
                x[idx] = buy * best;
                if let Some(cash_best) = compute_cash_state(sims, cash0, x) {
                    cash_now = cash_best;
                }
                stats.cash_scalings += 1;
                changed = true;
            }
        }

        if changed {
            stats.repair_passes += 1;
        } else {
            break;
        }
    }

    stats
}

fn run_projected_newton(
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    bounds: &BoxBounds,
    warm_start: Option<&[f64]>,
    cfg: GlobalSolveConfig,
) -> Option<SolveTrace> {
    let n = sims.len();
    let pred_sum: f64 = sims.iter().map(|s| s.prediction).sum();
    let mut x = interior_start(cash0, n, bounds, warm_start)?;
    let mut lbfgs_state = LbfgsState::default();
    let mut feasibility_stats = FeasibilityRepairStats::default();
    feasibility_stats.merge(restore_coupled_feasibility(
        &mut x, sims, holdings0, cash0, cfg,
    ));
    let mut eval = evaluate_objective(sims, holdings0, cash0, pred_sum, &x, cfg)?;
    let mut objective_trace = vec![eval.f];

    let mut converged = false;
    let mut outer_iters = 0usize;
    let mut line_search_trials = 0usize;
    let mut line_search_accepts = 0usize;
    let mut line_search_invalid_evals = 0usize;
    let mut line_search_rescue_attempts = 0usize;
    let mut line_search_rescue_accepts = 0usize;
    let mut active_dims = x.len();
    for _ in 0..cfg.max_iters {
        outer_iters += 1;
        let mut grad = eval.grad.clone();
        apply_zero_trade_clamp(&x, &mut grad, bounds, cfg, n);
        let pg = projected_grad_norm(&x, &grad, bounds);
        if pg <= cfg.pg_tol {
            converged = true;
            break;
        }

        let p = match cfg.optimizer {
            GlobalOptimizer::DiagonalProjectedNewton => grad
                .iter()
                .zip(eval.hdiag.iter())
                .map(|(&g, &h)| -g / h.max(cfg.hess_floor))
                .collect(),
            GlobalOptimizer::LbfgsbProjected
            | GlobalOptimizer::DualDecompositionPrototype
            | GlobalOptimizer::DualRouterV1 => {
                let free_mask = build_free_mask(&x, &grad, bounds, cfg.active_set_eps);
                active_dims = free_mask.iter().filter(|&&is_free| is_free).count();
                let mut direction =
                    compute_lbfgs_direction(&grad, &eval.hdiag, &free_mask, &lbfgs_state, cfg);
                let directional_derivative = dot(&grad, &direction);
                let finite_direction = direction.iter().all(|v| v.is_finite());
                if !finite_direction
                    || !directional_derivative.is_finite()
                    || directional_derivative >= 0.0
                {
                    direction =
                        scaled_negative_gradient_direction(&grad, &eval.hdiag, &free_mask, cfg);
                }
                direction
            }
        };
        if matches!(cfg.optimizer, GlobalOptimizer::DiagonalProjectedNewton) {
            active_dims = p.len();
        }

        let mut alpha = 1.0_f64;
        let mut accepted = false;
        let prev_grad = eval.grad.clone();
        for _ in 0..cfg.max_line_search_trials {
            line_search_trials += 1;
            let mut x_trial = x.clone();
            for (xt, (&xi, &pi)) in x_trial.iter_mut().zip(x.iter().zip(p.iter())) {
                *xt = xi + alpha * pi;
            }
            project_box(&mut x_trial, bounds);
            let mut eval_trial =
                evaluate_objective(sims, holdings0, cash0, pred_sum, &x_trial, cfg);
            if eval_trial.is_none() {
                let repair_stats =
                    restore_coupled_feasibility(&mut x_trial, sims, holdings0, cash0, cfg);
                feasibility_stats.merge(repair_stats);
                eval_trial = evaluate_objective(sims, holdings0, cash0, pred_sum, &x_trial, cfg);
            }

            let Some(eval_trial) = eval_trial else {
                line_search_invalid_evals += 1;
                alpha *= cfg.backtrack_beta;
                if alpha <= 1e-10 {
                    break;
                }
                continue;
            };

            let mut step = vec![0.0_f64; x.len()];
            for i in 0..x.len() {
                step[i] = x_trial[i] - x[i];
            }
            let step_norm = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
            if step_norm <= 1e-14 {
                break;
            }

            let rhs = eval.f + cfg.armijo_c1 * dot(&grad, &step);
            if eval_trial.f <= rhs {
                if matches!(
                    cfg.optimizer,
                    GlobalOptimizer::LbfgsbProjected
                        | GlobalOptimizer::DualDecompositionPrototype
                        | GlobalOptimizer::DualRouterV1
                ) {
                    let y: Vec<f64> = eval_trial
                        .grad
                        .iter()
                        .zip(prev_grad.iter())
                        .map(|(&new_g, &old_g)| new_g - old_g)
                        .collect();
                    lbfgs_state.update(step.clone(), y, cfg);
                }
                x = x_trial;
                eval = eval_trial;
                objective_trace.push(eval.f);
                accepted = true;
                line_search_accepts += 1;
                break;
            }

            alpha *= cfg.backtrack_beta;
            if alpha <= 1e-10 {
                break;
            }
        }

        if !accepted {
            let try_rescue = matches!(
                cfg.optimizer,
                GlobalOptimizer::LbfgsbProjected
                    | GlobalOptimizer::DualDecompositionPrototype
                    | GlobalOptimizer::DualRouterV1
            ) && cfg.line_search_rescue_trials > 0;

            if try_rescue {
                line_search_rescue_attempts += 1;
                lbfgs_state.clear();
                let free_mask = build_free_mask(&x, &grad, bounds, cfg.active_set_eps);
                let rescue_direction =
                    scaled_negative_gradient_direction(&grad, &eval.hdiag, &free_mask, cfg);
                let directional_derivative = dot(&grad, &rescue_direction);
                let finite_direction = rescue_direction.iter().all(|v| v.is_finite());
                let min_decrease =
                    cfg.line_search_rescue_min_decrease.max(0.0) * (1.0 + eval.f.abs());

                if finite_direction
                    && directional_derivative.is_finite()
                    && directional_derivative < 0.0
                {
                    let mut alpha_rescue = 1.0_f64;
                    for _ in 0..cfg.line_search_rescue_trials {
                        line_search_trials += 1;
                        let mut x_trial = x.clone();
                        for (xt, (&xi, &pi)) in x_trial
                            .iter_mut()
                            .zip(x.iter().zip(rescue_direction.iter()))
                        {
                            *xt = xi + alpha_rescue * pi;
                        }
                        project_box(&mut x_trial, bounds);
                        let mut eval_trial =
                            evaluate_objective(sims, holdings0, cash0, pred_sum, &x_trial, cfg);
                        if eval_trial.is_none() {
                            let repair_stats = restore_coupled_feasibility(
                                &mut x_trial,
                                sims,
                                holdings0,
                                cash0,
                                cfg,
                            );
                            feasibility_stats.merge(repair_stats);
                            eval_trial =
                                evaluate_objective(sims, holdings0, cash0, pred_sum, &x_trial, cfg);
                        }

                        let Some(eval_trial) = eval_trial else {
                            line_search_invalid_evals += 1;
                            alpha_rescue *= cfg.backtrack_beta;
                            if alpha_rescue <= 1e-10 {
                                break;
                            }
                            continue;
                        };

                        let step_norm = x_trial
                            .iter()
                            .zip(x.iter())
                            .map(|(&trial, &curr)| (trial - curr).abs())
                            .fold(0.0, f64::max);
                        if step_norm <= 1e-14 {
                            break;
                        }

                        if eval_trial.f <= eval.f - min_decrease {
                            x = x_trial;
                            eval = eval_trial;
                            objective_trace.push(eval.f);
                            accepted = true;
                            line_search_accepts += 1;
                            line_search_rescue_accepts += 1;
                            break;
                        }

                        alpha_rescue *= cfg.backtrack_beta;
                        if alpha_rescue <= 1e-10 {
                            break;
                        }
                    }
                }
            }

            if !accepted {
                break;
            }
        }
    }

    let mut grad = eval.grad.clone();
    apply_zero_trade_clamp(&x, &mut grad, bounds, cfg, n);
    let projected_grad = projected_grad_norm(&x, &grad, bounds);
    let coupled_residual = coupled_feasibility_residual(&x, sims, holdings0, cash0, cfg);
    if !converged && projected_grad <= cfg.pg_tol * 10.0 {
        converged = true;
    }

    Some(SolveTrace {
        optimizer: cfg.optimizer,
        x,
        eval,
        projected_grad_norm: projected_grad,
        coupled_residual,
        converged,
        objective_trace,
        outer_iters,
        line_search_trials,
        line_search_accepts,
        line_search_invalid_evals,
        line_search_rescue_attempts,
        line_search_rescue_accepts,
        feasibility_repairs: feasibility_stats.repair_passes,
        feasibility_hold_clamps: feasibility_stats.hold_clamps,
        feasibility_cash_scales: feasibility_stats.cash_scalings,
        active_dims,
        curvature_skips: lbfgs_state.curvature_skips,
    })
}

fn emit_mint_bracket_with_sells(
    sims: &mut [PoolSim],
    actions: &mut Vec<Action>,
    amount: f64,
    desired_sells: &[f64],
    sim_balances: &BalanceMap,
    cfg: GlobalSolveConfig,
) -> Result<(Vec<f64>, f64), GlobalCandidateInvalidReason> {
    if amount <= DUST {
        return Ok((vec![0.0; sims.len()], 0.0));
    }
    if sims.len() < 2 {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    let target_idx = sims
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.prediction.total_cmp(&b.prediction))
        .map_or(0usize, |(i, _)| i);

    let (contract_1, contract_2) = action_contract_pair(sims);
    actions.push(Action::FlashLoan { amount });
    actions.push(Action::Mint {
        contract_1,
        contract_2,
        amount,
        target_market: sims[target_idx].market_name,
    });

    let mut executed = vec![0.0_f64; sims.len()];
    let mut total_proceeds = 0.0_f64;
    let mut has_sell = false;
    for (i, sim) in sims.iter_mut().enumerate() {
        let desired = desired_sells.get(i).copied().unwrap_or(0.0).max(0.0);
        if desired <= DUST {
            continue;
        }

        let available = sim_balances
            .get(sim.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0)
            + amount;
        let sell_tol = cfg.solver_budget_eps * (1.0 + desired.abs().max(available.abs()));
        if desired > available + sell_tol {
            return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
        }
        let safe_desired = desired.min(available.max(0.0));
        if safe_desired <= DUST {
            continue;
        }

        let Some((sold, proceeds, new_price)) = sim.sell_exact(safe_desired) else {
            return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
        };
        if sold <= DUST || !proceeds.is_finite() {
            return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
        }
        if !within_execution_tol(safe_desired, sold, cfg.solver_budget_eps) {
            return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
        }

        sim.set_price(new_price);
        actions.push(Action::Sell {
            market_name: sim.market_name,
            amount: sold,
            proceeds,
        });
        executed[i] = sold;
        total_proceeds += proceeds;
        has_sell = true;
    }

    if !has_sell {
        return Err(GlobalCandidateInvalidReason::MintWithoutSellFlow);
    }

    actions.push(Action::RepayFlashLoan { amount });
    Ok((executed, total_proceeds))
}

fn emit_uniform_merge(
    sims: &[PoolSim],
    actions: &mut Vec<Action>,
    amount: f64,
    sim_balances: &BalanceMap,
    cfg: GlobalSolveConfig,
) -> Result<(), GlobalCandidateInvalidReason> {
    if amount <= DUST {
        return Ok(());
    }

    let min_hold = sims
        .iter()
        .map(|s| {
            sim_balances
                .get(s.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        })
        .fold(f64::INFINITY, f64::min);
    if min_hold + cfg.solver_budget_eps < amount {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    let source_idx = sims
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let ha = sim_balances.get(a.market_name).copied().unwrap_or(0.0);
            let hb = sim_balances.get(b.market_name).copied().unwrap_or(0.0);
            ha.total_cmp(&hb)
        })
        .map_or(0usize, |(i, _)| i);

    let (contract_1, contract_2) = action_contract_pair(sims);
    actions.push(Action::Merge {
        contract_1,
        contract_2,
        amount,
        source_market: sims[source_idx].market_name,
    });

    Ok(())
}

fn emit_direct_sell_exact(
    sims: &mut [PoolSim],
    idx: usize,
    desired: f64,
    actions: &mut Vec<Action>,
    budget: &mut f64,
    cfg: GlobalSolveConfig,
) -> Result<f64, GlobalCandidateInvalidReason> {
    if desired <= DUST {
        return Ok(0.0);
    }
    if idx >= sims.len() {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    let Some((sold, proceeds, new_price)) = sims[idx].sell_exact(desired) else {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    };
    if sold <= DUST || !proceeds.is_finite() {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }
    if !within_execution_tol(desired, sold, cfg.solver_budget_eps) {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    sims[idx].set_price(new_price);
    *budget += proceeds;
    actions.push(Action::Sell {
        market_name: sims[idx].market_name,
        amount: sold,
        proceeds,
    });
    Ok(sold)
}

fn emit_direct_buy_exact(
    sims: &mut [PoolSim],
    idx: usize,
    desired: f64,
    actions: &mut Vec<Action>,
    budget: &mut f64,
    cfg: GlobalSolveConfig,
) -> Result<f64, GlobalCandidateInvalidReason> {
    if desired <= DUST {
        return Ok(0.0);
    }
    if idx >= sims.len() {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    let Some((bought, cost, new_price)) = sims[idx].buy_exact(desired) else {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    };
    if bought <= DUST || !cost.is_finite() {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }
    if !within_execution_tol(desired, bought, cfg.solver_budget_eps) {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    let budget_tol = cfg.solver_budget_eps * (1.0 + budget.abs().max(cost.abs()));
    if cost > *budget + budget_tol {
        return Err(GlobalCandidateInvalidReason::BudgetEpsilonViolation);
    }

    sims[idx].set_price(new_price);
    *budget -= cost;
    actions.push(Action::Buy {
        market_name: sims[idx].market_name,
        amount: bought,
        cost,
    });
    Ok(bought)
}

fn append_direct_sell_cleanup(
    sims: &mut [PoolSim],
    sim_balances: &mut BalanceMap,
    budget: &mut f64,
    actions: &mut Vec<Action>,
) {
    for _ in 0..3 {
        let mut improved = false;
        let mut order: Vec<usize> = (0..sims.len()).collect();
        order.sort_by(|lhs, rhs| {
            let edge_l = sims[*lhs].price() * FEE_FACTOR - sims[*lhs].prediction;
            let edge_r = sims[*rhs].price() * FEE_FACTOR - sims[*rhs].prediction;
            edge_r.total_cmp(&edge_l)
        });

        for idx in order {
            let edge = sims[idx].price() * FEE_FACTOR - sims[idx].prediction;
            if !edge.is_finite() || edge <= 1e-9 {
                continue;
            }

            let available = sim_balances
                .get(sims[idx].market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let max_sell = sims[idx].max_sell_tokens().max(0.0);
            let cap = available.min(max_sell);
            if cap <= DUST {
                continue;
            }

            let target_price = (sims[idx].prediction / FEE_FACTOR).max(DOMAIN_EPS);
            let desired = sims[idx]
                .sell_to_price(target_price)
                .map(|(amount, _, _)| amount)
                .unwrap_or(cap)
                .max(0.0);
            let amount = desired.min(cap);
            if amount <= DUST {
                continue;
            }

            let Some((sold, proceeds, new_price)) = sims[idx].sell_exact(amount) else {
                continue;
            };
            if sold <= DUST || !proceeds.is_finite() {
                continue;
            }

            sims[idx].set_price(new_price);
            *budget += proceeds;
            let entry = sim_balances.entry(sims[idx].market_name).or_insert(0.0);
            *entry = (*entry - sold).max(0.0);
            actions.push(Action::Sell {
                market_name: sims[idx].market_name,
                amount: sold,
                proceeds,
            });
            improved = true;
        }

        if !improved {
            break;
        }
    }
}

fn held_balance(sim_balances: &BalanceMap, market_name: &'static str) -> f64 {
    sim_balances
        .get(market_name)
        .copied()
        .unwrap_or(0.0)
        .max(0.0)
}

fn run_route_phase1_sell_overpriced(
    sims: &mut [PoolSim],
    sim_balances: &mut BalanceMap,
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
) {
    const MAX_ROUTE_PHASE1_ITERS: usize = 128;
    for i in 0..sims.len() {
        for _ in 0..MAX_ROUTE_PHASE1_ITERS {
            let price = sims[i].price();
            let pred = sims[i].prediction;
            if price <= pred + EPS {
                break;
            }

            let held = held_balance(sim_balances, sims[i].market_name);
            if held <= EPS {
                break;
            }

            let (tokens_needed, _, _) = sims[i]
                .sell_to_price(pred)
                .unwrap_or((0.0, 0.0, sims[i].price()));
            let sell_amount = if tokens_needed > EPS {
                tokens_needed.min(held)
            } else {
                held
            };
            if sell_amount <= EPS {
                break;
            }

            let sold_total = {
                let mut exec = ExecutionState::new(sims, budget, actions, sim_balances);
                exec.execute_optimal_sell(i, sell_amount, 0.0, mint_available)
            };
            if sold_total <= EPS {
                break;
            }

            let new_price = sims[i].price();
            let new_held = held_balance(sim_balances, sims[i].market_name);
            if new_price >= price - EPS
                && new_price > pred + EPS
                && new_held + EPS < held
                && new_held > EPS
            {
                let sold_remaining = {
                    let mut exec = ExecutionState::new(sims, budget, actions, sim_balances);
                    exec.execute_optimal_sell(i, new_held, 0.0, mint_available)
                };
                if sold_remaining <= EPS {
                    break;
                }
            }
        }
    }
}

fn run_route_phase3_recycling(
    sims: &mut [PoolSim],
    sim_balances: &mut BalanceMap,
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mut phase3_prof: f64,
    mint_available: bool,
) {
    const MAX_ROUTE_PHASE3_ITERS: usize = 8;
    const ROUTE_PHASE3_EV_REL_TOL: f64 = 1e-10;

    for _ in 0..MAX_ROUTE_PHASE3_ITERS {
        if phase3_prof <= 0.0 {
            break;
        }

        let mut liquidation_candidates: Vec<(usize, f64)> = Vec::new();
        for (i, sim) in sims.iter().enumerate() {
            if held_balance(sim_balances, sim.market_name) <= EPS {
                continue;
            }
            let prof = profitability(sim.prediction, sim.price());
            if prof < phase3_prof {
                liquidation_candidates.push((i, prof));
            }
        }
        if liquidation_candidates.is_empty() {
            break;
        }
        liquidation_candidates.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));

        let ev_before = portfolio_expected_value(sims, sim_balances, *budget);
        if !ev_before.is_finite() {
            break;
        }

        let sims_before = sims.to_vec();
        let balances_before = sim_balances.clone();
        let budget_before = *budget;
        let actions_before = actions.len();

        let mut sold_any = false;
        for (idx, _) in liquidation_candidates {
            let sell_amount = held_balance(sim_balances, sims[idx].market_name);
            if sell_amount <= EPS {
                continue;
            }
            let sold_total = {
                let mut exec = ExecutionState::new(sims, budget, actions, sim_balances);
                exec.execute_optimal_sell(idx, sell_amount, phase3_prof, mint_available)
            };
            if sold_total > EPS {
                sold_any = true;
            }
        }
        if !sold_any {
            sims.clone_from_slice(&sims_before);
            *sim_balances = balances_before;
            *budget = budget_before;
            actions.truncate(actions_before);
            break;
        }

        let actions_before_waterfall = actions.len();
        let next_prof = waterfall(sims, budget, actions, mint_available);
        apply_actions_to_sim_balances(
            &actions[actions_before_waterfall..],
            sims,
            sim_balances,
        );

        let ev_after = portfolio_expected_value(sims, sim_balances, *budget);
        let ev_tol = ROUTE_PHASE3_EV_REL_TOL * (1.0 + ev_before.abs() + ev_after.abs());
        if !ev_after.is_finite() || ev_after <= ev_before + ev_tol {
            sims.clone_from_slice(&sims_before);
            *sim_balances = balances_before;
            *budget = budget_before;
            actions.truncate(actions_before);
            break;
        }

        if next_prof > 0.0 {
            phase3_prof = next_prof;
        }
    }
}

fn run_route_polish_reoptimization(
    sims: &mut Vec<PoolSim>,
    sim_balances: &mut BalanceMap,
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
) {
    const MAX_ROUTE_POLISH_PASSES: usize = 3;
    const ROUTE_POLISH_EV_REL_TOL: f64 = 1e-10;

    for _ in 0..MAX_ROUTE_POLISH_PASSES {
        if *budget <= EPS {
            break;
        }

        let ev_before = portfolio_expected_value(sims, sim_balances, *budget);
        if !ev_before.is_finite() {
            break;
        }

        let mut trial_sims = sims.clone();
        let mut trial_balances = sim_balances.clone();
        let mut trial_budget = *budget;
        let mut trial_actions = Vec::new();

        if mint_available {
            let _ = {
                let mut exec = ExecutionState::new(
                    &mut trial_sims,
                    &mut trial_budget,
                    &mut trial_actions,
                    &mut trial_balances,
                );
                exec.execute_complete_set_arb()
            };
        }

        run_route_phase1_sell_overpriced(
            &mut trial_sims,
            &mut trial_balances,
            &mut trial_budget,
            &mut trial_actions,
            mint_available,
        );

        let actions_before = trial_actions.len();
        let last_bought_prof = waterfall(
            &mut trial_sims,
            &mut trial_budget,
            &mut trial_actions,
            mint_available,
        );
        apply_actions_to_sim_balances(
            &trial_actions[actions_before..],
            &trial_sims,
            &mut trial_balances,
        );

        if mint_available && last_bought_prof > 0.0 {
            run_route_phase3_recycling(
                &mut trial_sims,
                &mut trial_balances,
                &mut trial_budget,
                &mut trial_actions,
                last_bought_prof,
                true,
            );
        }

        if trial_actions.is_empty() {
            break;
        }

        let ev_after = portfolio_expected_value(&trial_sims, &trial_balances, trial_budget);
        let ev_tol = ROUTE_POLISH_EV_REL_TOL * (1.0 + ev_before.abs() + ev_after.abs());
        if !ev_after.is_finite() || ev_after <= ev_before + ev_tol {
            break;
        }

        actions.extend(trial_actions);
        *sims = trial_sims;
        *sim_balances = trial_balances;
        *budget = trial_budget;
    }
}

fn replay_expected_value_for_actions(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
) -> Option<f64> {
    let preds = crate::pools::prediction_map();
    let sims = build_sims(slot0_results, &preds).ok()?;
    let (holdings, cash) =
        replay_actions_to_portfolio_state(actions, slot0_results, balances, susds_balance);
    let holdings_ev: f64 = sims
        .iter()
        .map(|sim| {
            holdings
                .get(sim.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
                * sim.prediction
        })
        .sum();
    Some(cash + holdings_ev)
}

fn build_route_refinement_actions(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Option<Vec<Action>> {
    if !susds_balance.is_finite() || susds_balance <= 0.0 {
        return None;
    }

    let preds = crate::pools::prediction_map();
    let mut sims = build_sims(slot0_results, &preds).ok()?;
    if sims.is_empty() {
        return None;
    }
    let allow_complete_set = sims.len() == crate::predictions::PREDICTIONS_L1.len();

    let mut sim_balances: BalanceMap = sims
        .iter()
        .map(|sim| (sim.market_name, lookup_balance(balances, sim.market_name)))
        .collect();
    let mut budget = susds_balance;
    let mut actions = Vec::new();

    if allow_complete_set {
        let _ = {
            let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
            exec.execute_complete_set_arb()
        };
    }

    run_route_phase1_sell_overpriced(
        &mut sims,
        &mut sim_balances,
        &mut budget,
        &mut actions,
        allow_complete_set,
    );

    let actions_before = actions.len();
    let last_bought_prof = waterfall(&mut sims, &mut budget, &mut actions, allow_complete_set);
    apply_actions_to_sim_balances(&actions[actions_before..], &sims, &mut sim_balances);
    if allow_complete_set && last_bought_prof > 0.0 {
        run_route_phase3_recycling(
            &mut sims,
            &mut sim_balances,
            &mut budget,
            &mut actions,
            last_bought_prof,
            true,
        );
    }
    run_route_polish_reoptimization(
        &mut sims,
        &mut sim_balances,
        &mut budget,
        &mut actions,
        allow_complete_set,
    );

    for _ in 0..4 {
        let loop_start = actions.len();
        run_route_phase1_sell_overpriced(
            &mut sims,
            &mut sim_balances,
            &mut budget,
            &mut actions,
            allow_complete_set,
        );
        let actions_before_direct = actions.len();
        let _ = waterfall(&mut sims, &mut budget, &mut actions, false);
        apply_actions_to_sim_balances(&actions[actions_before_direct..], &sims, &mut sim_balances);
        if actions.len() == loop_start {
            break;
        }
    }

    if allow_complete_set {
        let actions_before_mixed = actions.len();
        let mixed_last_prof = waterfall(&mut sims, &mut budget, &mut actions, true);
        apply_actions_to_sim_balances(&actions[actions_before_mixed..], &sims, &mut sim_balances);
        if mixed_last_prof > 0.0 {
            run_route_phase3_recycling(
                &mut sims,
                &mut sim_balances,
                &mut budget,
                &mut actions,
                mixed_last_prof,
                true,
            );
        }

        for _ in 0..2 {
            let loop_start = actions.len();
            run_route_phase1_sell_overpriced(
                &mut sims,
                &mut sim_balances,
                &mut budget,
                &mut actions,
                true,
            );
            let actions_before_direct = actions.len();
            let _ = waterfall(&mut sims, &mut budget, &mut actions, false);
            apply_actions_to_sim_balances(
                &actions[actions_before_direct..],
                &sims,
                &mut sim_balances,
            );
            if actions.len() == loop_start {
                break;
            }
        }
    }

    Some(actions)
}

pub(super) fn project_solution_to_actions(
    sims: &mut [PoolSim],
    balances: &HashMap<&str, f64>,
    initial_susd: f64,
    solution: &GlobalSolveResult,
    cfg: GlobalSolveConfig,
) -> Result<Vec<Action>, GlobalCandidateInvalidReason> {
    if !initial_susd.is_finite() || initial_susd < 0.0 {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    let mut actions = Vec::new();
    let mut budget = initial_susd;

    let mut sim_balances: BalanceMap = HashMap::new();
    for sim in sims.iter() {
        sim_balances.insert(sim.market_name, lookup_balance(balances, sim.market_name));
    }

    let mut remaining_direct_sells = solution.direct_sells.clone();
    let theta = solution.net_complete_set;

    if theta > DUST {
        let start = actions.len();
        let (embedded_sells, mint_sell_proceeds) = emit_mint_bracket_with_sells(
            sims,
            &mut actions,
            theta.max(0.0),
            &remaining_direct_sells,
            &sim_balances,
            cfg,
        )?;
        for (remaining, embedded) in remaining_direct_sells.iter_mut().zip(embedded_sells.iter()) {
            let residual = (*remaining - *embedded).max(0.0);
            let residual_tol = cfg.solver_budget_eps * (1.0 + remaining.abs().max(embedded.abs()));
            *remaining = if residual <= residual_tol {
                0.0
            } else {
                residual
            };
        }
        budget += mint_sell_proceeds - theta.max(0.0);
        if !budget.is_finite() {
            return Err(GlobalCandidateInvalidReason::BudgetEpsilonViolation);
        }
        apply_actions_to_sim_balances(&actions[start..], sims, &mut sim_balances);
    }

    for (idx, remaining_sell) in remaining_direct_sells.iter().copied().enumerate() {
        if remaining_sell <= DUST {
            continue;
        }
        let available = sim_balances
            .get(sims[idx].market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let sell_tol = cfg.solver_budget_eps * (1.0 + remaining_sell.abs().max(available.abs()));
        if remaining_sell > available + sell_tol {
            return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
        }
        let safe_sell = remaining_sell.min(available.max(0.0));
        if safe_sell <= DUST {
            continue;
        }

        let start = actions.len();
        emit_direct_sell_exact(sims, idx, safe_sell, &mut actions, &mut budget, cfg)?;
        apply_actions_to_sim_balances(&actions[start..], sims, &mut sim_balances);
    }

    for (idx, buy) in solution.direct_buys.iter().copied().enumerate() {
        if buy <= DUST {
            continue;
        }
        let start = actions.len();
        emit_direct_buy_exact(sims, idx, buy, &mut actions, &mut budget, cfg)?;
        apply_actions_to_sim_balances(&actions[start..], sims, &mut sim_balances);
    }

    if theta < -DUST {
        let merge_amount = (-theta).max(0.0);
        let start = actions.len();
        emit_uniform_merge(sims, &mut actions, merge_amount, &sim_balances, cfg)?;
        budget += merge_amount;
        apply_actions_to_sim_balances(&actions[start..], sims, &mut sim_balances);
    }

    if !budget.is_finite() || budget < -cfg.solver_budget_eps * (1.0 + budget.abs()) {
        return Err(GlobalCandidateInvalidReason::BudgetEpsilonViolation);
    }

    if sim_balances
        .values()
        .any(|v| !v.is_finite() || *v < -cfg.solver_budget_eps * (1.0 + v.abs()))
    {
        return Err(GlobalCandidateInvalidReason::ProjectionUnavailable);
    }

    Ok(actions)
}

fn warm_start_from_actions(actions: &[Action], sims: &[PoolSim]) -> Vec<f64> {
    let n = sims.len();
    let mut x = vec![0.0_f64; 2 * n + 1];
    let mut idx_by_market = HashMap::with_capacity(n);
    for (i, sim) in sims.iter().enumerate() {
        idx_by_market.insert(sim.market_name, i);
    }

    let mut theta = 0.0_f64;
    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name) {
                    x[idx] += amount.max(0.0);
                }
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name) {
                    x[n + idx] += amount.max(0.0);
                }
            }
            Action::Mint { amount, .. } => {
                theta += amount.max(0.0);
            }
            Action::Merge { amount, .. } => {
                theta -= amount.max(0.0);
            }
            Action::FlashLoan { .. } | Action::RepayFlashLoan { .. } => {}
        }
    }

    x[2 * n] = theta;
    x
}

pub(super) fn build_global_candidate_plan(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    warm_start_actions: Option<&[Action]>,
    cfg: GlobalSolveConfig,
) -> Option<GlobalCandidatePlan> {
    match cfg.optimizer {
        GlobalOptimizer::DualDecompositionPrototype => {
            super::global_solver_dual::build_global_candidate_plan_dual(
                balances,
                susds_balance,
                slot0_results,
                warm_start_actions,
                cfg,
            )
        }
        GlobalOptimizer::DualRouterV1 => {
            super::global_solver_dual_router::build_global_candidate_plan_dual_router(
                balances,
                susds_balance,
                slot0_results,
                warm_start_actions,
                cfg,
            )
            .or_else(|| {
                let mut fallback_cfg = cfg;
                fallback_cfg.optimizer = GlobalOptimizer::LbfgsbProjected;
                build_global_candidate_plan_primal(
                    balances,
                    susds_balance,
                    slot0_results,
                    warm_start_actions,
                    fallback_cfg,
                )
            })
        }
        GlobalOptimizer::DiagonalProjectedNewton | GlobalOptimizer::LbfgsbProjected => {
            build_global_candidate_plan_primal(
                balances,
                susds_balance,
                slot0_results,
                warm_start_actions,
                cfg,
            )
        }
    }
}

pub(super) fn build_global_candidate_plan_primal(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    warm_start_actions: Option<&[Action]>,
    cfg: GlobalSolveConfig,
) -> Option<GlobalCandidatePlan> {
    if !susds_balance.is_finite() || susds_balance <= 0.0 {
        return None;
    }

    let preds = crate::pools::prediction_map();
    let mut sims = build_sims(slot0_results, &preds).ok()?;
    if sims.is_empty() {
        return None;
    }

    let holdings0: Vec<f64> = sims
        .iter()
        .map(|sim| lookup_balance(balances, sim.market_name))
        .collect();

    let allow_complete_set = sims.len() == crate::predictions::PREDICTIONS_L1.len();
    let bounds = build_bounds(&sims, &holdings0, susds_balance, allow_complete_set);

    let warm_start = warm_start_actions.map(|actions| warm_start_from_actions(actions, &sims));
    let trace = run_projected_newton(
        &sims,
        &holdings0,
        susds_balance,
        &bounds,
        warm_start.as_deref(),
        cfg,
    )
    .or_else(|| run_projected_newton(&sims, &holdings0, susds_balance, &bounds, None, cfg))?;

    let n = sims.len();
    let solution = GlobalSolveResult {
        optimizer: trace.optimizer,
        direct_buys: trace.x[..n].to_vec(),
        direct_sells: trace.x[n..(2 * n)].to_vec(),
        net_complete_set: trace.x[2 * n],
        dual_residual_norm: 0.0,
        primal_restore_iters: 0,
        projected_grad_norm: trace.projected_grad_norm,
        coupled_residual: trace.coupled_residual,
        converged: trace.converged,
        outer_iters: trace.outer_iters,
        line_search_trials: trace.line_search_trials,
        line_search_accepts: trace.line_search_accepts,
        line_search_invalid_evals: trace.line_search_invalid_evals,
        line_search_rescue_attempts: trace.line_search_rescue_attempts,
        line_search_rescue_accepts: trace.line_search_rescue_accepts,
        feasibility_repairs: trace.feasibility_repairs,
        feasibility_hold_clamps: trace.feasibility_hold_clamps,
        feasibility_cash_scales: trace.feasibility_cash_scales,
        active_dims: trace.active_dims,
        curvature_skips: trace.curvature_skips,
    };

    let (mut actions, mut invalid_reason) =
        match project_solution_to_actions(&mut sims, balances, susds_balance, &solution, cfg) {
            Ok(actions) => (actions, None),
            Err(reason) => (Vec::new(), Some(reason)),
        };

    if invalid_reason.is_none() && cfg.enable_route_refinement {
        let (post_holdings, mut post_cash) =
            replay_actions_to_portfolio_state(&actions, slot0_results, balances, susds_balance);
        if post_cash.is_finite() && post_cash > DUST {
            let mut sim_balances: BalanceMap = sims
                .iter()
                .map(|sim| {
                    (
                        sim.market_name,
                        post_holdings
                            .get(sim.market_name)
                            .copied()
                            .unwrap_or(0.0)
                            .max(0.0),
                    )
                })
                .collect();
            append_direct_sell_cleanup(&mut sims, &mut sim_balances, &mut post_cash, &mut actions);
            let _ = waterfall(&mut sims, &mut post_cash, &mut actions, allow_complete_set);
            append_direct_sell_cleanup(&mut sims, &mut sim_balances, &mut post_cash, &mut actions);
            let _ = waterfall(&mut sims, &mut post_cash, &mut actions, allow_complete_set);
        }
    }

    let (replay_holdings, replay_cash) =
        replay_actions_to_portfolio_state(&actions, slot0_results, balances, susds_balance);
    let target_cash = replay_cash;
    let target_holds: Vec<f64> = sims
        .iter()
        .map(|sim| {
            replay_holdings
                .get(sim.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        })
        .collect();

    let cash_tol = cfg
        .solver_budget_eps
        .max(1e-6 * (1.0 + target_cash.abs().max(replay_cash.abs())));
    let replay_cash_delta = (target_cash - replay_cash).abs();

    let mut max_hold_delta = 0.0_f64;
    for (i, sim) in sims.iter().enumerate() {
        let replay_hold = replay_holdings
            .get(sim.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let target_hold = target_holds.get(i).copied().unwrap_or(0.0).max(0.0);
        let delta = (target_hold - replay_hold).abs();
        if delta > max_hold_delta {
            max_hold_delta = delta;
        }
    }

    let hold_tol_scale = 1.0
        + target_holds
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(replay_holdings.values().copied().fold(0.0_f64, f64::max));
    let hold_tol = cfg.solver_budget_eps.max(1e-6 * hold_tol_scale);
    let coupled_tol = cfg.solver_budget_eps.max(1e-6);

    if invalid_reason.is_none() && cfg.enable_route_refinement {
        if let Some(route_actions) =
            build_route_refinement_actions(balances, susds_balance, slot0_results)
        {
            let candidate_ev = replay_expected_value_for_actions(
                &actions,
                slot0_results,
                balances,
                susds_balance,
            )
            .unwrap_or(f64::NEG_INFINITY);
            let route_ev = replay_expected_value_for_actions(
                &route_actions,
                slot0_results,
                balances,
                susds_balance,
            )
            .unwrap_or(f64::NEG_INFINITY);
            let tol = 1e-9 * (1.0 + candidate_ev.abs().max(route_ev.abs()));
            if route_ev > candidate_ev + tol {
                actions = route_actions;
            }
        }
    }

    if invalid_reason.is_none() {
        if !trace.eval.cost_sum.is_finite()
            || trace.objective_trace.iter().any(|value| !value.is_finite())
        {
            invalid_reason = Some(GlobalCandidateInvalidReason::NonFiniteSolveState);
        } else if !solution.projected_grad_norm.is_finite()
            || !solution.coupled_residual.is_finite()
        {
            invalid_reason = Some(GlobalCandidateInvalidReason::ProjectedGradientTooLarge);
        } else if solution.projected_grad_norm > cfg.pg_tol * 10.0
            && solution.coupled_residual > coupled_tol
        {
            invalid_reason = Some(GlobalCandidateInvalidReason::ProjectedGradientTooLarge);
        } else if replay_cash_delta > cash_tol {
            invalid_reason = Some(GlobalCandidateInvalidReason::ReplayCashMismatch);
        } else if max_hold_delta > hold_tol {
            invalid_reason = Some(GlobalCandidateInvalidReason::ReplayHoldingsMismatch);
        }
    }

    let candidate_valid = invalid_reason.is_none();

    Some(GlobalCandidatePlan {
        actions,
        solve: solution,
        replay_cash_delta,
        replay_holdings_delta: max_hold_delta,
        candidate_valid,
        invalid_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::{Address, U256};
    use std::collections::HashMap;

    fn sample_sims(prices: &[f64], preds: &[f64]) -> Vec<PoolSim> {
        assert_eq!(prices.len(), preds.len());
        let markets: Vec<_> = crate::markets::MARKETS_L1
            .iter()
            .filter(|m| m.pool.is_some())
            .take(prices.len())
            .collect();
        assert_eq!(
            markets.len(),
            prices.len(),
            "insufficient pooled sample markets"
        );
        let slot0_results: Vec<_> = markets
            .iter()
            .enumerate()
            .map(|(i, market)| {
                let pool = market.pool.as_ref().expect("pooled market required");
                let is_token1_outcome =
                    pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
                let sqrt_price =
                    crate::pools::prediction_to_sqrt_price_x96(prices[i], is_token1_outcome)
                        .unwrap_or(U256::from(1u128 << 96));
                (
                    Slot0Result {
                        pool_id: Address::ZERO,
                        sqrt_price_x96: sqrt_price,
                        tick: 0,
                        observation_index: 0,
                        observation_cardinality: 0,
                        observation_cardinality_next: 0,
                        fee_protocol: 0,
                        unlocked: true,
                    },
                    *market,
                )
            })
            .collect();

        slot0_results
            .iter()
            .enumerate()
            .map(|(i, (slot0, market))| {
                PoolSim::from_slot0(slot0, market, preds[i]).expect("valid pooled sample sim")
            })
            .collect()
    }

    #[test]
    fn projected_grad_norm_honors_box_boundaries() {
        let bounds = BoxBounds {
            lower: vec![0.0, 0.0, -1.0],
            upper: vec![1.0, 2.0, 1.0],
        };
        let x = vec![0.0, 0.5, -1.0];
        let g = vec![2.0, -3.0, 4.0];
        let norm = projected_grad_norm(&x, &g, &bounds);
        assert!(norm <= 3.0 + 1e-12);
        assert!(norm >= 2.9);
    }

    #[test]
    fn objective_trace_is_monotone_under_line_search() {
        let sims = sample_sims(&[0.08, 0.07, 0.06], &[0.6, 0.25, 0.15]);
        let holdings0 = vec![0.1, 0.2, 0.3];
        let bounds = build_bounds(&sims, &holdings0, 10.0, false);
        let cfg = GlobalSolveConfig {
            max_iters: 64,
            ..GlobalSolveConfig::default()
        };
        let trace = run_projected_newton(&sims, &holdings0, 10.0, &bounds, None, cfg)
            .expect("projected newton should produce a trace");

        for win in trace.objective_trace.windows(2) {
            assert!(
                win[1] <= win[0] + 1e-12,
                "line search must be monotone: {} -> {}",
                win[0],
                win[1]
            );
        }
    }

    #[test]
    fn solver_stays_within_box_bounds() {
        let sims = sample_sims(&[0.07], &[0.42]);
        let holdings0 = vec![0.5];
        let bounds = build_bounds(&sims, &holdings0, 1.0, false);
        let cfg = GlobalSolveConfig {
            max_iters: 32,
            ..GlobalSolveConfig::default()
        };

        let trace = run_projected_newton(&sims, &holdings0, 1.0, &bounds, None, cfg)
            .expect("solver should return a trace");

        for i in 0..trace.x.len() {
            assert!(trace.x[i] + 1e-12 >= bounds.lower[i]);
            assert!(trace.x[i] <= bounds.upper[i] + 1e-12);
        }
    }

    #[test]
    fn mint_bracket_requires_sell_flow() {
        let mut sims = sample_sims(&[0.08, 0.07, 0.06], &[0.6, 0.25, 0.15]);
        let mut actions = Vec::new();
        let desired_sells = vec![0.0_f64; sims.len()];
        let mut sim_balances: BalanceMap = HashMap::new();
        for sim in &sims {
            sim_balances.insert(sim.market_name, 0.0);
        }

        let err = emit_mint_bracket_with_sells(
            &mut sims,
            &mut actions,
            0.2,
            &desired_sells,
            &sim_balances,
            GlobalSolveConfig::default(),
        )
        .expect_err("mint bracket without sell flow should fail");
        assert_eq!(err, GlobalCandidateInvalidReason::MintWithoutSellFlow);
    }

    #[test]
    fn project_solution_reports_budget_epsilon_violation() {
        let mut sims = sample_sims(&[0.08], &[0.6]);
        let balances: HashMap<&str, f64> = HashMap::new();
        let solution = GlobalSolveResult {
            optimizer: GlobalOptimizer::LbfgsbProjected,
            direct_buys: vec![0.5],
            direct_sells: vec![0.0],
            net_complete_set: 0.0,
            dual_residual_norm: 0.0,
            primal_restore_iters: 0,
            projected_grad_norm: 0.0,
            coupled_residual: 0.0,
            converged: true,
            outer_iters: 0,
            line_search_trials: 0,
            line_search_accepts: 0,
            line_search_invalid_evals: 0,
            line_search_rescue_attempts: 0,
            line_search_rescue_accepts: 0,
            feasibility_repairs: 0,
            feasibility_hold_clamps: 0,
            feasibility_cash_scales: 0,
            active_dims: 0,
            curvature_skips: 0,
        };
        let cfg = GlobalSolveConfig {
            solver_budget_eps: 1e-12,
            ..GlobalSolveConfig::default()
        };

        let err = project_solution_to_actions(&mut sims, &balances, 1e-12, &solution, cfg)
            .expect_err("unaffordable exact buy should fail");
        assert_eq!(err, GlobalCandidateInvalidReason::BudgetEpsilonViolation);
    }

    #[test]
    fn zero_trade_clamp_pins_near_zero_gradients() {
        let cfg = GlobalSolveConfig {
            zero_trade_band_eps: 1e-8,
            ..GlobalSolveConfig::default()
        };
        let bounds = BoxBounds {
            lower: vec![0.0, 0.0, 0.0, 0.0, -1.0],
            upper: vec![1.0, 1.0, 1.0, 1.0, 1.0],
        };
        let x = vec![0.0, 0.2, 0.0, 0.1, 0.0];
        let mut grad = vec![1e-10, 0.5, -1e-10, -0.5, 0.0];

        apply_zero_trade_clamp(&x, &mut grad, &bounds, cfg, 2);

        assert_eq!(grad[0], 0.0);
        assert_eq!(grad[2], 0.0);
        assert!(grad[1] > 0.0);
        assert!(grad[3] < 0.0);
    }

    #[test]
    fn lbfgs_direction_is_descent_or_falls_back() {
        let cfg = GlobalSolveConfig::default();
        let grad = vec![1.0, 2.0];
        let hdiag = vec![1.0, 2.0];
        let free_mask = vec![true, true];
        let lbfgs = LbfgsState {
            pairs: vec![LbfgsHistoryPair {
                s: vec![1.0, 0.0],
                y: vec![f64::NAN, 0.0],
                rho: 1.0,
            }],
            curvature_skips: 0,
        };

        let direction = compute_lbfgs_direction(&grad, &hdiag, &free_mask, &lbfgs, cfg);
        assert!(
            direction.iter().any(|v| !v.is_finite()),
            "test setup should produce a non-finite L-BFGS direction"
        );

        let fallback = scaled_negative_gradient_direction(&grad, &hdiag, &free_mask, cfg);
        assert!(
            dot(&grad, &fallback) < 0.0,
            "scaled negative gradient fallback must be descent"
        );
    }

    #[test]
    fn active_set_zeroes_bound_pinned_coordinates() {
        let cfg = GlobalSolveConfig::default();
        let bounds = BoxBounds {
            lower: vec![0.0, 0.0, 0.0],
            upper: vec![1.0, 1.0, 1.0],
        };
        let x = vec![0.0, 1.0, 0.5];
        let grad = vec![0.5, -0.3, 0.2];
        let hdiag = vec![1.0, 1.0, 1.0];

        let free_mask = build_free_mask(&x, &grad, &bounds, cfg.active_set_eps);
        assert!(
            !free_mask[0],
            "lower-bound pinned coordinate should be active"
        );
        assert!(
            !free_mask[1],
            "upper-bound pinned coordinate should be active"
        );
        assert!(free_mask[2], "interior coordinate should remain free");

        let direction = scaled_negative_gradient_direction(&grad, &hdiag, &free_mask, cfg);
        assert_eq!(direction[0], 0.0);
        assert_eq!(direction[1], 0.0);
        assert!(direction[2].abs() > 0.0);
    }

    #[test]
    fn curvature_guard_skips_bad_pairs() {
        let cfg = GlobalSolveConfig {
            lbfgs_min_curvature: 1e-12,
            ..GlobalSolveConfig::default()
        };
        let mut lbfgs = LbfgsState::default();

        lbfgs.update(vec![1.0, 0.0], vec![-1.0, 0.0], cfg);

        assert_eq!(lbfgs.pairs.len(), 0);
        assert_eq!(lbfgs.curvature_skips, 1);
    }

    #[test]
    fn history_bounded_by_lbfgs_history() {
        let cfg = GlobalSolveConfig {
            lbfgs_history: 2,
            lbfgs_min_curvature: 0.0,
            ..GlobalSolveConfig::default()
        };
        let mut lbfgs = LbfgsState::default();

        lbfgs.update(vec![1.0, 0.0], vec![1.0, 0.0], cfg);
        lbfgs.update(vec![0.0, 1.0], vec![0.0, 1.0], cfg);
        lbfgs.update(vec![1.0, 1.0], vec![1.0, 1.0], cfg);

        assert_eq!(lbfgs.pairs.len(), 2);
        assert_eq!(lbfgs.pairs[0].s, vec![0.0, 1.0]);
        assert_eq!(lbfgs.pairs[1].s, vec![1.0, 1.0]);
    }

    #[test]
    fn line_search_rescue_recovers_from_over_strict_armijo() {
        let sims = sample_sims(&[0.08, 0.07, 0.06], &[0.6, 0.25, 0.15]);
        let holdings0 = vec![0.1, 0.2, 0.3];
        let bounds = build_bounds(&sims, &holdings0, 10.0, false);
        let cfg = GlobalSolveConfig {
            max_iters: 16,
            armijo_c1: 10.0,
            max_line_search_trials: 4,
            line_search_rescue_trials: 16,
            line_search_rescue_min_decrease: 1e-14,
            ..GlobalSolveConfig::default()
        };

        let trace = run_projected_newton(&sims, &holdings0, 10.0, &bounds, None, cfg)
            .expect("solver should still produce a trace with rescue enabled");
        assert!(
            trace.line_search_rescue_attempts > 0,
            "rescue must be attempted after Armijo failure"
        );
        assert!(
            trace.line_search_rescue_accepts > 0,
            "rescue should recover at least one step"
        );
        let first = trace.objective_trace[0];
        let best = trace
            .objective_trace
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert!(
            best <= first - 1e-12,
            "rescue path should find at least one strict objective improvement"
        );
    }

    #[test]
    fn line_search_rescue_is_bounded_when_no_step_is_acceptable() {
        let sims = sample_sims(&[0.08, 0.07, 0.06], &[0.6, 0.25, 0.15]);
        let holdings0 = vec![0.1, 0.2, 0.3];
        let bounds = build_bounds(&sims, &holdings0, 10.0, false);
        let cfg = GlobalSolveConfig {
            max_iters: 8,
            armijo_c1: 10.0,
            max_line_search_trials: 1,
            line_search_rescue_trials: 2,
            line_search_rescue_min_decrease: 1e6,
            ..GlobalSolveConfig::default()
        };

        let trace = run_projected_newton(&sims, &holdings0, 10.0, &bounds, None, cfg)
            .expect("solver should return trace even when rescue fails");
        assert_eq!(trace.line_search_rescue_attempts, 1);
        assert_eq!(trace.line_search_rescue_accepts, 0);
        assert!(
            trace.outer_iters <= 2,
            "solver should terminate quickly when no step can be accepted"
        );
    }
}
