use std::collections::HashMap;

use crate::markets::MarketData;
use crate::pools::Slot0Result;

use super::Action;
use super::diagnostics::replay_actions_to_portfolio_state;
use super::global_solver::{
    BoxBounds, GlobalCandidateInvalidReason, GlobalCandidatePlan, GlobalOptimizer, GlobalSolveConfig,
    GlobalSolveResult, project_solution_to_actions,
};
use super::sim::{DUST, EPS, FEE_FACTOR, PoolSim, build_sims};
use super::types::lookup_balance;

const DOMAIN_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
struct RouteFaithfulEval {
    f: f64,
    grad: Vec<f64>,
    cash: f64,
    buys: Vec<f64>,
    sells: Vec<f64>,
    holds: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct FlowSubproblem {
    buy: f64,
    sell: f64,
    objective: f64,
    token_flow: f64,
}

#[derive(Debug, Clone)]
struct DualIterate {
    shadow_prices: Vec<f64>,
    subproblems: Vec<FlowSubproblem>,
    theta_plus: f64,
    theta_minus: f64,
    objective: f64,
    gradient: Vec<f64>,
}

#[derive(Debug, Clone)]
struct RouteFaithfulSolve {
    x: Vec<f64>,
    eval: RouteFaithfulEval,
    projected_grad_norm: f64,
    converged: bool,
    outer_iters: usize,
    line_search_trials: usize,
    line_search_accepts: usize,
    line_search_invalid_evals: usize,
    active_dims: usize,
}

trait MarketArbSolver {
    fn solve_arb(&self, local_price: f64) -> FlowSubproblem;
}

impl MarketArbSolver for PoolSim {
    fn solve_arb(&self, local_price: f64) -> FlowSubproblem {
        if !local_price.is_finite() || local_price <= 0.0 {
            return FlowSubproblem::default();
        }

        let ask0 = self.price() / FEE_FACTOR;
        let bid0 = self.price() * FEE_FACTOR;

        let mut best = FlowSubproblem::default();

        if local_price > ask0 + 1e-12 {
            let lam = self.lambda();
            if lam > 0.0 {
                let raw = (1.0 - (self.price() / (FEE_FACTOR * local_price)).sqrt()) / lam;
                let buy = raw.clamp(0.0, self.max_buy_tokens().max(0.0));
                if buy > DUST {
                    if let Some((actual, cost, _new_price)) = self.buy_exact(buy)
                        && actual > DUST
                        && cost.is_finite()
                    {
                        let objective = local_price * actual - cost;
                        if objective > best.objective {
                            best = FlowSubproblem {
                                buy: actual,
                                sell: 0.0,
                                objective,
                                token_flow: actual,
                            };
                        }
                    }
                }
            }
        }

        if local_price + 1e-12 < bid0 {
            let kappa = self.kappa();
            if kappa > 0.0 {
                let raw = ((self.price() * FEE_FACTOR / local_price).sqrt() - 1.0) / kappa;
                let sell = raw.clamp(0.0, self.max_sell_tokens().max(0.0));
                if sell > DUST {
                    if let Some((actual, proceeds, _new_price)) = self.sell_exact(sell)
                        && actual > DUST
                        && proceeds.is_finite()
                    {
                        let objective = proceeds - local_price * actual;
                        if objective > best.objective {
                            best = FlowSubproblem {
                                buy: 0.0,
                                sell: actual,
                                objective,
                                token_flow: -actual,
                            };
                        }
                    }
                }
            }
        }

        best
    }
}

fn idx_buy(i: usize) -> usize {
    i
}

fn idx_sell_ratio(n: usize, i: usize) -> usize {
    n + i
}

fn idx_theta_plus(n: usize) -> usize {
    2 * n
}

fn idx_theta_minus(n: usize) -> usize {
    2 * n + 1
}

fn variable_count(n: usize) -> usize {
    2 * n + 2
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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

fn buy_cost_and_derivatives(sim: &PoolSim, buy: f64) -> Option<(f64, f64)> {
    if buy < -DOMAIN_EPS {
        return None;
    }
    let price = sim.price();
    if !price.is_finite() || price <= 0.0 {
        return None;
    }
    let lambda = sim.lambda();
    if !lambda.is_finite() || lambda < 0.0 {
        return None;
    }
    let d = 1.0 - buy.max(0.0) * lambda;
    if d <= 1e-15 || !d.is_finite() {
        return None;
    }
    let cost = buy.max(0.0) * price / (FEE_FACTOR * d);
    let d1 = price / (FEE_FACTOR * d * d);
    (cost.is_finite() && d1.is_finite()).then_some((cost, d1))
}

fn sell_proceeds_and_derivatives(sim: &PoolSim, sell: f64) -> Option<(f64, f64)> {
    if sell < -DOMAIN_EPS {
        return None;
    }
    let price = sim.price();
    if !price.is_finite() || price <= 0.0 {
        return None;
    }
    let kappa = sim.kappa();
    if !kappa.is_finite() || kappa < 0.0 {
        return None;
    }
    let d = 1.0 + sell.max(0.0) * kappa;
    if d <= 1e-15 || !d.is_finite() {
        return None;
    }
    let proceeds = price * sell.max(0.0) * FEE_FACTOR / d;
    let d1 = price * FEE_FACTOR / (d * d);
    (proceeds.is_finite() && d1.is_finite()).then_some((proceeds, d1))
}

fn build_route_faithful_bounds(
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    allow_complete_set: bool,
) -> BoxBounds {
    let n = sims.len();
    let mut lower = vec![0.0_f64; variable_count(n)];
    let mut upper = vec![0.0_f64; variable_count(n)];

    let mut buy_caps = vec![0.0_f64; n];
    let mut depth_budget = 0.0_f64;
    for i in 0..n {
        let lambda = sims[i].lambda().max(0.0);
        let mut buy_cap = sims[i].max_buy_tokens().max(0.0);
        if lambda > 0.0 {
            let safe_cap = ((1.0 - 1e-9) / lambda).max(0.0);
            buy_cap = buy_cap.min(safe_cap);
        }
        buy_caps[i] = buy_cap;
        let local_depth = buy_cap + sims[i].max_sell_tokens().max(0.0) + holdings0[i].max(0.0);
        if local_depth.is_finite() {
            depth_budget += local_depth;
        }
    }

    let theta_pos_cap = if allow_complete_set {
        (cash0.max(0.0) + depth_budget.max(0.0)).max(0.0)
    } else {
        0.0
    };

    let mut theta_neg_cap = 0.0_f64;
    if allow_complete_set {
        for i in 0..n {
            let candidate = holdings0[i].max(0.0) + buy_caps[i] + theta_pos_cap;
            theta_neg_cap = if i == 0 {
                candidate
            } else {
                theta_neg_cap.min(candidate)
            };
        }
        theta_neg_cap = theta_neg_cap.max(0.0);
    }

    for i in 0..n {
        lower[idx_buy(i)] = 0.0;
        upper[idx_buy(i)] = buy_caps[i];
        lower[idx_sell_ratio(n, i)] = 0.0;
        upper[idx_sell_ratio(n, i)] = 1.0;
    }

    lower[idx_theta_plus(n)] = 0.0;
    upper[idx_theta_plus(n)] = theta_pos_cap.max(0.0);
    lower[idx_theta_minus(n)] = 0.0;
    upper[idx_theta_minus(n)] = theta_neg_cap.max(0.0);

    BoxBounds { lower, upper }
}

fn evaluate_route_faithful(
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    predictions: &[f64],
    x: &[f64],
    cfg: GlobalSolveConfig,
) -> Option<RouteFaithfulEval> {
    let n = sims.len();
    if x.len() != variable_count(n) || holdings0.len() != n || predictions.len() != n {
        return None;
    }

    let theta_plus = x[idx_theta_plus(n)].max(0.0);
    let theta_minus = x[idx_theta_minus(n)].max(0.0);
    let theta = theta_plus - theta_minus;

    let mut buys = vec![0.0_f64; n];
    let mut ratios = vec![0.0_f64; n];
    let mut sells = vec![0.0_f64; n];
    let mut holds = vec![0.0_f64; n];
    let mut buy_cost = vec![0.0_f64; n];
    let mut buy_d1 = vec![0.0_f64; n];
    let mut sell_proc = vec![0.0_f64; n];
    let mut sell_d1 = vec![0.0_f64; n];

    let mut buy_cost_sum = 0.0_f64;
    let mut sell_proc_sum = 0.0_f64;
    let mut holdings_ev = 0.0_f64;
    let mut churn_term = 0.0_f64;

    for i in 0..n {
        let buy = x[idx_buy(i)].max(0.0);
        let ratio = x[idx_sell_ratio(n, i)].clamp(0.0, 1.0);
        let inventory = holdings0[i] + buy + theta;
        if !inventory.is_finite() || inventory < -DOMAIN_EPS {
            return None;
        }
        let inv = inventory.max(0.0);
        let sell = ratio * inv;
        let hold = inv - sell;
        if hold < -DOMAIN_EPS {
            return None;
        }

        let (cost, d1_buy) = buy_cost_and_derivatives(&sims[i], buy)?;
        let (proceeds, d1_sell) = sell_proceeds_and_derivatives(&sims[i], sell)?;

        buys[i] = buy;
        ratios[i] = ratio;
        sells[i] = sell;
        holds[i] = hold;
        buy_cost[i] = cost;
        buy_d1[i] = d1_buy;
        sell_proc[i] = proceeds;
        sell_d1[i] = d1_sell;

        buy_cost_sum += cost;
        sell_proc_sum += proceeds;
        holdings_ev += hold * predictions[i];
        churn_term += buy * sell;
    }

    let cash = cash0 - theta_plus + theta_minus - buy_cost_sum + sell_proc_sum;
    if !cash.is_finite() {
        return None;
    }
    let cash_margin = cash - cfg.solver_budget_eps;
    if cash_margin < -DOMAIN_EPS {
        return None;
    }

    let barrier_shift = cfg.barrier_shift.max(DOMAIN_EPS);
    let cash_bar = cash_margin.max(0.0) + barrier_shift;
    let mut f = -(
        cash + holdings_ev
            - cfg.theta_l2_reg * theta * theta
            - cfg.buy_sell_churn_reg * churn_term
    ) - cfg.barrier_mu_cash * cash_bar.ln();

    for hold in &holds {
        let hold_bar = hold.max(0.0) + barrier_shift;
        f -= cfg.barrier_mu_hold * hold_bar.ln();
    }
    if !f.is_finite() {
        return None;
    }

    let mut grad = vec![0.0_f64; variable_count(n)];

    let mut dc_dtheta_common = -1.0_f64;
    let mut hold_tp_sum = 0.0_f64;
    let mut hold_tm_sum = 0.0_f64;
    let mut churn_tp_sum = 0.0_f64;

    for i in 0..n {
        let inv = holdings0[i] + buys[i] + theta;
        let ratio = ratios[i];
        let hold = holds[i].max(0.0);
        let hold_bar = hold + barrier_shift;

        let ds_db = ratio;
        let ds_dr = inv.max(0.0);
        let ds_dtp = ratio;

        let dhold_db = 1.0 - ratio;
        let dhold_dr = -inv.max(0.0);
        let dhold_dtp = 1.0 - ratio;
        let dhold_dtm = -(1.0 - ratio);

        let dc_db = -buy_d1[i] + sell_d1[i] * ds_db;
        let dc_dr = sell_d1[i] * ds_dr;

        dc_dtheta_common += sell_d1[i] * ds_dtp;

        let churn_db = sells[i] + buys[i] * ratio;
        let churn_dr = buys[i] * inv.max(0.0);

        let dev_db = dc_db + predictions[i] * dhold_db - cfg.buy_sell_churn_reg * churn_db;
        let dev_dr = dc_dr + predictions[i] * dhold_dr - cfg.buy_sell_churn_reg * churn_dr;

        grad[idx_buy(i)] = -dev_db
            - (cfg.barrier_mu_cash * dc_db / cash_bar)
            - (cfg.barrier_mu_hold * dhold_db / hold_bar);
        grad[idx_sell_ratio(n, i)] = -dev_dr
            - (cfg.barrier_mu_cash * dc_dr / cash_bar)
            - (cfg.barrier_mu_hold * dhold_dr / hold_bar);

        hold_tp_sum += dhold_dtp / hold_bar;
        hold_tm_sum += dhold_dtm / hold_bar;
        churn_tp_sum += buys[i] * ratio;
    }

    let dev_tp = dc_dtheta_common
        + predictions
            .iter()
            .zip(ratios.iter())
            .map(|(pred, ratio)| pred * (1.0 - ratio))
            .sum::<f64>()
        - 2.0 * cfg.theta_l2_reg * theta
        - cfg.buy_sell_churn_reg * churn_tp_sum;

    let dev_tm = (-dc_dtheta_common)
        + predictions
            .iter()
            .zip(ratios.iter())
            .map(|(pred, ratio)| -pred * (1.0 - ratio))
            .sum::<f64>()
        + 2.0 * cfg.theta_l2_reg * theta
        + cfg.buy_sell_churn_reg * churn_tp_sum;

    grad[idx_theta_plus(n)] =
        -dev_tp - (cfg.barrier_mu_cash * dc_dtheta_common / cash_bar) - (cfg.barrier_mu_hold * hold_tp_sum);
    grad[idx_theta_minus(n)] =
        -dev_tm - (cfg.barrier_mu_cash * (-dc_dtheta_common) / cash_bar) - (cfg.barrier_mu_hold * hold_tm_sum);

    if grad.iter().any(|value| !value.is_finite()) {
        return None;
    }

    Some(RouteFaithfulEval {
        f,
        grad,
        cash,
        buys,
        sells,
        holds,
    })
}

fn build_route_faithful_seed(
    initial_buys: &[f64],
    initial_sells: &[f64],
    initial_theta: f64,
    holdings0: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut x = vec![0.0_f64; variable_count(n)];
    let theta_plus = initial_theta.max(0.0);
    let theta_minus = (-initial_theta).max(0.0);
    x[idx_theta_plus(n)] = theta_plus;
    x[idx_theta_minus(n)] = theta_minus;

    for i in 0..n {
        let buy = initial_buys.get(i).copied().unwrap_or(0.0).max(0.0);
        let sell = initial_sells.get(i).copied().unwrap_or(0.0).max(0.0);
        let inventory = holdings0[i] + buy + theta_plus - theta_minus;
        let ratio = if inventory > DUST {
            (sell / inventory).clamp(0.0, 1.0)
        } else {
            0.0
        };
        x[idx_buy(i)] = buy;
        x[idx_sell_ratio(n, i)] = ratio;
    }

    x
}

fn run_route_faithful_primal(
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    predictions: &[f64],
    bounds: &BoxBounds,
    seed: &[f64],
    cfg: GlobalSolveConfig,
) -> Option<RouteFaithfulSolve> {
    let n = sims.len();
    let mut x = seed.to_vec();
    if x.len() != variable_count(n) {
        return None;
    }
    project_box(&mut x, bounds);

    let mut eval = evaluate_route_faithful(sims, holdings0, cash0, predictions, &x, cfg);
    if eval.is_none() {
        let mut candidate = x.clone();
        for _ in 0..32 {
            for i in 0..n {
                candidate[idx_buy(i)] *= 0.5;
                candidate[idx_sell_ratio(n, i)] *= 0.5;
            }
            candidate[idx_theta_plus(n)] *= 0.5;
            candidate[idx_theta_minus(n)] *= 0.5;

            project_box(&mut candidate, bounds);

            let theta_plus = candidate[idx_theta_plus(n)].max(0.0);
            let mut inv_floor = f64::INFINITY;
            for i in 0..n {
                inv_floor = inv_floor.min(holdings0[i] + candidate[idx_buy(i)].max(0.0) + theta_plus);
            }
            if inv_floor.is_finite() {
                candidate[idx_theta_minus(n)] =
                    candidate[idx_theta_minus(n)].min(inv_floor.max(0.0));
            }
            project_box(&mut candidate, bounds);

            if let Some(ok) = evaluate_route_faithful(sims, holdings0, cash0, predictions, &candidate, cfg) {
                x = candidate;
                eval = Some(ok);
                break;
            }
        }
    }
    let mut eval = eval?;
    let mut converged = false;
    let mut outer_iters = 0usize;
    let mut line_search_trials = 0usize;
    let mut line_search_accepts = 0usize;
    let mut line_search_invalid_evals = 0usize;

    let pg_tol = cfg.dual_router_pg_tol.max(1e-8);
    for _ in 0..cfg.max_iters {
        outer_iters += 1;
        let pg = projected_grad_norm(&x, &eval.grad, bounds);
        if pg <= pg_tol {
            converged = true;
            break;
        }

        let direction: Vec<f64> = eval.grad.iter().map(|g| -g).collect();
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..cfg.max_line_search_trials {
            line_search_trials += 1;
            let mut x_trial = x.clone();
            for (xt, (&xi, &di)) in x_trial.iter_mut().zip(x.iter().zip(direction.iter())) {
                *xt = xi + alpha * di;
            }
            project_box(&mut x_trial, bounds);
            let Some(eval_trial) = evaluate_route_faithful(
                sims,
                holdings0,
                cash0,
                predictions,
                &x_trial,
                cfg,
            ) else {
                line_search_invalid_evals += 1;
                alpha *= cfg.backtrack_beta;
                if alpha <= 1e-10 {
                    break;
                }
                continue;
            };

            let step: Vec<f64> = x_trial.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
            let step_norm = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
            if step_norm <= 1e-14 {
                break;
            }

            let rhs = eval.f + cfg.armijo_c1 * dot(&eval.grad, &step);
            if eval_trial.f <= rhs {
                x = x_trial;
                eval = eval_trial;
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
            break;
        }
    }

    let projected = projected_grad_norm(&x, &eval.grad, bounds);
    Some(RouteFaithfulSolve {
        x,
        eval,
        projected_grad_norm: projected,
        converged,
        outer_iters,
        line_search_trials,
        line_search_accepts,
        line_search_invalid_evals,
        active_dims: variable_count(n),
    })
}

fn dual_complete_set_caps(sims: &[PoolSim], holdings0: &[f64], cash0: f64) -> (f64, f64) {
    let n = sims.len();
    let mut depth_budget = 0.0_f64;
    let mut buy_caps = vec![0.0_f64; n];
    for (i, sim) in sims.iter().enumerate() {
        let buy_cap = sim.max_buy_tokens().max(0.0);
        buy_caps[i] = buy_cap;
        depth_budget += buy_cap + sim.max_sell_tokens().max(0.0) + holdings0[i].max(0.0);
    }

    let theta_plus_cap = (cash0.max(0.0) + depth_budget.max(0.0)).max(0.0);
    let mut theta_minus_cap = 0.0_f64;
    for i in 0..n {
        let cap = holdings0[i].max(0.0) + buy_caps[i] + theta_plus_cap;
        theta_minus_cap = if i == 0 { cap } else { theta_minus_cap.min(cap) };
    }

    (theta_plus_cap, theta_minus_cap.max(0.0))
}

fn solve_complete_set_subproblem(
    shadow_prices: &[f64],
    allow_complete_set: bool,
    theta_plus_cap: f64,
    theta_minus_cap: f64,
) -> (f64, f64, f64) {
    if !allow_complete_set {
        return (0.0, 0.0, 0.0);
    }
    let _ = shadow_prices;
    let _ = theta_plus_cap;
    let _ = theta_minus_cap;
    // The dual complete-set subproblem is highly non-smooth around ties; seeding
    // cap-saturating mint/merge flows destabilizes the route-faithful primal.
    // Let the primal stage discover theta directly from local gradients.
    (0.0, 0.0, 0.0)
}

fn build_dual_iterate(
    sims: &[PoolSim],
    predictions: &[f64],
    shadow_prices: &[f64],
    theta_plus_cap: f64,
    theta_minus_cap: f64,
    allow_complete_set: bool,
    cfg: GlobalSolveConfig,
) -> DualIterate {
    let mut subproblems = Vec::with_capacity(sims.len());
    let mut gradient = vec![0.0_f64; sims.len()];
    let mut objective = 0.0_f64;

    for (i, sim) in sims.iter().enumerate() {
        let local_price = shadow_prices
            .get(i)
            .copied()
            .unwrap_or(cfg.dual_router_price_floor)
            .max(cfg.dual_router_price_floor);
        let solved = sim.solve_arb(local_price);
        objective += solved.objective;
        gradient[i] = solved.token_flow + 1e-3 * (local_price - predictions[i]);
        subproblems.push(solved);
    }

    let (theta_plus, theta_minus, theta_obj) = solve_complete_set_subproblem(
        shadow_prices,
        allow_complete_set,
        theta_plus_cap,
        theta_minus_cap,
    );
    objective += theta_obj;

    for value in &mut gradient {
        *value += theta_plus - theta_minus;
    }

    for (i, price) in shadow_prices.iter().enumerate() {
        objective += 5e-4 * (*price - predictions[i]) * (*price - predictions[i]);
    }

    DualIterate {
        shadow_prices: shadow_prices.to_vec(),
        subproblems,
        theta_plus,
        theta_minus,
        objective,
        gradient,
    }
}

fn projected_dual_grad_norm(prices: &[f64], grad: &[f64], floor: f64) -> f64 {
    prices
        .iter()
        .zip(grad.iter())
        .map(|(price, g)| {
            if *price <= floor + EPS {
                g.min(0.0).abs()
            } else {
                g.abs()
            }
        })
        .fold(0.0, f64::max)
}

fn run_dual_router(
    sims: &[PoolSim],
    predictions: &[f64],
    theta_plus_cap: f64,
    theta_minus_cap: f64,
    allow_complete_set: bool,
    cfg: GlobalSolveConfig,
) -> (DualIterate, f64) {
    let floor = cfg.dual_router_price_floor.max(1e-9);
    let mut prices: Vec<f64> = predictions
        .iter()
        .map(|pred| pred.max(floor))
        .collect();

    let mut best = build_dual_iterate(
        sims,
        predictions,
        &prices,
        theta_plus_cap,
        theta_minus_cap,
        allow_complete_set,
        cfg,
    );
    let mut best_norm = projected_dual_grad_norm(&best.shadow_prices, &best.gradient, floor);

    let mut step = 0.35_f64;
    for _ in 0..cfg.dual_router_max_iters.max(1) {
        let iterate = build_dual_iterate(
            sims,
            predictions,
            &prices,
            theta_plus_cap,
            theta_minus_cap,
            allow_complete_set,
            cfg,
        );
        let norm = projected_dual_grad_norm(&iterate.shadow_prices, &iterate.gradient, floor);
        if iterate.objective < best.objective || norm < best_norm {
            best = iterate.clone();
            best_norm = norm;
        }
        if norm <= cfg.dual_router_pg_tol.max(1e-8) {
            break;
        }

        for (price, grad) in prices.iter_mut().zip(iterate.gradient.iter()) {
            *price = (*price - step * grad).max(floor);
        }
        step *= 0.92;
        if step < 1e-3 {
            step = 1e-3;
        }
    }

    (best, best_norm)
}

fn restore_primal_feasibility(
    buys: &mut [f64],
    sells: &mut [f64],
    theta: &mut f64,
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    cfg: GlobalSolveConfig,
) -> usize {
    let n = buys.len();
    let mut restore_iters = 0usize;

    for _ in 0..cfg.dual_router_primal_restore_iters.max(1) {
        restore_iters += 1;

        for i in 0..n {
            buys[i] = buys[i].clamp(0.0, sims[i].max_buy_tokens().max(0.0));
            let cap = (holdings0[i] + buys[i] + *theta).max(0.0).min(sims[i].max_sell_tokens());
            sells[i] = sells[i].clamp(0.0, cap.max(0.0));
        }

        let mut buy_cost_sum = 0.0_f64;
        let mut sell_proc_sum = 0.0_f64;
        for i in 0..n {
            let Some((cost, _)) = buy_cost_and_derivatives(&sims[i], buys[i]) else {
                continue;
            };
            let Some((proc, _)) = sell_proceeds_and_derivatives(&sims[i], sells[i]) else {
                continue;
            };
            buy_cost_sum += cost;
            sell_proc_sum += proc;
        }

        let cash = cash0 - theta.max(0.0) + (-*theta).max(0.0) - buy_cost_sum + sell_proc_sum;
        let hold_violation = (0..n)
            .map(|i| (-(holdings0[i] + buys[i] - sells[i] + *theta)).max(0.0))
            .fold(0.0_f64, f64::max);
        let cash_violation = (cfg.solver_budget_eps - cash).max(0.0);
        let residual = hold_violation.max(cash_violation);

        if residual <= cfg.dual_router_primal_residual_tol.max(1e-10) {
            break;
        }

        if cash_violation > 0.0 {
            let total_buy: f64 = buys.iter().copied().sum();
            if total_buy > DUST {
                let mut lo = 0.0_f64;
                let mut hi = 1.0_f64;
                let mut best = 0.0_f64;
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let mut trial_buy_cost = 0.0_f64;
                    for i in 0..n {
                        if let Some((cost, _)) = buy_cost_and_derivatives(&sims[i], buys[i] * mid) {
                            trial_buy_cost += cost;
                        }
                    }
                    let trial_cash = cash0
                        - theta.max(0.0)
                        + (-*theta).max(0.0)
                        - trial_buy_cost
                        + sell_proc_sum;
                    if trial_cash + DOMAIN_EPS >= cfg.solver_budget_eps {
                        best = mid;
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    if (hi - lo).abs() <= 1e-9 {
                        break;
                    }
                }
                for value in buys.iter_mut() {
                    *value *= best;
                }
            } else {
                *theta = theta.min(0.0);
            }
        }
    }

    restore_iters
}

fn coupled_residual_from_solution(
    sims: &[PoolSim],
    holdings0: &[f64],
    cash0: f64,
    buys: &[f64],
    sells: &[f64],
    theta: f64,
    cfg: GlobalSolveConfig,
) -> f64 {
    let mut max_violation = 0.0_f64;
    let mut buy_cost_sum = 0.0_f64;
    let mut sell_proc_sum = 0.0_f64;

    for i in 0..sims.len() {
        let hold = holdings0[i] + buys[i] - sells[i] + theta;
        max_violation = max_violation.max((-hold).max(0.0));
        let sell_cap = (holdings0[i] + buys[i] + theta).max(0.0).min(sims[i].max_sell_tokens());
        max_violation = max_violation.max((sells[i] - sell_cap).max(0.0));
        if let Some((cost, _)) = buy_cost_and_derivatives(&sims[i], buys[i]) {
            buy_cost_sum += cost;
        }
        if let Some((proc, _)) = sell_proceeds_and_derivatives(&sims[i], sells[i]) {
            sell_proc_sum += proc;
        }
    }

    let cash = cash0 - theta.max(0.0) + (-theta).max(0.0) - buy_cost_sum + sell_proc_sum;
    max_violation.max((cfg.solver_budget_eps - cash).max(0.0))
}

pub(super) fn build_global_candidate_plan_dual_router(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    warm_start_actions: Option<&[Action]>,
    cfg: GlobalSolveConfig,
) -> Option<GlobalCandidatePlan> {
    if !susds_balance.is_finite() || susds_balance <= 0.0 {
        return None;
    }

    let preds_map = crate::pools::prediction_map();
    let mut sims = build_sims(slot0_results, &preds_map).ok()?;
    if sims.is_empty() {
        return None;
    }

    let holdings0: Vec<f64> = sims
        .iter()
        .map(|sim| lookup_balance(balances, sim.market_name))
        .collect();
    let predictions: Vec<f64> = sims.iter().map(|sim| sim.prediction).collect();

    let allow_complete_set = sims.len() == crate::predictions::PREDICTIONS_L1.len();
    let (theta_plus_cap, theta_minus_cap) = dual_complete_set_caps(&sims, &holdings0, susds_balance);

    let (dual_iterate, dual_residual_norm) = run_dual_router(
        &sims,
        &predictions,
        theta_plus_cap,
        theta_minus_cap,
        allow_complete_set,
        cfg,
    );

    let mut initial_buys = vec![0.0_f64; sims.len()];
    let mut initial_sells = vec![0.0_f64; sims.len()];
    for (i, sub) in dual_iterate.subproblems.iter().enumerate() {
        initial_buys[i] = sub.buy.max(0.0);
        initial_sells[i] = sub.sell.max(0.0);
    }
    let initial_theta = dual_iterate.theta_plus - dual_iterate.theta_minus;

    let bounds = build_route_faithful_bounds(&sims, &holdings0, susds_balance, allow_complete_set);

    let mut seed = build_route_faithful_seed(
        &initial_buys,
        &initial_sells,
        initial_theta,
        &holdings0,
        sims.len(),
    );

    // Warm-start from incumbent actions if provided.
    if let Some(actions) = warm_start_actions {
        let mut buy_from_actions = vec![0.0_f64; sims.len()];
        let mut sell_from_actions = vec![0.0_f64; sims.len()];
        let mut theta = 0.0_f64;
        let idx_by_market: HashMap<&'static str, usize> = sims
            .iter()
            .enumerate()
            .map(|(i, sim)| (sim.market_name, i))
            .collect();

        for action in actions {
            match action {
                Action::Buy {
                    market_name,
                    amount,
                    ..
                } => {
                    if let Some(idx) = idx_by_market.get(market_name) {
                        buy_from_actions[*idx] += amount.max(0.0);
                    }
                }
                Action::Sell {
                    market_name,
                    amount,
                    ..
                } => {
                    if let Some(idx) = idx_by_market.get(market_name) {
                        sell_from_actions[*idx] += amount.max(0.0);
                    }
                }
                Action::Mint { amount, .. } => theta += amount.max(0.0),
                Action::Merge { amount, .. } => theta -= amount.max(0.0),
                Action::FlashLoan { .. } | Action::RepayFlashLoan { .. } => {}
            }
        }

        let warm_seed = build_route_faithful_seed(
            &buy_from_actions,
            &sell_from_actions,
            theta,
            &holdings0,
            sims.len(),
        );

        if warm_seed.len() == seed.len() {
            seed = warm_seed;
        }
    }

    let solved = run_route_faithful_primal(
        &sims,
        &holdings0,
        susds_balance,
        &predictions,
        &bounds,
        &seed,
        cfg,
    )?;

    let n = sims.len();
    let theta_plus = solved.x[idx_theta_plus(n)].max(0.0);
    let theta_minus = solved.x[idx_theta_minus(n)].max(0.0);
    let mut net_theta = theta_plus - theta_minus;
    let mut direct_buys = solved.eval.buys.clone();
    let mut direct_sells = solved.eval.sells.clone();

    let primal_restore_iters = restore_primal_feasibility(
        &mut direct_buys,
        &mut direct_sells,
        &mut net_theta,
        &sims,
        &holdings0,
        susds_balance,
        cfg,
    );

    let mut solve = GlobalSolveResult {
        optimizer: GlobalOptimizer::DualRouterV1,
        direct_buys,
        direct_sells,
        net_complete_set: net_theta,
        dual_residual_norm,
        primal_restore_iters,
        projected_grad_norm: solved.projected_grad_norm,
        coupled_residual: 0.0,
        converged: solved.converged,
        outer_iters: solved.outer_iters,
        line_search_trials: solved.line_search_trials,
        line_search_accepts: solved.line_search_accepts,
        line_search_invalid_evals: solved.line_search_invalid_evals,
        line_search_rescue_attempts: 0,
        line_search_rescue_accepts: 0,
        feasibility_repairs: primal_restore_iters,
        feasibility_hold_clamps: 0,
        feasibility_cash_scales: 0,
        active_dims: solved.active_dims,
        curvature_skips: 0,
    };

    solve.coupled_residual = coupled_residual_from_solution(
        &sims,
        &holdings0,
        susds_balance,
        &solve.direct_buys,
        &solve.direct_sells,
        solve.net_complete_set,
        cfg,
    );

    let (actions, mut invalid_reason) =
        match project_solution_to_actions(&mut sims, balances, susds_balance, &solve, cfg) {
            Ok(actions) => (actions, None),
            Err(reason) => (Vec::new(), Some(reason)),
        };

    let (replay_holdings, replay_cash) =
        replay_actions_to_portfolio_state(&actions, slot0_results, balances, susds_balance);

    let target_cash = solved.eval.cash;
    let target_holds = solved.eval.holds;

    let replay_cash_delta = (target_cash - replay_cash).abs();
    let cash_tol = cfg
        .solver_budget_eps
        .max(1e-6 * (1.0 + target_cash.abs().max(replay_cash.abs())));

    let mut max_hold_delta = 0.0_f64;
    for (i, sim) in sims.iter().enumerate() {
        let replay_hold = replay_holdings
            .get(sim.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let target_hold = target_holds.get(i).copied().unwrap_or(0.0).max(0.0);
        max_hold_delta = max_hold_delta.max((target_hold - replay_hold).abs());
    }

    let hold_tol_scale = 1.0
        + target_holds
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(replay_holdings.values().copied().fold(0.0_f64, f64::max));
    let hold_tol = cfg.solver_budget_eps.max(1e-6 * hold_tol_scale);
    let coupled_tol = cfg.solver_budget_eps.max(1e-6);

    if invalid_reason.is_none() {
        if !dual_residual_norm.is_finite() {
            invalid_reason = Some(GlobalCandidateInvalidReason::NonFiniteSolveState);
        } else if !solve.projected_grad_norm.is_finite() || !solve.coupled_residual.is_finite() {
            invalid_reason = Some(GlobalCandidateInvalidReason::ProjectedGradientTooLarge);
        } else if solve.projected_grad_norm > cfg.pg_tol * 10.0
            && solve.coupled_residual > coupled_tol
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
        solve,
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

    fn sample_sims(prices: &[f64], preds: &[f64]) -> Vec<PoolSim> {
        assert_eq!(prices.len(), preds.len());
        let markets: Vec<_> = crate::markets::MARKETS_L1
            .iter()
            .filter(|market| market.pool.is_some())
            .take(prices.len())
            .collect();
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
    fn pool_subproblem_no_trade_band_holds() {
        let sims = sample_sims(&[0.08], &[0.62]);
        let local = sims[0].price();
        let solved = sims[0].solve_arb(local);
        assert_eq!(solved.buy, 0.0);
        assert_eq!(solved.sell, 0.0);
        assert!(solved.objective.abs() <= 1e-12);
    }

    #[test]
    fn complete_set_subproblem_selects_merge_or_mint() {
        let prices_hi = vec![0.8, 0.4, 0.3];
        let (tp_hi, tm_hi, _) = solve_complete_set_subproblem(&prices_hi, true, 2.0, 2.0);
        assert!(tp_hi > 0.0);
        assert_eq!(tm_hi, 0.0);

        let prices_lo = vec![0.1, 0.2, 0.3];
        let (tp_lo, tm_lo, _) = solve_complete_set_subproblem(&prices_lo, true, 2.0, 2.0);
        assert_eq!(tp_lo, 0.0);
        assert!(tm_lo > 0.0);
    }
}
