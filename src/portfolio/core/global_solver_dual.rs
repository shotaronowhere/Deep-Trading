use std::collections::HashMap;

use crate::markets::MarketData;
use crate::pools::Slot0Result;

use super::Action;
use super::diagnostics::replay_expected_value;
use super::global_solver::{
    GlobalCandidatePlan, GlobalOptimizer, GlobalSolveConfig, build_global_candidate_plan_primal,
};

pub(super) fn build_global_candidate_plan_dual(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    warm_start_actions: Option<&[Action]>,
    cfg: GlobalSolveConfig,
) -> Option<GlobalCandidatePlan> {
    let outer_iters = cfg.dual_outer_iters.max(1);
    let mut lambda = 0.0_f64;
    let mut step = cfg.dual_lambda_step.max(0.0);
    let decay = cfg.dual_lambda_decay.clamp(0.0, 1.0);
    let theta_tol = cfg.dual_theta_tolerance.max(0.0);

    let mut warm_actions = warm_start_actions.map(|actions| actions.to_vec());
    let mut best_valid: Option<GlobalCandidatePlan> = None;
    let mut best_valid_ev = f64::NEG_INFINITY;
    let mut first_available: Option<GlobalCandidatePlan> = None;

    for _ in 0..outer_iters {
        let mut subcfg = cfg;
        subcfg.optimizer = GlobalOptimizer::LbfgsbProjected;
        subcfg.theta_l2_reg = (cfg.theta_l2_reg + lambda).max(0.0);
        subcfg.dual_outer_iters = 0;

        let Some(candidate) = build_global_candidate_plan_primal(
            balances,
            susds_balance,
            slot0_results,
            warm_actions.as_deref(),
            subcfg,
        ) else {
            step *= decay;
            continue;
        };

        if first_available.is_none() {
            first_available = Some(candidate.clone());
        }

        let ev = replay_expected_value(&candidate.actions, slot0_results, balances, susds_balance)
            .unwrap_or(f64::NEG_INFINITY);
        if candidate.candidate_valid && ev.is_finite() && ev > best_valid_ev {
            best_valid_ev = ev;
            best_valid = Some(candidate.clone());
        }

        warm_actions = Some(candidate.actions.clone());
        let theta = candidate.solve.net_complete_set;
        if theta.abs() <= theta_tol {
            break;
        }

        if step > 0.0 {
            lambda = (lambda + step * theta).max(0.0);
            step *= decay;
        }
    }

    let mut selected = best_valid.or(first_available)?;
    selected.solve.optimizer = GlobalOptimizer::DualDecompositionPrototype;
    Some(selected)
}
