use std::collections::{HashMap, HashSet};

use crate::pools::Slot0Result;

use super::Action;
use super::diagnostics::replay_expected_value;
use super::global_solver::{
    GlobalCandidateInvalidReason, GlobalOptimizer, GlobalSolveConfig, GlobalSolveResult,
    build_global_candidate_plan,
};
#[cfg(test)]
use super::sim::DUST;
use super::sim::{
    EPS, PoolSim, SimBuildError, build_sims, build_sims_without_predictions, profitability,
    target_price_for_prof,
};
use super::trading::{ExecutionState, portfolio_expected_value};
use super::types::BalanceMap;
use super::types::{apply_actions_to_sim_balances, lookup_balance};
use super::waterfall::waterfall;

const MAX_PHASE1_ITERS: usize = 128;
const MAX_PHASE3_ITERS: usize = 8;
const PHASE3_PROF_REL_TOL: f64 = 1e-9;
const PHASE3_EV_GUARD_REL_TOL: f64 = 1e-10;
const MAX_POLISH_PASSES: usize = 64;
const POLISH_EV_REL_TOL: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceMode {
    Full,
    ArbOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceEngine {
    Incumbent,
    GlobalCandidate,
    AutoBestReplay,
}

#[derive(Debug, Clone, Copy)]
pub struct RebalanceConfig {
    pub mode: RebalanceMode,
    pub engine: RebalanceEngine,
    pub global: GlobalSolveConfig,
}

impl RebalanceConfig {
    pub const fn new(mode: RebalanceMode, engine: RebalanceEngine) -> Self {
        Self {
            mode,
            engine,
            global: GlobalSolveConfig {
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
                barrier_shift: 1e-4,
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
            },
        }
    }
}

impl Default for RebalanceConfig {
    fn default() -> Self {
        Self {
            mode: RebalanceMode::Full,
            engine: RebalanceEngine::Incumbent,
            global: GlobalSolveConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RebalanceDecisionDiagnostics {
    pub incumbent_ev_after: f64,
    pub candidate_ev_after: f64,
    pub chosen_engine: RebalanceEngine,
    pub candidate_optimizer: GlobalOptimizer,
    pub candidate_valid: bool,
    pub candidate_invalid_reason: Option<GlobalCandidateInvalidReason>,
    pub candidate_projected_grad_norm: f64,
    pub candidate_coupled_residual: f64,
    pub candidate_replay_cash_delta: f64,
    pub candidate_replay_holdings_delta: f64,
    pub candidate_solver_iters: usize,
    pub candidate_line_search_trials: usize,
    pub candidate_line_search_accepts: usize,
    pub candidate_line_search_invalid_evals: usize,
    pub candidate_line_search_rescue_attempts: usize,
    pub candidate_line_search_rescue_accepts: usize,
    pub candidate_feasibility_repairs: usize,
    pub candidate_feasibility_hold_clamps: usize,
    pub candidate_feasibility_cash_scales: usize,
    pub candidate_active_dims: usize,
    pub candidate_curvature_skips: usize,
    pub candidate_dual_residual_norm: f64,
    pub candidate_primal_restore_iters: usize,
    pub candidate_net_theta: f64,
    pub candidate_total_buy: f64,
    pub candidate_total_sell: f64,
    pub candidate_buy_sell_overlap: f64,
}

#[derive(Debug, Clone, Copy)]
enum RebalanceInitError {
    NonFiniteBudget {
        susds_balance: f64,
    },
    SimBuildFailed {
        err: SimBuildError,
    },
    NoEligibleSims {
        slot0_result_count: usize,
        prediction_count: usize,
        expected_outcome_count: usize,
    },
}

fn log_rebalance_init_error(err: RebalanceInitError) {
    match err {
        RebalanceInitError::NonFiniteBudget { susds_balance } => {
            tracing::warn!(
                init_failure = "non_finite_budget",
                susds_balance,
                "rebalance initialization failed"
            );
        }
        RebalanceInitError::SimBuildFailed { err } => {
            tracing::warn!(
                init_failure = "sim_build_failed",
                %err,
                "rebalance initialization failed"
            );
        }
        RebalanceInitError::NoEligibleSims {
            slot0_result_count,
            prediction_count,
            expected_outcome_count,
        } => {
            tracing::warn!(
                init_failure = "no_eligible_sims",
                slot0_result_count,
                prediction_count,
                expected_outcome_count,
                "rebalance initialization failed"
            );
        }
    }
}

fn pooled_l1_outcome_names() -> HashSet<&'static str> {
    crate::markets::MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .map(|m| m.name)
        .collect()
}

fn held_total(sim_balances: &BalanceMap, market_name: &'static str) -> f64 {
    *sim_balances.get(market_name).unwrap_or(&0.0)
}

fn held_legacy(
    sim_balances: &BalanceMap,
    legacy_remaining: &BalanceMap,
    market_name: &'static str,
) -> f64 {
    legacy_remaining
        .get(market_name)
        .copied()
        .unwrap_or(0.0)
        .min(held_total(sim_balances, market_name))
        .max(0.0)
}

struct RebalanceContext {
    actions: Vec<Action>,
    budget: f64,
    sims: Vec<PoolSim>,
    mint_available: bool,
    sim_balances: BalanceMap,
    legacy_remaining: BalanceMap,
}

struct Phase3Trial {
    sims: Vec<PoolSim>,
    balances: BalanceMap,
    legacy_remaining: BalanceMap,
    budget: f64,
    actions: Vec<Action>,
}

impl Phase3Trial {
    fn from_context(ctx: &RebalanceContext) -> Self {
        Self {
            sims: ctx.sims.clone(),
            balances: ctx.sim_balances.clone(),
            legacy_remaining: ctx.legacy_remaining.clone(),
            budget: ctx.budget,
            actions: Vec::new(),
        }
    }

    fn execute_optimal_sell(
        &mut self,
        source_idx: usize,
        sell_amount: f64,
        inventory_keep_prof: f64,
        mint_available: bool,
    ) -> f64 {
        let mut exec = ExecutionState::new(
            &mut self.sims,
            &mut self.budget,
            &mut self.actions,
            &mut self.balances,
        );
        exec.execute_optimal_sell(source_idx, sell_amount, inventory_keep_prof, mint_available)
    }
}

impl RebalanceContext {
    fn from_inputs(
        balances: &HashMap<&str, f64>,
        susds_balance: f64,
        slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
        predictions: &std::collections::HashMap<String, f64>,
        expected_outcome_count: usize,
    ) -> Result<Self, RebalanceInitError> {
        if !susds_balance.is_finite() {
            return Err(RebalanceInitError::NonFiniteBudget { susds_balance });
        }

        let sims = build_sims(slot0_results, predictions)
            .map_err(|err| RebalanceInitError::SimBuildFailed { err })?;
        if sims.is_empty() {
            return Err(RebalanceInitError::NoEligibleSims {
                slot0_result_count: slot0_results.len(),
                prediction_count: predictions.len(),
                expected_outcome_count,
            });
        }

        // Mint/merge routes require all tradeable outcomes to have liquid pools.
        // sims may be smaller if pools have zero liquidity or slot0_results is partial (RPC failures).
        let mint_available = sims.len() == expected_outcome_count;

        let mut sim_balances: BalanceMap = HashMap::new();
        let mut legacy_remaining: BalanceMap = HashMap::new();
        for sim in &sims {
            let held = lookup_balance(balances, sim.market_name);
            sim_balances.insert(sim.market_name, held);
            legacy_remaining.insert(sim.market_name, held);
        }

        Ok(Self {
            actions: Vec::new(),
            budget: susds_balance,
            sims,
            mint_available,
            sim_balances,
            legacy_remaining,
        })
    }

    fn has_legacy_holdings(&self) -> bool {
        self.legacy_remaining.values().any(|&v| v > EPS)
    }

    fn cap_legacy_to_current_holdings(&mut self) -> bool {
        for sim in &self.sims {
            let current = held_total(&self.sim_balances, sim.market_name);
            let legacy = self.legacy_remaining.entry(sim.market_name).or_insert(0.0);
            *legacy = (*legacy).min(current.max(0.0));
            if *legacy < EPS {
                *legacy = 0.0;
            }
        }
        self.has_legacy_holdings()
    }

    fn collect_liquidation_candidates(&self, phase3_prof: f64) -> Vec<(usize, f64)> {
        let mut liquidation_candidates: Vec<(usize, f64)> = Vec::new();
        for (i, sim) in self.sims.iter().enumerate() {
            if held_legacy(&self.sim_balances, &self.legacy_remaining, sim.market_name) <= EPS {
                continue;
            }
            let prof = profitability(sim.prediction, sim.price());
            if prof < phase3_prof {
                liquidation_candidates.push((i, prof));
            }
        }
        liquidation_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        liquidation_candidates
    }

    fn commit_phase3_trial(&mut self, trial: Phase3Trial) {
        self.actions.extend(trial.actions);
        self.sims = trial.sims;
        self.sim_balances = trial.balances;
        self.legacy_remaining = trial.legacy_remaining;
        self.budget = trial.budget;
    }

    fn execute_optimal_sell(
        &mut self,
        source_idx: usize,
        sell_amount: f64,
        inventory_keep_prof: f64,
    ) -> f64 {
        let mut exec = ExecutionState::new(
            &mut self.sims,
            &mut self.budget,
            &mut self.actions,
            &mut self.sim_balances,
        );
        exec.execute_optimal_sell(
            source_idx,
            sell_amount,
            inventory_keep_prof,
            self.mint_available,
        )
    }

    fn run_phase1_sell_overpriced(&mut self) {
        for i in 0..self.sims.len() {
            for _ in 0..MAX_PHASE1_ITERS {
                let price = self.sims[i].price();
                let pred = self.sims[i].prediction;
                if price <= pred + EPS {
                    break;
                }

                let held = held_total(&self.sim_balances, self.sims[i].market_name);
                if held <= EPS {
                    break;
                }

                // Target the direct sell amount needed to bring price to prediction.
                let (tokens_needed, _, _) =
                    self.sims[i]
                        .sell_to_price(pred)
                        .unwrap_or((0.0, 0.0, self.sims[i].price()));
                let sell_amount = if tokens_needed > EPS {
                    tokens_needed.min(held)
                } else {
                    held
                };
                if sell_amount <= EPS {
                    break;
                }

                let sold_total = self.execute_optimal_sell(i, sell_amount, 0.0);
                if sold_total <= EPS {
                    break;
                }

                let new_price = self.sims[i].price();
                let new_held = held_total(&self.sim_balances, self.sims[i].market_name);
                // Merge-heavy splits may not move the source pool price. If that happens while
                // still overpriced, force one full inventory attempt to avoid leaving residual
                // overpriced holdings due one-shot sizing.
                if new_price >= price - EPS
                    && new_price > pred + EPS
                    && new_held + EPS < held
                    && new_held > EPS
                {
                    let sold_remaining = self.execute_optimal_sell(i, new_held, 0.0);
                    if sold_remaining <= EPS {
                        break;
                    }
                }
            }
        }
    }

    fn run_phase3_recycling(&mut self, initial_prof: f64) {
        let mut phase3_prof = initial_prof;
        for _ in 0..MAX_PHASE3_ITERS {
            if phase3_prof <= 0.0 {
                break;
            }
            if !self.has_legacy_holdings() {
                break;
            }

            let liquidation_candidates = self.collect_liquidation_candidates(phase3_prof);
            if liquidation_candidates.is_empty() {
                break;
            }

            let ev_before_iter =
                portfolio_expected_value(&self.sims, &self.sim_balances, self.budget);
            if !ev_before_iter.is_finite() {
                break;
            }

            let mut trial = Phase3Trial::from_context(self);
            let budget_before_liq = trial.budget;
            for (idx, _) in liquidation_candidates {
                let market_name = trial.sims[idx].market_name;
                let legacy_amount =
                    held_legacy(&trial.balances, &trial.legacy_remaining, market_name);
                if legacy_amount <= EPS {
                    continue;
                }
                let target_price = target_price_for_prof(trial.sims[idx].prediction, phase3_prof);
                let (tokens_needed, _, _) = trial.sims[idx]
                    .sell_to_price(target_price)
                    .unwrap_or((0.0, 0.0, trial.sims[idx].price()));
                let sell_target = tokens_needed.min(legacy_amount);
                if sell_target <= EPS {
                    continue;
                }

                let sold_total =
                    trial.execute_optimal_sell(idx, sell_target, phase3_prof, self.mint_available);
                if sold_total > EPS {
                    let legacy = trial.legacy_remaining.entry(market_name).or_insert(0.0);
                    *legacy = (*legacy - sold_total).max(0.0);
                }
            }

            let recovered_budget = trial.budget - budget_before_liq;
            if trial.actions.is_empty() || recovered_budget <= EPS || trial.budget <= EPS {
                break;
            }

            // Reallocate recovered capital and fold the acquired positions into simulated balances.
            let actions_before_realloc = trial.actions.len();
            let new_prof = waterfall(
                &mut trial.sims,
                &mut trial.budget,
                &mut trial.actions,
                self.mint_available,
            );
            apply_actions_to_sim_balances(
                &trial.actions[actions_before_realloc..],
                &trial.sims,
                &mut trial.balances,
            );
            if trial.actions.len() == actions_before_realloc || new_prof <= 0.0 {
                break;
            }

            let ev_after_iter =
                portfolio_expected_value(&trial.sims, &trial.balances, trial.budget);
            if !ev_after_iter.is_finite() {
                break;
            }
            let ev_tol =
                PHASE3_EV_GUARD_REL_TOL * (1.0 + ev_before_iter.abs() + ev_after_iter.abs());
            if ev_after_iter + ev_tol < ev_before_iter {
                break;
            }

            self.commit_phase3_trial(trial);

            let prof_delta = (new_prof - phase3_prof).abs();
            if prof_delta <= PHASE3_PROF_REL_TOL * (1.0 + phase3_prof.abs()) {
                break;
            }
            phase3_prof = new_prof;
        }
    }

    fn commit_trial_actions(&mut self, mut trial: RebalanceContext) {
        self.actions.append(&mut trial.actions);
        self.sims = trial.sims;
        self.sim_balances = trial.sim_balances;
        self.legacy_remaining = trial.legacy_remaining;
        self.budget = trial.budget;
    }

    fn run_polish_reoptimization(&mut self) {
        for _ in 0..MAX_POLISH_PASSES {
            if self.budget <= EPS {
                break;
            }

            let ev_before = portfolio_expected_value(&self.sims, &self.sim_balances, self.budget);
            if !ev_before.is_finite() {
                break;
            }

            // Build a trial context from the current simulated state; commit only if EV improves.
            let mut trial = RebalanceContext {
                actions: Vec::new(),
                budget: self.budget,
                sims: self.sims.clone(),
                mint_available: self.mint_available,
                sim_balances: self.sim_balances.clone(),
                // In polish mode, recycle across current inventory (not only initial legacy).
                legacy_remaining: self.sim_balances.clone(),
            };

            if trial.mint_available {
                let _arb_profit = {
                    let mut exec = ExecutionState::new(
                        &mut trial.sims,
                        &mut trial.budget,
                        &mut trial.actions,
                        &mut trial.sim_balances,
                    );
                    exec.execute_complete_set_arb()
                };
            }

            trial.run_phase1_sell_overpriced();

            let actions_before = trial.actions.len();
            let last_bought_prof = waterfall(
                &mut trial.sims,
                &mut trial.budget,
                &mut trial.actions,
                trial.mint_available,
            );
            apply_actions_to_sim_balances(
                &trial.actions[actions_before..],
                &trial.sims,
                &mut trial.sim_balances,
            );

            // Recycle lower-profit legacy candidates and reallocate recovered budget.
            if trial.cap_legacy_to_current_holdings() {
                trial.run_phase3_recycling(last_bought_prof);
            }

            if trial.actions.is_empty() {
                break;
            }

            let ev_after = portfolio_expected_value(&trial.sims, &trial.sim_balances, trial.budget);
            if !ev_after.is_finite() {
                break;
            }
            let ev_tol = POLISH_EV_REL_TOL * (1.0 + ev_before.abs() + ev_after.abs());
            if ev_after <= ev_before + ev_tol {
                break;
            }

            self.commit_trial_actions(trial);
        }
    }
}

fn rebalance_full(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<Action> {
    let preds = crate::pools::prediction_map();
    let expected_count = crate::predictions::PREDICTIONS_L1.len();
    let mut ctx = match RebalanceContext::from_inputs(
        balances,
        susds_balance,
        slot0_results,
        &preds,
        expected_count,
    ) {
        Ok(ctx) => ctx,
        Err(err) => {
            log_rebalance_init_error(err);
            return Vec::new();
        }
    };

    // ── Phase 0: Complete-set arbitrage (buy-merge only) ──
    // Execute before any discretionary rebalancing so free budget is harvested first.
    if ctx.mint_available {
        let _arb_profit = {
            let mut exec = ExecutionState::new(
                &mut ctx.sims,
                &mut ctx.budget,
                &mut ctx.actions,
                &mut ctx.sim_balances,
            );
            exec.execute_complete_set_arb()
        };
    }

    // ── Phase 1: Sell overpriced holdings ──
    ctx.run_phase1_sell_overpriced();

    // Legacy inventory available for phase-3 recycling cannot exceed current holdings after phase 1.
    let has_legacy_holdings = ctx.cap_legacy_to_current_holdings();

    // ── Phase 2: Waterfall allocation ──
    let actions_before = ctx.actions.len();
    let last_bought_prof = waterfall(
        &mut ctx.sims,
        &mut ctx.budget,
        &mut ctx.actions,
        ctx.mint_available,
    );

    // Update simulated holdings from the initial waterfall pass.
    apply_actions_to_sim_balances(
        &ctx.actions[actions_before..],
        &ctx.sims,
        &mut ctx.sim_balances,
    );

    // ── Phase 3: Post-allocation liquidation ──
    // Iterate liquidation/reallocation until convergence, but recycle legacy inventory only.
    if has_legacy_holdings {
        ctx.run_phase3_recycling(last_bought_prof);
    }

    // ── Phase 4: Short polish re-optimization loop ──
    // Run bounded extra passes only when they increase EV.
    ctx.run_polish_reoptimization();

    // ── Phase 5: Final local cleanup pass ──
    // Run one final sell-overpriced + waterfall pass so the terminal state
    // better aligns with first-order local optimality checks used in tests.
    ctx.run_phase1_sell_overpriced();
    let actions_before_cleanup = ctx.actions.len();
    let cleanup_last_prof = waterfall(
        &mut ctx.sims,
        &mut ctx.budget,
        &mut ctx.actions,
        ctx.mint_available,
    );
    apply_actions_to_sim_balances(
        &ctx.actions[actions_before_cleanup..],
        &ctx.sims,
        &mut ctx.sim_balances,
    );
    if ctx.mint_available && !ctx.actions[actions_before_cleanup..].is_empty() {
        // Recycle across full current inventory once more to reduce residual local gradients.
        ctx.legacy_remaining = ctx.sim_balances.clone();
        if ctx.has_legacy_holdings() {
            ctx.run_phase3_recycling(cleanup_last_prof);
        }
    }
    for _ in 0..4 {
        let start_actions = ctx.actions.len();
        ctx.run_phase1_sell_overpriced();
        let actions_before_direct_cleanup = ctx.actions.len();
        let _ = waterfall(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            false, // direct-only terminal sweep for residual direct alpha
        );
        apply_actions_to_sim_balances(
            &ctx.actions[actions_before_direct_cleanup..],
            &ctx.sims,
            &mut ctx.sim_balances,
        );
        if ctx.actions.len() == start_actions {
            break;
        }
    }

    if ctx.mint_available {
        let actions_before_mixed_cleanup = ctx.actions.len();
        let mixed_last_prof = waterfall(&mut ctx.sims, &mut ctx.budget, &mut ctx.actions, true);
        apply_actions_to_sim_balances(
            &ctx.actions[actions_before_mixed_cleanup..],
            &ctx.sims,
            &mut ctx.sim_balances,
        );
        if !ctx.actions[actions_before_mixed_cleanup..].is_empty() {
            ctx.legacy_remaining = ctx.sim_balances.clone();
            if ctx.has_legacy_holdings() {
                ctx.run_phase3_recycling(mixed_last_prof);
            }
        }
        for _ in 0..2 {
            let start_actions = ctx.actions.len();
            ctx.run_phase1_sell_overpriced();
            let actions_before_direct_cleanup = ctx.actions.len();
            let _ = waterfall(&mut ctx.sims, &mut ctx.budget, &mut ctx.actions, false);
            apply_actions_to_sim_balances(
                &ctx.actions[actions_before_direct_cleanup..],
                &ctx.sims,
                &mut ctx.sim_balances,
            );
            if ctx.actions.len() == start_actions {
                break;
            }
        }
    }

    #[cfg(test)]
    assert_internal_state_matches_replay(&ctx, slot0_results, balances, susds_balance);

    ctx.actions
}

#[cfg(test)]
fn assert_internal_state_matches_replay(
    ctx: &RebalanceContext,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    balances: &HashMap<&str, f64>,
    initial_susd: f64,
) {
    let (replay_holdings, replay_cash) = super::diagnostics::replay_actions_to_portfolio_state(
        &ctx.actions,
        slot0_results,
        balances,
        initial_susd,
    );

    let cash_tol = 1e-6 * (1.0 + replay_cash.abs().max(ctx.budget.abs()));
    assert!(
        (ctx.budget - replay_cash).abs() <= cash_tol,
        "internal/replay cash mismatch: internal={:.12}, replay={:.12}, tol={:.12}",
        ctx.budget,
        replay_cash,
        cash_tol
    );

    for sim in &ctx.sims {
        let internal = ctx
            .sim_balances
            .get(sim.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let replay = replay_holdings
            .get(sim.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let tol = 1e-6 * (1.0 + internal.abs().max(replay.abs()));
        assert!(
            (internal - replay).abs() <= tol,
            "internal/replay holding mismatch: market={}, internal={:.12}, replay={:.12}, tol={:.12}",
            sim.market_name,
            internal,
            replay,
            tol
        );
    }

    const LOCAL_GRAD_EPS: f64 = 1e-6;
    const LOCAL_GRAD_TOL: f64 = 1e-6;
    let gradients = estimate_local_gradients_for_context(ctx, LOCAL_GRAD_EPS);
    println!(
        "[rebalance][post-grad] eps={:.3e} max_direct_grad={:.9} ({}) max_indirect_grad={:.9} ({})",
        gradients.eps,
        gradients.max_direct_grad,
        gradients.best_direct_label,
        gradients.max_indirect_grad,
        gradients.best_indirect_label
    );
    assert!(
        gradients.max_direct_grad <= LOCAL_GRAD_TOL,
        "post-waterfall direct local gradient still positive: grad={:.12}, best={}, tol={:.3e}",
        gradients.max_direct_grad,
        gradients.best_direct_label,
        LOCAL_GRAD_TOL
    );
    assert!(
        gradients.max_indirect_grad <= LOCAL_GRAD_TOL,
        "post-waterfall indirect local gradient still positive: grad={:.12}, best={}, tol={:.3e}",
        gradients.max_indirect_grad,
        gradients.best_indirect_label,
        LOCAL_GRAD_TOL
    );
}

#[cfg(test)]
#[derive(Clone)]
struct LocalGradientState {
    sims: Vec<PoolSim>,
    holdings: Vec<f64>,
    cash: f64,
    allow_indirect: bool,
}

#[cfg(test)]
impl LocalGradientState {
    fn ev(&self) -> f64 {
        let holdings_ev: f64 = self
            .holdings
            .iter()
            .zip(self.sims.iter())
            .map(|(held, sim)| held.max(0.0) * sim.prediction)
            .sum();
        self.cash + holdings_ev
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct LocalGradientSummary {
    eps: f64,
    max_direct_grad: f64,
    max_indirect_grad: f64,
    best_direct_label: String,
    best_indirect_label: String,
}

#[cfg(test)]
fn gradient_state_is_valid(state: &LocalGradientState) -> bool {
    if !state.cash.is_finite() || state.cash < -1e-9 {
        return false;
    }
    state
        .holdings
        .iter()
        .all(|held| held.is_finite() && *held >= -1e-9)
}

#[cfg(test)]
fn eval_direct_buy_gradient(state: &LocalGradientState, idx: usize, eps: f64) -> Option<f64> {
    if idx >= state.sims.len() {
        return None;
    }
    let cap = state.sims[idx].max_buy_tokens().max(0.0);
    if cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST {
        return None;
    }
    let (bought, cost, _) = state.sims[idx].buy_exact(amount)?;
    if bought <= DUST || !cost.is_finite() || cost > state.cash + EPS {
        return None;
    }
    Some((state.sims[idx].prediction * bought - cost) / bought.max(DUST))
}

#[cfg(test)]
fn eval_direct_sell_gradient(state: &LocalGradientState, idx: usize, eps: f64) -> Option<f64> {
    if idx >= state.sims.len() {
        return None;
    }
    let held = state.holdings[idx].max(0.0);
    let cap = held.min(state.sims[idx].max_sell_tokens().max(0.0));
    if cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST {
        return None;
    }
    let (sold, proceeds, _) = state.sims[idx].sell_exact(amount)?;
    if sold <= DUST || !proceeds.is_finite() {
        return None;
    }
    Some((proceeds - state.sims[idx].prediction * sold) / sold.max(DUST))
}

#[cfg(test)]
fn eval_mint_sell_gradient(state: &LocalGradientState, target_idx: usize, eps: f64) -> Option<f64> {
    let n = state.sims.len();
    if n < 2 || target_idx >= n {
        return None;
    }

    let mut cap = f64::INFINITY;
    for i in 0..n {
        if i == target_idx {
            continue;
        }
        cap = cap.min(state.sims[i].max_sell_tokens().max(0.0));
    }
    if !cap.is_finite() || cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST || amount > state.cash + EPS {
        return None;
    }

    let base_ev = state.ev();
    let mut trial = state.clone();
    for held in &mut trial.holdings {
        *held += amount;
    }
    trial.cash -= amount;

    for i in 0..n {
        if i == target_idx {
            continue;
        }
        let (sold, proceeds, new_price) = trial.sims[i].sell_exact(amount)?;
        if sold <= DUST || !proceeds.is_finite() {
            return None;
        }
        trial.sims[i].set_price(new_price);
        trial.holdings[i] = (trial.holdings[i] - sold).max(0.0);
        trial.cash += proceeds;
    }
    if !gradient_state_is_valid(&trial) {
        return None;
    }
    Some((trial.ev() - base_ev) / amount.max(DUST))
}

#[cfg(test)]
fn eval_buy_merge_gradient(state: &LocalGradientState, source_idx: usize, eps: f64) -> Option<f64> {
    let n = state.sims.len();
    if n < 2 || source_idx >= n {
        return None;
    }
    if state.holdings[source_idx] <= DUST {
        return None;
    }

    let mut cap = state.holdings[source_idx].max(0.0);
    for i in 0..n {
        if i == source_idx {
            continue;
        }
        cap = cap.min(state.holdings[i].max(0.0) + state.sims[i].max_buy_tokens().max(0.0));
    }
    if !cap.is_finite() || cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST {
        return None;
    }

    let base_ev = state.ev();
    let mut trial = state.clone();
    for i in 0..n {
        if i == source_idx {
            continue;
        }
        let shortfall = (amount - trial.holdings[i].max(0.0)).max(0.0);
        if shortfall <= DUST {
            continue;
        }
        let (bought, cost, new_price) = trial.sims[i].buy_exact(shortfall)?;
        if bought + EPS < shortfall || !cost.is_finite() || cost > trial.cash + EPS {
            return None;
        }
        trial.sims[i].set_price(new_price);
        trial.holdings[i] += bought;
        trial.cash -= cost;
    }

    for held in &mut trial.holdings {
        *held -= amount;
        if *held < 0.0 && *held > -1e-9 {
            *held = 0.0;
        }
    }
    trial.cash += amount;
    if !gradient_state_is_valid(&trial) {
        return None;
    }
    Some((trial.ev() - base_ev) / amount.max(DUST))
}

#[cfg(test)]
fn estimate_local_gradients(state: &LocalGradientState, eps: f64) -> LocalGradientSummary {
    let mut max_direct = f64::NEG_INFINITY;
    let mut max_indirect = f64::NEG_INFINITY;
    let mut best_direct = "none".to_string();
    let mut best_indirect = "none".to_string();

    for i in 0..state.sims.len() {
        if let Some(g) = eval_direct_buy_gradient(state, i, eps) {
            if g > max_direct {
                max_direct = g;
                best_direct = format!("direct_buy:{}", state.sims[i].market_name);
            }
        }
        if let Some(g) = eval_direct_sell_gradient(state, i, eps) {
            if g > max_direct {
                max_direct = g;
                best_direct = format!("direct_sell:{}", state.sims[i].market_name);
            }
        }
        if state.allow_indirect {
            if let Some(g) = eval_mint_sell_gradient(state, i, eps) {
                if g > max_indirect {
                    max_indirect = g;
                    best_indirect = format!("mint_sell_target:{}", state.sims[i].market_name);
                }
            }
            if let Some(g) = eval_buy_merge_gradient(state, i, eps) {
                if g > max_indirect {
                    max_indirect = g;
                    best_indirect = format!("buy_merge_source:{}", state.sims[i].market_name);
                }
            }
        }
    }

    LocalGradientSummary {
        eps,
        max_direct_grad: if max_direct.is_finite() {
            max_direct
        } else {
            0.0
        },
        max_indirect_grad: if max_indirect.is_finite() {
            max_indirect
        } else {
            0.0
        },
        best_direct_label: best_direct,
        best_indirect_label: best_indirect,
    }
}

#[cfg(test)]
fn estimate_local_gradients_for_context(ctx: &RebalanceContext, eps: f64) -> LocalGradientSummary {
    let holdings: Vec<f64> = ctx
        .sims
        .iter()
        .map(|sim| {
            ctx.sim_balances
                .get(sim.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        })
        .collect();
    let state = LocalGradientState {
        sims: ctx.sims.clone(),
        holdings,
        cash: ctx.budget,
        allow_indirect: ctx.mint_available,
    };
    estimate_local_gradients(&state, eps)
}

fn rebalance_arb_only(
    _balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<Action> {
    if !susds_balance.is_finite() {
        log_rebalance_init_error(RebalanceInitError::NonFiniteBudget { susds_balance });
        return Vec::new();
    }

    let expected_names = pooled_l1_outcome_names();
    let expected_count = expected_names.len();
    let mut sims = build_sims_without_predictions(slot0_results);
    if sims.is_empty() {
        log_rebalance_init_error(RebalanceInitError::NoEligibleSims {
            slot0_result_count: slot0_results.len(),
            prediction_count: 0,
            expected_outcome_count: expected_count,
        });
        return Vec::new();
    }
    let sim_names: HashSet<&'static str> = sims.iter().map(|s| s.market_name).collect();
    if sims.len() != expected_count || sim_names != expected_names {
        let missing_count = expected_names.difference(&sim_names).count();
        let unexpected_count = sim_names.difference(&expected_names).count();
        tracing::warn!(
            init_failure = "arb_only_incomplete_outcome_set",
            slot0_result_count = slot0_results.len(),
            sim_count = sims.len(),
            sim_unique_count = sim_names.len(),
            expected_outcome_count = expected_count,
            missing_count,
            unexpected_count,
            "rebalance initialization failed"
        );
        return Vec::new();
    }

    let mut sim_balances: BalanceMap = HashMap::new();

    let mut actions = Vec::new();
    let mut budget = susds_balance;
    {
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
        let _arb_profit = exec.execute_two_sided_complete_set_arb();
    }

    actions
}

fn rebalance_incumbent_with_mode(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
) -> Vec<Action> {
    match mode {
        RebalanceMode::Full => rebalance_full(balances, susds_balance, slot0_results),
        RebalanceMode::ArbOnly => rebalance_arb_only(balances, susds_balance, slot0_results),
    }
}

fn choose_higher_ev_plan(
    incumbent_actions: Vec<Action>,
    candidate_actions: Vec<Action>,
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> (Vec<Action>, f64, f64, RebalanceEngine) {
    let incumbent_ev =
        replay_expected_value(&incumbent_actions, slot0_results, balances, susds_balance)
            .unwrap_or(f64::NEG_INFINITY);
    let candidate_ev =
        replay_expected_value(&candidate_actions, slot0_results, balances, susds_balance)
            .unwrap_or(f64::NEG_INFINITY);

    if candidate_ev.is_finite() && !incumbent_ev.is_finite() {
        return (
            candidate_actions,
            incumbent_ev,
            candidate_ev,
            RebalanceEngine::GlobalCandidate,
        );
    }
    if incumbent_ev.is_finite() && !candidate_ev.is_finite() {
        return (
            incumbent_actions,
            incumbent_ev,
            candidate_ev,
            RebalanceEngine::Incumbent,
        );
    }
    if !incumbent_ev.is_finite() && !candidate_ev.is_finite() {
        return (
            incumbent_actions,
            incumbent_ev,
            candidate_ev,
            RebalanceEngine::Incumbent,
        );
    }

    let tol = 1e-9 * (1.0 + incumbent_ev.abs().max(candidate_ev.abs()));
    if candidate_ev > incumbent_ev + tol {
        (
            candidate_actions,
            incumbent_ev,
            candidate_ev,
            RebalanceEngine::GlobalCandidate,
        )
    } else {
        (
            incumbent_actions,
            incumbent_ev,
            candidate_ev,
            RebalanceEngine::Incumbent,
        )
    }
}

fn candidate_trade_totals(solve: &GlobalSolveResult) -> (f64, f64, f64) {
    let total_buy: f64 = solve.direct_buys.iter().copied().sum();
    let total_sell: f64 = solve.direct_sells.iter().copied().sum();
    let overlap: f64 = solve
        .direct_buys
        .iter()
        .zip(solve.direct_sells.iter())
        .map(|(buy, sell)| buy.min(*sell))
        .sum();
    (total_buy, total_sell, overlap)
}

#[cfg(test)]
#[test]
fn choose_higher_ev_plan_prefers_finite_candidate_when_incumbent_ev_is_nonfinite() {
    let balances: HashMap<&str, f64> = HashMap::new();
    let slot0_results: Vec<(Slot0Result, &'static crate::markets::MarketData)> = Vec::new();

    let incumbent_actions = vec![Action::Buy {
        market_name: "dummy",
        amount: 1.0,
        cost: f64::INFINITY,
    }];
    let candidate_actions: Vec<Action> = Vec::new();

    let (selected, incumbent_ev, candidate_ev, chosen) = choose_higher_ev_plan(
        incumbent_actions,
        candidate_actions,
        &balances,
        1.0,
        &slot0_results,
    );

    assert!(incumbent_ev.is_infinite() && incumbent_ev.is_sign_negative());
    assert!(candidate_ev.is_finite());
    assert_eq!(chosen, RebalanceEngine::GlobalCandidate);
    assert!(selected.is_empty());
}

pub fn rebalance_with_config_and_diagnostics(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    config: RebalanceConfig,
) -> (Vec<Action>, RebalanceDecisionDiagnostics) {
    match config.mode {
        RebalanceMode::ArbOnly => {
            let actions = rebalance_arb_only(balances, susds_balance, slot0_results);
            (
                actions,
                RebalanceDecisionDiagnostics {
                    incumbent_ev_after: f64::NAN,
                    candidate_ev_after: f64::NAN,
                    chosen_engine: RebalanceEngine::Incumbent,
                    candidate_optimizer: config.global.optimizer,
                    candidate_valid: false,
                    candidate_invalid_reason: None,
                    candidate_projected_grad_norm: f64::NAN,
                    candidate_coupled_residual: f64::NAN,
                    candidate_replay_cash_delta: f64::NAN,
                    candidate_replay_holdings_delta: f64::NAN,
                    candidate_solver_iters: 0,
                    candidate_line_search_trials: 0,
                    candidate_line_search_accepts: 0,
                    candidate_line_search_invalid_evals: 0,
                    candidate_line_search_rescue_attempts: 0,
                    candidate_line_search_rescue_accepts: 0,
                    candidate_feasibility_repairs: 0,
                    candidate_feasibility_hold_clamps: 0,
                    candidate_feasibility_cash_scales: 0,
                    candidate_active_dims: 0,
                    candidate_curvature_skips: 0,
                    candidate_dual_residual_norm: f64::NAN,
                    candidate_primal_restore_iters: 0,
                    candidate_net_theta: 0.0,
                    candidate_total_buy: 0.0,
                    candidate_total_sell: 0.0,
                    candidate_buy_sell_overlap: 0.0,
                },
            )
        }
        RebalanceMode::Full => {
            let incumbent_actions =
                rebalance_incumbent_with_mode(balances, susds_balance, slot0_results, config.mode);
            let incumbent_ev_after =
                replay_expected_value(&incumbent_actions, slot0_results, balances, susds_balance)
                    .unwrap_or(f64::NEG_INFINITY);

            match config.engine {
                RebalanceEngine::Incumbent => (
                    incumbent_actions,
                    RebalanceDecisionDiagnostics {
                        incumbent_ev_after,
                        candidate_ev_after: f64::NEG_INFINITY,
                        chosen_engine: RebalanceEngine::Incumbent,
                        candidate_optimizer: config.global.optimizer,
                        candidate_valid: false,
                        candidate_invalid_reason: None,
                        candidate_projected_grad_norm: f64::NAN,
                        candidate_coupled_residual: f64::NAN,
                        candidate_replay_cash_delta: f64::NAN,
                        candidate_replay_holdings_delta: f64::NAN,
                        candidate_solver_iters: 0,
                        candidate_line_search_trials: 0,
                        candidate_line_search_accepts: 0,
                        candidate_line_search_invalid_evals: 0,
                        candidate_line_search_rescue_attempts: 0,
                        candidate_line_search_rescue_accepts: 0,
                        candidate_feasibility_repairs: 0,
                        candidate_feasibility_hold_clamps: 0,
                        candidate_feasibility_cash_scales: 0,
                        candidate_active_dims: 0,
                        candidate_curvature_skips: 0,
                        candidate_dual_residual_norm: f64::NAN,
                        candidate_primal_restore_iters: 0,
                        candidate_net_theta: 0.0,
                        candidate_total_buy: 0.0,
                        candidate_total_sell: 0.0,
                        candidate_buy_sell_overlap: 0.0,
                    },
                ),
                RebalanceEngine::GlobalCandidate => {
                    if let Some(candidate) = build_global_candidate_plan(
                        balances,
                        susds_balance,
                        slot0_results,
                        None,
                        config.global,
                    ) {
                        let candidate_ev_after = replay_expected_value(
                            &candidate.actions,
                            slot0_results,
                            balances,
                            susds_balance,
                        )
                        .unwrap_or(f64::NEG_INFINITY);
                        let (candidate_total_buy, candidate_total_sell, candidate_buy_sell_overlap) =
                            candidate_trade_totals(&candidate.solve);

                        if candidate.candidate_valid {
                            tracing::info!(
                                candidate_pg_norm = candidate.solve.projected_grad_norm,
                                candidate_coupled_residual = candidate.solve.coupled_residual,
                                candidate_optimizer = ?candidate.solve.optimizer,
                                candidate_active_dims = candidate.solve.active_dims,
                                candidate_curvature_skips = candidate.solve.curvature_skips,
                                replay_cash_delta = candidate.replay_cash_delta,
                                replay_holdings_delta = candidate.replay_holdings_delta,
                                solver_iters = candidate.solve.outer_iters,
                                line_search_trials = candidate.solve.line_search_trials,
                                line_search_accepts = candidate.solve.line_search_accepts,
                                line_search_invalid_evals = candidate.solve.line_search_invalid_evals,
                                line_search_rescue_attempts = candidate.solve.line_search_rescue_attempts,
                                line_search_rescue_accepts = candidate.solve.line_search_rescue_accepts,
                                feasibility_repairs = candidate.solve.feasibility_repairs,
                                feasibility_hold_clamps = candidate.solve.feasibility_hold_clamps,
                                feasibility_cash_scales = candidate.solve.feasibility_cash_scales,
                                incumbent_ev_after,
                                candidate_ev_after,
                                chosen_engine = ?RebalanceEngine::GlobalCandidate,
                                "global candidate rebalance decision"
                            );
                            (
                                candidate.actions,
                                RebalanceDecisionDiagnostics {
                                    incumbent_ev_after,
                                    candidate_ev_after,
                                    chosen_engine: RebalanceEngine::GlobalCandidate,
                                    candidate_optimizer: candidate.solve.optimizer,
                                    candidate_valid: true,
                                    candidate_invalid_reason: None,
                                    candidate_projected_grad_norm: candidate
                                        .solve
                                        .projected_grad_norm,
                                    candidate_coupled_residual: candidate.solve.coupled_residual,
                                    candidate_replay_cash_delta: candidate.replay_cash_delta,
                                    candidate_replay_holdings_delta: candidate
                                        .replay_holdings_delta,
                                    candidate_solver_iters: candidate.solve.outer_iters,
                                    candidate_line_search_trials: candidate
                                        .solve
                                        .line_search_trials,
                                    candidate_line_search_accepts: candidate
                                        .solve
                                        .line_search_accepts,
                                    candidate_line_search_invalid_evals: candidate
                                        .solve
                                        .line_search_invalid_evals,
                                    candidate_line_search_rescue_attempts: candidate
                                        .solve
                                        .line_search_rescue_attempts,
                                    candidate_line_search_rescue_accepts: candidate
                                        .solve
                                        .line_search_rescue_accepts,
                                    candidate_feasibility_repairs: candidate
                                        .solve
                                        .feasibility_repairs,
                                    candidate_feasibility_hold_clamps: candidate
                                        .solve
                                        .feasibility_hold_clamps,
                                    candidate_feasibility_cash_scales: candidate
                                        .solve
                                        .feasibility_cash_scales,
                                    candidate_active_dims: candidate.solve.active_dims,
                                    candidate_curvature_skips: candidate.solve.curvature_skips,
                                    candidate_dual_residual_norm: candidate.solve.dual_residual_norm,
                                    candidate_primal_restore_iters: candidate
                                        .solve
                                        .primal_restore_iters,
                                    candidate_net_theta: candidate.solve.net_complete_set,
                                    candidate_total_buy,
                                    candidate_total_sell,
                                    candidate_buy_sell_overlap,
                                },
                            )
                        } else {
                            tracing::warn!(
                                candidate_pg_norm = candidate.solve.projected_grad_norm,
                                candidate_coupled_residual = candidate.solve.coupled_residual,
                                candidate_optimizer = ?candidate.solve.optimizer,
                                candidate_active_dims = candidate.solve.active_dims,
                                candidate_curvature_skips = candidate.solve.curvature_skips,
                                replay_cash_delta = candidate.replay_cash_delta,
                                replay_holdings_delta = candidate.replay_holdings_delta,
                                solver_iters = candidate.solve.outer_iters,
                                line_search_trials = candidate.solve.line_search_trials,
                                line_search_accepts = candidate.solve.line_search_accepts,
                                line_search_invalid_evals = candidate.solve.line_search_invalid_evals,
                                line_search_rescue_attempts = candidate.solve.line_search_rescue_attempts,
                                line_search_rescue_accepts = candidate.solve.line_search_rescue_accepts,
                                feasibility_repairs = candidate.solve.feasibility_repairs,
                                feasibility_hold_clamps = candidate.solve.feasibility_hold_clamps,
                                feasibility_cash_scales = candidate.solve.feasibility_cash_scales,
                                invalid_reason = ?candidate.invalid_reason,
                                "global candidate invalid; falling back to incumbent"
                            );
                            (
                                incumbent_actions,
                                RebalanceDecisionDiagnostics {
                                    incumbent_ev_after,
                                    candidate_ev_after,
                                    chosen_engine: RebalanceEngine::Incumbent,
                                    candidate_optimizer: candidate.solve.optimizer,
                                    candidate_valid: false,
                                    candidate_invalid_reason: candidate.invalid_reason,
                                    candidate_projected_grad_norm: candidate
                                        .solve
                                        .projected_grad_norm,
                                    candidate_coupled_residual: candidate.solve.coupled_residual,
                                    candidate_replay_cash_delta: candidate.replay_cash_delta,
                                    candidate_replay_holdings_delta: candidate
                                        .replay_holdings_delta,
                                    candidate_solver_iters: candidate.solve.outer_iters,
                                    candidate_line_search_trials: candidate
                                        .solve
                                        .line_search_trials,
                                    candidate_line_search_accepts: candidate
                                        .solve
                                        .line_search_accepts,
                                    candidate_line_search_invalid_evals: candidate
                                        .solve
                                        .line_search_invalid_evals,
                                    candidate_line_search_rescue_attempts: candidate
                                        .solve
                                        .line_search_rescue_attempts,
                                    candidate_line_search_rescue_accepts: candidate
                                        .solve
                                        .line_search_rescue_accepts,
                                    candidate_feasibility_repairs: candidate
                                        .solve
                                        .feasibility_repairs,
                                    candidate_feasibility_hold_clamps: candidate
                                        .solve
                                        .feasibility_hold_clamps,
                                    candidate_feasibility_cash_scales: candidate
                                        .solve
                                        .feasibility_cash_scales,
                                    candidate_active_dims: candidate.solve.active_dims,
                                    candidate_curvature_skips: candidate.solve.curvature_skips,
                                    candidate_dual_residual_norm: candidate.solve.dual_residual_norm,
                                    candidate_primal_restore_iters: candidate
                                        .solve
                                        .primal_restore_iters,
                                    candidate_net_theta: candidate.solve.net_complete_set,
                                    candidate_total_buy,
                                    candidate_total_sell,
                                    candidate_buy_sell_overlap,
                                },
                            )
                        }
                    } else {
                        tracing::warn!("global candidate unavailable; falling back to incumbent");
                        (
                            incumbent_actions,
                            RebalanceDecisionDiagnostics {
                                incumbent_ev_after,
                                candidate_ev_after: f64::NEG_INFINITY,
                                chosen_engine: RebalanceEngine::Incumbent,
                                candidate_optimizer: config.global.optimizer,
                                candidate_valid: false,
                                candidate_invalid_reason: None,
                                candidate_projected_grad_norm: f64::NAN,
                                candidate_coupled_residual: f64::NAN,
                                candidate_replay_cash_delta: f64::NAN,
                                candidate_replay_holdings_delta: f64::NAN,
                                candidate_solver_iters: 0,
                                candidate_line_search_trials: 0,
                                candidate_line_search_accepts: 0,
                                candidate_line_search_invalid_evals: 0,
                                candidate_line_search_rescue_attempts: 0,
                                candidate_line_search_rescue_accepts: 0,
                                candidate_feasibility_repairs: 0,
                                candidate_feasibility_hold_clamps: 0,
                                candidate_feasibility_cash_scales: 0,
                                candidate_active_dims: 0,
                                candidate_curvature_skips: 0,
                                candidate_dual_residual_norm: f64::NAN,
                                candidate_primal_restore_iters: 0,
                                candidate_net_theta: 0.0,
                                candidate_total_buy: 0.0,
                                candidate_total_sell: 0.0,
                                candidate_buy_sell_overlap: 0.0,
                            },
                        )
                    }
                }
                RebalanceEngine::AutoBestReplay => {
                    if let Some(candidate) = build_global_candidate_plan(
                        balances,
                        susds_balance,
                        slot0_results,
                        Some(&incumbent_actions),
                        config.global,
                    ) {
                        if candidate.candidate_valid {
                            let (candidate_total_buy, candidate_total_sell, candidate_buy_sell_overlap) =
                                candidate_trade_totals(&candidate.solve);
                            let (selected, inc_ev, cand_ev, chosen) = choose_higher_ev_plan(
                                incumbent_actions,
                                candidate.actions,
                                balances,
                                susds_balance,
                                slot0_results,
                            );
                            tracing::info!(
                                incumbent_ev_after = inc_ev,
                                candidate_ev_after = cand_ev,
                                chosen_engine = ?chosen,
                                candidate_pg_norm = candidate.solve.projected_grad_norm,
                                candidate_coupled_residual = candidate.solve.coupled_residual,
                                candidate_optimizer = ?candidate.solve.optimizer,
                                candidate_active_dims = candidate.solve.active_dims,
                                candidate_curvature_skips = candidate.solve.curvature_skips,
                                solver_iters = candidate.solve.outer_iters,
                                line_search_trials = candidate.solve.line_search_trials,
                                line_search_accepts = candidate.solve.line_search_accepts,
                                line_search_invalid_evals = candidate.solve.line_search_invalid_evals,
                                line_search_rescue_attempts = candidate.solve.line_search_rescue_attempts,
                                line_search_rescue_accepts = candidate.solve.line_search_rescue_accepts,
                                feasibility_repairs = candidate.solve.feasibility_repairs,
                                feasibility_hold_clamps = candidate.solve.feasibility_hold_clamps,
                                feasibility_cash_scales = candidate.solve.feasibility_cash_scales,
                                "auto-best replay decision"
                            );
                            (
                                selected,
                                RebalanceDecisionDiagnostics {
                                    incumbent_ev_after: inc_ev,
                                    candidate_ev_after: cand_ev,
                                    chosen_engine: chosen,
                                    candidate_optimizer: candidate.solve.optimizer,
                                    candidate_valid: true,
                                    candidate_invalid_reason: None,
                                    candidate_projected_grad_norm: candidate
                                        .solve
                                        .projected_grad_norm,
                                    candidate_coupled_residual: candidate.solve.coupled_residual,
                                    candidate_replay_cash_delta: candidate.replay_cash_delta,
                                    candidate_replay_holdings_delta: candidate
                                        .replay_holdings_delta,
                                    candidate_solver_iters: candidate.solve.outer_iters,
                                    candidate_line_search_trials: candidate
                                        .solve
                                        .line_search_trials,
                                    candidate_line_search_accepts: candidate
                                        .solve
                                        .line_search_accepts,
                                    candidate_line_search_invalid_evals: candidate
                                        .solve
                                        .line_search_invalid_evals,
                                    candidate_line_search_rescue_attempts: candidate
                                        .solve
                                        .line_search_rescue_attempts,
                                    candidate_line_search_rescue_accepts: candidate
                                        .solve
                                        .line_search_rescue_accepts,
                                    candidate_feasibility_repairs: candidate
                                        .solve
                                        .feasibility_repairs,
                                    candidate_feasibility_hold_clamps: candidate
                                        .solve
                                        .feasibility_hold_clamps,
                                    candidate_feasibility_cash_scales: candidate
                                        .solve
                                        .feasibility_cash_scales,
                                    candidate_active_dims: candidate.solve.active_dims,
                                    candidate_curvature_skips: candidate.solve.curvature_skips,
                                    candidate_dual_residual_norm: candidate.solve.dual_residual_norm,
                                    candidate_primal_restore_iters: candidate
                                        .solve
                                        .primal_restore_iters,
                                    candidate_net_theta: candidate.solve.net_complete_set,
                                    candidate_total_buy,
                                    candidate_total_sell,
                                    candidate_buy_sell_overlap,
                                },
                            )
                        } else {
                            let (candidate_total_buy, candidate_total_sell, candidate_buy_sell_overlap) =
                                candidate_trade_totals(&candidate.solve);
                            tracing::warn!(
                                candidate_pg_norm = candidate.solve.projected_grad_norm,
                                candidate_coupled_residual = candidate.solve.coupled_residual,
                                candidate_optimizer = ?candidate.solve.optimizer,
                                candidate_active_dims = candidate.solve.active_dims,
                                candidate_curvature_skips = candidate.solve.curvature_skips,
                                replay_cash_delta = candidate.replay_cash_delta,
                                replay_holdings_delta = candidate.replay_holdings_delta,
                                solver_iters = candidate.solve.outer_iters,
                                line_search_trials = candidate.solve.line_search_trials,
                                line_search_accepts = candidate.solve.line_search_accepts,
                                line_search_invalid_evals = candidate.solve.line_search_invalid_evals,
                                line_search_rescue_attempts = candidate.solve.line_search_rescue_attempts,
                                line_search_rescue_accepts = candidate.solve.line_search_rescue_accepts,
                                feasibility_repairs = candidate.solve.feasibility_repairs,
                                feasibility_hold_clamps = candidate.solve.feasibility_hold_clamps,
                                feasibility_cash_scales = candidate.solve.feasibility_cash_scales,
                                invalid_reason = ?candidate.invalid_reason,
                                "auto-best replay candidate invalid; using incumbent"
                            );
                            (
                                incumbent_actions,
                                RebalanceDecisionDiagnostics {
                                    incumbent_ev_after,
                                    candidate_ev_after: f64::NEG_INFINITY,
                                    chosen_engine: RebalanceEngine::Incumbent,
                                    candidate_optimizer: candidate.solve.optimizer,
                                    candidate_valid: false,
                                    candidate_invalid_reason: candidate.invalid_reason,
                                    candidate_projected_grad_norm: candidate
                                        .solve
                                        .projected_grad_norm,
                                    candidate_coupled_residual: candidate.solve.coupled_residual,
                                    candidate_replay_cash_delta: candidate.replay_cash_delta,
                                    candidate_replay_holdings_delta: candidate
                                        .replay_holdings_delta,
                                    candidate_solver_iters: candidate.solve.outer_iters,
                                    candidate_line_search_trials: candidate
                                        .solve
                                        .line_search_trials,
                                    candidate_line_search_accepts: candidate
                                        .solve
                                        .line_search_accepts,
                                    candidate_line_search_invalid_evals: candidate
                                        .solve
                                        .line_search_invalid_evals,
                                    candidate_line_search_rescue_attempts: candidate
                                        .solve
                                        .line_search_rescue_attempts,
                                    candidate_line_search_rescue_accepts: candidate
                                        .solve
                                        .line_search_rescue_accepts,
                                    candidate_feasibility_repairs: candidate
                                        .solve
                                        .feasibility_repairs,
                                    candidate_feasibility_hold_clamps: candidate
                                        .solve
                                        .feasibility_hold_clamps,
                                    candidate_feasibility_cash_scales: candidate
                                        .solve
                                        .feasibility_cash_scales,
                                    candidate_active_dims: candidate.solve.active_dims,
                                    candidate_curvature_skips: candidate.solve.curvature_skips,
                                    candidate_dual_residual_norm: candidate.solve.dual_residual_norm,
                                    candidate_primal_restore_iters: candidate
                                        .solve
                                        .primal_restore_iters,
                                    candidate_net_theta: candidate.solve.net_complete_set,
                                    candidate_total_buy,
                                    candidate_total_sell,
                                    candidate_buy_sell_overlap,
                                },
                            )
                        }
                    } else {
                        tracing::warn!("auto-best replay candidate unavailable; using incumbent");
                        (
                            incumbent_actions,
                            RebalanceDecisionDiagnostics {
                                incumbent_ev_after,
                                candidate_ev_after: f64::NEG_INFINITY,
                                chosen_engine: RebalanceEngine::Incumbent,
                                candidate_optimizer: config.global.optimizer,
                                candidate_valid: false,
                                candidate_invalid_reason: None,
                                candidate_projected_grad_norm: f64::NAN,
                                candidate_coupled_residual: f64::NAN,
                                candidate_replay_cash_delta: f64::NAN,
                                candidate_replay_holdings_delta: f64::NAN,
                                candidate_solver_iters: 0,
                                candidate_line_search_trials: 0,
                                candidate_line_search_accepts: 0,
                                candidate_line_search_invalid_evals: 0,
                                candidate_line_search_rescue_attempts: 0,
                                candidate_line_search_rescue_accepts: 0,
                                candidate_feasibility_repairs: 0,
                                candidate_feasibility_hold_clamps: 0,
                                candidate_feasibility_cash_scales: 0,
                                candidate_active_dims: 0,
                                candidate_curvature_skips: 0,
                                candidate_dual_residual_norm: f64::NAN,
                                candidate_primal_restore_iters: 0,
                                candidate_net_theta: 0.0,
                                candidate_total_buy: 0.0,
                                candidate_total_sell: 0.0,
                                candidate_buy_sell_overlap: 0.0,
                            },
                        )
                    }
                }
            }
        }
    }
}

pub fn rebalance_with_config(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    config: RebalanceConfig,
) -> Vec<Action> {
    rebalance_with_config_and_diagnostics(balances, susds_balance, slot0_results, config).0
}

/// Computes optimal trades for L1 markets using the requested mode.
pub fn rebalance_with_mode(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
) -> Vec<Action> {
    rebalance_with_config(
        balances,
        susds_balance,
        slot0_results,
        RebalanceConfig {
            mode,
            engine: RebalanceEngine::Incumbent,
            global: GlobalSolveConfig::default(),
        },
    )
}

/// Computes optimal rebalancing trades for L1 markets.
///
/// Implemented full-mode flow:
/// 0. Complete-set arbitrage pre-pass (`buy-all -> merge`) when mint routes are available
/// 1. Iterative sell-overpriced liquidation
/// 2. Waterfall allocation to equalize marginal profitability
/// 3. Legacy-inventory recycling with EV-guarded trial commits
/// 4. Bounded polish re-optimization loop (commit only when EV improves)
/// 5. Terminal cleanup sweeps (mixed + direct-only bounded passes)
pub fn rebalance(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<Action> {
    rebalance_with_mode(balances, susds_balance, slot0_results, RebalanceMode::Full)
}
