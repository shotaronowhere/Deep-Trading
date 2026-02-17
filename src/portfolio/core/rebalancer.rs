use std::collections::{HashMap, HashSet};

use crate::pools::Slot0Result;

use super::Action;
use super::sim::{
    EPS, PoolSim, SimBuildError, build_sims, build_sims_without_predictions, profitability,
    target_price_for_prof,
};
#[cfg(test)]
use super::sim::DUST;
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
        let mixed_last_prof = waterfall(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            true,
        );
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
            let _ = waterfall(
                &mut ctx.sims,
                &mut ctx.budget,
                &mut ctx.actions,
                false,
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
        let internal = ctx.sim_balances.get(sim.market_name).copied().unwrap_or(0.0).max(0.0);
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
        .map(|sim| ctx.sim_balances.get(sim.market_name).copied().unwrap_or(0.0).max(0.0))
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

/// Computes optimal trades for L1 markets using the requested mode.
pub fn rebalance_with_mode(
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

/// Computes optimal rebalancing trades for L1 markets.
///
/// 1. Sell overpriced holdings (price > prediction) via swap simulation
/// 2. Waterfall allocation: deploy capital to highest profitability outcomes,
///    equalizing profitability progressively
/// 3. Post-allocation liquidation: sell held outcomes less profitable than
///    the last bought outcome, reallocate via waterfall
pub fn rebalance(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<Action> {
    rebalance_with_mode(balances, susds_balance, slot0_results, RebalanceMode::Full)
}
