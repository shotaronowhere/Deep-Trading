use std::collections::{HashMap, HashSet};

use crate::pools::Slot0Result;

use super::Action;
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

    ctx.actions
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
