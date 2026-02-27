use std::collections::{HashMap, HashSet};

use crate::execution::GroupKind;
use crate::execution::gas::{GasAssumptions, estimate_min_gas_susd_for_group};
use crate::pools::Slot0Result;

use super::Action;
#[cfg(test)]
use super::sim::DUST;
use super::sim::{
    EPS, PoolSim, SimBuildError, build_sims, build_sims_without_predictions, profitability,
    target_price_for_prof,
};
use super::trading::{ExecutionState, portfolio_expected_value};
use super::types::BalanceMap;
use super::types::{apply_actions_to_sim_balances, lookup_balance};
use super::waterfall::{WaterfallGateStats, waterfall_with_execution_gate};

const MAX_PHASE1_ITERS: usize = 128;
const MAX_PHASE3_ITERS: usize = 8;
const PHASE3_PROF_REL_TOL: f64 = 1e-9;
const PHASE3_EV_GUARD_REL_TOL: f64 = 1e-10;
const PHASE3_ESCALATION_PROF_REL_GAP: f64 = 1e-6;
const PHASE3_ESCALATION_MIN_REMAINING_FRAC: f64 = 0.20;
const PHASE3_ESCALATION_MIN_REMAINING_ABS: f64 = 1e-6;
const MAX_POLISH_PASSES: usize = 64;
const POLISH_EV_REL_TOL: f64 = 1e-10;
const DEFAULT_GATE_BUFFER_FRAC: f64 = 0.20;
const DEFAULT_GATE_BUFFER_MIN_SUSD: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceMode {
    Full,
    ArbOnly,
}

#[derive(Debug, Clone, Copy)]
struct RouteGateThresholds {
    direct_buy: f64,
    mint_sell: f64,
    direct_sell: f64,
    buy_merge: f64,
    direct_merge: f64,
    buffer_frac: f64,
    buffer_min_susd: f64,
}

impl RouteGateThresholds {
    fn disabled() -> Self {
        Self {
            direct_buy: 0.0,
            mint_sell: 0.0,
            direct_sell: 0.0,
            buy_merge: 0.0,
            direct_merge: 0.0,
            buffer_frac: 0.0,
            buffer_min_susd: 0.0,
        }
    }

    fn is_legacy_compat_mode(&self) -> bool {
        gate_is_disabled(self.direct_buy, self.buffer_frac, self.buffer_min_susd)
            && gate_is_disabled(self.mint_sell, self.buffer_frac, self.buffer_min_susd)
            && gate_is_disabled(self.direct_sell, self.buffer_frac, self.buffer_min_susd)
            && gate_is_disabled(self.buy_merge, self.buffer_frac, self.buffer_min_susd)
            && gate_is_disabled(self.direct_merge, self.buffer_frac, self.buffer_min_susd)
    }

    fn two_sided_complete_set_arb_enabled(&self) -> bool {
        !self.is_legacy_compat_mode()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct RebalanceGateCounters {
    skipped_by_gate_direct_buy: u64,
    skipped_by_gate_mint_sell: u64,
    skipped_by_gate_direct_sell: u64,
    skipped_by_gate_buy_merge: u64,
    skipped_by_gate_direct_merge: u64,
    waterfall_steps_pruned_subgas: u64,
    phase1_candidates_skipped_subgas: u64,
    phase3_candidates_skipped_subgas: u64,
}

fn sanitize_gate_threshold(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        f64::INFINITY
    }
}

fn gate_is_disabled(gas_susd: f64, buffer_frac: f64, buffer_min_susd: f64) -> bool {
    gas_susd.is_finite()
        && gas_susd <= 0.0
        && buffer_frac.is_finite()
        && buffer_frac <= 0.0
        && buffer_min_susd.is_finite()
        && buffer_min_susd <= 0.0
}

pub(super) fn passes_execution_gate(
    edge_susd: f64,
    gas_susd: f64,
    buffer_frac: f64,
    buffer_min_susd: f64,
) -> bool {
    if gate_is_disabled(gas_susd, buffer_frac, buffer_min_susd) {
        // Preserve legacy behavior when route thresholds and buffers are disabled.
        return true;
    }
    if !edge_susd.is_finite() || edge_susd <= 0.0 {
        return false;
    }
    if !gas_susd.is_finite() || gas_susd < 0.0 {
        return false;
    }
    let effective_frac = if buffer_frac.is_finite() {
        buffer_frac.max(0.0)
    } else {
        0.0
    };
    let effective_min = if buffer_min_susd.is_finite() {
        buffer_min_susd.max(0.0)
    } else {
        0.0
    };
    let profit_buffer = effective_min.max(effective_frac * edge_susd);
    edge_susd > gas_susd + profit_buffer + EPS
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
    route_gates: RouteGateThresholds,
    gate_counters: RebalanceGateCounters,
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
        allow_buy_merge: bool,
        allow_direct_merge: bool,
    ) -> f64 {
        let mut exec = ExecutionState::new(
            &mut self.sims,
            &mut self.budget,
            &mut self.actions,
            &mut self.balances,
        );
        exec.execute_optimal_sell_with_merge_gates(
            source_idx,
            sell_amount,
            inventory_keep_prof,
            mint_available,
            allow_buy_merge,
            allow_direct_merge,
        )
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
            route_gates: RouteGateThresholds::disabled(),
            gate_counters: RebalanceGateCounters::default(),
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

    fn merge_waterfall_gate_stats(&mut self, stats: WaterfallGateStats) {
        self.gate_counters.skipped_by_gate_direct_buy += stats.skipped_direct as u64;
        self.gate_counters.skipped_by_gate_mint_sell += stats.skipped_mint as u64;
        self.gate_counters.waterfall_steps_pruned_subgas += stats.steps_pruned_subgas as u64;
    }

    fn log_gate_summary(&self) {
        tracing::info!(
            skipped_by_gate_direct_buy = self.gate_counters.skipped_by_gate_direct_buy,
            skipped_by_gate_mint_sell = self.gate_counters.skipped_by_gate_mint_sell,
            skipped_by_gate_direct_sell = self.gate_counters.skipped_by_gate_direct_sell,
            skipped_by_gate_buy_merge = self.gate_counters.skipped_by_gate_buy_merge,
            skipped_by_gate_direct_merge = self.gate_counters.skipped_by_gate_direct_merge,
            waterfall_steps_pruned_subgas = self.gate_counters.waterfall_steps_pruned_subgas,
            phase1_candidates_skipped_subgas = self.gate_counters.phase1_candidates_skipped_subgas,
            phase3_candidates_skipped_subgas = self.gate_counters.phase3_candidates_skipped_subgas,
            "rebalance gate summary"
        );
    }

    fn phase0_route_kind(&self, two_sided: bool) -> Option<GroupKind> {
        let price_sum: f64 = self.sims.iter().map(|s| s.price()).sum();
        if !price_sum.is_finite() {
            return None;
        }
        if price_sum < 1.0 - EPS {
            Some(GroupKind::BuyMerge)
        } else if two_sided && price_sum > 1.0 + EPS {
            Some(GroupKind::MintSell)
        } else {
            None
        }
    }

    fn estimate_phase0_edge_susd(&self, two_sided: bool) -> Option<(GroupKind, f64)> {
        let route_kind = self.phase0_route_kind(two_sided)?;
        let mut trial_sims = self.sims.clone();
        let mut trial_budget = self.budget;
        let mut trial_actions = Vec::new();
        let mut trial_balances: BalanceMap = HashMap::new();
        let mut exec = ExecutionState::new(
            &mut trial_sims,
            &mut trial_budget,
            &mut trial_actions,
            &mut trial_balances,
        );
        let edge_susd = if two_sided {
            exec.execute_two_sided_complete_set_arb()
        } else {
            exec.execute_complete_set_arb()
        };
        if edge_susd.is_finite() && edge_susd > 0.0 {
            Some((route_kind, edge_susd))
        } else {
            None
        }
    }

    fn route_gate_threshold(&self, kind: GroupKind) -> f64 {
        match kind {
            GroupKind::DirectBuy => self.route_gates.direct_buy,
            GroupKind::MintSell => self.route_gates.mint_sell,
            GroupKind::DirectSell => self.route_gates.direct_sell,
            GroupKind::BuyMerge => self.route_gates.buy_merge,
            GroupKind::DirectMerge => self.route_gates.direct_merge,
        }
    }

    fn merge_route_gate_flags(&mut self, edge_susd: f64) -> (bool, bool) {
        let buy_merge_allowed = passes_execution_gate(
            edge_susd,
            self.route_gates.buy_merge,
            self.route_gates.buffer_frac,
            self.route_gates.buffer_min_susd,
        );
        let direct_merge_allowed = passes_execution_gate(
            edge_susd,
            self.route_gates.direct_merge,
            self.route_gates.buffer_frac,
            self.route_gates.buffer_min_susd,
        );
        if self.mint_available && !buy_merge_allowed {
            self.gate_counters.skipped_by_gate_buy_merge += 1;
        }
        if self.mint_available && !direct_merge_allowed {
            self.gate_counters.skipped_by_gate_direct_merge += 1;
        }
        (buy_merge_allowed, direct_merge_allowed)
    }

    fn run_phase0_complete_set_arb(&mut self) {
        if !self.mint_available {
            return;
        }
        let two_sided = self.route_gates.two_sided_complete_set_arb_enabled();
        let Some((route_kind, edge_susd)) = self.estimate_phase0_edge_susd(two_sided) else {
            return;
        };
        let threshold = self.route_gate_threshold(route_kind);
        if !passes_execution_gate(
            edge_susd,
            threshold,
            self.route_gates.buffer_frac,
            self.route_gates.buffer_min_susd,
        ) {
            match route_kind {
                GroupKind::MintSell => self.gate_counters.skipped_by_gate_mint_sell += 1,
                GroupKind::BuyMerge => self.gate_counters.skipped_by_gate_buy_merge += 1,
                _ => {}
            }
            return;
        }
        let mut exec = ExecutionState::new(
            &mut self.sims,
            &mut self.budget,
            &mut self.actions,
            &mut self.sim_balances,
        );
        if two_sided {
            exec.execute_two_sided_complete_set_arb();
        } else {
            // Keep zero-threshold paths behavior-compatible with legacy one-sided arb.
            exec.execute_complete_set_arb();
        }
    }

    fn execute_optimal_sell(
        &mut self,
        source_idx: usize,
        sell_amount: f64,
        inventory_keep_prof: f64,
        mint_available: bool,
        allow_buy_merge: bool,
        allow_direct_merge: bool,
    ) -> f64 {
        let mut exec = ExecutionState::new(
            &mut self.sims,
            &mut self.budget,
            &mut self.actions,
            &mut self.sim_balances,
        );
        exec.execute_optimal_sell_with_merge_gates(
            source_idx,
            sell_amount,
            inventory_keep_prof,
            mint_available,
            allow_buy_merge,
            allow_direct_merge,
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

                let approx_edge = sell_amount * (price - pred).max(0.0);
                if !passes_execution_gate(
                    approx_edge,
                    self.route_gates.direct_sell,
                    self.route_gates.buffer_frac,
                    self.route_gates.buffer_min_susd,
                ) {
                    self.gate_counters.skipped_by_gate_direct_sell += 1;
                    self.gate_counters.phase1_candidates_skipped_subgas += 1;
                    break;
                }

                let (buy_merge_allowed, direct_merge_allowed) =
                    self.merge_route_gate_flags(approx_edge);
                let sold_total = self.execute_optimal_sell(
                    i,
                    sell_amount,
                    0.0,
                    self.mint_available,
                    buy_merge_allowed,
                    direct_merge_allowed,
                );
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
                    let remaining_edge = new_held * (new_price - pred).max(0.0);
                    if !passes_execution_gate(
                        remaining_edge,
                        self.route_gates.direct_sell,
                        self.route_gates.buffer_frac,
                        self.route_gates.buffer_min_susd,
                    ) {
                        self.gate_counters.skipped_by_gate_direct_sell += 1;
                        self.gate_counters.phase1_candidates_skipped_subgas += 1;
                        break;
                    }
                    let (buy_merge_allowed, direct_merge_allowed) =
                        self.merge_route_gate_flags(remaining_edge);
                    let sold_remaining = self.execute_optimal_sell(
                        i,
                        new_held,
                        0.0,
                        self.mint_available,
                        buy_merge_allowed,
                        direct_merge_allowed,
                    );
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

                let current_prof =
                    profitability(trial.sims[idx].prediction, trial.sims[idx].price());
                // Conservative first-order edge estimate for gating; ignores slippage by design.
                let approx_edge =
                    sell_target * trial.sims[idx].price() * (phase3_prof - current_prof).max(0.0);
                if !passes_execution_gate(
                    approx_edge,
                    self.route_gates.direct_sell,
                    self.route_gates.buffer_frac,
                    self.route_gates.buffer_min_susd,
                ) {
                    self.gate_counters.skipped_by_gate_direct_sell += 1;
                    self.gate_counters.phase3_candidates_skipped_subgas += 1;
                    continue;
                }

                let (buy_merge_allowed, direct_merge_allowed) =
                    self.merge_route_gate_flags(approx_edge);
                let sold_total = trial.execute_optimal_sell(
                    idx,
                    sell_target,
                    phase3_prof,
                    self.mint_available,
                    buy_merge_allowed,
                    direct_merge_allowed,
                );
                if sold_total > EPS {
                    let legacy = trial.legacy_remaining.entry(market_name).or_insert(0.0);
                    *legacy = (*legacy - sold_total).max(0.0);

                    let remaining_legacy =
                        held_legacy(&trial.balances, &trial.legacy_remaining, market_name);
                    let post_prof =
                        profitability(trial.sims[idx].prediction, trial.sims[idx].price());
                    let prof_gap_tol = PHASE3_ESCALATION_PROF_REL_GAP
                        * (1.0 + phase3_prof.abs().max(post_prof.abs()));
                    let remaining_min = PHASE3_ESCALATION_MIN_REMAINING_ABS
                        .max(PHASE3_ESCALATION_MIN_REMAINING_FRAC * legacy_amount);
                    if remaining_legacy > remaining_min && post_prof + prof_gap_tol < phase3_prof {
                        let extra_prof =
                            profitability(trial.sims[idx].prediction, trial.sims[idx].price());
                        // Conservative first-order edge estimate for gating; ignores slippage by design.
                        let extra_edge = remaining_legacy
                            * trial.sims[idx].price()
                            * (phase3_prof - extra_prof).max(0.0);
                        if !passes_execution_gate(
                            extra_edge,
                            self.route_gates.direct_sell,
                            self.route_gates.buffer_frac,
                            self.route_gates.buffer_min_susd,
                        ) {
                            self.gate_counters.skipped_by_gate_direct_sell += 1;
                            self.gate_counters.phase3_candidates_skipped_subgas += 1;
                            continue;
                        }
                        let (buy_merge_allowed, direct_merge_allowed) =
                            self.merge_route_gate_flags(extra_edge);
                        let sold_extra = trial.execute_optimal_sell(
                            idx,
                            remaining_legacy,
                            phase3_prof,
                            self.mint_available,
                            buy_merge_allowed,
                            direct_merge_allowed,
                        );
                        if sold_extra > EPS {
                            let legacy = trial.legacy_remaining.entry(market_name).or_insert(0.0);
                            *legacy = (*legacy - sold_extra).max(0.0);
                        }
                    }
                }
            }

            let recovered_budget = trial.budget - budget_before_liq;
            if trial.actions.is_empty() || recovered_budget <= EPS || trial.budget <= EPS {
                break;
            }

            // Reallocate recovered capital and fold the acquired positions into simulated balances.
            let actions_before_realloc = trial.actions.len();
            let mut wf_stats = WaterfallGateStats::default();
            let new_prof = waterfall_with_execution_gate(
                &mut trial.sims,
                &mut trial.budget,
                &mut trial.actions,
                self.mint_available,
                self.route_gates.direct_buy,
                self.route_gates.mint_sell,
                self.route_gates.buffer_frac,
                self.route_gates.buffer_min_susd,
                Some(&mut wf_stats),
            );
            self.merge_waterfall_gate_stats(wf_stats);
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
        self.gate_counters.skipped_by_gate_direct_buy +=
            trial.gate_counters.skipped_by_gate_direct_buy;
        self.gate_counters.skipped_by_gate_mint_sell +=
            trial.gate_counters.skipped_by_gate_mint_sell;
        self.gate_counters.skipped_by_gate_direct_sell +=
            trial.gate_counters.skipped_by_gate_direct_sell;
        self.gate_counters.skipped_by_gate_buy_merge +=
            trial.gate_counters.skipped_by_gate_buy_merge;
        self.gate_counters.skipped_by_gate_direct_merge +=
            trial.gate_counters.skipped_by_gate_direct_merge;
        self.gate_counters.waterfall_steps_pruned_subgas +=
            trial.gate_counters.waterfall_steps_pruned_subgas;
        self.gate_counters.phase1_candidates_skipped_subgas +=
            trial.gate_counters.phase1_candidates_skipped_subgas;
        self.gate_counters.phase3_candidates_skipped_subgas +=
            trial.gate_counters.phase3_candidates_skipped_subgas;
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
                route_gates: self.route_gates,
                gate_counters: RebalanceGateCounters::default(),
            };

            trial.run_phase0_complete_set_arb();

            trial.run_phase1_sell_overpriced();

            let actions_before = trial.actions.len();
            let mut wf_stats = WaterfallGateStats::default();
            let last_bought_prof = waterfall_with_execution_gate(
                &mut trial.sims,
                &mut trial.budget,
                &mut trial.actions,
                trial.mint_available,
                trial.route_gates.direct_buy,
                trial.route_gates.mint_sell,
                trial.route_gates.buffer_frac,
                trial.route_gates.buffer_min_susd,
                Some(&mut wf_stats),
            );
            trial.merge_waterfall_gate_stats(wf_stats);
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
    route_gates: RouteGateThresholds,
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
    ctx.route_gates = route_gates;

    // ── Phase 0: Complete-set arbitrage (two-sided) ──
    // Execute before any discretionary rebalancing so free budget is harvested first.
    ctx.run_phase0_complete_set_arb();

    // ── Phase 1: Sell overpriced holdings ──
    ctx.run_phase1_sell_overpriced();

    // Legacy inventory available for phase-3 recycling cannot exceed current holdings after phase 1.
    let has_legacy_holdings = ctx.cap_legacy_to_current_holdings();

    // ── Phase 2: Waterfall allocation ──
    let actions_before = ctx.actions.len();
    let mut wf_stats = WaterfallGateStats::default();
    let last_bought_prof = waterfall_with_execution_gate(
        &mut ctx.sims,
        &mut ctx.budget,
        &mut ctx.actions,
        ctx.mint_available,
        ctx.route_gates.direct_buy,
        ctx.route_gates.mint_sell,
        ctx.route_gates.buffer_frac,
        ctx.route_gates.buffer_min_susd,
        Some(&mut wf_stats),
    );
    ctx.merge_waterfall_gate_stats(wf_stats);

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
    let mut cleanup_wf_stats = WaterfallGateStats::default();
    let cleanup_last_prof = waterfall_with_execution_gate(
        &mut ctx.sims,
        &mut ctx.budget,
        &mut ctx.actions,
        ctx.mint_available,
        ctx.route_gates.direct_buy,
        ctx.route_gates.mint_sell,
        ctx.route_gates.buffer_frac,
        ctx.route_gates.buffer_min_susd,
        Some(&mut cleanup_wf_stats),
    );
    ctx.merge_waterfall_gate_stats(cleanup_wf_stats);
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
        let mut direct_cleanup_wf_stats = WaterfallGateStats::default();
        let _ = waterfall_with_execution_gate(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            false, // direct-only terminal sweep for residual direct alpha
            ctx.route_gates.direct_buy,
            ctx.route_gates.mint_sell,
            ctx.route_gates.buffer_frac,
            ctx.route_gates.buffer_min_susd,
            Some(&mut direct_cleanup_wf_stats),
        );
        ctx.merge_waterfall_gate_stats(direct_cleanup_wf_stats);
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
        let mut mixed_wf_stats = WaterfallGateStats::default();
        let mixed_last_prof = waterfall_with_execution_gate(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            true,
            ctx.route_gates.direct_buy,
            ctx.route_gates.mint_sell,
            ctx.route_gates.buffer_frac,
            ctx.route_gates.buffer_min_susd,
            Some(&mut mixed_wf_stats),
        );
        ctx.merge_waterfall_gate_stats(mixed_wf_stats);
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
            let mut final_direct_wf_stats = WaterfallGateStats::default();
            let _ = waterfall_with_execution_gate(
                &mut ctx.sims,
                &mut ctx.budget,
                &mut ctx.actions,
                false,
                ctx.route_gates.direct_buy,
                ctx.route_gates.mint_sell,
                ctx.route_gates.buffer_frac,
                ctx.route_gates.buffer_min_susd,
                Some(&mut final_direct_wf_stats),
            );
            ctx.merge_waterfall_gate_stats(final_direct_wf_stats);
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

    ctx.log_gate_summary();
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

    let gates_disabled = ctx.route_gates.is_legacy_compat_mode();
    if !gates_disabled {
        return;
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

/// Computes optimal trades for L1 markets using the requested mode.
/// Gas thresholds default to 0.0 (no filtering). Use `rebalance_with_gas` for gas-aware scheduling.
pub fn rebalance_with_mode(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
) -> Vec<Action> {
    rebalance_with_mode_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        RouteGateThresholds::disabled(),
    )
}

fn rebalance_with_mode_and_thresholds(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    route_gates: RouteGateThresholds,
) -> Vec<Action> {
    match mode {
        RebalanceMode::Full => rebalance_full(balances, susds_balance, slot0_results, route_gates),
        RebalanceMode::ArbOnly => rebalance_arb_only(balances, susds_balance, slot0_results),
    }
}

/// Default gas price assumptions for minimum-trade-threshold computation.
/// Conservative: 1 gwei L2, $3000/ETH.
const THRESHOLD_GAS_PRICE_ETH: f64 = 1e-9;
const THRESHOLD_ETH_USD: f64 = 3000.0;

fn compute_gas_thresholds(
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
    cross_route_legs: usize,
) -> RouteGateThresholds {
    let direct_buy =
        estimate_min_gas_susd_for_group(gas, GroupKind::DirectBuy, 0, 0, gas_price_eth, eth_usd);
    let mint_sell = estimate_min_gas_susd_for_group(
        gas,
        GroupKind::MintSell,
        0,
        cross_route_legs,
        gas_price_eth,
        eth_usd,
    );
    let direct_sell =
        estimate_min_gas_susd_for_group(gas, GroupKind::DirectSell, 0, 0, gas_price_eth, eth_usd);
    let buy_merge = estimate_min_gas_susd_for_group(
        gas,
        GroupKind::BuyMerge,
        cross_route_legs,
        0,
        gas_price_eth,
        eth_usd,
    );
    let direct_merge =
        estimate_min_gas_susd_for_group(gas, GroupKind::DirectMerge, 0, 0, gas_price_eth, eth_usd);
    RouteGateThresholds {
        direct_buy: sanitize_gate_threshold(direct_buy),
        mint_sell: sanitize_gate_threshold(mint_sell),
        direct_sell: sanitize_gate_threshold(direct_sell),
        buy_merge: sanitize_gate_threshold(buy_merge),
        direct_merge: sanitize_gate_threshold(direct_merge),
        buffer_frac: DEFAULT_GATE_BUFFER_FRAC,
        buffer_min_susd: DEFAULT_GATE_BUFFER_MIN_SUSD,
    }
}

/// Gas-aware entry point for `main.rs`.
/// Computes per-route thresholds from `GasAssumptions` and threads them through `waterfall`.
/// The existing `rebalance_with_mode` keeps its current signature (used by all tests unchanged).
pub fn rebalance_with_gas(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    gas: &GasAssumptions,
) -> Vec<Action> {
    rebalance_with_gas_pricing(
        balances,
        susds_balance,
        slot0_results,
        mode,
        gas,
        THRESHOLD_GAS_PRICE_ETH,
        THRESHOLD_ETH_USD,
    )
}

/// Gas-aware entry point with explicit gas price and ETH/USD assumptions.
pub fn rebalance_with_gas_pricing(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
) -> Vec<Action> {
    let n_sims = slot0_results.len();
    let route_gates = compute_gas_thresholds(gas, gas_price_eth, eth_usd, n_sims.saturating_sub(1));
    rebalance_with_mode_and_thresholds(balances, susds_balance, slot0_results, mode, route_gates)
}

/// Computes optimal rebalancing trades for L1 markets.
///
/// Implemented full-mode flow:
/// 0. Complete-set arbitrage pre-pass (two-sided) when mint routes are available
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
