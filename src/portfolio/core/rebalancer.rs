use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::execution::GroupKind;
use crate::execution::gas::{GasAssumptions, estimate_min_gas_susd_for_group};
use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};

use super::Action;
use super::bundle::BundleRouteKind;
#[cfg(test)]
use super::sim::DUST;
use super::sim::{
    EPS, PoolSim, SimBuildError, build_sims, build_sims_without_predictions, profitability,
    target_price_for_prof,
};
use super::trading::{ExecutionState, portfolio_expected_value};
use super::types::BalanceMap;
use super::types::{apply_actions_to_sim_balances, lookup_balance};
use super::waterfall::{
    WaterfallGateStats, waterfall_with_execution_gate_and_forced_first_frontier_and_preserve,
    waterfall_with_execution_gate_and_preserve,
};

const MAX_PHASE1_ITERS: usize = 128;
const MAX_PHASE3_ITERS: usize = 8;
const PHASE3_PROF_REL_TOL: f64 = 1e-9;
const PHASE3_EV_GUARD_REL_TOL: f64 = 1e-10;
const PHASE3_ESCALATION_PROF_REL_GAP: f64 = 1e-6;
const PHASE3_ESCALATION_MIN_REMAINING_FRAC: f64 = 0.20;
const PHASE3_ESCALATION_MIN_REMAINING_ABS: f64 = 1e-6;
const MAX_POLISH_PASSES: usize = 64;
const POLISH_EV_REL_TOL: f64 = 1e-10;
const PRESERVE_SELECTION_EV_REL_TOL: f64 = 1e-13;
const MAX_ONLINE_PRESERVE_CANDIDATES: usize = 4;
const ARB_OPERATOR_EV_REL_TOL: f64 = 1e-10;
const MAX_META_SOLVER_INVOCATIONS: usize = 192;
const MAX_PRESERVE_SEARCH_CANDIDATES: usize = 12;
const MAX_EXACT_PRESERVE_SUBSET_CANDIDATES: usize = 6;
const MAX_PRESERVE_LOCAL_SEARCH_ROUNDS: usize = 2;
const RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS: usize = 2;
const MAX_CYCLIC_LATE_ARB_CYCLES: usize = 8;
const MAX_CYCLIC_LATE_ARB_INVOCATIONS: usize = MAX_CYCLIC_LATE_ARB_CYCLES * 2;
const CYCLIC_LATE_ARB_EV_REL_TOL: f64 = 1e-10;
const DEFAULT_GATE_BUFFER_FRAC: f64 = 0.20;
const DEFAULT_GATE_BUFFER_MIN_SUSD: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceMode {
    Full,
    ArbOnly,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebalanceFlags {
    pub enable_ev_guarded_greedy_churn_pruning: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhaseOrderVariant {
    ArbFirst,
    ArbLast,
    NoArb,
    CyclicLateArb,
}

impl PhaseOrderVariant {
    const STATIC_ALL: [Self; 3] = [Self::ArbFirst, Self::ArbLast, Self::NoArb];

    fn params(self) -> (bool, bool, bool) {
        match self {
            Self::ArbFirst => (true, false, true),
            Self::ArbLast => (false, true, false),
            Self::NoArb => (false, false, false),
            Self::CyclicLateArb => (false, false, false),
        }
    }

    fn stable_rank(self) -> usize {
        match self {
            Self::ArbFirst => 0,
            Self::ArbLast => 1,
            Self::NoArb => 2,
            Self::CyclicLateArb => 3,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ArbFirst => "arb_first",
            Self::ArbLast => "arb_last",
            Self::NoArb => "no_arb",
            Self::CyclicLateArb => "cyclic_late_arb",
        }
    }
}

fn first_frontier_family_label(family: Option<BundleRouteKind>) -> &'static str {
    match family {
        None => "default",
        Some(BundleRouteKind::Direct) => "direct",
        Some(BundleRouteKind::Mint) => "mint",
    }
}

fn first_frontier_family_stable_rank(family: Option<BundleRouteKind>) -> usize {
    match family {
        None => 0,
        Some(BundleRouteKind::Direct) => 1,
        Some(BundleRouteKind::Mint) => 2,
    }
}

#[derive(Debug, Clone, Copy)]
struct MetaSearchBudget {
    remaining: usize,
}

impl MetaSearchBudget {
    fn new(remaining: usize) -> Self {
        Self { remaining }
    }

    fn try_take(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }
}

fn with_budget_slice<T>(
    meta_budget: &mut MetaSearchBudget,
    allowance: usize,
    refine: impl FnOnce(&mut MetaSearchBudget) -> T,
) -> T {
    let granted = allowance.min(meta_budget.remaining);
    let mut slice = MetaSearchBudget::new(granted);
    let result = refine(&mut slice);
    let used = granted.saturating_sub(slice.remaining);
    meta_budget.remaining = meta_budget.remaining.saturating_sub(used);
    result
}

#[derive(Debug, Clone)]
struct CandidateResult {
    actions: Vec<Action>,
    ev: f64,
    variant: PhaseOrderVariant,
    preserve_markets: HashSet<&'static str>,
    forced_first_frontier_family: Option<BundleRouteKind>,
}

#[derive(Debug, Clone)]
struct PhaseOrderVariantEvaluation {
    baseline: CandidateResult,
    greedy_best: CandidateResult,
    preserve_candidates: Vec<&'static str>,
}

impl PhaseOrderVariantEvaluation {
    fn best_candidate(&self) -> &CandidateResult {
        if candidate_is_better(&self.greedy_best, &self.baseline) {
            &self.greedy_best
        } else {
            &self.baseline
        }
    }

    fn best_source_label(&self) -> &'static str {
        if candidate_is_better(&self.greedy_best, &self.baseline) {
            "greedy_prescan"
        } else {
            "baseline"
        }
    }
}

#[derive(Debug, Clone)]
struct RefinedCandidateResult {
    candidate: CandidateResult,
    source_label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverFamily {
    Plain,
    ArbPrimed,
}

impl SolverFamily {
    fn stable_rank(self) -> usize {
        match self {
            Self::Plain => 0,
            Self::ArbPrimed => 1,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::ArbPrimed => "arb_primed",
        }
    }
}

#[derive(Debug, Clone)]
struct SolverStateSnapshot {
    slot0_results: Vec<(Slot0Result, &'static crate::markets::MarketData)>,
    holdings: BalanceMap,
    cash: f64,
}

#[derive(Debug, Clone)]
struct PlanResult {
    actions: Vec<Action>,
    terminal_state: SolverStateSnapshot,
    ev: f64,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct SolverRunStats {
    pub(super) exact_rebalance_calls: usize,
    pub(super) exact_rebalance_candidate_evals: usize,
    pub(super) arb_operator_evals: usize,
    pub(super) arb_corrections_taken: usize,
    pub(super) arb_primed_root_taken: bool,
}

#[derive(Debug, Clone, Copy)]
struct PreserveCandidateScore {
    market_name: &'static str,
    churn_amount: f64,
    sold_amount: f64,
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
    mint_sell_preserve_markets: HashSet<&'static str>,
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
            mint_sell_preserve_markets: HashSet::new(),
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

    fn liquidation_merge_route_flags(&mut self, edge_susd: f64) -> (bool, bool) {
        let _ = edge_susd;
        if self.mint_available {
            self.gate_counters.skipped_by_gate_direct_merge += 1;
        }
        // Liquidation intentionally avoids merge-based routes for now.
        (false, false)
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
                    self.liquidation_merge_route_flags(approx_edge);
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
                        self.liquidation_merge_route_flags(remaining_edge);
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
                    self.liquidation_merge_route_flags(approx_edge);
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
                            self.liquidation_merge_route_flags(extra_edge);
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
            let new_prof = waterfall_with_execution_gate_and_preserve(
                &mut trial.sims,
                &mut trial.budget,
                &mut trial.actions,
                self.mint_available,
                self.route_gates.direct_buy,
                self.route_gates.mint_sell,
                self.route_gates.buffer_frac,
                self.route_gates.buffer_min_susd,
                Some(&self.mint_sell_preserve_markets),
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
        self.mint_sell_preserve_markets = trial.mint_sell_preserve_markets;
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

    fn run_polish_reoptimization(&mut self, run_phase0_in_polish: bool) {
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
                mint_sell_preserve_markets: self.mint_sell_preserve_markets.clone(),
                sim_balances: self.sim_balances.clone(),
                // In polish mode, recycle across current inventory (not only initial legacy).
                legacy_remaining: self.sim_balances.clone(),
                route_gates: self.route_gates,
                gate_counters: RebalanceGateCounters::default(),
            };

            if run_phase0_in_polish {
                trial.run_phase0_complete_set_arb();
            }

            trial.run_phase1_sell_overpriced();

            let actions_before = trial.actions.len();
            let mut wf_stats = WaterfallGateStats::default();
            let last_bought_prof = waterfall_with_execution_gate_and_preserve(
                &mut trial.sims,
                &mut trial.budget,
                &mut trial.actions,
                trial.mint_available,
                trial.route_gates.direct_buy,
                trial.route_gates.mint_sell,
                trial.route_gates.buffer_frac,
                trial.route_gates.buffer_min_susd,
                Some(&trial.mint_sell_preserve_markets),
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

#[cfg(test)]
fn finish_rebalance_full_inner(
    ctx: RebalanceContext,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    run_phase0_arb: bool,
    run_phase0_arb_at_end: bool,
    run_phase0_in_polish: bool,
    verify_internal_state: bool,
) -> Vec<Action> {
    finish_rebalance_full_inner_with_forced_first_frontier(
        ctx,
        slot0_results,
        balances,
        susds_balance,
        run_phase0_arb,
        run_phase0_arb_at_end,
        run_phase0_in_polish,
        None,
        verify_internal_state,
    )
    .unwrap_or_default()
}

fn finish_rebalance_full_inner_with_forced_first_frontier(
    mut ctx: RebalanceContext,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    run_phase0_arb: bool,
    run_phase0_arb_at_end: bool,
    run_phase0_in_polish: bool,
    forced_first_phase2_frontier_family: Option<BundleRouteKind>,
    verify_internal_state: bool,
) -> Option<Vec<Action>> {
    // ── Phase 0: Complete-set arbitrage (two-sided) ──
    // Execute before any discretionary rebalancing so free budget is harvested first.
    if run_phase0_arb {
        ctx.run_phase0_complete_set_arb();
    }

    // ── Phase 1: Sell overpriced holdings ──
    ctx.run_phase1_sell_overpriced();

    // Legacy inventory available for phase-3 recycling cannot exceed current holdings after phase 1.
    let has_legacy_holdings = ctx.cap_legacy_to_current_holdings();

    // ── Phase 2: Waterfall allocation ──
    let actions_before = ctx.actions.len();
    let mut wf_stats = WaterfallGateStats::default();
    let last_bought_prof = if let Some(frontier_family) = forced_first_phase2_frontier_family {
        waterfall_with_execution_gate_and_forced_first_frontier_and_preserve(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            ctx.mint_available,
            ctx.route_gates.direct_buy,
            ctx.route_gates.mint_sell,
            ctx.route_gates.buffer_frac,
            ctx.route_gates.buffer_min_susd,
            Some(&ctx.mint_sell_preserve_markets),
            frontier_family,
            Some(&mut wf_stats),
        )?
    } else {
        waterfall_with_execution_gate_and_preserve(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            ctx.mint_available,
            ctx.route_gates.direct_buy,
            ctx.route_gates.mint_sell,
            ctx.route_gates.buffer_frac,
            ctx.route_gates.buffer_min_susd,
            Some(&ctx.mint_sell_preserve_markets),
            Some(&mut wf_stats),
        )
    };
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
    ctx.run_polish_reoptimization(run_phase0_in_polish);

    // ── Phase 5: Final local cleanup pass ──
    // Run one final sell-overpriced + waterfall pass so the terminal state
    // better aligns with first-order local optimality checks used in tests.
    ctx.run_phase1_sell_overpriced();
    let actions_before_cleanup = ctx.actions.len();
    let mut cleanup_wf_stats = WaterfallGateStats::default();
    let cleanup_last_prof = waterfall_with_execution_gate_and_preserve(
        &mut ctx.sims,
        &mut ctx.budget,
        &mut ctx.actions,
        ctx.mint_available,
        ctx.route_gates.direct_buy,
        ctx.route_gates.mint_sell,
        ctx.route_gates.buffer_frac,
        ctx.route_gates.buffer_min_susd,
        Some(&ctx.mint_sell_preserve_markets),
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
        let _ = waterfall_with_execution_gate_and_preserve(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            false, // direct-only terminal sweep for residual direct alpha
            ctx.route_gates.direct_buy,
            ctx.route_gates.mint_sell,
            ctx.route_gates.buffer_frac,
            ctx.route_gates.buffer_min_susd,
            Some(&ctx.mint_sell_preserve_markets),
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
        let mixed_last_prof = waterfall_with_execution_gate_and_preserve(
            &mut ctx.sims,
            &mut ctx.budget,
            &mut ctx.actions,
            true,
            ctx.route_gates.direct_buy,
            ctx.route_gates.mint_sell,
            ctx.route_gates.buffer_frac,
            ctx.route_gates.buffer_min_susd,
            Some(&ctx.mint_sell_preserve_markets),
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
            let _ = waterfall_with_execution_gate_and_preserve(
                &mut ctx.sims,
                &mut ctx.budget,
                &mut ctx.actions,
                false,
                ctx.route_gates.direct_buy,
                ctx.route_gates.mint_sell,
                ctx.route_gates.buffer_frac,
                ctx.route_gates.buffer_min_susd,
                Some(&ctx.mint_sell_preserve_markets),
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

    if run_phase0_arb_at_end {
        ctx.run_phase0_complete_set_arb();
    }

    #[cfg(test)]
    if verify_internal_state {
        assert_internal_state_matches_replay(&ctx, slot0_results, balances, susds_balance);
    }

    #[cfg(not(test))]
    let _ = (
        slot0_results,
        balances,
        susds_balance,
        verify_internal_state,
    );

    ctx.log_gate_summary();
    Some(ctx.actions)
}

fn build_rebalance_context(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
) -> Result<RebalanceContext, RebalanceInitError> {
    let mut ctx = RebalanceContext::from_inputs(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_outcome_count,
    )?;
    ctx.route_gates = route_gates;
    Ok(ctx)
}

fn build_rebalance_context_with_options(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
) -> Result<RebalanceContext, RebalanceInitError> {
    let mut ctx = build_rebalance_context(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_outcome_count,
        route_gates,
    )?;
    if let Some(force) = force_mint_available {
        ctx.mint_available = force;
    }
    Ok(ctx)
}

fn collect_mint_sell_preserve_candidate_scores(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
    limit: usize,
) -> Vec<PreserveCandidateScore> {
    let mut sold_amounts: HashMap<&'static str, f64> = HashMap::new();
    let mut rebought_amounts: HashMap<&'static str, f64> = HashMap::new();
    let mut seen_sell: HashSet<&'static str> = HashSet::new();
    let mut sold_before_buy: HashSet<&'static str> = HashSet::new();

    for action in actions {
        match action {
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                seen_sell.insert(*market_name);
                *sold_amounts.entry(*market_name).or_insert(0.0) += amount.max(0.0);
            }
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                if seen_sell.contains(market_name) {
                    sold_before_buy.insert(*market_name);
                    *rebought_amounts.entry(*market_name).or_insert(0.0) += amount.max(0.0);
                }
            }
            _ => {}
        }
    }

    if sold_before_buy.is_empty() {
        return Vec::new();
    }

    let _ = (slot0_results, initial_balances, initial_susd);
    let mut candidates: Vec<PreserveCandidateScore> = sold_before_buy
        .into_iter()
        .filter_map(|market_name| {
            let sold_amount = sold_amounts
                .get(market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let rebought_amount = rebought_amounts
                .get(market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let churn_amount = sold_amount.min(rebought_amount);
            if churn_amount <= EPS {
                return None;
            }
            Some(PreserveCandidateScore {
                market_name,
                churn_amount,
                sold_amount,
            })
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .churn_amount
            .total_cmp(&left.churn_amount)
            .then_with(|| right.sold_amount.total_cmp(&left.sold_amount))
            .then_with(|| left.market_name.cmp(right.market_name))
    });
    candidates.truncate(limit);
    candidates
}

fn collect_mint_sell_preserve_candidates(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
    limit: usize,
) -> Vec<&'static str> {
    collect_mint_sell_preserve_candidate_scores(
        actions,
        slot0_results,
        initial_balances,
        initial_susd,
        limit,
    )
    .into_iter()
    .map(|score| score.market_name)
    .collect()
}

fn action_plan_expected_value(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
    predictions: &HashMap<String, f64>,
) -> f64 {
    let (final_holdings, final_cash) = super::diagnostics::replay_actions_to_portfolio_state(
        actions,
        slot0_results,
        initial_balances,
        initial_susd,
    );
    if !final_cash.is_finite() {
        return f64::NEG_INFINITY;
    }

    let mut ev = final_cash;
    for (_, market) in slot0_results {
        let held = final_holdings
            .get(market.name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let key = crate::pools::normalize_market_name(market.name);
        let pred = predictions.get(&key).copied().unwrap_or(0.0);
        ev += pred * held;
    }
    if ev.is_finite() {
        ev
    } else {
        f64::NEG_INFINITY
    }
}

fn build_solver_state_snapshot(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> SolverStateSnapshot {
    let mut holdings: BalanceMap = HashMap::new();
    for (_, market) in slot0_results {
        holdings.insert(market.name, lookup_balance(balances, market.name));
    }
    SolverStateSnapshot {
        slot0_results: slot0_results.to_vec(),
        holdings,
        cash: susds_balance,
    }
}

fn state_snapshot_expected_value(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
) -> f64 {
    let mut ev = state.cash;
    for (_, market) in &state.slot0_results {
        let held = state
            .holdings
            .get(market.name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let pred = predictions
            .get(&crate::pools::normalize_market_name(market.name))
            .copied()
            .unwrap_or(0.0);
        ev += pred * held;
    }
    if ev.is_finite() {
        ev
    } else {
        f64::NEG_INFINITY
    }
}

fn apply_actions_to_solver_state(
    state: &SolverStateSnapshot,
    actions: &[Action],
    predictions: &HashMap<String, f64>,
) -> Option<SolverStateSnapshot> {
    let (next_holdings, next_cash) = super::diagnostics::replay_actions_to_portfolio_state(
        actions,
        &state.slot0_results,
        &state.holdings,
        state.cash,
    );
    let next_slot0 = replay_actions_to_market_state_with_predictions(
        actions,
        &state.slot0_results,
        predictions,
    )?;
    Some(SolverStateSnapshot {
        slot0_results: next_slot0,
        holdings: next_holdings,
        cash: next_cash,
    })
}

fn sorted_preserve_markets(markets: &[&'static str]) -> Vec<&'static str> {
    let mut sorted = markets.to_vec();
    sorted.sort_unstable();
    sorted
}

fn frontier_family_candidates() -> [Option<BundleRouteKind>; 3] {
    [
        None,
        Some(BundleRouteKind::Direct),
        Some(BundleRouteKind::Mint),
    ]
}

fn solver_identity_plan(state: &SolverStateSnapshot, family: SolverFamily) -> PlanResult {
    PlanResult {
        actions: Vec::new(),
        terminal_state: state.clone(),
        ev: f64::NEG_INFINITY,
        frontier_family: None,
        preserve_markets: Vec::new(),
        family,
    }
}

fn plan_result_is_better(candidate: &PlanResult, incumbent: &PlanResult) -> bool {
    let ev_tol = PRESERVE_SELECTION_EV_REL_TOL
        * (1.0
            + candidate.ev.abs().max(incumbent.ev.abs())
            + candidate.ev.abs().min(incumbent.ev.abs()));
    if candidate.ev > incumbent.ev + ev_tol {
        return true;
    }
    if incumbent.ev > candidate.ev + ev_tol {
        return false;
    }
    if candidate.actions.len() != incumbent.actions.len() {
        return candidate.actions.len() < incumbent.actions.len();
    }
    if candidate.family != incumbent.family {
        return candidate.family.stable_rank() < incumbent.family.stable_rank();
    }
    if candidate.frontier_family != incumbent.frontier_family {
        return first_frontier_family_stable_rank(candidate.frontier_family)
            < first_frontier_family_stable_rank(incumbent.frontier_family);
    }
    if candidate.preserve_markets.len() != incumbent.preserve_markets.len() {
        return candidate.preserve_markets.len() < incumbent.preserve_markets.len();
    }
    if candidate.preserve_markets != incumbent.preserve_markets {
        return candidate.preserve_markets < incumbent.preserve_markets;
    }
    candidate.ev.total_cmp(&incumbent.ev).is_gt()
}

fn plan_result_cmp(left: &PlanResult, right: &PlanResult) -> Ordering {
    if plan_result_is_better(left, right) {
        Ordering::Less
    } else if plan_result_is_better(right, left) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn compose_rebalance_step(
    prefix_actions: &[Action],
    rebalance: &PlanResult,
    family: SolverFamily,
) -> PlanResult {
    let mut actions = prefix_actions.to_vec();
    actions.extend(rebalance.actions.iter().cloned());
    PlanResult {
        actions,
        terminal_state: rebalance.terminal_state.clone(),
        ev: rebalance.ev,
        frontier_family: rebalance.frontier_family,
        preserve_markets: rebalance.preserve_markets.clone(),
        family,
    }
}

fn compose_arb_step(prefix: &PlanResult, arb_step: &PlanResult) -> PlanResult {
    let mut actions = prefix.actions.clone();
    actions.extend(arb_step.actions.iter().cloned());
    PlanResult {
        actions,
        terminal_state: arb_step.terminal_state.clone(),
        ev: arb_step.ev,
        frontier_family: prefix.frontier_family,
        preserve_markets: prefix.preserve_markets.clone(),
        family: prefix.family,
    }
}

fn run_no_arb_rebalance_plan_from_state(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    preserve_markets: &[&'static str],
    frontier_family: Option<BundleRouteKind>,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    family: SolverFamily,
) -> Option<PlanResult> {
    let mut ctx = build_rebalance_context_with_options(
        &state.holdings,
        state.cash,
        &state.slot0_results,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
    )
    .ok()?;
    ctx.mint_sell_preserve_markets = preserve_markets.iter().copied().collect();
    let actions = finish_rebalance_full_inner_with_forced_first_frontier(
        ctx,
        &state.slot0_results,
        &state.holdings,
        state.cash,
        false,
        false,
        run_phase0_in_polish,
        frontier_family,
        verify_internal_state,
    )?;
    let terminal_state = apply_actions_to_solver_state(state, &actions, predictions)?;
    Some(PlanResult {
        actions,
        ev: state_snapshot_expected_value(&terminal_state, predictions),
        terminal_state,
        frontier_family,
        preserve_markets: sorted_preserve_markets(preserve_markets),
        family,
    })
}

fn run_positive_arb_plan_from_state(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    family: SolverFamily,
    stats: &mut SolverRunStats,
) -> Option<PlanResult> {
    stats.arb_operator_evals += 1;
    let mut ctx = build_rebalance_context_with_options(
        &state.holdings,
        state.cash,
        &state.slot0_results,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
    )
    .ok()?;
    ctx.run_phase0_complete_set_arb();
    if ctx.actions.is_empty() {
        return None;
    }

    let terminal_state = apply_actions_to_solver_state(state, &ctx.actions, predictions)?;
    let ev_before = state_snapshot_expected_value(state, predictions);
    let ev_after = state_snapshot_expected_value(&terminal_state, predictions);
    let ev_tol = ARB_OPERATOR_EV_REL_TOL * (1.0 + ev_before.abs() + ev_after.abs());
    if ev_after <= ev_before + ev_tol {
        return None;
    }

    Some(PlanResult {
        actions: ctx.actions,
        terminal_state,
        ev: ev_after,
        frontier_family: None,
        preserve_markets: Vec::new(),
        family,
    })
}

fn aggregate_preserve_candidate_scores(
    seed_plans: &[PlanResult],
    state: &SolverStateSnapshot,
) -> Vec<PreserveCandidateScore> {
    let mut aggregated: HashMap<&'static str, (f64, f64)> = HashMap::new();
    for seed in seed_plans {
        for score in collect_mint_sell_preserve_candidate_scores(
            &seed.actions,
            &state.slot0_results,
            &state.holdings,
            state.cash,
            MAX_PRESERVE_SEARCH_CANDIDATES,
        ) {
            let entry = aggregated
                .entry(score.market_name)
                .or_insert((0.0_f64, 0.0_f64));
            entry.0 = entry.0.max(score.churn_amount);
            entry.1 = entry.1.max(score.sold_amount);
        }
    }

    let mut aggregated: Vec<PreserveCandidateScore> = aggregated
        .into_iter()
        .map(
            |(market_name, (churn_amount, sold_amount))| PreserveCandidateScore {
                market_name,
                churn_amount,
                sold_amount,
            },
        )
        .collect();
    aggregated.sort_by(|left, right| {
        right
            .churn_amount
            .total_cmp(&left.churn_amount)
            .then_with(|| right.sold_amount.total_cmp(&left.sold_amount))
            .then_with(|| left.market_name.cmp(right.market_name))
    });
    aggregated
}

fn cap_preserve_candidate_universe(
    mut scores: Vec<PreserveCandidateScore>,
    limit: usize,
) -> Vec<&'static str> {
    scores.truncate(limit);
    scores.into_iter().map(|score| score.market_name).collect()
}

fn merge_preserve_candidate_scores(
    score_sets: &[Vec<PreserveCandidateScore>],
) -> Vec<PreserveCandidateScore> {
    let mut merged: HashMap<&'static str, (f64, f64)> = HashMap::new();
    for scores in score_sets {
        for score in scores {
            let entry = merged
                .entry(score.market_name)
                .or_insert((0.0_f64, 0.0_f64));
            entry.0 = entry.0.max(score.churn_amount);
            entry.1 = entry.1.max(score.sold_amount);
        }
    }

    let mut merged: Vec<PreserveCandidateScore> = merged
        .into_iter()
        .map(
            |(market_name, (churn_amount, sold_amount))| PreserveCandidateScore {
                market_name,
                churn_amount,
                sold_amount,
            },
        )
        .collect();
    merged.sort_by(|left, right| {
        right
            .churn_amount
            .total_cmp(&left.churn_amount)
            .then_with(|| right.sold_amount.total_cmp(&left.sold_amount))
            .then_with(|| left.market_name.cmp(right.market_name))
    });
    merged
}

fn enumerate_preserve_subsets(preserve_universe: &[&'static str]) -> Vec<Vec<&'static str>> {
    let subset_count = 1usize << preserve_universe.len();
    let mut subsets = Vec::with_capacity(subset_count);
    for mask in 0..subset_count {
        let mut subset = Vec::new();
        for (bit, market_name) in preserve_universe.iter().enumerate() {
            if (mask & (1usize << bit)) != 0 {
                subset.push(*market_name);
            }
        }
        subsets.push(subset);
    }
    subsets
}

fn enumerate_exact_no_arb_candidates_over_preserve_universe(
    mut candidates: Vec<PlanResult>,
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    family: SolverFamily,
    preserve_universe: &[&'static str],
    stats: &mut SolverRunStats,
) -> Vec<PlanResult> {
    if preserve_universe.is_empty() {
        candidates.sort_by(plan_result_cmp);
        return candidates;
    }

    let frontier_modes = frontier_family_candidates();
    let preserve_subsets = enumerate_preserve_subsets(preserve_universe);
    let tasks: Vec<(Option<BundleRouteKind>, Vec<&'static str>)> = frontier_modes
        .into_iter()
        .flat_map(|frontier_family| {
            preserve_subsets
                .iter()
                .filter(|subset| !subset.is_empty())
                .cloned()
                .map(move |subset| (frontier_family, subset))
        })
        .collect();
    stats.exact_rebalance_candidate_evals += tasks.len();
    let mut extra_candidates: Vec<PlanResult> = tasks
        .into_par_iter()
        .filter_map(|(frontier_family, preserve_markets)| {
            run_no_arb_rebalance_plan_from_state(
                state,
                predictions,
                expected_outcome_count,
                route_gates,
                &preserve_markets,
                frontier_family,
                force_mint_available,
                verify_internal_state,
                run_phase0_in_polish,
                family,
            )
        })
        .collect();
    candidates.append(&mut extra_candidates);
    candidates.sort_by(plan_result_cmp);
    candidates
}

fn collect_no_preserve_frontier_seed_plans(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    family: SolverFamily,
    stats: &mut SolverRunStats,
) -> Vec<PlanResult> {
    let mut seed_plans = Vec::new();
    for frontier_family in frontier_family_candidates() {
        stats.exact_rebalance_candidate_evals += 1;
        let Some(candidate) = run_no_arb_rebalance_plan_from_state(
            state,
            predictions,
            expected_outcome_count,
            route_gates,
            &[],
            frontier_family,
            force_mint_available,
            verify_internal_state,
            run_phase0_in_polish,
            family,
        ) else {
            continue;
        };
        seed_plans.push(candidate);
    }
    seed_plans
}

fn enumerate_exact_no_arb_candidates_with_options(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    family: SolverFamily,
    extra_preserve_seed_states: &[SolverStateSnapshot],
    online_preserve_cap: usize,
    stats: &mut SolverRunStats,
) -> (Vec<PlanResult>, usize) {
    stats.exact_rebalance_calls += 1;

    let candidates = collect_no_preserve_frontier_seed_plans(
        state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        run_phase0_in_polish,
        family,
        stats,
    );
    let mut seed_plans = candidates.clone();
    for seed_state in extra_preserve_seed_states {
        seed_plans.extend(collect_no_preserve_frontier_seed_plans(
            seed_state,
            predictions,
            expected_outcome_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            run_phase0_in_polish,
            family,
            stats,
        ));
    }

    let seed_scores = aggregate_preserve_candidate_scores(&seed_plans, state);
    let expansion_seed_universe =
        cap_preserve_candidate_universe(seed_scores.clone(), MAX_PRESERVE_SEARCH_CANDIDATES);
    let mut preserve_score_sets = vec![seed_scores];
    if !expansion_seed_universe.is_empty() {
        let singleton_tasks: Vec<(Option<BundleRouteKind>, &'static str)> =
            frontier_family_candidates()
                .into_iter()
                .flat_map(|frontier_family| {
                    expansion_seed_universe
                        .iter()
                        .copied()
                        .map(move |market_name| (frontier_family, market_name))
                })
                .collect();
        stats.exact_rebalance_candidate_evals += singleton_tasks.len();
        let singleton_candidates: Vec<PlanResult> = singleton_tasks
            .into_par_iter()
            .filter_map(|(frontier_family, market_name)| {
                run_no_arb_rebalance_plan_from_state(
                    state,
                    predictions,
                    expected_outcome_count,
                    route_gates,
                    &[market_name],
                    frontier_family,
                    force_mint_available,
                    verify_internal_state,
                    run_phase0_in_polish,
                    family,
                )
            })
            .collect();
        if !singleton_candidates.is_empty() {
            preserve_score_sets.push(aggregate_preserve_candidate_scores(
                &singleton_candidates,
                state,
            ));
        }
    }
    let preserve_universe = cap_preserve_candidate_universe(
        merge_preserve_candidate_scores(&preserve_score_sets),
        online_preserve_cap,
    );
    let candidates = enumerate_exact_no_arb_candidates_over_preserve_universe(
        candidates,
        state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        run_phase0_in_polish,
        family,
        &preserve_universe,
        stats,
    );

    tracing::debug!(
        family = family.as_str(),
        exact_rebalance_calls = stats.exact_rebalance_calls,
        exact_rebalance_candidate_evals = stats.exact_rebalance_candidate_evals,
        preserve_universe_size = preserve_universe.len(),
        candidate_count = candidates.len(),
        run_phase0_in_polish,
        "ultimate solver exact no-arb candidates"
    );

    (candidates, preserve_universe.len())
}

fn enumerate_exact_no_arb_candidates(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    family: SolverFamily,
    stats: &mut SolverRunStats,
) -> (Vec<PlanResult>, usize) {
    enumerate_exact_no_arb_candidates_with_options(
        state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        run_phase0_in_polish,
        family,
        &[],
        MAX_ONLINE_PRESERVE_CANDIDATES,
        stats,
    )
}

fn run_exact_no_arb_plan_with_options(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    family: SolverFamily,
    extra_preserve_seed_states: &[SolverStateSnapshot],
    online_preserve_cap: usize,
    stats: &mut SolverRunStats,
) -> PlanResult {
    let mut best = solver_identity_plan(state, family);
    best.ev = state_snapshot_expected_value(state, predictions);
    let (candidates, preserve_universe_size) = enumerate_exact_no_arb_candidates_with_options(
        state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        run_phase0_in_polish,
        family,
        extra_preserve_seed_states,
        online_preserve_cap,
        stats,
    );
    for candidate in candidates {
        if plan_result_is_better(&candidate, &best) {
            best = candidate;
        }
    }

    tracing::debug!(
        family = family.as_str(),
        exact_rebalance_calls = stats.exact_rebalance_calls,
        exact_rebalance_candidate_evals = stats.exact_rebalance_candidate_evals,
        preserve_universe_size,
        best_ev = best.ev,
        best_actions = best.actions.len(),
        best_frontier_family = first_frontier_family_label(best.frontier_family),
        best_preserve_markets = best.preserve_markets.len(),
        run_phase0_in_polish,
        online_preserve_cap,
        extra_preserve_seed_states = extra_preserve_seed_states.len(),
        "ultimate solver exact no-arb result"
    );

    best
}

fn run_exact_no_arb_plan(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    family: SolverFamily,
    stats: &mut SolverRunStats,
) -> PlanResult {
    run_exact_no_arb_plan_with_options(
        state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        run_phase0_in_polish,
        family,
        &[],
        MAX_ONLINE_PRESERVE_CANDIDATES,
        stats,
    )
}

fn maybe_apply_late_correction_tail(
    base: &PlanResult,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    run_phase0_in_polish: bool,
    stats: &mut SolverRunStats,
) -> PlanResult {
    let Some(late_arb) = run_positive_arb_plan_from_state(
        &base.terminal_state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        base.family,
        stats,
    ) else {
        return base.clone();
    };

    stats.arb_corrections_taken += 1;
    let arb_extended = compose_arb_step(base, &late_arb);
    let tail_rebalance = run_exact_no_arb_plan(
        &late_arb.terminal_state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        run_phase0_in_polish,
        base.family,
        stats,
    );

    if tail_rebalance.actions.is_empty() {
        arb_extended
    } else {
        compose_rebalance_step(&arb_extended.actions, &tail_rebalance, base.family)
    }
}

fn run_plain_family_plan(
    initial_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    stats: &mut SolverRunStats,
) -> PlanResult {
    let mut best = solver_identity_plan(initial_state, SolverFamily::Plain);
    best.ev = state_snapshot_expected_value(initial_state, predictions);
    let (base_candidates, _) = enumerate_exact_no_arb_candidates(
        initial_state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        false,
        SolverFamily::Plain,
        stats,
    );
    for base in base_candidates {
        if plan_result_is_better(&base, &best) {
            best = base.clone();
        }
        let corrected = maybe_apply_late_correction_tail(
            &base,
            predictions,
            expected_outcome_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            false,
            stats,
        );
        if plan_result_is_better(&corrected, &best) {
            best = corrected;
        }
    }
    best
}

fn run_arb_primed_family_plan(
    initial_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    stats: &mut SolverRunStats,
) -> Option<PlanResult> {
    let root_arb = run_positive_arb_plan_from_state(
        initial_state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        SolverFamily::ArbPrimed,
        stats,
    )?;
    stats.arb_primed_root_taken = true;

    let mut best = root_arb.clone();
    let (exact_rebalance_candidates, _) = enumerate_exact_no_arb_candidates(
        &root_arb.terminal_state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        true,
        SolverFamily::ArbPrimed,
        stats,
    );
    for exact_rebalance in exact_rebalance_candidates {
        let base = if exact_rebalance.actions.is_empty() {
            root_arb.clone()
        } else {
            compose_rebalance_step(&root_arb.actions, &exact_rebalance, SolverFamily::ArbPrimed)
        };
        if plan_result_is_better(&base, &best) {
            best = base.clone();
        }
        let corrected = maybe_apply_late_correction_tail(
            &base,
            predictions,
            expected_outcome_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            true,
            stats,
        );
        if plan_result_is_better(&corrected, &best) {
            best = corrected;
        }
    }

    Some(best)
}

fn rebalance_full_ultimate_with_predictions_and_stats(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    route_gates: RouteGateThresholds,
    _flags: RebalanceFlags,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
) -> (Vec<Action>, SolverRunStats) {
    let _ = _flags.enable_ev_guarded_greedy_churn_pruning;
    let mut stats = SolverRunStats::default();

    if let Err(err) = build_rebalance_context_with_options(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_count,
        route_gates,
        force_mint_available,
    ) {
        log_rebalance_init_error(err);
        return (Vec::new(), stats);
    }

    let initial_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let mut best = run_plain_family_plan(
        &initial_state,
        predictions,
        expected_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        &mut stats,
    );
    if let Some(arb_primed) = run_arb_primed_family_plan(
        &initial_state,
        predictions,
        expected_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        &mut stats,
    ) && plan_result_is_better(&arb_primed, &best)
    {
        best = arb_primed;
    }

    let staged_reference_actions = rebalance_full_with_predictions_staged_reference(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_count,
        route_gates,
        RebalanceFlags::default(),
        force_mint_available,
        verify_internal_state,
    );
    let staged_reference_ev = action_plan_expected_value(
        &staged_reference_actions,
        slot0_results,
        balances,
        susds_balance,
        predictions,
    );
    if let Some(staged_terminal_state) =
        apply_actions_to_solver_state(&initial_state, &staged_reference_actions, predictions)
    {
        let staged_reference = PlanResult {
            actions: staged_reference_actions,
            terminal_state: staged_terminal_state,
            ev: staged_reference_ev,
            frontier_family: None,
            preserve_markets: Vec::new(),
            family: SolverFamily::Plain,
        };
        if plan_result_is_better(&staged_reference, &best) {
            best = staged_reference;
        }
    }

    tracing::info!(
        chosen_family = best.family.as_str(),
        chosen_frontier_family = first_frontier_family_label(best.frontier_family),
        chosen_preserve_markets = best.preserve_markets.len(),
        final_ev = best.ev,
        final_actions = best.actions.len(),
        exact_rebalance_calls = stats.exact_rebalance_calls,
        exact_rebalance_candidate_evals = stats.exact_rebalance_candidate_evals,
        arb_operator_evals = stats.arb_operator_evals,
        arb_corrections_taken = stats.arb_corrections_taken,
        arb_primed_root_taken = stats.arb_primed_root_taken,
        "ultimate solver result"
    );

    (best.actions, stats)
}

fn candidate_is_better(candidate: &CandidateResult, incumbent: &CandidateResult) -> bool {
    let ev_tol = PRESERVE_SELECTION_EV_REL_TOL
        * (1.0
            + candidate.ev.abs().max(incumbent.ev.abs())
            + candidate.ev.abs().min(incumbent.ev.abs()));
    if candidate.ev > incumbent.ev + ev_tol {
        return true;
    }
    if incumbent.ev > candidate.ev + ev_tol {
        return false;
    }
    if candidate.actions.len() != incumbent.actions.len() {
        return candidate.actions.len() < incumbent.actions.len();
    }
    if candidate.variant != incumbent.variant {
        return candidate.variant.stable_rank() < incumbent.variant.stable_rank();
    }
    if candidate.forced_first_frontier_family != incumbent.forced_first_frontier_family {
        return first_frontier_family_stable_rank(candidate.forced_first_frontier_family)
            < first_frontier_family_stable_rank(incumbent.forced_first_frontier_family);
    }
    let candidate_preserve = sorted_preserve_markets(
        &candidate
            .preserve_markets
            .iter()
            .copied()
            .collect::<Vec<&'static str>>(),
    );
    let incumbent_preserve = sorted_preserve_markets(
        &incumbent
            .preserve_markets
            .iter()
            .copied()
            .collect::<Vec<&'static str>>(),
    );
    if candidate_preserve != incumbent_preserve {
        return candidate_preserve < incumbent_preserve;
    }
    candidate.ev.total_cmp(&incumbent.ev).is_gt()
}

fn candidate_cmp(left: &CandidateResult, right: &CandidateResult) -> Ordering {
    if candidate_is_better(left, right) {
        Ordering::Less
    } else if candidate_is_better(right, left) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn candidate_uses_mint(candidate: &CandidateResult) -> bool {
    candidate
        .actions
        .iter()
        .any(|action| matches!(action, Action::Mint { .. }))
}

fn run_rebalance_full_variant_candidate(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    variant: PhaseOrderVariant,
    preserve_markets: &HashSet<&'static str>,
    forced_first_frontier_family: Option<BundleRouteKind>,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    meta_budget: &mut MetaSearchBudget,
) -> Option<CandidateResult> {
    if !meta_budget.try_take() {
        return None;
    }

    let mut ctx = build_rebalance_context_with_options(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
    )
    .ok()?;
    ctx.mint_sell_preserve_markets = preserve_markets.clone();
    let (run_phase0_arb, run_phase0_arb_at_end, run_phase0_in_polish) = variant.params();
    let actions = finish_rebalance_full_inner_with_forced_first_frontier(
        ctx,
        slot0_results,
        balances,
        susds_balance,
        run_phase0_arb,
        run_phase0_arb_at_end,
        run_phase0_in_polish,
        forced_first_frontier_family,
        verify_internal_state,
    )?;
    let ev = action_plan_expected_value(
        &actions,
        slot0_results,
        balances,
        susds_balance,
        predictions,
    );
    Some(CandidateResult {
        actions,
        ev,
        variant,
        preserve_markets: preserve_markets.clone(),
        forced_first_frontier_family,
    })
}

fn phase_order_evaluations(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    meta_budget: &mut MetaSearchBudget,
) -> Vec<PhaseOrderVariantEvaluation> {
    let preserve_markets = HashSet::new();
    let mut evaluations = Vec::with_capacity(PhaseOrderVariant::STATIC_ALL.len());
    for variant in PhaseOrderVariant::STATIC_ALL {
        let Some(baseline) = run_rebalance_full_variant_candidate(
            balances,
            susds_balance,
            slot0_results,
            predictions,
            expected_outcome_count,
            route_gates,
            variant,
            &preserve_markets,
            None,
            force_mint_available,
            verify_internal_state,
            meta_budget,
        ) else {
            continue;
        };
        evaluations.push(PhaseOrderVariantEvaluation {
            greedy_best: baseline.clone(),
            baseline,
            preserve_candidates: Vec::new(),
        });
    }

    let mint_eval_indices: Vec<usize> = evaluations
        .iter()
        .enumerate()
        .filter_map(|(idx, evaluation)| {
            evaluation
                .baseline
                .actions
                .iter()
                .any(|action| matches!(action, Action::Mint { .. }))
                .then_some(idx)
        })
        .collect();
    for (position, eval_idx) in mint_eval_indices.iter().enumerate() {
        if meta_budget.remaining == 0 {
            break;
        }
        let remaining_variants = mint_eval_indices.len().saturating_sub(position);
        if remaining_variants == 0 {
            break;
        }
        let max_trials = meta_budget
            .remaining
            .div_ceil(remaining_variants)
            .min(MAX_PRESERVE_SEARCH_CANDIDATES);
        if max_trials == 0 {
            continue;
        }
        let evaluation = &mut evaluations[*eval_idx];
        let Some((greedy_best, preserve_candidates)) = greedy_preserve_candidate(
            &evaluation.baseline,
            balances,
            susds_balance,
            slot0_results,
            predictions,
            expected_outcome_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            max_trials,
            meta_budget,
        ) else {
            continue;
        };
        evaluation.greedy_best = greedy_best;
        evaluation.preserve_candidates = preserve_candidates;
    }

    for evaluation in &evaluations {
        tracing::debug!(
            phase_order_variant = evaluation.baseline.variant.as_str(),
            baseline_ev = evaluation.baseline.ev,
            baseline_actions = evaluation.baseline.actions.len(),
            greedy_prescan_ev = evaluation.greedy_best.ev,
            greedy_prescan_actions = evaluation.greedy_best.actions.len(),
            preserve_candidate_count = evaluation.preserve_candidates.len(),
            preserve_market_count = evaluation.greedy_best.preserve_markets.len(),
            selected_source = evaluation.best_source_label(),
            "rebalance meta-solver stage1 evaluation"
        );
    }

    evaluations
}

fn greedy_preserve_candidate(
    baseline: &CandidateResult,
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    max_trials: usize,
    meta_budget: &mut MetaSearchBudget,
) -> Option<(CandidateResult, Vec<&'static str>)> {
    let preserve_candidates = collect_mint_sell_preserve_candidates(
        &baseline.actions,
        slot0_results,
        balances,
        susds_balance,
        MAX_PRESERVE_SEARCH_CANDIDATES,
    );
    if preserve_candidates.is_empty() {
        return None;
    }

    let mut greedy_best = baseline.clone();
    let mut active_preserve: HashSet<&'static str> = HashSet::new();
    for market_name in preserve_candidates.iter().take(max_trials) {
        let mut trial_preserve = active_preserve.clone();
        trial_preserve.insert(*market_name);
        let Some(candidate) = run_rebalance_full_variant_candidate(
            balances,
            susds_balance,
            slot0_results,
            predictions,
            expected_outcome_count,
            route_gates,
            baseline.variant,
            &trial_preserve,
            None,
            force_mint_available,
            verify_internal_state,
            meta_budget,
        ) else {
            continue;
        };
        if candidate_is_better(&candidate, &greedy_best) {
            greedy_best = candidate;
            active_preserve = trial_preserve;
        }
    }

    Some((greedy_best, preserve_candidates))
}

fn exact_preserve_subset_candidate(
    baseline: &CandidateResult,
    incumbent: CandidateResult,
    preserve_candidates: &[&'static str],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    meta_budget: &mut MetaSearchBudget,
) -> CandidateResult {
    let exact_candidates: Vec<&'static str> = preserve_candidates
        .iter()
        .copied()
        .take(MAX_EXACT_PRESERVE_SUBSET_CANDIDATES)
        .collect();
    if exact_candidates.is_empty() {
        return incumbent;
    }

    let subset_count = 1usize << exact_candidates.len();
    let mut best = if candidate_is_better(&incumbent, baseline) {
        incumbent
    } else {
        baseline.clone()
    };

    for mask in 0..subset_count {
        if meta_budget.remaining <= RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS {
            break;
        }

        let mut trial_preserve: HashSet<&'static str> = HashSet::new();
        for (bit, market_name) in exact_candidates.iter().enumerate() {
            if (mask & (1usize << bit)) != 0 {
                trial_preserve.insert(*market_name);
            }
        }

        let Some(candidate) = run_rebalance_full_variant_candidate(
            balances,
            susds_balance,
            slot0_results,
            predictions,
            expected_outcome_count,
            route_gates,
            baseline.variant,
            &trial_preserve,
            None,
            force_mint_available,
            verify_internal_state,
            meta_budget,
        ) else {
            continue;
        };
        if candidate_is_better(&candidate, &best) {
            best = candidate;
        }
    }

    best
}

fn preserve_local_search_candidate(
    baseline: &CandidateResult,
    greedy_start: CandidateResult,
    preserve_candidates: &[&'static str],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    meta_budget: &mut MetaSearchBudget,
) -> CandidateResult {
    let mut current = greedy_start;
    let mut best = if candidate_is_better(&current, baseline) {
        current.clone()
    } else {
        baseline.clone()
    };

    for _ in 0..MAX_PRESERVE_LOCAL_SEARCH_ROUNDS {
        if meta_budget.remaining <= RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS {
            break;
        }
        let mut round_best: Option<CandidateResult> = None;
        let mut preserved: Vec<_> = current.preserve_markets.iter().copied().collect();
        preserved.sort_unstable();

        for removed in &preserved {
            if meta_budget.remaining <= RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS {
                break;
            }
            let mut trial_preserve = current.preserve_markets.clone();
            trial_preserve.remove(removed);
            let Some(candidate) = run_rebalance_full_variant_candidate(
                balances,
                susds_balance,
                slot0_results,
                predictions,
                expected_outcome_count,
                route_gates,
                current.variant,
                &trial_preserve,
                None,
                force_mint_available,
                verify_internal_state,
                meta_budget,
            ) else {
                continue;
            };
            if round_best
                .as_ref()
                .is_none_or(|incumbent| candidate_is_better(&candidate, incumbent))
            {
                round_best = Some(candidate);
            }
        }

        for removed in &preserved {
            for added in preserve_candidates {
                if meta_budget.remaining <= RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS {
                    break;
                }
                if current.preserve_markets.contains(added) {
                    continue;
                }
                let mut trial_preserve = current.preserve_markets.clone();
                trial_preserve.remove(removed);
                trial_preserve.insert(*added);
                let Some(candidate) = run_rebalance_full_variant_candidate(
                    balances,
                    susds_balance,
                    slot0_results,
                    predictions,
                    expected_outcome_count,
                    route_gates,
                    current.variant,
                    &trial_preserve,
                    None,
                    force_mint_available,
                    verify_internal_state,
                    meta_budget,
                ) else {
                    continue;
                };
                if round_best
                    .as_ref()
                    .is_none_or(|incumbent| candidate_is_better(&candidate, incumbent))
                {
                    round_best = Some(candidate);
                }
            }
            if meta_budget.remaining <= RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS {
                break;
            }
        }

        let Some(candidate) = round_best else {
            break;
        };
        if !candidate_is_better(&candidate, &current) {
            break;
        }
        current = candidate;
        if candidate_is_better(&current, &best) {
            best = current.clone();
        }
    }

    best
}

fn first_frontier_branch_candidate(
    base: &CandidateResult,
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    frontier_family: BundleRouteKind,
    meta_budget: &mut MetaSearchBudget,
) -> Option<CandidateResult> {
    run_rebalance_full_variant_candidate(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_outcome_count,
        route_gates,
        base.variant,
        &base.preserve_markets,
        Some(frontier_family),
        force_mint_available,
        verify_internal_state,
        meta_budget,
    )
}

fn run_arb_only_candidate(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    meta_budget: &mut MetaSearchBudget,
) -> Option<Vec<Action>> {
    if !meta_budget.try_take() {
        return None;
    }

    let mut ctx = build_rebalance_context_with_options(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
    )
    .ok()?;
    ctx.run_phase0_complete_set_arb();
    Some(ctx.actions)
}

fn replay_actions_to_market_state_with_predictions(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
) -> Option<Vec<(Slot0Result, &'static crate::markets::MarketData)>> {
    let mut sims = Vec::with_capacity(slot0_results.len());
    let mut idx_by_market = HashMap::with_capacity(slot0_results.len());

    for (slot0, market) in slot0_results {
        let pred = predictions
            .get(&crate::pools::normalize_market_name(market.name))
            .copied()
            .unwrap_or(0.0);
        let sim = PoolSim::from_slot0(slot0, market, pred)?;
        idx_by_market.insert(market.name, sims.len());
        sims.push(sim);
    }

    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name)
                    && let Some((bought, _, new_price)) = sims[idx].buy_exact(*amount)
                    && bought > 0.0
                {
                    sims[idx].set_price(new_price);
                }
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name)
                    && let Some((sold, _, new_price)) = sims[idx].sell_exact(*amount)
                    && sold > 0.0
                {
                    sims[idx].set_price(new_price);
                }
            }
            Action::Mint { .. } | Action::Merge { .. } => {}
        }
    }

    Some(
        slot0_results
            .iter()
            .map(|(slot0, market)| {
                let mut next = slot0.clone();
                if let Some(&idx) = idx_by_market.get(market.name)
                    && let Some(pool) = market.pool.as_ref()
                {
                    let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
                    let next_price = sims[idx].price().max(1e-12);
                    next.sqrt_price_x96 =
                        prediction_to_sqrt_price_x96(next_price, is_token1_outcome)
                            .unwrap_or(slot0.sqrt_price_x96);
                }
                (next, *market)
            })
            .collect(),
    )
}

fn apply_actions_to_cyclic_state(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    predictions: &HashMap<String, f64>,
) -> Option<(
    Vec<(Slot0Result, &'static crate::markets::MarketData)>,
    HashMap<&'static str, f64>,
    f64,
)> {
    let (next_balances, next_cash) = super::diagnostics::replay_actions_to_portfolio_state(
        actions,
        slot0_results,
        balances,
        susds_balance,
    );
    let next_slot0 =
        replay_actions_to_market_state_with_predictions(actions, slot0_results, predictions)?;
    Some((next_slot0, next_balances, next_cash))
}

fn cyclic_late_arb_candidate(
    seed: &CandidateResult,
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    meta_budget: &mut MetaSearchBudget,
) -> Option<CandidateResult> {
    let mut current_slot0 = slot0_results.to_vec();
    let (mut current_balances, mut current_cash) =
        super::diagnostics::replay_actions_to_portfolio_state(
            &[],
            slot0_results,
            balances,
            susds_balance,
        );
    let mut committed_actions = Vec::new();
    let mut committed_ev = action_plan_expected_value(
        &committed_actions,
        slot0_results,
        balances,
        susds_balance,
        predictions,
    );

    for cycle_idx in 0..MAX_CYCLIC_LATE_ARB_CYCLES {
        if meta_budget.remaining == 0 {
            break;
        }

        let mut cycle_actions = Vec::new();
        let mut trial_slot0 = current_slot0.clone();
        let mut trial_balances = current_balances.clone();
        let mut trial_cash = current_cash;

        let Some(rebalance_candidate) = run_rebalance_full_variant_candidate(
            &trial_balances,
            trial_cash,
            &trial_slot0,
            predictions,
            expected_outcome_count,
            route_gates,
            PhaseOrderVariant::NoArb,
            &seed.preserve_markets,
            seed.forced_first_frontier_family,
            force_mint_available,
            verify_internal_state,
            meta_budget,
        ) else {
            break;
        };
        if !rebalance_candidate.actions.is_empty() {
            cycle_actions.extend(rebalance_candidate.actions.iter().cloned());
            let Some((next_slot0, next_balances, next_cash)) = apply_actions_to_cyclic_state(
                &rebalance_candidate.actions,
                &trial_slot0,
                &trial_balances,
                trial_cash,
                predictions,
            ) else {
                break;
            };
            trial_slot0 = next_slot0;
            trial_balances = next_balances;
            trial_cash = next_cash;
        }

        let arb_actions = run_arb_only_candidate(
            &trial_balances,
            trial_cash,
            &trial_slot0,
            predictions,
            expected_outcome_count,
            route_gates,
            force_mint_available,
            meta_budget,
        )
        .unwrap_or_default();
        if !arb_actions.is_empty() {
            cycle_actions.extend(arb_actions.iter().cloned());
            let Some((next_slot0, next_balances, next_cash)) = apply_actions_to_cyclic_state(
                &arb_actions,
                &trial_slot0,
                &trial_balances,
                trial_cash,
                predictions,
            ) else {
                break;
            };
            trial_slot0 = next_slot0;
            trial_balances = next_balances;
            trial_cash = next_cash;
        }

        if cycle_actions.is_empty() {
            break;
        }

        let mut trial_actions = committed_actions.clone();
        trial_actions.extend(cycle_actions.iter().cloned());
        let trial_ev = action_plan_expected_value(
            &trial_actions,
            slot0_results,
            balances,
            susds_balance,
            predictions,
        );
        let ev_tol = CYCLIC_LATE_ARB_EV_REL_TOL * (1.0 + committed_ev.abs() + trial_ev.abs());
        tracing::debug!(
            seed_preserve_markets = seed.preserve_markets.len(),
            seed_first_frontier_family =
                first_frontier_family_label(seed.forced_first_frontier_family),
            cycle = cycle_idx + 1,
            cycle_actions = cycle_actions.len(),
            cycle_ev = trial_ev,
            committed_ev,
            "rebalance meta-solver cyclic late-arb candidate"
        );
        if trial_ev <= committed_ev + ev_tol {
            break;
        }

        committed_actions = trial_actions;
        committed_ev = trial_ev;
        current_slot0 = trial_slot0;
        current_balances = trial_balances;
        current_cash = trial_cash;
    }

    if committed_actions.is_empty() {
        return None;
    }

    Some(CandidateResult {
        actions: committed_actions,
        ev: committed_ev,
        variant: PhaseOrderVariant::CyclicLateArb,
        preserve_markets: seed.preserve_markets.clone(),
        forced_first_frontier_family: seed.forced_first_frontier_family,
    })
}

fn refine_phase_order_candidate(
    evaluation: &PhaseOrderVariantEvaluation,
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    enable_first_frontier_branch: bool,
    meta_budget: &mut MetaSearchBudget,
) -> RefinedCandidateResult {
    let mut best = evaluation.best_candidate().clone();
    let mut source_label = evaluation.best_source_label();

    if candidate_uses_mint(&best)
        && !evaluation.preserve_candidates.is_empty()
        && meta_budget.remaining > RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS
    {
        let local_candidate = preserve_local_search_candidate(
            &evaluation.baseline,
            best.clone(),
            &evaluation.preserve_candidates,
            balances,
            susds_balance,
            slot0_results,
            predictions,
            expected_outcome_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            meta_budget,
        );
        if candidate_is_better(&local_candidate, &best) {
            best = local_candidate;
            source_label = "local_search";
        }
    }

    if enable_first_frontier_branch && meta_budget.remaining > 0 {
        for frontier_family in [BundleRouteKind::Direct, BundleRouteKind::Mint] {
            let Some(candidate) = first_frontier_branch_candidate(
                &best,
                balances,
                susds_balance,
                slot0_results,
                predictions,
                expected_outcome_count,
                route_gates,
                force_mint_available,
                verify_internal_state,
                frontier_family,
                meta_budget,
            ) else {
                continue;
            };
            tracing::debug!(
                chosen_variant = candidate.variant.as_str(),
                forced_first_frontier_family =
                    first_frontier_family_label(candidate.forced_first_frontier_family),
                candidate_ev = candidate.ev,
                candidate_actions = candidate.actions.len(),
                candidate_preserve_markets = candidate.preserve_markets.len(),
                "rebalance meta-solver first-frontier branch candidate"
            );
            if candidate_is_better(&candidate, &best) {
                best = candidate;
                source_label = match frontier_family {
                    BundleRouteKind::Direct => "first_frontier_branch_direct",
                    BundleRouteKind::Mint => "first_frontier_branch_mint",
                };
            }
        }
    }

    RefinedCandidateResult {
        candidate: best,
        source_label,
    }
}

fn preserve_candidates_for_candidate(
    candidate: &CandidateResult,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
) -> Vec<&'static str> {
    collect_mint_sell_preserve_candidates(
        &candidate.actions,
        slot0_results,
        balances,
        susds_balance,
        MAX_PRESERVE_SEARCH_CANDIDATES,
    )
}

fn rebalance_full_with_predictions_and_budget_candidate(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    route_gates: RouteGateThresholds,
    _flags: RebalanceFlags,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    max_meta_solver_invocations: usize,
    enable_first_frontier_branch: bool,
) -> Option<CandidateResult> {
    // Retained for API compatibility; full-mode meta search always evaluates
    // the bounded preserve search when the winning plan contains mint activity.
    let _ = _flags.enable_ev_guarded_greedy_churn_pruning;
    let mut meta_budget = MetaSearchBudget::new(max_meta_solver_invocations);
    let phase_order_evaluations = phase_order_evaluations(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_count,
        route_gates,
        force_mint_available,
        verify_internal_state,
        &mut meta_budget,
    );
    if phase_order_evaluations.is_empty() {
        return None;
    }

    let stage1_selection = phase_order_evaluations
        .iter()
        .min_by(|left, right| candidate_cmp(left.best_candidate(), right.best_candidate()))
        .expect("checked non-empty stage1 phase-order evaluations");
    let stage1_best = stage1_selection.best_candidate().clone();
    let stage1_source = stage1_selection.best_source_label();

    let mut best = stage1_best.clone();
    let mut final_source = stage1_source;
    let mut refined_results: Vec<RefinedCandidateResult> =
        Vec::with_capacity(phase_order_evaluations.len());

    let mut refinement_order: Vec<usize> = (0..phase_order_evaluations.len()).collect();
    refinement_order.sort_by(|left, right| {
        candidate_cmp(
            phase_order_evaluations[*left].best_candidate(),
            phase_order_evaluations[*right].best_candidate(),
        )
    });
    let refinement_allowance = if refinement_order.is_empty() {
        0
    } else {
        meta_budget
            .remaining
            .div_ceil(refinement_order.len().saturating_add(1))
            .max(1)
    };

    for eval_idx in refinement_order {
        if meta_budget.remaining == 0 {
            break;
        }
        let refined = with_budget_slice(&mut meta_budget, refinement_allowance, |variant_budget| {
            refine_phase_order_candidate(
                &phase_order_evaluations[eval_idx],
                balances,
                susds_balance,
                slot0_results,
                predictions,
                expected_count,
                route_gates,
                force_mint_available,
                verify_internal_state,
                enable_first_frontier_branch,
                variant_budget,
            )
        });
        tracing::debug!(
            phase_order_variant = refined.candidate.variant.as_str(),
            refinement_source = refined.source_label,
            refined_ev = refined.candidate.ev,
            refined_actions = refined.candidate.actions.len(),
            refined_preserve_markets = refined.candidate.preserve_markets.len(),
            refined_first_frontier_family =
                first_frontier_family_label(refined.candidate.forced_first_frontier_family),
            "rebalance meta-solver phase-order refinement candidate"
        );
        refined_results.push(refined.clone());
        if candidate_is_better(&refined.candidate, &best) {
            best = refined.candidate;
            final_source = refined.source_label;
        }
    }

    if let Some(best_variant_baseline) = phase_order_evaluations
        .iter()
        .find(|evaluation| evaluation.baseline.variant == best.variant)
        .map(|evaluation| &evaluation.baseline)
    {
        let exact_preserve_candidates =
            preserve_candidates_for_candidate(&best, slot0_results, balances, susds_balance);
        let exact_subset_count = 1usize
            << exact_preserve_candidates
                .len()
                .min(MAX_EXACT_PRESERVE_SUBSET_CANDIDATES);
        if meta_budget.remaining >= exact_subset_count {
            let exact_candidate = exact_preserve_subset_candidate(
                best_variant_baseline,
                best.clone(),
                &exact_preserve_candidates,
                balances,
                susds_balance,
                slot0_results,
                predictions,
                expected_count,
                route_gates,
                force_mint_available,
                verify_internal_state,
                &mut meta_budget,
            );
            if candidate_is_better(&exact_candidate, &best) {
                best = exact_candidate;
                final_source = "exact_preserve_subset";
            }
        }
    }

    if enable_first_frontier_branch && meta_budget.remaining > 0 {
        let final_coordination = refine_phase_order_candidate(
            &PhaseOrderVariantEvaluation {
                baseline: phase_order_evaluations
                    .iter()
                    .find(|evaluation| evaluation.baseline.variant == best.variant)
                    .map(|evaluation| evaluation.baseline.clone())
                    .unwrap_or_else(|| best.clone()),
                greedy_best: best.clone(),
                preserve_candidates: preserve_candidates_for_candidate(
                    &best,
                    slot0_results,
                    balances,
                    susds_balance,
                ),
            },
            balances,
            susds_balance,
            slot0_results,
            predictions,
            expected_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            true,
            &mut meta_budget,
        );
        if candidate_is_better(&final_coordination.candidate, &best) {
            best = final_coordination.candidate;
            final_source = final_coordination.source_label;
        }
    }

    let cyclic_seed = if best.variant == PhaseOrderVariant::NoArb {
        Some(best.clone())
    } else {
        refined_results
            .iter()
            .find(|result| result.candidate.variant == PhaseOrderVariant::NoArb)
            .map(|result| result.candidate.clone())
            .or_else(|| {
                phase_order_evaluations
                    .iter()
                    .find(|evaluation| evaluation.baseline.variant == PhaseOrderVariant::NoArb)
                    .map(|evaluation| evaluation.best_candidate().clone())
            })
    };
    if let Some(seed) = cyclic_seed.as_ref() {
        let cyclic_candidate = with_budget_slice(
            &mut meta_budget,
            MAX_CYCLIC_LATE_ARB_INVOCATIONS,
            |cyclic_budget| {
                cyclic_late_arb_candidate(
                    seed,
                    balances,
                    susds_balance,
                    slot0_results,
                    predictions,
                    expected_count,
                    route_gates,
                    force_mint_available,
                    verify_internal_state,
                    cyclic_budget,
                )
            },
        );
        if let Some(candidate) = cyclic_candidate {
            tracing::debug!(
                phase_order_variant = seed.variant.as_str(),
                seed_ev = seed.ev,
                seed_actions = seed.actions.len(),
                seed_preserve_markets = seed.preserve_markets.len(),
                seed_first_frontier_family =
                    first_frontier_family_label(seed.forced_first_frontier_family),
                candidate_ev = candidate.ev,
                candidate_actions = candidate.actions.len(),
                "rebalance meta-solver cyclic late-arb candidate result"
            );
            if candidate_is_better(&candidate, &best) {
                best = candidate;
                final_source = "cyclic_late_arb";
            }
        }
    }

    tracing::info!(
        chosen_variant = best.variant.as_str(),
        chosen_first_frontier_family =
            first_frontier_family_label(best.forced_first_frontier_family),
        stage1_source,
        stage1_first_frontier_family =
            first_frontier_family_label(stage1_best.forced_first_frontier_family),
        stage1_ev = stage1_best.ev,
        stage1_actions = stage1_best.actions.len(),
        stage1_preserve_markets = stage1_best.preserve_markets.len(),
        preserve_candidate_count = stage1_selection.preserve_candidates.len(),
        final_source,
        final_ev = best.ev,
        final_actions = best.actions.len(),
        final_preserve_markets = best.preserve_markets.len(),
        final_first_frontier_family =
            first_frontier_family_label(best.forced_first_frontier_family),
        meta_budget_total = max_meta_solver_invocations,
        meta_budget_remaining = meta_budget.remaining,
        meta_budget_exhausted = meta_budget.remaining == 0,
        "rebalance meta-solver result"
    );

    Some(best)
}

fn rebalance_full_with_predictions_and_budget(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    route_gates: RouteGateThresholds,
    flags: RebalanceFlags,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    max_meta_solver_invocations: usize,
    enable_first_frontier_branch: bool,
) -> Vec<Action> {
    let Some(best) = rebalance_full_with_predictions_and_budget_candidate(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_count,
        route_gates,
        flags,
        force_mint_available,
        verify_internal_state,
        max_meta_solver_invocations,
        enable_first_frontier_branch,
    ) else {
        let err = build_rebalance_context_with_options(
            balances,
            susds_balance,
            slot0_results,
            predictions,
            expected_count,
            route_gates,
            force_mint_available,
        )
        .err()
        .unwrap_or(RebalanceInitError::NoEligibleSims {
            slot0_result_count: slot0_results.len(),
            prediction_count: predictions.len(),
            expected_outcome_count: expected_count,
        });
        log_rebalance_init_error(err);
        return Vec::new();
    };
    best.actions
}

fn rebalance_full_with_predictions_staged_reference(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    route_gates: RouteGateThresholds,
    flags: RebalanceFlags,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
) -> Vec<Action> {
    rebalance_full_with_predictions_and_budget(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_count,
        route_gates,
        flags,
        force_mint_available,
        verify_internal_state,
        MAX_META_SOLVER_INVOCATIONS,
        true,
    )
}

fn rebalance_full_with_predictions(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    route_gates: RouteGateThresholds,
    flags: RebalanceFlags,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
) -> Vec<Action> {
    rebalance_full_ultimate_with_predictions_and_stats(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_count,
        route_gates,
        flags,
        force_mint_available,
        verify_internal_state,
    )
    .0
}

fn rebalance_full(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    route_gates: RouteGateThresholds,
    flags: RebalanceFlags,
) -> Vec<Action> {
    let preds = crate::pools::prediction_map();
    let expected_count = crate::predictions::PREDICTIONS_L1.len();
    rebalance_full_with_predictions(
        balances,
        susds_balance,
        slot0_results,
        &preds,
        expected_count,
        route_gates,
        flags,
        None,
        false,
    )
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TestPhaseOrderVariant {
    ArbFirst,
    ArbLast,
    NoArb,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct StagedReferenceChoice {
    pub(super) variant_label: &'static str,
    pub(super) frontier_family: Option<BundleRouteKind>,
    pub(super) preserve_markets: Vec<&'static str>,
}

#[cfg(test)]
impl From<TestPhaseOrderVariant> for PhaseOrderVariant {
    fn from(value: TestPhaseOrderVariant) -> Self {
        match value {
            TestPhaseOrderVariant::ArbFirst => Self::ArbFirst,
            TestPhaseOrderVariant::ArbLast => Self::ArbLast,
            TestPhaseOrderVariant::NoArb => Self::NoArb,
        }
    }
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_full_with_predictions(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_and_stats_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> (Vec<Action>, SolverRunStats) {
    rebalance_full_ultimate_with_predictions_and_stats(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_staged_reference_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_full_with_predictions_staged_reference(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
    )
}

#[cfg(test)]
pub(super) fn staged_reference_choice_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Option<StagedReferenceChoice> {
    rebalance_full_with_predictions_and_budget_candidate(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
        MAX_META_SOLVER_INVOCATIONS,
        true,
    )
    .map(|best| StagedReferenceChoice {
        variant_label: best.variant.as_str(),
        frontier_family: best.forced_first_frontier_family,
        preserve_markets: sorted_preserve_markets(
            &best
                .preserve_markets
                .iter()
                .copied()
                .collect::<Vec<&'static str>>(),
        ),
    })
}

#[cfg(test)]
pub(super) fn phase_order_exact_subset_choice_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    variant: TestPhaseOrderVariant,
    force_mint_available: bool,
) -> Option<StagedReferenceChoice> {
    let mut meta_budget = MetaSearchBudget::new(MAX_META_SOLVER_INVOCATIONS);
    let preserve_markets = HashSet::new();
    let baseline = run_rebalance_full_variant_candidate(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        PhaseOrderVariant::from(variant),
        &preserve_markets,
        None,
        Some(force_mint_available),
        false,
        &mut meta_budget,
    )?;
    let mut best = baseline.clone();
    if let Some((greedy_best, preserve_candidates)) = greedy_preserve_candidate(
        &baseline,
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        MAX_PRESERVE_SEARCH_CANDIDATES,
        &mut meta_budget,
    ) {
        if candidate_is_better(&greedy_best, &best) {
            best = greedy_best.clone();
        }
        let exact_candidate = exact_preserve_subset_candidate(
            &baseline,
            best.clone(),
            &preserve_candidates,
            balances,
            susds_balance,
            slot0_results,
            predictions,
            slot0_results.len(),
            RouteGateThresholds::disabled(),
            Some(force_mint_available),
            false,
            &mut meta_budget,
        );
        if candidate_is_better(&exact_candidate, &best) {
            best = exact_candidate;
        }
    }
    Some(StagedReferenceChoice {
        variant_label: best.variant.as_str(),
        frontier_family: best.forced_first_frontier_family,
        preserve_markets: sorted_preserve_markets(
            &best
                .preserve_markets
                .iter()
                .copied()
                .collect::<Vec<&'static str>>(),
        ),
    })
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_exact_no_arb_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let mut stats = SolverRunStats::default();
    run_exact_no_arb_plan(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        false,
        SolverFamily::Plain,
        &mut stats,
    )
    .actions
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_exact_no_arb_with_explicit_choice_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    preserve_markets: &[&'static str],
    frontier_family: Option<BundleRouteKind>,
) -> Vec<Action> {
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    run_no_arb_rebalance_plan_from_state(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        preserve_markets,
        frontier_family,
        Some(force_mint_available),
        false,
        false,
        SolverFamily::Plain,
    )
    .map(|plan| plan.actions)
    .unwrap_or_default()
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_exact_no_arb_with_search_config_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    online_preserve_cap: usize,
    include_positive_root_arb_seed: bool,
) -> Vec<Action> {
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let mut stats = SolverRunStats::default();
    let extra_seed_states = if include_positive_root_arb_seed {
        run_positive_arb_plan_from_state(
            &state,
            predictions,
            slot0_results.len(),
            RouteGateThresholds::disabled(),
            Some(force_mint_available),
            SolverFamily::Plain,
            &mut stats,
        )
        .map(|plan| vec![plan.terminal_state])
        .unwrap_or_default()
    } else {
        Vec::new()
    };
    run_exact_no_arb_plan_with_options(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        false,
        SolverFamily::Plain,
        &extra_seed_states,
        online_preserve_cap,
        &mut stats,
    )
    .actions
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_exact_no_arb_with_explicit_preserve_universe_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    preserve_universe: &[&'static str],
) -> Vec<Action> {
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let mut stats = SolverRunStats::default();
    let seed_candidates = collect_no_preserve_frontier_seed_plans(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        false,
        SolverFamily::Plain,
        &mut stats,
    );
    let mut best = solver_identity_plan(&state, SolverFamily::Plain);
    best.ev = state_snapshot_expected_value(&state, predictions);
    for candidate in enumerate_exact_no_arb_candidates_over_preserve_universe(
        seed_candidates,
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        false,
        SolverFamily::Plain,
        preserve_universe,
        &mut stats,
    ) {
        if plan_result_is_better(&candidate, &best) {
            best = candidate;
        }
    }
    best.actions
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_plain_family_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let mut stats = SolverRunStats::default();
    run_plain_family_plan(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        &mut stats,
    )
    .actions
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_arb_primed_family_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Option<Vec<Action>> {
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let mut stats = SolverRunStats::default();
    run_arb_primed_family_plan(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        &mut stats,
    )
    .map(|plan| plan.actions)
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_and_budget_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    max_meta_solver_invocations: usize,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_full_with_predictions_and_budget(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
        max_meta_solver_invocations,
        true,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_without_first_frontier_branch_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_full_with_predictions_and_budget(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
        MAX_META_SOLVER_INVOCATIONS,
        false,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_arb_first_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_with_custom_predictions_variant_and_preserve_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        TestPhaseOrderVariant::ArbFirst,
        &HashSet::new(),
        force_mint_available,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_arb_last_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_with_custom_predictions_variant_and_preserve_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        TestPhaseOrderVariant::ArbLast,
        &HashSet::new(),
        force_mint_available,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_variant_and_preserve_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    variant: TestPhaseOrderVariant,
    preserve_sell_markets: &HashSet<&'static str>,
    force_mint_available: bool,
) -> Vec<Action> {
    let mut ctx = match RebalanceContext::from_inputs(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
    ) {
        Ok(ctx) => ctx,
        Err(err) => {
            log_rebalance_init_error(err);
            return Vec::new();
        }
    };
    ctx.route_gates = RouteGateThresholds::disabled();
    ctx.mint_available = force_mint_available;
    ctx.mint_sell_preserve_markets = preserve_sell_markets.clone();
    let (run_phase0_arb, run_phase0_arb_at_end, run_phase0_in_polish) =
        PhaseOrderVariant::from(variant).params();
    finish_rebalance_full_inner(
        ctx,
        slot0_results,
        balances,
        susds_balance,
        run_phase0_arb,
        run_phase0_arb_at_end,
        run_phase0_in_polish,
        false,
    )
}

#[cfg(test)]
pub(super) fn collect_mint_sell_preserve_candidates_for_test(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
    limit: usize,
) -> Vec<&'static str> {
    collect_mint_sell_preserve_candidates(
        actions,
        slot0_results,
        initial_balances,
        initial_susd,
        limit,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_rebalance_only_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    let mut ctx = match RebalanceContext::from_inputs(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
    ) {
        Ok(ctx) => ctx,
        Err(err) => {
            log_rebalance_init_error(err);
            return Vec::new();
        }
    };
    ctx.route_gates = RouteGateThresholds::disabled();
    ctx.mint_available = force_mint_available;
    finish_rebalance_full_inner(
        ctx,
        slot0_results,
        balances,
        susds_balance,
        false,
        false,
        true,
        false,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_and_preserve_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    preserve_sell_markets: &HashSet<&'static str>,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_with_custom_predictions_variant_and_preserve_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        TestPhaseOrderVariant::NoArb,
        preserve_sell_markets,
        force_mint_available,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_rebalance_strict_no_arb_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    rebalance_with_custom_predictions_variant_and_preserve_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        TestPhaseOrderVariant::NoArb,
        &HashSet::new(),
        force_mint_available,
    )
}

#[cfg(test)]
pub(super) fn arb_only_with_custom_predictions_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    let mut ctx = match RebalanceContext::from_inputs(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
    ) {
        Ok(ctx) => ctx,
        Err(err) => {
            log_rebalance_init_error(err);
            return Vec::new();
        }
    };
    ctx.route_gates = RouteGateThresholds::disabled();
    ctx.mint_available = force_mint_available;
    ctx.run_phase0_complete_set_arb();
    ctx.actions
}

#[cfg(test)]
pub(super) fn phase1_liquidation_actions_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    let mut ctx = match RebalanceContext::from_inputs(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
    ) {
        Ok(ctx) => ctx,
        Err(err) => {
            log_rebalance_init_error(err);
            return Vec::new();
        }
    };
    ctx.route_gates = RouteGateThresholds::disabled();
    ctx.mint_available = force_mint_available;
    ctx.run_phase1_sell_overpriced();
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
    rebalance_with_mode_and_flags(
        balances,
        susds_balance,
        slot0_results,
        mode,
        RebalanceFlags::default(),
    )
}

/// Computes optimal trades for L1 markets using explicit strategy flags.
pub fn rebalance_with_mode_and_flags(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    flags: RebalanceFlags,
) -> Vec<Action> {
    rebalance_with_mode_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        RouteGateThresholds::disabled(),
        flags,
    )
}

fn rebalance_with_mode_and_thresholds(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    route_gates: RouteGateThresholds,
    flags: RebalanceFlags,
) -> Vec<Action> {
    match mode {
        RebalanceMode::Full => {
            rebalance_full(balances, susds_balance, slot0_results, route_gates, flags)
        }
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
    rebalance_with_gas_and_flags(
        balances,
        susds_balance,
        slot0_results,
        mode,
        gas,
        RebalanceFlags::default(),
    )
}

/// Gas-aware entry point for `main.rs` with explicit strategy flags.
pub fn rebalance_with_gas_and_flags(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    gas: &GasAssumptions,
    flags: RebalanceFlags,
) -> Vec<Action> {
    rebalance_with_gas_pricing_and_flags(
        balances,
        susds_balance,
        slot0_results,
        mode,
        gas,
        THRESHOLD_GAS_PRICE_ETH,
        THRESHOLD_ETH_USD,
        flags,
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
    rebalance_with_gas_pricing_and_flags(
        balances,
        susds_balance,
        slot0_results,
        mode,
        gas,
        gas_price_eth,
        eth_usd,
        RebalanceFlags::default(),
    )
}

/// Gas-aware entry point with explicit gas price/ETHUSD and strategy flags.
pub fn rebalance_with_gas_pricing_and_flags(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
    flags: RebalanceFlags,
) -> Vec<Action> {
    let n_sims = slot0_results.len();
    let route_gates = compute_gas_thresholds(gas, gas_price_eth, eth_usd, n_sims.saturating_sub(1));
    rebalance_with_mode_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        route_gates,
        flags,
    )
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
