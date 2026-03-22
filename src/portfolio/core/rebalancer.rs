#![allow(clippy::similar_names)]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::thread;

use rayon::prelude::*;
#[cfg(test)]
use serde::Serialize;

use alloy::primitives::Address;

use crate::execution::bounds::{
    ConservativeExecutionConfig, build_group_plans_for_gas_replay_with_market_context,
};
use crate::execution::gas::{
    GasAssumptions, LiveOptimismFeeInputs, estimate_group_calldata_bytes,
    estimate_group_incremental_calldata_bytes, estimate_group_l2_gas_units,
    estimate_l1_data_fee_susd_for_tx_bytes_len, estimate_l2_gas_susd,
    estimate_min_incremental_gas_susd_for_group, gas_price_eth_to_wei, resolve_l1_fee_per_byte_wei,
};
use crate::execution::grouping::{
    group_execution_actions, group_execution_actions_by_profitability_step,
};
use crate::execution::program::{
    ExecutionProgramPlan, MAX_PACKED_TX_L2_GAS_UNITS, compile_execution_program_unchecked,
};
use crate::execution::runtime::{
    DEFAULT_EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK, DEFAULT_EXECUTION_QUOTE_LATENCY_BLOCKS,
};
use crate::execution::{ExecutionMode, GroupKind};
use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};

use super::Action;
use super::bundle::BundleRouteKind;
use super::forecastflows::{
    self, ForecastFlowsCandidateVariant, ForecastFlowsFamilyCandidate, ForecastFlowsTelemetry,
};
use super::merge::action_contract_pair;
use super::sim::{
    DUST, EPS, PoolSim, SimBuildError, alt_price, build_sims, build_sims_without_predictions,
    profitability, target_price_for_prof,
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
const MAX_DISTILLED_PRESERVE_SET_SIZE: usize = 5;
const MAX_DISTILLED_EXTRA_PROPOSAL_SETS: usize = 3;
const ARB_OPERATOR_EV_REL_TOL: f64 = 1e-10;
#[cfg(test)]
const MAX_META_SOLVER_INVOCATIONS: usize = 192;
const MAX_PRESERVE_SEARCH_CANDIDATES: usize = 12;
#[cfg(test)]
const MAX_EXACT_PRESERVE_SUBSET_CANDIDATES: usize = 6;
#[cfg(test)]
const MAX_PRESERVE_LOCAL_SEARCH_ROUNDS: usize = 2;
#[cfg(test)]
const RESERVED_FIRST_FRONTIER_BRANCH_INVOCATIONS: usize = 2;
#[cfg(test)]
const MAX_CYCLIC_LATE_ARB_CYCLES: usize = 8;
#[cfg(test)]
const MAX_CYCLIC_LATE_ARB_INVOCATIONS: usize = MAX_CYCLIC_LATE_ARB_CYCLES * 2;
#[cfg(test)]
const CYCLIC_LATE_ARB_EV_REL_TOL: f64 = 1e-10;
const DEFAULT_GATE_BUFFER_FRAC: f64 = 0.20;
const DEFAULT_GATE_BUFFER_MIN_SUSD: f64 = 0.25;
const PLANNER_SYNTHETIC_CHAIN_ID: u64 = 10;
const PLANNER_SYNTHETIC_NONCE: u64 = 0;
const CONSTANT_L_SOLVE_ITERS: usize = 64;
const CONSTANT_L_EXTRA_SEED_MAX_PROFITABLE: usize = 16;
const CONSTANT_L_LOCAL_SEARCH_MAX_PROFITABLE: usize = 16;
#[cfg(test)]
const K1_SMALL_ORACLE_MAX_MARKETS: usize = 12;
#[cfg(test)]
const K1_MEDIUM_ORACLE_MAX_MARKETS: usize = 13;
#[cfg(test)]
const K2_ORACLE_MAX_MARKETS: usize = 8;
const STAGED_CONSTANT_L_RAW_EV_GAP_SUSD: f64 = 1e-6;
const STAGED_CONSTANT_L_FRACTION_REFINEMENT_EPS: f64 = 1e-12;
const STAGED_CONSTANT_L_FRACTIONS: [f64; 9] =
    [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
#[cfg(test)]
const BENCHMARK_OP_GAS_PRICE_ETH: f64 = 1.002325e-12;
#[cfg(test)]
const BENCHMARK_OP_ETH_USD: f64 = 3000.0;
#[cfg(test)]
const BENCHMARK_OP_L1_FEE_PER_BYTE_WEI: f64 = 1_643_855.3414634147;
#[cfg(test)]
const BENCHMARK_L1_DATA_FEE_FLOOR_SUSD: f64 = 0.0;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(super) struct PlannerPricingSnapshot {
    pub(super) gas_price_eth: f64,
    pub(super) eth_usd: f64,
    pub(super) l1_fee_per_byte_wei: f64,
    pub(super) source_label: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PlannerCostConfig {
    pub(super) gas_assumptions: GasAssumptions,
    pub(super) pricing: PlannerPricingSnapshot,
    pub(super) conservative_execution: ConservativeExecutionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeeEstimateSource {
    ZeroAction,
    ReplayPackedProgram,
    ReplayStrictProgram,
    StructuralFallback,
    Unavailable,
}

impl FeeEstimateSource {
    fn label(self) -> &'static str {
        match self {
            Self::ZeroAction => "zero_action",
            Self::ReplayPackedProgram => "replay_packed_program",
            Self::ReplayStrictProgram => "replay_strict_program",
            Self::StructuralFallback => "structural_fallback",
            Self::Unavailable => "unavailable",
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct PlanCostEstimate {
    group_count: usize,
    tx_count: usize,
    l2_fee_susd: f64,
    l1_fee_susd: f64,
    total_fee_susd: f64,
    source: FeeEstimateSource,
}

#[derive(Debug, Default, Clone, Copy)]
struct ForecastFlowsLocalTimingTotals {
    candidate_build_ms: u128,
    step_prune_ms: u128,
    route_prune_ms: u128,
}

#[derive(Debug, Clone)]
struct ForecastFlowsEvaluatedActionSet {
    actions: Vec<Action>,
    terminal_state: SolverStateSnapshot,
    raw_ev: f64,
    plan_cost: PlanCostEstimate,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ForecastFlowsActionKey {
    Mint {
        contract_1: &'static str,
        contract_2: &'static str,
        amount_bits: u64,
        target_market: &'static str,
    },
    Buy {
        market_name: &'static str,
        amount_bits: u64,
        cost_bits: u64,
    },
    Sell {
        market_name: &'static str,
        amount_bits: u64,
        proceeds_bits: u64,
    },
    Merge {
        contract_1: &'static str,
        contract_2: &'static str,
        amount_bits: u64,
        source_market: &'static str,
    },
}

#[derive(Debug, Default)]
struct ForecastFlowsActionEvaluationCache {
    evaluations: HashMap<Vec<ForecastFlowsActionKey>, Option<ForecastFlowsEvaluatedActionSet>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceMode {
    Full,
    ArbOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceSolver {
    Native,
    ForecastFlows,
    HeadToHead,
}

impl RebalanceSolver {
    pub fn from_env() -> Self {
        match std::env::var("REBALANCE_SOLVER")
            .ok()
            .map(|raw| raw.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("native") => Self::Native,
            Some("forecastflows") | Some("forecast_flows") | Some("forecast") => {
                Self::ForecastFlows
            }
            _ => Self::HeadToHead,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::ForecastFlows => "forecastflows",
            Self::HeadToHead => "head_to_head",
        }
    }

    pub fn uses_forecastflows(self) -> bool {
        matches!(self, Self::ForecastFlows | Self::HeadToHead)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebalanceFlags {
    pub enable_ev_guarded_greedy_churn_pruning: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ForecastFlowsRunTelemetry {
    pub strategy: Option<String>,
    pub worker_roundtrip_ms: Option<u128>,
    pub driver_overhead_ms: Option<u128>,
    pub translation_replay_ms: Option<u128>,
    pub local_candidate_build_ms: Option<u128>,
    pub local_step_prune_ms: Option<u128>,
    pub local_route_prune_ms: Option<u128>,
    pub workspace_reused: bool,
    pub direct_solver_time_ms: Option<u128>,
    pub mixed_solver_time_ms: Option<u128>,
    pub estimated_execution_cost_susd: Option<f64>,
    pub estimated_net_ev_susd: Option<f64>,
    pub validated_total_fee_susd: Option<f64>,
    pub validated_net_ev_susd: Option<f64>,
    pub fee_estimate_error_susd: Option<f64>,
    pub validation_only: bool,
    pub solve_tuning: Option<String>,
    pub sysimage_status: Option<String>,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RebalancePlanSummary {
    pub raw_ev: f64,
    pub estimated_total_fee_susd: Option<f64>,
    pub estimated_net_ev: Option<f64>,
    pub estimated_group_count: Option<usize>,
    pub estimated_tx_count: Option<usize>,
    pub action_count: usize,
    pub forecastflows_telemetry: ForecastFlowsRunTelemetry,
    family_stable_rank: usize,
    frontier_family_stable_rank: usize,
    preserve_markets: Vec<String>,
    compiler_variant_stable_rank: usize,
    selected_common_shift: Option<f64>,
    selected_mixed_lambda: Option<f64>,
    selected_active_set_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RebalancePlanDecision {
    pub actions: Vec<Action>,
    pub summary: RebalancePlanSummary,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhaseOrderVariant {
    ArbFirst,
    ArbLast,
    NoArb,
    CyclicLateArb,
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct MetaSearchBudget {
    remaining: usize,
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CandidateResult {
    actions: Vec<Action>,
    raw_ev: f64,
    estimated_total_fee_susd: Option<f64>,
    estimated_net_ev: Option<f64>,
    estimated_group_count: Option<usize>,
    estimated_tx_count: Option<usize>,
    fee_estimate_source: FeeEstimateSource,
    variant: PhaseOrderVariant,
    preserve_markets: HashSet<&'static str>,
    forced_first_frontier_family: Option<BundleRouteKind>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct PhaseOrderVariantEvaluation {
    baseline: CandidateResult,
    greedy_best: CandidateResult,
    preserve_candidates: Vec<&'static str>,
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Debug, Clone)]
struct RefinedCandidateResult {
    candidate: CandidateResult,
    source_label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SolverFamily {
    Plain,
    ArbPrimed,
    ForecastFlows,
}

impl SolverFamily {
    fn stable_rank(self) -> usize {
        match self {
            Self::Plain => 0,
            Self::ArbPrimed => 1,
            Self::ForecastFlows => 2,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::ArbPrimed => "arb_primed",
            Self::ForecastFlows => "forecastflows",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanCompilerVariant {
    BaselineStepPrune,
    TargetDelta,
    AnalyticMixed,
    ConstantLMixed,
    CoupledMixed,
    StagedConstantLMixed2,
    DirectOnly,
    ForecastFlowsDirect,
    ForecastFlowsMixed,
    NoOp,
}

impl PlanCompilerVariant {
    fn stable_rank(self) -> usize {
        match self {
            Self::BaselineStepPrune => 0,
            Self::TargetDelta => 1,
            Self::ConstantLMixed => 2,
            Self::AnalyticMixed => 3,
            Self::CoupledMixed => 4,
            Self::StagedConstantLMixed2 => 5,
            Self::DirectOnly => 6,
            Self::ForecastFlowsDirect => 7,
            Self::ForecastFlowsMixed => 8,
            Self::NoOp => 9,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::BaselineStepPrune => "baseline_step_prune",
            Self::TargetDelta => "target_delta",
            Self::AnalyticMixed => "analytic_mixed",
            Self::ConstantLMixed => "constant_l_mixed",
            Self::CoupledMixed => "coupled_mixed",
            Self::StagedConstantLMixed2 => "staged_constant_l_2",
            Self::DirectOnly => "direct_only",
            Self::ForecastFlowsDirect => "forecastflows_direct",
            Self::ForecastFlowsMixed => "forecastflows_mixed",
            Self::NoOp => "noop",
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
struct ConstantLMixedFixedActiveSolution {
    actions: Vec<Action>,
    terminal_state: SolverStateSnapshot,
    raw_ev: f64,
    certificate: MixedCertificate,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConstantLMixedSolveKey {
    state_fingerprint: u64,
    budget_cap_bits: u64,
    active_mask: Vec<bool>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstantLSeedMode {
    RuntimeCapped,
    TeacherBestKnown,
}

#[derive(Debug, Clone)]
struct ConstantLSeedSet {
    profitable_indices: Vec<usize>,
    seed_masks: Vec<Vec<bool>>,
}

#[derive(Debug, Default)]
struct ConstantLMixedSolveCache {
    fixed_active_solutions:
        HashMap<ConstantLMixedSolveKey, Option<ConstantLMixedFixedActiveSolution>>,
    seed_sets: HashMap<(u64, ConstantLSeedMode, SolverFamily), Option<ConstantLSeedSet>>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(super) struct MixedCertificate {
    pub(super) active_mask: Vec<bool>,
    pub(super) active_set_size: usize,
    pub(super) pi: f64,
    pub(super) mint_amount: f64,
    pub(super) direct_cost: f64,
    pub(super) sell_proceeds: f64,
    pub(super) mint_net_cost: f64,
    pub(super) budget_used: f64,
    pub(super) budget_residual: f64,
    pub(super) delta_target: f64,
    pub(super) delta_realized: f64,
    pub(super) raw_ev: f64,
    pub(super) estimated_fee_susd: Option<f64>,
    pub(super) estimated_net_ev: Option<f64>,
}

#[cfg(test)]
impl Serialize for MixedCertificate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("MixedCertificate", 14)?;
        state.serialize_field("active_mask", &self.active_mask)?;
        state.serialize_field("active_set_size", &self.active_set_size)?;
        state.serialize_field("pi", &self.pi)?;
        state.serialize_field("mint_amount", &self.mint_amount)?;
        state.serialize_field("direct_cost", &self.direct_cost)?;
        state.serialize_field("sell_proceeds", &self.sell_proceeds)?;
        state.serialize_field("mint_net_cost", &self.mint_net_cost)?;
        state.serialize_field("budget_used", &self.budget_used)?;
        state.serialize_field("budget_residual", &self.budget_residual)?;
        state.serialize_field("delta_target", &self.delta_target)?;
        state.serialize_field("delta_realized", &self.delta_realized)?;
        state.serialize_field("raw_ev", &self.raw_ev)?;
        state.serialize_field("estimated_fee_susd", &self.estimated_fee_susd)?;
        state.serialize_field("estimated_net_ev", &self.estimated_net_ev)?;
        state.end()
    }
}

#[derive(Debug, Clone)]
struct PlanResult {
    actions: Vec<Action>,
    terminal_state: SolverStateSnapshot,
    raw_ev: f64,
    estimated_total_fee_susd: Option<f64>,
    estimated_net_ev: Option<f64>,
    estimated_group_count: Option<usize>,
    estimated_tx_count: Option<usize>,
    fee_estimate_source: FeeEstimateSource,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    compiler_variant: PlanCompilerVariant,
    selected_common_shift: Option<f64>,
    selected_mixed_lambda: Option<f64>,
    selected_active_set_size: Option<usize>,
    selected_stage_count: Option<usize>,
    selected_stage1_budget_fraction: Option<f64>,
    mixed_certificates: Vec<MixedCertificate>,
}

#[derive(Debug, Default, Clone)]
pub(super) struct SolverRunStats {
    pub(super) exact_rebalance_calls: usize,
    pub(super) exact_rebalance_candidate_evals: usize,
    pub(super) distilled_proposal_sets: usize,
    pub(super) distilled_proposal_candidate_evals: usize,
    pub(super) arb_operator_evals: usize,
    pub(super) arb_primed_root_taken: bool,
    pub(super) fee_estimate_unavailable_results: usize,
    pub(super) forecastflows_requests: usize,
    pub(super) forecastflows_worker_available: bool,
    pub(super) forecastflows_fallback_reason: Option<&'static str>,
    pub(super) forecastflows_telemetry: ForecastFlowsTelemetry,
    pub(super) forecastflows_winning_variant: Option<ForecastFlowsCandidateVariant>,
}

#[derive(Debug, Clone, Copy)]
struct PreserveCandidateScore {
    market_name: &'static str,
    churn_amount: f64,
    sold_amount: f64,
}

#[derive(Debug, Clone)]
struct DistilledMarketFeature {
    market_name: &'static str,
    holding: f64,
    churn_amount: f64,
    sold_amount: f64,
    direct_profitability: f64,
    direct_profitability_rank: usize,
    prediction_minus_price: f64,
    mint_profitability: f64,
    hold_through_mint_value: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize)]
pub(super) struct TeacherMarketFeatureRow {
    market_name: &'static str,
    holding: f64,
    churn_amount: f64,
    sold_amount: f64,
    direct_profitability: f64,
    direct_profitability_rank: usize,
    prediction_minus_price: f64,
    mint_profitability: f64,
    hold_through_mint_value: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize)]
pub(super) struct TeacherDecisionSnapshot {
    winning_family: &'static str,
    winning_frontier_family: &'static str,
    preserve_markets: Vec<&'static str>,
    root_arb_fired: bool,
    late_arb_improved: bool,
    raw_ev: f64,
    action_count: usize,
    features: Vec<TeacherMarketFeatureRow>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct TestPlanEconomics {
    pub(super) raw_ev: f64,
    pub(super) estimated_total_fee_susd: Option<f64>,
    pub(super) estimated_net_ev: Option<f64>,
    pub(super) estimated_group_count: Option<usize>,
    pub(super) estimated_tx_count: Option<usize>,
    pub(super) fee_estimate_source: &'static str,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize)]
pub(super) struct TestSelectedPlanSummary {
    pub(super) family: &'static str,
    pub(super) frontier_family: &'static str,
    pub(super) compiler_variant: &'static str,
    pub(super) selected_common_shift: Option<f64>,
    pub(super) selected_mixed_lambda: Option<f64>,
    pub(super) selected_active_set_size: Option<usize>,
    pub(super) selected_stage_count: Option<usize>,
    pub(super) selected_stage1_budget_fraction: Option<f64>,
    pub(super) raw_ev: f64,
    pub(super) estimated_total_fee_susd: Option<f64>,
    pub(super) estimated_net_ev: Option<f64>,
    pub(super) estimated_group_count: Option<usize>,
    pub(super) estimated_tx_count: Option<usize>,
    pub(super) fee_estimate_source: &'static str,
    pub(super) action_count: usize,
    pub(super) direct_buy_count: usize,
    pub(super) direct_sell_count: usize,
    pub(super) mint_count: usize,
    pub(super) merge_count: usize,
    pub(super) total_calldata_bytes: Option<usize>,
    pub(super) mixed_certificates: Vec<MixedCertificate>,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize)]
pub(super) struct TestForecastFlowsPlanVariants {
    pub(super) raw: TestSelectedPlanSummary,
    pub(super) polished: TestSelectedPlanSummary,
    pub(super) raw_replayable: bool,
    pub(super) polished_replayable: bool,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize)]
pub(super) struct TestK1OracleComparison {
    pub(super) runtime_best: TestSelectedPlanSummary,
    pub(super) oracle_best: TestSelectedPlanSummary,
    pub(super) oracle_best_is_direct_prefix: bool,
    pub(super) oracle_best_active_set_size: Option<usize>,
    pub(super) runtime_k1_gap_net_ev: Option<f64>,
    pub(super) runtime_k1_gap_raw_ev: Option<f64>,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize)]
pub(super) struct TestK1BestKnownComparison {
    pub(super) runtime_best: TestSelectedPlanSummary,
    pub(super) best_known_best: TestSelectedPlanSummary,
    pub(super) runtime_k1_gap_net_ev: Option<f64>,
    pub(super) runtime_k1_gap_raw_ev: Option<f64>,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize)]
pub(super) struct TestK2OracleComparison {
    pub(super) runtime_best: TestSelectedPlanSummary,
    pub(super) k1_oracle_best: TestSelectedPlanSummary,
    pub(super) k2_oracle_best: TestSelectedPlanSummary,
    pub(super) k2_gain_net_ev: Option<f64>,
    pub(super) k2_gain_raw_ev: Option<f64>,
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

fn planner_cost_config_with_pricing(
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
    source_label: &'static str,
) -> PlannerCostConfig {
    let mut effective_gas_assumptions = *gas_assumptions;
    if let Some(l1_fee_per_byte_wei) = resolve_l1_fee_per_byte_wei(&effective_gas_assumptions) {
        effective_gas_assumptions.l1_fee_per_byte_wei = l1_fee_per_byte_wei;
    }
    PlannerCostConfig {
        gas_assumptions: effective_gas_assumptions,
        pricing: PlannerPricingSnapshot {
            gas_price_eth,
            eth_usd,
            l1_fee_per_byte_wei: effective_gas_assumptions.l1_fee_per_byte_wei,
            source_label,
        },
        conservative_execution: ConservativeExecutionConfig {
            quote_latency_blocks: DEFAULT_EXECUTION_QUOTE_LATENCY_BLOCKS,
            adverse_move_bps_per_block: DEFAULT_EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK,
        },
    }
}

fn default_planner_cost_config() -> PlannerCostConfig {
    planner_cost_config_with_pricing(
        &GasAssumptions::default(),
        THRESHOLD_GAS_PRICE_ETH,
        THRESHOLD_ETH_USD,
        "default_conservative_thresholds",
    )
}

#[cfg(test)]
pub(super) fn benchmark_gas_assumptions_for_test() -> GasAssumptions {
    let mut assumptions = GasAssumptions::default();
    assumptions.l1_fee_per_byte_wei = BENCHMARK_OP_L1_FEE_PER_BYTE_WEI;
    assumptions.l1_data_fee_floor_susd = BENCHMARK_L1_DATA_FEE_FLOOR_SUSD;
    assumptions
}

#[cfg(test)]
pub(super) fn benchmark_planner_cost_config_for_test() -> PlannerCostConfig {
    planner_cost_config_with_pricing(
        &benchmark_gas_assumptions_for_test(),
        BENCHMARK_OP_GAS_PRICE_ETH,
        BENCHMARK_OP_ETH_USD,
        "benchmark_snapshot_op_2026_03_08_fixed_l1",
    )
}

#[cfg(test)]
fn benchmark_planner_cost_config_with_pricing_for_test(
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
) -> PlannerCostConfig {
    planner_cost_config_with_pricing(
        gas_assumptions,
        gas_price_eth,
        eth_usd,
        "benchmark_test_explicit_pricing",
    )
}

fn planner_synthetic_fee_inputs(pricing: PlannerPricingSnapshot) -> LiveOptimismFeeInputs {
    LiveOptimismFeeInputs {
        chain_id: PLANNER_SYNTHETIC_CHAIN_ID,
        sender_nonce: PLANNER_SYNTHETIC_NONCE,
        gas_price_wei: gas_price_eth_to_wei(pricing.gas_price_eth).unwrap_or(0),
    }
}

fn better_execution_program(
    candidate: &ExecutionProgramPlan,
    incumbent: &ExecutionProgramPlan,
) -> bool {
    let fee_tol = raw_value_tol(candidate.total_fee_susd, incumbent.total_fee_susd);
    if candidate.total_fee_susd + fee_tol < incumbent.total_fee_susd {
        return true;
    }
    if incumbent.total_fee_susd + fee_tol < candidate.total_fee_susd {
        return false;
    }
    if candidate.tx_count != incumbent.tx_count {
        return candidate.tx_count < incumbent.tx_count;
    }
    if candidate.strict_subgroup_count != incumbent.strict_subgroup_count {
        return candidate.strict_subgroup_count < incumbent.strict_subgroup_count;
    }
    matches!(
        (candidate.mode, incumbent.mode),
        (ExecutionMode::Packed, ExecutionMode::Strict)
    )
}

fn plan_cost_from_program(
    program: &ExecutionProgramPlan,
    source: FeeEstimateSource,
) -> PlanCostEstimate {
    PlanCostEstimate {
        group_count: program.strict_subgroup_count,
        tx_count: program.tx_count,
        l2_fee_susd: program.total_l2_fee_susd,
        l1_fee_susd: program.total_l1_fee_susd,
        total_fee_susd: program.total_fee_susd,
        source,
    }
}

fn estimate_plan_cost_from_replay(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    cost_config: PlannerCostConfig,
) -> Option<PlanCostEstimate> {
    if actions.is_empty() {
        return Some(PlanCostEstimate {
            group_count: 0,
            tx_count: 0,
            l2_fee_susd: 0.0,
            l1_fee_susd: 0.0,
            total_fee_susd: 0.0,
            source: FeeEstimateSource::ZeroAction,
        });
    }

    let replay = build_group_plans_for_gas_replay_with_market_context(
        actions,
        slot0_results,
        cost_config.conservative_execution,
        &cost_config.gas_assumptions,
        cost_config.pricing.gas_price_eth,
        cost_config.pricing.eth_usd,
    )
    .ok()?;
    if replay.plans.is_empty() || !replay.skipped_groups.is_empty() {
        return None;
    }

    let fee_inputs = planner_synthetic_fee_inputs(cost_config.pricing);
    let strict_program = compile_execution_program_unchecked(
        ExecutionMode::Strict,
        Address::ZERO,
        actions,
        &replay.plans,
        fee_inputs,
        &cost_config.gas_assumptions,
        cost_config.pricing.eth_usd,
    )
    .ok()?;
    let packed_program = compile_execution_program_unchecked(
        ExecutionMode::Packed,
        Address::ZERO,
        actions,
        &replay.plans,
        fee_inputs,
        &cost_config.gas_assumptions,
        cost_config.pricing.eth_usd,
    )
    .ok()?;

    let (program, source) = if better_execution_program(&packed_program, &strict_program) {
        (&packed_program, FeeEstimateSource::ReplayPackedProgram)
    } else {
        (&strict_program, FeeEstimateSource::ReplayStrictProgram)
    };
    Some(plan_cost_from_program(program, source))
}

fn estimate_plan_cost_structural(
    actions: &[Action],
    cost_config: PlannerCostConfig,
) -> Option<PlanCostEstimate> {
    if actions.is_empty() {
        return Some(PlanCostEstimate {
            group_count: 0,
            tx_count: 0,
            l2_fee_susd: 0.0,
            l1_fee_susd: 0.0,
            total_fee_susd: 0.0,
            source: FeeEstimateSource::ZeroAction,
        });
    }

    let groups = group_execution_actions(actions).ok()?;
    if groups.is_empty() {
        return None;
    }

    let mut l2_fee_susd = 0.0;
    let mut l1_fee_susd = 0.0;
    let mut tx_count = 0usize;
    let mut chunk_l2_gas_units = 0u64;
    let mut chunk_tx_bytes_len = 0u64;

    let flush_chunk = |chunk_l2_gas_units: &mut u64,
                       chunk_tx_bytes_len: &mut u64,
                       tx_count: &mut usize,
                       l2_fee_susd: &mut f64,
                       l1_fee_susd: &mut f64|
     -> Option<()> {
        if *chunk_tx_bytes_len == 0 {
            return Some(());
        }
        *l2_fee_susd += estimate_l2_gas_susd(
            *chunk_l2_gas_units,
            cost_config.pricing.gas_price_eth,
            cost_config.pricing.eth_usd,
        );
        *l1_fee_susd += estimate_l1_data_fee_susd_for_tx_bytes_len(
            &cost_config.gas_assumptions,
            (*chunk_tx_bytes_len).try_into().ok()?,
            cost_config.pricing.eth_usd,
        );
        *tx_count += 1;
        *chunk_l2_gas_units = 0;
        *chunk_tx_bytes_len = 0;
        Some(())
    };

    for group in &groups {
        let l2_gas_units = estimate_group_l2_gas_units(
            &cost_config.gas_assumptions,
            group.kind,
            group.buy_legs,
            group.sell_legs,
        );
        let group_base_bytes =
            estimate_group_calldata_bytes(group.kind, group.buy_legs, group.sell_legs);
        let group_incremental_bytes =
            estimate_group_incremental_calldata_bytes(group.kind, group.buy_legs, group.sell_legs);
        if chunk_tx_bytes_len == 0 {
            if l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
                return None;
            }
            chunk_l2_gas_units = l2_gas_units;
            chunk_tx_bytes_len = group_base_bytes;
            continue;
        }

        let tentative_l2_gas_units = chunk_l2_gas_units.saturating_add(l2_gas_units);
        if tentative_l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
            flush_chunk(
                &mut chunk_l2_gas_units,
                &mut chunk_tx_bytes_len,
                &mut tx_count,
                &mut l2_fee_susd,
                &mut l1_fee_susd,
            )?;
            if l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
                return None;
            }
            chunk_l2_gas_units = l2_gas_units;
            chunk_tx_bytes_len = group_base_bytes;
            continue;
        }

        chunk_l2_gas_units = tentative_l2_gas_units;
        chunk_tx_bytes_len = chunk_tx_bytes_len.saturating_add(group_incremental_bytes);
    }

    flush_chunk(
        &mut chunk_l2_gas_units,
        &mut chunk_tx_bytes_len,
        &mut tx_count,
        &mut l2_fee_susd,
        &mut l1_fee_susd,
    )?;

    if !l2_fee_susd.is_finite() || !l1_fee_susd.is_finite() {
        return None;
    }

    Some(PlanCostEstimate {
        group_count: groups.len(),
        tx_count,
        l2_fee_susd,
        l1_fee_susd,
        total_fee_susd: l2_fee_susd + l1_fee_susd,
        source: FeeEstimateSource::StructuralFallback,
    })
}

#[cfg(test)]
fn estimate_structural_packed_total_calldata_bytes(
    actions: &[Action],
    cost_config: PlannerCostConfig,
) -> Option<usize> {
    if actions.is_empty() {
        return Some(0);
    }

    let groups = group_execution_actions(actions).ok()?;
    if groups.is_empty() {
        return None;
    }

    let mut total_bytes = 0u64;
    let mut chunk_l2_gas_units = 0u64;
    let mut chunk_tx_bytes_len = 0u64;
    for group in &groups {
        let l2_gas_units = estimate_group_l2_gas_units(
            &cost_config.gas_assumptions,
            group.kind,
            group.buy_legs,
            group.sell_legs,
        );
        let group_base_bytes =
            estimate_group_calldata_bytes(group.kind, group.buy_legs, group.sell_legs);
        let group_incremental_bytes =
            estimate_group_incremental_calldata_bytes(group.kind, group.buy_legs, group.sell_legs);
        if chunk_tx_bytes_len == 0 {
            if l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
                return None;
            }
            chunk_l2_gas_units = l2_gas_units;
            chunk_tx_bytes_len = group_base_bytes;
            continue;
        }
        let tentative_l2_gas_units = chunk_l2_gas_units.saturating_add(l2_gas_units);
        if tentative_l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
            total_bytes = total_bytes.saturating_add(chunk_tx_bytes_len);
            if l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
                return None;
            }
            chunk_l2_gas_units = l2_gas_units;
            chunk_tx_bytes_len = group_base_bytes;
            continue;
        }
        chunk_l2_gas_units = tentative_l2_gas_units;
        chunk_tx_bytes_len = chunk_tx_bytes_len.saturating_add(group_incremental_bytes);
    }
    total_bytes = total_bytes.saturating_add(chunk_tx_bytes_len);
    total_bytes.try_into().ok()
}

fn estimate_plan_cost(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    cost_config: PlannerCostConfig,
) -> Option<PlanCostEstimate> {
    estimate_plan_cost_from_replay(actions, slot0_results, cost_config)
        .or_else(|| estimate_plan_cost_structural(actions, cost_config))
}

fn plan_cost_fields(
    raw_ev: f64,
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    cost_config: PlannerCostConfig,
) -> (
    Option<f64>,
    Option<f64>,
    Option<usize>,
    Option<usize>,
    FeeEstimateSource,
    bool,
) {
    if let Some(cost) = estimate_plan_cost(actions, slot0_results, cost_config) {
        return plan_cost_fields_from_estimate(raw_ev, cost);
    }

    (None, None, None, None, FeeEstimateSource::Unavailable, true)
}

fn plan_cost_fields_from_estimate(
    raw_ev: f64,
    cost: PlanCostEstimate,
) -> (
    Option<f64>,
    Option<f64>,
    Option<usize>,
    Option<usize>,
    FeeEstimateSource,
    bool,
) {
    (
        Some(cost.total_fee_susd),
        Some(raw_ev - cost.total_fee_susd),
        Some(cost.group_count),
        Some(cost.tx_count),
        cost.source,
        false,
    )
}

fn raw_value_tol(left: f64, right: f64) -> f64 {
    PRESERVE_SELECTION_EV_REL_TOL
        * (1.0 + left.abs().max(right.abs()) + left.abs().min(right.abs()))
}

fn net_ev_cmp(
    candidate_net_ev: Option<f64>,
    incumbent_net_ev: Option<f64>,
    candidate_raw_ev: f64,
    incumbent_raw_ev: f64,
) -> Ordering {
    if let (Some(candidate_net_ev), Some(incumbent_net_ev)) = (candidate_net_ev, incumbent_net_ev) {
        let ev_tol = raw_value_tol(candidate_net_ev, incumbent_net_ev);
        if candidate_net_ev > incumbent_net_ev + ev_tol {
            return Ordering::Less;
        }
        if incumbent_net_ev > candidate_net_ev + ev_tol {
            return Ordering::Greater;
        }
    }

    let ev_tol = raw_value_tol(candidate_raw_ev, incumbent_raw_ev);
    if candidate_raw_ev > incumbent_raw_ev + ev_tol {
        Ordering::Less
    } else if incumbent_raw_ev > candidate_raw_ev + ev_tol {
        Ordering::Greater
    } else {
        Ordering::Equal
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

fn build_plan_result(
    starting_state: &SolverStateSnapshot,
    terminal_state: SolverStateSnapshot,
    actions: Vec<Action>,
    raw_ev: f64,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    compiler_variant: PlanCompilerVariant,
    selected_common_shift: Option<f64>,
    selected_mixed_lambda: Option<f64>,
    selected_active_set_size: Option<usize>,
) -> PlanResult {
    let (
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source,
        _fee_estimate_unavailable,
    ) = plan_cost_fields(raw_ev, &actions, &starting_state.slot0_results, cost_config);
    PlanResult {
        actions,
        terminal_state,
        raw_ev,
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source,
        frontier_family,
        preserve_markets,
        family,
        compiler_variant,
        selected_common_shift,
        selected_mixed_lambda,
        selected_active_set_size,
        selected_stage_count: None,
        selected_stage1_budget_fraction: None,
        mixed_certificates: Vec::new(),
    }
}

fn with_mixed_certificates(
    mut plan: PlanResult,
    mut mixed_certificates: Vec<MixedCertificate>,
) -> PlanResult {
    for certificate in &mut mixed_certificates {
        certificate.raw_ev = plan.raw_ev;
        certificate.estimated_fee_susd = plan.estimated_total_fee_susd;
        certificate.estimated_net_ev = plan.estimated_net_ev;
    }
    plan.mixed_certificates = mixed_certificates;
    plan
}

#[cfg(test)]
fn build_candidate_result(
    starting_state: &SolverStateSnapshot,
    actions: Vec<Action>,
    raw_ev: f64,
    variant: PhaseOrderVariant,
    preserve_markets: HashSet<&'static str>,
    forced_first_frontier_family: Option<BundleRouteKind>,
    cost_config: PlannerCostConfig,
) -> CandidateResult {
    let (
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source,
        _fee_estimate_unavailable,
    ) = plan_cost_fields(raw_ev, &actions, &starting_state.slot0_results, cost_config);
    CandidateResult {
        actions,
        raw_ev,
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source,
        variant,
        preserve_markets,
        forced_first_frontier_family,
    }
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

#[cfg(test)]
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

#[cfg(test)]
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

fn solver_state_fingerprint(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    state.cash.to_bits().hash(&mut hasher);
    state.slot0_results.len().hash(&mut hasher);
    for (slot0, market) in &state.slot0_results {
        market.name.hash(&mut hasher);
        slot0.pool_id.hash(&mut hasher);
        slot0.tick.hash(&mut hasher);
        slot0.sqrt_price_x96.to_string().hash(&mut hasher);
        state
            .holdings
            .get(market.name)
            .copied()
            .unwrap_or(0.0)
            .to_bits()
            .hash(&mut hasher);
        predictions
            .get(&crate::pools::normalize_market_name(market.name))
            .copied()
            .unwrap_or(0.0)
            .to_bits()
            .hash(&mut hasher);
    }
    hasher.finish()
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

fn solver_identity_plan(
    state: &SolverStateSnapshot,
    family: SolverFamily,
    raw_ev: f64,
    cost_config: PlannerCostConfig,
) -> PlanResult {
    solver_identity_plan_with_variant(
        state,
        family,
        raw_ev,
        cost_config,
        PlanCompilerVariant::NoOp,
    )
}

fn solver_identity_plan_with_variant(
    state: &SolverStateSnapshot,
    family: SolverFamily,
    raw_ev: f64,
    cost_config: PlannerCostConfig,
    compiler_variant: PlanCompilerVariant,
) -> PlanResult {
    build_plan_result(
        state,
        state.clone(),
        Vec::new(),
        raw_ev,
        None,
        Vec::new(),
        family,
        cost_config,
        compiler_variant,
        None,
        None,
        None,
    )
}

fn representative_complete_set_market(state: &SolverStateSnapshot) -> Option<&'static str> {
    let mut markets: Vec<&'static str> = state
        .slot0_results
        .iter()
        .map(|(_, market)| market.name)
        .collect();
    markets.sort_unstable();
    markets.first().copied()
}

fn sorted_market_deltas(
    starting_state: &SolverStateSnapshot,
    target_state: &SolverStateSnapshot,
) -> Vec<(&'static str, f64)> {
    let mut deltas: Vec<(&'static str, f64)> = starting_state
        .slot0_results
        .iter()
        .map(|(_, market)| {
            let start = starting_state
                .holdings
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let target = target_state
                .holdings
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            (market.name, target - start)
        })
        .collect();
    deltas.sort_by(|left, right| left.0.cmp(right.0));
    deltas
}

fn values_match_with_tol(left: f64, right: f64) -> bool {
    (left - right).abs() <= raw_value_tol(left, right)
}

fn holdings_match_within_tol(
    candidate_state: &SolverStateSnapshot,
    target_state: &SolverStateSnapshot,
) -> bool {
    for (_, market) in &target_state.slot0_results {
        let candidate = candidate_state
            .holdings
            .get(market.name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let target = target_state
            .holdings
            .get(market.name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        if !values_match_with_tol(candidate, target) {
            return false;
        }
    }
    true
}

fn deduped_target_delta_common_shifts(deltas: &[(&'static str, f64)]) -> Vec<f64> {
    let mut shifts = vec![0.0_f64];
    let mut unique_deltas: Vec<f64> = deltas.iter().map(|(_, delta)| *delta).collect();
    unique_deltas.sort_by(|left, right| left.total_cmp(right));
    for delta in unique_deltas {
        if shifts
            .iter()
            .any(|existing| values_match_with_tol(*existing, delta))
        {
            continue;
        }
        shifts.push(delta);
    }
    shifts
}

#[derive(Debug, Clone, Copy)]
struct FrontierDirectAdjustment {
    market_name: &'static str,
    buy_amount: f64,
    sell_amount: f64,
}

#[derive(Debug, Clone, Copy)]
struct CoupledMixedFrontierEntry {
    idx: usize,
    market_name: &'static str,
    current_price: f64,
    holding: f64,
    active: bool,
    buy_amount: f64,
    buy_cost: f64,
    sell_cap: f64,
}

#[derive(Debug, Clone, Copy)]
enum ActiveBuyLimitMode {
    ClampToBuyLimit,
    RequireWithinBuyLimit,
}

#[derive(Debug, Clone, Copy)]
struct MixedFrontierAggregates {
    active_target_sum: f64,
    inactive_current_sum: f64,
    max_required_shift: f64,
}

#[derive(Debug, Clone, Copy)]
struct MixedSequentialBudget {
    required_initial_cash: f64,
    mint_net_cost: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct ConstantLMixedPiEvaluation {
    required_initial_cash: f64,
    mint_amount: f64,
    direct_cost: f64,
    sell_proceeds: f64,
    mint_net_cost: f64,
    budget_used: f64,
    budget_residual: f64,
    delta_target: f64,
    delta_realized: f64,
}

fn deduped_numeric_candidates(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|left, right| left.total_cmp(right));
    let mut deduped = Vec::with_capacity(values.len());
    for value in values {
        if !value.is_finite() {
            continue;
        }
        if deduped
            .last()
            .is_some_and(|existing| values_match_with_tol(*existing, value))
        {
            continue;
        }
        deduped.push(value.max(0.0));
    }
    if deduped.is_empty() || !values_match_with_tol(deduped[0], 0.0) {
        deduped.insert(0, 0.0);
    }
    deduped
}

fn sorted_profitable_direct_prefixes(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
) -> Option<Vec<(usize, f64)>> {
    let sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
    let mut profitable: Vec<(usize, f64, &'static str)> = sims
        .iter()
        .enumerate()
        .filter_map(|(idx, sim)| {
            let direct_prof = profitability(sim.prediction, sim.price());
            (direct_prof.is_finite() && direct_prof > 0.0).then_some((
                idx,
                direct_prof,
                sim.market_name,
            ))
        })
        .collect();
    profitable.sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.2.cmp(right.2)));
    Some(
        profitable
            .into_iter()
            .map(|(idx, prof, _)| (idx, prof))
            .collect(),
    )
}

fn mixed_frontier_entries_for_target_prof(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    active_mask: &[bool],
    target_prof: f64,
    active_buy_limit_mode: ActiveBuyLimitMode,
) -> Option<(
    Vec<PoolSim>,
    Vec<CoupledMixedFrontierEntry>,
    MixedFrontierAggregates,
)> {
    let sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
    let mut entries = Vec::with_capacity(sims.len());
    let mut active_target_sum = 0.0;
    let mut inactive_current_sum = 0.0;
    let mut max_required_shift = 0.0_f64;

    for (idx, sim) in sims.iter().enumerate() {
        let raw_target_price = target_price_for_prof(sim.prediction, target_prof);
        let holding = starting_state
            .holdings
            .get(sim.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let tol = 1e-9 * (1.0 + sim.price().abs().max(raw_target_price.abs()));
        if active_mask.get(idx).copied().unwrap_or(false) {
            if matches!(
                active_buy_limit_mode,
                ActiveBuyLimitMode::RequireWithinBuyLimit
            ) && raw_target_price > sim.buy_limit_price + tol
            {
                return None;
            }
            let target_price = match active_buy_limit_mode {
                ActiveBuyLimitMode::ClampToBuyLimit => raw_target_price.min(sim.buy_limit_price),
                ActiveBuyLimitMode::RequireWithinBuyLimit => raw_target_price,
            }
            .max(sim.price());
            let (buy_cost, buy_amount, achieved_price) = if target_price > sim.price() + tol {
                sim.cost_to_price(target_price)?
            } else {
                (0.0, 0.0, sim.price())
            };
            active_target_sum += achieved_price.max(sim.price());
            entries.push(CoupledMixedFrontierEntry {
                idx,
                market_name: sim.market_name,
                current_price: sim.price(),
                holding,
                active: true,
                buy_amount: buy_amount.max(0.0),
                buy_cost: buy_cost.max(0.0),
                sell_cap: 0.0,
            });
            continue;
        }

        inactive_current_sum += sim.price();
        let target_price = raw_target_price.max(sim.sell_limit_price).min(sim.price());
        let (sell_cap, _proceeds, _achieved_price) = if target_price + tol < sim.price() {
            sim.sell_to_price(target_price)?
        } else {
            (0.0, 0.0, sim.price())
        };
        max_required_shift = max_required_shift.max((sell_cap.max(0.0) - holding).max(0.0));
        entries.push(CoupledMixedFrontierEntry {
            idx,
            market_name: sim.market_name,
            current_price: sim.price(),
            holding,
            active: false,
            buy_amount: 0.0,
            buy_cost: 0.0,
            sell_cap: sell_cap.max(0.0),
        });
    }

    Some((
        sims,
        entries,
        MixedFrontierAggregates {
            active_target_sum,
            inactive_current_sum,
            max_required_shift,
        },
    ))
}

fn coupled_mixed_frontier_entries_for_target_prof(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    active_mask: &[bool],
    target_prof: f64,
) -> Option<(
    Vec<PoolSim>,
    Vec<CoupledMixedFrontierEntry>,
    MixedFrontierAggregates,
)> {
    mixed_frontier_entries_for_target_prof(
        starting_state,
        predictions,
        active_mask,
        target_prof,
        ActiveBuyLimitMode::ClampToBuyLimit,
    )
}

fn coupled_mixed_delta_and_proceeds_for_common_shift(
    sims: &[PoolSim],
    entries: &[CoupledMixedFrontierEntry],
    common_shift: f64,
) -> Option<(f64, f64)> {
    let mut delta = 0.0;
    let mut proceeds = 0.0;
    for entry in entries {
        if entry.active || entry.sell_cap <= DUST {
            continue;
        }
        let amount = entry.sell_cap.min(entry.holding + common_shift);
        if amount <= DUST {
            continue;
        }
        let (sold, leg_proceeds, new_price) = sims[entry.idx].sell_exact(amount)?;
        if sold + EPS < amount || !leg_proceeds.is_finite() || !new_price.is_finite() {
            return None;
        }
        delta += (entry.current_price - new_price).max(0.0);
        proceeds += leg_proceeds;
    }
    Some((delta, proceeds))
}

fn sequential_mixed_budget_for_action_order(
    mint_amount: f64,
    sell_proceeds: f64,
    direct_cost: f64,
) -> MixedSequentialBudget {
    let mint_amount = mint_amount.max(0.0);
    let sell_proceeds = sell_proceeds.max(0.0);
    let direct_cost = direct_cost.max(0.0);
    let mint_net_cost = mint_amount - sell_proceeds;
    let required_initial_cash = mint_amount.max(direct_cost + mint_net_cost).max(0.0);
    MixedSequentialBudget {
        required_initial_cash,
        mint_net_cost,
    }
}

fn build_mixed_certificate(
    active_mask: &[bool],
    pi: f64,
    mint_amount: f64,
    direct_cost: f64,
    sell_proceeds: f64,
    budget_cap: f64,
    delta_target: f64,
    delta_realized: f64,
    raw_ev: f64,
) -> MixedCertificate {
    let sequential_budget =
        sequential_mixed_budget_for_action_order(mint_amount, sell_proceeds, direct_cost);
    MixedCertificate {
        active_mask: active_mask.to_vec(),
        active_set_size: active_mask.iter().filter(|active| **active).count(),
        pi,
        mint_amount,
        direct_cost,
        sell_proceeds,
        mint_net_cost: sequential_budget.mint_net_cost,
        budget_used: sequential_budget.required_initial_cash,
        budget_residual: budget_cap.max(0.0) - sequential_budget.required_initial_cash,
        delta_target,
        delta_realized,
        raw_ev,
        estimated_fee_susd: None,
        estimated_net_ev: None,
    }
}

fn solve_coupled_mixed_common_shift(
    sims: &[PoolSim],
    entries: &[CoupledMixedFrontierEntry],
    delta_target: f64,
    max_common_shift: f64,
) -> Option<f64> {
    if delta_target <= DUST {
        return Some(0.0);
    }
    let (delta_zero, _) = coupled_mixed_delta_and_proceeds_for_common_shift(sims, entries, 0.0)?;
    if delta_zero + raw_value_tol(delta_zero, delta_target) >= delta_target {
        return Some(0.0);
    }
    if max_common_shift <= DUST {
        return None;
    }
    let (delta_hi, _) =
        coupled_mixed_delta_and_proceeds_for_common_shift(sims, entries, max_common_shift)?;
    if delta_hi + raw_value_tol(delta_hi, delta_target) < delta_target {
        return None;
    }

    let mut lo = 0.0;
    let mut hi = max_common_shift;
    let mut guess = (delta_target / delta_hi).clamp(0.0, 1.0) * max_common_shift;
    for _ in 0..4 {
        if guess <= lo + EPS || guess >= hi - EPS {
            break;
        }
        let (delta_guess, _) =
            coupled_mixed_delta_and_proceeds_for_common_shift(sims, entries, guess)?;
        let tol = raw_value_tol(delta_guess, delta_target);
        if delta_guess + tol < delta_target {
            lo = guess;
        } else {
            hi = guess;
        }
        let step = (max_common_shift * 1e-6).max(1e-9);
        let next = (guess + step).min(max_common_shift);
        if next <= guess + EPS {
            break;
        }
        let (delta_next, _) =
            coupled_mixed_delta_and_proceeds_for_common_shift(sims, entries, next)?;
        let slope = (delta_next - delta_guess) / (next - guess);
        if !slope.is_finite() || slope <= EPS {
            break;
        }
        let new_guess = (guess - (delta_guess - delta_target) / slope).clamp(lo, hi);
        if (new_guess - guess).abs() <= raw_value_tol(new_guess, guess) {
            break;
        }
        guess = new_guess;
    }

    for _ in 0..64 {
        if (hi - lo).abs() <= raw_value_tol(hi, lo) {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let (delta_mid, _) = coupled_mixed_delta_and_proceeds_for_common_shift(sims, entries, mid)?;
        let tol = raw_value_tol(delta_mid, delta_target);
        if delta_mid + tol < delta_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Some(hi.max(0.0))
}

fn compile_coupled_mixed_actions_for_entries_and_shift(
    starting_state: &SolverStateSnapshot,
    sims: &[PoolSim],
    entries: &[CoupledMixedFrontierEntry],
    common_shift: f64,
) -> Option<Vec<Action>> {
    let representative_market = representative_complete_set_market(starting_state)?;
    let (contract_1, contract_2) = action_contract_pair(sims);
    let mut cash = starting_state.cash;
    let mut actions = Vec::new();

    if common_shift > DUST {
        if cash + EPS < common_shift {
            return None;
        }
        cash -= common_shift;
        actions.push(Action::Mint {
            contract_1,
            contract_2,
            amount: common_shift,
            target_market: representative_market,
        });
    }

    for entry in entries {
        if entry.active || entry.sell_cap <= DUST {
            continue;
        }
        let amount = entry.sell_cap.min(entry.holding + common_shift);
        if amount <= DUST {
            continue;
        }
        let (sold, proceeds, _new_price) = sims[entry.idx].sell_exact(amount)?;
        if sold + EPS < amount || sold <= DUST || !proceeds.is_finite() {
            return None;
        }
        cash += proceeds;
        actions.push(Action::Sell {
            market_name: entry.market_name,
            amount: sold,
            proceeds,
        });
    }

    for entry in entries {
        if !entry.active || entry.buy_amount <= DUST {
            continue;
        }
        if cash + EPS < entry.buy_cost {
            return None;
        }
        cash -= entry.buy_cost;
        actions.push(Action::Buy {
            market_name: entry.market_name,
            amount: entry.buy_amount,
            cost: entry.buy_cost,
        });
    }

    (!actions.is_empty()).then_some(actions)
}

fn compile_coupled_mixed_candidate_for_target_prof(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    active_mask: &[bool],
    target_prof: f64,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    active_set_size: usize,
) -> Option<PlanResult> {
    let (sims, entries, aggregates) = coupled_mixed_frontier_entries_for_target_prof(
        starting_state,
        predictions,
        active_mask,
        target_prof,
    )?;
    let base = (1.0 - aggregates.inactive_current_sum).max(0.0);
    let delta_target = (aggregates.active_target_sum - base).max(0.0);
    let max_common_shift = starting_state
        .cash
        .max(0.0)
        .min(aggregates.max_required_shift.max(0.0));
    let common_shift =
        solve_coupled_mixed_common_shift(&sims, &entries, delta_target, max_common_shift)?;
    let (delta_realized, sell_proceeds) =
        coupled_mixed_delta_and_proceeds_for_common_shift(&sims, &entries, common_shift)?;
    let direct_cost: f64 = entries
        .iter()
        .filter(|entry| entry.active)
        .map(|entry| entry.buy_cost.max(0.0))
        .sum();
    let actions = compile_coupled_mixed_actions_for_entries_and_shift(
        starting_state,
        &sims,
        &entries,
        common_shift,
    )?;
    let candidate_terminal_state =
        apply_actions_to_solver_state(starting_state, &actions, predictions)?;
    let candidate_raw_ev = state_snapshot_expected_value(&candidate_terminal_state, predictions);
    let candidate = build_plan_result(
        starting_state,
        candidate_terminal_state,
        actions,
        candidate_raw_ev,
        None,
        Vec::new(),
        family,
        cost_config,
        PlanCompilerVariant::CoupledMixed,
        Some(common_shift),
        Some(target_prof),
        Some(active_set_size),
    );
    Some(with_mixed_certificates(
        candidate,
        vec![build_mixed_certificate(
            active_mask,
            target_prof,
            common_shift,
            direct_cost,
            sell_proceeds,
            starting_state.cash.max(0.0),
            delta_target,
            delta_realized,
            candidate_raw_ev,
        )],
    ))
}

fn compile_coupled_mixed_candidate_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
) -> Option<PlanResult> {
    let profitable = sorted_profitable_direct_prefixes(starting_state, predictions)?;
    if profitable.is_empty() {
        return None;
    }

    let mut best: Option<PlanResult> = None;
    for prefix_end in 0..profitable.len() {
        let active_set_size = prefix_end + 1;
        let mut active_mask = vec![false; starting_state.slot0_results.len()];
        for (idx, _) in profitable.iter().take(active_set_size) {
            if let Some(slot) = active_mask.get_mut(*idx) {
                *slot = true;
            }
        }

        let pi_hi = profitable[prefix_end].1.max(0.0);
        let pi_lo = profitable
            .get(prefix_end + 1)
            .map(|(_, prof)| (*prof).max(0.0))
            .unwrap_or(0.0);
        if pi_hi <= DUST {
            continue;
        }

        let Some(mut best_prefix) = compile_coupled_mixed_candidate_for_target_prof(
            starting_state,
            predictions,
            &active_mask,
            pi_hi,
            family,
            cost_config,
            active_set_size,
        ) else {
            continue;
        };

        if let Some(candidate_at_lo) = compile_coupled_mixed_candidate_for_target_prof(
            starting_state,
            predictions,
            &active_mask,
            pi_lo,
            family,
            cost_config,
            active_set_size,
        ) {
            best_prefix = candidate_at_lo;
        } else {
            let mut lo = pi_lo;
            let mut hi = pi_hi;
            for _ in 0..56 {
                if (hi - lo).abs() <= raw_value_tol(hi, lo) {
                    break;
                }
                let mid = 0.5 * (lo + hi);
                if let Some(candidate_mid) = compile_coupled_mixed_candidate_for_target_prof(
                    starting_state,
                    predictions,
                    &active_mask,
                    mid,
                    family,
                    cost_config,
                    active_set_size,
                ) {
                    best_prefix = candidate_mid;
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
        }

        if best
            .as_ref()
            .is_none_or(|incumbent| plan_result_is_better(&best_prefix, incumbent))
        {
            best = Some(best_prefix);
        }
    }

    best
}

fn constant_l_mixed_frontier_entries_for_target_prof(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    active_mask: &[bool],
    target_prof: f64,
) -> Option<(
    Vec<PoolSim>,
    Vec<CoupledMixedFrontierEntry>,
    MixedFrontierAggregates,
)> {
    mixed_frontier_entries_for_target_prof(
        starting_state,
        predictions,
        active_mask,
        target_prof,
        ActiveBuyLimitMode::RequireWithinBuyLimit,
    )
}

fn evaluate_constant_l_mixed_cost_at_target_prof(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    active_mask: &[bool],
    target_prof: f64,
    budget_cap: f64,
) -> Option<ConstantLMixedPiEvaluation> {
    let budget_cap = budget_cap.max(0.0).min(starting_state.cash.max(0.0));
    let (sims, entries, aggregates) = constant_l_mixed_frontier_entries_for_target_prof(
        starting_state,
        predictions,
        active_mask,
        target_prof,
    )?;
    let base = (1.0 - aggregates.inactive_current_sum).max(0.0);
    let delta_target = (aggregates.active_target_sum - base).max(0.0);
    let max_common_shift = budget_cap.min(aggregates.max_required_shift.max(0.0));
    if delta_target > DUST
        && (!entries
            .iter()
            .any(|entry| !entry.active && entry.sell_cap > DUST)
            || max_common_shift <= DUST)
    {
        return None;
    }
    let mint_amount =
        solve_coupled_mixed_common_shift(&sims, &entries, delta_target, max_common_shift)?;
    let (delta_realized, sell_proceeds) =
        coupled_mixed_delta_and_proceeds_for_common_shift(&sims, &entries, mint_amount)?;
    let direct_cost: f64 = entries
        .iter()
        .filter(|entry| entry.active)
        .map(|entry| entry.buy_cost.max(0.0))
        .sum();
    let sequential_budget =
        sequential_mixed_budget_for_action_order(mint_amount, sell_proceeds, direct_cost);
    let required_initial_cash = sequential_budget.required_initial_cash;
    let cost_tol = raw_value_tol(required_initial_cash, budget_cap);
    (required_initial_cash <= budget_cap + cost_tol).then_some(ConstantLMixedPiEvaluation {
        required_initial_cash,
        mint_amount,
        direct_cost,
        sell_proceeds,
        mint_net_cost: sequential_budget.mint_net_cost,
        budget_used: required_initial_cash,
        budget_residual: budget_cap - required_initial_cash,
        delta_target,
        delta_realized,
    })
}

fn build_plan_result_from_constant_l_fixed_active_solution(
    starting_state: &SolverStateSnapshot,
    solution: ConstantLMixedFixedActiveSolution,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
) -> PlanResult {
    let certificate = solution.certificate.clone();
    let plan = build_plan_result(
        starting_state,
        solution.terminal_state,
        solution.actions,
        solution.raw_ev,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
        PlanCompilerVariant::ConstantLMixed,
        Some(certificate.mint_amount),
        Some(certificate.pi),
        Some(certificate.active_set_size),
    );
    with_mixed_certificates(plan, vec![certificate])
}

fn compile_constant_l_mixed_fixed_active_solution_for_target_prof(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    budget_cap: f64,
    active_mask: &[bool],
    target_prof: f64,
    active_set_size: usize,
) -> Option<ConstantLMixedFixedActiveSolution> {
    debug_assert_eq!(
        active_set_size,
        active_mask.iter().filter(|active| **active).count()
    );
    let evaluation = evaluate_constant_l_mixed_cost_at_target_prof(
        starting_state,
        predictions,
        active_mask,
        target_prof,
        budget_cap,
    )?;
    let (sims, entries, _aggregates) = constant_l_mixed_frontier_entries_for_target_prof(
        starting_state,
        predictions,
        active_mask,
        target_prof,
    )?;
    let actions = compile_coupled_mixed_actions_for_entries_and_shift(
        starting_state,
        &sims,
        &entries,
        evaluation.mint_amount,
    )?;
    let candidate_terminal_state =
        apply_actions_to_solver_state(starting_state, &actions, predictions)?;
    let candidate_raw_ev = state_snapshot_expected_value(&candidate_terminal_state, predictions);
    Some(ConstantLMixedFixedActiveSolution {
        actions,
        terminal_state: candidate_terminal_state,
        raw_ev: candidate_raw_ev,
        certificate: build_mixed_certificate(
            active_mask,
            target_prof,
            evaluation.mint_amount,
            evaluation.direct_cost,
            evaluation.sell_proceeds,
            budget_cap,
            evaluation.delta_target,
            evaluation.delta_realized,
            candidate_raw_ev,
        ),
    })
}

fn solve_constant_l_target_prof_for_active_mask(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    budget_cap: f64,
    active_mask: &[bool],
) -> Option<(f64, usize)> {
    let active_set_size = active_mask.iter().filter(|active| **active).count();
    if active_set_size == 0 {
        return None;
    }

    let sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
    let mut pi_hi = f64::INFINITY;
    for (idx, active) in active_mask.iter().copied().enumerate() {
        if !active {
            continue;
        }
        let sim = sims.get(idx)?;
        let direct_prof = profitability(sim.prediction, sim.price());
        if !direct_prof.is_finite() || direct_prof <= DUST {
            return None;
        }
        pi_hi = pi_hi.min(direct_prof);
    }
    if !pi_hi.is_finite() || pi_hi <= DUST {
        return None;
    }

    let budget_cap = budget_cap.max(0.0).min(starting_state.cash.max(0.0));
    let cost_lo = evaluate_constant_l_mixed_cost_at_target_prof(
        starting_state,
        predictions,
        active_mask,
        0.0,
        budget_cap,
    );
    let target_prof = if let Some(cost_lo) = cost_lo {
        let tol = raw_value_tol(cost_lo.required_initial_cash, budget_cap);
        if cost_lo.required_initial_cash <= budget_cap + tol {
            0.0
        } else {
            let cost_hi = evaluate_constant_l_mixed_cost_at_target_prof(
                starting_state,
                predictions,
                active_mask,
                pi_hi,
                budget_cap,
            )?;
            let tol = raw_value_tol(cost_hi.required_initial_cash, budget_cap);
            if cost_hi.required_initial_cash > budget_cap + tol {
                return None;
            }

            let mut lo = 0.0;
            let mut hi = pi_hi;
            for _ in 0..CONSTANT_L_SOLVE_ITERS {
                if (hi - lo).abs() <= raw_value_tol(hi, lo) {
                    break;
                }
                let mid = 0.5 * (lo + hi);
                let Some(cost_mid) = evaluate_constant_l_mixed_cost_at_target_prof(
                    starting_state,
                    predictions,
                    active_mask,
                    mid,
                    budget_cap,
                ) else {
                    lo = mid;
                    continue;
                };
                let tol = raw_value_tol(cost_mid.required_initial_cash, budget_cap);
                if cost_mid.required_initial_cash > budget_cap + tol {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            hi
        }
    } else {
        let cost_hi = evaluate_constant_l_mixed_cost_at_target_prof(
            starting_state,
            predictions,
            active_mask,
            pi_hi,
            budget_cap,
        )?;
        let tol = raw_value_tol(cost_hi.required_initial_cash, budget_cap);
        if cost_hi.required_initial_cash > budget_cap + tol {
            return None;
        }
        let mut lo = 0.0;
        let mut hi = pi_hi;
        for _ in 0..CONSTANT_L_SOLVE_ITERS {
            if (hi - lo).abs() <= raw_value_tol(hi, lo) {
                break;
            }
            let mid = 0.5 * (lo + hi);
            let Some(cost_mid) = evaluate_constant_l_mixed_cost_at_target_prof(
                starting_state,
                predictions,
                active_mask,
                mid,
                budget_cap,
            ) else {
                lo = mid;
                continue;
            };
            let tol = raw_value_tol(cost_mid.required_initial_cash, budget_cap);
            if cost_mid.required_initial_cash > budget_cap + tol {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    };

    Some((target_prof, active_set_size))
}

fn compile_constant_l_mixed_fixed_active_solution_for_active_mask_with_budget_cap(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    budget_cap: f64,
    active_mask: &[bool],
) -> Option<ConstantLMixedFixedActiveSolution> {
    let (target_prof, active_set_size) = solve_constant_l_target_prof_for_active_mask(
        starting_state,
        predictions,
        budget_cap,
        active_mask,
    )?;
    compile_constant_l_mixed_fixed_active_solution_for_target_prof(
        starting_state,
        predictions,
        budget_cap,
        active_mask,
        target_prof,
        active_set_size,
    )
}

fn compile_constant_l_mixed_candidate_for_active_mask_with_budget_cap_cached(
    cache: &mut ConstantLMixedSolveCache,
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    budget_cap: f64,
    active_mask: &[bool],
) -> Option<PlanResult> {
    let budget_cap = budget_cap.max(0.0).min(starting_state.cash.max(0.0));
    let key = ConstantLMixedSolveKey {
        state_fingerprint: solver_state_fingerprint(starting_state, predictions),
        budget_cap_bits: budget_cap.to_bits(),
        active_mask: active_mask.to_vec(),
    };
    let solution = if let Some(cached) = cache.fixed_active_solutions.get(&key) {
        cached.clone()
    } else {
        let computed =
            compile_constant_l_mixed_fixed_active_solution_for_active_mask_with_budget_cap(
                starting_state,
                predictions,
                budget_cap,
                active_mask,
            );
        cache.fixed_active_solutions.insert(key, computed.clone());
        computed
    }?;
    Some(build_plan_result_from_constant_l_fixed_active_solution(
        starting_state,
        solution,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
    ))
}

fn active_mask_from_actions(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<bool> {
    let mut index_by_market = HashMap::new();
    for (idx, (_, market)) in slot0_results.iter().enumerate() {
        index_by_market.insert(market.name, idx);
    }
    let mut active_mask = vec![false; slot0_results.len()];
    for action in actions {
        if let Action::Buy { market_name, .. } = action
            && let Some(slot) = index_by_market.get(market_name)
            && let Some(active) = active_mask.get_mut(*slot)
        {
            *active = true;
        }
    }
    active_mask
}

fn active_mask_from_plan_result(
    plan: &PlanResult,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<bool> {
    plan.mixed_certificates
        .first()
        .map(|certificate| certificate.active_mask.clone())
        .unwrap_or_else(|| active_mask_from_actions(&plan.actions, slot0_results))
}

fn dedup_active_masks(masks: Vec<Vec<bool>>) -> Vec<Vec<bool>> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for mask in masks {
        if !mask.iter().any(|active| *active) || !seen.insert(mask.clone()) {
            continue;
        }
        deduped.push(mask);
    }
    deduped
}

fn constant_l_seed_active_masks(
    cache: &mut ConstantLMixedSolveCache,
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    seed_mode: ConstantLSeedMode,
) -> Option<(Vec<usize>, Vec<Vec<bool>>)> {
    let cache_key = (
        solver_state_fingerprint(starting_state, predictions),
        seed_mode,
        family,
    );
    if let Some(cached) = cache.seed_sets.get(&cache_key) {
        let cached = cached.clone()?;
        return Some((cached.profitable_indices, cached.seed_masks));
    }

    let profitable = sorted_profitable_direct_prefixes(starting_state, predictions)?;
    if profitable.is_empty() {
        cache.seed_sets.insert(cache_key, None);
        return None;
    }
    let profitable_indices: Vec<usize> = profitable.iter().map(|(idx, _)| *idx).collect();
    let mut seed_masks = Vec::new();

    for prefix_len in 1..=profitable_indices.len() {
        let mut active_mask = vec![false; starting_state.slot0_results.len()];
        for idx in profitable_indices.iter().take(prefix_len) {
            if let Some(slot) = active_mask.get_mut(*idx) {
                *slot = true;
            }
        }
        seed_masks.push(active_mask);
    }

    for idx in &profitable_indices {
        let mut active_mask = vec![false; starting_state.slot0_results.len()];
        if let Some(slot) = active_mask.get_mut(*idx) {
            *slot = true;
        }
        seed_masks.push(active_mask);
    }

    let allow_extra_solver_seeds = match seed_mode {
        ConstantLSeedMode::RuntimeCapped => {
            profitable_indices.len() <= CONSTANT_L_EXTRA_SEED_MAX_PROFITABLE
        }
        ConstantLSeedMode::TeacherBestKnown => true,
    };

    if allow_extra_solver_seeds {
        if let Some(analytic_seed) = compile_best_frontier_candidate_for_program_net_ev(
            starting_state,
            predictions,
            None,
            Vec::new(),
            family,
            cost_config,
            PlanCompilerVariant::AnalyticMixed,
            true,
            true,
        ) {
            seed_masks.push(active_mask_from_plan_result(
                &analytic_seed,
                &starting_state.slot0_results,
            ));
        }

        if let Some(direct_only_seed) = compile_best_frontier_candidate_for_program_net_ev(
            starting_state,
            predictions,
            None,
            Vec::new(),
            family,
            cost_config,
            PlanCompilerVariant::DirectOnly,
            false,
            false,
        ) {
            seed_masks.push(active_mask_from_plan_result(
                &direct_only_seed,
                &starting_state.slot0_results,
            ));
        }
    }

    let seed_set = ConstantLSeedSet {
        profitable_indices,
        seed_masks: dedup_active_masks(seed_masks),
    };
    cache.seed_sets.insert(cache_key, Some(seed_set.clone()));
    Some((seed_set.profitable_indices, seed_set.seed_masks))
}

fn improve_constant_l_candidate_with_local_search(
    cache: &mut ConstantLMixedSolveCache,
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    budget_cap: f64,
    profitable_indices: &[usize],
    initial_best: PlanResult,
) -> PlanResult {
    let mut best = initial_best;
    let mut visited: HashSet<Vec<bool>> = HashSet::new();
    visited.insert(active_mask_from_plan_result(
        &best,
        &starting_state.slot0_results,
    ));

    loop {
        let current_mask = active_mask_from_plan_result(&best, &starting_state.slot0_results);
        let active_indices: Vec<usize> = current_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, active)| active.then_some(idx))
            .collect();
        let inactive_profitable: Vec<usize> = profitable_indices
            .iter()
            .copied()
            .filter(|idx| !current_mask[*idx])
            .collect();
        let mut best_neighbor: Option<PlanResult> = None;

        for idx in &inactive_profitable {
            let mut neighbor = current_mask.clone();
            neighbor[*idx] = true;
            if !visited.insert(neighbor.clone()) {
                continue;
            }
            if let Some(candidate) =
                compile_constant_l_mixed_candidate_for_active_mask_with_budget_cap_cached(
                    cache,
                    starting_state,
                    predictions,
                    frontier_family,
                    preserve_markets.clone(),
                    family,
                    cost_config,
                    budget_cap,
                    &neighbor,
                )
                && best_neighbor
                    .as_ref()
                    .is_none_or(|incumbent| plan_result_is_better(&candidate, incumbent))
            {
                best_neighbor = Some(candidate);
            }
        }

        if active_indices.len() > 1 {
            for idx in &active_indices {
                let mut neighbor = current_mask.clone();
                neighbor[*idx] = false;
                if !visited.insert(neighbor.clone()) {
                    continue;
                }
                if let Some(candidate) =
                    compile_constant_l_mixed_candidate_for_active_mask_with_budget_cap_cached(
                        cache,
                        starting_state,
                        predictions,
                        frontier_family,
                        preserve_markets.clone(),
                        family,
                        cost_config,
                        budget_cap,
                        &neighbor,
                    )
                    && best_neighbor
                        .as_ref()
                        .is_none_or(|incumbent| plan_result_is_better(&candidate, incumbent))
                {
                    best_neighbor = Some(candidate);
                }
            }
        }

        for drop_idx in &active_indices {
            for add_idx in &inactive_profitable {
                let mut neighbor = current_mask.clone();
                neighbor[*drop_idx] = false;
                neighbor[*add_idx] = true;
                if !visited.insert(neighbor.clone()) {
                    continue;
                }
                if let Some(candidate) =
                    compile_constant_l_mixed_candidate_for_active_mask_with_budget_cap_cached(
                        cache,
                        starting_state,
                        predictions,
                        frontier_family,
                        preserve_markets.clone(),
                        family,
                        cost_config,
                        budget_cap,
                        &neighbor,
                    )
                    && best_neighbor
                        .as_ref()
                        .is_none_or(|incumbent| plan_result_is_better(&candidate, incumbent))
                {
                    best_neighbor = Some(candidate);
                }
            }
        }

        let Some(neighbor) = best_neighbor else {
            break;
        };
        if !plan_result_is_better(&neighbor, &best) {
            break;
        }
        best = neighbor;
    }

    best
}

fn compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap_and_seed_mode(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    budget_cap: f64,
    seed_mode: ConstantLSeedMode,
) -> Option<PlanResult> {
    let budget_cap = budget_cap.max(0.0).min(starting_state.cash.max(0.0));
    if budget_cap <= DUST {
        return None;
    }

    let mut cache = ConstantLMixedSolveCache::default();
    let (profitable_indices, seed_masks) = constant_l_seed_active_masks(
        &mut cache,
        starting_state,
        predictions,
        family,
        cost_config,
        seed_mode,
    )?;
    let mut best: Option<PlanResult> = None;
    for active_mask in seed_masks {
        let Some(candidate) =
            compile_constant_l_mixed_candidate_for_active_mask_with_budget_cap_cached(
                &mut cache,
                starting_state,
                predictions,
                frontier_family,
                preserve_markets.clone(),
                family,
                cost_config,
                budget_cap,
                &active_mask,
            )
        else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|incumbent| plan_result_is_better(&candidate, incumbent))
        {
            best = Some(candidate);
        }
    }

    best.map(|seed_best| {
        let allow_local_search = match seed_mode {
            ConstantLSeedMode::RuntimeCapped => {
                profitable_indices.len() <= CONSTANT_L_LOCAL_SEARCH_MAX_PROFITABLE
            }
            ConstantLSeedMode::TeacherBestKnown => true,
        };
        if !allow_local_search {
            seed_best
        } else {
            improve_constant_l_candidate_with_local_search(
                &mut cache,
                starting_state,
                predictions,
                frontier_family,
                preserve_markets,
                family,
                cost_config,
                budget_cap,
                &profitable_indices,
                seed_best,
            )
        }
    })
}

fn compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    budget_cap: f64,
) -> Option<PlanResult> {
    compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap_and_seed_mode(
        starting_state,
        predictions,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
        budget_cap,
        ConstantLSeedMode::RuntimeCapped,
    )
}

fn compile_constant_l_mixed_candidate_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
) -> Option<PlanResult> {
    compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap(
        starting_state,
        predictions,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
        starting_state.cash.max(0.0),
    )
}

#[cfg(test)]
fn compile_constant_l_mixed_best_known_candidate_for_program_net_ev_with_budget_cap(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    budget_cap: f64,
    cost_config: PlannerCostConfig,
) -> Option<PlanResult> {
    compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap_and_seed_mode(
        starting_state,
        predictions,
        None,
        Vec::new(),
        SolverFamily::Plain,
        cost_config,
        budget_cap,
        ConstantLSeedMode::TeacherBestKnown,
    )
}

fn profitability_step_count(actions: &[Action]) -> usize {
    group_execution_actions_by_profitability_step(actions)
        .map(|groups| groups.len())
        .unwrap_or(0)
}

fn staged_constant_l_stage_count(stage_1: &PlanResult, stage_2: &PlanResult) -> usize {
    usize::from(!stage_1.actions.is_empty()) + usize::from(!stage_2.actions.is_empty())
}

fn compose_staged_constant_l_plan(
    starting_state: &SolverStateSnapshot,
    stage_1: &PlanResult,
    stage_2: &PlanResult,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    stage_1_budget_fraction: f64,
) -> Option<PlanResult> {
    let mut actions = stage_1.actions.clone();
    actions.extend(stage_2.actions.iter().cloned());
    if actions.is_empty() {
        return None;
    }

    let terminal_state = if stage_2.actions.is_empty() {
        stage_1.terminal_state.clone()
    } else {
        stage_2.terminal_state.clone()
    };
    let raw_ev = if stage_2.actions.is_empty() {
        stage_1.raw_ev
    } else {
        stage_2.raw_ev
    };
    let mut plan = build_plan_result(
        starting_state,
        terminal_state,
        actions,
        raw_ev,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
        PlanCompilerVariant::StagedConstantLMixed2,
        stage_2
            .selected_common_shift
            .or(stage_1.selected_common_shift),
        stage_2
            .selected_mixed_lambda
            .or(stage_1.selected_mixed_lambda),
        stage_2
            .selected_active_set_size
            .or(stage_1.selected_active_set_size),
    );
    plan.selected_stage_count = Some(staged_constant_l_stage_count(stage_1, stage_2));
    plan.selected_stage1_budget_fraction = Some(stage_1_budget_fraction);
    plan.mixed_certificates = stage_1
        .mixed_certificates
        .iter()
        .chain(stage_2.mixed_certificates.iter())
        .cloned()
        .collect();
    Some(plan)
}

fn compile_staged_constant_l_mixed_candidate_for_fraction(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    stage_1_budget_fraction: f64,
) -> Option<PlanResult> {
    let stage_1_budget_cap = starting_state.cash.max(0.0) * stage_1_budget_fraction;
    let stage_1_raw_ev = state_snapshot_expected_value(starting_state, predictions);
    let stage_1 = compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap(
        starting_state,
        predictions,
        frontier_family,
        preserve_markets.clone(),
        family,
        cost_config,
        stage_1_budget_cap,
    )
    .unwrap_or_else(|| solver_identity_plan(starting_state, family, stage_1_raw_ev, cost_config));
    let stage_2_state = if stage_1.actions.is_empty() {
        starting_state.clone()
    } else {
        stage_1.terminal_state.clone()
    };
    let stage_2_raw_ev = state_snapshot_expected_value(&stage_2_state, predictions);
    let stage_2 = compile_constant_l_mixed_candidate_for_program_net_ev(
        &stage_2_state,
        predictions,
        frontier_family,
        preserve_markets.clone(),
        family,
        cost_config,
    )
    .unwrap_or_else(|| solver_identity_plan(&stage_2_state, family, stage_2_raw_ev, cost_config));
    compose_staged_constant_l_plan(
        starting_state,
        &stage_1,
        &stage_2,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
        stage_1_budget_fraction,
    )
}

fn compile_staged_constant_l_mixed_candidate_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    rich_raw_ev: f64,
    rich_actions: &[Action],
    best_compact_candidate: &PlanResult,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    constant_l_candidate: Option<&PlanResult>,
) -> Option<PlanResult> {
    if constant_l_candidate.is_none()
        || rich_raw_ev <= best_compact_candidate.raw_ev + STAGED_CONSTANT_L_RAW_EV_GAP_SUSD
        || profitability_step_count(rich_actions) <= 1
    {
        return None;
    }

    let mut evaluated_fractions: Vec<f64> = Vec::new();
    let mut staged_candidates: Vec<(f64, PlanResult)> = Vec::new();
    let evaluate_fraction =
        |fraction: f64,
         evaluated_fractions: &mut Vec<f64>,
         staged_candidates: &mut Vec<(f64, PlanResult)>| {
            if evaluated_fractions.iter().any(|existing| {
                (existing - fraction).abs() <= STAGED_CONSTANT_L_FRACTION_REFINEMENT_EPS
            }) {
                return;
            }
            evaluated_fractions.push(fraction);
            if let Some(candidate) = compile_staged_constant_l_mixed_candidate_for_fraction(
                starting_state,
                predictions,
                frontier_family,
                preserve_markets.clone(),
                family,
                cost_config,
                fraction,
            ) {
                staged_candidates.push((fraction, candidate));
            }
        };

    for fraction in STAGED_CONSTANT_L_FRACTIONS {
        evaluate_fraction(fraction, &mut evaluated_fractions, &mut staged_candidates);
    }

    if staged_candidates.is_empty() {
        return None;
    }

    let mut best_index = 0usize;
    for idx in 1..staged_candidates.len() {
        if plan_result_is_better(&staged_candidates[idx].1, &staged_candidates[best_index].1) {
            best_index = idx;
        }
    }

    let best_fraction = staged_candidates[best_index].0;
    if let Some(base_index) = STAGED_CONSTANT_L_FRACTIONS
        .iter()
        .position(|fraction| values_match_with_tol(*fraction, best_fraction))
        && base_index > 0
        && base_index + 1 < STAGED_CONSTANT_L_FRACTIONS.len()
    {
        let left = 0.5 * (STAGED_CONSTANT_L_FRACTIONS[base_index - 1] + best_fraction);
        let right = 0.5 * (best_fraction + STAGED_CONSTANT_L_FRACTIONS[base_index + 1]);
        evaluate_fraction(left, &mut evaluated_fractions, &mut staged_candidates);
        evaluate_fraction(right, &mut evaluated_fractions, &mut staged_candidates);
    }

    staged_candidates
        .into_iter()
        .map(|(_, candidate)| candidate)
        .min_by(plan_result_cmp)
}

fn frontier_profitability_candidates(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    include_mint_profitability: bool,
) -> Option<Vec<f64>> {
    let sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
    let price_sum: f64 = sims.iter().map(PoolSim::price).sum();
    let mut candidates = vec![0.0_f64];
    for (idx, sim) in sims.iter().enumerate() {
        let direct_prof = profitability(sim.prediction, sim.price());
        if direct_prof.is_finite() && direct_prof > 0.0 {
            candidates.push(direct_prof);
        }
        if include_mint_profitability {
            let mint_prof = profitability(sim.prediction, alt_price(&sims, idx, price_sum));
            if mint_prof.is_finite() && mint_prof > 0.0 {
                candidates.push(mint_prof);
            }
        }
    }
    Some(deduped_numeric_candidates(candidates))
}

fn frontier_adjustments_for_target_prof(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    target_prof: f64,
) -> Option<Vec<FrontierDirectAdjustment>> {
    let sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
    let mut adjustments = Vec::with_capacity(sims.len());
    for sim in &sims {
        let target_price = target_price_for_prof(sim.prediction, target_prof);
        let tol = EPS * (1.0 + sim.price().abs().max(target_price.abs()));
        if target_price > sim.price() + tol {
            let (_buy_cost, buy_amount, _new_price) = sim.cost_to_price(target_price)?;
            adjustments.push(FrontierDirectAdjustment {
                market_name: sim.market_name,
                buy_amount: buy_amount.max(0.0),
                sell_amount: 0.0,
            });
            continue;
        }
        if target_price + tol < sim.price() {
            let (sell_amount, _proceeds, _new_price) = sim.sell_to_price(target_price)?;
            adjustments.push(FrontierDirectAdjustment {
                market_name: sim.market_name,
                buy_amount: 0.0,
                sell_amount: sell_amount.max(0.0),
            });
            continue;
        }
        adjustments.push(FrontierDirectAdjustment {
            market_name: sim.market_name,
            buy_amount: 0.0,
            sell_amount: 0.0,
        });
    }
    Some(adjustments)
}

fn required_common_shift_candidates(
    starting_state: &SolverStateSnapshot,
    adjustments: &[FrontierDirectAdjustment],
    allow_common_shift: bool,
) -> Vec<f64> {
    if !allow_common_shift {
        return vec![0.0];
    }
    let max_common_shift = starting_state.cash.max(0.0);
    let mut shifts = vec![0.0_f64];
    for adjustment in adjustments {
        if adjustment.sell_amount <= DUST {
            continue;
        }
        let holding = starting_state
            .holdings
            .get(adjustment.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let shortage = (adjustment.sell_amount - holding).max(0.0);
        if shortage > max_common_shift + EPS {
            continue;
        }
        shifts.push(shortage);
    }
    deduped_numeric_candidates(shifts)
}

fn compile_frontier_actions_for_adjustments_and_shift(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    adjustments: &[FrontierDirectAdjustment],
    common_shift: f64,
    allow_common_shift: bool,
) -> Option<Vec<Action>> {
    if !allow_common_shift && common_shift > DUST {
        return None;
    }

    let mut sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
    let mut sim_idx_by_market: HashMap<&'static str, usize> = HashMap::new();
    for (idx, sim) in sims.iter().enumerate() {
        sim_idx_by_market.insert(sim.market_name, idx);
    }

    let (contract_1, contract_2) = action_contract_pair(&sims);
    let representative_market = representative_complete_set_market(starting_state)?;
    let mut cash = starting_state.cash;
    let mut actions = Vec::new();

    if common_shift > DUST {
        if cash + EPS < common_shift {
            return None;
        }
        cash -= common_shift;
        actions.push(Action::Mint {
            contract_1,
            contract_2,
            amount: common_shift,
            target_market: representative_market,
        });
    }

    for adjustment in adjustments {
        if adjustment.sell_amount <= DUST {
            continue;
        }
        let starting_holding = starting_state
            .holdings
            .get(adjustment.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let available = starting_holding + common_shift;
        let amount = adjustment.sell_amount.min(available);
        if amount <= DUST {
            continue;
        }
        let idx = *sim_idx_by_market.get(adjustment.market_name)?;
        let (sold, proceeds, new_price) = sims[idx].sell_exact(amount)?;
        if sold <= DUST || !proceeds.is_finite() || !new_price.is_finite() {
            return None;
        }
        sims[idx].set_price(new_price);
        cash += proceeds;
        actions.push(Action::Sell {
            market_name: adjustment.market_name,
            amount: sold,
            proceeds,
        });
    }

    for adjustment in adjustments {
        if adjustment.buy_amount <= DUST {
            continue;
        }
        let idx = *sim_idx_by_market.get(adjustment.market_name)?;
        let (bought, cost, new_price) = sims[idx].buy_exact(adjustment.buy_amount)?;
        if bought + EPS < adjustment.buy_amount
            || bought <= DUST
            || !cost.is_finite()
            || !new_price.is_finite()
            || cash + EPS < cost
        {
            return None;
        }
        sims[idx].set_price(new_price);
        cash -= cost;
        actions.push(Action::Buy {
            market_name: adjustment.market_name,
            amount: bought,
            cost,
        });
    }

    if actions.is_empty() {
        return None;
    }
    Some(actions)
}

fn compile_best_frontier_candidate_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    compiler_variant: PlanCompilerVariant,
    allow_common_shift: bool,
    include_mint_profitability: bool,
) -> Option<PlanResult> {
    let profitability_candidates =
        frontier_profitability_candidates(starting_state, predictions, include_mint_profitability)?;
    let mut best: Option<PlanResult> = None;

    for target_prof in profitability_candidates {
        let Some(adjustments) =
            frontier_adjustments_for_target_prof(starting_state, predictions, target_prof)
        else {
            continue;
        };
        let common_shift_candidates =
            required_common_shift_candidates(starting_state, &adjustments, allow_common_shift);
        for common_shift in common_shift_candidates {
            let Some(actions) = compile_frontier_actions_for_adjustments_and_shift(
                starting_state,
                predictions,
                &adjustments,
                common_shift,
                allow_common_shift,
            ) else {
                continue;
            };
            let Some(candidate_terminal_state) =
                apply_actions_to_solver_state(starting_state, &actions, predictions)
            else {
                continue;
            };
            let candidate_raw_ev =
                state_snapshot_expected_value(&candidate_terminal_state, predictions);
            let mut candidate = build_plan_result(
                starting_state,
                candidate_terminal_state.clone(),
                actions,
                candidate_raw_ev,
                frontier_family,
                preserve_markets.clone(),
                family,
                cost_config,
                compiler_variant,
                allow_common_shift.then_some(common_shift),
                Some(target_prof),
                None,
            );
            if compiler_variant == PlanCompilerVariant::AnalyticMixed {
                let start_sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
                let end_sims =
                    build_sims(&candidate_terminal_state.slot0_results, predictions).ok()?;
                let active_mask: Vec<bool> = adjustments
                    .iter()
                    .map(|adjustment| adjustment.buy_amount > DUST)
                    .collect();
                let direct_cost: f64 = candidate
                    .actions
                    .iter()
                    .filter_map(|action| match action {
                        Action::Buy { cost, .. } => Some(*cost),
                        _ => None,
                    })
                    .sum();
                let sell_proceeds: f64 = candidate
                    .actions
                    .iter()
                    .filter_map(|action| match action {
                        Action::Sell { proceeds, .. } => Some(*proceeds),
                        _ => None,
                    })
                    .sum();
                let delta_target: f64 = start_sims
                    .iter()
                    .zip(end_sims.iter())
                    .filter(|(start, _)| {
                        adjustments
                            .iter()
                            .find(|adjustment| adjustment.market_name == start.market_name)
                            .is_some_and(|adjustment| adjustment.buy_amount > DUST)
                    })
                    .map(|(start, end)| (end.price() - start.price()).max(0.0))
                    .sum();
                let delta_realized: f64 = start_sims
                    .iter()
                    .zip(end_sims.iter())
                    .filter(|(start, _)| {
                        adjustments
                            .iter()
                            .find(|adjustment| adjustment.market_name == start.market_name)
                            .is_some_and(|adjustment| adjustment.sell_amount > DUST)
                    })
                    .map(|(start, end)| (start.price() - end.price()).max(0.0))
                    .sum();
                candidate = with_mixed_certificates(
                    candidate,
                    vec![build_mixed_certificate(
                        &active_mask,
                        target_prof,
                        common_shift.max(0.0),
                        direct_cost,
                        sell_proceeds,
                        starting_state.cash.max(0.0),
                        delta_target,
                        delta_realized,
                        candidate_raw_ev,
                    )],
                );
            }
            if best
                .as_ref()
                .is_none_or(|incumbent| plan_result_is_better(&candidate, incumbent))
            {
                best = Some(candidate);
            }
        }
    }

    best
}

fn compile_target_delta_actions_for_common_shift(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    target_state: &SolverStateSnapshot,
    common_shift: f64,
) -> Option<Vec<Action>> {
    let deltas = sorted_market_deltas(starting_state, target_state);
    let representative_market = representative_complete_set_market(starting_state)?;
    let mut sims = build_sims(&starting_state.slot0_results, predictions).ok()?;
    let mut sim_idx_by_market: HashMap<&'static str, usize> = HashMap::new();
    for (idx, sim) in sims.iter().enumerate() {
        sim_idx_by_market.insert(sim.market_name, idx);
    }
    let (contract_1, contract_2) = action_contract_pair(&sims);
    let mut cash = starting_state.cash;
    let mut actions = Vec::new();

    if common_shift > EPS {
        if cash + EPS < common_shift {
            return None;
        }
        cash -= common_shift;
        actions.push(Action::Mint {
            contract_1,
            contract_2,
            amount: common_shift,
            target_market: representative_market,
        });
        for (market_name, delta) in &deltas {
            if *delta + EPS >= common_shift {
                continue;
            }
            let amount = common_shift - *delta;
            if amount <= DUST {
                continue;
            }
            let idx = *sim_idx_by_market.get(market_name)?;
            let (sold, proceeds, new_price) = sims[idx].sell_exact(amount)?;
            if sold + EPS < amount
                || sold <= DUST
                || !proceeds.is_finite()
                || !new_price.is_finite()
            {
                return None;
            }
            sims[idx].set_price(new_price);
            cash += proceeds;
            actions.push(Action::Sell {
                market_name: *market_name,
                amount: sold,
                proceeds,
            });
        }
        for (market_name, delta) in &deltas {
            if *delta <= common_shift + EPS {
                continue;
            }
            let amount = *delta - common_shift;
            if amount <= DUST {
                continue;
            }
            let idx = *sim_idx_by_market.get(market_name)?;
            let (bought, cost, new_price) = sims[idx].buy_exact(amount)?;
            if bought + EPS < amount
                || bought <= DUST
                || !cost.is_finite()
                || !new_price.is_finite()
            {
                return None;
            }
            if cash + EPS < cost {
                return None;
            }
            sims[idx].set_price(new_price);
            cash -= cost;
            actions.push(Action::Buy {
                market_name: *market_name,
                amount: bought,
                cost,
            });
        }
    } else if common_shift < -EPS {
        for (market_name, delta) in &deltas {
            if *delta <= common_shift + EPS {
                continue;
            }
            let amount = *delta - common_shift;
            if amount <= DUST {
                continue;
            }
            let idx = *sim_idx_by_market.get(market_name)?;
            let (bought, cost, new_price) = sims[idx].buy_exact(amount)?;
            if bought + EPS < amount
                || bought <= DUST
                || !cost.is_finite()
                || !new_price.is_finite()
            {
                return None;
            }
            if cash + EPS < cost {
                return None;
            }
            sims[idx].set_price(new_price);
            cash -= cost;
            actions.push(Action::Buy {
                market_name: *market_name,
                amount: bought,
                cost,
            });
        }
        let merge_amount = -common_shift;
        if merge_amount <= DUST {
            return None;
        }
        actions.push(Action::Merge {
            contract_1,
            contract_2,
            amount: merge_amount,
            source_market: representative_market,
        });
        for (market_name, delta) in &deltas {
            if *delta + EPS >= common_shift {
                continue;
            }
            let amount = common_shift - *delta;
            if amount <= DUST {
                continue;
            }
            let idx = *sim_idx_by_market.get(market_name)?;
            let (sold, proceeds, new_price) = sims[idx].sell_exact(amount)?;
            if sold + EPS < amount
                || sold <= DUST
                || !proceeds.is_finite()
                || !new_price.is_finite()
            {
                return None;
            }
            sims[idx].set_price(new_price);
            actions.push(Action::Sell {
                market_name: *market_name,
                amount: sold,
                proceeds,
            });
        }
    } else {
        for (market_name, delta) in &deltas {
            if *delta >= -EPS {
                continue;
            }
            let amount = -*delta;
            if amount <= DUST {
                continue;
            }
            let idx = *sim_idx_by_market.get(market_name)?;
            let (sold, proceeds, new_price) = sims[idx].sell_exact(amount)?;
            if sold + EPS < amount
                || sold <= DUST
                || !proceeds.is_finite()
                || !new_price.is_finite()
            {
                return None;
            }
            sims[idx].set_price(new_price);
            cash += proceeds;
            actions.push(Action::Sell {
                market_name: *market_name,
                amount: sold,
                proceeds,
            });
        }
        for (market_name, delta) in &deltas {
            if *delta <= EPS {
                continue;
            }
            let amount = *delta;
            if amount <= DUST {
                continue;
            }
            let idx = *sim_idx_by_market.get(market_name)?;
            let (bought, cost, new_price) = sims[idx].buy_exact(amount)?;
            if bought + EPS < amount
                || bought <= DUST
                || !cost.is_finite()
                || !new_price.is_finite()
            {
                return None;
            }
            if cash + EPS < cost {
                return None;
            }
            sims[idx].set_price(new_price);
            cash -= cost;
            actions.push(Action::Buy {
                market_name: *market_name,
                amount: bought,
                cost,
            });
        }
    }

    if actions.is_empty() {
        return None;
    }
    Some(actions)
}

fn compile_target_delta_candidate_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    target_state: &SolverStateSnapshot,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    allow_common_shift: bool,
) -> Option<PlanResult> {
    let deltas = sorted_market_deltas(starting_state, target_state);
    let common_shifts = if allow_common_shift {
        deduped_target_delta_common_shifts(&deltas)
    } else {
        vec![0.0]
    };
    let mut best: Option<PlanResult> = None;

    for common_shift in common_shifts {
        let Some(actions) = compile_target_delta_actions_for_common_shift(
            starting_state,
            predictions,
            target_state,
            common_shift,
        ) else {
            continue;
        };
        let Some(candidate_terminal_state) =
            apply_actions_to_solver_state(starting_state, &actions, predictions)
        else {
            continue;
        };
        if !holdings_match_within_tol(&candidate_terminal_state, target_state) {
            continue;
        }
        let candidate_raw_ev =
            state_snapshot_expected_value(&candidate_terminal_state, predictions);
        let candidate = build_plan_result(
            starting_state,
            candidate_terminal_state,
            actions,
            candidate_raw_ev,
            frontier_family,
            preserve_markets.clone(),
            family,
            cost_config,
            PlanCompilerVariant::TargetDelta,
            allow_common_shift.then_some(common_shift),
            None,
            None,
        );
        if best
            .as_ref()
            .is_none_or(|incumbent| plan_result_is_better(&candidate, incumbent))
        {
            best = Some(candidate);
        }
    }

    best
}

fn baseline_step_prune_candidate_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    initial_actions: Vec<Action>,
    compiler_variant: PlanCompilerVariant,
) -> PlanResult {
    let Some(initial_terminal_state) =
        apply_actions_to_solver_state(starting_state, &initial_actions, predictions)
    else {
        return solver_identity_plan_with_variant(
            starting_state,
            family,
            state_snapshot_expected_value(starting_state, predictions),
            cost_config,
            compiler_variant,
        );
    };
    let initial_raw_ev = state_snapshot_expected_value(&initial_terminal_state, predictions);
    let mut best = build_plan_result(
        starting_state,
        initial_terminal_state,
        initial_actions,
        initial_raw_ev,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
        compiler_variant,
        None,
        None,
        None,
    );

    // Single-pass reverse prune: try dropping each step group from lowest
    // profitability upward. Because reverse iteration monotonically frees
    // cash, restarting the loop provides near-zero additional compaction
    // at massive cost. A single pass is sufficient.
    let Ok(step_groups) = group_execution_actions_by_profitability_step(&best.actions) else {
        return best;
    };
    if step_groups.is_empty() {
        return best;
    }

    // Collect indices to exclude as a set for O(1) lookup.
    let mut excluded: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for step_group in step_groups.iter().rev() {
        // Tentatively add this group's indices to the excluded set.
        for &idx in &step_group.action_indices {
            excluded.insert(idx);
        }
        let keep_actions: Vec<Action> = best
            .actions
            .iter()
            .enumerate()
            .filter(|(i, _)| !excluded.contains(i))
            .map(|(_, a)| a.clone())
            .collect();
        let Some(candidate_terminal_state) =
            apply_actions_to_solver_state(starting_state, &keep_actions, predictions)
        else {
            // Can't drop this group — remove from excluded and continue.
            for &idx in &step_group.action_indices {
                excluded.remove(&idx);
            }
            continue;
        };
        // Reject pruned plans that spend more cash than available.
        if candidate_terminal_state.cash < -EPS {
            for &idx in &step_group.action_indices {
                excluded.remove(&idx);
            }
            continue;
        }
        let candidate_raw_ev =
            state_snapshot_expected_value(&candidate_terminal_state, predictions);
        let candidate = build_plan_result(
            starting_state,
            candidate_terminal_state,
            keep_actions,
            candidate_raw_ev,
            frontier_family,
            best.preserve_markets.clone(),
            family,
            cost_config,
            PlanCompilerVariant::BaselineStepPrune,
            None,
            None,
            None,
        );
        if plan_result_is_better(&candidate, &best) {
            best = candidate;
            // Keep excluded — this group stays dropped for subsequent tries.
        } else {
            // Dropping this group didn't help — restore it.
            for &idx in &step_group.action_indices {
                excluded.remove(&idx);
            }
        }
    }

    best
}

#[allow(dead_code)]
fn route_group_prune_candidate_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    initial_actions: Vec<Action>,
    compiler_variant: PlanCompilerVariant,
) -> PlanResult {
    let Some(initial_terminal_state) =
        apply_actions_to_solver_state(starting_state, &initial_actions, predictions)
    else {
        return solver_identity_plan_with_variant(
            starting_state,
            family,
            state_snapshot_expected_value(starting_state, predictions),
            cost_config,
            compiler_variant,
        );
    };
    let initial_raw_ev = state_snapshot_expected_value(&initial_terminal_state, predictions);
    let mut best = build_plan_result(
        starting_state,
        initial_terminal_state,
        initial_actions,
        initial_raw_ev,
        frontier_family,
        preserve_markets,
        family,
        cost_config,
        compiler_variant,
        None,
        None,
        None,
    );

    loop {
        let Ok(route_groups) = group_execution_actions(&best.actions) else {
            break;
        };
        if route_groups.is_empty() {
            break;
        }

        let mut improved = false;
        for route_group in route_groups.iter().rev() {
            let mut keep_actions = Vec::with_capacity(best.actions.len());
            for (action_index, action) in best.actions.iter().enumerate() {
                if !route_group.action_indices.contains(&action_index) {
                    keep_actions.push(action.clone());
                }
            }
            let Some(candidate_terminal_state) =
                apply_actions_to_solver_state(starting_state, &keep_actions, predictions)
            else {
                continue;
            };
            let candidate_raw_ev =
                state_snapshot_expected_value(&candidate_terminal_state, predictions);
            let candidate = build_plan_result(
                starting_state,
                candidate_terminal_state,
                keep_actions,
                candidate_raw_ev,
                frontier_family,
                best.preserve_markets.clone(),
                family,
                cost_config,
                compiler_variant,
                None,
                None,
                None,
            );
            if plan_result_is_better(&candidate, &best) {
                best = candidate;
                improved = true;
                break;
            }
        }

        if !improved {
            break;
        }
    }

    best
}

fn compact_raw_no_arb_plan_for_program_net_ev(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
    initial_actions: Vec<Action>,
    allow_common_shift: bool,
) -> PlanResult {
    let starting_raw_ev = state_snapshot_expected_value(starting_state, predictions);
    let Some(rich_terminal_state) =
        apply_actions_to_solver_state(starting_state, &initial_actions, predictions)
    else {
        return solver_identity_plan(starting_state, family, starting_raw_ev, cost_config);
    };
    let rich_raw_ev = state_snapshot_expected_value(&rich_terminal_state, predictions);
    let baseline = baseline_step_prune_candidate_for_program_net_ev(
        starting_state,
        predictions,
        frontier_family,
        preserve_markets.clone(),
        family,
        cost_config,
        initial_actions.clone(),
        PlanCompilerVariant::BaselineStepPrune,
    );
    let mut best = baseline;
    let constant_l_candidate = allow_common_shift
        .then(|| {
            compile_constant_l_mixed_candidate_for_program_net_ev(
                starting_state,
                predictions,
                frontier_family,
                preserve_markets.clone(),
                family,
                cost_config,
            )
        })
        .flatten();
    if let Some(constant_l) = constant_l_candidate.as_ref()
        && plan_result_is_better(constant_l, &best)
    {
        best = constant_l.clone();
    }
    if let Some(target_delta) = compile_target_delta_candidate_for_program_net_ev(
        starting_state,
        predictions,
        &rich_terminal_state,
        frontier_family,
        preserve_markets.clone(),
        family,
        cost_config,
        allow_common_shift,
    ) && plan_result_is_better(&target_delta, &best)
    {
        best = target_delta;
    }
    if let Some(staged_constant_l) = compile_staged_constant_l_mixed_candidate_for_program_net_ev(
        starting_state,
        predictions,
        rich_raw_ev,
        &initial_actions,
        &best,
        frontier_family,
        preserve_markets.clone(),
        family,
        cost_config,
        constant_l_candidate.as_ref(),
    ) && plan_result_is_better(&staged_constant_l, &best)
    {
        best = staged_constant_l;
    }

    let noop = solver_identity_plan(starting_state, family, starting_raw_ev, cost_config);
    if plan_result_is_better(&noop, &best) {
        best = noop;
    }

    best
}

fn action_cmp(left: &Action, right: &Action) -> Ordering {
    fn cmp_f64(left: f64, right: f64) -> Ordering {
        left.total_cmp(&right)
    }

    match (left, right) {
        (
            Action::Mint {
                contract_1: left_contract_1,
                contract_2: left_contract_2,
                amount: left_amount,
                target_market: left_target_market,
            },
            Action::Mint {
                contract_1: right_contract_1,
                contract_2: right_contract_2,
                amount: right_amount,
                target_market: right_target_market,
            },
        ) => left_target_market
            .cmp(right_target_market)
            .then_with(|| left_contract_1.cmp(right_contract_1))
            .then_with(|| left_contract_2.cmp(right_contract_2))
            .then_with(|| cmp_f64(*left_amount, *right_amount)),
        (
            Action::Buy {
                market_name: left_market_name,
                amount: left_amount,
                cost: left_cost,
            },
            Action::Buy {
                market_name: right_market_name,
                amount: right_amount,
                cost: right_cost,
            },
        ) => left_market_name
            .cmp(right_market_name)
            .then_with(|| cmp_f64(*left_amount, *right_amount))
            .then_with(|| cmp_f64(*left_cost, *right_cost)),
        (
            Action::Sell {
                market_name: left_market_name,
                amount: left_amount,
                proceeds: left_proceeds,
            },
            Action::Sell {
                market_name: right_market_name,
                amount: right_amount,
                proceeds: right_proceeds,
            },
        ) => left_market_name
            .cmp(right_market_name)
            .then_with(|| cmp_f64(*left_amount, *right_amount))
            .then_with(|| cmp_f64(*left_proceeds, *right_proceeds)),
        (
            Action::Merge {
                contract_1: left_contract_1,
                contract_2: left_contract_2,
                amount: left_amount,
                source_market: left_source_market,
            },
            Action::Merge {
                contract_1: right_contract_1,
                contract_2: right_contract_2,
                amount: right_amount,
                source_market: right_source_market,
            },
        ) => left_source_market
            .cmp(right_source_market)
            .then_with(|| left_contract_1.cmp(right_contract_1))
            .then_with(|| left_contract_2.cmp(right_contract_2))
            .then_with(|| cmp_f64(*left_amount, *right_amount)),
        (Action::Mint { .. }, _) => Ordering::Less,
        (_, Action::Mint { .. }) => Ordering::Greater,
        (Action::Buy { .. }, _) => Ordering::Less,
        (_, Action::Buy { .. }) => Ordering::Greater,
        (Action::Sell { .. }, _) => Ordering::Less,
        (_, Action::Sell { .. }) => Ordering::Greater,
    }
}

fn actions_cmp(left: &[Action], right: &[Action]) -> Ordering {
    for (left_action, right_action) in left.iter().zip(right.iter()) {
        let ordering = action_cmp(left_action, right_action);
        if !ordering.is_eq() {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn plan_result_is_better(candidate: &PlanResult, incumbent: &PlanResult) -> bool {
    let economic_cmp = net_ev_cmp(
        candidate.estimated_net_ev,
        incumbent.estimated_net_ev,
        candidate.raw_ev,
        incumbent.raw_ev,
    );
    if !economic_cmp.is_eq() {
        return economic_cmp.is_lt();
    }
    if candidate.estimated_tx_count != incumbent.estimated_tx_count {
        return candidate.estimated_tx_count.unwrap_or(usize::MAX)
            < incumbent.estimated_tx_count.unwrap_or(usize::MAX);
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
    if candidate.compiler_variant != incumbent.compiler_variant {
        return candidate.compiler_variant.stable_rank() < incumbent.compiler_variant.stable_rank();
    }
    if candidate.selected_common_shift != incumbent.selected_common_shift {
        return candidate
            .selected_common_shift
            .unwrap_or(0.0)
            .total_cmp(&incumbent.selected_common_shift.unwrap_or(0.0))
            .is_lt();
    }
    if candidate.selected_mixed_lambda != incumbent.selected_mixed_lambda {
        return candidate
            .selected_mixed_lambda
            .unwrap_or(0.0)
            .total_cmp(&incumbent.selected_mixed_lambda.unwrap_or(0.0))
            .is_lt();
    }
    if candidate.selected_active_set_size != incumbent.selected_active_set_size {
        return candidate.selected_active_set_size.unwrap_or(usize::MAX)
            < incumbent.selected_active_set_size.unwrap_or(usize::MAX);
    }
    if candidate.actions != incumbent.actions {
        return actions_cmp(&candidate.actions, &incumbent.actions).is_lt();
    }
    false
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

fn public_forecastflows_run_telemetry(
    telemetry: &ForecastFlowsTelemetry,
    fallback_reason: Option<&'static str>,
) -> ForecastFlowsRunTelemetry {
    ForecastFlowsRunTelemetry {
        strategy: telemetry.strategy.clone(),
        worker_roundtrip_ms: telemetry.worker_roundtrip_ms,
        driver_overhead_ms: telemetry.driver_overhead_ms,
        translation_replay_ms: telemetry.translation_replay_ms,
        local_candidate_build_ms: telemetry.local_candidate_build_ms,
        local_step_prune_ms: telemetry.local_step_prune_ms,
        local_route_prune_ms: telemetry.local_route_prune_ms,
        workspace_reused: telemetry.workspace_reused,
        direct_solver_time_ms: telemetry.direct_solver_time_ms,
        mixed_solver_time_ms: telemetry.mixed_solver_time_ms,
        estimated_execution_cost_susd: telemetry.estimated_execution_cost_susd,
        estimated_net_ev_susd: telemetry.estimated_net_ev_susd,
        validated_total_fee_susd: telemetry.validated_total_fee_susd,
        validated_net_ev_susd: telemetry.validated_net_ev_susd,
        fee_estimate_error_susd: telemetry.fee_estimate_error_susd,
        validation_only: telemetry.validation_only,
        solve_tuning: telemetry.solve_tuning.clone(),
        sysimage_status: telemetry.sysimage_status.clone(),
        fallback_reason: fallback_reason.map(str::to_string),
    }
}

fn summary_from_plan_result(plan: &PlanResult, stats: &SolverRunStats) -> RebalancePlanSummary {
    RebalancePlanSummary {
        raw_ev: plan.raw_ev,
        estimated_total_fee_susd: plan.estimated_total_fee_susd,
        estimated_net_ev: plan.estimated_net_ev,
        estimated_group_count: plan.estimated_group_count,
        estimated_tx_count: plan.estimated_tx_count,
        action_count: plan.actions.len(),
        forecastflows_telemetry: public_forecastflows_run_telemetry(
            &stats.forecastflows_telemetry,
            stats.forecastflows_fallback_reason,
        ),
        family_stable_rank: plan.family.stable_rank(),
        frontier_family_stable_rank: first_frontier_family_stable_rank(plan.frontier_family),
        preserve_markets: plan
            .preserve_markets
            .iter()
            .map(|market| (*market).to_string())
            .collect(),
        compiler_variant_stable_rank: plan.compiler_variant.stable_rank(),
        selected_common_shift: plan.selected_common_shift,
        selected_mixed_lambda: plan.selected_mixed_lambda,
        selected_active_set_size: plan.selected_active_set_size,
    }
}

fn decision_from_plan_result(plan: PlanResult, stats: &SolverRunStats) -> RebalancePlanDecision {
    let summary = summary_from_plan_result(&plan, stats);
    RebalancePlanDecision {
        actions: plan.actions,
        summary,
    }
}

pub fn compare_rebalance_plan_decisions(
    left: &RebalancePlanDecision,
    right: &RebalancePlanDecision,
) -> Ordering {
    let economic_cmp = net_ev_cmp(
        left.summary.estimated_net_ev,
        right.summary.estimated_net_ev,
        left.summary.raw_ev,
        right.summary.raw_ev,
    );
    if !economic_cmp.is_eq() {
        return economic_cmp;
    }
    if left.summary.estimated_tx_count != right.summary.estimated_tx_count {
        return left
            .summary
            .estimated_tx_count
            .unwrap_or(usize::MAX)
            .cmp(&right.summary.estimated_tx_count.unwrap_or(usize::MAX));
    }
    if left.summary.action_count != right.summary.action_count {
        return left.summary.action_count.cmp(&right.summary.action_count);
    }
    if left.summary.family_stable_rank != right.summary.family_stable_rank {
        return left
            .summary
            .family_stable_rank
            .cmp(&right.summary.family_stable_rank);
    }
    if left.summary.frontier_family_stable_rank != right.summary.frontier_family_stable_rank {
        return left
            .summary
            .frontier_family_stable_rank
            .cmp(&right.summary.frontier_family_stable_rank);
    }
    if left.summary.preserve_markets.len() != right.summary.preserve_markets.len() {
        return left
            .summary
            .preserve_markets
            .len()
            .cmp(&right.summary.preserve_markets.len());
    }
    if left.summary.preserve_markets != right.summary.preserve_markets {
        return left
            .summary
            .preserve_markets
            .cmp(&right.summary.preserve_markets);
    }
    if left.summary.compiler_variant_stable_rank != right.summary.compiler_variant_stable_rank {
        return left
            .summary
            .compiler_variant_stable_rank
            .cmp(&right.summary.compiler_variant_stable_rank);
    }
    if left.summary.selected_common_shift != right.summary.selected_common_shift {
        return left
            .summary
            .selected_common_shift
            .unwrap_or(0.0)
            .total_cmp(&right.summary.selected_common_shift.unwrap_or(0.0));
    }
    if left.summary.selected_mixed_lambda != right.summary.selected_mixed_lambda {
        return left
            .summary
            .selected_mixed_lambda
            .unwrap_or(0.0)
            .total_cmp(&right.summary.selected_mixed_lambda.unwrap_or(0.0));
    }
    if left.summary.selected_active_set_size != right.summary.selected_active_set_size {
        return left
            .summary
            .selected_active_set_size
            .unwrap_or(usize::MAX)
            .cmp(&right.summary.selected_active_set_size.unwrap_or(usize::MAX));
    }
    actions_cmp(&left.actions, &right.actions)
}

fn choose_head_to_head_plan_result(
    native: PlanResult,
    forecastflows: Option<PlanResult>,
) -> PlanResult {
    let mut chosen = native;
    if let Some(forecastflows) = forecastflows
        && plan_result_is_better(&forecastflows, &chosen)
    {
        chosen = forecastflows;
    }
    chosen
}

fn compose_rebalance_step(
    starting_state: &SolverStateSnapshot,
    prefix_actions: &[Action],
    rebalance: &PlanResult,
    family: SolverFamily,
    cost_config: PlannerCostConfig,
) -> PlanResult {
    let mut actions = prefix_actions.to_vec();
    actions.extend(rebalance.actions.iter().cloned());
    let mut plan = build_plan_result(
        starting_state,
        rebalance.terminal_state.clone(),
        actions,
        rebalance.raw_ev,
        rebalance.frontier_family,
        rebalance.preserve_markets.clone(),
        family,
        cost_config,
        rebalance.compiler_variant,
        rebalance.selected_common_shift,
        rebalance.selected_mixed_lambda,
        rebalance.selected_active_set_size,
    );
    plan.selected_stage_count = rebalance.selected_stage_count;
    plan.selected_stage1_budget_fraction = rebalance.selected_stage1_budget_fraction;
    plan.mixed_certificates = rebalance.mixed_certificates.clone();
    plan
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
    cost_config: PlannerCostConfig,
) -> Option<PlanResult> {
    let allow_common_shift = force_mint_available
        .unwrap_or(state.slot0_results.len() == expected_outcome_count);
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
    Some(compact_raw_no_arb_plan_for_program_net_ev(
        state,
        predictions,
        frontier_family,
        sorted_preserve_markets(preserve_markets),
        family,
        cost_config,
        actions,
        allow_common_shift,
    ))
}

fn run_positive_arb_plan_from_state(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    family: SolverFamily,
    stats: &mut SolverRunStats,
    cost_config: PlannerCostConfig,
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

    Some(build_plan_result(
        state,
        terminal_state,
        ctx.actions,
        ev_after,
        None,
        Vec::new(),
        family,
        cost_config,
        PlanCompilerVariant::NoOp,
        None,
        None,
        None,
    ))
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

fn descending_rank_map(
    features: &[DistilledMarketFeature],
    value: impl Fn(&DistilledMarketFeature) -> f64,
) -> HashMap<&'static str, usize> {
    let mut ranked: Vec<(&'static str, f64)> = features
        .iter()
        .map(|feature| (feature.market_name, value(feature)))
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    ranked
        .into_iter()
        .enumerate()
        .map(|(rank, (market_name, _))| (market_name, rank))
        .collect()
}

fn rank_weight(rank: usize, total: usize) -> f64 {
    total.saturating_sub(rank) as f64
}

fn collect_distilled_market_features(
    state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    preserve_scores: &[PreserveCandidateScore],
) -> Vec<DistilledMarketFeature> {
    let Ok(sims) = build_sims(&state.slot0_results, predictions) else {
        return Vec::new();
    };
    let price_sum: f64 = sims.iter().map(|sim| sim.price()).sum();
    let preserve_score_map: HashMap<&'static str, (f64, f64)> = preserve_scores
        .iter()
        .map(|score| (score.market_name, (score.churn_amount, score.sold_amount)))
        .collect();

    let mut features: Vec<DistilledMarketFeature> = sims
        .iter()
        .map(|sim| {
            let holding = state
                .holdings
                .get(sim.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let (churn_amount, sold_amount) = preserve_score_map
                .get(sim.market_name)
                .copied()
                .unwrap_or((0.0, 0.0));
            let direct_profitability = profitability(sim.prediction, sim.price());
            let prediction_minus_price = sim.prediction - sim.price();
            let mint_price = alt_price(
                &sims,
                sims.iter()
                    .position(|candidate| candidate.market_name == sim.market_name)
                    .expect("sim index should exist for current market"),
                price_sum,
            );
            let mint_profitability = profitability(sim.prediction, mint_price);
            let hold_through_mint_value = holding * (sim.prediction - mint_price).max(0.0);
            DistilledMarketFeature {
                market_name: sim.market_name,
                holding,
                churn_amount,
                sold_amount,
                direct_profitability,
                direct_profitability_rank: 0,
                prediction_minus_price,
                mint_profitability,
                hold_through_mint_value,
            }
        })
        .collect();

    let direct_ranks = descending_rank_map(&features, |feature| feature.direct_profitability);
    let feature_count = features.len();
    for feature in &mut features {
        feature.direct_profitability_rank = direct_ranks
            .get(feature.market_name)
            .copied()
            .unwrap_or(feature_count);
    }
    features.sort_by(|left, right| left.market_name.cmp(right.market_name));
    features
}

#[cfg(test)]
fn distilled_feature_rows(features: &[DistilledMarketFeature]) -> Vec<TeacherMarketFeatureRow> {
    features
        .iter()
        .map(|feature| TeacherMarketFeatureRow {
            market_name: feature.market_name,
            holding: feature.holding,
            churn_amount: feature.churn_amount,
            sold_amount: feature.sold_amount,
            direct_profitability: feature.direct_profitability,
            direct_profitability_rank: feature.direct_profitability_rank,
            prediction_minus_price: feature.prediction_minus_price,
            mint_profitability: feature.mint_profitability,
            hold_through_mint_value: feature.hold_through_mint_value,
        })
        .collect()
}

fn capped_top_markets_by_score(
    features: &[DistilledMarketFeature],
    score: impl Fn(&DistilledMarketFeature) -> f64,
) -> Vec<&'static str> {
    let mut ranked: Vec<(&'static str, f64)> = features
        .iter()
        .filter(|feature| {
            feature.holding > EPS
                && (feature.churn_amount > EPS
                    || feature.direct_profitability > 0.0
                    || feature.hold_through_mint_value > 0.0)
        })
        .map(|feature| (feature.market_name, score(feature)))
        .filter(|(_, score)| score.is_finite() && *score > 0.0)
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    ranked.truncate(MAX_DISTILLED_PRESERVE_SET_SIZE);
    ranked
        .into_iter()
        .map(|(market_name, _)| market_name)
        .collect()
}

fn choose_distilled_frontier_family(
    features: &[DistilledMarketFeature],
    preserve_markets: &[&'static str],
) -> Option<BundleRouteKind> {
    if preserve_markets.is_empty() {
        return None;
    }
    let preserve_set: HashSet<&'static str> = preserve_markets.iter().copied().collect();
    let mut direct_score = 0.0;
    let mut mint_score = 0.0;
    for feature in features {
        if !preserve_set.contains(feature.market_name) {
            continue;
        }
        let weight = 1.0 + feature.holding.sqrt();
        direct_score += feature.direct_profitability.max(0.0) * weight;
        mint_score +=
            (feature.mint_profitability.max(0.0) + feature.hold_through_mint_value) * weight;
    }
    if mint_score > direct_score + EPS {
        Some(BundleRouteKind::Mint)
    } else if direct_score > EPS {
        Some(BundleRouteKind::Direct)
    } else {
        None
    }
}

fn distilled_preserve_proposals(features: &[DistilledMarketFeature]) -> Vec<Vec<&'static str>> {
    if features.is_empty() {
        return Vec::new();
    }
    let total = features.len();
    let churn_ranks = descending_rank_map(features, |feature| feature.churn_amount);
    let holding_ranks = descending_rank_map(features, |feature| feature.holding);
    let direct_ranks = descending_rank_map(features, |feature| feature.direct_profitability);
    let hold_through_ranks =
        descending_rank_map(features, |feature| feature.hold_through_mint_value);

    let p1 = capped_top_markets_by_score(features, |feature| {
        6.0 * rank_weight(
            churn_ranks
                .get(feature.market_name)
                .copied()
                .unwrap_or(total),
            total,
        ) + 3.0
            * rank_weight(
                holding_ranks
                    .get(feature.market_name)
                    .copied()
                    .unwrap_or(total),
                total,
            )
            + 2.0
                * rank_weight(
                    direct_ranks
                        .get(feature.market_name)
                        .copied()
                        .unwrap_or(total),
                    total,
                )
            + 0.5 * feature.sold_amount
            + feature.prediction_minus_price.max(0.0)
    });
    let p2 = capped_top_markets_by_score(features, |feature| {
        6.0 * rank_weight(
            direct_ranks
                .get(feature.market_name)
                .copied()
                .unwrap_or(total),
            total,
        ) + 5.0
            * rank_weight(
                hold_through_ranks
                    .get(feature.market_name)
                    .copied()
                    .unwrap_or(total),
                total,
            )
            + 2.0
                * rank_weight(
                    holding_ranks
                        .get(feature.market_name)
                        .copied()
                        .unwrap_or(total),
                    total,
                )
            + rank_weight(
                churn_ranks
                    .get(feature.market_name)
                    .copied()
                    .unwrap_or(total),
                total,
            )
    });

    let mut proposals = Vec::new();
    if !p1.is_empty() {
        proposals.push(sorted_preserve_markets(&p1));
    }
    if !p2.is_empty() {
        let proposal = sorted_preserve_markets(&p2);
        if !proposals.contains(&proposal) {
            proposals.push(proposal);
        }
    }

    let p1_set: HashSet<&'static str> = p1.iter().copied().collect();
    let p2_set: HashSet<&'static str> = p2.iter().copied().collect();
    let symmetric_diff = p1_set.symmetric_difference(&p2_set).count();
    if symmetric_diff >= 2 {
        let union = capped_top_markets_by_score(features, |feature| {
            let p1_rank = rank_weight(
                churn_ranks
                    .get(feature.market_name)
                    .copied()
                    .unwrap_or(total),
                total,
            ) + rank_weight(
                direct_ranks
                    .get(feature.market_name)
                    .copied()
                    .unwrap_or(total),
                total,
            );
            let p2_rank = rank_weight(
                direct_ranks
                    .get(feature.market_name)
                    .copied()
                    .unwrap_or(total),
                total,
            ) + rank_weight(
                hold_through_ranks
                    .get(feature.market_name)
                    .copied()
                    .unwrap_or(total),
                total,
            );
            p1_rank.max(p2_rank) + feature.holding.max(0.0)
        });
        if !union.is_empty() {
            let union = sorted_preserve_markets(&union);
            if !proposals.contains(&union) {
                proposals.push(union);
            }
        }
    }

    proposals.truncate(MAX_DISTILLED_EXTRA_PROPOSAL_SETS);
    proposals
}

fn distilled_proposal_tasks(
    features: &[DistilledMarketFeature],
) -> Vec<(Option<BundleRouteKind>, Vec<&'static str>)> {
    let mut tasks = Vec::new();
    for preserve_markets in distilled_preserve_proposals(features) {
        tasks.push((None, preserve_markets.clone()));
        if let Some(frontier_family) = choose_distilled_frontier_family(features, &preserve_markets)
        {
            tasks.push((Some(frontier_family), preserve_markets));
        }
    }
    tasks
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
    cost_config: PlannerCostConfig,
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
                cost_config,
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
    cost_config: PlannerCostConfig,
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
            cost_config,
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
    include_distilled_proposals: bool,
    stats: &mut SolverRunStats,
    cost_config: PlannerCostConfig,
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
        cost_config,
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
            cost_config,
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
                    cost_config,
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
    let merged_preserve_scores = merge_preserve_candidate_scores(&preserve_score_sets);
    let preserve_universe =
        cap_preserve_candidate_universe(merged_preserve_scores.clone(), online_preserve_cap);
    let mut candidates = enumerate_exact_no_arb_candidates_over_preserve_universe(
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
        cost_config,
    );

    if include_distilled_proposals {
        let features =
            collect_distilled_market_features(state, predictions, &merged_preserve_scores);
        let proposal_tasks = distilled_proposal_tasks(&features);
        stats.distilled_proposal_sets += proposal_tasks.len();
        stats.distilled_proposal_candidate_evals += proposal_tasks.len();
        let mut proposal_candidates: Vec<PlanResult> = proposal_tasks
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
                    cost_config,
                )
            })
            .collect();
        stats.exact_rebalance_candidate_evals += proposal_candidates.len();
        candidates.append(&mut proposal_candidates);
        candidates.sort_by(plan_result_cmp);
    }

    let auto_mint_available =
        force_mint_available.unwrap_or(state.slot0_results.len() == expected_outcome_count);
    if auto_mint_available {
        if let Some(analytic_mixed) = compile_best_frontier_candidate_for_program_net_ev(
            state,
            predictions,
            None,
            Vec::new(),
            family,
            cost_config,
            PlanCompilerVariant::AnalyticMixed,
            true,
            true,
        ) {
            candidates.push(analytic_mixed);
        }
        if let Some(coupled_mixed) = compile_coupled_mixed_candidate_for_program_net_ev(
            state,
            predictions,
            family,
            cost_config,
        ) {
            candidates.push(coupled_mixed);
        }
    }
    if let Some(direct_only) = compile_best_frontier_candidate_for_program_net_ev(
        state,
        predictions,
        None,
        Vec::new(),
        family,
        cost_config,
        PlanCompilerVariant::DirectOnly,
        false,
        false,
    ) {
        candidates.push(direct_only);
    }
    candidates.sort_by(plan_result_cmp);

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
    cost_config: PlannerCostConfig,
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
        true,
        stats,
        cost_config,
    )
}

#[cfg(test)]
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
    include_distilled_proposals: bool,
    stats: &mut SolverRunStats,
    cost_config: PlannerCostConfig,
) -> PlanResult {
    let mut best = solver_identity_plan(
        state,
        family,
        state_snapshot_expected_value(state, predictions),
        cost_config,
    );
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
        include_distilled_proposals,
        stats,
        cost_config,
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
        best_raw_ev = best.raw_ev,
        best_estimated_net_ev = best.estimated_net_ev,
        best_estimated_total_fee_susd = best.estimated_total_fee_susd,
        best_actions = best.actions.len(),
        best_frontier_family = first_frontier_family_label(best.frontier_family),
        best_preserve_markets = best.preserve_markets.len(),
        best_compiler_variant = best.compiler_variant.as_str(),
        best_selected_common_shift = best.selected_common_shift,
        best_selected_mixed_lambda = best.selected_mixed_lambda,
        best_selected_active_set_size = best.selected_active_set_size,
        best_selected_stage_count = best.selected_stage_count,
        best_selected_stage1_budget_fraction = best.selected_stage1_budget_fraction,
        run_phase0_in_polish,
        online_preserve_cap,
        extra_preserve_seed_states = extra_preserve_seed_states.len(),
        "ultimate solver exact no-arb result"
    );

    best
}

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
        false,
        stats,
        cost_config,
    )
}

fn run_plain_family_plan(
    initial_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    route_gates: RouteGateThresholds,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
    stats: &mut SolverRunStats,
    cost_config: PlannerCostConfig,
) -> PlanResult {
    let mut best = solver_identity_plan(
        initial_state,
        SolverFamily::Plain,
        state_snapshot_expected_value(initial_state, predictions),
        cost_config,
    );
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
        cost_config,
    );
    for base in base_candidates {
        if plan_result_is_better(&base, &best) {
            best = base;
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
    cost_config: PlannerCostConfig,
) -> Option<PlanResult> {
    let root_arb = run_positive_arb_plan_from_state(
        initial_state,
        predictions,
        expected_outcome_count,
        route_gates,
        force_mint_available,
        SolverFamily::ArbPrimed,
        stats,
        cost_config,
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
        cost_config,
    );
    for exact_rebalance in exact_rebalance_candidates {
        let base = if exact_rebalance.actions.is_empty() {
            root_arb.clone()
        } else {
            compose_rebalance_step(
                initial_state,
                &root_arb.actions,
                &exact_rebalance,
                SolverFamily::ArbPrimed,
                cost_config,
            )
        };
        if plan_result_is_better(&base, &best) {
            best = base;
        }
    }

    Some(best)
}

fn forecastflows_plan_compiler_variant(
    variant: ForecastFlowsCandidateVariant,
) -> PlanCompilerVariant {
    match variant {
        ForecastFlowsCandidateVariant::Direct => PlanCompilerVariant::ForecastFlowsDirect,
        ForecastFlowsCandidateVariant::Mixed => PlanCompilerVariant::ForecastFlowsMixed,
    }
}

fn forecastflows_action_set_key(actions: &[Action]) -> Vec<ForecastFlowsActionKey> {
    actions
        .iter()
        .map(|action| match action {
            Action::Mint {
                contract_1,
                contract_2,
                amount,
                target_market,
            } => ForecastFlowsActionKey::Mint {
                contract_1: *contract_1,
                contract_2: *contract_2,
                amount_bits: amount.to_bits(),
                target_market: *target_market,
            },
            Action::Buy {
                market_name,
                amount,
                cost,
            } => ForecastFlowsActionKey::Buy {
                market_name: *market_name,
                amount_bits: amount.to_bits(),
                cost_bits: cost.to_bits(),
            },
            Action::Sell {
                market_name,
                amount,
                proceeds,
            } => ForecastFlowsActionKey::Sell {
                market_name: *market_name,
                amount_bits: amount.to_bits(),
                proceeds_bits: proceeds.to_bits(),
            },
            Action::Merge {
                contract_1,
                contract_2,
                amount,
                source_market,
            } => ForecastFlowsActionKey::Merge {
                contract_1: *contract_1,
                contract_2: *contract_2,
                amount_bits: amount.to_bits(),
                source_market: *source_market,
            },
        })
        .collect()
}

fn evaluate_forecastflows_action_set(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    actions: Vec<Action>,
    cost_config: PlannerCostConfig,
    evaluation_cache: &mut ForecastFlowsActionEvaluationCache,
) -> Option<ForecastFlowsEvaluatedActionSet> {
    let key = forecastflows_action_set_key(&actions);
    if let Some(cached) = evaluation_cache.evaluations.get(&key) {
        return cached.clone();
    }

    let evaluated =
        estimate_plan_cost_from_replay(&actions, &starting_state.slot0_results, cost_config)
            .and_then(|plan_cost| {
                let terminal_state =
                    apply_actions_to_solver_state(starting_state, &actions, predictions)?;
                let raw_ev = state_snapshot_expected_value(&terminal_state, predictions);
                Some(ForecastFlowsEvaluatedActionSet {
                    actions,
                    terminal_state,
                    raw_ev,
                    plan_cost,
                })
            });
    evaluation_cache.evaluations.insert(key, evaluated.clone());
    evaluated
}

fn build_plan_result_from_forecastflows_evaluation(
    evaluated: ForecastFlowsEvaluatedActionSet,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    compiler_variant: PlanCompilerVariant,
) -> PlanResult {
    let (
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source,
        _fee_estimate_unavailable,
    ) = plan_cost_fields_from_estimate(evaluated.raw_ev, evaluated.plan_cost);
    PlanResult {
        actions: evaluated.actions,
        terminal_state: evaluated.terminal_state,
        raw_ev: evaluated.raw_ev,
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source,
        frontier_family,
        preserve_markets,
        family: SolverFamily::ForecastFlows,
        compiler_variant,
        selected_common_shift: None,
        selected_mixed_lambda: None,
        selected_active_set_size: None,
        selected_stage_count: None,
        selected_stage1_budget_fraction: None,
        mixed_certificates: Vec::new(),
    }
}

fn build_forecastflows_raw_plan_result(
    initial_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    candidate: ForecastFlowsFamilyCandidate,
    cost_config: PlannerCostConfig,
    evaluation_cache: &mut ForecastFlowsActionEvaluationCache,
) -> Option<PlanResult> {
    let evaluated = evaluate_forecastflows_action_set(
        initial_state,
        predictions,
        candidate.actions,
        cost_config,
        evaluation_cache,
    )?;
    Some(build_plan_result_from_forecastflows_evaluation(
        evaluated,
        None,
        Vec::new(),
        forecastflows_plan_compiler_variant(candidate.variant),
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
fn baseline_step_prune_forecastflows_candidate_for_program_net_ev(
    initial_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    cost_config: PlannerCostConfig,
    initial_actions: Vec<Action>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    compiler_variant: PlanCompilerVariant,
    evaluation_cache: &mut ForecastFlowsActionEvaluationCache,
) -> PlanResult {
    let Some(initial_evaluated) = evaluate_forecastflows_action_set(
        initial_state,
        predictions,
        initial_actions,
        cost_config,
        evaluation_cache,
    ) else {
        return solver_identity_plan_with_variant(
            initial_state,
            SolverFamily::ForecastFlows,
            state_snapshot_expected_value(initial_state, predictions),
            cost_config,
            compiler_variant,
        );
    };
    let mut best = build_plan_result_from_forecastflows_evaluation(
        initial_evaluated,
        frontier_family,
        preserve_markets.clone(),
        compiler_variant,
    );

    loop {
        let Ok(step_groups) = group_execution_actions_by_profitability_step(&best.actions) else {
            break;
        };
        if step_groups.is_empty() {
            break;
        }

        let mut improved = false;
        for step_group in step_groups.iter().rev() {
            let mut keep_actions = Vec::with_capacity(best.actions.len());
            for (action_index, action) in best.actions.iter().enumerate() {
                if !step_group.action_indices.contains(&action_index) {
                    keep_actions.push(action.clone());
                }
            }
            let Some(candidate_evaluated) = evaluate_forecastflows_action_set(
                initial_state,
                predictions,
                keep_actions,
                cost_config,
                evaluation_cache,
            ) else {
                continue;
            };
            let candidate = build_plan_result_from_forecastflows_evaluation(
                candidate_evaluated,
                frontier_family,
                preserve_markets.clone(),
                compiler_variant,
            );
            if plan_result_is_better(&candidate, &best) {
                best = candidate;
                improved = true;
                break;
            }
        }

        if !improved {
            break;
        }
    }

    best
}

#[cfg_attr(not(test), allow(dead_code))]
fn route_group_prune_forecastflows_candidate_for_program_net_ev(
    initial_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    cost_config: PlannerCostConfig,
    initial_actions: Vec<Action>,
    frontier_family: Option<BundleRouteKind>,
    preserve_markets: Vec<&'static str>,
    compiler_variant: PlanCompilerVariant,
    evaluation_cache: &mut ForecastFlowsActionEvaluationCache,
) -> PlanResult {
    let Some(initial_evaluated) = evaluate_forecastflows_action_set(
        initial_state,
        predictions,
        initial_actions,
        cost_config,
        evaluation_cache,
    ) else {
        return solver_identity_plan_with_variant(
            initial_state,
            SolverFamily::ForecastFlows,
            state_snapshot_expected_value(initial_state, predictions),
            cost_config,
            compiler_variant,
        );
    };
    let mut best = build_plan_result_from_forecastflows_evaluation(
        initial_evaluated,
        frontier_family,
        preserve_markets.clone(),
        compiler_variant,
    );

    loop {
        let Ok(route_groups) = group_execution_actions(&best.actions) else {
            break;
        };
        if route_groups.is_empty() {
            break;
        }

        let mut improved = false;
        for route_group in route_groups.iter().rev() {
            let mut keep_actions = Vec::with_capacity(best.actions.len());
            for (action_index, action) in best.actions.iter().enumerate() {
                if !route_group.action_indices.contains(&action_index) {
                    keep_actions.push(action.clone());
                }
            }
            let Some(candidate_evaluated) = evaluate_forecastflows_action_set(
                initial_state,
                predictions,
                keep_actions,
                cost_config,
                evaluation_cache,
            ) else {
                continue;
            };
            let candidate = build_plan_result_from_forecastflows_evaluation(
                candidate_evaluated,
                frontier_family,
                preserve_markets.clone(),
                compiler_variant,
            );
            if plan_result_is_better(&candidate, &best) {
                best = candidate;
                improved = true;
                break;
            }
        }

        if !improved {
            break;
        }
    }

    best
}

#[cfg_attr(not(test), allow(dead_code))]
fn prune_forecastflows_plan_for_program_net_ev(
    initial_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    raw: &PlanResult,
    cost_config: PlannerCostConfig,
    evaluation_cache: &mut ForecastFlowsActionEvaluationCache,
    local_timing: &mut ForecastFlowsLocalTimingTotals,
) -> PlanResult {
    let step_prune_started = std::time::Instant::now();
    let step_pruned = baseline_step_prune_forecastflows_candidate_for_program_net_ev(
        initial_state,
        predictions,
        cost_config,
        raw.actions.clone(),
        raw.frontier_family,
        raw.preserve_markets.clone(),
        raw.compiler_variant,
        evaluation_cache,
    );
    local_timing.step_prune_ms += step_prune_started.elapsed().as_millis();
    let route_prune_started = std::time::Instant::now();
    let route_pruned = route_group_prune_forecastflows_candidate_for_program_net_ev(
        initial_state,
        predictions,
        cost_config,
        step_pruned.actions.clone(),
        step_pruned.frontier_family,
        step_pruned.preserve_markets.clone(),
        step_pruned.compiler_variant,
        evaluation_cache,
    );
    local_timing.route_prune_ms += route_prune_started.elapsed().as_millis();
    if plan_result_is_better(&route_pruned, &step_pruned) {
        route_pruned
    } else {
        step_pruned
    }
}

fn apply_forecastflows_run_stats(combined: &mut SolverRunStats, forecastflows: SolverRunStats) {
    combined.forecastflows_requests = forecastflows.forecastflows_requests;
    combined.forecastflows_worker_available = forecastflows.forecastflows_worker_available;
    combined.forecastflows_fallback_reason = forecastflows.forecastflows_fallback_reason;
    combined.forecastflows_telemetry = forecastflows.forecastflows_telemetry;
    combined.forecastflows_winning_variant = forecastflows.forecastflows_winning_variant;
}

fn apply_forecastflows_telemetry(stats: &mut SolverRunStats, telemetry: &ForecastFlowsTelemetry) {
    stats.forecastflows_telemetry = telemetry.clone();
}

fn run_forecastflows_family_plan(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    cost_config: PlannerCostConfig,
) -> (Option<PlanResult>, SolverRunStats) {
    let initial_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let mut stats = SolverRunStats {
        forecastflows_worker_available: false,
        ..SolverRunStats::default()
    };
    stats.forecastflows_telemetry.strategy = Some("rust_prune".to_string());
    let report = match forecastflows::solve_family_candidates(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_count,
        cost_config,
    ) {
        Ok(report) => report,
        Err(err) => {
            stats.forecastflows_requests = err.request_count();
            stats.forecastflows_worker_available = err.worker_available();
            if let Some(telemetry) = err.telemetry() {
                apply_forecastflows_telemetry(&mut stats, telemetry);
            }
            stats.forecastflows_fallback_reason = Some(err.fallback_reason());
            tracing::warn!(error = %err, "ForecastFlows candidate unavailable; falling back to native planner");
            return (None, stats);
        }
    };
    stats.forecastflows_requests = report.request_count;
    stats.forecastflows_worker_available = true;
    apply_forecastflows_telemetry(&mut stats, &report.telemetry);
    stats.forecastflows_telemetry.local_candidate_build_ms = Some(0);
    stats.forecastflows_telemetry.validation_only = false;

    if report.candidates.is_empty() {
        stats.forecastflows_fallback_reason = Some("no_certified_candidate");
        return (None, stats);
    }

    let mut best: Option<PlanResult> = None;
    let mut local_timing = ForecastFlowsLocalTimingTotals::default();
    for candidate in report.candidates {
        let estimated_execution_cost_susd = candidate.estimated_execution_cost_susd;
        let estimated_net_ev_susd = candidate.estimated_net_ev_susd;
        let mut evaluation_cache = ForecastFlowsActionEvaluationCache::default();
        let candidate_build_started = std::time::Instant::now();
        let Some(raw_plan) = build_forecastflows_raw_plan_result(
            &initial_state,
            predictions,
            candidate,
            cost_config,
            &mut evaluation_cache,
        ) else {
            continue;
        };
        local_timing.candidate_build_ms += candidate_build_started.elapsed().as_millis();
        let plan = prune_forecastflows_plan_for_program_net_ev(
            &initial_state,
            predictions,
            &raw_plan,
            cost_config,
            &mut evaluation_cache,
            &mut local_timing,
        );
        if best
            .as_ref()
            .is_none_or(|incumbent| plan_result_is_better(&plan, incumbent))
        {
            stats.forecastflows_telemetry.estimated_execution_cost_susd =
                estimated_execution_cost_susd;
            stats.forecastflows_telemetry.estimated_net_ev_susd = estimated_net_ev_susd;
            stats.forecastflows_telemetry.validated_total_fee_susd = plan.estimated_total_fee_susd;
            stats.forecastflows_telemetry.validated_net_ev_susd = plan.estimated_net_ev;
            stats.forecastflows_telemetry.fee_estimate_error_susd =
                match (plan.estimated_total_fee_susd, estimated_execution_cost_susd) {
                    (Some(validated_total_fee_susd), Some(estimated_execution_cost_susd)) => {
                        Some(validated_total_fee_susd - estimated_execution_cost_susd)
                    }
                    _ => None,
                };
            best = Some(plan);
        }
    }
    stats.forecastflows_telemetry.local_candidate_build_ms = Some(local_timing.candidate_build_ms);
    stats.forecastflows_telemetry.local_step_prune_ms = Some(local_timing.step_prune_ms);
    stats.forecastflows_telemetry.local_route_prune_ms = Some(local_timing.route_prune_ms);

    if best.is_none() {
        stats.forecastflows_fallback_reason = Some("no_replayable_candidate");
    }

    (best, stats)
}

fn rebalance_full_ultimate_with_predictions_and_stats(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    route_gates: RouteGateThresholds,
    cost_config: PlannerCostConfig,
    solver: RebalanceSolver,
    _flags: RebalanceFlags,
    force_mint_available: Option<bool>,
    verify_internal_state: bool,
) -> (PlanResult, SolverRunStats) {
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
        let initial_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
        return (
            solver_identity_plan(
                &initial_state,
                SolverFamily::Plain,
                state_snapshot_expected_value(&initial_state, predictions),
                cost_config,
            ),
            stats,
        );
    }

    let initial_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let solve_native = || {
        let mut native_stats = SolverRunStats::default();
        let mut best = run_plain_family_plan(
            &initial_state,
            predictions,
            expected_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            &mut native_stats,
            cost_config,
        );
        if let Some(arb_primed) = run_arb_primed_family_plan(
            &initial_state,
            predictions,
            expected_count,
            route_gates,
            force_mint_available,
            verify_internal_state,
            &mut native_stats,
            cost_config,
        ) && plan_result_is_better(&arb_primed, &best)
        {
            best = arb_primed;
        }
        (best, native_stats)
    };

    let (best, native_stats) = match solver {
        RebalanceSolver::Native => solve_native(),
        RebalanceSolver::ForecastFlows => {
            let (forecastflows, forecastflows_stats) = run_forecastflows_family_plan(
                balances,
                susds_balance,
                slot0_results,
                predictions,
                expected_count,
                cost_config,
            );
            if let Some(plan) = forecastflows {
                (plan, forecastflows_stats)
            } else {
                let native = solve_native();
                let mut combined = native.1;
                apply_forecastflows_run_stats(&mut combined, forecastflows_stats);
                (native.0, combined)
            }
        }
        RebalanceSolver::HeadToHead => {
            let (native, forecastflows) = thread::scope(|scope| {
                let forecastflows = scope.spawn(|| {
                    run_forecastflows_family_plan(
                        balances,
                        susds_balance,
                        slot0_results,
                        predictions,
                        expected_count,
                        cost_config,
                    )
                });
                let native = solve_native();
                let forecastflows = forecastflows
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                (native, forecastflows)
            });
            let mut combined = native.1;
            apply_forecastflows_run_stats(&mut combined, forecastflows.1);
            let chosen = choose_head_to_head_plan_result(native.0, forecastflows.0);
            (chosen, combined)
        }
    };
    stats = native_stats;

    stats.forecastflows_winning_variant = match best.compiler_variant {
        PlanCompilerVariant::ForecastFlowsDirect => Some(ForecastFlowsCandidateVariant::Direct),
        PlanCompilerVariant::ForecastFlowsMixed => Some(ForecastFlowsCandidateVariant::Mixed),
        _ => None,
    };

    if matches!(solver, RebalanceSolver::ForecastFlows)
        && best.family != SolverFamily::ForecastFlows
    {
        tracing::warn!(
            "ForecastFlows solver requested but no valid external candidate was available; using native plan"
        );
    }

    tracing::info!(
        requested_solver = solver.as_str(),
        chosen_family = best.family.as_str(),
        chosen_frontier_family = first_frontier_family_label(best.frontier_family),
        chosen_compiler_variant = best.compiler_variant.as_str(),
        chosen_common_shift = best.selected_common_shift,
        chosen_mixed_lambda = best.selected_mixed_lambda,
        chosen_active_set_size = best.selected_active_set_size,
        chosen_preserve_markets = best.preserve_markets.len(),
        final_raw_ev = best.raw_ev,
        final_estimated_total_fee_susd = best.estimated_total_fee_susd,
        final_estimated_net_ev = best.estimated_net_ev,
        final_group_count = best.estimated_group_count,
        final_tx_count = best.estimated_tx_count,
        final_fee_source = best.fee_estimate_source.label(),
        final_actions = best.actions.len(),
        exact_rebalance_calls = stats.exact_rebalance_calls,
        exact_rebalance_candidate_evals = stats.exact_rebalance_candidate_evals,
        arb_operator_evals = stats.arb_operator_evals,
        arb_primed_root_taken = stats.arb_primed_root_taken,
        fee_estimate_unavailable_results = stats.fee_estimate_unavailable_results,
        forecastflows_requests = stats.forecastflows_requests,
        forecastflows_worker_available = stats.forecastflows_worker_available,
        forecastflows_worker_roundtrip_ms = stats.forecastflows_telemetry.worker_roundtrip_ms,
        forecastflows_driver_overhead_ms = stats.forecastflows_telemetry.driver_overhead_ms,
        forecastflows_strategy = stats
            .forecastflows_telemetry
            .strategy
            .as_deref()
            .unwrap_or("none"),
        forecastflows_translation_replay_ms = stats.forecastflows_telemetry.translation_replay_ms,
        forecastflows_local_candidate_build_ms =
            stats.forecastflows_telemetry.local_candidate_build_ms,
        forecastflows_local_step_prune_ms = stats.forecastflows_telemetry.local_step_prune_ms,
        forecastflows_local_route_prune_ms = stats.forecastflows_telemetry.local_route_prune_ms,
        forecastflows_estimated_execution_cost_susd =
            stats.forecastflows_telemetry.estimated_execution_cost_susd,
        forecastflows_estimated_net_ev_susd = stats.forecastflows_telemetry.estimated_net_ev_susd,
        forecastflows_validated_total_fee_susd =
            stats.forecastflows_telemetry.validated_total_fee_susd,
        forecastflows_validated_net_ev_susd = stats.forecastflows_telemetry.validated_net_ev_susd,
        forecastflows_fee_estimate_error_susd =
            stats.forecastflows_telemetry.fee_estimate_error_susd,
        forecastflows_validation_only = stats.forecastflows_telemetry.validation_only,
        forecastflows_workspace_reused = stats.forecastflows_telemetry.workspace_reused,
        forecastflows_fallback_reason = stats.forecastflows_fallback_reason.unwrap_or("none"),
        forecastflows_direct_status = stats
            .forecastflows_telemetry
            .direct_status
            .as_deref()
            .unwrap_or("none"),
        forecastflows_mixed_status = stats
            .forecastflows_telemetry
            .mixed_status
            .as_deref()
            .unwrap_or("none"),
        forecastflows_direct_solver_time_ms = stats.forecastflows_telemetry.direct_solver_time_ms,
        forecastflows_mixed_solver_time_ms = stats.forecastflows_telemetry.mixed_solver_time_ms,
        forecastflows_certified_drop_reason = stats
            .forecastflows_telemetry
            .certified_drop_reason
            .as_deref()
            .unwrap_or("none"),
        forecastflows_replay_drop_reason = stats
            .forecastflows_telemetry
            .replay_drop_reason
            .as_deref()
            .unwrap_or("none"),
        forecastflows_replay_tolerance_clamp_used =
            stats.forecastflows_telemetry.replay_tolerance_clamp_used,
        forecastflows_sysimage_status = stats
            .forecastflows_telemetry
            .sysimage_status
            .as_deref()
            .unwrap_or("none"),
        forecastflows_julia_threads = stats
            .forecastflows_telemetry
            .julia_threads
            .as_deref()
            .unwrap_or("none"),
        forecastflows_solve_tuning = stats
            .forecastflows_telemetry
            .solve_tuning
            .as_deref()
            .unwrap_or("none"),
        forecastflows_winning_variant = stats
            .forecastflows_winning_variant
            .map(ForecastFlowsCandidateVariant::as_str)
            .unwrap_or("none"),
        "ultimate solver result"
    );

    (best, stats)
}

#[cfg(test)]
fn candidate_is_better(candidate: &CandidateResult, incumbent: &CandidateResult) -> bool {
    let economic_cmp = net_ev_cmp(
        candidate.estimated_net_ev,
        incumbent.estimated_net_ev,
        candidate.raw_ev,
        incumbent.raw_ev,
    );
    if !economic_cmp.is_eq() {
        return economic_cmp.is_lt();
    }
    if candidate.estimated_tx_count != incumbent.estimated_tx_count {
        return candidate.estimated_tx_count.unwrap_or(usize::MAX)
            < incumbent.estimated_tx_count.unwrap_or(usize::MAX);
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
    candidate.raw_ev.total_cmp(&incumbent.raw_ev).is_gt()
}

#[cfg(test)]
fn candidate_cmp(left: &CandidateResult, right: &CandidateResult) -> Ordering {
    if candidate_is_better(left, right) {
        Ordering::Less
    } else if candidate_is_better(right, left) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

#[cfg(test)]
fn candidate_uses_mint(candidate: &CandidateResult) -> bool {
    candidate
        .actions
        .iter()
        .any(|action| matches!(action, Action::Mint { .. }))
}

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
    let starting_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    Some(build_candidate_result(
        &starting_state,
        actions,
        ev,
        variant,
        preserve_markets.clone(),
        forced_first_frontier_family,
        cost_config,
    ))
}

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
            cost_config,
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
            cost_config,
        ) else {
            continue;
        };
        evaluation.greedy_best = greedy_best;
        evaluation.preserve_candidates = preserve_candidates;
    }

    for evaluation in &evaluations {
        tracing::debug!(
            phase_order_variant = evaluation.baseline.variant.as_str(),
            baseline_raw_ev = evaluation.baseline.raw_ev,
            baseline_estimated_net_ev = evaluation.baseline.estimated_net_ev,
            baseline_actions = evaluation.baseline.actions.len(),
            greedy_prescan_raw_ev = evaluation.greedy_best.raw_ev,
            greedy_prescan_estimated_net_ev = evaluation.greedy_best.estimated_net_ev,
            greedy_prescan_actions = evaluation.greedy_best.actions.len(),
            preserve_candidate_count = evaluation.preserve_candidates.len(),
            preserve_market_count = evaluation.greedy_best.preserve_markets.len(),
            selected_source = evaluation.best_source_label(),
            "rebalance meta-solver stage1 evaluation"
        );
    }

    evaluations
}

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
            cost_config,
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

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
            cost_config,
        ) else {
            continue;
        };
        if candidate_is_better(&candidate, &best) {
            best = candidate;
        }
    }

    best
}

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
                cost_config,
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
                    cost_config,
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

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
        cost_config,
    )
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
            cost_config,
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

    let starting_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    Some(build_candidate_result(
        &starting_state,
        committed_actions,
        committed_ev,
        PhaseOrderVariant::CyclicLateArb,
        seed.preserve_markets.clone(),
        seed.forced_first_frontier_family,
        cost_config,
    ))
}

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
            cost_config,
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
                cost_config,
            ) else {
                continue;
            };
            tracing::debug!(
                chosen_variant = candidate.variant.as_str(),
                forced_first_frontier_family =
                    first_frontier_family_label(candidate.forced_first_frontier_family),
                candidate_raw_ev = candidate.raw_ev,
                candidate_estimated_net_ev = candidate.estimated_net_ev,
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

#[cfg(test)]
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

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
        cost_config,
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
                cost_config,
            )
        });
        tracing::debug!(
            phase_order_variant = refined.candidate.variant.as_str(),
            refinement_source = refined.source_label,
            refined_raw_ev = refined.candidate.raw_ev,
            refined_estimated_net_ev = refined.candidate.estimated_net_ev,
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
                cost_config,
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
            cost_config,
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
                    cost_config,
                )
            },
        );
        if let Some(candidate) = cyclic_candidate {
            tracing::debug!(
                phase_order_variant = seed.variant.as_str(),
                seed_raw_ev = seed.raw_ev,
                seed_estimated_net_ev = seed.estimated_net_ev,
                seed_actions = seed.actions.len(),
                seed_preserve_markets = seed.preserve_markets.len(),
                seed_first_frontier_family =
                    first_frontier_family_label(seed.forced_first_frontier_family),
                candidate_raw_ev = candidate.raw_ev,
                candidate_estimated_net_ev = candidate.estimated_net_ev,
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
        stage1_raw_ev = stage1_best.raw_ev,
        stage1_estimated_net_ev = stage1_best.estimated_net_ev,
        stage1_actions = stage1_best.actions.len(),
        stage1_preserve_markets = stage1_best.preserve_markets.len(),
        preserve_candidate_count = stage1_selection.preserve_candidates.len(),
        final_source,
        final_raw_ev = best.raw_ev,
        final_estimated_net_ev = best.estimated_net_ev,
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

#[cfg(test)]
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
    cost_config: PlannerCostConfig,
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
        cost_config,
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

#[cfg(test)]
fn rebalance_full_with_predictions_staged_reference(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    route_gates: RouteGateThresholds,
    cost_config: PlannerCostConfig,
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
        cost_config,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn rebalance_full_with_predictions(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    expected_count: usize,
    solver: RebalanceSolver,
    route_gates: RouteGateThresholds,
    cost_config: PlannerCostConfig,
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
        cost_config,
        solver,
        flags,
        force_mint_available,
        verify_internal_state,
    )
    .0
    .actions
}

#[allow(dead_code)]
fn rebalance_full(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    solver: RebalanceSolver,
    route_gates: RouteGateThresholds,
    cost_config: PlannerCostConfig,
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
        solver,
        route_gates,
        cost_config,
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
    rebalance_with_custom_predictions_and_solver_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        force_mint_available,
        RebalanceSolver::Native,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_and_solver_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    solver: RebalanceSolver,
) -> Vec<Action> {
    rebalance_full_with_predictions(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        solver,
        RouteGateThresholds::disabled(),
        benchmark_planner_cost_config_for_test(),
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
    rebalance_with_custom_predictions_and_solver_and_stats_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        force_mint_available,
        RebalanceSolver::Native,
    )
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_and_solver_and_stats_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    solver: RebalanceSolver,
) -> (Vec<Action>, SolverRunStats) {
    let (plan, stats) = rebalance_full_ultimate_with_predictions_and_stats(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        benchmark_planner_cost_config_for_test(),
        solver,
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
    );
    (plan.actions, stats)
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
        benchmark_planner_cost_config_for_test(),
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
    let cost_config = benchmark_planner_cost_config_for_test();
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
        cost_config,
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
fn action_is_complete_set_arb_marker(action: &Action) -> bool {
    match action {
        Action::Merge { source_market, .. } => *source_market == "complete_set_arb",
        Action::Mint { target_market, .. } => *target_market == "complete_set_arb",
        _ => false,
    }
}

#[cfg(test)]
fn analyze_complete_set_arb_usage(actions: &[Action]) -> (bool, bool) {
    let mut root_arb_fired = false;
    let mut late_arb_improved = false;
    let mut seen_non_arb = false;

    for action in actions {
        if action_is_complete_set_arb_marker(action) {
            if seen_non_arb {
                late_arb_improved = true;
            } else {
                root_arb_fired = true;
            }
        } else {
            seen_non_arb = true;
        }
    }

    (root_arb_fired, late_arb_improved)
}

#[cfg(test)]
pub(super) fn staged_teacher_snapshot_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Option<TeacherDecisionSnapshot> {
    let cost_config = benchmark_planner_cost_config_for_test();
    let candidate = rebalance_full_with_predictions_and_budget_candidate(
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
        cost_config,
    )?;
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let preserve_scores = collect_mint_sell_preserve_candidate_scores(
        &candidate.actions,
        slot0_results,
        balances,
        susds_balance,
        slot0_results.len(),
    );
    let features = collect_distilled_market_features(&state, predictions, &preserve_scores);
    let (root_arb_fired, late_arb_improved) = analyze_complete_set_arb_usage(&candidate.actions);
    Some(TeacherDecisionSnapshot {
        winning_family: candidate.variant.as_str(),
        winning_frontier_family: first_frontier_family_label(
            candidate.forced_first_frontier_family,
        ),
        preserve_markets: sorted_preserve_markets(
            &candidate
                .preserve_markets
                .iter()
                .copied()
                .collect::<Vec<&'static str>>(),
        ),
        root_arb_fired,
        late_arb_improved,
        raw_ev: candidate.raw_ev,
        action_count: candidate.actions.len(),
        features: distilled_feature_rows(&features),
    })
}

#[cfg(test)]
pub(super) fn estimate_plan_economics_for_test(
    actions: &[Action],
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
) -> TestPlanEconomics {
    let raw_ev =
        action_plan_expected_value(actions, slot0_results, balances, susds_balance, predictions);
    let cost_config = benchmark_planner_cost_config_for_test();
    let (
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source,
        _fee_estimate_unavailable,
    ) = plan_cost_fields(raw_ev, actions, slot0_results, cost_config);
    TestPlanEconomics {
        raw_ev,
        estimated_total_fee_susd,
        estimated_net_ev,
        estimated_group_count,
        estimated_tx_count,
        fee_estimate_source: fee_estimate_source.label(),
    }
}

#[cfg(test)]
fn summarize_action_route_counts(actions: &[Action]) -> (usize, usize, usize, usize) {
    let mut direct_buy_count = 0;
    let mut direct_sell_count = 0;
    let mut mint_count = 0;
    let mut merge_count = 0;
    for action in actions {
        match action {
            Action::Buy { .. } => direct_buy_count += 1,
            Action::Sell { .. } => direct_sell_count += 1,
            Action::Mint { .. } => mint_count += 1,
            Action::Merge { .. } => merge_count += 1,
        }
    }
    (direct_buy_count, direct_sell_count, mint_count, merge_count)
}

#[cfg(test)]
fn total_calldata_bytes_for_actions_for_test(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    cost_config: PlannerCostConfig,
) -> Option<usize> {
    if actions.is_empty() {
        return Some(0);
    }
    let replay = build_group_plans_for_gas_replay_with_market_context(
        actions,
        slot0_results,
        cost_config.conservative_execution,
        &cost_config.gas_assumptions,
        cost_config.pricing.gas_price_eth,
        cost_config.pricing.eth_usd,
    )
    .ok()?;
    if replay.plans.is_empty() || !replay.skipped_groups.is_empty() {
        return estimate_structural_packed_total_calldata_bytes(actions, cost_config);
    }
    let fee_inputs = planner_synthetic_fee_inputs(cost_config.pricing);
    let strict_program = compile_execution_program_unchecked(
        ExecutionMode::Strict,
        Address::ZERO,
        actions,
        &replay.plans,
        fee_inputs,
        &cost_config.gas_assumptions,
        cost_config.pricing.eth_usd,
    )
    .ok()?;
    let packed_program = compile_execution_program_unchecked(
        ExecutionMode::Packed,
        Address::ZERO,
        actions,
        &replay.plans,
        fee_inputs,
        &cost_config.gas_assumptions,
        cost_config.pricing.eth_usd,
    )
    .ok()?;
    let program = if better_execution_program(&packed_program, &strict_program) {
        packed_program
    } else {
        strict_program
    };
    Some(
        program
            .chunks
            .iter()
            .map(|chunk| chunk.unsigned_tx_bytes_len)
            .sum(),
    )
}

#[cfg(test)]
fn selected_plan_summary_from_plan_for_test(
    plan: &PlanResult,
    cost_config: PlannerCostConfig,
) -> TestSelectedPlanSummary {
    let (direct_buy_count, direct_sell_count, mint_count, merge_count) =
        summarize_action_route_counts(&plan.actions);
    TestSelectedPlanSummary {
        family: plan.family.as_str(),
        frontier_family: first_frontier_family_label(plan.frontier_family),
        compiler_variant: plan.compiler_variant.as_str(),
        selected_common_shift: plan.selected_common_shift,
        selected_mixed_lambda: plan.selected_mixed_lambda,
        selected_active_set_size: plan.selected_active_set_size,
        selected_stage_count: plan.selected_stage_count,
        selected_stage1_budget_fraction: plan.selected_stage1_budget_fraction,
        raw_ev: plan.raw_ev,
        estimated_total_fee_susd: plan.estimated_total_fee_susd,
        estimated_net_ev: plan.estimated_net_ev,
        estimated_group_count: plan.estimated_group_count,
        estimated_tx_count: plan.estimated_tx_count,
        fee_estimate_source: plan.fee_estimate_source.label(),
        action_count: plan.actions.len(),
        direct_buy_count,
        direct_sell_count,
        mint_count,
        merge_count,
        total_calldata_bytes: total_calldata_bytes_for_actions_for_test(
            &plan.actions,
            &plan.terminal_state.slot0_results,
            cost_config,
        ),
        mixed_certificates: plan.mixed_certificates.clone(),
    }
}

#[cfg(test)]
pub(super) fn forecastflows_candidate_plan_variants_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    candidate: ForecastFlowsFamilyCandidate,
) -> Option<TestForecastFlowsPlanVariants> {
    let initial_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let cost_config = benchmark_planner_cost_config_for_test();
    let mut evaluation_cache = ForecastFlowsActionEvaluationCache::default();
    let mut local_timing = ForecastFlowsLocalTimingTotals::default();
    let raw = build_forecastflows_raw_plan_result(
        &initial_state,
        predictions,
        candidate,
        cost_config,
        &mut evaluation_cache,
    )?;
    let polished = prune_forecastflows_plan_for_program_net_ev(
        &initial_state,
        predictions,
        &raw,
        cost_config,
        &mut evaluation_cache,
        &mut local_timing,
    );
    Some(TestForecastFlowsPlanVariants {
        raw_replayable: estimate_plan_cost_from_replay(&raw.actions, slot0_results, cost_config)
            .is_some(),
        polished_replayable: estimate_plan_cost_from_replay(
            &polished.actions,
            slot0_results,
            cost_config,
        )
        .is_some(),
        raw: selected_plan_summary_from_plan_for_test(&raw, cost_config),
        polished: selected_plan_summary_from_plan_for_test(&polished, cost_config),
    })
}

#[cfg(test)]
pub(super) fn selected_plan_run_with_solver_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    solver: RebalanceSolver,
) -> (Vec<Action>, TestSelectedPlanSummary, SolverRunStats) {
    let cost_config = benchmark_planner_cost_config_for_test();
    let (plan, stats) = rebalance_full_ultimate_with_predictions_and_stats(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        cost_config,
        solver,
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
    );
    let actions = plan.actions.clone();
    let summary = selected_plan_summary_from_plan_for_test(&plan, cost_config);
    (actions, summary, stats)
}

#[cfg(test)]
pub(super) fn selected_plan_run_with_solver_and_pricing_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    solver: RebalanceSolver,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
) -> (Vec<Action>, TestSelectedPlanSummary, SolverRunStats) {
    let cost_config = benchmark_planner_cost_config_with_pricing_for_test(
        gas_assumptions,
        gas_price_eth,
        eth_usd,
    );
    let (plan, stats) = rebalance_full_ultimate_with_predictions_and_stats(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        cost_config,
        solver,
        RebalanceFlags::default(),
        Some(force_mint_available),
        false,
    );
    let actions = plan.actions.clone();
    let summary = selected_plan_summary_from_plan_for_test(&plan, cost_config);
    (actions, summary, stats)
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_selected_plan_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> TestSelectedPlanSummary {
    selected_plan_run_with_solver_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        force_mint_available,
        RebalanceSolver::Native,
    )
    .1
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_selected_plan_with_pricing_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
) -> TestSelectedPlanSummary {
    selected_plan_run_with_solver_and_pricing_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        force_mint_available,
        RebalanceSolver::Native,
        gas_assumptions,
        gas_price_eth,
        eth_usd,
    )
    .1
}

#[cfg(test)]
pub(super) fn compile_constant_l_mixed_selected_plan_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    budget_fraction: Option<f64>,
) -> Option<TestSelectedPlanSummary> {
    let starting_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let cost_config = benchmark_planner_cost_config_for_test();
    let budget_cap = starting_state.cash.max(0.0) * budget_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap(
        &starting_state,
        predictions,
        None,
        Vec::new(),
        SolverFamily::Plain,
        cost_config,
        budget_cap,
    )
    .map(|plan| selected_plan_summary_from_plan_for_test(&plan, cost_config))
}

#[cfg(test)]
fn active_mask_is_direct_profitability_prefix_for_test(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    active_mask: &[bool],
) -> Option<bool> {
    let profitable = sorted_profitable_direct_prefixes(starting_state, predictions)?;
    for prefix_len in 1..=profitable.len() {
        let mut prefix_mask = vec![false; starting_state.slot0_results.len()];
        for (idx, _) in profitable.iter().take(prefix_len) {
            if let Some(slot) = prefix_mask.get_mut(*idx) {
                *slot = true;
            }
        }
        if prefix_mask == active_mask {
            return Some(true);
        }
    }
    Some(false)
}

#[cfg(test)]
fn enumerate_constant_l_mixed_oracle_candidates_for_test(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    budget_cap: f64,
    cost_config: PlannerCostConfig,
    max_markets: usize,
) -> Option<Vec<PlanResult>> {
    let market_count = starting_state.slot0_results.len();
    if market_count == 0 || market_count > max_markets {
        return None;
    }

    let mut cache = ConstantLMixedSolveCache::default();
    let mut candidates = Vec::new();
    for subset_bits in 1u64..(1u64 << market_count) {
        let active_mask: Vec<bool> = (0..market_count)
            .map(|idx| ((subset_bits >> idx) & 1) == 1)
            .collect();
        let Some(candidate) =
            compile_constant_l_mixed_candidate_for_active_mask_with_budget_cap_cached(
                &mut cache,
                starting_state,
                predictions,
                None,
                Vec::new(),
                SolverFamily::Plain,
                cost_config,
                budget_cap,
                &active_mask,
            )
        else {
            continue;
        };
        candidates.push(candidate);
    }

    (!candidates.is_empty()).then_some(candidates)
}

#[cfg(test)]
fn compile_constant_l_mixed_oracle_plan_for_test_with_limit(
    starting_state: &SolverStateSnapshot,
    predictions: &HashMap<String, f64>,
    budget_cap: f64,
    cost_config: PlannerCostConfig,
    max_markets: usize,
) -> Option<PlanResult> {
    enumerate_constant_l_mixed_oracle_candidates_for_test(
        starting_state,
        predictions,
        budget_cap,
        cost_config,
        max_markets,
    )?
    .into_iter()
    .min_by(plan_result_cmp)
}

#[cfg(test)]
fn test_selected_plan_gap_net_ev(
    runtime_best: &PlanResult,
    teacher_best: &PlanResult,
) -> Option<f64> {
    let runtime_net = runtime_best.estimated_net_ev.unwrap_or(runtime_best.raw_ev);
    let teacher_net = teacher_best.estimated_net_ev.unwrap_or(teacher_best.raw_ev);
    Some((teacher_net - runtime_net).max(0.0))
}

#[cfg(test)]
fn test_selected_plan_gap_raw_ev(
    runtime_best: &PlanResult,
    teacher_best: &PlanResult,
) -> Option<f64> {
    Some((teacher_best.raw_ev - runtime_best.raw_ev).max(0.0))
}

#[cfg(test)]
pub(super) fn compare_constant_l_runtime_vs_oracle_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    budget_fraction: Option<f64>,
) -> Option<TestK1OracleComparison> {
    compare_constant_l_runtime_vs_oracle_with_limit_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        budget_fraction,
        K1_SMALL_ORACLE_MAX_MARKETS,
    )
}

#[cfg(test)]
pub(super) fn compare_constant_l_runtime_vs_medium_oracle_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    budget_fraction: Option<f64>,
) -> Option<TestK1OracleComparison> {
    compare_constant_l_runtime_vs_oracle_with_limit_for_test(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        budget_fraction,
        K1_MEDIUM_ORACLE_MAX_MARKETS,
    )
}

#[cfg(test)]
fn compare_constant_l_runtime_vs_oracle_with_limit_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    budget_fraction: Option<f64>,
    max_markets: usize,
) -> Option<TestK1OracleComparison> {
    let starting_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let cost_config = benchmark_planner_cost_config_for_test();
    let budget_cap = starting_state.cash.max(0.0) * budget_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    let runtime_best = compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap(
        &starting_state,
        predictions,
        None,
        Vec::new(),
        SolverFamily::Plain,
        cost_config,
        budget_cap,
    )?;
    let oracle_best = compile_constant_l_mixed_oracle_plan_for_test_with_limit(
        &starting_state,
        predictions,
        budget_cap,
        cost_config,
        max_markets,
    )?;
    let oracle_best_is_direct_prefix = oracle_best
        .mixed_certificates
        .first()
        .and_then(|certificate| {
            active_mask_is_direct_profitability_prefix_for_test(
                &starting_state,
                predictions,
                &certificate.active_mask,
            )
        })
        .unwrap_or(false);
    let runtime_summary = selected_plan_summary_from_plan_for_test(&runtime_best, cost_config);
    let oracle_summary = selected_plan_summary_from_plan_for_test(&oracle_best, cost_config);
    Some(TestK1OracleComparison {
        runtime_best: runtime_summary,
        oracle_best: oracle_summary,
        oracle_best_is_direct_prefix,
        oracle_best_active_set_size: oracle_best.selected_active_set_size,
        runtime_k1_gap_net_ev: test_selected_plan_gap_net_ev(&runtime_best, &oracle_best),
        runtime_k1_gap_raw_ev: test_selected_plan_gap_raw_ev(&runtime_best, &oracle_best),
    })
}

#[cfg(test)]
pub(super) fn compare_constant_l_runtime_vs_best_known_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    budget_fraction: Option<f64>,
) -> Option<TestK1BestKnownComparison> {
    let starting_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let cost_config = benchmark_planner_cost_config_for_test();
    let budget_cap = starting_state.cash.max(0.0) * budget_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    let runtime_best = compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap(
        &starting_state,
        predictions,
        None,
        Vec::new(),
        SolverFamily::Plain,
        cost_config,
        budget_cap,
    )?;
    let best_known_best =
        compile_constant_l_mixed_best_known_candidate_for_program_net_ev_with_budget_cap(
            &starting_state,
            predictions,
            budget_cap,
            cost_config,
        )?;
    Some(TestK1BestKnownComparison {
        runtime_best: selected_plan_summary_from_plan_for_test(&runtime_best, cost_config),
        best_known_best: selected_plan_summary_from_plan_for_test(&best_known_best, cost_config),
        runtime_k1_gap_net_ev: test_selected_plan_gap_net_ev(&runtime_best, &best_known_best),
        runtime_k1_gap_raw_ev: test_selected_plan_gap_raw_ev(&runtime_best, &best_known_best),
    })
}

#[cfg(test)]
pub(super) fn compare_constant_l_runtime_vs_k2_oracle_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    budget_fraction: Option<f64>,
) -> Option<TestK2OracleComparison> {
    let starting_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    if starting_state.slot0_results.is_empty()
        || starting_state.slot0_results.len() > K2_ORACLE_MAX_MARKETS
    {
        return None;
    }
    let cost_config = benchmark_planner_cost_config_for_test();
    let budget_cap = starting_state.cash.max(0.0) * budget_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    let runtime_best = compile_constant_l_mixed_candidate_for_program_net_ev_with_budget_cap(
        &starting_state,
        predictions,
        None,
        Vec::new(),
        SolverFamily::Plain,
        cost_config,
        budget_cap,
    )?;
    let k1_oracle_best = compile_constant_l_mixed_oracle_plan_for_test_with_limit(
        &starting_state,
        predictions,
        budget_cap,
        cost_config,
        K2_ORACLE_MAX_MARKETS,
    )?;
    let stage_1_candidates = enumerate_constant_l_mixed_oracle_candidates_for_test(
        &starting_state,
        predictions,
        budget_cap,
        cost_config,
        K2_ORACLE_MAX_MARKETS,
    )?;
    let mut k2_oracle_best = k1_oracle_best.clone();
    for stage_1 in stage_1_candidates {
        let stage_1_budget_fraction = if starting_state.cash.abs() <= DUST {
            0.0
        } else {
            stage_1
                .mixed_certificates
                .first()
                .map(|certificate| {
                    (certificate.budget_used / starting_state.cash.max(0.0)).clamp(0.0, 1.0)
                })
                .unwrap_or(0.0)
        };
        let stage_2_state = stage_1.terminal_state.clone();
        let stage_2_raw_ev = state_snapshot_expected_value(&stage_2_state, predictions);
        let stage_2 = compile_constant_l_mixed_oracle_plan_for_test_with_limit(
            &stage_2_state,
            predictions,
            stage_2_state.cash.max(0.0),
            cost_config,
            K2_ORACLE_MAX_MARKETS,
        )
        .unwrap_or_else(|| {
            solver_identity_plan(
                &stage_2_state,
                SolverFamily::Plain,
                stage_2_raw_ev,
                cost_config,
            )
        });
        let Some(candidate) = compose_staged_constant_l_plan(
            &starting_state,
            &stage_1,
            &stage_2,
            None,
            Vec::new(),
            SolverFamily::Plain,
            cost_config,
            stage_1_budget_fraction,
        ) else {
            continue;
        };
        if plan_result_is_better(&candidate, &k2_oracle_best) {
            k2_oracle_best = candidate;
        }
    }
    Some(TestK2OracleComparison {
        runtime_best: selected_plan_summary_from_plan_for_test(&runtime_best, cost_config),
        k1_oracle_best: selected_plan_summary_from_plan_for_test(&k1_oracle_best, cost_config),
        k2_oracle_best: selected_plan_summary_from_plan_for_test(&k2_oracle_best, cost_config),
        k2_gain_net_ev: test_selected_plan_gap_net_ev(&k1_oracle_best, &k2_oracle_best),
        k2_gain_raw_ev: test_selected_plan_gap_raw_ev(&k1_oracle_best, &k2_oracle_best),
    })
}

#[cfg(test)]
pub(super) fn compile_target_delta_actions_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    rich_actions: &[Action],
) -> Option<Vec<Action>> {
    let starting_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let rich_terminal_state =
        apply_actions_to_solver_state(&starting_state, rich_actions, predictions)?;
    let deltas = sorted_market_deltas(&starting_state, &rich_terminal_state);
    let common_shifts = deduped_target_delta_common_shifts(&deltas);
    let mut best_actions = None;
    let mut best: Option<PlanResult> = None;
    let cost_config = benchmark_planner_cost_config_for_test();
    for common_shift in common_shifts {
        let Some(actions) = compile_target_delta_actions_for_common_shift(
            &starting_state,
            predictions,
            &rich_terminal_state,
            common_shift,
        ) else {
            continue;
        };
        let Some(candidate_terminal_state) =
            apply_actions_to_solver_state(&starting_state, &actions, predictions)
        else {
            continue;
        };
        if !holdings_match_within_tol(&candidate_terminal_state, &rich_terminal_state) {
            continue;
        }
        let raw_ev = state_snapshot_expected_value(&candidate_terminal_state, predictions);
        let candidate = build_plan_result(
            &starting_state,
            candidate_terminal_state,
            actions.clone(),
            raw_ev,
            None,
            Vec::new(),
            SolverFamily::Plain,
            cost_config,
            PlanCompilerVariant::TargetDelta,
            Some(common_shift),
            None,
            None,
        );
        if best
            .as_ref()
            .is_none_or(|incumbent| plan_result_is_better(&candidate, incumbent))
        {
            best = Some(candidate);
            best_actions = Some(actions);
        }
    }
    best_actions
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
    let cost_config = benchmark_planner_cost_config_for_test();
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
        cost_config,
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
        cost_config,
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
            cost_config,
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
    let cost_config = benchmark_planner_cost_config_for_test();
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
        cost_config,
    )
    .actions
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_exact_no_arb_with_distilled_proposals_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> Vec<Action> {
    let state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let cost_config = benchmark_planner_cost_config_for_test();
    let mut stats = SolverRunStats::default();
    run_exact_no_arb_plan_with_options(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        false,
        SolverFamily::Plain,
        &[],
        MAX_ONLINE_PRESERVE_CANDIDATES,
        true,
        &mut stats,
        cost_config,
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
    let cost_config = benchmark_planner_cost_config_for_test();
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
        cost_config,
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
    let cost_config = benchmark_planner_cost_config_for_test();
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
            cost_config,
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
        false,
        &mut stats,
        cost_config,
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
    let cost_config = benchmark_planner_cost_config_for_test();
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
        cost_config,
    );
    let mut best = solver_identity_plan(
        &state,
        SolverFamily::Plain,
        state_snapshot_expected_value(&state, predictions),
        cost_config,
    );
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
        cost_config,
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
    let cost_config = benchmark_planner_cost_config_for_test();
    let mut stats = SolverRunStats::default();
    run_plain_family_plan(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        &mut stats,
        cost_config,
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
    let cost_config = benchmark_planner_cost_config_for_test();
    let mut stats = SolverRunStats::default();
    run_arb_primed_family_plan(
        &state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        &mut stats,
        cost_config,
    )
    .map(|plan| plan.actions)
}

#[cfg(test)]
pub(super) fn rebalance_with_custom_predictions_operator_only_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    predictions: &HashMap<String, f64>,
    force_mint_available: bool,
) -> (Vec<Action>, SolverRunStats) {
    let initial_state = build_solver_state_snapshot(balances, susds_balance, slot0_results);
    let cost_config = benchmark_planner_cost_config_for_test();
    let mut stats = SolverRunStats::default();
    let mut best = run_plain_family_plan(
        &initial_state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        &mut stats,
        cost_config,
    );
    if let Some(arb_primed) = run_arb_primed_family_plan(
        &initial_state,
        predictions,
        slot0_results.len(),
        RouteGateThresholds::disabled(),
        Some(force_mint_available),
        false,
        &mut stats,
        cost_config,
    ) && plan_result_is_better(&arb_primed, &best)
    {
        best = arb_primed;
    }
    (best.actions, stats)
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
    rebalance_with_solver_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        RebalanceSolver::Native,
        RouteGateThresholds::disabled(),
        default_planner_cost_config(),
        flags,
    )
}

pub fn rebalance_with_solver_and_flags(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    solver: RebalanceSolver,
    flags: RebalanceFlags,
) -> Vec<Action> {
    rebalance_with_solver_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        solver,
        RouteGateThresholds::disabled(),
        default_planner_cost_config(),
        flags,
    )
}

fn rebalance_with_solver_and_thresholds(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    solver: RebalanceSolver,
    route_gates: RouteGateThresholds,
    cost_config: PlannerCostConfig,
    flags: RebalanceFlags,
) -> Vec<Action> {
    rebalance_decision_with_solver_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        solver,
        route_gates,
        cost_config,
        flags,
    )
    .actions
}

fn rebalance_decision_with_solver_and_thresholds(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    solver: RebalanceSolver,
    route_gates: RouteGateThresholds,
    cost_config: PlannerCostConfig,
    flags: RebalanceFlags,
) -> RebalancePlanDecision {
    match mode {
        RebalanceMode::Full => {
            let predictions = crate::pools::prediction_map();
            let expected_count = crate::predictions::PREDICTIONS_L1.len();
            let (plan, stats) = rebalance_full_ultimate_with_predictions_and_stats(
                balances,
                susds_balance,
                slot0_results,
                predictions,
                expected_count,
                route_gates,
                cost_config,
                solver,
                flags,
                None,
                false,
            );
            decision_from_plan_result(plan, &stats)
        }
        RebalanceMode::ArbOnly => {
            let actions = rebalance_arb_only(balances, susds_balance, slot0_results);
            RebalancePlanDecision {
                summary: RebalancePlanSummary {
                    raw_ev: 0.0,
                    estimated_total_fee_susd: None,
                    estimated_net_ev: None,
                    estimated_group_count: None,
                    estimated_tx_count: None,
                    action_count: actions.len(),
                    forecastflows_telemetry: ForecastFlowsRunTelemetry::default(),
                    family_stable_rank: 0,
                    frontier_family_stable_rank: 0,
                    preserve_markets: Vec::new(),
                    compiler_variant_stable_rank: 0,
                    selected_common_shift: None,
                    selected_mixed_lambda: None,
                    selected_active_set_size: None,
                },
                actions,
            }
        }
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
    let direct_buy = estimate_min_incremental_gas_susd_for_group(
        gas,
        GroupKind::DirectBuy,
        0,
        0,
        gas_price_eth,
        eth_usd,
    );
    let mint_sell = estimate_min_incremental_gas_susd_for_group(
        gas,
        GroupKind::MintSell,
        0,
        cross_route_legs,
        gas_price_eth,
        eth_usd,
    );
    let direct_sell = estimate_min_incremental_gas_susd_for_group(
        gas,
        GroupKind::DirectSell,
        0,
        0,
        gas_price_eth,
        eth_usd,
    );
    let buy_merge = estimate_min_incremental_gas_susd_for_group(
        gas,
        GroupKind::BuyMerge,
        cross_route_legs,
        0,
        gas_price_eth,
        eth_usd,
    );
    let direct_merge = estimate_min_incremental_gas_susd_for_group(
        gas,
        GroupKind::DirectMerge,
        0,
        0,
        gas_price_eth,
        eth_usd,
    );
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
    rebalance_with_solver_and_gas_pricing_and_flags(
        balances,
        susds_balance,
        slot0_results,
        mode,
        RebalanceSolver::Native,
        gas,
        gas_price_eth,
        eth_usd,
        flags,
    )
}

pub fn rebalance_with_solver_and_gas_pricing_and_flags(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    solver: RebalanceSolver,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
    flags: RebalanceFlags,
) -> Vec<Action> {
    let n_sims = slot0_results.len();
    let route_gates = compute_gas_thresholds(gas, gas_price_eth, eth_usd, n_sims.saturating_sub(1));
    let cost_config =
        planner_cost_config_with_pricing(gas, gas_price_eth, eth_usd, "explicit_gas_pricing");
    rebalance_decision_with_solver_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        solver,
        route_gates,
        cost_config,
        flags,
    )
    .actions
}

pub fn rebalance_with_solver_and_gas_pricing_and_flags_and_decision(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    mode: RebalanceMode,
    solver: RebalanceSolver,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
    flags: RebalanceFlags,
) -> RebalancePlanDecision {
    let n_sims = slot0_results.len();
    let route_gates = compute_gas_thresholds(gas, gas_price_eth, eth_usd, n_sims.saturating_sub(1));
    let cost_config =
        planner_cost_config_with_pricing(gas, gas_price_eth, eth_usd, "explicit_gas_pricing");
    rebalance_decision_with_solver_and_thresholds(
        balances,
        susds_balance,
        slot0_results,
        mode,
        solver,
        route_gates,
        cost_config,
        flags,
    )
}

pub fn warm_forecastflows_worker() -> Result<(), String> {
    forecastflows::warm_worker().map_err(|err| err.to_string())
}

pub fn shutdown_forecastflows_worker() -> Result<(), String> {
    forecastflows::shutdown_worker().map_err(|err| err.to_string())
}

pub fn forecastflows_doctor_report() -> Result<forecastflows::ForecastFlowsDoctorReport, String> {
    forecastflows::doctor_report().map_err(|err| err.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_state() -> SolverStateSnapshot {
        SolverStateSnapshot {
            slot0_results: Vec::new(),
            holdings: BalanceMap::new(),
            cash: 0.0,
        }
    }

    fn dummy_plan_result(
        family: SolverFamily,
        compiler_variant: PlanCompilerVariant,
        raw_ev: f64,
        estimated_net_ev: Option<f64>,
        actions: Vec<Action>,
    ) -> PlanResult {
        PlanResult {
            actions,
            terminal_state: dummy_state(),
            raw_ev,
            estimated_total_fee_susd: estimated_net_ev.map(|net| (raw_ev - net).max(0.0)),
            estimated_net_ev,
            estimated_group_count: Some(1),
            estimated_tx_count: Some(1),
            fee_estimate_source: FeeEstimateSource::StructuralFallback,
            frontier_family: None,
            preserve_markets: Vec::new(),
            family,
            compiler_variant,
            selected_common_shift: None,
            selected_mixed_lambda: None,
            selected_active_set_size: None,
            selected_stage_count: None,
            selected_stage1_budget_fraction: None,
            mixed_certificates: Vec::new(),
        }
    }

    #[test]
    fn plan_result_prefers_higher_net_ev_over_higher_raw_ev() {
        let state = dummy_state();
        let higher_raw_lower_net = PlanResult {
            actions: vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "a",
            }],
            terminal_state: state.clone(),
            raw_ev: 10.0,
            estimated_total_fee_susd: Some(3.0),
            estimated_net_ev: Some(7.0),
            estimated_group_count: Some(1),
            estimated_tx_count: Some(1),
            fee_estimate_source: FeeEstimateSource::StructuralFallback,
            frontier_family: None,
            preserve_markets: Vec::new(),
            family: SolverFamily::Plain,
            compiler_variant: PlanCompilerVariant::BaselineStepPrune,
            selected_common_shift: None,
            selected_mixed_lambda: None,
            selected_active_set_size: None,
            selected_stage_count: None,
            selected_stage1_budget_fraction: None,
            mixed_certificates: Vec::new(),
        };
        let lower_raw_higher_net = PlanResult {
            actions: vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "b",
            }],
            terminal_state: state,
            raw_ev: 9.9,
            estimated_total_fee_susd: Some(1.0),
            estimated_net_ev: Some(8.9),
            estimated_group_count: Some(1),
            estimated_tx_count: Some(1),
            fee_estimate_source: FeeEstimateSource::StructuralFallback,
            frontier_family: None,
            preserve_markets: Vec::new(),
            family: SolverFamily::Plain,
            compiler_variant: PlanCompilerVariant::BaselineStepPrune,
            selected_common_shift: None,
            selected_mixed_lambda: None,
            selected_active_set_size: None,
            selected_stage_count: None,
            selected_stage1_budget_fraction: None,
            mixed_certificates: Vec::new(),
        };

        assert!(plan_result_is_better(
            &lower_raw_higher_net,
            &higher_raw_lower_net
        ));
    }

    #[test]
    fn forecastflows_head_to_head_helper_prefers_native_when_forecastflows_is_absent_or_worse() {
        let native = dummy_plan_result(
            SolverFamily::Plain,
            PlanCompilerVariant::BaselineStepPrune,
            10.0,
            Some(9.0),
            vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "native",
            }],
        );
        let worse_forecastflows = dummy_plan_result(
            SolverFamily::ForecastFlows,
            PlanCompilerVariant::ForecastFlowsDirect,
            10.0,
            Some(8.0),
            vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "forecastflows",
            }],
        );

        let chosen_with_worse =
            choose_head_to_head_plan_result(native.clone(), Some(worse_forecastflows));
        assert_eq!(chosen_with_worse.family, SolverFamily::Plain);

        let chosen_without_external = choose_head_to_head_plan_result(native.clone(), None);
        assert_eq!(chosen_without_external.family, SolverFamily::Plain);
    }

    #[test]
    fn forecastflows_head_to_head_helper_prefers_forecastflows_when_local_net_ev_is_better() {
        let native = dummy_plan_result(
            SolverFamily::Plain,
            PlanCompilerVariant::BaselineStepPrune,
            10.0,
            Some(9.0),
            vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "native",
            }],
        );
        let forecastflows = dummy_plan_result(
            SolverFamily::ForecastFlows,
            PlanCompilerVariant::ForecastFlowsMixed,
            10.0,
            Some(9.5),
            vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "forecastflows",
            }],
        );

        let chosen = choose_head_to_head_plan_result(native, Some(forecastflows));
        assert_eq!(chosen.family, SolverFamily::ForecastFlows);
        assert_eq!(
            chosen.compiler_variant,
            PlanCompilerVariant::ForecastFlowsMixed
        );
    }

    #[test]
    fn forecastflows_head_to_head_helper_keeps_native_on_tie_via_existing_stable_order() {
        let native = dummy_plan_result(
            SolverFamily::Plain,
            PlanCompilerVariant::BaselineStepPrune,
            10.0,
            Some(9.0),
            vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "native",
            }],
        );
        let forecastflows = dummy_plan_result(
            SolverFamily::ForecastFlows,
            PlanCompilerVariant::ForecastFlowsDirect,
            10.0,
            Some(9.0),
            vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "forecastflows",
            }],
        );

        let chosen = choose_head_to_head_plan_result(native, Some(forecastflows));
        assert_eq!(chosen.family, SolverFamily::Plain);
        assert_eq!(
            chosen.compiler_variant,
            PlanCompilerVariant::BaselineStepPrune
        );
    }

    #[test]
    fn candidate_result_falls_back_to_raw_ev_when_fee_estimate_is_unavailable() {
        let higher_raw = CandidateResult {
            actions: vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "a",
            }],
            raw_ev: 10.0,
            estimated_total_fee_susd: None,
            estimated_net_ev: None,
            estimated_group_count: None,
            estimated_tx_count: None,
            fee_estimate_source: FeeEstimateSource::Unavailable,
            variant: PhaseOrderVariant::NoArb,
            preserve_markets: HashSet::new(),
            forced_first_frontier_family: None,
        };
        let lower_raw = CandidateResult {
            actions: vec![Action::Mint {
                contract_1: "",
                contract_2: "",
                amount: 1.0,
                target_market: "b",
            }],
            raw_ev: 9.0,
            estimated_total_fee_susd: Some(0.1),
            estimated_net_ev: Some(8.9),
            estimated_group_count: Some(1),
            estimated_tx_count: Some(1),
            fee_estimate_source: FeeEstimateSource::StructuralFallback,
            variant: PhaseOrderVariant::NoArb,
            preserve_markets: HashSet::new(),
            forced_first_frontier_family: None,
        };

        assert!(candidate_is_better(&higher_raw, &lower_raw));
    }

    #[test]
    fn benchmark_planner_cost_config_pins_l1_fee_snapshot() {
        let cost_config = benchmark_planner_cost_config_for_test();
        assert!(
            (cost_config.gas_assumptions.l1_fee_per_byte_wei - BENCHMARK_OP_L1_FEE_PER_BYTE_WEI)
                .abs()
                <= 1e-9,
            "benchmark planner should use the fixed L1 byte-fee snapshot"
        );
        assert_eq!(
            cost_config.gas_assumptions.l1_data_fee_floor_susd, BENCHMARK_L1_DATA_FEE_FLOOR_SUSD,
            "benchmark planner should disable the conservative L1 floor"
        );
        assert_eq!(
            cost_config.pricing.source_label,
            "benchmark_snapshot_op_2026_03_08_fixed_l1"
        );
    }
}
