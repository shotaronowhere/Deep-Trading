use std::collections::HashMap;
use std::fmt;

use alloy::primitives::{U160, U256};
use uniswap_v3_math::tick_math::get_sqrt_ratio_at_tick;

use crate::markets::MarketData;
use crate::pools::{Slot0Result, sqrt_price_x96_to_price_outcome, u256_to_f64};
use crate::portfolio::Action;

pub use super::batch_bounds::{
    BatchBoundsError, derive_batch_quote_bounds, derive_batch_quote_bounds_unchecked,
    stamp_plan_with_block, stamp_plans_with_block,
};
pub use super::edge::{planned_cashflow_edge_susd, planned_edge_from_prediction_map_susd};
use super::gas::{
    GasAssumptions, estimate_group_l2_gas_units, estimate_l2_gas_susd, estimate_total_gas_susd,
    hydrate_cached_optimism_l1_fee_per_byte, resolve_l1_fee_per_byte_wei,
};
use super::grouping::{
    ExecutionGroup, GroupingError, group_execution_actions_by_profitability_step,
};
use super::{ExecutionGroupPlan, ExecutionLegPlan, GroupKind, LegKind};

#[derive(Debug, Clone, Copy)]
pub struct ConservativeExecutionConfig {
    pub quote_latency_blocks: u64,
    pub adverse_move_bps_per_block: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferConfig {
    pub buffer_frac: f64,
    pub buffer_min_susd: f64,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            buffer_frac: 0.20,
            buffer_min_susd: 0.25,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupPlanningError {
    Grouping(GroupingError),
    MissingEdge {
        group_index: usize,
        group_kind: GroupKind,
        first_action_index: usize,
    },
    L1FeeHydration {
        rpc_url: String,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupSkipReason {
    NonPositiveEdge,
    MissingDexLegs,
    InvalidL2GasEstimate,
    InvalidTotalGasEstimate,
    NonPositivePostBufferMargin,
    InvalidLegNotional,
}

impl fmt::Display for GroupPlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grouping(err) => write!(f, "{err}"),
            Self::MissingEdge {
                group_index,
                group_kind,
                first_action_index,
            } => write!(
                f,
                "missing edge for group #{group_index} ({group_kind:?}) starting at action index {first_action_index}"
            ),
            Self::L1FeeHydration { rpc_url, reason } => write!(
                f,
                "failed to hydrate Optimism L1 fee-per-byte from RPC '{}': {}",
                rpc_url, reason
            ),
        }
    }
}

impl std::error::Error for GroupPlanningError {}

impl fmt::Display for GroupSkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveEdge => write!(f, "group edge is non-positive or non-finite"),
            Self::MissingDexLegs => write!(f, "group has no DEX legs and is not a direct merge"),
            Self::InvalidL2GasEstimate => write!(f, "l2 gas estimate is non-finite"),
            Self::InvalidTotalGasEstimate => write!(f, "total gas estimate is non-finite"),
            Self::NonPositivePostBufferMargin => {
                write!(f, "edge does not clear gas + profit buffer gate")
            }
            Self::InvalidLegNotional => write!(f, "leg notionals are non-positive or non-finite"),
        }
    }
}

impl From<GroupingError> for GroupPlanningError {
    fn from(value: GroupingError) -> Self {
        Self::Grouping(value)
    }
}

pub fn build_group_plan(
    group: &ExecutionGroup,
    edge_plan_susd: f64,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Option<ExecutionGroupPlan> {
    build_group_plan_with_reason(
        group,
        edge_plan_susd,
        gas_assumptions,
        gas_price_eth,
        eth_usd_assumed,
        buffer,
    )
    .ok()
}

pub fn build_group_plan_with_reason(
    group: &ExecutionGroup,
    edge_plan_susd: f64,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Result<ExecutionGroupPlan, GroupSkipReason> {
    if !edge_plan_susd.is_finite() || edge_plan_susd <= 0.0 {
        return Err(GroupSkipReason::NonPositiveEdge);
    }

    if group.legs.is_empty() && group.kind != GroupKind::DirectMerge {
        return Err(GroupSkipReason::MissingDexLegs);
    }

    let l2_gas_units =
        estimate_group_l2_gas_units(gas_assumptions, group.kind, group.buy_legs, group.sell_legs);
    let gas_l2_susd = estimate_l2_gas_susd(l2_gas_units, gas_price_eth, eth_usd_assumed);
    if !gas_l2_susd.is_finite() {
        return Err(GroupSkipReason::InvalidL2GasEstimate);
    }
    let gas_total_susd = estimate_total_gas_susd(
        gas_assumptions,
        group.kind,
        group.buy_legs,
        group.sell_legs,
        l2_gas_units,
        gas_price_eth,
        eth_usd_assumed,
    );
    if !gas_total_susd.is_finite() {
        return Err(GroupSkipReason::InvalidTotalGasEstimate);
    }

    let profit_buffer_susd = buffer
        .buffer_min_susd
        .max(buffer.buffer_frac.max(0.0) * edge_plan_susd);
    let post_buffer_margin_susd = edge_plan_susd - gas_total_susd - profit_buffer_susd;
    if !post_buffer_margin_susd.is_finite() || post_buffer_margin_susd <= 0.0 {
        return Err(GroupSkipReason::NonPositivePostBufferMargin);
    }
    let slippage_budget_susd = if group.kind == GroupKind::DirectMerge {
        0.0
    } else {
        post_buffer_margin_susd
    };

    let leg_plans = if group.legs.is_empty() {
        Vec::new()
    } else {
        let total_notional: f64 = group.legs.iter().map(|l| l.planned_quote_susd).sum();
        if total_notional <= 0.0 || !total_notional.is_finite() {
            return Err(GroupSkipReason::InvalidLegNotional);
        }

        let mut leg_plans = Vec::with_capacity(group.legs.len());
        for leg in &group.legs {
            let allocated_slippage_susd =
                slippage_budget_susd * (leg.planned_quote_susd / total_notional);
            let (max_cost_susd, min_proceeds_susd) = match leg.kind {
                LegKind::Buy => (Some(leg.planned_quote_susd + allocated_slippage_susd), None),
                LegKind::Sell => (
                    None,
                    Some((leg.planned_quote_susd - allocated_slippage_susd).max(0.0)),
                ),
            };

            leg_plans.push(ExecutionLegPlan {
                action_index: leg.action_index,
                market_name: leg.market_name,
                kind: leg.kind,
                planned_quote_susd: leg.planned_quote_susd,
                conservative_quote_susd: leg.planned_quote_susd,
                adverse_notional_susd: leg.planned_quote_susd,
                allocated_slippage_susd,
                max_cost_susd,
                min_proceeds_susd,
                sqrt_price_limit_x96: None,
            });
        }
        leg_plans
    };

    let guaranteed_profit_floor_susd = if group.kind == GroupKind::DirectMerge {
        // Direct merges have no DEX price-risk legs; profit floor is residual after gas + buffer.
        post_buffer_margin_susd
    } else {
        // For DEX-leg groups, this is the profit floor after gas and worst-case tolerated slippage.
        profit_buffer_susd
    };

    Ok(ExecutionGroupPlan {
        kind: group.kind,
        action_indices: group.action_indices.clone(),
        profitability_step_index: 0,
        step_subgroup_index: 0,
        step_subgroup_count: 1,
        legs: leg_plans,
        planned_at_block: None,
        edge_plan_susd,
        l2_gas_units,
        gas_l2_susd,
        gas_total_susd,
        profit_buffer_susd,
        slippage_budget_susd,
        guaranteed_profit_floor_susd,
    })
}

fn conservative_move_frac(config: ConservativeExecutionConfig) -> f64 {
    ((config.quote_latency_blocks as f64) * (config.adverse_move_bps_per_block as f64) / 10_000.0)
        .max(0.0)
}

fn conservative_quote(planned_quote_susd: f64, kind: LegKind, move_frac: f64) -> f64 {
    match kind {
        LegKind::Buy => planned_quote_susd * (1.0 + move_frac),
        LegKind::Sell => (planned_quote_susd * (1.0 - move_frac)).max(0.0),
    }
}

fn conservative_quote_delta(planned_quote_susd: f64, kind: LegKind, move_frac: f64) -> f64 {
    match kind {
        LegKind::Buy => (planned_quote_susd * move_frac).max(0.0),
        LegKind::Sell => {
            (planned_quote_susd - conservative_quote(planned_quote_susd, kind, move_frac)).max(0.0)
        }
    }
}

fn outcome_price_to_sqrt_limit_x96(outcome_price: f64, is_token1_outcome: bool) -> Option<U160> {
    if !outcome_price.is_finite() || outcome_price <= 0.0 {
        return None;
    }
    let raw_price = if is_token1_outcome {
        1.0 / outcome_price
    } else {
        outcome_price
    };
    if !raw_price.is_finite() || raw_price <= 0.0 {
        return None;
    }

    let scaled = raw_price.sqrt() * 2f64.powi(96);
    if !scaled.is_finite() || scaled <= 0.0 || scaled >= 2f64.powi(128) {
        return None;
    }
    Some(U160::from(scaled.round() as u128))
}

#[derive(Debug, Clone, Copy)]
struct MarketPriceState {
    is_token1_outcome: bool,
    current_price: f64,
    buy_limit_price: f64,
    sell_limit_price: f64,
    liquidity_raw: f64,
}

fn market_price_state(
    slot0: &Slot0Result,
    market: &'static MarketData,
) -> Option<MarketPriceState> {
    let pool = market.pool.as_ref()?;
    let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
    let zero_for_one_buy = pool.token0.eq_ignore_ascii_case(market.quote_token);
    let liquidity: u128 = pool.liquidity.parse().ok()?;
    if liquidity == 0 {
        return None;
    }

    let tick_0 = pool.ticks.first()?.tick_idx;
    let tick_1 = pool.ticks.get(1)?.tick_idx;
    let sqrt_lo = U256::from(get_sqrt_ratio_at_tick(tick_0.min(tick_1)).ok()?);
    let sqrt_hi = U256::from(get_sqrt_ratio_at_tick(tick_0.max(tick_1)).ok()?);
    let (sqrt_buy_limit, sqrt_sell_limit) = if zero_for_one_buy {
        (sqrt_lo, sqrt_hi)
    } else {
        (sqrt_hi, sqrt_lo)
    };

    let current_price = u256_to_f64(sqrt_price_x96_to_price_outcome(
        slot0.sqrt_price_x96,
        is_token1_outcome,
    )?);
    if !current_price.is_finite() || current_price <= 0.0 {
        return None;
    }

    let buy_limit_price = u256_to_f64(sqrt_price_x96_to_price_outcome(
        sqrt_buy_limit,
        is_token1_outcome,
    )?);
    let sell_limit_price = u256_to_f64(sqrt_price_x96_to_price_outcome(
        sqrt_sell_limit,
        is_token1_outcome,
    )?);
    if !buy_limit_price.is_finite()
        || !sell_limit_price.is_finite()
        || buy_limit_price <= 0.0
        || sell_limit_price <= 0.0
    {
        return None;
    }

    Some(MarketPriceState {
        is_token1_outcome,
        current_price,
        buy_limit_price,
        sell_limit_price,
        liquidity_raw: (liquidity as f64) / 1e18,
    })
}

fn action_amount_for_leg(actions: &[Action], leg: &ExecutionLegPlan) -> Option<f64> {
    let action = actions.get(leg.action_index)?;
    match (leg.kind, action) {
        (LegKind::Buy, Action::Buy { amount, .. }) => Some(*amount),
        (LegKind::Sell, Action::Sell { amount, .. }) => Some(*amount),
        _ => None,
    }
}

fn conservative_sqrt_price_limit(
    actions: &[Action],
    leg: &ExecutionLegPlan,
    slot0: &Slot0Result,
    market: &'static MarketData,
    move_frac: f64,
) -> Option<U160> {
    let state = market_price_state(slot0, market)?;
    let amount = action_amount_for_leg(actions, leg)?;
    if !amount.is_finite() || amount <= 0.0 || state.liquidity_raw <= 0.0 {
        return None;
    }

    let sqrt_current = state.current_price.sqrt();
    if !sqrt_current.is_finite() || sqrt_current <= 0.0 {
        return None;
    }

    let terminal_price = match leg.kind {
        LegKind::Buy => {
            let lambda = sqrt_current / state.liquidity_raw;
            let denom = 1.0 - amount * lambda;
            if !denom.is_finite() || denom <= EPSILON {
                state.buy_limit_price
            } else {
                (state.current_price / (denom * denom))
                    .max(state.current_price)
                    .min(state.buy_limit_price)
            }
        }
        LegKind::Sell => {
            const EXECUTION_FEE_FACTOR: f64 = 1.0 - (100.0 / 1_000_000.0);
            let kappa = EXECUTION_FEE_FACTOR * sqrt_current / state.liquidity_raw;
            let denom = 1.0 + amount * kappa;
            if !denom.is_finite() || denom <= 0.0 {
                return None;
            }
            (state.current_price / (denom * denom))
                .max(state.sell_limit_price)
                .min(state.current_price)
        }
    };

    let adverse_price = match leg.kind {
        LegKind::Buy => (terminal_price * (1.0 + move_frac)).min(state.buy_limit_price),
        LegKind::Sell => {
            (terminal_price * (1.0 - move_frac).max(EPSILON)).max(state.sell_limit_price)
        }
    };
    outcome_price_to_sqrt_limit_x96(adverse_price, state.is_token1_outcome)
}

const EPSILON: f64 = 1e-9;

fn apply_market_context_to_plan(
    actions: &[Action],
    plan: &mut ExecutionGroupPlan,
    slot0_by_market: &HashMap<&'static str, (&Slot0Result, &'static MarketData)>,
    move_frac: f64,
) -> bool {
    if plan.legs.is_empty() {
        return true;
    }

    let mut total_conservative_notional = 0.0_f64;
    let mut conservative_delta_susd = 0.0_f64;

    for leg in &mut plan.legs {
        leg.conservative_quote_susd =
            conservative_quote(leg.planned_quote_susd, leg.kind, move_frac);
        leg.adverse_notional_susd = leg.conservative_quote_susd;
        conservative_delta_susd +=
            conservative_quote_delta(leg.planned_quote_susd, leg.kind, move_frac);
        total_conservative_notional += leg.conservative_quote_susd.max(0.0);
        leg.sqrt_price_limit_x96 = leg
            .market_name
            .and_then(|market_name| slot0_by_market.get(market_name))
            .and_then(|(slot0, market)| {
                conservative_sqrt_price_limit(actions, leg, slot0, *market, move_frac)
            });
        if leg.market_name.is_some() && leg.sqrt_price_limit_x96.is_none() {
            return false;
        }
    }

    if !conservative_delta_susd.is_finite()
        || !total_conservative_notional.is_finite()
        || total_conservative_notional <= 0.0
    {
        return false;
    }

    let remaining_profit_after_conservative =
        plan.edge_plan_susd - plan.gas_total_susd - conservative_delta_susd;
    if !remaining_profit_after_conservative.is_finite()
        || remaining_profit_after_conservative <= 0.0
    {
        return false;
    }

    plan.slippage_budget_susd = (plan.slippage_budget_susd - conservative_delta_susd).max(0.0);
    plan.guaranteed_profit_floor_susd =
        remaining_profit_after_conservative - plan.slippage_budget_susd;
    if !plan.guaranteed_profit_floor_susd.is_finite() || plan.guaranteed_profit_floor_susd <= 0.0 {
        return false;
    }
    plan.profit_buffer_susd = plan.guaranteed_profit_floor_susd;

    for leg in &mut plan.legs {
        leg.allocated_slippage_susd =
            plan.slippage_budget_susd * (leg.conservative_quote_susd / total_conservative_notional);
        match leg.kind {
            LegKind::Buy => {
                leg.max_cost_susd = Some(leg.conservative_quote_susd + leg.allocated_slippage_susd);
                leg.min_proceeds_susd = None;
            }
            LegKind::Sell => {
                leg.max_cost_susd = None;
                leg.min_proceeds_susd =
                    Some((leg.conservative_quote_susd - leg.allocated_slippage_susd).max(0.0));
            }
        }
    }

    true
}

fn apply_market_context_to_plans(
    actions: &[Action],
    plans: &mut Vec<ExecutionGroupPlan>,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    config: ConservativeExecutionConfig,
) {
    let move_frac = conservative_move_frac(config);
    let slot0_by_market: HashMap<&'static str, (&Slot0Result, &'static MarketData)> = slot0_results
        .iter()
        .map(|(slot0, market)| (market.name, (slot0, *market)))
        .collect();

    let mut truncate_from = plans.len();
    for idx in 0..plans.len() {
        let failed_step = {
            let plan = &mut plans[idx];
            if apply_market_context_to_plan(actions, plan, &slot0_by_market, move_frac) {
                None
            } else {
                Some(plan.profitability_step_index)
            }
        };
        if let Some(step_index) = failed_step {
            truncate_from = plans
                .iter()
                .position(|plan| plan.profitability_step_index == step_index)
                .unwrap_or(idx);
            tracing::info!(
                step_index,
                "stopping planning at first profitability step invalidated by conservative execution repricing"
            );
            break;
        }
    }
    plans.truncate(truncate_from);
}

pub fn build_group_plans_with_edge_provider<F>(
    actions: &[Action],
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
    mut edge_provider: F,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError>
where
    F: FnMut(usize, &ExecutionGroup) -> Option<f64>,
{
    // DirectBuy groups have no route-local cashflow edge and must receive
    // externally supplied expected-value edge from the caller.
    let step_groups = group_execution_actions_by_profitability_step(actions)?;
    let mut effective_gas_assumptions = *gas_assumptions;
    if let Some(l1_fee_per_byte_wei) = resolve_l1_fee_per_byte_wei(gas_assumptions) {
        effective_gas_assumptions.l1_fee_per_byte_wei = l1_fee_per_byte_wei;
    }
    if !effective_gas_assumptions.l1_fee_per_byte_wei.is_finite()
        || effective_gas_assumptions.l1_fee_per_byte_wei <= 0.0
    {
        tracing::warn!(
            l1_fee_per_byte_wei = effective_gas_assumptions.l1_fee_per_byte_wei,
            "no usable Optimism L1 fee-per-byte estimate; strict planning will skip all groups"
        );
    }
    let mut plans = Vec::new();
    let mut group_index = 0usize;

    for (step_index, step_group) in step_groups.iter().enumerate() {
        let step_subgroup_count = step_group.strict_groups.len();
        let mut step_plans = Vec::with_capacity(step_subgroup_count);
        let mut step_failed = false;
        for (step_subgroup_index, group) in step_group.strict_groups.iter().enumerate() {
            let subgroup_index = group_index;
            group_index += 1;
            let first_action_index =
                group
                    .first_action_index()
                    .ok_or(GroupPlanningError::MissingEdge {
                        group_index: subgroup_index,
                        group_kind: group.kind,
                        first_action_index: 0,
                    })?;

            let edge =
                edge_provider(subgroup_index, group).ok_or(GroupPlanningError::MissingEdge {
                    group_index: subgroup_index,
                    group_kind: group.kind,
                    first_action_index,
                })?;

            match build_group_plan_with_reason(
                group,
                edge,
                &effective_gas_assumptions,
                gas_price_eth,
                eth_usd_assumed,
                buffer,
            ) {
                Ok(mut plan) => {
                    plan.profitability_step_index = step_index;
                    plan.step_subgroup_index = step_subgroup_index;
                    plan.step_subgroup_count = step_subgroup_count;
                    step_plans.push(plan);
                }
                Err(skip_reason) => {
                    tracing::info!(
                        group_index = subgroup_index,
                        step_index,
                        step_subgroup_index,
                        group_kind = ?group.kind,
                        first_action_index,
                        %skip_reason,
                        "skipped group"
                    );
                    step_failed = true;
                    break;
                }
            }
        }

        if step_failed {
            tracing::info!(
                step_index,
                step_kind = ?step_group.kind,
                "stopping planning at first unplannable profitability step to avoid partial-step execution"
            );
            break;
        }
        plans.extend(step_plans);
    }

    Ok(plans)
}

/// Convenience helper for groups where edge can be derived from route-local cashflow.
///
/// This intentionally rejects `DirectBuy` groups: standalone buy actions have negative
/// cashflow by construction (`proceeds - cost`) and require externally supplied edge from
/// the optimizer's expected-value model.
pub fn build_group_plans_from_cashflow(
    actions: &[Action],
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError> {
    build_group_plans_with_edge_provider(
        actions,
        gas_assumptions,
        gas_price_eth,
        eth_usd_assumed,
        buffer,
        |_, group| match group.kind {
            GroupKind::DirectBuy => None,
            _ => Some(planned_cashflow_edge_susd(group)),
        },
    )
}

/// Convenience helper that derives `DirectBuy` edge from prediction EV delta and
/// non-buy groups from route-local cashflow.
pub fn build_group_plans_with_prediction_edges(
    actions: &[Action],
    prediction_by_market: &HashMap<String, f64>,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError> {
    build_group_plans_with_edge_provider(
        actions,
        gas_assumptions,
        gas_price_eth,
        eth_usd_assumed,
        buffer,
        |_, group| planned_edge_from_prediction_map_susd(group, prediction_by_market),
    )
}

/// Uses the built-in L1 prediction table for direct-buy edge attribution.
pub fn build_group_plans_with_default_edges(
    actions: &[Action],
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError> {
    let predictions = crate::pools::prediction_map();
    build_group_plans_with_prediction_edges(
        actions,
        predictions,
        gas_assumptions,
        gas_price_eth,
        eth_usd_assumed,
        buffer,
    )
}

pub fn build_group_plans_with_market_context<F>(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    conservative_config: ConservativeExecutionConfig,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
    edge_provider: F,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError>
where
    F: FnMut(usize, &ExecutionGroup) -> Option<f64>,
{
    let mut plans = build_group_plans_with_edge_provider(
        actions,
        gas_assumptions,
        gas_price_eth,
        eth_usd_assumed,
        buffer,
        edge_provider,
    )?;
    apply_market_context_to_plans(actions, &mut plans, slot0_results, conservative_config);
    Ok(plans)
}

pub fn build_group_plans_with_default_edges_and_market_context(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    conservative_config: ConservativeExecutionConfig,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError> {
    let mut plans = build_group_plans_with_default_edges(
        actions,
        gas_assumptions,
        gas_price_eth,
        eth_usd_assumed,
        buffer,
    )?;
    apply_market_context_to_plans(actions, &mut plans, slot0_results, conservative_config);
    Ok(plans)
}

/// Hydration-first planning helper: refresh Optimism L1 fee-per-byte (cache-aware)
/// before building market-context plans with default edges.
pub async fn build_group_plans_with_default_edges_and_l1_hydration(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    conservative_config: ConservativeExecutionConfig,
    gas_assumptions: &GasAssumptions,
    rpc_url: &str,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError> {
    let mut hydrated = *gas_assumptions;
    hydrate_cached_optimism_l1_fee_per_byte(&mut hydrated, rpc_url)
        .await
        .map_err(|err| GroupPlanningError::L1FeeHydration {
            rpc_url: rpc_url.to_string(),
            reason: err.to_string(),
        })?;

    build_group_plans_with_default_edges_and_market_context(
        actions,
        slot0_results,
        conservative_config,
        &hydrated,
        gas_price_eth,
        eth_usd_assumed,
        buffer,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::execution::BatchQuoteBounds;
    use crate::execution::GroupKind;
    use crate::execution::grouping::{GroupingError, group_execution_actions};
    use crate::markets::MARKETS_L1;
    use crate::pools::{Slot0Result, normalize_market_name, prediction_to_sqrt_price_x96};
    use crate::portfolio::{self, RebalanceMode};
    use alloy::primitives::{Address, U256};

    fn buy(name: &'static str, amount: f64, cost: f64) -> Action {
        Action::Buy {
            market_name: name,
            amount,
            cost,
        }
    }

    fn sell(name: &'static str, amount: f64, proceeds: f64) -> Action {
        Action::Sell {
            market_name: name,
            amount,
            proceeds,
        }
    }

    fn merge(amount: f64) -> Action {
        Action::Merge {
            contract_1: "c1",
            contract_2: "c2",
            amount,
            source_market: "s",
        }
    }

    fn test_gas_assumptions() -> GasAssumptions {
        GasAssumptions {
            l1_fee_per_byte_wei: 1.0e11,
            ..GasAssumptions::default()
        }
    }

    fn full_slot0_results_with_prediction_multiplier(
        multiplier: f64,
    ) -> Vec<(Slot0Result, &'static crate::markets::MarketData)> {
        let preds = crate::pools::prediction_map();
        MARKETS_L1
            .iter()
            .filter(|m| m.pool.is_some())
            .filter(|m| preds.contains_key(&normalize_market_name(m.name)))
            .map(|market| {
                let pool = market.pool.as_ref().expect("pooled market required");
                let key = normalize_market_name(market.name);
                let pred = preds[&key];
                let is_token1_outcome =
                    pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
                let price = (pred * multiplier).max(1e-9);
                let sqrt_price = prediction_to_sqrt_price_x96(price, is_token1_outcome)
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
                    market,
                )
            })
            .collect()
    }

    #[test]
    fn runtime_actions_produce_nonempty_plan_prefix_under_realistic_gas() {
        let slot0_results = full_slot0_results_with_prediction_multiplier(0.35);
        let gas = GasAssumptions {
            l1_fee_per_byte_wei: 1.0e8,
            ..GasAssumptions::default()
        };
        let actions = portfolio::rebalance_with_gas_pricing(
            &HashMap::new(),
            200.0,
            &slot0_results,
            RebalanceMode::Full,
            &gas,
            1e-9,
            3000.0,
        );
        assert!(
            !actions.is_empty(),
            "runtime rebalance fixture should emit actions"
        );

        let plans = build_group_plans_with_default_edges(
            &actions,
            &gas,
            1e-9,
            3000.0,
            BufferConfig::default(),
        )
        .expect("planning should succeed for runtime action stream");
        assert!(
            !plans.is_empty(),
            "runtime action stream should produce non-empty strict execution prefix"
        );
    }

    #[test]
    fn skips_group_when_edge_below_gas_plus_buffer() {
        let actions = vec![sell("x", 1.0, 1.0)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        let plan = build_group_plan(
            &groups[0],
            0.5,
            &test_gas_assumptions(),
            1e-9,
            3000.0,
            BufferConfig::default(),
        );
        assert!(plan.is_none(), "group should be skipped by strict gate");
    }

    #[test]
    fn build_group_plan_with_reason_reports_non_positive_edge() {
        let actions = vec![sell("x", 1.0, 1.0)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        let err = build_group_plan_with_reason(
            &groups[0],
            0.0,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect_err("non-positive edge should report explicit skip reason");
        assert_eq!(err, GroupSkipReason::NonPositiveEdge);
    }

    #[test]
    fn build_group_plan_with_reason_reports_invalid_leg_notional() {
        let actions = vec![sell("x", 1.0, 0.0)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        let gas = GasAssumptions {
            direct_sell_l2_units: 0,
            l1_data_fee_floor_susd: 0.0,
            l1_fee_per_byte_wei: 1.0,
            ..GasAssumptions::default()
        };
        let err = build_group_plan_with_reason(
            &groups[0],
            10.0,
            &gas,
            0.0,
            0.0,
            BufferConfig {
                buffer_frac: 0.0,
                buffer_min_susd: 0.0,
            },
        )
        .expect_err("zero notional legs should report explicit skip reason");
        assert_eq!(err, GroupSkipReason::InvalidLegNotional);
    }

    #[test]
    fn leg_allocations_sum_to_group_budget() {
        let actions = vec![
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "target",
            },
            sell("a", 10.0, 7.0),
            sell("b", 10.0, 7.0),
        ];

        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::MintSell);

        let plan = build_group_plan(
            &groups[0],
            8.0,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("plan should be feasible");

        let alloc_sum: f64 = plan.legs.iter().map(|l| l.allocated_slippage_susd).sum();
        let tol = 1e-9 * (1.0 + alloc_sum.abs() + plan.slippage_budget_susd.abs());
        assert!(
            (alloc_sum - plan.slippage_budget_susd).abs() <= tol,
            "allocations should sum to group budget"
        );
    }

    #[test]
    fn plans_preserve_profitability_step_and_subgroup_order() {
        let actions = vec![
            sell("a", 1.0, 10.0),
            sell("b", 1.0, 2.0),
            buy("c", 1.0, 1.0),
        ];
        let gas = GasAssumptions {
            direct_sell_l2_units: 0,
            direct_buy_l2_units: 0,
            l1_data_fee_floor_susd: 0.0,
            l1_fee_per_byte_wei: 1.0,
            ..GasAssumptions::default()
        };
        let plans = build_group_plans_with_edge_provider(
            &actions,
            &gas,
            0.0,
            0.0,
            BufferConfig {
                buffer_frac: 0.0,
                buffer_min_susd: 0.0,
            },
            |_, _| Some(10.0),
        )
        .expect("planning should succeed");

        assert_eq!(plans.len(), 3, "expected one plan per strict subgroup");
        assert_eq!(plans[0].profitability_step_index, 0);
        assert_eq!(plans[0].step_subgroup_index, 0);
        assert_eq!(plans[0].step_subgroup_count, 2);
        assert_eq!(plans[0].action_indices, vec![0]);
        assert_eq!(plans[1].profitability_step_index, 0);
        assert_eq!(plans[1].step_subgroup_index, 1);
        assert_eq!(plans[1].step_subgroup_count, 2);
        assert_eq!(plans[1].action_indices, vec![1]);
        assert_eq!(plans[2].profitability_step_index, 1);
        assert_eq!(plans[2].step_subgroup_index, 0);
        assert_eq!(plans[2].step_subgroup_count, 1);
        assert_eq!(plans[2].action_indices, vec![2]);
    }

    #[test]
    fn drops_partial_profitability_step_and_halts_following_steps() {
        let actions = vec![
            sell("a", 1.0, 10.0),
            sell("b", 1.0, 0.0),
            buy("c", 1.0, 1.0),
        ];
        let gas = GasAssumptions {
            direct_sell_l2_units: 0,
            direct_buy_l2_units: 0,
            l1_data_fee_floor_susd: 0.0,
            l1_fee_per_byte_wei: 1.0,
            ..GasAssumptions::default()
        };
        let plans = build_group_plans_with_edge_provider(
            &actions,
            &gas,
            0.0,
            0.0,
            BufferConfig {
                buffer_frac: 0.0,
                buffer_min_susd: 0.0,
            },
            |_, _| Some(10.0),
        )
        .expect("planning should not hard-fail on skip");

        assert!(
            plans.is_empty(),
            "step with unplannable subgroup must be dropped atomically, and later steps must not be planned"
        );
    }

    #[test]
    fn cashflow_builder_rejects_direct_buy_groups_without_external_edge() {
        let actions = vec![buy("c", 1.0, 1.0)];
        let err = build_group_plans_from_cashflow(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect_err("direct buy should require external edge");
        assert!(matches!(
            err,
            GroupPlanningError::MissingEdge {
                group_kind: GroupKind::DirectBuy,
                ..
            }
        ));
    }

    #[test]
    fn edge_provider_rejects_missing_direct_buy_edge() {
        let actions = vec![buy("c", 1.0, 1.0), sell("d", 1.0, 1.0)];
        let err = build_group_plans_with_edge_provider(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
            |_, group| match group.kind {
                GroupKind::DirectBuy => None,
                _ => Some(planned_cashflow_edge_susd(group)),
            },
        )
        .expect_err("direct buy should require an externally supplied edge");
        assert!(matches!(
            err,
            GroupPlanningError::MissingEdge {
                group_kind: GroupKind::DirectBuy,
                ..
            }
        ));
    }

    #[test]
    fn planning_fails_closed_on_invalid_mint_sell_shape() {
        let actions = vec![Action::Mint {
            contract_1: "c1",
            contract_2: "c2",
            amount: 10.0,
            target_market: "a",
        }];
        let err = build_group_plans_from_cashflow(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect_err("invalid mint-sell shape should fail closed");
        assert!(matches!(
            err,
            GroupPlanningError::Grouping(GroupingError::InvalidMintSellGroup {
                start_index: 0,
                end_index: 0,
            })
        ));
    }

    #[test]
    fn planning_fails_closed_on_standalone_mint_action() {
        let actions = vec![Action::Mint {
            contract_1: "c1",
            contract_2: "c2",
            amount: 1.0,
            target_market: "x",
        }];
        let err = build_group_plans_from_cashflow(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect_err("standalone mint should fail closed");
        assert!(matches!(
            err,
            GroupPlanningError::Grouping(GroupingError::InvalidMintSellGroup {
                start_index: 0,
                end_index: 0
            })
        ));
    }

    #[test]
    fn prediction_edge_builder_plans_direct_buy_groups() {
        let actions = vec![buy("mkt", 2.0, 0.9)];
        let mut prediction_by_market = HashMap::new();
        prediction_by_market.insert("mkt".to_string(), 0.6);

        let gas = GasAssumptions {
            direct_buy_l2_units: 0,
            l1_data_fee_floor_susd: 0.0,
            l1_fee_per_byte_wei: 1.0e11,
            ..GasAssumptions::default()
        };
        let plans = build_group_plans_with_prediction_edges(
            &actions,
            &prediction_by_market,
            &gas,
            0.0,
            3000.0,
            BufferConfig {
                buffer_frac: 0.0,
                buffer_min_susd: 0.0,
            },
        )
        .expect("prediction edge planning should succeed");

        assert_eq!(plans.len(), 1, "direct buy should receive predicted edge");
        assert_eq!(plans[0].kind, GroupKind::DirectBuy);
        let expected_edge = 2.0 * 0.6 - 0.9;
        assert!(
            (plans[0].edge_plan_susd - expected_edge).abs() <= 1e-12,
            "unexpected direct-buy edge attribution"
        );
    }

    #[test]
    fn prediction_edge_builder_normalizes_market_name_lookup() {
        let actions = vec![buy("MKT\t", 2.0, 0.9)];
        let mut prediction_by_market = HashMap::new();
        prediction_by_market.insert("mkt".to_string(), 0.6);

        let gas = GasAssumptions {
            direct_buy_l2_units: 0,
            l1_data_fee_floor_susd: 0.0,
            l1_fee_per_byte_wei: 1.0e11,
            ..GasAssumptions::default()
        };
        let plans = build_group_plans_with_prediction_edges(
            &actions,
            &prediction_by_market,
            &gas,
            0.0,
            3000.0,
            BufferConfig {
                buffer_frac: 0.0,
                buffer_min_susd: 0.0,
            },
        )
        .expect("normalized prediction lookup should succeed");

        assert_eq!(plans.len(), 1, "normalized market key should match");
        let expected_edge = 2.0 * 0.6 - 0.9;
        assert!((plans[0].edge_plan_susd - expected_edge).abs() <= 1e-12);
    }

    #[test]
    fn prediction_edge_builder_skips_direct_buy_without_prediction() {
        let actions = vec![buy("missing", 1.0, 1.0)];
        let prediction_by_market = HashMap::new();
        let gas = GasAssumptions {
            direct_buy_l2_units: 0,
            l1_data_fee_floor_susd: 0.0,
            l1_fee_per_byte_wei: 1.0e11,
            ..GasAssumptions::default()
        };

        let plans = build_group_plans_with_prediction_edges(
            &actions,
            &prediction_by_market,
            &gas,
            0.0,
            3000.0,
            BufferConfig {
                buffer_frac: 0.0,
                buffer_min_susd: 0.0,
            },
        )
        .expect("missing prediction should not hard-fail planning");
        assert!(
            plans.is_empty(),
            "direct buy with no predicted edge should be skipped"
        );
    }

    #[tokio::test]
    async fn hydration_entrypoint_returns_error_when_hydration_fails() {
        let actions = vec![sell("x", 1.0, 2.0)];
        let slot0_results = Vec::new();
        let err = build_group_plans_with_default_edges_and_l1_hydration(
            &actions,
            &slot0_results,
            ConservativeExecutionConfig {
                quote_latency_blocks: 1,
                adverse_move_bps_per_block: 15,
            },
            &GasAssumptions::default(),
            "",
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .await
        .expect_err("empty RPC URL should fail hydration");
        assert!(
            matches!(err, GroupPlanningError::L1FeeHydration { .. }),
            "expected typed hydration error, got: {err}"
        );
    }

    #[test]
    fn cashflow_builder_drops_negative_edge_mint_sell_group() {
        let actions = vec![
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "target",
            },
            sell("a", 10.0, 2.0),
            sell("b", 10.0, 2.0),
        ];
        let plans = build_group_plans_from_cashflow(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("planning should not fail for mint-sell");
        assert!(
            plans.is_empty(),
            "negative-edge mint-sell group should be filtered out"
        );
    }

    #[test]
    fn direct_merge_is_plannable_without_dex_legs() {
        let actions = vec![merge(10.0)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::DirectMerge);

        let edge = planned_cashflow_edge_susd(&groups[0]);
        let plan = build_group_plan(
            &groups[0],
            edge,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("direct merge should be plannable without dex legs");

        assert_eq!(plan.kind, GroupKind::DirectMerge);
        assert!(
            plan.legs.is_empty(),
            "direct merge plan should not contain slippage legs"
        );
        assert!(
            plan.slippage_budget_susd == 0.0,
            "direct merge should not reserve unenforceable slippage budget"
        );
        assert!(
            plan.guaranteed_profit_floor_susd > 0.0,
            "direct merge should retain positive post-buffer profit floor"
        );
    }

    #[test]
    fn cashflow_builder_includes_direct_merge_when_profitable() {
        let actions = vec![merge(10.0)];
        let plans = build_group_plans_from_cashflow(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("cashflow planning should succeed for direct merge");
        assert_eq!(plans.len(), 1, "direct merge should produce one group plan");
        assert_eq!(plans[0].kind, GroupKind::DirectMerge);
        assert!(plans[0].legs.is_empty());
    }

    #[test]
    fn direct_merge_is_skipped_when_edge_is_below_gas_plus_buffer() {
        let actions = vec![merge(0.01)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::DirectMerge);

        let edge = planned_cashflow_edge_susd(&groups[0]);
        let plan = build_group_plan(
            &groups[0],
            edge,
            &test_gas_assumptions(),
            1e-9,
            3000.0,
            BufferConfig::default(),
        );
        assert!(
            plan.is_none(),
            "tiny direct merge should be skipped when gas + buffer exceeds edge"
        );
    }

    #[test]
    fn derives_batch_sell_bounds_from_sell_legs() {
        let actions = vec![
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "target",
            },
            sell("a", 10.0, 7.0),
            sell("b", 10.0, 7.0),
        ];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        let plan = build_group_plan(
            &groups[0],
            8.0,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("plan should be feasible");

        let mut plan = plan;
        stamp_plan_with_block(&mut plan, 100);
        let bounds = derive_batch_quote_bounds(&plan, 100, 2)
            .expect("sell bounds derivation should not error")
            .expect("sell bounds should exist");
        match bounds {
            BatchQuoteBounds::Sell {
                planned_total_out_susd,
                min_total_out_susd,
            } => {
                assert!((planned_total_out_susd - 14.0).abs() <= 1e-12);
                let tol = 1e-9 * (1.0 + planned_total_out_susd + min_total_out_susd);
                assert!(
                    ((planned_total_out_susd - min_total_out_susd) - plan.slippage_budget_susd)
                        .abs()
                        <= tol
                );
            }
            BatchQuoteBounds::Buy { .. } => panic!("expected sell bounds"),
        }
    }

    #[test]
    fn derives_batch_buy_bounds_from_buy_legs() {
        let actions = vec![buy("c", 1.0, 1.0)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        let plan = build_group_plan(
            &groups[0],
            2.0,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("direct buy should be feasible with explicit edge");

        let mut plan = plan;
        stamp_plan_with_block(&mut plan, 100);
        let bounds = derive_batch_quote_bounds(&plan, 100, 2)
            .expect("buy bounds derivation should not error")
            .expect("buy bounds should exist");
        match bounds {
            BatchQuoteBounds::Buy {
                planned_total_in_susd,
                max_total_in_susd,
            } => {
                assert!((planned_total_in_susd - 1.0).abs() <= 1e-12);
                let tol = 1e-9 * (1.0 + planned_total_in_susd + max_total_in_susd);
                assert!(
                    ((max_total_in_susd - planned_total_in_susd) - plan.slippage_budget_susd).abs()
                        <= tol
                );
            }
            BatchQuoteBounds::Sell { .. } => panic!("expected buy bounds"),
        }
    }

    #[test]
    fn derives_batch_buy_bounds_from_buy_merge_group() {
        let actions = vec![buy("a", 1.0, 0.4), buy("b", 1.0, 0.5), merge(1.0)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::BuyMerge);

        let plan = build_group_plan(
            &groups[0],
            3.0,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("buy-merge group should be feasible");

        let mut plan = plan;
        stamp_plan_with_block(&mut plan, 100);
        let bounds = derive_batch_quote_bounds(&plan, 100, 2)
            .expect("buy-merge bounds derivation should not error")
            .expect("buy-merge bounds should exist");
        match bounds {
            BatchQuoteBounds::Buy {
                planned_total_in_susd,
                max_total_in_susd,
            } => {
                assert!((planned_total_in_susd - 0.9).abs() <= 1e-12);
                let tol = 1e-9 * (1.0 + planned_total_in_susd + max_total_in_susd);
                assert!(
                    ((max_total_in_susd - planned_total_in_susd) - plan.slippage_budget_susd).abs()
                        <= tol
                );
            }
            BatchQuoteBounds::Sell { .. } => panic!("expected buy bounds"),
        }
    }

    #[test]
    fn direct_merge_has_no_batch_router_bounds() {
        let actions = vec![merge(10.0)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        let edge = planned_cashflow_edge_susd(&groups[0]);
        let plan = build_group_plan(
            &groups[0],
            edge,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect("direct merge should be plannable");

        let mut plan = plan;
        stamp_plan_with_block(&mut plan, 100);
        assert!(
            derive_batch_quote_bounds(&plan, 100, 2)
                .expect("direct merge bounds derivation should not error")
                .is_none(),
            "direct merge should bypass batch-router buy/sell bounds"
        );
    }

    #[test]
    fn mixed_leg_directions_error_in_batch_bounds_derivation() {
        let plan = ExecutionGroupPlan {
            kind: GroupKind::MintSell,
            action_indices: vec![0, 1],
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![
                ExecutionLegPlan {
                    action_index: 0,
                    market_name: Some("a"),
                    kind: LegKind::Buy,
                    planned_quote_susd: 1.0,
                    conservative_quote_susd: 1.0,
                    adverse_notional_susd: 1.0,
                    allocated_slippage_susd: 0.1,
                    max_cost_susd: Some(1.1),
                    min_proceeds_susd: None,
                    sqrt_price_limit_x96: Some(U160::from(1u8)),
                },
                ExecutionLegPlan {
                    action_index: 1,
                    market_name: Some("b"),
                    kind: LegKind::Sell,
                    planned_quote_susd: 1.5,
                    conservative_quote_susd: 1.5,
                    adverse_notional_susd: 1.5,
                    allocated_slippage_susd: 0.1,
                    max_cost_susd: None,
                    min_proceeds_susd: Some(1.4),
                    sqrt_price_limit_x96: Some(U160::from(1u8)),
                },
            ],
            planned_at_block: Some(100),
            edge_plan_susd: 1.0,
            l2_gas_units: 1,
            gas_l2_susd: 0.1,
            gas_total_susd: 0.2,
            profit_buffer_susd: 0.25,
            slippage_budget_susd: 0.65,
            guaranteed_profit_floor_susd: 0.25,
        };

        let err = derive_batch_quote_bounds(&plan, 100, 2).expect_err("mixed legs must error");
        assert_eq!(err, BatchBoundsError::MixedLegDirections);
    }

    #[test]
    fn stale_plan_errors_in_batch_bounds_derivation() {
        let mut plan = ExecutionGroupPlan {
            kind: GroupKind::DirectBuy,
            action_indices: vec![0],
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some("a"),
                kind: LegKind::Buy,
                planned_quote_susd: 1.0,
                conservative_quote_susd: 1.0,
                adverse_notional_susd: 1.0,
                allocated_slippage_susd: 0.1,
                max_cost_susd: Some(1.1),
                min_proceeds_susd: None,
                sqrt_price_limit_x96: Some(U160::from(1u8)),
            }],
            planned_at_block: None,
            edge_plan_susd: 1.0,
            l2_gas_units: 1,
            gas_l2_susd: 0.1,
            gas_total_susd: 0.2,
            profit_buffer_susd: 0.25,
            slippage_budget_susd: 0.65,
            guaranteed_profit_floor_susd: 0.25,
        };
        let err =
            derive_batch_quote_bounds(&plan, 100, 2).expect_err("unstamped plans should fail");
        assert_eq!(err, BatchBoundsError::StalePlan);

        stamp_plan_with_block(&mut plan, 90);
        let err = derive_batch_quote_bounds(&plan, 100, 2).expect_err("stale plans should fail");
        assert_eq!(err, BatchBoundsError::StalePlan);
    }

    #[test]
    fn batch_bounds_use_conservative_quotes() {
        let plan = ExecutionGroupPlan {
            kind: GroupKind::DirectBuy,
            action_indices: vec![0],
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some("a"),
                kind: LegKind::Buy,
                planned_quote_susd: 1.0,
                conservative_quote_susd: 1.2,
                adverse_notional_susd: 1.2,
                allocated_slippage_susd: 0.1,
                max_cost_susd: Some(1.3),
                min_proceeds_susd: None,
                sqrt_price_limit_x96: Some(U160::from(1u8)),
            }],
            planned_at_block: Some(100),
            edge_plan_susd: 1.0,
            l2_gas_units: 1,
            gas_l2_susd: 0.1,
            gas_total_susd: 0.2,
            profit_buffer_susd: 0.25,
            slippage_budget_susd: 0.4,
            guaranteed_profit_floor_susd: 0.25,
        };
        let bounds = derive_batch_quote_bounds(&plan, 100, 2)
            .expect("bounds derivation should succeed")
            .expect("buy bounds expected");
        assert_eq!(
            bounds,
            BatchQuoteBounds::Buy {
                planned_total_in_susd: 1.2,
                max_total_in_susd: 1.6,
            }
        );
    }

    #[test]
    fn market_context_populates_conservative_quotes_and_price_limits() {
        let slot0_results = full_slot0_results_with_prediction_multiplier(0.35);
        let gas = GasAssumptions {
            l1_fee_per_byte_wei: 1.0e8,
            ..GasAssumptions::default()
        };
        let actions = portfolio::rebalance_with_gas_pricing(
            &HashMap::new(),
            200.0,
            &slot0_results,
            RebalanceMode::Full,
            &gas,
            1e-9,
            3000.0,
        );
        let baseline_plans = build_group_plans_with_default_edges(
            &actions,
            &gas,
            1e-9,
            3000.0,
            BufferConfig::default(),
        )
        .expect("baseline planning should succeed");
        let plans = build_group_plans_with_default_edges_and_market_context(
            &actions,
            &slot0_results,
            ConservativeExecutionConfig {
                quote_latency_blocks: 1,
                adverse_move_bps_per_block: 15,
            },
            &gas,
            1e-9,
            3000.0,
            BufferConfig::default(),
        )
        .expect("planning with market context should succeed");
        let dex_leg = plans
            .iter()
            .flat_map(|plan| plan.legs.iter())
            .find(|leg| leg.kind == LegKind::Buy && leg.market_name.is_some())
            .expect("expected at least one buy dex leg");
        assert!(
            dex_leg.conservative_quote_susd >= dex_leg.planned_quote_susd,
            "buy legs should widen conservatively"
        );
        assert!(
            dex_leg.sqrt_price_limit_x96.is_some(),
            "market-aware planning should populate price limits"
        );

        let baseline_plan = baseline_plans
            .iter()
            .find(|plan| !plan.legs.is_empty())
            .expect("expected at least one baseline dex plan");
        let repriced_plan = plans
            .iter()
            .find(|plan| plan.action_indices == baseline_plan.action_indices)
            .expect("repriced plan should preserve the same leading execution group");
        let expected_delta: f64 = baseline_plan
            .legs
            .iter()
            .map(|leg| conservative_quote_delta(leg.planned_quote_susd, leg.kind, 0.0015))
            .sum();
        let expected_slippage = (baseline_plan.slippage_budget_susd - expected_delta).max(0.0);
        let tol = 1e-9 * (1.0 + expected_slippage.abs());
        assert!(
            (repriced_plan.slippage_budget_susd - expected_slippage).abs() <= tol,
            "conservative widening should be debited from residual slippage budget"
        );
        assert!(
            repriced_plan.guaranteed_profit_floor_susd > 0.0,
            "repriced plans should preserve a positive profit floor or be dropped"
        );
    }

    #[test]
    fn buy_price_limit_tracks_terminal_price_not_spot() {
        let slot0_results = full_slot0_results_with_prediction_multiplier(0.35);
        let gas = GasAssumptions {
            l1_fee_per_byte_wei: 1.0e8,
            ..GasAssumptions::default()
        };
        let actions = portfolio::rebalance_with_gas_pricing(
            &HashMap::new(),
            200.0,
            &slot0_results,
            RebalanceMode::Full,
            &gas,
            1e-9,
            3000.0,
        );
        let plans = build_group_plans_with_default_edges_and_market_context(
            &actions,
            &slot0_results,
            ConservativeExecutionConfig {
                quote_latency_blocks: 1,
                adverse_move_bps_per_block: 15,
            },
            &gas,
            1e-9,
            3000.0,
            BufferConfig::default(),
        )
        .expect("planning with market context should succeed");

        let buy_leg = plans
            .iter()
            .flat_map(|plan| plan.legs.iter())
            .filter(|leg| leg.kind == LegKind::Buy)
            .max_by(|lhs, rhs| {
                let lhs_amount = action_amount_for_leg(&actions, lhs).unwrap_or(0.0);
                let rhs_amount = action_amount_for_leg(&actions, rhs).unwrap_or(0.0);
                lhs_amount
                    .partial_cmp(&rhs_amount)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("expected at least one buy leg");
        let market_name = buy_leg
            .market_name
            .expect("buy leg should have market context");
        let (slot0, market) = slot0_results
            .iter()
            .find(|(_, market)| market.name == market_name)
            .expect("buy leg market should have slot0 data");
        let state = market_price_state(slot0, *market).expect("market state should be available");
        let amount =
            action_amount_for_leg(&actions, buy_leg).expect("buy leg should map to action");
        let lambda = state.current_price.sqrt() / state.liquidity_raw;
        let denom = 1.0 - amount * lambda;
        let terminal_price = if denom <= EPSILON {
            state.buy_limit_price
        } else {
            (state.current_price / (denom * denom))
                .max(state.current_price)
                .min(state.buy_limit_price)
        };
        let expected_limit_price = (terminal_price * 1.0015).min(state.buy_limit_price);
        let limit_price = u256_to_f64(
            sqrt_price_x96_to_price_outcome(
                U256::from(
                    buy_leg
                        .sqrt_price_limit_x96
                        .expect("buy leg should have limit"),
                ),
                state.is_token1_outcome,
            )
            .expect("limit should convert back into an outcome price"),
        );
        let tol = 1e-9 * (1.0 + expected_limit_price.abs() + limit_price.abs());
        assert!(
            (limit_price - expected_limit_price).abs() <= tol,
            "buy-side limit should be based on the planned terminal price, not spot"
        );
    }
}
