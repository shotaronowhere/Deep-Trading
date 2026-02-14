use std::collections::HashMap;
use std::fmt;

use crate::pools::normalize_market_name;
use crate::portfolio::Action;

use super::gas::{
    GasAssumptions, estimate_group_l2_gas_units, estimate_l2_gas_susd, estimate_total_gas_susd,
    hydrate_cached_optimism_l1_fee_per_byte, resolve_l1_fee_per_byte_wei,
};
use super::grouping::{ActionGroup, GroupingError, group_actions};
use super::{
    BatchQuoteBounds, ExecutionGroupPlan, ExecutionLegPlan, GroupKind, LegKind, is_plan_stale,
};

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchBoundsError {
    InvalidSlippageBudget,
    MixedLegDirections,
    StalePlan,
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
        }
    }
}

impl std::error::Error for GroupPlanningError {}

impl fmt::Display for BatchBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSlippageBudget => write!(f, "invalid slippage budget on group plan"),
            Self::MixedLegDirections => {
                write!(f, "mixed buy/sell legs cannot map to one batch bound")
            }
            Self::StalePlan => write!(f, "plan is stale or missing reference block"),
        }
    }
}

impl std::error::Error for BatchBoundsError {}

impl From<GroupingError> for GroupPlanningError {
    fn from(value: GroupingError) -> Self {
        Self::Grouping(value)
    }
}

#[derive(Debug, Clone, Copy)]
struct GroupLegSummary {
    action_index: usize,
    market_name: Option<&'static str>,
    kind: LegKind,
    planned_quote_susd: f64,
}

#[derive(Debug, Clone, Copy)]
struct GroupCashflowSummary {
    planned_cost_susd: f64,
    planned_proceeds_susd: f64,
    buy_legs: usize,
    sell_legs: usize,
}

pub fn planned_cashflow_edge_susd(actions: &[Action], group: &ActionGroup) -> f64 {
    let (summary, _) = summarize_group(actions, group, false);
    summary.planned_proceeds_susd - summary.planned_cost_susd
}

pub fn planned_edge_from_prediction_map_susd(
    actions: &[Action],
    group: &ActionGroup,
    prediction_by_market: &HashMap<String, f64>,
) -> Option<f64> {
    match group.kind {
        GroupKind::DirectBuy => {
            let action_index = *group.action_indices.first()?;
            match actions.get(action_index)? {
                Action::Buy {
                    market_name,
                    amount,
                    cost,
                } => {
                    let key = normalize_market_name(market_name);
                    let prediction = match prediction_by_market.get(&key).copied() {
                        Some(prediction) => prediction,
                        None => {
                            eprintln!(
                                "info: missing prediction for direct-buy market '{}' (normalized '{}'); using prediction=0.0 and relying on edge gate",
                                market_name, key
                            );
                            0.0
                        }
                    };
                    debug_assert!(
                        amount.is_finite() && *amount >= 0.0,
                        "invalid buy amount at action index {}: {}",
                        action_index,
                        amount
                    );
                    debug_assert!(
                        cost.is_finite() && *cost >= 0.0,
                        "invalid buy cost at action index {}: {}",
                        action_index,
                        cost
                    );
                    debug_assert!(
                        prediction.is_finite(),
                        "invalid prediction for direct-buy market '{}' (normalized '{}'): {}",
                        market_name,
                        key,
                        prediction
                    );
                    Some(amount.max(0.0) * prediction - cost.max(0.0))
                }
                _ => None,
            }
        }
        _ => Some(planned_cashflow_edge_susd(actions, group)),
    }
}

pub fn build_group_plan(
    actions: &[Action],
    group: &ActionGroup,
    edge_plan_susd: f64,
    gas_assumptions: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Option<ExecutionGroupPlan> {
    if !edge_plan_susd.is_finite() || edge_plan_susd <= 0.0 {
        return None;
    }

    let (cashflow, legs) = summarize_group(actions, group, true);
    if legs.is_empty() && group.kind != GroupKind::DirectMerge {
        return None;
    }

    let l2_gas_units = estimate_group_l2_gas_units(
        gas_assumptions,
        group.kind,
        cashflow.buy_legs,
        cashflow.sell_legs,
    );
    let gas_l2_susd = estimate_l2_gas_susd(l2_gas_units, gas_price_eth, eth_usd_assumed);
    if !gas_l2_susd.is_finite() {
        return None;
    }
    let gas_total_susd = estimate_total_gas_susd(
        gas_assumptions,
        group.kind,
        cashflow.buy_legs,
        cashflow.sell_legs,
        l2_gas_units,
        gas_price_eth,
        eth_usd_assumed,
    );
    if !gas_total_susd.is_finite() {
        return None;
    }

    let profit_buffer_susd = buffer
        .buffer_min_susd
        .max(buffer.buffer_frac.max(0.0) * edge_plan_susd);
    let post_buffer_margin_susd = edge_plan_susd - gas_total_susd - profit_buffer_susd;
    if !post_buffer_margin_susd.is_finite() || post_buffer_margin_susd <= 0.0 {
        return None;
    }
    let slippage_budget_susd = if group.kind == GroupKind::DirectMerge {
        0.0
    } else {
        post_buffer_margin_susd
    };

    let leg_plans = if legs.is_empty() {
        Vec::new()
    } else {
        let total_notional: f64 = legs.iter().map(|l| l.planned_quote_susd).sum();
        if total_notional <= 0.0 || !total_notional.is_finite() {
            return None;
        }

        let mut leg_plans = Vec::with_capacity(legs.len());
        for leg in legs {
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
                adverse_notional_susd: leg.planned_quote_susd,
                allocated_slippage_susd,
                max_cost_susd,
                min_proceeds_susd,
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

    Some(ExecutionGroupPlan {
        kind: group.kind,
        action_indices: group.action_indices.clone(),
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

/// Derive aggregate quote-token bounds for BatchRouter execution.
///
/// Returns:
/// - `Sell { min_total_out_susd }` for sell-only leg groups
/// - `Buy { max_total_in_susd }` for buy-only leg groups
/// - `None` for groups with no DEX legs (e.g. DirectMerge)
pub fn derive_batch_quote_bounds(
    plan: &ExecutionGroupPlan,
    current_block: u64,
    max_stale_blocks: u64,
) -> Result<Option<BatchQuoteBounds>, BatchBoundsError> {
    if is_plan_stale(plan, current_block, max_stale_blocks) {
        return Err(BatchBoundsError::StalePlan);
    }
    derive_batch_quote_bounds_unchecked(plan)
}

/// Derive aggregate quote-token bounds without applying staleness checks.
/// Prefer `derive_batch_quote_bounds` on the execution path.
pub fn derive_batch_quote_bounds_unchecked(
    plan: &ExecutionGroupPlan,
) -> Result<Option<BatchQuoteBounds>, BatchBoundsError> {
    if !plan.slippage_budget_susd.is_finite() || plan.slippage_budget_susd < 0.0 {
        return Err(BatchBoundsError::InvalidSlippageBudget);
    }

    let mut planned_total_in_susd = 0.0_f64;
    let mut planned_total_out_susd = 0.0_f64;
    for leg in &plan.legs {
        match leg.kind {
            LegKind::Buy => planned_total_in_susd += leg.planned_quote_susd.max(0.0),
            LegKind::Sell => planned_total_out_susd += leg.planned_quote_susd.max(0.0),
        }
    }

    if planned_total_in_susd > 0.0 && planned_total_out_susd > 0.0 {
        return Err(BatchBoundsError::MixedLegDirections);
    }

    if planned_total_out_susd > 0.0 {
        let min_total_out_susd = (planned_total_out_susd - plan.slippage_budget_susd).max(0.0);
        return Ok(Some(BatchQuoteBounds::Sell {
            planned_total_out_susd,
            min_total_out_susd,
        }));
    }

    if planned_total_in_susd > 0.0 {
        let max_total_in_susd = planned_total_in_susd + plan.slippage_budget_susd;
        return Ok(Some(BatchQuoteBounds::Buy {
            planned_total_in_susd,
            max_total_in_susd,
        }));
    }

    Ok(None)
}

pub fn stamp_plan_with_block(plan: &mut ExecutionGroupPlan, planned_at_block: u64) {
    plan.planned_at_block = Some(planned_at_block);
}

pub fn stamp_plans_with_block(plans: &mut [ExecutionGroupPlan], planned_at_block: u64) {
    for plan in plans {
        stamp_plan_with_block(plan, planned_at_block);
    }
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
    F: FnMut(usize, &ActionGroup, &[Action]) -> Option<f64>,
{
    // DirectBuy groups have no route-local cashflow edge and must receive
    // externally supplied expected-value edge from the caller.
    let groups = group_actions(actions)?;
    let mut effective_gas_assumptions = *gas_assumptions;
    if let Some(l1_fee_per_byte_wei) = resolve_l1_fee_per_byte_wei(gas_assumptions) {
        effective_gas_assumptions.l1_fee_per_byte_wei = l1_fee_per_byte_wei;
    }
    if !effective_gas_assumptions.l1_fee_per_byte_wei.is_finite()
        || effective_gas_assumptions.l1_fee_per_byte_wei <= 0.0
    {
        eprintln!(
            "warning: no usable Optimism L1 fee-per-byte estimate is available (value: {}); strict planning will fail closed and skip all groups. Configure a conservative positive GasAssumptions.l1_fee_per_byte_wei fallback to preserve liveness during oracle outages",
            effective_gas_assumptions.l1_fee_per_byte_wei
        );
    }
    let mut plans = Vec::new();

    for (group_index, group) in groups.iter().enumerate() {
        let first_action_index =
            group
                .action_indices
                .first()
                .copied()
                .ok_or(GroupPlanningError::MissingEdge {
                    group_index,
                    group_kind: group.kind,
                    first_action_index: 0,
                })?;

        let edge =
            edge_provider(group_index, group, actions).ok_or(GroupPlanningError::MissingEdge {
                group_index,
                group_kind: group.kind,
                first_action_index,
            })?;

        if let Some(plan) = build_group_plan(
            actions,
            group,
            edge,
            &effective_gas_assumptions,
            gas_price_eth,
            eth_usd_assumed,
            buffer,
        ) {
            plans.push(plan);
        }
    }

    plans.sort_by(|a, b| {
        b.guaranteed_profit_floor_susd
            .total_cmp(&a.guaranteed_profit_floor_susd)
            .then_with(|| b.edge_plan_susd.total_cmp(&a.edge_plan_susd))
    });

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
        |_, group, actions| match group.kind {
            GroupKind::DirectBuy => None,
            _ => Some(planned_cashflow_edge_susd(actions, group)),
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
        |_, group, actions| {
            planned_edge_from_prediction_map_susd(actions, group, prediction_by_market)
        },
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

/// Hydration-first planning helper: refresh Optimism L1 fee-per-byte (cache-aware)
/// before building plans with default edges.
///
/// Panics if hydration fails because execution cannot proceed safely without
/// reliable RPC-derived fee data.
pub async fn build_group_plans_with_default_edges_and_l1_hydration(
    actions: &[Action],
    gas_assumptions: &GasAssumptions,
    rpc_url: &str,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
    buffer: BufferConfig,
) -> Result<Vec<ExecutionGroupPlan>, GroupPlanningError> {
    let mut hydrated = *gas_assumptions;
    hydrate_cached_optimism_l1_fee_per_byte(&mut hydrated, rpc_url)
        .await
        .unwrap_or_else(|err| {
            panic!(
                "failed to hydrate Optimism L1 fee-per-byte from RPC; cannot continue safe execution planning: {err}"
            )
        });

    build_group_plans_with_default_edges(actions, &hydrated, gas_price_eth, eth_usd_assumed, buffer)
}

fn summarize_group(
    actions: &[Action],
    group: &ActionGroup,
    include_legs: bool,
) -> (GroupCashflowSummary, Vec<GroupLegSummary>) {
    let mut planned_cost_susd = 0.0_f64;
    let mut planned_proceeds_susd = 0.0_f64;
    let mut buy_legs = 0usize;
    let mut sell_legs = 0usize;
    let mut legs = Vec::new();

    for idx in &group.action_indices {
        match actions[*idx] {
            Action::Buy {
                market_name,
                amount,
                cost,
            } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid buy amount at action index {}: {}",
                    idx,
                    amount
                );
                debug_assert!(
                    cost.is_finite() && cost >= 0.0,
                    "invalid buy cost at action index {}: {}",
                    idx,
                    cost
                );
                planned_cost_susd += cost.max(0.0);
                buy_legs += 1;
                if include_legs {
                    legs.push(GroupLegSummary {
                        action_index: *idx,
                        market_name: Some(market_name),
                        kind: LegKind::Buy,
                        planned_quote_susd: cost.max(0.0),
                    });
                }
            }
            Action::Sell {
                market_name,
                amount,
                proceeds,
                ..
            } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid sell amount at action index {}: {}",
                    idx,
                    amount
                );
                debug_assert!(
                    proceeds.is_finite() && proceeds >= 0.0,
                    "invalid sell proceeds at action index {}: {}",
                    idx,
                    proceeds
                );
                planned_proceeds_susd += proceeds.max(0.0);
                sell_legs += 1;
                if include_legs {
                    legs.push(GroupLegSummary {
                        action_index: *idx,
                        market_name: Some(market_name),
                        kind: LegKind::Sell,
                        planned_quote_susd: proceeds.max(0.0),
                    });
                }
            }
            Action::Mint { amount, .. } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid mint amount at action index {}: {}",
                    idx,
                    amount
                );
                planned_cost_susd += amount.max(0.0);
            }
            Action::Merge { amount, .. } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid merge amount at action index {}: {}",
                    idx,
                    amount
                );
                planned_proceeds_susd += amount.max(0.0);
            }
            Action::FlashLoan { .. } | Action::RepayFlashLoan { .. } => {}
        }
    }

    (
        GroupCashflowSummary {
            planned_cost_susd,
            planned_proceeds_susd,
            buy_legs,
            sell_legs,
        },
        legs,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::execution::GroupKind;
    use crate::execution::grouping::{GroupingError, group_actions};

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

    #[test]
    fn skips_group_when_edge_below_gas_plus_buffer() {
        let actions = vec![sell("x", 1.0, 1.0)];
        let groups = group_actions(&actions).expect("grouping should succeed");
        let plan = build_group_plan(
            &actions,
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
    fn leg_allocations_sum_to_group_budget() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "target",
            },
            sell("a", 10.0, 7.0),
            sell("b", 10.0, 7.0),
            Action::RepayFlashLoan { amount: 10.0 },
        ];

        let groups = group_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::MintSell);

        let plan = build_group_plan(
            &actions,
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
    fn groups_are_sorted_by_guaranteed_profit_floor() {
        let actions = vec![
            sell("a", 1.0, 10.0),
            sell("b", 1.0, 2.0),
            buy("c", 1.0, 1.0),
        ];
        let plans = build_group_plans_with_edge_provider(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
            |_, group, actions| Some(planned_cashflow_edge_susd(actions, group).max(0.1)),
        )
        .expect("planning should succeed");

        assert!(!plans.is_empty());
        for w in plans.windows(2) {
            assert!(
                w[0].guaranteed_profit_floor_susd >= w[1].guaranteed_profit_floor_susd,
                "plans should be sorted by guaranteed profit floor"
            );
        }
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
            |_, group, actions| match group.kind {
                GroupKind::DirectBuy => None,
                _ => Some(planned_cashflow_edge_susd(actions, group)),
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
    fn planning_fails_closed_on_unsupported_flash_arb_shape() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            buy("a", 1.0, 0.8),
            sell("b", 1.0, 0.9),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = build_group_plans_from_cashflow(
            &actions,
            &test_gas_assumptions(),
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .expect_err("unsupported flash-arb shape should fail closed");
        assert!(matches!(
            err,
            GroupPlanningError::Grouping(GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3,
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
            GroupPlanningError::Grouping(GroupingError::UnexpectedActionOutsideGroup { index: 0 })
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
    #[should_panic(
        expected = "failed to hydrate Optimism L1 fee-per-byte from RPC; cannot continue safe execution planning"
    )]
    async fn hydration_entrypoint_panics_when_hydration_fails() {
        let actions = vec![sell("x", 1.0, 2.0)];
        let _ = build_group_plans_with_default_edges_and_l1_hydration(
            &actions,
            &GasAssumptions::default(),
            "",
            1e-10,
            3000.0,
            BufferConfig::default(),
        )
        .await;
    }

    #[test]
    fn cashflow_builder_drops_negative_edge_mint_sell_group() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "target",
            },
            sell("a", 10.0, 2.0),
            sell("b", 10.0, 2.0),
            Action::RepayFlashLoan { amount: 10.0 },
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
        let groups = group_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::DirectMerge);

        let edge = planned_cashflow_edge_susd(&actions, &groups[0]);
        let plan = build_group_plan(
            &actions,
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
        let groups = group_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::DirectMerge);

        let edge = planned_cashflow_edge_susd(&actions, &groups[0]);
        let plan = build_group_plan(
            &actions,
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
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "target",
            },
            sell("a", 10.0, 7.0),
            sell("b", 10.0, 7.0),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let groups = group_actions(&actions).expect("grouping should succeed");
        let plan = build_group_plan(
            &actions,
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
        let groups = group_actions(&actions).expect("grouping should succeed");
        let plan = build_group_plan(
            &actions,
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
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            buy("a", 1.0, 0.4),
            buy("b", 1.0, 0.5),
            merge(1.0),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let groups = group_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups[0].kind, GroupKind::BuyMerge);

        let plan = build_group_plan(
            &actions,
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
        let groups = group_actions(&actions).expect("grouping should succeed");
        let edge = planned_cashflow_edge_susd(&actions, &groups[0]);
        let plan = build_group_plan(
            &actions,
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
            legs: vec![
                ExecutionLegPlan {
                    action_index: 0,
                    market_name: Some("a"),
                    kind: LegKind::Buy,
                    planned_quote_susd: 1.0,
                    adverse_notional_susd: 1.0,
                    allocated_slippage_susd: 0.1,
                    max_cost_susd: Some(1.1),
                    min_proceeds_susd: None,
                },
                ExecutionLegPlan {
                    action_index: 1,
                    market_name: Some("b"),
                    kind: LegKind::Sell,
                    planned_quote_susd: 1.5,
                    adverse_notional_susd: 1.5,
                    allocated_slippage_susd: 0.1,
                    max_cost_susd: None,
                    min_proceeds_susd: Some(1.4),
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
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some("a"),
                kind: LegKind::Buy,
                planned_quote_susd: 1.0,
                adverse_notional_susd: 1.0,
                allocated_slippage_susd: 0.1,
                max_cost_susd: Some(1.1),
                min_proceeds_susd: None,
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
}
