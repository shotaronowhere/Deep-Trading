use std::fmt;

use super::{BatchQuoteBounds, ExecutionGroupPlan, LegKind, is_plan_stale};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchBoundsError {
    InvalidSlippageBudget,
    MixedLegDirections,
    StalePlan,
}

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
