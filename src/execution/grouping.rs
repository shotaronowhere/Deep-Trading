use std::{collections::HashSet, fmt};

use crate::portfolio::Action;

use super::{GroupKind, LegKind};

#[derive(Debug, Clone)]
pub struct ActionGroup {
    pub kind: GroupKind,
    pub step_kind: ProfitabilityStepKind,
    pub action_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct GroupLeg {
    pub action_index: usize,
    pub market_name: Option<&'static str>,
    pub kind: LegKind,
    pub planned_quote_susd: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct DirectBuyInput {
    pub action_index: usize,
    pub market_name: &'static str,
    pub amount: f64,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionGroup {
    pub kind: GroupKind,
    pub action_indices: Vec<usize>,
    pub legs: Vec<GroupLeg>,
    pub planned_cost_susd: f64,
    pub planned_proceeds_susd: f64,
    pub buy_legs: usize,
    pub sell_legs: usize,
    pub direct_buy: Option<DirectBuyInput>,
}

impl ExecutionGroup {
    pub fn first_action_index(&self) -> Option<usize> {
        self.action_indices.first().copied()
    }
}

pub const ZERO_FEE_REL_TOL: f64 = 1e-8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfitabilityStepKind {
    ArbMintSell,
    ArbBuyMerge,
    PureDirectBuy,
    PureDirectSell,
    PureDirectMerge,
    MixedDirectBuyMintSell,
    MixedDirectSellBuyMerge,
}

impl ProfitabilityStepKind {
    pub fn display_group_kind(self) -> GroupKind {
        match self {
            Self::ArbMintSell | Self::MixedDirectBuyMintSell => GroupKind::MintSell,
            Self::ArbBuyMerge | Self::MixedDirectSellBuyMerge => GroupKind::BuyMerge,
            Self::PureDirectBuy => GroupKind::DirectBuy,
            Self::PureDirectSell => GroupKind::DirectSell,
            Self::PureDirectMerge => GroupKind::DirectMerge,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProfitabilityStepGroup {
    pub kind: ProfitabilityStepKind,
    pub action_indices: Vec<usize>,
    pub strict_groups: Vec<ExecutionGroup>,
}

impl ProfitabilityStepGroup {
    pub fn first_action_index(&self) -> Option<usize> {
        self.action_indices.first().copied()
    }
}

fn classify_step_kind(groups: &[ExecutionGroup]) -> ProfitabilityStepKind {
    debug_assert!(!groups.is_empty(), "step group cannot be empty");
    let has = |kind: GroupKind| groups.iter().any(|g| g.kind == kind);
    let all = |kind: GroupKind| groups.iter().all(|g| g.kind == kind);

    if all(GroupKind::MintSell) {
        return ProfitabilityStepKind::ArbMintSell;
    }
    if all(GroupKind::BuyMerge) {
        return ProfitabilityStepKind::ArbBuyMerge;
    }
    if all(GroupKind::DirectBuy) {
        return ProfitabilityStepKind::PureDirectBuy;
    }
    if all(GroupKind::DirectSell) {
        return ProfitabilityStepKind::PureDirectSell;
    }
    if all(GroupKind::DirectMerge) {
        return ProfitabilityStepKind::PureDirectMerge;
    }
    if has(GroupKind::MintSell)
        && has(GroupKind::DirectBuy)
        && groups
            .iter()
            .all(|g| matches!(g.kind, GroupKind::MintSell | GroupKind::DirectBuy))
    {
        return ProfitabilityStepKind::MixedDirectBuyMintSell;
    }
    if has(GroupKind::BuyMerge)
        && has(GroupKind::DirectSell)
        && groups
            .iter()
            .all(|g| matches!(g.kind, GroupKind::BuyMerge | GroupKind::DirectSell))
    {
        return ProfitabilityStepKind::MixedDirectSellBuyMerge;
    }

    unreachable!("unsupported profitability-step kind composition")
}

fn step_group_from_route_groups(route_groups: &[ExecutionGroup]) -> ProfitabilityStepGroup {
    let mut action_indices = Vec::new();
    for group in route_groups {
        action_indices.extend_from_slice(&group.action_indices);
    }
    ProfitabilityStepGroup {
        kind: classify_step_kind(route_groups),
        action_indices,
        strict_groups: route_groups.to_vec(),
    }
}

fn merge_profitability_step_groups(
    actions: &[Action],
    route_groups: &[ExecutionGroup],
) -> Vec<ProfitabilityStepGroup> {
    let mut merged = Vec::new();
    let mut i = 0usize;

    fn group_anchor_market(actions: &[Action], group: &ExecutionGroup) -> Option<&'static str> {
        match group.kind {
            GroupKind::DirectBuy => {
                group
                    .action_indices
                    .iter()
                    .find_map(|&idx| match &actions[idx] {
                        Action::Buy { market_name, .. } => Some(*market_name),
                        _ => None,
                    })
            }
            GroupKind::DirectSell => {
                group
                    .action_indices
                    .iter()
                    .find_map(|&idx| match &actions[idx] {
                        Action::Sell { market_name, .. } => Some(*market_name),
                        _ => None,
                    })
            }
            GroupKind::MintSell => {
                group
                    .action_indices
                    .iter()
                    .find_map(|&idx| match &actions[idx] {
                        Action::Mint { target_market, .. } => Some(*target_market),
                        _ => None,
                    })
            }
            GroupKind::BuyMerge => {
                group
                    .action_indices
                    .iter()
                    .find_map(|&idx| match &actions[idx] {
                        Action::Merge { source_market, .. } => Some(*source_market),
                        _ => None,
                    })
            }
            GroupKind::DirectMerge => None,
        }
    }

    fn merge_same_kind_unique_market_block(
        actions: &[Action],
        route_groups: &[ExecutionGroup],
        start: usize,
        kind: GroupKind,
    ) -> Option<(ProfitabilityStepGroup, usize)> {
        if route_groups[start].kind != kind {
            return None;
        }

        let mut seen: HashSet<&'static str> = HashSet::new();
        let mut end = start;
        while end < route_groups.len() && route_groups[end].kind == kind {
            let Some(anchor) = group_anchor_market(actions, &route_groups[end]) else {
                break;
            };
            if seen.contains(anchor) {
                break;
            }
            seen.insert(anchor);
            end += 1;
        }

        if end > start + 1 {
            Some((step_group_from_route_groups(&route_groups[start..end]), end))
        } else {
            None
        }
    }

    fn block_is_route_coupled(
        actions: &[Action],
        block: &[ExecutionGroup],
        a: GroupKind,
        b: GroupKind,
    ) -> bool {
        let mut route_markets: HashSet<&'static str> = HashSet::new();
        let mut direct_markets: HashSet<&'static str> = HashSet::new();

        for group in block {
            for &idx in &group.action_indices {
                match (a, b, &actions[idx]) {
                    (
                        GroupKind::MintSell,
                        GroupKind::DirectBuy,
                        Action::Mint { target_market, .. },
                    ) if group.kind == GroupKind::MintSell => {
                        route_markets.insert(*target_market);
                    }
                    (
                        GroupKind::MintSell,
                        GroupKind::DirectBuy,
                        Action::Buy { market_name, .. },
                    ) if group.kind == GroupKind::DirectBuy => {
                        direct_markets.insert(*market_name);
                    }
                    (
                        GroupKind::BuyMerge,
                        GroupKind::DirectSell,
                        Action::Merge { source_market, .. },
                    ) if group.kind == GroupKind::BuyMerge => {
                        route_markets.insert(*source_market);
                    }
                    (
                        GroupKind::BuyMerge,
                        GroupKind::DirectSell,
                        Action::Sell { market_name, .. },
                    ) if group.kind == GroupKind::DirectSell => {
                        direct_markets.insert(*market_name);
                    }
                    _ => {}
                }
            }
        }

        !route_markets.is_empty()
            && !direct_markets.is_empty()
            && route_markets
                .iter()
                .any(|market| direct_markets.contains(market))
    }

    fn merge_two_kind_block(
        actions: &[Action],
        route_groups: &[ExecutionGroup],
        start: usize,
        a: GroupKind,
        b: GroupKind,
    ) -> Option<(ProfitabilityStepGroup, usize)> {
        let first_kind = route_groups[start].kind;
        if first_kind != a && first_kind != b {
            return None;
        }

        let mut saw_a = first_kind == a;
        let mut saw_b = first_kind == b;
        let mut current_kind = first_kind;
        let mut transitions = 0usize;
        let mut end = start + 1;

        while end < route_groups.len() {
            let kind = route_groups[end].kind;
            if kind != a && kind != b {
                break;
            }
            if kind != current_kind {
                transitions += 1;
                if transitions > 1 {
                    break;
                }
                current_kind = kind;
            }
            saw_a |= kind == a;
            saw_b |= kind == b;
            end += 1;
        }

        if saw_a && saw_b && block_is_route_coupled(actions, &route_groups[start..end], a, b) {
            Some((step_group_from_route_groups(&route_groups[start..end]), end))
        } else {
            None
        }
    }

    while i < route_groups.len() {
        if let Some((group, next_i)) = merge_two_kind_block(
            actions,
            route_groups,
            i,
            GroupKind::MintSell,
            GroupKind::DirectBuy,
        ) {
            merged.push(group);
            i = next_i;
            continue;
        }

        if let Some((group, next_i)) = merge_two_kind_block(
            actions,
            route_groups,
            i,
            GroupKind::BuyMerge,
            GroupKind::DirectSell,
        ) {
            merged.push(group);
            i = next_i;
            continue;
        }

        if let Some((group, next_i)) =
            merge_same_kind_unique_market_block(actions, route_groups, i, route_groups[i].kind)
        {
            merged.push(group);
            i = next_i;
            continue;
        }

        merged.push(step_group_from_route_groups(&route_groups[i..=i]));
        i += 1;
    }

    merged
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupingError {
    UnexpectedActionOutsideGroup {
        index: usize,
    },
    NestedFlashLoan {
        index: usize,
    },
    MissingRepayFlashLoan {
        flash_loan_index: usize,
    },
    InvalidFlashLoanBracket {
        start_index: usize,
        end_index: usize,
    },
    UnsupportedFlashLoanFee {
        start_index: usize,
        end_index: usize,
    },
}

impl fmt::Display for GroupingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedActionOutsideGroup { index } => {
                write!(
                    f,
                    "unsupported action shape outside execution groups at index {index}"
                )
            }
            Self::NestedFlashLoan { index } => {
                write!(f, "nested flash loan bracket at index {index}")
            }
            Self::MissingRepayFlashLoan { flash_loan_index } => {
                write!(
                    f,
                    "missing repay flash loan for bracket starting at index {flash_loan_index}"
                )
            }
            Self::InvalidFlashLoanBracket {
                start_index,
                end_index,
            } => write!(
                f,
                "unsupported flash-loan bracket shape between indices {start_index} and {end_index}"
            ),
            Self::UnsupportedFlashLoanFee {
                start_index,
                end_index,
            } => write!(
                f,
                "flash-loan fee is not supported between indices {start_index} and {end_index}"
            ),
        }
    }
}

impl std::error::Error for GroupingError {}

fn collect_flash_loan_end(
    actions: &[Action],
    flash_loan_index: usize,
) -> Result<usize, GroupingError> {
    let mut i = flash_loan_index + 1;
    while i < actions.len() {
        match &actions[i] {
            Action::FlashLoan { .. } => {
                return Err(GroupingError::NestedFlashLoan { index: i });
            }
            Action::RepayFlashLoan { .. } => {
                return Ok(i);
            }
            _ => i += 1,
        }
    }
    Err(GroupingError::MissingRepayFlashLoan { flash_loan_index })
}

fn typed_group_from_actions(
    actions: &[Action],
    kind: GroupKind,
    action_indices: Vec<usize>,
) -> ExecutionGroup {
    let mut legs = Vec::new();
    let mut planned_cost_susd = 0.0_f64;
    let mut planned_proceeds_susd = 0.0_f64;
    let mut buy_legs = 0usize;
    let mut sell_legs = 0usize;
    let mut direct_buy = None;

    for action_index in &action_indices {
        match actions[*action_index] {
            Action::Buy {
                market_name,
                amount,
                cost,
            } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid buy amount: {amount}"
                );
                debug_assert!(cost.is_finite() && cost >= 0.0, "invalid buy cost: {cost}");
                let quote = cost.max(0.0);
                planned_cost_susd += quote;
                buy_legs += 1;
                legs.push(GroupLeg {
                    action_index: *action_index,
                    market_name: Some(market_name),
                    kind: LegKind::Buy,
                    planned_quote_susd: quote,
                });
                if kind == GroupKind::DirectBuy {
                    direct_buy = Some(DirectBuyInput {
                        action_index: *action_index,
                        market_name,
                        amount,
                        cost,
                    });
                }
            }
            Action::Sell {
                market_name,
                amount,
                proceeds,
            } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid sell amount: {amount}"
                );
                debug_assert!(
                    proceeds.is_finite() && proceeds >= 0.0,
                    "invalid sell proceeds: {proceeds}"
                );
                let quote = proceeds.max(0.0);
                planned_proceeds_susd += quote;
                sell_legs += 1;
                legs.push(GroupLeg {
                    action_index: *action_index,
                    market_name: Some(market_name),
                    kind: LegKind::Sell,
                    planned_quote_susd: quote,
                });
            }
            Action::Mint { amount, .. } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid mint amount: {amount}"
                );
                planned_cost_susd += amount.max(0.0);
            }
            Action::Merge { amount, .. } => {
                debug_assert!(
                    amount.is_finite() && amount >= 0.0,
                    "invalid merge amount: {amount}"
                );
                planned_proceeds_susd += amount.max(0.0);
            }
            Action::FlashLoan { .. } | Action::RepayFlashLoan { .. } => {}
        }
    }

    ExecutionGroup {
        kind,
        action_indices,
        legs,
        planned_cost_susd,
        planned_proceeds_susd,
        buy_legs,
        sell_legs,
        direct_buy,
    }
}

/// Groups optimizer actions into execution units and converts each unit into a typed IR
/// consumed by execution planning.
///
/// Supported shapes:
/// - `DirectBuy`, `DirectSell`, `DirectMerge`
/// - `MintSell` flash bracket: `FlashLoan -> Mint -> Sell+ -> RepayFlashLoan`
/// - `BuyMerge` flash bracket: `FlashLoan -> Buy+ -> Merge -> RepayFlashLoan`
///
/// Any unsupported shape returns an error (fail closed) so callers can skip submission.
pub fn group_execution_actions(actions: &[Action]) -> Result<Vec<ExecutionGroup>, GroupingError> {
    let mut groups = Vec::new();
    let mut i = 0usize;

    while i < actions.len() {
        match &actions[i] {
            Action::Buy { .. } => {
                groups.push(typed_group_from_actions(
                    actions,
                    GroupKind::DirectBuy,
                    vec![i],
                ));
                i += 1;
            }
            Action::Sell { .. } => {
                groups.push(typed_group_from_actions(
                    actions,
                    GroupKind::DirectSell,
                    vec![i],
                ));
                i += 1;
            }
            Action::Merge { .. } => {
                groups.push(typed_group_from_actions(
                    actions,
                    GroupKind::DirectMerge,
                    vec![i],
                ));
                i += 1;
            }
            Action::FlashLoan { .. } => {
                let start = i;
                let end_index = collect_flash_loan_end(actions, start)?;
                let kind = classify_flash_loan_bracket(actions, start, end_index)?;
                let action_indices = (start..=end_index).collect();
                groups.push(typed_group_from_actions(actions, kind, action_indices));
                i = end_index + 1;
            }
            Action::Mint { .. } | Action::RepayFlashLoan { .. } => {
                return Err(GroupingError::UnexpectedActionOutsideGroup { index: i });
            }
        }
    }

    Ok(groups)
}

/// Groups actions into profitability-step units.
///
/// Starts from strict route groups and merges adjacent coupled patterns:
/// - contiguous `MintSell` + `DirectBuy` blocks in either order
/// - contiguous `BuyMerge` + `DirectSell` blocks in either order
/// - contiguous same-kind blocks (`DirectBuy`, `DirectSell`, `MintSell`, `BuyMerge`) with
///   unique anchor markets, split when anchor markets repeat
///
/// Mixed blocks are merged only when there is a single kind transition
/// (for example `MintSell... -> DirectBuy...` or `DirectBuy... -> MintSell...`),
/// so repeated mixed phases remain split.
///
/// This preserves strict subgroup boundaries for execution planning while exposing a step-level
/// grouping view for optimizer diagnostics and summaries.
pub fn group_execution_actions_by_profitability_step(
    actions: &[Action],
) -> Result<Vec<ProfitabilityStepGroup>, GroupingError> {
    let route_groups = group_execution_actions(actions)?;
    Ok(merge_profitability_step_groups(actions, &route_groups))
}

/// Maps typed profitability-step groups into lightweight action-index groups.
fn map_step_groups_to_action_groups(groups: Vec<ProfitabilityStepGroup>) -> Vec<ActionGroup> {
    groups
        .into_iter()
        .map(|group| ActionGroup {
            kind: group.kind.display_group_kind(),
            step_kind: group.kind,
            action_indices: group.action_indices,
        })
        .collect()
}

/// Groups actions into profitability-step `ActionGroup`s for diagnostics and step-aware planning.
pub fn group_actions_by_profitability_step(
    actions: &[Action],
) -> Result<Vec<ActionGroup>, GroupingError> {
    let groups = group_execution_actions_by_profitability_step(actions)?;
    Ok(map_step_groups_to_action_groups(groups))
}

/// Legacy alias for profitability-step grouping.
pub fn group_actions(actions: &[Action]) -> Result<Vec<ActionGroup>, GroupingError> {
    group_actions_by_profitability_step(actions)
}

fn classify_flash_loan_bracket(
    actions: &[Action],
    start_index: usize,
    end_index: usize,
) -> Result<GroupKind, GroupingError> {
    let borrowed = match actions.get(start_index) {
        Some(Action::FlashLoan { amount }) => *amount,
        _ => {
            return Err(GroupingError::InvalidFlashLoanBracket {
                start_index,
                end_index,
            });
        }
    };
    let repaid = match actions.get(end_index) {
        Some(Action::RepayFlashLoan { amount }) => *amount,
        _ => {
            return Err(GroupingError::InvalidFlashLoanBracket {
                start_index,
                end_index,
            });
        }
    };
    let tol = ZERO_FEE_REL_TOL * (1.0 + borrowed.abs() + repaid.abs());
    if !borrowed.is_finite()
        || !repaid.is_finite()
        || borrowed < 0.0
        || repaid < 0.0
        || (borrowed - repaid).abs() > tol
    {
        return Err(GroupingError::UnsupportedFlashLoanFee {
            start_index,
            end_index,
        });
    }

    let inner = &actions[(start_index + 1)..end_index];

    if let Some((first, rest)) = inner.split_first()
        && matches!(first, Action::Mint { .. })
        && !rest.is_empty()
        && rest.iter().all(|a| matches!(a, Action::Sell { .. }))
    {
        return Ok(GroupKind::MintSell);
    }

    if let Some((last, rest)) = inner.split_last()
        && matches!(last, Action::Merge { .. })
        && !rest.is_empty()
        && rest.iter().all(|a| matches!(a, Action::Buy { .. }))
    {
        return Ok(GroupKind::BuyMerge);
    }

    Err(GroupingError::InvalidFlashLoanBracket {
        start_index,
        end_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn groups_direct_and_bracket_actions() {
        let actions = vec![
            buy("a", 1.0, 2.0),
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            sell("b", 10.0, 4.0),
            Action::RepayFlashLoan { amount: 10.0 },
            Action::FlashLoan { amount: 5.0 },
            buy("x", 5.0, 1.5),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 5.0,
                source_market: "x",
            },
            Action::RepayFlashLoan { amount: 5.0 },
            sell("z", 1.0, 1.0),
        ];

        let groups = group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(groups.len(), 4);
        assert_eq!(groups[0].kind, GroupKind::DirectBuy);
        assert_eq!(groups[1].kind, GroupKind::MintSell);
        assert_eq!(groups[2].kind, GroupKind::BuyMerge);
        assert_eq!(groups[3].kind, GroupKind::DirectSell);
    }

    #[test]
    fn typed_groups_capture_cashflow_and_dex_legs() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            sell("a", 10.0, 7.0),
            sell("b", 10.0, 6.0),
            Action::RepayFlashLoan { amount: 10.0 },
        ];

        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::MintSell);
        assert_eq!(groups[0].planned_cost_susd, 10.0);
        assert_eq!(groups[0].planned_proceeds_susd, 13.0);
        assert_eq!(groups[0].buy_legs, 0);
        assert_eq!(groups[0].sell_legs, 2);
        assert_eq!(groups[0].legs.len(), 2);
    }

    #[test]
    fn strict_grouping_keeps_mixed_route_stream_split() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            sell("a", 10.0, 7.0),
            Action::RepayFlashLoan { amount: 10.0 },
            buy("x", 1.0, 0.6),
            buy("y", 1.0, 0.5),
        ];

        let groups = group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].kind, GroupKind::MintSell);
        assert_eq!(groups[1].kind, GroupKind::DirectBuy);
        assert_eq!(groups[2].kind, GroupKind::DirectBuy);
    }

    #[test]
    fn profitability_step_grouping_exposes_explicit_mixed_step_kind() {
        let actions = vec![
            buy("x", 1.0, 0.6),
            buy("y", 1.0, 0.5),
            Action::FlashLoan { amount: 6.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 6.0,
                target_market: "x",
            },
            sell("a", 6.0, 4.0),
            Action::RepayFlashLoan { amount: 6.0 },
        ];

        let steps = group_execution_actions_by_profitability_step(&actions)
            .expect("profitability-step grouping should succeed");
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].kind, ProfitabilityStepKind::MixedDirectBuyMintSell);
        assert_eq!(steps[0].strict_groups.len(), 3);
        assert_eq!(steps[0].strict_groups[0].kind, GroupKind::DirectBuy);
        assert_eq!(steps[0].strict_groups[1].kind, GroupKind::DirectBuy);
        assert_eq!(steps[0].strict_groups[2].kind, GroupKind::MintSell);
    }

    #[test]
    fn profitability_step_grouping_merges_mint_sell_with_trailing_direct_buys() {
        let actions = vec![
            Action::FlashLoan { amount: 6.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 6.0,
                target_market: "x",
            },
            sell("a", 6.0, 4.0),
            Action::RepayFlashLoan { amount: 6.0 },
            Action::FlashLoan { amount: 5.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 5.0,
                target_market: "x",
            },
            sell("b", 5.0, 3.0),
            Action::RepayFlashLoan { amount: 5.0 },
            buy("x", 1.0, 0.6),
            buy("y", 1.0, 0.5),
            sell("z", 1.0, 0.4),
        ];

        let groups = group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].kind, GroupKind::MintSell);
        assert_eq!(groups[0].action_indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(groups[1].kind, GroupKind::DirectSell);
        assert_eq!(groups[1].action_indices, vec![10]);
    }

    #[test]
    fn profitability_step_grouping_merges_buy_merge_with_trailing_direct_sell() {
        let actions = vec![
            Action::FlashLoan { amount: 5.0 },
            buy("x", 1.0, 0.7),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "x",
            },
            Action::RepayFlashLoan { amount: 5.0 },
            sell("x", 0.4, 0.3),
        ];

        let strict_groups =
            group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(strict_groups.len(), 2);
        assert_eq!(strict_groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(strict_groups[1].kind, GroupKind::DirectSell);

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 1);
        assert_eq!(step_groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn profitability_step_grouping_merges_leading_direct_buys_with_mint_sell() {
        let actions = vec![
            buy("x", 1.0, 0.6),
            buy("y", 1.0, 0.5),
            Action::FlashLoan { amount: 6.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 6.0,
                target_market: "x",
            },
            sell("a", 6.0, 4.0),
            Action::RepayFlashLoan { amount: 6.0 },
        ];

        let strict_groups =
            group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(strict_groups.len(), 3);
        assert_eq!(strict_groups[0].kind, GroupKind::DirectBuy);
        assert_eq!(strict_groups[1].kind, GroupKind::DirectBuy);
        assert_eq!(strict_groups[2].kind, GroupKind::MintSell);

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 1);
        assert_eq!(step_groups[0].kind, GroupKind::MintSell);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn profitability_step_grouping_merges_leading_direct_sell_with_buy_merge() {
        let actions = vec![
            sell("x", 0.4, 0.3),
            Action::FlashLoan { amount: 5.0 },
            buy("x", 1.0, 0.7),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "x",
            },
            Action::RepayFlashLoan { amount: 5.0 },
        ];

        let strict_groups =
            group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(strict_groups.len(), 2);
        assert_eq!(strict_groups[0].kind, GroupKind::DirectSell);
        assert_eq!(strict_groups[1].kind, GroupKind::BuyMerge);

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 1);
        assert_eq!(step_groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn profitability_step_grouping_keeps_repeated_mixed_mint_buy_phases_separate() {
        let actions = vec![
            Action::FlashLoan { amount: 6.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 6.0,
                target_market: "x",
            },
            sell("a", 6.0, 4.0),
            Action::RepayFlashLoan { amount: 6.0 },
            buy("x", 1.0, 0.6),
            Action::FlashLoan { amount: 5.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 5.0,
                target_market: "y",
            },
            sell("b", 5.0, 3.0),
            Action::RepayFlashLoan { amount: 5.0 },
            buy("y", 1.0, 0.5),
        ];

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 2);
        assert_eq!(step_groups[0].kind, GroupKind::MintSell);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(step_groups[1].kind, GroupKind::MintSell);
        assert_eq!(step_groups[1].action_indices, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn profitability_step_grouping_keeps_repeated_mixed_sell_merge_phases_separate() {
        let actions = vec![
            Action::FlashLoan { amount: 5.0 },
            buy("x", 1.0, 0.7),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "x",
            },
            Action::RepayFlashLoan { amount: 5.0 },
            sell("x", 0.4, 0.3),
            Action::FlashLoan { amount: 4.0 },
            buy("y", 1.0, 0.6),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "y",
            },
            Action::RepayFlashLoan { amount: 4.0 },
            sell("y", 0.3, 0.2),
        ];

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 2);
        assert_eq!(step_groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(step_groups[1].kind, GroupKind::BuyMerge);
        assert_eq!(step_groups[1].action_indices, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn profitability_step_grouping_does_not_merge_unrelated_mint_and_direct_buy_blocks() {
        let actions = vec![
            Action::FlashLoan { amount: 6.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 6.0,
                target_market: "x",
            },
            sell("a", 6.0, 4.0),
            Action::RepayFlashLoan { amount: 6.0 },
            buy("y", 1.0, 0.5),
        ];

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 2);
        assert_eq!(step_groups[0].kind, GroupKind::MintSell);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3]);
        assert_eq!(step_groups[1].kind, GroupKind::DirectBuy);
        assert_eq!(step_groups[1].action_indices, vec![4]);
    }

    #[test]
    fn profitability_step_grouping_does_not_merge_unrelated_buy_merge_and_direct_sell_blocks() {
        let actions = vec![
            Action::FlashLoan { amount: 5.0 },
            buy("x", 1.0, 0.7),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "x",
            },
            Action::RepayFlashLoan { amount: 5.0 },
            sell("y", 0.4, 0.3),
        ];

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 2);
        assert_eq!(step_groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3]);
        assert_eq!(step_groups[1].kind, GroupKind::DirectSell);
        assert_eq!(step_groups[1].action_indices, vec![4]);
    }

    #[test]
    fn profitability_step_grouping_merges_pure_direct_buy_waterfall_steps() {
        // Step-like direct buy stream:
        // step1: [a], step2: [a, b], step3: [a, b, c]
        let actions = vec![
            buy("a", 1.0, 0.1),
            buy("a", 1.0, 0.1),
            buy("b", 1.0, 0.1),
            buy("a", 1.0, 0.1),
            buy("b", 1.0, 0.1),
            buy("c", 1.0, 0.1),
        ];

        let strict_groups =
            group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(strict_groups.len(), 6);
        assert!(strict_groups.iter().all(|g| g.kind == GroupKind::DirectBuy));

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 3);
        assert!(step_groups.iter().all(|g| g.kind == GroupKind::DirectBuy));
        assert_eq!(step_groups[0].action_indices, vec![0]);
        assert_eq!(step_groups[1].action_indices, vec![1, 2]);
        assert_eq!(step_groups[2].action_indices, vec![3, 4, 5]);

        let typed_steps = group_execution_actions_by_profitability_step(&actions)
            .expect("typed profitability-step grouping should succeed");
        assert_eq!(typed_steps.len(), 3);
        assert!(
            typed_steps
                .iter()
                .all(|step| step.kind == ProfitabilityStepKind::PureDirectBuy)
        );
        assert_eq!(typed_steps[0].strict_groups.len(), 1);
        assert_eq!(typed_steps[1].strict_groups.len(), 2);
        assert_eq!(typed_steps[2].strict_groups.len(), 3);
    }

    #[test]
    fn profitability_step_grouping_merges_pure_mint_sell_waterfall_steps() {
        // Step-like mint stream:
        // step1 targets [a], step2 targets [a, b]
        let actions = vec![
            Action::FlashLoan { amount: 2.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 2.0,
                target_market: "a",
            },
            sell("x", 2.0, 0.9),
            Action::RepayFlashLoan { amount: 2.0 },
            Action::FlashLoan { amount: 2.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 2.0,
                target_market: "a",
            },
            sell("x", 2.0, 0.9),
            Action::RepayFlashLoan { amount: 2.0 },
            Action::FlashLoan { amount: 2.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 2.0,
                target_market: "b",
            },
            sell("x", 2.0, 0.9),
            Action::RepayFlashLoan { amount: 2.0 },
        ];

        let strict_groups =
            group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(strict_groups.len(), 3);
        assert!(strict_groups.iter().all(|g| g.kind == GroupKind::MintSell));

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 2);
        assert!(step_groups.iter().all(|g| g.kind == GroupKind::MintSell));
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3]);
        assert_eq!(
            step_groups[1].action_indices,
            vec![4, 5, 6, 7, 8, 9, 10, 11]
        );
    }

    #[test]
    fn typed_direct_buy_group_captures_prediction_inputs() {
        let actions = vec![buy("mkt", 2.0, 0.9)];
        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::DirectBuy);
        let direct_buy = groups[0]
            .direct_buy
            .expect("direct-buy metadata should be present");
        assert_eq!(direct_buy.market_name, "mkt");
        assert_eq!(direct_buy.amount, 2.0);
        assert_eq!(direct_buy.cost, 0.9);
    }

    #[test]
    fn errors_when_flash_loan_bracket_is_unclosed() {
        let actions = vec![Action::FlashLoan { amount: 1.0 }, buy("a", 1.0, 1.0)];
        let err = group_actions(&actions).expect_err("missing repay must error");
        assert_eq!(
            err,
            GroupingError::MissingRepayFlashLoan {
                flash_loan_index: 0
            }
        );
    }

    #[test]
    fn errors_when_mint_appears_outside_bracket() {
        let actions = vec![Action::Mint {
            contract_1: "c1",
            contract_2: "c2",
            amount: 1.0,
            target_market: "m",
        }];
        let err = group_actions(&actions).expect_err("standalone mint should error");
        assert_eq!(
            err,
            GroupingError::UnexpectedActionOutsideGroup { index: 0 }
        );
    }

    #[test]
    fn groups_standalone_merge_as_direct_merge() {
        let actions = vec![Action::Merge {
            contract_1: "c1",
            contract_2: "c2",
            amount: 1.0,
            source_market: "s",
        }];
        let groups = group_actions(&actions).expect("standalone merge should group");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::DirectMerge);
        assert_eq!(groups[0].action_indices, vec![0]);
    }

    #[test]
    fn errors_on_flash_arb_pattern_without_mint_or_merge() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            buy("a", 1.0, 0.8),
            sell("b", 1.0, 0.9),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("unsupported flash arb must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_mint_bracket_contains_buy_leg() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            buy("a", 1.0, 0.8),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("mint bracket with buy leg must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_buy_merge_bracket_contains_sell_leg() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            buy("a", 1.0, 0.8),
            sell("b", 1.0, 0.9),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "s",
            },
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("buy-merge bracket with sell leg must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 4
            }
        );
    }

    #[test]
    fn errors_when_mint_sell_bracket_is_out_of_order() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            sell("a", 1.0, 0.8),
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("out-of-order mint-sell bracket must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_buy_merge_bracket_is_out_of_order() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "s",
            },
            buy("a", 1.0, 0.8),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("out-of-order buy-merge bracket must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_flash_loan_fee_is_non_zero() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            sell("a", 10.0, 7.0),
            Action::RepayFlashLoan { amount: 10.001 },
        ];
        let err = group_actions(&actions).expect_err("non-zero flash-loan fee must error");
        assert_eq!(
            err,
            GroupingError::UnsupportedFlashLoanFee {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn accepts_flash_loan_fee_at_tolerance_boundary() {
        let borrowed = 10.0;
        // Solve d = tol * (1 + |borrowed| + |borrowed + d|) exactly for boundary d.
        let boundary_diff = ZERO_FEE_REL_TOL * (1.0 + 2.0 * borrowed) / (1.0 - ZERO_FEE_REL_TOL);
        let actions = vec![
            Action::FlashLoan { amount: borrowed },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: borrowed,
                target_market: "t",
            },
            sell("a", borrowed, 7.0),
            Action::RepayFlashLoan {
                amount: borrowed + boundary_diff,
            },
        ];
        let groups = group_actions(&actions).expect("boundary fee tolerance should be accepted");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::MintSell);
    }

    #[test]
    fn rejects_flash_loan_fee_just_above_tolerance_boundary() {
        let borrowed = 10.0;
        let boundary_diff = ZERO_FEE_REL_TOL * (1.0 + 2.0 * borrowed) / (1.0 - ZERO_FEE_REL_TOL);
        let actions = vec![
            Action::FlashLoan { amount: borrowed },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: borrowed,
                target_market: "t",
            },
            sell("a", borrowed, 7.0),
            Action::RepayFlashLoan {
                amount: borrowed + boundary_diff * (1.0 + 1e-6),
            },
        ];
        let err = group_actions(&actions).expect_err("fee just above tolerance should be rejected");
        assert_eq!(
            err,
            GroupingError::UnsupportedFlashLoanFee {
                start_index: 0,
                end_index: 3
            }
        );
    }
}
