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

fn classify_step_kind(groups: &[ExecutionGroup]) -> Option<ProfitabilityStepKind> {
    debug_assert!(!groups.is_empty(), "step group cannot be empty");
    let has = |kind: GroupKind| groups.iter().any(|g| g.kind == kind);
    let all = |kind: GroupKind| groups.iter().all(|g| g.kind == kind);

    if all(GroupKind::MintSell) {
        return Some(ProfitabilityStepKind::ArbMintSell);
    }
    if all(GroupKind::BuyMerge) {
        return Some(ProfitabilityStepKind::ArbBuyMerge);
    }
    if all(GroupKind::DirectBuy) {
        return Some(ProfitabilityStepKind::PureDirectBuy);
    }
    if all(GroupKind::DirectSell) {
        return Some(ProfitabilityStepKind::PureDirectSell);
    }
    if all(GroupKind::DirectMerge) {
        return Some(ProfitabilityStepKind::PureDirectMerge);
    }
    if has(GroupKind::MintSell)
        && has(GroupKind::DirectBuy)
        && groups
            .iter()
            .all(|g| matches!(g.kind, GroupKind::MintSell | GroupKind::DirectBuy))
    {
        return Some(ProfitabilityStepKind::MixedDirectBuyMintSell);
    }
    if has(GroupKind::BuyMerge)
        && has(GroupKind::DirectSell)
        && groups
            .iter()
            .all(|g| matches!(g.kind, GroupKind::BuyMerge | GroupKind::DirectSell))
    {
        return Some(ProfitabilityStepKind::MixedDirectSellBuyMerge);
    }

    None
}

fn step_group_from_route_groups(
    route_groups: &[ExecutionGroup],
) -> Result<ProfitabilityStepGroup, GroupingError> {
    let mut action_indices = Vec::new();
    for group in route_groups {
        action_indices.extend_from_slice(&group.action_indices);
    }
    let kind = classify_step_kind(route_groups).ok_or_else(|| {
        let start_index = action_indices.first().copied().unwrap_or(0);
        let end_index = action_indices.last().copied().unwrap_or(start_index);
        GroupingError::UnsupportedProfitabilityStepComposition {
            start_index,
            end_index,
        }
    })?;
    Ok(ProfitabilityStepGroup {
        kind,
        action_indices,
        strict_groups: route_groups.to_vec(),
    })
}

fn merge_profitability_step_groups(
    actions: &[Action],
    route_groups: &[ExecutionGroup],
) -> Result<Vec<ProfitabilityStepGroup>, GroupingError> {
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
    ) -> Option<usize> {
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

        if end > start + 1 { Some(end) } else { None }
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
    ) -> Option<usize> {
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
            Some(end)
        } else {
            None
        }
    }

    while i < route_groups.len() {
        if let Some(next_i) = merge_two_kind_block(
            actions,
            route_groups,
            i,
            GroupKind::MintSell,
            GroupKind::DirectBuy,
        ) {
            merged.push(step_group_from_route_groups(&route_groups[i..next_i])?);
            i = next_i;
            continue;
        }

        if let Some(next_i) = merge_two_kind_block(
            actions,
            route_groups,
            i,
            GroupKind::BuyMerge,
            GroupKind::DirectSell,
        ) {
            merged.push(step_group_from_route_groups(&route_groups[i..next_i])?);
            i = next_i;
            continue;
        }

        if let Some(next_i) =
            merge_same_kind_unique_market_block(actions, route_groups, i, route_groups[i].kind)
        {
            merged.push(step_group_from_route_groups(&route_groups[i..next_i])?);
            i = next_i;
            continue;
        }

        merged.push(step_group_from_route_groups(&route_groups[i..=i])?);
        i += 1;
    }

    Ok(merged)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupingError {
    UnexpectedActionOutsideGroup {
        index: usize,
    },
    InvalidMintSellGroup {
        start_index: usize,
        end_index: usize,
    },
    InvalidBuyMergeGroup {
        start_index: usize,
        end_index: usize,
    },
    UnsupportedProfitabilityStepComposition {
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
            Self::InvalidMintSellGroup {
                start_index,
                end_index,
            } => write!(
                f,
                "unsupported mint-sell group shape between indices {start_index} and {end_index}"
            ),
            Self::InvalidBuyMergeGroup {
                start_index,
                end_index,
            } => write!(
                f,
                "unsupported buy-merge group shape between indices {start_index} and {end_index}"
            ),
            Self::UnsupportedProfitabilityStepComposition {
                start_index,
                end_index,
            } => write!(
                f,
                "unsupported profitability-step composition between indices {start_index} and {end_index}"
            ),
        }
    }
}

impl std::error::Error for GroupingError {}

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
/// - `MintSell`: `Mint -> Sell+`
/// - `BuyMerge`: `Buy+ -> Merge`
///
/// Any unsupported shape returns an error (fail closed) so callers can skip submission.
pub fn group_execution_actions(actions: &[Action]) -> Result<Vec<ExecutionGroup>, GroupingError> {
    let mut groups = Vec::new();
    let mut i = 0usize;

    while i < actions.len() {
        match &actions[i] {
            Action::Buy { .. } => {
                let start = i;
                let mut end = i;
                while end < actions.len() && matches!(actions[end], Action::Buy { .. }) {
                    end += 1;
                }
                if end < actions.len() && matches!(actions[end], Action::Merge { .. }) {
                    let action_indices = (start..=end).collect();
                    groups.push(typed_group_from_actions(
                        actions,
                        GroupKind::BuyMerge,
                        action_indices,
                    ));
                    i = end + 1;
                } else {
                    groups.push(typed_group_from_actions(
                        actions,
                        GroupKind::DirectBuy,
                        vec![i],
                    ));
                    i += 1;
                }
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
            Action::Mint { .. } => {
                let start = i;
                let mut end = i + 1;
                while end < actions.len() && matches!(actions[end], Action::Sell { .. }) {
                    end += 1;
                }
                if end == start + 1 {
                    return Err(GroupingError::InvalidMintSellGroup {
                        start_index: start,
                        end_index: start,
                    });
                }
                let action_indices = (start..end).collect();
                groups.push(typed_group_from_actions(
                    actions,
                    GroupKind::MintSell,
                    action_indices,
                ));
                i = end;
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
    merge_profitability_step_groups(actions, &route_groups)
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

    fn mint(target_market: &'static str, amount: f64) -> Action {
        Action::Mint {
            contract_1: "c1",
            contract_2: "c2",
            amount,
            target_market,
        }
    }

    fn merge(source_market: &'static str, amount: f64) -> Action {
        Action::Merge {
            contract_1: "c1",
            contract_2: "c2",
            amount,
            source_market,
        }
    }

    #[test]
    fn groups_direct_and_indirect_actions() {
        let actions = vec![
            buy("a", 1.0, 2.0),
            mint("t", 10.0),
            sell("b", 10.0, 4.0),
            buy("x", 5.0, 1.5),
            merge("x", 5.0),
            sell("z", 1.0, 1.0),
        ];

        let groups = group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(groups.len(), 4);
        assert_eq!(groups[0].kind, GroupKind::DirectBuy);
        assert_eq!(groups[1].kind, GroupKind::MintSell);
        assert_eq!(groups[2].kind, GroupKind::BuyMerge);
        assert_eq!(groups[3].kind, GroupKind::DirectSell);
        assert_eq!(groups[0].action_indices, vec![0]);
        assert_eq!(groups[1].action_indices, vec![1, 2]);
        assert_eq!(groups[2].action_indices, vec![3, 4]);
        assert_eq!(groups[3].action_indices, vec![5]);
    }

    #[test]
    fn typed_groups_capture_cashflow_and_dex_legs() {
        let actions = vec![mint("t", 10.0), sell("a", 10.0, 7.0), sell("b", 10.0, 6.0)];

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
    fn typed_buy_merge_group_tracks_cost_and_proceeds() {
        let actions = vec![buy("x", 2.0, 0.7), buy("y", 2.0, 0.8), merge("x", 2.0)];

        let groups = group_execution_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(groups[0].planned_cost_susd, 1.5);
        assert_eq!(groups[0].planned_proceeds_susd, 2.0);
        assert_eq!(groups[0].buy_legs, 2);
        assert_eq!(groups[0].sell_legs, 0);
        assert_eq!(groups[0].legs.len(), 2);
    }

    #[test]
    fn errors_when_mint_group_has_no_sell_leg() {
        let actions = vec![mint("m", 1.0)];
        let err = group_actions(&actions).expect_err("standalone mint should fail");
        assert_eq!(
            err,
            GroupingError::InvalidMintSellGroup {
                start_index: 0,
                end_index: 0,
            }
        );
    }

    #[test]
    fn groups_standalone_merge_as_direct_merge() {
        let actions = vec![merge("s", 1.0)];
        let groups = group_actions(&actions).expect("standalone merge should group");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::DirectMerge);
        assert_eq!(groups[0].action_indices, vec![0]);
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
    fn profitability_step_grouping_merges_leading_direct_buys_with_mint_sell_when_coupled() {
        let actions = vec![
            buy("x", 1.0, 0.6),
            buy("y", 1.0, 0.5),
            mint("x", 6.0),
            sell("a", 6.0, 4.0),
        ];

        let strict_groups =
            group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(strict_groups.len(), 3);
        assert_eq!(strict_groups[0].kind, GroupKind::DirectBuy);
        assert_eq!(strict_groups[1].kind, GroupKind::DirectBuy);
        assert_eq!(strict_groups[2].kind, GroupKind::MintSell);

        let step_groups = group_execution_actions_by_profitability_step(&actions)
            .expect("step grouping should succeed");
        assert_eq!(step_groups.len(), 1);
        assert_eq!(
            step_groups[0].kind,
            ProfitabilityStepKind::MixedDirectBuyMintSell
        );
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn profitability_step_grouping_does_not_merge_unrelated_mint_and_direct_buy_blocks() {
        let actions = vec![mint("x", 6.0), sell("a", 6.0, 4.0), buy("y", 1.0, 0.5)];

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 2);
        assert_eq!(step_groups[0].kind, GroupKind::MintSell);
        assert_eq!(step_groups[0].action_indices, vec![0, 1]);
        assert_eq!(step_groups[1].kind, GroupKind::DirectBuy);
        assert_eq!(step_groups[1].action_indices, vec![2]);
    }

    #[test]
    fn profitability_step_grouping_merges_buy_merge_with_trailing_direct_sell_when_coupled() {
        let actions = vec![buy("x", 1.0, 0.7), merge("x", 1.0), sell("x", 0.4, 0.3)];

        let strict_groups =
            group_execution_actions(&actions).expect("strict grouping should succeed");
        assert_eq!(strict_groups.len(), 2);
        assert_eq!(strict_groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(strict_groups[1].kind, GroupKind::DirectSell);

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 1);
        assert_eq!(step_groups[0].kind, GroupKind::BuyMerge);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2]);
    }

    #[test]
    fn profitability_step_grouping_keeps_repeated_mixed_mint_buy_phases_separate() {
        let actions = vec![
            mint("x", 6.0),
            sell("a", 6.0, 4.0),
            buy("x", 1.0, 0.6),
            mint("y", 5.0),
            sell("b", 5.0, 3.0),
            buy("y", 1.0, 0.5),
        ];

        let step_groups =
            group_actions(&actions).expect("profitability-step grouping should succeed");
        assert_eq!(step_groups.len(), 2);
        assert_eq!(step_groups[0].kind, GroupKind::MintSell);
        assert_eq!(step_groups[0].action_indices, vec![0, 1, 2]);
        assert_eq!(step_groups[1].kind, GroupKind::MintSell);
        assert_eq!(step_groups[1].action_indices, vec![3, 4, 5]);
    }

    #[test]
    fn step_group_classification_fails_closed_on_unsupported_kind_mix() {
        let actions = vec![merge("m", 1.0), buy("m", 1.0, 0.5)];
        let route_groups = vec![
            typed_group_from_actions(&actions, GroupKind::DirectMerge, vec![0]),
            typed_group_from_actions(&actions, GroupKind::DirectBuy, vec![1]),
        ];
        let err = step_group_from_route_groups(&route_groups)
            .expect_err("unsupported mixed composition must fail closed");
        assert_eq!(
            err,
            GroupingError::UnsupportedProfitabilityStepComposition {
                start_index: 0,
                end_index: 1,
            }
        );
    }
}
