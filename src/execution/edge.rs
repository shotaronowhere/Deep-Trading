use std::collections::HashMap;

use crate::pools::normalize_market_name;

use super::GroupKind;
use super::grouping::ExecutionGroup;

pub fn planned_cashflow_edge_susd(group: &ExecutionGroup) -> f64 {
    group.planned_proceeds_susd - group.planned_cost_susd
}

pub fn planned_edge_from_prediction_map_susd(
    group: &ExecutionGroup,
    prediction_by_market: &HashMap<String, f64>,
) -> Option<f64> {
    match group.kind {
        GroupKind::DirectBuy => {
            let buy = group.direct_buy?;
            let key = normalize_market_name(buy.market_name);
            let prediction = match prediction_by_market.get(&key).copied() {
                Some(prediction) => prediction,
                None => {
                    tracing::info!(
                        market_name = buy.market_name,
                        normalized_key = %key,
                        "missing prediction for direct-buy market; using prediction=0.0"
                    );
                    0.0
                }
            };
            debug_assert!(
                buy.amount.is_finite() && buy.amount >= 0.0,
                "invalid buy amount at action index {}: {}",
                buy.action_index,
                buy.amount
            );
            debug_assert!(
                buy.cost.is_finite() && buy.cost >= 0.0,
                "invalid buy cost at action index {}: {}",
                buy.action_index,
                buy.cost
            );
            debug_assert!(
                prediction.is_finite(),
                "invalid prediction for direct-buy market '{}' (normalized '{}'): {}",
                buy.market_name,
                key,
                prediction
            );
            Some(buy.amount.max(0.0) * prediction - buy.cost.max(0.0))
        }
        _ => Some(planned_cashflow_edge_susd(group)),
    }
}
