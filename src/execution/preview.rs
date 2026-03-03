use super::{BatchQuoteBounds, ExecutionGroupPlan, ExecutionLegPlan};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionGroupPreview {
    lines: Vec<String>,
}

impl ExecutionGroupPreview {
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    pub fn rendered_lines(&self, prefix: &str) -> Vec<String> {
        self.lines
            .iter()
            .map(|body| {
                if prefix.is_empty() {
                    body.clone()
                } else {
                    format!("{prefix}{body}")
                }
            })
            .collect()
    }

    pub fn print(&self, prefix: &str) {
        for line in self.rendered_lines(prefix) {
            println!("{line}");
        }
    }
}

fn push_leg(lines: &mut Vec<String>, idx: usize, leg: &ExecutionLegPlan) {
    let limit = leg
        .sqrt_price_limit_x96
        .map(|value| format!("{value:#x}"))
        .unwrap_or_else(|| "none".to_string());
    lines.push(format!(
            "first execution leg[{idx}]: kind={:?} market={:?} planned_quote_susd={:.9} conservative_quote_susd={:.9} allocated_slippage_susd={:.9} max_cost_susd={:?} min_proceeds_susd={:?} sqrt_price_limit_x96={}",
            leg.kind,
            leg.market_name,
            leg.planned_quote_susd,
            leg.conservative_quote_susd,
            leg.allocated_slippage_susd,
            leg.max_cost_susd,
            leg.min_proceeds_susd,
            limit
        ));
}

fn push_leg_summary(lines: &mut Vec<String>, legs: &[ExecutionLegPlan]) {
    if legs.is_empty() {
        lines.push("first execution legs: none".to_string());
        return;
    }

    lines.push(format!("first execution legs: {} total", legs.len()));
    if legs.len() <= 4 {
        for (idx, leg) in legs.iter().enumerate() {
            push_leg(lines, idx, leg);
        }
        return;
    }

    for (idx, leg) in legs.iter().take(2).enumerate() {
        push_leg(lines, idx, leg);
    }
    lines.push(format!(
        "first execution legs: ... omitted {} middle legs ...",
        legs.len() - 4
    ));
    for (idx, leg) in legs.iter().enumerate().skip(legs.len() - 2) {
        push_leg(lines, idx, leg);
    }
}

fn push_batch_bounds(lines: &mut Vec<String>, bounds: Option<BatchQuoteBounds>) {
    match bounds {
        Some(BatchQuoteBounds::Buy {
            planned_total_in_susd,
            max_total_in_susd,
        }) => lines.push(format!(
                "first execution batch bounds: kind=buy planned_total_in_susd={:.9} max_total_in_susd={:.9}",
                planned_total_in_susd, max_total_in_susd
            )),
        Some(BatchQuoteBounds::Sell {
            planned_total_out_susd,
            min_total_out_susd,
        }) => lines.push(format!(
                "first execution batch bounds: kind=sell planned_total_out_susd={:.9} min_total_out_susd={:.9}",
                planned_total_out_susd, min_total_out_susd
            )),
        None => lines.push("first execution batch bounds: none".to_string()),
    }
}

pub fn build_execution_group_preview(
    plan: &ExecutionGroupPlan,
    quote_latency_blocks: u64,
    adverse_move_bps_per_block: u64,
    batch_bounds: Option<BatchQuoteBounds>,
) -> ExecutionGroupPreview {
    let mut lines = Vec::with_capacity(plan.legs.len().min(4) + 5);
    lines.push(format!(
        "first execution group: kind={:?} step={}/{} subgroup={}/{} action_indices={:?}",
        plan.kind,
        plan.profitability_step_index + 1,
        plan.profitability_step_index + plan.step_subgroup_count,
        plan.step_subgroup_index + 1,
        plan.step_subgroup_count,
        plan.action_indices
    ));
    lines.push(format!(
            "first execution economics: edge_plan_susd={:.9} gas_total_susd={:.9} slippage_budget_susd={:.9} guaranteed_profit_floor_susd={:.9}",
            plan.edge_plan_susd,
            plan.gas_total_susd,
            plan.slippage_budget_susd,
            plan.guaranteed_profit_floor_susd
        ));
    lines.push(format!(
        "conservative execution: quote_latency_blocks={} adverse_move_bps_per_block={}",
        quote_latency_blocks, adverse_move_bps_per_block
    ));
    push_leg_summary(&mut lines, &plan.legs);
    push_batch_bounds(&mut lines, batch_bounds);
    ExecutionGroupPreview { lines }
}

pub fn print_execution_group_preview(
    prefix: &str,
    plan: &ExecutionGroupPlan,
    quote_latency_blocks: u64,
    adverse_move_bps_per_block: u64,
    batch_bounds: Option<BatchQuoteBounds>,
) {
    build_execution_group_preview(
        plan,
        quote_latency_blocks,
        adverse_move_bps_per_block,
        batch_bounds,
    )
    .print(prefix);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::U160;

    use crate::execution::{GroupKind, LegKind};

    fn sample_leg(
        action_index: usize,
        market_name: Option<&'static str>,
        kind: LegKind,
        sqrt_price_limit_x96: Option<U160>,
    ) -> ExecutionLegPlan {
        let action_index_f64 = action_index as f64;
        ExecutionLegPlan {
            action_index,
            market_name,
            kind,
            planned_quote_susd: 1.0 + action_index_f64,
            conservative_quote_susd: 1.25 + action_index_f64,
            adverse_notional_susd: 0.5 + action_index_f64,
            allocated_slippage_susd: 0.75 + action_index_f64,
            max_cost_susd: (kind == LegKind::Buy).then_some(2.0 + action_index_f64),
            min_proceeds_susd: (kind == LegKind::Sell).then_some(1.5 + action_index_f64),
            sqrt_price_limit_x96,
        }
    }

    fn sample_plan(legs: Vec<ExecutionLegPlan>) -> ExecutionGroupPlan {
        ExecutionGroupPlan {
            kind: GroupKind::DirectBuy,
            action_indices: vec![2, 4],
            profitability_step_index: 0,
            step_subgroup_index: 1,
            step_subgroup_count: 2,
            legs,
            planned_at_block: Some(123),
            edge_plan_susd: 6.2,
            l2_gas_units: 42_000,
            gas_l2_susd: 0.12,
            gas_total_susd: 0.76,
            profit_buffer_susd: 1.24,
            slippage_budget_susd: 4.2,
            guaranteed_profit_floor_susd: 1.24,
        }
    }

    #[test]
    fn renders_exact_lines_for_small_preview() {
        let plan = sample_plan(vec![
            sample_leg(0, Some("alpha"), LegKind::Buy, Some(U160::from(0xffu16))),
            sample_leg(1, None, LegKind::Sell, None),
        ]);

        let preview = build_execution_group_preview(
            &plan,
            3,
            15,
            Some(BatchQuoteBounds::Buy {
                planned_total_in_susd: 2.25,
                max_total_in_susd: 6.95,
            }),
        );

        let expected = vec![
            "first execution group: kind=DirectBuy step=1/2 subgroup=2/2 action_indices=[2, 4]"
                .to_string(),
            "first execution economics: edge_plan_susd=6.200000000 gas_total_susd=0.760000000 slippage_budget_susd=4.200000000 guaranteed_profit_floor_susd=1.240000000".to_string(),
            "conservative execution: quote_latency_blocks=3 adverse_move_bps_per_block=15"
                .to_string(),
            "first execution legs: 2 total".to_string(),
            "first execution leg[0]: kind=Buy market=Some(\"alpha\") planned_quote_susd=1.000000000 conservative_quote_susd=1.250000000 allocated_slippage_susd=0.750000000 max_cost_susd=Some(2.0) min_proceeds_susd=None sqrt_price_limit_x96=0xff".to_string(),
            "first execution leg[1]: kind=Sell market=None planned_quote_susd=2.000000000 conservative_quote_susd=2.250000000 allocated_slippage_susd=1.750000000 max_cost_susd=None min_proceeds_susd=Some(2.5) sqrt_price_limit_x96=none".to_string(),
            "first execution batch bounds: kind=buy planned_total_in_susd=2.250000000 max_total_in_susd=6.950000000".to_string(),
        ];

        assert_eq!(preview.lines(), expected.as_slice());
        assert_eq!(
            preview.rendered_lines("[test] "),
            expected
                .iter()
                .map(|line| format!("[test] {line}"))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn renders_compact_middle_omission_for_large_leg_sets() {
        let plan = sample_plan(vec![
            sample_leg(0, Some("m0"), LegKind::Buy, Some(U160::from(0x10u8))),
            sample_leg(1, Some("m1"), LegKind::Sell, Some(U160::from(0x11u8))),
            sample_leg(2, Some("m2"), LegKind::Buy, Some(U160::from(0x12u8))),
            sample_leg(3, Some("m3"), LegKind::Sell, Some(U160::from(0x13u8))),
            sample_leg(4, Some("m4"), LegKind::Buy, Some(U160::from(0x14u8))),
            sample_leg(5, Some("m5"), LegKind::Sell, None),
        ]);

        let preview = build_execution_group_preview(
            &plan,
            1,
            15,
            Some(BatchQuoteBounds::Sell {
                planned_total_out_susd: 12.5,
                min_total_out_susd: 10.75,
            }),
        );

        assert_eq!(preview.lines().len(), 10);
        assert_eq!(preview.lines()[3], "first execution legs: 6 total");
        assert_eq!(
            preview.lines()[4],
            "first execution leg[0]: kind=Buy market=Some(\"m0\") planned_quote_susd=1.000000000 conservative_quote_susd=1.250000000 allocated_slippage_susd=0.750000000 max_cost_susd=Some(2.0) min_proceeds_susd=None sqrt_price_limit_x96=0x10"
        );
        assert_eq!(
            preview.lines()[5],
            "first execution leg[1]: kind=Sell market=Some(\"m1\") planned_quote_susd=2.000000000 conservative_quote_susd=2.250000000 allocated_slippage_susd=1.750000000 max_cost_susd=None min_proceeds_susd=Some(2.5) sqrt_price_limit_x96=0x11"
        );
        assert_eq!(
            preview.lines()[6],
            "first execution legs: ... omitted 2 middle legs ..."
        );
        assert_eq!(
            preview.lines()[7],
            "first execution leg[4]: kind=Buy market=Some(\"m4\") planned_quote_susd=5.000000000 conservative_quote_susd=5.250000000 allocated_slippage_susd=4.750000000 max_cost_susd=Some(6.0) min_proceeds_susd=None sqrt_price_limit_x96=0x14"
        );
        assert_eq!(
            preview.lines()[8],
            "first execution leg[5]: kind=Sell market=Some(\"m5\") planned_quote_susd=6.000000000 conservative_quote_susd=6.250000000 allocated_slippage_susd=5.750000000 max_cost_susd=None min_proceeds_susd=Some(6.5) sqrt_price_limit_x96=none"
        );
        assert_eq!(
            preview
                .rendered_lines("")
                .last()
                .expect("batch bounds line must be present"),
            "first execution batch bounds: kind=sell planned_total_out_susd=12.500000000 min_total_out_susd=10.750000000"
        );
    }
}
