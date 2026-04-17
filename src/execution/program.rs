use alloy::primitives::Address;

use super::batch_bounds::derive_batch_quote_bounds_unchecked;
use super::gas::{
    ExactGasQuoteError, GasAssumptions, LiveOptimismFeeInputs,
    build_unsigned_batch_execute_tx_bytes, estimate_l1_data_fee_susd_for_tx_bytes_len,
    estimate_l2_gas_susd,
};
use super::tx_builder::{
    ExecutionAddressBook, TxBuildError, build_trade_executor_calls_with_address_book,
};
use super::{ExecutionGroupPlan, ExecutionMode, ITradeExecutor, SUSD_DECIMALS, is_plan_stale};
use crate::portfolio::Action;

pub const MAX_PACKED_TX_L2_GAS_UNITS: u64 = 40_000_000;

#[derive(Debug, Clone)]
pub struct ExecutionChunkPlan {
    pub plans: Vec<ExecutionGroupPlan>,
    pub total_l2_gas_units: u64,
    pub unsigned_tx_bytes_len: usize,
    pub estimated_total_fee_susd: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionProgramPlan {
    pub mode: ExecutionMode,
    pub chunks: Vec<ExecutionChunkPlan>,
    pub total_l2_fee_susd: f64,
    pub total_l1_fee_susd: f64,
    pub total_fee_susd: f64,
    pub strict_subgroup_count: usize,
    pub tx_count: usize,
}

#[derive(Debug, Clone)]
pub enum ProgramBuildError {
    MissingBounds(String),
    TxBuild(TxBuildError),
    TxBytes(ExactGasQuoteError),
    GasCapExceeded { l2_gas_units: u64, cap: u64 },
}

impl std::fmt::Display for ProgramBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBounds(message) => write!(f, "{message}"),
            Self::TxBuild(err) => write!(f, "{err}"),
            Self::TxBytes(err) => write!(f, "{err}"),
            Self::GasCapExceeded { l2_gas_units, cap } => {
                write!(f, "plan gas {l2_gas_units} exceeds packed tx cap {cap}")
            }
        }
    }
}

impl std::error::Error for ProgramBuildError {}

impl From<TxBuildError> for ProgramBuildError {
    fn from(value: TxBuildError) -> Self {
        Self::TxBuild(value)
    }
}

impl From<ExactGasQuoteError> for ProgramBuildError {
    fn from(value: ExactGasQuoteError) -> Self {
        Self::TxBytes(value)
    }
}

fn append_plan_calls_unchecked_with_address_book(
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    calls: &mut Vec<ITradeExecutor::Call>,
    address_book: &ExecutionAddressBook,
) -> Result<(), ProgramBuildError> {
    let batch_bounds = derive_batch_quote_bounds_unchecked(plan)
        .map_err(|err| ProgramBuildError::MissingBounds(err.to_string()))?
        .map(|bounds| bounds.to_token_bounds(SUSD_DECIMALS))
        .transpose()
        .map_err(|err| ProgramBuildError::MissingBounds(err.to_string()))?;
    calls.extend(build_trade_executor_calls_with_address_book(
        executor,
        actions,
        plan,
        batch_bounds,
        address_book,
    )?);
    Ok(())
}

pub fn build_chunk_calls_checked(
    executor: Address,
    actions: &[Action],
    plans: &[ExecutionGroupPlan],
    current_block: u64,
    max_stale_blocks: u64,
) -> Result<Vec<ITradeExecutor::Call>, ProgramBuildError> {
    build_chunk_calls_checked_with_address_book(
        executor,
        actions,
        plans,
        current_block,
        max_stale_blocks,
        &ExecutionAddressBook::default(),
    )
}

pub fn build_chunk_calls_checked_with_address_book(
    executor: Address,
    actions: &[Action],
    plans: &[ExecutionGroupPlan],
    current_block: u64,
    max_stale_blocks: u64,
    address_book: &ExecutionAddressBook,
) -> Result<Vec<ITradeExecutor::Call>, ProgramBuildError> {
    let mut calls = Vec::new();
    for plan in plans {
        if is_plan_stale(plan, current_block, max_stale_blocks) {
            return Err(ProgramBuildError::MissingBounds(format!(
                "stale or invalid bounds for group {:?} at profitability step {} subgroup {}",
                plan.kind, plan.profitability_step_index, plan.step_subgroup_index
            )));
        }
        let batch_bounds =
            super::batch_bounds::derive_batch_quote_bounds(plan, current_block, max_stale_blocks)
                .map_err(|err| ProgramBuildError::MissingBounds(err.to_string()))?
                .map(|bounds| bounds.to_token_bounds(SUSD_DECIMALS))
                .transpose()
                .map_err(|err| ProgramBuildError::MissingBounds(err.to_string()))?;
        calls.extend(build_trade_executor_calls_with_address_book(
            executor,
            actions,
            plan,
            batch_bounds,
            address_book,
        )?);
    }
    Ok(calls)
}

fn finalize_chunk(
    plans: &[ExecutionGroupPlan],
    calls: &[ITradeExecutor::Call],
    total_l2_gas_units: u64,
    fee_inputs: LiveOptimismFeeInputs,
    gas_assumptions: &GasAssumptions,
    eth_usd: f64,
    executor: Address,
) -> Result<ExecutionChunkPlan, ProgramBuildError> {
    let unsigned_tx_data =
        build_unsigned_batch_execute_tx_bytes(executor, calls, fee_inputs, total_l2_gas_units)?;
    let l2_fee = estimate_l2_gas_susd(
        total_l2_gas_units,
        fee_inputs.gas_price_wei as f64 / 1e18,
        eth_usd,
    );
    let l1_fee = estimate_l1_data_fee_susd_for_tx_bytes_len(
        gas_assumptions,
        unsigned_tx_data.len(),
        eth_usd,
    );
    Ok(ExecutionChunkPlan {
        plans: plans.to_vec(),
        total_l2_gas_units,
        unsigned_tx_bytes_len: unsigned_tx_data.len(),
        estimated_total_fee_susd: l2_fee + l1_fee,
    })
}

pub fn compile_execution_program_unchecked(
    mode: ExecutionMode,
    executor: Address,
    actions: &[Action],
    plans: &[ExecutionGroupPlan],
    fee_inputs: LiveOptimismFeeInputs,
    gas_assumptions: &GasAssumptions,
    eth_usd: f64,
) -> Result<ExecutionProgramPlan, ProgramBuildError> {
    compile_execution_program_unchecked_with_address_book(
        mode,
        executor,
        actions,
        plans,
        fee_inputs,
        gas_assumptions,
        eth_usd,
        &ExecutionAddressBook::default(),
    )
}

pub fn compile_execution_program_unchecked_with_address_book(
    mode: ExecutionMode,
    executor: Address,
    actions: &[Action],
    plans: &[ExecutionGroupPlan],
    fee_inputs: LiveOptimismFeeInputs,
    gas_assumptions: &GasAssumptions,
    eth_usd: f64,
    address_book: &ExecutionAddressBook,
) -> Result<ExecutionProgramPlan, ProgramBuildError> {
    if plans.is_empty() {
        return Ok(ExecutionProgramPlan {
            mode,
            chunks: Vec::new(),
            total_l2_fee_susd: 0.0,
            total_l1_fee_susd: 0.0,
            total_fee_susd: 0.0,
            strict_subgroup_count: 0,
            tx_count: 0,
        });
    }

    let mut chunks = Vec::new();
    let mut current_plans = Vec::new();
    let mut current_calls = Vec::new();
    let mut current_l2_gas_units = 0u64;

    let flush_current = |chunks: &mut Vec<ExecutionChunkPlan>,
                         current_plans: &mut Vec<ExecutionGroupPlan>,
                         current_calls: &mut Vec<ITradeExecutor::Call>,
                         current_l2_gas_units: &mut u64|
     -> Result<(), ProgramBuildError> {
        if current_plans.is_empty() {
            return Ok(());
        }
        let chunk = finalize_chunk(
            current_plans,
            current_calls,
            *current_l2_gas_units,
            fee_inputs,
            gas_assumptions,
            eth_usd,
            executor,
        )?;
        chunks.push(chunk);
        current_plans.clear();
        current_calls.clear();
        *current_l2_gas_units = 0;
        Ok(())
    };

    for plan in plans {
        if plan.l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
            return Err(ProgramBuildError::GasCapExceeded {
                l2_gas_units: plan.l2_gas_units,
                cap: MAX_PACKED_TX_L2_GAS_UNITS,
            });
        }

        let mut plan_calls = Vec::new();
        append_plan_calls_unchecked_with_address_book(
            executor,
            actions,
            plan,
            &mut plan_calls,
            address_book,
        )?;

        match mode {
            ExecutionMode::Strict => {
                flush_current(
                    &mut chunks,
                    &mut current_plans,
                    &mut current_calls,
                    &mut current_l2_gas_units,
                )?;
                current_l2_gas_units = plan.l2_gas_units;
                current_plans.push(plan.clone());
                current_calls = plan_calls;
                flush_current(
                    &mut chunks,
                    &mut current_plans,
                    &mut current_calls,
                    &mut current_l2_gas_units,
                )?;
            }
            ExecutionMode::Packed => {
                let tentative_l2_gas_units = current_l2_gas_units.saturating_add(plan.l2_gas_units);
                if tentative_l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS && !current_plans.is_empty()
                {
                    flush_current(
                        &mut chunks,
                        &mut current_plans,
                        &mut current_calls,
                        &mut current_l2_gas_units,
                    )?;
                }

                let tentative_l2_gas_units = current_l2_gas_units.saturating_add(plan.l2_gas_units);
                if tentative_l2_gas_units >= MAX_PACKED_TX_L2_GAS_UNITS {
                    return Err(ProgramBuildError::GasCapExceeded {
                        l2_gas_units: tentative_l2_gas_units,
                        cap: MAX_PACKED_TX_L2_GAS_UNITS,
                    });
                }

                let mut tentative_calls = current_calls.clone();
                tentative_calls.extend(plan_calls.iter().cloned());
                if build_unsigned_batch_execute_tx_bytes(
                    executor,
                    &tentative_calls,
                    fee_inputs,
                    tentative_l2_gas_units,
                )
                .is_err()
                    && !current_plans.is_empty()
                {
                    flush_current(
                        &mut chunks,
                        &mut current_plans,
                        &mut current_calls,
                        &mut current_l2_gas_units,
                    )?;
                    current_calls = plan_calls;
                    current_plans.push(plan.clone());
                    current_l2_gas_units = plan.l2_gas_units;
                } else {
                    current_calls = tentative_calls;
                    current_plans.push(plan.clone());
                    current_l2_gas_units = tentative_l2_gas_units;
                }
            }
        }
    }

    flush_current(
        &mut chunks,
        &mut current_plans,
        &mut current_calls,
        &mut current_l2_gas_units,
    )?;

    let total_l2_fee_susd = chunks.iter().fold(0.0, |sum, chunk| {
        sum + estimate_l2_gas_susd(
            chunk.total_l2_gas_units,
            fee_inputs.gas_price_wei as f64 / 1e18,
            eth_usd,
        )
    });
    let total_fee_susd = chunks
        .iter()
        .fold(0.0, |sum, chunk| sum + chunk.estimated_total_fee_susd);
    let total_l1_fee_susd = total_fee_susd - total_l2_fee_susd;

    Ok(ExecutionProgramPlan {
        mode,
        strict_subgroup_count: plans.len(),
        tx_count: chunks.len(),
        chunks,
        total_l2_fee_susd,
        total_l1_fee_susd,
        total_fee_susd,
    })
}

#[cfg(test)]
mod tests {
    use alloy::primitives::{Address, U160};

    use super::*;
    use crate::execution::GroupKind;
    use crate::execution::gas::GasAssumptions;
    use crate::execution::{ExecutionGroupPlan, ExecutionLegPlan, LegKind};
    use crate::portfolio::Action;

    fn sample_plan(kind: GroupKind, action_index: usize, l2_gas_units: u64) -> ExecutionGroupPlan {
        let leg_kind = match kind {
            GroupKind::DirectSell | GroupKind::MintSell => LegKind::Sell,
            _ => LegKind::Buy,
        };
        ExecutionGroupPlan {
            kind,
            action_indices: vec![action_index],
            profitability_step_index: 0,
            step_subgroup_index: action_index,
            step_subgroup_count: 2,
            legs: vec![ExecutionLegPlan {
                action_index,
                market_name: Some("m"),
                kind: leg_kind,
                planned_quote_susd: 1.0,
                conservative_quote_susd: 1.0,
                adverse_notional_susd: 1.0,
                allocated_slippage_susd: 0.01,
                max_cost_susd: (leg_kind == LegKind::Buy).then_some(1.1),
                min_proceeds_susd: (leg_kind == LegKind::Sell).then_some(0.9),
                sqrt_price_limit_x96: Some(U160::from(123u64)),
            }],
            planned_at_block: Some(100),
            edge_plan_susd: 10.0,
            l2_gas_units,
            gas_l2_susd: 0.1,
            gas_total_susd: 0.2,
            profit_buffer_susd: 0.0,
            slippage_budget_susd: 0.0,
            guaranteed_profit_floor_susd: 0.0,
        }
    }

    #[test]
    fn packed_program_combines_multiple_groups_when_under_gas_cap() {
        let actions = vec![
            Action::Buy {
                market_name: crate::markets::MARKETS_L1[0].name,
                amount: 1.0,
                cost: 1.0,
            },
            Action::Buy {
                market_name: crate::markets::MARKETS_L1[1].name,
                amount: 1.0,
                cost: 1.0,
            },
        ];
        let plans = vec![
            sample_plan(GroupKind::DirectBuy, 0, 100_000),
            sample_plan(GroupKind::DirectBuy, 1, 100_000),
        ];
        let program = compile_execution_program_unchecked(
            ExecutionMode::Packed,
            Address::ZERO,
            &actions,
            &plans,
            LiveOptimismFeeInputs {
                chain_id: 10,
                sender_nonce: 0,
                gas_price_wei: 1_000_000,
            },
            &GasAssumptions {
                l1_fee_per_byte_wei: 1.0,
                ..GasAssumptions::default()
            },
            3000.0,
        )
        .expect("packed compile should succeed");
        assert_eq!(program.chunks.len(), 1);
        assert_eq!(program.strict_subgroup_count, 2);
    }
}
