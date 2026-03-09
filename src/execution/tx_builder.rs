use std::fmt;
use std::str::FromStr;

use alloy::primitives::{Address, U160, U256};
use alloy::sol_types::SolCall;
use alloy_primitives::aliases::U24;

use super::{
    BASE_COLLATERAL, BatchTokenBounds, CTF_ROUTER_ADDRESS, ICTFRouter, ITradeExecutor,
    IV3SwapRouter, MARKET_1_ADDRESS, MARKET_2_ADDRESS, MARKET_2_COLLATERAL,
    QuoteAmountConversionError, SWAP_ROUTER_ADDRESS as SWAP_ROUTER, quote_to_u256_ceil,
    quote_to_u256_floor,
};
use crate::markets::MARKETS_L1;
use crate::portfolio::Action;
use crate::{execution::ExecutionGroupPlan, execution::GroupKind, execution::LegKind};

const TOKEN_DECIMALS: u8 = 18;
const SWAP_FEE_TIER: u16 = 100;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxBuildError {
    ActionIndexOutOfBounds {
        index: usize,
    },
    UnsupportedGroupShape {
        kind: GroupKind,
        reason: &'static str,
    },
    MissingBounds {
        kind: GroupKind,
    },
    UnexpectedBounds {
        kind: GroupKind,
    },
    BoundsKindMismatch {
        kind: GroupKind,
        expected: &'static str,
    },
    UnknownMarket {
        market_name: &'static str,
    },
    InvalidMarketTokenAddress {
        market_name: &'static str,
        raw: String,
    },
    MissingPriceLimit {
        kind: GroupKind,
        action_index: usize,
    },
    AmountConversion(QuoteAmountConversionError),
}

impl fmt::Display for TxBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActionIndexOutOfBounds { index } => {
                write!(f, "action index {index} is out of bounds")
            }
            Self::UnsupportedGroupShape { kind, reason } => {
                write!(f, "unsupported {kind:?} shape: {reason}")
            }
            Self::MissingBounds { kind } => write!(f, "missing batch bounds for {kind:?} group"),
            Self::UnexpectedBounds { kind } => {
                write!(f, "unexpected batch bounds for {kind:?} group")
            }
            Self::BoundsKindMismatch { kind, expected } => {
                write!(f, "invalid bounds kind for {kind:?}; expected {expected}")
            }
            Self::UnknownMarket { market_name } => {
                write!(f, "unknown market name in action: {market_name}")
            }
            Self::InvalidMarketTokenAddress { market_name, raw } => {
                write!(
                    f,
                    "invalid outcome token address for market {market_name}: {raw}"
                )
            }
            Self::MissingPriceLimit { kind, action_index } => {
                write!(
                    f,
                    "missing sqrtPriceLimitX96 for {kind:?} leg at action index {action_index}"
                )
            }
            Self::AmountConversion(err) => write!(f, "amount conversion error: {err:?}"),
        }
    }
}

impl std::error::Error for TxBuildError {}

impl From<QuoteAmountConversionError> for TxBuildError {
    fn from(value: QuoteAmountConversionError) -> Self {
        Self::AmountConversion(value)
    }
}

pub fn build_trade_executor_calls(
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    match plan.kind {
        GroupKind::DirectBuy => build_direct_buy_calls(executor, actions, plan, batch_bounds),
        GroupKind::DirectSell => build_direct_sell_calls(executor, actions, plan, batch_bounds),
        GroupKind::MintSell => build_mint_sell_calls(executor, actions, plan, batch_bounds),
        GroupKind::BuyMerge => build_buy_merge_calls(executor, actions, plan, batch_bounds),
        GroupKind::DirectMerge => build_direct_merge_calls(actions, plan, batch_bounds),
    }
}

fn leg_for_action(
    plan: &ExecutionGroupPlan,
    expected_kind: LegKind,
    action_index: usize,
) -> Result<&super::ExecutionLegPlan, TxBuildError> {
    plan.legs
        .iter()
        .find(|leg| leg.action_index == action_index && leg.kind == expected_kind)
        .ok_or(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "group is missing the execution leg for an action",
        })
}

fn price_limit_for_leg(
    plan: &ExecutionGroupPlan,
    leg: &super::ExecutionLegPlan,
) -> Result<U160, TxBuildError> {
    leg.sqrt_price_limit_x96
        .ok_or(TxBuildError::MissingPriceLimit {
            kind: plan.kind,
            action_index: leg.action_index,
        })
}

fn max_cost_for_leg(
    plan: &ExecutionGroupPlan,
    leg: &super::ExecutionLegPlan,
) -> Result<U256, TxBuildError> {
    let Some(max_cost) = leg.max_cost_susd else {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "buy leg is missing a max cost bound",
        });
    };
    amount_to_wei_ceil(max_cost)
}

fn min_proceeds_for_leg(
    plan: &ExecutionGroupPlan,
    leg: &super::ExecutionLegPlan,
) -> Result<U256, TxBuildError> {
    let Some(min_proceeds) = leg.min_proceeds_susd else {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "sell leg is missing a min proceeds bound",
        });
    };
    amount_to_wei_floor(min_proceeds)
}

fn build_exact_output_single_call(
    recipient: Address,
    token_out: Address,
    amount_out: U256,
    amount_in_max: U256,
    sqrt_price_limit_x96: U160,
) -> ITradeExecutor::Call {
    let calldata = IV3SwapRouter::exactOutputSingleCall {
        params: IV3SwapRouter::ExactOutputSingleParams {
            tokenIn: BASE_COLLATERAL,
            tokenOut: token_out,
            fee: U24::from(SWAP_FEE_TIER),
            recipient,
            amountOut: amount_out,
            amountInMaximum: amount_in_max,
            sqrtPriceLimitX96: sqrt_price_limit_x96,
        },
    }
    .abi_encode();
    executor_call(SWAP_ROUTER, calldata)
}

fn build_exact_input_single_call(
    recipient: Address,
    token_in: Address,
    amount_in: U256,
    amount_out_min: U256,
    sqrt_price_limit_x96: U160,
) -> ITradeExecutor::Call {
    let calldata = IV3SwapRouter::exactInputSingleCall {
        params: IV3SwapRouter::ExactInputSingleParams {
            tokenIn: token_in,
            tokenOut: BASE_COLLATERAL,
            fee: U24::from(SWAP_FEE_TIER),
            recipient,
            amountIn: amount_in,
            amountOutMinimum: amount_out_min,
            sqrtPriceLimitX96: sqrt_price_limit_x96,
        },
    }
    .abi_encode();
    executor_call(SWAP_ROUTER, calldata)
}

fn build_direct_buy_calls(
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let _ = expect_buy_bounds(plan.kind, batch_bounds)?;
    if plan.action_indices.len() != 1 {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "DirectBuy must contain exactly one action",
        });
    }

    let action = action_at(actions, plan.action_indices[0])?;
    let Action::Buy {
        market_name,
        amount,
        ..
    } = action
    else {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "DirectBuy must contain only Buy action",
        });
    };

    let token_out = outcome_token_for_market(*market_name)?;
    let amount_out_wei = amount_to_wei_ceil(*amount)?;
    let leg = leg_for_action(plan, super::LegKind::Buy, plan.action_indices[0])?;
    let max_cost_wei = max_cost_for_leg(plan, leg)?;
    let sqrt_price_limit_x96 = price_limit_for_leg(plan, leg)?;

    Ok(vec![build_exact_output_single_call(
        executor,
        token_out,
        amount_out_wei,
        max_cost_wei,
        sqrt_price_limit_x96,
    )])
}

fn build_direct_sell_calls(
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let _ = expect_sell_bounds(plan.kind, batch_bounds)?;
    if plan.action_indices.len() != 1 {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "DirectSell must contain exactly one action",
        });
    }

    let action = action_at(actions, plan.action_indices[0])?;
    let Action::Sell {
        market_name,
        amount,
        ..
    } = action
    else {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "DirectSell must contain only Sell action",
        });
    };

    let token_in = outcome_token_for_market(*market_name)?;
    let amount_in_wei = amount_to_wei_floor(*amount)?;
    let leg = leg_for_action(plan, super::LegKind::Sell, plan.action_indices[0])?;
    let min_proceeds_wei = min_proceeds_for_leg(plan, leg)?;
    let sqrt_price_limit_x96 = price_limit_for_leg(plan, leg)?;

    Ok(vec![build_exact_input_single_call(
        executor,
        token_in,
        amount_in_wei,
        min_proceeds_wei,
        sqrt_price_limit_x96,
    )])
}

fn build_mint_sell_calls(
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let _ = expect_sell_bounds(plan.kind, batch_bounds)?;
    if plan.action_indices.len() < 2 {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "MintSell must contain Mint followed by one or more Sell actions",
        });
    }

    let first = action_at(actions, plan.action_indices[0])?;
    let Action::Mint { amount, .. } = first else {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "MintSell must start with Mint action",
        });
    };
    let mint_amount_wei = amount_to_wei_floor(*amount)?;

    let mut swap_calls = Vec::new();
    for action_index in plan.action_indices.iter().skip(1) {
        let action = action_at(actions, *action_index)?;
        let Action::Sell {
            market_name,
            amount,
            ..
        } = action
        else {
            return Err(TxBuildError::UnsupportedGroupShape {
                kind: plan.kind,
                reason: "MintSell tail must contain only Sell actions",
            });
        };
        let leg = leg_for_action(plan, super::LegKind::Sell, *action_index)?;
        let token_in = outcome_token_for_market(*market_name)?;
        let amount_in_wei = amount_to_wei_floor(*amount)?;
        let min_proceeds_wei = min_proceeds_for_leg(plan, leg)?;
        let sqrt_price_limit_x96 = price_limit_for_leg(plan, leg)?;
        swap_calls.push(build_exact_input_single_call(
            executor,
            token_in,
            amount_in_wei,
            min_proceeds_wei,
            sqrt_price_limit_x96,
        ));
    }
    if swap_calls.is_empty() {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "MintSell requires at least one sell leg",
        });
    }

    let split_market_1 = ICTFRouter::splitPositionCall {
        collateralToken: BASE_COLLATERAL,
        market: MARKET_1_ADDRESS,
        amount: mint_amount_wei,
    }
    .abi_encode();
    // Action::Mint semantics are cross-contract complete-set minting, so both
    // market splits are intentional even if this subgroup only sells a subset.
    let split_market_2 = ICTFRouter::splitPositionCall {
        collateralToken: MARKET_2_COLLATERAL,
        market: MARKET_2_ADDRESS,
        amount: mint_amount_wei,
    }
    .abi_encode();
    let mut calls = vec![
        executor_call(CTF_ROUTER_ADDRESS, split_market_1),
        executor_call(CTF_ROUTER_ADDRESS, split_market_2),
    ];
    calls.extend(swap_calls);
    Ok(calls)
}

fn build_buy_merge_calls(
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let _ = expect_buy_bounds(plan.kind, batch_bounds)?;
    if plan.action_indices.len() < 2 {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "BuyMerge must contain one or more Buy actions then Merge",
        });
    }

    let merge_index = *plan.action_indices.last().expect("length checked");
    let merge_action = action_at(actions, merge_index)?;
    let Action::Merge { amount, .. } = merge_action else {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "BuyMerge must end with Merge action",
        });
    };
    let merge_amount_wei = amount_to_wei_floor(*amount)?;

    let mut swap_calls = Vec::new();
    for action_index in plan
        .action_indices
        .iter()
        .take(plan.action_indices.len() - 1)
    {
        let action = action_at(actions, *action_index)?;
        let Action::Buy {
            market_name,
            amount,
            ..
        } = action
        else {
            return Err(TxBuildError::UnsupportedGroupShape {
                kind: plan.kind,
                reason: "BuyMerge prefix must contain only Buy actions",
            });
        };
        let leg = leg_for_action(plan, super::LegKind::Buy, *action_index)?;
        let token_out = outcome_token_for_market(*market_name)?;
        let amount_out_wei = amount_to_wei_ceil(*amount)?;
        let max_cost_wei = max_cost_for_leg(plan, leg)?;
        let sqrt_price_limit_x96 = price_limit_for_leg(plan, leg)?;
        swap_calls.push(build_exact_output_single_call(
            executor,
            token_out,
            amount_out_wei,
            max_cost_wei,
            sqrt_price_limit_x96,
        ));
    }
    if swap_calls.is_empty() {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "BuyMerge requires at least one buy leg",
        });
    }
    // Action::Merge semantics are cross-contract complete-set merge/burn.
    let merge_market_2 = ICTFRouter::mergePositionsCall {
        collateralToken: MARKET_2_COLLATERAL,
        market: MARKET_2_ADDRESS,
        amount: merge_amount_wei,
    }
    .abi_encode();
    let merge_market_1 = ICTFRouter::mergePositionsCall {
        collateralToken: BASE_COLLATERAL,
        market: MARKET_1_ADDRESS,
        amount: merge_amount_wei,
    }
    .abi_encode();

    let mut calls = swap_calls;
    calls.extend([
        executor_call(CTF_ROUTER_ADDRESS, merge_market_2),
        executor_call(CTF_ROUTER_ADDRESS, merge_market_1),
    ]);
    Ok(calls)
}

fn build_direct_merge_calls(
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    if batch_bounds.is_some() {
        return Err(TxBuildError::UnexpectedBounds { kind: plan.kind });
    }
    if plan.action_indices.len() != 1 {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "DirectMerge must contain exactly one Merge action",
        });
    }

    let action = action_at(actions, plan.action_indices[0])?;
    let Action::Merge { amount, .. } = action else {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "DirectMerge must contain only Merge action",
        });
    };
    let merge_amount_wei = amount_to_wei_floor(*amount)?;

    let merge_market_2 = ICTFRouter::mergePositionsCall {
        collateralToken: MARKET_2_COLLATERAL,
        market: MARKET_2_ADDRESS,
        amount: merge_amount_wei,
    }
    .abi_encode();
    let merge_market_1 = ICTFRouter::mergePositionsCall {
        collateralToken: BASE_COLLATERAL,
        market: MARKET_1_ADDRESS,
        amount: merge_amount_wei,
    }
    .abi_encode();

    Ok(vec![
        executor_call(CTF_ROUTER_ADDRESS, merge_market_2),
        executor_call(CTF_ROUTER_ADDRESS, merge_market_1),
    ])
}

fn expect_buy_bounds(
    kind: GroupKind,
    bounds: Option<BatchTokenBounds>,
) -> Result<U256, TxBuildError> {
    let Some(bounds) = bounds else {
        return Err(TxBuildError::MissingBounds { kind });
    };
    let BatchTokenBounds::Buy {
        max_total_in_wei, ..
    } = bounds
    else {
        return Err(TxBuildError::BoundsKindMismatch {
            kind,
            expected: "Buy",
        });
    };
    Ok(max_total_in_wei)
}

fn expect_sell_bounds(
    kind: GroupKind,
    bounds: Option<BatchTokenBounds>,
) -> Result<U256, TxBuildError> {
    let Some(bounds) = bounds else {
        return Err(TxBuildError::MissingBounds { kind });
    };
    let BatchTokenBounds::Sell {
        min_total_out_wei, ..
    } = bounds
    else {
        return Err(TxBuildError::BoundsKindMismatch {
            kind,
            expected: "Sell",
        });
    };
    Ok(min_total_out_wei)
}

fn action_at(actions: &[Action], index: usize) -> Result<&Action, TxBuildError> {
    actions
        .get(index)
        .ok_or(TxBuildError::ActionIndexOutOfBounds { index })
}

fn outcome_token_for_market(market_name: &'static str) -> Result<Address, TxBuildError> {
    let market = match MARKETS_L1.iter().find(|market| market.name == market_name) {
        Some(market) => market,
        None => {
            #[cfg(test)]
            {
                // Synthetic benchmark fixtures use test-only market names that do not
                // exist in the static L1 registry. For exact tx-byte fee diagnostics,
                // a deterministic pseudo-address preserves calldata entropy and size
                // without weakening runtime safety in non-test builds.
                let hash = alloy::primitives::keccak256(market_name.as_bytes());
                return Ok(Address::from_slice(&hash[12..]));
            }
            #[cfg(not(test))]
            {
                return Err(TxBuildError::UnknownMarket { market_name });
            }
        }
    };
    Address::from_str(market.outcome_token).map_err(|_| TxBuildError::InvalidMarketTokenAddress {
        market_name,
        raw: market.outcome_token.to_string(),
    })
}

fn amount_to_wei_floor(amount: f64) -> Result<U256, TxBuildError> {
    Ok(quote_to_u256_floor(amount, TOKEN_DECIMALS)?)
}

fn amount_to_wei_ceil(amount: f64) -> Result<U256, TxBuildError> {
    Ok(quote_to_u256_ceil(amount, TOKEN_DECIMALS)?)
}

fn executor_call(to: Address, data: Vec<u8>) -> ITradeExecutor::Call {
    ITradeExecutor::Call {
        to,
        data: data.into(),
    }
}

#[cfg(test)]
mod tests {
    use alloy::primitives::U256;

    use super::*;
    use crate::execution::{BatchTokenBounds, ExecutionGroupPlan, ExecutionLegPlan, LegKind};

    fn plan(kind: GroupKind, action_indices: Vec<usize>) -> ExecutionGroupPlan {
        let leg_kind = match kind {
            GroupKind::DirectSell | GroupKind::MintSell => LegKind::Sell,
            _ => LegKind::Buy,
        };
        ExecutionGroupPlan {
            kind,
            action_indices,
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some("m"),
                kind: leg_kind,
                planned_quote_susd: 1.0,
                conservative_quote_susd: 1.01,
                adverse_notional_susd: 1.0,
                allocated_slippage_susd: 0.01,
                max_cost_susd: (leg_kind == LegKind::Buy).then_some(1.01),
                min_proceeds_susd: (leg_kind == LegKind::Sell).then_some(0.99),
                sqrt_price_limit_x96: Some(U160::from(123u64)),
            }],
            planned_at_block: Some(100),
            edge_plan_susd: 1.0,
            l2_gas_units: 1,
            gas_l2_susd: 0.01,
            gas_total_susd: 0.02,
            profit_buffer_susd: 0.01,
            slippage_budget_susd: 0.05,
            guaranteed_profit_floor_susd: 0.01,
        }
    }

    fn first_two_markets() -> (&'static str, &'static str) {
        let mut iter = MARKETS_L1
            .iter()
            .filter(|market| market.pool.is_some() && !market.outcome_token.is_empty());
        let first = iter.next().expect("expected first market").name;
        let second = iter.next().expect("expected second market").name;
        (first, second)
    }

    #[test]
    fn buy_merge_preserves_per_leg_buy_amounts() {
        let (m0, m1) = first_two_markets();
        let actions = vec![
            Action::Buy {
                market_name: m0,
                amount: 1.25,
                cost: 0.8,
            },
            Action::Buy {
                market_name: m1,
                amount: 2.75,
                cost: 1.7,
            },
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.25,
                source_market: m0,
            },
        ];
        let mut plan = plan(GroupKind::BuyMerge, vec![0, 1, 2]);
        plan.legs.push(ExecutionLegPlan {
            action_index: 1,
            market_name: Some("m"),
            kind: LegKind::Buy,
            planned_quote_susd: 1.0,
            conservative_quote_susd: 1.01,
            adverse_notional_susd: 1.0,
            allocated_slippage_susd: 0.01,
            max_cost_susd: Some(1.01),
            min_proceeds_susd: None,
            sqrt_price_limit_x96: Some(U160::from(123u64)),
        });
        let bounds = Some(BatchTokenBounds::Buy {
            planned_total_in_wei: U256::from(2u64),
            max_total_in_wei: U256::from(3u64),
        });

        let calls = build_trade_executor_calls(Address::ZERO, &actions, &plan, bounds)
            .expect("build should succeed");
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0].to, SWAP_ROUTER);
        let decoded = IV3SwapRouter::exactOutputSingleCall::abi_decode(&calls[0].data)
            .expect("decode should succeed");
        assert_eq!(
            decoded.params.amountOut,
            U256::from(1_250_000_000_000_000_000u128)
        );
        assert_eq!(
            decoded.params.amountInMaximum,
            U256::from(1_010_000_000_000_000_000u128)
        );
        assert_eq!(decoded.params.sqrtPriceLimitX96, U160::from(123u64));
        assert_eq!(decoded.params.recipient, Address::ZERO);
        let decoded_second = IV3SwapRouter::exactOutputSingleCall::abi_decode(&calls[1].data)
            .expect("decode should succeed");
        assert_ne!(
            decoded.params.amountOut, decoded_second.params.amountOut,
            "heterogeneous buy amounts must be preserved across per-leg swap calls"
        );
        assert_eq!(calls[2].to, CTF_ROUTER_ADDRESS);
        assert_eq!(calls[3].to, CTF_ROUTER_ADDRESS);
    }

    #[test]
    fn direct_merge_rejects_unexpected_bounds() {
        let actions = vec![Action::Merge {
            contract_1: "c1",
            contract_2: "c2",
            amount: 1.0,
            source_market: "m",
        }];
        let plan = plan(GroupKind::DirectMerge, vec![0]);
        let err = match build_trade_executor_calls(
            Address::ZERO,
            &actions,
            &plan,
            Some(BatchTokenBounds::Sell {
                planned_total_out_wei: U256::from(1u64),
                min_total_out_wei: U256::from(1u64),
            }),
        ) {
            Ok(_) => panic!("unexpected bounds must fail closed"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            TxBuildError::UnexpectedBounds {
                kind: GroupKind::DirectMerge
            }
        ));
    }

    #[test]
    fn direct_buy_requires_buy_bounds() {
        let (market, _) = first_two_markets();
        let actions = vec![Action::Buy {
            market_name: market,
            amount: 1.0,
            cost: 0.5,
        }];
        let plan = plan(GroupKind::DirectBuy, vec![0]);
        let err = match build_trade_executor_calls(Address::ZERO, &actions, &plan, None) {
            Ok(_) => panic!("missing bounds must fail closed"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            TxBuildError::MissingBounds {
                kind: GroupKind::DirectBuy
            }
        ));
    }

    #[test]
    fn direct_sell_emits_single_swap_router_call() {
        let (market, _) = first_two_markets();
        let actions = vec![Action::Sell {
            market_name: market,
            amount: 1.0,
            proceeds: 0.4,
        }];
        let plan = plan(GroupKind::DirectSell, vec![0]);
        let bounds = Some(BatchTokenBounds::Sell {
            planned_total_out_wei: U256::from(4u64),
            min_total_out_wei: U256::from(3u64),
        });
        let calls = build_trade_executor_calls(Address::ZERO, &actions, &plan, bounds)
            .expect("build should succeed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].to, SWAP_ROUTER);
        let decoded = IV3SwapRouter::exactInputSingleCall::abi_decode(&calls[0].data)
            .expect("decode should succeed");
        assert_eq!(
            decoded.params.amountIn,
            U256::from(1_000_000_000_000_000_000u128)
        );
        assert_eq!(decoded.params.sqrtPriceLimitX96, U160::from(123u64));
        assert_eq!(
            decoded.params.amountOutMinimum,
            U256::from(990_000_000_000_000_000u128)
        );
        assert_eq!(decoded.params.tokenOut, BASE_COLLATERAL);
        assert_eq!(decoded.params.recipient, Address::ZERO);
    }

    #[test]
    fn direct_buy_fails_closed_when_price_limit_missing() {
        let (market, _) = first_two_markets();
        let actions = vec![Action::Buy {
            market_name: market,
            amount: 1.0,
            cost: 0.5,
        }];
        let mut plan = plan(GroupKind::DirectBuy, vec![0]);
        plan.legs[0].sqrt_price_limit_x96 = None;
        let bounds = Some(BatchTokenBounds::Buy {
            planned_total_in_wei: U256::from(1u64),
            max_total_in_wei: U256::from(2u64),
        });
        let err = match build_trade_executor_calls(Address::ZERO, &actions, &plan, bounds) {
            Ok(_) => panic!("missing price limit must fail closed"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            TxBuildError::MissingPriceLimit {
                kind: GroupKind::DirectBuy,
                action_index: 0
            }
        ));
    }

    #[test]
    fn conversion_rounding_policy_is_directional() {
        let amount = 1.0 / 3.0;
        let floor = amount_to_wei_floor(amount).expect("floor conversion");
        let ceil = amount_to_wei_ceil(amount).expect("ceil conversion");
        assert!(ceil >= floor);
        assert!(ceil - floor <= U256::from(1u8));
        assert_eq!(
            quote_to_u256_floor(amount, TOKEN_DECIMALS).expect("quote floor"),
            floor
        );
    }
}
