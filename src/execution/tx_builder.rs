use std::fmt;
use std::str::FromStr;

use alloy::primitives::{Address, U160, U256};
use alloy::sol_types::SolCall;
use alloy_primitives::aliases::U24;

use super::{
    BASE_COLLATERAL, BATCH_SWAP_ROUTER_ADDRESS as BATCH_ROUTER, BatchTokenBounds,
    CTF_ROUTER_ADDRESS, IBatchSwapRouter, ICTFRouter, ITradeExecutor, MARKET_1_ADDRESS,
    MARKET_2_ADDRESS, MARKET_2_COLLATERAL, QuoteAmountConversionError, quote_to_u256_ceil,
    quote_to_u256_floor,
};
use crate::markets::MARKETS_L1;
use crate::portfolio::Action;
use crate::{execution::ExecutionGroupPlan, execution::GroupKind};

const TOKEN_DECIMALS: u8 = 18;
const SWAP_FEE_TIER: u16 = 500;
const SQRT_PRICE_LIMIT_X96: U160 = U160::ZERO;

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
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    match plan.kind {
        GroupKind::DirectBuy => build_direct_buy_calls(actions, plan, batch_bounds),
        GroupKind::DirectSell => build_direct_sell_calls(actions, plan, batch_bounds),
        GroupKind::MintSell => build_mint_sell_calls(actions, plan, batch_bounds),
        GroupKind::BuyMerge => build_buy_merge_calls(actions, plan, batch_bounds),
        GroupKind::DirectMerge => build_direct_merge_calls(actions, plan, batch_bounds),
    }
}

fn build_direct_buy_calls(
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let max_total_in_wei = expect_buy_bounds(plan.kind, batch_bounds)?;
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
    let calldata = IBatchSwapRouter::exactOutputCall {
        tokenOuts: vec![token_out],
        amountOut: vec![amount_out_wei],
        tokenIn: BASE_COLLATERAL,
        amountInTotalMax: max_total_in_wei,
        fee: U24::from(SWAP_FEE_TIER),
        sqrtPriceLimitX96: SQRT_PRICE_LIMIT_X96,
    }
    .abi_encode();

    Ok(vec![executor_call(BATCH_ROUTER, calldata)])
}

fn build_direct_sell_calls(
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let min_total_out_wei = expect_sell_bounds(plan.kind, batch_bounds)?;
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
    let calldata = IBatchSwapRouter::exactInputCall {
        tokenIns: vec![token_in],
        amountIn: vec![amount_in_wei],
        tokenOut: BASE_COLLATERAL,
        amountOutTotalMinimum: min_total_out_wei,
        fee: U24::from(SWAP_FEE_TIER),
        sqrtPriceLimitX96: SQRT_PRICE_LIMIT_X96,
    }
    .abi_encode();

    Ok(vec![executor_call(BATCH_ROUTER, calldata)])
}

fn build_mint_sell_calls(
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let min_total_out_wei = expect_sell_bounds(plan.kind, batch_bounds)?;
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

    let mut token_ins = Vec::new();
    let mut amount_ins = Vec::new();
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
        token_ins.push(outcome_token_for_market(*market_name)?);
        amount_ins.push(amount_to_wei_floor(*amount)?);
    }
    if token_ins.is_empty() {
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
    let batch_sell = IBatchSwapRouter::exactInputCall {
        tokenIns: token_ins,
        amountIn: amount_ins,
        tokenOut: BASE_COLLATERAL,
        amountOutTotalMinimum: min_total_out_wei,
        fee: U24::from(SWAP_FEE_TIER),
        sqrtPriceLimitX96: SQRT_PRICE_LIMIT_X96,
    }
    .abi_encode();

    Ok(vec![
        executor_call(CTF_ROUTER_ADDRESS, split_market_1),
        executor_call(CTF_ROUTER_ADDRESS, split_market_2),
        executor_call(BATCH_ROUTER, batch_sell),
    ])
}

fn build_buy_merge_calls(
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    batch_bounds: Option<BatchTokenBounds>,
) -> Result<Vec<ITradeExecutor::Call>, TxBuildError> {
    let max_total_in_wei = expect_buy_bounds(plan.kind, batch_bounds)?;
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

    let mut token_outs = Vec::new();
    let mut amount_outs = Vec::new();
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
        token_outs.push(outcome_token_for_market(*market_name)?);
        amount_outs.push(amount_to_wei_ceil(*amount)?);
    }
    if token_outs.is_empty() {
        return Err(TxBuildError::UnsupportedGroupShape {
            kind: plan.kind,
            reason: "BuyMerge requires at least one buy leg",
        });
    }

    let batch_buy = IBatchSwapRouter::exactOutputCall {
        tokenOuts: token_outs,
        amountOut: amount_outs,
        tokenIn: BASE_COLLATERAL,
        amountInTotalMax: max_total_in_wei,
        fee: U24::from(SWAP_FEE_TIER),
        sqrtPriceLimitX96: SQRT_PRICE_LIMIT_X96,
    }
    .abi_encode();
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

    Ok(vec![
        executor_call(BATCH_ROUTER, batch_buy),
        executor_call(CTF_ROUTER_ADDRESS, merge_market_2),
        executor_call(CTF_ROUTER_ADDRESS, merge_market_1),
    ])
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
    let Some(market) = MARKETS_L1.iter().find(|market| market.name == market_name) else {
        return Err(TxBuildError::UnknownMarket { market_name });
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
        ExecutionGroupPlan {
            kind,
            action_indices,
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some("m"),
                kind: LegKind::Buy,
                planned_quote_susd: 1.0,
                adverse_notional_susd: 1.0,
                allocated_slippage_susd: 0.01,
                max_cost_susd: Some(1.01),
                min_proceeds_susd: None,
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
    fn buy_merge_uses_heterogeneous_array_amounts() {
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
        let plan = plan(GroupKind::BuyMerge, vec![0, 1, 2]);
        let bounds = Some(BatchTokenBounds::Buy {
            planned_total_in_wei: U256::from(2u64),
            max_total_in_wei: U256::from(3u64),
        });

        let calls =
            build_trade_executor_calls(&actions, &plan, bounds).expect("build should succeed");
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].to, BATCH_ROUTER);
        let decoded = IBatchSwapRouter::exactOutputCall::abi_decode(&calls[0].data)
            .expect("decode should succeed");
        assert_eq!(decoded.amountOut.len(), 2);
        assert_ne!(
            decoded.amountOut[0], decoded.amountOut[1],
            "heterogeneous buy amounts must be preserved in calldata arrays"
        );
        assert_eq!(decoded.tokenIn, BASE_COLLATERAL);
        assert_eq!(decoded.amountInTotalMax, U256::from(3u64));
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
        let err = match build_trade_executor_calls(&actions, &plan, None) {
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
    fn direct_sell_emits_single_batch_router_call() {
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
        let calls =
            build_trade_executor_calls(&actions, &plan, bounds).expect("build should succeed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].to, BATCH_ROUTER);
        let decoded = IBatchSwapRouter::exactInputCall::abi_decode(&calls[0].data)
            .expect("decode should succeed");
        assert_eq!(decoded.amountOutTotalMinimum, U256::from(3u64));
        assert_eq!(decoded.tokenOut, BASE_COLLATERAL);
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
