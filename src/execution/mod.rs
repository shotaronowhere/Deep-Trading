use alloy::primitives::{Address, U256, address};
use alloy::sol;

pub mod approvals;
mod batch_bounds;
pub mod bounds;
mod edge;
pub mod gas;
pub mod grouping;
pub mod runtime;
pub mod tx_builder;

// https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Router.sol
sol! {
    #[sol(rpc)]
    interface ICTFRouter {
        function splitPosition(address collateralToken, address market, uint256 amount) external;
        function mergePositions(address collateralToken, address market, uint256 amount) external;
    }
}

pub const CTF_ROUTER_ADDRESS: Address = address!("179d8F8c811B8C759c33809dbc6c5ceDc62D05DD");

// https://github.com/Uniswap/swap-router-contracts/blob/70bc2e40dfca294c1cea9bf67a4036732ee54303/contracts/interfaces/IV3SwapRouter.sol
sol! {
    #[sol(rpc)]
    interface IV3SwapRouter {
        struct ExactInputSingleParams {
            address tokenIn;
            address tokenOut;
            uint24 fee;
            address recipient;
            uint256 amountIn;
            uint256 amountOutMinimum;
            uint160 sqrtPriceLimitX96;
        }

        // Sell: spend exact tokenIn, receive at least amountOutMinimum of tokenOut.
        // Action::Sell → exactInputSingle(outcome_token → sUSDS, min_proceeds_wei)
        function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);

        struct ExactOutputSingleParams {
            address tokenIn;
            address tokenOut;
            uint24 fee;
            address recipient;
            uint256 amountOut;
            uint256 amountInMaximum;
            uint160 sqrtPriceLimitX96;
        }

        // Buy: receive exact amountOut of tokenOut, spend at most amountInMaximum of tokenIn.
        // Action::Buy → exactOutputSingle(sUSDS → outcome_token, max_cost_wei)
        function exactOutputSingle(ExactOutputSingleParams calldata params) external payable returns (uint256 amountIn);
    }
}

/// sUSDS — base collateral for splitPosition/mergePositions on market 1.
pub const BASE_COLLATERAL: Address = address!("b5B2dc7fd34C249F4be7fB1fCea07950784229e0");

/// SwapRouter02 — direct Buy/Sell (exactOutputSingle / exactInputSingle).
pub const SWAP_ROUTER_ADDRESS: Address = address!("68b3465833fb72A70ecDF485E0e4C7bD8665Fc45");

/// Seer batch swap router — multi-leg MintSell / BuyMerge groups.
pub const BATCH_SWAP_ROUTER_ADDRESS: Address = address!("4081136d23FEeCD324a420A54635e007F51fd94a");

/// Market 1 (67 outcomes) — split with BASE_COLLATERAL (sUSDS).
pub const MARKET_1_ADDRESS: Address = address!("3220a208aaf4d2ceecde5a2e21ec0c9145f40ba6");
/// Market 2 (33 outcomes) — conditional on MARKET_2_COLLATERAL.
pub const MARKET_2_ADDRESS: Address = address!("fea47428981f70110c64dd678889826c3627245b");
/// "Other repos" ERC1155 outcome token from market 1 — collateral for market 2 splits.
pub const MARKET_2_COLLATERAL: Address = address!("63a4f76ef5846f68d069054c271465b7118e8ed9");

// https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Router.sol
sol! {
    #[sol(rpc)]
    interface IBatchSwapRouter {
        // MintSell / DirectSell: sell one or more outcome tokens for sUSDS.
        function exactInput(
            address[] memory tokenIns,
            uint256[] memory amountIn,
            address tokenOut,
            uint256 amountOutTotalMinimum,
            uint24 fee,
            uint160 sqrtPriceLimitX96
        ) external returns (uint256 amountOut);

        // BuyMerge / DirectBuy: buy one or more outcome tokens with sUSDS.
        function exactOutput(
            address[] memory tokenOuts,
            uint256[] memory amountOut,
            address tokenIn,
            uint256 amountInTotalMax,
            uint24 fee,
            uint160 sqrtPriceLimitX96
        ) external returns (uint256 amountIn);
    }
}

sol! {
    #[sol(rpc)]
    interface IERC20 {
        function approve(address spender, uint256 amount) external returns (bool);
        function allowance(address owner, address spender) external view returns (uint256);
    }
}

sol! {
    #[sol(rpc)]
    interface ITradeExecutor {
        struct Call {
            address to;
            bytes data;
        }

        function batchExecute(Call[] calldata calls) external;
        function setTemporaryPermission(address _permitted, uint256 _expire) external;
        function owner() external view returns (address);
        function permitted() external view returns (address);
        function expire() external view returns (uint256);
    }
}

// https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Market.sol
sol! {
    #[sol(rpc)]
    interface IMarket {
        // Returns the ERC1155 wrapper address for a given outcome index.
        // Used to resolve the intermediate collateral token for market 2 splits.
        function wrappedOutcome(uint256 index) external view returns (address wrapped1155, bytes memory data);
        // Returns (_, _, parentOutcome, parentMarket, _) for conditional markets.
        // parentOutcome is the index in parentMarket whose wrapped token is this market's collateral.
        function conditionalTokensParams() external view returns (bytes32, bytes32, uint256 parentOutcome, address parentMarket, bytes32);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Strict,
    Aggressive,
}

impl ExecutionMode {
    pub fn groups_per_tx(self) -> usize {
        match self {
            Self::Strict => 1,
            Self::Aggressive => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    DirectBuy,
    DirectSell,
    DirectMerge,
    MintSell,
    BuyMerge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegKind {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct ExecutionLegPlan {
    pub action_index: usize,
    pub market_name: Option<&'static str>,
    pub kind: LegKind,
    pub planned_quote_susd: f64,
    pub adverse_notional_susd: f64,
    pub allocated_slippage_susd: f64,
    pub max_cost_susd: Option<f64>,
    pub min_proceeds_susd: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ExecutionGroupPlan {
    pub kind: GroupKind,
    pub action_indices: Vec<usize>,
    pub profitability_step_index: usize,
    pub step_subgroup_index: usize,
    pub step_subgroup_count: usize,
    pub legs: Vec<ExecutionLegPlan>,
    pub planned_at_block: Option<u64>,
    pub edge_plan_susd: f64,
    pub l2_gas_units: u64,
    pub gas_l2_susd: f64,
    pub gas_total_susd: f64,
    pub profit_buffer_susd: f64,
    pub slippage_budget_susd: f64,
    pub guaranteed_profit_floor_susd: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchQuoteBounds {
    Sell {
        planned_total_out_susd: f64,
        min_total_out_susd: f64,
    },
    Buy {
        planned_total_in_susd: f64,
        max_total_in_susd: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchTokenBounds {
    Sell {
        planned_total_out_wei: U256,
        min_total_out_wei: U256,
    },
    Buy {
        planned_total_in_wei: U256,
        max_total_in_wei: U256,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteAmountConversionError {
    NonFinite,
    Negative,
    Overflow,
}

impl std::fmt::Display for QuoteAmountConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "amount is not finite"),
            Self::Negative => write!(f, "amount is negative"),
            Self::Overflow => write!(f, "amount overflow during conversion"),
        }
    }
}

impl std::error::Error for QuoteAmountConversionError {}

pub const SUSD_DECIMALS: u8 = 18;

pub fn is_plan_stale(plan: &ExecutionGroupPlan, current_block: u64, max_stale_blocks: u64) -> bool {
    let Some(planned_at_block) = plan.planned_at_block else {
        // Fail closed until caller stamps plans with a reference block.
        return true;
    };
    current_block.saturating_sub(planned_at_block) > max_stale_blocks
}

fn quote_to_u256_with_rounding(
    amount_quote: f64,
    quote_decimals: u8,
    round_up: bool,
) -> Result<U256, QuoteAmountConversionError> {
    if !amount_quote.is_finite() {
        return Err(QuoteAmountConversionError::NonFinite);
    }
    if amount_quote < 0.0 {
        return Err(QuoteAmountConversionError::Negative);
    }

    let scale = 10f64.powi(quote_decimals as i32);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(QuoteAmountConversionError::Overflow);
    }

    let scaled = amount_quote * scale;
    if !scaled.is_finite() {
        return Err(QuoteAmountConversionError::Overflow);
    }

    let rounded = if round_up {
        scaled.ceil()
    } else {
        scaled.floor()
    };
    if rounded >= 2f64.powi(128) {
        return Err(QuoteAmountConversionError::Overflow);
    }

    Ok(U256::from(rounded as u128))
}

pub fn quote_to_u256_floor(
    amount_quote: f64,
    quote_decimals: u8,
) -> Result<U256, QuoteAmountConversionError> {
    quote_to_u256_with_rounding(amount_quote, quote_decimals, false)
}

pub fn quote_to_u256_ceil(
    amount_quote: f64,
    quote_decimals: u8,
) -> Result<U256, QuoteAmountConversionError> {
    quote_to_u256_with_rounding(amount_quote, quote_decimals, true)
}

impl BatchQuoteBounds {
    pub fn to_token_bounds(
        self,
        quote_decimals: u8,
    ) -> Result<BatchTokenBounds, QuoteAmountConversionError> {
        match self {
            Self::Sell {
                planned_total_out_susd,
                min_total_out_susd,
            } => Ok(BatchTokenBounds::Sell {
                planned_total_out_wei: quote_to_u256_floor(planned_total_out_susd, quote_decimals)?,
                min_total_out_wei: quote_to_u256_floor(min_total_out_susd, quote_decimals)?,
            }),
            Self::Buy {
                planned_total_in_susd,
                max_total_in_susd,
            } => Ok(BatchTokenBounds::Buy {
                planned_total_in_wei: quote_to_u256_ceil(planned_total_in_susd, quote_decimals)?,
                max_total_in_wei: quote_to_u256_ceil(max_total_in_susd, quote_decimals)?,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_sell_min_down_and_buy_max_up() {
        let amount = 1.0 / 3.0;
        let floor =
            quote_to_u256_floor(amount, SUSD_DECIMALS).expect("floor conversion should succeed");
        let ceil =
            quote_to_u256_ceil(amount, SUSD_DECIMALS).expect("ceil conversion should succeed");
        assert!(ceil >= floor, "ceil should not be below floor");
        assert!(
            ceil - floor <= U256::from(1u8),
            "ceil/floor gap should be at most 1 wei"
        );

        let sell = BatchQuoteBounds::Sell {
            planned_total_out_susd: amount,
            min_total_out_susd: amount,
        }
        .to_token_bounds(SUSD_DECIMALS)
        .expect("sell bounds conversion should succeed");
        match sell {
            BatchTokenBounds::Sell {
                min_total_out_wei, ..
            } => assert_eq!(min_total_out_wei, floor, "sell min must round down"),
            BatchTokenBounds::Buy { .. } => panic!("expected sell token bounds"),
        }

        let buy = BatchQuoteBounds::Buy {
            planned_total_in_susd: amount,
            max_total_in_susd: amount,
        }
        .to_token_bounds(SUSD_DECIMALS)
        .expect("buy bounds conversion should succeed");
        match buy {
            BatchTokenBounds::Buy {
                max_total_in_wei, ..
            } => assert_eq!(max_total_in_wei, ceil, "buy max must round up"),
            BatchTokenBounds::Sell { .. } => panic!("expected buy token bounds"),
        }
    }

    #[test]
    fn rejects_invalid_quote_amounts() {
        assert_eq!(
            quote_to_u256_floor(f64::NAN, SUSD_DECIMALS),
            Err(QuoteAmountConversionError::NonFinite)
        );
        assert_eq!(
            quote_to_u256_ceil(f64::INFINITY, SUSD_DECIMALS),
            Err(QuoteAmountConversionError::NonFinite)
        );
        assert_eq!(
            quote_to_u256_floor(-0.1, SUSD_DECIMALS),
            Err(QuoteAmountConversionError::Negative)
        );
        assert_eq!(
            quote_to_u256_floor(1e30, SUSD_DECIMALS),
            Err(QuoteAmountConversionError::Overflow)
        );
        assert_eq!(
            quote_to_u256_floor(2f64.powi(128), 0),
            Err(QuoteAmountConversionError::Overflow)
        );
    }

    #[test]
    fn supports_non_18_decimal_quote_tokens() {
        let usdc_decimals = 6;
        let floor = quote_to_u256_floor(1.234_567_89, usdc_decimals)
            .expect("floor conversion should succeed");
        let ceil = quote_to_u256_ceil(1.234_567_89, usdc_decimals)
            .expect("ceil conversion should succeed");
        assert_eq!(floor, U256::from(1_234_567u64));
        assert_eq!(ceil, U256::from(1_234_568u64));
    }

    #[test]
    fn staleness_check_fails_closed_without_reference_block() {
        let plan = ExecutionGroupPlan {
            kind: GroupKind::DirectBuy,
            action_indices: vec![0],
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some("x"),
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
            profit_buffer_susd: 0.1,
            slippage_budget_susd: 0.7,
            guaranteed_profit_floor_susd: 0.1,
        };
        assert!(
            is_plan_stale(&plan, 100, 2),
            "unstamped plans should be treated as stale"
        );
    }

    #[test]
    fn staleness_check_respects_max_stale_blocks() {
        let mut plan = ExecutionGroupPlan {
            kind: GroupKind::DirectSell,
            action_indices: vec![0],
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some("x"),
                kind: LegKind::Sell,
                planned_quote_susd: 1.0,
                adverse_notional_susd: 1.0,
                allocated_slippage_susd: 0.1,
                max_cost_susd: None,
                min_proceeds_susd: Some(0.9),
            }],
            planned_at_block: Some(100),
            edge_plan_susd: 1.0,
            l2_gas_units: 1,
            gas_l2_susd: 0.1,
            gas_total_susd: 0.2,
            profit_buffer_susd: 0.1,
            slippage_budget_susd: 0.7,
            guaranteed_profit_floor_susd: 0.1,
        };
        assert!(
            !is_plan_stale(&plan, 102, 2),
            "plan at exact stale bound should still be fresh"
        );
        assert!(
            is_plan_stale(&plan, 103, 2),
            "plan past stale bound should be stale"
        );
        plan.planned_at_block = Some(105);
        assert!(
            !is_plan_stale(&plan, 103, 2),
            "future reference block should not trip stale check"
        );
    }
}
