use alloy_primitives::{I256, U256};
use uniswap_v3_math::{swap_math::compute_swap_step, tick_math::get_sqrt_ratio_at_tick};

use crate::markets::Pool;

/// Fee tier in pips (100 = 0.01%)
pub(crate) const FEE_PIPS: u32 = 100;

#[derive(Debug)]
pub enum SwapSimulationError {
    MissingTicks,
    InvalidLiquidity,
    Math(uniswap_v3_math::error::UniswapV3MathError),
}

impl std::fmt::Display for SwapSimulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTicks => write!(f, "pool is missing required tick bounds"),
            Self::InvalidLiquidity => write!(f, "pool liquidity is missing, invalid, or zero"),
            Self::Math(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for SwapSimulationError {}

impl From<uniswap_v3_math::error::UniswapV3MathError> for SwapSimulationError {
    fn from(value: uniswap_v3_math::error::UniswapV3MathError) -> Self {
        Self::Math(value)
    }
}

/// Swap result containing amounts and new price state
#[derive(Debug, Clone)]
pub struct SwapResult {
    pub amount_in: U256,
    pub amount_out: U256,
    pub sqrt_price_next: U256,
    pub fee_amount: U256,
    pub crossed_tick: bool,
}

/// Simulates a swap using the pool's current state.
///
/// * `amount` - Positive for exact input, negative for exact output
/// * `zero_for_one` - true if swapping token0 for token1 (price decreases)
pub fn simulate_swap(
    pool: &Pool,
    sqrt_price_x96: U256,
    amount: I256,
    zero_for_one: bool,
) -> Result<SwapResult, SwapSimulationError> {
    // we don't know which tick is low or high
    let tick_0 = pool
        .ticks
        .first()
        .ok_or(SwapSimulationError::MissingTicks)?;
    let tick_1 = pool.ticks.get(1).ok_or(SwapSimulationError::MissingTicks)?;
    let liquidity: u128 = pool
        .liquidity
        .parse()
        .map_err(|_| SwapSimulationError::InvalidLiquidity)?;
    if liquidity == 0 {
        return Err(SwapSimulationError::InvalidLiquidity);
    }

    // Target price is the tick boundary in the swap direction
    let target_tick = if zero_for_one {
        tick_0.tick_idx.min(tick_1.tick_idx)
    } else {
        tick_0.tick_idx.max(tick_1.tick_idx)
    };
    let sqrt_price_target =
        get_sqrt_ratio_at_tick(target_tick).map_err(SwapSimulationError::Math)?;

    let (sqrt_price_next, amount_in, amount_out, fee_amount) = compute_swap_step(
        sqrt_price_x96,
        sqrt_price_target,
        liquidity,
        amount,
        FEE_PIPS,
    )
    .map_err(SwapSimulationError::Math)?;

    // Check if we hit the tick boundary
    let crossed_tick = sqrt_price_next == sqrt_price_target;

    Ok(SwapResult {
        amount_in,
        amount_out,
        sqrt_price_next,
        fee_amount,
        crossed_tick,
    })
}

/// Convenience: simulate buying outcome tokens with quote tokens (exact input)
pub fn simulate_buy(
    pool: &Pool,
    sqrt_price_x96: U256,
    quote_token: &str,
    amount: I256,
) -> Result<SwapResult, SwapSimulationError> {
    // zero_for_one means selling token0 for token1
    // If quote is token0, we're selling quote (token0) to buy outcome (token1) -> zero_for_one = true
    // If quote is token1, we're selling quote (token1) to buy outcome (token0) -> zero_for_one = false
    let zero_for_one = pool.token0.eq_ignore_ascii_case(quote_token);
    simulate_swap(pool, sqrt_price_x96, amount, zero_for_one)
}

#[cfg(test)]
mod tests {
    use alloy_primitives::{I256, U256};

    use super::{SwapSimulationError, simulate_swap};
    use crate::markets::{Pool, Tick};

    static EMPTY_TICKS: [Tick; 0] = [];
    static TWO_TICKS: [Tick; 2] = [
        Tick {
            tick_idx: -1,
            liquidity_net: 100,
        },
        Tick {
            tick_idx: 1,
            liquidity_net: -100,
        },
    ];

    #[test]
    fn simulate_swap_returns_error_when_pool_has_missing_ticks() {
        let pool = Pool {
            token0: "0x0",
            token1: "0x1",
            pool_id: "0x2",
            liquidity: "1000",
            ticks: &EMPTY_TICKS,
        };
        let amount = I256::try_from(1u8).expect("test amount should fit into I256");
        let result = simulate_swap(&pool, U256::from(1u8), amount, true);
        assert!(matches!(result, Err(SwapSimulationError::MissingTicks)));
    }

    #[test]
    fn simulate_swap_returns_error_when_pool_liquidity_is_invalid() {
        let pool = Pool {
            token0: "0x0",
            token1: "0x1",
            pool_id: "0x2",
            liquidity: "not-a-number",
            ticks: &TWO_TICKS,
        };
        let amount = I256::try_from(1u8).expect("test amount should fit into I256");
        let result = simulate_swap(&pool, U256::from(1u8), amount, true);
        assert!(matches!(result, Err(SwapSimulationError::InvalidLiquidity)));
    }
}
