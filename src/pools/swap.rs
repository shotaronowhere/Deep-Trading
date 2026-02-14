use alloy_primitives::{I256, U256};
use uniswap_v3_math::{swap_math::compute_swap_step, tick_math::get_sqrt_ratio_at_tick};

use crate::markets::Pool;

/// Fee tier in pips (100 = 0.01%)
pub(crate) const FEE_PIPS: u32 = 100;

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
) -> Result<SwapResult, uniswap_v3_math::error::UniswapV3MathError> {
    // we don't know which tick is low or high
    let tick_0 = pool.ticks.first().unwrap();
    let tick_1 = pool.ticks.get(1).unwrap();
    let liquidity: u128 = pool.liquidity.parse().unwrap();

    // Target price is the tick boundary in the swap direction
    let target_tick = if zero_for_one {
        tick_0.tick_idx.min(tick_1.tick_idx)
    } else {
        tick_0.tick_idx.max(tick_1.tick_idx)
    };
    let sqrt_price_target = get_sqrt_ratio_at_tick(target_tick)?;

    let (sqrt_price_next, amount_in, amount_out, fee_amount) = compute_swap_step(
        sqrt_price_x96,
        sqrt_price_target,
        liquidity,
        amount,
        FEE_PIPS,
    )?;

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
) -> Result<SwapResult, uniswap_v3_math::error::UniswapV3MathError> {
    // zero_for_one means selling token0 for token1
    // If quote is token0, we're selling quote (token0) to buy outcome (token1) -> zero_for_one = true
    // If quote is token1, we're selling quote (token1) to buy outcome (token0) -> zero_for_one = false
    let zero_for_one = pool.token0.to_lowercase() == quote_token.to_lowercase();
    simulate_swap(pool, sqrt_price_x96, amount, zero_for_one)
}
