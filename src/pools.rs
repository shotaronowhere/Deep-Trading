use alloy::{
    primitives::{Address, Bytes},
    providers::Provider,
    sol,
    sol_types::SolCall,
};
use alloy_primitives::{I256, U256};
use futures_util::future;
use std::str::FromStr;
use uniswap_v3_math::{swap_math::compute_swap_step, tick, tick_math::get_sqrt_ratio_at_tick};

use crate::markets::{MARKETS_L1, MarketData, Pool};

// Multicall3 contract interface
sol! {
    #[sol(rpc)]
    contract Multicall3 {
        struct Call3 {
            address target;
            bool allowFailure;
            bytes callData;
        }

        struct Result {
            bool success;
            bytes returnData;
        }

        function aggregate3(Call3[] calldata calls) external payable returns (Result[] memory returnData);
    }
}

// slot0 function for encoding/decoding
sol! {
    function slot0() external view returns (
        uint160 sqrtPriceX96,
        int24 tick,
        uint16 observationIndex,
        uint16 observationCardinality,
        uint16 observationCardinalityNext,
        uint8 feeProtocol,
        bool unlocked
    );
}

// Multicall3 address (same on all EVM chains)
const MULTICALL3_ADDRESS: Address =
    alloy::primitives::address!("cA11bde05977b3631167028862bE2a173976CA11");
const MULTICALL_BATCH_SIZE: usize = 200;

/// Returns (base_token, quote_token) addresses for price calculation
pub fn base_quote_tokens(market: &MarketData) -> (Address, Address) {
    let pool = market.pool.as_ref().unwrap();
    let token0 = Address::from_str(pool.token0).unwrap();
    let token1 = Address::from_str(pool.token1).unwrap();
    let quote_token = Address::from_str(market.quote_token).unwrap();
    if token0 == quote_token {
        (token1, token0)
    } else {
        (token0, token1)
    }
}

/// Result of a slot0 query for a single pool
#[derive(Debug, Clone)]
pub struct Slot0Result {
    pub pool_id: Address,
    pub sqrt_price_x96: U256,
    pub tick: i32,
    pub observation_index: u16,
    pub observation_cardinality: u16,
    pub observation_cardinality_next: u16,
    pub fee_protocol: u8,
    pub unlocked: bool,
}

/// Fee tier in pips (100 = 0.01%)
const FEE_PIPS: u32 = 100;

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
    let tick_0 = pool.ticks.get(0).unwrap();
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
pub fn simulate_swap_direction(
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

/// Collects all markets with valid pools from MARKETS_L1, with pre-parsed pool_id
fn collect_markets_with_pools() -> Vec<(Address, &'static MarketData)> {
    MARKETS_L1
        .iter()
        .filter_map(|m| {
            let pool = m.pool.as_ref()?;
            let pool_id = Address::from_str(pool.pool_id).ok()?;
            Some((pool_id, m))
        })
        .collect()
}

/// Parses the slot0 return data from a multicall result
fn parse_slot0_result(pool_id: Address, result: &Multicall3::Result) -> Option<Slot0Result> {
    if !result.success || result.returnData.is_empty() {
        return None;
    }

    let decoded = slot0Call::abi_decode_returns(&result.returnData).ok()?;

    Some(Slot0Result {
        pool_id,
        sqrt_price_x96: U256::from(decoded.sqrtPriceX96),
        tick: decoded.tick.as_i32(),
        observation_index: decoded.observationIndex,
        observation_cardinality: decoded.observationCardinality,
        observation_cardinality_next: decoded.observationCardinalityNext,
        fee_protocol: decoded.feeProtocol,
        unlocked: decoded.unlocked,
    })
}

/// Fetches slot0 data for all pools in MARKETS_L1 using batched multicall
/// Returns tuple of (Slot0Result, &MarketData) for each successful fetch
pub async fn fetch_all_slot0<P: Provider + Clone>(
    provider: P,
) -> Result<Vec<(Slot0Result, &'static MarketData)>, Box<dyn std::error::Error>> {
    let markets = collect_markets_with_pools();
    if markets.is_empty() {
        return Ok(Vec::new());
    }

    let slot0_calldata = Bytes::from(slot0Call {}.abi_encode());
    let calls: Vec<Multicall3::Call3> = markets
        .iter()
        .map(|(pool_id, _)| Multicall3::Call3 {
            target: *pool_id,
            allowFailure: true,
            callData: slot0_calldata.clone(),
        })
        .collect();

    let multicall = Multicall3::new(MULTICALL3_ADDRESS, provider);

    // Execute all batches concurrently
    let batch_futures: Vec<_> = calls
        .chunks(MULTICALL_BATCH_SIZE)
        .map(|chunk| {
            let call = multicall.aggregate3(chunk.to_vec());
            async move { call.call().await }
        })
        .collect();

    let all_results = future::try_join_all(batch_futures).await?;

    // Process results
    let mut slot0_results = Vec::with_capacity(markets.len());
    for (results_batch, markets_batch) in all_results
        .into_iter()
        .zip(markets.chunks(MULTICALL_BATCH_SIZE))
    {
        for (result, (pool_id, market)) in results_batch.iter().zip(markets_batch.iter()) {
            match parse_slot0_result(*pool_id, result) {
                Some(slot0_data) => slot0_results.push((slot0_data, *market)),
                None => eprintln!("Failed to fetch slot0 for pool {}", pool_id),
            }
        }
    }

    Ok(slot0_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::providers::ProviderBuilder;

    #[tokio::test]
    async fn test_fetch_all_slot0() {
        dotenvy::dotenv().ok();
        let rpc_url = std::env::var("RPC").expect("RPC environment variable not set");
        let provider = ProviderBuilder::new().connect_http(rpc_url.parse().unwrap());

        let results = fetch_all_slot0(provider).await.unwrap();

        println!("Fetched {} pool slot0 results", results.len());
        for (slot0, market) in results.iter().take(10) {
            let pool = market.pool.as_ref().unwrap();

            println!(
                "Pool {}: tick={}, token0={}, token1={}, sqrtPriceX96={}, liquidity={}",
                slot0.pool_id,
                slot0.tick,
                pool.token0,
                pool.token1,
                slot0.sqrt_price_x96,
                pool.liquidity
            );

            // Simulate buying outcome with 1e15 quote tokens (exact input)
            let result = simulate_swap_direction(
                pool,
                slot0.sqrt_price_x96,
                market.quote_token,
                I256::try_from(-1_000_000_000_000_000i128).unwrap(),
            )
            .unwrap();
            println!(
                "Swap 1e15 quote: amount_in(zeikomi)={}, amount_out={}, crossed={}",
                result.amount_in + result.fee_amount,
                result.amount_out,
                result.crossed_tick
            );
            // print outcome token and quote token addresses
            println!(
                "    outcome_token={}, quote_token={}",
                market.outcome_token, market.quote_token
            );
        }
        assert!(!results.is_empty());
    }

    #[test]
    fn test_collect_markets_with_pools() {
        let markets = collect_markets_with_pools();
        println!("Collected {} markets with pools", markets.len());
        assert!(!markets.is_empty());
    }
}
