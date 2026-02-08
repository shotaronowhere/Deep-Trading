use alloy::{
    primitives::{Address, Bytes},
    providers::Provider,
    sol,
    sol_types::SolCall,
};
use alloy_primitives::{I256, U256};
use futures_util::future;
use std::str::FromStr;
use uniswap_v3_math::{
    full_math::mul_div, swap_math::compute_swap_step, tick, tick_math::get_sqrt_ratio_at_tick,
};

use crate::markets::{MARKETS_L1, MarketData, Pool};
use crate::predictions::PREDICTIONS_L1;

/// Price scale factor: 10^18 (18 decimals of precision)
const PRICE_SCALE: U256 = U256::from_limbs([1_000_000_000_000_000_000u64, 0, 0, 0]);
const TWO_96: U256 = U256::from_limbs([0, 0x1_0000_0000, 0, 0]); // 2^96

/// Converts sqrtPriceX96 to price (token1/token0) scaled by 10^18.
/// Returns None on overflow (shouldn't happen with valid sqrtPriceX96).
pub fn sqrt_price_x96_to_price(sqrt_price_x96: U256) -> Option<U256> {
    // price = sqrtPriceX96² / 2^192
    // price_scaled = sqrtPriceX96² * 10^18 / 2^192
    //              = mul_div(sqrtPriceX96, sqrtPriceX96 * 10^18, 2^192)
    let scaled = sqrt_price_x96.checked_mul(PRICE_SCALE)?;
    mul_div(sqrt_price_x96, scaled, TWO_96 * TWO_96).ok()
}

/// Converts sqrtPriceX96 to inverse price (token0/token1) scaled by 10^18.
/// Returns None on overflow or division by zero.
pub fn sqrt_price_x96_to_inv_price(sqrt_price_x96: U256) -> Option<U256> {
    // inv_price = 2^192 / sqrtPriceX96²
    // inv_price_scaled = 2^192 * 10^18 / sqrtPriceX96²
    // Compute as: mul_div(2^96, 2^96, sqrtPriceX96) * 10^18 / sqrtPriceX96
    let intermediate = mul_div(TWO_96, TWO_96, sqrt_price_x96).ok()?;
    mul_div(intermediate, PRICE_SCALE, sqrt_price_x96).ok()
}

/// Returns price of outcome token in quote token, scaled by 10^18.
pub fn sqrt_price_x96_to_price_outcome(
    sqrt_price_x96: U256,
    is_token1_outcome: bool,
) -> Option<U256> {
    if is_token1_outcome {
        // outcome is token1, quote is token0 -> price = token0/token1 = 1/price
        sqrt_price_x96_to_inv_price(sqrt_price_x96)
    } else {
        // outcome is token0, quote is token1 -> price = token1/token0
        sqrt_price_x96_to_price(sqrt_price_x96)
    }
}

/// Entry in the profitability result: how much the prediction exceeds the market price.
#[derive(Debug, Clone)]
pub struct ProfitabilityEntry {
    pub market_name: &'static str,
    pub prediction: f64,
    pub market_price: f64,
    pub diff: f64,
}

/// Returns the difference between predictions (PREDICTIONS_L1) and current
/// outcome prices for each matched market. Matches by market name (case-insensitive).
///
/// `slot0_results` should come from `fetch_all_slot0`.
pub fn profitability(slot0_results: &[(Slot0Result, &MarketData)]) -> Vec<ProfitabilityEntry> {
    let mut entries = Vec::new();

    for prediction in PREDICTIONS_L1.iter() {
        let pred_name = prediction.market.to_lowercase();

        // Find the matching slot0 result by market name
        let Some((slot0, market)) = slot0_results
            .iter()
            .find(|(_, m)| m.name.trim_end_matches("\\t").to_lowercase() == pred_name)
        else {
            continue;
        };

        let pool = match market.pool.as_ref() {
            Some(p) => p,
            None => continue,
        };

        let is_token1_outcome = pool.token1.to_lowercase() == market.outcome_token.to_lowercase();

        let price = match sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome) {
            Some(p) => p,
            None => continue,
        };

        let price_f64: f64 = price.to_string().parse::<f64>().unwrap() / 1e18;
        let diff = (prediction.prediction - price_f64) / price_f64;

        entries.push(ProfitabilityEntry {
            market_name: market.name,
            prediction: prediction.prediction,
            market_price: price_f64,
            diff,
        });
    }

    entries.sort_by(|a, b| b.diff.total_cmp(&a.diff));
    entries
}

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

/// Calculates the implied long price for an outcome token by summing the prices
/// of all other outcome tokens. This represents the cost of selling all tokens
/// except the target, effectively going long on it.
pub fn price_alt(outcome_token: &str, prices: &[(f64, &str)]) -> f64 {
    let others_sum: f64 = prices
        .iter()
        .filter(|(_, token)| !token.eq_ignore_ascii_case(outcome_token))
        .map(|(price, _)| price)
        .sum();
    1.0 - others_sum
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
        let mut total_price = U256::ZERO;
        for (slot0, market) in results.iter().take(1000) {
            let pool = market.pool.as_ref().unwrap();
            let price = sqrt_price_x96_to_price_outcome(
                slot0.sqrt_price_x96,
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase(),
            )
            .unwrap();
            total_price += price;
            // Convert to f64 for display (price is scaled by 10^18)
            let price_f64: f64 = price.to_string().parse::<f64>().unwrap() / 1e18;
            println!(
                "Pool {}: tick={}, token0={}, token1={}, sqrtPriceX96={}, liquidity={}, price={}",
                slot0.pool_id,
                slot0.tick,
                pool.token0,
                pool.token1,
                slot0.sqrt_price_x96,
                pool.liquidity,
                price_f64
            );
        }
        let total_f64: f64 = total_price.to_string().parse::<f64>().unwrap() / 1e18;
        println!("Total price: {}", total_f64);

        if total_f64 < 1.0 {
            println!("Arbitrage opportunity detected!");

            // Binary search for optimal amount to buy (exact output)
            // We want to find the max amount where sum of price_next <= 1.0 and no tick crossed
            let mut lo: u128 = 1;
            let mut hi: u128 = 1_000_000_000_000_000_000_000u128; // 1000 tokens max
            let mut best_amount: u128 = 0;
            let mut best_cost: u128 = 0;

            while lo <= hi {
                let mid = lo + (hi - lo) / 2;

                // Simulate buying `mid` amount of each outcome token
                let mut valid = true;
                let mut total_cost: u128 = 0;
                let mut sum_price_next = U256::ZERO;

                for (slot0, market) in results.iter() {
                    let pool = market.pool.as_ref().unwrap();
                    let is_token1_outcome =
                        pool.token1.to_lowercase() == market.outcome_token.to_lowercase();

                    // Exact output: negative amount
                    let amount = I256::try_from(mid).unwrap().checked_neg().unwrap();
                    let result = match simulate_buy(
                        pool,
                        slot0.sqrt_price_x96,
                        market.quote_token,
                        amount,
                    ) {
                        Ok(r) => r,
                        Err(_) => {
                            valid = false;
                            break;
                        }
                    };

                    if result.crossed_tick {
                        valid = false;
                        break;
                    }

                    let price_next =
                        sqrt_price_x96_to_price_outcome(result.sqrt_price_next, is_token1_outcome)
                            .unwrap_or(U256::MAX);
                    sum_price_next += price_next;
                    total_cost += (result.amount_in + result.fee_amount).to::<u128>();
                }

                // Check if sum of prices <= 1.0 (using PRICE_SCALE for precision)
                if valid && sum_price_next <= PRICE_SCALE {
                    best_amount = mid;
                    best_cost = total_cost;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }

            println!(
                "Optimal amount: {} (cost: {}, profit: {})",
                best_amount,
                best_cost,
                if best_amount > best_cost {
                    best_amount - best_cost
                } else {
                    0
                }
            );

            // Print final state for each pool at optimal amount
            if best_amount > 0 {
                let amount = I256::try_from(best_amount).unwrap().checked_neg().unwrap();
                for (slot0, market) in results.iter() {
                    let pool = market.pool.as_ref().unwrap();
                    let result =
                        simulate_buy(pool, slot0.sqrt_price_x96, market.quote_token, amount)
                            .unwrap();
                    let is_token1_outcome =
                        pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
                    let price_next =
                        sqrt_price_x96_to_price_outcome(result.sqrt_price_next, is_token1_outcome)
                            .unwrap();
                    let price_f64: f64 = price_next.to_string().parse::<f64>().unwrap() / 1e18;
                    println!(
                        "  outcome={}: cost={}, price_next={:.6}, crossed={}",
                        market.outcome_token,
                        result.amount_in + result.fee_amount,
                        price_f64,
                        result.crossed_tick
                    );
                }
            }
        }
    }

    #[test]
    fn test_collect_markets_with_pools() {
        let markets = collect_markets_with_pools();
        println!("Collected {} markets with pools", markets.len());
        assert!(!markets.is_empty());
    }

    #[tokio::test]
    async fn test_profitability() {
        dotenvy::dotenv().ok();
        let rpc_url = std::env::var("RPC").expect("RPC environment variable not set");
        let provider = ProviderBuilder::new().connect_http(rpc_url.parse().unwrap());

        let slot0_results = fetch_all_slot0(provider).await.unwrap();
        let entries = profitability(&slot0_results);

        println!("Profitability for {} matched markets:", entries.len());
        for entry in &entries {
            println!(
                "  {}: prediction={:.4}, market_price={:.4}, diff={:+.4}",
                entry.market_name, entry.prediction, entry.market_price, entry.diff
            );
        }

        assert!(!entries.is_empty());
    }

    #[tokio::test]
    async fn test_price_alt() {
        dotenvy::dotenv().ok();
        let rpc_url = std::env::var("RPC").expect("RPC environment variable not set");
        let provider = ProviderBuilder::new().connect_http(rpc_url.parse().unwrap());

        let results = fetch_all_slot0(provider).await.unwrap();

        // Build (price, outcome_token) pairs
        let prices: Vec<(f64, &str)> = results
            .iter()
            .filter_map(|(slot0, market)| {
                let pool = market.pool.as_ref()?;
                let is_token1_outcome =
                    pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
                let price =
                    sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome)?;
                let price_f64: f64 = price.to_string().parse::<f64>().unwrap() / 1e18;
                Some((price_f64, market.outcome_token))
            })
            .collect();

        println!("Prices for {} outcomes:", prices.len());
        for (price, token) in &prices {
            let long_price = price_alt(token, &prices);
            println!(
                "  token={}, price={:.6}, price_alt={:.6}",
                token, price, long_price
            );
        }

        // price_alt should equal 1 - (total - own price)
        let total: f64 = prices.iter().map(|(p, _)| p).sum();
        for (price, token) in &prices {
            let long_price = price_alt(token, &prices);
            let expected = 1.0 - (total - price);
            assert!(
                (long_price - expected).abs() < 1e-12,
                "price_alt mismatch for {}: got {}, expected {}",
                token,
                long_price,
                expected
            );
        }
    }
}
