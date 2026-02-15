use std::collections::HashMap;
use std::str::FromStr;

use alloy::{
    primitives::{Address, Bytes},
    providers::Provider,
    sol,
    sol_types::SolCall,
};
use alloy_primitives::U256;
use futures_util::future;

use crate::markets::{MARKETS_L1, MarketData};

use super::pricing::u256_to_f64;

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

// ERC20 balanceOf for reading token balances
sol! {
    function balanceOf(address owner) external view returns (uint256);
}

// Multicall3 address (same on all EVM chains)
const MULTICALL3_ADDRESS: Address =
    alloy::primitives::address!("cA11bde05977b3631167028862bE2a173976CA11");
const MULTICALL_BATCH_SIZE: usize = 200;

/// Returns (base_token, quote_token) addresses for price calculation
pub fn base_quote_tokens(market: &MarketData) -> Option<(Address, Address)> {
    let pool = market.pool.as_ref()?;
    let token0 = Address::from_str(pool.token0).ok()?;
    let token1 = Address::from_str(pool.token1).ok()?;
    let quote_token = Address::from_str(market.quote_token).ok()?;
    if token0 == quote_token {
        Some((token1, token0))
    } else {
        Some((token0, token1))
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
                None => tracing::warn!(%pool_id, "failed to fetch slot0"),
            }
        }
    }

    Ok(slot0_results)
}

/// Fetches sUSD and all outcome token balances for a wallet via batched multicall.
/// Returns (susds_balance, outcome_balances_by_market_name).
pub async fn fetch_balances<P: Provider + Clone>(
    provider: P,
    wallet: Address,
) -> Result<(f64, HashMap<&'static str, f64>), Box<dyn std::error::Error>> {
    let calldata = Bytes::from(balanceOfCall { owner: wallet }.abi_encode());

    // First call: sUSD balance. Rest: one per market in MARKETS_L1.
    let quote_token = Address::from_str(MARKETS_L1[0].quote_token)?;
    let mut calls = vec![Multicall3::Call3 {
        target: quote_token,
        allowFailure: true,
        callData: calldata.clone(),
    }];

    let markets: Vec<&'static MarketData> = MARKETS_L1.iter().collect();

    for market in &markets {
        let token = Address::from_str(market.outcome_token)?;
        calls.push(Multicall3::Call3 {
            target: token,
            allowFailure: true,
            callData: calldata.clone(),
        });
    }

    let multicall = Multicall3::new(MULTICALL3_ADDRESS, provider);
    let batch_futures: Vec<_> = calls
        .chunks(MULTICALL_BATCH_SIZE)
        .map(|chunk| {
            let call = multicall.aggregate3(chunk.to_vec());
            async move { call.call().await }
        })
        .collect();
    let all_results = future::try_join_all(batch_futures).await?;
    let flat: Vec<_> = all_results.into_iter().flatten().collect();

    // Parse sUSD balance (first result)
    let susds = flat
        .first()
        .and_then(|r| {
            if r.success {
                balanceOfCall::abi_decode_returns(&r.returnData).ok()
            } else {
                None
            }
        })
        .map(u256_to_f64)
        .unwrap_or(0.0);

    // Parse outcome balances
    let mut balances = HashMap::new();
    for (i, market) in markets.iter().enumerate() {
        let bal = flat
            .get(i + 1)
            .and_then(|r| {
                if r.success {
                    balanceOfCall::abi_decode_returns(&r.returnData).ok()
                } else {
                    None
                }
            })
            .map(u256_to_f64)
            .unwrap_or(0.0);
        balances.insert(market.name, bal);
    }

    Ok((susds, balances))
}

#[cfg(test)]
mod tests {
    use super::base_quote_tokens;
    use crate::markets::{MarketData, Pool, Tick};

    static TICKS: [Tick; 2] = [
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
    fn base_quote_tokens_returns_none_when_pool_is_missing() {
        let market = MarketData {
            name: "m",
            market_id: "id",
            outcome_token: "0x1",
            pool: None,
            quote_token: "0x2",
        };
        assert!(base_quote_tokens(&market).is_none());
    }

    #[test]
    fn base_quote_tokens_returns_none_on_invalid_token_address() {
        let market = MarketData {
            name: "m",
            market_id: "id",
            outcome_token: "0x1",
            pool: Some(Pool {
                token0: "not-an-address",
                token1: "0x0000000000000000000000000000000000000001",
                pool_id: "0x0000000000000000000000000000000000000002",
                liquidity: "1000",
                ticks: &TICKS,
            }),
            quote_token: "0x0000000000000000000000000000000000000003",
        };
        assert!(base_quote_tokens(&market).is_none());
    }
}
