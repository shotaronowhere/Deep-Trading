use crate::markets::{MarketData, Pool, Tick};
use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};
use alloy::primitives::{Address, U256};

use super::PoolSim;

/// Mock pool: token0=quote(sUSD), token1=outcome → is_token1_outcome=true.
/// Uses positive tick range matching real L1 pools with this ordering.
/// Price range: ~[0.0001, 0.2] for outcome token.
pub(super) fn mock_pool() -> Pool {
    Pool {
        token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0", // quote (sUSD)
        token1: "0x1111111111111111111111111111111111111111", // outcome
        pool_id: "0x0000000000000000000000000000000000000001",
        liquidity: "1000000000000000000000", // 1000 tokens
        ticks: &[
            Tick {
                tick_idx: 16095,
                liquidity_net: 1000000000000000000000,
            },
            Tick {
                tick_idx: 92108,
                liquidity_net: -1000000000000000000000,
            },
        ],
    }
}

/// Leak a pool into a static reference for tests.
pub(super) fn leak_pool(pool: Pool) -> &'static Pool {
    Box::leak(Box::new(pool))
}

pub(super) fn leak_market(market: MarketData) -> &'static MarketData {
    Box::leak(Box::new(market))
}

/// Create slot0 + market data pair for testing.
/// `price_fraction` sets the outcome price approximately (0.0 to 1.0).
pub(super) fn mock_slot0_market(
    name: &'static str,
    outcome_token: &'static str,
    price_fraction: f64,
) -> (Slot0Result, &'static MarketData) {
    let mut p = mock_pool();
    p.token1 = outcome_token;
    let pool = leak_pool(p);

    // token0 is quote (sUSD), token1 is outcome
    // is_token1_outcome = true
    // outcome price = inv_price = 2^192 / sqrtPriceX96^2
    // To get outcome price = price_fraction:
    // sqrtPriceX96 = sqrt(2^192 / (price_fraction * 1e18)) * sqrt(1e18)
    // Simpler: use prediction_to_sqrt_price_x96
    let sqrt_price =
        prediction_to_sqrt_price_x96(price_fraction, true).unwrap_or(U256::from(1u128 << 96));

    let market = leak_market(MarketData {
        name,
        market_id: crate::markets::MARKETS_L1[0].market_id,
        outcome_token,
        pool: Some(*pool),
        quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
    });

    let slot0 = Slot0Result {
        pool_id: Address::ZERO,
        sqrt_price_x96: sqrt_price,
        tick: 0,
        observation_index: 0,
        observation_cardinality: 0,
        observation_cardinality_next: 0,
        fee_protocol: 0,
        unlocked: true,
    };

    (slot0, market)
}

pub(super) fn build_three_sims(prices: [f64; 3]) -> Vec<PoolSim> {
    build_three_sims_with_preds(prices, [0.3, 0.3, 0.3])
}

pub(super) fn build_three_sims_with_preds(prices: [f64; 3], preds: [f64; 3]) -> Vec<PoolSim> {
    let tokens = [
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
        "0x3333333333333333333333333333333333333333",
    ];
    let names = ["M1", "M2", "M3"];
    let slot0_results: Vec<_> = tokens
        .iter()
        .zip(names.iter())
        .zip(prices.iter())
        .map(|((tok, name), price)| mock_slot0_market(name, tok, *price))
        .collect();
    slot0_results
        .iter()
        .zip(preds.iter())
        .map(|((s, m), pred)| PoolSim::from_slot0(s, m, *pred).unwrap())
        .collect()
}

pub(super) fn mock_slot0_market_with_liquidity_and_ticks(
    name: &'static str,
    outcome_token: &'static str,
    price_fraction: f64,
    liquidity: u128,
    tick_lo: i32,
    tick_hi: i32,
) -> (Slot0Result, &'static MarketData) {
    let mut p = mock_pool();
    p.token1 = outcome_token;

    let liq_str = Box::leak(liquidity.to_string().into_boxed_str());
    p.liquidity = liq_str;
    let liq_i128 = i128::try_from(liquidity).unwrap_or(i128::MAX);
    let lo = tick_lo.min(tick_hi);
    let hi = tick_lo.max(tick_hi);
    let ticks = Box::leak(Box::new([
        Tick {
            tick_idx: lo,
            liquidity_net: liq_i128,
        },
        Tick {
            tick_idx: hi,
            liquidity_net: -liq_i128,
        },
    ]));
    p.ticks = ticks;

    let pool = leak_pool(p);
    let sqrt_price =
        prediction_to_sqrt_price_x96(price_fraction, true).unwrap_or(U256::from(1u128 << 96));

    let market = leak_market(MarketData {
        name,
        market_id: crate::markets::MARKETS_L1[0].market_id,
        outcome_token,
        pool: Some(*pool),
        quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
    });

    let slot0 = Slot0Result {
        pool_id: Address::ZERO,
        sqrt_price_x96: sqrt_price,
        tick: 0,
        observation_index: 0,
        observation_cardinality: 0,
        observation_cardinality_next: 0,
        fee_protocol: 0,
        unlocked: true,
    };

    (slot0, market)
}

pub(super) fn mock_slot0_market_with_liquidity(
    name: &'static str,
    outcome_token: &'static str,
    price_fraction: f64,
    liquidity: u128,
) -> (Slot0Result, &'static MarketData) {
    mock_slot0_market_with_liquidity_and_ticks(
        name,
        outcome_token,
        price_fraction,
        liquidity,
        16095,
        92108,
    )
}
