mod analytics;
mod cache;
mod pricing;
mod rpc;
mod swap;

pub use analytics::{DepthResult, ProfitabilityEntry, price_long_simple_alt, profitability_simple};
pub use cache::{BalanceCache, cache_to_balances, load_balance_cache, save_balance_cache};
pub use pricing::{
    normalize_market_name, prediction_map, sqrt_price_x96_to_inv_price, sqrt_price_x96_to_price,
    sqrt_price_x96_to_price_outcome,
};
pub use rpc::{Slot0Result, base_quote_tokens, fetch_all_slot0, fetch_balances};
pub use swap::{SwapResult, simulate_buy, simulate_swap};

pub(crate) use pricing::u256_to_f64;
pub(crate) use swap::FEE_PIPS;

#[cfg(test)]
pub(crate) use pricing::prediction_to_sqrt_price_x96;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use alloy::providers::ProviderBuilder;
    use alloy_primitives::{I256, U256};

    use super::*;

    const PRICE_SCALE_TEST: U256 = U256::from_limbs([1_000_000_000_000_000_000u64, 0, 0, 0]);

    fn network_tests_enabled() -> bool {
        std::env::var("RUN_NETWORK_TESTS").ok().as_deref() == Some("1")
    }

    #[tokio::test]
    async fn test_fetch_all_slot0() {
        if !network_tests_enabled() {
            return;
        }
        dotenvy::dotenv().ok();
        let rpc_url = std::env::var("RPC").expect("RPC environment variable not set");
        let provider = ProviderBuilder::new().with_reqwest(rpc_url.parse().unwrap(), |builder| {
            builder
                .no_proxy()
                .build()
                .expect("failed to build reqwest client for tests")
        });

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

                for (slot0, market) in &results {
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
                if valid && sum_price_next <= PRICE_SCALE_TEST {
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
                for (slot0, market) in &results {
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
        let markets_with_pools = crate::markets::MARKETS_L1
            .iter()
            .filter(|m| m.pool.is_some())
            .count();
        println!("Collected {} markets with pools", markets_with_pools);
        assert!(markets_with_pools > 0);
    }

    #[tokio::test]
    async fn test_profitability_simple() {
        if !network_tests_enabled() {
            return;
        }
        dotenvy::dotenv().ok();
        let rpc_url = std::env::var("RPC").expect("RPC environment variable not set");
        let provider = ProviderBuilder::new().with_reqwest(rpc_url.parse().unwrap(), |builder| {
            builder
                .no_proxy()
                .build()
                .expect("failed to build reqwest client for tests")
        });

        let slot0_results = fetch_all_slot0(provider).await.unwrap();
        let entries = profitability_simple(&slot0_results);

        println!("Profitability for {} matched markets:", entries.len());
        for entry in &entries {
            let d = &entry.depth;
            println!(
                "  {}: prediction={:.4}, market_price={:.4}, diff={:+.4}, liq={}, tick_out={:.4}, tick_cost={:.4}, be_out={:.4}, be_cost={:.4}",
                entry.market_name,
                entry.prediction,
                entry.market_price,
                entry.diff,
                entry.has_liquidity,
                d.outcome_at_tick,
                d.cost_at_tick,
                d.outcome_at_breakeven,
                d.cost_at_breakeven,
            );
        }

        assert!(!entries.is_empty());
    }

    #[tokio::test]
    async fn test_price_long_simple_alt() {
        if !network_tests_enabled() {
            return;
        }
        dotenvy::dotenv().ok();
        let rpc_url = std::env::var("RPC").expect("RPC environment variable not set");
        let provider = ProviderBuilder::new().with_reqwest(rpc_url.parse().unwrap(), |builder| {
            builder
                .no_proxy()
                .build()
                .expect("failed to build reqwest client for tests")
        });

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
            let long_price = price_long_simple_alt(token, &prices);
            println!(
                "  token={}, price={:.6}, price_long_simple_alt={:.6}",
                token, price, long_price
            );
        }

        // price_long_simple_alt should equal 1 - (total - own price)
        let total: f64 = prices.iter().map(|(p, _)| p).sum();
        for (price, token) in &prices {
            let long_price = price_long_simple_alt(token, &prices);
            let expected = 1.0 - (total - price);
            assert!(
                (long_price - expected).abs() < 1e-12,
                "price_long_simple_alt mismatch for {}: got {}, expected {}",
                token,
                long_price,
                expected
            );
        }
    }

    #[test]
    fn test_cache_round_trip() {
        let dir = std::env::temp_dir().join("deep_trading_test_cache");
        let path = dir.join("balances.json");
        let _ = std::fs::create_dir_all(&dir);

        let mut outcomes = HashMap::new();
        outcomes.insert("outcome_a", 1.5);
        outcomes.insert("outcome_b", 0.0);

        save_balance_cache(&path, "0xABC", 42.0, &outcomes).unwrap();
        let loaded = load_balance_cache(&path, "0xABC", None).unwrap();

        assert_eq!(loaded.wallet, "0xABC");
        assert_eq!(loaded.susds, 42.0);
        assert_eq!(loaded.outcomes.get("outcome_a"), Some(&1.5));
        assert_eq!(loaded.outcomes.get("outcome_b"), Some(&0.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cache_expiration() {
        let dir = std::env::temp_dir().join("deep_trading_test_cache");
        let path = dir.join("balances_exp.json");
        let _ = std::fs::create_dir_all(&dir);

        // Write a cache with timestamp far in the past
        let cache = BalanceCache {
            timestamp: 1000,
            wallet: "0xABC".to_string(),
            susds: 1.0,
            outcomes: HashMap::new(),
        };
        let file = std::fs::File::create(&path).unwrap();
        serde_json::to_writer(std::io::BufWriter::new(file), &cache).unwrap();

        let loaded = load_balance_cache(&path, "0xABC", Some(60));
        assert!(loaded.is_none(), "cache should be expired");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cache_wallet_mismatch() {
        let dir = std::env::temp_dir().join("deep_trading_test_cache");
        let path = dir.join("balances_wm.json");
        let _ = std::fs::create_dir_all(&dir);

        let outcomes = HashMap::new();
        save_balance_cache(&path, "0xABC", 1.0, &outcomes).unwrap();

        let loaded = load_balance_cache(&path, "0xDEF", Some(3600));
        assert!(loaded.is_none(), "cache should reject different wallet");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cache_to_balances() {
        let market_name = crate::markets::MARKETS_L1[0].name;
        let mut outcomes = HashMap::new();
        outcomes.insert(market_name.to_string(), 7.77);
        outcomes.insert("nonexistent_market".to_string(), 99.0);

        let cache = BalanceCache {
            timestamp: 0,
            wallet: "0xABC".to_string(),
            susds: 0.0,
            outcomes,
        };

        let balances = cache_to_balances(&cache);
        assert_eq!(balances.get(market_name), Some(&7.77));
        assert!(!balances.contains_key("nonexistent_market"));
    }

    #[tokio::test]
    async fn test_fetch_balances() {
        if !network_tests_enabled() {
            return;
        }
        dotenvy::dotenv().ok();
        let rpc_url = match std::env::var("RPC") {
            Ok(url) => url,
            Err(_) => return,
        };
        let provider = ProviderBuilder::new().with_reqwest(rpc_url.parse().unwrap(), |builder| {
            builder
                .no_proxy()
                .build()
                .expect("failed to build reqwest client for tests")
        });

        // Use zero address — should return 0 balances but not error
        let wallet = alloy::primitives::Address::ZERO;
        let (susds, balances) = fetch_balances(provider, wallet).await.unwrap();
        println!("sUSD balance: {}", susds);
        println!("Outcome balances: {} markets", balances.len());
        for (name, bal) in &balances {
            if *bal > 0.0 {
                println!("  {}: {}", name, bal);
            }
        }
    }
}
