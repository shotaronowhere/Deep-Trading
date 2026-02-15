use alloy::primitives::{Address, Bytes, Uint, address};
use alloy::providers::ProviderBuilder;
use alloy::sol;
use alloy::sol_types::SolCall;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::Path;

use super::common::rustfmt_generated_file;

// Uniswap V3 Factory getPool call
sol! {
    function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool);
}

// Market parentWrappedOutcome call (for L2 base token resolution)
sol! {
    function parentWrappedOutcome() external view returns (address wrapped1155, bytes data);
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

type U24 = Uint<24, 1>;

// Optimism Uniswap V3 Factory and Multicall3 addresses
const FACTORY_ADDRESS: Address = address!("1F98431c8aD98523631AE4a59f267346ea31F984");
const MULTICALL3_ADDRESS: Address = address!("cA11bde05977b3631167028862bE2a173976CA11");
const FEE_TIER: u32 = 100;
const MULTICALL_BATCH_SIZE: usize = 16000;
pub(super) const DEFAULT_OPTIMISM_RPC_URL: &str = "https://optimism.drpc.org";

// L1 base token (hardcoded)
const L1_QUOTE_TOKEN: &str = "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MarketsFile {
    markets_data: BTreeMap<String, MarketInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MarketInfo {
    market_id: Option<String>,
    outcome_token: String,
    pools: Option<Vec<PoolInfo>>,
    up_token: Option<String>,
    down_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TickInfo {
    tick_idx: String,
    liquidity_net: String,
}

#[derive(Debug, Deserialize)]
struct PoolInfo {
    token0: String,
    token1: String,
    liquidity: Option<String>,
    ticks: Option<Vec<TickInfo>>,
}

// Tick with resolved data (filtered for non-zero liquidityNet)
#[derive(Clone)]
struct ResolvedTick {
    tick_idx: i32,
    liquidity_net: i128,
}

// Pool with resolved pool_id from factory
#[derive(Clone)]
struct ResolvedPool {
    token0: String,
    token1: String,
    pool_id: Option<String>,
    liquidity: Option<String>,
    ticks: Vec<ResolvedTick>,
}

// Track which call corresponds to which market and pool index
struct CallInfo<'a> {
    market_name: &'a str,
    pool_index: usize,
    token0: &'a str,
    token1: &'a str,
    liquidity: Option<&'a str>,
    ticks: Vec<ResolvedTick>,
}

// Helper to parse the address from a multicall result
fn parse_pool_address(result: &Multicall3::Result) -> Option<String> {
    if !result.success || result.returnData.len() < 32 {
        return None;
    }
    let addr = Address::from_slice(&result.returnData[12..32]);
    (addr != Address::ZERO).then(|| format!("{addr:?}"))
}

// Generate L1 markets array (single pool per market)
fn generate_markets_l1_array(
    markets: &BTreeMap<String, MarketInfo>,
    resolved_pools: &BTreeMap<String, Vec<ResolvedPool>>,
    quote_token: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    let mut tick_definitions = String::new();
    let mut market_entries = String::new();

    for (name, market_data) in markets {
        let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");
        let lower_name = escaped_name.to_lowercase();
        let safe_name = lower_name.replace(['/', '-', '.', '\\'], "_");
        let market_id = market_data.market_id.as_deref().unwrap_or("");

        writeln!(&mut market_entries, "    MarketData {{")?;
        writeln!(&mut market_entries, "        name: \"{}\",", lower_name)?;
        writeln!(&mut market_entries, "        market_id: \"{}\",", market_id)?;
        writeln!(
            &mut market_entries,
            "        outcome_token: \"{}\",",
            market_data.outcome_token
        )?;

        // Take first pool from the vec
        match resolved_pools.get(name).and_then(|pools| pools.first()) {
            Some(pool) => {
                let pool_id_str = pool.pool_id.as_deref().unwrap_or("");
                let liquidity_str = pool.liquidity.as_deref().unwrap_or("");

                // Generate tick array if pool has ticks
                let ticks_ref = if pool.ticks.is_empty() {
                    "&[]".to_string()
                } else {
                    let tick_static_name = format!("TICKS_L1_{}", safe_name.to_uppercase());
                    writeln!(
                        &mut tick_definitions,
                        "static {}: [Tick; {}] = [",
                        tick_static_name,
                        pool.ticks.len()
                    )?;
                    for tick in &pool.ticks {
                        writeln!(
                            &mut tick_definitions,
                            "    Tick {{ tick_idx: {}, liquidity_net: {} }},",
                            tick.tick_idx, tick.liquidity_net
                        )?;
                    }
                    writeln!(&mut tick_definitions, "];")?;
                    format!("&{}", tick_static_name)
                };

                writeln!(&mut market_entries, "        pool: Some(Pool {{")?;
                writeln!(
                    &mut market_entries,
                    "            token0: \"{}\",",
                    pool.token0
                )?;
                writeln!(
                    &mut market_entries,
                    "            token1: \"{}\",",
                    pool.token1
                )?;
                writeln!(
                    &mut market_entries,
                    "            pool_id: \"{}\",",
                    pool_id_str
                )?;
                writeln!(
                    &mut market_entries,
                    "            liquidity: \"{}\",",
                    liquidity_str
                )?;
                writeln!(&mut market_entries, "            ticks: {},", ticks_ref)?;
                writeln!(&mut market_entries, "        }}),")?;
            }
            None => {
                writeln!(&mut market_entries, "        pool: None,")?;
            }
        }

        writeln!(
            &mut market_entries,
            "        quote_token: \"{}\",",
            quote_token
        )?;
        writeln!(&mut market_entries, "    }},")?;
    }

    output.push_str(&tick_definitions);
    if !tick_definitions.is_empty() {
        writeln!(&mut output)?;
    }
    writeln!(
        &mut output,
        "pub static MARKETS_L1: [MarketData; {}] = [",
        markets.len()
    )?;
    output.push_str(&market_entries);
    writeln!(&mut output, "];")?;

    Ok(output)
}

/// Generate tick statics + pool static array for a slice of resolved pools.
/// Returns the Rust expression referencing the pool array (e.g. `&POOLS_L2_FOO` or `&[]`).
fn generate_pool_statics(
    pools: &[ResolvedPool],
    safe_name: &str,
    prefix: &str,
    out: &mut String,
) -> std::fmt::Result {
    if pools.is_empty() {
        return Ok(());
    }
    let upper = safe_name.to_uppercase();

    for (idx, pool) in pools.iter().enumerate() {
        if !pool.ticks.is_empty() {
            let tick_name = format!("TICKS_{prefix}{upper}_{idx}");
            writeln!(out, "static {tick_name}: [Tick; {}] = [", pool.ticks.len())?;
            for tick in &pool.ticks {
                writeln!(
                    out,
                    "    Tick {{ tick_idx: {}, liquidity_net: {} }},",
                    tick.tick_idx, tick.liquidity_net
                )?;
            }
            writeln!(out, "];")?;
        }
    }

    let static_name = format!("POOLS_{prefix}{upper}");
    writeln!(out, "static {static_name}: [Pool; {}] = [", pools.len())?;
    for (idx, pool) in pools.iter().enumerate() {
        let pool_id = pool.pool_id.as_deref().unwrap_or("");
        let liquidity = pool.liquidity.as_deref().unwrap_or("");
        let ticks_ref = if pool.ticks.is_empty() {
            "&[]".to_string()
        } else {
            format!("&TICKS_{prefix}{upper}_{idx}")
        };
        writeln!(
            out,
            "    Pool {{ token0: \"{}\", token1: \"{}\", pool_id: \"{pool_id}\", liquidity: \"{liquidity}\", ticks: {ticks_ref} }},",
            pool.token0, pool.token1
        )?;
    }
    writeln!(out, "];")?;
    Ok(())
}

fn pool_ref_expr(pools: Option<&Vec<ResolvedPool>>, safe_name: &str, prefix: &str) -> String {
    match pools.filter(|p| !p.is_empty()) {
        Some(_) => format!("&POOLS_{prefix}{}", safe_name.to_uppercase()),
        None => "&[]".to_string(),
    }
}

// Generate originality markets array (multiple pools per market: upPool, downPool)
fn generate_markets_originality_array(
    markets: &BTreeMap<String, MarketInfo>,
    resolved_pools: &BTreeMap<String, Vec<ResolvedPool>>,
    quote_token: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    let mut pool_definitions = String::new();
    let mut market_entries = String::new();

    for (name, market_data) in markets {
        let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");
        let lower_name = escaped_name.to_lowercase();
        let safe_name = lower_name.replace(['/', '-', '.', '\\'], "_");
        let market_id = market_data.market_id.as_deref().unwrap_or("");

        let pools = resolved_pools.get(name);
        if let Some(p) = pools.filter(|p| !p.is_empty()) {
            generate_pool_statics(p, &safe_name, "ORI_", &mut pool_definitions)?;
        }
        let pool_ref = pool_ref_expr(pools, &safe_name, "ORI_");

        let up_token = market_data.up_token.as_deref().unwrap_or("");
        let down_token = market_data.down_token.as_deref().unwrap_or("");

        writeln!(&mut market_entries, "    MarketDataOriginality {{")?;
        writeln!(&mut market_entries, "        name: \"{lower_name}\",")?;
        writeln!(&mut market_entries, "        market_id: \"{market_id}\",")?;
        writeln!(&mut market_entries, "        pools: {pool_ref},")?;
        writeln!(&mut market_entries, "        up_token: \"{up_token}\",")?;
        writeln!(&mut market_entries, "        down_token: \"{down_token}\",")?;
        writeln!(
            &mut market_entries,
            "        quote_token: \"{quote_token}\","
        )?;
        writeln!(&mut market_entries, "    }},")?;
    }

    output.push_str(&pool_definitions);
    if !pool_definitions.is_empty() {
        writeln!(&mut output)?;
    }
    writeln!(
        &mut output,
        "pub static MARKETS_ORIGINALITY: [MarketDataOriginality; {}] = [",
        markets.len()
    )?;
    output.push_str(&market_entries);
    writeln!(&mut output, "];")?;

    Ok(output)
}

// Generate L2 markets array (multiple pools per market) - single pass
fn generate_markets_l2_array(
    markets: &BTreeMap<String, MarketInfo>,
    resolved_pools: &BTreeMap<String, Vec<ResolvedPool>>,
    quote_tokens: &BTreeMap<String, String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    let mut pool_definitions = String::new();
    let mut market_entries = String::new();

    for (name, market_data) in markets {
        let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");
        let lower_name = escaped_name.to_lowercase();
        let safe_name = lower_name.replace(['/', '-', '.', '\\'], "_");
        let market_id = market_data.market_id.as_deref().unwrap_or("");
        let quote_token = quote_tokens.get(name).map(|s| s.as_str()).unwrap_or("");

        let pools = resolved_pools.get(name);
        if let Some(p) = pools.filter(|p| !p.is_empty()) {
            generate_pool_statics(p, &safe_name, "", &mut pool_definitions)?;
        }
        let pool_ref = pool_ref_expr(pools, &safe_name, "");

        writeln!(&mut market_entries, "    MarketDataL2 {{")?;
        writeln!(&mut market_entries, "        name: \"{lower_name}\",")?;
        writeln!(&mut market_entries, "        market_id: \"{market_id}\",")?;
        writeln!(
            &mut market_entries,
            "        outcome_token: \"{}\",",
            market_data.outcome_token
        )?;
        writeln!(&mut market_entries, "        pools: {pool_ref},")?;
        writeln!(
            &mut market_entries,
            "        quote_token: \"{quote_token}\","
        )?;
        writeln!(&mut market_entries, "    }},")?;
    }

    output.push_str(&pool_definitions);
    if !pool_definitions.is_empty() {
        writeln!(&mut output)?;
    }
    writeln!(
        &mut output,
        "pub static MARKETS_L2: [MarketDataL2; {}] = [",
        markets.len()
    )?;
    output.push_str(&market_entries);
    writeln!(&mut output, "];")?;

    Ok(output)
}

// Helper to filter ticks and create ResolvedTick vec
fn filter_ticks(pool: &PoolInfo) -> Vec<ResolvedTick> {
    pool.ticks
        .as_ref()
        .map(|ticks| {
            ticks
                .iter()
                .filter_map(|t| {
                    let liquidity_net: i128 = t.liquidity_net.parse().ok()?;
                    if liquidity_net == 0 {
                        return None;
                    }
                    let tick_idx: i32 = t.tick_idx.parse().ok()?;
                    Some(ResolvedTick {
                        tick_idx,
                        liquidity_net,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

// Helper to build a single call and info
fn build_single_call<'a>(
    name: &'a str,
    pool_idx: usize,
    pool: &'a PoolInfo,
    calls: &mut Vec<Multicall3::Call3>,
    call_infos: &mut Vec<CallInfo<'a>>,
) {
    let (Ok(token0), Ok(token1)) = (
        pool.token0.parse::<Address>(),
        pool.token1.parse::<Address>(),
    ) else {
        eprintln!(
            "warning: Skipping pool {} for market '{}': invalid token address",
            pool_idx, name
        );
        return;
    };

    let call_data = getPoolCall {
        tokenA: token0,
        tokenB: token1,
        fee: U24::from(FEE_TIER),
    }
    .abi_encode();

    calls.push(Multicall3::Call3 {
        target: FACTORY_ADDRESS,
        allowFailure: true,
        callData: Bytes::from(call_data),
    });

    call_infos.push(CallInfo {
        market_name: name,
        pool_index: pool_idx,
        token0: &pool.token0,
        token1: &pool.token1,
        liquidity: pool.liquidity.as_deref(),
        ticks: filter_ticks(pool),
    });
}

// Helper to build calls and infos for a set of pools
fn build_pool_calls<'a>(
    markets: &'a BTreeMap<String, MarketInfo>,
    first_only: bool,
) -> (Vec<Multicall3::Call3>, Vec<CallInfo<'a>>) {
    let mut calls = Vec::new();
    let mut call_infos = Vec::new();

    for (name, market_data) in markets {
        let Some(pools) = market_data.pools.as_ref() else {
            continue;
        };

        if first_only {
            if let Some(pool) = pools.first() {
                build_single_call(name, 0, pool, &mut calls, &mut call_infos);
            }
        } else {
            for (pool_idx, pool) in pools.iter().enumerate() {
                build_single_call(name, pool_idx, pool, &mut calls, &mut call_infos);
            }
        }
    }
    (calls, call_infos)
}

// Unified pool address resolution - returns Vec per market, caller extracts first if needed
async fn resolve_pool_addresses(
    markets: &BTreeMap<String, MarketInfo>,
    rpc_url: &str,
    first_only: bool,
) -> Result<BTreeMap<String, Vec<ResolvedPool>>, Box<dyn std::error::Error>> {
    let provider = ProviderBuilder::new().with_reqwest(rpc_url.parse()?, |builder| {
        builder
            .no_proxy()
            .build()
            .expect("failed to build reqwest client")
    });
    let multicall = Multicall3::new(MULTICALL3_ADDRESS, provider);

    let (calls, call_infos) = build_pool_calls(markets, first_only);
    eprintln!("Batching {} getPool calls via Multicall3...", calls.len());

    // Use BTreeMap<usize, ResolvedPool> to correctly order pools by index
    let mut resolved: BTreeMap<String, BTreeMap<usize, ResolvedPool>> = BTreeMap::new();

    for (batch_idx, chunk) in calls.chunks(MULTICALL_BATCH_SIZE).enumerate() {
        let start_idx = batch_idx * MULTICALL_BATCH_SIZE;
        let results = multicall.aggregate3(chunk.to_vec()).call().await?;

        for (i, result) in results.iter().enumerate() {
            let info = &call_infos[start_idx + i];
            let pool_id = parse_pool_address(result);

            resolved
                .entry(info.market_name.to_string())
                .or_default()
                .insert(
                    info.pool_index,
                    ResolvedPool {
                        token0: info.token0.to_string(),
                        token1: info.token1.to_string(),
                        pool_id,
                        liquidity: info.liquidity.map(String::from),
                        ticks: info.ticks.clone(),
                    },
                );
        }
    }

    // Convert BTreeMap<usize, Pool> to Vec<Pool>, preserving order
    Ok(resolved
        .into_iter()
        .map(|(name, pools_map)| {
            let pools: Vec<_> = pools_map.into_values().collect();
            (name, pools)
        })
        .collect())
}

// Resolve base tokens for L2 markets by calling parentWrappedOutcome on each market contract
async fn resolve_l2_quote_tokens(
    markets: &BTreeMap<String, MarketInfo>,
    rpc_url: &str,
) -> Result<BTreeMap<String, String>, Box<dyn std::error::Error>> {
    let provider = ProviderBuilder::new().with_reqwest(rpc_url.parse()?, |builder| {
        builder
            .no_proxy()
            .build()
            .expect("failed to build reqwest client")
    });
    let multicall = Multicall3::new(MULTICALL3_ADDRESS, provider);

    let mut calls = Vec::new();
    let mut market_names = Vec::new();

    for (name, market_data) in markets {
        let Some(market_id) = &market_data.market_id else {
            eprintln!(
                "warning: Skipping base token resolution for '{}': no market_id",
                name
            );
            continue;
        };
        let Ok(market_addr) = market_id.parse::<Address>() else {
            eprintln!(
                "warning: Skipping base token resolution for '{}': invalid market_id '{}'",
                name, market_id
            );
            continue;
        };
        let call_data = parentWrappedOutcomeCall {}.abi_encode();
        calls.push(Multicall3::Call3 {
            target: market_addr,
            allowFailure: true,
            callData: Bytes::from(call_data),
        });
        market_names.push(name.clone());
    }

    eprintln!(
        "Batching {} parentWrappedOutcome calls via Multicall3...",
        calls.len()
    );

    let mut quote_tokens: BTreeMap<String, String> = BTreeMap::new();

    for (batch_idx, chunk) in calls.chunks(MULTICALL_BATCH_SIZE).enumerate() {
        let start_idx = batch_idx * MULTICALL_BATCH_SIZE;
        let results = multicall.aggregate3(chunk.to_vec()).call().await?;

        for (i, result) in results.iter().enumerate() {
            let market_name = &market_names[start_idx + i];
            if result.success && result.returnData.len() >= 32 {
                // The first 32 bytes contain the address (padded)
                let addr = Address::from_slice(&result.returnData[12..32]);
                if addr != Address::ZERO {
                    quote_tokens.insert(market_name.clone(), format!("{addr:?}"));
                }
            }
        }
    }

    Ok(quote_tokens)
}

pub(super) async fn generate_markets_rs(
    l1_json_path: &Path,
    l2_json_path: &Path,
    originality_json_path: &Path,
    output_path: &Path,
    rpc_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let l1_content = fs::read_to_string(l1_json_path)?;
    let l1_json: MarketsFile = serde_json::from_str(&l1_content)?;

    let l2_content = fs::read_to_string(l2_json_path)?;
    let l2_json: MarketsFile = serde_json::from_str(&l2_content)?;

    let originality_content = fs::read_to_string(originality_json_path)?;
    let originality_json: MarketsFile = serde_json::from_str(&originality_content)?;

    // Resolve pool addresses and base tokens concurrently
    eprintln!("Resolving pool addresses and base tokens for all markets...");
    let (l1_res, l2_res, ori_res, quote_res) = tokio::join!(
        resolve_pool_addresses(&l1_json.markets_data, rpc_url, true),
        resolve_pool_addresses(&l2_json.markets_data, rpc_url, false),
        resolve_pool_addresses(&originality_json.markets_data, rpc_url, false),
        resolve_l2_quote_tokens(&l2_json.markets_data, rpc_url)
    );

    let l1_resolved_pools = l1_res?;
    let l2_resolved_pools = l2_res?;
    let originality_resolved_pools = ori_res?;
    let l2_quote_tokens = quote_res?;

    let mut output = String::new();
    writeln!(
        &mut output,
        "// AUTO-GENERATED — do not edit manually.\n// Regenerate with: cargo run --bin regenerate\n"
    )?;

    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct Tick {{")?;
    writeln!(&mut output, "    pub tick_idx: i32,")?;
    writeln!(&mut output, "    pub liquidity_net: i128,")?;
    writeln!(&mut output, "}}\n")?;

    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct Pool {{")?;
    writeln!(&mut output, "    pub token0: &'static str,")?;
    writeln!(&mut output, "    pub token1: &'static str,")?;
    writeln!(&mut output, "    pub pool_id: &'static str,")?;
    writeln!(&mut output, "    pub liquidity: &'static str,")?;
    writeln!(&mut output, "    pub ticks: &'static [Tick],")?;
    writeln!(&mut output, "}}\n")?;

    // L1 MarketData (single pool)
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct MarketData {{")?;
    writeln!(&mut output, "    pub name: &'static str,")?;
    writeln!(&mut output, "    pub market_id: &'static str,")?;
    writeln!(&mut output, "    pub outcome_token: &'static str,")?;
    writeln!(&mut output, "    pub pool: Option<Pool>,")?;
    writeln!(&mut output, "    pub quote_token: &'static str,")?;
    writeln!(&mut output, "}}\n")?;

    // L2 MarketDataL2 (multiple pools)
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct MarketDataL2 {{")?;
    writeln!(&mut output, "    pub name: &'static str,")?;
    writeln!(&mut output, "    pub market_id: &'static str,")?;
    writeln!(&mut output, "    pub outcome_token: &'static str,")?;
    writeln!(&mut output, "    pub pools: &'static [Pool],")?;
    writeln!(&mut output, "    pub quote_token: &'static str,")?;
    writeln!(&mut output, "}}\n")?;

    // Originality MarketDataOriginality (upPool/downPool with explicit tokens)
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct MarketDataOriginality {{")?;
    writeln!(&mut output, "    pub name: &'static str,")?;
    writeln!(&mut output, "    pub market_id: &'static str,")?;
    writeln!(&mut output, "    pub pools: &'static [Pool],")?;
    writeln!(&mut output, "    pub up_token: &'static str,")?;
    writeln!(&mut output, "    pub down_token: &'static str,")?;
    writeln!(&mut output, "    pub quote_token: &'static str,")?;
    writeln!(&mut output, "}}\n")?;

    output.push_str(&generate_markets_l1_array(
        &l1_json.markets_data,
        &l1_resolved_pools,
        L1_QUOTE_TOKEN,
    )?);
    writeln!(&mut output)?;

    output.push_str(&generate_markets_l2_array(
        &l2_json.markets_data,
        &l2_resolved_pools,
        &l2_quote_tokens,
    )?);
    writeln!(&mut output)?;

    output.push_str(&generate_markets_originality_array(
        &originality_json.markets_data,
        &originality_resolved_pools,
        L1_QUOTE_TOKEN,
    )?);

    fs::write(output_path, output)?;
    rustfmt_generated_file(output_path);
    eprintln!(
        "Generated {} with {} L1 markets, {} L2 markets, {} originality markets",
        output_path.display(),
        l1_json.markets_data.len(),
        l2_json.markets_data.len(),
        originality_json.markets_data.len()
    );

    Ok(())
}
