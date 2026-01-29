use alloy::primitives::{address, Address, Bytes, Uint};
use alloy::providers::ProviderBuilder;
use alloy::sol;
use alloy::sol_types::SolCall;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fmt::Write;
use std::fs;
use std::path::Path;

// Uniswap V3 Factory getPool call
sol! {
    function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool);
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

// --- Structs for CSV parsing ---

#[derive(Debug, Deserialize)]
struct WeightRecord {
    repo: String,
    #[serde(rename = "pareant")]
    _parent: String,
    weight: f64,
}

// --- Structs for JSON parsing ---

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
    tick_idx: String,
    liquidity_net: String,
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


fn generate_weights_rs(csv_path: &Path, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(csv_path)?;
    let mut weights = Vec::new();

    for result in reader.deserialize() {
        let record: WeightRecord = result?;
        let repo = record.repo.trim();
        let market = repo.strip_prefix("https://github.com/").unwrap_or(repo);
        weights.push((market.to_string(), record.weight));
    }

    let mut output = String::new();
    writeln!(&mut output, "// Auto-generated from weights.csv - do not edit manually\n")?;
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct Weight {{")?;
    writeln!(&mut output, "    pub market: &'static str,")?;
    writeln!(&mut output, "    pub prediction: f64,")?;
    writeln!(&mut output, "}}\n")?;
    writeln!(
        &mut output,
        "pub static WEIGHTS: [Weight; {}] = [",
        weights.len()
    )?;

    for (repo, weight) in &weights {
        writeln!(
            &mut output,
            "    Weight {{ market: \"{}\", prediction: {} }},",
            repo, weight
        )?;
    }

    writeln!(&mut output, "];")?;

    fs::write(output_path, output)?;
    println!("cargo::warning=Generated {} with {} weights", output_path.display(), weights.len());

    Ok(())
}

// Generate L1 markets array (single pool per market)
fn generate_markets_l1_array(
    markets: &BTreeMap<String, MarketInfo>,
    resolved_pools: &BTreeMap<String, Vec<ResolvedPool>>,
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
                            "    Tick {{ tick_idx: \"{}\", liquidity_net: \"{}\" }},",
                            tick.tick_idx, tick.liquidity_net
                        )?;
                    }
                    writeln!(&mut tick_definitions, "];")?;
                    format!("&{}", tick_static_name)
                };

                writeln!(&mut market_entries, "        pool: Some(Pool {{")?;
                writeln!(&mut market_entries, "            token0: \"{}\",", pool.token0)?;
                writeln!(&mut market_entries, "            token1: \"{}\",", pool.token1)?;
                writeln!(&mut market_entries, "            pool_id: \"{}\",", pool_id_str)?;
                writeln!(&mut market_entries, "            liquidity: \"{}\",", liquidity_str)?;
                writeln!(&mut market_entries, "            ticks: {},", ticks_ref)?;
                writeln!(&mut market_entries, "        }}),")?;
            }
            None => {
                writeln!(&mut market_entries, "        pool: None,")?;
            }
        }

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

// Generate L2 markets array (multiple pools per market) - single pass
fn generate_markets_l2_array(
    markets: &BTreeMap<String, MarketInfo>,
    resolved_pools: &BTreeMap<String, Vec<ResolvedPool>>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    let mut pool_definitions = String::new();
    let mut market_entries = String::new();

    for (name, market_data) in markets {
        let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");
        let lower_name = escaped_name.to_lowercase();
        let safe_name = lower_name.replace(['/', '-', '.', '\\'], "_");
        let market_id = market_data.market_id.as_deref().unwrap_or("");

        let pool_ref = if let Some(pools) = resolved_pools.get(name).filter(|p| !p.is_empty()) {
            let static_name = format!("POOLS_{}", safe_name.to_uppercase());

            // Generate tick arrays for each pool first
            for (pool_idx, pool) in pools.iter().enumerate() {
                if !pool.ticks.is_empty() {
                    let tick_static_name = format!("TICKS_{}_{}", safe_name.to_uppercase(), pool_idx);
                    writeln!(&mut pool_definitions, "static {}: [Tick; {}] = [", tick_static_name, pool.ticks.len())?;
                    for tick in &pool.ticks {
                        writeln!(&mut pool_definitions,
                            "    Tick {{ tick_idx: \"{}\", liquidity_net: \"{}\" }},",
                            tick.tick_idx, tick.liquidity_net
                        )?;
                    }
                    writeln!(&mut pool_definitions, "];")?;
                }
            }

            // Generate pool array
            writeln!(&mut pool_definitions, "static {}: [Pool; {}] = [", static_name, pools.len())?;
            for (pool_idx, pool) in pools.iter().enumerate() {
                let pool_id_str = pool.pool_id.as_deref().unwrap_or("");
                let liquidity_str = pool.liquidity.as_deref().unwrap_or("");
                let ticks_ref = if pool.ticks.is_empty() {
                    "&[]".to_string()
                } else {
                    format!("&TICKS_{}_{}", safe_name.to_uppercase(), pool_idx)
                };
                writeln!(&mut pool_definitions,
                    "    Pool {{ token0: \"{}\", token1: \"{}\", pool_id: \"{}\", liquidity: \"{}\", ticks: {} }},",
                    pool.token0, pool.token1, pool_id_str, liquidity_str, ticks_ref
                )?;
            }
            writeln!(&mut pool_definitions, "];")?;
            format!("&{}", static_name)
        } else {
            "&[]".to_string()
        };

        writeln!(&mut market_entries, "    MarketDataL2 {{")?;
        writeln!(&mut market_entries, "        name: \"{}\",", lower_name)?;
        writeln!(&mut market_entries, "        market_id: \"{}\",", market_id)?;
        writeln!(&mut market_entries, "        outcome_token: \"{}\",", market_data.outcome_token)?;
        writeln!(&mut market_entries, "        pools: {},", pool_ref)?;
        writeln!(&mut market_entries, "    }},")?;
    }

    output.push_str(&pool_definitions);
    if !pool_definitions.is_empty() {
        writeln!(&mut output)?;
    }
    writeln!(&mut output, "pub static MARKETS_L2: [MarketDataL2; {}] = [", markets.len())?;
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
                .filter(|t| t.liquidity_net != "0")
                .map(|t| ResolvedTick {
                    tick_idx: t.tick_idx.clone(),
                    liquidity_net: t.liquidity_net.clone(),
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
    let (Ok(token0), Ok(token1)) = (pool.token0.parse::<Address>(), pool.token1.parse::<Address>()) else {
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
    let provider = ProviderBuilder::new().connect_http(rpc_url.parse()?);
    let multicall = Multicall3::new(MULTICALL3_ADDRESS, provider);

    let (calls, call_infos) = build_pool_calls(markets, first_only);
    println!(
        "cargo::warning=Batching {} getPool calls via Multicall3...",
        calls.len()
    );

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

async fn generate_markets_rs(
    l1_json_path: &Path,
    l2_json_path: &Path,
    output_path: &Path,
    rpc_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let l1_content = fs::read_to_string(l1_json_path)?;
    let l1_json: MarketsFile = serde_json::from_str(&l1_content)?;

    let l2_content = fs::read_to_string(l2_json_path)?;
    let l2_json: MarketsFile = serde_json::from_str(&l2_content)?;

    // Resolve pool addresses: first pool only for L1, all pools for L2
    println!("cargo::warning=Resolving pool addresses for L1 markets (first pool only)...");
    let l1_resolved_pools = resolve_pool_addresses(&l1_json.markets_data, rpc_url, true).await?;

    println!("cargo::warning=Resolving pool addresses for L2 markets (all pools)...");
    let l2_resolved_pools = resolve_pool_addresses(&l2_json.markets_data, rpc_url, false).await?;

    let mut output = String::new();
    writeln!(&mut output, "// Auto-generated from markets_data_l1.json and markets_data_l2.json - do not edit manually\n")?;

    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct Tick {{")?;
    writeln!(&mut output, "    pub tick_idx: &'static str,")?;
    writeln!(&mut output, "    pub liquidity_net: &'static str,")?;
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
    writeln!(&mut output, "}}\n")?;

    // L2 MarketDataL2 (multiple pools)
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct MarketDataL2 {{")?;
    writeln!(&mut output, "    pub name: &'static str,")?;
    writeln!(&mut output, "    pub market_id: &'static str,")?;
    writeln!(&mut output, "    pub outcome_token: &'static str,")?;
    writeln!(&mut output, "    pub pools: &'static [Pool],")?;
    writeln!(&mut output, "}}\n")?;

    output.push_str(&generate_markets_l1_array(&l1_json.markets_data, &l1_resolved_pools)?);
    writeln!(&mut output)?;

    output.push_str(&generate_markets_l2_array(&l2_json.markets_data, &l2_resolved_pools)?);

    fs::write(output_path, output)?;
    println!(
        "cargo::warning=Generated {} with {} L1 markets and {} L2 markets",
        output_path.display(),
        l1_json.markets_data.len(),
        l2_json.markets_data.len()
    );

    Ok(())
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = Path::new(&manifest_dir);
    let env_path = manifest_path.join(".env");

    if env_path.exists() {
        dotenvy::from_path(&env_path).ok();
    }

    let rpc_url = env::var("RPC").unwrap_or_else(|_| {
        panic!("RPC environment variable not set. Please set it in .env file.");
    });

    let weights_csv = manifest_path.join("weights.csv");
    let markets_l1_json = manifest_path.join("markets_data_l1.json");
    let markets_l2_json = manifest_path.join("markets_data_l2.json");
    let weights_rs = manifest_path.join("src/weights.rs");
    let markets_rs = manifest_path.join("src/markets.rs");

    println!("cargo::rerun-if-changed=weights.csv");
    println!("cargo::rerun-if-changed=markets_data_l1.json");
    println!("cargo::rerun-if-changed=markets_data_l2.json");
    println!("cargo::rerun-if-changed=.env");

    if weights_csv.exists() {
        if let Err(e) = generate_weights_rs(&weights_csv, &weights_rs) {
            panic!("Failed to generate weights.rs: {}", e);
        }
    }
    if !markets_l1_json.exists() {
        panic!("markets_data_l1.json not found. Run `cargo test test_prepare` to fetch market data.");
    }
    if !markets_l2_json.exists() {
        panic!("markets_data_l2.json not found. Run `cargo test test_prepare` to fetch market data.");
    }

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    if let Err(e) = rt.block_on(generate_markets_rs(&markets_l1_json, &markets_l2_json, &markets_rs, &rpc_url)) {
        panic!("Failed to generate markets.rs: {}", e);
    }
}
