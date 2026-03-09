use std::collections::HashMap;
use std::fmt;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use alloy::consensus::SignableTransaction;
use alloy::hex;
use alloy::network::TransactionBuilder;
use alloy::primitives::{Address, Bytes, U256};
use alloy::rpc::types::TransactionRequest;
use alloy::sol;
use alloy::sol_types::SolCall;
use tokio::sync::Mutex;

use super::batch_bounds::derive_batch_quote_bounds_unchecked;
use super::tx_builder::{TxBuildError, build_trade_executor_calls};
use super::{ExecutionGroupPlan, GroupKind, ITradeExecutor, SUSD_DECIMALS};
use crate::portfolio::Action;

const OP_GAS_PRICE_ORACLE_ADDR: &str = "0x420000000000000000000000000000000000000F";
const WEI_PER_ETH: f64 = 1e18;
const L1_FEE_CACHE_TTL: Duration = Duration::from_secs(60);
const L1_FEE_RPC_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const L1_FEE_SLOPE_SAMPLE_SMALL_BYTES: usize = 256;
const L1_FEE_SLOPE_SAMPLE_LARGE_BYTES: usize = 512;
/// Build a non-uniform probe payload that approximates real calldata byte entropy.
/// Uses a deterministic LCG so the result is reproducible and non-trivially compressible.
/// Any single repeated byte (0x00, 0x01, 0xAB, etc.) compresses to ~11 bytes under
/// Fjord/Brotli regardless of value — this sequence avoids that collapse.
fn make_l1_fee_probe_bytes(len: usize) -> Vec<u8> {
    let mut state: u32 = 0xDEAD_BEEF;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        })
        .collect()
}
static OP_L1_FEE_PER_BYTE_WEI_CACHE: OnceLock<RwLock<HashMap<String, CachedL1FeePerByteWei>>> =
    OnceLock::new();
static OP_L1_FEE_PER_BYTE_FETCH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

sol! {
    function getL1Fee(bytes payload) external view returns (uint256);
}

#[derive(Debug, Clone, Copy)]
pub struct GasAssumptions {
    pub direct_buy_l2_units: u64,
    pub direct_sell_l2_units: u64,
    pub direct_merge_l2_units: u64,
    pub mint_sell_base_l2_units: u64,
    pub mint_sell_per_sell_leg_l2_units: u64,
    pub buy_merge_base_l2_units: u64,
    pub buy_merge_per_buy_leg_l2_units: u64,
    pub l1_data_fee_floor_susd: f64,
    pub l1_fee_per_byte_wei: f64,
}

impl Default for GasAssumptions {
    fn default() -> Self {
        Self {
            direct_buy_l2_units: 57_542,
            direct_sell_l2_units: 38_099,
            direct_merge_l2_units: 21_502,
            mint_sell_base_l2_units: 17_783,
            mint_sell_per_sell_leg_l2_units: 50_649,
            buy_merge_base_l2_units: 37_370,
            buy_merge_per_buy_leg_l2_units: 29_670,
            l1_data_fee_floor_susd: 0.001,
            l1_fee_per_byte_wei: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveOptimismFeeInputs {
    pub chain_id: u64,
    pub sender_nonce: u64,
    pub gas_price_wei: u128,
}

#[derive(Debug, Clone)]
pub struct ExactExecutionGasQuote {
    pub l2_gas_units: u64,
    pub gas_price_wei: u128,
    pub l2_fee_wei: U256,
    pub l1_fee_wei: U256,
    pub l2_fee_susd: f64,
    pub l1_fee_susd: f64,
    pub total_fee_susd: f64,
    pub net_ev_susd: f64,
    pub unsigned_tx_data: Bytes,
}

#[derive(Debug, Clone)]
pub struct ExactGasQuoteError(String);

impl ExactGasQuoteError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ExactGasQuoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ExactGasQuoteError {}

impl From<L1FeeOracleError> for ExactGasQuoteError {
    fn from(value: L1FeeOracleError) -> Self {
        Self::new(value.to_string())
    }
}

impl From<TxBuildError> for ExactGasQuoteError {
    fn from(value: TxBuildError) -> Self {
        Self::new(value.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct L1FeeOracleError(String);

impl L1FeeOracleError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for L1FeeOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for L1FeeOracleError {}

#[derive(Debug, Clone, Copy)]
struct CachedL1FeePerByteWei {
    fee_per_byte_wei: f64,
    fetched_at: Instant,
}

fn l1_fee_cache() -> &'static RwLock<HashMap<String, CachedL1FeePerByteWei>> {
    OP_L1_FEE_PER_BYTE_WEI_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn l1_fee_fetch_lock() -> &'static Mutex<()> {
    OP_L1_FEE_PER_BYTE_FETCH_LOCK.get_or_init(|| Mutex::new(()))
}

fn fresh_cached_l1_fee_per_byte_wei(rpc_url: &str) -> Option<f64> {
    let guard = l1_fee_cache().read().ok()?;
    let cached = guard.get(rpc_url)?;
    if cached.fetched_at.elapsed() > L1_FEE_CACHE_TTL {
        return None;
    }
    Some(cached.fee_per_byte_wei)
}

fn fresh_single_cached_l1_fee_per_byte_wei() -> Option<f64> {
    let guard = l1_fee_cache().read().ok()?;
    let mut fresh = guard
        .values()
        .filter(|cached| cached.fetched_at.elapsed() <= L1_FEE_CACHE_TTL)
        .map(|cached| cached.fee_per_byte_wei);
    let value = fresh.next()?;
    if fresh.next().is_some() {
        return None;
    }
    Some(value)
}

pub fn cached_optimism_l1_fee_per_byte_wei_value() -> Option<f64> {
    fresh_single_cached_l1_fee_per_byte_wei()
}

fn effective_l1_fee_per_byte_wei_with_cache(
    assumptions: &GasAssumptions,
    cached_fee_per_byte_wei: Option<f64>,
) -> Option<f64> {
    if assumptions.l1_fee_per_byte_wei.is_finite() && assumptions.l1_fee_per_byte_wei > 0.0 {
        return Some(assumptions.l1_fee_per_byte_wei);
    }
    cached_fee_per_byte_wei
}

pub fn resolve_l1_fee_per_byte_wei(assumptions: &GasAssumptions) -> Option<f64> {
    effective_l1_fee_per_byte_wei_with_cache(
        assumptions,
        cached_optimism_l1_fee_per_byte_wei_value(),
    )
}

pub fn estimate_group_l2_gas_units(
    assumptions: &GasAssumptions,
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
) -> u64 {
    match kind {
        GroupKind::DirectBuy => assumptions.direct_buy_l2_units,
        GroupKind::DirectSell => assumptions.direct_sell_l2_units,
        GroupKind::DirectMerge => assumptions.direct_merge_l2_units,
        GroupKind::MintSell => assumptions.mint_sell_base_l2_units.saturating_add(
            assumptions
                .mint_sell_per_sell_leg_l2_units
                .saturating_mul(sell_legs as u64),
        ),
        GroupKind::BuyMerge => assumptions.buy_merge_base_l2_units.saturating_add(
            assumptions
                .buy_merge_per_buy_leg_l2_units
                .saturating_mul(buy_legs as u64),
        ),
    }
}

pub fn estimate_group_incremental_calldata_bytes(
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
) -> u64 {
    const SWAP_BYTES: u64 = 224;
    const FLASH_ROUTE_EXTRA_BYTES: u64 = 160;
    const DIRECT_MERGE_CALL_BYTES: u64 = 220;

    match kind {
        GroupKind::DirectBuy | GroupKind::DirectSell => SWAP_BYTES,
        GroupKind::MintSell => {
            let legs = sell_legs.max(1) as u64;
            FLASH_ROUTE_EXTRA_BYTES + SWAP_BYTES.saturating_mul(legs)
        }
        GroupKind::BuyMerge => {
            let legs = buy_legs.max(1) as u64;
            FLASH_ROUTE_EXTRA_BYTES + SWAP_BYTES.saturating_mul(legs)
        }
        GroupKind::DirectMerge => DIRECT_MERGE_CALL_BYTES,
    }
}

pub fn estimate_group_calldata_bytes(kind: GroupKind, buy_legs: usize, sell_legs: usize) -> u64 {
    const TX_ENVELOPE_BYTES: u64 = 110;
    const BATCH_CALL_BASE_BYTES: u64 = 100;
    TX_ENVELOPE_BYTES
        + BATCH_CALL_BASE_BYTES
        + estimate_group_incremental_calldata_bytes(kind, buy_legs, sell_legs)
}

pub fn estimate_l2_gas_susd(gas_units: u64, gas_price_eth: f64, eth_usd_assumed: f64) -> f64 {
    if !gas_price_eth.is_finite()
        || !eth_usd_assumed.is_finite()
        || gas_price_eth < 0.0
        || eth_usd_assumed < 0.0
    {
        return f64::INFINITY;
    }
    gas_units as f64 * gas_price_eth * eth_usd_assumed
}

pub fn estimate_group_l1_data_fee_susd(
    assumptions: &GasAssumptions,
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
    eth_usd_assumed: f64,
) -> f64 {
    if !eth_usd_assumed.is_finite()
        || eth_usd_assumed < 0.0
        || !assumptions.l1_data_fee_floor_susd.is_finite()
        || assumptions.l1_data_fee_floor_susd < 0.0
    {
        return f64::INFINITY;
    }

    let l1_fee_per_byte_wei = assumptions.l1_fee_per_byte_wei;
    if !l1_fee_per_byte_wei.is_finite() || l1_fee_per_byte_wei < 0.0 {
        return f64::INFINITY;
    }
    if l1_fee_per_byte_wei == 0.0 {
        return f64::INFINITY;
    }

    let calldata_bytes = estimate_group_calldata_bytes(kind, buy_legs, sell_legs) as f64;
    let l1_scaled_susd = calldata_bytes * l1_fee_per_byte_wei * eth_usd_assumed / WEI_PER_ETH;
    if !l1_scaled_susd.is_finite() {
        return f64::INFINITY;
    }

    assumptions.l1_data_fee_floor_susd.max(l1_scaled_susd)
}

pub fn estimate_group_incremental_l1_data_fee_susd(
    assumptions: &GasAssumptions,
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
    eth_usd_assumed: f64,
) -> f64 {
    if !eth_usd_assumed.is_finite() || eth_usd_assumed < 0.0 {
        return f64::INFINITY;
    }

    let l1_fee_per_byte_wei = assumptions.l1_fee_per_byte_wei;
    if !l1_fee_per_byte_wei.is_finite() || l1_fee_per_byte_wei < 0.0 {
        return f64::INFINITY;
    }
    if l1_fee_per_byte_wei == 0.0 {
        return 0.0;
    }

    let calldata_bytes =
        estimate_group_incremental_calldata_bytes(kind, buy_legs, sell_legs) as f64;
    let l1_scaled_susd = calldata_bytes * l1_fee_per_byte_wei * eth_usd_assumed / WEI_PER_ETH;
    if !l1_scaled_susd.is_finite() || l1_scaled_susd < 0.0 {
        return f64::INFINITY;
    }

    l1_scaled_susd
}

pub fn estimate_total_gas_susd(
    assumptions: &GasAssumptions,
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
    gas_units: u64,
    gas_price_eth: f64,
    eth_usd_assumed: f64,
) -> f64 {
    let gas_l2_susd = estimate_l2_gas_susd(gas_units, gas_price_eth, eth_usd_assumed);
    let gas_l1_susd =
        estimate_group_l1_data_fee_susd(assumptions, kind, buy_legs, sell_legs, eth_usd_assumed);
    if !gas_l2_susd.is_finite() || !gas_l1_susd.is_finite() {
        return f64::INFINITY;
    }
    gas_l2_susd + gas_l1_susd
}

/// Estimated total gas cost in sUSD for a single trade group of the given kind.
///
/// Uses the L1 data fee floor when no L1 fee-per-byte is set.
/// `sell_legs` / `buy_legs` are the number of swap legs in the group.
pub fn estimate_min_gas_susd_for_group(
    assumptions: &GasAssumptions,
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
    gas_price_eth: f64,
    eth_usd: f64,
) -> f64 {
    let l2_units = estimate_group_l2_gas_units(assumptions, kind, buy_legs, sell_legs);
    let gas_l2 = estimate_l2_gas_susd(l2_units, gas_price_eth, eth_usd);
    if !gas_l2.is_finite() || gas_l2 < 0.0 {
        return f64::INFINITY;
    }

    let gas_l1 = if assumptions.l1_fee_per_byte_wei > 0.0 {
        estimate_group_l1_data_fee_susd(assumptions, kind, buy_legs, sell_legs, eth_usd)
    } else {
        assumptions.l1_data_fee_floor_susd
    };
    if !gas_l1.is_finite() || gas_l1 < 0.0 {
        return f64::INFINITY;
    }
    gas_l2 + gas_l1
}

pub fn estimate_min_incremental_gas_susd_for_group(
    assumptions: &GasAssumptions,
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
    gas_price_eth: f64,
    eth_usd: f64,
) -> f64 {
    let l2_units = estimate_group_l2_gas_units(assumptions, kind, buy_legs, sell_legs);
    let gas_l2 = estimate_l2_gas_susd(l2_units, gas_price_eth, eth_usd);
    if !gas_l2.is_finite() || gas_l2 < 0.0 {
        return f64::INFINITY;
    }
    let gas_l1 = estimate_group_incremental_l1_data_fee_susd(
        assumptions,
        kind,
        buy_legs,
        sell_legs,
        eth_usd,
    );
    if !gas_l1.is_finite() || gas_l1 < 0.0 {
        return f64::INFINITY;
    }
    gas_l2 + gas_l1
}

fn build_rpc_client() -> Result<reqwest::Client, L1FeeOracleError> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(L1_FEE_RPC_REQUEST_TIMEOUT)
        .build()
        .map_err(|e| L1FeeOracleError::new(format!("failed to build rpc client: {e}")))
}

fn encode_unsigned_transaction(tx: TransactionRequest) -> Result<Bytes, ExactGasQuoteError> {
    let typed_tx = tx
        .build_unsigned()
        .map_err(|err| ExactGasQuoteError::new(format!("failed to build unsigned tx: {err}")))?;
    let mut encoded = Vec::new();
    typed_tx.encode_for_signing(&mut encoded);
    Ok(Bytes::from(encoded))
}

pub fn estimate_l1_data_fee_susd_for_tx_bytes_len(
    assumptions: &GasAssumptions,
    tx_data_len: usize,
    eth_usd_assumed: f64,
) -> f64 {
    if !eth_usd_assumed.is_finite()
        || eth_usd_assumed < 0.0
        || !assumptions.l1_data_fee_floor_susd.is_finite()
        || assumptions.l1_data_fee_floor_susd < 0.0
    {
        return f64::INFINITY;
    }

    let byte_scaled_susd =
        if assumptions.l1_fee_per_byte_wei.is_finite() && assumptions.l1_fee_per_byte_wei > 0.0 {
            tx_data_len as f64 * assumptions.l1_fee_per_byte_wei * eth_usd_assumed / WEI_PER_ETH
        } else {
            0.0
        };
    if !byte_scaled_susd.is_finite() || byte_scaled_susd < 0.0 {
        return f64::INFINITY;
    }

    assumptions.l1_data_fee_floor_susd.max(byte_scaled_susd)
}

pub fn gas_price_eth_to_wei(gas_price_eth: f64) -> Option<u128> {
    if !gas_price_eth.is_finite() || gas_price_eth < 0.0 {
        return None;
    }
    let gas_price_wei = gas_price_eth * WEI_PER_ETH;
    if !gas_price_wei.is_finite() || gas_price_wei < 0.0 || gas_price_wei > u128::MAX as f64 {
        return None;
    }
    Some(gas_price_wei.round() as u128)
}

pub fn build_unsigned_contract_call_tx_bytes(
    to: Address,
    calldata: Bytes,
    fee_inputs: LiveOptimismFeeInputs,
    gas_limit: u64,
) -> Result<Bytes, ExactGasQuoteError> {
    let tx = TransactionRequest::default()
        .to(to)
        .nonce(fee_inputs.sender_nonce)
        .with_chain_id(fee_inputs.chain_id)
        .gas_limit(gas_limit.max(21_000))
        .max_fee_per_gas(fee_inputs.gas_price_wei)
        .max_priority_fee_per_gas(0)
        .input(calldata.into());
    encode_unsigned_transaction(tx)
}

pub fn wei_to_susd(wei: U256, eth_usd_assumed: f64) -> f64 {
    if !eth_usd_assumed.is_finite() || eth_usd_assumed < 0.0 {
        return f64::INFINITY;
    }
    let wei_f64 = f64::from(wei);
    if !wei_f64.is_finite() || wei_f64 < 0.0 {
        return f64::INFINITY;
    }
    wei_f64 * eth_usd_assumed / WEI_PER_ETH
}

fn parse_hex_u256_result(body: serde_json::Value) -> Result<U256, L1FeeOracleError> {
    let result_hex = body
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or_else(|| L1FeeOracleError::new("missing rpc result"))?;
    let raw_hex = result_hex.trim_start_matches("0x");
    let padded_hex = if raw_hex.len() % 2 == 1 {
        format!("0{raw_hex}")
    } else {
        raw_hex.to_string()
    };
    let raw_bytes = hex::decode(padded_hex)
        .map_err(|e| L1FeeOracleError::new(format!("invalid rpc result hex: {e}")))?;
    U256::try_from_be_slice(&raw_bytes).ok_or_else(|| {
        L1FeeOracleError::new(format!(
            "rpc result does not fit into U256 ({} bytes)",
            raw_bytes.len()
        ))
    })
}

fn parse_hex_u64_result(body: serde_json::Value, label: &str) -> Result<u64, L1FeeOracleError> {
    let value = parse_hex_u256_result(body)?;
    u64::try_from(value).map_err(|_| {
        L1FeeOracleError::new(format!("{label} result does not fit into u64: {value}"))
    })
}

fn parse_hex_u128_result(body: serde_json::Value, label: &str) -> Result<u128, L1FeeOracleError> {
    let value = parse_hex_u256_result(body)?;
    u128::try_from(value).map_err(|_| {
        L1FeeOracleError::new(format!("{label} result does not fit into u128: {value}"))
    })
}

async fn post_rpc_json(
    client: &reqwest::Client,
    rpc_url: &str,
    payload: serde_json::Value,
) -> Result<serde_json::Value, L1FeeOracleError> {
    let response = client
        .post(rpc_url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| L1FeeOracleError::new(format!("rpc request failed: {e}")))?
        .error_for_status()
        .map_err(|e| L1FeeOracleError::new(format!("rpc bad http status: {e}")))?;

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| L1FeeOracleError::new(format!("failed to decode rpc body: {e}")))?;
    if let Some(err) = body.get("error") {
        return Err(L1FeeOracleError::new(format!("rpc returned error: {err}")));
    }
    Ok(body)
}

async fn rpc_get_hex_u128(
    client: &reqwest::Client,
    rpc_url: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<u128, L1FeeOracleError> {
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    });
    parse_hex_u128_result(post_rpc_json(client, rpc_url, payload).await?, method)
}

async fn rpc_get_hex_u64(
    client: &reqwest::Client,
    rpc_url: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<u64, L1FeeOracleError> {
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    });
    parse_hex_u64_result(post_rpc_json(client, rpc_url, payload).await?, method)
}

pub async fn fetch_live_optimism_fee_inputs(
    rpc_url: &str,
    sender: Address,
) -> Result<LiveOptimismFeeInputs, ExactGasQuoteError> {
    if rpc_url.trim().is_empty() {
        return Err(ExactGasQuoteError::new("rpc url is empty"));
    }

    let client = build_rpc_client()?;
    let gas_price_wei =
        rpc_get_hex_u128(&client, rpc_url, "eth_gasPrice", serde_json::json!([])).await?;
    let chain_id = rpc_get_hex_u64(&client, rpc_url, "eth_chainId", serde_json::json!([])).await?;
    let sender_nonce = rpc_get_hex_u64(
        &client,
        rpc_url,
        "eth_getTransactionCount",
        serde_json::json!([format!("{sender:#x}"), "latest"]),
    )
    .await?;

    Ok(LiveOptimismFeeInputs {
        chain_id,
        sender_nonce,
        gas_price_wei,
    })
}

pub fn build_unsigned_batch_execute_tx_bytes(
    executor: Address,
    calls: &[ITradeExecutor::Call],
    fee_inputs: LiveOptimismFeeInputs,
    gas_limit: u64,
) -> Result<Bytes, ExactGasQuoteError> {
    let calldata = ITradeExecutor::batchExecuteCall {
        calls: calls.to_vec(),
    }
    .abi_encode();
    build_unsigned_contract_call_tx_bytes(executor, calldata.into(), fee_inputs, gas_limit)
}

pub fn build_unsigned_group_plan_batch_execute_tx_bytes(
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    fee_inputs: LiveOptimismFeeInputs,
) -> Result<Bytes, ExactGasQuoteError> {
    let batch_bounds = derive_batch_quote_bounds_unchecked(plan)
        .map_err(|err| ExactGasQuoteError::new(format!("failed to derive batch bounds: {err}")))?
        .map(|bounds| bounds.to_token_bounds(SUSD_DECIMALS))
        .transpose()
        .map_err(|err| ExactGasQuoteError::new(format!("failed to convert batch bounds: {err}")))?;
    let calls = build_trade_executor_calls(executor, actions, plan, batch_bounds)?;
    build_unsigned_batch_execute_tx_bytes(executor, &calls, fee_inputs, plan.l2_gas_units)
}

async fn fetch_optimism_l1_fee_wei_for_tx_data(
    client: &reqwest::Client,
    rpc_url: &str,
    tx_data: Bytes,
) -> Result<U256, ExactGasQuoteError> {
    let call_data = getL1FeeCall { payload: tx_data }.abi_encode();
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [
            {
                "to": OP_GAS_PRICE_ORACLE_ADDR,
                "data": format!("0x{}", hex::encode(call_data)),
            },
            "latest"
        ]
    });
    let body = post_rpc_json(client, rpc_url, payload).await?;
    parse_hex_u256_result(body).map_err(ExactGasQuoteError::from)
}

pub async fn fetch_exact_optimism_l1_fee_wei_for_tx_data(
    rpc_url: &str,
    tx_data: Bytes,
) -> Result<U256, ExactGasQuoteError> {
    let client = build_rpc_client()?;
    fetch_optimism_l1_fee_wei_for_tx_data(&client, rpc_url, tx_data).await
}

pub async fn quote_group_plan_exact_gas(
    rpc_url: &str,
    sender: Address,
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    eth_usd_assumed: f64,
) -> Result<ExactExecutionGasQuote, ExactGasQuoteError> {
    let fee_inputs = fetch_live_optimism_fee_inputs(rpc_url, sender).await?;
    quote_group_plan_exact_gas_with_fee_inputs(
        rpc_url,
        executor,
        actions,
        plan,
        fee_inputs,
        eth_usd_assumed,
    )
    .await
}

pub async fn quote_group_plan_exact_gas_with_fee_inputs(
    rpc_url: &str,
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
    fee_inputs: LiveOptimismFeeInputs,
    eth_usd_assumed: f64,
) -> Result<ExactExecutionGasQuote, ExactGasQuoteError> {
    let unsigned_tx_data =
        build_unsigned_group_plan_batch_execute_tx_bytes(executor, actions, plan, fee_inputs)?;
    let client = build_rpc_client()?;
    let l1_fee_wei =
        fetch_optimism_l1_fee_wei_for_tx_data(&client, rpc_url, unsigned_tx_data.clone()).await?;
    let l2_fee_wei = U256::from(fee_inputs.gas_price_wei) * U256::from(plan.l2_gas_units);
    let l2_fee_susd = wei_to_susd(l2_fee_wei, eth_usd_assumed);
    let l1_fee_susd = wei_to_susd(l1_fee_wei, eth_usd_assumed);
    let total_fee_susd = l1_fee_susd + l2_fee_susd;

    Ok(ExactExecutionGasQuote {
        l2_gas_units: plan.l2_gas_units,
        gas_price_wei: fee_inputs.gas_price_wei,
        l2_fee_wei,
        l1_fee_wei,
        l2_fee_susd,
        l1_fee_susd,
        total_fee_susd,
        net_ev_susd: plan.edge_plan_susd - total_fee_susd,
        unsigned_tx_data,
    })
}

pub async fn estimate_batch_execute_l2_gas_units_live(
    rpc_url: &str,
    sender: Address,
    executor: Address,
    calls: &[ITradeExecutor::Call],
) -> Result<u64, ExactGasQuoteError> {
    let calldata = ITradeExecutor::batchExecuteCall {
        calls: calls.to_vec(),
    }
    .abi_encode();
    let client = build_rpc_client()?;
    rpc_get_hex_u64(
        &client,
        rpc_url,
        "eth_estimateGas",
        serde_json::json!([{
            "from": format!("{sender:#x}"),
            "to": format!("{executor:#x}"),
            "data": format!("0x{}", hex::encode(calldata)),
        }]),
    )
    .await
    .map_err(ExactGasQuoteError::from)
}

pub async fn estimate_group_plan_l2_gas_units_live(
    rpc_url: &str,
    sender: Address,
    executor: Address,
    actions: &[Action],
    plan: &ExecutionGroupPlan,
) -> Result<u64, ExactGasQuoteError> {
    let batch_bounds = derive_batch_quote_bounds_unchecked(plan)
        .map_err(|err| ExactGasQuoteError::new(format!("failed to derive batch bounds: {err}")))?
        .map(|bounds| bounds.to_token_bounds(SUSD_DECIMALS))
        .transpose()
        .map_err(|err| ExactGasQuoteError::new(format!("failed to convert batch bounds: {err}")))?;
    let calls = build_trade_executor_calls(executor, actions, plan, batch_bounds)?;
    estimate_batch_execute_l2_gas_units_live(rpc_url, sender, executor, &calls).await
}

pub async fn cached_optimism_l1_fee_per_byte_wei(rpc_url: &str) -> Result<f64, L1FeeOracleError> {
    if let Some(cached) = fresh_cached_l1_fee_per_byte_wei(rpc_url) {
        return Ok(cached);
    }

    let _refresh_guard = l1_fee_fetch_lock().lock().await;
    if let Some(cached) = fresh_cached_l1_fee_per_byte_wei(rpc_url) {
        return Ok(cached);
    }

    let fetched = fetch_optimism_l1_fee_per_byte_wei(rpc_url).await?;
    let mut guard = l1_fee_cache()
        .write()
        .map_err(|_| L1FeeOracleError::new("failed to acquire l1 fee cache write lock"))?;
    guard.insert(
        rpc_url.to_owned(),
        CachedL1FeePerByteWei {
            fee_per_byte_wei: fetched,
            fetched_at: Instant::now(),
        },
    );
    Ok(fetched)
}

pub async fn hydrate_cached_optimism_l1_fee_per_byte(
    assumptions: &mut GasAssumptions,
    rpc_url: &str,
) -> Result<f64, L1FeeOracleError> {
    let fee_per_byte = cached_optimism_l1_fee_per_byte_wei(rpc_url).await?;
    assumptions.l1_fee_per_byte_wei = fee_per_byte;
    Ok(fee_per_byte)
}

pub async fn default_gas_assumptions_with_optimism_l1_fee(
    rpc_url: &str,
) -> Result<GasAssumptions, L1FeeOracleError> {
    let mut assumptions = GasAssumptions::default();
    hydrate_cached_optimism_l1_fee_per_byte(&mut assumptions, rpc_url).await?;
    Ok(assumptions)
}

async fn fetch_optimism_l1_fee_per_byte_wei(rpc_url: &str) -> Result<f64, L1FeeOracleError> {
    if rpc_url.trim().is_empty() {
        return Err(L1FeeOracleError::new("rpc url is empty"));
    }

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(L1_FEE_RPC_REQUEST_TIMEOUT)
        .build()
        .map_err(|e| L1FeeOracleError::new(format!("failed to build rpc client: {e}")))?;

    let small_fee_wei = fetch_optimism_l1_fee_wei_for_payload_len(
        &client,
        rpc_url,
        L1_FEE_SLOPE_SAMPLE_SMALL_BYTES,
    )
    .await?;
    let large_fee_wei = fetch_optimism_l1_fee_wei_for_payload_len(
        &client,
        rpc_url,
        L1_FEE_SLOPE_SAMPLE_LARGE_BYTES,
    )
    .await?;
    marginal_fee_per_byte_wei(
        small_fee_wei,
        large_fee_wei,
        L1_FEE_SLOPE_SAMPLE_SMALL_BYTES,
        L1_FEE_SLOPE_SAMPLE_LARGE_BYTES,
    )
}

fn marginal_fee_per_byte_wei(
    smaller_fee_wei: f64,
    larger_fee_wei: f64,
    smaller_payload_bytes: usize,
    larger_payload_bytes: usize,
) -> Result<f64, L1FeeOracleError> {
    if larger_payload_bytes <= smaller_payload_bytes {
        return Err(L1FeeOracleError::new(format!(
            "invalid payload span for l1 fee slope: {smaller_payload_bytes} -> {larger_payload_bytes}"
        )));
    }
    if !smaller_fee_wei.is_finite()
        || !larger_fee_wei.is_finite()
        || smaller_fee_wei < 0.0
        || larger_fee_wei < 0.0
    {
        return Err(L1FeeOracleError::new(format!(
            "invalid l1 fee samples: small={smaller_fee_wei}, large={larger_fee_wei}"
        )));
    }

    let delta_fee_wei = larger_fee_wei - smaller_fee_wei;
    let delta_payload_bytes = (larger_payload_bytes - smaller_payload_bytes) as f64;
    let fee_per_byte_wei = delta_fee_wei / delta_payload_bytes;
    if !fee_per_byte_wei.is_finite() || fee_per_byte_wei <= 0.0 {
        return Err(L1FeeOracleError::new(format!(
            "invalid l1 fee-per-byte slope: small={smaller_fee_wei}, large={larger_fee_wei}, bytes={smaller_payload_bytes}->{larger_payload_bytes}"
        )));
    }
    Ok(fee_per_byte_wei)
}

async fn fetch_optimism_l1_fee_wei_for_payload_len(
    client: &reqwest::Client,
    rpc_url: &str,
    payload_len: usize,
) -> Result<f64, L1FeeOracleError> {
    if payload_len == 0 {
        return Err(L1FeeOracleError::new("l1 fee payload length must be > 0"));
    }

    let call_data = getL1FeeCall {
        payload: Bytes::from(make_l1_fee_probe_bytes(payload_len)),
    }
    .abi_encode();
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [
            {
                "to": OP_GAS_PRICE_ORACLE_ADDR,
                "data": format!("0x{}", hex::encode(call_data)),
            },
            "latest"
        ]
    });

    let response = client
        .post(rpc_url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| L1FeeOracleError::new(format!("eth_call request failed: {e}")))?
        .error_for_status()
        .map_err(|e| L1FeeOracleError::new(format!("eth_call bad http status: {e}")))?;

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| L1FeeOracleError::new(format!("failed to decode eth_call body: {e}")))?;
    if let Some(err) = body.get("error") {
        return Err(L1FeeOracleError::new(format!(
            "eth_call returned error: {err}"
        )));
    }
    parse_eth_call_fee_wei(body, payload_len)
}

fn parse_eth_call_fee_wei(
    body: serde_json::Value,
    payload_len: usize,
) -> Result<f64, L1FeeOracleError> {
    let result_hex = body
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or_else(|| L1FeeOracleError::new("missing eth_call result"))?;
    let raw_hex = result_hex.trim_start_matches("0x");
    let padded_hex = if raw_hex.len() % 2 == 1 {
        format!("0{raw_hex}")
    } else {
        raw_hex.to_string()
    };
    let raw_bytes = hex::decode(padded_hex)
        .map_err(|e| L1FeeOracleError::new(format!("invalid eth_call result hex: {e}")))?;
    let wei = U256::try_from_be_slice(&raw_bytes).ok_or_else(|| {
        L1FeeOracleError::new(format!(
            "eth_call result does not fit into U256 ({} bytes)",
            raw_bytes.len()
        ))
    })?;
    let fee_wei = f64::from(wei);
    if !fee_wei.is_finite() || fee_wei <= 0.0 {
        return Err(L1FeeOracleError::new(format!(
            "invalid l1 fee for payload length {payload_len}: {fee_wei}"
        )));
    }
    Ok(fee_wei)
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use super::*;
    use crate::execution::{ExecutionGroupPlan, ExecutionLegPlan, LegKind};
    use crate::portfolio::Action;

    #[test]
    fn probe_bytes_are_non_uniform() {
        // Regression guard: the L1 fee probe must NOT be a single repeated byte.
        // Under Fjord Brotli, ANY uniform vec![b; N] compresses to ~11 bytes regardless
        // of which byte `b` is. This causes the marginal fee slope to collapse to ~0.
        // The probe must have enough byte diversity to simulate real calldata entropy.
        let small = make_l1_fee_probe_bytes(L1_FEE_SLOPE_SAMPLE_SMALL_BYTES);
        let large = make_l1_fee_probe_bytes(L1_FEE_SLOPE_SAMPLE_LARGE_BYTES);
        // At least 4 distinct byte values
        let distinct_small: std::collections::HashSet<u8> = small.iter().copied().collect();
        let distinct_large: std::collections::HashSet<u8> = large.iter().copied().collect();
        assert!(
            distinct_small.len() >= 4,
            "probe bytes have only {} distinct values (need ≥4 for realistic entropy)",
            distinct_small.len()
        );
        assert_eq!(small.len(), L1_FEE_SLOPE_SAMPLE_SMALL_BYTES);
        assert_eq!(large.len(), L1_FEE_SLOPE_SAMPLE_LARGE_BYTES);
        let _ = distinct_large; // large inherits same generator
    }

    fn with_cache_lock<T>(f: impl FnOnce() -> T) -> T {
        static CACHE_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = CACHE_TEST_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().expect("cache test lock poisoned");
        f()
    }

    fn clear_l1_fee_cache_for_test() {
        let mut guard = l1_fee_cache()
            .write()
            .expect("failed to acquire l1 fee cache write lock");
        guard.clear();
    }

    #[test]
    fn estimates_match_spec_constants() {
        let gas = GasAssumptions::default();
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::DirectBuy, 0, 0),
            57_542
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::DirectSell, 0, 0),
            38_099
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::DirectMerge, 0, 0),
            21_502
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::MintSell, 0, 3),
            169_730
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::BuyMerge, 2, 0),
            96_710
        );
    }

    #[test]
    fn calldata_estimate_scales_with_leg_count() {
        let one_leg = estimate_group_calldata_bytes(GroupKind::MintSell, 0, 1);
        let five_legs = estimate_group_calldata_bytes(GroupKind::MintSell, 0, 5);
        assert!(five_legs > one_leg, "more legs should imply more calldata");
    }

    #[test]
    fn converts_units_to_susd() {
        let susd = estimate_l2_gas_susd(57_542, 5e-10, 3000.0);
        assert!(
            (susd - 0.086_313).abs() < 1e-12,
            "unexpected gas conversion"
        );
    }

    #[test]
    fn l1_fee_scales_with_estimated_calldata() {
        let assumptions = GasAssumptions {
            l1_data_fee_floor_susd: 0.0,
            l1_fee_per_byte_wei: 1.0e12,
            ..GasAssumptions::default()
        };
        let direct =
            estimate_group_l1_data_fee_susd(&assumptions, GroupKind::DirectBuy, 1, 0, 3000.0);
        let routed =
            estimate_group_l1_data_fee_susd(&assumptions, GroupKind::MintSell, 0, 5, 3000.0);
        assert!(routed > direct, "larger calldata should cost more l1 fee");
    }

    #[test]
    fn marginal_l1_fee_per_byte_uses_two_point_slope() {
        let fee_per_byte =
            marginal_fee_per_byte_wei(1000.0, 2200.0, 100, 200).expect("expected positive slope");
        assert!((fee_per_byte - 12.0).abs() < 1e-12);
    }

    #[test]
    fn marginal_l1_fee_per_byte_rejects_non_positive_slope() {
        let err = marginal_fee_per_byte_wei(2000.0, 2000.0, 256, 512)
            .expect_err("expected non-positive slope to fail");
        assert!(
            err.to_string().contains("invalid l1 fee-per-byte slope"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn total_gas_requires_l1_fee_per_byte_estimate() {
        let gas = GasAssumptions::default();
        let total =
            estimate_total_gas_susd(&gas, GroupKind::DirectBuy, 1, 0, 220_000, 5e-10, 3000.0);
        assert!(
            !total.is_finite(),
            "missing l1 fee-per-byte estimate should fail closed"
        );
    }

    #[test]
    fn total_gas_includes_l1_floor_when_fee_per_byte_is_set() {
        let gas = GasAssumptions {
            l1_data_fee_floor_susd: 0.001,
            l1_fee_per_byte_wei: 1.0,
            ..GasAssumptions::default()
        };
        let total =
            estimate_total_gas_susd(&gas, GroupKind::DirectBuy, 1, 0, 57_542, 5e-10, 3000.0);
        assert!(
            (total - 0.087_313).abs() < 1e-12,
            "expected calibrated l2 + 0.001 l1 floor"
        );
    }

    #[test]
    fn effective_l1_fee_prefers_assumption_over_cache() {
        let assumptions = GasAssumptions {
            l1_fee_per_byte_wei: 111.0,
            ..GasAssumptions::default()
        };
        let effective = effective_l1_fee_per_byte_wei_with_cache(&assumptions, Some(222.0));
        assert_eq!(effective, Some(111.0));
    }

    #[test]
    fn effective_l1_fee_falls_back_to_cache_without_assumption() {
        let assumptions = GasAssumptions::default();
        let effective = effective_l1_fee_per_byte_wei_with_cache(&assumptions, Some(222.0));
        assert_eq!(effective, Some(222.0));
    }

    #[test]
    fn effective_l1_fee_falls_back_to_assumption_without_cache() {
        let assumptions = GasAssumptions {
            l1_fee_per_byte_wei: 111.0,
            ..GasAssumptions::default()
        };
        let effective = effective_l1_fee_per_byte_wei_with_cache(&assumptions, None);
        assert_eq!(effective, Some(111.0));
    }

    #[test]
    fn rejects_eth_call_result_larger_than_u256() {
        let body = serde_json::json!({
            "result": format!("0x{}", "11".repeat(33)),
        });
        let err = parse_eth_call_fee_wei(body, 256)
            .expect_err("oversized eth_call result should fail instead of panicking");
        assert!(
            err.to_string().contains("does not fit into U256"),
            "unexpected parse error: {err}"
        );
    }

    #[test]
    fn cached_fee_lookup_is_scoped_by_rpc_url() {
        with_cache_lock(|| {
            clear_l1_fee_cache_for_test();
            {
                let mut guard = l1_fee_cache()
                    .write()
                    .expect("failed to acquire l1 fee cache write lock");
                guard.insert(
                    "https://rpc-a.example".to_string(),
                    CachedL1FeePerByteWei {
                        fee_per_byte_wei: 111.0,
                        fetched_at: Instant::now(),
                    },
                );
            }

            assert_eq!(
                fresh_cached_l1_fee_per_byte_wei("https://rpc-a.example"),
                Some(111.0)
            );
            assert_eq!(
                fresh_cached_l1_fee_per_byte_wei("https://rpc-b.example"),
                None
            );
        });
    }

    #[test]
    fn cache_value_helper_fails_closed_when_multiple_endpoints_are_fresh() {
        with_cache_lock(|| {
            clear_l1_fee_cache_for_test();
            {
                let mut guard = l1_fee_cache()
                    .write()
                    .expect("failed to acquire l1 fee cache write lock");
                guard.insert(
                    "https://rpc-a.example".to_string(),
                    CachedL1FeePerByteWei {
                        fee_per_byte_wei: 111.0,
                        fetched_at: Instant::now(),
                    },
                );
                guard.insert(
                    "https://rpc-b.example".to_string(),
                    CachedL1FeePerByteWei {
                        fee_per_byte_wei: 222.0,
                        fetched_at: Instant::now(),
                    },
                );
            }

            assert_eq!(cached_optimism_l1_fee_per_byte_wei_value(), None);
        });
    }

    #[test]
    fn min_gas_susd_direct_buy_at_floor() {
        let gas = GasAssumptions {
            l1_data_fee_floor_susd: 0.001,
            l1_fee_per_byte_wei: 0.0, // unknown — fall back to floor
            ..GasAssumptions::default()
        };
        // L2: 57_542 gas × 5e-10 ETH/gas × 3000 $/ETH = 0.086313 sUSD
        // L1: floor = 0.001 sUSD
        // Total: 0.087313 sUSD
        let cost = estimate_min_gas_susd_for_group(&gas, GroupKind::DirectBuy, 0, 0, 5e-10, 3000.0);
        assert!(
            (cost - 0.087_313).abs() < 1e-9,
            "expected ~0.087313, got {cost}"
        );
    }

    #[test]
    fn min_gas_susd_is_finite_for_97_leg_mint_sell() {
        let gas = GasAssumptions {
            l1_data_fee_floor_susd: 0.001,
            l1_fee_per_byte_wei: 0.0,
            ..GasAssumptions::default()
        };
        let cost = estimate_min_gas_susd_for_group(&gas, GroupKind::MintSell, 0, 97, 5e-10, 3000.0);
        assert!(
            cost.is_finite(),
            "97-leg MintSell gas estimate must be finite"
        );
        assert!(cost > 0.0, "97-leg MintSell gas must be positive");
        assert!(
            cost > 1.0,
            "97-leg MintSell at 5e-10 gas price should be > $1: {cost}"
        );
    }

    #[test]
    fn exact_input_single_payload_without_selector_is_224_bytes() {
        // Verify that 7 EVM word fields = 7 × 32 = 224 bytes.
        // SWAP_BYTES = 224 excludes the 4-byte selector (which lives in BATCH_CALL_BASE_BYTES).
        use alloy::sol;
        use alloy::sol_types::SolCall;

        sol! {
            struct ExactInputSingleParams {
                address tokenIn;
                address tokenOut;
                uint24 fee;
                address recipient;
                uint256 amountIn;
                uint256 amountOutMinimum;
                uint160 sqrtPriceLimitX96;
            }
            function exactInputSingle(ExactInputSingleParams params) external returns (uint256);
        }

        let call = exactInputSingleCall {
            params: ExactInputSingleParams {
                tokenIn: alloy::primitives::Address::ZERO,
                tokenOut: alloy::primitives::Address::ZERO,
                fee: alloy::primitives::Uint::from(100u32),
                recipient: alloy::primitives::Address::ZERO,
                amountIn: alloy::primitives::U256::ZERO,
                amountOutMinimum: alloy::primitives::U256::ZERO,
                sqrtPriceLimitX96: alloy::primitives::U160::ZERO,
            },
        };
        let encoded = call.abi_encode();
        // selector(4) + 7 × 32 = 228 total; payload without selector = 224
        assert_eq!(
            encoded.len() - 4,
            224,
            "exactInputSingle payload (no selector) must be 224 bytes; got {}",
            encoded.len() - 4
        );
    }

    #[test]
    fn split_position_full_call_is_100_bytes() {
        // Verify CTF splitPosition: selector(4) + 3 × address/uint256 words = 100 bytes.
        // Internal L2 calls (splitPosition is called by BatchSwapRouter internally) do NOT
        // contribute to L1 calldata — only the outer EOA tx bytes matter for L1 data fee.
        use alloy::sol;
        use alloy::sol_types::SolCall;

        sol! {
            function splitPosition(address collateralToken, address conditionId, uint256 amount) external;
        }

        let call = splitPositionCall {
            collateralToken: alloy::primitives::Address::ZERO,
            conditionId: alloy::primitives::Address::ZERO,
            amount: alloy::primitives::U256::ZERO,
        };
        let encoded = call.abi_encode();
        assert_eq!(
            encoded.len(),
            100,
            "splitPosition ABI encoding must be 100 bytes"
        );
    }

    #[test]
    fn direct_buy_calldata_estimate_is_434_bytes() {
        // DirectBuy: TX_ENVELOPE(110) + BATCH_CALL_BASE(100) + SWAP_BYTES(224) = 434
        let estimate = estimate_group_calldata_bytes(GroupKind::DirectBuy, 0, 0);
        assert_eq!(
            estimate, 434,
            "DirectBuy calldata estimate should be 434 bytes"
        );
    }

    #[test]
    fn unsigned_batch_execute_tx_bytes_are_buildable() {
        let call = ITradeExecutor::Call {
            to: Address::repeat_byte(0x11),
            data: Bytes::from(vec![0xde, 0xad, 0xbe, 0xef]),
        };
        let bytes = build_unsigned_batch_execute_tx_bytes(
            Address::repeat_byte(0x22),
            &[call],
            LiveOptimismFeeInputs {
                chain_id: 10,
                sender_nonce: 7,
                gas_price_wei: 1_002_325,
            },
            220_000,
        )
        .expect("unsigned tx bytes should build");
        assert!(
            !bytes.is_empty(),
            "unsigned batchExecute tx bytes should not be empty"
        );
    }

    #[test]
    fn unsigned_group_plan_tx_bytes_are_buildable_from_execution_calls() {
        let market = crate::markets::MARKETS_L1
            .iter()
            .find(|market| market.pool.is_some())
            .expect("pooled market required");
        let actions = vec![Action::Buy {
            market_name: market.name,
            amount: 1.0,
            cost: 0.55,
        }];
        let plan = ExecutionGroupPlan {
            kind: GroupKind::DirectBuy,
            action_indices: vec![0],
            profitability_step_index: 0,
            step_subgroup_index: 0,
            step_subgroup_count: 1,
            legs: vec![ExecutionLegPlan {
                action_index: 0,
                market_name: Some(market.name),
                kind: LegKind::Buy,
                planned_quote_susd: 0.55,
                conservative_quote_susd: 0.55,
                adverse_notional_susd: 0.55,
                allocated_slippage_susd: 0.01,
                max_cost_susd: Some(0.56),
                min_proceeds_susd: None,
                sqrt_price_limit_x96: Some(alloy::primitives::U160::from(1u128 << 96)),
            }],
            planned_at_block: Some(100),
            edge_plan_susd: 0.15,
            l2_gas_units: 220_000,
            gas_l2_susd: 0.0,
            gas_total_susd: 0.0,
            profit_buffer_susd: 0.01,
            slippage_budget_susd: 0.01,
            guaranteed_profit_floor_susd: 0.01,
        };
        let bytes = build_unsigned_group_plan_batch_execute_tx_bytes(
            Address::repeat_byte(0x22),
            &actions,
            &plan,
            LiveOptimismFeeInputs {
                chain_id: 10,
                sender_nonce: 0,
                gas_price_wei: 1_002_325,
            },
        )
        .expect("group-plan batchExecute tx bytes should build");
        assert!(
            !bytes.is_empty(),
            "unsigned group-plan tx bytes should not be empty"
        );
    }

    #[tokio::test]
    #[ignore = "live OP RPC integration helper; run explicitly"]
    async fn live_op_get_l1_fee_succeeds_for_unsigned_batch_execute_tx() {
        let rpc_url =
            std::env::var("RPC").unwrap_or_else(|_| "https://optimism.drpc.org".to_string());
        let fee_inputs = fetch_live_optimism_fee_inputs(&rpc_url, Address::ZERO)
            .await
            .expect("live fee inputs should resolve");
        let tx_data = build_unsigned_batch_execute_tx_bytes(
            Address::repeat_byte(0x22),
            &[ITradeExecutor::Call {
                to: Address::repeat_byte(0x11),
                data: Bytes::from(vec![0xde, 0xad, 0xbe, 0xef]),
            }],
            fee_inputs,
            220_000,
        )
        .expect("unsigned tx bytes should build");
        let client = build_rpc_client().expect("rpc client should build");
        let l1_fee_wei = fetch_optimism_l1_fee_wei_for_tx_data(&client, &rpc_url, tx_data)
            .await
            .expect("getL1Fee should succeed against live OP RPC");
        assert!(l1_fee_wei > U256::ZERO, "l1 fee should be positive");
    }
}
