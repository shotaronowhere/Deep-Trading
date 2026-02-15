use std::collections::HashMap;
use std::fmt;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use alloy::hex;
use alloy::primitives::{Bytes, U256};
use alloy::sol;
use alloy::sol_types::SolCall;
use tokio::sync::Mutex;

use super::GroupKind;

const OP_GAS_PRICE_ORACLE_ADDR: &str = "0x420000000000000000000000000000000000000F";
const WEI_PER_ETH: f64 = 1e18;
const L1_FEE_CACHE_TTL: Duration = Duration::from_secs(60);
const L1_FEE_RPC_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const L1_FEE_SLOPE_SAMPLE_SMALL_BYTES: usize = 256;
const L1_FEE_SLOPE_SAMPLE_LARGE_BYTES: usize = 512;
const L1_FEE_SAMPLE_FILL_BYTE: u8 = 1;
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
            direct_buy_l2_units: 220_000,
            direct_sell_l2_units: 200_000,
            direct_merge_l2_units: 150_000,
            mint_sell_base_l2_units: 550_000,
            mint_sell_per_sell_leg_l2_units: 170_000,
            buy_merge_base_l2_units: 500_000,
            buy_merge_per_buy_leg_l2_units: 180_000,
            l1_data_fee_floor_susd: 0.10,
            l1_fee_per_byte_wei: 0.0,
        }
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

pub fn estimate_group_calldata_bytes(kind: GroupKind, buy_legs: usize, sell_legs: usize) -> u64 {
    const TX_ENVELOPE_BYTES: u64 = 110;
    const BATCH_CALL_BASE_BYTES: u64 = 100;
    const SWAP_BYTES: u64 = 224;
    const FLASH_ROUTE_EXTRA_BYTES: u64 = 160;
    const DIRECT_MERGE_CALL_BYTES: u64 = 220;

    match kind {
        GroupKind::DirectBuy | GroupKind::DirectSell => {
            TX_ENVELOPE_BYTES + BATCH_CALL_BASE_BYTES + SWAP_BYTES
        }
        GroupKind::MintSell => {
            let legs = sell_legs.max(1) as u64;
            TX_ENVELOPE_BYTES
                + FLASH_ROUTE_EXTRA_BYTES
                + BATCH_CALL_BASE_BYTES
                + SWAP_BYTES.saturating_mul(legs)
        }
        GroupKind::BuyMerge => {
            let legs = buy_legs.max(1) as u64;
            TX_ENVELOPE_BYTES
                + FLASH_ROUTE_EXTRA_BYTES
                + BATCH_CALL_BASE_BYTES
                + SWAP_BYTES.saturating_mul(legs)
        }
        GroupKind::DirectMerge => TX_ENVELOPE_BYTES + DIRECT_MERGE_CALL_BYTES,
    }
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
        payload: Bytes::from(vec![L1_FEE_SAMPLE_FILL_BYTE; payload_len]),
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
            220_000
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::DirectSell, 0, 0),
            200_000
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::DirectMerge, 0, 0),
            150_000
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::MintSell, 0, 3),
            1_060_000
        );
        assert_eq!(
            estimate_group_l2_gas_units(&gas, GroupKind::BuyMerge, 2, 0),
            860_000
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
        let susd = estimate_l2_gas_susd(220_000, 5e-10, 3000.0);
        assert!((susd - 0.33).abs() < 1e-12, "unexpected gas conversion");
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
            l1_data_fee_floor_susd: 0.10,
            l1_fee_per_byte_wei: 1.0,
            ..GasAssumptions::default()
        };
        let total =
            estimate_total_gas_susd(&gas, GroupKind::DirectBuy, 1, 0, 220_000, 5e-10, 3000.0);
        assert!(
            (total - 0.43).abs() < 1e-12,
            "expected 0.33 l2 + 0.10 l1 floor"
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
}
