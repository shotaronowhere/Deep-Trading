use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use alloy::dyn_abi::DynSolValue;
use alloy::hex;
use alloy::network::{Ethereum, ReceiptResponse};
use alloy::primitives::{Address, Bytes};
use alloy::providers::Provider;
use alloy::rpc::types::TransactionRequest;
use serde::{Deserialize, Serialize};

pub const DEFAULT_RPC_URL: &str = "https://optimism.drpc.org";
pub const DEFAULT_EXECUTE_SUBMIT: bool = false;
pub const DEFAULT_EXECUTION_MAX_STEPS: usize = 32;
pub const DEFAULT_EXECUTION_MAX_STALE_BLOCKS: u64 = 2;
pub const DEFAULT_EXECUTION_DEADLINE_SECS: u64 = 20;
pub const TRADE_EXECUTOR_CACHE_PATH: &str = "cache/trade_executor.json";
pub const TRADE_EXECUTOR_ARTIFACT_PATH: &str = "out/TradeExecutor.sol/TradeExecutor.json";

const OWNER_SELECTOR: [u8; 4] = [0x8d, 0xa5, 0xcb, 0x5b];

#[derive(Debug, Clone)]
pub struct ExecutionRuntimeConfig {
    pub private_key: String,
    pub rpc_url: String,
    pub execute_submit: bool,
    pub execution_max_steps: usize,
    pub execution_max_stale_blocks: u64,
    pub execution_deadline_secs: u64,
}

pub fn is_submission_deadline_exceeded(elapsed: Duration, deadline_secs: u64) -> bool {
    elapsed > Duration::from_secs(deadline_secs)
}

impl ExecutionRuntimeConfig {
    pub fn from_env() -> Result<Self, RuntimeError> {
        let private_key = std::env::var("PRIVATE_KEY")
            .map_err(|_| RuntimeError::MissingEnv("PRIVATE_KEY"))?
            .trim()
            .to_string();
        if private_key.is_empty() {
            return Err(RuntimeError::InvalidEnvValue {
                name: "PRIVATE_KEY",
                value: "<empty>".to_string(),
            });
        }

        let rpc_url = std::env::var("RPC")
            .unwrap_or_else(|_| DEFAULT_RPC_URL.to_string())
            .trim()
            .to_string();
        if rpc_url.is_empty() {
            return Err(RuntimeError::InvalidEnvValue {
                name: "RPC",
                value: "<empty>".to_string(),
            });
        }

        let execute_submit = parse_env_bool("EXECUTE_SUBMIT", DEFAULT_EXECUTE_SUBMIT)?;
        let execution_max_steps =
            parse_env_usize("EXECUTION_MAX_STEPS", DEFAULT_EXECUTION_MAX_STEPS)?;
        let execution_max_stale_blocks = parse_env_u64(
            "EXECUTION_MAX_STALE_BLOCKS",
            DEFAULT_EXECUTION_MAX_STALE_BLOCKS,
        )?;
        let execution_deadline_secs =
            parse_env_u64("EXECUTION_DEADLINE_SECS", DEFAULT_EXECUTION_DEADLINE_SECS)?;

        Ok(Self {
            private_key,
            rpc_url,
            execute_submit,
            execution_max_steps,
            execution_max_stale_blocks,
            execution_deadline_secs,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedTradeExecutor {
    pub chain_id: u64,
    pub owner: Address,
    pub executor: Address,
    pub reused_cache: bool,
}

#[derive(Debug)]
pub enum RuntimeError {
    MissingEnv(&'static str),
    InvalidEnvValue { name: &'static str, value: String },
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidArtifact(String),
    InvalidHex(String),
    Provider(String),
    DeploymentFailed(String),
    MissingContractAddress,
    InvalidOwnerReturn { executor: Address, len: usize },
    InvalidCachedAddress(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEnv(name) => write!(f, "required env var {name} is not set"),
            Self::InvalidEnvValue { name, value } => {
                write!(f, "invalid env var {name}={value}")
            }
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::InvalidArtifact(message) => write!(f, "invalid artifact: {message}"),
            Self::InvalidHex(message) => write!(f, "invalid hex: {message}"),
            Self::Provider(message) => write!(f, "provider error: {message}"),
            Self::DeploymentFailed(message) => write!(f, "deployment failed: {message}"),
            Self::MissingContractAddress => {
                write!(f, "deployment receipt missing contract address")
            }
            Self::InvalidOwnerReturn { executor, len } => write!(
                f,
                "invalid owner() return payload from executor {executor}: {len} bytes"
            ),
            Self::InvalidCachedAddress(raw) => {
                write!(f, "invalid cached address in trade executor cache: {raw}")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<std::io::Error> for RuntimeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for RuntimeError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TradeExecutorCacheFile {
    entries: Vec<TradeExecutorCacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TradeExecutorCacheEntry {
    chain_id: u64,
    owner: String,
    executor: String,
}

pub async fn resolve_trade_executor<P>(
    provider: P,
    owner: Address,
) -> Result<ResolvedTradeExecutor, RuntimeError>
where
    P: Provider<Ethereum> + Clone,
{
    let chain_id = provider
        .get_chain_id()
        .await
        .map_err(|err| RuntimeError::Provider(err.to_string()))?;

    if let Some(cached_executor) = load_cached_executor(chain_id, owner)? {
        match validate_cached_executor(provider.clone(), cached_executor, owner).await {
            Ok(true) => {
                return Ok(ResolvedTradeExecutor {
                    chain_id,
                    owner,
                    executor: cached_executor,
                    reused_cache: true,
                });
            }
            Ok(false) => {
                tracing::warn!(
                    chain_id,
                    owner = %owner,
                    executor = %cached_executor,
                    "cached trade executor invalid; redeploying"
                );
            }
            Err(err) => {
                tracing::warn!(
                    chain_id,
                    owner = %owner,
                    executor = %cached_executor,
                    error = %err,
                    "failed to validate cached trade executor; redeploying"
                );
            }
        }
    }

    let executor = deploy_trade_executor(provider.clone(), owner).await?;
    save_cached_executor(chain_id, owner, executor)?;
    Ok(ResolvedTradeExecutor {
        chain_id,
        owner,
        executor,
        reused_cache: false,
    })
}

pub async fn resolve_trade_executor_readonly<P>(
    provider: P,
    owner: Address,
) -> Result<Option<ResolvedTradeExecutor>, RuntimeError>
where
    P: Provider<Ethereum> + Clone,
{
    let chain_id = provider
        .get_chain_id()
        .await
        .map_err(|err| RuntimeError::Provider(err.to_string()))?;

    let Some(cached_executor) = load_cached_executor(chain_id, owner)? else {
        return Ok(None);
    };

    match validate_cached_executor(provider, cached_executor, owner).await {
        Ok(true) => Ok(Some(ResolvedTradeExecutor {
            chain_id,
            owner,
            executor: cached_executor,
            reused_cache: true,
        })),
        Ok(false) => Ok(None),
        Err(err) => {
            tracing::warn!(
                chain_id,
                owner = %owner,
                executor = %cached_executor,
                error = %err,
                "failed to validate cached trade executor in readonly mode"
            );
            Ok(None)
        }
    }
}

async fn validate_cached_executor<P>(
    provider: P,
    executor: Address,
    expected_owner: Address,
) -> Result<bool, RuntimeError>
where
    P: Provider<Ethereum> + Clone,
{
    let code = provider
        .get_code_at(executor)
        .await
        .map_err(|err| RuntimeError::Provider(err.to_string()))?;
    if code.is_empty() {
        return Ok(false);
    }

    let owner = read_executor_owner(provider, executor).await?;
    Ok(owner == expected_owner)
}

async fn deploy_trade_executor<P>(provider: P, owner: Address) -> Result<Address, RuntimeError>
where
    P: Provider<Ethereum> + Clone,
{
    let init_code = build_trade_executor_init_code(owner)?;
    let tx = TransactionRequest::default()
        .from(owner)
        .input(Bytes::from(init_code).into());

    let receipt = provider
        .send_transaction(tx)
        .await
        .map_err(|err| RuntimeError::Provider(err.to_string()))?
        .get_receipt()
        .await
        .map_err(|err| RuntimeError::Provider(err.to_string()))?;

    if !receipt.status() {
        return Err(RuntimeError::DeploymentFailed(format!(
            "tx {} reverted",
            receipt.transaction_hash()
        )));
    }

    receipt
        .contract_address()
        .ok_or(RuntimeError::MissingContractAddress)
}

async fn read_executor_owner<P>(provider: P, executor: Address) -> Result<Address, RuntimeError>
where
    P: Provider<Ethereum> + Clone,
{
    let call = TransactionRequest::default()
        .to(executor)
        .input(Bytes::copy_from_slice(&OWNER_SELECTOR).into());
    let out = provider
        .call(call)
        .await
        .map_err(|err| RuntimeError::Provider(err.to_string()))?;
    if out.len() < 32 {
        return Err(RuntimeError::InvalidOwnerReturn {
            executor,
            len: out.len(),
        });
    }
    Ok(Address::from_slice(&out[12..32]))
}

fn build_trade_executor_init_code(owner: Address) -> Result<Vec<u8>, RuntimeError> {
    let raw = fs::read_to_string(TRADE_EXECUTOR_ARTIFACT_PATH)?;
    let json: serde_json::Value = serde_json::from_str(&raw)?;
    let Some(bytecode_hex) = json
        .get("bytecode")
        .and_then(|bytecode| bytecode.get("object"))
        .and_then(serde_json::Value::as_str)
    else {
        return Err(RuntimeError::InvalidArtifact(
            "missing bytecode.object in TradeExecutor artifact".to_string(),
        ));
    };

    let mut init_code = decode_hex(bytecode_hex)?;
    if init_code.is_empty() {
        return Err(RuntimeError::InvalidArtifact(
            "TradeExecutor bytecode is empty".to_string(),
        ));
    }

    let constructor_args =
        DynSolValue::Tuple(vec![DynSolValue::Address(owner)]).abi_encode_params();
    init_code.extend_from_slice(&constructor_args);
    Ok(init_code)
}

fn decode_hex(raw: &str) -> Result<Vec<u8>, RuntimeError> {
    let trimmed = raw.strip_prefix("0x").unwrap_or(raw);
    hex::decode(trimmed).map_err(|err| RuntimeError::InvalidHex(err.to_string()))
}

fn parse_env_bool(name: &'static str, default: bool) -> Result<bool, RuntimeError> {
    let Some(raw) = std::env::var(name).ok() else {
        return Ok(default);
    };
    parse_bool_literal(raw.trim()).ok_or_else(|| RuntimeError::InvalidEnvValue { name, value: raw })
}

fn parse_env_u64(name: &'static str, default: u64) -> Result<u64, RuntimeError> {
    let Some(raw) = std::env::var(name).ok() else {
        return Ok(default);
    };
    raw.trim()
        .parse::<u64>()
        .map_err(|_| RuntimeError::InvalidEnvValue { name, value: raw })
}

fn parse_env_usize(name: &'static str, default: usize) -> Result<usize, RuntimeError> {
    let Some(raw) = std::env::var(name).ok() else {
        return Ok(default);
    };
    raw.trim()
        .parse::<usize>()
        .map_err(|_| RuntimeError::InvalidEnvValue { name, value: raw })
}

fn parse_bool_literal(raw: &str) -> Option<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn load_cached_executor(chain_id: u64, owner: Address) -> Result<Option<Address>, RuntimeError> {
    let path = Path::new(TRADE_EXECUTOR_CACHE_PATH);
    load_cached_executor_from_path(path, chain_id, owner)
}

fn load_cached_executor_from_path(
    path: &Path,
    chain_id: u64,
    owner: Address,
) -> Result<Option<Address>, RuntimeError> {
    if !path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(path)?;
    let cache: TradeExecutorCacheFile = serde_json::from_str(&raw)?;
    let owner_key = normalize_address(owner);

    let Some(entry) = cache
        .entries
        .iter()
        .find(|entry| entry.chain_id == chain_id && entry.owner.eq_ignore_ascii_case(&owner_key))
    else {
        return Ok(None);
    };

    Address::from_str(entry.executor.trim())
        .map(Some)
        .map_err(|_| RuntimeError::InvalidCachedAddress(entry.executor.clone()))
}

fn save_cached_executor(
    chain_id: u64,
    owner: Address,
    executor: Address,
) -> Result<(), RuntimeError> {
    let path = PathBuf::from(TRADE_EXECUTOR_CACHE_PATH);
    save_cached_executor_to_path(&path, chain_id, owner, executor)
}

fn save_cached_executor_to_path(
    path: &Path,
    chain_id: u64,
    owner: Address,
    executor: Address,
) -> Result<(), RuntimeError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut cache = if path.exists() {
        let raw = fs::read_to_string(path)?;
        serde_json::from_str::<TradeExecutorCacheFile>(&raw)?
    } else {
        TradeExecutorCacheFile::default()
    };

    let owner_key = normalize_address(owner);
    let executor_key = normalize_address(executor);
    if let Some(entry) = cache
        .entries
        .iter_mut()
        .find(|entry| entry.chain_id == chain_id && entry.owner.eq_ignore_ascii_case(&owner_key))
    {
        entry.executor = executor_key;
    } else {
        cache.entries.push(TradeExecutorCacheEntry {
            chain_id,
            owner: owner_key,
            executor: executor_key,
        });
    }

    let json = serde_json::to_string_pretty(&cache)?;
    fs::write(path, json)?;
    Ok(())
}

fn normalize_address(address: Address) -> String {
    address.to_string().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::time::Duration;

    use alloy::primitives::Address;

    use super::{
        is_submission_deadline_exceeded, load_cached_executor_from_path, parse_bool_literal,
        save_cached_executor_to_path,
    };

    #[test]
    fn parse_bool_literal_accepts_common_truthy_values() {
        assert_eq!(parse_bool_literal("1"), Some(true));
        assert_eq!(parse_bool_literal("true"), Some(true));
        assert_eq!(parse_bool_literal("YES"), Some(true));
        assert_eq!(parse_bool_literal("On"), Some(true));
    }

    #[test]
    fn parse_bool_literal_accepts_common_falsy_values() {
        assert_eq!(parse_bool_literal("0"), Some(false));
        assert_eq!(parse_bool_literal("false"), Some(false));
        assert_eq!(parse_bool_literal("NO"), Some(false));
        assert_eq!(parse_bool_literal("off"), Some(false));
    }

    #[test]
    fn parse_bool_literal_rejects_unknown_values() {
        assert_eq!(parse_bool_literal(""), None);
        assert_eq!(parse_bool_literal("maybe"), None);
    }

    #[test]
    fn deadline_gate_is_fail_closed_past_threshold() {
        assert!(!is_submission_deadline_exceeded(
            Duration::from_secs(20),
            20
        ));
        assert!(is_submission_deadline_exceeded(Duration::from_secs(21), 20));
    }

    #[test]
    fn cache_round_trip_is_keyed_by_chain_and_owner() {
        let temp_name = format!(
            "trade_executor_cache_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be valid")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(temp_name);
        let owner = Address::from_str("0x1111111111111111111111111111111111111111")
            .expect("owner should parse");
        let executor = Address::from_str("0x2222222222222222222222222222222222222222")
            .expect("executor should parse");

        save_cached_executor_to_path(&path, 10, owner, executor).expect("cache save should work");

        let hit = load_cached_executor_from_path(&path, 10, owner)
            .expect("cache load should work")
            .expect("expected cache hit");
        assert_eq!(hit, executor);

        let miss_chain = load_cached_executor_from_path(&path, 11, owner)
            .expect("cache load should work for mismatched chain");
        assert!(miss_chain.is_none(), "different chain id should miss cache");

        let other_owner = Address::from_str("0x3333333333333333333333333333333333333333")
            .expect("other owner should parse");
        let miss_owner = load_cached_executor_from_path(&path, 10, other_owner)
            .expect("cache load should work for mismatched owner");
        assert!(miss_owner.is_none(), "different owner should miss cache");

        let _ = std::fs::remove_file(path);
    }
}
