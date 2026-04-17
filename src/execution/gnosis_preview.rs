//! Algebra solver preview via SolverQuoter on Gnosis chain.
//!
//! Mirrors the pattern in `onchain_preview.rs` but uses `IRebalancerAlgebra`
//! (no `fee` field in RebalanceParams) and deploys `RebalancerAlgebra.sol`.

use std::collections::HashMap;
use std::fmt;
use std::fs;

use alloy::dyn_abi::DynSolValue;
use alloy::hex;
use alloy::network::{Ethereum, ReceiptResponse};
use alloy::primitives::{Address, Bytes, U160, U256, address};
use alloy::providers::Provider;
use alloy::rpc::types::TransactionRequest;
use alloy::sol;
use alloy::sol_types::SolCall;
use serde::{Deserialize, Serialize};

use crate::gnosis::{GNOSIS_ROUTER, SDAI, SWAPR_ROUTER};
use crate::pools::prediction_to_sqrt_price_x96;

use super::ITradeExecutor;

// ── Sol! bindings (Algebra variant — no fee in RebalanceParams) ──

sol! {
    #[sol(rpc)]
    interface IRebalancerAlgebra {
        struct RebalanceParams {
            address[] tokens;
            address[] pools;
            bool[] isToken1;
            uint256[] balances;
            uint256 collateralAmount;
            uint160[] sqrtPredX96;
            address collateral;
        }

        function rebalanceExact(
            RebalanceParams calldata params,
            uint256 maxBisectionIterations,
            uint256 maxTickCrossingsPerPool
        ) external returns (uint256 totalProceeds, uint256 totalSpent);

        function rebalanceAndArbExact(
            RebalanceParams calldata params,
            address market,
            uint256 maxArbRounds,
            uint256 maxRecycleRounds,
            uint256 maxBisectionIterations,
            uint256 maxTickCrossingsPerPool
        ) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);
    }
}

sol! {
    #[sol(rpc)]
    interface ISolverQuoter {
        function quote(
            address solver,
            bytes calldata solverCall,
            address[] calldata tokens,
            address collateral
        ) external;
    }
}

sol! {
    #[sol(rpc)]
    interface IERC20Transfer {
        function transfer(address to, uint256 amount) external returns (bool);
    }
}

// ── Constants ──

const QUOTER_ADDRESS: Address = address!("00000000000000000000000000000000DeAdBeef");
const REBALANCER_ALGEBRA_ARTIFACT: &str = "out/RebalancerAlgebra.sol/RebalancerAlgebra.json";
const SOLVER_QUOTER_ARTIFACT: &str = "out/SolverQuoter.sol/SolverQuoter.json";
const GNOSIS_SOLVER_CACHE: &str = "cache/gnosis_solver_preview.json";

const MAX_BISECTION: u64 = 64;
const MAX_TICK_CROSSINGS: u64 = 20;
const MAX_ARB_ROUNDS: u64 = 3;
const MAX_RECYCLE_ROUNDS: u64 = 2;

// ── Types ──

#[derive(Debug, Clone)]
pub struct SolverPreviewResult {
    pub name: &'static str,
    pub success: bool,
    pub post_cash_wei: U256,
    pub post_balances_wei: Vec<U256>,
    pub raw_ev: f64,
}

/// A pool that was verified to have on-chain code.
#[derive(Debug, Clone)]
pub struct ActivePool {
    pub token: Address,
    pub pool: Address,
    pub is_token1: bool,
    pub prediction: f64,
    pub collateral: Address, // underlying_token paired in the pool
    pub market_id: Address,  // conditional market for splitPosition arb
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GnosisSolverCache {
    chain_id: u64,
    rebalancer_algebra: String,
}

#[derive(Debug)]
pub enum PreviewError {
    Provider(String),
    Build(String),
    Cache(String),
    Deploy(String),
}

impl fmt::Display for PreviewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(msg) => write!(f, "provider: {msg}"),
            Self::Build(msg) => write!(f, "build: {msg}"),
            Self::Cache(msg) => write!(f, "cache: {msg}"),
            Self::Deploy(msg) => write!(f, "deploy: {msg}"),
        }
    }
}

impl std::error::Error for PreviewError {}

// ── Cache ──

fn load_cache() -> Option<GnosisSolverCache> {
    let raw = fs::read_to_string(GNOSIS_SOLVER_CACHE).ok()?;
    serde_json::from_str(&raw).ok()
}

fn save_cache(entry: &GnosisSolverCache) -> Result<(), PreviewError> {
    let dir = std::path::Path::new(GNOSIS_SOLVER_CACHE).parent().unwrap();
    fs::create_dir_all(dir).map_err(|e| PreviewError::Cache(e.to_string()))?;
    let json =
        serde_json::to_string_pretty(entry).map_err(|e| PreviewError::Cache(e.to_string()))?;
    fs::write(GNOSIS_SOLVER_CACHE, json).map_err(|e| PreviewError::Cache(e.to_string()))
}

// ── Deploy helpers ──

fn build_init_code(artifact_path: &str, constructor_args: &[u8]) -> Result<Vec<u8>, PreviewError> {
    let raw = fs::read_to_string(artifact_path)
        .map_err(|e| PreviewError::Build(format!("{artifact_path}: {e}")))?;
    let json: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| PreviewError::Build(e.to_string()))?;
    let bytecode_hex = json
        .get("bytecode")
        .and_then(|b| b.get("object"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PreviewError::Build(format!("missing bytecode in {artifact_path}")))?;
    let trimmed = bytecode_hex.strip_prefix("0x").unwrap_or(bytecode_hex);
    let mut code = hex::decode(trimmed).map_err(|e| PreviewError::Build(format!("hex: {e}")))?;
    code.extend_from_slice(constructor_args);
    Ok(code)
}

fn get_deployed_bytecode(artifact_path: &str) -> Result<Vec<u8>, PreviewError> {
    let raw = fs::read_to_string(artifact_path)
        .map_err(|e| PreviewError::Build(format!("{artifact_path}: {e}")))?;
    let json: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| PreviewError::Build(e.to_string()))?;
    let bytecode_hex = json
        .get("deployedBytecode")
        .and_then(|b| b.get("object"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PreviewError::Build(format!("missing deployedBytecode in {artifact_path}"))
        })?;
    let trimmed = bytecode_hex.strip_prefix("0x").unwrap_or(bytecode_hex);
    hex::decode(trimmed).map_err(|e| PreviewError::Build(format!("hex: {e}")))
}

async fn deploy_contract<P: Provider<Ethereum> + Clone>(
    provider: P,
    owner: Address,
    init_code: Vec<u8>,
) -> Result<Address, PreviewError> {
    let tx = TransactionRequest::default()
        .from(owner)
        .input(Bytes::from(init_code).into());
    let receipt = provider
        .send_transaction(tx)
        .await
        .map_err(|e| PreviewError::Deploy(e.to_string()))?
        .get_receipt()
        .await
        .map_err(|e| PreviewError::Deploy(e.to_string()))?;
    if !receipt.status() {
        return Err(PreviewError::Deploy(format!(
            "tx {} reverted",
            receipt.transaction_hash()
        )));
    }
    receipt
        .contract_address()
        .ok_or(PreviewError::Deploy("missing contract address".into()))
}

/// Deploy (or reuse cached) RebalancerAlgebra on Gnosis.
pub async fn resolve_algebra_solver<P: Provider<Ethereum> + Clone>(
    provider: P,
    owner: Address,
) -> Result<Address, PreviewError> {
    let chain_id = provider
        .get_chain_id()
        .await
        .map_err(|e| PreviewError::Provider(e.to_string()))?;

    if let Some(cached) = load_cache() {
        if cached.chain_id == chain_id {
            let addr = cached
                .rebalancer_algebra
                .parse::<Address>()
                .map_err(|e| PreviewError::Cache(e.to_string()))?;
            let code = provider
                .get_code_at(addr)
                .await
                .map_err(|e| PreviewError::Provider(e.to_string()))?;
            if !code.is_empty() {
                tracing::info!(%addr, "reused cached RebalancerAlgebra");
                return Ok(addr);
            }
            tracing::info!("cached Algebra solver has no code; redeploying");
        }
    }

    // Constructor: (address _router, address _ctfRouter)
    let constructor_args = DynSolValue::Tuple(vec![
        DynSolValue::Address(SWAPR_ROUTER),
        DynSolValue::Address(GNOSIS_ROUTER),
    ])
    .abi_encode_params();

    let init_code = build_init_code(REBALANCER_ALGEBRA_ARTIFACT, &constructor_args)?;
    let addr = deploy_contract(provider, owner, init_code).await?;
    tracing::info!(%addr, "deployed RebalancerAlgebra");

    save_cache(&GnosisSolverCache {
        chain_id,
        rebalancer_algebra: format!("{addr}"),
    })?;

    Ok(addr)
}

/// Read cached solver address without deploying.
pub fn resolve_algebra_solver_readonly() -> Option<Address> {
    let cached = load_cache()?;
    cached.rebalancer_algebra.parse::<Address>().ok()
}

// ── RebalanceParams builder ──

pub fn build_algebra_params(
    active_pools: &[ActivePool],
    token_balances: &HashMap<Address, U256>,
    collateral_balance: U256,
) -> IRebalancerAlgebra::RebalanceParams {
    let mut tokens = Vec::with_capacity(active_pools.len());
    let mut pools = Vec::with_capacity(active_pools.len());
    let mut is_token1 = Vec::with_capacity(active_pools.len());
    let mut balances = Vec::with_capacity(active_pools.len());
    let mut sqrt_pred_x96 = Vec::with_capacity(active_pools.len());

    // All pools in a single call must share the same collateral
    let collateral = active_pools.first().map(|ap| ap.collateral).unwrap_or(SDAI);

    for ap in active_pools {
        let sqrt_pred = U160::from(
            prediction_to_sqrt_price_x96(ap.prediction, ap.is_token1).unwrap_or_else(|| {
                panic!("invalid prediction {} for pool {}", ap.prediction, ap.pool)
            }),
        );
        let held = token_balances.get(&ap.token).copied().unwrap_or(U256::ZERO);
        tokens.push(ap.token);
        pools.push(ap.pool);
        is_token1.push(ap.is_token1);
        balances.push(held);
        sqrt_pred_x96.push(sqrt_pred);
    }

    IRebalancerAlgebra::RebalanceParams {
        tokens,
        pools,
        isToken1: is_token1,
        balances,
        collateralAmount: collateral_balance,
        sqrtPredX96: sqrt_pred_x96,
        collateral,
    }
}

// ── Quoter call builder ──

fn build_batch_execute_calls(
    quoter: Address,
    solver: Address,
    solver_calldata: Bytes,
    params: &IRebalancerAlgebra::RebalanceParams,
) -> Vec<ITradeExecutor::Call> {
    let mut calls: Vec<ITradeExecutor::Call> = Vec::new();

    for (i, token) in params.tokens.iter().enumerate() {
        if params.balances[i].is_zero() {
            continue;
        }
        let data = IERC20Transfer::transferCall {
            to: quoter,
            amount: params.balances[i],
        }
        .abi_encode();
        calls.push(ITradeExecutor::Call {
            to: *token,
            data: Bytes::from(data),
        });
    }

    if !params.collateralAmount.is_zero() {
        let data = IERC20Transfer::transferCall {
            to: quoter,
            amount: params.collateralAmount,
        }
        .abi_encode();
        calls.push(ITradeExecutor::Call {
            to: params.collateral,
            data: Bytes::from(data),
        });
    }

    let quote_data = ISolverQuoter::quoteCall {
        solver,
        solverCall: solver_calldata,
        tokens: params.tokens.clone(),
        collateral: params.collateral,
    }
    .abi_encode();
    calls.push(ITradeExecutor::Call {
        to: quoter,
        data: Bytes::from(quote_data),
    });

    calls
}

fn parse_quoter_revert(revert_data: &[u8]) -> Option<(bool, U256, Vec<U256>)> {
    let decoded = alloy::dyn_abi::DynSolType::Tuple(vec![
        alloy::dyn_abi::DynSolType::Bool,
        alloy::dyn_abi::DynSolType::Bytes,
        alloy::dyn_abi::DynSolType::Uint(256),
        alloy::dyn_abi::DynSolType::Array(Box::new(alloy::dyn_abi::DynSolType::Uint(256))),
    ])
    .abi_decode(revert_data)
    .ok()?;

    let tuple = match decoded {
        DynSolValue::Tuple(t) => t,
        _ => return None,
    };
    if tuple.len() != 4 {
        return None;
    }

    let success = tuple[0].as_bool()?;
    let post_cash = tuple[2].as_uint()?.0;
    let post_balances = match &tuple[3] {
        DynSolValue::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_uint().map(|(val, _)| val))
            .collect(),
        _ => return None,
    };
    Some((success, post_cash, post_balances))
}

fn extract_revert_data(
    err: &alloy::transports::RpcError<alloy::transports::TransportErrorKind>,
) -> Option<Vec<u8>> {
    let err_str = format!("{err:?}");
    if let Some(pos) = err_str.find("0x") {
        let hex_part = &err_str[pos..];
        let end = hex_part
            .find(|c: char| !c.is_ascii_hexdigit() && c != 'x')
            .unwrap_or(hex_part.len());
        let trimmed = hex_part[..end]
            .strip_prefix("0x")
            .unwrap_or(&hex_part[..end]);
        return alloy::hex::decode(trimmed).ok();
    }
    None
}

// ── Compute EV ──

fn u256_to_f64(v: U256) -> f64 {
    v.to_string().parse::<f64>().unwrap_or(0.0) / 1e18
}

fn compute_raw_ev(
    post_cash_wei: U256,
    post_balances_wei: &[U256],
    active_pools: &[ActivePool],
) -> f64 {
    let cash = u256_to_f64(post_cash_wei);
    let holdings_ev: f64 = post_balances_wei
        .iter()
        .zip(active_pools.iter())
        .map(|(balance, ap)| u256_to_f64(*balance) * ap.prediction)
        .sum();
    cash + holdings_ev
}

// ── Public preview API ──

/// Run all Algebra solver variants via static call. Returns results sorted by raw EV descending.
/// `arb_market` is the conditional market for splitPosition (movie's market_id).
pub async fn preview_algebra_solvers<P: Provider<Ethereum> + Clone>(
    provider: P,
    executor: Address,
    owner: Address,
    rebalancer: Address,
    params: &IRebalancerAlgebra::RebalanceParams,
    active_pools: &[ActivePool],
    arb_market: Address,
) -> Vec<SolverPreviewResult> {
    let quoter_bytecode = match get_deployed_bytecode(SOLVER_QUOTER_ARTIFACT) {
        Ok(code) => code,
        Err(err) => {
            tracing::warn!(error = %err, "failed to load SolverQuoter bytecode");
            return Vec::new();
        }
    };

    let solvers: Vec<(&str, Bytes)> = vec![
        (
            "rebalanceExact",
            Bytes::from(
                IRebalancerAlgebra::rebalanceExactCall {
                    params: params.clone(),
                    maxBisectionIterations: U256::from(MAX_BISECTION),
                    maxTickCrossingsPerPool: U256::from(MAX_TICK_CROSSINGS),
                }
                .abi_encode(),
            ),
        ),
        (
            "rebalanceAndArbExact",
            Bytes::from(
                IRebalancerAlgebra::rebalanceAndArbExactCall {
                    params: params.clone(),
                    market: arb_market,
                    maxArbRounds: U256::from(MAX_ARB_ROUNDS),
                    maxRecycleRounds: U256::from(MAX_RECYCLE_ROUNDS),
                    maxBisectionIterations: U256::from(MAX_BISECTION),
                    maxTickCrossingsPerPool: U256::from(MAX_TICK_CROSSINGS),
                }
                .abi_encode(),
            ),
        ),
    ];

    let mut results = Vec::new();

    for (name, solver_calldata) in solvers {
        let calls = build_batch_execute_calls(QUOTER_ADDRESS, rebalancer, solver_calldata, params);
        let batch_calldata = ITradeExecutor::batchExecuteCall { calls }.abi_encode();

        let tx = TransactionRequest::default()
            .from(owner)
            .to(executor)
            .input(Bytes::from(batch_calldata).into());

        let state_override = alloy::rpc::types::state::AccountOverride {
            code: Some(Bytes::from(quoter_bytecode.clone())),
            ..Default::default()
        };
        let mut overrides = HashMap::new();
        overrides.insert(QUOTER_ADDRESS, state_override);

        let call_result = provider
            .call(tx)
            .overrides(alloy::rpc::types::state::StateOverride::from_iter(
                overrides,
            ))
            .await;

        match call_result {
            Ok(_) => {
                tracing::warn!(solver = name, "unexpected success (expected revert)");
            }
            Err(err) => match extract_revert_data(&err).and_then(|d| parse_quoter_revert(&d)) {
                Some((success, post_cash, post_balances)) => {
                    let raw_ev = compute_raw_ev(post_cash, &post_balances, active_pools);
                    results.push(SolverPreviewResult {
                        name,
                        success,
                        post_cash_wei: post_cash,
                        post_balances_wei: post_balances,
                        raw_ev,
                    });
                }
                None => {
                    tracing::warn!(solver = name, "failed to parse quoter revert");
                }
            },
        }
    }

    results.sort_by(|a, b| {
        b.raw_ev
            .partial_cmp(&a.raw_ev)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Print solver comparison table.
pub fn print_solver_comparison(results: &[SolverPreviewResult], pre_ev: f64, gas_cost_sdai: f64) {
    if results.is_empty() {
        return;
    }

    eprintln!();
    eprintln!("  === Gnosis Algebra Solver Comparison (static call) ===");
    eprintln!(
        "  {:<30} {:>12} {:>12} {:>12} {:>8}",
        "Solver", "Raw EV", "EV Delta", "Net EV \u{0394}", "Status"
    );
    eprintln!(
        "  {:-<30} {:-<12} {:-<12} {:-<12} {:-<8}",
        "", "", "", "", ""
    );
    for r in results {
        let delta = r.raw_ev - pre_ev;
        let net_delta = delta - gas_cost_sdai;
        let status = if r.success { "OK" } else { "REVERT" };
        eprintln!(
            "  {:<30} {:>12.4} {:>+12.4} {:>+12.4} {:>8}",
            r.name, r.raw_ev, delta, net_delta, status
        );
    }
    eprintln!("  Gas cost: {:.4} sDAI", gas_cost_sdai);
    if let Some(best) = results.iter().find(|r| r.success) {
        let net = best.raw_ev - pre_ev - gas_cost_sdai;
        eprintln!("  Best: {} (net EV \u{0394} {:+.4} sDAI)", best.name, net);
    }
    eprintln!("  ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnosis::MOVIES;
    use std::collections::HashMap;

    #[test]
    fn test_build_algebra_params_basic() {
        let balances: HashMap<Address, U256> = HashMap::new();
        let collateral_balance = U256::from(10_000_000_000_000_000_000u128); // 10e18

        // Use first movie's up/down pools (same underlying_token)
        let movie = &MOVIES[0];
        let active: Vec<ActivePool> = vec![
            ActivePool {
                token: movie.up_token,
                pool: movie.up_pool(),
                is_token1: movie.up_is_token1(),
                prediction: movie.up_prediction(),
                collateral: movie.underlying_token,
                market_id: movie.market_id,
            },
            ActivePool {
                token: movie.down_token,
                pool: movie.down_pool(),
                is_token1: movie.down_is_token1(),
                prediction: movie.down_prediction(),
                collateral: movie.underlying_token,
                market_id: movie.market_id,
            },
        ];

        let params = build_algebra_params(&active, &balances, collateral_balance);
        assert_eq!(params.tokens.len(), 2);
        assert_eq!(params.collateral, movie.underlying_token);
        assert!(params.collateralAmount > U256::ZERO);
    }
}
