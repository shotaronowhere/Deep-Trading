//! Multi-solver static call preview via SolverQuoter revert-encoding pattern.
//!
//! Runs each on-chain solver variant as an `eth_call` and returns post-rebalance
//! portfolio balances. The SolverQuoter bytecode is injected via state override
//! (no deployment needed). The Rebalancer / RebalancerMixed must be deployed and
//! cached.

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

use crate::markets::MarketData;
use crate::pools::{Slot0Result, prediction_map, prediction_to_sqrt_price_x96, u256_to_f64};

use super::{
    BASE_COLLATERAL, CTF_ROUTER_ADDRESS, ITradeExecutor, MARKET_1_ADDRESS, SWAP_ROUTER_ADDRESS,
};

// ── Sol! bindings ──

sol! {
    #[sol(rpc)]
    interface IRebalancer {
        struct RebalanceParams {
            address[] tokens;
            address[] pools;
            bool[] isToken1;
            uint256[] balances;
            uint256 collateralAmount;
            uint160[] sqrtPredX96;
            address collateral;
            uint24 fee;
        }

        function rebalanceExact(
            RebalanceParams calldata params,
            uint256 maxBisectionIterations,
            uint256 maxTickCrossingsPerPool
        ) external returns (uint256 totalProceeds, uint256 totalSpent);

        function rebalanceAndArb(
            RebalanceParams calldata params,
            address market,
            uint256 maxArbRounds,
            uint256 maxRecycleRounds
        ) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);

        function rebalanceMixedConstantL(
            RebalanceParams calldata params,
            address market,
            uint256 maxOuterIterations,
            uint256 maxInnerIterations,
            uint256 maxMintCollateral
        ) external returns (uint256 totalProceeds, uint256 totalSpent);
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

/// Fixed address for the SolverQuoter injected via state override.
const QUOTER_ADDRESS: Address = address!("00000000000000000000000000000000DeAdBeef");

const REBALANCER_ARTIFACT_PATH: &str = "out/Rebalancer.sol/Rebalancer.json";
const REBALANCER_MIXED_ARTIFACT_PATH: &str = "out/RebalancerMixed.sol/RebalancerMixed.json";
const SOLVER_CACHE_PATH: &str = "cache/solver_preview.json";

const SOLVER_QUOTER_ARTIFACT_PATH: &str = "out/SolverQuoter.sol/SolverQuoter.json";

const FEE_TIER: u32 = 3000;
const MAX_BISECTION: u64 = 64;
const MAX_TICK_CROSSINGS: u64 = 20;
const MAX_ARB_ROUNDS: u64 = 3;
const MAX_RECYCLE_ROUNDS: u64 = 2;
const MAX_OUTER_ITER: u64 = 8;
const MAX_INNER_ITER: u64 = 64;

// ── Types ──

#[derive(Debug, Clone)]
pub struct SolverPreviewResult {
    pub name: &'static str,
    pub success: bool,
    pub post_cash_wei: U256,
    pub post_balances_wei: Vec<U256>,
    pub raw_ev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SolverCacheEntry {
    chain_id: u64,
    rebalancer: String,
    rebalancer_mixed: String,
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
            Self::Provider(msg) => write!(f, "provider error: {msg}"),
            Self::Build(msg) => write!(f, "build error: {msg}"),
            Self::Cache(msg) => write!(f, "cache error: {msg}"),
            Self::Deploy(msg) => write!(f, "deploy error: {msg}"),
        }
    }
}

impl std::error::Error for PreviewError {}

// ── Cache ──

fn load_solver_cache() -> Option<SolverCacheEntry> {
    let raw = fs::read_to_string(SOLVER_CACHE_PATH).ok()?;
    serde_json::from_str(&raw).ok()
}

fn save_solver_cache(entry: &SolverCacheEntry) -> Result<(), PreviewError> {
    let dir = std::path::Path::new(SOLVER_CACHE_PATH).parent().unwrap();
    fs::create_dir_all(dir).map_err(|e| PreviewError::Cache(e.to_string()))?;
    let json = serde_json::to_string_pretty(entry).map_err(|e| PreviewError::Cache(e.to_string()))?;
    fs::write(SOLVER_CACHE_PATH, json).map_err(|e| PreviewError::Cache(e.to_string()))
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
    let mut code =
        hex::decode(trimmed).map_err(|e| PreviewError::Build(format!("hex decode: {e}")))?;
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
    hex::decode(trimmed).map_err(|e| PreviewError::Build(format!("hex decode: {e}")))
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

/// Resolve (and deploy if needed) the Rebalancer + RebalancerMixed contracts.
pub async fn resolve_solvers<P: Provider<Ethereum> + Clone>(
    provider: P,
    owner: Address,
) -> Result<(Address, Address), PreviewError> {
    let chain_id = provider
        .get_chain_id()
        .await
        .map_err(|e| PreviewError::Provider(e.to_string()))?;

    if let Some(cached) = load_solver_cache() {
        if cached.chain_id == chain_id {
            let rebalancer = cached
                .rebalancer
                .parse::<Address>()
                .map_err(|e| PreviewError::Cache(e.to_string()))?;
            let rebalancer_mixed = cached
                .rebalancer_mixed
                .parse::<Address>()
                .map_err(|e| PreviewError::Cache(e.to_string()))?;
            // Quick validation: check code exists
            let code = provider
                .get_code_at(rebalancer)
                .await
                .map_err(|e| PreviewError::Provider(e.to_string()))?;
            if !code.is_empty() {
                tracing::info!(
                    %rebalancer,
                    %rebalancer_mixed,
                    "reused cached on-chain solvers"
                );
                return Ok((rebalancer, rebalancer_mixed));
            }
            tracing::info!("cached solver has no code; redeploying");
        }
    }

    let constructor_args = DynSolValue::Tuple(vec![
        DynSolValue::Address(SWAP_ROUTER_ADDRESS),
        DynSolValue::Address(CTF_ROUTER_ADDRESS),
    ])
    .abi_encode_params();

    let rebalancer_init = build_init_code(REBALANCER_ARTIFACT_PATH, &constructor_args)?;
    let rebalancer = deploy_contract(provider.clone(), owner, rebalancer_init).await?;
    tracing::info!(%rebalancer, "deployed Rebalancer");

    let mixed_init = build_init_code(REBALANCER_MIXED_ARTIFACT_PATH, &constructor_args)?;
    let rebalancer_mixed = deploy_contract(provider.clone(), owner, mixed_init).await?;
    tracing::info!(%rebalancer_mixed, "deployed RebalancerMixed");

    save_solver_cache(&SolverCacheEntry {
        chain_id,
        rebalancer: format!("{rebalancer}"),
        rebalancer_mixed: format!("{rebalancer_mixed}"),
    })?;

    Ok((rebalancer, rebalancer_mixed))
}

/// Read cached solver addresses without deploying.
pub fn resolve_solvers_readonly() -> Option<(Address, Address)> {
    let cached = load_solver_cache()?;
    let rebalancer = cached.rebalancer.parse::<Address>().ok()?;
    let rebalancer_mixed = cached.rebalancer_mixed.parse::<Address>().ok()?;
    Some((rebalancer, rebalancer_mixed))
}

// ── RebalanceParams builder ──

/// Build the on-chain RebalanceParams from current executor state.
pub fn build_rebalance_params(
    slot0_results: &[(Slot0Result, &'static MarketData)],
    balances_owned: &HashMap<&'static str, f64>,
    susds_balance: f64,
) -> Result<IRebalancer::RebalanceParams, PreviewError> {
    let predictions = prediction_map();

    let mut tokens: Vec<Address> = Vec::new();
    let mut pools: Vec<Address> = Vec::new();
    let mut is_token1: Vec<bool> = Vec::new();
    let mut balances: Vec<U256> = Vec::new();
    let mut sqrt_pred_x96: Vec<U160> = Vec::new();

    for (_slot0, market) in slot0_results {
        let pool = match market.pool.as_ref() {
            Some(p) => p,
            None => continue,
        };
        let outcome_addr: Address = market
            .outcome_token
            .parse()
            .map_err(|e| PreviewError::Build(format!("bad token addr {}: {e}", market.name)))?;
        let pool_addr: Address = pool
            .pool_id
            .parse()
            .map_err(|e| PreviewError::Build(format!("bad pool addr {}: {e}", market.name)))?;
        let is_t1 = pool.token1.eq_ignore_ascii_case(market.outcome_token);

        let pred = predictions
            .get(market.name)
            .copied()
            .unwrap_or(0.0);
        let sqrt_pred = match prediction_to_sqrt_price_x96(pred, is_t1) {
            Some(v) => U160::from(v),
            None => continue, // skip zero-prediction markets
        };

        let held = balances_owned
            .get(market.name)
            .copied()
            .unwrap_or(0.0);
        let balance_wei = U256::from((held * 1e18) as u128);

        tokens.push(outcome_addr);
        pools.push(pool_addr);
        is_token1.push(is_t1);
        balances.push(balance_wei);
        sqrt_pred_x96.push(sqrt_pred);
    }

    let collateral_amount = U256::from((susds_balance * 1e18) as u128);

    Ok(IRebalancer::RebalanceParams {
        tokens,
        pools,
        isToken1: is_token1,
        balances,
        collateralAmount: collateral_amount,
        sqrtPredX96: sqrt_pred_x96,
        collateral: BASE_COLLATERAL,
        fee: FEE_TIER.try_into().unwrap(),
    })
}

// ── Quoter call builder ──

fn build_batch_execute_calls(
    quoter: Address,
    solver: Address,
    solver_calldata: Bytes,
    params: &IRebalancer::RebalanceParams,
) -> Vec<ITradeExecutor::Call> {
    let mut calls: Vec<ITradeExecutor::Call> = Vec::new();

    // Transfer all tokens from executor to quoter
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

    // Transfer collateral
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

    // Call quoter.quote(solver, solverCalldata, tokens, collateral)
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

/// Parse the revert data from a SolverQuoter.quote() call.
/// Decodes: abi.encode(bool success, bytes returnData, uint256 postCash, uint256[] postBalances)
fn parse_quoter_revert(revert_data: &[u8]) -> Option<(bool, U256, Vec<U256>)> {
    let decoded =
        alloy::dyn_abi::DynSolType::Tuple(vec![
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
    let post_balances_dyn = match &tuple[3] {
        DynSolValue::Array(arr) => arr,
        _ => return None,
    };
    let post_balances: Vec<U256> = post_balances_dyn
        .iter()
        .filter_map(|v| v.as_uint().map(|(val, _)| val))
        .collect();

    Some((success, post_cash, post_balances))
}

/// Compute raw EV from post-rebalance balances.
fn compute_raw_ev(
    post_cash_wei: U256,
    post_balances_wei: &[U256],
    market_names: &[&'static str],
) -> f64 {
    let predictions = prediction_map();
    let cash = u256_to_f64(post_cash_wei);
    let holdings_ev: f64 = post_balances_wei
        .iter()
        .zip(market_names.iter())
        .map(|(balance, name)| {
            let pred = predictions.get(*name).copied().unwrap_or(0.0);
            u256_to_f64(*balance) * pred
        })
        .sum();
    cash + holdings_ev
}

// ── Public preview API ──

/// Run all on-chain solver variants via static call and return results sorted by raw EV descending.
pub async fn preview_all_solvers<P: Provider<Ethereum> + Clone>(
    provider: P,
    executor: Address,
    owner: Address,
    rebalancer: Address,
    rebalancer_mixed: Address,
    params: &IRebalancer::RebalanceParams,
    market_names: &[&'static str],
) -> Vec<SolverPreviewResult> {
    let quoter_bytecode = match get_deployed_bytecode(SOLVER_QUOTER_ARTIFACT_PATH) {
        Ok(code) => code,
        Err(err) => {
            tracing::warn!(error = %err, "failed to load SolverQuoter bytecode; skipping preview");
            return Vec::new();
        }
    };

    // Build solver calldata variants
    let solvers: Vec<(&str, Address, Bytes)> = vec![
        (
            "rebalanceExact",
            rebalancer,
            Bytes::from(
                IRebalancer::rebalanceExactCall {
                    params: params.clone(),
                    maxBisectionIterations: U256::from(MAX_BISECTION),
                    maxTickCrossingsPerPool: U256::from(MAX_TICK_CROSSINGS),
                }
                .abi_encode(),
            ),
        ),
        (
            "rebalanceAndArb",
            rebalancer,
            Bytes::from(
                IRebalancer::rebalanceAndArbCall {
                    params: params.clone(),
                    market: MARKET_1_ADDRESS,
                    maxArbRounds: U256::from(MAX_ARB_ROUNDS),
                    maxRecycleRounds: U256::from(MAX_RECYCLE_ROUNDS),
                }
                .abi_encode(),
            ),
        ),
        (
            "rebalanceMixedConstantL",
            rebalancer_mixed,
            Bytes::from(
                IRebalancer::rebalanceMixedConstantLCall {
                    params: params.clone(),
                    market: MARKET_1_ADDRESS,
                    maxOuterIterations: U256::from(MAX_OUTER_ITER),
                    maxInnerIterations: U256::from(MAX_INNER_ITER),
                    maxMintCollateral: params.collateralAmount,
                }
                .abi_encode(),
            ),
        ),
    ];

    let mut results = Vec::new();

    for (name, solver_addr, solver_calldata) in solvers {
        let calls = build_batch_execute_calls(
            QUOTER_ADDRESS,
            solver_addr,
            solver_calldata,
            params,
        );

        let batch_calldata = ITradeExecutor::batchExecuteCall { calls }.abi_encode();

        // Build eth_call with state override to inject SolverQuoter bytecode
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
            .overrides(alloy::rpc::types::state::StateOverride::from_iter(overrides))
            .await;

        match call_result {
            Ok(_) => {
                // SolverQuoter always reverts — if we get Ok, something is wrong
                tracing::warn!(solver = name, "unexpected success from quoter (expected revert)");
            }
            Err(err) => {
                // Parse revert data from the error
                let revert_data = extract_revert_data(&err);
                match revert_data.and_then(|data| parse_quoter_revert(&data)) {
                    Some((success, post_cash, post_balances)) => {
                        let raw_ev = compute_raw_ev(post_cash, &post_balances, market_names);
                        results.push(SolverPreviewResult {
                            name,
                            success,
                            post_cash_wei: post_cash,
                            post_balances_wei: post_balances,
                            raw_ev,
                        });
                    }
                    None => {
                        tracing::warn!(
                            solver = name,
                            "failed to parse quoter revert data"
                        );
                    }
                }
            }
        }
    }

    results.sort_by(|a, b| b.raw_ev.partial_cmp(&a.raw_ev).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Extract raw revert bytes from a provider error.
fn extract_revert_data(err: &alloy::transports::RpcError<alloy::transports::TransportErrorKind>) -> Option<Vec<u8>> {
    // The error message often contains the revert data as hex after "revert" or "execution reverted"
    let err_str = format!("{err:?}");

    // Try to find hex data in the error — common patterns:
    // "execution reverted: 0x..."
    // "revert: 0x..."
    // Or the error payload might contain it directly
    if let Some(pos) = err_str.find("0x") {
        // Extract the hex string
        let hex_part = &err_str[pos..];
        let end = hex_part
            .find(|c: char| !c.is_ascii_hexdigit() && c != 'x')
            .unwrap_or(hex_part.len());
        let hex_str = &hex_part[..end];
        let trimmed = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        return alloy::hex::decode(trimmed).ok();
    }
    None
}

/// Print a comparison table of solver preview results.
pub fn print_solver_comparison(results: &[SolverPreviewResult], pre_ev: f64) {
    if results.is_empty() {
        return;
    }

    eprintln!();
    eprintln!("  === On-Chain Solver Comparison (static call) ===");
    eprintln!(
        "  {:<30} {:>12} {:>12} {:>8}",
        "Solver", "Raw EV", "EV Delta", "Status"
    );
    eprintln!("  {:-<30} {:-<12} {:-<12} {:-<8}", "", "", "", "");
    for r in results {
        let delta = r.raw_ev - pre_ev;
        let status = if r.success { "OK" } else { "REVERT" };
        eprintln!(
            "  {:<30} {:>12.4} {:>+12.4} {:>8}",
            r.name, r.raw_ev, delta, status
        );
    }
    if let Some(best) = results.first() {
        eprintln!("  Best: {} (raw EV {:.4})", best.name, best.raw_ev);
    }
    eprintln!("  ===");
}
