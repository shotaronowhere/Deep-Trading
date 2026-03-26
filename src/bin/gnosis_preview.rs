//! Gnosis chain dry-run: poll blocks, static-call Algebra solvers, report best net-EV.
//!
//! Usage:
//!   cargo run --release --bin gnosis_preview
//!
//! Required .env:
//!   RPC_GNOSIS          Gnosis chain RPC endpoint
//!   PRIVATE_KEY          Signer private key (no tx sent in dry-run)
//!   TRADE_EXECUTOR       Deployed TradeExecutor address on Gnosis

use std::collections::HashMap;
use std::error::Error;
use std::time::Duration;

use alloy::network::{Ethereum, EthereumWallet};
use alloy::primitives::{Address, Bytes, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use alloy::sol_types::SolCall;
use tokio::time::sleep;

use deep_trading_bot::execution::gnosis_preview::{
    self, ActivePool, resolve_algebra_solver, resolve_algebra_solver_readonly,
};
use deep_trading_bot::execution::ITradeExecutor;
use deep_trading_bot::gnosis::MOVIES;

sol! {
    #[sol(rpc)]
    interface IERC20 {
        function balanceOf(address account) external view returns (uint256);
        function allowance(address owner, address spender) external view returns (uint256);
        function approve(address spender, uint256 amount) external returns (bool);
    }
}

const BLOCK_POLL_MS: u64 = 5000; // Gnosis ~5s block time
const GAS_ESTIMATE_REBALANCE: u64 = 3_000_000;

fn u256_to_f64(v: U256) -> f64 {
    v.to_string().parse::<f64>().unwrap_or(0.0) / 1e18
}

/// Check which computed pools actually exist on-chain.
async fn discover_active_pools<P: Provider<Ethereum> + Clone>(
    provider: P,
) -> Result<Vec<ActivePool>, Box<dyn Error>> {
    let mut active = Vec::new();

    for movie in MOVIES.iter() {
        let up_pool = movie.up_pool();
        let code = provider.get_code_at(up_pool).await?;
        if !code.is_empty() {
            active.push(ActivePool {
                token: movie.up_token,
                pool: up_pool,
                is_token1: movie.up_is_token1(),
                prediction: movie.up_prediction(),
                collateral: movie.underlying_token,
                market_id: movie.market_id,
            });
        } else {
            tracing::warn!(movie = movie.name, %up_pool, "upToken pool has no code");
        }

        let down_pool = movie.down_pool();
        let code = provider.get_code_at(down_pool).await?;
        if !code.is_empty() {
            active.push(ActivePool {
                token: movie.down_token,
                pool: down_pool,
                is_token1: movie.down_is_token1(),
                prediction: movie.down_prediction(),
                collateral: movie.underlying_token,
                market_id: movie.market_id,
            });
        } else {
            tracing::warn!(movie = movie.name, %down_pool, "downToken pool has no code");
        }
    }

    Ok(active)
}

/// Read all token balances (outcome tokens + underlying tokens) for the executor.
async fn fetch_balances<P: Provider<Ethereum> + Clone>(
    provider: P,
    executor: Address,
    active_pools: &[ActivePool],
) -> Result<HashMap<Address, U256>, Box<dyn Error>> {
    let mut balances = HashMap::new();

    // Collect unique tokens: outcome tokens + collateral (underlying) tokens
    let mut tokens_to_check: Vec<Address> = Vec::new();
    for ap in active_pools {
        if !tokens_to_check.contains(&ap.token) {
            tokens_to_check.push(ap.token);
        }
        if !tokens_to_check.contains(&ap.collateral) {
            tokens_to_check.push(ap.collateral);
        }
    }

    for token in tokens_to_check {
        let bal = IERC20::new(token, provider.clone())
            .balanceOf(executor)
            .call()
            .await?;
        if !bal.is_zero() {
            balances.insert(token, bal);
        }
    }

    Ok(balances)
}

fn portfolio_ev(
    token_balances: &HashMap<Address, U256>,
    active_pools: &[ActivePool],
) -> f64 {
    // TODO: underlying_token value depends on parent market probability, not 1:1 with sDAI.
    // Relative per-movie deltas are still valid; absolute EV is approximate.
    let mut collateral_ev = 0.0;
    let mut counted_collaterals = Vec::new();
    for ap in active_pools {
        if !counted_collaterals.contains(&ap.collateral) {
            collateral_ev += u256_to_f64(
                token_balances.get(&ap.collateral).copied().unwrap_or(U256::ZERO),
            );
            counted_collaterals.push(ap.collateral);
        }
    }
    let holdings_ev: f64 = active_pools
        .iter()
        .map(|ap| {
            u256_to_f64(token_balances.get(&ap.token).copied().unwrap_or(U256::ZERO))
                * ap.prediction
        })
        .sum();
    collateral_ev + holdings_ev
}

/// Ensure infinite approvals from executor to solver + routers.
async fn ensure_approvals<P: Provider<Ethereum> + Clone>(
    provider: P,
    executor: Address,
    solver: Address,
    active_pools: &[ActivePool],
) -> Result<(), Box<dyn Error>> {
    let max_approval = U256::MAX;
    // Solver handles its own router approvals internally via _safeApprove
    let spenders = [solver];

    let mut approval_calls: Vec<ITradeExecutor::Call> = Vec::new();

    // Collect unique tokens (outcome tokens + underlying/collateral tokens)
    let mut tokens: Vec<Address> = Vec::new();
    for ap in active_pools {
        if !tokens.contains(&ap.token) {
            tokens.push(ap.token);
        }
        if !tokens.contains(&ap.collateral) {
            tokens.push(ap.collateral);
        }
    }

    // Threshold: consider "infinite" if > 2^200
    let threshold = U256::from(1u128) << 127;

    for &token in &tokens {
        for &spender in &spenders {
            let allowance = IERC20::new(token, provider.clone())
                .allowance(executor, spender)
                .call()
                .await?
                ;
            if allowance < threshold {
                let data = IERC20::approveCall {
                    spender,
                    amount: max_approval,
                }
                .abi_encode();
                approval_calls.push(ITradeExecutor::Call {
                    to: token,
                    data: Bytes::from(data),
                });
            }
        }
    }

    if approval_calls.is_empty() {
        tracing::info!("all approvals already set");
        return Ok(());
    }

    tracing::info!(count = approval_calls.len(), "sending infinite approval batch");
    let batch_data = ITradeExecutor::batchExecuteCall {
        calls: approval_calls,
    }
    .abi_encode();
    let tx = alloy::rpc::types::TransactionRequest::default()
        .to(executor)
        .input(Bytes::from(batch_data).into());
    use alloy::network::ReceiptResponse;
    let receipt = provider
        .send_transaction(tx)
        .await
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })?
        .get_receipt()
        .await
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })?;
    if !receipt.status() {
        return Err(format!("approval batch reverted: {}", receipt.transaction_hash()).into());
    }
    tracing::info!(tx = %receipt.transaction_hash(), "approval batch confirmed");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // ── Config ──
    let rpc_url = std::env::var("RPC_GNOSIS")
        .unwrap_or_else(|_| "https://rpc.gnosischain.com".into());
    let private_key = std::env::var("PRIVATE_KEY").expect("PRIVATE_KEY must be set");
    let executor_str = std::env::var("TRADE_EXECUTOR").expect("TRADE_EXECUTOR must be set");
    let executor: Address = executor_str
        .parse()
        .expect("TRADE_EXECUTOR must be a valid address");
    let is_live = std::env::var("EXECUTE_SUBMIT")
        .map(|v| v == "Live")
        .unwrap_or(false);

    let signer: PrivateKeySigner = private_key.parse()?;
    let owner = signer.address();
    let wallet = EthereumWallet::from(signer);

    let provider = ProviderBuilder::new()
        .wallet(wallet)
        .with_reqwest(rpc_url.parse()?, |b| {
            b.no_proxy()
                .build()
                .expect("failed to build reqwest client")
        });

    let chain_id = provider.get_chain_id().await?;
    tracing::info!(%chain_id, %owner, %executor, "connected to Gnosis");

    // ── Discover active pools ──
    let active_pools = discover_active_pools(provider.clone()).await?;
    tracing::info!(count = active_pools.len(), "active pools discovered");
    if active_pools.is_empty() {
        tracing::error!("no active pools found — check CREATE2 derivation or pool pairing");
        return Ok(());
    }

    // ── Resolve solver ──
    let solver = if is_live {
        resolve_algebra_solver(provider.clone(), owner).await?
    } else {
        let addr = resolve_algebra_solver_readonly()
            .ok_or("no cached solver — run once with EXECUTE_SUBMIT=Live to deploy")?;
        // Verify cached solver has on-chain code
        let code = provider.get_code_at(addr).await?;
        if code.is_empty() {
            return Err("cached solver has no on-chain code — redeploy with EXECUTE_SUBMIT=Live".into());
        }
        addr
    };
    tracing::info!(%solver, "RebalancerAlgebra resolved");

    // ── Approvals pre-check (live mode only) ──
    if is_live {
        ensure_approvals(provider.clone(), executor, solver, &active_pools).await?;
    }

    // ── Block polling loop ──
    let poll_interval = Duration::from_millis(BLOCK_POLL_MS);
    let mut last_block: u64 = 0;

    tracing::info!("entering block polling loop (dry-run)");

    loop {
        let latest = match provider.get_block_number().await {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(error = %e, "RPC error fetching block number");
                sleep(poll_interval).await;
                continue;
            }
        };
        if latest <= last_block {
            sleep(poll_interval).await;
            continue;
        }
        last_block = latest;

        // Fetch balances
        let token_bals =
            match fetch_balances(provider.clone(), executor, &active_pools).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to fetch balances");
                    sleep(poll_interval).await;
                    continue;
                }
            };
        let pre_ev = portfolio_ev(&token_bals, &active_pools);

        tracing::info!(
            block = latest,
            pre_ev = format!("{pre_ev:.4}"),
            "new block"
        );

        // Group pools by collateral (underlying_token) — one solver call per movie
        let mut groups: HashMap<Address, Vec<&ActivePool>> = HashMap::new();
        for ap in &active_pools {
            groups.entry(ap.collateral).or_default().push(ap);
        }

        // Estimate gas cost in sDAI (~ xDAI ~ $1)
        let gas_price = provider.get_gas_price().await.unwrap_or(2_000_000_000);
        let gas_cost_xdai = (GAS_ESTIMATE_REBALANCE as f64) * (gas_price as f64) / 1e18;

        for (collateral, group) in &groups {
            let group_owned: Vec<ActivePool> = group.iter().map(|ap| (*ap).clone()).collect();
            let collateral_bal = token_bals.get(collateral).copied().unwrap_or(U256::ZERO);
            let params = gnosis_preview::build_algebra_params(&group_owned, &token_bals, collateral_bal);

            let arb_market = group_owned[0].market_id;
            let results = gnosis_preview::preview_algebra_solvers(
                provider.clone(),
                executor,
                owner,
                solver,
                &params,
                &group_owned,
                arb_market,
            )
            .await;

            let group_pre_ev = portfolio_ev(&token_bals, &group_owned);
            gnosis_preview::print_solver_comparison(&results, group_pre_ev, gas_cost_xdai);
        }

        sleep(poll_interval).await;
    }
}
