use std::collections::HashMap;
use std::error::Error;
use std::str::FromStr;

use alloy::primitives::Address;
use deep_trading_bot::execution::gas::default_gas_assumptions_with_optimism_l1_fee;
use deep_trading_bot::pools;
use deep_trading_bot::portfolio::{self, RebalanceMode, TraceConfig};

fn parse_rebalance_mode() -> RebalanceMode {
    match std::env::var("REBALANCE_MODE")
        .ok()
        .map(|raw| raw.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("arb") | Some("arb_only") | Some("arbonly") => RebalanceMode::ArbOnly,
        _ => RebalanceMode::Full,
    }
}

fn parse_starting_susd() -> f64 {
    std::env::var("STARTING_SUSD")
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(100.0)
}

fn parse_eth_usd() -> f64 {
    std::env::var("ETH_USD")
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(3000.0)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let rpc_url = std::env::var("RPC").unwrap_or_else(|_| "https://optimism.drpc.org".to_string());

    let provider =
        alloy::providers::ProviderBuilder::new().with_reqwest(rpc_url.parse()?, |builder| {
            builder
                .no_proxy()
                .build()
                .expect("failed to build reqwest client")
        });

    let slot0_results = pools::fetch_all_slot0(provider.clone()).await?;
    if slot0_results.is_empty() {
        tracing::warn!("no pooled markets were fetched; skipping rebalance run");
        return Ok(());
    }

    let mode = parse_rebalance_mode();
    let eth_usd = parse_eth_usd();

    let gas_assumptions = if mode == RebalanceMode::Full {
        match default_gas_assumptions_with_optimism_l1_fee(&rpc_url).await {
            Ok(ga) => ga,
            Err(err) => {
                tracing::warn!(error = %err, "failed to fetch L1 fee oracle; using default GasAssumptions");
                deep_trading_bot::execution::gas::GasAssumptions::default()
            }
        }
    } else {
        deep_trading_bot::execution::gas::GasAssumptions::default()
    };

    let (initial_susd, balances_owned): (f64, HashMap<&'static str, f64>) =
        if let Ok(wallet_raw) = std::env::var("WALLET") {
            let wallet = Address::from_str(wallet_raw.trim())?;
            tracing::info!(wallet = %wallet, "fetching wallet balances");
            pools::fetch_balances(provider.clone(), wallet).await?
        } else {
            let starting_susd = parse_starting_susd();
            tracing::info!(
                starting_susd,
                "WALLET not set; using synthetic starting state with zero outcome holdings"
            );
            (starting_susd, HashMap::new())
        };

    let balances_view: HashMap<&str, f64> = balances_owned
        .iter()
        .map(|(market, units)| (*market as &str, *units))
        .collect();

    let actions = portfolio::rebalance_with_gas_pricing(
        &balances_view,
        initial_susd,
        &slot0_results,
        mode,
        &gas_assumptions,
        1e-9,
        eth_usd,
    );
    let (final_holdings, final_susd) = portfolio::replay_actions_to_portfolio_state(
        &actions,
        &slot0_results,
        &balances_view,
        initial_susd,
    );
    let trace_config = TraceConfig::from_env();

    tracing::info!(
        mode = ?mode,
        eth_usd,
        markets = slot0_results.len(),
        actions = actions.len(),
        "completed rebalance planning"
    );

    portfolio::print_portfolio_snapshot(
        "runtime",
        "initial",
        &balances_owned,
        initial_susd,
        trace_config,
    );
    portfolio::print_rebalance_execution_summary("runtime", &actions, &slot0_results, trace_config);
    portfolio::print_portfolio_snapshot(
        "runtime",
        "final",
        &final_holdings,
        final_susd,
        trace_config,
    );

    Ok(())
}
