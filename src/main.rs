use std::collections::HashMap;
use std::error::Error;
use std::str::FromStr;

use alloy::primitives::Address;
use alloy::providers::{Provider, ProviderBuilder};
use deep_trading_bot::execution::bounds::{
    BufferConfig, ConservativeExecutionConfig,
    build_group_plans_with_default_edges_and_market_context, derive_batch_quote_bounds,
    stamp_plans_with_block,
};
use deep_trading_bot::execution::gas::default_gas_assumptions_with_optimism_l1_fee;
use deep_trading_bot::execution::preview::print_execution_group_preview;
use deep_trading_bot::execution::runtime::{
    DEFAULT_EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK, DEFAULT_EXECUTION_MAX_STALE_BLOCKS,
    DEFAULT_EXECUTION_QUOTE_LATENCY_BLOCKS,
};
use deep_trading_bot::pools;
use deep_trading_bot::portfolio::{self, RebalanceFlags, RebalanceMode, TraceConfig};

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

fn parse_quote_latency_blocks() -> u64 {
    std::env::var("EXECUTION_QUOTE_LATENCY_BLOCKS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_EXECUTION_QUOTE_LATENCY_BLOCKS)
}

fn parse_adverse_move_bps_per_block() -> u64 {
    std::env::var("EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK)
}

fn parse_execution_max_stale_blocks() -> u64 {
    std::env::var("EXECUTION_MAX_STALE_BLOCKS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_EXECUTION_MAX_STALE_BLOCKS)
}

fn parse_enable_greedy_churn_pruning() -> bool {
    matches!(
        std::env::var("REBALANCE_ENABLE_GREEDY_CHURN_PRUNING")
            .ok()
            .map(|raw| raw.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on") | Some("enabled")
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let rpc_url = std::env::var("RPC").unwrap_or_else(|_| "https://optimism.drpc.org".to_string());

    let provider = ProviderBuilder::new().with_reqwest(rpc_url.parse()?, |builder| {
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
    let enable_greedy_churn_pruning = parse_enable_greedy_churn_pruning();
    let conservative_execution = ConservativeExecutionConfig {
        quote_latency_blocks: parse_quote_latency_blocks(),
        adverse_move_bps_per_block: parse_adverse_move_bps_per_block(),
    };
    let execution_max_stale_blocks = parse_execution_max_stale_blocks();

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

    let actions = portfolio::rebalance_with_gas_pricing_and_flags(
        &balances_view,
        initial_susd,
        &slot0_results,
        mode,
        &gas_assumptions,
        1e-9,
        eth_usd,
        RebalanceFlags {
            enable_ev_guarded_greedy_churn_pruning: enable_greedy_churn_pruning,
        },
    );
    let mut plans = build_group_plans_with_default_edges_and_market_context(
        &actions,
        &slot0_results,
        conservative_execution,
        &gas_assumptions,
        1e-9,
        eth_usd,
        BufferConfig::default(),
    )?;
    if plans.is_empty() {
        println!("[rebalance][runtime] execution planner: no executable groups");
    } else {
        let planning_block = provider
            .get_block_number()
            .await
            .map_err(|err| format!("failed to fetch planning block: {err}"))?;
        stamp_plans_with_block(&mut plans, planning_block);

        let first_plan = &plans[0];
        let batch_bounds =
            derive_batch_quote_bounds(first_plan, planning_block, execution_max_stale_blocks)?;
        print_execution_group_preview(
            "[rebalance][runtime] ",
            first_plan,
            conservative_execution.quote_latency_blocks,
            conservative_execution.adverse_move_bps_per_block,
            batch_bounds,
        );
    }

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
        enable_greedy_churn_pruning,
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
