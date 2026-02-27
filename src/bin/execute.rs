use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

use alloy::network::EthereumWallet;
use alloy::network::ReceiptResponse;
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;

use deep_trading_bot::execution::approvals::ensure_executor_approvals;
use deep_trading_bot::execution::bounds::{
    BufferConfig, build_group_plans_with_default_edges, derive_batch_quote_bounds,
    stamp_plans_with_block,
};
use deep_trading_bot::execution::gas::default_gas_assumptions_with_optimism_l1_fee;
use deep_trading_bot::execution::runtime::{
    ExecutionRuntimeConfig, is_submission_deadline_exceeded, resolve_trade_executor,
    resolve_trade_executor_readonly,
};
use deep_trading_bot::execution::tx_builder::build_trade_executor_calls;
use deep_trading_bot::execution::{ITradeExecutor, SUSD_DECIMALS};
use deep_trading_bot::pools;
use deep_trading_bot::portfolio::{self, RebalanceMode};

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

    let config = ExecutionRuntimeConfig::from_env()?;
    let mode = parse_rebalance_mode();
    let eth_usd = parse_eth_usd();

    let signer: PrivateKeySigner = config.private_key.parse()?;
    let signer_address = signer.address();
    let wallet = EthereumWallet::from(signer);
    let provider =
        ProviderBuilder::new()
            .wallet(wallet)
            .with_reqwest(config.rpc_url.parse()?, |builder| {
                builder
                    .no_proxy()
                    .build()
                    .expect("failed to build reqwest client")
            });

    let resolved = if config.execute_submit {
        resolve_trade_executor(provider.clone(), signer_address).await?
    } else {
        resolve_trade_executor_readonly(provider.clone(), signer_address)
            .await?
            .ok_or_else(|| {
                "dry-run mode requires a valid cached TradeExecutor; run once with EXECUTE_SUBMIT=1 to deploy/cache one"
                    .to_string()
            })?
    };
    tracing::info!(
        chain_id = resolved.chain_id,
        signer = %signer_address,
        executor = %resolved.executor,
        reused_cache = resolved.reused_cache,
        execute_submit = config.execute_submit,
        mode = ?mode,
        "trade executor resolved"
    );
    if resolved.owner != signer_address {
        return Err(format!(
            "signer {} is not executor owner {}",
            signer_address, resolved.owner
        )
        .into());
    }

    if config.execute_submit {
        let approval_summary =
            ensure_executor_approvals(provider.clone(), resolved.executor).await?;
        tracing::info!(
            checked_pairs = approval_summary.checked_pairs,
            missing_pairs = approval_summary.missing_pairs,
            sent_txs = approval_summary.sent_txs,
            "approval preprocessing complete"
        );
    } else {
        tracing::info!("dry-run mode: skipping approval preprocessing");
    }

    let gas_assumptions = match default_gas_assumptions_with_optimism_l1_fee(&config.rpc_url).await
    {
        Ok(assumptions) => assumptions,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "failed to fetch L1 fee oracle; using default gas assumptions"
            );
            deep_trading_bot::execution::gas::GasAssumptions::default()
        }
    };

    let mut steps = 0usize;
    while steps < config.execution_max_steps {
        steps += 1;
        let step_started_at = Instant::now();

        let slot0_results = pools::fetch_all_slot0(provider.clone()).await?;
        if slot0_results.is_empty() {
            tracing::warn!("no pooled markets fetched; stopping execute loop");
            break;
        }

        let (susds_balance, balances_owned) =
            pools::fetch_balances(provider.clone(), resolved.executor).await?;
        let balances_view: HashMap<&str, f64> = balances_owned
            .iter()
            .map(|(market, units)| (*market as &str, *units))
            .collect();

        let actions = portfolio::rebalance_with_gas_pricing(
            &balances_view,
            susds_balance,
            &slot0_results,
            mode,
            &gas_assumptions,
            1e-9,
            eth_usd,
        );
        if actions.is_empty() {
            tracing::info!(step = steps, "no actions generated; stopping execute loop");
            break;
        }

        let mut plans = build_group_plans_with_default_edges(
            &actions,
            &gas_assumptions,
            1e-9,
            eth_usd,
            BufferConfig::default(),
        )?;
        if plans.is_empty() {
            use deep_trading_bot::execution::bounds::planned_edge_from_prediction_map_susd;
            use deep_trading_bot::execution::grouping::group_execution_actions;
            tracing::info!(
                step = steps,
                action_count = actions.len(),
                eth_usd,
                "no executable groups planned; stopping execute loop"
            );
            if let Ok(groups) = group_execution_actions(&actions) {
                let preds = deep_trading_bot::pools::prediction_map();
                for (idx, group) in groups.iter().take(3).enumerate() {
                    let edge = planned_edge_from_prediction_map_susd(group, &preds).unwrap_or(0.0);
                    let gas_est = deep_trading_bot::execution::gas::estimate_min_gas_susd_for_group(
                        &gas_assumptions,
                        group.kind,
                        group.buy_legs,
                        group.sell_legs,
                        1e-9,
                        eth_usd,
                    );
                    tracing::info!(
                        step = steps,
                        group_index = idx,
                        group_kind = ?group.kind,
                        buy_legs = group.buy_legs,
                        sell_legs = group.sell_legs,
                        edge_plan_susd = edge,
                        gas_estimate_susd = gas_est,
                        "empty-plan diagnostics"
                    );
                }
            }
            break;
        }

        let planning_block = provider
            .get_block_number()
            .await
            .map_err(|err| format!("failed to fetch planning block: {err}"))?;
        stamp_plans_with_block(&mut plans, planning_block);

        let plan = plans
            .first()
            .cloned()
            .ok_or_else(|| "strict execution expected one first group".to_string())?;

        let submit_block = provider
            .get_block_number()
            .await
            .map_err(|err| format!("failed to fetch submit block: {err}"))?;
        let quote_bounds =
            derive_batch_quote_bounds(&plan, submit_block, config.execution_max_stale_blocks)
                .map_err(|err| {
                    format!(
                        "stale or invalid bounds for group {:?} at step {}: {err}",
                        plan.kind, steps
                    )
                })?;
        let token_bounds = quote_bounds
            .map(|bounds| bounds.to_token_bounds(SUSD_DECIMALS))
            .transpose()?;
        if is_submission_deadline_exceeded(
            step_started_at.elapsed(),
            config.execution_deadline_secs,
        ) {
            return Err(format!(
                "deadline exceeded before submit ({}s)",
                config.execution_deadline_secs
            )
            .into());
        }

        let calls = build_trade_executor_calls(&actions, &plan, token_bounds).map_err(|err| {
            format!(
                "failed to build executor calls for {:?} step {}: {err}",
                plan.kind, steps
            )
        })?;
        if calls.is_empty() {
            return Err(format!("group {:?} produced zero executor calls", plan.kind).into());
        }

        tracing::info!(
            step = steps,
            block = submit_block,
            group_kind = ?plan.kind,
            action_indices = ?plan.action_indices,
            calls = calls.len(),
            quote_bounds = ?quote_bounds,
            token_bounds = ?token_bounds,
            "prepared strict subgroup transaction"
        );

        if !config.execute_submit {
            for (call_index, call) in calls.iter().enumerate() {
                tracing::info!(
                    step = steps,
                    call_index,
                    target = %call.to,
                    calldata = %alloy::hex::encode_prefixed(&call.data),
                    "dry-run call"
                );
            }
            tracing::info!(
                step = steps,
                "dry-run mode active (set EXECUTE_SUBMIT=1 to broadcast); stopping after first subgroup"
            );
            break;
        }

        let executor_contract = ITradeExecutor::new(resolved.executor, provider.clone());
        let receipt = executor_contract
            .batchExecute(calls)
            .send()
            .await?
            .get_receipt()
            .await?;
        if !receipt.status() {
            return Err(format!("executor tx {} reverted", receipt.transaction_hash()).into());
        }

        tracing::info!(
            step = steps,
            tx_hash = %receipt.transaction_hash(),
            block = ?receipt.block_number(),
            gas_used = receipt.gas_used(),
            "strict subgroup submitted"
        );
    }

    if steps >= config.execution_max_steps {
        tracing::info!(
            max_steps = config.execution_max_steps,
            "execution step cap reached; stopping"
        );
    }

    Ok(())
}
