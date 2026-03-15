use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};

use alloy::network::EthereumWallet;
use alloy::network::ReceiptResponse;
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;
use sha2::{Digest, Sha256};
use tokio::task::JoinHandle;
use tokio::time::sleep;

use deep_trading_bot::execution::approvals::ensure_executor_approvals;
use deep_trading_bot::execution::bounds::{
    BufferConfig, ConservativeExecutionConfig,
    build_group_plans_with_default_edges_and_market_context, derive_batch_quote_bounds,
    stamp_plans_with_block,
};
use deep_trading_bot::execution::gas::{
    GasAssumptions, default_gas_assumptions_with_optimism_l1_fee, fetch_live_optimism_fee_inputs,
};
use deep_trading_bot::execution::program::{
    build_chunk_calls_checked, compile_execution_program_unchecked,
};
use deep_trading_bot::execution::runtime::{
    ExecutionRuntimeConfig, is_submission_deadline_exceeded, resolve_trade_executor,
    resolve_trade_executor_readonly,
};
use deep_trading_bot::execution::tx_builder::build_trade_executor_calls;
use deep_trading_bot::execution::{ExecutionMode, ITradeExecutor, SUSD_DECIMALS};
use deep_trading_bot::markets::MarketData;
use deep_trading_bot::pools::{self, Slot0Result};
use deep_trading_bot::portfolio::{
    self, RebalanceFlags, RebalanceMode, RebalancePlanDecision, RebalanceSolver,
    compare_rebalance_plan_decisions,
};

const EXPLICIT_GAS_PRICE_ETH: f64 = 1e-9;

struct ForecastFlowsShutdownGuard {
    enabled: bool,
}

impl ForecastFlowsShutdownGuard {
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }
}

impl Drop for ForecastFlowsShutdownGuard {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        if let Err(err) = portfolio::shutdown_forecastflows_worker() {
            tracing::warn!(error = %err, "failed to shut down ForecastFlows worker");
        }
    }
}

#[derive(Debug)]
struct NativeAuditOutcome {
    cycle_block: u64,
    winner: &'static str,
    native_minus_live_net_ev: Option<f64>,
    native_minus_live_tx_count: Option<i64>,
    native_minus_live_action_count: i64,
    forecastflows_fallback_reason: Option<String>,
    duration_ms: u128,
}

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

fn parse_rebalance_solver() -> RebalanceSolver {
    RebalanceSolver::from_env()
}

fn parse_eth_usd() -> f64 {
    std::env::var("ETH_USD")
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(3000.0)
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

fn balances_view<'a>(balances_owned: &'a HashMap<&'static str, f64>) -> HashMap<&'a str, f64> {
    balances_owned
        .iter()
        .map(|(market, units)| (*market as &str, *units))
        .collect()
}

fn skipped_blocks_while_busy(cycle_block: u64, latest_block: u64) -> u64 {
    latest_block.saturating_sub(cycle_block.saturating_add(1))
}

fn native_audit_due(
    last_audit_anchor_block: u64,
    current_block: u64,
    interval_blocks: u64,
) -> bool {
    interval_blocks > 0 && current_block.saturating_sub(last_audit_anchor_block) >= interval_blocks
}

fn option_f64_delta(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) if left.is_finite() && right.is_finite() => Some(left - right),
        _ => None,
    }
}

fn option_usize_delta(left: Option<usize>, right: Option<usize>) -> Option<i64> {
    Some(i64::try_from(left?).ok()? - i64::try_from(right?).ok()?)
}

fn should_cache_snapshot_after_forecastflows_attempt(fallback_reason: Option<&str>) -> bool {
    !matches!(
        fallback_reason,
        Some(
            "worker_spawn_error"
                | "worker_io_error"
                | "worker_timeout"
                | "worker_closed"
                | "worker_protocol_error"
                | "worker_cooldown"
        )
    )
}

fn snapshot_fingerprint(
    susds_balance: f64,
    balances_owned: &HashMap<&'static str, f64>,
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(susds_balance.to_le_bytes());

    let mut sorted_balances: Vec<_> = balances_owned.iter().collect();
    sorted_balances.sort_unstable_by_key(|(market, _)| **market);
    for (market, units) in sorted_balances {
        hasher.update(market.as_bytes());
        hasher.update([0]);
        hasher.update(units.to_le_bytes());
    }

    let mut sorted_slot0_results: Vec<_> = slot0_results.iter().collect();
    sorted_slot0_results.sort_unstable_by_key(|(_, market)| market.name);
    for (slot0, market) in sorted_slot0_results {
        hasher.update(market.name.as_bytes());
        hasher.update([0]);
        hasher.update(slot0.pool_id.to_string().as_bytes());
        hasher.update(slot0.sqrt_price_x96.to_string().as_bytes());
        hasher.update(slot0.tick.to_le_bytes());
        hasher.update(slot0.observation_index.to_le_bytes());
        hasher.update(slot0.observation_cardinality.to_le_bytes());
        hasher.update(slot0.observation_cardinality_next.to_le_bytes());
        hasher.update([slot0.fee_protocol]);
        hasher.update([u8::from(slot0.unlocked)]);
    }

    format!("{:x}", hasher.finalize())
}

fn spawn_native_audit(
    cycle_block: u64,
    balances_owned: HashMap<&'static str, f64>,
    susds_balance: f64,
    slot0_results: Vec<(Slot0Result, &'static MarketData)>,
    gas_assumptions: GasAssumptions,
    eth_usd: f64,
    flags: RebalanceFlags,
    live_decision: RebalancePlanDecision,
) -> JoinHandle<NativeAuditOutcome> {
    tokio::task::spawn_blocking(move || {
        let audit_started_at = Instant::now();
        let balances_view = balances_view(&balances_owned);
        let native_decision =
            portfolio::rebalance_with_solver_and_gas_pricing_and_flags_and_decision(
                &balances_view,
                susds_balance,
                &slot0_results,
                RebalanceMode::Full,
                RebalanceSolver::Native,
                &gas_assumptions,
                EXPLICIT_GAS_PRICE_ETH,
                eth_usd,
                flags,
            );

        let winner = match compare_rebalance_plan_decisions(&native_decision, &live_decision) {
            Ordering::Less => "native",
            Ordering::Greater => "forecastflows",
            Ordering::Equal => "tie",
        };

        NativeAuditOutcome {
            cycle_block,
            winner,
            native_minus_live_net_ev: option_f64_delta(
                native_decision.summary.estimated_net_ev,
                live_decision.summary.estimated_net_ev,
            ),
            native_minus_live_tx_count: option_usize_delta(
                native_decision.summary.estimated_tx_count,
                live_decision.summary.estimated_tx_count,
            ),
            native_minus_live_action_count: i64::try_from(native_decision.actions.len())
                .unwrap_or(i64::MAX)
                - i64::try_from(live_decision.actions.len()).unwrap_or(i64::MAX),
            forecastflows_fallback_reason: live_decision
                .summary
                .forecastflows_telemetry
                .fallback_reason
                .clone(),
            duration_ms: audit_started_at.elapsed().as_millis(),
        }
    })
}

async fn log_finished_native_audit(handle: JoinHandle<NativeAuditOutcome>, shutdown: bool) {
    match handle.await {
        Ok(outcome) => {
            if shutdown {
                tracing::info!(
                    audit_block = outcome.cycle_block,
                    audit_winner = outcome.winner,
                    native_minus_live_net_ev = outcome.native_minus_live_net_ev,
                    native_minus_live_tx_count = outcome.native_minus_live_tx_count,
                    native_minus_live_action_count = outcome.native_minus_live_action_count,
                    forecastflows_fallback_reason = outcome
                        .forecastflows_fallback_reason
                        .as_deref()
                        .unwrap_or("none"),
                    audit_duration_ms = outcome.duration_ms,
                    "native audit completed during shutdown"
                );
            } else {
                tracing::info!(
                    audit_block = outcome.cycle_block,
                    audit_winner = outcome.winner,
                    native_minus_live_net_ev = outcome.native_minus_live_net_ev,
                    native_minus_live_tx_count = outcome.native_minus_live_tx_count,
                    native_minus_live_action_count = outcome.native_minus_live_action_count,
                    forecastflows_fallback_reason = outcome
                        .forecastflows_fallback_reason
                        .as_deref()
                        .unwrap_or("none"),
                    audit_duration_ms = outcome.duration_ms,
                    "native audit completed"
                );
            }
        }
        Err(err) => {
            if shutdown {
                tracing::warn!(error = %err, "native audit task failed during shutdown");
            } else {
                tracing::warn!(error = %err, "native audit task failed");
            }
        }
    }
}

fn log_skipped_blocks(step: usize, cycle_block: u64, latest_block: u64, total: &mut u64) {
    let skipped = skipped_blocks_while_busy(cycle_block, latest_block);
    if skipped == 0 {
        return;
    }
    *total += skipped;
    tracing::info!(
        step,
        cycle_block,
        latest_block,
        skipped_blocks_while_busy = skipped,
        skipped_blocks_while_busy_total = *total,
        "planning cycle skipped stale blocks while busy"
    );
}

fn log_live_plan_summary(
    step: usize,
    cycle_block: u64,
    solver: RebalanceSolver,
    planning_started_at: Instant,
    live_decision: &RebalancePlanDecision,
) {
    tracing::info!(
        step,
        block = cycle_block,
        requested_solver = solver.as_str(),
        action_count = live_decision.actions.len(),
        estimated_net_ev = live_decision.summary.estimated_net_ev,
        estimated_tx_count = live_decision.summary.estimated_tx_count,
        forecastflows_live_solve_ms = live_decision
            .summary
            .forecastflows_telemetry
            .worker_roundtrip_ms,
        forecastflows_translation_replay_ms = live_decision
            .summary
            .forecastflows_telemetry
            .translation_replay_ms,
        forecastflows_local_candidate_build_ms = live_decision
            .summary
            .forecastflows_telemetry
            .local_candidate_build_ms,
        forecastflows_local_step_prune_ms = live_decision
            .summary
            .forecastflows_telemetry
            .local_step_prune_ms,
        forecastflows_local_route_prune_ms = live_decision
            .summary
            .forecastflows_telemetry
            .local_route_prune_ms,
        forecastflows_estimated_execution_cost_susd = live_decision
            .summary
            .forecastflows_telemetry
            .estimated_execution_cost_susd,
        forecastflows_estimated_net_ev_susd = live_decision
            .summary
            .forecastflows_telemetry
            .estimated_net_ev_susd,
        forecastflows_validated_total_fee_susd = live_decision
            .summary
            .forecastflows_telemetry
            .validated_total_fee_susd,
        forecastflows_validated_net_ev_susd = live_decision
            .summary
            .forecastflows_telemetry
            .validated_net_ev_susd,
        forecastflows_fee_estimate_error_susd = live_decision
            .summary
            .forecastflows_telemetry
            .fee_estimate_error_susd,
        forecastflows_validation_only = live_decision
            .summary
            .forecastflows_telemetry
            .validation_only,
        forecastflows_workspace_reused = live_decision
            .summary
            .forecastflows_telemetry
            .workspace_reused,
        forecastflows_driver_overhead_ms = live_decision
            .summary
            .forecastflows_telemetry
            .driver_overhead_ms,
        forecastflows_strategy = live_decision
            .summary
            .forecastflows_telemetry
            .strategy
            .as_deref()
            .unwrap_or("none"),
        forecastflows_fallback_reason = live_decision
            .summary
            .forecastflows_telemetry
            .fallback_reason
            .as_deref()
            .unwrap_or("none"),
        end_to_end_planning_ms = planning_started_at.elapsed().as_millis(),
        "planner cycle completed"
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = ExecutionRuntimeConfig::from_env()?;
    let mode = parse_rebalance_mode();
    let solver = parse_rebalance_solver();
    let native_audit_enabled = mode == RebalanceMode::Full
        && solver == RebalanceSolver::ForecastFlows
        && config.forecastflows_native_audit_interval_blocks > 0;
    let _forecastflows_shutdown_guard =
        ForecastFlowsShutdownGuard::new(mode == RebalanceMode::Full && solver.uses_forecastflows());
    let eth_usd = parse_eth_usd();
    let enable_greedy_churn_pruning = parse_enable_greedy_churn_pruning();
    let flags = RebalanceFlags {
        enable_ev_guarded_greedy_churn_pruning: enable_greedy_churn_pruning,
    };
    let conservative_execution = ConservativeExecutionConfig {
        quote_latency_blocks: config.execution_quote_latency_blocks,
        adverse_move_bps_per_block: config.execution_adverse_move_bps_per_block,
    };
    let poll_interval = Duration::from_millis(config.execution_block_poll_ms);

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
        requested_solver = solver.as_str(),
        native_audit_enabled,
        native_audit_interval_blocks = config.forecastflows_native_audit_interval_blocks,
        execution_block_poll_ms = config.execution_block_poll_ms,
        enable_greedy_churn_pruning,
        quote_latency_blocks = conservative_execution.quote_latency_blocks,
        adverse_move_bps_per_block = conservative_execution.adverse_move_bps_per_block,
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

    if mode == RebalanceMode::Full && solver.uses_forecastflows() {
        if let Err(err) = portfolio::warm_forecastflows_worker() {
            tracing::warn!(error = %err, "ForecastFlows worker warmup failed; native fallback remains enabled");
        }
    }

    let mut steps = 0usize;
    let mut skipped_blocks_while_busy_total = 0u64;
    let mut unchanged_snapshot_skips = 0u64;
    let mut last_snapshot_fingerprint: Option<String> = None;
    let mut native_audit_handle: Option<JoinHandle<NativeAuditOutcome>> = None;
    let mut last_seen_block = provider
        .get_block_number()
        .await
        .map_err(|err| format!("failed to fetch initial block number: {err}"))?;
    let mut last_audit_anchor_block = last_seen_block;

    while steps < config.execution_max_steps {
        if let Some(handle) = native_audit_handle.as_ref()
            && handle.is_finished()
        {
            let handle = native_audit_handle
                .take()
                .expect("native audit handle should be present when finished");
            log_finished_native_audit(handle, false).await;
        }

        let latest_block = provider
            .get_block_number()
            .await
            .map_err(|err| format!("failed to poll latest block: {err}"))?;
        if latest_block <= last_seen_block {
            sleep(poll_interval).await;
            continue;
        }

        let cycle_block = latest_block;
        last_seen_block = cycle_block;
        steps += 1;
        let planning_started_at = Instant::now();

        let slot0_results = pools::fetch_all_slot0(provider.clone()).await?;
        if slot0_results.is_empty() {
            tracing::warn!(
                step = steps,
                block = cycle_block,
                "no pooled markets fetched; waiting for next block"
            );
            continue;
        }

        let (susds_balance, balances_owned) =
            pools::fetch_balances(provider.clone(), resolved.executor).await?;
        let fingerprint = snapshot_fingerprint(susds_balance, &balances_owned, &slot0_results);
        if last_snapshot_fingerprint.as_deref() == Some(fingerprint.as_str()) {
            unchanged_snapshot_skips += 1;
            tracing::info!(
                step = steps,
                block = cycle_block,
                unchanged_snapshot_skips,
                "snapshot unchanged; skipping planning cycle"
            );
            let latest_after_cycle = provider.get_block_number().await.unwrap_or(cycle_block);
            log_skipped_blocks(
                steps,
                cycle_block,
                latest_after_cycle,
                &mut skipped_blocks_while_busy_total,
            );
            last_seen_block = last_seen_block.max(latest_after_cycle);
            continue;
        }

        let balances_view = balances_view(&balances_owned);
        let live_decision = portfolio::rebalance_with_solver_and_gas_pricing_and_flags_and_decision(
            &balances_view,
            susds_balance,
            &slot0_results,
            mode,
            solver,
            &gas_assumptions,
            EXPLICIT_GAS_PRICE_ETH,
            eth_usd,
            flags,
        );
        let cache_snapshot = !solver.uses_forecastflows()
            || should_cache_snapshot_after_forecastflows_attempt(
                live_decision
                    .summary
                    .forecastflows_telemetry
                    .fallback_reason
                    .as_deref(),
            );
        if cache_snapshot {
            last_snapshot_fingerprint = Some(fingerprint);
        }
        log_live_plan_summary(
            steps,
            cycle_block,
            solver,
            planning_started_at,
            &live_decision,
        );

        if native_audit_enabled
            && native_audit_handle.is_none()
            && native_audit_due(
                last_audit_anchor_block,
                cycle_block,
                config.forecastflows_native_audit_interval_blocks,
            )
        {
            native_audit_handle = Some(spawn_native_audit(
                cycle_block,
                balances_owned.clone(),
                susds_balance,
                slot0_results.clone(),
                gas_assumptions,
                eth_usd,
                flags,
                live_decision.clone(),
            ));
            last_audit_anchor_block = cycle_block;
            tracing::info!(
                step = steps,
                block = cycle_block,
                audit_interval_blocks = config.forecastflows_native_audit_interval_blocks,
                "scheduled native audit"
            );
        }

        if live_decision.actions.is_empty() {
            tracing::info!(
                step = steps,
                block = cycle_block,
                "no actions generated; waiting for next block"
            );
            let latest_after_cycle = provider.get_block_number().await.unwrap_or(cycle_block);
            log_skipped_blocks(
                steps,
                cycle_block,
                latest_after_cycle,
                &mut skipped_blocks_while_busy_total,
            );
            last_seen_block = last_seen_block.max(latest_after_cycle);
            continue;
        }

        let mut plans = build_group_plans_with_default_edges_and_market_context(
            &live_decision.actions,
            &slot0_results,
            conservative_execution,
            &gas_assumptions,
            EXPLICIT_GAS_PRICE_ETH,
            eth_usd,
            BufferConfig::default(),
        )?;
        if plans.is_empty() {
            use deep_trading_bot::execution::bounds::planned_edge_from_prediction_map_susd;
            use deep_trading_bot::execution::grouping::group_execution_actions;

            tracing::info!(
                step = steps,
                block = cycle_block,
                action_count = live_decision.actions.len(),
                eth_usd,
                "no executable groups planned; waiting for next block"
            );
            if let Ok(groups) = group_execution_actions(&live_decision.actions) {
                let predictions = deep_trading_bot::pools::prediction_map();
                for (idx, group) in groups.iter().take(3).enumerate() {
                    let edge =
                        planned_edge_from_prediction_map_susd(group, predictions).unwrap_or(0.0);
                    let gas_estimate =
                        deep_trading_bot::execution::gas::estimate_min_gas_susd_for_group(
                            &gas_assumptions,
                            group.kind,
                            group.buy_legs,
                            group.sell_legs,
                            EXPLICIT_GAS_PRICE_ETH,
                            eth_usd,
                        );
                    tracing::info!(
                        step = steps,
                        group_index = idx,
                        group_kind = ?group.kind,
                        buy_legs = group.buy_legs,
                        sell_legs = group.sell_legs,
                        edge_plan_susd = edge,
                        gas_estimate_susd = gas_estimate,
                        "empty-plan diagnostics"
                    );
                }
            }
            let latest_after_cycle = provider.get_block_number().await.unwrap_or(cycle_block);
            log_skipped_blocks(
                steps,
                cycle_block,
                latest_after_cycle,
                &mut skipped_blocks_while_busy_total,
            );
            last_seen_block = last_seen_block.max(latest_after_cycle);
            continue;
        }

        let planning_block = provider
            .get_block_number()
            .await
            .map_err(|err| format!("failed to fetch planning block: {err}"))?;
        stamp_plans_with_block(&mut plans, planning_block);

        let fee_inputs = fetch_live_optimism_fee_inputs(&config.rpc_url, signer_address).await?;
        let packed_program = compile_execution_program_unchecked(
            ExecutionMode::Packed,
            resolved.executor,
            &live_decision.actions,
            &plans,
            fee_inputs,
            &gas_assumptions,
            eth_usd,
        );

        let submit_block = provider
            .get_block_number()
            .await
            .map_err(|err| format!("failed to fetch submit block: {err}"))?;
        if is_submission_deadline_exceeded(
            planning_started_at.elapsed(),
            config.execution_deadline_secs,
        ) {
            return Err(format!(
                "deadline exceeded before submit ({}s)",
                config.execution_deadline_secs
            )
            .into());
        }

        let (execution_mode, chunk_plans, calls) = match packed_program {
            Ok(program) => {
                let first_chunk = program
                    .chunks
                    .first()
                    .cloned()
                    .ok_or_else(|| "packed execution expected a first chunk".to_string())?;
                match build_chunk_calls_checked(
                    resolved.executor,
                    &live_decision.actions,
                    &first_chunk.plans,
                    submit_block,
                    config.execution_max_stale_blocks,
                ) {
                    Ok(calls) if !calls.is_empty() => {
                        tracing::info!(
                            step = steps,
                            block = submit_block,
                            tx_mode = "packed",
                            packed_tx_count = program.tx_count,
                            packed_strict_subgroup_count = program.strict_subgroup_count,
                            first_chunk_subgroups = first_chunk.plans.len(),
                            first_chunk_l2_gas_units = first_chunk.total_l2_gas_units,
                            first_chunk_estimated_fee_susd = first_chunk.estimated_total_fee_susd,
                            end_to_end_planning_ms = planning_started_at.elapsed().as_millis(),
                            "prepared packed chunk transaction"
                        );
                        (ExecutionMode::Packed, first_chunk.plans, calls)
                    }
                    Ok(_) => {
                        return Err("packed chunk produced zero executor calls".into());
                    }
                    Err(err) => {
                        tracing::warn!(
                            step = steps,
                            error = %err,
                            "packed chunk assembly failed; falling back to strict subgroup submission"
                        );
                        let plan = plans.first().cloned().ok_or_else(|| {
                            "strict execution expected one first group".to_string()
                        })?;
                        let quote_bounds = derive_batch_quote_bounds(
                            &plan,
                            submit_block,
                            config.execution_max_stale_blocks,
                        )
                        .map_err(|bound_err| {
                            format!(
                                "stale or invalid bounds for group {:?} at step {}: {bound_err}",
                                plan.kind, steps
                            )
                        })?;
                        let token_bounds = quote_bounds
                            .map(|bounds| bounds.to_token_bounds(SUSD_DECIMALS))
                            .transpose()?;
                        let calls = build_trade_executor_calls(
                            resolved.executor,
                            &live_decision.actions,
                            &plan,
                            token_bounds,
                        )
                        .map_err(|tx_err| {
                            format!(
                                "failed to build executor calls for {:?} step {}: {tx_err}",
                                plan.kind, steps
                            )
                        })?;
                        tracing::info!(
                            step = steps,
                            block = submit_block,
                            tx_mode = "strict_fallback",
                            group_kind = ?plan.kind,
                            action_indices = ?plan.action_indices,
                            calls = calls.len(),
                            quote_bounds = ?quote_bounds,
                            token_bounds = ?token_bounds,
                            end_to_end_planning_ms = planning_started_at.elapsed().as_millis(),
                            "prepared strict subgroup transaction"
                        );
                        (ExecutionMode::Strict, vec![plan], calls)
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    step = steps,
                    error = %err,
                    "packed program compilation failed; falling back to strict subgroup submission"
                );
                let plan = plans
                    .first()
                    .cloned()
                    .ok_or_else(|| "strict execution expected one first group".to_string())?;
                let quote_bounds = derive_batch_quote_bounds(
                    &plan,
                    submit_block,
                    config.execution_max_stale_blocks,
                )
                .map_err(|bound_err| {
                    format!(
                        "stale or invalid bounds for group {:?} at step {}: {bound_err}",
                        plan.kind, steps
                    )
                })?;
                let token_bounds = quote_bounds
                    .map(|bounds| bounds.to_token_bounds(SUSD_DECIMALS))
                    .transpose()?;
                let calls = build_trade_executor_calls(
                    resolved.executor,
                    &live_decision.actions,
                    &plan,
                    token_bounds,
                )
                .map_err(|tx_err| {
                    format!(
                        "failed to build executor calls for {:?} step {}: {tx_err}",
                        plan.kind, steps
                    )
                })?;
                tracing::info!(
                    step = steps,
                    block = submit_block,
                    tx_mode = "strict_fallback",
                    group_kind = ?plan.kind,
                    action_indices = ?plan.action_indices,
                    calls = calls.len(),
                    quote_bounds = ?quote_bounds,
                    token_bounds = ?token_bounds,
                    end_to_end_planning_ms = planning_started_at.elapsed().as_millis(),
                    "prepared strict subgroup transaction"
                );
                (ExecutionMode::Strict, vec![plan], calls)
            }
        };

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
                tx_mode = ?execution_mode,
                subgroups = chunk_plans.len(),
                "dry-run mode active (set EXECUTE_SUBMIT=1 to broadcast); stopping after first chunk"
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
            tx_mode = ?execution_mode,
            submitted_subgroups = chunk_plans.len(),
            "execution chunk submitted"
        );

        let latest_after_cycle = provider.get_block_number().await.unwrap_or(submit_block);
        log_skipped_blocks(
            steps,
            cycle_block,
            latest_after_cycle,
            &mut skipped_blocks_while_busy_total,
        );
        last_seen_block = last_seen_block.max(latest_after_cycle);
    }

    if let Some(handle) = native_audit_handle {
        if handle.is_finished() {
            log_finished_native_audit(handle, true).await;
        } else {
            tracing::info!("native audit still running at shutdown; dropping handle");
        }
    }

    if steps >= config.execution_max_steps {
        tracing::info!(
            max_steps = config.execution_max_steps,
            "execution step cap reached; stopping"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        native_audit_due, should_cache_snapshot_after_forecastflows_attempt,
        skipped_blocks_while_busy, snapshot_fingerprint,
    };
    use alloy::primitives::{U256, address};
    use deep_trading_bot::markets::MarketData;
    use deep_trading_bot::pools::Slot0Result;
    use std::collections::HashMap;

    static TEST_MARKET_A: MarketData = MarketData {
        name: "alpha",
        market_id: "market-alpha",
        outcome_token: "outcome-alpha",
        pool: None,
        quote_token: "quote",
    };

    static TEST_MARKET_B: MarketData = MarketData {
        name: "beta",
        market_id: "market-beta",
        outcome_token: "outcome-beta",
        pool: None,
        quote_token: "quote",
    };

    fn slot0(pool_id: alloy::primitives::Address, sqrt_price_x96: u64, tick: i32) -> Slot0Result {
        Slot0Result {
            pool_id,
            sqrt_price_x96: U256::from(sqrt_price_x96),
            tick,
            observation_index: 1,
            observation_cardinality: 2,
            observation_cardinality_next: 3,
            fee_protocol: 4,
            unlocked: true,
        }
    }

    #[test]
    fn skipped_blocks_while_busy_only_counts_dropped_backlog() {
        assert_eq!(skipped_blocks_while_busy(100, 100), 0);
        assert_eq!(skipped_blocks_while_busy(100, 101), 0);
        assert_eq!(skipped_blocks_while_busy(100, 103), 2);
    }

    #[test]
    fn native_audit_due_respects_interval_blocks() {
        assert!(!native_audit_due(100, 111, 12));
        assert!(native_audit_due(100, 112, 12));
        assert!(!native_audit_due(100, 120, 0));
    }

    #[test]
    fn snapshot_cache_retries_transient_forecastflows_failures() {
        assert!(!should_cache_snapshot_after_forecastflows_attempt(Some(
            "worker_timeout"
        )));
        assert!(!should_cache_snapshot_after_forecastflows_attempt(Some(
            "worker_cooldown"
        )));
        assert!(should_cache_snapshot_after_forecastflows_attempt(Some(
            "no_certified_candidate"
        )));
        assert!(should_cache_snapshot_after_forecastflows_attempt(Some(
            "translation_error"
        )));
        assert!(should_cache_snapshot_after_forecastflows_attempt(None));
    }

    #[test]
    fn snapshot_fingerprint_is_order_invariant() {
        let mut balances_one = HashMap::new();
        balances_one.insert("alpha", 2.0);
        balances_one.insert("beta", 3.0);

        let mut balances_two = HashMap::new();
        balances_two.insert("beta", 3.0);
        balances_two.insert("alpha", 2.0);

        let slot0_one = vec![
            (
                slot0(address!("0000000000000000000000000000000000000001"), 11, -7),
                &TEST_MARKET_A,
            ),
            (
                slot0(address!("0000000000000000000000000000000000000002"), 13, 9),
                &TEST_MARKET_B,
            ),
        ];
        let slot0_two = vec![
            (
                slot0(address!("0000000000000000000000000000000000000002"), 13, 9),
                &TEST_MARKET_B,
            ),
            (
                slot0(address!("0000000000000000000000000000000000000001"), 11, -7),
                &TEST_MARKET_A,
            ),
        ];

        assert_eq!(
            snapshot_fingerprint(5.0, &balances_one, &slot0_one),
            snapshot_fingerprint(5.0, &balances_two, &slot0_two)
        );
    }
}
