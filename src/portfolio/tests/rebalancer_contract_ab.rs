use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::super::rebalancer::{
    TestPhaseOrderVariant, arb_only_with_custom_predictions_for_test,
    benchmark_gas_assumptions_for_test, collect_mint_sell_preserve_candidates_for_test,
    compare_constant_l_runtime_vs_best_known_for_test,
    compare_constant_l_runtime_vs_k2_oracle_for_test,
    compare_constant_l_runtime_vs_medium_oracle_for_test,
    compare_constant_l_runtime_vs_oracle_for_test, compile_constant_l_mixed_selected_plan_for_test,
    compile_target_delta_actions_for_test, estimate_plan_economics_for_test,
    phase_order_exact_subset_choice_for_test, rebalance_with_custom_predictions_and_stats_for_test,
    rebalance_with_custom_predictions_arb_first_for_test,
    rebalance_with_custom_predictions_arb_last_for_test,
    rebalance_with_custom_predictions_arb_primed_family_for_test,
    rebalance_with_custom_predictions_exact_no_arb_for_test,
    rebalance_with_custom_predictions_exact_no_arb_with_distilled_proposals_for_test,
    rebalance_with_custom_predictions_exact_no_arb_with_explicit_choice_for_test,
    rebalance_with_custom_predictions_exact_no_arb_with_explicit_preserve_universe_for_test,
    rebalance_with_custom_predictions_exact_no_arb_with_search_config_for_test,
    rebalance_with_custom_predictions_for_test,
    rebalance_with_custom_predictions_operator_only_for_test,
    rebalance_with_custom_predictions_plain_family_for_test,
    rebalance_with_custom_predictions_rebalance_only_for_test,
    rebalance_with_custom_predictions_rebalance_strict_no_arb_for_test,
    rebalance_with_custom_predictions_selected_plan_for_test,
    rebalance_with_custom_predictions_selected_plan_with_pricing_for_test,
    rebalance_with_custom_predictions_staged_reference_for_test, staged_reference_choice_for_test,
    staged_teacher_snapshot_for_test,
};
use super::fixtures::mock_slot0_market_with_orientation_liquidity_and_ticks;
use super::replay_actions_to_state;
use crate::execution::bounds::{
    ConservativeExecutionConfig, build_group_plans_for_gas_replay_with_market_context,
};
use crate::execution::gas::{
    build_unsigned_contract_call_tx_bytes, build_unsigned_group_plan_batch_execute_tx_bytes,
    default_gas_assumptions_with_optimism_l1_fee, estimate_l1_data_fee_susd_for_tx_bytes_len,
    fetch_exact_optimism_l1_fee_wei_for_tx_data, fetch_live_optimism_fee_inputs, wei_to_susd,
};
use crate::execution::runtime::{
    DEFAULT_EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK, DEFAULT_EXECUTION_QUOTE_LATENCY_BLOCKS,
};
use crate::pools::{
    normalize_market_name, prediction_map, prediction_to_sqrt_price_x96,
    sqrt_price_x96_to_price_outcome,
};
use crate::portfolio::Action;
use crate::portfolio::core::sim::PoolSim;
use alloy::hex;
use alloy::primitives::{Address, Bytes, U256};

#[derive(Debug, Deserialize)]
struct BenchmarkCaseRow {
    predictions_wad: Vec<u128>,
    is_token1: Vec<bool>,
    starting_prices_wad: Vec<u128>,
    liquidity: Vec<u128>,
    ticks: Vec<Vec<i32>>,
    initial_holdings_wad: Vec<u128>,
    initial_cash_budget_wad: u128,
    uniform_count: usize,
    uniform_price_bps: u64,
    uniform_liquidity: u128,
    uniform_tick_lo: i32,
    uniform_tick_hi: i32,
    fee_tier: u64,
    direct_only_reference: bool,
}

#[derive(Debug)]
struct BenchmarkCase {
    case_id: String,
    predictions_wad: Vec<u128>,
    is_token1: Vec<bool>,
    starting_prices_wad: Vec<u128>,
    liquidity: Vec<u128>,
    ticks: Vec<Vec<i32>>,
    initial_holdings_wad: Vec<u128>,
    initial_cash_budget_wad: u128,
    uniform_count: usize,
    uniform_price_bps: u64,
    uniform_liquidity: u128,
    uniform_tick_lo: i32,
    uniform_tick_hi: i32,
    fee_tier: u64,
    direct_only_reference: bool,
}

#[derive(Debug, Deserialize)]
struct BenchmarkExpectedRow {
    offchain_direct_ev_wei: u128,
    offchain_mixed_ev_wei: u128,
    offchain_full_rebalance_only_ev_wei: u128,
    offchain_action_count: u64,
}

#[derive(Debug, Serialize)]
struct LiveBenchmarkReport {
    case_id: String,
    predictions_wad: Vec<u128>,
    is_token1: Vec<bool>,
    starting_prices_wad: Vec<u128>,
    liquidity: Vec<u128>,
    ticks: Vec<Vec<i32>>,
    initial_holdings_wad: Vec<u128>,
    initial_cash_budget_wad: u128,
    fee_tier: u64,
    offchain_direct_ev_wei: u128,
    offchain_mixed_ev_wei: u128,
    offchain_full_rebalance_only_ev_wei: u128,
    offchain_action_count: u64,
}

#[derive(Debug, Serialize)]
struct GasAblationRow {
    case_id: String,
    layer: String,
    raw_ev_wei: String,
    raw_ev_susd: f64,
    action_count: usize,
    group_count: usize,
    skipped_group_count: usize,
    runtime_millis: u128,
    total_fee_susd: f64,
    net_ev_susd: f64,
    incremental_raw_ev_susd: f64,
    incremental_net_ev_susd: f64,
}

#[derive(Debug, Clone)]
struct OnchainBenchmarkCallArtifact {
    target: Address,
    calldata: Vec<u8>,
    gas_units: u64,
    raw_ev_wei: u128,
}

#[derive(Debug, Clone)]
struct OnchainBenchmarkCaseArtifact {
    exact: OnchainBenchmarkCallArtifact,
    mixed: OnchainBenchmarkCallArtifact,
}

#[derive(Debug, Serialize)]
struct SharedSnapshotSolverRow {
    case_id: String,
    flavor: String,
    scenario_family: String,
    outcome_count: usize,
    profitable_count: usize,
    profitable_count_bucket: String,
    budget_regime: String,
    fee_regime: String,
    tick_regime: String,
    raw_ev_wei: String,
    raw_ev_susd: f64,
    total_fee_susd: f64,
    net_ev_susd: f64,
    action_count: usize,
    group_or_tx_count: usize,
    runtime_millis: Option<u128>,
    fee_source: String,
    family: Option<String>,
    compiler_variant: Option<String>,
    selected_common_shift: Option<f64>,
    selected_mixed_lambda: Option<f64>,
    selected_active_set_size: Option<usize>,
    selected_stage_count: Option<usize>,
    selected_stage1_budget_fraction: Option<f64>,
    k1_teacher_source: Option<String>,
    runtime_k1_gap_net_ev: Option<f64>,
    runtime_k1_gap_raw_ev: Option<f64>,
    oracle_best_is_direct_prefix: Option<bool>,
    oracle_best_active_set_size: Option<usize>,
    direct_buy_count: Option<usize>,
    direct_sell_count: Option<usize>,
    mint_count: Option<usize>,
    merge_count: Option<usize>,
    total_calldata_bytes: Option<usize>,
}

type Slot0Case = Vec<(
    crate::pools::Slot0Result,
    &'static crate::markets::MarketData,
)>;

const BENCHMARK_OP_CHAIN_ID: u64 = 10;
const BENCHMARK_OP_GAS_PRICE_WEI: u128 = 1_002_325;
const BENCHMARK_OP_ETH_USD: f64 = 3000.0;
const BENCHMARK_OP_GAS_PRICE_ETH: f64 = BENCHMARK_OP_GAS_PRICE_WEI as f64 / 1e18;

#[derive(Debug, Clone)]
struct BuiltCase {
    slot0_results: Slot0Case,
    balances_view: HashMap<&'static str, f64>,
    predictions: HashMap<String, f64>,
    cash_budget: f64,
}

#[derive(Debug, Clone, Copy)]
struct CaseRowMetadata {
    scenario_family: &'static str,
    outcome_count: usize,
    profitable_count: usize,
    profitable_count_bucket: &'static str,
    budget_regime: &'static str,
    fee_regime: &'static str,
    tick_regime: &'static str,
}

#[derive(Debug, Clone)]
struct StressCaseDefinition {
    case_id: String,
    built: BuiltCase,
    force_mint_available: bool,
    metadata: CaseRowMetadata,
}

#[derive(Debug, Default, Clone)]
struct K1TeacherRowDiagnostics {
    k1_teacher_source: Option<String>,
    runtime_k1_gap_net_ev: Option<f64>,
    runtime_k1_gap_raw_ev: Option<f64>,
    oracle_best_is_direct_prefix: Option<bool>,
    oracle_best_active_set_size: Option<usize>,
}

fn k1_teacher_row_diagnostics_for_built_case(built: &BuiltCase) -> K1TeacherRowDiagnostics {
    if let Some(comparison) = compare_constant_l_runtime_vs_medium_oracle_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        None,
    ) {
        return K1TeacherRowDiagnostics {
            k1_teacher_source: Some("oracle".to_string()),
            runtime_k1_gap_net_ev: comparison.runtime_k1_gap_net_ev,
            runtime_k1_gap_raw_ev: comparison.runtime_k1_gap_raw_ev,
            oracle_best_is_direct_prefix: Some(comparison.oracle_best_is_direct_prefix),
            oracle_best_active_set_size: comparison.oracle_best_active_set_size,
        };
    }

    if let Some(comparison) = compare_constant_l_runtime_vs_best_known_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        None,
    ) {
        return K1TeacherRowDiagnostics {
            k1_teacher_source: Some("best_known".to_string()),
            runtime_k1_gap_net_ev: comparison.runtime_k1_gap_net_ev,
            runtime_k1_gap_raw_ev: comparison.runtime_k1_gap_raw_ev,
            oracle_best_is_direct_prefix: None,
            oracle_best_active_set_size: None,
        };
    }

    K1TeacherRowDiagnostics::default()
}

fn profitable_count_bucket(count: usize) -> &'static str {
    match count {
        0 => "0",
        1..=4 => "1-4",
        5..=8 => "5-8",
        9..=13 => "9-13",
        14..=16 => "14-16",
        17..=20 => "17-20",
        21..=24 => "21-24",
        25..=32 => "25-32",
        _ => "33+",
    }
}

fn budget_regime_for_cash_budget(cash_budget: f64) -> &'static str {
    if cash_budget <= 10.0 + 1e-12 {
        "tiny"
    } else if cash_budget <= 40.0 + 1e-12 {
        "medium"
    } else {
        "full"
    }
}

fn fee_regime_label(multiplier: f64) -> &'static str {
    if (multiplier - 0.5).abs() <= 1e-12 {
        "0.5x"
    } else if (multiplier - 2.0).abs() <= 1e-12 {
        "2.0x"
    } else {
        "1.0x"
    }
}

fn direct_gross_profitable_outcome_count(built: &BuiltCase) -> usize {
    built
        .slot0_results
        .iter()
        .filter(|(slot0, market)| {
            let pool = match market.pool.as_ref() {
                Some(pool) => pool,
                None => return false,
            };
            let is_token1 = pool.token1.eq_ignore_ascii_case(market.outcome_token);
            let Some(price_wad) = sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1)
                .and_then(|value| u128::try_from(value).ok())
            else {
                return false;
            };
            let price = wad_to_f64(price_wad);
            let pred = built
                .predictions
                .get(&normalize_market_name(market.name))
                .copied()
                .unwrap_or(0.0);
            pred > price + 1e-12
        })
        .count()
}

fn fixture_scenario_family(case_id: &str) -> &'static str {
    match case_id {
        "mixed_route_favorable_synthetic_case" => "synthetic_adversarial",
        _ => "release_fixture",
    }
}

fn fixture_tick_regime(case: &BenchmarkCase) -> &'static str {
    if case.case_id.contains("multitick") {
        "crossing_light"
    } else {
        "single_tick"
    }
}

fn case_row_metadata(
    built: &BuiltCase,
    scenario_family: &'static str,
    budget_regime: &'static str,
    fee_regime: &'static str,
    tick_regime: &'static str,
) -> CaseRowMetadata {
    let profitable_count = direct_gross_profitable_outcome_count(built);
    CaseRowMetadata {
        scenario_family,
        outcome_count: built.slot0_results.len(),
        profitable_count,
        profitable_count_bucket: profitable_count_bucket(profitable_count),
        budget_regime,
        fee_regime,
        tick_regime,
    }
}

fn fixture_case_row_metadata(
    case: &BenchmarkCase,
    built: &BuiltCase,
    fee_multiplier: f64,
) -> CaseRowMetadata {
    case_row_metadata(
        built,
        fixture_scenario_family(&case.case_id),
        budget_regime_for_cash_budget(built.cash_budget),
        fee_regime_label(fee_multiplier),
        fixture_tick_regime(case),
    )
}

fn scaled_benchmark_gas_assumptions(multiplier: f64) -> crate::execution::gas::GasAssumptions {
    let mut assumptions = benchmark_gas_assumptions_for_test();
    assumptions.l1_fee_per_byte_wei *= multiplier;
    assumptions.l1_data_fee_floor_susd *= multiplier;
    assumptions
}

fn benchmark_fee_inputs_with_multiplier(
    multiplier: f64,
) -> crate::execution::gas::LiveOptimismFeeInputs {
    crate::execution::gas::LiveOptimismFeeInputs {
        sender_nonce: 0,
        chain_id: BENCHMARK_OP_CHAIN_ID,
        gas_price_wei: (BENCHMARK_OP_GAS_PRICE_WEI as f64 * multiplier).round() as u128,
    }
}

fn modeled_total_fee_susd_for_onchain_artifact(
    artifact: &OnchainBenchmarkCallArtifact,
    gas_assumptions: &crate::execution::gas::GasAssumptions,
    fee_inputs: crate::execution::gas::LiveOptimismFeeInputs,
    eth_usd: f64,
) -> f64 {
    let unsigned_tx_data = build_unsigned_contract_call_tx_bytes(
        artifact.target,
        artifact.calldata.clone().into(),
        fee_inputs,
        artifact.gas_units,
    )
    .expect("unsigned tx bytes should build for benchmark call");
    let l1_fee_susd =
        modeled_l1_fee_susd_for_tx_bytes(gas_assumptions, unsigned_tx_data.len(), eth_usd);
    let l2_fee_susd =
        (artifact.gas_units as f64 * fee_inputs.gas_price_wei as f64) * eth_usd / 1e18;
    l1_fee_susd + l2_fee_susd
}

fn stress_case(
    case_id: String,
    built: BuiltCase,
    scenario_family: &'static str,
    budget_regime: &'static str,
    tick_regime: &'static str,
) -> StressCaseDefinition {
    let metadata = case_row_metadata(&built, scenario_family, budget_regime, "1.0x", tick_regime);
    StressCaseDefinition {
        case_id,
        built,
        force_mint_available: true,
        metadata,
    }
}

fn solver_layer_actions_for_case(
    built: &BuiltCase,
    force_mint_available: bool,
) -> Vec<(&'static str, Vec<super::Action>)> {
    vec![
        (
            "r_exact_baseline_k4",
            rebalance_with_custom_predictions_exact_no_arb_with_search_config_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
                4,
                false,
            ),
        ),
        (
            "distilled_proposals",
            rebalance_with_custom_predictions_exact_no_arb_with_distilled_proposals_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
            ),
        ),
        (
            "operator_only",
            rebalance_with_custom_predictions_operator_only_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
            )
            .0,
        ),
        (
            "staged_fallback",
            rebalance_with_custom_predictions_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
            ),
        ),
    ]
}

async fn total_exact_gas_susd_for_actions(
    built: &BuiltCase,
    actions: &[super::Action],
    gas_assumptions: &crate::execution::gas::GasAssumptions,
    fee_inputs: crate::execution::gas::LiveOptimismFeeInputs,
    executor: Address,
    eth_usd: f64,
) -> (usize, usize, f64) {
    let replay = build_group_plans_for_gas_replay_with_market_context(
        actions,
        &built.slot0_results,
        ConservativeExecutionConfig {
            quote_latency_blocks: DEFAULT_EXECUTION_QUOTE_LATENCY_BLOCKS,
            adverse_move_bps_per_block: DEFAULT_EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK,
        },
        gas_assumptions,
        fee_inputs.gas_price_wei as f64 / 1e18,
        eth_usd,
    )
    .expect("gas replay planning should succeed for benchmark gas ablation");

    let mut total_fee_susd = 0.0;
    for plan in &replay.plans {
        let unsigned_tx_data =
            build_unsigned_group_plan_batch_execute_tx_bytes(executor, actions, plan, fee_inputs)
                .expect("unsigned tx bytes should be buildable for benchmark gas ablation");
        let l1_fee_susd = gas_assumptions.l1_data_fee_floor_susd.max(
            (unsigned_tx_data.len() as f64 * gas_assumptions.l1_fee_per_byte_wei) * eth_usd / 1e18,
        );
        let l2_fee_susd =
            (plan.l2_gas_units as f64 * fee_inputs.gas_price_wei as f64) * eth_usd / 1e18;
        total_fee_susd += l1_fee_susd + l2_fee_susd;
    }

    (
        replay.plans.len(),
        replay.skipped_groups.len(),
        total_fee_susd,
    )
}

async fn total_exact_fee_susd_for_actions_with_exact_l1(
    rpc_url: &str,
    built: &BuiltCase,
    actions: &[super::Action],
    gas_assumptions: &crate::execution::gas::GasAssumptions,
    fee_inputs: crate::execution::gas::LiveOptimismFeeInputs,
    executor: Address,
    eth_usd: f64,
) -> (usize, usize, f64) {
    if actions.is_empty() {
        return (0, 0, 0.0);
    }

    let replay = build_group_plans_for_gas_replay_with_market_context(
        actions,
        &built.slot0_results,
        ConservativeExecutionConfig {
            quote_latency_blocks: DEFAULT_EXECUTION_QUOTE_LATENCY_BLOCKS,
            adverse_move_bps_per_block: DEFAULT_EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK,
        },
        gas_assumptions,
        fee_inputs.gas_price_wei as f64 / 1e18,
        eth_usd,
    )
    .expect("gas replay planning should succeed for exact benchmark pricing");

    let mut total_fee_susd = 0.0;
    for plan in &replay.plans {
        let unsigned_tx_data =
            build_unsigned_group_plan_batch_execute_tx_bytes(executor, actions, plan, fee_inputs)
                .expect("unsigned tx bytes should be buildable for exact benchmark pricing");
        let l1_fee_wei = fetch_exact_l1_fee_wei_with_retry(rpc_url, unsigned_tx_data.clone()).await;
        let l1_fee_susd = wei_to_susd(l1_fee_wei, eth_usd);
        let l2_fee_susd =
            (plan.l2_gas_units as f64 * fee_inputs.gas_price_wei as f64) * eth_usd / 1e18;
        total_fee_susd += l1_fee_susd + l2_fee_susd;
    }

    (
        replay.plans.len(),
        replay.skipped_groups.len(),
        total_fee_susd,
    )
}

fn modeled_l1_fee_susd_for_tx_bytes(
    gas_assumptions: &crate::execution::gas::GasAssumptions,
    tx_data_len: usize,
    eth_usd: f64,
) -> f64 {
    estimate_l1_data_fee_susd_for_tx_bytes_len(gas_assumptions, tx_data_len, eth_usd)
}

async fn fetch_exact_l1_fee_wei_with_retry(rpc_url: &str, tx_data: Bytes) -> U256 {
    let mut last_error = None;
    for attempt in 0..3 {
        match fetch_exact_optimism_l1_fee_wei_for_tx_data(rpc_url, tx_data.clone()).await {
            Ok(value) => return value,
            Err(err) => {
                last_error = Some(err);
                if attempt < 2 {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        250 * (attempt as u64 + 1),
                    ))
                    .await;
                }
            }
        }
    }
    panic!(
        "exact getL1Fee should succeed after retries: {}",
        last_error
            .map(|err| err.to_string())
            .unwrap_or_else(|| "unknown error".to_string())
    );
}

fn onchain_call_report_path() -> String {
    format!(
        "{}/test/fixtures/rebalancer_ab_onchain_call_report.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn stress_onchain_call_report_path() -> String {
    format!(
        "{}/test/fixtures/rebalancer_ab_stress_onchain_call_report.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn parse_onchain_benchmark_call_artifact(
    value: &serde_json::Value,
) -> OnchainBenchmarkCallArtifact {
    let target = value
        .get("target")
        .and_then(serde_json::Value::as_str)
        .and_then(|raw| Address::from_str(raw).ok())
        .expect("artifact target should be a valid address");
    let calldata_hex = value
        .get("calldata")
        .and_then(serde_json::Value::as_str)
        .expect("artifact calldata should exist");
    let calldata = hex::decode(calldata_hex.trim_start_matches("0x"))
        .expect("artifact calldata should decode from hex");
    let gas_units = value
        .get("gas_units")
        .and_then(serde_json::Value::as_u64)
        .expect("artifact gas_units should exist");
    let raw_ev_wei = value
        .get("raw_ev_wei")
        .and_then(serde_json::Value::as_str)
        .and_then(|raw| raw.parse::<u128>().ok())
        .expect("artifact raw_ev_wei should exist");

    OnchainBenchmarkCallArtifact {
        target,
        calldata,
        gas_units,
        raw_ev_wei,
    }
}

fn load_onchain_call_artifacts_from_path(
    path: &str,
) -> HashMap<String, OnchainBenchmarkCaseArtifact> {
    let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read on-chain call artifact at {}: {}. Run `forge test --match-test test_write_rebalancer_ab_onchain_call_report -vv` first.",
            path, err
        )
    });
    let root: serde_json::Value =
        serde_json::from_str(&raw).expect("on-chain call artifact json should decode");
    let object = root
        .as_object()
        .expect("on-chain call artifact root should be an object");

    object
        .iter()
        .map(|(case_id, value)| {
            let case_value = if let Some(case_object) = value.as_object() {
                serde_json::Value::Object(case_object.clone())
            } else if let Some(case_json) = value.as_str() {
                serde_json::from_str::<serde_json::Value>(case_json)
                    .expect("nested case artifact json should decode")
            } else {
                panic!("unexpected on-chain case artifact shape for {}", case_id);
            };
            let exact_value = case_value
                .get("exact")
                .cloned()
                .expect("case artifact should contain exact");
            let mixed_value = case_value
                .get("mixed")
                .cloned()
                .expect("case artifact should contain mixed");
            let exact = if exact_value.is_string() {
                parse_onchain_benchmark_call_artifact(
                    &serde_json::from_str::<serde_json::Value>(
                        exact_value
                            .as_str()
                            .expect("exact nested artifact should be string"),
                    )
                    .expect("exact nested artifact json should decode"),
                )
            } else {
                parse_onchain_benchmark_call_artifact(&exact_value)
            };
            let mixed = if mixed_value.is_string() {
                parse_onchain_benchmark_call_artifact(
                    &serde_json::from_str::<serde_json::Value>(
                        mixed_value
                            .as_str()
                            .expect("mixed nested artifact should be string"),
                    )
                    .expect("mixed nested artifact json should decode"),
                )
            } else {
                parse_onchain_benchmark_call_artifact(&mixed_value)
            };
            (
                case_id.clone(),
                OnchainBenchmarkCaseArtifact { exact, mixed },
            )
        })
        .collect()
}

fn load_onchain_call_artifacts() -> HashMap<String, OnchainBenchmarkCaseArtifact> {
    let path = onchain_call_report_path();
    load_onchain_call_artifacts_from_path(&path)
}

fn load_stress_onchain_call_artifacts() -> HashMap<String, OnchainBenchmarkCaseArtifact> {
    let path = stress_onchain_call_report_path();
    let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read stress on-chain call artifact at {}: {}. Run `forge test --match-test test_write_rebalancer_ab_stress_onchain_call_report -vv` first.",
            path, err
        )
    });
    let root: serde_json::Value =
        serde_json::from_str(&raw).expect("stress on-chain call artifact json should decode");
    let object = root
        .as_object()
        .expect("stress on-chain call artifact root should be an object");
    if object.is_empty() {
        panic!(
            "stress on-chain call artifact at {} should contain at least one case",
            path
        );
    }
    load_onchain_call_artifacts_from_path(&path)
}

#[derive(Debug, Clone)]
struct ExplicitCaseSpec {
    case_id: String,
    predictions_wad: Vec<u128>,
    is_token1: Vec<bool>,
    starting_prices_wad: Vec<u128>,
    initial_holdings_wad: Vec<u128>,
    liquidity: Vec<u128>,
    ticks: Vec<Vec<i32>>,
    initial_cash_budget_wad: u128,
}

#[derive(Debug, Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_range_u64(&mut self, start_inclusive: u64, end_exclusive: u64) -> u64 {
        assert!(start_inclusive < end_exclusive, "invalid range");
        let span = end_exclusive - start_inclusive;
        start_inclusive + (self.next_u64() % span)
    }
}

fn leak_string(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn ev_to_wei(ev: f64) -> u128 {
    assert!(ev.is_finite(), "EV must be finite");
    assert!(ev >= 0.0, "EV must be non-negative");
    (ev * 1e18).round() as u128
}

fn wad_to_f64(value: u128) -> f64 {
    value as f64 / 1e18
}

fn f64_to_wad(value: f64) -> u128 {
    assert!(value.is_finite(), "WAD value must be finite");
    assert!(value >= 0.0, "WAD value must be non-negative");
    (value * 1e18).round() as u128
}

fn live_report_path() -> String {
    format!(
        "{}/test/fixtures/rebalancer_ab_live_l1_snapshot_report.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fallback_single_tick_geometry(
    market: &'static crate::markets::MarketData,
) -> Option<(u128, i32, i32)> {
    let pool = market.pool.as_ref()?;
    let liquidity = pool.liquidity.parse::<u128>().ok()?;
    let tick_lo = pool.ticks.iter().map(|tick| tick.tick_idx).min()?;
    let tick_hi = pool.ticks.iter().map(|tick| tick.tick_idx).max()?;
    (liquidity > 0 && tick_lo < tick_hi).then_some((liquidity, tick_lo, tick_hi))
}

fn current_single_tick_geometry(
    slot0: &crate::pools::Slot0Result,
    market: &'static crate::markets::MarketData,
) -> Option<(u128, i32, i32, bool)> {
    let pool = market.pool.as_ref()?;
    let mut tick_lo = None;
    let mut tick_hi = None;
    let mut active_liquidity = 0i128;

    for tick in pool.ticks {
        if tick.tick_idx <= slot0.tick {
            active_liquidity = active_liquidity.checked_add(tick.liquidity_net)?;
            if tick_lo.is_none_or(|best| tick.tick_idx > best) {
                tick_lo = Some(tick.tick_idx);
            }
        } else if tick_hi.is_none_or(|best| tick.tick_idx < best) {
            tick_hi = Some(tick.tick_idx);
        }
    }

    if let (Some(lo), Some(hi)) = (tick_lo, tick_hi)
        && active_liquidity > 0
    {
        return Some((active_liquidity as u128, lo, hi, false));
    }

    fallback_single_tick_geometry(market).map(|(liquidity, lo, hi)| (liquidity, lo, hi, true))
}

fn build_explicit_case_from_spec(spec: &ExplicitCaseSpec) -> BuiltCase {
    let prices: Vec<f64> = spec
        .starting_prices_wad
        .iter()
        .copied()
        .map(wad_to_f64)
        .collect();
    let preds: Vec<f64> = spec
        .predictions_wad
        .iter()
        .copied()
        .map(wad_to_f64)
        .collect();
    let holdings: Vec<f64> = spec
        .initial_holdings_wad
        .iter()
        .copied()
        .map(wad_to_f64)
        .collect();
    build_explicit_case(
        &spec.case_id,
        &prices,
        &preds,
        &spec.is_token1,
        &holdings,
        &spec.liquidity,
        &spec.ticks,
        wad_to_f64(spec.initial_cash_budget_wad),
    )
}

fn build_explicit_case(
    case_id: &str,
    prices: &[f64],
    preds: &[f64],
    is_token1: &[bool],
    holdings: &[f64],
    liquidities: &[u128],
    ticks: &[Vec<i32>],
    cash_budget: f64,
) -> BuiltCase {
    assert_eq!(
        prices.len(),
        preds.len(),
        "price/prediction length mismatch"
    );
    assert_eq!(
        prices.len(),
        holdings.len(),
        "price/holding length mismatch"
    );
    assert_eq!(
        prices.len(),
        is_token1.len(),
        "price/orientation length mismatch"
    );
    assert_eq!(
        prices.len(),
        liquidities.len(),
        "price/liquidity length mismatch"
    );
    assert_eq!(prices.len(), ticks.len(), "price/tick length mismatch");

    let mut slot0_results = Vec::with_capacity(prices.len());
    let mut balances = HashMap::new();
    let mut predictions = HashMap::new();

    for i in 0..prices.len() {
        let tick_pair = &ticks[i];
        assert_eq!(
            tick_pair.len(),
            2,
            "tick range must have exactly two entries"
        );
        let name = leak_string(format!("{case_id}_{i}"));
        let token = leak_string(format!("0x{:040x}", i + 1));
        let (slot0, market) = mock_slot0_market_with_orientation_liquidity_and_ticks(
            name,
            token,
            prices[i],
            liquidities[i],
            tick_pair[0],
            tick_pair[1],
            is_token1[i],
        );
        slot0_results.push((slot0, market));
        balances.insert(name, holdings[i]);
        predictions.insert(normalize_market_name(name), preds[i]);
    }

    let balances_view = balances.clone();
    BuiltCase {
        slot0_results,
        balances_view,
        predictions,
        cash_budget,
    }
}

fn benchmark_ev_wei(actions: &[super::Action], built: &BuiltCase) -> u128 {
    let (holdings, cash) = replay_actions_to_state(
        actions,
        &built.slot0_results,
        &built.balances_view,
        built.cash_budget,
    );
    let holdings_ev: f64 = holdings
        .iter()
        .map(|(name, units)| {
            let pred = built
                .predictions
                .get(&normalize_market_name(name))
                .copied()
                .unwrap_or(0.0);
            pred * units
        })
        .sum();
    ev_to_wei(cash + holdings_ev)
}

fn normalize_weights_to_wad(weights: &[u64]) -> Vec<u128> {
    assert!(!weights.is_empty(), "weights must be non-empty");
    let total_weight: u128 = weights.iter().map(|&w| u128::from(w)).sum();
    assert!(total_weight > 0, "weight sum must be positive");

    let mut values: Vec<u128> = weights
        .iter()
        .map(|&w| u128::from(w) * 1_000_000_000_000_000_000u128 / total_weight)
        .collect();
    let assigned: u128 = values.iter().sum();
    values[0] += 1_000_000_000_000_000_000u128 - assigned;
    values
}

fn current_result_for_built_case(
    built: &BuiltCase,
    force_mint_available: bool,
) -> (u128, u128, u128, u64) {
    let direct_actions = rebalance_with_custom_predictions_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        false,
    );
    let direct_ev = benchmark_ev_wei(&direct_actions, built);

    let mixed_actions = rebalance_with_custom_predictions_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let mixed_ev = benchmark_ev_wei(&mixed_actions, built);

    let full_rebalance_only_actions = rebalance_with_custom_predictions_rebalance_only_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let full_rebalance_only_ev = benchmark_ev_wei(&full_rebalance_only_actions, built);

    (
        direct_ev,
        mixed_ev,
        full_rebalance_only_ev,
        mixed_actions.len() as u64,
    )
}

const MAX_ARB_END_CYCLES: usize = 128;
const ARB_END_CYCLE_EV_REL_TOL: f64 = 1e-10;
const HELD_SANITY_MIN_BALANCE: f64 = 1e-6;
const HELD_PROFITABILITY_SPREAD_SANITY_TOL: f64 = 1e-3;

#[derive(Debug, Clone, Copy)]
struct HeldProfitabilityStats {
    held_count: usize,
    min_prof: f64,
    max_prof: f64,
    spread: f64,
}

fn apply_actions_to_case_state(
    actions: &[super::Action],
    slot0_results: &Slot0Case,
    balances: &HashMap<&'static str, f64>,
    cash_budget: f64,
    predictions: &HashMap<String, f64>,
) -> (Slot0Case, HashMap<&'static str, f64>, f64) {
    let (next_balances, next_cash) =
        replay_actions_to_state(actions, slot0_results, balances, cash_budget);
    let next_slot0 = replay_actions_to_case_market_state(actions, slot0_results, predictions);
    (next_slot0, next_balances, next_cash)
}

fn replay_actions_to_case_market_state(
    actions: &[super::Action],
    slot0_results: &Slot0Case,
    predictions: &HashMap<String, f64>,
) -> Slot0Case {
    let mut sims = Vec::with_capacity(slot0_results.len());
    let mut idx_by_market = HashMap::with_capacity(slot0_results.len());

    for (slot0, market) in slot0_results {
        let pred = predictions
            .get(&normalize_market_name(market.name))
            .copied()
            .unwrap_or(0.0);
        let sim =
            PoolSim::from_slot0(slot0, market, pred).expect("benchmark replay should build sims");
        idx_by_market.insert(market.name, sims.len());
        sims.push(sim);
    }

    for action in actions {
        match action {
            super::Action::Buy {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name)
                    && let Some((bought, _cost, new_price)) = sims[idx].buy_exact(*amount)
                    && bought > 0.0
                {
                    sims[idx].set_price(new_price);
                }
            }
            super::Action::Sell {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name)
                    && let Some((sold, _proceeds, new_price)) = sims[idx].sell_exact(*amount)
                    && sold > 0.0
                {
                    sims[idx].set_price(new_price);
                }
            }
            super::Action::Mint { .. } | super::Action::Merge { .. } => {}
        }
    }

    slot0_results
        .iter()
        .map(|(slot0, market)| {
            let mut next = slot0.clone();
            if let Some(&idx) = idx_by_market.get(market.name)
                && let Some(pool) = market.pool.as_ref()
            {
                let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
                let next_price = sims[idx].price().max(1e-12);
                next.sqrt_price_x96 = prediction_to_sqrt_price_x96(next_price, is_token1_outcome)
                    .unwrap_or(slot0.sqrt_price_x96);
            }
            (next, *market)
        })
        .collect()
}

fn state_ev(
    balances: &HashMap<&'static str, f64>,
    cash_budget: f64,
    predictions: &HashMap<String, f64>,
) -> f64 {
    let holdings_ev: f64 = balances
        .iter()
        .map(|(name, units)| {
            let pred = predictions
                .get(&normalize_market_name(name))
                .copied()
                .unwrap_or(0.0);
            pred * units
        })
        .sum();
    cash_budget + holdings_ev
}

fn held_profitability_stats_for_state(
    slot0_results: &Slot0Case,
    balances: &HashMap<&'static str, f64>,
    predictions: &HashMap<String, f64>,
) -> Option<HeldProfitabilityStats> {
    let mut held_count = 0usize;
    let mut min_prof = f64::INFINITY;
    let mut max_prof = f64::NEG_INFINITY;

    for (slot0, market) in slot0_results {
        let held = balances.get(market.name).copied().unwrap_or(0.0);
        if held <= HELD_SANITY_MIN_BALANCE {
            continue;
        }
        let Some(pool) = market.pool.as_ref() else {
            continue;
        };
        let is_token1 = pool.token1.eq_ignore_ascii_case(market.outcome_token);
        let Some(price_wad) = sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1)
            .and_then(|value| u128::try_from(value).ok())
        else {
            continue;
        };
        let price = wad_to_f64(price_wad);
        if !price.is_finite() || price <= 0.0 {
            continue;
        }
        let pred = predictions
            .get(&normalize_market_name(market.name))
            .copied()
            .unwrap_or(0.0);
        let prof = (pred - price) / price;
        if !prof.is_finite() {
            continue;
        }
        held_count += 1;
        min_prof = min_prof.min(prof);
        max_prof = max_prof.max(prof);
    }

    if held_count >= 2 && min_prof.is_finite() && max_prof.is_finite() {
        Some(HeldProfitabilityStats {
            held_count,
            min_prof,
            max_prof,
            spread: max_prof - min_prof,
        })
    } else {
        None
    }
}

fn arb_end_cyclic_result_for_built_case(
    built: &BuiltCase,
    force_mint_available: bool,
) -> (u128, u64, Option<HeldProfitabilityStats>) {
    let mut slot0_results = built.slot0_results.clone();
    let mut balances = built.balances_view.clone();
    let mut cash_budget = built.cash_budget;
    let mut total_actions = 0u64;
    let mut converged = false;

    for _ in 0..MAX_ARB_END_CYCLES {
        let ev_before_cycle = state_ev(&balances, cash_budget, &built.predictions);

        let rebalance_actions = rebalance_with_custom_predictions_rebalance_strict_no_arb_for_test(
            &balances,
            cash_budget,
            &slot0_results,
            &built.predictions,
            force_mint_available,
        );
        total_actions += rebalance_actions.len() as u64;
        if !rebalance_actions.is_empty() {
            (slot0_results, balances, cash_budget) = apply_actions_to_case_state(
                &rebalance_actions,
                &slot0_results,
                &balances,
                cash_budget,
                &built.predictions,
            );
        }

        let arb_actions = arb_only_with_custom_predictions_for_test(
            &balances,
            cash_budget,
            &slot0_results,
            &built.predictions,
            force_mint_available,
        );
        total_actions += arb_actions.len() as u64;
        if !arb_actions.is_empty() {
            (slot0_results, balances, cash_budget) = apply_actions_to_case_state(
                &arb_actions,
                &slot0_results,
                &balances,
                cash_budget,
                &built.predictions,
            );
        }

        if rebalance_actions.is_empty() && arb_actions.is_empty() {
            converged = true;
            break;
        }

        let ev_after_cycle = state_ev(&balances, cash_budget, &built.predictions);
        let ev_tol =
            ARB_END_CYCLE_EV_REL_TOL * (1.0 + ev_before_cycle.abs() + ev_after_cycle.abs());
        if ev_after_cycle <= ev_before_cycle + ev_tol {
            converged = true;
            break;
        }
    }

    assert!(
        converged,
        "arb-end cyclic experiment did not converge within {} cycles",
        MAX_ARB_END_CYCLES
    );

    let ev_wei = ev_to_wei(state_ev(&balances, cash_budget, &built.predictions));
    let held_stats =
        held_profitability_stats_for_state(&slot0_results, &balances, &built.predictions);
    (ev_wei, total_actions, held_stats)
}

fn held_profitability_stats_for_actions(
    actions: &[super::Action],
    built: &BuiltCase,
) -> Option<HeldProfitabilityStats> {
    let (balances, _cash) = replay_actions_to_state(
        actions,
        &built.slot0_results,
        &built.balances_view,
        built.cash_budget,
    );
    let slot0_results =
        replay_actions_to_case_market_state(actions, &built.slot0_results, &built.predictions);
    held_profitability_stats_for_state(&slot0_results, &balances, &built.predictions)
}

fn start_arb_result_for_built_case(
    built: &BuiltCase,
    force_mint_available: bool,
) -> (u128, u64, Option<HeldProfitabilityStats>) {
    let actions = rebalance_with_custom_predictions_arb_first_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let ev_wei = benchmark_ev_wei(&actions, built);
    let held_stats = held_profitability_stats_for_actions(&actions, built);
    (ev_wei, actions.len() as u64, held_stats)
}

fn sample_random_case(rng: &mut Lcg, case_idx: usize, outcomes: usize) -> ExplicitCaseSpec {
    const ONE_WAD: u128 = 1_000_000_000_000_000_000u128;
    assert!(outcomes >= 2, "sweep cases need at least two outcomes");

    let mut weights = vec![0_u64; outcomes];
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight = if i == 0 {
            if outcomes >= 32 {
                rng.next_range_u64(450, 1_300)
            } else {
                rng.next_range_u64(180, 420)
            }
        } else if outcomes >= 32 && i < 8 {
            rng.next_range_u64(140, 520)
        } else {
            rng.next_range_u64(90, 320)
        };
    }
    let predictions_wad = normalize_weights_to_wad(&weights);

    let mut starting_prices_wad = Vec::with_capacity(outcomes);
    let mut liquidity = Vec::with_capacity(outcomes);
    let mut ticks = Vec::with_capacity(outcomes);

    for (i, &pred) in predictions_wad.iter().enumerate() {
        let factor_bps = if i == 0 {
            rng.next_range_u64(5_500, 9_400)
        } else if i == 1 {
            rng.next_range_u64(9_200, 12_800)
        } else if outcomes >= 32 && i < 8 {
            rng.next_range_u64(9_800, 12_200)
        } else {
            rng.next_range_u64(8_500, 12_200)
        };
        let price = (pred * u128::from(factor_bps)) / 10_000;
        starting_prices_wad.push(price.max(ONE_WAD / 10_000));

        let liq = if i == 0 {
            if outcomes >= 32 {
                u128::from(rng.next_range_u64(120_000_000_000_000_000, 650_000_000_000_000_000))
            } else {
                u128::from(rng.next_range_u64(250_000_000_000_000_000, 1_200_000_000_000_000_000))
            }
        } else if outcomes >= 32 && i < 8 {
            u128::from(rng.next_range_u64(3_000_000_000_000_000_000, 9_000_000_000_000_000_000))
        } else {
            u128::from(rng.next_range_u64(1_000_000_000_000_000_000, 8_000_000_000_000_000_000))
        };
        liquidity.push(liq);
        ticks.push(vec![1, 92_108]);
    }

    let mut initial_holdings_wad = vec![0; outcomes];
    if outcomes >= 32 {
        let seed_positions = [1usize, 2, 7, 17, 31, 46, 63, outcomes - 1];
        for &idx in &seed_positions {
            if idx < outcomes {
                initial_holdings_wad[idx] = u128::from(rng.next_range_u64(1, 5)) * ONE_WAD;
            }
        }
    }

    ExplicitCaseSpec {
        case_id: format!("sweep_{outcomes}_case_{case_idx}"),
        predictions_wad,
        is_token1: vec![true; outcomes],
        starting_prices_wad,
        initial_holdings_wad,
        liquidity,
        ticks,
        initial_cash_budget_wad: if outcomes >= 32 {
            150 * ONE_WAD
        } else {
            100 * ONE_WAD
        },
    }
}

fn build_uniform_large_case(case: &BenchmarkCase) -> BuiltCase {
    let count = case.uniform_count;
    assert!(count > 0, "uniform_count must be positive");
    let pred = 1.0 / count as f64;
    let prices = vec![pred * (case.uniform_price_bps as f64) / 10_000.0; count];
    let preds = vec![pred; count];
    let is_token1 = vec![true; count];
    let holdings = vec![0.0; count];
    let liquidities = vec![case.uniform_liquidity; count];
    let ticks = vec![vec![case.uniform_tick_lo, case.uniform_tick_hi]; count];
    build_explicit_case(
        &case.case_id,
        &prices,
        &preds,
        &is_token1,
        &holdings,
        &liquidities,
        &ticks,
        wad_to_f64(case.initial_cash_budget_wad),
    )
}

fn build_case(case: &BenchmarkCase) -> BuiltCase {
    assert_eq!(
        case.fee_tier,
        crate::pools::FEE_PIPS as u64,
        "benchmark harness should match the off-chain hardcoded pool fee assumption"
    );
    assert!(
        case.is_token1.iter().all(|flag| *flag),
        "benchmark harness currently supports token1-oriented pools only"
    );

    if case.uniform_count > 0 {
        build_uniform_large_case(case)
    } else {
        let prices: Vec<f64> = case
            .starting_prices_wad
            .iter()
            .copied()
            .map(wad_to_f64)
            .collect();
        let preds: Vec<f64> = case
            .predictions_wad
            .iter()
            .copied()
            .map(wad_to_f64)
            .collect();
        let holdings: Vec<f64> = case
            .initial_holdings_wad
            .iter()
            .copied()
            .map(wad_to_f64)
            .collect();
        build_explicit_case(
            &case.case_id,
            &prices,
            &preds,
            &case.is_token1,
            &holdings,
            &case.liquidity,
            &case.ticks,
            wad_to_f64(case.initial_cash_budget_wad),
        )
    }
}

#[derive(Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_usize(&mut self, low: usize, high_inclusive: usize) -> usize {
        assert!(low <= high_inclusive, "invalid deterministic rng range");
        low + (self.next_u64() as usize % (high_inclusive - low + 1))
    }
}

fn build_random_small_case(seed: u64) -> BuiltCase {
    let mut rng = DeterministicRng::new(seed);
    let outcome_count = rng.next_usize(3, 7);
    let mut weights = Vec::with_capacity(outcome_count);
    for _ in 0..outcome_count {
        weights.push(1u64 + (rng.next_u64() % 200));
    }
    let predictions: Vec<f64> = normalize_weights_to_wad(&weights)
        .into_iter()
        .map(wad_to_f64)
        .collect();
    let prices: Vec<f64> = predictions
        .iter()
        .map(|pred| {
            let scale = 0.82 + 0.36 * rng.next_f64();
            (*pred * scale).clamp(0.0025, 0.95)
        })
        .collect();
    let holdings = vec![0.0; outcome_count];
    let is_token1 = vec![true; outcome_count];
    let liquidities: Vec<u128> = (0..outcome_count)
        .map(|_| {
            3_000_000_000_000_000_000_000u128
                + u128::from(rng.next_u64() % 2_000_000_000_000_000_000u64)
        })
        .collect();
    let ticks = vec![vec![-120_000, 120_000]; outcome_count];
    build_explicit_case(
        &format!("oracle_random_{seed}"),
        &prices,
        &predictions,
        &is_token1,
        &holdings,
        &liquidities,
        &ticks,
        100.0,
    )
}

fn build_random_medium_case(seed: u64) -> BuiltCase {
    let mut rng = DeterministicRng::new(seed ^ 0x9e3779b97f4a7c15);
    let outcome_count = 13;
    let mut weights = Vec::with_capacity(outcome_count);
    for _ in 0..outcome_count {
        weights.push(1u64 + (rng.next_u64() % 200));
    }
    let predictions: Vec<f64> = normalize_weights_to_wad(&weights)
        .into_iter()
        .map(wad_to_f64)
        .collect();
    let prices: Vec<f64> = predictions
        .iter()
        .map(|pred| {
            let scale = 0.80 + 0.40 * rng.next_f64();
            (*pred * scale).clamp(0.0015, 0.95)
        })
        .collect();
    let holdings = vec![0.0; outcome_count];
    let is_token1 = vec![true; outcome_count];
    let liquidities: Vec<u128> = (0..outcome_count)
        .map(|_| {
            2_500_000_000_000_000_000_000u128
                + u128::from(rng.next_u64() % 3_000_000_000_000_000_000u64)
        })
        .collect();
    let ticks = vec![vec![-120_000, 120_000]; outcome_count];
    build_explicit_case(
        &format!("oracle_medium_{seed}"),
        &prices,
        &predictions,
        &is_token1,
        &holdings,
        &liquidities,
        &ticks,
        100.0,
    )
}

fn build_boundary_profitable_case(
    profitable_count: usize,
    budget_scale: f64,
    tick_regime: &'static str,
) -> StressCaseDefinition {
    assert!(
        matches!(
            tick_regime,
            "single_tick" | "crossing_light" | "crossing_heavy"
        ),
        "unexpected tick regime"
    );
    let outcome_count = profitable_count + 2;
    let budget_regime = budget_regime_for_cash_budget(100.0 * budget_scale);
    let predictions: Vec<f64> = normalize_weights_to_wad(&vec![100u64; outcome_count])
        .into_iter()
        .map(wad_to_f64)
        .collect();
    let prices: Vec<f64> = predictions
        .iter()
        .enumerate()
        .map(|(idx, pred)| {
            if idx < profitable_count {
                pred * (0.76 + 0.03 * ((idx % 4) as f64))
            } else {
                pred * (1.03 + 0.02 * ((idx % 3) as f64))
            }
        })
        .collect();
    let liquidities: Vec<u128> = (0..outcome_count)
        .map(|idx| 3_500_000_000_000_000_000_000u128 + (idx as u128) * 50_000_000_000_000_000u128)
        .collect();
    let tick_span = match tick_regime {
        "single_tick" => 120_000,
        "crossing_light" => 40_000,
        "crossing_heavy" => 28_000,
        _ => unreachable!(),
    };
    let ticks = vec![vec![-tick_span, tick_span]; outcome_count];
    let case_id = format!("boundary_profitable_{profitable_count}_{budget_regime}_{tick_regime}");
    let built = build_explicit_case(
        &case_id,
        &prices,
        &predictions,
        &vec![true; outcome_count],
        &vec![0.0; outcome_count],
        &liquidities,
        &ticks,
        100.0 * budget_scale,
    );
    stress_case(
        case_id,
        built,
        "boundary_cutoff",
        budget_regime,
        tick_regime,
    )
}

fn build_complementary_inactive_sell_case() -> StressCaseDefinition {
    let case_id = "preserve_frontier_inactive_sell_case".to_string();
    let built = build_explicit_case(
        &case_id,
        &[0.13, 0.16, 0.34, 0.25, 0.12],
        &[0.27, 0.24, 0.18, 0.16, 0.15],
        &[true, true, true, true, true],
        &[0.0, 0.0, 7.0, 4.0, 0.0],
        &[
            2_400_000_000_000_000_000_000u128,
            2_300_000_000_000_000_000_000u128,
            3_100_000_000_000_000_000_000u128,
            2_700_000_000_000_000_000_000u128,
            2_500_000_000_000_000_000_000u128,
        ],
        &[
            vec![-120_000, 120_000],
            vec![-120_000, 120_000],
            vec![-120_000, 120_000],
            vec![-120_000, 120_000],
            vec![-120_000, 120_000],
        ],
        18.0,
    );
    stress_case(
        case_id,
        built,
        "synthetic_adversarial",
        "medium",
        "single_tick",
    )
}

fn build_mint_dominant_case() -> StressCaseDefinition {
    let case_id = "mint_dominant_case".to_string();
    let built = build_explicit_case(
        &case_id,
        &[0.84, 0.08, 0.08],
        &[0.36, 0.32, 0.32],
        &[true, true, true],
        &[0.0, 0.0, 0.0],
        &[
            2_800_000_000_000_000_000_000u128,
            2_800_000_000_000_000_000_000u128,
            2_800_000_000_000_000_000_000u128,
        ],
        &[
            vec![-180_000, 180_000],
            vec![-180_000, 180_000],
            vec![-180_000, 180_000],
        ],
        30.0,
    );
    stress_case(
        case_id,
        built,
        "synthetic_adversarial",
        "medium",
        "single_tick",
    )
}

fn build_tick_scope_cases() -> Vec<StressCaseDefinition> {
    ["crossing_light", "crossing_heavy"]
        .into_iter()
        .map(|tick_regime| {
            let boundary = build_boundary_profitable_case(8, 1.0, tick_regime);
            stress_case(
                boundary.case_id,
                boundary.built,
                "tick_scope",
                boundary.metadata.budget_regime,
                tick_regime,
            )
        })
        .collect()
}

fn shared_single_tick_onchain_stress_cases() -> Vec<StressCaseDefinition> {
    vec![
        build_complementary_inactive_sell_case(),
        build_mint_dominant_case(),
        build_boundary_profitable_case(16, 1.0, "single_tick"),
        build_boundary_profitable_case(20, 0.05, "single_tick"),
    ]
}

fn cases_from_fixture() -> Vec<BenchmarkCase> {
    let rows: HashMap<String, BenchmarkCaseRow> = serde_json::from_str(include_str!(
        "../../../test/fixtures/rebalancer_ab_cases.json"
    ))
    .expect("cases fixture should parse");
    let mut keys: Vec<String> = rows.keys().cloned().collect();
    keys.sort();
    keys.into_iter()
        .map(|case_id| {
            let row = rows
                .get(&case_id)
                .expect("sorted key should still exist in parsed case rows");
            BenchmarkCase {
                case_id,
                predictions_wad: row.predictions_wad.clone(),
                is_token1: row.is_token1.clone(),
                starting_prices_wad: row.starting_prices_wad.clone(),
                liquidity: row.liquidity.clone(),
                ticks: row.ticks.clone(),
                initial_holdings_wad: row.initial_holdings_wad.clone(),
                initial_cash_budget_wad: row.initial_cash_budget_wad,
                uniform_count: row.uniform_count,
                uniform_price_bps: row.uniform_price_bps,
                uniform_liquidity: row.uniform_liquidity,
                uniform_tick_lo: row.uniform_tick_lo,
                uniform_tick_hi: row.uniform_tick_hi,
                fee_tier: row.fee_tier,
                direct_only_reference: row.direct_only_reference,
            }
        })
        .collect()
}

fn expected_from_fixture() -> HashMap<String, BenchmarkExpectedRow> {
    serde_json::from_str(include_str!(
        "../../../test/fixtures/rebalancer_ab_expected.json"
    ))
    .expect("expected fixture should parse")
}

fn current_result_for_case(case: &BenchmarkCase) -> (u128, u128, u128, u64) {
    let built = build_case(case);
    current_result_for_built_case(&built, !case.direct_only_reference)
}

#[test]
fn benchmark_snapshot_matches_current_optimizer() {
    const HETEROGENEOUS_MIXED_WOBBLE_TOLERANCE_WEI: u128 = 65_536;
    let cases = cases_from_fixture();
    let expected = expected_from_fixture();
    let mut failures = Vec::new();

    for case in &cases {
        let expected_row = expected
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing snapshot row for {}", case.case_id));
        let (direct_ev, mixed_ev, full_rebalance_only_ev, action_count) =
            current_result_for_case(case);

        if direct_ev != expected_row.offchain_direct_ev_wei {
            failures.push(format!(
                "{} direct expected={} actual={}",
                case.case_id, expected_row.offchain_direct_ev_wei, direct_ev
            ));
        }
        let mixed_ev_matches_snapshot =
            if case.case_id == "heterogeneous_ninety_eight_outcome_l1_like_case" {
                mixed_ev.abs_diff(expected_row.offchain_mixed_ev_wei)
                    <= HETEROGENEOUS_MIXED_WOBBLE_TOLERANCE_WEI
            } else {
                mixed_ev == expected_row.offchain_mixed_ev_wei
            };
        if !mixed_ev_matches_snapshot {
            failures.push(format!(
                "{} mixed expected={} actual={}",
                case.case_id, expected_row.offchain_mixed_ev_wei, mixed_ev
            ));
        }
        if full_rebalance_only_ev != expected_row.offchain_full_rebalance_only_ev_wei {
            failures.push(format!(
                "{} full_rebalance_only expected={} actual={}",
                case.case_id,
                expected_row.offchain_full_rebalance_only_ev_wei,
                full_rebalance_only_ev
            ));
        }
        if action_count != expected_row.offchain_action_count {
            failures.push(format!(
                "{} actions expected={} actual={}",
                case.case_id, expected_row.offchain_action_count, action_count
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "benchmark snapshot mismatches:\n{}",
        failures.join("\n")
    );
}

#[test]
fn benchmark_ev_non_decreasing_vs_fixture() {
    const HETEROGENEOUS_MIXED_WOBBLE_TOLERANCE_WEI: u128 = 65_536;
    let cases = cases_from_fixture();
    let expected = expected_from_fixture();

    for case in &cases {
        let expected_row = expected
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing snapshot row for {}", case.case_id));
        let (direct_ev, mixed_ev, full_rebalance_only_ev, action_count) =
            current_result_for_case(case);

        assert!(
            direct_ev >= expected_row.offchain_direct_ev_wei,
            "{} direct regressed: expected_floor={} actual={}",
            case.case_id,
            expected_row.offchain_direct_ev_wei,
            direct_ev
        );
        let mixed_ev_non_regressive = mixed_ev >= expected_row.offchain_mixed_ev_wei
            || (case.case_id == "heterogeneous_ninety_eight_outcome_l1_like_case"
                && expected_row.offchain_mixed_ev_wei.abs_diff(mixed_ev)
                    <= HETEROGENEOUS_MIXED_WOBBLE_TOLERANCE_WEI);
        assert!(
            mixed_ev_non_regressive,
            "{} mixed regressed: expected_floor={} actual={}",
            case.case_id, expected_row.offchain_mixed_ev_wei, mixed_ev
        );
        assert!(
            full_rebalance_only_ev >= expected_row.offchain_full_rebalance_only_ev_wei,
            "{} full_rebalance_only regressed: expected_floor={} actual={}",
            case.case_id,
            expected_row.offchain_full_rebalance_only_ev_wei,
            full_rebalance_only_ev
        );
        let _ = action_count;
    }
}

#[test]
fn shared_snapshot_metadata_classifies_committed_fixtures() {
    let mut saw_crossing_light = false;
    let mut saw_release_fixture = false;

    for case in cases_from_fixture() {
        let built = build_case(&case);
        let metadata = fixture_case_row_metadata(&case, &built, 1.0);

        assert_eq!(metadata.outcome_count, built.slot0_results.len());
        assert_eq!(
            metadata.profitable_count_bucket,
            profitable_count_bucket(metadata.profitable_count)
        );
        assert_eq!(
            metadata.budget_regime,
            budget_regime_for_cash_budget(built.cash_budget)
        );
        if metadata.tick_regime == "crossing_light" {
            saw_crossing_light = true;
        }
        if metadata.scenario_family == "release_fixture" {
            saw_release_fixture = true;
        }
    }

    assert!(
        saw_crossing_light,
        "expected at least one crossing-light fixture"
    );
    assert!(saw_release_fixture, "expected at least one release fixture");
}

#[test]
fn offchain_default_net_ev_matches_or_beats_onchain_under_fee_sweeps_on_committed_cases() {
    const NET_EV_TOL: f64 = 1e-9;

    let onchain_artifacts = load_onchain_call_artifacts();
    for case in cases_from_fixture() {
        let built = build_case(&case);
        let force_mint_available = !case.direct_only_reference;
        let onchain = onchain_artifacts
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing on-chain artifact for {}", case.case_id));

        for fee_multiplier in [0.5, 1.0, 2.0] {
            let gas_assumptions = scaled_benchmark_gas_assumptions(fee_multiplier);
            let fee_inputs = benchmark_fee_inputs_with_multiplier(fee_multiplier);
            let metadata = fixture_case_row_metadata(&case, &built, fee_multiplier);
            let summary = rebalance_with_custom_predictions_selected_plan_with_pricing_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
                &gas_assumptions,
                BENCHMARK_OP_GAS_PRICE_ETH * fee_multiplier,
                BENCHMARK_OP_ETH_USD,
            );
            let offchain_net_ev = summary
                .estimated_net_ev
                .expect("off-chain benchmark plan should have modeled net EV under fee sweep");
            let onchain_exact_net_ev = wad_to_f64(onchain.exact.raw_ev_wei)
                - modeled_total_fee_susd_for_onchain_artifact(
                    &onchain.exact,
                    &gas_assumptions,
                    fee_inputs,
                    BENCHMARK_OP_ETH_USD,
                );
            let onchain_mixed_net_ev = wad_to_f64(onchain.mixed.raw_ev_wei)
                - modeled_total_fee_susd_for_onchain_artifact(
                    &onchain.mixed,
                    &gas_assumptions,
                    fee_inputs,
                    BENCHMARK_OP_ETH_USD,
                );

            assert_eq!(metadata.fee_regime, fee_regime_label(fee_multiplier));
            assert!(
                offchain_net_ev + NET_EV_TOL >= onchain_exact_net_ev,
                "{} {} off-chain net EV lost to on-chain exact: offchain={} onchain_exact={} summary={:?}",
                case.case_id,
                metadata.fee_regime,
                offchain_net_ev,
                onchain_exact_net_ev,
                summary
            );
            assert!(
                offchain_net_ev + NET_EV_TOL >= onchain_mixed_net_ev,
                "{} {} off-chain net EV lost to on-chain mixed: offchain={} onchain_mixed={} summary={:?}",
                case.case_id,
                metadata.fee_regime,
                offchain_net_ev,
                onchain_mixed_net_ev,
                summary
            );
        }
    }
}

#[test]
#[ignore = "heavy release validation; run explicitly with --release"]
fn offchain_default_net_ev_matches_or_beats_onchain_under_fee_sweeps_on_shared_single_tick_stress_cases()
 {
    const NET_EV_TOL: f64 = 1e-9;

    let onchain_artifacts = load_stress_onchain_call_artifacts();
    for stress in shared_single_tick_onchain_stress_cases() {
        let onchain = onchain_artifacts
            .get(&stress.case_id)
            .unwrap_or_else(|| panic!("missing stress on-chain artifact for {}", stress.case_id));

        for fee_multiplier in [0.5, 1.0, 2.0] {
            let gas_assumptions = scaled_benchmark_gas_assumptions(fee_multiplier);
            let fee_inputs = benchmark_fee_inputs_with_multiplier(fee_multiplier);
            let metadata = case_row_metadata(
                &stress.built,
                stress.metadata.scenario_family,
                stress.metadata.budget_regime,
                fee_regime_label(fee_multiplier),
                stress.metadata.tick_regime,
            );
            let summary = rebalance_with_custom_predictions_selected_plan_with_pricing_for_test(
                &stress.built.balances_view,
                stress.built.cash_budget,
                &stress.built.slot0_results,
                &stress.built.predictions,
                stress.force_mint_available,
                &gas_assumptions,
                BENCHMARK_OP_GAS_PRICE_ETH * fee_multiplier,
                BENCHMARK_OP_ETH_USD,
            );
            let offchain_net_ev = summary
                .estimated_net_ev
                .expect("off-chain stress plan should have modeled net EV under fee sweep");
            let onchain_exact_net_ev = wad_to_f64(onchain.exact.raw_ev_wei)
                - modeled_total_fee_susd_for_onchain_artifact(
                    &onchain.exact,
                    &gas_assumptions,
                    fee_inputs,
                    BENCHMARK_OP_ETH_USD,
                );
            let onchain_mixed_net_ev = wad_to_f64(onchain.mixed.raw_ev_wei)
                - modeled_total_fee_susd_for_onchain_artifact(
                    &onchain.mixed,
                    &gas_assumptions,
                    fee_inputs,
                    BENCHMARK_OP_ETH_USD,
                );

            assert_eq!(metadata.tick_regime, "single_tick");
            assert_eq!(metadata.fee_regime, fee_regime_label(fee_multiplier));
            assert_ne!(
                summary.fee_estimate_source, "unavailable",
                "{} {} should retain modeled fee coverage",
                stress.case_id, metadata.fee_regime
            );
            assert!(
                offchain_net_ev + NET_EV_TOL >= onchain_exact_net_ev,
                "{} {} off-chain net EV lost to on-chain exact: offchain={} onchain_exact={} summary={:?}",
                stress.case_id,
                metadata.fee_regime,
                offchain_net_ev,
                onchain_exact_net_ev,
                summary
            );
            assert!(
                offchain_net_ev + NET_EV_TOL >= onchain_mixed_net_ev,
                "{} {} off-chain net EV lost to on-chain mixed: offchain={} onchain_mixed={} summary={:?}",
                stress.case_id,
                metadata.fee_regime,
                offchain_net_ev,
                onchain_mixed_net_ev,
                summary
            );
        }
    }
}

#[test]
fn complementary_inactive_sell_case_never_loses_to_direct_only_or_rebalance_only() {
    let stress = build_complementary_inactive_sell_case();
    let selected = rebalance_with_custom_predictions_selected_plan_for_test(
        &stress.built.balances_view,
        stress.built.cash_budget,
        &stress.built.slot0_results,
        &stress.built.predictions,
        stress.force_mint_available,
    );
    let direct_only = rebalance_with_custom_predictions_selected_plan_for_test(
        &stress.built.balances_view,
        stress.built.cash_budget,
        &stress.built.slot0_results,
        &stress.built.predictions,
        false,
    );
    let rebalance_only_actions = rebalance_with_custom_predictions_rebalance_only_for_test(
        &stress.built.balances_view,
        stress.built.cash_budget,
        &stress.built.slot0_results,
        &stress.built.predictions,
        stress.force_mint_available,
    );
    let rebalance_only = estimate_plan_economics_for_test(
        &rebalance_only_actions,
        &stress.built.balances_view,
        stress.built.cash_budget,
        &stress.built.slot0_results,
        &stress.built.predictions,
    );

    assert_eq!(stress.metadata.scenario_family, "synthetic_adversarial");
    assert!(
        selected.direct_sell_count > 0,
        "{} should exercise inactive-sell recovery: {:?}",
        stress.case_id,
        selected
    );
    assert!(
        selected.estimated_net_ev.unwrap_or(f64::NEG_INFINITY) + 1e-9
            >= direct_only.estimated_net_ev.unwrap_or(f64::NEG_INFINITY),
        "{} selected plan should not lose to direct-only on net EV: selected={:?} direct_only={:?}",
        stress.case_id,
        selected,
        direct_only
    );
    assert!(
        selected.estimated_net_ev.unwrap_or(f64::NEG_INFINITY) + 1e-9
            >= rebalance_only.estimated_net_ev.unwrap_or(f64::NEG_INFINITY),
        "{} selected plan should not lose to rebalance-only on net EV: selected={:?} rebalance_only={:?}",
        stress.case_id,
        selected,
        rebalance_only
    );
}

#[test]
fn boundary_cutoff_smoke_suite_stays_net_ev_non_regressive() {
    assert_boundary_cutoff_suite_non_regressive(&[16], &[1.0]);
}

#[test]
#[ignore = "heavy deterministic cutoff sweep; run explicitly"]
fn boundary_cutoff_full_sweep_stays_net_ev_non_regressive() {
    assert_boundary_cutoff_suite_non_regressive(&[14, 16, 18, 20, 24, 32], &[0.05, 0.25, 1.0]);
}

#[test]
#[ignore = "heavy release validation; run explicitly with --release"]
fn shared_single_tick_stress_cases_do_not_require_staged_reference() {
    assert_stress_cases_do_not_require_staged_reference(&shared_single_tick_onchain_stress_cases());
}

#[test]
#[ignore = "heavy deterministic cutoff staged comparison; run explicitly"]
fn boundary_cutoff_full_sweep_does_not_require_staged_reference() {
    let cases: Vec<StressCaseDefinition> = [14usize, 16, 18, 20, 24, 32]
        .into_iter()
        .flat_map(|profitable_count| {
            [0.05, 0.25, 1.0].into_iter().map(move |budget_scale| {
                build_boundary_profitable_case(profitable_count, budget_scale, "single_tick")
            })
        })
        .collect();
    assert_stress_cases_do_not_require_staged_reference(&cases);
}

fn assert_boundary_cutoff_suite_non_regressive(profitable_counts: &[usize], budget_scales: &[f64]) {
    let mut checked = 0usize;

    for &profitable_count in profitable_counts {
        for &budget_scale in budget_scales {
            let stress =
                build_boundary_profitable_case(profitable_count, budget_scale, "single_tick");
            let selected = rebalance_with_custom_predictions_selected_plan_for_test(
                &stress.built.balances_view,
                stress.built.cash_budget,
                &stress.built.slot0_results,
                &stress.built.predictions,
                stress.force_mint_available,
            );
            let direct_only = rebalance_with_custom_predictions_selected_plan_for_test(
                &stress.built.balances_view,
                stress.built.cash_budget,
                &stress.built.slot0_results,
                &stress.built.predictions,
                false,
            );

            assert_eq!(
                stress.metadata.profitable_count_bucket,
                profitable_count_bucket(profitable_count)
            );
            assert_eq!(
                stress.metadata.budget_regime,
                budget_regime_for_cash_budget(stress.built.cash_budget)
            );
            assert_ne!(
                selected.fee_estimate_source, "unavailable",
                "{} should retain modeled fee coverage",
                stress.case_id
            );
            assert!(
                selected.estimated_net_ev.unwrap_or(f64::NEG_INFINITY) + 1e-9
                    >= direct_only.estimated_net_ev.unwrap_or(f64::NEG_INFINITY),
                "{} selected plan should not lose to direct-only: selected={:?} direct_only={:?}",
                stress.case_id,
                selected,
                direct_only
            );
            if let Some(constant_l) = compile_constant_l_mixed_selected_plan_for_test(
                &stress.built.balances_view,
                stress.built.cash_budget,
                &stress.built.slot0_results,
                &stress.built.predictions,
                None,
            ) {
                assert!(
                    selected.estimated_net_ev.unwrap_or(f64::NEG_INFINITY) + 1e-9
                        >= constant_l.estimated_net_ev.unwrap_or(f64::NEG_INFINITY),
                    "{} selected plan should not lose to its native constant-L candidate: selected={:?} constant_l={:?}",
                    stress.case_id,
                    selected,
                    constant_l
                );
            }
            checked += 1;
        }
    }

    assert_eq!(
        checked,
        profitable_counts.len() * budget_scales.len(),
        "boundary budget sweep should cover all requested cases"
    );
}

fn assert_stress_cases_do_not_require_staged_reference(cases: &[StressCaseDefinition]) {
    for stress in cases {
        let ultimate_actions = rebalance_with_custom_predictions_for_test(
            &stress.built.balances_view,
            stress.built.cash_budget,
            &stress.built.slot0_results,
            &stress.built.predictions,
            stress.force_mint_available,
        );
        let staged_actions = rebalance_with_custom_predictions_staged_reference_for_test(
            &stress.built.balances_view,
            stress.built.cash_budget,
            &stress.built.slot0_results,
            &stress.built.predictions,
            stress.force_mint_available,
        );
        let ultimate_economics = estimate_plan_economics_for_test(
            &ultimate_actions,
            &stress.built.balances_view,
            stress.built.cash_budget,
            &stress.built.slot0_results,
            &stress.built.predictions,
        );
        let staged_economics = estimate_plan_economics_for_test(
            &staged_actions,
            &stress.built.balances_view,
            stress.built.cash_budget,
            &stress.built.slot0_results,
            &stress.built.predictions,
        );
        assert!(
            ultimate_economics
                .estimated_net_ev
                .unwrap_or(f64::NEG_INFINITY)
                + 1e-9
                >= staged_economics
                    .estimated_net_ev
                    .unwrap_or(f64::NEG_INFINITY),
            "{} staged reference still beats the default solver: ultimate={:?} staged={:?}",
            stress.case_id,
            ultimate_economics,
            staged_economics
        );
    }
}

#[test]
fn explicit_mint_dominant_case_prefers_mint_frontier() {
    let stress = build_mint_dominant_case();
    let summary = compile_constant_l_mixed_selected_plan_for_test(
        &stress.built.balances_view,
        stress.built.cash_budget,
        &stress.built.slot0_results,
        &stress.built.predictions,
        None,
    )
    .expect("mint dominant case should compile a constant-L candidate");
    let direct_only = rebalance_with_custom_predictions_selected_plan_for_test(
        &stress.built.balances_view,
        stress.built.cash_budget,
        &stress.built.slot0_results,
        &stress.built.predictions,
        false,
    );

    assert_eq!(stress.metadata.scenario_family, "synthetic_adversarial");
    assert!(
        summary.mint_count > 0 && summary.direct_sell_count > 0,
        "{} should retain a mint-dominant mixed frontier: {:?}",
        stress.case_id,
        summary
    );
    assert_eq!(
        summary.compiler_variant, "constant_l_mixed",
        "{} should remain in the native constant-L path: {:?}",
        stress.case_id, summary
    );
    assert!(
        summary.direct_buy_count <= 2,
        "{} should only need a small residual direct-buy tail: {:?}",
        stress.case_id,
        summary
    );
    assert!(
        summary.estimated_net_ev.unwrap_or(f64::NEG_INFINITY) + 1e-9
            >= direct_only.estimated_net_ev.unwrap_or(f64::NEG_INFINITY),
        "{} mint-dominant plan should not lose to direct-only: summary={:?} direct_only={:?}",
        stress.case_id,
        summary,
        direct_only
    );
}

#[test]
fn tick_scope_cases_remain_finite_and_are_marked_noncanonical() {
    for stress in build_tick_scope_cases() {
        let selected = rebalance_with_custom_predictions_selected_plan_for_test(
            &stress.built.balances_view,
            stress.built.cash_budget,
            &stress.built.slot0_results,
            &stress.built.predictions,
            stress.force_mint_available,
        );

        assert_eq!(stress.metadata.scenario_family, "tick_scope");
        assert_ne!(stress.metadata.tick_regime, "single_tick");
        assert!(
            selected.raw_ev.is_finite(),
            "{} raw EV should stay finite",
            stress.case_id
        );
        assert!(
            selected.estimated_net_ev.is_some(),
            "{} should keep modeled net EV even in tick-scope stress",
            stress.case_id
        );
    }
}

#[test]
fn ultimate_solver_mixed_ev_dominates_staged_reference() {
    let cases = cases_from_fixture();

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let ultimate_actions = rebalance_with_custom_predictions_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let staged_actions = rebalance_with_custom_predictions_staged_reference_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let ultimate_economics = estimate_plan_economics_for_test(
            &ultimate_actions,
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
        );
        let staged_economics = estimate_plan_economics_for_test(
            &staged_actions,
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
        );
        let ultimate_net_ev = ultimate_economics
            .estimated_net_ev
            .expect("ultimate plan should have a finite fee estimate");
        let staged_net_ev = staged_economics
            .estimated_net_ev
            .expect("staged plan should have a finite fee estimate");
        let net_ev_tol = 1e-10 * (1.0 + ultimate_net_ev.abs() + staged_net_ev.abs());
        let ultimate_non_regressive = ultimate_net_ev + net_ev_tol >= staged_net_ev;
        assert!(
            ultimate_non_regressive,
            "{} ultimate solver underperformed staged reference on net EV: ultimate_net={} staged_net={} ultimate_raw={} staged_raw={}",
            case.case_id,
            ultimate_net_ev,
            staged_net_ev,
            ultimate_economics.raw_ev,
            staged_economics.raw_ev
        );
        if (ultimate_net_ev - staged_net_ev).abs() <= net_ev_tol {
            assert!(
                ultimate_actions.len() <= staged_actions.len(),
                "{} ultimate solver matched staged net EV but used more actions: ultimate_actions={} staged_actions={}",
                case.case_id,
                ultimate_actions.len(),
                staged_actions.len()
            );
        }
    }
}

#[test]
fn exact_no_arb_dominates_heuristic_no_arb() {
    let cases = cases_from_fixture();

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let exact_actions = rebalance_with_custom_predictions_exact_no_arb_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let heuristic_actions = rebalance_with_custom_predictions_rebalance_strict_no_arb_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let exact_ev = benchmark_ev_wei(&exact_actions, &built);
        let heuristic_ev = benchmark_ev_wei(&heuristic_actions, &built);

        assert!(
            exact_ev >= heuristic_ev,
            "{} exact no-arb underperformed heuristic no-arb: exact={} heuristic={}",
            case.case_id,
            exact_ev,
            heuristic_ev
        );
        if exact_ev == heuristic_ev {
            assert!(
                exact_actions.len() <= heuristic_actions.len(),
                "{} exact no-arb matched heuristic EV but used more actions: exact_actions={} heuristic_actions={}",
                case.case_id,
                exact_actions.len(),
                heuristic_actions.len()
            );
        }
    }
}

#[test]
fn distilled_exact_no_arb_is_non_regressive_vs_baseline_k4_search() {
    let cases = cases_from_fixture();

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let distilled_actions =
            rebalance_with_custom_predictions_exact_no_arb_with_distilled_proposals_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
            );
        let baseline_k4_actions =
            rebalance_with_custom_predictions_exact_no_arb_with_search_config_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
                4,
                false,
            );
        let distilled_ev = benchmark_ev_wei(&distilled_actions, &built);
        let baseline_k4_ev = benchmark_ev_wei(&baseline_k4_actions, &built);

        assert!(
            distilled_ev >= baseline_k4_ev,
            "{} distilled exact no-arb regressed versus baseline k4 exact search: distilled={} baseline_k4={}",
            case.case_id,
            distilled_ev,
            baseline_k4_ev
        );
        if distilled_ev == baseline_k4_ev {
            assert!(
                distilled_actions.len() <= baseline_k4_actions.len(),
                "{} distilled exact no-arb matched baseline k4 EV but used more actions: distilled_actions={} baseline_k4_actions={}",
                case.case_id,
                distilled_actions.len(),
                baseline_k4_actions.len()
            );
        }
    }
}

#[test]
fn plain_family_is_non_regressive_vs_exact_no_arb() {
    let cases = cases_from_fixture();

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let plain_actions = rebalance_with_custom_predictions_plain_family_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let exact_actions = rebalance_with_custom_predictions_exact_no_arb_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );

        let plain_ev = benchmark_ev_wei(&plain_actions, &built);
        let exact_ev = benchmark_ev_wei(&exact_actions, &built);
        assert!(
            plain_ev >= exact_ev,
            "{} plain family regressed versus exact no-arb: plain={} exact={}",
            case.case_id,
            plain_ev,
            exact_ev
        );
        if plain_ev == exact_ev {
            assert!(
                plain_actions.len() <= exact_actions.len(),
                "{} plain family matched exact no-arb EV but used more actions: plain_actions={} exact_actions={}",
                case.case_id,
                plain_actions.len(),
                exact_actions.len()
            );
        }
    }
}

#[test]
#[ignore = "debug helper; run explicitly"]
fn print_teacher_distillation_benchmark_cases_jsonl() {
    for case in cases_from_fixture() {
        let built = build_case(&case);
        let force_mint_available = !case.direct_only_reference;
        let teacher = staged_teacher_snapshot_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        )
        .expect("teacher snapshot should exist for benchmark case");
        let operator_only_actions = rebalance_with_custom_predictions_operator_only_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        )
        .0;
        let staged_actions = rebalance_with_custom_predictions_staged_reference_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let operator_only_ev = benchmark_ev_wei(&operator_only_actions, &built);
        let staged_ev = benchmark_ev_wei(&staged_actions, &built);
        let gap_wei = staged_ev.saturating_sub(operator_only_ev);

        println!(
            "{}",
            serde_json::json!({
                "case_id": case.case_id,
                "teacher": teacher,
                "operator_only_ev_wei": operator_only_ev.to_string(),
                "staged_ev_wei": staged_ev.to_string(),
                "gap_wei": gap_wei.to_string(),
                "within_wobble": gap_wei <= 65_536u128,
            })
        );
    }
}

#[tokio::test]
#[ignore = "debug helper; run explicitly"]
async fn print_gas_aware_solver_ablation_jsonl() {
    let rpc_url = std::env::var("RPC").unwrap_or_else(|_| "https://optimism.drpc.org".to_string());
    let gas_assumptions = default_gas_assumptions_with_optimism_l1_fee(&rpc_url)
        .await
        .unwrap_or_else(|_| crate::execution::gas::GasAssumptions::default());
    let fee_inputs = fetch_live_optimism_fee_inputs(&rpc_url, Address::ZERO)
        .await
        .expect("live OP fee inputs should resolve");
    let executor = Address::repeat_byte(0x44);
    let eth_usd = 3000.0;

    for case in cases_from_fixture() {
        let built = build_case(&case);
        let force_mint_available = !case.direct_only_reference;
        let mut previous_raw_ev_susd: Option<f64> = None;
        let mut previous_net_ev_susd: Option<f64> = None;

        for (layer, actions) in solver_layer_actions_for_case(&built, force_mint_available) {
            let started = std::time::Instant::now();
            let (group_count, skipped_group_count, total_fee_susd) =
                total_exact_gas_susd_for_actions(
                    &built,
                    &actions,
                    &gas_assumptions,
                    fee_inputs,
                    executor,
                    eth_usd,
                )
                .await;
            let runtime_millis = started.elapsed().as_millis();
            let raw_ev_wei = benchmark_ev_wei(&actions, &built);
            let raw_ev_susd = wad_to_f64(raw_ev_wei);
            let net_ev_susd = raw_ev_susd - total_fee_susd;
            let incremental_raw_ev_susd = previous_raw_ev_susd
                .map(|previous| raw_ev_susd - previous)
                .unwrap_or(0.0);
            let incremental_net_ev_susd = previous_net_ev_susd
                .map(|previous| net_ev_susd - previous)
                .unwrap_or(0.0);
            previous_raw_ev_susd = Some(raw_ev_susd);
            previous_net_ev_susd = Some(net_ev_susd);

            println!(
                "{}",
                serde_json::to_string(&GasAblationRow {
                    case_id: case.case_id.clone(),
                    layer: layer.to_string(),
                    raw_ev_wei: raw_ev_wei.to_string(),
                    raw_ev_susd,
                    action_count: actions.len(),
                    group_count,
                    skipped_group_count,
                    runtime_millis,
                    total_fee_susd,
                    net_ev_susd,
                    incremental_raw_ev_susd,
                    incremental_net_ev_susd,
                })
                .expect("gas ablation row should serialize")
            );
        }
    }
}

#[tokio::test]
#[ignore = "debug helper; run explicitly"]
async fn print_shared_op_snapshot_solver_matrix_jsonl() {
    let rpc_url = std::env::var("RPC").unwrap_or_else(|_| "https://optimism.drpc.org".to_string());
    let gas_assumptions = default_gas_assumptions_with_optimism_l1_fee(&rpc_url)
        .await
        .unwrap_or_else(|_| crate::execution::gas::GasAssumptions::default());
    let fee_inputs = crate::execution::gas::LiveOptimismFeeInputs {
        sender_nonce: 0,
        chain_id: BENCHMARK_OP_CHAIN_ID,
        gas_price_wei: BENCHMARK_OP_GAS_PRICE_WEI,
    };
    let executor = Address::repeat_byte(0x44);
    let eth_usd = BENCHMARK_OP_ETH_USD;
    let onchain_artifacts = load_onchain_call_artifacts();

    for case in cases_from_fixture() {
        let built = build_case(&case);
        let shared_metadata = fixture_case_row_metadata(&case, &built, 1.0);
        let force_mint_available = !case.direct_only_reference;

        let offchain_direct_actions = rebalance_with_custom_predictions_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            false,
        );
        let offchain_direct_summary = rebalance_with_custom_predictions_selected_plan_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            false,
        );
        let offchain_default_actions = rebalance_with_custom_predictions_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let offchain_default_summary = rebalance_with_custom_predictions_selected_plan_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let k1_teacher_diagnostics = k1_teacher_row_diagnostics_for_built_case(&built);

        let started_direct = std::time::Instant::now();
        let (direct_group_count, _direct_skipped, direct_fee_susd) =
            total_exact_fee_susd_for_actions_with_exact_l1(
                &rpc_url,
                &built,
                &offchain_direct_actions,
                &gas_assumptions,
                fee_inputs,
                executor,
                eth_usd,
            )
            .await;
        let direct_runtime_millis = started_direct.elapsed().as_millis();
        let direct_raw_ev_wei = benchmark_ev_wei(&offchain_direct_actions, &built);
        let direct_raw_ev_susd = wad_to_f64(direct_raw_ev_wei);
        println!(
            "{}",
            serde_json::to_string(&SharedSnapshotSolverRow {
                case_id: case.case_id.clone(),
                flavor: "offchain_direct".to_string(),
                scenario_family: shared_metadata.scenario_family.to_string(),
                outcome_count: shared_metadata.outcome_count,
                profitable_count: shared_metadata.profitable_count,
                profitable_count_bucket: shared_metadata.profitable_count_bucket.to_string(),
                budget_regime: shared_metadata.budget_regime.to_string(),
                fee_regime: shared_metadata.fee_regime.to_string(),
                tick_regime: shared_metadata.tick_regime.to_string(),
                raw_ev_wei: direct_raw_ev_wei.to_string(),
                raw_ev_susd: direct_raw_ev_susd,
                total_fee_susd: direct_fee_susd,
                net_ev_susd: direct_raw_ev_susd - direct_fee_susd,
                action_count: offchain_direct_actions.len(),
                group_or_tx_count: direct_group_count,
                runtime_millis: Some(direct_runtime_millis),
                fee_source: "exact_l1_exact_replay_l2".to_string(),
                family: Some(offchain_direct_summary.family.to_string()),
                compiler_variant: Some(offchain_direct_summary.compiler_variant.to_string()),
                selected_common_shift: offchain_direct_summary.selected_common_shift,
                selected_mixed_lambda: offchain_direct_summary.selected_mixed_lambda,
                selected_active_set_size: offchain_direct_summary.selected_active_set_size,
                selected_stage_count: offchain_direct_summary.selected_stage_count,
                selected_stage1_budget_fraction: offchain_direct_summary
                    .selected_stage1_budget_fraction,
                k1_teacher_source: None,
                runtime_k1_gap_net_ev: None,
                runtime_k1_gap_raw_ev: None,
                oracle_best_is_direct_prefix: None,
                oracle_best_active_set_size: None,
                direct_buy_count: Some(offchain_direct_summary.direct_buy_count),
                direct_sell_count: Some(offchain_direct_summary.direct_sell_count),
                mint_count: Some(offchain_direct_summary.mint_count),
                merge_count: Some(offchain_direct_summary.merge_count),
                total_calldata_bytes: offchain_direct_summary.total_calldata_bytes,
            })
            .expect("shared snapshot row should serialize")
        );

        let started_default = std::time::Instant::now();
        let (default_group_count, _default_skipped, default_fee_susd) =
            total_exact_fee_susd_for_actions_with_exact_l1(
                &rpc_url,
                &built,
                &offchain_default_actions,
                &gas_assumptions,
                fee_inputs,
                executor,
                eth_usd,
            )
            .await;
        let default_runtime_millis = started_default.elapsed().as_millis();
        let default_raw_ev_wei = benchmark_ev_wei(&offchain_default_actions, &built);
        let default_raw_ev_susd = wad_to_f64(default_raw_ev_wei);
        println!(
            "{}",
            serde_json::to_string(&SharedSnapshotSolverRow {
                case_id: case.case_id.clone(),
                flavor: "offchain_default".to_string(),
                scenario_family: shared_metadata.scenario_family.to_string(),
                outcome_count: shared_metadata.outcome_count,
                profitable_count: shared_metadata.profitable_count,
                profitable_count_bucket: shared_metadata.profitable_count_bucket.to_string(),
                budget_regime: shared_metadata.budget_regime.to_string(),
                fee_regime: shared_metadata.fee_regime.to_string(),
                tick_regime: shared_metadata.tick_regime.to_string(),
                raw_ev_wei: default_raw_ev_wei.to_string(),
                raw_ev_susd: default_raw_ev_susd,
                total_fee_susd: default_fee_susd,
                net_ev_susd: default_raw_ev_susd - default_fee_susd,
                action_count: offchain_default_actions.len(),
                group_or_tx_count: default_group_count,
                runtime_millis: Some(default_runtime_millis),
                fee_source: "exact_l1_exact_replay_l2".to_string(),
                family: Some(offchain_default_summary.family.to_string()),
                compiler_variant: Some(offchain_default_summary.compiler_variant.to_string()),
                selected_common_shift: offchain_default_summary.selected_common_shift,
                selected_mixed_lambda: offchain_default_summary.selected_mixed_lambda,
                selected_active_set_size: offchain_default_summary.selected_active_set_size,
                selected_stage_count: offchain_default_summary.selected_stage_count,
                selected_stage1_budget_fraction: offchain_default_summary
                    .selected_stage1_budget_fraction,
                k1_teacher_source: k1_teacher_diagnostics.k1_teacher_source.clone(),
                runtime_k1_gap_net_ev: k1_teacher_diagnostics.runtime_k1_gap_net_ev,
                runtime_k1_gap_raw_ev: k1_teacher_diagnostics.runtime_k1_gap_raw_ev,
                oracle_best_is_direct_prefix: k1_teacher_diagnostics.oracle_best_is_direct_prefix,
                oracle_best_active_set_size: k1_teacher_diagnostics.oracle_best_active_set_size,
                direct_buy_count: Some(offchain_default_summary.direct_buy_count),
                direct_sell_count: Some(offchain_default_summary.direct_sell_count),
                mint_count: Some(offchain_default_summary.mint_count),
                merge_count: Some(offchain_default_summary.merge_count),
                total_calldata_bytes: offchain_default_summary.total_calldata_bytes,
            })
            .expect("shared snapshot row should serialize")
        );

        let onchain = onchain_artifacts
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing on-chain artifact for {}", case.case_id));
        for (flavor, artifact) in [
            ("onchain_exact", &onchain.exact),
            ("onchain_mixed", &onchain.mixed),
        ] {
            let unsigned_tx_data = build_unsigned_contract_call_tx_bytes(
                artifact.target,
                artifact.calldata.clone().into(),
                fee_inputs,
                artifact.gas_units,
            )
            .expect("unsigned tx bytes should build for on-chain benchmark call");
            let l1_fee_wei =
                fetch_exact_l1_fee_wei_with_retry(&rpc_url, unsigned_tx_data.clone()).await;
            let l1_fee_susd = wei_to_susd(l1_fee_wei, eth_usd);
            let l2_fee_susd =
                (artifact.gas_units as f64 * fee_inputs.gas_price_wei as f64) * eth_usd / 1e18;
            let total_fee_susd = l1_fee_susd + l2_fee_susd;
            let raw_ev_susd = wad_to_f64(artifact.raw_ev_wei);
            println!(
                "{}",
                serde_json::to_string(&SharedSnapshotSolverRow {
                    case_id: case.case_id.clone(),
                    flavor: flavor.to_string(),
                    scenario_family: shared_metadata.scenario_family.to_string(),
                    outcome_count: shared_metadata.outcome_count,
                    profitable_count: shared_metadata.profitable_count,
                    profitable_count_bucket: shared_metadata.profitable_count_bucket.to_string(),
                    budget_regime: shared_metadata.budget_regime.to_string(),
                    fee_regime: shared_metadata.fee_regime.to_string(),
                    tick_regime: shared_metadata.tick_regime.to_string(),
                    raw_ev_wei: artifact.raw_ev_wei.to_string(),
                    raw_ev_susd,
                    total_fee_susd,
                    net_ev_susd: raw_ev_susd - total_fee_susd,
                    action_count: 0,
                    group_or_tx_count: 1,
                    runtime_millis: None,
                    fee_source: "exact_l1_foundry_gas_units".to_string(),
                    family: None,
                    compiler_variant: None,
                    selected_common_shift: None,
                    selected_mixed_lambda: None,
                    selected_active_set_size: None,
                    selected_stage_count: None,
                    selected_stage1_budget_fraction: None,
                    k1_teacher_source: None,
                    runtime_k1_gap_net_ev: None,
                    runtime_k1_gap_raw_ev: None,
                    oracle_best_is_direct_prefix: None,
                    oracle_best_active_set_size: None,
                    direct_buy_count: None,
                    direct_sell_count: None,
                    mint_count: None,
                    merge_count: None,
                    total_calldata_bytes: None,
                })
                .expect("shared snapshot row should serialize")
            );
        }
    }
}

#[tokio::test]
#[ignore = "debug helper; run explicitly"]
async fn print_shared_op_snapshot_onchain_benchmark_rows_jsonl() {
    let gas_assumptions = benchmark_gas_assumptions_for_test();
    let fee_inputs = crate::execution::gas::LiveOptimismFeeInputs {
        sender_nonce: 0,
        chain_id: BENCHMARK_OP_CHAIN_ID,
        gas_price_wei: BENCHMARK_OP_GAS_PRICE_WEI,
    };
    let eth_usd = BENCHMARK_OP_ETH_USD;
    let onchain_artifacts = load_onchain_call_artifacts();

    for case in cases_from_fixture() {
        let built = build_case(&case);
        let shared_metadata = fixture_case_row_metadata(&case, &built, 1.0);
        let onchain = onchain_artifacts
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing on-chain artifact for {}", case.case_id));
        for (flavor, artifact) in [
            ("onchain_exact", &onchain.exact),
            ("onchain_mixed", &onchain.mixed),
        ] {
            let unsigned_tx_data = build_unsigned_contract_call_tx_bytes(
                artifact.target,
                artifact.calldata.clone().into(),
                fee_inputs,
                artifact.gas_units,
            )
            .expect("unsigned tx bytes should build for on-chain benchmark call");
            let l1_fee_susd =
                modeled_l1_fee_susd_for_tx_bytes(&gas_assumptions, unsigned_tx_data.len(), eth_usd);
            let l2_fee_susd =
                (artifact.gas_units as f64 * fee_inputs.gas_price_wei as f64) * eth_usd / 1e18;
            let total_fee_susd = l1_fee_susd + l2_fee_susd;
            let raw_ev_susd = wad_to_f64(artifact.raw_ev_wei);
            println!(
                "{}",
                serde_json::to_string(&SharedSnapshotSolverRow {
                    case_id: case.case_id.clone(),
                    flavor: flavor.to_string(),
                    scenario_family: shared_metadata.scenario_family.to_string(),
                    outcome_count: shared_metadata.outcome_count,
                    profitable_count: shared_metadata.profitable_count,
                    profitable_count_bucket: shared_metadata.profitable_count_bucket.to_string(),
                    budget_regime: shared_metadata.budget_regime.to_string(),
                    fee_regime: shared_metadata.fee_regime.to_string(),
                    tick_regime: shared_metadata.tick_regime.to_string(),
                    raw_ev_wei: artifact.raw_ev_wei.to_string(),
                    raw_ev_susd,
                    total_fee_susd,
                    net_ev_susd: raw_ev_susd - total_fee_susd,
                    action_count: 0,
                    group_or_tx_count: 1,
                    runtime_millis: None,
                    fee_source: "modeled_l1_foundry_gas_units".to_string(),
                    family: None,
                    compiler_variant: None,
                    selected_common_shift: None,
                    selected_mixed_lambda: None,
                    selected_active_set_size: None,
                    selected_stage_count: None,
                    selected_stage1_budget_fraction: None,
                    k1_teacher_source: None,
                    runtime_k1_gap_net_ev: None,
                    runtime_k1_gap_raw_ev: None,
                    oracle_best_is_direct_prefix: None,
                    oracle_best_active_set_size: None,
                    direct_buy_count: None,
                    direct_sell_count: None,
                    mint_count: None,
                    merge_count: None,
                    total_calldata_bytes: None,
                })
                .expect("shared snapshot row should serialize")
            );
        }
    }
}

#[tokio::test]
#[ignore = "debug helper; run explicitly"]
async fn print_shared_op_snapshot_offchain_selected_rows_jsonl() {
    for case in cases_from_fixture() {
        let built = build_case(&case);
        let shared_metadata = fixture_case_row_metadata(&case, &built, 1.0);
        let force_mint_available = !case.direct_only_reference;
        let k1_teacher_diagnostics = k1_teacher_row_diagnostics_for_built_case(&built);
        for (flavor, force_mint) in [
            ("offchain_direct", false),
            ("offchain_default", force_mint_available),
        ] {
            let started = std::time::Instant::now();
            let actions = rebalance_with_custom_predictions_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint,
            );
            let summary = rebalance_with_custom_predictions_selected_plan_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint,
            );
            let runtime_millis = started.elapsed().as_millis();
            let raw_ev_wei = benchmark_ev_wei(&actions, &built);
            let raw_ev_susd = wad_to_f64(raw_ev_wei);
            let total_fee_susd = summary
                .estimated_total_fee_susd
                .expect("selected benchmark plan should have a modeled fee estimate");
            let group_or_tx_count = summary.estimated_tx_count.unwrap_or(0);
            println!(
                "{}",
                serde_json::to_string(&SharedSnapshotSolverRow {
                    case_id: case.case_id.clone(),
                    flavor: flavor.to_string(),
                    scenario_family: shared_metadata.scenario_family.to_string(),
                    outcome_count: shared_metadata.outcome_count,
                    profitable_count: shared_metadata.profitable_count,
                    profitable_count_bucket: shared_metadata.profitable_count_bucket.to_string(),
                    budget_regime: shared_metadata.budget_regime.to_string(),
                    fee_regime: shared_metadata.fee_regime.to_string(),
                    tick_regime: shared_metadata.tick_regime.to_string(),
                    raw_ev_wei: raw_ev_wei.to_string(),
                    raw_ev_susd,
                    total_fee_susd,
                    net_ev_susd: raw_ev_susd - total_fee_susd,
                    action_count: actions.len(),
                    group_or_tx_count,
                    runtime_millis: Some(runtime_millis),
                    fee_source: summary.fee_estimate_source.to_string(),
                    family: Some(summary.family.to_string()),
                    compiler_variant: Some(summary.compiler_variant.to_string()),
                    selected_common_shift: summary.selected_common_shift,
                    selected_mixed_lambda: summary.selected_mixed_lambda,
                    selected_active_set_size: summary.selected_active_set_size,
                    selected_stage_count: summary.selected_stage_count,
                    selected_stage1_budget_fraction: summary.selected_stage1_budget_fraction,
                    k1_teacher_source: if flavor == "offchain_default" {
                        k1_teacher_diagnostics.k1_teacher_source.clone()
                    } else {
                        None
                    },
                    runtime_k1_gap_net_ev: if flavor == "offchain_default" {
                        k1_teacher_diagnostics.runtime_k1_gap_net_ev
                    } else {
                        None
                    },
                    runtime_k1_gap_raw_ev: if flavor == "offchain_default" {
                        k1_teacher_diagnostics.runtime_k1_gap_raw_ev
                    } else {
                        None
                    },
                    oracle_best_is_direct_prefix: if flavor == "offchain_default" {
                        k1_teacher_diagnostics.oracle_best_is_direct_prefix
                    } else {
                        None
                    },
                    oracle_best_active_set_size: if flavor == "offchain_default" {
                        k1_teacher_diagnostics.oracle_best_active_set_size
                    } else {
                        None
                    },
                    direct_buy_count: Some(summary.direct_buy_count),
                    direct_sell_count: Some(summary.direct_sell_count),
                    mint_count: Some(summary.mint_count),
                    merge_count: Some(summary.merge_count),
                    total_calldata_bytes: summary.total_calldata_bytes,
                })
                .expect("shared snapshot row should serialize")
            );
        }
    }
}

#[test]
fn arb_primed_root_is_only_taken_when_start_arb_is_positive() {
    let cases = cases_from_fixture();

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let (_actions, stats) = rebalance_with_custom_predictions_and_stats_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let start_arb_actions = arb_only_with_custom_predictions_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let start_arb_ev = benchmark_ev_wei(&start_arb_actions, &built);
        let arb_primed_actions = rebalance_with_custom_predictions_arb_primed_family_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let initial_ev = ev_to_wei(state_ev(
            &built.balances_view,
            built.cash_budget,
            &built.predictions,
        ));
        let should_take_root = start_arb_ev > initial_ev;

        assert!(
            stats.arb_primed_root_taken == should_take_root,
            "{} arb-primed root mismatch: took_root={} should_take_root={} start_arb_ev={} initial_ev={}",
            case.case_id,
            stats.arb_primed_root_taken,
            should_take_root,
            start_arb_ev,
            initial_ev
        );
        assert_eq!(
            arb_primed_actions.is_some(),
            stats.arb_primed_root_taken,
            "{} arb-primed family helper/root-taken stats diverged",
            case.case_id
        );
    }
}

#[test]
fn ultimate_solver_is_deterministic_under_parallel_candidate_evaluation() {
    let cases = cases_from_fixture();

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let baseline = rebalance_with_custom_predictions_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        for _ in 0..3 {
            let rerun = rebalance_with_custom_predictions_for_test(
                &built.balances_view,
                built.cash_budget,
                &built.slot0_results,
                &built.predictions,
                force_mint_available,
            );
            assert_eq!(
                rerun, baseline,
                "{} solver produced nondeterministic actions under parallel exact enumeration",
                case.case_id
            );
        }
    }
}

#[test]
fn target_delta_compiler_direct_ninety_eight_case_stays_direct() {
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "ninety_eight_outcome_multitick_direct_only")
        .expect("fixture should include ninety_eight_outcome_multitick_direct_only");
    let built = build_case(&case);
    let rich_actions = rebalance_with_custom_predictions_rebalance_only_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        false,
    );
    let compiled_actions = compile_target_delta_actions_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        &rich_actions,
    )
    .expect("target-delta compiler should build a direct-only compact candidate");
    assert!(
        !compiled_actions
            .iter()
            .any(|action| matches!(action, Action::Mint { .. } | Action::Merge { .. })),
        "direct-only compiler output should not use mint/merge: {:?}",
        compiled_actions
    );
}

#[test]
fn analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev() {
    const PREVIOUS_NET_EV_FLOOR: f64 = 150.36371245961456;
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "heterogeneous_ninety_eight_outcome_l1_like_case")
        .expect("fixture should include heterogeneous_ninety_eight_outcome_l1_like_case");
    let built = build_case(&case);
    let selected_summary = rebalance_with_custom_predictions_selected_plan_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        true,
    );
    assert!(
        matches!(
            selected_summary.compiler_variant,
            "analytic_mixed" | "constant_l_mixed"
        ),
        "realistic 98 solver should keep the best compact mixed compiler family: {:?}",
        selected_summary
    );
    assert_eq!(
        selected_summary.estimated_tx_count,
        Some(1),
        "realistic 98 selected plan should remain one packed tx"
    );
    assert!(
        selected_summary.action_count <= 197,
        "realistic 98 selected plan should stay compact: actions={}",
        selected_summary.action_count
    );
    assert!(
        selected_summary
            .estimated_net_ev
            .unwrap_or(f64::NEG_INFINITY)
            > PREVIOUS_NET_EV_FLOOR,
        "realistic 98 selected plan should improve the previous net-EV floor: {:?}",
        selected_summary
    );
    if selected_summary.compiler_variant != "analytic_mixed" {
        assert!(
            selected_summary.compiler_variant == "constant_l_mixed",
            "realistic 98 winner should only displace analytic_mixed if the new K=1 solver is strictly better: {:?}",
            selected_summary
        );
    }
}

#[test]
fn mixed_route_favorable_selection_never_loses_to_direct_only() {
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "mixed_route_favorable_synthetic_case")
        .expect("fixture should include mixed_route_favorable_synthetic_case");
    let built = build_case(&case);
    let onchain_artifacts = load_onchain_call_artifacts();
    let onchain_mixed_raw_ev = wad_to_f64(
        onchain_artifacts
            .get(&case.case_id)
            .expect("fixture should have an on-chain mixed artifact")
            .mixed
            .raw_ev_wei,
    );
    let selected_summary = rebalance_with_custom_predictions_selected_plan_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        true,
    );
    let constant_l_summary = compile_constant_l_mixed_selected_plan_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        None,
    )
    .expect("constant-L compiler should build a mixed-route candidate");
    let direct_only_summary = rebalance_with_custom_predictions_selected_plan_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        false,
    );
    let rich_actions = rebalance_with_custom_predictions_rebalance_only_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        true,
    );
    let compiled_actions = compile_target_delta_actions_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        &rich_actions,
    )
    .expect("target-delta compiler should build a mixed-route candidate");
    let compiled_economics = estimate_plan_economics_for_test(
        &compiled_actions,
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
    );

    assert!(
        matches!(
            selected_summary.compiler_variant,
            "constant_l_mixed" | "staged_constant_l_2"
        ),
        "synthetic mixed-route case should now be selected by the native constant-L compiler family: {:?}",
        selected_summary
    );
    assert_ne!(
        selected_summary.compiler_variant, "baseline_step_prune",
        "synthetic mixed-route case should no longer fall back to baseline_step_prune: {:?}",
        selected_summary
    );
    assert!(
        selected_summary.action_count <= 8,
        "synthetic mixed-route case should stay compact: {:?}",
        selected_summary
    );
    assert!(
        selected_summary
            .estimated_net_ev
            .unwrap_or(f64::NEG_INFINITY)
            + 1e-12
            >= compiled_economics
                .estimated_net_ev
                .unwrap_or(f64::NEG_INFINITY),
        "solver should keep the higher-net-EV baseline compact candidate over target-delta: selected={:?} compiled={:?}",
        selected_summary,
        compiled_economics
    );
    assert!(
        selected_summary
            .estimated_net_ev
            .unwrap_or(f64::NEG_INFINITY)
            + 1e-12
            >= direct_only_summary
                .estimated_net_ev
                .unwrap_or(f64::NEG_INFINITY),
        "mixed/default selection should never lose to the independently solved direct-only guard: selected={:?} direct_only={:?}",
        selected_summary,
        direct_only_summary
    );
    if selected_summary.compiler_variant == "constant_l_mixed" {
        assert!(
            (selected_summary.raw_ev - onchain_mixed_raw_ev).abs() <= 1e-5,
            "K=1 constant-L compiler should converge to the on-chain mixed raw EV on the synthetic case: selected={:?} onchain_raw_ev={}",
            selected_summary,
            onchain_mixed_raw_ev
        );
    } else {
        assert!(
            selected_summary.raw_ev + 1e-12 >= constant_l_summary.raw_ev,
            "staged constant-L compiler should be at least as strong as K=1 on raw EV: selected={:?} constant_l={:?}",
            selected_summary,
            constant_l_summary
        );
    }
}

#[test]
fn mixed_route_favorable_constant_l_certificate_matches_expected_equilibrium() {
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "mixed_route_favorable_synthetic_case")
        .expect("fixture should include mixed_route_favorable_synthetic_case");
    let built = build_case(&case);
    let onchain_artifacts = load_onchain_call_artifacts();
    let onchain_mixed_raw_ev = wad_to_f64(
        onchain_artifacts
            .get(&case.case_id)
            .expect("fixture should have an on-chain mixed artifact")
            .mixed
            .raw_ev_wei,
    );
    let constant_l_summary = compile_constant_l_mixed_selected_plan_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        None,
    )
    .expect("constant-L compiler should build the synthetic mixed certificate");

    assert_eq!(
        constant_l_summary.compiler_variant, "constant_l_mixed",
        "direct constant-L compilation should expose the new compiler variant: {:?}",
        constant_l_summary
    );
    assert_eq!(
        constant_l_summary.selected_active_set_size,
        Some(1),
        "synthetic constant-L certificate should keep a single active outcome: {:?}",
        constant_l_summary
    );
    assert!(
        (constant_l_summary.selected_mixed_lambda.unwrap_or(f64::NAN) - 0.0).abs() <= 1e-12,
        "synthetic constant-L certificate should converge at pi=0: {:?}",
        constant_l_summary
    );
    assert!(
        (constant_l_summary.selected_common_shift.unwrap_or(f64::NAN) - 1.1598556159322904).abs()
            <= 1e-9,
        "synthetic constant-L certificate should recover the expected mint amount: {:?}",
        constant_l_summary
    );
    assert!(
        (constant_l_summary.raw_ev - onchain_mixed_raw_ev).abs() <= 1e-5,
        "synthetic constant-L certificate should match on-chain mixed raw EV: {:?} onchain_raw_ev={}",
        constant_l_summary,
        onchain_mixed_raw_ev
    );
}

#[test]
fn small_bundle_constant_l_certificate_has_multi_active_support() {
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "small_bundle_mixed_case")
        .expect("fixture should include small_bundle_mixed_case");
    let built = build_case(&case);
    let constant_l_summary = compile_constant_l_mixed_selected_plan_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        None,
    )
    .expect("constant-L compiler should build the small-bundle mixed certificate");
    let certificate = constant_l_summary
        .mixed_certificates
        .first()
        .expect("constant-L summary should expose a mixed certificate");

    assert!(
        certificate.active_set_size > 1,
        "small bundle mixed certificate should exercise a multi-active K=1 state: {:?}",
        constant_l_summary
    );
}

#[test]
fn constant_l_random_corpus_contains_positive_pi_multi_active_certificate() {
    let found = (1..=256u64).find_map(|seed| {
        let built = build_random_small_case(seed);
        let summary = compile_constant_l_mixed_selected_plan_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        )?;
        let certificate = summary.mixed_certificates.first()?;
        (certificate.active_set_size > 1 && certificate.pi > 1e-9).then_some((seed, summary))
    });

    assert!(
        found.is_some(),
        "expected the deterministic small-case corpus to contain at least one positive-pi, multi-active K=1 certificate"
    );
}

#[test]
fn constant_l_random_corpus_exercises_self_financing_budget_accounting() {
    let found = (1..=256u64).find_map(|seed| {
        let built = build_random_small_case(seed);
        let summary = compile_constant_l_mixed_selected_plan_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        )?;
        let certificate = summary.mixed_certificates.first()?;
        (certificate.sell_proceeds > certificate.mint_amount + 1e-9)
            .then_some((seed, built, summary))
    });
    let (seed, built, summary) = found.expect(
        "expected the deterministic small-case corpus to contain at least one self-financing K=1 certificate",
    );
    let certificate = summary
        .mixed_certificates
        .first()
        .expect("found self-financing certificate should still exist");
    let expected_budget_used = certificate
        .mint_amount
        .max(certificate.direct_cost + certificate.mint_amount - certificate.sell_proceeds)
        .max(0.0);
    let legacy_clamped_budget =
        certificate.direct_cost + (certificate.mint_amount - certificate.sell_proceeds).max(0.0);

    assert!(
        (certificate.budget_used - expected_budget_used).abs() <= 1e-9,
        "certificate should report the exact sequential starting-cash requirement on self-financing seed {}: {:?}",
        seed,
        summary
    );
    assert!(
        certificate.budget_used <= built.cash_budget + 1e-9,
        "self-financing mixed certificate should remain budget-feasible on seed {}: {:?}",
        seed,
        summary
    );
    assert!(
        (legacy_clamped_budget - certificate.budget_used).abs() > 1e-9,
        "self-financing regression should differ from the legacy clamped budget proxy on seed {}: {:?}",
        seed,
        summary
    );
}

#[test]
fn constant_l_runtime_search_matches_oracle_on_committed_small_mixed_cases() {
    let mut checked = 0usize;
    for case in cases_from_fixture() {
        if case.direct_only_reference || case.uniform_count > 0 || case.predictions_wad.len() > 12 {
            continue;
        }
        let built = build_case(&case);
        let comparison = compare_constant_l_runtime_vs_oracle_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        )
        .unwrap_or_else(|| panic!("K=1 oracle comparison should exist for {}", case.case_id));
        checked += 1;
        assert!(
            comparison.runtime_k1_gap_net_ev.unwrap_or(f64::INFINITY) <= 1e-9,
            "runtime K=1 search should match the K=1 oracle on committed small mixed case {}: {:?}",
            case.case_id,
            comparison
        );
    }
    assert!(
        checked > 0,
        "expected at least one committed small mixed case"
    );
}

#[test]
fn constant_l_runtime_search_matches_oracle_on_random_small_corpus() {
    let mut checked = 0usize;
    let mut failures = Vec::new();

    for seed in 1..=96u64 {
        let built = build_random_small_case(seed);
        let Some(comparison) = compare_constant_l_runtime_vs_oracle_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        ) else {
            continue;
        };
        checked += 1;
        if comparison.runtime_k1_gap_net_ev.unwrap_or(0.0) > 1e-9 {
            failures.push(format!(
                "seed={} gap={} runtime_best={:?} oracle_best={:?}",
                seed,
                comparison.runtime_k1_gap_net_ev.unwrap_or(f64::NAN),
                comparison.runtime_best,
                comparison.oracle_best
            ));
        }
        if checked >= 24 {
            break;
        }
    }

    assert!(
        checked >= 24,
        "expected at least 24 oracle-comparable randomized small cases, got {}",
        checked
    );
    assert!(
        failures.is_empty(),
        "randomized K=1 oracle found runtime gaps:\n{}",
        failures.join("\n")
    );
}

#[test]
fn constant_l_runtime_search_matches_medium_oracle_on_random_medium_corpus() {
    let mut checked = 0usize;
    let mut failures = Vec::new();

    for seed in 1..=3u64 {
        let built = build_random_medium_case(seed);
        let Some(comparison) = compare_constant_l_runtime_vs_medium_oracle_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        ) else {
            continue;
        };
        checked += 1;
        if comparison.runtime_k1_gap_net_ev.unwrap_or(0.0) > 1e-9 {
            failures.push(format!(
                "seed={} gap_net={} gap_raw={} runtime_best={:?} oracle_best={:?}",
                seed,
                comparison.runtime_k1_gap_net_ev.unwrap_or(f64::NAN),
                comparison.runtime_k1_gap_raw_ev.unwrap_or(f64::NAN),
                comparison.runtime_best,
                comparison.oracle_best
            ));
        }
    }

    assert_eq!(
        checked, 3,
        "expected the deterministic medium corpus to remain oracle-comparable"
    );
    assert!(
        failures.is_empty(),
        "medium deterministic K=1 oracle found runtime gaps:\n{}",
        failures.join("\n")
    );
}

#[test]
fn constant_l_medium_oracle_reports_active_set_metadata() {
    let built = build_random_medium_case(1);
    let comparison = compare_constant_l_runtime_vs_medium_oracle_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        None,
    )
    .expect("medium K=1 oracle comparison should exist");

    assert!(
        comparison.oracle_best_active_set_size.is_some(),
        "medium oracle comparison should report the oracle best active-set size: {:?}",
        comparison
    );
}

#[test]
fn constant_l_oracle_random_corpus_finds_non_prefix_optima() {
    let mut found = None;
    for seed in 1..=96u64 {
        let built = build_random_small_case(seed);
        let Some(comparison) = compare_constant_l_runtime_vs_oracle_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        ) else {
            continue;
        };
        if !comparison.oracle_best_is_direct_prefix {
            found = Some((seed, comparison));
            break;
        }
    }

    assert!(
        found.is_some(),
        "expected the deterministic oracle corpus to contain at least one non-prefix K=1 optimum"
    );
}

#[test]
#[ignore = "teacher-only large-case diagnostic; run explicitly"]
fn constant_l_best_known_teacher_never_loses_to_runtime_on_large_case() {
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "heterogeneous_ninety_eight_outcome_l1_like_case")
        .expect("fixture should include heterogeneous_ninety_eight_outcome_l1_like_case");
    let built = build_case(&case);
    let comparison = compare_constant_l_runtime_vs_best_known_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        None,
    )
    .expect("best-known K=1 comparison should exist for the large benchmark case");
    assert!(
        comparison.runtime_k1_gap_net_ev.unwrap_or(f64::INFINITY) >= 0.0,
        "best-known teacher comparison should report a non-negative K=1 gap: {:?}",
        comparison
    );
}

#[test]
fn constant_l_k2_oracle_never_loses_to_k1_or_runtime_on_small_cases() {
    let mut checked = 0usize;
    for case in cases_from_fixture() {
        if case.direct_only_reference || case.uniform_count > 0 || case.predictions_wad.len() > 8 {
            continue;
        }
        let built = build_case(&case);
        let Some(comparison) = compare_constant_l_runtime_vs_k2_oracle_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        ) else {
            continue;
        };
        checked += 1;
        let runtime_net = comparison
            .runtime_best
            .estimated_net_ev
            .unwrap_or(comparison.runtime_best.raw_ev);
        let k1_net = comparison
            .k1_oracle_best
            .estimated_net_ev
            .unwrap_or(comparison.k1_oracle_best.raw_ev);
        let k2_net = comparison
            .k2_oracle_best
            .estimated_net_ev
            .unwrap_or(comparison.k2_oracle_best.raw_ev);
        assert!(
            k2_net + 1e-12 >= k1_net,
            "K2 oracle should dominate K1 oracle on oracle-capable case {}: {:?}",
            case.case_id,
            comparison
        );
        assert!(
            k2_net + 1e-12 >= runtime_net,
            "K2 oracle should never lose to runtime K1 on oracle-capable case {}: {:?}",
            case.case_id,
            comparison
        );
    }
    assert!(
        checked > 0,
        "expected at least one K2-oracle-capable benchmark case"
    );
}

#[test]
fn constant_l_k2_oracle_gain_is_negligible_on_current_small_benchmarks() {
    let mut max_gain = 0.0_f64;
    let mut checked = 0usize;
    for case in cases_from_fixture() {
        if case.direct_only_reference || case.uniform_count > 0 || case.predictions_wad.len() > 8 {
            continue;
        }
        let built = build_case(&case);
        let Some(comparison) = compare_constant_l_runtime_vs_k2_oracle_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            None,
        ) else {
            continue;
        };
        checked += 1;
        max_gain = max_gain.max(comparison.k2_gain_net_ev.unwrap_or(0.0));
    }

    assert!(
        checked > 0,
        "expected at least one K2-oracle-capable benchmark case"
    );
    assert!(
        max_gain <= 1e-9,
        "current committed small benchmark suite should not show a material K2 teacher gain before a runtime redesign: max_gain={}",
        max_gain
    );
}

#[test]
fn ultimate_solver_mixed_ev_dominates_end_arb_cyclic_hypothesis() {
    let cases = cases_from_fixture();

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let ultimate_actions = rebalance_with_custom_predictions_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
        );
        let ultimate_ev = benchmark_ev_wei(&ultimate_actions, &built);
        let (cyclic_ev, cyclic_action_count, _) =
            arb_end_cyclic_result_for_built_case(&built, force_mint_available);

        assert!(
            ultimate_ev >= cyclic_ev,
            "{} ultimate solver underperformed end-arb cyclic hypothesis: ultimate={} cyclic={}",
            case.case_id,
            ultimate_ev,
            cyclic_ev
        );
        if ultimate_ev == cyclic_ev {
            assert!(
                (ultimate_actions.len() as u64) <= cyclic_action_count,
                "{} ultimate solver matched end-arb cyclic EV but used more actions: ultimate_actions={} cyclic_actions={}",
                case.case_id,
                ultimate_actions.len(),
                cyclic_action_count
            );
        }
    }
}

#[test]
#[ignore = "snapshot helper; run explicitly"]
fn print_current_optimizer_benchmark_rows() {
    let cases = cases_from_fixture();

    for case in &cases {
        let (direct_ev, mixed_ev, full_rebalance_only_ev, action_count) =
            current_result_for_case(case);
        println!("{}:", case.case_id);
        println!("  offchain_direct_ev_wei={}", direct_ev);
        println!("  offchain_mixed_ev_wei={}", mixed_ev);
        println!(
            "  offchain_full_rebalance_only_ev_wei={}",
            full_rebalance_only_ev
        );
        println!("  offchain_action_count={}", action_count);
    }
}

#[test]
#[ignore = "debug helper; run explicitly"]
fn print_heterogeneous_ninety_eight_exact_preserve_oracle_breakdown() {
    const ORACLE_WOBBLE_TOLERANCE_WEI: u128 = 65_536;

    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "heterogeneous_ninety_eight_outcome_l1_like_case")
        .expect("heterogeneous benchmark case should exist");
    let built = build_case(&case);
    let force_mint_available = !case.direct_only_reference;

    let exact_k4 = rebalance_with_custom_predictions_exact_no_arb_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let exact_k8 = rebalance_with_custom_predictions_exact_no_arb_with_search_config_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
        8,
        false,
    );
    let exact_k8_with_arb_seed =
        rebalance_with_custom_predictions_exact_no_arb_with_search_config_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
            8,
            true,
        );
    let staged = rebalance_with_custom_predictions_staged_reference_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let staged_preserve_candidates = collect_mint_sell_preserve_candidates_for_test(
        &staged,
        &built.slot0_results,
        &built.balances_view,
        built.cash_budget,
        8,
    );
    let exact_staged_seed_k8 =
        rebalance_with_custom_predictions_exact_no_arb_with_explicit_preserve_universe_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
            &staged_preserve_candidates,
        );
    let staged_choice = staged_reference_choice_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    )
    .expect("staged reference should produce a candidate on the heterogeneous benchmark case");
    let exact_staged_choice =
        rebalance_with_custom_predictions_exact_no_arb_with_explicit_choice_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
            &staged_choice.preserve_markets,
            staged_choice.frontier_family,
        );

    let exact_k4_ev = benchmark_ev_wei(&exact_k4, &built);
    let exact_k8_ev = benchmark_ev_wei(&exact_k8, &built);
    let exact_k8_with_arb_seed_ev = benchmark_ev_wei(&exact_k8_with_arb_seed, &built);
    let exact_staged_seed_k8_ev = benchmark_ev_wei(&exact_staged_seed_k8, &built);
    let exact_staged_choice_ev = benchmark_ev_wei(&exact_staged_choice, &built);
    let staged_ev = benchmark_ev_wei(&staged, &built);

    println!("exact_k4 ev={} actions={}", exact_k4_ev, exact_k4.len());
    println!("exact_k8 ev={} actions={}", exact_k8_ev, exact_k8.len());
    println!(
        "exact_k8_with_arb_seed ev={} actions={}",
        exact_k8_with_arb_seed_ev,
        exact_k8_with_arb_seed.len()
    );
    println!(
        "exact_staged_seed_k8 ev={} actions={}",
        exact_staged_seed_k8_ev,
        exact_staged_seed_k8.len()
    );
    println!(
        "exact_staged_choice ev={} actions={}",
        exact_staged_choice_ev,
        exact_staged_choice.len()
    );
    println!("staged_choice_variant={}", staged_choice.variant_label);
    println!(
        "staged_choice_frontier_family={:?}",
        staged_choice.frontier_family
    );
    println!(
        "staged_choice_preserve_markets={:?}",
        staged_choice.preserve_markets
    );
    println!(
        "staged_preserve_candidates={:?}",
        staged_preserve_candidates
    );
    println!("staged ev={} actions={}", staged_ev, staged.len());

    assert!(
        exact_k8_ev.saturating_add(ORACLE_WOBBLE_TOLERANCE_WEI) >= exact_k4_ev,
        "expanded preserve cap should not underperform k4 on the heterogeneous oracle case"
    );
    assert!(
        exact_k8_with_arb_seed_ev.saturating_add(ORACLE_WOBBLE_TOLERANCE_WEI) >= exact_k8_ev,
        "adding a positive root-arb preserve seed should not underperform the same expanded cap without it"
    );
    assert!(
        exact_staged_seed_k8_ev >= exact_k4_ev,
        "staged-derived preserve universe should not underperform the baseline k4 exact solver on the heterogeneous oracle case"
    );
}

#[test]
#[ignore = "debug helper; run explicitly"]
fn print_heterogeneous_ninety_eight_variant_proposal_breakdown() {
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "heterogeneous_ninety_eight_outcome_l1_like_case")
        .expect("heterogeneous benchmark case should exist");
    let built = build_case(&case);
    let force_mint_available = !case.direct_only_reference;

    let staged_choice = staged_reference_choice_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    )
    .expect("staged reference should produce a candidate on the heterogeneous benchmark case");
    println!(
        "staged_choice variant={} frontier={:?} preserve={:?}",
        staged_choice.variant_label, staged_choice.frontier_family, staged_choice.preserve_markets
    );

    for variant in [
        TestPhaseOrderVariant::ArbFirst,
        TestPhaseOrderVariant::ArbLast,
        TestPhaseOrderVariant::NoArb,
    ] {
        let choice = phase_order_exact_subset_choice_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            variant,
            force_mint_available,
        )
        .expect("variant proposal helper should produce a candidate");
        let replayed = rebalance_with_custom_predictions_exact_no_arb_with_explicit_choice_for_test(
            &built.balances_view,
            built.cash_budget,
            &built.slot0_results,
            &built.predictions,
            force_mint_available,
            &choice.preserve_markets,
            choice.frontier_family,
        );
        println!(
            "variant={} frontier={:?} preserve={:?} exact_no_arb_replay_ev={} actions={}",
            choice.variant_label,
            choice.frontier_family,
            choice.preserve_markets,
            benchmark_ev_wei(&replayed, &built),
            replayed.len()
        );
    }
}

#[test]
#[ignore = "debug helper; run explicitly"]
fn print_heterogeneous_ninety_eight_case_family_breakdown() {
    let case = cases_from_fixture()
        .into_iter()
        .find(|case| case.case_id == "heterogeneous_ninety_eight_outcome_l1_like_case")
        .expect("heterogeneous benchmark case should exist");
    let built = build_case(&case);
    let force_mint_available = !case.direct_only_reference;

    let exact_no_arb = rebalance_with_custom_predictions_exact_no_arb_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let plain = rebalance_with_custom_predictions_plain_family_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let arb_primed = rebalance_with_custom_predictions_arb_primed_family_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let no_arb = rebalance_with_custom_predictions_rebalance_strict_no_arb_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let arb_first = rebalance_with_custom_predictions_arb_first_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let arb_last = rebalance_with_custom_predictions_arb_last_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let rebalance_only = rebalance_with_custom_predictions_rebalance_only_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let staged = rebalance_with_custom_predictions_staged_reference_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let ultimate = rebalance_with_custom_predictions_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        force_mint_available,
    );
    let (cyclic_ev, cyclic_actions, _) =
        arb_end_cyclic_result_for_built_case(&built, force_mint_available);

    println!(
        "exact_no_arb ev={} actions={}",
        benchmark_ev_wei(&exact_no_arb, &built),
        exact_no_arb.len()
    );
    println!(
        "plain ev={} actions={}",
        benchmark_ev_wei(&plain, &built),
        plain.len()
    );
    println!(
        "arb_primed ev={} actions={}",
        arb_primed
            .as_ref()
            .map(|actions| benchmark_ev_wei(actions, &built))
            .unwrap_or(0),
        arb_primed
            .as_ref()
            .map(|actions| actions.len())
            .unwrap_or(0)
    );
    println!(
        "no_arb ev={} actions={}",
        benchmark_ev_wei(&no_arb, &built),
        no_arb.len()
    );
    println!(
        "arb_first ev={} actions={}",
        benchmark_ev_wei(&arb_first, &built),
        arb_first.len()
    );
    println!(
        "arb_last ev={} actions={}",
        benchmark_ev_wei(&arb_last, &built),
        arb_last.len()
    );
    println!(
        "rebalance_only ev={} actions={}",
        benchmark_ev_wei(&rebalance_only, &built),
        rebalance_only.len()
    );
    println!(
        "staged ev={} actions={}",
        benchmark_ev_wei(&staged, &built),
        staged.len()
    );
    println!(
        "ultimate ev={} actions={}",
        benchmark_ev_wei(&ultimate, &built),
        ultimate.len()
    );
    println!("cyclic ev={} actions={}", cyclic_ev, cyclic_actions);
}

#[test]
#[ignore = "hypothesis helper; run explicitly"]
fn print_phase0_arb_start_vs_end_cyclic_hypothesis() {
    let cases = cases_from_fixture();
    let mut improved = 0usize;
    let mut tied = 0usize;
    let mut worse = 0usize;

    for case in &cases {
        let built = build_case(case);
        let force_mint_available = !case.direct_only_reference;
        let (start_arb_ev, start_action_count, start_held_stats) =
            start_arb_result_for_built_case(&built, force_mint_available);
        if let Some(stats) = start_held_stats {
            assert!(
                stats.spread <= HELD_PROFITABILITY_SPREAD_SANITY_TOL,
                "{} start-arb held-profitability spread too wide: spread={:.9}, min={:.9}, max={:.9}, held_count={}",
                case.case_id,
                stats.spread,
                stats.min_prof,
                stats.max_prof,
                stats.held_count
            );
        }

        let (end_arb_cycle_ev, end_action_count, end_held_stats) =
            arb_end_cyclic_result_for_built_case(&built, force_mint_available);
        if let Some(stats) = end_held_stats {
            assert!(
                stats.spread <= HELD_PROFITABILITY_SPREAD_SANITY_TOL,
                "{} end-arb-cyclic held-profitability spread too wide: spread={:.9}, min={:.9}, max={:.9}, held_count={}",
                case.case_id,
                stats.spread,
                stats.min_prof,
                stats.max_prof,
                stats.held_count
            );
        }

        let delta = end_arb_cycle_ev as i128 - start_arb_ev as i128;

        if delta > 0 {
            improved += 1;
        } else if delta < 0 {
            worse += 1;
        } else {
            tied += 1;
        }

        println!(
            "{}: start_arb_ev_wei={} end_arb_cycle_ev_wei={} end_minus_start_wei={} start_actions={} end_actions={} start_held_prof_spread={:.9} end_held_prof_spread={:.9}",
            case.case_id,
            start_arb_ev,
            end_arb_cycle_ev,
            delta,
            start_action_count,
            end_action_count,
            start_held_stats.map(|s| s.spread).unwrap_or(0.0),
            end_held_stats.map(|s| s.spread).unwrap_or(0.0)
        );
    }

    println!(
        "[arb-timing] improved={} tied={} worse={}",
        improved, tied, worse
    );
}

#[tokio::test]
#[ignore = "networked live snapshot helper; run explicitly"]
async fn write_live_l1_single_tick_benchmark_report() {
    let case_id = "live_full_l1_single_tick_snapshot_case".to_string();
    let Some(slot0_results) =
        super::execution::fetch_live_expected_l1_slot0_results("rebalancer_ab_live_l1_snapshot")
            .await
    else {
        return;
    };

    let preds = prediction_map();
    let mut predictions_wad = Vec::with_capacity(slot0_results.len());
    let mut is_token1 = Vec::with_capacity(slot0_results.len());
    let mut starting_prices_wad = Vec::with_capacity(slot0_results.len());
    let mut liquidity = Vec::with_capacity(slot0_results.len());
    let mut ticks = Vec::with_capacity(slot0_results.len());
    let initial_holdings_wad = vec![0u128; slot0_results.len()];
    let initial_cash_budget_wad = 100_000_000_000_000_000_000u128;
    let mut fallback_geometry_count = 0usize;
    let mut token0_outcome_count = 0usize;

    for (slot0, market) in &slot0_results {
        let pool = market
            .pool
            .as_ref()
            .expect("live benchmark markets must have pools");
        let orientation_is_token1 = pool.token1.eq_ignore_ascii_case(market.outcome_token);
        if !orientation_is_token1 {
            token0_outcome_count += 1;
        }

        let price_wad =
            sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, orientation_is_token1)
                .and_then(|value| u128::try_from(value).ok())
                .expect("live slot0 outcome price must fit in u128 WAD");
        let pred = preds
            .get(&normalize_market_name(market.name))
            .copied()
            .expect("live benchmark market must have a prediction");
        let (active_liquidity, tick_lo, tick_hi, used_fallback) =
            current_single_tick_geometry(slot0, *market)
                .expect("live benchmark market must have single-tick geometry");
        if used_fallback {
            fallback_geometry_count += 1;
        }

        predictions_wad.push(f64_to_wad(pred));
        is_token1.push(orientation_is_token1);
        starting_prices_wad.push(price_wad);
        liquidity.push(active_liquidity);
        ticks.push(vec![tick_lo, tick_hi]);
    }

    let spec = ExplicitCaseSpec {
        case_id: case_id.clone(),
        predictions_wad: predictions_wad.clone(),
        is_token1: is_token1.clone(),
        starting_prices_wad: starting_prices_wad.clone(),
        initial_holdings_wad: initial_holdings_wad.clone(),
        liquidity: liquidity.clone(),
        ticks: ticks.clone(),
        initial_cash_budget_wad,
    };
    let built = build_explicit_case_from_spec(&spec);
    let (direct_ev, mixed_ev, full_rebalance_only_ev, action_count) =
        current_result_for_built_case(&built, true);

    let report = LiveBenchmarkReport {
        case_id: case_id.clone(),
        predictions_wad,
        is_token1,
        starting_prices_wad,
        liquidity,
        ticks,
        initial_holdings_wad,
        initial_cash_budget_wad,
        fee_tier: crate::pools::FEE_PIPS as u64,
        offchain_direct_ev_wei: direct_ev,
        offchain_mixed_ev_wei: mixed_ev,
        offchain_full_rebalance_only_ev_wei: full_rebalance_only_ev,
        offchain_action_count: action_count,
    };

    let json = serde_json::to_string_pretty(&report)
        .expect("live benchmark report should serialize to valid JSON");
    let output_path = live_report_path();
    fs::write(&output_path, format!("{json}\n")).expect("failed to write live benchmark report");

    println!(
        "[ab-live] wrote={} markets={} token0_outcomes={} fallback_geometry={} offchain_direct_ev_wei={} offchain_full_rebalance_only_ev_wei={}",
        output_path,
        slot0_results.len(),
        token0_outcome_count,
        fallback_geometry_count,
        direct_ev,
        full_rebalance_only_ev
    );
}

#[test]
#[ignore = "search helper; run explicitly"]
fn sweep_randomized_rebalance_only_dominance_cases() {
    let case_count = std::env::var("AB_SWEEP_CASES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(500);
    let outcomes = std::env::var("AB_SWEEP_OUTCOMES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(4);
    let seed = std::env::var("AB_SWEEP_SEED")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(1);

    let mut rng = Lcg::new(seed);
    let mut improved = 0usize;
    let mut tied = 0usize;
    let mut worse = 0usize;
    let mut best_delta = i128::MIN;
    let mut best_spec: Option<ExplicitCaseSpec> = None;
    let mut best_direct = 0u128;
    let mut best_full = 0u128;

    for case_idx in 0..case_count {
        let spec = sample_random_case(&mut rng, case_idx, outcomes);
        let built = build_explicit_case_from_spec(&spec);
        let (direct_ev, _mixed_ev, full_rebalance_only_ev, _actions) =
            current_result_for_built_case(&built, true);
        let delta = full_rebalance_only_ev as i128 - direct_ev as i128;

        if delta > 0 {
            improved += 1;
        } else if delta < 0 {
            worse += 1;
        } else {
            tied += 1;
        }

        if delta > best_delta {
            best_delta = delta;
            best_spec = Some(spec);
            best_direct = direct_ev;
            best_full = full_rebalance_only_ev;
        }
    }

    println!(
        "[ab-sweep] cases={} outcomes={} seed={} improved={} tied={} worse={}",
        case_count, outcomes, seed, improved, tied, worse
    );

    if let Some(spec) = best_spec {
        println!(
            "[ab-sweep] best_case={} full_minus_direct_wei={} direct_ev={} full_rebalance_only_ev={}",
            spec.case_id, best_delta, best_direct, best_full
        );
        println!("[ab-sweep] best predictions_wad={:?}", spec.predictions_wad);
        println!(
            "[ab-sweep] best starting_prices_wad={:?}",
            spec.starting_prices_wad
        );
        println!(
            "[ab-sweep] best initial_holdings_wad={:?}",
            spec.initial_holdings_wad
        );
        println!("[ab-sweep] best liquidity={:?}", spec.liquidity);
        println!("[ab-sweep] best ticks={:?}", spec.ticks);
        println!(
            "[ab-sweep] best initial_cash_budget_wad={}",
            spec.initial_cash_budget_wad
        );
    }
}

#[test]
#[ignore = "hypothesis helper; run explicitly"]
fn sweep_phase0_arb_start_vs_end_cyclic_hypothesis() {
    let case_count = std::env::var("AB_SWEEP_CASES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(500);
    let outcomes = std::env::var("AB_SWEEP_OUTCOMES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(4);
    let seed = std::env::var("AB_SWEEP_SEED")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(1);

    let mut rng = Lcg::new(seed);
    let mut improved = 0usize;
    let mut tied = 0usize;
    let mut worse = 0usize;

    let mut best_delta = i128::MIN;
    let mut best_case_id = String::new();
    let mut best_start_ev = 0u128;
    let mut best_end_ev = 0u128;

    let mut worst_delta = i128::MAX;
    let mut worst_case_id = String::new();
    let mut worst_start_ev = 0u128;
    let mut worst_end_ev = 0u128;
    let mut max_start_spread = 0.0_f64;
    let mut max_end_spread = 0.0_f64;

    for case_idx in 0..case_count {
        let spec = sample_random_case(&mut rng, case_idx, outcomes);
        let built = build_explicit_case_from_spec(&spec);
        let (start_arb_ev, _start_action_count, start_held_stats) =
            start_arb_result_for_built_case(&built, true);
        if let Some(stats) = start_held_stats {
            max_start_spread = max_start_spread.max(stats.spread);
            assert!(
                stats.spread <= HELD_PROFITABILITY_SPREAD_SANITY_TOL,
                "{} start-arb held-profitability spread too wide: spread={:.9}, min={:.9}, max={:.9}, held_count={}",
                spec.case_id,
                stats.spread,
                stats.min_prof,
                stats.max_prof,
                stats.held_count
            );
        }

        let (end_arb_cycle_ev, _end_action_count, end_held_stats) =
            arb_end_cyclic_result_for_built_case(&built, true);
        if let Some(stats) = end_held_stats {
            max_end_spread = max_end_spread.max(stats.spread);
            assert!(
                stats.spread <= HELD_PROFITABILITY_SPREAD_SANITY_TOL,
                "{} end-arb-cyclic held-profitability spread too wide: spread={:.9}, min={:.9}, max={:.9}, held_count={}",
                spec.case_id,
                stats.spread,
                stats.min_prof,
                stats.max_prof,
                stats.held_count
            );
        }

        let delta = end_arb_cycle_ev as i128 - start_arb_ev as i128;

        if delta > 0 {
            improved += 1;
        } else if delta < 0 {
            worse += 1;
        } else {
            tied += 1;
        }

        if delta > best_delta {
            best_delta = delta;
            best_case_id = spec.case_id.clone();
            best_start_ev = start_arb_ev;
            best_end_ev = end_arb_cycle_ev;
        }
        if delta < worst_delta {
            worst_delta = delta;
            worst_case_id = spec.case_id.clone();
            worst_start_ev = start_arb_ev;
            worst_end_ev = end_arb_cycle_ev;
        }
    }

    println!(
        "[arb-timing-sweep] cases={} outcomes={} seed={} improved={} tied={} worse={} max_start_held_prof_spread={:.9} max_end_held_prof_spread={:.9}",
        case_count, outcomes, seed, improved, tied, worse, max_start_spread, max_end_spread
    );
    println!(
        "[arb-timing-sweep] best_case={} end_minus_start_wei={} start_arb_ev_wei={} end_arb_cycle_ev_wei={}",
        best_case_id, best_delta, best_start_ev, best_end_ev
    );
    println!(
        "[arb-timing-sweep] worst_case={} end_minus_start_wei={} start_arb_ev_wei={} end_arb_cycle_ev_wei={}",
        worst_case_id, worst_delta, worst_start_ev, worst_end_ev
    );
}
