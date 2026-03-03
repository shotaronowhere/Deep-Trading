use std::collections::HashMap;

use serde::Deserialize;

use super::super::rebalancer::rebalance_with_custom_predictions_for_test;
use super::fixtures::mock_slot0_market_with_liquidity_and_ticks;
use super::replay_actions_to_state;
use crate::pools::normalize_market_name;

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
    offchain_action_count: u64,
}

type Slot0Case = Vec<(
    crate::pools::Slot0Result,
    &'static crate::markets::MarketData,
)>;

struct BuiltCase {
    slot0_results: Slot0Case,
    balances_view: HashMap<&'static str, f64>,
    predictions: HashMap<String, f64>,
    cash_budget: f64,
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

fn build_explicit_case(
    case_id: &str,
    prices: &[f64],
    preds: &[f64],
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
        let (slot0, market) = mock_slot0_market_with_liquidity_and_ticks(
            name,
            token,
            prices[i],
            liquidities[i],
            tick_pair[0],
            tick_pair[1],
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

fn build_uniform_large_case(case: &BenchmarkCase) -> BuiltCase {
    let count = case.uniform_count;
    assert!(count > 0, "uniform_count must be positive");
    let pred = 1.0 / count as f64;
    let prices = vec![pred * (case.uniform_price_bps as f64) / 10_000.0; count];
    let preds = vec![pred; count];
    let holdings = vec![0.0; count];
    let liquidities = vec![case.uniform_liquidity; count];
    let ticks = vec![vec![case.uniform_tick_lo, case.uniform_tick_hi]; count];
    build_explicit_case(
        &case.case_id,
        &prices,
        &preds,
        &holdings,
        &liquidities,
        &ticks,
        wad_to_f64(case.initial_cash_budget_wad),
    )
}

fn build_case(case: &BenchmarkCase) -> BuiltCase {
    assert_eq!(
        case.fee_tier, 500,
        "benchmark harness currently assumes 5 bps pools"
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
            &holdings,
            &case.liquidity,
            &case.ticks,
            wad_to_f64(case.initial_cash_budget_wad),
        )
    }
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

fn current_result_for_case(case: &BenchmarkCase) -> (u128, u128, u64) {
    let built = build_case(case);

    let direct_actions = rebalance_with_custom_predictions_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        false,
    );
    let direct_ev = benchmark_ev_wei(&direct_actions, &built);

    let mixed_actions = rebalance_with_custom_predictions_for_test(
        &built.balances_view,
        built.cash_budget,
        &built.slot0_results,
        &built.predictions,
        !case.direct_only_reference,
    );
    let mixed_ev = benchmark_ev_wei(&mixed_actions, &built);

    (direct_ev, mixed_ev, mixed_actions.len() as u64)
}

#[test]
fn benchmark_snapshot_matches_current_optimizer() {
    let cases = cases_from_fixture();
    let expected = expected_from_fixture();
    let mut failures = Vec::new();

    for case in &cases {
        let expected_row = expected
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing snapshot row for {}", case.case_id));
        let (direct_ev, mixed_ev, action_count) = current_result_for_case(case);

        if direct_ev != expected_row.offchain_direct_ev_wei {
            failures.push(format!(
                "{} direct expected={} actual={}",
                case.case_id, expected_row.offchain_direct_ev_wei, direct_ev
            ));
        }
        if mixed_ev != expected_row.offchain_mixed_ev_wei {
            failures.push(format!(
                "{} mixed expected={} actual={}",
                case.case_id, expected_row.offchain_mixed_ev_wei, mixed_ev
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
