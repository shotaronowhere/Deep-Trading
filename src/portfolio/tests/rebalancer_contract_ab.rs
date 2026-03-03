use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs;

use serde::{Deserialize, Serialize};

use super::super::rebalancer::{
    arb_only_with_custom_predictions_for_test, rebalance_with_custom_predictions_for_test,
    rebalance_with_custom_predictions_rebalance_only_for_test,
    rebalance_with_custom_predictions_rebalance_strict_no_arb_for_test,
};
use super::fixtures::mock_slot0_market_with_orientation_liquidity_and_ticks;
use super::replay_actions_to_state;
use crate::pools::{
    normalize_market_name, prediction_map, prediction_to_sqrt_price_x96,
    sqrt_price_x96_to_price_outcome,
};
use crate::portfolio::core::sim::PoolSim;

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
    let actions = rebalance_with_custom_predictions_for_test(
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
        if mixed_ev != expected_row.offchain_mixed_ev_wei {
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
