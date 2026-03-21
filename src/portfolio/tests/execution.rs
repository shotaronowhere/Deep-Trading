use std::collections::{BTreeMap, HashMap, HashSet};

use super::super::merge::{
    execute_merge_sell, merge_sell_cap, merge_sell_proceeds, optimal_sell_split,
};
use super::super::rebalancer::{
    RebalanceMode, rebalance, rebalance_with_gas_pricing, rebalance_with_mode,
};
use super::super::sim::PoolSim;
use super::super::trading::{ExecutionState, solve_complete_set_arb_amount};
use super::{
    Action, assert_rebalance_action_invariants, assert_strict_ev_gain_with_portfolio_trace,
    brute_force_best_split, build_slot0_results_for_markets, build_three_sims,
    build_three_sims_with_preds, eligible_l1_markets_with_predictions, ev_from_state, leak_market,
    leak_pool, mock_slot0_market, print_rebalance_execution_summary,
    replay_actions_to_market_state, replay_actions_to_state,
};
use crate::execution::GroupKind;
use crate::execution::bounds::{
    BufferConfig, ConservativeExecutionConfig,
    build_group_plans_for_gas_replay_with_market_context, build_group_plans_from_cashflow,
    build_group_plans_with_default_edges, derive_batch_quote_bounds, stamp_plans_with_block,
};
use crate::execution::gas::{
    GasAssumptions, default_gas_assumptions_with_optimism_l1_fee,
    estimate_group_plan_l2_gas_units_live, fetch_live_optimism_fee_inputs,
    quote_group_plan_exact_gas_with_fee_inputs,
};
use crate::execution::grouping::{group_actions, group_execution_actions};
use crate::markets::{MarketData, Pool, Tick};
use crate::pools::{
    Slot0Result, normalize_market_name, prediction_map, prediction_to_sqrt_price_x96,
    sqrt_price_x96_to_price_outcome,
};
use alloy::primitives::{Address, I256, U256};
use uniswap_v3_math::{
    liquidity_math::add_delta,
    swap_math::compute_swap_step,
    tick_math::{MAX_TICK, MIN_TICK, get_sqrt_ratio_at_tick, get_tick_at_sqrt_ratio},
};

fn sample_rebalance_actions() -> Vec<Action> {
    let markets = eligible_l1_markets_with_predictions();
    let selected: Vec<_> = markets.into_iter().take(8).collect();
    let multipliers = [1.35, 0.55, 1.40, 0.60, 1.25, 0.70, 1.15, 0.75];
    let slot0_results = build_slot0_results_for_markets(&selected, &multipliers);
    rebalance(&HashMap::new(), 100.0, &slot0_results)
}

fn full_pooled_slot0_results_with_uniform_price(
    price: f64,
) -> Vec<(Slot0Result, &'static crate::markets::MarketData)> {
    crate::markets::MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .map(|market| {
            let pool = market.pool.as_ref().expect("pooled market required");
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let sqrt_price = prediction_to_sqrt_price_x96(price, is_token1_outcome)
                .unwrap_or(U256::from(1u128 << 96));
            (
                Slot0Result {
                    pool_id: Address::ZERO,
                    sqrt_price_x96: sqrt_price,
                    tick: 0,
                    observation_index: 0,
                    observation_cardinality: 0,
                    observation_cardinality_next: 0,
                    fee_protocol: 0,
                    unlocked: true,
                },
                market,
            )
        })
        .collect()
}

fn test_gas_assumptions() -> GasAssumptions {
    GasAssumptions {
        l1_fee_per_byte_wei: 1.0e11,
        ..GasAssumptions::default()
    }
}

fn gas_replay_assumptions() -> GasAssumptions {
    GasAssumptions {
        l1_data_fee_floor_susd: 0.0,
        l1_fee_per_byte_wei: 1.0,
        ..GasAssumptions::default()
    }
}

fn canonical_small_shape_actions() -> Vec<(&'static str, Vec<Action>)> {
    let markets = eligible_l1_markets_with_predictions();
    let selected: Vec<_> = markets.into_iter().take(3).collect();
    let market_a = selected[0].name;
    let market_b = selected[1].name;

    vec![
        (
            "DirectBuy",
            vec![Action::Buy {
                market_name: market_a,
                amount: 1.0,
                cost: 0.25,
            }],
        ),
        (
            "DirectSell",
            vec![Action::Sell {
                market_name: market_a,
                amount: 1.0,
                proceeds: 0.25,
            }],
        ),
        (
            "DirectMerge",
            vec![Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: market_a,
            }],
        ),
        (
            "MintSell(1)",
            vec![
                Action::Mint {
                    contract_1: "c1",
                    contract_2: "c2",
                    amount: 1.0,
                    target_market: market_a,
                },
                Action::Sell {
                    market_name: market_b,
                    amount: 1.0,
                    proceeds: 0.25,
                },
            ],
        ),
        (
            "BuyMerge(1)",
            vec![
                Action::Buy {
                    market_name: market_b,
                    amount: 1.0,
                    cost: 0.25,
                },
                Action::Merge {
                    contract_1: "c1",
                    contract_2: "c2",
                    amount: 1.0,
                    source_market: market_a,
                },
            ],
        ),
    ]
}

#[test]
fn gas_replay_helper_reconstructs_canonical_small_shapes() {
    let slot0_results = full_pooled_slot0_results_with_uniform_price(0.02);

    for (shape, actions) in canonical_small_shape_actions() {
        let replay = build_group_plans_for_gas_replay_with_market_context(
            &actions,
            &slot0_results,
            ConservativeExecutionConfig {
                quote_latency_blocks: 1,
                adverse_move_bps_per_block: 15,
            },
            &gas_replay_assumptions(),
            1e-9,
            3000.0,
        )
        .unwrap_or_else(|err| panic!("gas replay planning should succeed for {shape}: {err}"));
        assert_eq!(
            replay.plans.len(),
            1,
            "{} should produce exactly one replayable subgroup plan",
            shape
        );
        assert!(
            replay.skipped_groups.is_empty(),
            "{} should not skip replay groups: {:?}",
            shape,
            replay.skipped_groups
        );
    }
}

fn assert_flash_brackets_are_well_formed(actions: &[Action]) {
    group_execution_actions(actions)
        .expect("action stream should satisfy no-flash grouping shapes");
}

pub(super) async fn fetch_live_expected_l1_slot0_results(
    label: &str,
) -> Option<Vec<(Slot0Result, &'static crate::markets::MarketData)>> {
    dotenvy::dotenv().ok();
    let default_rpc = "https://optimism.drpc.org".to_string();
    let mut rpc_candidates: Vec<String> = Vec::new();
    if let Ok(rpc) = std::env::var("RPC") {
        rpc_candidates.push(rpc);
    }
    if !rpc_candidates.iter().any(|rpc| rpc == &default_rpc) {
        rpc_candidates.push(default_rpc.clone());
    }

    let expected_markets = eligible_l1_markets_with_predictions();
    let expected_market_names: HashSet<&'static str> =
        expected_markets.iter().map(|m| m.name).collect();

    let mut slot0_results_all: Option<Vec<(Slot0Result, &'static crate::markets::MarketData)>> =
        None;
    let mut last_err = String::new();
    for rpc_url in &rpc_candidates {
        let Ok(parsed_url) = rpc_url.parse() else {
            eprintln!(
                "[rebalance][{}] skipping invalid RPC URL: {}",
                label, rpc_url
            );
            continue;
        };
        let provider =
            alloy::providers::ProviderBuilder::new().with_reqwest(parsed_url, |builder| {
                builder
                    .no_proxy()
                    .build()
                    .expect("failed to build reqwest client for tests")
            });
        match crate::pools::fetch_all_slot0(provider).await {
            Ok(results) => {
                println!("[rebalance][{}] using RPC endpoint: {}", label, rpc_url);
                slot0_results_all = Some(results);
                break;
            }
            Err(err) => {
                last_err = err.to_string();
                eprintln!(
                    "[rebalance][{}] failed RPC endpoint {}: {}",
                    label, rpc_url, last_err
                );
            }
        }
    }

    let Some(slot0_results_all) = slot0_results_all else {
        eprintln!(
            "[rebalance][{}] skipping test: no reachable RPC endpoint; last error: {}",
            label, last_err
        );
        return None;
    };

    let slot0_results: Vec<_> = slot0_results_all
        .into_iter()
        .filter(|(_, market)| expected_market_names.contains(market.name))
        .collect();

    assert_eq!(
        slot0_results.len(),
        expected_markets.len(),
        "live slot0 fetch must include every tradeable L1 market with predictions"
    );

    Some(slot0_results)
}

fn live_subset_ranked_by_abs_gap(
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    count: usize,
    largest_first: bool,
) -> Vec<(Slot0Result, &'static crate::markets::MarketData)> {
    let preds = prediction_map();
    let mut ranked: Vec<_> = slot0_results
        .iter()
        .filter_map(|(slot0, market)| {
            let pool = market.pool.as_ref()?;
            let pred = *preds.get(&normalize_market_name(market.name))?;
            let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
            let current_price =
                sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome)?;
            let current = current_price.to_string().parse::<f64>().ok()? / 1e18;
            let gap = (current - pred).abs();
            Some((gap, current, pred, slot0.clone(), *market))
        })
        .collect();

    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if largest_first {
        ranked.reverse();
    }

    ranked
        .into_iter()
        .take(count)
        .map(|(gap, current, pred, slot0, market)| {
            println!(
                "[rebalance][live_subset] selected {} current={:.6} pred={:.6} abs_gap={:.6}",
                market.name, current, pred, gap
            );
            (slot0, market)
        })
        .collect()
}

fn live_subset_ranked_by_buy_gap(
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    count: usize,
) -> Vec<(Slot0Result, &'static crate::markets::MarketData)> {
    let preds = prediction_map();
    let mut ranked: Vec<_> = slot0_results
        .iter()
        .filter_map(|(slot0, market)| {
            let pool = market.pool.as_ref()?;
            let pred = *preds.get(&normalize_market_name(market.name))?;
            let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
            let current_price =
                sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome)?;
            let current = current_price.to_string().parse::<f64>().ok()? / 1e18;
            let buy_gap = pred - current;
            (buy_gap > 0.0).then_some((buy_gap, current, pred, slot0.clone(), *market))
        })
        .collect();

    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    ranked
        .into_iter()
        .take(count)
        .map(|(buy_gap, current, pred, slot0, market)| {
            println!(
                "[rebalance][live_subset] selected {} current={:.6} pred={:.6} buy_gap={:.6}",
                market.name, current, pred, buy_gap
            );
            (slot0, market)
        })
        .collect()
}

fn assert_live_subset_rebalance_non_decreasing_ev(
    label: &str,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_susd: f64,
) {
    let initial_balances: HashMap<&str, f64> = HashMap::new();
    let actions = rebalance(&initial_balances, initial_susd, slot0_results);
    let actions_repeat = rebalance(&initial_balances, initial_susd, slot0_results);

    assert_eq!(
        format!("{:?}", actions),
        format!("{:?}", actions_repeat),
        "rebalance should be deterministic for a fixed live snapshot"
    );

    assert_rebalance_action_invariants(&actions, slot0_results, &initial_balances, initial_susd);
    print_rebalance_execution_summary(label, &actions, slot0_results);

    let (final_holdings, final_cash) =
        replay_actions_to_state(&actions, slot0_results, &initial_balances, initial_susd);
    let ev_after = ev_from_state(&final_holdings, final_cash);
    let ev_gain = ev_after - initial_susd;
    println!(
        "[rebalance][{}] expected value: before={:.9}, after={:.9}, gain={:.9}",
        label, initial_susd, ev_after, ev_gain
    );

    assert!(
        ev_after + 1e-6 >= initial_susd,
        "{} should not reduce expected value on a fixed live snapshot: before={:.9}, after={:.9}",
        label,
        initial_susd,
        ev_after
    );
}

#[tokio::test]
#[ignore = "live OP gas-aware report helper; run explicitly"]
async fn print_live_op_first_group_exact_gas_report() {
    let rpc_url = std::env::var("RPC").unwrap_or_else(|_| "https://optimism.drpc.org".to_string());
    let Some(slot0_results) =
        fetch_live_expected_l1_slot0_results("live_op_exact_gas_report").await
    else {
        return;
    };
    let gas_assumptions = default_gas_assumptions_with_optimism_l1_fee(&rpc_url)
        .await
        .unwrap_or_else(|_| GasAssumptions::default());
    let initial_balances: HashMap<&str, f64> = HashMap::new();
    let actions = rebalance_with_gas_pricing(
        &initial_balances,
        100.0,
        &slot0_results,
        RebalanceMode::Full,
        &gas_assumptions,
        1e-9,
        3000.0,
    );
    assert!(
        !actions.is_empty(),
        "live gas-aware report requires at least one planned action"
    );
    let replay = build_group_plans_for_gas_replay_with_market_context(
        &actions,
        &slot0_results,
        ConservativeExecutionConfig {
            quote_latency_blocks: 1,
            adverse_move_bps_per_block: 15,
        },
        &gas_replay_assumptions(),
        1e-9,
        3000.0,
    )
    .expect("gas replay plans should build for live gas report");
    if replay.plans.is_empty() {
        println!(
            "{}",
            serde_json::json!({
                "planned_action_count": actions.len(),
                "planned_groups": 0,
                "skipped_groups": replay.skipped_groups.len(),
                "note": "no executable group plan under current live snapshot",
            })
        );
        return;
    }
    let mut plans = replay.plans;
    stamp_plans_with_block(&mut plans, 0);
    let first = plans.first().expect("first group plan should exist");
    let fee_inputs = fetch_live_optimism_fee_inputs(&rpc_url, Address::ZERO)
        .await
        .expect("live OP fee inputs should resolve");
    let executor = Address::repeat_byte(0x44);
    let exact_quote = quote_group_plan_exact_gas_with_fee_inputs(
        &rpc_url, executor, &actions, first, fee_inputs, 3000.0,
    )
    .await
    .expect("exact gas quote should succeed");
    let live_l2_gas_units =
        estimate_group_plan_l2_gas_units_live(&rpc_url, Address::ZERO, executor, &actions, first)
            .await
            .expect("live eth_estimateGas should succeed");

    println!(
        "{}",
        serde_json::json!({
            "planned_action_count": actions.len(),
            "planned_groups": plans.len(),
            "skipped_groups": replay.skipped_groups.len(),
            "first_group_kind": format!("{:?}", first.kind),
            "first_group_action_count": first.action_indices.len(),
            "unsigned_tx_bytes": exact_quote.unsigned_tx_data.len(),
            "gas_price_wei": exact_quote.gas_price_wei.to_string(),
            "calibrated_l2_gas_units": exact_quote.l2_gas_units,
            "live_estimated_l2_gas_units": live_l2_gas_units,
            "l2_fee_wei": exact_quote.l2_fee_wei.to_string(),
            "l1_fee_wei": exact_quote.l1_fee_wei.to_string(),
            "l2_fee_susd": exact_quote.l2_fee_susd,
            "l1_fee_susd": exact_quote.l1_fee_susd,
            "total_fee_susd": exact_quote.total_fee_susd,
            "net_ev_susd": exact_quote.net_ev_susd,
        })
    );
}

#[tokio::test]
#[ignore = "live OP canonical shape L1 fee calibration helper; run explicitly"]
async fn print_live_op_canonical_small_shape_l1_fee_floor_calibration() {
    let rpc_url = std::env::var("RPC").unwrap_or_else(|_| "https://optimism.drpc.org".to_string());
    let fee_inputs = fetch_live_optimism_fee_inputs(&rpc_url, Address::ZERO)
        .await
        .expect("live OP fee inputs should resolve");
    let executor = Address::repeat_byte(0x44);
    let slot0_results = full_pooled_slot0_results_with_uniform_price(0.02);
    let mut max_total_fee_susd = 0.0_f64;

    for (shape, actions) in canonical_small_shape_actions() {
        let mut replay = build_group_plans_for_gas_replay_with_market_context(
            &actions,
            &slot0_results,
            ConservativeExecutionConfig {
                quote_latency_blocks: 1,
                adverse_move_bps_per_block: 15,
            },
            &gas_replay_assumptions(),
            fee_inputs.gas_price_wei as f64 / 1e18,
            3000.0,
        )
        .unwrap_or_else(|err| panic!("gas replay planning should succeed for {shape}: {err}"));
        assert_eq!(
            replay.plans.len(),
            1,
            "{} should produce exactly one replayable subgroup plan",
            shape
        );
        assert!(
            replay.skipped_groups.is_empty(),
            "{} should not skip replay groups: {:?}",
            shape,
            replay.skipped_groups
        );
        stamp_plans_with_block(&mut replay.plans, 0);
        let plan = replay
            .plans
            .first()
            .expect("canonical shape replay plan should exist");
        let exact_quote = quote_group_plan_exact_gas_with_fee_inputs(
            &rpc_url, executor, &actions, plan, fee_inputs, 3000.0,
        )
        .await
        .expect("exact gas quote should succeed for canonical shape");
        max_total_fee_susd = max_total_fee_susd.max(exact_quote.total_fee_susd);
        println!(
            "{}",
            serde_json::json!({
                "shape": shape,
                "unsigned_tx_bytes": exact_quote.unsigned_tx_data.len(),
                "gas_price_wei": exact_quote.gas_price_wei.to_string(),
                "l1_fee_wei": exact_quote.l1_fee_wei.to_string(),
                "l1_fee_susd": exact_quote.l1_fee_susd,
                "l2_gas_units": exact_quote.l2_gas_units,
                "l2_fee_susd": exact_quote.l2_fee_susd,
                "total_fee_susd": exact_quote.total_fee_susd,
            })
        );
    }

    let recommended_floor = (max_total_fee_susd * 2.0 * 1000.0).ceil() / 1000.0;
    println!(
        "{}",
        serde_json::json!({
            "recommended_l1_data_fee_floor_susd": recommended_floor,
            "max_small_shape_total_fee_susd": max_total_fee_susd,
        })
    );
}

#[derive(Clone)]
struct ExactReplayPoolState {
    pool: Pool,
    quote_token: &'static str,
    outcome_token: &'static str,
    sqrt_price_x96: U256,
    liquidity: u128,
    ticks: Vec<Tick>,
}

struct ExactReplayResult {
    holdings: HashMap<&'static str, f64>,
    cash: f64,
    crossed_initialized_ticks: usize,
}

fn amount_to_wei(amount: f64) -> Option<u128> {
    if !amount.is_finite() || amount <= 0.0 {
        return Some(0);
    }
    let scaled = (amount * 1e18).round();
    if !scaled.is_finite() || scaled <= 0.0 || scaled > u128::MAX as f64 {
        return None;
    }
    Some(scaled as u128)
}

fn merge_and_sort_ticks(pool: &Pool) -> Vec<Tick> {
    let mut merged: BTreeMap<i32, i128> = BTreeMap::new();
    for tick in pool.ticks {
        *merged.entry(tick.tick_idx).or_insert(0) += tick.liquidity_net;
    }
    merged
        .into_iter()
        .filter_map(|(tick_idx, liquidity_net)| {
            (liquidity_net != 0).then_some(Tick {
                tick_idx,
                liquidity_net,
            })
        })
        .collect()
}

fn next_initialized_tick(
    ticks: &[Tick],
    current_tick: i32,
    zero_for_one: bool,
    at_boundary: bool,
) -> Option<Tick> {
    if zero_for_one {
        let cutoff = if at_boundary {
            current_tick.saturating_sub(1)
        } else {
            current_tick
        };
        ticks
            .iter()
            .rev()
            .copied()
            .find(|tick| tick.tick_idx <= cutoff)
    } else {
        ticks
            .iter()
            .copied()
            .find(|tick| tick.tick_idx > current_tick)
    }
}

impl ExactReplayPoolState {
    fn from_slot0(slot0: &Slot0Result, market: &'static MarketData) -> Option<Self> {
        let pool = market.pool?;
        let liquidity = pool.liquidity.parse::<u128>().ok()?;
        Some(Self {
            pool,
            quote_token: market.quote_token,
            outcome_token: market.outcome_token,
            sqrt_price_x96: slot0.sqrt_price_x96,
            liquidity,
            ticks: merge_and_sort_ticks(&pool),
        })
    }

    fn swap_exact_input(
        &mut self,
        input_amount: f64,
        zero_for_one: bool,
    ) -> Option<(f64, f64, usize)> {
        let raw_input = amount_to_wei(input_amount)?;
        if raw_input == 0 || self.liquidity == 0 {
            return Some((0.0, 0.0, 0));
        }

        let mut remaining = I256::try_from(raw_input).ok()?;
        let mut total_in = U256::ZERO;
        let mut total_out = U256::ZERO;
        let mut crossed_initialized_ticks = 0usize;

        while remaining > I256::ZERO && self.liquidity > 0 {
            let current_tick = get_tick_at_sqrt_ratio(self.sqrt_price_x96).ok()?;
            let at_boundary =
                get_sqrt_ratio_at_tick(current_tick).ok() == Some(self.sqrt_price_x96);
            let next_tick =
                next_initialized_tick(&self.ticks, current_tick, zero_for_one, at_boundary);
            let target_tick = next_tick
                .map(|tick| tick.tick_idx)
                .unwrap_or(if zero_for_one { MIN_TICK } else { MAX_TICK });
            let target_sqrt = get_sqrt_ratio_at_tick(target_tick).ok()?;
            let (sqrt_next, amount_in, amount_out, fee_amount) = compute_swap_step(
                self.sqrt_price_x96,
                target_sqrt,
                self.liquidity,
                remaining,
                crate::pools::FEE_PIPS,
            )
            .ok()?;

            let step_total_in = amount_in + fee_amount;
            if step_total_in == U256::ZERO && amount_out == U256::ZERO {
                break;
            }

            total_in += step_total_in;
            total_out += amount_out;
            self.sqrt_price_x96 = sqrt_next;

            let step_in_i256 = I256::try_from(step_total_in.to::<u128>()).ok()?;
            if step_in_i256 >= remaining {
                remaining = I256::ZERO;
            } else {
                remaining -= step_in_i256;
            }

            if sqrt_next != target_sqrt {
                break;
            }

            let Some(crossed_tick) = next_tick else {
                break;
            };

            let liquidity_delta = if zero_for_one {
                crossed_tick.liquidity_net.checked_neg()?
            } else {
                crossed_tick.liquidity_net
            };
            self.liquidity = add_delta(self.liquidity, liquidity_delta).ok()?;
            crossed_initialized_ticks += 1;
        }

        Some((
            crate::pools::u256_to_f64(total_in),
            crate::pools::u256_to_f64(total_out),
            crossed_initialized_ticks,
        ))
    }

    fn buy_exact_input(&mut self, quote_amount: f64) -> Option<(f64, f64, usize)> {
        let zero_for_one = self.pool.token0.eq_ignore_ascii_case(self.quote_token);
        self.swap_exact_input(quote_amount, zero_for_one)
    }

    fn sell_exact_input(&mut self, outcome_amount: f64) -> Option<(f64, f64, usize)> {
        let zero_for_one = self.pool.token0.eq_ignore_ascii_case(self.outcome_token);
        self.swap_exact_input(outcome_amount, zero_for_one)
    }
}

fn replay_actions_to_state_multitick_exact(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
) -> ExactReplayResult {
    let mut holdings: HashMap<&'static str, f64> = HashMap::new();
    let mut pools: HashMap<&'static str, ExactReplayPoolState> = HashMap::new();
    for (slot0, market) in slot0_results {
        holdings.insert(
            market.name,
            initial_balances
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0),
        );
        if let Some(state) = ExactReplayPoolState::from_slot0(slot0, *market) {
            pools.insert(market.name, state);
        }
    }

    let mut cash = initial_susd.max(0.0);
    let mut crossed_initialized_ticks = 0usize;

    for action in actions {
        match action {
            Action::Buy {
                market_name, cost, ..
            } => {
                let spend = (*cost).min(cash.max(0.0));
                if spend <= 0.0 {
                    continue;
                }
                let Some(pool) = pools.get_mut(market_name) else {
                    continue;
                };
                let Some((actual_spend, received, crossed)) = pool.buy_exact_input(spend) else {
                    continue;
                };
                cash -= actual_spend;
                *holdings.entry(*market_name).or_insert(0.0) += received;
                crossed_initialized_ticks += crossed;
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                let available = holdings.get(market_name).copied().unwrap_or(0.0).max(0.0);
                let sell_amount = (*amount).min(available);
                if sell_amount <= 0.0 {
                    continue;
                }
                let Some(pool) = pools.get_mut(market_name) else {
                    continue;
                };
                let Some((sold, proceeds, crossed)) = pool.sell_exact_input(sell_amount) else {
                    continue;
                };
                *holdings.entry(*market_name).or_insert(0.0) -= sold;
                cash += proceeds;
                crossed_initialized_ticks += crossed;
            }
            Action::Mint { amount, .. } => {
                let mint_amount = (*amount).min(cash.max(0.0));
                if mint_amount <= 0.0 {
                    continue;
                }
                for (_, market) in slot0_results {
                    *holdings.entry(market.name).or_insert(0.0) += mint_amount;
                }
                cash -= mint_amount;
            }
            Action::Merge { amount, .. } => {
                let merge_amount = slot0_results
                    .iter()
                    .map(|(_, market)| holdings.get(market.name).copied().unwrap_or(0.0).max(0.0))
                    .fold(f64::INFINITY, f64::min)
                    .min(*amount);
                if !merge_amount.is_finite() || merge_amount <= 0.0 {
                    continue;
                }
                for (_, market) in slot0_results {
                    *holdings.entry(market.name).or_insert(0.0) -= merge_amount;
                }
                cash += merge_amount;
            }
        }
    }

    ExactReplayResult {
        holdings,
        cash,
        crossed_initialized_ticks,
    }
}

fn normalized_full_l1_price_multipliers(markets: &[&'static MarketData]) -> Vec<f64> {
    let preds = prediction_map();
    let mut rng = super::TestRng::new(0x5eed_98_1f);
    let raw: Vec<f64> = markets.iter().map(|_| rng.in_range(0.72, 1.28)).collect();
    let weighted_sum: f64 = markets
        .iter()
        .zip(raw.iter())
        .map(|(market, mult)| {
            let pred = preds
                .get(&normalize_market_name(market.name))
                .copied()
                .expect("eligible L1 market must have prediction");
            pred * *mult
        })
        .sum();

    raw.into_iter().map(|mult| mult / weighted_sum).collect()
}

fn pick_injected_liquidity_band(
    current_tick: i32,
    final_tick: i32,
    rng: &mut super::TestRng,
) -> Option<(i32, i32)> {
    if current_tick == final_tick {
        return None;
    }

    let delta = final_tick - current_tick;
    let distance = delta.unsigned_abs() as i32;
    if distance < 6 {
        return None;
    }

    let start_frac = rng.in_range(0.18, 0.35);
    let width_frac = rng.in_range(0.08, 0.20);
    let mut offset = ((distance as f64) * start_frac).round() as i32;
    let mut width = ((distance as f64) * width_frac).round() as i32;
    offset = offset.clamp(1, distance.saturating_sub(2));
    width = width.clamp(2, distance.saturating_sub(offset));

    if delta < 0 {
        let band_hi = current_tick - offset;
        let band_lo = band_hi - width;
        Some((band_lo.min(band_hi), band_lo.max(band_hi)))
    } else {
        let band_lo = current_tick + offset;
        let band_hi = band_lo + width;
        Some((band_lo.min(band_hi), band_lo.max(band_hi)))
    }
}

fn significant_liquidity_delta(base_liquidity: u128, rng: &mut super::TestRng) -> u128 {
    let scale_bps = 15_000u128 + (rng.next_u64() % 20_001) as u128;
    let scaled = base_liquidity.saturating_mul(scale_bps) / 10_000;
    scaled.max(base_liquidity / 2).max(1)
}

fn build_augmented_multitick_fixture(
    slot0_results: &[(Slot0Result, &'static MarketData)],
    actions: &[Action],
) -> Vec<(Slot0Result, &'static MarketData)> {
    let post_trade_market_state = replay_actions_to_market_state(actions, slot0_results);
    let final_slot0_by_name: HashMap<&'static str, Slot0Result> = post_trade_market_state
        .into_iter()
        .map(|(slot0, market)| (market.name, slot0))
        .collect();
    let buy_markets: HashSet<&'static str> = actions
        .iter()
        .filter_map(|action| match action {
            Action::Buy { market_name, .. } => Some(*market_name),
            Action::Sell { .. } | Action::Mint { .. } | Action::Merge { .. } => None,
        })
        .collect();

    let mut rng = super::TestRng::new(0xa11c_e98);
    let mut injected = 0usize;
    let mut augmented = Vec::with_capacity(slot0_results.len());

    for (slot0, market) in slot0_results {
        let Some(pool) = market.pool.as_ref() else {
            augmented.push((slot0.clone(), *market));
            continue;
        };
        if !buy_markets.contains(market.name) {
            augmented.push((slot0.clone(), *market));
            continue;
        }

        let Some(final_slot0) = final_slot0_by_name.get(market.name) else {
            augmented.push((slot0.clone(), *market));
            continue;
        };
        let Ok(current_tick) = get_tick_at_sqrt_ratio(slot0.sqrt_price_x96) else {
            augmented.push((slot0.clone(), *market));
            continue;
        };
        let Ok(final_tick) = get_tick_at_sqrt_ratio(final_slot0.sqrt_price_x96) else {
            augmented.push((slot0.clone(), *market));
            continue;
        };
        let Some((band_lo, band_hi)) =
            pick_injected_liquidity_band(current_tick, final_tick, &mut rng)
        else {
            augmented.push((slot0.clone(), *market));
            continue;
        };

        let Ok(base_liquidity) = pool.liquidity.parse::<u128>() else {
            augmented.push((slot0.clone(), *market));
            continue;
        };
        let extra_liquidity = significant_liquidity_delta(base_liquidity, &mut rng);
        let extra_i128 =
            i128::try_from(extra_liquidity.min(i128::MAX as u128)).expect("extra liquidity fits");

        let delta_lo;
        let delta_hi;
        if final_tick < current_tick {
            // zeroForOne buy path: crossing down enters at upper bound, exits at lower bound.
            delta_lo = extra_i128;
            delta_hi = -extra_i128;
        } else {
            // oneForZero buy path: crossing up enters at lower bound, exits at upper bound.
            delta_lo = extra_i128;
            delta_hi = -extra_i128;
        }

        let mut augmented_ticks = pool.ticks.to_vec();
        augmented_ticks.push(Tick {
            tick_idx: band_lo,
            liquidity_net: delta_lo,
        });
        augmented_ticks.push(Tick {
            tick_idx: band_hi,
            liquidity_net: delta_hi,
        });
        let leaked_ticks = Box::leak(augmented_ticks.into_boxed_slice());

        let mut augmented_pool = *pool;
        augmented_pool.ticks = leaked_ticks;
        let leaked_pool = leak_pool(augmented_pool);
        let leaked_market = leak_market(MarketData {
            name: market.name,
            market_id: market.market_id,
            outcome_token: market.outcome_token,
            pool: Some(*leaked_pool),
            quote_token: market.quote_token,
        });

        injected += 1;
        augmented.push((slot0.clone(), leaked_market));
    }

    assert!(
        injected >= 1,
        "expected at least one buy market with a traversable price range for injected liquidity"
    );

    augmented
}

#[test]
fn test_optimal_sell_split_matches_bruteforce() {
    let source_prices = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18];
    let other_prices = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16];
    let sell_amount = 5.0;

    for &p0 in &source_prices {
        for &p1 in &other_prices {
            for &p2 in &other_prices {
                let sims = build_three_sims([p0, p1, p2]);
                let upper = merge_sell_cap(&sims, 0).min(sell_amount);
                if upper <= 1e-9 {
                    continue;
                }

                let (grid_m, grid_total) = brute_force_best_split(&sims, 0, sell_amount, 4000);
                let (opt_m, opt_total) = optimal_sell_split(&sims, 0, sell_amount);

                // Solver should match brute-force objective very closely.
                assert!(
                    (opt_total - grid_total).abs() <= 5e-5,
                    "split solver mismatch: p0={:.3}, p1={:.3}, p2={:.3}, grid_m={:.6}, opt_m={:.6}, grid={:.9}, opt={:.9}",
                    p0,
                    p1,
                    p2,
                    grid_m,
                    opt_m,
                    grid_total,
                    opt_total
                );

                assert!(
                    opt_m >= -1e-9 && opt_m <= upper + 1e-9,
                    "optimal merge amount out of bounds: p0={:.3}, p1={:.3}, p2={:.3}, opt_m={:.9}, upper={:.9}",
                    p0,
                    p1,
                    p2,
                    opt_m,
                    upper
                );
            }
        }
    }
}

#[test]
fn test_arb_only_mint_sell_output_is_groupable_and_plannable_from_cashflow() {
    let slot0_results = full_pooled_slot0_results_with_uniform_price(0.02);
    let actions = rebalance_with_mode(
        &HashMap::new(),
        100.0,
        &slot0_results,
        RebalanceMode::ArbOnly,
    );
    assert!(
        !actions.is_empty(),
        "arb-only mode should emit actions in a profitable mint-sell setup"
    );

    let groups = group_actions(&actions).expect("arb-only action stream should be groupable");
    assert!(
        !groups.is_empty(),
        "expected at least one grouped mint-sell step"
    );
    assert!(groups.iter().all(|g| g.kind == GroupKind::MintSell));

    let plans = build_group_plans_from_cashflow(
        &actions,
        &test_gas_assumptions(),
        1e-8,
        3000.0,
        BufferConfig::default(),
    )
    .expect("mint-sell arb stream should be plannable from cashflow");
    assert!(
        plans.is_empty() || plans.iter().all(|p| p.kind == GroupKind::MintSell),
        "any generated plans should stay within mint-sell group kind"
    );
}

#[test]
fn test_merge_sell_single_pool_is_disabled() {
    let (slot0, market) =
        mock_slot0_market("M1", "0x1111111111111111111111111111111111111111", 0.4);
    let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.3).unwrap()];

    let cap = merge_sell_cap(&sims, 0);
    assert!(
        cap == 0.0,
        "merge cap should be zero when no non-source pools exist"
    );

    let (net, actual) = merge_sell_proceeds(&sims, 0, 5.0);
    assert!(net == 0.0 && actual == 0.0, "merge should be infeasible");

    let mut budget = 10.0;
    let mut actions = Vec::new();
    let merged = execute_merge_sell(&mut sims, 0, 5.0, &mut actions, &mut budget);
    assert_eq!(merged, 0.0, "execution should not merge");
    assert_eq!(budget, 10.0, "budget must remain unchanged");
    assert!(actions.is_empty(), "no actions should be emitted");
}

#[test]
fn test_execute_optimal_sell_uses_inventory_for_merge() {
    let mut sims = build_three_sims([0.8, 0.05, 0.05]);
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("M1", 5.0);
    sim_balances.insert("M2", 5.0);
    sim_balances.insert("M3", 5.0);

    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell(0, 5.0, f64::INFINITY, true)
    };

    assert!((sold - 5.0).abs() < 1e-9, "should sell full source amount");
    assert!(
        (budget - 5.0).abs() < 1e-9,
        "full inventory merge should recover full 1 sUSD per token"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "should include merge action"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "should not buy complements when inventory covers all merge legs"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "no pool buys needed when inventory covers all merge legs"
    );

    assert!((*sim_balances.get("M1").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M2").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M3").unwrap() - 0.0).abs() < 1e-9);
    assert!(
        (sims[1].price() - 0.05).abs() < 1e-9,
        "no buy => no price move"
    );
    assert!(
        (sims[2].price() - 0.05).abs() < 1e-9,
        "no buy => no price move"
    );
}

#[test]
fn test_execute_optimal_sell_buys_only_shortfall() {
    let mut sims = build_three_sims([0.8, 0.05, 0.05]);
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("M1", 5.0);
    sim_balances.insert("M2", 2.0);
    sim_balances.insert("M3", 7.0);

    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell(0, 5.0, f64::INFINITY, true)
    };

    assert!((sold - 5.0).abs() < 1e-9, "should sell full source amount");
    let buys: Vec<_> = actions
        .iter()
        .filter_map(|a| match a {
            Action::Buy {
                market_name,
                amount,
                ..
            } => Some((*market_name, *amount)),
            _ => None,
        })
        .collect();
    let mut buy_totals: HashMap<&str, f64> = HashMap::new();
    for (market, amount) in buys {
        *buy_totals.entry(market).or_insert(0.0) += amount;
    }
    assert_eq!(buy_totals.len(), 1, "should only buy shortfall leg");
    assert!(
        buy_totals.contains_key("M2"),
        "M2 had the shortfall and should be the only bought leg"
    );
    let m2_bought = buy_totals.get("M2").copied().unwrap_or(0.0);
    assert!(
        (m2_bought - 3.0).abs() < 1e-6,
        "shortfall should be 3 tokens"
    );
    assert!(
        budget > 0.0,
        "merge with partial inventory should still recover budget"
    );

    assert!((*sim_balances.get("M1").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M2").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M3").unwrap() - 2.0).abs() < 1e-9);
}

#[test]
fn test_execute_optimal_sell_allows_direct_merge_when_buy_merge_is_blocked() {
    let mut sims = build_three_sims_with_preds([0.8, 0.05, 0.05], [0.3, 0.01, 0.01]);
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("M1", 5.0);
    sim_balances.insert("M2", 5.0);
    sim_balances.insert("M3", 5.0);

    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell_with_merge_gates(
            0, 5.0, 0.0, true, false, // buy-merge blocked
            true,  // direct-merge allowed
        )
    };

    assert!((sold - 5.0).abs() < 1e-9, "should sell full source amount");
    assert!(
        actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "direct-merge route should still execute when buy-merge is blocked"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "no buy legs should be emitted for direct-merge-only liquidation"
    );
}

#[test]
fn test_execute_optimal_sell_keeps_profitable_complement_inventory() {
    let mut sims = build_three_sims([0.8, 0.05, 0.05]);
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("M1", 5.0);
    sim_balances.insert("M2", 5.0);
    sim_balances.insert("M3", 5.0);

    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell(
            0, 5.0, 0.0, // phase-1 behavior: preserve profitable inventory
            true,
        )
    };

    assert!((sold - 5.0).abs() < 1e-9, "should sell full source amount");
    assert!(
        actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "merge should still be used when economically optimal"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "keeping profitable inventory forces pool buys for merge legs"
    );
    let buys: Vec<_> = actions
        .iter()
        .filter_map(|a| match a {
            Action::Buy {
                market_name,
                amount,
                ..
            } => Some((*market_name, *amount)),
            _ => None,
        })
        .collect();
    let mut buy_totals: HashMap<&str, f64> = HashMap::new();
    for (market, amount) in buys {
        *buy_totals.entry(market).or_insert(0.0) += amount;
    }
    assert_eq!(
        buy_totals.len(),
        2,
        "both complementary legs should be bought"
    );
    assert!(
        buy_totals.contains_key("M2") && buy_totals.contains_key("M3"),
        "merge complements should buy both M2 and M3"
    );
    assert!(
        buy_totals.get("M2").copied().unwrap_or(0.0) > 4.9
            && buy_totals.get("M3").copied().unwrap_or(0.0) > 4.9
    );

    assert!(
        budget < 5.0 - 1e-6,
        "keeping profitable inventory should reduce immediate merge proceeds vs free-consume case"
    );
    assert!((*sim_balances.get("M1").unwrap() - 0.0).abs() < 1e-9);
    assert!(
        (*sim_balances.get("M2").unwrap() - 5.0).abs() < 1e-9,
        "profitable complement inventory should be preserved"
    );
    assert!(
        (*sim_balances.get("M3").unwrap() - 5.0).abs() < 1e-9,
        "profitable complement inventory should be preserved"
    );
}

#[test]
fn test_execute_optimal_sell_consumes_low_profit_complements() {
    let mut sims = build_three_sims_with_preds([0.8, 0.05, 0.05], [0.3, 0.01, 0.01]);
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("M1", 5.0);
    sim_balances.insert("M2", 5.0);
    sim_balances.insert("M3", 5.0);

    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell(
            0, 5.0, 0.0, // complements are unprofitable, so inventory is consumable
            true,
        )
    };

    assert!((sold - 5.0).abs() < 1e-9, "should sell full source amount");
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "consumable inventory should avoid unnecessary buy legs"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "consumable inventory should avoid pool buys"
    );
    assert!(
        (budget - 5.0).abs() < 1e-9,
        "using low-profit complements should recover full merge value"
    );
    assert!((*sim_balances.get("M1").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M2").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M3").unwrap() - 0.0).abs() < 1e-9);
}

#[test]
fn test_merge_route_sells() {
    // 3 outcomes: M1 at price 0.8 (overpriced, pred=0.3), M2 and M3 at price 0.05 each.
    // Merge sell of M1: buy M2+M3 (cheap), merge → 1 sUSD. Cost ≈ 0.05/0.9999 × 2 ≈ 0.10.
    // Net merge proceeds per token ≈ 1 - 0.10 = 0.90.
    // Direct sell proceeds per token ≈ 0.8 × 0.9999 ≈ 0.80.
    // Merge should be chosen (0.90 > 0.80).
    let tokens = [
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
        "0x3333333333333333333333333333333333333333",
    ];
    let names = ["M1", "M2", "M3"];
    let prices = [0.8, 0.05, 0.05];
    let preds = [0.3, 0.3, 0.3];

    let slot0_results: Vec<_> = tokens
        .iter()
        .zip(names.iter())
        .zip(prices.iter())
        .map(|((tok, name), price)| mock_slot0_market(name, tok, *price))
        .collect();

    let mut sims: Vec<_> = slot0_results
        .iter()
        .zip(preds.iter())
        .map(|((s, m), pred)| PoolSim::from_slot0(s, m, *pred).unwrap())
        .collect();

    // Verify merge is better than direct for M1
    let sell_amount = 5.0;
    let (merge_net, merge_actual) = merge_sell_proceeds(&sims, 0, sell_amount);
    let (_, direct_proceeds, _) = sims[0].sell_exact(sell_amount).unwrap();

    println!(
        "Merge net: {:.6}, Direct proceeds: {:.6}",
        merge_net, direct_proceeds
    );
    assert!(
        merge_net > direct_proceeds,
        "merge ({:.6}) should beat direct ({:.6}) for high-price outcome",
        merge_net,
        direct_proceeds
    );
    assert!(merge_actual > 0.0, "merge should be feasible");

    // Execute merge sell and verify actions
    let mut budget = 0.0;
    let mut actions = Vec::new();
    let merged = execute_merge_sell(&mut sims, 0, sell_amount, &mut actions, &mut budget);

    assert!(merged > 0.0, "should have merged tokens");
    assert!(budget > 0.0, "budget should increase from merge proceeds");
    assert!(
        (budget - merge_net).abs() < 1e-9,
        "execution budget delta should match dry-run merge proceeds"
    );

    // Should have: Buy×2 and Merge.
    let has_merge = actions.iter().any(|a| matches!(a, Action::Merge { .. }));
    let mut buy_totals: HashMap<&str, f64> = HashMap::new();
    for action in &actions {
        if let Action::Buy {
            market_name,
            amount,
            ..
        } = action
        {
            *buy_totals.entry(*market_name).or_insert(0.0) += *amount;
        }
    }
    assert!(has_merge, "should have Merge action");
    assert_eq!(
        buy_totals.len(),
        2,
        "should buy both non-source outcomes (possibly across multiple rounds)"
    );
    assert!(
        buy_totals.contains_key("M2") && buy_totals.contains_key("M3"),
        "should buy both complementary markets"
    );
    let m2_total = buy_totals.get("M2").copied().unwrap_or(0.0);
    let m3_total = buy_totals.get("M3").copied().unwrap_or(0.0);
    assert!(
        (m2_total - merged).abs() < 1e-6 && (m3_total - merged).abs() < 1e-6,
        "total buy amount per complement should match merged amount"
    );

    // Other pool prices should have increased (we bought into them)
    assert!(
        sims[1].price() > 0.05,
        "M2 price should increase after buying"
    );
    assert!(
        sims[2].price() > 0.05,
        "M3 price should increase after buying"
    );
}

#[test]
fn test_merge_not_chosen_for_low_price() {
    // M1 at price 0.1 (overpriced, pred=0.05), M2 and M3 at price 0.4 each.
    // Direct sell price ≈ 0.1×0.9999 ≈ 0.10/token.
    // Merge cost: buy M2+M3 ≈ 0.4/0.9999 × 2 ≈ 0.80. Net ≈ 1 - 0.80 = 0.20/token.
    // Actually merge is still better here. Let me pick prices where it's not.
    // M1 at price 0.1, M2+M3 at 0.45 each → merge cost ≈ 0.90, net ≈ 0.10.
    // Direct ≈ 0.10. Very close. Let M2+M3 be 0.48 → merge cost ≈ 0.96, net ≈ 0.04.
    // Direct ≈ 0.10. Direct wins.
    let tokens = [
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
        "0x3333333333333333333333333333333333333333",
    ];
    let names = ["M1", "M2", "M3"];
    let prices = [0.1, 0.48, 0.48];
    let preds = [0.05, 0.3, 0.3];

    let slot0_results: Vec<_> = tokens
        .iter()
        .zip(names.iter())
        .zip(prices.iter())
        .map(|((tok, name), price)| mock_slot0_market(name, tok, *price))
        .collect();

    let sims: Vec<_> = slot0_results
        .iter()
        .zip(preds.iter())
        .map(|((s, m), pred)| PoolSim::from_slot0(s, m, *pred).unwrap())
        .collect();

    let sell_amount = 5.0;
    let (merge_net, _) = merge_sell_proceeds(&sims, 0, sell_amount);
    let (_, direct_proceeds, _) = sims[0].sell_exact(sell_amount).unwrap();

    println!(
        "Merge net: {:.6}, Direct proceeds: {:.6}",
        merge_net, direct_proceeds
    );
    assert!(
        direct_proceeds > merge_net,
        "direct ({:.6}) should beat merge ({:.6}) for low-price outcome with expensive others",
        direct_proceeds,
        merge_net
    );
}

#[test]
#[ignore = "release perf helper; run explicitly"]
fn test_rebalance_perf_full_l1() {
    use crate::markets::MARKETS_L1;
    use std::time::Instant;

    let preds = crate::pools::prediction_map();

    // Build slot0 results for all 98 tradeable markets (those with pools + predictions).
    // Set each price to 50% of prediction to create buy opportunities for the waterfall.
    let slot0_results: Vec<(Slot0Result, &'static crate::markets::MarketData)> = MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .filter(|m| {
            let key = normalize_market_name(m.name);
            preds.contains_key(&key)
        })
        .map(|market| {
            let pool = market.pool.as_ref().unwrap();
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let key = normalize_market_name(market.name);
            let pred = preds[&key];
            // Price = 50% of prediction → profitable to buy
            let price = pred * 0.5;
            let sqrt_price = prediction_to_sqrt_price_x96(price, is_token1_outcome)
                .unwrap_or(U256::from(1u128 << 96));
            let slot0 = Slot0Result {
                pool_id: Address::ZERO,
                sqrt_price_x96: sqrt_price,
                tick: 0,
                observation_index: 0,
                observation_cardinality: 0,
                observation_cardinality_next: 0,
                fee_protocol: 0,
                unlocked: true,
            };
            (slot0, market)
        })
        .collect();

    println!("Markets: {}", slot0_results.len());

    // Warm up
    let _ = rebalance(&HashMap::new(), 100.0, &slot0_results);

    // Benchmark: 10 iterations
    let iters = 10;
    let start = Instant::now();
    let mut actions = Vec::new();
    for _ in 0..iters {
        actions = rebalance(&HashMap::new(), 100.0, &slot0_results);
    }
    let elapsed = start.elapsed();

    let buys = actions
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .count();
    let sells = actions
        .iter()
        .filter(|a| matches!(a, Action::Sell { .. }))
        .count();
    let mints = actions
        .iter()
        .filter(|a| matches!(a, Action::Mint { .. }))
        .count();
    let merges = actions
        .iter()
        .filter(|a| matches!(a, Action::Merge { .. }))
        .count();

    println!(
        "=== Rebalance Performance (full L1, {} outcomes) ===",
        slot0_results.len()
    );
    println!("  Total: {:?} for {} iterations", elapsed, iters);
    println!("  Per call: {:?}", elapsed / iters as u32);
    println!(
        "  Actions: {} total ({} buys, {} sells, {} mints, {} merges)",
        actions.len(),
        buys,
        sells,
        mints,
        merges
    );

    // Sanity: should produce actions when everything is underpriced
    assert!(
        !actions.is_empty(),
        "should produce actions for underpriced markets"
    );

    let initial_balances: HashMap<&str, f64> = HashMap::new();
    let (_, _, remaining_budget) = assert_strict_ev_gain_with_portfolio_trace(
        "rebalance_perf_full_l1",
        &actions,
        &slot0_results,
        &initial_balances,
        100.0,
    );

    // Budget accounting: remaining should be non-negative.
    // It may exceed initial budget if complete-set arbitrage is executed.
    assert!(
        remaining_budget >= -1e-9,
        "remaining budget should be non-negative: {:.6}",
        remaining_budget
    );
}

#[test]
#[ignore = "release perf helper; run explicitly"]
fn test_rebalance_perf_full_l1_with_gas_pricing() {
    use crate::markets::MARKETS_L1;
    use std::time::Instant;

    let preds = crate::pools::prediction_map();
    let slot0_results: Vec<(Slot0Result, &'static crate::markets::MarketData)> = MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .filter(|m| preds.contains_key(&normalize_market_name(m.name)))
        .map(|market| {
            let pool = market.pool.as_ref().unwrap();
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let key = normalize_market_name(market.name);
            let pred = preds[&key];
            let price = pred * 0.5;
            let sqrt_price = prediction_to_sqrt_price_x96(price, is_token1_outcome)
                .unwrap_or(U256::from(1u128 << 96));
            (
                Slot0Result {
                    pool_id: Address::ZERO,
                    sqrt_price_x96: sqrt_price,
                    tick: 0,
                    observation_index: 0,
                    observation_cardinality: 0,
                    observation_cardinality_next: 0,
                    fee_protocol: 0,
                    unlocked: true,
                },
                market,
            )
        })
        .collect();

    let gas = test_gas_assumptions();
    println!("Markets: {}", slot0_results.len());

    let _ = rebalance_with_gas_pricing(
        &HashMap::new(),
        100.0,
        &slot0_results,
        RebalanceMode::Full,
        &gas,
        1e-9,
        3000.0,
    );

    let iters = 10;
    let start = Instant::now();
    let mut actions = Vec::new();
    for _ in 0..iters {
        actions = rebalance_with_gas_pricing(
            &HashMap::new(),
            100.0,
            &slot0_results,
            RebalanceMode::Full,
            &gas,
            1e-9,
            3000.0,
        );
    }
    let elapsed = start.elapsed();

    let buys = actions
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .count();
    let sells = actions
        .iter()
        .filter(|a| matches!(a, Action::Sell { .. }))
        .count();
    let mints = actions
        .iter()
        .filter(|a| matches!(a, Action::Mint { .. }))
        .count();
    let merges = actions
        .iter()
        .filter(|a| matches!(a, Action::Merge { .. }))
        .count();

    println!(
        "=== Rebalance Performance (full L1 + gas pricing, {} outcomes) ===",
        slot0_results.len()
    );
    println!("  Total: {:?} for {} iterations", elapsed, iters);
    println!("  Per call: {:?}", elapsed / iters as u32);
    println!(
        "  Actions: {} total ({} buys, {} sells, {} mints, {} merges)",
        actions.len(),
        buys,
        sells,
        mints,
        merges
    );

    assert!(
        !actions.is_empty(),
        "gas-aware full-L1 fixture should produce actions"
    );

    let initial_balances: HashMap<&str, f64> = HashMap::new();
    let (_, _, remaining_budget) = assert_strict_ev_gain_with_portfolio_trace(
        "rebalance_perf_full_l1_with_gas_pricing",
        &actions,
        &slot0_results,
        &initial_balances,
        100.0,
    );
    assert!(
        remaining_budget >= -1e-9,
        "remaining budget should be non-negative: {:.6}",
        remaining_budget
    );
}

#[test]
#[ignore = "profiling helper; run explicitly"]
fn profile_rebalance_scenarios() {
    use crate::markets::MARKETS_L1;
    use std::time::Instant;

    fn build_slot0_with<F>(
        limit: Option<usize>,
        mut price_for_pred: F,
    ) -> Vec<(Slot0Result, &'static crate::markets::MarketData)>
    where
        F: FnMut(f64, usize) -> f64,
    {
        let preds = crate::pools::prediction_map();
        let mut rows = Vec::new();
        for market in MARKETS_L1.iter().filter(|m| m.pool.is_some()) {
            let key = normalize_market_name(market.name);
            let Some(&pred) = preds.get(&key) else {
                continue;
            };
            let pool = market.pool.as_ref().unwrap();
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let idx = rows.len();
            let price = price_for_pred(pred, idx).max(1e-6);
            let sqrt_price = prediction_to_sqrt_price_x96(price, is_token1_outcome)
                .unwrap_or(U256::from(1u128 << 96));
            rows.push((
                Slot0Result {
                    pool_id: Address::ZERO,
                    sqrt_price_x96: sqrt_price,
                    tick: 0,
                    observation_index: 0,
                    observation_cardinality: 0,
                    observation_cardinality_next: 0,
                    fee_protocol: 0,
                    unlocked: true,
                },
                market,
            ));
            if let Some(max_rows) = limit {
                if rows.len() >= max_rows {
                    break;
                }
            }
        }
        rows
    }

    let scenarios: Vec<(
        &str,
        Vec<(Slot0Result, &'static crate::markets::MarketData)>,
        HashMap<&str, f64>,
        f64,
    )> = vec![
        (
            "full_underpriced_with_arb",
            build_slot0_with(None, |pred, _| pred * 0.5),
            HashMap::new(),
            100.0,
        ),
        (
            "full_near_fair",
            build_slot0_with(None, |pred, _| pred * 0.98),
            HashMap::new(),
            100.0,
        ),
        (
            "partial_underpriced_no_mint_route",
            build_slot0_with(Some(64), |pred, _| pred * 0.5),
            HashMap::new(),
            100.0,
        ),
    ];

    for (name, slot0_results, balances, susd) in scenarios {
        // Warm up
        let _ = rebalance(&balances, susd, &slot0_results);

        let iters = 3;
        let start = Instant::now();
        let mut actions = Vec::new();
        for _ in 0..iters {
            actions = rebalance(&balances, susd, &slot0_results);
        }
        let elapsed = start.elapsed();
        let buys = actions
            .iter()
            .filter(|a| matches!(a, Action::Buy { .. }))
            .count();
        let sells = actions
            .iter()
            .filter(|a| matches!(a, Action::Sell { .. }))
            .count();
        let mints = actions
            .iter()
            .filter(|a| matches!(a, Action::Mint { .. }))
            .count();
        let merges = actions
            .iter()
            .filter(|a| matches!(a, Action::Merge { .. }))
            .count();
        println!(
            "[profile] {}: outcomes={}, per_call={:?}, actions={} (buys={}, sells={}, mints={}, merges={})",
            name,
            slot0_results.len(),
            elapsed / iters as u32,
            actions.len(),
            buys,
            sells,
            mints,
            merges
        );
    }
}

#[test]
#[ignore = "profiling helper; run explicitly"]
fn profile_complete_set_arb_solver() {
    let sims = build_three_sims_with_preds([0.2, 0.2, 0.2], [0.3, 0.3, 0.3]);
    let iters = 2000;
    let start = std::time::Instant::now();
    let mut last = 0.0;
    for _ in 0..iters {
        last = solve_complete_set_arb_amount(&sims);
    }
    let elapsed = start.elapsed();
    println!(
        "[profile] complete_set_arb_solver: iters={}, total={:?}, per_iter={:?}, amount={:.12}",
        iters,
        elapsed,
        elapsed / iters as u32,
        last
    );
    assert!(last >= 0.0);
}

#[test]
fn test_rebalance_output_is_groupable() {
    let actions = sample_rebalance_actions();
    let groups = group_actions(&actions).expect("rebalance output should be groupable");

    let covered_actions = groups.iter().map(|g| g.action_indices.len()).sum::<usize>();
    assert_eq!(
        covered_actions,
        actions.len(),
        "grouping should cover every action exactly once"
    );
    assert!(
        groups.iter().all(|g| !g.action_indices.is_empty()),
        "groups should never be empty"
    );
}

#[test]
fn test_rebalance_mixed_groups_are_route_coupled() {
    let actions = sample_rebalance_actions();
    let groups = group_actions(&actions).expect("rebalance output should be groupable");

    for (group_idx, group) in groups.iter().enumerate() {
        match group.kind {
            GroupKind::MintSell => {
                let mut direct_buy_markets: HashSet<&'static str> = HashSet::new();
                let mut mint_targets: HashSet<&'static str> = HashSet::new();
                for &action_idx in &group.action_indices {
                    match actions[action_idx] {
                        Action::Buy { market_name, .. } => {
                            direct_buy_markets.insert(market_name);
                        }
                        Action::Mint { target_market, .. } => {
                            mint_targets.insert(target_market);
                        }
                        Action::Sell { .. } | Action::Merge { .. } => {}
                    }
                }

                if !direct_buy_markets.is_empty() {
                    assert!(
                        direct_buy_markets
                            .iter()
                            .any(|market| mint_targets.contains(market)),
                        "group #{group_idx} mixes direct buys with mint-sell legs without shared target market"
                    );
                }
            }
            GroupKind::BuyMerge => {
                let mut direct_sell_markets: HashSet<&'static str> = HashSet::new();
                let mut merge_sources: HashSet<&'static str> = HashSet::new();
                for &action_idx in &group.action_indices {
                    match actions[action_idx] {
                        Action::Sell { market_name, .. } => {
                            direct_sell_markets.insert(market_name);
                        }
                        Action::Merge { source_market, .. } => {
                            merge_sources.insert(source_market);
                        }
                        Action::Buy { .. } | Action::Mint { .. } => {}
                    }
                }

                if !direct_sell_markets.is_empty() {
                    assert!(
                        direct_sell_markets
                            .iter()
                            .any(|market| merge_sources.contains(market)),
                        "group #{group_idx} mixes direct sells with buy-merge legs without shared source market"
                    );
                }
            }
            GroupKind::DirectBuy | GroupKind::DirectSell | GroupKind::DirectMerge => {}
        }
    }
}

#[test]
fn test_rebalance_output_is_plannable_with_default_edge_model() {
    let actions = sample_rebalance_actions();
    let strict_groups =
        group_execution_actions(&actions).expect("rebalance output should be strictly groupable");

    let plans = build_group_plans_with_default_edges(
        &actions,
        &test_gas_assumptions(),
        1e-10,
        3000.0,
        BufferConfig::default(),
    )
    .expect("planning should succeed with built-in direct-buy edge mapping");

    let direct_buy_groups = strict_groups
        .iter()
        .filter(|g| g.kind == GroupKind::DirectBuy)
        .count();
    let direct_buy_plans = plans
        .iter()
        .filter(|p| p.kind == GroupKind::DirectBuy)
        .count();
    assert_eq!(
        direct_buy_plans, direct_buy_groups,
        "all direct-buy groups should be plannable with explicit edge input"
    );
}

#[test]
fn test_rebalance_non_direct_merge_plans_have_batch_bounds() {
    let actions = sample_rebalance_actions();
    let mut plans = build_group_plans_with_default_edges(
        &actions,
        &test_gas_assumptions(),
        1e-10,
        3000.0,
        BufferConfig::default(),
    )
    .expect("planning should succeed");
    stamp_plans_with_block(&mut plans, 100);

    assert!(!plans.is_empty(), "expected at least one executable plan");
    for plan in plans {
        let bounds = derive_batch_quote_bounds(&plan, 100, 2)
            .expect("planned groups should never have mixed buy/sell leg directions");
        match plan.kind {
            GroupKind::DirectMerge => assert!(
                bounds.is_none(),
                "direct merge should not map to batch buy/sell bounds"
            ),
            GroupKind::DirectBuy
            | GroupKind::DirectSell
            | GroupKind::MintSell
            | GroupKind::BuyMerge => assert!(
                bounds.is_some(),
                "dex-leg groups must map to aggregate batch bounds"
            ),
        }
    }
}

#[test]
fn test_rebalance_never_emits_naked_mint_or_repay() {
    let actions = sample_rebalance_actions();
    assert_flash_brackets_are_well_formed(&actions);
}

#[tokio::test]
async fn test_rebalance_optimization_full_l1_live_prices() {
    // Full-L1 optimization test on live pool prices.
    let Some(slot0_results) = fetch_live_expected_l1_slot0_results("live_full_l1").await else {
        return;
    };

    let initial_balances: HashMap<&str, f64> = HashMap::new();
    let initial_susd = 100.0;
    let actions = rebalance(&initial_balances, initial_susd, &slot0_results);
    let actions_repeat = rebalance(&initial_balances, initial_susd, &slot0_results);

    assert_eq!(
        format!("{:?}", actions),
        format!("{:?}", actions_repeat),
        "rebalance should be deterministic for a fixed live slot0 snapshot"
    );

    assert_rebalance_action_invariants(&actions, &slot0_results, &initial_balances, initial_susd);
    print_rebalance_execution_summary("live_full_l1", &actions, &slot0_results);
    let (final_holdings, final_cash) =
        replay_actions_to_state(&actions, &slot0_results, &initial_balances, initial_susd);

    let ev_before = initial_susd;
    let ev_after = ev_from_state(&final_holdings, final_cash);
    let ev_gain = ev_after - ev_before;
    println!(
        "[rebalance][live_full_l1] expected value: before={:.9}, after={:.9}, gain={:.9}",
        ev_before, ev_after, ev_gain
    );

    assert!(
        ev_after + 1e-6 >= ev_before,
        "live full-L1 optimization should not reduce expected value: before={:.9}, after={:.9}",
        ev_before,
        ev_after
    );
}

#[tokio::test]
async fn test_rebalance_arb_only_full_l1_live_prices() {
    // Full-L1 arb-only test on live pool prices.
    let Some(slot0_results) = fetch_live_expected_l1_slot0_results("live_full_l1_arb_only").await
    else {
        return;
    };

    let initial_balances: HashMap<&str, f64> = HashMap::new();
    let initial_susd = 100.0;
    let actions = rebalance_with_mode(
        &initial_balances,
        initial_susd,
        &slot0_results,
        RebalanceMode::ArbOnly,
    );
    let actions_repeat = rebalance_with_mode(
        &initial_balances,
        initial_susd,
        &slot0_results,
        RebalanceMode::ArbOnly,
    );

    println!(
        "[rebalance][live_full_l1_arb_only] actions count={}",
        actions.len()
    );

    assert_eq!(
        format!("{:?}", actions),
        format!("{:?}", actions_repeat),
        "arb-only rebalance should be deterministic for a fixed live slot0 snapshot"
    );

    assert_rebalance_action_invariants(&actions, &slot0_results, &initial_balances, initial_susd);
    print_rebalance_execution_summary("live_full_l1_arb_only", &actions, &slot0_results);

    let (final_holdings, final_cash) =
        replay_actions_to_state(&actions, &slot0_results, &initial_balances, initial_susd);

    let ev_before = initial_susd;
    let ev_after = ev_from_state(&final_holdings, final_cash);
    let ev_gain = ev_after - ev_before;
    println!(
        "[rebalance][live_full_l1_arb_only] expected value: before={:.9}, after={:.9}, gain={:.9}",
        ev_before, ev_after, ev_gain
    );

    assert!(
        ev_after + 1e-6 >= ev_before,
        "live full-L1 arb-only should not reduce expected value: before={:.9}, after={:.9}",
        ev_before,
        ev_after
    );
}

#[tokio::test]
async fn test_rebalance_top_buy_gap_subset_l1_live_prices() {
    let Some(full_slot0_results) =
        fetch_live_expected_l1_slot0_results("live_top_buy_gap_subset").await
    else {
        return;
    };

    let slot0_results = live_subset_ranked_by_buy_gap(&full_slot0_results, 4);
    assert!(
        slot0_results.len() >= 4,
        "expected at least four live underpriced markets with valid price + prediction data"
    );

    assert_live_subset_rebalance_non_decreasing_ev("live_top_buy_gap_subset", &slot0_results, 25.0);
}

#[tokio::test]
async fn test_rebalance_near_fair_subset_l1_live_prices() {
    let Some(full_slot0_results) =
        fetch_live_expected_l1_slot0_results("live_near_fair_subset").await
    else {
        return;
    };

    let slot0_results = live_subset_ranked_by_abs_gap(&full_slot0_results, 4, false);
    assert!(
        slot0_results.len() >= 4,
        "expected at least four live markets with valid price + prediction data"
    );

    assert_live_subset_rebalance_non_decreasing_ev("live_near_fair_subset", &slot0_results, 25.0);
}

#[test]
fn test_rebalance_regression_full_l1_augmented_multitick_liquidity_changes_exact_replay() {
    let markets = eligible_l1_markets_with_predictions();
    let price_multipliers = normalized_full_l1_price_multipliers(&markets);
    let slot0_results = build_slot0_results_for_markets(&markets, &price_multipliers);
    let initial_balances: HashMap<&str, f64> = HashMap::new();
    let initial_susd = 250.0;

    let actions = rebalance(&initial_balances, initial_susd, &slot0_results);
    assert!(
        actions
            .iter()
            .any(|action| matches!(action, Action::Buy { .. })),
        "expected the realistic full-L1 fixture to include at least one buy"
    );
    assert_rebalance_action_invariants(&actions, &slot0_results, &initial_balances, initial_susd);

    let augmented_slot0_results = build_augmented_multitick_fixture(&slot0_results, &actions);
    let augmented_actions = rebalance(&initial_balances, initial_susd, &augmented_slot0_results);

    assert_eq!(
        format!("{:?}", actions),
        format!("{:?}", augmented_actions),
        "appending extra in-path ticks should not change the current planner because it still only reads the current liquidity band"
    );

    let base_exact = replay_actions_to_state_multitick_exact(
        &actions,
        &slot0_results,
        &initial_balances,
        initial_susd,
    );
    let augmented_exact = replay_actions_to_state_multitick_exact(
        &actions,
        &augmented_slot0_results,
        &initial_balances,
        initial_susd,
    );

    let base_ev = ev_from_state(&base_exact.holdings, base_exact.cash);
    let augmented_ev = ev_from_state(&augmented_exact.holdings, augmented_exact.cash);
    let ev_delta = augmented_ev - base_ev;
    let holdings_changed = markets.iter().any(|market| {
        let base_units = base_exact.holdings.get(market.name).copied().unwrap_or(0.0);
        let augmented_units = augmented_exact
            .holdings
            .get(market.name)
            .copied()
            .unwrap_or(0.0);
        (augmented_units - base_units).abs() > 1e-9
    });
    let cash_changed = (augmented_exact.cash - base_exact.cash).abs() > 1e-9;

    println!(
        "[rebalance][full_l1_augmented_multitick] actions={} base_crossings={} augmented_crossings={} base_ev={:.9} augmented_ev={:.9} delta={:.9}",
        actions.len(),
        base_exact.crossed_initialized_ticks,
        augmented_exact.crossed_initialized_ticks,
        base_ev,
        augmented_ev,
        ev_delta
    );

    assert!(
        augmented_exact.crossed_initialized_ticks > base_exact.crossed_initialized_ticks,
        "expected injected in-path liquidity to add at least one extra initialized-tick crossing"
    );
    assert!(
        holdings_changed || cash_changed,
        "expected injected in-path liquidity to change realized fills or cash"
    );
    assert!(
        ev_delta.abs() > 0.1,
        "expected injected multi-tick liquidity to materially change exact replay EV: base={:.9}, augmented={:.9}, delta={:.9}",
        base_ev,
        augmented_ev,
        ev_delta
    );
}

/// Validates that `execute_bundle_step` records live AMM costs in Actions,
/// not stale pre-planned costs, when pool prices have moved since planning.
#[test]
fn test_execute_bundle_step_records_live_cost_not_stale_planned_cost() {
    use super::super::bundle::{BundleRouteKind, BundleSegmentPlan, BundleStepPlan};
    use super::super::trading::ExecutionState;

    // 1. Build sims at initial prices
    let mut sims = build_three_sims_with_preds([0.05, 0.08, 0.12], [0.15, 0.25, 0.35]);

    // 2. Compute planned costs at initial prices
    let idx = 0;
    let target_price = 0.10; // buy to push price from 0.05 → 0.10
    let (planned_cost, planned_amount, planned_new_price) =
        sims[idx].cost_to_price(target_price).unwrap();
    assert!(planned_amount > 1e-6, "should have non-trivial buy amount");
    assert!(planned_cost > 1e-6, "should have non-trivial cost");

    // 3. Shift pool price to simulate polish loop interference.
    //    Buy a small amount to move the price up, making subsequent buys more expensive.
    let (_, shift_cost, shift_price) = sims[idx].buy_exact(planned_amount * 0.3).unwrap();
    assert!(shift_cost > 0.0);
    sims[idx].set_price(shift_price);
    let price_after_shift = sims[idx].price();
    assert!(
        price_after_shift > 0.05 + 1e-6,
        "price should have moved up from shift"
    );

    // 4. Create a bundle step plan with the STALE planned cost (from pre-shift state)
    let stale_plan = BundleStepPlan {
        segments: vec![BundleSegmentPlan {
            kind: BundleRouteKind::Direct,
            target_prof: 0.5,
            cash_cost: planned_cost,
            mint_amount: 0.0,
            direct_member_plans: vec![(idx, planned_amount, planned_cost, planned_new_price)],
            mint_sell_leg_plans: vec![],
        }],
        final_prof: 0.5,
    };

    // 5. Compute what live cost SHOULD be (at current shifted price)
    let (_, live_cost_expected, _) = sims[idx].buy_exact(planned_amount).unwrap();
    assert!(
        (live_cost_expected - planned_cost).abs() > 0.001,
        "live cost should differ meaningfully from planned cost: live={}, planned={}",
        live_cost_expected,
        planned_cost
    );

    // 6. Execute the bundle step
    let mut budget = 1000.0;
    let mut actions = Vec::new();
    let mut sim_balances = HashMap::new();
    let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
    let executed = exec.execute_bundle_step(&stale_plan, &[idx]);
    assert!(executed, "step should execute");
    assert_eq!(actions.len(), 1, "should produce one Buy action");

    // 7. Verify: Action::Buy.cost should be the LIVE cost, not the stale planned cost
    match &actions[0] {
        Action::Buy { cost, amount, .. } => {
            let diff_from_live = (*cost - live_cost_expected).abs();
            let diff_from_planned = (*cost - planned_cost).abs();
            assert!(
                diff_from_live < 1e-6,
                "Action cost should match live cost: action={:.6}, live={:.6}, planned={:.6}",
                cost,
                live_cost_expected,
                planned_cost
            );
            assert!(
                diff_from_planned > 0.001,
                "Action cost should differ from stale planned cost: action={:.6}, planned={:.6}",
                cost,
                planned_cost
            );
            assert!(
                (*amount - planned_amount).abs() < 1e-12,
                "amount should match planned amount"
            );
        }
        other => panic!("expected Action::Buy, got {:?}", other),
    }
}
