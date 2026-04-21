use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::str::FromStr;

use alloy::primitives::{Address, Bytes, U160, U256};
use alloy::sol;
use alloy::sol_types::{SolCall, SolValue};
use serde::{Deserialize, Serialize};

use deep_trading_bot::execution::bounds::{
    BufferConfig, ConservativeExecutionConfig, build_group_plans_with_market_context,
    build_group_plans_with_prediction_edges, planned_edge_from_prediction_map_susd,
    stamp_plans_with_block,
};
use deep_trading_bot::execution::gas::{GasAssumptions, LiveOptimismFeeInputs};
use deep_trading_bot::execution::program::{
    build_chunk_calls_checked_with_address_book,
    compile_execution_program_unchecked_with_address_book,
};
use deep_trading_bot::execution::tx_builder::ExecutionAddressBook;
use deep_trading_bot::execution::{ExecutionMode, ITradeExecutor};
use deep_trading_bot::markets::{MarketData, Pool, Tick};
use deep_trading_bot::pools::Slot0Result;
use deep_trading_bot::portfolio::{
    Action, RebalanceFlags, RebalanceSolver,
    rebalance_with_custom_predictions_and_solver_and_gas_pricing_and_flags_and_decision,
};

sol! {
    struct LocalFixtureCall {
        address to;
        bytes data;
    }

    struct LocalFixtureChunk {
        LocalFixtureCall[] calls;
        uint256 estimatedL2Gas;
        uint256 estimatedCalldataBytes;
        uint256 estimatedTotalFeeWad;
    }

    struct LocalFixtureResult {
        LocalFixtureChunk[] chunks;
        uint256 actionCount;
        uint256 preRawEvWad;
        uint256 expectedRawEvWad;
        uint256 estimatedTotalFeeWad;
        uint256 estimatedNetEvWad;
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct FixtureInput {
    scenario_id: String,
    solver: Option<String>,
    executor: String,
    starting_cash_wad: String,
    current_block: Option<u64>,
    max_stale_blocks: Option<u64>,
    gas_price_wei: Option<u64>,
    eth_usd_wad: Option<String>,
    l1_fee_per_byte_wei: Option<f64>,
    force_mint_available: Option<bool>,
    address_book: FixtureAddressBook,
    markets: Vec<FixtureMarket>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct FixtureAddressBook {
    collateral: String,
    seer_router: String,
    swap_router02: String,
    market1: String,
    market2: String,
    market2_collateral_connector: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct FixtureMarket {
    name: String,
    market_id: String,
    outcome_token: String,
    pool: String,
    token0: String,
    token1: String,
    quote_token: String,
    sqrt_price_x96: String,
    tick: i32,
    tick_lower: i32,
    tick_upper: i32,
    liquidity: String,
    price_wad: String,
    prediction_wad: String,
    initial_balance_wad: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct FixtureOutput {
    scenario_id: String,
    solver: String,
    actions: Vec<ActionOutput>,
    chunks: Vec<ChunkOutput>,
    abi: String,
    pre_raw_ev_wad: String,
    expected_raw_ev_wad: String,
    estimated_total_fee_wad: String,
    estimated_net_ev_wad: String,
    calldata_bytes: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct ActionOutput {
    kind: &'static str,
    market_name: Option<&'static str>,
    amount_wad: String,
    quote_wad: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct ChunkOutput {
    calls: Vec<CallOutput>,
    estimated_l2_gas: u64,
    estimated_calldata_bytes: usize,
    estimated_total_fee_wad: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct CallOutput {
    to: String,
    data: String,
}

fn parse_address(raw: &str) -> Result<Address, Box<dyn Error>> {
    Ok(Address::from_str(raw.trim())?)
}

fn parse_u256(raw: &str) -> Result<U256, Box<dyn Error>> {
    Ok(U256::from_str(raw.trim())?)
}

fn wad_to_f64(raw: &str) -> Result<f64, Box<dyn Error>> {
    let value = parse_u256(raw)?;
    let as_f64 = value.to_string().parse::<f64>()?;
    Ok(as_f64 / 1e18)
}

fn u256_wad_to_f64(value: &U256) -> Result<f64, Box<dyn Error>> {
    let as_f64 = value.to_string().parse::<f64>()?;
    Ok(as_f64 / 1e18)
}

fn susd_to_wad(value: f64) -> U256 {
    if !value.is_finite() || value <= 0.0 {
        return U256::ZERO;
    }
    U256::from((value * 1e18).round() as u128)
}

fn leak_str(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn leak_market(input: &FixtureMarket) -> Result<&'static MarketData, Box<dyn Error>> {
    let liquidity_raw = parse_u256(&input.liquidity)?;
    let liquidity_i128 = liquidity_raw
        .to_string()
        .parse::<i128>()
        .unwrap_or(i128::MAX);
    let ticks: &'static [Tick] = Box::leak(
        vec![
            Tick {
                tick_idx: input.tick_lower,
                liquidity_net: liquidity_i128,
            },
            Tick {
                tick_idx: input.tick_upper,
                liquidity_net: -liquidity_i128,
            },
        ]
        .into_boxed_slice(),
    );
    let pool = Pool {
        token0: leak_str(input.token0.clone()),
        token1: leak_str(input.token1.clone()),
        pool_id: leak_str(input.pool.clone()),
        liquidity: leak_str(input.liquidity.clone()),
        ticks,
    };
    Ok(Box::leak(Box::new(MarketData {
        name: leak_str(input.name.clone()),
        market_id: leak_str(input.market_id.clone()),
        outcome_token: leak_str(input.outcome_token.clone()),
        pool: Some(pool),
        quote_token: leak_str(input.quote_token.clone()),
    })))
}

fn solver_from_input(raw: Option<&str>) -> RebalanceSolver {
    match raw.unwrap_or("native").trim().to_ascii_lowercase().as_str() {
        "forecastflows" | "forecast_flows" | "forecast" => RebalanceSolver::ForecastFlows,
        "head_to_head" | "headtohead" => RebalanceSolver::HeadToHead,
        _ => RebalanceSolver::Native,
    }
}

fn action_output(action: &Action) -> ActionOutput {
    match action {
        Action::Mint {
            amount,
            target_market,
            ..
        } => ActionOutput {
            kind: "mint",
            market_name: Some(*target_market),
            amount_wad: susd_to_wad(*amount).to_string(),
            quote_wad: None,
        },
        Action::Buy {
            market_name,
            amount,
            cost,
        } => ActionOutput {
            kind: "buy",
            market_name: Some(*market_name),
            amount_wad: susd_to_wad(*amount).to_string(),
            quote_wad: Some(susd_to_wad(*cost).to_string()),
        },
        Action::Sell {
            market_name,
            amount,
            proceeds,
        } => ActionOutput {
            kind: "sell",
            market_name: Some(*market_name),
            amount_wad: susd_to_wad(*amount).to_string(),
            quote_wad: Some(susd_to_wad(*proceeds).to_string()),
        },
        Action::Merge {
            amount,
            source_market,
            ..
        } => ActionOutput {
            kind: "merge",
            market_name: Some(*source_market),
            amount_wad: susd_to_wad(*amount).to_string(),
            quote_wad: None,
        },
    }
}

fn pre_raw_ev_wad(input: &FixtureInput) -> Result<U256, Box<dyn Error>> {
    let mut total = parse_u256(&input.starting_cash_wad)?;
    for market in &input.markets {
        let Some(balance_raw) = &market.initial_balance_wad else {
            continue;
        };
        let balance = parse_u256(balance_raw)?;
        let prediction = parse_u256(&market.prediction_wad)?;
        total = total.saturating_add(
            balance.saturating_mul(prediction) / U256::from(1_000_000_000_000_000_000u128),
        );
    }
    Ok(total)
}

fn scripted_route_probe_actions(
    scenario_id: &str,
    markets: &[(&'static str, f64)],
    starting_cash: f64,
    pre_raw_susd: f64,
) -> Option<(Vec<Action>, f64)> {
    let amount = starting_cash.min(10.0);
    if amount <= 0.0 || markets.is_empty() {
        return None;
    }

    if scenario_id.contains("mint_sell") {
        let mut actions = vec![Action::Mint {
            contract_1: "local_market_1",
            contract_2: "local_market_2",
            amount,
            target_market: "complete_set_arb",
        }];
        let mut proceeds = 0.0;
        for (market_name, price) in markets {
            let leg_proceeds = amount * price * 0.999;
            proceeds += leg_proceeds;
            actions.push(Action::Sell {
                market_name,
                amount,
                proceeds: leg_proceeds,
            });
        }
        return Some((actions, pre_raw_susd - amount + proceeds));
    }

    if scenario_id.contains("buy_merge") {
        // Local pool math can under-deliver the final few wei of an exact-output buy on some
        // legs. Keep the scripted merge slightly below the requested buy amount so the synthetic
        // route probe remains executable under real local contract math.
        let merge_amount = amount * 0.999;
        let mut actions = Vec::with_capacity(markets.len() + 1);
        let mut cost = 0.0;
        for (market_name, price) in markets {
            let leg_cost = amount * price * 1.001;
            cost += leg_cost;
            actions.push(Action::Buy {
                market_name,
                amount,
                cost: leg_cost,
            });
        }
        actions.push(Action::Merge {
            contract_1: "local_market_1",
            contract_2: "local_market_2",
            amount: merge_amount,
            source_market: "complete_set_arb",
        });
        return Some((actions, pre_raw_susd - cost + merge_amount));
    }

    None
}

fn build_address_book(input: &FixtureInput) -> Result<ExecutionAddressBook, Box<dyn Error>> {
    let mut outcome_tokens = HashMap::new();
    for market in &input.markets {
        outcome_tokens.insert(market.name.clone(), parse_address(&market.outcome_token)?);
    }
    Ok(ExecutionAddressBook {
        collateral: parse_address(&input.address_book.collateral)?,
        seer_router: parse_address(&input.address_book.seer_router)?,
        swap_router: parse_address(&input.address_book.swap_router02)?,
        market1: parse_address(&input.address_book.market1)?,
        market2: parse_address(&input.address_book.market2)?,
        market2_collateral: parse_address(&input.address_book.market2_collateral_connector)?,
        outcome_tokens,
    })
}

fn should_zero_price_limits_for_benchmark_forecastflows(solver: RebalanceSolver) -> bool {
    matches!(solver, RebalanceSolver::ForecastFlows)
        && std::env::var("FORECASTFLOWS_REQUEST_PROFILE")
            .is_ok_and(|value| value.eq_ignore_ascii_case("benchmark"))
}

fn execution_mode_for_fixture(solver: RebalanceSolver) -> ExecutionMode {
    if should_zero_price_limits_for_benchmark_forecastflows(solver) {
        ExecutionMode::Strict
    } else {
        ExecutionMode::Packed
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("RUST_LOG").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();
    }
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: local_foundry_e2e_fixture <input-json-path>")?;
    let input: FixtureInput = serde_json::from_str(&fs::read_to_string(path)?)?;
    let executor = parse_address(&input.executor)?;
    let address_book = build_address_book(&input)?;

    let mut predictions = HashMap::new();
    let mut balances_owned: HashMap<&'static str, f64> = HashMap::new();
    let mut slot0_results = Vec::with_capacity(input.markets.len());
    let mut market_prices = Vec::with_capacity(input.markets.len());

    for market in &input.markets {
        let leaked = leak_market(market)?;
        let prediction = wad_to_f64(&market.prediction_wad)?;
        predictions.insert(leaked.name.to_string(), prediction);
        market_prices.push((leaked.name, wad_to_f64(&market.price_wad)?));
        let balance = market
            .initial_balance_wad
            .as_deref()
            .map(wad_to_f64)
            .transpose()?
            .unwrap_or(0.0);
        if balance > 0.0 {
            balances_owned.insert(leaked.name, balance);
        }
        slot0_results.push((
            Slot0Result {
                pool_id: parse_address(&market.pool)?,
                sqrt_price_x96: parse_u256(&market.sqrt_price_x96)?,
                tick: market.tick,
                observation_index: 0,
                observation_cardinality: 1,
                observation_cardinality_next: 1,
                fee_protocol: 0,
                unlocked: true,
            },
            leaked,
        ));
    }

    let balances_view: HashMap<&str, f64> = balances_owned
        .iter()
        .map(|(market, balance)| (*market as &str, *balance))
        .collect();
    let starting_cash = wad_to_f64(&input.starting_cash_wad)?;
    let eth_usd = input
        .eth_usd_wad
        .as_deref()
        .map(wad_to_f64)
        .transpose()?
        .unwrap_or(3000.0);
    let gas_price_wei = input.gas_price_wei.unwrap_or(1_002_325);
    let gas_price_eth = (gas_price_wei as f64) / 1e18;
    let gas_assumptions = GasAssumptions {
        l1_fee_per_byte_wei: input.l1_fee_per_byte_wei.unwrap_or(1_643_855.0),
        l1_data_fee_floor_susd: 0.0,
        ..GasAssumptions::default()
    };
    let current_block = input.current_block.unwrap_or(1);
    let max_stale_blocks = input.max_stale_blocks.unwrap_or(2);
    let solver = solver_from_input(input.solver.as_deref());
    let pre_raw = pre_raw_ev_wad(&input)?;
    let pre_raw_susd = u256_wad_to_f64(&pre_raw)?;

    let decision =
        rebalance_with_custom_predictions_and_solver_and_gas_pricing_and_flags_and_decision(
            &balances_view,
            starting_cash,
            &slot0_results,
            &predictions,
            solver,
            &gas_assumptions,
            gas_price_eth,
            eth_usd,
            RebalanceFlags::default(),
            input.force_mint_available.unwrap_or(true),
        );
    if matches!(solver, RebalanceSolver::ForecastFlows) {
        if let Some(reason) = decision
            .summary
            .forecastflows_telemetry
            .fallback_reason
            .as_deref()
        {
            let certified = decision
                .summary
                .forecastflows_telemetry
                .certified_drop_reason
                .as_deref()
                .unwrap_or("-");
            let replay = decision
                .summary
                .forecastflows_telemetry
                .replay_drop_reason
                .as_deref()
                .unwrap_or("-");
            return Err(format!(
                "forecastflows fallback={reason}; benchmark refuses silent fallback. \
                 certified_drop_reason={certified}; replay_drop_reason={replay}; \
                 Set FORECASTFLOWS_WORKER_BIN to a built rust_worker binary."
            )
            .into());
        }
    }
    let mut actions = decision.actions.clone();
    let mut expected_raw_susd = decision.summary.raw_ev;
    if let Some((scripted_actions, scripted_raw)) = scripted_route_probe_actions(
        &input.scenario_id,
        &market_prices,
        starting_cash,
        pre_raw_susd,
    ) {
        actions = scripted_actions;
        expected_raw_susd = scripted_raw;
    }

    let conservative_execution = ConservativeExecutionConfig {
        quote_latency_blocks: 0,
        adverse_move_bps_per_block: 0,
    };
    let buffer = BufferConfig {
        buffer_frac: 0.0,
        buffer_min_susd: 1e-9,
    };
    let mut plans = build_group_plans_with_market_context(
        &actions,
        &slot0_results,
        conservative_execution,
        &gas_assumptions,
        gas_price_eth,
        eth_usd,
        buffer,
        |_, group| planned_edge_from_prediction_map_susd(group, &predictions),
    )?;
    if plans.is_empty() && !actions.is_empty() {
        plans = build_group_plans_with_prediction_edges(
            &actions,
            &predictions,
            &gas_assumptions,
            gas_price_eth,
            eth_usd,
            buffer,
        )?;
        for plan in &mut plans {
            for leg in &mut plan.legs {
                leg.sqrt_price_limit_x96 = Some(U160::ZERO);
            }
        }
    }
    if should_zero_price_limits_for_benchmark_forecastflows(solver) {
        for plan in &mut plans {
            for leg in &mut plan.legs {
                leg.sqrt_price_limit_x96 = Some(U160::ZERO);
            }
        }
    }
    stamp_plans_with_block(&mut plans, current_block);

    let fee_inputs = LiveOptimismFeeInputs {
        chain_id: 31337,
        sender_nonce: 0,
        gas_price_wei: gas_price_wei.into(),
    };
    let program = compile_execution_program_unchecked_with_address_book(
        execution_mode_for_fixture(solver),
        executor,
        &actions,
        &plans,
        fee_inputs,
        &gas_assumptions,
        eth_usd,
        &address_book,
    )?;

    let mut output_chunks = Vec::with_capacity(program.chunks.len());
    let mut abi_chunks = Vec::with_capacity(program.chunks.len());
    let mut total_calldata_bytes = 0usize;
    for chunk in &program.chunks {
        let calls = build_chunk_calls_checked_with_address_book(
            executor,
            &actions,
            &chunk.plans,
            current_block,
            max_stale_blocks,
            &address_book,
        )?;
        let calldata = ITradeExecutor::batchExecuteCall {
            calls: calls.clone(),
        }
        .abi_encode();
        total_calldata_bytes += calldata.len();
        let fixture_calls: Vec<LocalFixtureCall> = calls
            .iter()
            .map(|call| LocalFixtureCall {
                to: call.to,
                data: Bytes::copy_from_slice(&call.data),
            })
            .collect();
        abi_chunks.push(LocalFixtureChunk {
            calls: fixture_calls,
            estimatedL2Gas: U256::from(chunk.total_l2_gas_units),
            estimatedCalldataBytes: U256::from(calldata.len()),
            estimatedTotalFeeWad: susd_to_wad(chunk.estimated_total_fee_susd),
        });
        output_chunks.push(ChunkOutput {
            calls: calls
                .iter()
                .map(|call| CallOutput {
                    to: call.to.to_string(),
                    data: alloy::hex::encode_prefixed(&call.data),
                })
                .collect(),
            estimated_l2_gas: chunk.total_l2_gas_units,
            estimated_calldata_bytes: calldata.len(),
            estimated_total_fee_wad: susd_to_wad(chunk.estimated_total_fee_susd).to_string(),
        });
    }

    let expected_raw = susd_to_wad(expected_raw_susd);
    let estimated_fee = susd_to_wad(program.total_fee_susd);
    let estimated_net = expected_raw
        .saturating_sub(pre_raw)
        .saturating_sub(estimated_fee);
    let payload = LocalFixtureResult {
        chunks: abi_chunks,
        actionCount: U256::from(actions.len()),
        preRawEvWad: pre_raw,
        expectedRawEvWad: expected_raw,
        estimatedTotalFeeWad: estimated_fee,
        estimatedNetEvWad: estimated_net,
    };
    let output = FixtureOutput {
        scenario_id: input.scenario_id,
        solver: solver.as_str().to_string(),
        actions: actions.iter().map(action_output).collect(),
        chunks: output_chunks,
        abi: alloy::hex::encode_prefixed(payload.abi_encode()),
        pre_raw_ev_wad: pre_raw.to_string(),
        expected_raw_ev_wad: expected_raw.to_string(),
        estimated_total_fee_wad: estimated_fee.to_string(),
        estimated_net_ev_wad: estimated_net.to_string(),
        calldata_bytes: total_calldata_bytes,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
