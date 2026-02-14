use std::collections::HashMap;

use super::super::merge::{
    execute_merge_sell, merge_sell_cap, merge_sell_proceeds, optimal_sell_split,
};
use super::super::rebalancer::rebalance;
use super::super::sim::PoolSim;
use super::super::trading::{ExecutionState, solve_complete_set_arb_amount};
use super::{
    Action, brute_force_best_split, build_slot0_results_for_markets, build_three_sims,
    build_three_sims_with_preds, eligible_l1_markets_with_predictions, mock_slot0_market,
};
use crate::execution::GroupKind;
use crate::execution::bounds::{
    BufferConfig, build_group_plans_with_default_edges, derive_batch_quote_bounds,
    stamp_plans_with_block,
};
use crate::execution::gas::GasAssumptions;
use crate::execution::grouping::group_actions;
use crate::pools::{Slot0Result, normalize_market_name, prediction_to_sqrt_price_x96};
use alloy::primitives::{Address, U256};

fn sample_rebalance_actions() -> Vec<Action> {
    let markets = eligible_l1_markets_with_predictions();
    let selected: Vec<_> = markets.into_iter().take(8).collect();
    let multipliers = [1.35, 0.55, 1.40, 0.60, 1.25, 0.70, 1.15, 0.75];
    let slot0_results = build_slot0_results_for_markets(&selected, &multipliers);
    rebalance(&HashMap::new(), 100.0, &slot0_results)
}

fn test_gas_assumptions() -> GasAssumptions {
    GasAssumptions {
        l1_fee_per_byte_wei: 1.0e11,
        ..GasAssumptions::default()
    }
}

fn assert_flash_brackets_are_well_formed(actions: &[Action]) {
    let mut in_flash_bracket = false;

    for (i, action) in actions.iter().enumerate() {
        match action {
            Action::FlashLoan { .. } => {
                assert!(
                    !in_flash_bracket,
                    "nested flash bracket at action index {i}"
                );
                in_flash_bracket = true;
            }
            Action::RepayFlashLoan { .. } => {
                assert!(
                    in_flash_bracket,
                    "repay outside flash bracket at action index {i}"
                );
                in_flash_bracket = false;
            }
            Action::Mint { .. } => {
                assert!(
                    in_flash_bracket,
                    "mint outside flash bracket at action index {i}"
                );
            }
            Action::Buy { .. } | Action::Sell { .. } | Action::Merge { .. } => {}
        }
    }

    assert!(
        !in_flash_bracket,
        "unterminated flash bracket at end of action stream"
    );
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
        let mut exec =
            ExecutionState::with_balances(&mut sims, &mut budget, &mut actions, &mut sim_balances);
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
        !actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. })),
        "no flash loan needed when no pool buys are required"
    );

    assert!((*sim_balances.get("M1").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M2").unwrap() - 0.0).abs() < 1e-9);
    assert!((*sim_balances.get("M3").unwrap() - 0.0).abs() < 1e-9);
    assert!(
        (sims[1].price - 0.05).abs() < 1e-9,
        "no buy => no price move"
    );
    assert!(
        (sims[2].price - 0.05).abs() < 1e-9,
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
        let mut exec =
            ExecutionState::with_balances(&mut sims, &mut budget, &mut actions, &mut sim_balances);
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
    assert_eq!(buys.len(), 1, "should only buy shortfall leg");
    assert_eq!(buys[0].0, "M2", "M2 had the shortfall");
    assert!(
        (buys[0].1 - 3.0).abs() < 1e-6,
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
fn test_execute_optimal_sell_keeps_profitable_complement_inventory() {
    let mut sims = build_three_sims([0.8, 0.05, 0.05]);
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("M1", 5.0);
    sim_balances.insert("M2", 5.0);
    sim_balances.insert("M3", 5.0);

    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec =
            ExecutionState::with_balances(&mut sims, &mut budget, &mut actions, &mut sim_balances);
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
        actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. })),
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
    assert_eq!(buys.len(), 2, "both complementary legs should be bought");
    assert!(buys.iter().all(|(_, amt)| *amt > 4.9));

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
        let mut exec =
            ExecutionState::with_balances(&mut sims, &mut budget, &mut actions, &mut sim_balances);
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
        !actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. })),
        "no pool buys means no flash loan"
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

    // Should have: FlashLoan, Buy×2, Merge, RepayFlashLoan
    let has_merge = actions.iter().any(|a| matches!(a, Action::Merge { .. }));
    let has_flash = actions
        .iter()
        .any(|a| matches!(a, Action::FlashLoan { .. }));
    let buy_count = actions
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .count();
    assert!(has_merge, "should have Merge action");
    assert!(has_flash, "should have FlashLoan action");
    assert_eq!(buy_count, 2, "should buy 2 non-source outcomes");

    // Other pool prices should have increased (we bought into them)
    assert!(
        sims[1].price > 0.05,
        "M2 price should increase after buying"
    );
    assert!(
        sims[2].price > 0.05,
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

    // === Expected value verification ===
    // Before: EV = 100.0 sUSD (no holdings)
    let initial_budget = 100.0;

    // Compute portfolio after rebalancing
    let mut holdings: HashMap<&str, f64> = HashMap::new();
    let mut total_cost = 0.0_f64;
    let mut total_sell_proceeds = 0.0_f64;
    for action in &actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                cost,
            } => {
                *holdings.entry(market_name).or_insert(0.0) += amount;
                total_cost += cost;
            }
            Action::Sell {
                market_name,
                amount,
                proceeds,
            } => {
                *holdings.entry(market_name).or_insert(0.0) -= amount;
                total_sell_proceeds += proceeds;
            }
            Action::Mint { amount, .. } => {
                // Mint gives all outcomes
                for (slot0, market) in &slot0_results {
                    let _ = slot0;
                    *holdings.entry(market.name).or_insert(0.0) += amount;
                }
            }
            Action::Merge { amount, .. } => {
                // Merge burns all outcomes, returns sUSD
                for (slot0, market) in &slot0_results {
                    let _ = slot0;
                    *holdings.entry(market.name).or_insert(0.0) -= amount;
                }
                total_sell_proceeds += amount; // sUSD recovered
            }
            _ => {}
        }
    }
    let remaining_budget = initial_budget - total_cost + total_sell_proceeds;

    // EV_after = remaining_sUSD + Σ prediction_i × holdings_i
    let ev_holdings: f64 = holdings
        .iter()
        .map(|(name, &units)| {
            let key = normalize_market_name(name);
            let pred = preds.get(&key).copied().unwrap_or(0.0);
            pred * units
        })
        .sum();
    let ev_after = remaining_budget + ev_holdings;

    // Verify no holdings are negative
    for (name, &units) in &holdings {
        assert!(units >= -1e-9, "negative holdings for {}: {}", name, units);
    }

    // Count unique outcomes bought
    let outcomes_bought: Vec<_> = holdings.iter().filter(|&(_, &u)| u > 1e-12).collect();

    println!("=== Expected Value Check ===");
    println!("  EV before:        {:.6} sUSD", initial_budget);
    println!("  EV after:         {:.6} sUSD", ev_after);
    println!(
        "  EV gain:          {:.6} sUSD ({:.2}%)",
        ev_after - initial_budget,
        (ev_after / initial_budget - 1.0) * 100.0
    );
    println!("  Remaining budget: {:.6} sUSD", remaining_budget);
    println!("  Holdings EV:      {:.6} sUSD", ev_holdings);
    println!(
        "  Outcomes held:    {}/{}",
        outcomes_bought.len(),
        slot0_results.len()
    );
    println!("  Total buy cost:   {:.6}", total_cost);
    println!("  Total sell proc:  {:.6}", total_sell_proceeds);

    // EV should increase (we're buying underpriced assets at 50% of prediction)
    assert!(
        ev_after > initial_budget,
        "EV should increase: before={:.6}, after={:.6}",
        initial_budget,
        ev_after
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
        let flash = actions
            .iter()
            .filter(|a| matches!(a, Action::FlashLoan { .. }))
            .count();
        println!(
            "[profile] {}: outcomes={}, per_call={:?}, actions={} (buys={}, sells={}, mints={}, merges={}, flash={})",
            name,
            slot0_results.len(),
            elapsed / iters as u32,
            actions.len(),
            buys,
            sells,
            mints,
            merges,
            flash
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
fn test_rebalance_output_is_plannable_with_default_edge_model() {
    let actions = sample_rebalance_actions();
    let groups = group_actions(&actions).expect("rebalance output should be groupable");

    let plans = build_group_plans_with_default_edges(
        &actions,
        &test_gas_assumptions(),
        1e-10,
        3000.0,
        BufferConfig::default(),
    )
    .expect("planning should succeed with built-in direct-buy edge mapping");

    let direct_buy_groups = groups
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
async fn test_rebalance_integration() {
    if std::env::var("RUN_NETWORK_TESTS").ok().as_deref() != Some("1") {
        return;
    }
    // Integration test with real pool data
    dotenvy::dotenv().ok();
    let rpc_url = match std::env::var("RPC") {
        Ok(url) => url,
        Err(_) => return, // skip if no RPC
    };
    let provider = alloy::providers::ProviderBuilder::new().with_reqwest(
        rpc_url.parse().unwrap(),
        |builder| {
            builder
                .no_proxy()
                .build()
                .expect("failed to build reqwest client for tests")
        },
    );

    let slot0_results = crate::pools::fetch_all_slot0(provider).await.unwrap();

    let actions = rebalance(&HashMap::new(), 100.0, &slot0_results);

    println!("Rebalance actions ({}):", actions.len());
    for action in &actions {
        match action {
            Action::Mint {
                contract_1,
                contract_2,
                amount,
                target_market,
            } => {
                println!(
                    "  MINT {} sets for {} (c1={}, c2={})",
                    amount, target_market, contract_1, contract_2
                )
            }
            Action::Buy {
                market_name,
                amount,
                cost,
            } => println!("  BUY {} {} (cost: {:.6})", amount, market_name, cost),
            Action::Sell {
                market_name,
                amount,
                proceeds,
            } => println!(
                "  SELL {} {} (proceeds: {:.6})",
                amount, market_name, proceeds
            ),
            Action::Merge {
                contract_1,
                contract_2,
                amount,
                source_market,
            } => {
                println!(
                    "  MERGE {} sets from {} (c1={}, c2={})",
                    amount, source_market, contract_1, contract_2
                )
            }
            Action::FlashLoan { amount } => println!("  FLASH_LOAN {:.6}", amount),
            Action::RepayFlashLoan { amount } => println!("  REPAY_FLASH_LOAN {:.6}", amount),
        }
    }
}
