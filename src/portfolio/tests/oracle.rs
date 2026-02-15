use std::collections::{HashMap, HashSet};

use super::super::merge::merge_sell_proceeds;
use super::super::planning::{
    PlannedRoute, active_skip_indices, plan_active_routes, plan_is_budget_feasible, solve_prof,
};
use super::super::rebalancer::rebalance;
use super::super::sim::{
    PoolSim, Route, alt_price, build_sims, profitability, target_price_for_prof,
};
use super::super::solver::mint_cost_to_prof;
use super::super::trading::ExecutionState;
use super::super::waterfall::{MAX_WATERFALL_ITERS, waterfall};
use super::{
    Action, TestRng, assert_action_values_are_finite, assert_flash_loan_ordering,
    assert_rebalance_action_invariants, brute_force_best_gain_mint_direct,
    build_rebalance_fuzz_case, build_slot0_results_for_markets, build_three_sims_with_preds,
    buy_totals, eligible_l1_markets_with_predictions, ev_from_state, flash_loan_totals,
    mock_slot0_market, mock_slot0_market_with_liquidity,
    mock_slot0_market_with_liquidity_and_ticks, oracle_direct_only_best_ev_grid,
    oracle_two_pool_direct_only_best_ev_with_holdings_grid, replay_actions_to_ev,
    replay_actions_to_market_state, replay_actions_to_state,
    slot0_for_market_with_multiplier_and_pool_liquidity,
};
use proptest::prelude::*;
use proptest::proptest;

#[test]
fn test_oracle_single_pool_overpriced_no_trade() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[1.35]);
    let budget = 75.0;
    let balances: HashMap<&str, f64> = HashMap::new();

    let actions = rebalance(&balances, budget, &slot0_results);
    assert!(
        actions.is_empty(),
        "overpriced single-pool case should not trade"
    );

    let ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
    assert!((ev - budget).abs() <= 1e-9);
}

#[test]
fn test_oracle_single_pool_direct_only_matches_grid_optimum() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[0.58]);
    let budget = 80.0;
    let balances: HashMap<&str, f64> = HashMap::new();

    let actions = rebalance(&balances, budget, &slot0_results);
    let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);

    let sims = {
        let preds = crate::pools::prediction_map();
        build_sims(&slot0_results, &preds).expect("fixture must include prediction for each market")
    };
    assert_eq!(sims.len(), 1);
    let oracle_ev = oracle_direct_only_best_ev_grid(&sims, budget, 2400);

    assert!(
        algo_ev + 1e-6 >= oracle_ev - 2e-3,
        "single-pool oracle gap too large: algo={:.9}, oracle={:.9}",
        algo_ev,
        oracle_ev
    );
}

#[test]
fn test_oracle_two_pool_direct_only_matches_grid_optimum() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[0.62, 0.74]);
    let budget = 120.0;
    let balances: HashMap<&str, f64> = HashMap::new();

    let actions = rebalance(&balances, budget, &slot0_results);
    let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);

    let sims = {
        let preds = crate::pools::prediction_map();
        build_sims(&slot0_results, &preds).expect("fixture must include prediction for each market")
    };
    assert_eq!(sims.len(), 2);
    let oracle_ev = oracle_direct_only_best_ev_grid(&sims, budget, 520);

    assert!(
        algo_ev + 1e-6 >= oracle_ev - 4e-3,
        "two-pool oracle gap too large: algo={:.9}, oracle={:.9}",
        algo_ev,
        oracle_ev
    );
}

#[test]
fn test_oracle_fuzz_two_pool_direct_only_not_worse_than_grid() {
    let mut rng = TestRng::new(0x1357_9BDF_2468_ACE0u64);
    let markets = eligible_l1_markets_with_predictions();
    for _ in 0..20 {
        let i = rng.pick(markets.len());
        let mut j = rng.pick(markets.len());
        while j == i {
            j = rng.pick(markets.len());
        }
        let selected = [markets[i], markets[j]];
        let multipliers = [rng.in_range(0.45, 1.45), rng.in_range(0.45, 1.45)];
        let slot0_results = build_slot0_results_for_markets(&selected, &multipliers);
        let budget = rng.in_range(1.0, 180.0);
        let balances: HashMap<&str, f64> = HashMap::new();

        let actions = rebalance(&balances, budget, &slot0_results);
        let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
        let sims = {
            let preds = crate::pools::prediction_map();
            build_sims(&slot0_results, &preds)
                .expect("fixture must include prediction for each market")
        };
        let oracle_ev = oracle_direct_only_best_ev_grid(&sims, budget, 260);

        assert!(
            algo_ev + 1e-6 >= oracle_ev - 1.2e-2,
            "oracle differential failed: algo={:.9}, oracle={:.9}, markets=({}, {}), multipliers=({:.4}, {:.4}), budget={:.4}",
            algo_ev,
            oracle_ev,
            selected[0].name,
            selected[1].name,
            multipliers[0],
            multipliers[1],
            budget
        );
    }
}

#[test]
fn test_oracle_two_pool_direct_only_with_legacy_holdings_matches_grid_optimum() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[1.35, 0.55]);
    let budget = 4.0;
    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(selected[0].name, 8.0);
    balances.insert(selected[1].name, 0.5);

    let actions = rebalance(&balances, budget, &slot0_results);
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::Mint { .. } | Action::Merge { .. })),
        "partial two-pool fixture should be direct-only"
    );
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. } | Action::RepayFlashLoan { .. })),
        "direct-only fixture should not use flash loans"
    );
    assert!(
        actions.iter().any(
            |a| matches!(a, Action::Sell { market_name, .. } if *market_name == selected[0].name)
        ),
        "overpriced legacy holding should trigger a sell"
    );
    assert!(
        actions.iter().any(
            |a| matches!(a, Action::Buy { market_name, .. } if *market_name == selected[1].name)
        ),
        "underpriced market should attract buy flow"
    );
    assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);

    let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
    let sims = {
        let preds = crate::pools::prediction_map();
        build_sims(&slot0_results, &preds).expect("fixture must include prediction for each market")
    };
    let initial_holdings = [
        balances.get(selected[0].name).copied().unwrap_or(0.0),
        balances.get(selected[1].name).copied().unwrap_or(0.0),
    ];
    let oracle_ev = oracle_two_pool_direct_only_best_ev_with_holdings_grid(
        &sims,
        &initial_holdings,
        budget,
        800,
    );
    assert!(
        algo_ev + 1e-6 >= oracle_ev - 7e-3,
        "legacy-holdings oracle gap too large: algo={:.9}, oracle={:.9}",
        algo_ev,
        oracle_ev
    );
}

#[test]
fn test_oracle_fuzz_two_pool_direct_only_with_legacy_holdings_not_worse_than_grid() {
    let mut rng = TestRng::new(0x7072_6F70_5F68_6F6Cu64);
    let markets = eligible_l1_markets_with_predictions();
    for _ in 0..24 {
        let i = rng.pick(markets.len());
        let mut j = rng.pick(markets.len());
        while j == i {
            j = rng.pick(markets.len());
        }
        let selected = [markets[i], markets[j]];
        let multipliers = [rng.in_range(0.40, 1.60), rng.in_range(0.40, 1.60)];
        let slot0_results = build_slot0_results_for_markets(&selected, &multipliers);
        let budget = rng.in_range(0.0, 180.0);

        let mut balances: HashMap<&str, f64> = HashMap::new();
        balances.insert(selected[0].name, rng.in_range(0.0, 14.0));
        balances.insert(selected[1].name, rng.in_range(0.0, 14.0));

        let actions = rebalance(&balances, budget, &slot0_results);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::Mint { .. } | Action::Merge { .. })),
            "partial two-pool fixture should be direct-only"
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::FlashLoan { .. } | Action::RepayFlashLoan { .. })),
            "direct-only fixture should not use flash loans"
        );
        assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);

        let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
        let sims = {
            let preds = crate::pools::prediction_map();
            build_sims(&slot0_results, &preds)
                .expect("fixture must include prediction for each market")
        };
        let initial_holdings = [
            balances.get(selected[0].name).copied().unwrap_or(0.0),
            balances.get(selected[1].name).copied().unwrap_or(0.0),
        ];
        let oracle_ev = oracle_two_pool_direct_only_best_ev_with_holdings_grid(
            &sims,
            &initial_holdings,
            budget,
            180,
        );

        assert!(
            algo_ev + 1e-6 >= oracle_ev - 1.5e-2,
            "legacy-holdings oracle differential failed: algo={:.9}, oracle={:.9}, markets=({}, {}), multipliers=({:.4}, {:.4}), budget={:.5}, holdings=({:.5}, {:.5})",
            algo_ev,
            oracle_ev,
            selected[0].name,
            selected[1].name,
            multipliers[0],
            multipliers[1],
            budget,
            initial_holdings[0],
            initial_holdings[1]
        );
    }
}

#[test]
fn test_oracle_two_pool_closed_form_direct_waterfall_matches_kkt_target() {
    let (slot0_a, market_a) =
        mock_slot0_market("CF_A", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 0.05);
    let (slot0_b, market_b) =
        mock_slot0_market("CF_B", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 0.04);

    let mut sims = vec![
        PoolSim::from_slot0(&slot0_a, market_a, 0.18).unwrap(),
        PoolSim::from_slot0(&slot0_b, market_b, 0.14).unwrap(),
    ];
    let sims_start = sims.clone();

    let initial_budget = 80.0;
    let mut budget = initial_budget;
    let mut actions = Vec::new();
    let last_prof = waterfall(&mut sims, &mut budget, &mut actions, false);

    assert!(
        actions.iter().all(|a| matches!(a, Action::Buy { .. })),
        "direct-only fixture should emit only buy actions"
    );
    let bought = buy_totals(&actions);
    assert!(
        bought.get("CF_A").copied().unwrap_or(0.0) > 1e-9,
        "first market should be in active set"
    );
    assert!(
        bought.get("CF_B").copied().unwrap_or(0.0) > 1e-9,
        "second market should be in active set"
    );

    let a_sum: f64 = sims_start
        .iter()
        .map(|s| s.l_eff() * s.prediction.sqrt())
        .sum();
    let b_sum: f64 = sims_start
        .iter()
        .map(|s| s.l_eff() * s.price().sqrt())
        .sum();
    let expected_prof = (a_sum / (initial_budget + b_sum)).powi(2) - 1.0;
    let prof_tol = 6e-6 * (1.0 + expected_prof.abs());
    assert!(
        (last_prof - expected_prof).abs() <= prof_tol,
        "closed-form and waterfall profitability should match: got={:.12}, expected={:.12}, tol={:.12}",
        last_prof,
        expected_prof,
        prof_tol
    );

    for sim in &sims {
        let target = target_price_for_prof(sim.prediction, expected_prof);
        assert!(
            target < sim.buy_limit_price - 1e-8,
            "fixture should stay uncapped to test pure KKT equalization"
        );
        let ptol = 8e-7 * (1.0 + target.abs());
        assert!(
            (sim.price() - target).abs() <= ptol,
            "final direct price should hit KKT target: market={}, got={:.12}, target={:.12}, tol={:.12}",
            sim.market_name,
            sim.price(),
            target,
            ptol
        );
    }

    let budget_tol = 4e-6 * (1.0 + initial_budget.abs());
    assert!(
        budget.abs() <= budget_tol,
        "waterfall should spend essentially all budget at boundary: leftover={:.12}, tol={:.12}",
        budget,
        budget_tol
    );
}

#[test]
fn test_mint_first_order_can_make_zero_cash_plan_feasible() {
    // Search adversarial mixed-route fixtures where:
    // - Mint leg is cash-positive (negative cost),
    // - Direct leg is cash-consuming,
    // - zero-cash feasibility holds only with mint-first ordering.
    let mut rng = TestRng::new(0x0FD3_A0A7_2026_4001u64);
    let mut witness: Option<(Vec<PoolSim>, f64, Vec<PlannedRoute>)> = None;

    'outer: for _ in 0..1400 {
        let p0 = rng.in_range(0.05, 0.40);
        let p1 = rng.in_range(0.52, 0.92);
        let p2 = rng.in_range(0.52, 0.92);
        let pred0 = (p0 + rng.in_range(0.08, 0.45)).min(0.98);
        let pred1 = rng.in_range(0.05, 0.45);
        let pred2 = rng.in_range(0.01, 0.30);

        let sims = build_three_sims_with_preds([p0, p1, p2], [pred0, pred1, pred2]);
        let active = vec![(0usize, Route::Mint), (0usize, Route::Direct)];
        let skip = active_skip_indices(&active);

        let p_direct = profitability(sims[0].prediction, sims[0].price());
        if p_direct <= 1e-9 {
            continue;
        }

        for k in 1..=20 {
            let target_prof = p_direct * (k as f64) / 22.0;
            let Some(plan) = plan_active_routes(&sims, &active, target_prof, &skip) else {
                continue;
            };
            if plan.len() != 2 {
                continue;
            }
            if plan[0].route != Route::Mint
                || plan[1].route != Route::Direct
                || plan[0].idx != 0
                || plan[1].idx != 0
            {
                continue;
            }
            let mut reversed = plan.clone();
            reversed.reverse();
            if plan[0].cost < -1e-6
                && plan[1].cost > 1e-6
                && plan_is_budget_feasible(&plan, 0.0)
                && !plan_is_budget_feasible(&reversed, 0.0)
            {
                witness = Some((sims.clone(), target_prof, plan));
                break 'outer;
            }
        }
    }

    let (sims, target_prof, plan) = witness.expect(
        "expected at least one order-sensitive zero-cash mixed-route fixture in sampled search",
    );
    let active = vec![(0usize, Route::Mint), (0usize, Route::Direct)];
    let skip = active_skip_indices(&active);
    assert!(
        plan[0].cost < 0.0 && plan[1].cost > 0.0,
        "witness should include cash-positive mint then cash-consuming direct step"
    );

    let mut exec_sims = sims.clone();
    let mut budget = 0.0;
    let mut actions = Vec::new();
    let ok = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec =
            ExecutionState::new(&mut exec_sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_planned_routes(&plan, &skip)
    };
    assert!(
        ok,
        "mint-first mixed plan should execute from zero cash at target_prof={:.9}",
        target_prof
    );
    assert!(budget >= -1e-9, "execution must not underflow cash");
    assert!(
        actions.iter().any(|a| matches!(a, Action::Mint { .. })),
        "execution should include mint leg"
    );
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::Buy { market_name, .. } if *market_name == "M1")),
        "execution should include direct buy leg on M1"
    );
}

#[test]
fn test_oracle_two_pool_direct_only_legacy_self_funding_budget_zero_matches_grid() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[1.45, 0.52]);
    let budget = 0.0;
    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(selected[0].name, 12.0);
    balances.insert(selected[1].name, 0.0);

    let ev_before = replay_actions_to_ev(&[], &slot0_results, &balances, budget);
    let actions = rebalance(&balances, budget, &slot0_results);
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::Mint { .. } | Action::Merge { .. })),
        "two-pool fixture should stay direct-only"
    );
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. } | Action::RepayFlashLoan { .. })),
        "direct-only fixture should not use flash loans"
    );
    assert!(
        actions.iter().any(
            |a| matches!(a, Action::Sell { market_name, .. } if *market_name == selected[0].name)
        ),
        "overpriced legacy holding should be sold"
    );
    assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);

    let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
    let sims = {
        let preds = crate::pools::prediction_map();
        build_sims(&slot0_results, &preds).expect("fixture must include prediction for each market")
    };
    let initial_holdings = [
        balances.get(selected[0].name).copied().unwrap_or(0.0),
        balances.get(selected[1].name).copied().unwrap_or(0.0),
    ];
    let oracle_ev = oracle_two_pool_direct_only_best_ev_with_holdings_grid(
        &sims,
        &initial_holdings,
        budget,
        1200,
    );

    assert!(
        algo_ev + 1e-6 >= oracle_ev - 1.1e-2,
        "self-funding legacy oracle gap too large: algo={:.9}, oracle={:.9}",
        algo_ev,
        oracle_ev
    );
    let ev_tol = 2e-6 * (1.0 + ev_before.abs() + algo_ev.abs());
    assert!(
        algo_ev + ev_tol >= ev_before,
        "self-funding rebalance should not reduce EV: before={:.9}, after={:.9}, tol={:.9}",
        ev_before,
        algo_ev,
        ev_tol
    );
}

#[test]
fn test_fuzz_rebalance_partial_direct_only_ev_non_decreasing() {
    let mut rng = TestRng::new(0xD15C_A5E0_2026_3001u64);
    for _ in 0..40 {
        let (slot0_results, balances_static, susd_balance) =
            build_rebalance_fuzz_case(&mut rng, true);
        let balances: HashMap<&str, f64> = balances_static
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();
        assert!(
            slot0_results.len() < crate::predictions::PREDICTIONS_L1.len(),
            "partial fixture should keep mint route disabled"
        );

        let ev_before = replay_actions_to_ev(&[], &slot0_results, &balances, susd_balance);
        let actions = rebalance(&balances, susd_balance, &slot0_results);
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::Mint { .. }
                    | Action::Merge { .. }
                    | Action::FlashLoan { .. }
                    | Action::RepayFlashLoan { .. }
            )),
            "partial direct-only fixture should not emit mint/merge/flash actions"
        );
        assert_rebalance_action_invariants(&actions, &slot0_results, &balances, susd_balance);

        let ev_after = replay_actions_to_ev(&actions, &slot0_results, &balances, susd_balance);
        let tol = 2e-4 * (1.0 + ev_before.abs() + ev_after.abs());
        assert!(
            ev_after + tol >= ev_before,
            "partial direct-only rebalance reduced EV: before={:.9}, after={:.9}, tol={:.9}",
            ev_before,
            ev_after,
            tol
        );
    }
}

#[test]
fn test_fuzz_rebalance_partial_no_legacy_holdings_emits_no_sells() {
    let mut rng = TestRng::new(0xA11C_EB00_2026_3002u64);
    for _ in 0..40 {
        let (slot0_results, _, susd_balance) = build_rebalance_fuzz_case(&mut rng, true);
        assert!(
            slot0_results.len() < crate::predictions::PREDICTIONS_L1.len(),
            "partial fixture should keep mint route disabled"
        );
        let balances: HashMap<&str, f64> = HashMap::new();

        let actions = rebalance(&balances, susd_balance, &slot0_results);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::Sell { .. } | Action::Merge { .. })),
            "without legacy inventory, rebalance should not emit sell/merge actions"
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::FlashLoan { .. } | Action::RepayFlashLoan { .. })),
            "partial fixture should not emit flash actions"
        );
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Mint { .. })),
            "partial fixture should not emit mint actions"
        );
        assert_rebalance_action_invariants(&actions, &slot0_results, &balances, susd_balance);
    }
}

#[test]
fn test_rebalance_negative_budget_legacy_sells_self_fund_rebalance() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[1.45, 0.52]);

    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(selected[0].name, 40.0);
    balances.insert(selected[1].name, 0.0);
    let budget = -0.5;

    let ev_before = replay_actions_to_ev(&[], &slot0_results, &balances, budget);
    let actions = rebalance(&balances, budget, &slot0_results);
    assert_action_values_are_finite(&actions);
    assert!(
        actions.iter().any(
            |a| matches!(a, Action::Sell { market_name, .. } if *market_name == selected[0].name)
        ),
        "negative-budget fixture should liquidate overpriced legacy holdings"
    );

    let (holdings_after, cash_after) =
        replay_actions_to_state(&actions, &slot0_results, &balances, budget);
    let ev_after = ev_from_state(&holdings_after, cash_after);
    let tol = 2e-6 * (1.0 + ev_before.abs() + ev_after.abs());
    assert!(
        ev_after + tol >= ev_before,
        "negative-budget rebalance should not reduce EV: before={:.9}, after={:.9}, tol={:.9}",
        ev_before,
        ev_after,
        tol
    );
    assert!(
        cash_after > budget + 1e-9,
        "phase-1 liquidation should improve cash from debt start: start={:.9}, end={:.9}",
        budget,
        cash_after
    );
}

#[test]
fn test_rebalance_handles_nan_and_infinite_budget_without_non_finite_actions() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[0.55, 0.60]);
    let balances: HashMap<&str, f64> = HashMap::new();

    let actions_nan = rebalance(&balances, f64::NAN, &slot0_results);
    assert!(
        actions_nan.is_empty(),
        "NaN budget should fail closed with no planned actions"
    );

    let actions_inf = rebalance(&balances, f64::INFINITY, &slot0_results);
    assert!(
        actions_inf.is_empty(),
        "infinite budget should fail closed with no planned actions"
    );
}

#[test]
fn test_rebalance_non_finite_balances_fail_closed_to_zero_inventory() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[1.35, 0.55]);

    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(selected[0].name, f64::NAN);
    balances.insert(selected[1].name, f64::INFINITY);

    let actions = rebalance(&balances, 0.0, &slot0_results);
    assert_action_values_are_finite(&actions);
    assert!(
        actions.iter().all(
            |a| !matches!(a, Action::Sell { market_name, .. } if *market_name == selected[0].name)
        ),
        "NaN legacy holdings should be sanitized to zero and never sold"
    );
    assert!(
        actions.iter().all(
            |a| !matches!(a, Action::Sell { market_name, .. } if *market_name == selected[1].name)
        ),
        "infinite legacy holdings should be sanitized to zero and never sold"
    );
}

#[test]
fn test_rebalance_zero_liquidity_outcome_disables_mint_merge_routes() {
    let markets = eligible_l1_markets_with_predictions();
    assert_eq!(
        markets.len(),
        crate::predictions::PREDICTIONS_L1.len(),
        "fixture should start from full-L1 coverage"
    );

    let multipliers = vec![0.55; markets.len()];
    let mut slot0_results = build_slot0_results_for_markets(&markets, &multipliers);
    // Force one entry to have zero liquidity so build_sims drops it.
    slot0_results[0] = slot0_for_market_with_multiplier_and_pool_liquidity(markets[0], 0.55, 0);

    let balances: HashMap<&str, f64> = HashMap::new();
    let budget = 35.0;
    let actions = rebalance(&balances, budget, &slot0_results);

    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::Mint { .. } | Action::Merge { .. })),
        "mint/merge must be disabled when any pooled outcome has zero liquidity"
    );
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. } | Action::RepayFlashLoan { .. })),
        "flash-loan legs should not appear when mint/merge routes are disabled"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "underpriced remaining outcomes should still trade via direct buys"
    );
    assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);
}

#[test]
fn test_phase3_near_tie_low_liquidity_avoids_ev_regression() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    // Build tiny-liquidity clones with nearly equal profitability, where churn risk is highest.
    let slot0_results = vec![
        slot0_for_market_with_multiplier_and_pool_liquidity(selected[0], 0.9950, 1_000_000_000_000),
        slot0_for_market_with_multiplier_and_pool_liquidity(selected[1], 0.9945, 1_000_000_000_000),
    ];

    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(selected[0].name, 30.0);
    let budget = 0.25;

    let ev_before = replay_actions_to_ev(&[], &slot0_results, &balances, budget);
    let actions = rebalance(&balances, budget, &slot0_results);
    assert!(
        !actions.iter().any(|a| matches!(
            a,
            Action::Mint { .. }
                | Action::Merge { .. }
                | Action::FlashLoan { .. }
                | Action::RepayFlashLoan { .. }
        )),
        "two-pool fixture should remain direct-only"
    );
    assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);

    let ev_after = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
    let tol = 2e-5 * (1.0 + ev_before.abs() + ev_after.abs());
    assert!(
        ev_after + tol >= ev_before,
        "near-tie low-liquidity scenario should not lose EV to churn: before={:.9}, after={:.9}, tol={:.9}",
        ev_before,
        ev_after,
        tol
    );
}

#[test]
fn test_phase3_recycling_full_l1_with_mint_routes_reduces_low_prof_legacy() {
    let markets = eligible_l1_markets_with_predictions();
    assert_eq!(
        markets.len(),
        crate::predictions::PREDICTIONS_L1.len(),
        "full fixture should include all tradeable L1 outcomes"
    );

    let multipliers: Vec<f64> = (0..markets.len())
        .map(|i| match i % 10 {
            0 => 0.46,
            1 => 0.58,
            2 => 0.72,
            3 => 0.87,
            4 => 0.995, // near-fair legacy bucket (low marginal profitability)
            5 => 1.08,
            6 => 1.19,
            7 => 1.31,
            8 => 0.64,
            _ => 0.53,
        })
        .collect();
    let slot0_results = build_slot0_results_for_markets(&markets, &multipliers);

    let mut balances: HashMap<&str, f64> = HashMap::new();
    let mut legacy_names: Vec<&str> = Vec::new();
    for (i, market) in markets.iter().enumerate() {
        if i % 10 == 4 {
            balances.insert(market.name, 3.5);
            legacy_names.push(market.name);
        }
    }
    let budget = 40.0;

    let ev_before = replay_actions_to_ev(&[], &slot0_results, &balances, budget);
    let actions = rebalance(&balances, budget, &slot0_results);
    assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);
    assert_action_values_are_finite(&actions);
    assert!(
        actions.iter().any(|a| matches!(a, Action::Mint { .. })),
        "full-L1 mixed fixture should exercise mint route"
    );
    assert!(
        actions.iter().any(
            |a| matches!(a, Action::Sell { market_name, .. } if legacy_names.contains(market_name))
        ),
        "expected liquidation from low-profitability legacy bucket in full-L1 fixture"
    );

    let (holdings_after, cash_after) =
        replay_actions_to_state(&actions, &slot0_results, &balances, budget);
    let ev_after = ev_from_state(&holdings_after, cash_after);
    let ev_tol = 2e-5 * (1.0 + ev_before.abs() + ev_after.abs());
    assert!(
        ev_after + ev_tol >= ev_before,
        "full-L1 phase3 recycling should not reduce EV: before={:.9}, after={:.9}, tol={:.9}",
        ev_before,
        ev_after,
        ev_tol
    );

    let reduced_legacy = legacy_names
        .iter()
        .filter(|name| {
            let before = balances.get(**name).copied().unwrap_or(0.0);
            let after = holdings_after.get(**name).copied().unwrap_or(0.0);
            after + 1e-8 < before
        })
        .count();
    assert!(
        reduced_legacy >= 1,
        "expected at least one legacy bucket holding to be reduced"
    );

    let (borrowed, repaid) = flash_loan_totals(&actions);
    let flash_tol = 1e-7 * (1.0 + borrowed.abs() + repaid.abs());
    assert!(
        (borrowed - repaid).abs() <= flash_tol,
        "flash totals must balance in full-L1 phase3 fixture"
    );
}

#[test]
fn test_fuzz_flash_loan_action_stream_ordering_invariants() {
    let mut rng = TestRng::new(0xF1A5_410A_2026_5001u64);
    let mut checked = 0usize;

    for _ in 0..24 {
        let (slot0_results, balances_static, susd_balance) =
            build_rebalance_fuzz_case(&mut rng, false);
        let balances: HashMap<&str, f64> = balances_static
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();

        let actions = rebalance(&balances, susd_balance, &slot0_results);
        let brackets = assert_flash_loan_ordering(&actions);
        if brackets > 0 {
            checked += 1;
        }
    }

    assert!(
        checked >= 1,
        "expected at least one full-L1 fuzz fixture to exercise flash-loan brackets"
    );
}

#[test]
fn test_waterfall_misnormalized_prediction_sums_remain_finite() {
    let scenarios = [
        // predictions sum > 1
        ([0.12, 0.11, 0.10], [0.60, 0.60, 0.60], 20.0, true),
        // predictions sum < 1
        ([0.03, 0.04, 0.05], [0.10, 0.10, 0.10], 20.0, true),
        // high-sum direct-only path
        ([0.08, 0.09, 0.07], [0.55, 0.65, 0.58], 12.0, false),
    ];

    for (prices, preds, start_budget, mint_available) in scenarios {
        let mut sims = build_three_sims_with_preds(prices, preds);
        let mut budget = start_budget;
        let mut actions = Vec::new();

        let prof = waterfall(&mut sims, &mut budget, &mut actions, mint_available);
        assert!(prof.is_finite());
        assert!(budget.is_finite() && budget >= -1e-7);
        assert_action_values_are_finite(&actions);
        for sim in &sims {
            assert!(sim.price().is_finite() && sim.price() > 0.0);
        }
    }
}

#[test]
fn test_oracle_phase3_recycling_two_pool_direct_only_matches_grid_optimum() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1]];
    // Prof(A) = ~2.04%, Prof(B) = 150% regardless of absolute prediction values.
    // This creates a known "legacy capital recycling" pressure from A -> B.
    let slot0_results = build_slot0_results_for_markets(&selected, &[0.98, 0.40]);
    let budget = 1.0;
    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(selected[0].name, 30.0);
    balances.insert(selected[1].name, 0.0);

    let actions = rebalance(&balances, budget, &slot0_results);
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::Mint { .. } | Action::Merge { .. })),
        "partial two-pool fixture should remain direct-only"
    );
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. } | Action::RepayFlashLoan { .. })),
        "direct-only fixture should not use flash loans"
    );
    assert!(
        actions.iter().any(
            |a| matches!(a, Action::Sell { market_name, .. } if *market_name == selected[0].name)
        ),
        "expected recycling sell from low-profitability legacy holding"
    );
    assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);

    let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
    let sims = {
        let preds = crate::pools::prediction_map();
        build_sims(&slot0_results, &preds).expect("fixture must include prediction for each market")
    };
    let initial_holdings = [
        balances.get(selected[0].name).copied().unwrap_or(0.0),
        balances.get(selected[1].name).copied().unwrap_or(0.0),
    ];
    let oracle_ev = oracle_two_pool_direct_only_best_ev_with_holdings_grid(
        &sims,
        &initial_holdings,
        budget,
        1800,
    );

    assert!(
        algo_ev + 1e-6 >= oracle_ev - 9e-3,
        "phase3 recycling oracle gap too large: algo={:.9}, oracle={:.9}",
        algo_ev,
        oracle_ev
    );
}

#[test]
fn test_phase1_merge_split_can_leave_source_pool_overpriced() {
    let mut sims = build_three_sims_with_preds([0.8, 0.05, 0.05], [0.3, 0.3, 0.3]);
    let source_idx = 0usize;
    let source_name = sims[source_idx].market_name;
    let prediction = sims[source_idx].prediction;
    let price_before = sims[source_idx].price();

    let (tokens_needed, _, _) = sims[source_idx]
        .sell_to_price(prediction)
        .expect("sell_to_price should compute a direct sell amount");
    assert!(tokens_needed > 0.0);

    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert(source_name, tokens_needed * 1.5);
    sim_balances.insert(sims[1].market_name, 0.0);
    sim_balances.insert(sims[2].market_name, 0.0);

    let mut actions = Vec::new();
    let mut budget = 0.0;
    let sold = {
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell(
            source_idx,
            tokens_needed, // Mirrors Phase 1's "sell until direct price reaches prediction" amount.
            0.0,
            true,
        )
    };

    assert!(sold > 0.0);
    assert!(
        actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "fixture should route at least part of Phase 1 sell through merge"
    );
    assert!(
        sims[source_idx].price() > prediction + 1e-5,
        "source pool can remain overpriced after Phase 1 split: before={:.9}, after={:.9}, pred={:.9}",
        price_before,
        sims[source_idx].price(),
        prediction
    );
}

#[test]
fn test_rebalance_phase1_clears_or_fairs_legacy_overpriced_source_full_l1() {
    let markets = eligible_l1_markets_with_predictions();
    let source_idx = 0usize;
    let source_name = markets[source_idx].name;

    // All outcomes overpriced (suppress phase-2 buying). Source is most overpriced.
    let mut multipliers = vec![1.22; markets.len()];
    multipliers[source_idx] = 1.45;
    let slot0_results = build_slot0_results_for_markets(&markets, &multipliers);

    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(source_name, 24.0);
    // Provide complementary inventory so merge can be exercised without pool buys.
    for market in markets.iter().skip(1) {
        balances.insert(market.name, 2.0);
    }
    let budget = 0.0;

    let actions = rebalance(&balances, budget, &slot0_results);
    assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::Sell { market_name, .. } if *market_name == source_name)),
        "overpriced legacy source should trigger sells"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "fixture should exercise merge path in full-L1 mode"
    );

    let (holdings_after, _) = replay_actions_to_state(&actions, &slot0_results, &balances, budget);
    let slot0_after = replay_actions_to_market_state(&actions, &slot0_results);
    let sims_after = {
        let preds = crate::pools::prediction_map();
        build_sims(&slot0_after, &preds).expect("fixture must include prediction for each market")
    };
    let source_sim_after = sims_after
        .iter()
        .find(|s| s.market_name == source_name)
        .expect("source market should exist in replayed sims");
    let source_held_after = holdings_after
        .get(source_name)
        .copied()
        .unwrap_or(0.0)
        .max(0.0);
    let mut legacy_remaining = balances.get(source_name).copied().unwrap_or(0.0).max(0.0);
    for action in &actions {
        match action {
            Action::Sell {
                market_name,
                amount,
                ..
            } if *market_name == source_name => {
                legacy_remaining = (legacy_remaining - *amount).max(0.0);
            }
            Action::Merge { amount, .. } => {
                legacy_remaining = (legacy_remaining - *amount).max(0.0);
            }
            _ => {}
        }
    }

    assert!(
        legacy_remaining <= 1e-8 || source_sim_after.price() <= source_sim_after.prediction + 1e-8,
        "legacy overpriced source should not remain both legacy-held and overpriced: legacy_remaining={:.9}, final_held={:.9}, price={:.9}, pred={:.9}",
        legacy_remaining,
        source_held_after,
        source_sim_after.price(),
        source_sim_after.prediction
    );
}

#[test]
fn test_fuzz_phase1_sell_order_budget_stability() {
    let mut rng = TestRng::new(0x0BAD_5E11_0123_4567u64);
    let mut max_gap = 0.0_f64;

    for _ in 0..220 {
        let p0 = rng.in_range(0.35, 0.75);
        let p1 = rng.in_range(0.12, 0.45);
        let p2 = rng.in_range(0.01, 0.10);
        let pred0 = p0 * rng.in_range(0.25, 0.75);
        let pred1 = p1 * rng.in_range(0.25, 0.75);
        let pred2 = rng.in_range(0.02, 0.35);
        let sims_base = build_three_sims_with_preds([p0, p1, p2], [pred0, pred1, pred2]);

        if !(sims_base[0].price() > sims_base[0].prediction
            && sims_base[1].price() > sims_base[1].prediction)
        {
            continue;
        }

        let mut base_balances: HashMap<&'static str, f64> = HashMap::new();
        base_balances.insert(sims_base[0].market_name, rng.in_range(8.0, 20.0));
        base_balances.insert(sims_base[1].market_name, rng.in_range(8.0, 20.0));
        base_balances.insert(sims_base[2].market_name, rng.in_range(0.0, 2.0));

        let run_phase1 = |order: [usize; 2]| -> f64 {
            let mut sims = sims_base.clone();
            let mut balances: HashMap<&'static str, f64> = base_balances.clone();

            let mut budget = 0.0_f64;
            let mut actions = Vec::new();
            for idx in order {
                let price = sims[idx].price();
                if price <= sims[idx].prediction {
                    continue;
                }
                let held = *balances.get(sims[idx].market_name).unwrap_or(&0.0);
                if held <= 0.0 {
                    continue;
                }
                let (tokens_needed, _, _) = sims[idx]
                    .sell_to_price(sims[idx].prediction)
                    .unwrap_or((0.0, 0.0, sims[idx].price()));
                let sell_amount = if tokens_needed > 0.0 && tokens_needed <= held {
                    tokens_needed
                } else {
                    held
                };
                if sell_amount <= 0.0 {
                    continue;
                }
                let _ = {
                    let mut exec =
                        ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut balances);
                    exec.execute_optimal_sell(idx, sell_amount, 0.0, true)
                };
            }
            budget
        };

        let budget_01 = run_phase1([0, 1]);
        let budget_10 = run_phase1([1, 0]);
        let gap = (budget_01 - budget_10).abs();
        if gap > max_gap {
            max_gap = gap;
        }
    }

    // With correct tick bounds, the merge route is fully functional, so sell ordering
    // has a measurable impact on recovered budget. Tolerance reflects this reality.
    assert!(
        max_gap <= 0.15,
        "sampled Phase 1 fixtures should be near order-stable; max_gap={:.12}",
        max_gap
    );
}

#[test]
fn test_fuzz_plan_execute_cost_consistency_near_mint_caps() {
    let mut rng = TestRng::new(0xFEED_C0DE_2026_1001u64);
    let mut checked = 0usize;
    for _ in 0..900 {
        let p0 = rng.in_range(0.18, 0.55);
        let p1 = rng.in_range(0.03, 0.15);
        let p2 = rng.in_range(0.55, 0.90);
        let alt0 = 1.0 - (p1 + p2);
        if alt0 <= 0.02 {
            continue;
        }
        let pred0_lo = (alt0 + 0.03).min(0.95);
        let pred1_lo = (p1 + 0.03).min(0.95);
        if pred0_lo >= 0.99 || pred1_lo >= 0.99 {
            continue;
        }
        let pred0 = rng.in_range(pred0_lo, 0.99);
        let pred1 = rng.in_range(pred1_lo, 0.99);
        let pred2 = rng.in_range(0.01, 0.60);

        let mut sims = build_three_sims_with_preds([p0, p1, p2], [pred0, pred1, pred2]);
        // With skip={0,1}, mint on idx=0 sells only idx=2. Shrink idx=2 range to make it cap-edge.
        let shrink = rng.in_range(1e-6, 2e-3);
        sims[2].sell_limit_price = (sims[2].price() * (1.0 - shrink)).max(1e-12);

        let active = vec![(0usize, Route::Mint), (1usize, Route::Direct)];
        let skip = active_skip_indices(&active);
        let price_sum: f64 = sims.iter().map(|s| s.price()).sum();
        let p_mint = profitability(sims[0].prediction, alt_price(&sims, 0, price_sum));
        let p_direct = profitability(sims[1].prediction, sims[1].price());
        if !(p_mint.is_finite() && p_direct.is_finite() && p_mint > 1e-8 && p_direct > 1e-8) {
            continue;
        }

        let target_prof = (p_mint.min(p_direct) * rng.in_range(0.80, 0.99)).max(0.0);
        let Some(plan) = plan_active_routes(&sims, &active, target_prof, &skip) else {
            continue;
        };
        let plan_cost: f64 = plan.iter().map(|s| s.cost).sum();
        if !plan_cost.is_finite() || plan_cost <= 1e-10 {
            continue;
        }

        let mut exec_sims = sims.clone();
        let start_budget = plan_cost + 0.2;
        let mut budget = start_budget;
        let mut actions = Vec::new();
        let ok = {
            let mut unused_bal: HashMap<&str, f64> = HashMap::new();
            let mut exec =
                ExecutionState::new(&mut exec_sims, &mut budget, &mut actions, &mut unused_bal);
            exec.execute_planned_routes(&plan, &skip)
        };
        assert!(ok, "feasible near-cap plan should execute");

        let spent = start_budget - budget;
        let tol = 2e-6 * (1.0 + plan_cost.abs() + spent.abs());
        assert!(
            (spent - plan_cost).abs() <= tol,
            "near-cap plan/execute cost drift too large: planned={:.12}, spent={:.12}, tol={:.12}",
            plan_cost,
            spent,
            tol
        );
        checked += 1;
    }

    assert!(
        checked >= 1,
        "insufficient valid near-cap mixed-route fixtures: {}",
        checked
    );
}

#[test]
fn test_fuzz_pool_sim_kappa_lambda_finite_difference_accuracy() {
    let mut rng = TestRng::new(0xBADC_AB1E_2026_2002u64);
    for _ in 0..320 {
        let liquidity = (10f64.powf(rng.in_range(17.0, 24.0))).round() as u128;
        let tick_span = 25_000 + (rng.pick(130_000) as i32);
        let price = rng.in_range(0.01, 0.9);
        let pred = rng.in_range(0.02, 0.95);
        let (slot0, market) = mock_slot0_market_with_liquidity_and_ticks(
            "FD_ACC",
            "0x1212121212121212121212121212121212121212",
            price,
            liquidity,
            -tick_span,
            tick_span,
        );
        let Some(sim) = PoolSim::from_slot0(&slot0, market, pred) else {
            continue;
        };
        let p0 = sim.price();

        let max_sell = sim.max_sell_tokens();
        let k = sim.kappa();
        if max_sell > 1e-12 && k > 0.0 {
            let req_sell = (1e-6 / k).clamp(1e-12, max_sell * 0.2);
            if let Some((sold, _, p_after_sell)) = sim.sell_exact(req_sell) {
                if sold > 1e-12 {
                    let d_num = (p_after_sell - p0) / sold;
                    let d_model = -2.0 * p0 * k;
                    let d_tol = 5e-4 * (1.0 + d_model.abs());
                    assert!(
                        (d_num - d_model).abs() <= d_tol,
                        "sell finite-difference mismatch: num={:.12}, model={:.12}, tol={:.12}, p0={:.9}, sold={:.9}, k={:.9}",
                        d_num,
                        d_model,
                        d_tol,
                        p0,
                        sold,
                        k
                    );

                    let p_model = p0 / (1.0 + sold * k).powi(2);
                    let p_tol = 2e-10 * (1.0 + p_after_sell.abs() + p_model.abs());
                    assert!(
                        (p_after_sell - p_model).abs() <= p_tol,
                        "sell price formula drift: actual={:.12}, model={:.12}, tol={:.12}",
                        p_after_sell,
                        p_model,
                        p_tol
                    );
                }
            }
        }

        let max_buy = sim.max_buy_tokens();
        let lam = sim.lambda();
        if max_buy > 1e-12 && lam > 0.0 {
            let req_buy = (1e-6 / lam).clamp(1e-12, max_buy * 0.2);
            if let Some((bought, _, p_after_buy)) = sim.buy_exact(req_buy) {
                if bought > 1e-12 {
                    let d_num = (p_after_buy - p0) / bought;
                    let d_model = 2.0 * p0 * lam;
                    let d_tol = 5e-4 * (1.0 + d_model.abs());
                    assert!(
                        (d_num - d_model).abs() <= d_tol,
                        "buy finite-difference mismatch: num={:.12}, model={:.12}, tol={:.12}, p0={:.9}, bought={:.9}, lam={:.9}",
                        d_num,
                        d_model,
                        d_tol,
                        p0,
                        bought,
                        lam
                    );

                    let p_model = p0 / (1.0 - bought * lam).powi(2);
                    let p_tol = 2e-10 * (1.0 + p_after_buy.abs() + p_model.abs());
                    assert!(
                        (p_after_buy - p_model).abs() <= p_tol,
                        "buy price formula drift: actual={:.12}, model={:.12}, tol={:.12}",
                        p_after_buy,
                        p_model,
                        p_tol
                    );
                }
            }
        }
    }
}

#[test]
fn test_direct_closed_form_target_can_overshoot_tick_boundary() {
    let (slot0, market) = mock_slot0_market_with_liquidity(
        "closed_form_cap",
        "0x9999999999999999999999999999999999999999",
        0.05,
        1_000_000_000_000_000_000,
    );
    let sims = vec![PoolSim::from_slot0(&slot0, market, 0.95).unwrap()];
    let active = vec![(0usize, Route::Direct)];
    let skip = active_skip_indices(&active);

    let prof_hi = profitability(sims[0].prediction, sims[0].price());
    let prof = solve_prof(&sims, &active, prof_hi, 0.0, 1_000_000.0, &skip);
    let target_price = target_price_for_prof(sims[0].prediction, prof);
    assert!(
        target_price > sims[0].buy_limit_price + 1e-9,
        "adversarial fixture expects closed-form target to exceed tick boundary"
    );

    let plan = plan_active_routes(&sims, &active, prof, &skip)
        .expect("direct plan should still clamp to executable boundary");
    let planned_price = plan[0]
        .new_price
        .expect("direct route should carry a target execution price");
    assert!(
        planned_price <= sims[0].buy_limit_price + 1e-12,
        "execution planning should clamp to tick boundary"
    );
}

#[test]
fn test_waterfall_tiny_liquidity_no_nan_no_overspend() {
    let (slot0, market) = mock_slot0_market_with_liquidity(
        "tiny_liq",
        "0x1111111111111111111111111111111111111111",
        0.05,
        1,
    );
    let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.9).unwrap()];
    let mut budget = 10.0;
    let mut actions = Vec::new();

    let last_prof = waterfall(&mut sims, &mut budget, &mut actions, false);
    assert!(last_prof.is_finite());
    assert!(budget.is_finite());
    assert!(budget >= -1e-6, "budget should not go negative");
    assert!(
        actions.len() <= MAX_WATERFALL_ITERS,
        "waterfall should not exceed iteration cap"
    );
    for a in &actions {
        if let Action::Buy { amount, cost, .. } = a {
            assert!(amount.is_finite() && *amount >= 0.0);
            assert!(cost.is_finite() && *cost >= 0.0);
        }
    }
}

#[test]
fn test_mint_cost_to_prof_all_legs_capped_is_unreachable() {
    let mut sims = build_three_sims_with_preds([0.08, 0.09, 0.10], [0.8, 0.1, 0.1]);
    for i in 1..3 {
        sims[i].sell_limit_price = sims[i].price();
    }
    let price_sum: f64 = sims.iter().map(|s| s.price()).sum();
    let current_alt = alt_price(&sims, 0, price_sum);
    let tp = (current_alt + 0.2).min(0.99);
    let target_prof = sims[0].prediction / tp - 1.0;

    let res = mint_cost_to_prof(&sims, 0, target_prof, &HashSet::new(), price_sum);
    assert!(
        res.is_none(),
        "when all non-target legs are capped and target alt is above cap, mint route should be unreachable"
    );
}

#[test]
fn test_mixed_route_plan_execute_budget_consistency() {
    let sims = build_three_sims_with_preds([0.12, 0.08, 0.07], [0.8, 0.45, 0.45]);
    let active = vec![(0, Route::Mint), (1, Route::Direct)];
    let skip = active_skip_indices(&active);
    let price_sum: f64 = sims.iter().map(|s| s.price()).sum();
    let p0 = profitability(sims[0].prediction, alt_price(&sims, 0, price_sum));
    let p1 = profitability(sims[1].prediction, sims[1].price());
    let target_prof = (p0.min(p1) * 0.85).max(0.0);

    let plan = plan_active_routes(&sims, &active, target_prof, &skip)
        .expect("plan should exist for mixed-route fixture");
    let plan_cost: f64 = plan.iter().map(|s| s.cost).sum();
    assert!(plan_cost.is_finite() && plan_cost >= 0.0);

    let mut exec_sims = sims.clone();
    let mut budget = plan_cost + 0.5;
    let mut actions = Vec::new();
    let ok = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec =
            ExecutionState::new(&mut exec_sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_planned_routes(&plan, &skip)
    };
    assert!(
        ok,
        "execution of a feasible mixed-route plan should succeed"
    );
    let spent = (plan_cost + 0.5) - budget;
    let tol = 1e-7 * (1.0 + plan_cost.abs());
    assert!(
        (spent - plan_cost).abs() <= tol,
        "executed spend should match planned spend: spent={:.12}, planned={:.12}, tol={:.12}",
        spent,
        plan_cost,
        tol
    );
    assert!(budget >= -1e-7);
}

#[test]
fn test_flash_loans_balance_in_full_rebalance_fuzz() {
    let mut rng = TestRng::new(0xDEAD_BEEF_2026_0001u64);
    for _ in 0..20 {
        let (slot0_results, balances_static, susd_balance) =
            build_rebalance_fuzz_case(&mut rng, false);
        let balances: HashMap<&str, f64> = balances_static
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();
        let actions = rebalance(&balances, susd_balance, &slot0_results);
        let (borrowed, repaid) = flash_loan_totals(&actions);
        let tol = 1e-7 * (1.0 + borrowed.abs() + repaid.abs());
        assert!(
            (borrowed - repaid).abs() <= tol,
            "flash loan mismatch: borrowed={:.12}, repaid={:.12}, tol={:.12}",
            borrowed,
            repaid,
            tol
        );
    }
}

#[test]
fn test_buy_sell_to_price_exact_tick_boundary_hits() {
    let (slot0, market) = mock_slot0_market(
        "boundary",
        "0x1111111111111111111111111111111111111111",
        0.05,
    );
    let sim = PoolSim::from_slot0(&slot0, market, 0.6).unwrap();

    let (buy_cost, buy_amount, buy_price) = sim.cost_to_price(sim.buy_limit_price).unwrap();
    assert!(buy_cost.is_finite() && buy_cost >= 0.0);
    assert!(buy_amount.is_finite() && buy_amount >= 0.0);
    assert!(
        (buy_price - sim.buy_limit_price).abs() <= 1e-12 * (1.0 + sim.buy_limit_price.abs()),
        "buy target at limit should clamp exactly to buy limit"
    );

    let (sell_tokens, sell_proceeds, sell_price) = sim.sell_to_price(sim.sell_limit_price).unwrap();
    assert!(sell_tokens.is_finite() && sell_tokens >= 0.0);
    assert!(sell_proceeds.is_finite() && sell_proceeds >= 0.0);
    assert!(
        (sell_price - sim.sell_limit_price).abs() <= 1e-12 * (1.0 + sim.sell_limit_price.abs()),
        "sell target at limit should clamp exactly to sell limit"
    );
}

#[test]
fn test_dust_budget_produces_no_actions() {
    let (slot0, market) = mock_slot0_market(
        "dust_budget",
        "0x1111111111111111111111111111111111111111",
        0.02,
    );
    let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.9).unwrap()];
    let mut budget = 1e-15;
    let mut actions = Vec::new();
    let prof = waterfall(&mut sims, &mut budget, &mut actions, false);
    assert_eq!(prof, 0.0);
    assert!(actions.is_empty());
    assert!((budget - 1e-15).abs() <= 1e-24);
}

#[test]
fn test_rebalance_permutation_invariance_by_ev() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1], markets[2], markets[3]];
    let multipliers = [0.55, 0.70, 1.20, 0.92];
    let slot0_results = build_slot0_results_for_markets(&selected, &multipliers);
    let mut reversed = slot0_results.clone();
    reversed.reverse();

    let mut balances: HashMap<&str, f64> = HashMap::new();
    balances.insert(selected[0].name, 2.5);
    balances.insert(selected[1].name, 0.9);
    let budget = 63.0;

    let actions_a = rebalance(&balances, budget, &slot0_results);
    let actions_b = rebalance(&balances, budget, &reversed);
    let ev_a = replay_actions_to_ev(&actions_a, &slot0_results, &balances, budget);
    let ev_b = replay_actions_to_ev(&actions_b, &reversed, &balances, budget);
    let tol = 2e-6 * (1.0 + ev_a.abs() + ev_b.abs());
    assert!(
        (ev_a - ev_b).abs() <= tol,
        "rebalance EV should be permutation-invariant: a={:.12}, b={:.12}, tol={:.12}",
        ev_a,
        ev_b,
        tol
    );
}

#[test]
fn test_waterfall_scale_invariance_direct_only() {
    let (s1a, m1a) = mock_slot0_market_with_liquidity(
        "SCALE_A1",
        "0x1111111111111111111111111111111111111111",
        0.05,
        1_000_000_000_000_000_000,
    );
    let (s1b, m1b) = mock_slot0_market_with_liquidity(
        "SCALE_B1",
        "0x2222222222222222222222222222222222222222",
        0.06,
        1_000_000_000_000_000_000,
    );
    let (s2a, m2a) = mock_slot0_market_with_liquidity(
        "SCALE_A2",
        "0x3333333333333333333333333333333333333333",
        0.05,
        100_000_000_000_000_000_000,
    );
    let (s2b, m2b) = mock_slot0_market_with_liquidity(
        "SCALE_B2",
        "0x4444444444444444444444444444444444444444",
        0.06,
        100_000_000_000_000_000_000,
    );

    let mut sims_small = vec![
        PoolSim::from_slot0(&s1a, m1a, 0.18).unwrap(),
        PoolSim::from_slot0(&s1b, m1b, 0.17).unwrap(),
    ];
    let mut sims_big = vec![
        PoolSim::from_slot0(&s2a, m2a, 0.18).unwrap(),
        PoolSim::from_slot0(&s2b, m2b, 0.17).unwrap(),
    ];

    let mut budget_small = 10.0;
    let mut budget_big = 1000.0;
    let mut actions_small = Vec::new();
    let mut actions_big = Vec::new();

    let prof_small = waterfall(
        &mut sims_small,
        &mut budget_small,
        &mut actions_small,
        false,
    );
    let prof_big = waterfall(&mut sims_big, &mut budget_big, &mut actions_big, false);

    let prof_tol = 5e-5 * (1.0 + prof_small.abs() + prof_big.abs());
    assert!(
        (prof_small - prof_big).abs() <= prof_tol,
        "scaled liquidity+budget should preserve target profitability: small={:.9}, big={:.9}, tol={:.9}",
        prof_small,
        prof_big,
        prof_tol
    );

    let small_totals = buy_totals(&actions_small);
    let big_totals = buy_totals(&actions_big);
    let small_a = small_totals.get("SCALE_A1").copied().unwrap_or(0.0);
    let small_b = small_totals.get("SCALE_B1").copied().unwrap_or(0.0);
    let big_a = big_totals.get("SCALE_A2").copied().unwrap_or(0.0);
    let big_b = big_totals.get("SCALE_B2").copied().unwrap_or(0.0);
    if small_a > 1e-9 {
        let ratio = big_a / small_a;
        assert!(
            (ratio - 100.0).abs() <= 1.0,
            "scaled amount ratio for A should be ~100, got {}",
            ratio
        );
    }
    if small_b > 1e-9 {
        let ratio = big_b / small_b;
        assert!(
            (ratio - 100.0).abs() <= 1.0,
            "scaled amount ratio for B should be ~100, got {}",
            ratio
        );
    }
}

#[test]
fn test_zero_prediction_market_is_not_bought() {
    let (slot0, market) = mock_slot0_market(
        "zero_pred",
        "0x1111111111111111111111111111111111111111",
        0.2,
    );
    let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.0).unwrap()];
    let mut budget = 100.0;
    let mut actions = Vec::new();

    let prof = waterfall(&mut sims, &mut budget, &mut actions, false);
    assert_eq!(prof, 0.0);
    assert!(actions.is_empty(), "zero prediction should never be bought");
    assert!((budget - 100.0).abs() < 1e-12);
}

#[test]
fn test_large_budget_rebalance_stays_finite() {
    let markets = eligible_l1_markets_with_predictions();
    let selected = [markets[0], markets[1], markets[2]];
    let slot0_results = build_slot0_results_for_markets(&selected, &[0.50, 0.65, 0.80]);
    let balances: HashMap<&str, f64> = HashMap::new();
    let budget = 1_000_000_000.0;
    let actions = rebalance(&balances, budget, &slot0_results);
    for a in &actions {
        match a {
            Action::Buy { amount, cost, .. } => {
                assert!(amount.is_finite() && cost.is_finite());
            }
            Action::Sell {
                amount, proceeds, ..
            } => {
                assert!(amount.is_finite() && proceeds.is_finite());
            }
            Action::Mint { amount, .. }
            | Action::Merge { amount, .. }
            | Action::FlashLoan { amount }
            | Action::RepayFlashLoan { amount } => {
                assert!(amount.is_finite());
            }
        }
    }
    let ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
    assert!(
        ev.is_finite(),
        "EV after large-budget rebalance must be finite"
    );
}

#[test]
fn test_rebalance_double_run_idempotent_after_market_replay_fuzz() {
    let mut rng = TestRng::new(0xC0DE_F00D_2026_0001u64);
    for _ in 0..20 {
        let (slot0_results, balances_static, susd_balance) =
            build_rebalance_fuzz_case(&mut rng, true);
        let balances: HashMap<&str, f64> = balances_static
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();
        assert!(
            slot0_results.len() < crate::predictions::PREDICTIONS_L1.len(),
            "idempotency fuzz fixture should disable mint/merge routes"
        );

        let mut initial_holdings: HashMap<&'static str, f64> = HashMap::new();
        for (_, market) in &slot0_results {
            initial_holdings.insert(
                market.name,
                balances.get(market.name).copied().unwrap_or(0.0).max(0.0),
            );
        }
        let ev_before = ev_from_state(&initial_holdings, susd_balance);

        let actions_first = rebalance(&balances, susd_balance, &slot0_results);
        assert!(
            !actions_first
                .iter()
                .any(|a| matches!(a, Action::Mint { .. } | Action::Merge { .. })),
            "partial fixture should not emit mint/merge actions"
        );
        assert_rebalance_action_invariants(&actions_first, &slot0_results, &balances, susd_balance);
        let (holdings_first, cash_first) =
            replay_actions_to_state(&actions_first, &slot0_results, &balances, susd_balance);
        let ev_after_first = ev_from_state(&holdings_first, cash_first);
        let first_gain = (ev_after_first - ev_before).max(0.0);

        let slot0_after_first = replay_actions_to_market_state(&actions_first, &slot0_results);
        let balances_after_first: HashMap<&str, f64> = holdings_first
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();
        let actions_second = rebalance(&balances_after_first, cash_first, &slot0_after_first);
        assert!(
            !actions_second
                .iter()
                .any(|a| matches!(a, Action::Mint { .. } | Action::Merge { .. })),
            "partial fixture should not emit mint/merge actions"
        );
        assert_rebalance_action_invariants(
            &actions_second,
            &slot0_after_first,
            &balances_after_first,
            cash_first,
        );
        let (holdings_second, cash_second) = replay_actions_to_state(
            &actions_second,
            &slot0_after_first,
            &balances_after_first,
            cash_first,
        );
        let ev_after_second = ev_from_state(&holdings_second, cash_second);
        let second_gain = (ev_after_second - ev_after_first).max(0.0);

        let monotone_tol = 1e-4 * (1.0 + ev_after_first.abs() + ev_after_second.abs());
        assert!(
            ev_after_second + monotone_tol >= ev_after_first,
            "second rebalance should not reduce EV after market replay: ev1={:.12}, ev2={:.12}, tol={:.12}",
            ev_after_first,
            ev_after_second,
            monotone_tol
        );

        let second_gain_cap = 0.05 * (1.0 + first_gain);
        assert!(
            second_gain <= second_gain_cap + 1e-6,
            "second rebalance should be near-idempotent after replayed market impact: gain1={:.9}, gain2={:.9}, cap={:.9}",
            first_gain,
            second_gain,
            second_gain_cap
        );
    }
}

#[test]
fn test_buy_sell_roundtrip_has_no_free_cash_profit_fuzz() {
    let mut rng = TestRng::new(0xBADC_0FFE_2026_0002u64);
    for _ in 0..280 {
        let liquidity = (10f64.powf(rng.in_range(15.0, 22.0))).round() as u128;
        let tick_span = 20_000 + (rng.pick(140_000) as i32);
        let price = rng.in_range(0.01, 0.9);
        let (slot0, market) = mock_slot0_market_with_liquidity_and_ticks(
            "ROUNDTRIP",
            "0x7777777777777777777777777777777777777777",
            price,
            liquidity,
            -tick_span,
            tick_span,
        );
        let Some(sim) = PoolSim::from_slot0(&slot0, market, 0.5) else {
            continue;
        };
        // Skip iterations where random tick/price combo places price outside tick bounds.
        if sim.price() < sim.sell_limit_price || sim.price() > sim.buy_limit_price {
            continue;
        }
        let max_buy = sim.max_buy_tokens();
        if max_buy <= 1e-10 {
            continue;
        }

        let req_buy = max_buy * rng.in_range(0.001, 0.5);
        let Some((bought, cost, new_price)) = sim.buy_exact(req_buy) else {
            continue;
        };
        if bought <= 1e-10 {
            continue;
        }

        let mut unwind = sim.clone();
        unwind.set_price(new_price);
        let mut remaining = bought;
        let mut proceeds_total = 0.0_f64;
        for _ in 0..4 {
            if remaining <= 1e-12 {
                break;
            }
            let Some((sold, proceeds, unwind_price)) = unwind.sell_exact(remaining) else {
                break;
            };
            if sold <= 1e-12 {
                break;
            }
            proceeds_total += proceeds;
            remaining = (remaining - sold).max(0.0);
            unwind.set_price(unwind_price);
        }
        let cash_tol = 1e-8 * (1.0 + cost.abs() + proceeds_total.abs());
        assert!(
            proceeds_total <= cost + cash_tol,
            "buy->sell roundtrip should not produce free cash even after iterative unwind: cost={:.12}, proceeds_total={:.12}, remaining={:.12}, tol={:.12}, start_price={:.6}, liquidity={}",
            cost,
            proceeds_total,
            remaining,
            cash_tol,
            price,
            liquidity
        );
    }
}

#[test]
fn test_merge_preferred_in_extreme_price_regime_wide_ticks() {
    let (s0, m0) = mock_slot0_market_with_liquidity_and_ticks(
        "WR_M1",
        "0x1111111111111111111111111111111111111111",
        0.90,
        1_000_000_000_000_000_000_000,
        -180_000,
        180_000,
    );
    let (s1, m1) = mock_slot0_market_with_liquidity_and_ticks(
        "WR_M2",
        "0x2222222222222222222222222222222222222222",
        0.03,
        1_000_000_000_000_000_000_000,
        -180_000,
        180_000,
    );
    let (s2, m2) = mock_slot0_market_with_liquidity_and_ticks(
        "WR_M3",
        "0x3333333333333333333333333333333333333333",
        0.04,
        1_000_000_000_000_000_000_000,
        -180_000,
        180_000,
    );

    let sims = vec![
        PoolSim::from_slot0(&s0, m0, 0.30).unwrap(),
        PoolSim::from_slot0(&s1, m1, 0.20).unwrap(),
        PoolSim::from_slot0(&s2, m2, 0.20).unwrap(),
    ];
    let sell_amount = 3.0;
    let (merge_net, merge_actual) = merge_sell_proceeds(&sims, 0, sell_amount);
    let (_, direct_proceeds, _) = sims[0].sell_exact(sell_amount).unwrap();
    assert!(merge_actual > 0.0);
    assert!(
        merge_net > direct_proceeds + 1e-6,
        "merge should dominate direct in high-source/cheap-complements regime: merge={:.9}, direct={:.9}",
        merge_net,
        direct_proceeds
    );

    let mut exec_sims = sims.clone();
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("WR_M1", sell_amount);
    sim_balances.insert("WR_M2", 0.0);
    sim_balances.insert("WR_M3", 0.0);
    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec =
            ExecutionState::new(&mut exec_sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell(0, sell_amount, f64::INFINITY, true)
    };
    assert!(sold > 0.0);
    assert!(
        actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "optimal sell should use merge in this regime"
    );
}

#[test]
fn test_direct_preferred_when_complements_expensive_wide_ticks() {
    let (s0, m0) = mock_slot0_market_with_liquidity_and_ticks(
        "WD_M1",
        "0x1111111111111111111111111111111111111111",
        0.08,
        1_000_000_000_000_000_000_000,
        -180_000,
        180_000,
    );
    let (s1, m1) = mock_slot0_market_with_liquidity_and_ticks(
        "WD_M2",
        "0x2222222222222222222222222222222222222222",
        0.92,
        1_000_000_000_000_000_000_000,
        -180_000,
        180_000,
    );
    let (s2, m2) = mock_slot0_market_with_liquidity_and_ticks(
        "WD_M3",
        "0x3333333333333333333333333333333333333333",
        0.92,
        1_000_000_000_000_000_000_000,
        -180_000,
        180_000,
    );

    let sims = vec![
        PoolSim::from_slot0(&s0, m0, 0.20).unwrap(),
        PoolSim::from_slot0(&s1, m1, 0.10).unwrap(),
        PoolSim::from_slot0(&s2, m2, 0.10).unwrap(),
    ];
    let sell_amount = 2.0;
    let (merge_net, merge_actual) = merge_sell_proceeds(&sims, 0, sell_amount);
    let (_, direct_proceeds, _) = sims[0].sell_exact(sell_amount).unwrap();
    assert!(merge_actual > 0.0);
    assert!(
        direct_proceeds > merge_net + 1e-6,
        "direct should dominate merge when complements are expensive: merge={:.9}, direct={:.9}",
        merge_net,
        direct_proceeds
    );

    let mut exec_sims = sims.clone();
    let mut sim_balances: HashMap<&'static str, f64> = HashMap::new();
    sim_balances.insert("WD_M1", sell_amount);
    sim_balances.insert("WD_M2", 0.0);
    sim_balances.insert("WD_M3", 0.0);
    let mut budget = 0.0;
    let mut actions = Vec::new();
    let sold = {
        let mut exec =
            ExecutionState::new(&mut exec_sims, &mut budget, &mut actions, &mut sim_balances);
        exec.execute_optimal_sell(0, sell_amount, f64::INFINITY, true)
    };
    assert!(sold > 0.0);
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "optimal sell should avoid merge in this regime"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Sell { .. })),
        "direct path should emit sell action"
    );
}

#[test]
fn test_mint_direct_mixed_route_matches_bruteforce_gain_fuzz() {
    let mut rng = TestRng::new(0xC105_EDCE_2026_0003u64);
    let mut checked = 0usize;
    for _ in 0..80 {
        let p0 = rng.in_range(0.18, 0.55);
        let p1 = rng.in_range(0.03, 0.15);
        let p2 = rng.in_range(0.55, 0.90);
        let alt0 = 1.0 - (p1 + p2);
        if alt0 <= 0.02 {
            continue;
        }
        let pred0_lo = (alt0 + 0.03).min(0.95);
        let pred1_lo = (p1 + 0.03).min(0.95);
        if pred0_lo >= 0.99 || pred1_lo >= 0.99 {
            continue;
        }
        let pred0 = rng.in_range(pred0_lo, 0.99);
        let pred1 = rng.in_range(pred1_lo, 0.99);
        let pred2 = rng.in_range(0.01, 0.60);

        let (s0, m0) = mock_slot0_market_with_liquidity_and_ticks(
            "MX_M1",
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            p0,
            1_000_000_000_000_000_000_000,
            -220_000,
            220_000,
        );
        let (s1, m1) = mock_slot0_market_with_liquidity_and_ticks(
            "MX_M2",
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            p1,
            1_000_000_000_000_000_000_000,
            -220_000,
            220_000,
        );
        let (s2, m2) = mock_slot0_market_with_liquidity_and_ticks(
            "MX_M3",
            "0xcccccccccccccccccccccccccccccccccccccccc",
            p2,
            1_000_000_000_000_000_000_000,
            -220_000,
            220_000,
        );
        let slot0_results = vec![(s0, m0), (s1, m1), (s2, m2)];
        let sims = vec![
            PoolSim::from_slot0(&slot0_results[0].0, slot0_results[0].1, pred0).unwrap(),
            PoolSim::from_slot0(&slot0_results[1].0, slot0_results[1].1, pred1).unwrap(),
            PoolSim::from_slot0(&slot0_results[2].0, slot0_results[2].1, pred2).unwrap(),
        ];

        let active = vec![(0usize, Route::Mint), (1usize, Route::Direct)];
        let skip = active_skip_indices(&active);
        let price_sum: f64 = sims.iter().map(|s| s.price()).sum();
        let p_mint = profitability(sims[0].prediction, alt_price(&sims, 0, price_sum));
        let p_direct = profitability(sims[1].prediction, sims[1].price());
        if !(p_mint > 1e-6 && p_direct > 1e-6) {
            continue;
        }

        let budget = rng.in_range(1.0, 40.0);
        let prof_hi = p_mint.max(p_direct);
        let prof_lo = 0.0;
        let achievable = solve_prof(&sims, &active, prof_hi, prof_lo, budget, &skip);
        let Some(plan) = plan_active_routes(&sims, &active, achievable, &skip) else {
            continue;
        };
        if !plan_is_budget_feasible(&plan, budget) {
            continue;
        }

        let mut exec_sims = sims.clone();
        let mut remaining_budget = budget;
        let mut actions = Vec::new();
        let ok = {
            let mut unused_bal: HashMap<&str, f64> = HashMap::new();
            let mut exec = ExecutionState::new(
                &mut exec_sims,
                &mut remaining_budget,
                &mut actions,
                &mut unused_bal,
            );
            exec.execute_planned_routes(&plan, &skip)
        };
        assert!(ok);

        let mut idx_by_market: HashMap<&str, usize> = HashMap::new();
        for (i, s) in sims.iter().enumerate() {
            idx_by_market.insert(s.market_name, i);
        }
        let mut holdings = vec![0.0_f64; sims.len()];
        let mut spent = 0.0_f64;
        for action in &actions {
            match action {
                Action::Buy {
                    market_name,
                    amount,
                    cost,
                } => {
                    if let Some(&idx) = idx_by_market.get(market_name) {
                        holdings[idx] += *amount;
                        spent += *cost;
                    }
                }
                Action::Sell {
                    market_name,
                    amount,
                    proceeds,
                } => {
                    if let Some(&idx) = idx_by_market.get(market_name) {
                        holdings[idx] -= *amount;
                        spent -= *proceeds;
                    }
                }
                Action::Mint { amount, .. } => {
                    for h in &mut holdings {
                        *h += *amount;
                    }
                    spent += *amount;
                }
                Action::Merge { amount, .. } => {
                    for h in &mut holdings {
                        *h -= *amount;
                    }
                    spent -= *amount;
                }
                Action::FlashLoan { .. } | Action::RepayFlashLoan { .. } => {}
            }
        }
        let algo_gain: f64 = holdings
            .iter()
            .enumerate()
            .map(|(i, h)| sims[i].prediction * *h)
            .sum::<f64>()
            - spent;
        let oracle_gain = brute_force_best_gain_mint_direct(&sims, 0, 1, budget, &skip, 320);
        let gap_tol = 3.0e-2 * (1.0 + oracle_gain.abs());
        assert!(
            algo_gain + gap_tol >= oracle_gain,
            "mint/direct differential oracle failed: algo_gain={:.9}, oracle_gain={:.9}, tol={:.9}, p=({:.4},{:.4},{:.4}), pred=({:.4},{:.4},{:.4}), budget={:.6}",
            algo_gain,
            oracle_gain,
            gap_tol,
            p0,
            p1,
            p2,
            pred0,
            pred1,
            pred2,
            budget
        );
        checked += 1;
    }
    assert!(
        checked >= 20,
        "insufficient valid mixed-route fuzz cases: {}",
        checked
    );
}

#[test]
fn test_exact_budget_match_plan_executes_without_underflow() {
    let (slot0, market) = mock_slot0_market(
        "exact_budget",
        "0x1111111111111111111111111111111111111111",
        0.05,
    );
    let sims = vec![PoolSim::from_slot0(&slot0, market, 0.30).unwrap()];
    let active = vec![(0, Route::Direct)];
    let skip = active_skip_indices(&active);
    let target_prof = 1.0;
    let plan = plan_active_routes(&sims, &active, target_prof, &skip)
        .expect("single direct route should be plannable");
    let required_budget: f64 = plan.iter().map(|s| s.cost).sum();
    assert!(required_budget.is_finite() && required_budget > 0.0);
    assert!(plan_is_budget_feasible(&plan, required_budget));

    let mut exec_sims = sims.clone();
    let mut budget = required_budget;
    let mut actions = Vec::new();
    let ok = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec =
            ExecutionState::new(&mut exec_sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_planned_routes(&plan, &skip)
    };
    assert!(ok, "exact-budget plan should execute");
    assert!(
        budget >= -1e-10,
        "budget should not underflow on exact match, got {}",
        budget
    );
    let tol = 1e-8 * (1.0 + required_budget.abs());
    assert!(
        budget.abs() <= tol,
        "exact-budget execution should leave near-zero residual: residual={}, tol={}",
        budget,
        tol
    );
}

#[test]
fn test_waterfall_idempotent_after_equilibrium() {
    let mut sims = build_three_sims_with_preds([0.03, 0.04, 0.05], [0.90, 0.85, 0.80]);
    let mut budget = 10_000.0;
    let mut actions_first = Vec::new();
    let _prof_first = waterfall(&mut sims, &mut budget, &mut actions_first, false);
    assert!(!actions_first.is_empty(), "first pass should trade");

    let budget_before_second = budget;
    let mut actions_second = Vec::new();
    let prof_second = waterfall(&mut sims, &mut budget, &mut actions_second, false);
    assert!(
        actions_second.is_empty(),
        "second pass at equilibrium should not emit new buy actions"
    );
    assert!(
        prof_second <= 1e-9,
        "second pass profitability should be exhausted, got {}",
        prof_second
    );
    assert!(
        (budget - budget_before_second).abs() <= 1e-12 * (1.0 + budget_before_second.abs()),
        "budget should be unchanged on idempotent pass"
    );
}

#[test]
fn test_waterfall_hard_caps_converges() {
    let (s0, m0) = mock_slot0_market_with_liquidity(
        "hard_cap_a",
        "0x1111111111111111111111111111111111111111",
        0.04,
        1_000,
    );
    let (s1, m1) = mock_slot0_market_with_liquidity(
        "hard_cap_b",
        "0x2222222222222222222222222222222222222222",
        0.045,
        1_000,
    );
    let (s2, m2) = mock_slot0_market_with_liquidity(
        "hard_cap_c",
        "0x3333333333333333333333333333333333333333",
        0.05,
        1_000,
    );
    let mut sims = vec![
        PoolSim::from_slot0(&s0, m0, 0.95).unwrap(),
        PoolSim::from_slot0(&s1, m1, 0.95).unwrap(),
        PoolSim::from_slot0(&s2, m2, 0.95).unwrap(),
    ];
    let mut budget = 1_000_000.0;
    let mut actions = Vec::new();
    let last_prof = waterfall(&mut sims, &mut budget, &mut actions, false);

    assert!(last_prof.is_finite());
    assert!(budget.is_finite());
    assert!(
        actions.len() <= MAX_WATERFALL_ITERS * sims.len(),
        "hard-cap convergence should not spin excessively"
    );
    let capped = sims
        .iter()
        .filter(|s| (s.price() - s.buy_limit_price).abs() <= 1e-9 * (1.0 + s.buy_limit_price))
        .count();
    assert!(
        capped >= 2,
        "expected most markets to hit hard caps under huge budget"
    );

    let mut second_actions = Vec::new();
    let second_prof = waterfall(&mut sims, &mut budget, &mut second_actions, false);
    assert!(
        second_actions.is_empty(),
        "after cap convergence, subsequent pass should not trade"
    );
    assert!(second_prof <= 1e-9);
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        max_shrink_iters: 512,
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_pool_sim_buy_sell_bounds(
        start_price in 0.005f64..0.18f64,
        pred in 0.02f64..0.95f64,
        buy_frac in 0.0f64..1.0f64,
        sell_frac in 0.0f64..1.0f64
    ) {
        let (slot0, market) = mock_slot0_market(
            "PROP_BOUNDS",
            "0x1111111111111111111111111111111111111111",
            start_price,
        );
        let sim = PoolSim::from_slot0(&slot0, market, pred).unwrap();

        let req_buy = sim.max_buy_tokens() * buy_frac.clamp(0.0, 1.0);
        let (bought, cost, p_after_buy) = sim.buy_exact(req_buy).unwrap();
        prop_assert!(bought.is_finite() && cost.is_finite() && p_after_buy.is_finite());
        prop_assert!(bought >= -1e-12 && bought <= sim.max_buy_tokens() + 1e-8);
        prop_assert!(cost >= -1e-12);
        prop_assert!(p_after_buy + 1e-12 >= sim.price());
        prop_assert!(p_after_buy <= sim.buy_limit_price + 1e-8);

        let req_sell = sim.max_sell_tokens() * sell_frac.clamp(0.0, 1.0);
        let (sold, proceeds, p_after_sell) = sim.sell_exact(req_sell).unwrap();
        prop_assert!(sold.is_finite() && proceeds.is_finite() && p_after_sell.is_finite());
        prop_assert!(sold >= -1e-12 && sold <= sim.max_sell_tokens() + 1e-8);
        prop_assert!(proceeds >= -1e-12);
        prop_assert!(p_after_sell <= sim.price() + 1e-12);
        prop_assert!(p_after_sell + 1e-8 >= sim.sell_limit_price);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_iters: 512,
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_solve_prof_budget_monotone_mixed(
        p0 in 0.01f64..0.18f64,
        p1 in 0.01f64..0.18f64,
        p2 in 0.01f64..0.18f64,
        pred0 in 0.03f64..0.95f64,
        pred1 in 0.03f64..0.95f64,
        pred2 in 0.03f64..0.95f64,
        lo_frac in 0.0f64..0.85f64,
        b_small_frac in 0.0f64..0.9f64,
        b_extra_frac in 0.02f64..0.6f64
    ) {
        let sims = build_three_sims_with_preds([p0, p1, p2], [pred0, pred1, pred2]);
        let active = vec![(0, Route::Direct), (1, Route::Mint)];
        let skip = active_skip_indices(&active);
        let price_sum: f64 = sims.iter().map(|s| s.price()).sum();
        let p_direct = profitability(sims[0].prediction, sims[0].price());
        let p_mint = profitability(sims[1].prediction, alt_price(&sims, 1, price_sum));
        prop_assume!(p_direct.is_finite() && p_mint.is_finite() && p_direct > 1e-6 && p_mint > 1e-6);

        let prof_hi = p_direct.max(p_mint);
        let prof_lo = (prof_hi * lo_frac).max(0.0);
        let plan_lo_opt = plan_active_routes(&sims, &active, prof_lo, &skip);
        prop_assume!(plan_lo_opt.is_some());
        let required_budget: f64 = plan_lo_opt.unwrap().iter().map(|s| s.cost).sum();
        prop_assume!(required_budget.is_finite() && required_budget > 1e-6);

        let budget_small = required_budget * b_small_frac;
        let budget_large = budget_small + required_budget * b_extra_frac;
        let prof_small = solve_prof(&sims, &active, prof_hi, prof_lo, budget_small, &skip);
        let prof_large = solve_prof(&sims, &active, prof_hi, prof_lo, budget_large, &skip);

        prop_assert!(prof_small.is_finite() && prof_large.is_finite());
        prop_assert!(prof_small >= prof_lo - 1e-9 && prof_small <= prof_hi + 1e-9);
        prop_assert!(prof_large >= prof_lo - 1e-9 && prof_large <= prof_hi + 1e-9);
        prop_assert!(
            prof_small + 1e-8 >= prof_large,
            "more budget should not require a higher target profitability: small={}, large={}",
            prof_small,
            prof_large
        );
    }
}
