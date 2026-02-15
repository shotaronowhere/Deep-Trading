use std::collections::{HashMap, HashSet};

use super::super::merge::{merge_sell_cap_with_inventory, optimal_sell_split_with_inventory};
use super::super::planning::{
    active_skip_indices, plan_active_routes, plan_is_budget_feasible, solve_prof,
};
use super::super::rebalancer::rebalance;
use super::super::sim::{PoolSim, Route, alt_price, profitability};
use super::super::solver::mint_cost_to_prof;
use super::super::waterfall::waterfall;
use super::{
    Action, TestRng, assert_rebalance_action_invariants, brute_force_best_split_with_inventory,
    build_rebalance_fuzz_case, build_slot0_results_for_markets, build_three_sims_with_preds,
    eligible_l1_markets_with_predictions, mock_slot0_market, replay_actions_to_ev,
};

#[test]
fn test_fuzz_pool_sim_swap_invariants() {
    let mut rng = TestRng::new(0xA5A5_1234_DEAD_BEEFu64);
    for _ in 0..400 {
        let start_price = rng.in_range(0.005, 0.18);
        let pred = rng.in_range(0.02, 0.95);
        let (slot0, market) = mock_slot0_market(
            "FUZZ_SWAP",
            "0x1111111111111111111111111111111111111111",
            start_price,
        );
        let sim = PoolSim::from_slot0(&slot0, market, pred).unwrap();

        let max_buy = sim.max_buy_tokens();
        let req_buy = rng.in_range(0.0, (1.5 * max_buy).max(1e-6));
        let (bought, cost, buy_price) = sim.buy_exact(req_buy).unwrap();
        assert!(bought >= -1e-12 && bought <= max_buy + 1e-9);
        assert!(cost.is_finite() && cost >= -1e-12);
        assert!(buy_price.is_finite());
        assert!(buy_price + 1e-12 >= sim.price());
        assert!(buy_price <= sim.buy_limit_price + 1e-8);

        let max_sell = sim.max_sell_tokens();
        let req_sell = rng.in_range(0.0, (1.5 * max_sell).max(1e-6));
        let (sold, proceeds, sell_price) = sim.sell_exact(req_sell).unwrap();
        assert!(sold >= -1e-12 && sold <= max_sell + 1e-9);
        assert!(proceeds.is_finite() && proceeds >= -1e-12);
        assert!(sell_price.is_finite());
        assert!(sell_price <= sim.price() + 1e-12);
        assert!(sell_price + 1e-8 >= sim.sell_limit_price);
    }
}

#[test]
fn test_fuzz_mint_newton_solver_hits_target_or_saturation() {
    let mut rng = TestRng::new(0xBADC_0FFE_1234_5678u64);
    for _ in 0..300 {
        let prices = [
            rng.in_range(0.01, 0.18),
            rng.in_range(0.01, 0.18),
            rng.in_range(0.01, 0.18),
        ];
        let preds = [
            rng.in_range(0.03, 0.95),
            rng.in_range(0.03, 0.95),
            rng.in_range(0.03, 0.95),
        ];
        let sims = build_three_sims_with_preds(prices, preds);
        let target_idx = rng.pick(3);
        let price_sum: f64 = sims.iter().map(|s| s.price()).sum();
        let current_alt = alt_price(&sims, target_idx, price_sum);

        let mut saturated = sims.to_vec();
        for i in 0..saturated.len() {
            if i == target_idx {
                continue;
            }
            let cap = saturated[i].max_sell_tokens();
            if cap <= 0.0 {
                continue;
            }
            if let Some((sold, _, p_new)) = saturated[i].sell_exact(cap) {
                if sold > 0.0 {
                    saturated[i].set_price(p_new);
                }
            }
        }
        let saturated_sum: f64 = saturated.iter().map(|s| s.price()).sum();
        let alt_cap = alt_price(&saturated, target_idx, saturated_sum);
        if alt_cap <= current_alt + 1e-8 {
            continue;
        }

        let tp_min = (current_alt + 1e-6).max(1e-5);
        if tp_min >= 0.995 {
            continue;
        }
        let reachable_hi = (alt_cap - 1e-6).min(0.995);

        let tp = if rng.chance(1, 4) && alt_cap + 1e-4 < 0.995 {
            rng.in_range((alt_cap + 1e-4).max(tp_min), 0.995)
        } else if reachable_hi > tp_min {
            rng.in_range(tp_min, reachable_hi)
        } else {
            continue;
        };

        let target_prof = sims[target_idx].prediction / tp - 1.0;
        let result = mint_cost_to_prof(&sims, target_idx, target_prof, &HashSet::new(), price_sum);

        let Some((cash_cost, value_cost, mint_amount, d_cost_d_pi)) = result else {
            // The solver can fail only for unreachable target alt-prices.
            assert!(tp > alt_cap + 1e-6);
            continue;
        };

        assert!(cash_cost.is_finite());
        assert!(value_cost.is_finite());
        assert!(mint_amount.is_finite() && mint_amount >= 0.0);
        assert!(d_cost_d_pi.is_finite());
        assert!(
            d_cost_d_pi <= 1e-8,
            "cash cost should be non-increasing in target profitability"
        );
        assert!(value_cost <= cash_cost + 1e-9);

        let mut simulated = sims.to_vec();
        let mut proceeds = 0.0_f64;
        for i in 0..simulated.len() {
            if i == target_idx {
                continue;
            }
            if let Some((sold, leg_proceeds, p_new)) = simulated[i].sell_exact(mint_amount) {
                if sold > 0.0 {
                    simulated[i].set_price(p_new);
                    proceeds += leg_proceeds;
                }
            }
        }
        let simulated_cost = mint_amount - proceeds;
        let simulated_sum: f64 = simulated.iter().map(|s| s.price()).sum();
        let alt_after = alt_price(&simulated, target_idx, simulated_sum);

        let cost_tol = 2e-7 * (1.0 + simulated_cost.abs() + cash_cost.abs());
        assert!(
            (simulated_cost - cash_cost).abs() <= cost_tol,
            "simulated and analytical mint cash costs diverged: sim={:.12}, analytical={:.12}, tol={:.12}",
            simulated_cost,
            cash_cost,
            cost_tol
        );
        assert!(alt_after + 1e-8 >= current_alt);
        assert!(alt_after <= alt_cap + 1e-8);

        if tp <= alt_cap - 1e-5 {
            let alt_tol = 3e-5 * (1.0 + tp.abs());
            assert!(
                (alt_after - tp).abs() <= alt_tol,
                "reachable target alt-price was not hit: target={:.9}, got={:.9}, tol={:.9}",
                tp,
                alt_after,
                alt_tol
            );
        } else if tp >= alt_cap + 1e-5 {
            let alt_tol = 3e-5 * (1.0 + alt_cap.abs());
            assert!(
                (alt_after - alt_cap).abs() <= alt_tol,
                "unreachable target should saturate near cap: cap={:.9}, got={:.9}, tol={:.9}",
                alt_cap,
                alt_after,
                alt_tol
            );
        }
    }
}

#[test]
fn test_fuzz_solve_prof_monotonic_with_budget_mixed_routes() {
    let mut rng = TestRng::new(0x1234_5678_9ABC_DEF0u64);
    for _ in 0..250 {
        let prices = [
            rng.in_range(0.01, 0.18),
            rng.in_range(0.01, 0.18),
            rng.in_range(0.01, 0.18),
        ];
        let preds = [
            rng.in_range(0.03, 0.95),
            rng.in_range(0.03, 0.95),
            rng.in_range(0.03, 0.95),
        ];
        let sims = build_three_sims_with_preds(prices, preds);

        let active = vec![(0, Route::Direct), (1, Route::Mint)];
        let skip = active_skip_indices(&active);
        let price_sum: f64 = sims.iter().map(|s| s.price()).sum();

        let p_direct = profitability(sims[0].prediction, sims[0].price());
        let p_mint = profitability(sims[1].prediction, alt_price(&sims, 1, price_sum));
        if !p_direct.is_finite() || !p_mint.is_finite() || p_direct <= 1e-6 || p_mint <= 1e-6 {
            continue;
        }
        // Mirror waterfall semantics: prof_hi is the current equalized level and must be affordable.
        let prof_hi = p_direct.max(p_mint);
        let prof_lo = (prof_hi * rng.in_range(0.0, 0.85)).max(0.0);

        let Some(plan_lo) = plan_active_routes(&sims, &active, prof_lo, &skip) else {
            continue;
        };
        let required_budget: f64 = plan_lo.iter().map(|s| s.cost).sum();
        if !required_budget.is_finite() || required_budget <= 1e-6 {
            continue;
        }

        let budget_small = rng.in_range(0.0, required_budget * 0.9);
        let budget_large =
            budget_small + rng.in_range(required_budget * 0.02, required_budget * 0.6);

        let prof_small = solve_prof(&sims, &active, prof_hi, prof_lo, budget_small, &skip);
        let prof_large = solve_prof(&sims, &active, prof_hi, prof_lo, budget_large, &skip);

        assert!(prof_small.is_finite() && prof_large.is_finite());
        assert!(prof_small >= prof_lo - 1e-9 && prof_small <= prof_hi + 1e-9);
        assert!(prof_large >= prof_lo - 1e-9 && prof_large <= prof_hi + 1e-9);
        assert!(
            prof_small + 1e-8 >= prof_large,
            "more budget should not force a higher target profitability: small={:.9}, large={:.9}",
            prof_small,
            prof_large
        );

        let plan_small = plan_active_routes(&sims, &active, prof_small, &skip).unwrap();
        let plan_large = plan_active_routes(&sims, &active, prof_large, &skip).unwrap();
        assert!(plan_is_budget_feasible(&plan_small, budget_small));
        assert!(plan_is_budget_feasible(&plan_large, budget_large));
    }
}

#[test]
fn test_fuzz_waterfall_direct_equalizes_uncapped_profitability() {
    let mut rng = TestRng::new(0x0DDC_0FFE_EE11_D00Du64);
    for _ in 0..250 {
        let mut prices = [0.0_f64; 3];
        let mut preds = [0.0_f64; 3];
        for i in 0..3 {
            let p = rng.in_range(0.01, 0.16);
            prices[i] = p;
            preds[i] = (p * rng.in_range(1.05, 2.2)).min(0.95);
        }
        let mut sims = build_three_sims_with_preds(prices, preds);
        let initial_budget = rng.in_range(0.01, 15.0);
        let mut budget = initial_budget;
        let mut actions = Vec::new();

        let last_prof = waterfall(&mut sims, &mut budget, &mut actions, false);

        assert!(last_prof.is_finite());
        assert!(budget.is_finite());
        assert!(budget >= -1e-7);
        assert!(
            actions.iter().all(|a| matches!(a, Action::Buy { .. })),
            "direct-only waterfall should emit only direct buys"
        );

        let mut running = initial_budget;
        let mut bought: HashMap<&str, f64> = HashMap::new();
        for action in &actions {
            if let Action::Buy {
                market_name,
                amount,
                cost,
            } = action
            {
                assert!(amount.is_finite() && *amount > 0.0);
                assert!(cost.is_finite() && *cost >= -1e-12);
                assert!(
                    *cost <= running + 1e-8,
                    "action cost should be affordable at execution time"
                );
                running -= *cost;
                *bought.entry(market_name).or_insert(0.0) += amount;
            }
        }
        assert!(
            (running - budget).abs() <= 5e-7 * (1.0 + running.abs() + budget.abs()),
            "budget accounting drift: replay={:.12}, final={:.12}",
            running,
            budget
        );

        if actions.is_empty() {
            continue;
        }

        let tol = 2e-4 * (1.0 + last_prof.abs());
        for sim in &sims {
            let prof = profitability(sim.prediction, sim.price());
            let was_bought = bought.get(sim.market_name).copied().unwrap_or(0.0) > 1e-12;

            if !was_bought {
                assert!(
                    prof <= last_prof + tol,
                    "non-purchased market left above threshold: market={}, prof={:.9}, threshold={:.9}",
                    sim.market_name,
                    prof,
                    last_prof
                );
            } else if sim.price() < sim.buy_limit_price - 1e-8 {
                // If not capped by tick boundary, bought outcomes should land near the common KKT threshold.
                assert!(
                    (prof - last_prof).abs() <= tol,
                    "uncapped purchased market did not equalize profitability: market={}, prof={:.9}, target={:.9}",
                    sim.market_name,
                    prof,
                    last_prof
                );
            }
        }
    }
}

#[test]
fn test_fuzz_optimal_sell_split_with_inventory_matches_bruteforce() {
    let mut rng = TestRng::new(0xCAFEBABE_D15EA5E5u64);
    for _ in 0..220 {
        let prices = [
            rng.in_range(0.08, 0.18),
            rng.in_range(0.01, 0.18),
            rng.in_range(0.01, 0.18),
        ];
        let preds = [
            rng.in_range(0.03, 0.95),
            rng.in_range(0.03, 0.95),
            rng.in_range(0.03, 0.95),
        ];
        let sims = build_three_sims_with_preds(prices, preds);

        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert("M1", rng.in_range(0.0, 8.0));
        sim_balances.insert("M2", rng.in_range(0.0, 8.0));
        sim_balances.insert("M3", rng.in_range(0.0, 8.0));

        let sell_amount = rng.in_range(0.0, sim_balances.get("M1").copied().unwrap_or(0.0) + 2.5);
        if sell_amount <= 1e-9 {
            continue;
        }
        let inventory_keep_prof = rng.in_range(-0.2, 1.0);
        let merge_upper =
            merge_sell_cap_with_inventory(&sims, 0, Some(&sim_balances), inventory_keep_prof)
                .min(sell_amount);
        if merge_upper <= 1e-9 {
            continue;
        }

        let (_grid_m, grid_total) = brute_force_best_split_with_inventory(
            &sims,
            0,
            sell_amount,
            &sim_balances,
            inventory_keep_prof,
            2500,
        );
        let (opt_m, opt_total) = optimal_sell_split_with_inventory(
            &sims,
            0,
            sell_amount,
            Some(&sim_balances),
            inventory_keep_prof,
        );

        let total_tol = 1e-4 * (1.0 + grid_total.abs());
        assert!(
            (opt_total - grid_total).abs() <= total_tol,
            "inventory split solver mismatch: opt_total={:.9}, grid_total={:.9}, tol={:.9}",
            opt_total,
            grid_total,
            total_tol
        );
        assert!(opt_m >= -1e-9 && opt_m <= merge_upper + 1e-9);
    }
}

#[test]
fn test_fuzz_rebalance_end_to_end_full_l1_invariants() {
    let mut rng = TestRng::new(0xFEED_FACE_1234_4321u64);
    for _ in 0..24 {
        let (slot0_results, balances_static, susd_balance) =
            build_rebalance_fuzz_case(&mut rng, false);
        let balances: HashMap<&str, f64> = balances_static
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();

        let actions_a = rebalance(&balances, susd_balance, &slot0_results);
        let actions_b = rebalance(&balances, susd_balance, &slot0_results);

        // Rebalance should be deterministic for identical inputs.
        assert_eq!(format!("{:?}", actions_a), format!("{:?}", actions_b));

        assert_rebalance_action_invariants(&actions_a, &slot0_results, &balances, susd_balance);
    }
}

#[test]
fn test_fuzz_rebalance_end_to_end_partial_l1_invariants() {
    let mut rng = TestRng::new(0xABCD_1234_EF99_7788u64);
    for _ in 0..24 {
        let (slot0_results, balances_static, susd_balance) =
            build_rebalance_fuzz_case(&mut rng, true);
        assert!(
            slot0_results.len() < crate::predictions::PREDICTIONS_L1.len(),
            "partial fuzz case must disable mint/merge route availability"
        );

        let balances: HashMap<&str, f64> = balances_static
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();
        let actions = rebalance(&balances, susd_balance, &slot0_results);

        assert!(
            !actions.iter().any(|a| matches!(a, Action::Mint { .. })),
            "mint actions should be disabled when not all L1 pools are present"
        );
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Merge { .. })),
            "merge actions should be disabled when not all L1 pools are present"
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::FlashLoan { .. } | Action::RepayFlashLoan { .. })),
            "flash loan actions should not appear when mint/merge routes are unavailable"
        );

        assert_rebalance_action_invariants(&actions, &slot0_results, &balances, susd_balance);
    }
}

#[test]
fn test_rebalance_regression_full_l1_snapshot_invariants() {
    let markets = eligible_l1_markets_with_predictions();
    assert_eq!(
        markets.len(),
        crate::predictions::PREDICTIONS_L1.len(),
        "full regression fixture should include all tradeable L1 outcomes"
    );

    let multipliers: Vec<f64> = (0..markets.len())
        .map(|i| match i % 10 {
            0 => 0.46,
            1 => 0.58,
            2 => 0.72,
            3 => 0.87,
            4 => 0.99,
            5 => 1.08,
            6 => 1.19,
            7 => 1.31,
            8 => 0.64,
            _ => 0.53,
        })
        .collect();
    let slot0_results = build_slot0_results_for_markets(&markets, &multipliers);

    let mut balances: HashMap<&str, f64> = HashMap::new();
    for (i, market) in markets.iter().enumerate() {
        if i % 9 == 0 {
            balances.insert(market.name, 1.25 + (i % 5) as f64 * 0.9);
        } else if i % 13 == 0 {
            balances.insert(market.name, 0.65);
        }
    }
    let budget = 83.0;

    let actions_a = rebalance(&balances, budget, &slot0_results);
    let actions_b = rebalance(&balances, budget, &slot0_results);
    assert_eq!(
        format!("{:?}", actions_a),
        format!("{:?}", actions_b),
        "full-L1 regression fixture should be deterministic"
    );
    assert_rebalance_action_invariants(&actions_a, &slot0_results, &balances, budget);

    let ev_before = replay_actions_to_ev(&[], &slot0_results, &balances, budget);
    let ev_after = replay_actions_to_ev(&actions_a, &slot0_results, &balances, budget);
    let gain = ev_after - ev_before;

    let buys = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .count();
    let sells = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Sell { .. }))
        .count();
    let mints = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Mint { .. }))
        .count();
    let merges = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Merge { .. }))
        .count();
    let flash = actions_a
        .iter()
        .filter(|a| matches!(a, Action::FlashLoan { .. }))
        .count();
    let repay = actions_a
        .iter()
        .filter(|a| matches!(a, Action::RepayFlashLoan { .. }))
        .count();

    const EXPECTED_ACTIONS: usize = 26_368;
    const EXPECTED_BUYS: usize = 1_091;
    const EXPECTED_SELLS: usize = 24_119;
    const EXPECTED_MINTS: usize = 376;
    const EXPECTED_MERGES: usize = 10;
    const EXPECTED_FLASH: usize = 386;
    const EXPECTED_REPAY: usize = 386;
    const EXPECTED_EV_BEFORE: f64 = 83.329_134_223;
    const EXPECTED_EV_AFTER: f64 = 305.747_156_758;
    const EV_TOL: f64 = 3e-6;

    assert_eq!(
        actions_a.len(),
        EXPECTED_ACTIONS,
        "full-L1 regression action count changed"
    );
    assert_eq!(buys, EXPECTED_BUYS, "buy action count drifted");
    assert_eq!(sells, EXPECTED_SELLS, "sell action count drifted");
    assert_eq!(mints, EXPECTED_MINTS, "mint action count drifted");
    assert_eq!(merges, EXPECTED_MERGES, "merge action count drifted");
    assert_eq!(flash, EXPECTED_FLASH, "flash-loan action count drifted");
    assert_eq!(
        repay, EXPECTED_REPAY,
        "flash repayment action count drifted"
    );
    assert!(
        (ev_before - EXPECTED_EV_BEFORE).abs() <= EV_TOL,
        "ev_before drifted: got={:.9}, expected={:.9}, tol={:.9}",
        ev_before,
        EXPECTED_EV_BEFORE,
        EV_TOL
    );
    assert!(
        (ev_after - EXPECTED_EV_AFTER).abs() <= EV_TOL,
        "ev_after drifted: got={:.9}, expected={:.9}, tol={:.9}",
        ev_after,
        EXPECTED_EV_AFTER,
        EV_TOL
    );
    assert!(
        gain > 0.0,
        "regression fixture should improve EV: before={:.9}, after={:.9}",
        ev_before,
        ev_after
    );
}

#[test]
fn test_rebalance_regression_full_l1_snapshot_variant_b_invariants() {
    let markets = eligible_l1_markets_with_predictions();
    assert_eq!(
        markets.len(),
        crate::predictions::PREDICTIONS_L1.len(),
        "full regression fixture should include all tradeable L1 outcomes"
    );

    let multipliers: Vec<f64> = (0..markets.len())
        .map(|i| match i % 12 {
            0 => 0.92,
            1 => 0.97,
            2 => 1.02,
            3 => 1.07,
            4 => 0.88,
            5 => 1.11,
            6 => 0.95,
            7 => 1.16,
            8 => 0.90,
            9 => 1.04,
            10 => 0.99,
            _ => 1.13,
        })
        .collect();
    let slot0_results = build_slot0_results_for_markets(&markets, &multipliers);

    let mut balances: HashMap<&str, f64> = HashMap::new();
    for (i, market) in markets.iter().enumerate() {
        if i % 7 == 0 {
            balances.insert(market.name, 0.8 + (i % 6) as f64 * 0.55);
        } else if i % 11 == 0 {
            balances.insert(market.name, 0.35);
        }
    }
    let budget = 41.0;

    let actions_a = rebalance(&balances, budget, &slot0_results);
    let actions_b = rebalance(&balances, budget, &slot0_results);
    assert_eq!(
        format!("{:?}", actions_a),
        format!("{:?}", actions_b),
        "full-L1 regression fixture should be deterministic"
    );
    assert_rebalance_action_invariants(&actions_a, &slot0_results, &balances, budget);

    let ev_before = replay_actions_to_ev(&[], &slot0_results, &balances, budget);
    let ev_after = replay_actions_to_ev(&actions_a, &slot0_results, &balances, budget);
    let gain = ev_after - ev_before;

    let buys = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .count();
    let sells = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Sell { .. }))
        .count();
    let mints = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Mint { .. }))
        .count();
    let merges = actions_a
        .iter()
        .filter(|a| matches!(a, Action::Merge { .. }))
        .count();
    let flash = actions_a
        .iter()
        .filter(|a| matches!(a, Action::FlashLoan { .. }))
        .count();
    let repay = actions_a
        .iter()
        .filter(|a| matches!(a, Action::RepayFlashLoan { .. }))
        .count();

    const EXPECTED_ACTIONS: usize = 29_032;
    const EXPECTED_BUYS: usize = 924;
    const EXPECTED_SELLS: usize = 26_935;
    const EXPECTED_MINTS: usize = 382;
    const EXPECTED_MERGES: usize = 9;
    const EXPECTED_FLASH: usize = 391;
    const EXPECTED_REPAY: usize = 391;
    const EXPECTED_EV_BEFORE: f64 = 41.229_354_975;
    const EXPECTED_EV_AFTER: f64 = 139.923_206_653;
    const EV_TOL: f64 = 3e-6;

    assert_eq!(
        actions_a.len(),
        EXPECTED_ACTIONS,
        "full-L1 regression variant-B action count changed"
    );
    assert_eq!(buys, EXPECTED_BUYS, "variant-B buy action count drifted");
    assert_eq!(sells, EXPECTED_SELLS, "variant-B sell action count drifted");
    assert_eq!(mints, EXPECTED_MINTS, "variant-B mint action count drifted");
    assert_eq!(
        merges, EXPECTED_MERGES,
        "variant-B merge action count drifted"
    );
    assert_eq!(
        flash, EXPECTED_FLASH,
        "variant-B flash-loan action count drifted"
    );
    assert_eq!(
        repay, EXPECTED_REPAY,
        "variant-B flash repayment action count drifted"
    );
    assert!(
        (ev_before - EXPECTED_EV_BEFORE).abs() <= EV_TOL,
        "variant-B ev_before drifted: got={:.9}, expected={:.9}, tol={:.9}",
        ev_before,
        EXPECTED_EV_BEFORE,
        EV_TOL
    );
    assert!(
        (ev_after - EXPECTED_EV_AFTER).abs() <= EV_TOL,
        "variant-B ev_after drifted: got={:.9}, expected={:.9}, tol={:.9}",
        ev_after,
        EXPECTED_EV_AFTER,
        EV_TOL
    );
    assert!(
        gain > 0.0,
        "regression fixture should improve EV: before={:.9}, after={:.9}",
        ev_before,
        ev_after
    );
}
