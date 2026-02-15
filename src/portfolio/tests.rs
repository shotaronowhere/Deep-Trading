use std::collections::{HashMap, HashSet};

use super::Action;
use super::merge::{
    merge_sell_cap, merge_sell_cap_with_inventory, split_sell_total_proceeds,
    split_sell_total_proceeds_with_inventory,
};
use super::rebalancer::{RebalanceMode, rebalance, rebalance_with_mode};
use super::sim::{PoolSim, Route, build_sims, profitability};
use super::trading::{ExecutionState, emit_mint_actions, execute_buy};
use super::waterfall::waterfall;
use crate::markets::MarketData;
use crate::pools::{Slot0Result, normalize_market_name, prediction_to_sqrt_price_x96};
use alloy::primitives::{Address, U256};

#[path = "tests/fixtures.rs"]
mod fixtures;
use fixtures::*;

#[test]
fn test_pool_sim_price_roundtrip() {
    // Create a sim and verify the price roughly matches what we set
    let (slot0, market) = mock_slot0_market(
        "test_market",
        "0x1111111111111111111111111111111111111111",
        0.3,
    );
    let sim = PoolSim::from_slot0(&slot0, market, 0.5).unwrap();
    let price = sim.price();
    assert!((price - 0.3).abs() < 0.01, "price {} should be ~0.3", price);
}

#[test]
fn test_sell_overpriced() {
    // Outcome priced at 0.5, prediction is 0.3 → overpriced, should sell
    let (slot0, market) = mock_slot0_market(
        "overpriced",
        "0x1111111111111111111111111111111111111111",
        0.5,
    );
    let slot0_results = vec![(slot0, market)];
    let mut balances = HashMap::new();
    balances.insert("overpriced", 100.0);

    // Temporarily add a matching prediction
    // Since we can't add to PREDICTIONS_L1 (static), we test via PoolSim directly
    let mut sims = vec![PoolSim::from_slot0(&slot0_results[0].0, slot0_results[0].1, 0.3).unwrap()];

    // Simulate sell phase
    let sim = &mut sims[0];
    let price = sim.price();
    assert!(price > 0.3, "price {} should be > 0.3", price);

    let (tokens_needed, proceeds, new_price) = sim.sell_to_price(0.3).unwrap();
    assert!(tokens_needed > 0.0, "should need to sell some tokens");
    assert!(proceeds > 0.0, "should receive proceeds");

    // Check price after sell
    assert!(
        (new_price - 0.3).abs() < 0.01,
        "price after sell {} should be ~0.3",
        new_price
    );
}

#[test]
fn test_cost_to_price() {
    // Price at 0.01, want to buy until price = 0.1 (within pool range ~[0.0001, 0.2])
    let (slot0, market) =
        mock_slot0_market("cheap", "0x1111111111111111111111111111111111111111", 0.01);
    let sim = PoolSim::from_slot0(&slot0, market, 0.5).unwrap();

    let (cost, amount, new_price) = sim.cost_to_price(0.1).unwrap();
    assert!(cost > 0.0, "should cost something to move price");
    assert!(amount > 0.0, "should receive outcome tokens");

    // Price should hit the exact target (within float tolerance), unless capped.
    let expected = 0.1_f64.min(sim.buy_limit_price);
    let tol = 1e-12 * (1.0 + expected.abs());
    assert!(
        (new_price - expected).abs() <= tol,
        "price after buy should match target/clamp: got={:.12}, expected={:.12}, tol={:.12}",
        new_price,
        expected,
        tol
    );
}

#[test]
fn test_waterfall_equalizes() {
    // Two outcomes: A at price 0.05 pred 0.1, B at price 0.05 pred 0.08
    // profitability(A) = (0.1 - 0.05)/0.05 = 1.0
    // profitability(B) = (0.08 - 0.05)/0.05 = 0.6
    // Waterfall should first buy A until prof(A) = 0.6, then buy both

    let (slot0_a, market_a) =
        mock_slot0_market("A", "0x1111111111111111111111111111111111111111", 0.05);
    let (slot0_b, market_b) =
        mock_slot0_market("B", "0x2222222222222222222222222222222222222222", 0.05);

    let mut sims = vec![
        PoolSim::from_slot0(&slot0_a, market_a, 0.10).unwrap(),
        PoolSim::from_slot0(&slot0_b, market_b, 0.08).unwrap(),
    ];

    let mut budget = 1000.0;
    let mut actions = Vec::new();

    waterfall(&mut sims, &mut budget, &mut actions, false);

    // Should have buy actions
    let buys: Vec<_> = actions
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .collect();
    assert!(!buys.is_empty(), "should have buy actions");

    // A should be bought first (higher profitability)
    if let Action::Buy { market_name, .. } = &buys[0] {
        assert_eq!(*market_name, "A", "A should be bought first");
    }
}

#[test]
fn test_no_action_when_all_overpriced_no_holdings() {
    // Price > prediction but no holdings → nothing to do
    let (slot0, market) = mock_slot0_market(
        "overpriced",
        "0x1111111111111111111111111111111111111111",
        0.5,
    );
    let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.3).unwrap()];
    let mut budget = 100.0;
    let mut actions = Vec::new();

    waterfall(&mut sims, &mut budget, &mut actions, false);

    assert!(actions.is_empty(), "no actions when everything overpriced");
    assert!((budget - 100.0).abs() < 1e-6, "budget should be unchanged");
}

#[test]
fn test_budget_exhaustion() {
    let (slot0, market) =
        mock_slot0_market("cheap", "0x1111111111111111111111111111111111111111", 0.01);
    let mut sims = vec![PoolSim::from_slot0(&slot0, market, 0.5).unwrap()];

    let mut budget = 0.001; // tiny budget
    let mut actions = Vec::new();

    waterfall(&mut sims, &mut budget, &mut actions, false);

    // Should have a buy but budget should be nearly exhausted
    let buys: Vec<_> = actions
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .collect();
    if !buys.is_empty() {
        assert!(budget < 0.001, "budget should be reduced");
    }
}

#[test]
fn test_mint_route_actions() {
    // Directly test emit_mint_actions and execute_buy with Route::Mint.
    // 3 outcomes: target=M1, non-targets M2,M3 will be sold.
    let tokens = [
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
        "0x3333333333333333333333333333333333333333",
    ];
    let names = ["M1", "M2", "M3"];

    let slot0_results: Vec<_> = tokens
        .iter()
        .zip(names.iter())
        .map(|(tok, name)| mock_slot0_market(name, tok, 0.05))
        .collect();

    let mut sims: Vec<_> = slot0_results
        .iter()
        .map(|(s, m)| PoolSim::from_slot0(s, m, 0.3).unwrap())
        .collect();

    // Test emit_mint_actions directly
    let mint_amount = 10.0;
    let mut actions = Vec::new();
    let proceeds = emit_mint_actions(&mut sims, 0, mint_amount, &mut actions, &HashSet::new());

    // First action: Mint with target_market = M1
    assert!(
        matches!(
            &actions[0],
            Action::Mint {
                target_market: "M1",
                ..
            }
        ),
        "first action should be Mint targeting M1"
    );
    if let Action::Mint { amount, .. } = &actions[0] {
        assert!((*amount - 10.0).abs() < 1e-12, "mint amount should be 10.0");
    }

    // Sells for non-target outcomes
    let sells: Vec<_> = actions
        .iter()
        .filter_map(|a| match a {
            Action::Sell {
                market_name,
                amount,
                ..
            } => Some((*market_name, *amount)),
            _ => None,
        })
        .collect();
    assert_eq!(sells.len(), 2, "should sell 2 non-target outcomes");
    for (name, amt) in &sells {
        assert!(*name == "M2" || *name == "M3");
        assert!(*amt > 0.0);
    }
    assert!(proceeds > 0.0, "selling non-targets should yield proceeds");

    // Test execute_buy with Route::Mint updates budget correctly
    let mut sims2: Vec<_> = slot0_results
        .iter()
        .map(|(s, m)| PoolSim::from_slot0(s, m, 0.3).unwrap())
        .collect();
    let mut budget = 100.0;
    let mut actions2 = Vec::new();
    let mut unused_bal: HashMap<&str, f64> = HashMap::new();
    let mut exec = ExecutionState::new(&mut sims2, &mut budget, &mut actions2, &mut unused_bal);
    execute_buy(&mut exec, 0, 5.0, 10.0, Route::Mint, None, &HashSet::new());
    assert!(budget < 100.0, "budget should decrease after mint");

    // Test sim_balances tracking: Mint adds to all, Sell subtracts
    let mut sim_balances: HashMap<&str, f64> = HashMap::new();
    for action in &actions2 {
        match action {
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                *sim_balances.entry(market_name).or_insert(0.0) += amount;
            }
            Action::Mint { amount, .. } => {
                for sim in sims2.iter() {
                    *sim_balances.entry(sim.market_name).or_insert(0.0) += amount;
                }
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                *sim_balances.entry(market_name).or_insert(0.0) -= amount;
            }
            Action::Merge { amount, .. } => {
                for sim in sims2.iter() {
                    *sim_balances.entry(sim.market_name).or_insert(0.0) -= amount;
                }
            }
            Action::FlashLoan { .. } | Action::RepayFlashLoan { .. } => {}
        }
    }
    // Target M1 should hold the full mint amount
    let m1_bal = *sim_balances.get("M1").unwrap_or(&0.0);
    assert!(
        (m1_bal - 10.0).abs() < 1e-12,
        "M1 should hold 10.0 minted tokens, got {}",
        m1_bal
    );
    // Non-targets should have residual >= 0 (mint - sold)
    for name in &["M2", "M3"] {
        let bal = *sim_balances.get(name).unwrap_or(&0.0);
        assert!(
            bal >= -1e-12,
            "{} balance should be >= 0 (residual), got {}",
            name,
            bal
        );
    }
}

#[test]
fn test_profitability_handles_nonpositive_prices() {
    let p_neg = profitability(0.3, -0.2);
    let p_zero = profitability(0.3, 0.0);
    let p_small = profitability(0.3, 1e-12);
    assert!(p_neg.is_finite() && p_neg > 0.0);
    assert!(p_zero.is_finite() && p_zero > 0.0);
    assert!(
        (p_zero - p_small).abs() < 1e-9,
        "zero and epsilon clamp should match"
    );
}

#[test]
fn test_waterfall_can_activate_mint_with_negative_alt_price() {
    // Sum(prices) > 1 => alt price for each target is negative.
    // Mint route should still be considered and executed.
    let mut sims = build_three_sims_with_preds([0.8, 0.8, 0.8], [0.3, 0.3, 0.3]);
    let mut budget = 1.0;
    let mut actions = Vec::new();

    let last_prof = waterfall(&mut sims, &mut budget, &mut actions, true);

    assert!(last_prof.is_finite() && last_prof >= 0.0);
    assert!(
        actions.iter().any(|a| matches!(a, Action::Mint { .. })),
        "negative alt-price setup should trigger mint route"
    );
}

#[test]
fn test_complete_set_arb_executes_when_profitable() {
    // Sum prices = 0.6 < 1.0 (before fees/slippage), so complete-set buy+merge should be profitable.
    let mut sims = build_three_sims_with_preds([0.2, 0.2, 0.2], [0.3, 0.3, 0.3]);
    let mut actions = Vec::new();
    let mut budget = 0.0;

    let profit = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_complete_set_arb()
    };
    assert!(profit > 0.0, "arb should produce positive profit");
    assert!(
        (budget - profit).abs() < 1e-9,
        "budget increase should match realized arb profit"
    );

    let buy_count = actions
        .iter()
        .filter(|a| matches!(a, Action::Buy { .. }))
        .count();
    assert_eq!(
        buy_count, 3,
        "complete-set arb should buy every pooled outcome"
    );
    assert!(
        actions.iter().any(|a| matches!(
            a,
            Action::Merge {
                source_market: "complete_set_arb",
                ..
            }
        )),
        "arb should emit merge action tagged as complete_set_arb"
    );
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::FlashLoan { .. })),
        "arb legs should be funded with a flash loan"
    );
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::RepayFlashLoan { .. })),
        "arb legs should repay the flash loan"
    );
}

#[test]
fn test_complete_set_arb_skips_when_unprofitable() {
    // Sum prices = 1.5 > 1.0, so buy-all-and-merge should be non-profitable.
    let mut sims = build_three_sims_with_preds([0.5, 0.5, 0.5], [0.3, 0.3, 0.3]);
    let mut actions = Vec::new();
    let mut budget = 7.0;

    let profit = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_complete_set_arb()
    };
    assert!(profit <= 1e-12, "unprofitable setup should not execute arb");
    assert!(
        actions.is_empty(),
        "no actions should be emitted when arb is skipped"
    );
    assert!(
        (budget - 7.0).abs() < 1e-12,
        "budget should remain unchanged when no arb trade is executed"
    );
}

#[test]
fn test_complete_set_mint_sell_arb_executes_when_profitable() {
    // Sum prices = 1.5 > 1.0, so mint-all-and-sell should be profitable.
    let mut sims = build_three_sims_with_preds([0.5, 0.5, 0.5], [0.3, 0.3, 0.3]);
    let mut actions = Vec::new();
    let mut budget = 0.0;

    let profit = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_complete_set_mint_sell_arb()
    };
    assert!(profit > 0.0, "mint-sell arb should produce positive profit");
    assert!(
        (budget - profit).abs() < 1e-9,
        "budget increase should match realized mint-sell arb profit"
    );

    assert!(
        matches!(actions.first(), Some(Action::FlashLoan { .. })),
        "first action should be flash-loan borrow"
    );
    assert!(
        matches!(
            actions.get(1),
            Some(Action::Mint {
                target_market: "complete_set_arb",
                ..
            })
        ),
        "second action should be mint tagged as complete_set_arb"
    );
    assert!(
        matches!(actions.last(), Some(Action::RepayFlashLoan { .. })),
        "last action should be flash-loan repay"
    );
    let sell_count = actions
        .iter()
        .filter(|a| matches!(a, Action::Sell { .. }))
        .count();
    assert_eq!(
        sell_count, 3,
        "mint-sell arb should sell every pooled outcome"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "mint-sell arb should not emit buy actions"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "mint-sell arb should not emit merge actions"
    );
}

#[test]
fn test_complete_set_mint_sell_arb_skips_when_unprofitable() {
    // Sum prices = 0.6 < 1.0, so mint-all-and-sell should be non-profitable.
    let mut sims = build_three_sims_with_preds([0.2, 0.2, 0.2], [0.3, 0.3, 0.3]);
    let mut actions = Vec::new();
    let mut budget = 5.0;

    let profit = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_complete_set_mint_sell_arb()
    };
    assert!(
        profit <= 1e-12,
        "unprofitable setup should not execute mint-sell arb"
    );
    assert!(
        actions.is_empty(),
        "no actions should be emitted when mint-sell arb is skipped"
    );
    assert!(
        (budget - 5.0).abs() < 1e-12,
        "budget should remain unchanged when no mint-sell trade is executed"
    );
}

#[test]
fn test_two_sided_complete_set_arb_selects_buy_merge_when_sum_below_one() {
    let mut sims = build_three_sims_with_preds([0.2, 0.2, 0.2], [0.3, 0.3, 0.3]);
    let mut actions = Vec::new();
    let mut budget = 0.0;

    let profit = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_two_sided_complete_set_arb()
    };
    assert!(
        profit > 0.0,
        "two-sided arb should execute buy-merge branch"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "buy-merge branch should emit buy legs"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "buy-merge branch should emit merge leg"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Mint { .. })),
        "buy-merge branch should not emit mint leg"
    );
}

#[test]
fn test_two_sided_complete_set_arb_selects_mint_sell_when_sum_above_one() {
    let mut sims = build_three_sims_with_preds([0.5, 0.5, 0.5], [0.3, 0.3, 0.3]);
    let mut actions = Vec::new();
    let mut budget = 0.0;

    let profit = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_two_sided_complete_set_arb()
    };
    assert!(
        profit > 0.0,
        "two-sided arb should execute mint-sell branch"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Mint { .. })),
        "mint-sell branch should emit mint leg"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Sell { .. })),
        "mint-sell branch should emit sell legs"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "mint-sell branch should not emit buy legs"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "mint-sell branch should not emit merge leg"
    );
}

#[test]
fn test_two_sided_complete_set_arb_skips_near_price_parity() {
    let mut sims = build_three_sims_with_preds([0.2, 0.3, 0.5], [0.3, 0.3, 0.3]);
    let mut actions = Vec::new();
    let mut budget = 4.0;

    let profit = {
        let mut unused_bal: HashMap<&str, f64> = HashMap::new();
        let mut exec = ExecutionState::new(&mut sims, &mut budget, &mut actions, &mut unused_bal);
        exec.execute_two_sided_complete_set_arb()
    };
    assert!(
        profit <= 1e-12,
        "no arb branch should execute near exact parity"
    );
    assert!(
        actions.is_empty(),
        "no actions should be emitted near exact parity"
    );
    assert!(
        (budget - 4.0).abs() < 1e-12,
        "budget should remain unchanged when no branch is taken"
    );
}

#[test]
fn test_rebalance_with_mode_full_matches_rebalance_default() {
    let markets = eligible_l1_markets_with_predictions();
    let selected: Vec<_> = markets.into_iter().take(8).collect();
    let multipliers = [1.35, 0.55, 1.40, 0.60, 1.25, 0.70, 1.15, 0.75];
    let slot0_results = build_slot0_results_for_markets(&selected, &multipliers);

    let mut balances = HashMap::new();
    balances.insert(selected[0].name, 1.0);
    balances.insert(selected[1].name, 2.0);
    let susd = 100.0;

    let default_actions = rebalance(&balances, susd, &slot0_results);
    let mode_actions = rebalance_with_mode(&balances, susd, &slot0_results, RebalanceMode::Full);

    assert_eq!(
        format!("{default_actions:?}"),
        format!("{mode_actions:?}"),
        "full mode should match default rebalance behavior"
    );
}

#[test]
fn test_rebalance_with_mode_arb_only_mint_sell_shape_only() {
    let markets: Vec<_> = crate::markets::MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .collect();
    let slot0_results: Vec<_> = markets
        .iter()
        .map(|market| {
            let pool = market.pool.as_ref().expect("pooled market required");
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let sqrt_price = prediction_to_sqrt_price_x96(0.02, is_token1_outcome)
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
                *market,
            )
        })
        .collect();

    let actions = rebalance_with_mode(&HashMap::new(), 0.0, &slot0_results, RebalanceMode::ArbOnly);
    assert!(!actions.is_empty(), "arb-only mode should emit arb actions");
    assert!(
        actions.iter().any(|a| matches!(a, Action::Mint { .. })),
        "arb-only mode should include mint for sum(prices)>1 case"
    );
    assert!(
        actions.iter().any(|a| matches!(a, Action::Sell { .. })),
        "arb-only mode should include sell legs in sum(prices)>1 case"
    );
    assert!(
        matches!(actions.first(), Some(Action::FlashLoan { .. }))
            && matches!(actions.last(), Some(Action::RepayFlashLoan { .. })),
        "arb-only action stream should be a single flash bracket"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Buy { .. })),
        "arb-only mint-sell path should not emit direct buy actions"
    );
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Merge { .. })),
        "arb-only mint-sell path should not emit merge actions"
    );
}

#[test]
fn test_rebalance_with_mode_arb_only_fails_closed_on_partial_snapshot() {
    let markets: Vec<_> = crate::markets::MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .collect();
    assert!(
        markets.len() > 1,
        "expected at least two pooled markets for partial snapshot test"
    );

    let slot0_results: Vec<_> = markets
        .iter()
        .take(markets.len() - 1)
        .map(|market| {
            let pool = market.pool.as_ref().expect("pooled market required");
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let sqrt_price = prediction_to_sqrt_price_x96(0.02, is_token1_outcome)
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
                *market,
            )
        })
        .collect();

    let actions = rebalance_with_mode(&HashMap::new(), 0.0, &slot0_results, RebalanceMode::ArbOnly);
    assert!(
        actions.is_empty(),
        "arb-only mode should fail closed on incomplete market snapshots"
    );
}

#[test]
fn test_rebalance_with_mode_arb_only_fails_closed_on_duplicate_membership() {
    let markets: Vec<_> = crate::markets::MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .collect();
    assert!(
        markets.len() > 1,
        "expected at least two pooled markets for duplicate-membership test"
    );

    let mut slot0_results: Vec<_> = markets
        .iter()
        .map(|market| {
            let pool = market.pool.as_ref().expect("pooled market required");
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let sqrt_price = prediction_to_sqrt_price_x96(0.02, is_token1_outcome)
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
                *market,
            )
        })
        .collect();

    // Keep vector length unchanged, but replace one member with a duplicate of another.
    let last = slot0_results.len() - 1;
    slot0_results[last] = slot0_results[0].clone();

    let actions = rebalance_with_mode(&HashMap::new(), 0.0, &slot0_results, RebalanceMode::ArbOnly);
    assert!(
        actions.is_empty(),
        "arb-only mode should fail closed when outcome membership is malformed"
    );
}

fn brute_force_best_split(
    sims: &[PoolSim],
    source_idx: usize,
    sell_amount: f64,
    steps: usize,
) -> (f64, f64) {
    let upper = merge_sell_cap(sims, source_idx).min(sell_amount);
    let mut best_m = 0.0_f64;
    let mut best_total = f64::NEG_INFINITY;
    for i in 0..=steps {
        let m = upper * (i as f64) / (steps as f64);
        let (total, _) = split_sell_total_proceeds(sims, source_idx, sell_amount, m);
        if total > best_total {
            best_total = total;
            best_m = m;
        }
    }
    (best_m, best_total)
}

fn brute_force_best_split_with_inventory(
    sims: &[PoolSim],
    source_idx: usize,
    sell_amount: f64,
    sim_balances: &HashMap<&str, f64>,
    inventory_keep_prof: f64,
    steps: usize,
) -> (f64, f64) {
    let upper =
        merge_sell_cap_with_inventory(sims, source_idx, Some(sim_balances), inventory_keep_prof)
            .min(sell_amount);
    let mut best_m = 0.0_f64;
    let mut best_total = f64::NEG_INFINITY;
    for i in 0..=steps {
        let m = upper * (i as f64) / (steps as f64);
        let (total, _) = split_sell_total_proceeds_with_inventory(
            sims,
            source_idx,
            sell_amount,
            m,
            Some(sim_balances),
            inventory_keep_prof,
        );
        if total > best_total {
            best_total = total;
            best_m = m;
        }
    }
    (best_m, best_total)
}

fn build_rebalance_fuzz_case(
    rng: &mut TestRng,
    force_partial: bool,
) -> (
    Vec<(Slot0Result, &'static crate::markets::MarketData)>,
    HashMap<&'static str, f64>,
    f64,
) {
    use crate::markets::MARKETS_L1;

    let preds = crate::pools::prediction_map();
    let mut candidates: Vec<&'static crate::markets::MarketData> = MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .filter(|m| {
            let key = normalize_market_name(m.name);
            preds.contains_key(&key)
        })
        .collect();
    assert!(
        !candidates.is_empty(),
        "fuzz scenario requires at least one eligible L1 market"
    );

    for i in (1..candidates.len()).rev() {
        let j = rng.pick(i + 1);
        candidates.swap(i, j);
    }

    let total = candidates.len();
    let selected_len = if force_partial && total > 1 {
        1 + rng.pick(total - 1)
    } else {
        total
    };
    candidates.truncate(selected_len);

    let mut balances: HashMap<&'static str, f64> = HashMap::new();
    let mut slot0_results: Vec<(Slot0Result, &'static crate::markets::MarketData)> =
        Vec::with_capacity(candidates.len());

    for market in candidates {
        let pool = market.pool.as_ref().expect("eligible pool must exist");
        let is_token1_outcome = pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
        let key = normalize_market_name(market.name);
        let pred = preds
            .get(&key)
            .copied()
            .expect("eligible market must have prediction");
        let multiplier = rng.in_range(0.35, 1.8);
        let price = (pred * multiplier).clamp(1e-6, 0.95);
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
        slot0_results.push((slot0, market));

        if rng.chance(3, 5) {
            balances.insert(market.name, rng.in_range(0.0, 10.0));
        }
    }

    let susd_balance = rng.in_range(0.0, 250.0);
    (slot0_results, balances, susd_balance)
}

fn assert_rebalance_action_invariants(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
) {
    let market_names: Vec<&str> = slot0_results.iter().map(|(_, m)| m.name).collect();
    let market_set: HashSet<&str> = market_names.iter().copied().collect();
    let mut holdings: HashMap<&str, f64> = HashMap::new();
    for name in &market_names {
        holdings.insert(
            *name,
            initial_balances.get(name).copied().unwrap_or(0.0).max(0.0),
        );
    }

    let mut cash = initial_susd;
    let mut flash_outstanding = 0.0_f64;

    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                cost,
            } => {
                assert!(
                    market_set.contains(market_name),
                    "unknown buy market {}",
                    market_name
                );
                assert!(amount.is_finite() && *amount >= 0.0);
                assert!(cost.is_finite() && *cost >= -1e-12);
                cash -= *cost;
                *holdings.entry(*market_name).or_insert(0.0) += *amount;
            }
            Action::Sell {
                market_name,
                amount,
                proceeds,
            } => {
                assert!(
                    market_set.contains(market_name),
                    "unknown sell market {}",
                    market_name
                );
                assert!(amount.is_finite() && *amount >= 0.0);
                assert!(proceeds.is_finite() && *proceeds >= -1e-12);
                let bal = holdings.entry(*market_name).or_insert(0.0);
                *bal -= *amount;
                assert!(
                    *bal >= -1e-6,
                    "sell over-consumed holdings for {}: {}",
                    market_name,
                    *bal
                );
                cash += *proceeds;
            }
            Action::Mint { amount, .. } => {
                assert!(amount.is_finite() && *amount >= 0.0);
                for name in &market_names {
                    *holdings.entry(*name).or_insert(0.0) += *amount;
                }
            }
            Action::Merge { amount, .. } => {
                assert!(amount.is_finite() && *amount >= 0.0);
                for name in &market_names {
                    let bal = holdings.entry(*name).or_insert(0.0);
                    *bal -= *amount;
                    assert!(
                        *bal >= -1e-6,
                        "merge over-consumed holdings for {}: {}",
                        name,
                        *bal
                    );
                }
                cash += *amount;
            }
            Action::FlashLoan { amount } => {
                assert!(amount.is_finite() && *amount >= 0.0);
                flash_outstanding += *amount;
                cash += *amount;
            }
            Action::RepayFlashLoan { amount } => {
                assert!(amount.is_finite() && *amount >= 0.0);
                flash_outstanding -= *amount;
                assert!(
                    flash_outstanding >= -1e-6,
                    "repaid more flash loan than borrowed: {}",
                    flash_outstanding
                );
                cash -= *amount;
            }
        }
        assert!(
            cash.is_finite(),
            "cash became non-finite while replaying action stream"
        );
    }

    assert!(
        flash_outstanding.abs() <= 1e-6,
        "flash loan should net to zero, got {}",
        flash_outstanding
    );
    assert!(cash >= -1e-6, "final cash should not be negative: {}", cash);
    for (name, bal) in holdings {
        assert!(
            bal >= -1e-6,
            "negative final holdings for {}: {}",
            name,
            bal
        );
    }
}

fn eligible_l1_markets_with_predictions() -> Vec<&'static crate::markets::MarketData> {
    use crate::markets::MARKETS_L1;
    let preds = crate::pools::prediction_map();
    MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .filter(|m| {
            let key = normalize_market_name(m.name);
            preds.contains_key(&key)
        })
        .collect()
}

fn build_slot0_results_for_markets(
    markets: &[&'static crate::markets::MarketData],
    price_multipliers: &[f64],
) -> Vec<(Slot0Result, &'static crate::markets::MarketData)> {
    assert_eq!(markets.len(), price_multipliers.len());
    let preds = crate::pools::prediction_map();
    markets
        .iter()
        .zip(price_multipliers.iter())
        .map(|(market, mult)| {
            let pool = market.pool.as_ref().expect("market must have pool");
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
            let key = normalize_market_name(market.name);
            let pred = preds
                .get(&key)
                .copied()
                .expect("market must have prediction");
            let price = (pred * *mult).clamp(1e-6, 0.95);
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
                *market,
            )
        })
        .collect()
}

fn slot0_for_market_with_multiplier_and_pool_liquidity(
    market: &'static crate::markets::MarketData,
    price_multiplier: f64,
    liquidity: u128,
) -> (Slot0Result, &'static crate::markets::MarketData) {
    let pool = market.pool.as_ref().expect("market must have pool");
    let mut custom_pool = *pool;
    let liq_str = Box::leak(liquidity.to_string().into_boxed_str());
    custom_pool.liquidity = liq_str;
    let leaked_pool = leak_pool(custom_pool);
    let leaked_market = leak_market(MarketData {
        name: market.name,
        market_id: market.market_id,
        outcome_token: market.outcome_token,
        pool: Some(*leaked_pool),
        quote_token: market.quote_token,
    });

    let preds = crate::pools::prediction_map();
    let key = normalize_market_name(market.name);
    let pred = preds
        .get(&key)
        .copied()
        .expect("market must have prediction");
    let price = (pred * price_multiplier).clamp(1e-6, 0.95);
    let is_token1_outcome =
        leaked_pool.token1.to_lowercase() == leaked_market.outcome_token.to_lowercase();
    let sqrt_price =
        prediction_to_sqrt_price_x96(price, is_token1_outcome).unwrap_or(U256::from(1u128 << 96));

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
        leaked_market,
    )
}

fn replay_actions_to_ev(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
) -> f64 {
    let (holdings, cash) =
        replay_actions_to_state(actions, slot0_results, initial_balances, initial_susd);
    ev_from_state(&holdings, cash)
}

fn replay_actions_to_state(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
) -> (HashMap<&'static str, f64>, f64) {
    let mut holdings: HashMap<&'static str, f64> = HashMap::new();
    for (_, market) in slot0_results {
        holdings.insert(
            market.name,
            initial_balances
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0),
        );
    }
    let mut cash = initial_susd;
    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                cost,
            } => {
                *holdings.entry(*market_name).or_insert(0.0) += *amount;
                cash -= *cost;
            }
            Action::Sell {
                market_name,
                amount,
                proceeds,
            } => {
                *holdings.entry(*market_name).or_insert(0.0) -= *amount;
                cash += *proceeds;
            }
            Action::Mint { amount, .. } => {
                for (_, market) in slot0_results {
                    *holdings.entry(market.name).or_insert(0.0) += *amount;
                }
            }
            Action::Merge { amount, .. } => {
                for (_, market) in slot0_results {
                    *holdings.entry(market.name).or_insert(0.0) -= *amount;
                }
                cash += *amount;
            }
            Action::FlashLoan { amount } => cash += *amount,
            Action::RepayFlashLoan { amount } => cash -= *amount,
        }
    }
    (holdings, cash)
}

fn replay_actions_to_market_state(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Vec<(Slot0Result, &'static MarketData)> {
    let preds = crate::pools::prediction_map();
    let mut sims =
        build_sims(slot0_results, &preds).expect("test fixtures should have prediction coverage");
    let mut idx_by_market: HashMap<&str, usize> = HashMap::new();
    for (i, sim) in sims.iter().enumerate() {
        idx_by_market.insert(sim.market_name, i);
    }

    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name) {
                    if let Some((bought, _, new_price)) = sims[idx].buy_exact(*amount) {
                        if bought > 0.0 {
                            sims[idx].set_price(new_price);
                        }
                    }
                }
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name) {
                    if let Some((sold, _, new_price)) = sims[idx].sell_exact(*amount) {
                        if sold > 0.0 {
                            sims[idx].set_price(new_price);
                        }
                    }
                }
            }
            Action::Mint { .. }
            | Action::Merge { .. }
            | Action::FlashLoan { .. }
            | Action::RepayFlashLoan { .. } => {}
        }
    }

    slot0_results
        .iter()
        .map(|(slot0, market)| {
            let mut next = slot0.clone();
            if let Some(&idx) = idx_by_market.get(market.name) {
                if let Some(pool) = market.pool.as_ref() {
                    let is_token1_outcome =
                        pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
                    let p = sims[idx].price().max(1e-12);
                    next.sqrt_price_x96 = prediction_to_sqrt_price_x96(p, is_token1_outcome)
                        .unwrap_or(slot0.sqrt_price_x96);
                }
            }
            (next, *market)
        })
        .collect()
}

fn sorted_market_prices(
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Option<Vec<(&'static str, f64)>> {
    let preds = crate::pools::prediction_map();
    let sims = build_sims(slot0_results, &preds).ok()?;
    let mut prices: Vec<(&'static str, f64)> = sims
        .iter()
        .map(|sim| (sim.market_name, sim.price()))
        .collect();
    prices.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    Some(prices)
}

fn print_market_price_changes(
    label: &str,
    prices_before: &[(&'static str, f64)],
    prices_after: &[(&'static str, f64)],
) {
    let after_by_market: HashMap<&'static str, f64> = prices_after.iter().copied().collect();
    const RED: &str = "\x1b[31m";
    const GREEN: &str = "\x1b[32m";
    const GRAY: &str = "\x1b[90m";
    const RESET: &str = "\x1b[0m";

    println!("[rebalance][{}] market price changes:", label);
    for &(market_name, before) in prices_before {
        let Some(after) = after_by_market.get(market_name).copied() else {
            println!(
                "  {}{}: {:.9} -> (missing){}",
                GRAY, market_name, before, RESET
            );
            continue;
        };
        let delta = after - before;
        let pct_change = if before.abs() <= 1e-12 {
            0.0
        } else {
            100.0 * delta / before
        };
        let (color, direction) = if delta > 1e-12 {
            (GREEN, "up")
        } else if delta < -1e-12 {
            (RED, "down")
        } else {
            (GRAY, "flat")
        };
        println!(
            "  {}{}: {:.9} -> {:.9} ({:+.4}%, {}){}",
            color, market_name, before, after, pct_change, direction, RESET
        );
    }
}

fn split_actions_by_complete_set_arb_phase(actions: &[Action]) -> (&[Action], &[Action]) {
    const COMPLETE_SET_ARB: &str = "complete_set_arb";
    let marker_idx = actions.iter().position(|action| match action {
        Action::Merge { source_market, .. } => *source_market == COMPLETE_SET_ARB,
        Action::Mint { target_market, .. } => *target_market == COMPLETE_SET_ARB,
        _ => false,
    });

    let Some(marker_idx) = marker_idx else {
        return (&actions[0..0], actions);
    };

    let mut arb_end = marker_idx;
    if matches!(actions.first(), Some(Action::FlashLoan { .. }))
        && let Some(repay_offset) = actions[marker_idx + 1..]
            .iter()
            .position(|action| matches!(action, Action::RepayFlashLoan { .. }))
    {
        arb_end = marker_idx + 1 + repay_offset;
    }

    (&actions[..=arb_end], &actions[arb_end + 1..])
}

fn print_rebalance_execution_summary(
    label: &str,
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
) {
    let Some(prices_before) = sorted_market_prices(slot0_results) else {
        print_trade_summary(label, actions);
        println!(
            "[rebalance][{}] market price deltas unavailable (snapshot could not be mapped to sims)",
            label
        );
        return;
    };
    let price_sum_before: f64 = prices_before.iter().map(|(_, p)| *p).sum();
    let (arb_actions, non_arb_actions) = split_actions_by_complete_set_arb_phase(actions);

    if !arb_actions.is_empty() {
        let arb_label = format!("{}_arb_phase", label);
        print_trade_summary(&arb_label, arb_actions);
        let slot0_after_arb = replay_actions_to_market_state(arb_actions, slot0_results);
        if let Some(prices_after_arb) = sorted_market_prices(&slot0_after_arb) {
            let price_sum_after_arb: f64 = prices_after_arb.iter().map(|(_, p)| *p).sum();
            print_market_price_changes(&arb_label, &prices_before, &prices_after_arb);
            println!(
                "[rebalance][{}] market price sum before={:.9}",
                arb_label, price_sum_before
            );
            println!(
                "[rebalance][{}] market price sum after={:.9}",
                arb_label, price_sum_after_arb
            );

            let non_arb_label = format!("{}_non_arb_phase", label);
            print_trade_summary(&non_arb_label, non_arb_actions);
            let slot0_after_non_arb =
                replay_actions_to_market_state(non_arb_actions, &slot0_after_arb);
            if let Some(prices_after_non_arb) = sorted_market_prices(&slot0_after_non_arb) {
                let price_sum_after_non_arb: f64 =
                    prices_after_non_arb.iter().map(|(_, p)| *p).sum();
                print_market_price_changes(
                    &non_arb_label,
                    &prices_after_arb,
                    &prices_after_non_arb,
                );
                println!(
                    "[rebalance][{}] market price sum before={:.9}",
                    non_arb_label, price_sum_after_arb
                );
                println!(
                    "[rebalance][{}] market price sum after={:.9}",
                    non_arb_label, price_sum_after_non_arb
                );

                let total_label = format!("{}_total", label);
                print_trade_summary(&total_label, actions);
                println!(
                    "[rebalance][{}] market price sum before={:.9}",
                    total_label, price_sum_before
                );
                println!(
                    "[rebalance][{}] market price sum after={:.9}",
                    total_label, price_sum_after_non_arb
                );
                return;
            }
        }
    }

    print_trade_summary(label, actions);
    let slot0_after = replay_actions_to_market_state(actions, slot0_results);
    if let Some(prices_after) = sorted_market_prices(&slot0_after) {
        let price_sum_after: f64 = prices_after.iter().map(|(_, p)| *p).sum();
        print_market_price_changes(label, &prices_before, &prices_after);
        println!(
            "[rebalance][{}] market price sum before={:.9}",
            label, price_sum_before
        );
        println!(
            "[rebalance][{}] market price sum after={:.9}",
            label, price_sum_after
        );
    } else {
        println!(
            "[rebalance][{}] market price deltas unavailable after replay",
            label
        );
    }
}

fn print_portfolio_snapshot(
    label: &str,
    stage: &str,
    holdings: &HashMap<&'static str, f64>,
    cash: f64,
) {
    let mut non_zero_positions: Vec<(&'static str, f64)> = holdings
        .iter()
        .map(|(name, units)| (*name, *units))
        .filter(|(_, units)| units.abs() > 1e-12)
        .collect();
    non_zero_positions.sort_by(|(lhs_name, _), (rhs_name, _)| lhs_name.cmp(rhs_name));

    println!(
        "[rebalance][{}] {} portfolio: cash={:.9}, non_zero_positions={}/{}",
        label,
        stage,
        cash,
        non_zero_positions.len(),
        holdings.len()
    );
    if non_zero_positions.is_empty() {
        println!("  (no non-zero holdings)");
        return;
    }

    for (name, units) in non_zero_positions {
        println!("  {}: {:.9}", name, units);
    }
}

fn print_trade_summary(label: &str, actions: &[Action]) {
    let mut buy_count = 0usize;
    let mut buy_units = 0.0_f64;
    let mut buy_cost = 0.0_f64;
    let mut sell_count = 0usize;
    let mut sell_units = 0.0_f64;
    let mut sell_proceeds = 0.0_f64;
    let mut mint_count = 0usize;
    let mut mint_amount = 0.0_f64;
    let mut merge_count = 0usize;
    let mut merge_amount = 0.0_f64;
    let mut flash_count = 0usize;
    let mut flash_amount = 0.0_f64;
    let mut repay_count = 0usize;
    let mut repay_amount = 0.0_f64;

    for action in actions {
        match action {
            Action::Buy { amount, cost, .. } => {
                buy_count += 1;
                buy_units += *amount;
                buy_cost += *cost;
            }
            Action::Sell {
                amount, proceeds, ..
            } => {
                sell_count += 1;
                sell_units += *amount;
                sell_proceeds += *proceeds;
            }
            Action::Mint { amount, .. } => {
                mint_count += 1;
                mint_amount += *amount;
            }
            Action::Merge { amount, .. } => {
                merge_count += 1;
                merge_amount += *amount;
            }
            Action::FlashLoan { amount } => {
                flash_count += 1;
                flash_amount += *amount;
            }
            Action::RepayFlashLoan { amount } => {
                repay_count += 1;
                repay_amount += *amount;
            }
        }
    }

    println!("[rebalance][{}] trade summary:", label);
    println!("  actions: {}", actions.len());
    println!(
        "  buy: count={}, units={:.9}, cost={:.9}",
        buy_count, buy_units, buy_cost
    );
    println!(
        "  sell: count={}, units={:.9}, proceeds={:.9}",
        sell_count, sell_units, sell_proceeds
    );
    println!("  mint: count={}, amount={:.9}", mint_count, mint_amount);
    println!("  merge: count={}, amount={:.9}", merge_count, merge_amount);
    println!(
        "  flash_loan: count={}, amount={:.9}",
        flash_count, flash_amount
    );
    println!(
        "  repay_flash_loan: count={}, amount={:.9}",
        repay_count, repay_amount
    );
}

fn ev_from_state(holdings: &HashMap<&'static str, f64>, cash: f64) -> f64 {
    let preds = crate::pools::prediction_map();
    let ev_holdings: f64 = holdings
        .iter()
        .map(|(name, &units)| {
            let key = normalize_market_name(name);
            let pred = preds.get(&key).copied().unwrap_or(0.0);
            pred * units
        })
        .sum();
    cash + ev_holdings
}

struct EvTrace {
    initial_holdings: HashMap<&'static str, f64>,
    final_holdings: HashMap<&'static str, f64>,
    ev_before: f64,
    ev_after: f64,
    final_cash: f64,
}

fn compute_ev_trace(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
) -> EvTrace {
    let mut initial_holdings: HashMap<&'static str, f64> = HashMap::new();
    for (_, market) in slot0_results {
        initial_holdings.insert(
            market.name,
            initial_balances
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0),
        );
    }
    let (final_holdings, final_cash) =
        replay_actions_to_state(actions, slot0_results, initial_balances, initial_susd);
    let ev_before = ev_from_state(&initial_holdings, initial_susd);
    let ev_after = ev_from_state(&final_holdings, final_cash);
    EvTrace {
        initial_holdings,
        final_holdings,
        ev_before,
        ev_after,
        final_cash,
    }
}

fn assert_strict_ev_gain_with_portfolio_trace(
    label: &str,
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
) -> (f64, f64, f64) {
    let trace = compute_ev_trace(actions, slot0_results, initial_balances, initial_susd);

    print_portfolio_snapshot(label, "initial", &trace.initial_holdings, initial_susd);
    print_rebalance_execution_summary(label, actions, slot0_results);
    print_portfolio_snapshot(label, "final", &trace.final_holdings, trace.final_cash);

    let ev_gain = trace.ev_after - trace.ev_before;
    println!(
        "[rebalance][{}] expected value: before={:.9}, after={:.9}, gain={:.9}",
        label, trace.ev_before, trace.ev_after, ev_gain
    );

    assert!(
        trace.ev_after > trace.ev_before,
        "expected value must strictly increase: before={:.9}, after={:.9}",
        trace.ev_before,
        trace.ev_after
    );

    (trace.ev_before, trace.ev_after, trace.final_cash)
}

fn brute_force_best_gain_mint_direct(
    sims: &[PoolSim],
    mint_idx: usize,
    direct_idx: usize,
    budget: f64,
    skip: &HashSet<usize>,
    steps: usize,
) -> f64 {
    let steps = steps.max(1);
    let mint_cap = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mint_idx && !skip.contains(i))
        .map(|(_, s)| s.max_sell_tokens())
        .fold(f64::INFINITY, f64::min);
    if !mint_cap.is_finite() {
        return 0.0;
    }
    let direct_cap = sims[direct_idx].max_buy_tokens().max(0.0);
    let mut best = 0.0_f64;

    for im in 0..=steps {
        let mint_amount = mint_cap * (im as f64) / (steps as f64);
        let mut state = sims.to_vec();
        let mut holdings = vec![0.0_f64; state.len()];
        let mut spent = 0.0_f64;

        if mint_amount > 0.0 {
            for h in &mut holdings {
                *h += mint_amount;
            }
            let mut proceeds = 0.0_f64;
            for i in 0..state.len() {
                if i == mint_idx || skip.contains(&i) {
                    continue;
                }
                if let Some((sold, leg_proceeds, new_p)) = state[i].sell_exact(mint_amount) {
                    if sold > 0.0 {
                        holdings[i] -= sold;
                        proceeds += leg_proceeds;
                        state[i].set_price(new_p);
                    }
                }
            }
            spent += mint_amount - proceeds;
        }

        if spent > budget + 1e-12 {
            continue;
        }

        for id in 0..=steps {
            let req_direct = direct_cap * (id as f64) / (steps as f64);
            let mut state_d = state.clone();
            let mut holdings_d = holdings.clone();
            let mut spent_d = spent;
            if req_direct > 0.0 {
                if let Some((bought, cost, new_p)) = state_d[direct_idx].buy_exact(req_direct) {
                    if bought > 0.0 {
                        holdings_d[direct_idx] += bought;
                        spent_d += cost;
                        state_d[direct_idx].set_price(new_p);
                    }
                }
            }
            if spent_d <= budget + 1e-12 {
                let ev_gain: f64 = holdings_d
                    .iter()
                    .enumerate()
                    .map(|(i, h)| state_d[i].prediction * *h)
                    .sum::<f64>()
                    - spent_d;
                if ev_gain > best {
                    best = ev_gain;
                }
            }
        }
    }

    best
}

fn oracle_direct_only_best_ev_grid(sims: &[PoolSim], budget: f64, steps: usize) -> f64 {
    let steps = steps.max(1);
    let leg_points: Vec<Vec<(f64, f64)>> = sims
        .iter()
        .map(|sim| {
            let max_buy = sim.max_buy_tokens().max(0.0);
            (0..=steps)
                .filter_map(|i| {
                    let t = (i as f64) / (steps as f64);
                    let req = max_buy * t * t;
                    sim.buy_exact(req).map(|(bought, cost, _)| {
                        let gain = sim.prediction * bought - cost;
                        (cost, gain)
                    })
                })
                .collect()
        })
        .collect();

    match leg_points.len() {
        0 => budget,
        1 => leg_points[0]
            .iter()
            .filter(|(cost, _)| *cost <= budget + 1e-12)
            .map(|(_, gain)| budget + *gain)
            .fold(budget, f64::max),
        2 => {
            let mut best = budget;
            for (c0, g0) in &leg_points[0] {
                if *c0 > budget + 1e-12 {
                    continue;
                }
                for (c1, g1) in &leg_points[1] {
                    let total_cost = *c0 + *c1;
                    if total_cost <= budget + 1e-12 {
                        let ev = budget + *g0 + *g1;
                        if ev > best {
                            best = ev;
                        }
                    }
                }
            }
            best
        }
        _ => panic!("oracle_direct_only_best_ev_grid currently supports up to 2 pools"),
    }
}

fn oracle_two_pool_direct_only_best_ev_with_holdings_grid(
    sims: &[PoolSim],
    initial_holdings: &[f64],
    budget: f64,
    steps: usize,
) -> f64 {
    assert_eq!(
        sims.len(),
        2,
        "oracle_two_pool_direct_only_best_ev_with_holdings_grid expects 2 pools"
    );
    assert_eq!(
        initial_holdings.len(),
        2,
        "oracle_two_pool_direct_only_best_ev_with_holdings_grid expects 2 holdings"
    );

    let steps = steps.max(1);
    let mut leg_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(2);
    for i in 0..2 {
        let sim = &sims[i];
        let held = initial_holdings[i].max(0.0);
        let sell_cap = held.min(sim.max_sell_tokens().max(0.0));
        let buy_cap = sim.max_buy_tokens().max(0.0);

        let mut points: Vec<(f64, f64)> = Vec::with_capacity(2 * steps + 3);
        points.push((0.0, 0.0)); // no-op leg
        for k in 0..=steps {
            let t = (k as f64) / (steps as f64);
            let scaled = t * t;

            let req_sell = sell_cap * scaled;
            if let Some((sold, proceeds, _)) = sim.sell_exact(req_sell) {
                if sold <= held + 1e-12 {
                    points.push((proceeds, -sold));
                }
            }

            let req_buy = buy_cap * scaled;
            if let Some((bought, cost, _)) = sim.buy_exact(req_buy) {
                points.push((-cost, bought));
            }
        }
        leg_points.push(points);
    }

    let mut best = f64::NEG_INFINITY;
    let pred0 = sims[0].prediction;
    let pred1 = sims[1].prediction;
    let h0 = initial_holdings[0].max(0.0);
    let h1 = initial_holdings[1].max(0.0);
    for (cash0, delta0) in &leg_points[0] {
        for (cash1, delta1) in &leg_points[1] {
            let final_cash = budget + *cash0 + *cash1;
            if final_cash < -1e-9 {
                continue;
            }
            let final_h0 = h0 + *delta0;
            let final_h1 = h1 + *delta1;
            if final_h0 < -1e-9 || final_h1 < -1e-9 {
                continue;
            }
            let ev = final_cash + pred0 * final_h0 + pred1 * final_h1;
            if ev > best {
                best = ev;
            }
        }
    }

    if best.is_finite() {
        best
    } else {
        budget + pred0 * h0 + pred1 * h1
    }
}

fn flash_loan_totals(actions: &[Action]) -> (f64, f64) {
    let mut borrowed = 0.0_f64;
    let mut repaid = 0.0_f64;
    for a in actions {
        match a {
            Action::FlashLoan { amount } => borrowed += *amount,
            Action::RepayFlashLoan { amount } => repaid += *amount,
            _ => {}
        }
    }
    (borrowed, repaid)
}

fn buy_totals(actions: &[Action]) -> HashMap<&'static str, f64> {
    let mut out: HashMap<&'static str, f64> = HashMap::new();
    for a in actions {
        if let Action::Buy {
            market_name,
            amount,
            ..
        } = a
        {
            *out.entry(*market_name).or_insert(0.0) += *amount;
        }
    }
    out
}

fn assert_action_values_are_finite(actions: &[Action]) {
    for action in actions {
        match action {
            Action::Buy { amount, cost, .. } => {
                assert!(amount.is_finite() && *amount >= 0.0);
                assert!(cost.is_finite() && *cost >= 0.0);
            }
            Action::Sell {
                amount, proceeds, ..
            } => {
                assert!(amount.is_finite() && *amount >= 0.0);
                assert!(proceeds.is_finite() && *proceeds >= 0.0);
            }
            Action::Mint { amount, .. }
            | Action::Merge { amount, .. }
            | Action::FlashLoan { amount }
            | Action::RepayFlashLoan { amount } => {
                assert!(amount.is_finite() && *amount >= 0.0);
            }
        }
    }
}

fn assert_flash_loan_ordering(actions: &[Action]) -> usize {
    let mut open_loan: Option<f64> = None;
    let mut steps_inside = 0usize;
    let mut brackets = 0usize;

    for action in actions {
        match action {
            Action::FlashLoan { amount } => {
                assert!(
                    open_loan.is_none(),
                    "nested FlashLoan bracket is not allowed"
                );
                open_loan = Some(*amount);
                steps_inside = 0;
                brackets += 1;
            }
            Action::RepayFlashLoan { amount } => {
                let borrowed = open_loan
                    .take()
                    .expect("RepayFlashLoan must close an open FlashLoan bracket");
                assert!(
                    steps_inside > 0,
                    "flash bracket should contain at least one operation"
                );
                let tol = 1e-8 * (1.0 + borrowed.abs() + amount.abs());
                assert!(
                    (borrowed - *amount).abs() <= tol,
                    "flash bracket amount mismatch: borrowed={:.12}, repaid={:.12}, tol={:.12}",
                    borrowed,
                    amount,
                    tol
                );
            }
            Action::Buy { .. }
            | Action::Sell { .. }
            | Action::Mint { .. }
            | Action::Merge { .. } => {
                if open_loan.is_some() {
                    steps_inside += 1;
                }
            }
        }
    }

    assert!(open_loan.is_none(), "unterminated FlashLoan bracket");
    brackets
}

#[derive(Clone)]
struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // [0, 1)
        (self.next_u64() as f64) / ((u64::MAX as f64) + 1.0)
    }

    fn in_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    fn pick(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() % (upper_exclusive as u64)) as usize
    }

    fn chance(&mut self, numer: u64, denom: u64) -> bool {
        (self.next_u64() % denom) < numer
    }
}

#[path = "tests/execution.rs"]
mod execution;
#[path = "tests/fuzz_rebalance.rs"]
mod fuzz_rebalance;
#[path = "tests/oracle.rs"]
mod oracle;
