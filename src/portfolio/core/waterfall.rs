use std::collections::HashMap;

use super::Action;
use super::bundle::{BundleRouteKind, bundle_frontier};
use super::planning::plan_bundle_step_with_scratch;
use super::rebalancer::passes_execution_gate;
use super::sim::{EPS, PoolSim};
#[cfg(test)]
use super::sim::{Route, alt_price, profitability};
use super::trading::ExecutionState;
#[cfg(test)]
use std::collections::HashSet;

#[cfg(test)]
/// Legacy test helper: find the highest-profitability (outcome, route) pair not already active.
/// The runtime waterfall no longer uses this directly, but the oracle tests still depend on the
/// historical route-level search surface for regression comparisons.
pub(super) fn best_non_active(
    sims: &[PoolSim],
    active_set: &HashSet<(usize, Route)>,
    mint_available: bool,
    price_sum: f64,
    remaining_budget: f64,
    gas_direct_susd: f64,
    gas_mint_susd: f64,
) -> Option<(usize, Route, f64)> {
    let mut best: Option<(usize, Route, f64)> = None;
    for (i, sim) in sims.iter().enumerate() {
        if !active_set.contains(&(i, Route::Direct)) {
            let prof = profitability(sim.prediction, sim.price());
            if prof > 0.0
                && remaining_budget * prof >= gas_direct_susd
                && best.is_none_or(|b| prof > b.2)
            {
                best = Some((i, Route::Direct, prof));
            }
        }
        if mint_available && !active_set.contains(&(i, Route::Mint)) {
            let prof = profitability(sim.prediction, alt_price(sims, i, price_sum));
            if prof > 0.0
                && remaining_budget * prof >= gas_mint_susd
                && best.is_none_or(|b| prof > b.2)
            {
                best = Some((i, Route::Mint, prof));
            }
        }
    }
    best
}

pub(super) const MAX_WATERFALL_ITERS: usize = 1000;
const MAX_STALLED_CONTINUES: usize = 4;

fn iteration_made_progress(
    prev_prof: f64,
    next_prof: f64,
    prev_budget: f64,
    next_budget: f64,
) -> bool {
    let prof_tol = EPS * (1.0 + prev_prof.abs().max(next_prof.abs()));
    if (next_prof - prev_prof).abs() > prof_tol {
        return true;
    }

    let budget_tol = EPS * (1.0 + prev_budget.abs().max(next_budget.abs()));
    (next_budget - prev_budget).abs() > budget_tol
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct WaterfallGateStats {
    pub(super) skipped_direct: usize,
    pub(super) skipped_mint: usize,
    pub(super) steps_pruned_subgas: usize,
}

fn passes_step_execution_gate(
    step: &super::bundle::BundleSegmentPlan,
    current_prof: f64,
    gas_direct_susd: f64,
    gas_mint_susd: f64,
    buffer_frac: f64,
    buffer_min_susd: f64,
) -> bool {
    if step.cash_cost <= 0.0 {
        return true;
    }
    let edge_susd = step.cash_cost * current_prof.max(0.0);
    let gas_susd = match step.kind {
        BundleRouteKind::Direct => gas_direct_susd,
        BundleRouteKind::Mint => gas_mint_susd,
    };
    passes_execution_gate(edge_susd, gas_susd, buffer_frac, buffer_min_susd)
}

/// Waterfall allocation: deploy capital to the highest profitability outcome.
/// As capital is deployed, profitability drops until it matches the next outcome.
/// Then deploy to both, then three, etc.
///
/// Returns the profitability level of the last bought outcome (for post-liquidation).
#[allow(dead_code)]
pub(super) fn waterfall(
    sims: &mut [PoolSim],
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
    gas_direct_susd: f64,
    gas_mint_susd: f64,
) -> f64 {
    waterfall_with_execution_gate(
        sims,
        budget,
        actions,
        mint_available,
        gas_direct_susd,
        gas_mint_susd,
        0.0,
        0.0,
        None,
    )
}

pub(super) fn waterfall_with_execution_gate(
    sims: &mut [PoolSim],
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
    gas_direct_susd: f64,
    gas_mint_susd: f64,
    buffer_frac: f64,
    buffer_min_susd: f64,
    mut gate_stats: Option<&mut WaterfallGateStats>,
) -> f64 {
    if *budget <= 0.0 {
        return 0.0;
    }

    let mut last_prof = 0.0;
    let mut waterfall_balances: HashMap<&str, f64> = HashMap::new();
    let mut planning_sim_state: Vec<PoolSim> = Vec::with_capacity(sims.len());
    let mut stalled_continues = 0usize;

    for _iter in 0..MAX_WATERFALL_ITERS {
        let Some(frontier) = bundle_frontier(
            sims,
            mint_available,
            *budget,
            gas_direct_susd,
            gas_mint_susd,
        ) else {
            break;
        };
        if *budget <= EPS || frontier.current_prof <= 0.0 {
            break;
        }

        let iter_start_prof = frontier.current_prof;
        let iter_start_budget = *budget;
        let Some(plan) = plan_bundle_step_with_scratch(
            sims,
            &frontier,
            mint_available,
            Some(*budget),
            &mut planning_sim_state,
        ) else {
            last_prof = frontier.current_prof;
            break;
        };

        let mut executable_plan = plan.clone();
        let mut passing_segments = executable_plan
            .segments
            .iter()
            .take_while(|segment| {
                passes_step_execution_gate(
                    segment,
                    frontier.current_prof,
                    gas_direct_susd,
                    gas_mint_susd,
                    buffer_frac,
                    buffer_min_susd,
                )
            })
            .count();

        if passing_segments == 0 {
            if executable_plan
                .segments
                .first()
                .is_some_and(|segment| segment.kind == BundleRouteKind::Mint)
            {
                if let Some(direct_fallback) = plan_bundle_step_with_scratch(
                    sims,
                    &frontier,
                    false,
                    Some(*budget),
                    &mut planning_sim_state,
                ) {
                    let fallback_passing = direct_fallback
                        .segments
                        .iter()
                        .take_while(|segment| {
                            passes_step_execution_gate(
                                segment,
                                frontier.current_prof,
                                gas_direct_susd,
                                gas_mint_susd,
                                buffer_frac,
                                buffer_min_susd,
                            )
                        })
                        .count();
                    if fallback_passing > 0 {
                        executable_plan = direct_fallback;
                        passing_segments = fallback_passing;
                    }
                }
            }
        }

        if passing_segments == 0 {
            if let Some(stats) = gate_stats.as_deref_mut() {
                match executable_plan.segments.first().map(|segment| segment.kind) {
                    Some(BundleRouteKind::Direct) => stats.skipped_direct += 1,
                    Some(BundleRouteKind::Mint) => stats.skipped_mint += 1,
                    None => {}
                }
                stats.steps_pruned_subgas += 1;
            }
            last_prof = frontier.current_prof;
            break;
        }

        if passing_segments < executable_plan.segments.len() {
            if let Some(stats) = gate_stats.as_deref_mut() {
                match executable_plan.segments[passing_segments].kind {
                    BundleRouteKind::Direct => stats.skipped_direct += 1,
                    BundleRouteKind::Mint => stats.skipped_mint += 1,
                }
                stats.steps_pruned_subgas += 1;
            }
            executable_plan.segments.truncate(passing_segments);
        }

        let executed = {
            waterfall_balances.clear();
            let mut exec = ExecutionState::new(sims, budget, actions, &mut waterfall_balances);
            exec.execute_bundle_step(&executable_plan, &frontier.members)
        };
        if !executed {
            last_prof = frontier.current_prof;
            break;
        }

        last_prof = executable_plan.final_prof;
        if !iteration_made_progress(
            iter_start_prof,
            executable_plan.final_prof,
            iter_start_budget,
            *budget,
        ) {
            stalled_continues += 1;
            if stalled_continues >= MAX_STALLED_CONTINUES {
                break;
            }
        } else {
            stalled_continues = 0;
        }
    }

    last_prof
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markets::{MARKETS_L1, MarketData, Pool, Tick};
    use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};
    use alloy::primitives::{Address, U256};

    /// Minimal PoolSim for waterfall tests: price ~= price_frac, prediction = pred.
    /// Uses Box::leak to produce 'static references (test process memory, not freed).
    fn make_sim(name: &'static str, token: &'static str, price_frac: f64, pred: f64) -> PoolSim {
        let liq_str: &'static str =
            Box::leak("1000000000000000000000".to_string().into_boxed_str());
        let ticks: &'static [Tick] = Box::leak(Box::new([
            Tick {
                tick_idx: 1,
                liquidity_net: 1_000_000_000_000_000_000_000,
            },
            Tick {
                tick_idx: 92108,
                liquidity_net: -1_000_000_000_000_000_000_000,
            },
        ]));
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
            token1: token,
            pool_id: "0x0000000000000000000000000000000000000001",
            liquidity: liq_str,
            ticks,
        }));
        let sqrt =
            prediction_to_sqrt_price_x96(price_frac, true).unwrap_or(U256::from(1u128 << 96));
        let market: &'static MarketData = Box::leak(Box::new(MarketData {
            name,
            market_id: MARKETS_L1[0].market_id,
            outcome_token: token,
            pool: Some(*pool),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        }));
        let slot0 = Slot0Result {
            pool_id: Address::ZERO,
            sqrt_price_x96: sqrt,
            tick: 0,
            observation_index: 0,
            observation_cardinality: 0,
            observation_cardinality_next: 0,
            fee_protocol: 0,
            unlocked: true,
        };
        PoolSim::from_slot0(&slot0, market, pred).unwrap()
    }

    #[test]
    fn waterfall_skips_outcome_when_budget_below_break_even() {
        // profitability ≈ (0.0101 - 0.01) / 0.01 = 1% = 0.01
        // gas_direct = $0.50, break-even min budget = 0.50 / 0.01 = $50
        // budget = $1 < $50 → outcome must be skipped → zero actions
        let mut sims = vec![make_sim(
            "m1",
            "0x1111111111111111111111111111111111111111",
            0.01,
            0.0101,
        )];
        let mut budget = 1.0_f64;
        let mut actions = vec![];

        let last_prof = waterfall(
            &mut sims,
            &mut budget,
            &mut actions,
            false,
            0.50, // gas_direct_susd: $0.50
            2.00, // gas_mint_susd: $2.00
        );
        assert!(
            actions.is_empty(),
            "budget $1 at 1% profitability cannot cover $0.50 gas; got {} actions, last_prof={last_prof}",
            actions.len()
        );
        assert!(
            (budget - 1.0).abs() < 1e-9,
            "budget must be unchanged when all trades skipped; got {budget}"
        );
    }

    #[test]
    fn waterfall_executes_when_budget_above_break_even() {
        // Use a much wider mispricing so the first executable step has enough edge to clear gas.
        let mut sims = vec![make_sim(
            "m2",
            "0x2222222222222222222222222222222222222222",
            0.01,
            0.04,
        )];
        let mut budget = 100.0_f64;
        let mut actions = vec![];

        let _last_prof = waterfall(&mut sims, &mut budget, &mut actions, false, 0.50, 2.00);
        assert!(
            !actions.is_empty(),
            "deeply underpriced direct alpha should clear the $0.50 gas gate; got 0 actions"
        );
    }

    #[test]
    fn waterfall_with_zero_gas_thresholds_behaves_as_before() {
        // gas_direct=0, gas_mint=0 → no filtering, same as old signature
        let mut sims = vec![make_sim(
            "m3",
            "0x3333333333333333333333333333333333333333",
            0.01,
            0.0101,
        )];
        let mut budget = 1.0_f64;
        let mut actions = vec![];

        let _last_prof = waterfall(&mut sims, &mut budget, &mut actions, false, 0.0, 0.0);
        assert!(
            !actions.is_empty(),
            "with zero gas thresholds, any positive profitability should produce actions"
        );
    }

    #[test]
    fn waterfall_prunes_subgas_steps() {
        let mut sims = vec![make_sim(
            "m4",
            "0x4444444444444444444444444444444444444444",
            0.01,
            0.0101,
        )];
        let mut budget = 1.0_f64;
        let mut actions = vec![];
        let mut gate_stats = WaterfallGateStats::default();

        let last_prof = waterfall_with_execution_gate(
            &mut sims,
            &mut budget,
            &mut actions,
            false,
            0.0,
            0.0,
            0.20,
            0.05,
            Some(&mut gate_stats),
        );
        assert!(
            last_prof.is_finite() && last_prof >= 0.0,
            "last profitability should remain finite after pruning"
        );
        assert!(actions.is_empty(), "sub-gas step should be pruned");
        assert!(
            (budget - 1.0).abs() <= 1e-9,
            "budget should remain unchanged after prune"
        );
        assert!(
            gate_stats.steps_pruned_subgas >= 1 && gate_stats.skipped_direct >= 1,
            "expected direct step to be pruned by gate: {:?}",
            gate_stats
        );
    }

    #[test]
    fn waterfall_subgas_prune_does_not_emit_nonfinite_or_overspend() {
        let mut sims = vec![
            make_sim(
                "m5",
                "0x5555555555555555555555555555555555555555",
                0.01,
                0.0101,
            ),
            make_sim(
                "m6",
                "0x6666666666666666666666666666666666666666",
                0.02,
                0.0202,
            ),
        ];
        let start_budget = 10.0_f64;
        let mut budget = start_budget;
        let mut actions = vec![];
        let mut gate_stats = WaterfallGateStats::default();

        let _last_prof = waterfall_with_execution_gate(
            &mut sims,
            &mut budget,
            &mut actions,
            false,
            0.0,
            0.0,
            0.20,
            0.25,
            Some(&mut gate_stats),
        );

        assert!(budget.is_finite(), "budget must remain finite");
        assert!(
            budget <= start_budget + 1e-9,
            "direct-only execution cannot overspend"
        );
        for action in &actions {
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
                Action::Mint { amount, .. } | Action::Merge { amount, .. } => {
                    assert!(amount.is_finite() && *amount >= 0.0);
                }
            }
        }
        assert!(
            gate_stats.steps_pruned_subgas >= 1,
            "expected at least one step pruning event"
        );
    }
}
