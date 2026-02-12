use std::collections::HashMap;

use super::Action;
use super::sim::{DUST, EPS, FEE_FACTOR, PoolSim, profitability, sanitize_nonnegative_finite};
use super::types::subtract_balance;

pub(super) fn action_contract_pair(sims: &[PoolSim]) -> (&'static str, &'static str) {
    if sims.is_empty() {
        return ("", "");
    }
    // We only need the first two lexicographically smallest unique contract ids.
    let mut c1 = sims[0].market_id;
    let mut c2 = c1;
    for sim in sims.iter().skip(1) {
        let c = sim.market_id;
        if c < c1 {
            c2 = c1;
            c1 = c;
        } else if c != c1 && (c2 == c1 || c < c2) {
            c2 = c;
        }
    }
    (c1, c2)
}

/// Compute merge sell proceeds without modifying state (dry run).
/// Merge route: buy all other outcomes, merge complete sets, get sUSD back.
/// Returns (net_proceeds, actual_merge_amount).
#[cfg(test)]
pub(super) fn merge_sell_proceeds(sims: &[PoolSim], source_idx: usize, amount: f64) -> (f64, f64) {
    merge_sell_proceeds_with_inventory(sims, source_idx, amount, None, f64::INFINITY)
}

/// Compute merge sell proceeds (dry run), optionally consuming existing complementary holdings
/// before buying shortfall from pools.
pub(super) fn merge_sell_proceeds_with_inventory(
    sims: &[PoolSim],
    source_idx: usize,
    amount: f64,
    sim_balances: Option<&HashMap<&str, f64>>,
    inventory_keep_prof: f64,
) -> (f64, f64) {
    let merge_cap =
        merge_sell_cap_with_inventory(sims, source_idx, sim_balances, inventory_keep_prof);
    let actual = amount.min(merge_cap);
    if actual <= 0.0 {
        return (0.0, 0.0);
    }

    let mut total_buy_cost = 0.0_f64;
    for (i, s) in sims.iter().enumerate() {
        if i == source_idx {
            continue;
        }
        let held = merge_usable_inventory(sim_balances, s, inventory_keep_prof);
        let shortfall = (actual - held).max(0.0);
        if shortfall <= DUST {
            continue;
        }
        match s.buy_exact(shortfall) {
            Some((bought, cost, _))
                if bought + EPS >= shortfall && bought > 0.0 && cost.is_finite() =>
            {
                total_buy_cost += cost;
            }
            _ => {
                // Any failed leg makes merge infeasible for this amount.
                return (0.0, 0.0);
            }
        }
    }

    (actual - total_buy_cost, actual)
}

/// Marginal proceeds of selling `amount_sold` source tokens directly.
/// d/dm [proceeds_direct(m)] = P0 * (1-fee) / (1 + m*kappa)^2.
pub(super) fn direct_sell_marginal_proceeds(sim: &PoolSim, amount_sold: f64) -> f64 {
    if amount_sold < 0.0 {
        return 0.0;
    }
    let k = sim.kappa();
    if k <= 0.0 || sim.price <= 0.0 {
        return 0.0;
    }
    let d = 1.0 + amount_sold * k;
    if d <= 0.0 {
        return 0.0;
    }
    sim.price * FEE_FACTOR / (d * d)
}

pub(super) fn merge_sell_marginal_proceeds_with_inventory(
    sims: &[PoolSim],
    source_idx: usize,
    merge_amount: f64,
    sim_balances: Option<&HashMap<&str, f64>>,
    inventory_keep_prof: f64,
) -> f64 {
    let mut marginal_buy_cost_sum = 0.0_f64;
    for (i, sim) in sims.iter().enumerate() {
        if i == source_idx {
            continue;
        }
        let held = merge_usable_inventory(sim_balances, sim, inventory_keep_prof);
        if merge_amount <= held + DUST {
            continue;
        }
        let buy_amount = merge_amount - held;
        let lam = sim.lambda();
        if lam <= 0.0 || sim.price <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let d = 1.0 - buy_amount * lam;
        if d <= 0.0 {
            return f64::NEG_INFINITY;
        }
        marginal_buy_cost_sum += sim.price / (FEE_FACTOR * d * d);
    }
    1.0 - marginal_buy_cost_sum
}

/// Max merge amount constrained by non-source pools' buy caps.
#[cfg(test)]
pub(super) fn merge_sell_cap(sims: &[PoolSim], source_idx: usize) -> f64 {
    merge_sell_cap_with_inventory(sims, source_idx, None, f64::INFINITY)
}

pub(super) fn merge_sell_cap_with_inventory(
    sims: &[PoolSim],
    source_idx: usize,
    sim_balances: Option<&HashMap<&str, f64>>,
    inventory_keep_prof: f64,
) -> f64 {
    if sims.len() < 2 || source_idx >= sims.len() {
        return 0.0;
    }
    sims.iter()
        .enumerate()
        .filter(|(i, _)| *i != source_idx)
        .map(|(_, s)| {
            merge_usable_inventory(sim_balances, s, inventory_keep_prof) + s.max_buy_tokens()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Total proceeds when selling `sell_amount` source tokens with a split:
/// merge `merge_amount`, direct-sell the remainder.
#[cfg(test)]
pub(super) fn split_sell_total_proceeds(
    sims: &[PoolSim],
    source_idx: usize,
    sell_amount: f64,
    merge_amount: f64,
) -> (f64, f64) {
    split_sell_total_proceeds_with_inventory(
        sims,
        source_idx,
        sell_amount,
        merge_amount,
        None,
        f64::INFINITY,
    )
}

pub(super) fn split_sell_total_proceeds_with_inventory(
    sims: &[PoolSim],
    source_idx: usize,
    sell_amount: f64,
    merge_amount: f64,
    sim_balances: Option<&HashMap<&str, f64>>,
    inventory_keep_prof: f64,
) -> (f64, f64) {
    let (merge_net, merged_actual) = merge_sell_proceeds_with_inventory(
        sims,
        source_idx,
        merge_amount,
        sim_balances,
        inventory_keep_prof,
    );
    let remainder = (sell_amount - merged_actual).max(0.0);
    let direct_proceeds = sims[source_idx]
        .sell_exact(remainder)
        .map(|(_, p, _)| p)
        .unwrap_or(0.0);
    (merge_net + direct_proceeds, merged_actual)
}

/// Find the optimal merge/direct split for selling a fixed source amount.
/// Returns (merge_amount, expected_total_proceeds) under current pool state.
#[cfg(test)]
pub(super) fn optimal_sell_split(
    sims: &[PoolSim],
    source_idx: usize,
    sell_amount: f64,
) -> (f64, f64) {
    optimal_sell_split_with_inventory(sims, source_idx, sell_amount, None, f64::INFINITY)
}

pub(super) fn optimal_sell_split_with_inventory(
    sims: &[PoolSim],
    source_idx: usize,
    sell_amount: f64,
    sim_balances: Option<&HashMap<&str, f64>>,
    inventory_keep_prof: f64,
) -> (f64, f64) {
    const OPT_ITERS: usize = 48;
    const OPT_EPS: f64 = EPS;

    if sell_amount <= 0.0 {
        return (0.0, 0.0);
    }

    let source = &sims[source_idx];
    let merge_upper =
        merge_sell_cap_with_inventory(sims, source_idx, sim_balances, inventory_keep_prof)
            .min(sell_amount);
    let (direct_total, _) = split_sell_total_proceeds_with_inventory(
        sims,
        source_idx,
        sell_amount,
        0.0,
        sim_balances,
        inventory_keep_prof,
    );
    if merge_upper <= OPT_EPS {
        return (0.0, direct_total);
    }

    // Candidate boundary m=upper.
    let (upper_total, _) = split_sell_total_proceeds_with_inventory(
        sims,
        source_idx,
        sell_amount,
        merge_upper,
        sim_balances,
        inventory_keep_prof,
    );

    // Derivative of total objective:
    // f'(m) = merge_marginal(m) - direct_marginal(sell_amount - m).
    let d0 = merge_sell_marginal_proceeds_with_inventory(
        sims,
        source_idx,
        0.0,
        sim_balances,
        inventory_keep_prof,
    ) - direct_sell_marginal_proceeds(source, sell_amount);

    let upper_left = (merge_upper - OPT_EPS * (1.0 + merge_upper)).max(0.0);
    let d_upper = merge_sell_marginal_proceeds_with_inventory(
        sims,
        source_idx,
        upper_left,
        sim_balances,
        inventory_keep_prof,
    ) - direct_sell_marginal_proceeds(source, (sell_amount - upper_left).max(0.0));

    // Concave objective: if derivative is non-positive at 0, best is at 0.
    if d0 <= 0.0 {
        if upper_total > direct_total {
            return (merge_upper, upper_total);
        }
        return (0.0, direct_total);
    }

    // If derivative remains non-negative through the right edge, best is at upper bound.
    if d_upper >= 0.0 {
        if upper_total > direct_total {
            return (merge_upper, upper_total);
        }
        return (0.0, direct_total);
    }

    // Interior optimum via bisection on f'(m)=0.
    let mut lo = 0.0_f64;
    let mut hi = merge_upper;
    for _ in 0..OPT_ITERS {
        let mid = 0.5 * (lo + hi);
        let dm = merge_sell_marginal_proceeds_with_inventory(
            sims,
            source_idx,
            mid,
            sim_balances,
            inventory_keep_prof,
        ) - direct_sell_marginal_proceeds(source, (sell_amount - mid).max(0.0));
        if dm > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= OPT_EPS * (1.0 + merge_upper) {
            break;
        }
    }
    let m_star = 0.5 * (lo + hi);
    let (star_total, star_merged) = split_sell_total_proceeds_with_inventory(
        sims,
        source_idx,
        sell_amount,
        m_star,
        sim_balances,
        inventory_keep_prof,
    );

    // Numerical guard: pick the best among interior + boundaries.
    if direct_total >= upper_total && direct_total >= star_total {
        (0.0, direct_total)
    } else if upper_total >= star_total {
        (merge_upper, upper_total)
    } else {
        (star_merged, star_total)
    }
}

pub(super) fn held_balance(sim_balances: Option<&HashMap<&str, f64>>, market_name: &str) -> f64 {
    sanitize_nonnegative_finite(
        sim_balances
            .and_then(|b| b.get(market_name).copied())
            .unwrap_or(0.0),
    )
}

/// Inventory used in merge legs is gated by profitability: keep holdings with
/// profitability above `inventory_keep_prof` and only consume lower-profitability balances.
pub(super) fn merge_usable_inventory(
    sim_balances: Option<&HashMap<&str, f64>>,
    sim: &PoolSim,
    inventory_keep_prof: f64,
) -> f64 {
    if profitability(sim.prediction, sim.price()) > inventory_keep_prof {
        return 0.0;
    }
    held_balance(sim_balances, sim.market_name)
}

/// Execute a merge sell while consuming existing complementary holdings first.
/// Missing shortfall is bought from pools. Updates `sim_balances`.
pub(super) fn execute_merge_sell_with_inventory(
    sims: &mut [PoolSim],
    source_idx: usize,
    amount: f64,
    sim_balances: &mut HashMap<&str, f64>,
    inventory_keep_prof: f64,
    actions: &mut Vec<Action>,
    budget: &mut f64,
) -> f64 {
    let merge_cap =
        merge_sell_cap_with_inventory(sims, source_idx, Some(sim_balances), inventory_keep_prof);
    let actual = amount.min(merge_cap);
    if actual <= 0.0 {
        return 0.0;
    }

    // (idx, bought, cost, new_price, consumed_from_inventory)
    let mut legs: Vec<(usize, f64, f64, f64, f64)> =
        Vec::with_capacity(sims.len().saturating_sub(1));
    let mut total_buy_cost = 0.0_f64;
    for (i, s) in sims.iter().enumerate() {
        if i == source_idx {
            continue;
        }
        let held = merge_usable_inventory(Some(sim_balances), s, inventory_keep_prof);
        let consumed_from_inventory = actual.min(held);
        let buy_amount = (actual - consumed_from_inventory).max(0.0);
        if buy_amount <= DUST {
            legs.push((i, 0.0, 0.0, s.price, consumed_from_inventory));
            continue;
        }
        match s.buy_exact(buy_amount) {
            Some((bought, cost, new_price))
                if bought + EPS >= buy_amount && bought > 0.0 && cost.is_finite() =>
            {
                legs.push((i, bought, cost, new_price, consumed_from_inventory));
                total_buy_cost += cost;
            }
            _ => return 0.0,
        }
    }

    if total_buy_cost > DUST {
        actions.push(Action::FlashLoan {
            amount: total_buy_cost,
        });
    }

    for (i, bought, cost, new_price, _) in &legs {
        if *bought > DUST {
            sims[*i].price = *new_price;
            actions.push(Action::Buy {
                market_name: sims[*i].market_name,
                amount: *bought,
                cost: *cost,
            });
        }
    }

    let (contract_1, contract_2) = action_contract_pair(sims);
    actions.push(Action::Merge {
        contract_1,
        contract_2,
        amount: actual,
        source_market: sims[source_idx].market_name,
    });

    if total_buy_cost > DUST {
        actions.push(Action::RepayFlashLoan {
            amount: total_buy_cost,
        });
    }

    for (i, _, _, _, consumed_from_inventory) in &legs {
        if *consumed_from_inventory > DUST {
            subtract_balance(sim_balances, sims[*i].market_name, *consumed_from_inventory);
        }
    }
    subtract_balance(sim_balances, sims[source_idx].market_name, actual);

    *budget += actual - total_buy_cost;
    actual
}

/// Execute a merge sell: buy all other outcomes, merge complete sets, recover sUSD.
/// Updates pool states, emits actions, updates budget.
/// Returns the actual amount merged (tokens consumed from source holding).
#[cfg(test)]
pub(super) fn execute_merge_sell(
    sims: &mut [PoolSim],
    source_idx: usize,
    amount: f64,
    actions: &mut Vec<Action>,
    budget: &mut f64,
) -> f64 {
    // Test-only helper with no starting complementary inventory.
    let mut no_inventory: HashMap<&str, f64> = HashMap::new();
    execute_merge_sell_with_inventory(
        sims,
        source_idx,
        amount,
        &mut no_inventory,
        f64::INFINITY,
        actions,
        budget,
    )
}
