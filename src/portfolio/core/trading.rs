use std::collections::{HashMap, HashSet};

use super::Action;
use super::merge::{
    action_contract_pair, execute_merge_sell_with_inventory, merge_usable_inventory,
    optimal_sell_split_with_inventory,
};
use super::planning::PlannedRoute;
use super::sim::{DUST, EPS, FEE_FACTOR, PoolSim, Route, sanitize_nonnegative_finite};
use super::types::{BalanceMap, apply_actions_to_sim_balances, subtract_balance};

const MAX_ALT_ROUTE_ROUNDS: usize = 256;

pub(super) struct ExecutionState<'a> {
    pub(super) sims: &'a mut [PoolSim],
    pub(super) budget: &'a mut f64,
    pub(super) actions: &'a mut Vec<Action>,
    pub(super) sim_balances: &'a mut BalanceMap,
}

impl<'a> ExecutionState<'a> {
    pub(super) fn new(
        sims: &'a mut [PoolSim],
        budget: &'a mut f64,
        actions: &'a mut Vec<Action>,
        sim_balances: &'a mut BalanceMap,
    ) -> Self {
        Self {
            sims,
            budget,
            actions,
            sim_balances,
        }
    }
}

pub(super) fn complete_set_arb_cap(sims: &[PoolSim]) -> f64 {
    if sims.is_empty() {
        return 0.0;
    }
    sims.iter()
        .map(|s| s.max_buy_tokens())
        .fold(f64::INFINITY, f64::min)
}

pub(super) fn complete_set_marginal_buy_cost(sims: &[PoolSim], amount: f64) -> f64 {
    let mut total = 0.0_f64;
    for s in sims {
        let lam = s.lambda();
        if lam <= 0.0 || s.price() <= 0.0 {
            return f64::INFINITY;
        }
        let d = 1.0 - amount * lam;
        if d <= 0.0 {
            return f64::INFINITY;
        }
        total += s.price() / (FEE_FACTOR * d * d);
    }
    total
}

fn bisect_boundary(mut lo: f64, mut hi: f64, mut go_right: impl FnMut(f64) -> bool) -> f64 {
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if go_right(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() <= EPS * (1.0 + hi.abs()) {
            break;
        }
    }
    0.5 * (lo + hi)
}

pub(super) fn solve_complete_set_arb_amount(sims: &[PoolSim]) -> f64 {
    let cap = complete_set_arb_cap(sims);
    if cap <= DUST {
        return 0.0;
    }

    let d0 = complete_set_marginal_buy_cost(sims, 0.0);
    if !d0.is_finite() || d0 >= 1.0 {
        return 0.0;
    }

    let cap_left = (cap - EPS * (1.0 + cap)).max(0.0);
    let d_cap = complete_set_marginal_buy_cost(sims, cap_left);
    if d_cap <= 1.0 {
        return cap;
    }

    bisect_boundary(0.0, cap_left, |mid| {
        complete_set_marginal_buy_cost(sims, mid) <= 1.0
    })
}

pub(super) fn complete_set_mint_sell_arb_cap(sims: &[PoolSim]) -> f64 {
    if sims.is_empty() {
        return 0.0;
    }
    sims.iter()
        .map(|s| s.max_sell_tokens())
        .fold(f64::INFINITY, f64::min)
}

pub(super) fn complete_set_marginal_sell_proceeds(sims: &[PoolSim], amount: f64) -> f64 {
    let mut total = 0.0_f64;
    for s in sims {
        let kappa = s.kappa();
        if kappa <= 0.0 || s.price() <= 0.0 {
            return 0.0;
        }
        let d = 1.0 + amount * kappa;
        if d <= 0.0 {
            return 0.0;
        }
        total += s.price() * FEE_FACTOR / (d * d);
    }
    total
}

pub(super) fn solve_complete_set_mint_sell_arb_amount(sims: &[PoolSim]) -> f64 {
    let cap = complete_set_mint_sell_arb_cap(sims);
    if cap <= DUST {
        return 0.0;
    }

    let d0 = complete_set_marginal_sell_proceeds(sims, 0.0);
    if !d0.is_finite() || d0 <= 1.0 {
        return 0.0;
    }

    let d_cap = complete_set_marginal_sell_proceeds(sims, cap);
    if d_cap >= 1.0 {
        return cap;
    }

    bisect_boundary(0.0, cap, |mid| {
        complete_set_marginal_sell_proceeds(sims, mid) >= 1.0
    })
}

fn complete_set_buy_cost(sims: &[PoolSim], amount: f64) -> Option<f64> {
    if amount <= DUST {
        return Some(0.0);
    }
    let mut total = 0.0_f64;
    for sim in sims {
        match sim.buy_exact(amount) {
            Some((bought, cost, _)) if bought + EPS >= amount && cost.is_finite() => {
                total += cost;
            }
            _ => return None,
        }
    }
    Some(total)
}

fn preview_complete_set_buy_round(
    sims: &[PoolSim],
    amount: f64,
) -> Option<(Vec<(usize, f64, f64)>, f64)> {
    if amount <= DUST {
        return Some((Vec::new(), 0.0));
    }
    let mut legs: Vec<(usize, f64, f64)> = Vec::with_capacity(sims.len());
    let mut total_buy_cost = 0.0_f64;
    for (i, sim) in sims.iter().enumerate() {
        match sim.buy_exact(amount) {
            Some((bought, cost, new_price))
                if bought + EPS >= amount && bought > 0.0 && cost.is_finite() =>
            {
                legs.push((i, cost, new_price));
                total_buy_cost += cost;
            }
            _ => return None,
        }
    }
    Some((legs, total_buy_cost))
}

fn affordable_complete_set_buy_amount(sims: &[PoolSim], upper: f64, budget: f64) -> f64 {
    if upper <= DUST || budget <= DUST {
        return 0.0;
    }
    let max_cost = match complete_set_buy_cost(sims, upper) {
        Some(cost) => cost,
        None => return 0.0,
    };
    if max_cost <= budget + EPS {
        return upper;
    }
    bisect_boundary(0.0, upper, |mid| {
        complete_set_buy_cost(sims, mid)
            .map(|cost| cost <= budget + EPS)
            .unwrap_or(false)
    })
}

fn find_profitable_complete_set_buy_round(
    sims: &[PoolSim],
    upper: f64,
    budget: f64,
) -> Option<(f64, Vec<(usize, f64, f64)>, f64)> {
    let mut round_amount = affordable_complete_set_buy_amount(sims, upper, budget);
    for _ in 0..64 {
        if round_amount <= DUST {
            return None;
        }
        let Some((legs, total_buy_cost)) = preview_complete_set_buy_round(sims, round_amount)
        else {
            round_amount *= 0.5;
            continue;
        };
        if total_buy_cost > budget + EPS {
            round_amount *= 0.5;
            continue;
        }
        let round_profit = round_amount - total_buy_cost;
        if round_profit.is_finite() && round_profit > EPS {
            return Some((round_amount, legs, total_buy_cost));
        }
        round_amount *= 0.5;
    }
    None
}

fn preview_complete_set_mint_sell_round(
    sims: &[PoolSim],
    amount: f64,
) -> Option<(Vec<(usize, f64, f64)>, f64)> {
    if amount <= DUST {
        return Some((Vec::new(), 0.0));
    }
    let mut legs: Vec<(usize, f64, f64)> = Vec::with_capacity(sims.len());
    let mut proceeds = 0.0_f64;
    for (i, sim) in sims.iter().enumerate() {
        match sim.sell_exact(amount) {
            Some((sold, leg_proceeds, new_price))
                if sold + EPS >= amount && sold > 0.0 && leg_proceeds.is_finite() =>
            {
                legs.push((i, leg_proceeds, new_price));
                proceeds += leg_proceeds;
            }
            _ => return None,
        }
    }
    Some((legs, proceeds))
}

fn find_profitable_complete_set_mint_sell_round(
    sims: &[PoolSim],
    upper: f64,
) -> Option<(f64, Vec<(usize, f64, f64)>, f64)> {
    let mut round_amount = upper;
    for _ in 0..64 {
        if round_amount <= DUST {
            return None;
        }
        let Some((legs, proceeds)) = preview_complete_set_mint_sell_round(sims, round_amount)
        else {
            round_amount *= 0.5;
            continue;
        };
        let round_profit = proceeds - round_amount;
        if round_profit.is_finite() && round_profit > EPS {
            return Some((round_amount, legs, proceeds));
        }
        round_amount *= 0.5;
    }
    None
}

impl<'a> ExecutionState<'a> {
    pub(super) fn execute_complete_set_arb(&mut self) -> f64 {
        let mut remaining = solve_complete_set_arb_amount(self.sims);
        if remaining <= DUST {
            return 0.0;
        }

        let mut realized_profit = 0.0_f64;
        for _ in 0..MAX_ALT_ROUTE_ROUNDS {
            if remaining <= DUST {
                break;
            }

            let liquidity_cap = complete_set_arb_cap(self.sims);
            if liquidity_cap <= DUST {
                break;
            }

            let round_upper = remaining.min(liquidity_cap);
            let Some((round_amount, legs, total_buy_cost)) =
                find_profitable_complete_set_buy_round(self.sims, round_upper, *self.budget)
            else {
                break;
            };
            let round_profit = round_amount - total_buy_cost;

            *self.budget -= total_buy_cost;
            for (i, cost, new_price) in legs {
                self.sims[i].set_price(new_price);
                self.actions.push(Action::Buy {
                    market_name: self.sims[i].market_name,
                    amount: round_amount,
                    cost,
                });
            }

            let (contract_1, contract_2) = action_contract_pair(self.sims);
            self.actions.push(Action::Merge {
                contract_1,
                contract_2,
                amount: round_amount,
                source_market: "complete_set_arb",
            });

            *self.budget += round_amount;
            realized_profit += round_profit;
            remaining -= round_amount;
        }

        realized_profit
    }

    pub(super) fn execute_complete_set_mint_sell_arb(&mut self) -> f64 {
        let mut remaining = solve_complete_set_mint_sell_arb_amount(self.sims);
        if remaining <= DUST {
            return 0.0;
        }

        let mut realized_profit = 0.0_f64;

        for _ in 0..MAX_ALT_ROUTE_ROUNDS {
            if remaining <= DUST {
                break;
            }

            let liquidity_cap = complete_set_mint_sell_arb_cap(self.sims);
            if liquidity_cap <= DUST {
                break;
            }

            let cash_cap = (*self.budget).max(0.0);
            if cash_cap <= DUST {
                break;
            }

            let round_amount = remaining.min(liquidity_cap).min(cash_cap);
            let Some((round_amount, legs, proceeds)) =
                find_profitable_complete_set_mint_sell_round(self.sims, round_amount)
            else {
                break;
            };
            let round_profit = proceeds - round_amount;

            *self.budget -= round_amount;
            *self.budget += proceeds;

            let (contract_1, contract_2) = action_contract_pair(self.sims);
            self.actions.push(Action::Mint {
                contract_1,
                contract_2,
                amount: round_amount,
                target_market: "complete_set_arb",
            });
            for (i, leg_proceeds, new_price) in legs {
                self.sims[i].set_price(new_price);
                self.actions.push(Action::Sell {
                    market_name: self.sims[i].market_name,
                    amount: round_amount,
                    proceeds: leg_proceeds,
                });
            }

            realized_profit += round_profit;
            remaining -= round_amount;
        }

        realized_profit
    }

    pub(super) fn execute_two_sided_complete_set_arb(&mut self) -> f64 {
        let price_sum: f64 = self.sims.iter().map(|s| s.price()).sum();
        if price_sum < 1.0 - EPS {
            self.execute_complete_set_arb()
        } else if price_sum > 1.0 + EPS {
            self.execute_complete_set_mint_sell_arb()
        } else {
            0.0
        }
    }

    pub(super) fn execute_planned_routes(
        &mut self,
        plan: &[PlannedRoute],
        skip: &HashSet<usize>,
    ) -> bool {
        for step in plan {
            if step.cost > *self.budget + EPS {
                return false;
            }
            if !execute_buy(
                self,
                step.idx,
                step.cost,
                step.amount,
                step.route,
                step.new_price,
                skip,
            ) {
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    pub(super) fn execute_optimal_sell(
        &mut self,
        source_idx: usize,
        sell_amount: f64,
        inventory_keep_prof: f64,
        mint_available: bool,
    ) -> f64 {
        self.execute_optimal_sell_with_merge_gates(
            source_idx,
            sell_amount,
            inventory_keep_prof,
            mint_available,
            true,
            true,
        )
    }

    pub(super) fn execute_optimal_sell_with_merge_gates(
        &mut self,
        source_idx: usize,
        sell_amount: f64,
        inventory_keep_prof: f64,
        mint_available: bool,
        allow_buy_merge: bool,
        allow_direct_merge: bool,
    ) -> f64 {
        if sell_amount <= 0.0 {
            return 0.0;
        }
        let sim_balances = &mut *self.sim_balances;

        let merge_target = if mint_available {
            let (m_opt, _) = optimal_sell_split_with_inventory(
                self.sims,
                source_idx,
                sell_amount,
                Some(sim_balances),
                inventory_keep_prof,
            );
            if m_opt > DUST {
                let needs_buy_legs = self
                    .sims
                    .iter()
                    .enumerate()
                    .any(|(i, sim)| {
                        i != source_idx
                            && m_opt
                                > merge_usable_inventory(
                                    Some(sim_balances),
                                    sim,
                                    inventory_keep_prof,
                                ) + DUST
                    });
                let route_allowed = if needs_buy_legs {
                    allow_buy_merge
                } else {
                    allow_direct_merge
                };
                if route_allowed { m_opt } else { 0.0 }
            } else {
                0.0
            }
        } else {
            0.0
        };

        let mut sold_total = 0.0_f64;
        if merge_target > DUST {
            let merged = execute_merge_sell_with_inventory(
                self.sims,
                source_idx,
                merge_target,
                sim_balances,
                inventory_keep_prof,
                self.actions,
                self.budget,
            );
            sold_total += merged;
        }

        let remainder = (sell_amount - sold_total).max(0.0);
        if remainder > DUST
            && let Some((sold, proceeds, new_price)) = self.sims[source_idx].sell_exact(remainder)
            && sold > 0.0
        {
            self.sims[source_idx].set_price(new_price);
            *self.budget += proceeds;
            sold_total += sold;
            subtract_balance(sim_balances, self.sims[source_idx].market_name, sold);
            self.actions.push(Action::Sell {
                market_name: self.sims[source_idx].market_name,
                amount: sold,
                proceeds,
            });
        }

        sold_total
    }
}

pub(super) fn portfolio_expected_value(
    sims: &[PoolSim],
    sim_balances: &HashMap<&str, f64>,
    cash: f64,
) -> f64 {
    if !cash.is_finite() {
        return f64::NEG_INFINITY;
    }
    let holdings_ev: f64 = sims
        .iter()
        .map(|sim| {
            let held = sanitize_nonnegative_finite(
                sim_balances.get(sim.market_name).copied().unwrap_or(0.0),
            );
            sim.prediction * held
        })
        .sum();
    let ev = cash + holdings_ev;
    if ev.is_finite() {
        ev
    } else {
        f64::NEG_INFINITY
    }
}

/// Emit mint actions: mint on both contracts, sell all non-target outcomes.
/// Contract addresses are derived from the distinct market_ids in sims.
pub(super) fn emit_mint_actions(
    sims: &mut [PoolSim],
    target_idx: usize,
    amount: f64,
    actions: &mut Vec<Action>,
    skip: &HashSet<usize>,
) -> Option<f64> {
    if amount <= DUST {
        return Some(0.0);
    }

    let mut legs: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(sims.len().saturating_sub(1));
    let mut total_proceeds = 0.0_f64;
    for (i, sim) in sims.iter().enumerate() {
        if i == target_idx || skip.contains(&i) {
            continue;
        }
        match sim.sell_exact(amount) {
            Some((sold, proceeds, new_price))
                if sold + EPS >= amount && sold > 0.0 && proceeds.is_finite() =>
            {
                total_proceeds += proceeds;
                legs.push((i, sold, proceeds, new_price));
            }
            _ => return None,
        }
    }

    if legs.is_empty() {
        return None;
    }

    let (contract_1, contract_2) = action_contract_pair(sims);
    actions.push(Action::Mint {
        contract_1,
        contract_2,
        amount,
        target_market: sims[target_idx].market_name,
    });

    // Sell all other outcomes across both contracts, update pool states
    for (i, sold, proceeds, new_price) in legs {
        sims[i].set_price(new_price);
        actions.push(Action::Sell {
            market_name: sims[i].market_name,
            amount: sold,
            proceeds,
        });
    }
    Some(total_proceeds)
}

fn mint_route_round_liquidity_cap(
    sims: &[PoolSim],
    target_idx: usize,
    skip: &HashSet<usize>,
) -> f64 {
    let mut cap = f64::INFINITY;
    let mut has_leg = false;
    for (i, sim) in sims.iter().enumerate() {
        if i == target_idx || skip.contains(&i) {
            continue;
        }
        has_leg = true;
        cap = cap.min(sim.max_sell_tokens().max(0.0));
    }
    if has_leg { cap } else { 0.0 }
}

fn mint_route_round_net_cost(
    sims: &[PoolSim],
    target_idx: usize,
    amount: f64,
    skip: &HashSet<usize>,
) -> Option<f64> {
    if amount <= DUST {
        return Some(0.0);
    }
    let mut proceeds = 0.0_f64;
    let mut has_leg = false;
    for (i, sim) in sims.iter().enumerate() {
        if i == target_idx || skip.contains(&i) {
            continue;
        }
        has_leg = true;
        let (sold, leg_proceeds, _) = sim.sell_exact(amount)?;
        if sold + EPS < amount || sold <= 0.0 || !leg_proceeds.is_finite() {
            return None;
        }
        proceeds += leg_proceeds;
    }
    if !has_leg {
        return None;
    }
    Some(amount - proceeds)
}

fn affordable_mint_route_round_amount(
    sims: &[PoolSim],
    target_idx: usize,
    upper: f64,
    skip: &HashSet<usize>,
    available_cash: f64,
) -> f64 {
    if upper <= DUST || available_cash <= DUST {
        return 0.0;
    }
    let affordable = |amount: f64| -> bool {
        mint_route_round_net_cost(sims, target_idx, amount, skip)
            .map(|net_cost| net_cost <= available_cash + EPS)
            .unwrap_or(false)
    };

    if affordable(upper) {
        return upper;
    }

    let mut lo = 0.0_f64;
    let mut hi = upper;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if affordable(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= EPS * (1.0 + hi.abs()) {
            break;
        }
    }
    lo
}

fn execute_mint_buy_in_rounds(
    exec: &mut ExecutionState<'_>,
    target_idx: usize,
    total_amount: f64,
    skip: &HashSet<usize>,
) -> f64 {
    let mut remaining = total_amount;
    let mut minted_total = 0.0_f64;

    for _ in 0..MAX_ALT_ROUTE_ROUNDS {
        if remaining <= DUST {
            break;
        }

        let liquidity_cap = mint_route_round_liquidity_cap(exec.sims, target_idx, skip);
        if liquidity_cap <= DUST {
            break;
        }

        let cash_cap = (*exec.budget).max(0.0);
        if cash_cap <= DUST {
            break;
        }

        let round_upper = remaining.min(liquidity_cap).min(cash_cap);
        let round_amount =
            affordable_mint_route_round_amount(exec.sims, target_idx, round_upper, skip, cash_cap);
        if round_amount <= DUST {
            break;
        }

        let action_start = exec.actions.len();
        *exec.budget -= round_amount;
        let proceeds =
            match emit_mint_actions(exec.sims, target_idx, round_amount, exec.actions, skip) {
                Some(value) => value,
                None => {
                    *exec.budget += round_amount;
                    break;
                }
            };
        *exec.budget += proceeds;
        apply_actions_to_sim_balances(&exec.actions[action_start..], exec.sims, exec.sim_balances);

        minted_total += round_amount;
        remaining -= round_amount;
    }

    minted_total
}

/// Execute a buy via the chosen route, updating state.
pub(super) fn execute_buy(
    exec: &mut ExecutionState<'_>,
    idx: usize,
    cost: f64,
    amount: f64,
    route: Route,
    new_price: Option<f64>,
    skip: &HashSet<usize>,
) -> bool {
    if amount <= 0.0 {
        return true;
    }
    match route {
        Route::Direct => {
            let Some(np) = new_price else {
                return false;
            };
            *exec.budget -= cost;
            exec.actions.push(Action::Buy {
                market_name: exec.sims[idx].market_name,
                amount,
                cost,
            });
            exec.sims[idx].set_price(np);
            *exec
                .sim_balances
                .entry(exec.sims[idx].market_name)
                .or_insert(0.0) += amount;
            true
        }
        Route::Mint => {
            // Keep mint execution atomic at step level: either satisfy the planned
            // amount, or revert all interim round mutations before failing closed.
            let budget_before = *exec.budget;
            let actions_len_before = exec.actions.len();
            let sims_before = exec.sims.to_vec();
            let balances_before = exec.sim_balances.clone();
            let minted = execute_mint_buy_in_rounds(exec, idx, amount, skip);
            if minted + EPS >= amount {
                true
            } else {
                *exec.budget = budget_before;
                exec.actions.truncate(actions_len_before);
                exec.sims.clone_from_slice(&sims_before);
                *exec.sim_balances = balances_before;
                false
            }
        }
    }
}
