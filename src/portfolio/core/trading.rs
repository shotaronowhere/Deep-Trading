use std::collections::{HashMap, HashSet};

use super::Action;
use super::merge::{
    action_contract_pair, execute_merge_sell_with_inventory, optimal_sell_split_with_inventory,
};
use super::planning::PlannedRoute;
use super::sim::{DUST, EPS, FEE_FACTOR, PoolSim, Route, sanitize_nonnegative_finite};
use super::types::{BalanceMap, subtract_balance};

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

impl<'a> ExecutionState<'a> {
    pub(super) fn execute_complete_set_arb(&mut self) -> f64 {
        let amount = solve_complete_set_arb_amount(self.sims);
        if amount <= DUST {
            return 0.0;
        }

        let mut legs: Vec<(usize, f64, f64)> = Vec::with_capacity(self.sims.len());
        let mut total_buy_cost = 0.0_f64;
        for (i, s) in self.sims.iter().enumerate() {
            match s.buy_exact(amount) {
                Some((bought, cost, new_price))
                    if bought + EPS >= amount && bought > 0.0 && cost.is_finite() =>
                {
                    legs.push((i, cost, new_price));
                    total_buy_cost += cost;
                }
                _ => return 0.0,
            }
        }

        let profit = amount - total_buy_cost;
        if !profit.is_finite() || profit <= EPS {
            return 0.0;
        }

        if total_buy_cost > DUST {
            self.actions.push(Action::FlashLoan {
                amount: total_buy_cost,
            });
        }

        for (i, cost, new_price) in legs {
            self.sims[i].set_price(new_price);
            self.actions.push(Action::Buy {
                market_name: self.sims[i].market_name,
                amount,
                cost,
            });
        }

        let (contract_1, contract_2) = action_contract_pair(self.sims);
        self.actions.push(Action::Merge {
            contract_1,
            contract_2,
            amount,
            source_market: "complete_set_arb",
        });

        if total_buy_cost > DUST {
            self.actions.push(Action::RepayFlashLoan {
                amount: total_buy_cost,
            });
        }

        *self.budget += profit;
        profit
    }

    pub(super) fn execute_complete_set_mint_sell_arb(&mut self) -> f64 {
        let amount = solve_complete_set_mint_sell_arb_amount(self.sims);
        if amount <= DUST {
            return 0.0;
        }

        let mut legs: Vec<(usize, f64, f64)> = Vec::with_capacity(self.sims.len());
        let mut total_proceeds = 0.0_f64;
        for (i, s) in self.sims.iter().enumerate() {
            match s.sell_exact(amount) {
                Some((sold, proceeds, new_price))
                    if sold + EPS >= amount && sold > 0.0 && proceeds.is_finite() =>
                {
                    legs.push((i, proceeds, new_price));
                    total_proceeds += proceeds;
                }
                _ => return 0.0,
            }
        }

        let profit = total_proceeds - amount;
        if !profit.is_finite() || profit <= EPS {
            return 0.0;
        }

        self.actions.push(Action::FlashLoan { amount });

        let (contract_1, contract_2) = action_contract_pair(self.sims);
        self.actions.push(Action::Mint {
            contract_1,
            contract_2,
            amount,
            target_market: "complete_set_arb",
        });

        for (i, proceeds, new_price) in legs {
            self.sims[i].set_price(new_price);
            self.actions.push(Action::Sell {
                market_name: self.sims[i].market_name,
                amount,
                proceeds,
            });
        }

        self.actions.push(Action::RepayFlashLoan { amount });
        *self.budget += profit;
        profit
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
            execute_buy(
                self,
                step.idx,
                step.cost,
                step.amount,
                step.route,
                step.new_price,
                skip,
            );
        }
        true
    }

    pub(super) fn execute_optimal_sell(
        &mut self,
        source_idx: usize,
        sell_amount: f64,
        inventory_keep_prof: f64,
        mint_available: bool,
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
            m_opt
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
) -> f64 {
    let (contract_1, contract_2) = action_contract_pair(sims);
    actions.push(Action::Mint {
        contract_1,
        contract_2,
        amount,
        target_market: sims[target_idx].market_name,
    });

    // Sell all other outcomes across both contracts, update pool states
    let mut total_proceeds = 0.0;
    for (i, sim) in sims.iter_mut().enumerate() {
        if i == target_idx || skip.contains(&i) {
            continue;
        }
        if let Some((sold, proceeds, new_price)) = sim.sell_exact(amount)
            && sold > 0.0
        {
            total_proceeds += proceeds;
            sim.set_price(new_price);
            actions.push(Action::Sell {
                market_name: sim.market_name,
                amount: sold,
                proceeds,
            });
        }
    }
    total_proceeds
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
) {
    if amount <= 0.0 {
        return;
    }
    match route {
        Route::Direct => {
            if let Some(np) = new_price {
                exec.sims[idx].set_price(np);
            }
            *exec.budget -= cost;
            exec.actions.push(Action::Buy {
                market_name: exec.sims[idx].market_name,
                amount,
                cost,
            });
        }
        Route::Mint => {
            // Flash loan funds the mint upfront; net cost comes from budget
            let upfront = amount; // 1 sUSD per set
            exec.actions.push(Action::FlashLoan { amount: upfront });
            let proceeds = emit_mint_actions(exec.sims, idx, amount, exec.actions, skip);
            exec.actions
                .push(Action::RepayFlashLoan { amount: upfront });
            let net_cost = upfront - proceeds;
            *exec.budget -= net_cost;
        }
    }
}
