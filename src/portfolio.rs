use std::collections::{HashMap, HashSet};

use uniswap_v3_math::tick_math::get_sqrt_ratio_at_tick;

use crate::pools::{
    FEE_PIPS, Slot0Result, normalize_market_name, prediction_map, sqrt_price_x96_to_price_outcome,
    u256_to_f64,
};

const NEWTON_ITERS: usize = 8;
const FEE_FACTOR: f64 = 1.0 - (FEE_PIPS as f64 / 1_000_000.0);

/// A rebalancing action to execute.
#[derive(Debug, Clone)]
pub enum Action {
    /// Mint complete sets across both L1 contracts.
    /// Contract 1 mint produces "other repos" token used as collateral for contract 2 mint.
    Mint {
        contract_1: &'static str,
        contract_2: &'static str,
        amount: f64,
        target_market: &'static str,
    },
    /// Buy outcome tokens directly from the pool.
    Buy {
        market_name: &'static str,
        amount: f64,
        cost: f64,
    },
    /// Sell outcome tokens back to the pool for sUSD.
    Sell {
        market_name: &'static str,
        amount: f64,
        proceeds: f64,
    },
    /// Merge (burn) complete sets across both L1 contracts to recover sUSD.
    /// Reverse of Mint: buy all other outcomes, merge into complete set, get sUSD back.
    Merge {
        contract_1: &'static str,
        contract_2: &'static str,
        amount: f64,
        source_market: &'static str,
    },
    /// Borrow sUSD via flash loan to fund minting.
    FlashLoan { amount: f64 },
    /// Repay flash loan after selling minted tokens.
    RepayFlashLoan { amount: f64 },
}

/// Mutable pool state for swap simulation during rebalancing.
/// All arithmetic is pure f64 — no U256 in the hot path.
#[derive(Clone)]
struct PoolSim {
    market_name: &'static str,
    market_id: &'static str,
    #[allow(dead_code)]
    outcome_token: &'static str,
    #[allow(dead_code)]
    pool: &'static crate::markets::Pool,
    price: f64,            // current outcome price (mutable)
    buy_limit_price: f64,  // max outcome price reachable via buying (tick boundary)
    sell_limit_price: f64, // min outcome price reachable via selling (tick boundary)
    liquidity: u128,
    l_eff: f64, // precomputed L / (1e18 × (1-fee))
    prediction: f64,
}

impl PoolSim {
    fn from_slot0(
        slot0: &Slot0Result,
        market: &'static crate::markets::MarketData,
        prediction: f64,
    ) -> Option<Self> {
        let pool = market.pool.as_ref()?;
        let is_token1_outcome = pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
        let zero_for_one_buy = pool.token0.to_lowercase() == market.quote_token.to_lowercase();
        let liquidity: u128 = pool.liquidity.parse().ok()?;
        if liquidity == 0 {
            return None;
        }
        let t0 = pool.ticks.get(0)?.tick_idx;
        let t1 = pool.ticks.get(1)?.tick_idx;
        let sqrt_lo = get_sqrt_ratio_at_tick(t0.min(t1)).ok()?;
        let sqrt_hi = get_sqrt_ratio_at_tick(t0.max(t1)).ok()?;

        let (sqrt_buy_limit, sqrt_sell_limit) = if zero_for_one_buy {
            (sqrt_lo, sqrt_hi)
        } else {
            (sqrt_hi, sqrt_lo)
        };

        // Convert all U256 prices to f64 once at construction
        let price = u256_to_f64(sqrt_price_x96_to_price_outcome(
            slot0.sqrt_price_x96,
            is_token1_outcome,
        )?);
        let buy_limit_price = u256_to_f64(sqrt_price_x96_to_price_outcome(
            sqrt_buy_limit,
            is_token1_outcome,
        )?);
        let sell_limit_price = u256_to_f64(sqrt_price_x96_to_price_outcome(
            sqrt_sell_limit,
            is_token1_outcome,
        )?);

        Some(PoolSim {
            market_name: market.name,
            market_id: market.market_id,
            outcome_token: market.outcome_token,
            pool,
            price,
            buy_limit_price,
            sell_limit_price,
            liquidity,
            l_eff: (liquidity as f64) / (1e18 * FEE_FACTOR),
            prediction,
        })
    }

    /// Current outcome price.
    fn price(&self) -> f64 {
        self.price
    }

    /// Max tokens sellable before hitting the tick boundary.
    fn max_sell_tokens(&self) -> f64 {
        let k = self.kappa();
        if k <= 0.0 {
            return 0.0;
        }
        let lp = self.sell_limit_price;
        if lp <= 0.0 || lp >= self.price {
            return 0.0;
        }
        ((self.price / lp).sqrt() - 1.0) / k
    }

    /// Price sensitivity for selling: κ = (1-fee) × √price × 1e18 / L.
    fn kappa(&self) -> f64 {
        if self.price <= 0.0 || self.liquidity == 0 {
            return 0.0;
        }
        FEE_FACTOR * self.price.sqrt() * 1e18 / (self.liquidity as f64)
    }

    /// Price sensitivity for buying: λ = √price × 1e18 / L.
    fn lambda(&self) -> f64 {
        if self.price <= 0.0 || self.liquidity == 0 {
            return 0.0;
        }
        self.price.sqrt() * 1e18 / (self.liquidity as f64)
    }

    /// Max tokens buyable before hitting the tick boundary.
    fn max_buy_tokens(&self) -> f64 {
        let lam = self.lambda();
        if lam <= 0.0 {
            return 0.0;
        }
        let blp = self.buy_limit_price;
        if blp <= self.price {
            return 0.0;
        }
        (1.0 - (self.price / blp).sqrt()) / lam
    }

    /// Buy exact amount of outcome tokens.
    /// Returns (tokens_actually_bought, quote_cost, new_price).
    /// Price after buying m: P(m) = P₀ / (1 - mλ)²
    /// Cost: m × P₀ / ((1-fee) × (1 - mλ))
    fn buy_exact(&self, amount: f64) -> Option<(f64, f64, f64)> {
        if amount <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let lam = self.lambda();
        let max_buy = self.max_buy_tokens();
        let actual = amount.min(max_buy);
        if actual <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let d = 1.0 - actual * lam;
        if d <= 0.0 {
            return None;
        }
        let new_price = self.price / (d * d);
        let cost = actual * self.price / (FEE_FACTOR * d);
        Some((actual, cost, new_price))
    }

    /// Effective liquidity: L_eff = L / (1e18 × (1-fee)).
    fn l_eff(&self) -> f64 {
        self.l_eff
    }

    /// Cost to move pool price to target_price via direct buy.
    /// Returns (quote_cost, outcome_received, new_price).
    fn cost_to_price(&self, target_price: f64) -> Option<(f64, f64, f64)> {
        let clamped = target_price.min(self.buy_limit_price);
        if clamped <= self.price {
            return Some((0.0, 0.0, self.price));
        }
        let cost = self.l_eff * (clamped.sqrt() - self.price.sqrt());
        let l_raw = self.liquidity as f64 / 1e18;
        let amount = l_raw * (1.0 / self.price.sqrt() - 1.0 / clamped.sqrt());
        Some((cost, amount, clamped))
    }

    /// Sell outcome tokens until price drops to target.
    /// Returns (tokens_sold, quote_received, new_price).
    fn sell_to_price(&self, target_price: f64) -> Option<(f64, f64, f64)> {
        let clamped = target_price.max(self.sell_limit_price);
        if clamped >= self.price {
            return Some((0.0, 0.0, self.price));
        }
        let k = self.kappa();
        if k <= 0.0 {
            return None;
        }
        let tokens = ((self.price / clamped).sqrt() - 1.0) / k;
        let d = 1.0 + tokens * k;
        let proceeds = self.price * tokens * FEE_FACTOR / d;
        Some((tokens, proceeds, clamped))
    }

    /// Sell exact amount of outcome tokens.
    /// Returns (tokens_actually_sold, quote_received, new_price).
    fn sell_exact(&self, amount: f64) -> Option<(f64, f64, f64)> {
        if amount <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let k = self.kappa();
        let max_sell = self.max_sell_tokens();
        let actual = amount.min(max_sell);
        if actual <= 0.0 {
            return Some((0.0, 0.0, self.price));
        }
        let d = 1.0 + actual * k;
        let new_price = self.price / (d * d);
        let proceeds = self.price * actual * FEE_FACTOR / d;
        Some((actual, proceeds, new_price))
    }
}

/// Build PoolSim entries from slot0 results, matching with predictions.
/// Panics if any outcome has no matching prediction — partial prediction sets are not supported.
fn build_sims(
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<PoolSim> {
    let preds = prediction_map();
    slot0_results
        .iter()
        .filter_map(|(slot0, market)| {
            let key = normalize_market_name(market.name);
            let pred = preds.get(&key).unwrap_or_else(|| {
                panic!(
                    "No prediction for market '{}'. All tradeable outcomes must have predictions.",
                    market.name
                )
            });
            PoolSim::from_slot0(slot0, market, *pred)
        })
        .collect()
}

/// Compute alt price for outcome at `idx`: 1 - sum(other outcome prices).
/// O(1) when price_sum is precomputed.
fn alt_price(sims: &[PoolSim], idx: usize, price_sum: f64) -> f64 {
    1.0 - (price_sum - sims[idx].price)
}

/// Profitability = (prediction - best_price) / best_price
fn profitability(prediction: f64, best_price: f64) -> f64 {
    let effective_price = best_price.max(1e-12);
    (prediction - effective_price) / effective_price
}

fn sanitize_nonnegative_finite(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Target price for a given target profitability: prediction / (1 + target_prof)
fn target_price_for_prof(prediction: f64, target_prof: f64) -> f64 {
    prediction / (1.0 + target_prof)
}

/// Which acquisition route is cheaper for an outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Route {
    Direct,
    Mint,
}

/// For the direct route, compute cost to bring an outcome's profitability to `target_prof`.
/// Returns (cost, outcome_amount, new_price).
fn direct_cost_to_prof(sim: &PoolSim, target_prof: f64) -> Option<(f64, f64, f64)> {
    let tp = target_price_for_prof(sim.prediction, target_prof);
    sim.cost_to_price(tp)
}

/// For the mint route, compute cost to bring an outcome's profitability to `target_prof`.
/// Uses Newton's method to find mint amount where alt price = target price.
/// Returns (cash_cost, value_cost, mint_amount, d_cash_cost_d_pi).
/// cash_cost = actual sUSD spent; value_cost = cash_cost minus expected value of unsold tokens.
fn mint_cost_to_prof(
    sims: &[PoolSim],
    target_idx: usize,
    target_prof: f64,
    skip: &HashSet<usize>,
    price_sum: f64,
) -> Option<(f64, f64, f64, f64)> {
    let tp = target_price_for_prof(sims[target_idx].prediction, target_prof);
    let current_alt = alt_price(sims, target_idx, price_sum);
    if current_alt >= tp {
        return Some((0.0, 0.0, 0.0, 0.0));
    }

    // Newton's method: solve g(m) = rhs where g(m) = Σⱼ pⱼ/(1+m×κⱼ)² (non-skip pools only).
    // Correct rhs accounts for skip pool prices frozen at P⁰_j:
    // rhs = (1 - tp) - Σ_{j∈skip, j≠target} P⁰_j
    let skip_price_sum: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx && skip.contains(i))
        .map(|(_, s)| s.price())
        .sum();
    let rhs = (1.0 - tp) - skip_price_sum;
    if rhs <= 0.0 {
        // Skip pools consume all probability mass — target alt price unreachable via minting.
        return None;
    }

    // Precompute per-pool parameters: (price, kappa, max_sell_tokens, prediction)
    let params: Vec<(f64, f64, f64, f64)> = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx && !skip.contains(i))
        .map(|(_, sim)| {
            (
                sim.price(),
                sim.kappa(),
                sim.max_sell_tokens(),
                sim.prediction,
            )
        })
        .collect();
    if params.is_empty() {
        return None;
    }

    // Reachability guard:
    // g(m) decreases from g0 (m=0) to g_cap (all legs at cap). If rhs is below g_cap,
    // target alt price is unreachable with available liquidity.
    let g0: f64 = params.iter().map(|&(p, _, _, _)| p).sum();
    let g_cap: f64 = params
        .iter()
        .map(|&(p, k, cap, _)| {
            let d = 1.0 + cap * k;
            p / (d * d)
        })
        .sum();
    let g_tol = 1e-12 * (1.0 + g0.abs() + g_cap.abs());
    if rhs < g_cap - g_tol {
        return None;
    }

    // Warm start: first Newton step from m=0 gives m = (g(0) - rhs) / (-g'(0))
    // = (tp - current_alt) / (2 × Σ Pⱼκⱼ), saving one iteration.
    let sum_pk: f64 = params.iter().map(|&(p, k, _, _)| p * k).sum();
    let m_upper: f64 = params.iter().map(|&(_, _, cap, _)| cap).fold(0.0, f64::max);
    let mut m = if sum_pk > 1e-30 {
        ((tp - current_alt) / (2.0 * sum_pk)).max(0.0)
    } else {
        0.0
    };
    if rhs <= g_cap + g_tol {
        m = m_upper;
    } else if m > m_upper {
        m = m_upper;
    }
    for _ in 0..NEWTON_ITERS {
        let mut g = 0.0_f64;
        let mut gp = 0.0_f64;
        for &(p, k, cap, _) in &params {
            let me = m.min(cap);
            let d = 1.0 + me * k;
            let d2 = d * d;
            g += p / d2;
            if m < cap {
                let d3 = d2 * d;
                gp += -2.0 * p * k / d3;
            }
        }
        if gp.abs() < 1e-30 {
            break;
        }
        let step = (g - rhs) / gp;
        m -= step;
        if m < 0.0 {
            m = 0.0;
        } else if m > m_upper {
            m = m_upper;
        }
        if step.abs() < 1e-12 {
            break;
        }
    }

    if m < 1e-18 {
        return Some((0.0, 0.0, 0.0, 0.0));
    }

    // Net cost and analytical derivative in one pass.
    // d(cash_cost)/dπ = d(cash_cost)/dm × dm/dπ
    // dm/dπ = P_target / ((1+π) × g'(m))  [implicit function theorem]
    // d(cash_cost)/dm = 1 - (1-f) × Σⱼ∈uncapped Pⱼ(m)
    let mut sum_marginal = 0.0_f64;
    let mut gp_final = 0.0_f64;
    let mut unsold_value = 0.0_f64;
    let total_proceeds: f64 = params
        .iter()
        .map(|&(p, k, cap, pred)| {
            let me = m.min(cap);
            let d = 1.0 + me * k;
            if m < cap {
                let d2 = d * d;
                let d3 = d2 * d;
                sum_marginal += p / d2;
                gp_final += -2.0 * p * k / d3;
            } else {
                unsold_value += pred * (m - me);
            }
            p * me * FEE_FACTOR / d
        })
        .sum();
    let cash_cost = m - total_proceeds;
    let value_cost = cash_cost - unsold_value;

    let d_cost_d_pi = if gp_final.abs() > 1e-30 {
        let dm_d_pi = tp / ((1.0 + target_prof) * gp_final);
        let d_cost_d_m = 1.0 - FEE_FACTOR * sum_marginal;
        d_cost_d_m * dm_d_pi
    } else {
        0.0
    };

    Some((cash_cost, value_cost, m, d_cost_d_pi))
}

/// Compute cost to reach target_prof for a given outcome via a specific route.
/// Returns (cash_cost, value_cost, amount, Option<new_price_for_direct>, d_cash_cost_d_pi).
/// cash_cost = actual sUSD spent; value_cost = cash_cost minus expected value of unsold tokens.
fn cost_for_route(
    sims: &[PoolSim],
    idx: usize,
    route: Route,
    target_prof: f64,
    skip: &HashSet<usize>,
    price_sum: f64,
) -> Option<(f64, f64, f64, Option<f64>, f64)> {
    match route {
        Route::Direct => {
            direct_cost_to_prof(&sims[idx], target_prof).map(|(cost, amount, new_price)| {
                let l = sims[idx].l_eff();
                let pred = sims[idx].prediction;
                let t = 1.0 + target_prof;
                let dcost = -l * pred.sqrt() / (2.0 * t * t.sqrt());
                (cost, cost, amount, Some(new_price), dcost)
            })
        }
        Route::Mint => mint_cost_to_prof(sims, idx, target_prof, skip, price_sum)
            .map(|(cash, value, amount, dcost)| (cash, value, amount, None, dcost)),
    }
}

/// Extract deduplicated outcome indices from active (outcome, route) pairs.
fn active_skip_indices(active: &[(usize, Route)]) -> HashSet<usize> {
    active.iter().map(|(idx, _)| *idx).collect()
}

#[derive(Debug, Clone, Copy)]
struct PlannedRoute {
    idx: usize,
    route: Route,
    cost: f64,
    amount: f64,
    new_price: Option<f64>,
}

fn execution_order(active: &[(usize, Route)]) -> Vec<(usize, Route)> {
    let mut ordered: Vec<(usize, Route)> = Vec::with_capacity(active.len());
    ordered.extend(
        active
            .iter()
            .copied()
            .filter(|(_, route)| *route == Route::Mint),
    );
    ordered.extend(
        active
            .iter()
            .copied()
            .filter(|(_, route)| *route == Route::Direct),
    );
    ordered
}

fn plan_active_routes(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    target_prof: f64,
    skip: &HashSet<usize>,
) -> Option<Vec<PlannedRoute>> {
    let mut sim_state = sims.to_vec();
    let mut price_sum: f64 = sim_state.iter().map(|s| s.price).sum();
    let mut plan: Vec<PlannedRoute> = Vec::with_capacity(active.len());

    for (idx, route) in execution_order(active) {
        let (direct_cost, _, amount, new_price, _) =
            cost_for_route(&sim_state, idx, route, target_prof, skip, price_sum)?;

        let (actual_cost, applied_new_price) = match route {
            Route::Direct => {
                let np = new_price?;
                sim_state[idx].price = np;
                (direct_cost, Some(np))
            }
            Route::Mint => {
                let mut proceeds = 0.0_f64;
                if amount > 0.0 {
                    for i in 0..sim_state.len() {
                        if i == idx || skip.contains(&i) {
                            continue;
                        }
                        if let Some((sold, leg_proceeds, new_leg_price)) =
                            sim_state[i].sell_exact(amount)
                        {
                            if sold > 0.0 {
                                proceeds += leg_proceeds;
                                sim_state[i].price = new_leg_price;
                            }
                        }
                    }
                }
                (amount - proceeds, None)
            }
        };

        if !actual_cost.is_finite() {
            return None;
        }
        plan.push(PlannedRoute {
            idx,
            route,
            cost: actual_cost,
            amount,
            new_price: applied_new_price,
        });

        price_sum = sim_state.iter().map(|s| s.price).sum();
    }

    Some(plan)
}

fn plan_is_budget_feasible(plan: &[PlannedRoute], budget: f64) -> bool {
    let mut running_budget = budget;
    for step in plan {
        if step.cost > running_budget + 1e-12 {
            return false;
        }
        running_budget -= step.cost;
        if !running_budget.is_finite() {
            return false;
        }
    }
    true
}

fn execute_planned_routes(
    sims: &mut [PoolSim],
    plan: &[PlannedRoute],
    budget: &mut f64,
    actions: &mut Vec<Action>,
    skip: &HashSet<usize>,
) -> bool {
    for step in plan {
        if step.cost > *budget + 1e-12 {
            return false;
        }
        execute_buy(
            sims,
            step.idx,
            step.cost,
            step.amount,
            step.route,
            step.new_price,
            budget,
            actions,
            skip,
        );
    }
    true
}

fn action_contract_pair(sims: &[PoolSim]) -> (&'static str, &'static str) {
    if sims.is_empty() {
        return ("", "");
    }
    let mut contracts: Vec<&'static str> = sims.iter().map(|s| s.market_id).collect();
    contracts.sort();
    contracts.dedup();
    let c1 = contracts[0];
    let c2 = if contracts.len() > 1 {
        contracts[1]
    } else {
        contracts[0]
    };
    (c1, c2)
}

fn apply_actions_to_sim_balances(
    actions: &[Action],
    sims: &[PoolSim],
    sim_balances: &mut HashMap<&str, f64>,
) {
    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                *sim_balances.entry(market_name).or_insert(0.0) += amount;
            }
            Action::Mint { amount, .. } => {
                for sim in sims.iter() {
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
                for sim in sims.iter() {
                    *sim_balances.entry(sim.market_name).or_insert(0.0) -= amount;
                }
            }
            Action::FlashLoan { .. } | Action::RepayFlashLoan { .. } => {}
        }
    }
}

fn complete_set_arb_cap(sims: &[PoolSim]) -> f64 {
    if sims.is_empty() {
        return 0.0;
    }
    sims.iter()
        .map(|s| s.max_buy_tokens())
        .fold(f64::INFINITY, f64::min)
}

fn complete_set_marginal_buy_cost(sims: &[PoolSim], amount: f64) -> f64 {
    let mut total = 0.0_f64;
    for s in sims {
        let lam = s.lambda();
        if lam <= 0.0 || s.price <= 0.0 {
            return f64::INFINITY;
        }
        let d = 1.0 - amount * lam;
        if d <= 0.0 {
            return f64::INFINITY;
        }
        total += s.price / (FEE_FACTOR * d * d);
    }
    total
}

fn solve_complete_set_arb_amount(sims: &[PoolSim]) -> f64 {
    let cap = complete_set_arb_cap(sims);
    if cap <= 1e-18 {
        return 0.0;
    }

    let d0 = complete_set_marginal_buy_cost(sims, 0.0);
    if !d0.is_finite() || d0 >= 1.0 {
        return 0.0;
    }

    let cap_left = (cap - 1e-12 * (1.0 + cap)).max(0.0);
    let d_cap = complete_set_marginal_buy_cost(sims, cap_left);
    if d_cap <= 1.0 {
        return cap;
    }

    let mut lo = 0.0_f64;
    let mut hi = cap_left;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let d_mid = complete_set_marginal_buy_cost(sims, mid);
        if d_mid > 1.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo).abs() <= 1e-12 * (1.0 + hi.abs()) {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn execute_complete_set_arb(
    sims: &mut [PoolSim],
    actions: &mut Vec<Action>,
    budget: &mut f64,
) -> f64 {
    let amount = solve_complete_set_arb_amount(sims);
    if amount <= 1e-18 {
        return 0.0;
    }

    let mut legs: Vec<(usize, f64, f64)> = Vec::with_capacity(sims.len());
    let mut total_buy_cost = 0.0_f64;
    for (i, s) in sims.iter().enumerate() {
        match s.buy_exact(amount) {
            Some((bought, cost, new_price))
                if bought + 1e-12 >= amount && bought > 0.0 && cost.is_finite() =>
            {
                legs.push((i, cost, new_price));
                total_buy_cost += cost;
            }
            _ => return 0.0,
        }
    }

    let profit = amount - total_buy_cost;
    if !profit.is_finite() || profit <= 1e-12 {
        return 0.0;
    }

    if total_buy_cost > 1e-18 {
        actions.push(Action::FlashLoan {
            amount: total_buy_cost,
        });
    }

    for (i, cost, new_price) in legs {
        sims[i].price = new_price;
        actions.push(Action::Buy {
            market_name: sims[i].market_name,
            amount,
            cost,
        });
    }

    let (contract_1, contract_2) = action_contract_pair(sims);
    actions.push(Action::Merge {
        contract_1,
        contract_2,
        amount,
        source_market: "complete_set_arb",
    });

    if total_buy_cost > 1e-18 {
        actions.push(Action::RepayFlashLoan {
            amount: total_buy_cost,
        });
    }

    *budget += profit;
    profit
}

fn lookup_balance(balances: &HashMap<&str, f64>, market_name: &str) -> f64 {
    sanitize_nonnegative_finite(balances.get(market_name).copied().unwrap_or(0.0))
}

fn portfolio_expected_value(
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
                sim_balances
                    .get(sim.market_name)
                    .copied()
                    .unwrap_or(0.0),
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
fn emit_mint_actions(
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
    for i in 0..sims.len() {
        if i == target_idx || skip.contains(&i) {
            continue;
        }
        if let Some((sold, proceeds, new_price)) = sims[i].sell_exact(amount) {
            if sold > 0.0 {
                total_proceeds += proceeds;
                sims[i].price = new_price;
                actions.push(Action::Sell {
                    market_name: sims[i].market_name,
                    amount: sold,
                    proceeds,
                });
            }
        }
    }
    total_proceeds
}

/// Compute merge sell proceeds without modifying state (dry run).
/// Merge route: buy all other outcomes, merge complete sets, get sUSD back.
/// Returns (net_proceeds, actual_merge_amount).
#[cfg(test)]
fn merge_sell_proceeds(sims: &[PoolSim], source_idx: usize, amount: f64) -> (f64, f64) {
    merge_sell_proceeds_with_inventory(sims, source_idx, amount, None, f64::INFINITY)
}

/// Compute merge sell proceeds (dry run), optionally consuming existing complementary holdings
/// before buying shortfall from pools.
fn merge_sell_proceeds_with_inventory(
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
        if shortfall <= 1e-18 {
            continue;
        }
        match s.buy_exact(shortfall) {
            Some((bought, cost, _))
                if bought + 1e-12 >= shortfall && bought > 0.0 && cost.is_finite() =>
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
fn direct_sell_marginal_proceeds(sim: &PoolSim, amount_sold: f64) -> f64 {
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

fn merge_sell_marginal_proceeds_with_inventory(
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
        if merge_amount <= held + 1e-18 {
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
fn merge_sell_cap(sims: &[PoolSim], source_idx: usize) -> f64 {
    merge_sell_cap_with_inventory(sims, source_idx, None, f64::INFINITY)
}

fn merge_sell_cap_with_inventory(
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
fn split_sell_total_proceeds(
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

fn split_sell_total_proceeds_with_inventory(
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
fn optimal_sell_split(sims: &[PoolSim], source_idx: usize, sell_amount: f64) -> (f64, f64) {
    optimal_sell_split_with_inventory(sims, source_idx, sell_amount, None, f64::INFINITY)
}

fn optimal_sell_split_with_inventory(
    sims: &[PoolSim],
    source_idx: usize,
    sell_amount: f64,
    sim_balances: Option<&HashMap<&str, f64>>,
    inventory_keep_prof: f64,
) -> (f64, f64) {
    const OPT_ITERS: usize = 48;
    const OPT_EPS: f64 = 1e-12;

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

/// Execute a fixed-amount sell using the optimal merge/direct split.
/// Returns the actual source tokens sold.
fn execute_optimal_sell(
    sims: &mut [PoolSim],
    source_idx: usize,
    sell_amount: f64,
    sim_balances: &mut HashMap<&str, f64>,
    inventory_keep_prof: f64,
    mint_available: bool,
    actions: &mut Vec<Action>,
    budget: &mut f64,
) -> f64 {
    if sell_amount <= 0.0 {
        return 0.0;
    }

    let merge_target = if mint_available {
        let (m_opt, _) = optimal_sell_split_with_inventory(
            sims,
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
    if merge_target > 1e-18 {
        let merged = execute_merge_sell_with_inventory(
            sims,
            source_idx,
            merge_target,
            sim_balances,
            inventory_keep_prof,
            actions,
            budget,
        );
        sold_total += merged;
    }

    let remainder = (sell_amount - sold_total).max(0.0);
    if remainder > 1e-18 {
        if let Some((sold, proceeds, new_price)) = sims[source_idx].sell_exact(remainder) {
            if sold > 0.0 {
                sims[source_idx].price = new_price;
                *budget += proceeds;
                sold_total += sold;
                subtract_balance(sim_balances, sims[source_idx].market_name, sold);
                actions.push(Action::Sell {
                    market_name: sims[source_idx].market_name,
                    amount: sold,
                    proceeds,
                });
            }
        }
    }

    sold_total
}

fn held_balance(sim_balances: Option<&HashMap<&str, f64>>, market_name: &str) -> f64 {
    sanitize_nonnegative_finite(
        sim_balances
            .and_then(|b| b.get(market_name).copied())
            .unwrap_or(0.0),
    )
}

/// Inventory used in merge legs is gated by profitability: keep holdings with
/// profitability above `inventory_keep_prof` and only consume lower-profitability balances.
fn merge_usable_inventory(
    sim_balances: Option<&HashMap<&str, f64>>,
    sim: &PoolSim,
    inventory_keep_prof: f64,
) -> f64 {
    if profitability(sim.prediction, sim.price()) > inventory_keep_prof {
        return 0.0;
    }
    held_balance(sim_balances, sim.market_name)
}

fn subtract_balance(sim_balances: &mut HashMap<&str, f64>, market_name: &'static str, amount: f64) {
    if amount <= 0.0 {
        return;
    }
    let bal = sim_balances.entry(market_name).or_insert(0.0);
    *bal -= amount;
    if *bal > -1e-12 && *bal < 0.0 {
        *bal = 0.0;
    }
}

/// Execute a merge sell while consuming existing complementary holdings first.
/// Missing shortfall is bought from pools. Updates `sim_balances`.
fn execute_merge_sell_with_inventory(
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
        if buy_amount <= 1e-18 {
            legs.push((i, 0.0, 0.0, s.price, consumed_from_inventory));
            continue;
        }
        match s.buy_exact(buy_amount) {
            Some((bought, cost, new_price))
                if bought + 1e-12 >= buy_amount && bought > 0.0 && cost.is_finite() =>
            {
                legs.push((i, bought, cost, new_price, consumed_from_inventory));
                total_buy_cost += cost;
            }
            _ => return 0.0,
        }
    }

    if total_buy_cost > 1e-18 {
        actions.push(Action::FlashLoan {
            amount: total_buy_cost,
        });
    }

    for (i, bought, cost, new_price, _) in &legs {
        if *bought > 1e-18 {
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

    if total_buy_cost > 1e-18 {
        actions.push(Action::RepayFlashLoan {
            amount: total_buy_cost,
        });
    }

    for (i, _, _, _, consumed_from_inventory) in &legs {
        if *consumed_from_inventory > 1e-18 {
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
fn execute_merge_sell(
    sims: &mut [PoolSim],
    source_idx: usize,
    amount: f64,
    actions: &mut Vec<Action>,
    budget: &mut f64,
) -> f64 {
    let merge_cap = merge_sell_cap(sims, source_idx);
    let actual = amount.min(merge_cap);
    if actual <= 0.0 {
        return 0.0;
    }

    // Pre-compute all legs and enforce all-or-nothing semantics:
    // if any leg cannot execute, abort merge entirely.
    let mut legs: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(sims.len().saturating_sub(1));
    let mut total_buy_cost = 0.0_f64;
    for (i, s) in sims.iter().enumerate() {
        if i == source_idx {
            continue;
        }
        match s.buy_exact(actual) {
            Some((bought, cost, new_price)) if bought > 0.0 && cost.is_finite() => {
                legs.push((i, bought, cost, new_price));
                total_buy_cost += cost;
            }
            _ => return 0.0,
        }
    }

    actions.push(Action::FlashLoan {
        amount: total_buy_cost,
    });

    // Apply all validated legs.
    for (i, bought, cost, new_price) in legs {
        sims[i].price = new_price;
        actions.push(Action::Buy {
            market_name: sims[i].market_name,
            amount: bought,
            cost,
        });
    }

    // Merge complete sets
    let (contract_1, contract_2) = action_contract_pair(sims);
    actions.push(Action::Merge {
        contract_1,
        contract_2,
        amount: actual,
        source_market: sims[source_idx].market_name,
    });

    actions.push(Action::RepayFlashLoan {
        amount: total_buy_cost,
    });

    // Net proceeds: merge returns `actual` sUSD, minus buy costs
    *budget += actual - total_buy_cost;
    actual
}

/// Computes optimal rebalancing trades for L1 markets.
///
/// 1. Sell overpriced holdings (price > prediction) via swap simulation
/// 2. Waterfall allocation: deploy capital to highest profitability outcomes,
///    equalizing profitability progressively
/// 3. Post-allocation liquidation: sell held outcomes less profitable than
///    the last bought outcome, reallocate via waterfall
pub fn rebalance(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> Vec<Action> {
    let mut actions: Vec<Action> = Vec::new();
    if !susds_balance.is_finite() {
        return actions;
    }
    let mut budget = susds_balance;
    let mut sims = build_sims(slot0_results);

    if sims.is_empty() {
        return actions;
    }

    // Mint/merge routes require all tradeable outcomes to have liquid pools.
    // sims may be smaller if pools have zero liquidity or slot0_results is partial (RPC failures).
    let mint_available = sims.len() == crate::predictions::PREDICTIONS_L1.len();

    // Track holdings changes during simulation.
    // `legacy_remaining` tracks inventory that existed before this rebalance call.
    let mut sim_balances: HashMap<&str, f64> = HashMap::new();
    let mut legacy_remaining: HashMap<&str, f64> = HashMap::new();
    for sim in &sims {
        let held = sanitize_nonnegative_finite(lookup_balance(balances, sim.market_name));
        sim_balances.insert(sim.market_name, held);
        legacy_remaining.insert(sim.market_name, held);
    }

    // ── Phase 0: Complete-set arbitrage (buy all outcomes, merge) ──
    // Execute before any discretionary rebalancing so free budget is harvested first.
    if mint_available {
        let _arb_profit = execute_complete_set_arb(&mut sims, &mut actions, &mut budget);
    }

    // ── Phase 1: Sell overpriced holdings ──
    const MAX_PHASE1_ITERS: usize = 128;
    const PHASE1_EPS: f64 = 1e-12;
    for i in 0..sims.len() {
        for _ in 0..MAX_PHASE1_ITERS {
            let price = sims[i].price();
            let pred = sims[i].prediction;
            if price <= pred + PHASE1_EPS {
                break;
            }

            let held = *sim_balances.get(sims[i].market_name).unwrap_or(&0.0);
            if held <= PHASE1_EPS {
                break;
            }

            // Target the direct sell amount needed to bring price to prediction.
            let (tokens_needed, _, _) = sims[i]
                .sell_to_price(pred)
                .unwrap_or((0.0, 0.0, sims[i].price));
            let sell_amount = if tokens_needed > PHASE1_EPS {
                tokens_needed.min(held)
            } else {
                held
            };
            if sell_amount <= PHASE1_EPS {
                break;
            }

            let sold_total = execute_optimal_sell(
                &mut sims,
                i,
                sell_amount,
                &mut sim_balances,
                0.0,
                mint_available,
                &mut actions,
                &mut budget,
            );
            if sold_total <= PHASE1_EPS {
                break;
            }

            let new_price = sims[i].price();
            let new_held = *sim_balances.get(sims[i].market_name).unwrap_or(&0.0);
            // Merge-heavy splits may not move the source pool price. If that happens while
            // still overpriced, force one full inventory attempt to avoid leaving residual
            // overpriced holdings due one-shot sizing.
            if new_price >= price - PHASE1_EPS
                && new_price > pred + PHASE1_EPS
                && new_held + PHASE1_EPS < held
                && new_held > PHASE1_EPS
            {
                let sold_remaining = execute_optimal_sell(
                    &mut sims,
                    i,
                    new_held,
                    &mut sim_balances,
                    0.0,
                    mint_available,
                    &mut actions,
                    &mut budget,
                );
                if sold_remaining <= PHASE1_EPS {
                    break;
                }
            }
        }
    }

    // Legacy inventory available for phase-3 recycling cannot exceed current holdings after phase 1.
    for sim in &sims {
        let current = *sim_balances.get(sim.market_name).unwrap_or(&0.0);
        let legacy = legacy_remaining.entry(sim.market_name).or_insert(0.0);
        *legacy = (*legacy).min(current.max(0.0));
        if *legacy < 1e-12 {
            *legacy = 0.0;
        }
    }
    let has_legacy_holdings = legacy_remaining.values().any(|&v| v > 1e-12);

    // ── Phase 2: Waterfall allocation ──
    let actions_before = actions.len();
    let last_bought_prof = waterfall(&mut sims, &mut budget, &mut actions, mint_available);

    // Update simulated holdings from the initial waterfall pass.
    apply_actions_to_sim_balances(&actions[actions_before..], &sims, &mut sim_balances);

    // ── Phase 3: Post-allocation liquidation ──
    // Iterate liquidation/reallocation until convergence, but recycle legacy inventory only.
    if has_legacy_holdings {
        const MAX_PHASE3_ITERS: usize = 8;
        const PHASE3_PROF_REL_TOL: f64 = 1e-9;
        const PHASE3_EV_GUARD_REL_TOL: f64 = 1e-10;
        let mut phase3_prof = last_bought_prof;
        for _ in 0..MAX_PHASE3_ITERS {
            if phase3_prof <= 0.0 {
                break;
            }
            if !legacy_remaining.values().any(|&v| v > 1e-12) {
                break;
            }

            // Collect legacy-held outcomes with profitability below the current threshold.
            let mut liquidation_candidates: Vec<(usize, f64)> = Vec::new();
            for (i, sim) in sims.iter().enumerate() {
                let held_total = *sim_balances.get(sim.market_name).unwrap_or(&0.0);
                let held_legacy = legacy_remaining
                    .get(sim.market_name)
                    .copied()
                    .unwrap_or(0.0)
                    .min(held_total)
                    .max(0.0);
                if held_legacy <= 0.0 {
                    continue;
                }
                let prof = profitability(sim.prediction, sim.price());
                if prof < phase3_prof {
                    liquidation_candidates.push((i, prof));
                }
            }
            if liquidation_candidates.is_empty() {
                break;
            }

            liquidation_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

            let ev_before_iter = portfolio_expected_value(&sims, &sim_balances, budget);
            if !ev_before_iter.is_finite() {
                break;
            }

            let mut trial_sims = sims.clone();
            let mut trial_balances = sim_balances.clone();
            let mut trial_legacy = legacy_remaining.clone();
            let mut trial_budget = budget;
            let mut trial_actions: Vec<Action> = Vec::new();

            let budget_before_liq = trial_budget;
            for (idx, _) in liquidation_candidates {
                let held_total = *trial_balances
                    .get(trial_sims[idx].market_name)
                    .unwrap_or(&0.0);
                let held_legacy = trial_legacy
                    .get(trial_sims[idx].market_name)
                    .copied()
                    .unwrap_or(0.0)
                    .min(held_total)
                    .max(0.0);
                if held_legacy <= 0.0 {
                    continue;
                }
                let target_price = target_price_for_prof(trial_sims[idx].prediction, phase3_prof);
                let (tokens_needed, _, _) =
                    trial_sims[idx]
                        .sell_to_price(target_price)
                        .unwrap_or((0.0, 0.0, trial_sims[idx].price));
                let sell_target = tokens_needed.min(held_legacy);
                if sell_target <= 0.0 {
                    continue;
                }

                let sold_total = execute_optimal_sell(
                    &mut trial_sims,
                    idx,
                    sell_target,
                    &mut trial_balances,
                    phase3_prof,
                    mint_available,
                    &mut trial_actions,
                    &mut trial_budget,
                );
                if sold_total > 0.0 {
                    let legacy = trial_legacy
                        .entry(trial_sims[idx].market_name)
                        .or_insert(0.0);
                    *legacy = (*legacy - sold_total).max(0.0);
                }
            }

            let recovered_budget = trial_budget - budget_before_liq;
            if trial_actions.is_empty() || recovered_budget <= 1e-12 {
                break;
            }
            if trial_budget <= 1e-12 {
                break;
            }

            // Reallocate recovered capital and fold the acquired positions into simulated balances.
            let actions_before_realloc = trial_actions.len();
            let new_prof = waterfall(
                &mut trial_sims,
                &mut trial_budget,
                &mut trial_actions,
                mint_available,
            );
            apply_actions_to_sim_balances(
                &trial_actions[actions_before_realloc..],
                &trial_sims,
                &mut trial_balances,
            );
            if trial_actions.len() == actions_before_realloc || new_prof <= 0.0 {
                break;
            }

            let ev_after_iter = portfolio_expected_value(&trial_sims, &trial_balances, trial_budget);
            if !ev_after_iter.is_finite() {
                break;
            }
            let ev_tol = PHASE3_EV_GUARD_REL_TOL * (1.0 + ev_before_iter.abs() + ev_after_iter.abs());
            if ev_after_iter + ev_tol < ev_before_iter {
                break;
            }

            actions.extend(trial_actions);
            sims = trial_sims;
            sim_balances = trial_balances;
            legacy_remaining = trial_legacy;
            budget = trial_budget;

            let prof_delta = (new_prof - phase3_prof).abs();
            if prof_delta <= PHASE3_PROF_REL_TOL * (1.0 + phase3_prof.abs()) {
                break;
            }
            phase3_prof = new_prof;
        }
    }

    actions
}

/// Find the highest-profitability (outcome, route) pair not already in the active set.
/// Scans current pool state each call, so mint perturbations are reflected immediately.
fn best_non_active(
    sims: &[PoolSim],
    active_set: &HashSet<(usize, Route)>,
    mint_available: bool,
    price_sum: f64,
) -> Option<(usize, Route, f64)> {
    let mut best: Option<(usize, Route, f64)> = None;
    for (i, sim) in sims.iter().enumerate() {
        if !active_set.contains(&(i, Route::Direct)) {
            let prof = profitability(sim.prediction, sim.price);
            if prof > 0.0 && best.map_or(true, |b| prof > b.2) {
                best = Some((i, Route::Direct, prof));
            }
        }
        if mint_available && !active_set.contains(&(i, Route::Mint)) {
            let mp = alt_price(sims, i, price_sum);
            let prof = profitability(sim.prediction, mp);
            if prof > 0.0 && best.map_or(true, |b| prof > b.2) {
                best = Some((i, Route::Mint, prof));
            }
        }
    }
    best
}

const MAX_WATERFALL_ITERS: usize = 1000;

/// Waterfall allocation: deploy capital to the highest profitability outcome.
/// As capital is deployed, profitability drops until it matches the next outcome.
/// Then deploy to both, then three, etc.
///
/// Returns the profitability level of the last bought outcome (for post-liquidation).
fn waterfall(
    sims: &mut [PoolSim],
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
) -> f64 {
    if *budget <= 0.0 {
        return 0.0;
    }

    // Precompute price_sum; maintained incrementally after executions.
    let mut price_sum: f64 = sims.iter().map(|s| s.price).sum();
    let mut active_set: HashSet<(usize, Route)> = HashSet::new();

    // Seed active set with the highest-profitability entry.
    let first = match best_non_active(sims, &active_set, mint_available, price_sum) {
        Some(entry) if entry.2 > 0.0 => entry,
        _ => return 0.0,
    };

    let mut active: Vec<(usize, Route)> = vec![(first.0, first.1)];
    active_set.insert((first.0, first.1));
    let mut current_prof = first.2;
    let mut last_prof = 0.0;

    for _iter in 0..MAX_WATERFALL_ITERS {
        if *budget <= 1e-12 || current_prof <= 0.0 {
            break;
        }

        // Dynamically find the next best entry from current pool state.
        // If a mint perturbed prices and pushed an entry above current_prof,
        // absorb it into active immediately (no cost step needed).
        loop {
            match best_non_active(sims, &active_set, mint_available, price_sum) {
                Some((idx, route, prof)) if prof > current_prof => {
                    active.push((idx, route));
                    active_set.insert((idx, route));
                }
                _ => break,
            }
        }

        let next = best_non_active(sims, &active_set, mint_available, price_sum);
        let target_prof = match next {
            Some((_, _, p)) if p > 0.0 => p,
            _ => 0.0,
        };

        // Prune entries that can't reach target_prof (e.g. tick boundary hit).
        // Re-derive skip after each removal so remaining entries see the correct set.
        loop {
            let skip = active_skip_indices(&active);
            let before = active.len();
            active.retain(|&(idx, route)| {
                cost_for_route(sims, idx, route, target_prof, &skip, price_sum).is_some()
            });
            if active.len() == before || active.is_empty() {
                break;
            }
            active_set = active.iter().copied().collect();
        }
        if active.is_empty() {
            break;
        }

        let skip = active_skip_indices(&active);
        let full_plan = match plan_active_routes(sims, &active, target_prof, &skip) {
            Some(plan) => plan,
            None => {
                last_prof = current_prof;
                break;
            }
        };

        if plan_is_budget_feasible(&full_plan, *budget) {
            if !execute_planned_routes(sims, &full_plan, budget, actions, &skip) {
                last_prof = current_prof;
                break;
            }

            // Refresh price_sum after executions
            price_sum = sims.iter().map(|s| s.price).sum();
            current_prof = target_prof;
            last_prof = target_prof;

            // Re-query best entry from post-execution state
            match best_non_active(sims, &active_set, mint_available, price_sum) {
                Some((idx, route, prof)) if prof > 0.0 => {
                    active.push((idx, route));
                    active_set.insert((idx, route));
                }
                _ => break,
            }
        } else {
            // Can't afford full step. Solve for lowest feasible profitability in [target_prof, current_prof].
            let mut achievable =
                solve_prof(sims, &active, current_prof, target_prof, *budget, &skip);
            let mut execution_plan = plan_active_routes(sims, &active, achievable, &skip);

            // Numerical guard: tighten toward current_prof until feasible.
            if execution_plan
                .as_ref()
                .map(|p| !plan_is_budget_feasible(p, *budget))
                .unwrap_or(true)
            {
                let mut lo = achievable;
                let mut hi = current_prof;
                let mut best: Option<(f64, Vec<PlannedRoute>)> = None;
                for _ in 0..32 {
                    let mid = 0.5 * (lo + hi);
                    if let Some(plan) = plan_active_routes(sims, &active, mid, &skip) {
                        if plan_is_budget_feasible(&plan, *budget) {
                            best = Some((mid, plan));
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    } else {
                        lo = mid;
                    }
                    if (hi - lo).abs() <= 1e-12 * (1.0 + hi.abs()) {
                        break;
                    }
                }
                if let Some((best_prof, plan)) = best {
                    achievable = best_prof;
                    execution_plan = Some(plan);
                }
            }

            let Some(execution_plan) = execution_plan else {
                last_prof = current_prof;
                break;
            };
            if !plan_is_budget_feasible(&execution_plan, *budget) {
                last_prof = current_prof;
                break;
            }
            if !execute_planned_routes(sims, &execution_plan, budget, actions, &skip) {
                last_prof = current_prof;
                break;
            }
            last_prof = achievable;
            break;
        }
    }

    last_prof
}

/// Execute a buy via the chosen route, updating state.
fn execute_buy(
    sims: &mut [PoolSim],
    idx: usize,
    cost: f64,
    amount: f64,
    route: Route,
    new_price: Option<f64>,
    budget: &mut f64,
    actions: &mut Vec<Action>,
    skip: &HashSet<usize>,
) {
    if amount <= 0.0 {
        return;
    }
    match route {
        Route::Direct => {
            if let Some(np) = new_price {
                sims[idx].price = np;
            }
            *budget -= cost;
            actions.push(Action::Buy {
                market_name: sims[idx].market_name,
                amount,
                cost,
            });
        }
        Route::Mint => {
            // Flash loan funds the mint upfront; net cost comes from budget
            let upfront = amount; // 1 sUSD per set
            actions.push(Action::FlashLoan { amount: upfront });
            let proceeds = emit_mint_actions(sims, idx, amount, actions, skip);
            actions.push(Action::RepayFlashLoan { amount: upfront });
            let net_cost = upfront - proceeds;
            *budget -= net_cost;
        }
    }
}

/// Find the lowest profitability level affordable with the available budget.
/// Uses closed-form for all-direct, and simulation-backed bisection for mixed routes.
fn solve_prof(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    prof_hi: f64,
    prof_lo: f64,
    budget: f64,
    skip: &HashSet<usize>,
) -> f64 {
    let all_direct = active.iter().all(|&(_, route)| route == Route::Direct);

    if all_direct {
        // Closed form: π = (A/B)² - 1
        let mut a_sum = 0.0_f64;
        let mut b_sum = 0.0_f64;
        for &(idx, _) in active {
            let l = sims[idx].l_eff();
            let p = sims[idx].price();
            a_sum += l * sims[idx].prediction.sqrt();
            b_sum += l * p.sqrt();
        }
        let b = budget + b_sum;
        if b <= 0.0 {
            return prof_hi;
        }
        let ratio = a_sum / b;
        let prof = ratio * ratio - 1.0;
        return prof.clamp(prof_lo, prof_hi);
    }

    let affordable = |prof: f64| -> bool {
        plan_active_routes(sims, active, prof, skip)
            .map(|plan| plan_is_budget_feasible(&plan, budget))
            .unwrap_or(false)
    };

    if affordable(prof_lo) {
        return prof_lo;
    }
    if !affordable(prof_hi) {
        return prof_hi;
    }

    let mut lo = prof_lo;
    let mut hi = prof_hi;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if affordable(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo).abs() <= 1e-12 * (1.0 + hi.abs()) {
            break;
        }
    }

    hi.clamp(prof_lo, prof_hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markets::{MarketData, Pool, Tick};
    use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};
    use alloy::primitives::{Address, U256};
    use proptest::prelude::*;

    /// Mock pool: token0=quote(sUSD), token1=outcome → is_token1_outcome=true.
    /// Uses positive tick range matching real L1 pools with this ordering.
    /// Price range: ~[0.0001, 0.2] for outcome token.
    fn mock_pool() -> Pool {
        Pool {
            token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0", // quote (sUSD)
            token1: "0x1111111111111111111111111111111111111111", // outcome
            pool_id: "0x0000000000000000000000000000000000000001",
            liquidity: "1000000000000000000000", // 1000 tokens
            ticks: &[
                Tick {
                    tick_idx: 16095,
                    liquidity_net: 1000000000000000000000,
                },
                Tick {
                    tick_idx: 92108,
                    liquidity_net: -1000000000000000000000,
                },
            ],
        }
    }

    /// Leak a pool into a static reference for tests
    fn leak_pool(pool: Pool) -> &'static Pool {
        Box::leak(Box::new(pool))
    }

    fn leak_market(market: MarketData) -> &'static MarketData {
        Box::leak(Box::new(market))
    }

    /// Create slot0 + market data pair for testing.
    /// `price_fraction` sets the outcome price approximately (0.0 to 1.0).
    fn mock_slot0_market(
        name: &'static str,
        outcome_token: &'static str,
        price_fraction: f64,
    ) -> (Slot0Result, &'static MarketData) {
        let mut p = mock_pool();
        p.token1 = outcome_token;
        let pool = leak_pool(p);

        // token0 is quote (sUSD), token1 is outcome
        // is_token1_outcome = true
        // outcome price = inv_price = 2^192 / sqrtPriceX96^2
        // To get outcome price = price_fraction:
        // sqrtPriceX96 = sqrt(2^192 / (price_fraction * 1e18)) * sqrt(1e18)
        // Simpler: use prediction_to_sqrt_price_x96
        let sqrt_price =
            prediction_to_sqrt_price_x96(price_fraction, true).unwrap_or(U256::from(1u128 << 96));

        let market = leak_market(MarketData {
            name,
            market_id: crate::markets::MARKETS_L1[0].market_id,
            outcome_token,
            pool: Some(*pool),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        });

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
    }

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
        let mut sims =
            vec![PoolSim::from_slot0(&slot0_results[0].0, slot0_results[0].1, 0.3).unwrap()];

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
        execute_buy(
            &mut sims2,
            0,
            5.0,
            10.0,
            Route::Mint,
            None,
            &mut budget,
            &mut actions2,
            &HashSet::new(),
        );
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

        let profit = execute_complete_set_arb(&mut sims, &mut actions, &mut budget);
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

        let profit = execute_complete_set_arb(&mut sims, &mut actions, &mut budget);
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

    fn build_three_sims(prices: [f64; 3]) -> Vec<PoolSim> {
        build_three_sims_with_preds(prices, [0.3, 0.3, 0.3])
    }

    fn build_three_sims_with_preds(prices: [f64; 3], preds: [f64; 3]) -> Vec<PoolSim> {
        let tokens = [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
        ];
        let names = ["M1", "M2", "M3"];
        let slot0_results: Vec<_> = tokens
            .iter()
            .zip(names.iter())
            .zip(prices.iter())
            .map(|((tok, name), price)| mock_slot0_market(name, tok, *price))
            .collect();
        slot0_results
            .iter()
            .zip(preds.iter())
            .map(|((s, m), pred)| PoolSim::from_slot0(s, m, *pred).unwrap())
            .collect()
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
        let upper = merge_sell_cap_with_inventory(
            sims,
            source_idx,
            Some(sim_balances),
            inventory_keep_prof,
        )
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
            let is_token1_outcome =
                pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
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
        let mut holdings: HashMap<&str, f64> = HashMap::new();
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
        let mut sims = build_sims(slot0_results);
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
                                sims[idx].price = new_price;
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
                                sims[idx].price = new_price;
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
                        let p = sims[idx].price.max(1e-12);
                        next.sqrt_price_x96 = prediction_to_sqrt_price_x96(p, is_token1_outcome)
                            .unwrap_or(slot0.sqrt_price_x96);
                    }
                }
                (next, *market)
            })
            .collect()
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
                            state[i].price = new_p;
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
                            state_d[direct_idx].price = new_p;
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

    fn mock_slot0_market_with_liquidity_and_ticks(
        name: &'static str,
        outcome_token: &'static str,
        price_fraction: f64,
        liquidity: u128,
        tick_lo: i32,
        tick_hi: i32,
    ) -> (Slot0Result, &'static MarketData) {
        let mut p = mock_pool();
        p.token1 = outcome_token;

        let liq_str = Box::leak(liquidity.to_string().into_boxed_str());
        p.liquidity = liq_str;
        let liq_i128 = i128::try_from(liquidity).unwrap_or(i128::MAX);
        let lo = tick_lo.min(tick_hi);
        let hi = tick_lo.max(tick_hi);
        let ticks = Box::leak(Box::new([
            Tick {
                tick_idx: lo,
                liquidity_net: liq_i128,
            },
            Tick {
                tick_idx: hi,
                liquidity_net: -liq_i128,
            },
        ]));
        p.ticks = ticks;

        let pool = leak_pool(p);
        let sqrt_price =
            prediction_to_sqrt_price_x96(price_fraction, true).unwrap_or(U256::from(1u128 << 96));

        let market = leak_market(MarketData {
            name,
            market_id: crate::markets::MARKETS_L1[0].market_id,
            outcome_token,
            pool: Some(*pool),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        });

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
    }

    fn mock_slot0_market_with_liquidity(
        name: &'static str,
        outcome_token: &'static str,
        price_fraction: f64,
        liquidity: u128,
    ) -> (Slot0Result, &'static MarketData) {
        mock_slot0_market_with_liquidity_and_ticks(
            name,
            outcome_token,
            price_fraction,
            liquidity,
            16095,
            92108,
        )
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
                Action::Buy { .. } | Action::Sell { .. } | Action::Mint { .. } | Action::Merge { .. } => {
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
                        saturated[i].price = p_new;
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
            let result =
                mint_cost_to_prof(&sims, target_idx, target_prof, &HashSet::new(), price_sum);

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
                        simulated[i].price = p_new;
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

            let sell_amount =
                rng.in_range(0.0, sim_balances.get("M1").copied().unwrap_or(0.0) + 2.5);
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
        assert_eq!(repay, EXPECTED_REPAY, "flash repayment action count drifted");
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
        assert_eq!(merges, EXPECTED_MERGES, "variant-B merge action count drifted");
        assert_eq!(flash, EXPECTED_FLASH, "variant-B flash-loan action count drifted");
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

        let sims = build_sims(&slot0_results);
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

        let sims = build_sims(&slot0_results);
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
            let sims = build_sims(&slot0_results);
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
        let sims = build_sims(&slot0_results);
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
            let sims = build_sims(&slot0_results);
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
        let b_sum: f64 = sims_start.iter().map(|s| s.l_eff() * s.price().sqrt()).sum();
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
        let ok = execute_planned_routes(&mut exec_sims, &plan, &mut budget, &mut actions, &skip);
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
        let sims = build_sims(&slot0_results);
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
            actions
                .iter()
                .all(|a| !matches!(a, Action::Sell { market_name, .. } if *market_name == selected[0].name)),
            "NaN legacy holdings should be sanitized to zero and never sold"
        );
        assert!(
            actions
                .iter()
                .all(|a| !matches!(a, Action::Sell { market_name, .. } if *market_name == selected[1].name)),
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
            actions
                .iter()
                .any(|a| matches!(a, Action::Sell { market_name, .. } if *market_name == selected[0].name)),
            "expected recycling sell from low-profitability legacy holding"
        );
        assert_rebalance_action_invariants(&actions, &slot0_results, &balances, budget);

        let algo_ev = replay_actions_to_ev(&actions, &slot0_results, &balances, budget);
        let sims = build_sims(&slot0_results);
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

        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert(source_name, tokens_needed * 1.5);
        sim_balances.insert(sims[1].market_name, 0.0);
        sim_balances.insert(sims[2].market_name, 0.0);

        let mut actions = Vec::new();
        let mut budget = 0.0;
        let sold = execute_optimal_sell(
            &mut sims,
            source_idx,
            tokens_needed, // Mirrors Phase 1's "sell until direct price reaches prediction" amount.
            &mut sim_balances,
            0.0,
            true,
            &mut actions,
            &mut budget,
        );

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
            actions.iter().any(
                |a| matches!(a, Action::Sell { market_name, .. } if *market_name == source_name)
            ),
            "overpriced legacy source should trigger sells"
        );
        assert!(
            actions.iter().any(|a| matches!(a, Action::Merge { .. })),
            "fixture should exercise merge path in full-L1 mode"
        );

        let (holdings_after, _) = replay_actions_to_state(&actions, &slot0_results, &balances, budget);
        let slot0_after = replay_actions_to_market_state(&actions, &slot0_results);
        let sims_after = build_sims(&slot0_after);
        let source_sim_after = sims_after
            .iter()
            .find(|s| s.market_name == source_name)
            .expect("source market should exist in replayed sims");
        let source_held_after = holdings_after.get(source_name).copied().unwrap_or(0.0).max(0.0);
        let mut legacy_remaining = balances.get(source_name).copied().unwrap_or(0.0).max(0.0);
        for action in &actions {
            match action {
                Action::Sell {
                    market_name, amount, ..
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

            let mut base_balances: HashMap<&str, f64> = HashMap::new();
            base_balances.insert(sims_base[0].market_name, rng.in_range(8.0, 20.0));
            base_balances.insert(sims_base[1].market_name, rng.in_range(8.0, 20.0));
            base_balances.insert(sims_base[2].market_name, rng.in_range(0.0, 2.0));

            let run_phase1 = |order: [usize; 2]| -> f64 {
                let mut sims = sims_base.clone();
                let mut balances = base_balances.clone();

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
                        .unwrap_or((0.0, 0.0, sims[idx].price));
                    let sell_amount = if tokens_needed > 0.0 && tokens_needed <= held {
                        tokens_needed
                    } else {
                        held
                    };
                    if sell_amount <= 0.0 {
                        continue;
                    }
                    let _ = execute_optimal_sell(
                        &mut sims,
                        idx,
                        sell_amount,
                        &mut balances,
                        0.0,
                        true,
                        &mut actions,
                        &mut budget,
                    );
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

        assert!(
            max_gap <= 1e-8,
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
            let ok = execute_planned_routes(
                &mut exec_sims,
                &plan,
                &mut budget,
                &mut actions,
                &skip,
            );
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
            sims[i].sell_limit_price = sims[i].price;
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
        let ok = execute_planned_routes(&mut exec_sims, &plan, &mut budget, &mut actions, &skip);
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

        let (sell_tokens, sell_proceeds, sell_price) =
            sim.sell_to_price(sim.sell_limit_price).unwrap();
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
            assert_rebalance_action_invariants(
                &actions_first,
                &slot0_results,
                &balances,
                susd_balance,
            );
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
            unwind.price = new_price;
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
                unwind.price = unwind_price;
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
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert("WR_M1", sell_amount);
        sim_balances.insert("WR_M2", 0.0);
        sim_balances.insert("WR_M3", 0.0);
        let mut budget = 0.0;
        let mut actions = Vec::new();
        let sold = execute_optimal_sell(
            &mut exec_sims,
            0,
            sell_amount,
            &mut sim_balances,
            f64::INFINITY,
            true,
            &mut actions,
            &mut budget,
        );
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
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert("WD_M1", sell_amount);
        sim_balances.insert("WD_M2", 0.0);
        sim_balances.insert("WD_M3", 0.0);
        let mut budget = 0.0;
        let mut actions = Vec::new();
        let sold = execute_optimal_sell(
            &mut exec_sims,
            0,
            sell_amount,
            &mut sim_balances,
            f64::INFINITY,
            true,
            &mut actions,
            &mut budget,
        );
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
            let ok = execute_planned_routes(
                &mut exec_sims,
                &plan,
                &mut remaining_budget,
                &mut actions,
                &skip,
            );
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
        let ok = execute_planned_routes(&mut exec_sims, &plan, &mut budget, &mut actions, &skip);
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
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert("M1", 5.0);
        sim_balances.insert("M2", 5.0);
        sim_balances.insert("M3", 5.0);

        let mut budget = 0.0;
        let mut actions = Vec::new();
        let sold = execute_optimal_sell(
            &mut sims,
            0,
            5.0,
            &mut sim_balances,
            f64::INFINITY,
            true,
            &mut actions,
            &mut budget,
        );

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
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert("M1", 5.0);
        sim_balances.insert("M2", 2.0);
        sim_balances.insert("M3", 7.0);

        let mut budget = 0.0;
        let mut actions = Vec::new();
        let sold = execute_optimal_sell(
            &mut sims,
            0,
            5.0,
            &mut sim_balances,
            f64::INFINITY,
            true,
            &mut actions,
            &mut budget,
        );

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
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert("M1", 5.0);
        sim_balances.insert("M2", 5.0);
        sim_balances.insert("M3", 5.0);

        let mut budget = 0.0;
        let mut actions = Vec::new();
        let sold = execute_optimal_sell(
            &mut sims,
            0,
            5.0,
            &mut sim_balances,
            0.0, // phase-1 behavior: preserve profitable inventory
            true,
            &mut actions,
            &mut budget,
        );

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
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        sim_balances.insert("M1", 5.0);
        sim_balances.insert("M2", 5.0);
        sim_balances.insert("M3", 5.0);

        let mut budget = 0.0;
        let mut actions = Vec::new();
        let sold = execute_optimal_sell(
            &mut sims,
            0,
            5.0,
            &mut sim_balances,
            0.0, // complements are unprofitable, so inventory is consumable
            true,
            &mut actions,
            &mut budget,
        );

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
}
