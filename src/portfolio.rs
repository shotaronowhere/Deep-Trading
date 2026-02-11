use std::collections::{HashMap, HashSet};

use uniswap_v3_math::tick_math::get_sqrt_ratio_at_tick;

use crate::pools::{
    normalize_market_name, prediction_map, sqrt_price_x96_to_price_outcome, u256_to_f64, FEE_PIPS,
    Slot0Result,
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
        let is_token1_outcome =
            pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
        let zero_for_one_buy =
            pool.token0.to_lowercase() == market.quote_token.to_lowercase();
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
fn build_sims(slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)]) -> Vec<PoolSim> {
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
    if best_price <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (prediction - best_price) / best_price
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
/// Returns (net_cost, mint_amount, d_net_cost_d_pi).
fn mint_cost_to_prof(
    sims: &[PoolSim],
    target_idx: usize,
    target_prof: f64,
    skip: &HashSet<usize>,
    price_sum: f64,
) -> Option<(f64, f64, f64)> {
    let tp = target_price_for_prof(sims[target_idx].prediction, target_prof);
    let current_alt = alt_price(sims, target_idx, price_sum);
    if current_alt >= tp {
        return Some((0.0, 0.0, 0.0));
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

    // Precompute per-pool parameters: (price, kappa, max_sell_tokens)
    let params: Vec<(f64, f64, f64)> = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx && !skip.contains(i))
        .map(|(_, sim)| (sim.price(), sim.kappa(), sim.max_sell_tokens()))
        .collect();

    // Warm start: first Newton step from m=0 gives m = (g(0) - rhs) / (-g'(0))
    // = (tp - current_alt) / (2 × Σ Pⱼκⱼ), saving one iteration.
    let sum_pk: f64 = params.iter().map(|&(p, k, _)| p * k).sum();
    let mut m = if sum_pk > 1e-30 {
        ((tp - current_alt) / (2.0 * sum_pk)).max(0.0)
    } else {
        0.0
    };
    for _ in 0..NEWTON_ITERS {
        let mut g = 0.0_f64;
        let mut gp = 0.0_f64;
        for &(p, k, cap) in &params {
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
        }
        if step.abs() < 1e-12 {
            break;
        }
    }

    if m < 1e-18 {
        return None;
    }

    // Net cost and analytical derivative in one pass.
    // d(net_cost)/dπ = d(net_cost)/dm × dm/dπ
    // dm/dπ = P_target / ((1+π) × g'(m))  [implicit function theorem]
    // d(net_cost)/dm = 1 - (1-f) × Σⱼ∈uncapped Pⱼ(m)
    let mut sum_marginal = 0.0_f64;
    let mut gp_final = 0.0_f64;
    let total_proceeds: f64 = params
        .iter()
        .map(|&(p, k, cap)| {
            let me = m.min(cap);
            let d = 1.0 + me * k;
            if m < cap {
                let d2 = d * d;
                let d3 = d2 * d;
                sum_marginal += p / d2;
                gp_final += -2.0 * p * k / d3;
            }
            p * me * FEE_FACTOR / d
        })
        .sum();
    let net_cost = m - total_proceeds;

    let d_cost_d_pi = if gp_final.abs() > 1e-30 {
        let dm_d_pi = tp / ((1.0 + target_prof) * gp_final);
        let d_cost_d_m = 1.0 - FEE_FACTOR * sum_marginal;
        d_cost_d_m * dm_d_pi
    } else {
        0.0
    };

    Some((net_cost, m, d_cost_d_pi))
}

/// Compute cost to reach target_prof for a given outcome via a specific route.
/// Returns (cost, amount, Option<new_price_for_direct>, d_cost_d_pi).
fn cost_for_route(
    sims: &[PoolSim],
    idx: usize,
    route: Route,
    target_prof: f64,
    skip: &HashSet<usize>,
    price_sum: f64,
) -> Option<(f64, f64, Option<f64>, f64)> {
    match route {
        Route::Direct => direct_cost_to_prof(&sims[idx], target_prof).map(|(cost, amount, new_price)| {
            let l = sims[idx].l_eff();
            let pred = sims[idx].prediction;
            let t = 1.0 + target_prof;
            let dcost = -l * pred.sqrt() / (2.0 * t * t.sqrt());
            (cost, amount, Some(new_price), dcost)
        }),
        Route::Mint => mint_cost_to_prof(sims, idx, target_prof, skip, price_sum)
            .map(|(cost, amount, dcost)| (cost, amount, None, dcost)),
    }
}

/// Extract deduplicated outcome indices from active (outcome, route) pairs.
fn active_skip_indices(active: &[(usize, Route)]) -> HashSet<usize> {
    active.iter().map(|(idx, _)| *idx).collect()
}

fn lookup_balance(balances: &HashMap<&str, f64>, market_name: &str) -> f64 {
    balances.get(market_name).copied().unwrap_or(0.0)
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
    let mut contracts: Vec<&'static str> = sims.iter().map(|s| s.market_id).collect();
    contracts.sort();
    contracts.dedup();
    actions.push(Action::Mint {
        contract_1: contracts[0],
        contract_2: if contracts.len() > 1 { contracts[1] } else { contracts[0] },
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
fn merge_sell_proceeds(sims: &[PoolSim], source_idx: usize, amount: f64) -> (f64, f64) {
    let merge_cap: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != source_idx)
        .map(|(_, s)| s.max_buy_tokens())
        .fold(f64::INFINITY, f64::min);

    let actual = amount.min(merge_cap);
    if actual <= 0.0 {
        return (0.0, 0.0);
    }

    let total_buy_cost: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != source_idx)
        .map(|(_, s)| s.buy_exact(actual).map(|(_, cost, _)| cost).unwrap_or(f64::MAX))
        .sum();

    (actual - total_buy_cost, actual)
}

/// Execute a merge sell: buy all other outcomes, merge complete sets, recover sUSD.
/// Updates pool states, emits actions, updates budget.
/// Returns the actual amount merged (tokens consumed from source holding).
fn execute_merge_sell(
    sims: &mut [PoolSim],
    source_idx: usize,
    amount: f64,
    actions: &mut Vec<Action>,
    budget: &mut f64,
) -> f64 {
    let merge_cap: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != source_idx)
        .map(|(_, s)| s.max_buy_tokens())
        .fold(f64::INFINITY, f64::min);

    let actual = amount.min(merge_cap);
    if actual <= 0.0 {
        return 0.0;
    }

    // Pre-compute total buy cost for flash loan amount
    let total_buy_cost: f64 = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != source_idx)
        .map(|(_, s)| s.buy_exact(actual).map(|(_, cost, _)| cost).unwrap_or(0.0))
        .sum();

    actions.push(Action::FlashLoan {
        amount: total_buy_cost,
    });

    // Buy all other outcomes, update pool states
    for i in 0..sims.len() {
        if i == source_idx {
            continue;
        }
        if let Some((bought, cost, new_price)) = sims[i].buy_exact(actual) {
            if bought > 0.0 {
                sims[i].price = new_price;
                actions.push(Action::Buy {
                    market_name: sims[i].market_name,
                    amount: bought,
                    cost,
                });
            }
        }
    }

    // Merge complete sets
    let mut contracts: Vec<&'static str> = sims.iter().map(|s| s.market_id).collect();
    contracts.sort();
    contracts.dedup();
    actions.push(Action::Merge {
        contract_1: contracts[0],
        contract_2: if contracts.len() > 1 {
            contracts[1]
        } else {
            contracts[0]
        },
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
    let mut budget = susds_balance;
    let mut sims = build_sims(slot0_results);

    if sims.is_empty() {
        return actions;
    }

    // Mint/merge routes require all tradeable outcomes to have liquid pools.
    // sims may be smaller if pools have zero liquidity or slot0_results is partial (RPC failures).
    let mint_available = sims.len() == crate::predictions::PREDICTIONS_L1.len();

    // Track holdings changes during simulation
    let mut sim_balances: HashMap<&str, f64> = HashMap::new();
    for sim in &sims {
        sim_balances.insert(sim.market_name, lookup_balance(balances, sim.market_name));
    }

    // ── Phase 1: Sell overpriced holdings ──
    for i in 0..sims.len() {
        let price = sims[i].price();
        if price <= sims[i].prediction {
            continue;
        }
        let held = *sim_balances.get(sims[i].market_name).unwrap_or(&0.0);
        if held <= 0.0 {
            continue;
        }

        // Sell until price = prediction, or sell all if holdings insufficient
        let (tokens_needed, proceeds_to_pred, price_at_pred) = sims[i]
            .sell_to_price(sims[i].prediction)
            .unwrap_or((0.0, 0.0, sims[i].price));

        // Determine sell amount and direct proceeds
        let (sell_amount, direct_proceeds, direct_new_price) =
            if tokens_needed > 0.0 && tokens_needed <= held {
                (tokens_needed, proceeds_to_pred, price_at_pred)
            } else {
                let (sold, proceeds, np) = sims[i]
                    .sell_exact(held)
                    .unwrap_or((0.0, 0.0, sims[i].price));
                (sold, proceeds, np)
            };

        if sell_amount <= 0.0 {
            continue;
        }

        // Compare total proceeds: merge path (merge + direct remainder) vs all-direct
        let use_merge = if mint_available {
            let (merge_net, merge_actual) = merge_sell_proceeds(&sims, i, sell_amount);
            if merge_actual > 0.0 {
                let remainder = sell_amount - merge_actual;
                let remainder_proceeds = if remainder > 1e-18 {
                    sims[i].sell_exact(remainder).map(|(_, p, _)| p).unwrap_or(0.0)
                } else {
                    0.0
                };
                (merge_net + remainder_proceeds) > direct_proceeds
            } else {
                false
            }
        } else {
            false
        };

        if use_merge {
            let merged = execute_merge_sell(&mut sims, i, sell_amount, &mut actions, &mut budget);
            *sim_balances.get_mut(sims[i].market_name).unwrap() -= merged;
            // Direct-sell any remainder that couldn't be merged (capacity-limited)
            let remainder = sell_amount - merged;
            if remainder > 1e-18 {
                if let Some((sold, proceeds, new_price)) = sims[i].sell_exact(remainder) {
                    if sold > 0.0 {
                        sims[i].price = new_price;
                        budget += proceeds;
                        *sim_balances.get_mut(sims[i].market_name).unwrap() -= sold;
                        actions.push(Action::Sell {
                            market_name: sims[i].market_name,
                            amount: sold,
                            proceeds,
                        });
                    }
                }
            }
        } else {
            sims[i].price = direct_new_price;
            budget += direct_proceeds;
            *sim_balances.get_mut(sims[i].market_name).unwrap() -= sell_amount;
            actions.push(Action::Sell {
                market_name: sims[i].market_name,
                amount: sell_amount,
                proceeds: direct_proceeds,
            });
        }
    }

    // ── Phase 2: Waterfall allocation ──
    let actions_before = actions.len();
    let last_bought_prof = waterfall(&mut sims, &mut budget, &mut actions, mint_available);

    // Update sim_balances with positions acquired during waterfall.
    // Mint gives `amount` of every outcome; subsequent Sells reduce non-targets.
    // Partial sells leave residual holdings (amount - sold) that must be tracked.
    for action in &actions[actions_before..] {
        match action {
            Action::Buy { market_name, amount, .. } => {
                *sim_balances.entry(market_name).or_insert(0.0) += amount;
            }
            Action::Mint { amount, .. } => {
                for sim in sims.iter() {
                    *sim_balances.entry(sim.market_name).or_insert(0.0) += amount;
                }
            }
            Action::Sell { market_name, amount, .. } => {
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

    // ── Phase 3: Post-allocation liquidation ──
    // If we bought something, check if any held outcomes are less profitable
    // than the last asset we bought. Sell lowest-profitability holdings first
    // and reallocate upward via waterfall.
    if last_bought_prof > 0.0 {
        // Collect held outcomes with profitability below last_bought_prof
        let mut liquidation_candidates: Vec<(usize, f64)> = Vec::new();
        for (i, sim) in sims.iter().enumerate() {
            let held = *sim_balances.get(sim.market_name).unwrap_or(&0.0);
            if held <= 0.0 {
                continue;
            }
            let prof = profitability(sim.prediction, sim.price());
            if prof < last_bought_prof {
                liquidation_candidates.push((i, prof));
            }
        }

        // Sort by profitability ascending (sell least profitable first)
        liquidation_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

        for (idx, _) in liquidation_candidates {
            let held = *sim_balances.get(sims[idx].market_name).unwrap_or(&0.0);
            if held <= 0.0 {
                continue;
            }
            // Sell only enough to raise profitability to last_bought_prof.
            // Target price = prediction / (1 + last_bought_prof).
            let target_price = target_price_for_prof(sims[idx].prediction, last_bought_prof);
            let (tokens_needed, proceeds_to_target, price_at_target) = sims[idx]
                .sell_to_price(target_price)
                .unwrap_or((0.0, 0.0, sims[idx].price));
            let sell_amount = tokens_needed.min(held);
            if sell_amount <= 0.0 {
                continue;
            }

            // Compute direct sell proceeds
            let (sold, direct_proceeds, direct_new_price) =
                if (sell_amount - held).abs() < 1e-18 {
                    sims[idx]
                        .sell_exact(held)
                        .unwrap_or((0.0, 0.0, sims[idx].price))
                } else if (sell_amount - tokens_needed).abs() < 1e-18 {
                    (tokens_needed, proceeds_to_target, price_at_target)
                } else {
                    sims[idx]
                        .sell_exact(sell_amount)
                        .unwrap_or((0.0, 0.0, sims[idx].price))
                };

            if sold <= 0.0 {
                continue;
            }

            // Compare total proceeds: merge path (merge + direct remainder) vs all-direct
            let use_merge = if mint_available {
                let (merge_net, merge_actual) = merge_sell_proceeds(&sims, idx, sold);
                if merge_actual > 0.0 {
                    let remainder = sold - merge_actual;
                    let remainder_proceeds = if remainder > 1e-18 {
                        sims[idx].sell_exact(remainder).map(|(_, p, _)| p).unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    (merge_net + remainder_proceeds) > direct_proceeds
                } else {
                    false
                }
            } else {
                false
            };

            if use_merge {
                let merged =
                    execute_merge_sell(&mut sims, idx, sold, &mut actions, &mut budget);
                *sim_balances.get_mut(sims[idx].market_name).unwrap() -= merged;
                // Direct-sell any remainder that couldn't be merged
                let remainder = sold - merged;
                if remainder > 1e-18 {
                    if let Some((rsold, rproceeds, rnew_price)) = sims[idx].sell_exact(remainder)
                    {
                        if rsold > 0.0 {
                            sims[idx].price = rnew_price;
                            budget += rproceeds;
                            *sim_balances.get_mut(sims[idx].market_name).unwrap() -= rsold;
                            actions.push(Action::Sell {
                                market_name: sims[idx].market_name,
                                amount: rsold,
                                proceeds: rproceeds,
                            });
                        }
                    }
                }
            } else {
                sims[idx].price = direct_new_price;
                budget += direct_proceeds;
                *sim_balances.get_mut(sims[idx].market_name).unwrap() -= sold;
                actions.push(Action::Sell {
                    market_name: sims[idx].market_name,
                    amount: sold,
                    proceeds: direct_proceeds,
                });
            }
        }

        // Reallocate recovered capital via waterfall
        if budget > 0.0 {
            waterfall(&mut sims, &mut budget, &mut actions, mint_available);
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
            if mp > 0.0 {
                let prof = profitability(sim.prediction, mp);
                if prof > 0.0 && best.map_or(true, |b| prof > b.2) {
                    best = Some((i, Route::Mint, prof));
                }
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

        // Compute total cost for all active entries to reach target_prof
        let total_cost: f64 = active
            .iter()
            .filter_map(|&(idx, route)| {
                cost_for_route(sims, idx, route, target_prof, &skip, price_sum).map(|(c, _, _, _)| c)
            })
            .sum();

        if total_cost <= *budget {
            // Can afford full step. Execute, recomputing costs right before
            // since mint actions mutate other pools.
            let mut any_skipped = false;
            let skip = active_skip_indices(&active);
            for &(idx, route) in &active {
                let ps_before: f64 = sims.iter().map(|s| s.price).sum();
                match cost_for_route(sims, idx, route, target_prof, &skip, ps_before) {
                    Some((cost, amount, new_price, _)) if cost <= *budget => {
                        execute_buy(sims, idx, cost, amount, route, new_price, budget, actions, &skip);
                    }
                    _ => {
                        any_skipped = true;
                    }
                }
            }
            // Refresh price_sum after executions
            price_sum = sims.iter().map(|s| s.price).sum();

            if any_skipped {
                last_prof = current_prof;
                break;
            }

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
            // Can't afford full step. Solve for achievable profitability.
            let skip = active_skip_indices(&active);
            let (achievable, mint_m) =
                solve_prof(sims, &active, current_prof, target_prof, *budget, &skip, price_sum);

            // Execute mints first with aggregate M (updates non-active pool states).
            if mint_m > 0.0 {
                let mint_entries: Vec<(usize, Route)> = active
                    .iter()
                    .filter(|&&(_, r)| r == Route::Mint)
                    .copied()
                    .collect();
                if !mint_entries.is_empty() {
                    let per_entry = mint_m / mint_entries.len() as f64;
                    for &(idx, _) in &mint_entries {
                        execute_buy(
                            sims, idx, 0.0, per_entry, Route::Mint, None, budget, actions, &skip,
                        );
                    }
                }
            }

            // Execute directs second (active pool prices unperturbed by mints)
            price_sum = sims.iter().map(|s| s.price).sum();
            for &(idx, route) in &active {
                if route == Route::Direct {
                    if let Some((cost, amount, new_price, _)) =
                        cost_for_route(sims, idx, route, achievable, &skip, price_sum)
                    {
                        if cost > *budget {
                            continue;
                        }
                        execute_buy(
                            sims, idx, cost, amount, route, new_price, budget, actions, &skip,
                        );
                    }
                }
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

/// Find the lowest profitability level affordable with the budget.
/// Returns (profitability, aggregate_mint_M).
/// Uses closed-form for all-direct; coupled (π, M) Newton for mixed routes.
fn solve_prof(
    sims: &[PoolSim],
    active: &[(usize, Route)],
    prof_hi: f64,
    prof_lo: f64,
    budget: f64,
    skip: &HashSet<usize>,
    price_sum: f64,
) -> (f64, f64) {
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
            return (prof_hi, 0.0);
        }
        let ratio = a_sum / b;
        let prof = ratio * ratio - 1.0;
        return (prof.clamp(prof_lo, prof_hi), 0.0);
    }

    // Coupled mixed-route solver: 2 unknowns (π, M).
    // Skip semantics collapse: all non-active pools see the same aggregate sell volume M.
    let direct_indices: HashSet<usize> = active
        .iter()
        .filter(|&&(_, r)| r == Route::Direct)
        .map(|&(i, _)| i)
        .collect();
    let mint_indices: Vec<usize> = active
        .iter()
        .filter(|&&(_, r)| r == Route::Mint)
        .map(|&(i, _)| i)
        .collect();

    // Pick binding mint target i*: tightest alt-price constraint (fixed for all iterations)
    let i_star = *mint_indices
        .iter()
        .max_by(|&&a, &&b| {
            let gap_a = sims[a].prediction / (1.0 + prof_hi) - alt_price(sims, a, price_sum);
            let gap_b = sims[b].prediction / (1.0 + prof_hi) - alt_price(sims, b, price_sum);
            gap_a
                .partial_cmp(&gap_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    // D* = D ∪ {i*}: outcomes with final price = pred/(1+π)
    let d_star: HashSet<usize> = direct_indices
        .iter()
        .copied()
        .chain(std::iter::once(i_star))
        .collect();
    let pi_star_sum: f64 = d_star.iter().map(|&j| sims[j].prediction).sum();
    let s0: f64 = sims
        .iter()
        .enumerate()
        .filter(|(j, _)| !d_star.contains(j))
        .map(|(_, s)| s.price())
        .sum();

    // Non-active pool parameters: (price, kappa, max_sell)
    let na: Vec<(f64, f64, f64)> = sims
        .iter()
        .enumerate()
        .filter(|(j, _)| !skip.contains(j))
        .map(|(_, s)| (s.price(), s.kappa(), s.max_sell_tokens()))
        .collect();

    // ΔG saturation ceiling
    let dg_max: f64 = na
        .iter()
        .map(|&(p, k, cap)| {
            let d = 1.0 + k * cap;
            p * (1.0 - 1.0 / (d * d))
        })
        .sum();

    let mut pi = prof_hi;
    let mut big_m = 0.0_f64;

    for _ in 0..15 {
        // 1. Direct costs (independent of M; active pool prices unperturbed by mints)
        let mut d_cost = 0.0_f64;
        let mut d_dcost = 0.0_f64;
        for &j in &direct_indices {
            let tsqrt = (sims[j].prediction / (1.0 + pi)).sqrt();
            let csqrt = sims[j].price().sqrt();
            if tsqrt > csqrt {
                d_cost += sims[j].l_eff() * (tsqrt - csqrt);
                let t = 1.0 + pi;
                d_dcost -= sims[j].l_eff() * sims[j].prediction.sqrt() / (2.0 * t * t.sqrt());
            }
        }

        // 2. δ(π) = Π*/(1+π) - (1 - S₀)
        let delta = pi_star_sum / (1.0 + pi) - (1.0 - s0);
        let d_delta = -pi_star_sum / ((1.0 + pi) * (1.0 + pi));

        // 3. Solve M(π), compute mint cost and d(cost)/dπ
        let mut c_mint = 0.0_f64;
        let mut dc_dpi = 0.0_f64;

        if delta <= 0.0 {
            big_m = 0.0;
        } else if delta >= dg_max && dg_max > 0.0 {
            // Saturated: all non-active pools capped
            let pi_bound = pi_star_sum / (1.0 - s0 + dg_max) - 1.0;
            big_m = na.iter().map(|&(_, _, cap)| cap).fold(0.0_f64, f64::max);
            c_mint = big_m
                - na.iter()
                    .map(|&(p, k, cap)| p * cap * FEE_FACTOR / (1.0 + k * cap))
                    .sum::<f64>();
            if direct_indices.is_empty() {
                if c_mint <= budget {
                    return (pi_bound.clamp(prof_lo, prof_hi), big_m);
                }
                // Can't afford saturation — nudge pi above boundary so next
                // iteration lands in the unsaturated regime with non-zero gradients.
                pi = pi_bound + 1e-10;
                continue;
            }
            if pi < pi_bound {
                pi = pi_bound;
                // Recompute direct costs for updated pi
                d_cost = 0.0;
                d_dcost = 0.0;
                for &j in &direct_indices {
                    let tsqrt = (sims[j].prediction / (1.0 + pi)).sqrt();
                    let csqrt = sims[j].price().sqrt();
                    if tsqrt > csqrt {
                        d_cost += sims[j].l_eff() * (tsqrt - csqrt);
                        let t = 1.0 + pi;
                        d_dcost -=
                            sims[j].l_eff() * sims[j].prediction.sqrt() / (2.0 * t * t.sqrt());
                    }
                }
            }
            // dc_dpi = 0 (M doesn't vary with π when saturated)
        } else {
            // Inner Newton: solve ΔG(M) = δ
            let dg0: f64 = na.iter().map(|&(p, k, _)| 2.0 * p * k).sum();
            big_m = if dg0 > 1e-30 {
                (delta / dg0).max(0.0)
            } else {
                0.0
            };
            for _ in 0..8 {
                let mut dg = 0.0_f64;
                let mut dgp = 0.0_f64;
                for &(p, k, cap) in &na {
                    let me = big_m.min(cap);
                    let d = 1.0 + k * me;
                    let d2 = d * d;
                    dg += p * (1.0 - 1.0 / d2);
                    if big_m < cap {
                        dgp += 2.0 * p * k / (d2 * d);
                    }
                }
                if dgp < 1e-30 {
                    break;
                }
                let step = (dg - delta) / dgp;
                big_m -= step;
                big_m = big_m.max(0.0);
                if step.abs() < 1e-10 * (1.0 + big_m) {
                    break;
                }
            }

            // Net cost and derivatives in one pass
            let mut dgp_final = 0.0_f64;
            let mut dc_dm = 1.0_f64;
            c_mint = big_m;
            for &(p, k, cap) in &na {
                let me = big_m.min(cap);
                let d = 1.0 + me * k;
                c_mint -= p * me * FEE_FACTOR / d;
                if big_m < cap {
                    let d2 = d * d;
                    dgp_final += 2.0 * p * k / (d2 * d);
                    dc_dm -= FEE_FACTOR * p / d2;
                }
            }
            let dm_dpi = if dgp_final > 1e-30 {
                d_delta / dgp_final
            } else {
                0.0
            };
            dc_dpi = dc_dm * dm_dpi;
        }

        // 4. Newton step on π
        let total = d_cost + c_mint;
        let dtotal = d_dcost + dc_dpi;
        if dtotal.abs() < 1e-30 {
            break;
        }
        let step = (total - budget) / dtotal;
        let pi_new = (pi - step).clamp(prof_lo, prof_hi);
        if (pi_new - pi).abs() < 1e-12 * (1.0 + pi.abs()) {
            break;
        }
        pi = pi_new;
    }

    // Final M recompute to ensure consistency with returned pi
    let delta_final = pi_star_sum / (1.0 + pi) - (1.0 - s0);
    if delta_final <= 0.0 {
        big_m = 0.0;
    } else if delta_final >= dg_max && dg_max > 0.0 {
        big_m = na.iter().map(|&(_, _, cap)| cap).fold(0.0_f64, f64::max);
    } else {
        let dg0: f64 = na.iter().map(|&(p, k, _)| 2.0 * p * k).sum();
        big_m = if dg0 > 1e-30 { (delta_final / dg0).max(0.0) } else { 0.0 };
        for _ in 0..8 {
            let mut dg = 0.0_f64;
            let mut dgp = 0.0_f64;
            for &(p, k, cap) in &na {
                let me = big_m.min(cap);
                let d = 1.0 + k * me;
                let d2 = d * d;
                dg += p * (1.0 - 1.0 / d2);
                if big_m < cap {
                    dgp += 2.0 * p * k / (d2 * d);
                }
            }
            if dgp < 1e-30 { break; }
            let step = (dg - delta_final) / dgp;
            big_m -= step;
            big_m = big_m.max(0.0);
            if step.abs() < 1e-10 * (1.0 + big_m) { break; }
        }
    }

    (pi, big_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markets::{MarketData, Pool, Tick};
    use crate::pools::{prediction_to_sqrt_price_x96, Slot0Result};
    use alloy::primitives::{Address, U256};

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
        let sqrt_price = prediction_to_sqrt_price_x96(price_fraction, true)
            .unwrap_or(U256::from(1u128 << 96));

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
        assert!(
            (price - 0.3).abs() < 0.01,
            "price {} should be ~0.3",
            price
        );
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
        let (slot0, market) = mock_slot0_market(
            "cheap",
            "0x1111111111111111111111111111111111111111",
            0.01,
        );
        let sim = PoolSim::from_slot0(&slot0, market, 0.5).unwrap();

        let (cost, amount, new_price) = sim.cost_to_price(0.1).unwrap();
        assert!(cost > 0.0, "should cost something to move price");
        assert!(amount > 0.0, "should receive outcome tokens");

        // Price should increase toward target, clamped to tick boundary
        assert!(
            new_price > 0.01,
            "price after buy {} should be > initial 0.01",
            new_price
        );
    }

    #[test]
    fn test_waterfall_equalizes() {
        // Two outcomes: A at price 0.05 pred 0.1, B at price 0.05 pred 0.08
        // profitability(A) = (0.1 - 0.05)/0.05 = 1.0
        // profitability(B) = (0.08 - 0.05)/0.05 = 0.6
        // Waterfall should first buy A until prof(A) = 0.6, then buy both

        let (slot0_a, market_a) = mock_slot0_market(
            "A",
            "0x1111111111111111111111111111111111111111",
            0.05,
        );
        let (slot0_b, market_b) = mock_slot0_market(
            "B",
            "0x2222222222222222222222222222222222222222",
            0.05,
        );

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
        assert!(
            (budget - 100.0).abs() < 1e-6,
            "budget should be unchanged"
        );
    }

    #[test]
    fn test_budget_exhaustion() {
        let (slot0, market) = mock_slot0_market(
            "cheap",
            "0x1111111111111111111111111111111111111111",
            0.01,
        );
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
            matches!(&actions[0], Action::Mint { target_market: "M1", .. }),
            "first action should be Mint targeting M1"
        );
        if let Action::Mint { amount, .. } = &actions[0] {
            assert!((*amount - 10.0).abs() < 1e-12, "mint amount should be 10.0");
        }

        // Sells for non-target outcomes
        let sells: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                Action::Sell { market_name, amount, .. } => Some((*market_name, *amount)),
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
        execute_buy(&mut sims2, 0, 5.0, 10.0, Route::Mint, None, &mut budget, &mut actions2, &HashSet::new());
        assert!(budget < 100.0, "budget should decrease after mint");

        // Test sim_balances tracking: Mint adds to all, Sell subtracts
        let mut sim_balances: HashMap<&str, f64> = HashMap::new();
        for action in &actions2 {
            match action {
                Action::Buy { market_name, amount, .. } => {
                    *sim_balances.entry(market_name).or_insert(0.0) += amount;
                }
                Action::Mint { amount, .. } => {
                    for sim in sims2.iter() {
                        *sim_balances.entry(sim.market_name).or_insert(0.0) += amount;
                    }
                }
                Action::Sell { market_name, amount, .. } => {
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
            "M1 should hold 10.0 minted tokens, got {}", m1_bal
        );
        // Non-targets should have residual >= 0 (mint - sold)
        for name in &["M2", "M3"] {
            let bal = *sim_balances.get(name).unwrap_or(&0.0);
            assert!(bal >= -1e-12, "{} balance should be >= 0 (residual), got {}", name, bal);
        }
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

        println!("Merge net: {:.6}, Direct proceeds: {:.6}", merge_net, direct_proceeds);
        assert!(merge_net > direct_proceeds,
            "merge ({:.6}) should beat direct ({:.6}) for high-price outcome", merge_net, direct_proceeds);
        assert!(merge_actual > 0.0, "merge should be feasible");

        // Execute merge sell and verify actions
        let mut budget = 0.0;
        let mut actions = Vec::new();
        let merged = execute_merge_sell(&mut sims, 0, sell_amount, &mut actions, &mut budget);

        assert!(merged > 0.0, "should have merged tokens");
        assert!(budget > 0.0, "budget should increase from merge proceeds");

        // Should have: FlashLoan, Buy×2, Merge, RepayFlashLoan
        let has_merge = actions.iter().any(|a| matches!(a, Action::Merge { .. }));
        let has_flash = actions.iter().any(|a| matches!(a, Action::FlashLoan { .. }));
        let buy_count = actions.iter().filter(|a| matches!(a, Action::Buy { .. })).count();
        assert!(has_merge, "should have Merge action");
        assert!(has_flash, "should have FlashLoan action");
        assert_eq!(buy_count, 2, "should buy 2 non-source outcomes");

        // Other pool prices should have increased (we bought into them)
        assert!(sims[1].price > 0.05, "M2 price should increase after buying");
        assert!(sims[2].price > 0.05, "M3 price should increase after buying");
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

        println!("Merge net: {:.6}, Direct proceeds: {:.6}", merge_net, direct_proceeds);
        assert!(direct_proceeds > merge_net,
            "direct ({:.6}) should beat merge ({:.6}) for low-price outcome with expensive others",
            direct_proceeds, merge_net);
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

        let buys = actions.iter().filter(|a| matches!(a, Action::Buy { .. })).count();
        let sells = actions.iter().filter(|a| matches!(a, Action::Sell { .. })).count();
        let mints = actions.iter().filter(|a| matches!(a, Action::Mint { .. })).count();
        let merges = actions.iter().filter(|a| matches!(a, Action::Merge { .. })).count();

        println!("=== Rebalance Performance (full L1, {} outcomes) ===", slot0_results.len());
        println!("  Total: {:?} for {} iterations", elapsed, iters);
        println!("  Per call: {:?}", elapsed / iters as u32);
        println!("  Actions: {} total ({} buys, {} sells, {} mints, {} merges)", actions.len(), buys, sells, mints, merges);

        // Sanity: should produce actions when everything is underpriced
        assert!(!actions.is_empty(), "should produce actions for underpriced markets");

        // === Expected value verification ===
        // Before: EV = 100.0 sUSD (no holdings)
        let initial_budget = 100.0;

        // Compute portfolio after rebalancing
        let mut holdings: HashMap<&str, f64> = HashMap::new();
        let mut total_cost = 0.0_f64;
        let mut total_sell_proceeds = 0.0_f64;
        for action in &actions {
            match action {
                Action::Buy { market_name, amount, cost } => {
                    *holdings.entry(market_name).or_insert(0.0) += amount;
                    total_cost += cost;
                }
                Action::Sell { market_name, amount, proceeds } => {
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
        let ev_holdings: f64 = holdings.iter().map(|(name, &units)| {
            let key = normalize_market_name(name);
            let pred = preds.get(&key).copied().unwrap_or(0.0);
            pred * units
        }).sum();
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
        println!("  EV gain:          {:.6} sUSD ({:.2}%)", ev_after - initial_budget, (ev_after / initial_budget - 1.0) * 100.0);
        println!("  Remaining budget: {:.6} sUSD", remaining_budget);
        println!("  Holdings EV:      {:.6} sUSD", ev_holdings);
        println!("  Outcomes held:    {}/{}", outcomes_bought.len(), slot0_results.len());
        println!("  Total buy cost:   {:.6}", total_cost);
        println!("  Total sell proc:  {:.6}", total_sell_proceeds);

        // EV should increase (we're buying underpriced assets at 50% of prediction)
        assert!(ev_after > initial_budget,
            "EV should increase: before={:.6}, after={:.6}", initial_budget, ev_after);

        // Budget accounting: remaining should be >= 0 and < initial
        assert!(remaining_budget >= -1e-9,
            "remaining budget should be non-negative: {:.6}", remaining_budget);
        assert!(remaining_budget < initial_budget,
            "should have spent some budget: remaining={:.6}", remaining_budget);
    }

    #[tokio::test]
    async fn test_rebalance_integration() {
        // Integration test with real pool data
        dotenvy::dotenv().ok();
        let rpc_url = match std::env::var("RPC") {
            Ok(url) => url,
            Err(_) => return, // skip if no RPC
        };
        let provider =
            alloy::providers::ProviderBuilder::new().connect_http(rpc_url.parse().unwrap());

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
                    println!("  MINT {} sets for {} (c1={}, c2={})", amount, target_market, contract_1, contract_2)
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
                    println!("  MERGE {} sets from {} (c1={}, c2={})", amount, source_market, contract_1, contract_2)
                }
                Action::FlashLoan { amount } => println!("  FLASH_LOAN {:.6}", amount),
                Action::RepayFlashLoan { amount } => println!("  REPAY_FLASH_LOAN {:.6}", amount),
            }
        }
    }
}
