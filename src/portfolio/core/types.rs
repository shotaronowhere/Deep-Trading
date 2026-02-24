use std::collections::HashMap;

use super::sim::{PoolSim, sanitize_nonnegative_finite};

/// Shared simulated holdings map keyed by market name.
pub(super) type BalanceMap = HashMap<&'static str, f64>;

/// A rebalancing action to execute.
#[derive(Debug, Clone, PartialEq)]
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
}

pub(super) fn lookup_balance(balances: &HashMap<&str, f64>, market_name: &str) -> f64 {
    sanitize_nonnegative_finite(balances.get(market_name).copied().unwrap_or(0.0))
}

pub(super) fn subtract_balance(
    sim_balances: &mut HashMap<&str, f64>,
    market_name: &'static str,
    amount: f64,
) {
    if amount <= 0.0 {
        return;
    }
    let bal = sim_balances.entry(market_name).or_insert(0.0);
    *bal -= amount;
    if *bal < 0.0 {
        *bal = 0.0;
    }
}

pub(super) fn apply_actions_to_sim_balances(
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
                for sim in sims {
                    *sim_balances.entry(sim.market_name).or_insert(0.0) += amount;
                }
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                subtract_balance(sim_balances, *market_name, *amount);
            }
            Action::Merge { amount, .. } => {
                for sim in sims {
                    subtract_balance(sim_balances, sim.market_name, *amount);
                }
            }
        }
    }
}
