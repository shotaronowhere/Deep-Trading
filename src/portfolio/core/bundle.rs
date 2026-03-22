use std::collections::HashSet;

use super::sim::{EPS, FEE_FACTOR, PoolSim, alt_price, profitability, target_price_for_prof};

pub(super) const ACTIVE_FRONTIER_REL_TOL: f64 = 1e-9;

#[derive(Debug, Clone)]
pub(super) struct BundleFrontier {
    pub(super) members: Vec<usize>,
    pub(super) current_prof: f64,
    pub(super) next_prof: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BundleRouteKind {
    Direct,
    Mint,
}

#[derive(Debug, Clone)]
pub(super) struct BundleDirectEstimate {
    pub(super) cash_cost: f64,
    pub(super) member_plans: Vec<(usize, f64, f64, f64)>,
}

#[derive(Debug, Clone)]
pub(super) struct BundleMintEstimate {
    pub(super) cash_cost: f64,
    pub(super) mint_amount: f64,
    pub(super) sell_leg_plans: Vec<(usize, f64, f64, f64)>,
}

#[derive(Debug, Clone)]
pub(super) struct BundleSegmentPlan {
    pub(super) kind: BundleRouteKind,
    #[allow(dead_code)]
    pub(super) target_prof: f64,
    pub(super) cash_cost: f64,
    pub(super) mint_amount: f64,
    pub(super) direct_member_plans: Vec<(usize, f64, f64, f64)>,
    pub(super) mint_sell_leg_plans: Vec<(usize, f64, f64, f64)>,
}

#[derive(Debug, Clone)]
pub(super) struct BundleStepPlan {
    pub(super) segments: Vec<BundleSegmentPlan>,
    pub(super) final_prof: f64,
    pub(super) fully_affordable: bool,
}

pub(super) fn profs_tied(a: f64, b: f64) -> bool {
    (a - b).abs() <= ACTIVE_FRONTIER_REL_TOL * (1.0 + a.abs().max(b.abs()))
}

pub(super) fn direct_profit(sim: &PoolSim) -> f64 {
    profitability(sim.prediction, sim.price())
}

fn sorted_preserve_indices(preserve_sell_indices: &HashSet<usize>) -> Vec<usize> {
    let mut indices: Vec<usize> = preserve_sell_indices.iter().copied().collect();
    indices.sort_unstable();
    indices
}

pub(super) fn direct_bundle_frontier(
    sims: &[PoolSim],
    remaining_budget: f64,
    gas_direct_susd: f64,
) -> Option<BundleFrontier> {
    let mut best_direct_prof = f64::NEG_INFINITY;
    for sim in sims {
        let direct_prof = direct_profit(sim);
        if direct_prof > 0.0
            && remaining_budget * direct_prof >= gas_direct_susd
            && direct_prof > best_direct_prof
        {
            best_direct_prof = direct_prof;
        }
    }

    if best_direct_prof.is_finite() && best_direct_prof > 0.0 {
        let mut members = Vec::new();
        let mut next_prof = 0.0_f64;
        for (idx, sim) in sims.iter().enumerate() {
            let direct_prof = direct_profit(sim);
            if direct_prof <= 0.0 || remaining_budget * direct_prof < gas_direct_susd {
                continue;
            }
            if profs_tied(direct_prof, best_direct_prof) {
                members.push(idx);
            } else if direct_prof > next_prof {
                next_prof = direct_prof;
            }
        }

        return (!members.is_empty()).then_some(BundleFrontier {
            members,
            current_prof: best_direct_prof,
            next_prof,
        });
    }

    None
}

pub(super) fn mint_bundle_frontier(
    sims: &[PoolSim],
    mint_available: bool,
    remaining_budget: f64,
    gas_mint_susd: f64,
    preserve_sell_indices: &HashSet<usize>,
) -> Option<BundleFrontier> {
    if !mint_available {
        return None;
    }

    let price_sum: f64 = sims.iter().map(|sim| sim.price()).sum();
    let sorted_preserve = sorted_preserve_indices(preserve_sell_indices);
    let preserved_price_sum: f64 = sorted_preserve
        .iter()
        .filter_map(|&idx| sims.get(idx))
        .map(|sim| sim.price())
        .sum();
    let preserved_prediction_sum: f64 = sorted_preserve
        .iter()
        .filter_map(|&idx| sims.get(idx))
        .map(|sim| sim.prediction)
        .sum();
    let mut best_mint_prof = f64::NEG_INFINITY;
    for (idx, sim) in sims.iter().enumerate() {
        let preserved_non_target_sum = if preserve_sell_indices.contains(&idx) {
            preserved_price_sum - sim.price()
        } else {
            preserved_price_sum
        };
        let preserved_non_target_prediction_sum = if preserve_sell_indices.contains(&idx) {
            preserved_prediction_sum - sim.prediction
        } else {
            preserved_prediction_sum
        };
        let mint_price = alt_price(sims, idx, price_sum) + preserved_non_target_sum.max(0.0);
        let mint_prediction = sim.prediction + preserved_non_target_prediction_sum.max(0.0);
        let mint_prof = profitability(mint_prediction, mint_price);
        if mint_prof > 0.0
            && remaining_budget * mint_prof >= gas_mint_susd
            && mint_prof > best_mint_prof
        {
            best_mint_prof = mint_prof;
        }
    }
    if !best_mint_prof.is_finite() || best_mint_prof <= 0.0 {
        return None;
    }

    let mut members = Vec::new();
    let mut next_prof = 0.0_f64;
    for (idx, sim) in sims.iter().enumerate() {
        let preserved_non_target_sum = if preserve_sell_indices.contains(&idx) {
            preserved_price_sum - sim.price()
        } else {
            preserved_price_sum
        };
        let preserved_non_target_prediction_sum = if preserve_sell_indices.contains(&idx) {
            preserved_prediction_sum - sim.prediction
        } else {
            preserved_prediction_sum
        };
        let mint_price = alt_price(sims, idx, price_sum) + preserved_non_target_sum.max(0.0);
        let mint_prediction = sim.prediction + preserved_non_target_prediction_sum.max(0.0);
        let mint_prof = profitability(mint_prediction, mint_price);
        if mint_prof <= 0.0 || remaining_budget * mint_prof < gas_mint_susd {
            continue;
        }
        if profs_tied(mint_prof, best_mint_prof) {
            members.push(idx);
        } else if mint_prof > next_prof {
            next_prof = mint_prof;
        }
    }

    (!members.is_empty()).then_some(BundleFrontier {
        members,
        current_prof: best_mint_prof,
        next_prof,
    })
}

pub(super) fn bundle_frontier(
    sims: &[PoolSim],
    mint_available: bool,
    remaining_budget: f64,
    gas_direct_susd: f64,
    gas_mint_susd: f64,
    preserve_sell_indices: &HashSet<usize>,
) -> Option<BundleFrontier> {
    let direct = direct_bundle_frontier(sims, remaining_budget, gas_direct_susd);
    let mint = mint_bundle_frontier(
        sims,
        mint_available,
        remaining_budget,
        gas_mint_susd,
        preserve_sell_indices,
    );
    match (direct, mint) {
        (Some(d), Some(m)) => {
            if m.current_prof > d.current_prof {
                Some(m)
            } else {
                Some(d)
            }
        }
        (Some(d), None) => Some(d),
        (None, m) => m,
    }
}

pub(super) fn direct_bundle_marginal_cost_at_prof(
    sims: &[PoolSim],
    bundle_members: &[usize],
    target_prof: f64,
) -> f64 {
    bundle_members
        .iter()
        .map(|&idx| target_price_for_prof(sims[idx].prediction, target_prof) / FEE_FACTOR)
        .sum()
}

pub(super) fn direct_preserve_marginal_cost_at_spot(
    sims: &[PoolSim],
    bundle_members: &[usize],
    preserve_sell_indices: &HashSet<usize>,
) -> f64 {
    sorted_preserve_indices(preserve_sell_indices)
        .into_iter()
        .filter_map(|idx| {
            if idx >= sims.len() || bundle_members.contains(&idx) {
                return None;
            }
            Some(sims[idx].price() / FEE_FACTOR)
        })
        .sum()
}

pub(super) fn mint_bundle_marginal_cost_at_prof(
    sims: &[PoolSim],
    bundle_members: &[usize],
    target_prof: f64,
    preserve_sell_indices: &HashSet<usize>,
) -> Option<f64> {
    let mut sellable_credit = 0.0_f64;
    let mut sellable = false;
    for (idx, sim) in sims.iter().enumerate() {
        if bundle_members.contains(&idx) || preserve_sell_indices.contains(&idx) {
            continue;
        }
        let frontier_price = target_price_for_prof(sim.prediction, target_prof);
        let tol = EPS * (1.0 + sim.price().abs().max(frontier_price.abs()));
        if sim.price() <= frontier_price + tol {
            continue;
        }
        sellable = true;
        sellable_credit += sim.price() * FEE_FACTOR;
    }
    sellable.then_some(1.0 - sellable_credit)
}
