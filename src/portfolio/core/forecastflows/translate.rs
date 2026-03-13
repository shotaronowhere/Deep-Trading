use std::collections::HashMap;
use std::fmt;

use uniswap_v3_math::tick_math::get_sqrt_ratio_at_tick;

use crate::markets::MarketData;
use crate::pools::{FEE_PIPS, Slot0Result, normalize_market_name, sqrt_price_x96_to_price_outcome};

use super::protocol::{
    CompareResult, MarketSpecRequest, OutcomeSpecRequest, PredictionMarketProblemRequest,
    PredictionMarketSolveResult, PredictionMarketTrade, UniV3LiquidityBandRequest,
};
use super::{ForecastFlowsCandidateVariant, ForecastFlowsFamilyCandidate};
use crate::portfolio::Action;

use super::super::merge::action_contract_pair;
use super::super::sim::{DUST, EPS, build_sims};
use super::super::types::{BalanceMap, lookup_balance};

#[derive(Debug)]
pub(super) enum ForecastFlowsTranslationError {
    UnsupportedSnapshot(String),
    InvalidResponse(String),
}

impl fmt::Display for ForecastFlowsTranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSnapshot(message) => write!(f, "{message}"),
            Self::InvalidResponse(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ForecastFlowsTranslationError {}

pub(super) fn build_problem_request(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
) -> Result<PredictionMarketProblemRequest, ForecastFlowsTranslationError> {
    if slot0_results.len() != expected_outcome_count {
        return Err(ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
            "ForecastFlows requires a full L1 snapshot with {expected_outcome_count} outcomes; got {}",
            slot0_results.len()
        )));
    }

    let mut outcomes = Vec::with_capacity(slot0_results.len());
    let mut markets = Vec::with_capacity(slot0_results.len());
    for (slot0, market) in slot0_results {
        let prediction_key = normalize_market_name(market.name);
        let fair_value = predictions.get(&prediction_key).copied().ok_or_else(|| {
            ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
                "missing prediction for market {}",
                market.name
            ))
        })?;
        let pool = market.pool.as_ref().ok_or_else(|| {
            ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
                "market {} is missing a pool",
                market.name
            ))
        })?;
        let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
        let (active_liquidity, tick_lo, tick_hi) = current_single_tick_geometry(slot0, market)
            .ok_or_else(|| {
                ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
                    "market {} does not have replayable single-range geometry",
                    market.name
                ))
            })?;
        let (buy_limit_price, sell_limit_price) =
            single_tick_limit_prices(market, is_token1_outcome, tick_lo, tick_hi).ok_or_else(
                || {
                    ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
                        "failed to derive active-range prices for market {}",
                        market.name
                    ))
                },
            )?;
        let current_price =
            sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome)
                .and_then(|value| u128::try_from(value).ok())
                .map(|wad| wad as f64 / 1e18)
                .ok_or_else(|| {
                    ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
                        "failed to derive current price for market {}",
                        market.name
                    ))
                })?;

        outcomes.push(OutcomeSpecRequest {
            outcome_id: market.outcome_token.to_string(),
            fair_value,
            initial_holding: lookup_balance(balances, market.name),
        });
        markets.push(MarketSpecRequest::UniV3 {
            market_id: market.name.to_string(),
            outcome_id: market.outcome_token.to_string(),
            current_price,
            bands: vec![
                UniV3LiquidityBandRequest {
                    lower_price: buy_limit_price,
                    liquidity_l: active_liquidity as f64 / 1e18,
                },
                UniV3LiquidityBandRequest {
                    lower_price: sell_limit_price,
                    liquidity_l: 0.0,
                },
            ],
            fee_multiplier: 1.0 - (FEE_PIPS as f64 / 1_000_000.0),
        });
    }

    Ok(PredictionMarketProblemRequest {
        outcomes,
        collateral_balance: susds_balance.max(0.0),
        markets,
        split_bound: None,
    })
}

#[cfg(test)]
pub(super) fn translate_compare_result(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    compare: CompareResult,
) -> Result<Vec<ForecastFlowsFamilyCandidate>, ForecastFlowsTranslationError> {
    Ok(translate_compare_result_report(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        compare,
    )?
    .candidates)
}

pub(super) struct CompareTranslationReport {
    pub(super) candidates: Vec<ForecastFlowsFamilyCandidate>,
    pub(super) certified_drop_reasons: Vec<VariantDropReason>,
}

pub(super) struct VariantDropReason {
    pub(super) variant: ForecastFlowsCandidateVariant,
    pub(super) reason: String,
}

pub(super) fn translate_compare_result_report(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    compare: CompareResult,
) -> Result<CompareTranslationReport, ForecastFlowsTranslationError> {
    let mut candidates = Vec::new();
    let mut certified_drop_reasons = Vec::new();

    collect_variant_translation(
        &mut candidates,
        &mut certified_drop_reasons,
        balances,
        susds_balance,
        slot0_results,
        predictions,
        &compare.direct_only,
        ForecastFlowsCandidateVariant::Direct,
    );
    collect_variant_translation(
        &mut candidates,
        &mut certified_drop_reasons,
        balances,
        susds_balance,
        slot0_results,
        predictions,
        &compare.mixed_enabled,
        ForecastFlowsCandidateVariant::Mixed,
    );

    if candidates.is_empty() && !certified_drop_reasons.is_empty() {
        let joined = certified_drop_reasons
            .iter()
            .map(|drop| format!("{}: {}", drop.variant.as_str(), drop.reason))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
            "all certified ForecastFlows candidates were dropped during translation: {joined}"
        )));
    }

    Ok(CompareTranslationReport {
        candidates,
        certified_drop_reasons,
    })
}

fn collect_variant_translation(
    candidates: &mut Vec<ForecastFlowsFamilyCandidate>,
    certified_drop_reasons: &mut Vec<VariantDropReason>,
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    result: &PredictionMarketSolveResult,
    variant: ForecastFlowsCandidateVariant,
) {
    match translate_solve_result(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        result,
        variant,
    ) {
        Ok(Some(candidate)) => candidates.push(candidate),
        Ok(None) => {}
        Err(err) => {
            if result.is_certified() {
                certified_drop_reasons.push(VariantDropReason {
                    variant,
                    reason: err.to_string(),
                });
            }
        }
    }
}

fn translate_solve_result(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    result: &PredictionMarketSolveResult,
    variant: ForecastFlowsCandidateVariant,
) -> Result<Option<ForecastFlowsFamilyCandidate>, ForecastFlowsTranslationError> {
    if !result.is_certified() {
        return Ok(None);
    }
    match variant {
        ForecastFlowsCandidateVariant::Direct if result.mode != "direct_only" => {
            return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
                "direct candidate returned unexpected mode {}",
                result.mode
            )));
        }
        ForecastFlowsCandidateVariant::Mixed if result.mode != "mixed_enabled" => {
            return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
                "mixed candidate returned unexpected mode {}",
                result.mode
            )));
        }
        _ => {}
    }

    let market_catalog = build_market_catalog(slot0_results);
    let net_trades = net_worker_trades(&market_catalog, &result.trades)?;
    let mint_amount = sanitize_nonnegative(result.split_merge.mint, "split_merge.mint")?;
    let merge_amount = sanitize_nonnegative(result.split_merge.merge, "split_merge.merge")?;
    if mint_amount > DUST && merge_amount > DUST {
        return Err(ForecastFlowsTranslationError::InvalidResponse(
            "mixed result cannot mint and merge in the same candidate".to_string(),
        ));
    }

    let mut sims = build_sims(slot0_results, predictions).map_err(|err| {
        ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
            "failed to build PoolSim replay state: {err}"
        ))
    })?;
    let mut sim_idx_by_market: HashMap<&'static str, usize> = HashMap::new();
    for (index, sim) in sims.iter().enumerate() {
        sim_idx_by_market.insert(sim.market_name, index);
    }
    let mut sim_balances = build_initial_holdings(balances, slot0_results);
    let mut cash = susds_balance.max(0.0);
    let representative_market =
        representative_complete_set_market(slot0_results).ok_or_else(|| {
            ForecastFlowsTranslationError::UnsupportedSnapshot(
                "cannot choose a representative complete-set market".to_string(),
            )
        })?;
    let (contract_1, contract_2) = action_contract_pair(&sims);
    let mut actions = Vec::new();

    if mint_amount > DUST {
        if cash + EPS < mint_amount {
            return Err(ForecastFlowsTranslationError::InvalidResponse(
                "local replay rejected mint amount due to insufficient cash".to_string(),
            ));
        }
        cash -= mint_amount;
        for (_, market) in slot0_results {
            *sim_balances.entry(market.name).or_insert(0.0) += mint_amount;
        }
        actions.push(Action::Mint {
            contract_1,
            contract_2,
            amount: mint_amount,
            target_market: representative_market,
        });
    }

    let mut sells = net_trades.sells;
    sells.sort_by_key(|trade| trade.market_name);
    let mut buys = net_trades.buys;
    buys.sort_by_key(|trade| trade.market_name);

    if merge_amount <= DUST {
        for trade in &sells {
            replay_sell(
                &mut sims,
                &sim_idx_by_market,
                &mut sim_balances,
                &mut cash,
                &mut actions,
                trade.market_name,
                trade.amount,
            )?;
        }
        for trade in &buys {
            replay_buy(
                &mut sims,
                &sim_idx_by_market,
                &mut sim_balances,
                &mut cash,
                &mut actions,
                trade.market_name,
                trade.amount,
            )?;
        }
    } else {
        for trade in &buys {
            replay_buy(
                &mut sims,
                &sim_idx_by_market,
                &mut sim_balances,
                &mut cash,
                &mut actions,
                trade.market_name,
                trade.amount,
            )?;
        }
        for (_, market) in slot0_results {
            let holding = sim_balances
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            if holding + EPS < merge_amount {
                return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
                    "local replay rejected merge amount due to insufficient holdings for {}",
                    market.name
                )));
            }
        }
        for (_, market) in slot0_results {
            let holding = sim_balances.entry(market.name).or_insert(0.0);
            *holding = (*holding - merge_amount).max(0.0);
        }
        cash += merge_amount;
        actions.push(Action::Merge {
            contract_1,
            contract_2,
            amount: merge_amount,
            source_market: representative_market,
        });
        for trade in &sells {
            replay_sell(
                &mut sims,
                &sim_idx_by_market,
                &mut sim_balances,
                &mut cash,
                &mut actions,
                trade.market_name,
                trade.amount,
            )?;
        }
    }

    Ok(Some(ForecastFlowsFamilyCandidate { actions, variant }))
}

#[derive(Debug, Clone, Copy)]
struct TradeAmount {
    market_name: &'static str,
    amount: f64,
}

#[derive(Default)]
struct NettedTrades {
    buys: Vec<TradeAmount>,
    sells: Vec<TradeAmount>,
}

fn build_market_catalog(
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> HashMap<&'static str, &'static str> {
    let mut market_catalog = HashMap::new();
    for (_, market) in slot0_results {
        market_catalog.insert(market.name, market.outcome_token);
    }
    market_catalog
}

fn net_worker_trades(
    market_catalog: &HashMap<&'static str, &'static str>,
    trades: &[PredictionMarketTrade],
) -> Result<NettedTrades, ForecastFlowsTranslationError> {
    let mut net_by_market: HashMap<&'static str, (f64, f64)> = HashMap::new();
    for trade in trades {
        if !trade.collateral_delta.is_finite() || !trade.outcome_delta.is_finite() {
            return Err(ForecastFlowsTranslationError::InvalidResponse(
                "worker returned a non-finite trade delta".to_string(),
            ));
        }
        let Some((&market_name, &outcome_id)) = market_catalog
            .iter()
            .find(|(market_name, _)| **market_name == trade.market_id)
        else {
            return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
                "worker returned unknown market_id {}",
                trade.market_id
            )));
        };
        if outcome_id != trade.outcome_id {
            return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
                "worker outcome_id {} does not match market {}",
                trade.outcome_id, trade.market_id
            )));
        }
        let entry = net_by_market.entry(market_name).or_insert((0.0, 0.0));
        entry.0 += trade.collateral_delta;
        entry.1 += trade.outcome_delta;
    }

    let mut netted = NettedTrades::default();
    for (market_name, (collateral_delta, outcome_delta)) in net_by_market {
        if collateral_delta.abs() <= EPS && outcome_delta.abs() <= EPS {
            continue;
        }
        if collateral_delta < -EPS && outcome_delta > EPS {
            netted.buys.push(TradeAmount {
                market_name,
                amount: outcome_delta,
            });
        } else if collateral_delta > EPS && outcome_delta < -EPS {
            netted.sells.push(TradeAmount {
                market_name,
                amount: -outcome_delta,
            });
        } else {
            return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
                "worker returned inconsistent trade signs for market {}",
                market_name
            )));
        }
    }

    Ok(netted)
}

fn build_initial_holdings(
    balances: &HashMap<&str, f64>,
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> BalanceMap {
    let mut holdings = BalanceMap::new();
    for (_, market) in slot0_results {
        holdings.insert(market.name, lookup_balance(balances, market.name));
    }
    holdings
}

fn replay_buy(
    sims: &mut [super::super::sim::PoolSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    market_name: &'static str,
    amount: f64,
) -> Result<(), ForecastFlowsTranslationError> {
    if amount <= DUST {
        return Ok(());
    }
    let idx = *sim_idx_by_market.get(market_name).ok_or_else(|| {
        ForecastFlowsTranslationError::InvalidResponse(format!(
            "missing sim for market {}",
            market_name
        ))
    })?;
    let (bought, cost, new_price) = sims[idx].buy_exact(amount).ok_or_else(|| {
        ForecastFlowsTranslationError::InvalidResponse(format!(
            "local replay buy failed for market {}",
            market_name
        ))
    })?;
    if bought + EPS < amount || bought <= DUST || !cost.is_finite() || !new_price.is_finite() {
        return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
            "local replay buy was infeasible for market {}",
            market_name
        )));
    }
    if *cash + EPS < cost {
        return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
            "local replay rejected buy due to insufficient cash for market {}",
            market_name
        )));
    }
    sims[idx].set_price(new_price);
    *cash -= cost;
    *sim_balances.entry(market_name).or_insert(0.0) += bought;
    actions.push(Action::Buy {
        market_name,
        amount: bought,
        cost,
    });
    Ok(())
}

fn replay_sell(
    sims: &mut [super::super::sim::PoolSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    market_name: &'static str,
    amount: f64,
) -> Result<(), ForecastFlowsTranslationError> {
    if amount <= DUST {
        return Ok(());
    }
    let available = sim_balances
        .get(market_name)
        .copied()
        .unwrap_or(0.0)
        .max(0.0);
    if available + EPS < amount {
        return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
            "local replay rejected sell due to insufficient holdings for market {}",
            market_name
        )));
    }
    let idx = *sim_idx_by_market.get(market_name).ok_or_else(|| {
        ForecastFlowsTranslationError::InvalidResponse(format!(
            "missing sim for market {}",
            market_name
        ))
    })?;
    let (sold, proceeds, new_price) = sims[idx].sell_exact(amount).ok_or_else(|| {
        ForecastFlowsTranslationError::InvalidResponse(format!(
            "local replay sell failed for market {}",
            market_name
        ))
    })?;
    if sold + EPS < amount || sold <= DUST || !proceeds.is_finite() || !new_price.is_finite() {
        return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
            "local replay sell was infeasible for market {}",
            market_name
        )));
    }
    sims[idx].set_price(new_price);
    *cash += proceeds;
    let entry = sim_balances.entry(market_name).or_insert(0.0);
    *entry = (*entry - sold).max(0.0);
    actions.push(Action::Sell {
        market_name,
        amount: sold,
        proceeds,
    });
    Ok(())
}

fn sanitize_nonnegative(value: f64, field: &str) -> Result<f64, ForecastFlowsTranslationError> {
    if !value.is_finite() {
        return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
            "{field} is not finite"
        )));
    }
    if value < -EPS {
        return Err(ForecastFlowsTranslationError::InvalidResponse(format!(
            "{field} must be nonnegative"
        )));
    }
    Ok(value.max(0.0))
}

fn representative_complete_set_market(
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Option<&'static str> {
    let mut markets: Vec<&'static str> = slot0_results
        .iter()
        .map(|(_, market)| market.name)
        .collect();
    markets.sort_unstable();
    markets.first().copied()
}

fn current_single_tick_geometry(
    slot0: &Slot0Result,
    market: &'static MarketData,
) -> Option<(u128, i32, i32)> {
    let pool = market.pool.as_ref()?;
    let mut tick_lo = None;
    let mut tick_hi = None;
    let mut active_liquidity = 0i128;

    for tick in pool.ticks {
        if tick.tick_idx <= slot0.tick {
            active_liquidity = active_liquidity.checked_add(tick.liquidity_net)?;
            if tick_lo.is_none_or(|best| tick.tick_idx > best) {
                tick_lo = Some(tick.tick_idx);
            }
        } else if tick_hi.is_none_or(|best| tick.tick_idx < best) {
            tick_hi = Some(tick.tick_idx);
        }
    }

    if let (Some(lo), Some(hi)) = (tick_lo, tick_hi)
        && active_liquidity > 0
        && lo < hi
    {
        return Some((active_liquidity as u128, lo, hi));
    }
    None
}

fn single_tick_limit_prices(
    market: &'static MarketData,
    is_token1_outcome: bool,
    tick_lo: i32,
    tick_hi: i32,
) -> Option<(f64, f64)> {
    let pool = market.pool.as_ref()?;
    let zero_for_one_buy = pool.token0.eq_ignore_ascii_case(market.quote_token);
    let sqrt_lo = get_sqrt_ratio_at_tick(tick_lo.min(tick_hi)).ok()?;
    let sqrt_hi = get_sqrt_ratio_at_tick(tick_lo.max(tick_hi)).ok()?;
    let (buy_limit_sqrt, sell_limit_sqrt) = if zero_for_one_buy {
        (sqrt_lo, sqrt_hi)
    } else {
        (sqrt_hi, sqrt_lo)
    };
    let buy_limit_price = sqrt_price_x96_to_price_outcome(buy_limit_sqrt, is_token1_outcome)
        .and_then(|value| u128::try_from(value).ok())
        .map(|wad| wad as f64 / 1e18)?;
    let sell_limit_price = sqrt_price_x96_to_price_outcome(sell_limit_sqrt, is_token1_outcome)
        .and_then(|value| u128::try_from(value).ok())
        .map(|wad| wad as f64 / 1e18)?;
    Some((buy_limit_price, sell_limit_price))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use alloy::primitives::{Address, U256};

    use super::*;
    use crate::markets::{MarketData, Pool, Tick};
    use crate::pools::prediction_to_sqrt_price_x96;

    fn leak_market(market: MarketData) -> &'static MarketData {
        Box::leak(Box::new(market))
    }

    fn mock_slot0_market_with_liquidity_and_ticks(
        name: &'static str,
        outcome_token: &'static str,
        price_fraction: f64,
        liquidity: u128,
        tick_lo: i32,
        tick_hi: i32,
    ) -> (Slot0Result, &'static MarketData) {
        let liq_i128 = i128::try_from(liquidity).unwrap_or(i128::MAX);
        let ticks = Box::leak(Box::new([
            Tick {
                tick_idx: tick_lo.min(tick_hi),
                liquidity_net: liq_i128,
            },
            Tick {
                tick_idx: tick_lo.max(tick_hi),
                liquidity_net: -liq_i128,
            },
        ]));
        let liquidity_str = Box::leak(liquidity.to_string().into_boxed_str());
        let market = leak_market(MarketData {
            name,
            market_id: "0xmarket-contract",
            outcome_token,
            pool: Some(Pool {
                token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
                token1: outcome_token,
                pool_id: "0xpool",
                liquidity: liquidity_str,
                ticks,
            }),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        });
        let slot0 = Slot0Result {
            pool_id: Address::ZERO,
            sqrt_price_x96: prediction_to_sqrt_price_x96(price_fraction, true)
                .unwrap_or(U256::from(1u128 << 96)),
            tick: 0,
            observation_index: 0,
            observation_cardinality: 0,
            observation_cardinality_next: 0,
            fee_protocol: 0,
            unlocked: true,
        };
        (slot0, market)
    }

    fn two_market_fixture() -> (
        Vec<(Slot0Result, &'static MarketData)>,
        HashMap<&'static str, f64>,
        HashMap<String, f64>,
    ) {
        let (slot0_a, market_a) = mock_slot0_market_with_liquidity_and_ticks(
            "M1",
            "0x1111111111111111111111111111111111111111",
            0.1,
            1_000_000_000_000_000_000_000u128,
            -16_096,
            92_108,
        );
        let (slot0_b, market_b) = mock_slot0_market_with_liquidity_and_ticks(
            "M2",
            "0x2222222222222222222222222222222222222222",
            0.9,
            1_000_000_000_000_000_000_000u128,
            -16_096,
            92_108,
        );
        let slot0_results = vec![(slot0_a, market_a), (slot0_b, market_b)];
        let balances = HashMap::from([("M1", 0.0), ("M2", 0.1)]);
        let predictions = HashMap::from([("m1".to_string(), 0.9), ("m2".to_string(), 0.1)]);
        (slot0_results, balances, predictions)
    }

    fn certified_direct_result(trades: Vec<PredictionMarketTrade>) -> CompareResult {
        CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "direct_only".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                trades,
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "uncertified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary {
                    passed: false,
                }),
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
        }
    }

    #[test]
    fn build_problem_request_uses_single_range_univ3_fee_mapping() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let problem = build_problem_request(&balances, 1.0, &slot0_results, &predictions, 2)
            .expect("problem request should build");

        assert_eq!(problem.outcomes.len(), 2);
        assert_eq!(problem.markets.len(), 2);
        let MarketSpecRequest::UniV3 {
            market_id,
            current_price,
            bands,
            fee_multiplier,
            ..
        } = &problem.markets[0];
        assert_eq!(market_id, "M1");
        assert!((*current_price - 0.1).abs() < 0.02);
        assert_eq!(bands.len(), 2);
        assert!(bands[0].lower_price > bands[1].lower_price);
        assert!(bands[0].liquidity_l > 0.0);
        assert_eq!(bands[1].liquidity_l, 0.0);
        assert!((*fee_multiplier - 0.9999).abs() < 1e-12);
    }

    #[test]
    fn build_problem_request_rejects_non_derivable_active_range_geometry() {
        let (mut slot0_results, balances, predictions) = two_market_fixture();
        slot0_results[0].0.tick = 100_000;
        let err = build_problem_request(&balances, 1.0, &slot0_results, &predictions, 2)
            .expect_err("non-derivable active range should be rejected");
        assert!(
            err.to_string()
                .contains("does not have replayable single-range geometry")
        );
    }

    #[test]
    fn translate_compare_result_maps_trade_signs_and_orders_direct_actions() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let compare = certified_direct_result(vec![
            PredictionMarketTrade {
                market_id: "M2".to_string(),
                outcome_id: "0x2222222222222222222222222222222222222222".to_string(),
                collateral_delta: 0.03,
                outcome_delta: -0.02,
            },
            PredictionMarketTrade {
                market_id: "M1".to_string(),
                outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                collateral_delta: -0.01,
                outcome_delta: 0.01,
            },
        ]);

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("translation should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Direct);
        assert!(matches!(
            candidates[0].actions.first(),
            Some(Action::Sell {
                market_name: "M2",
                ..
            })
        ));
        assert!(matches!(
            candidates[0].actions.get(1),
            Some(Action::Buy {
                market_name: "M1",
                ..
            })
        ));
    }

    #[test]
    fn translate_compare_result_rejects_dual_split_merge() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let compare = CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "uncertified".to_string(),
                mode: "direct_only".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary {
                    passed: false,
                }),
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan {
                    mint: 0.1,
                    merge: 0.1,
                },
            },
        };

        let err = translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
            .expect_err("translation should reject mixed mint+merge");
        assert!(
            err.to_string()
                .contains("cannot mint and merge in the same candidate")
        );
    }

    #[test]
    fn translate_compare_result_keeps_valid_mixed_when_direct_is_malformed() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let compare = CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                trades: vec![PredictionMarketTrade {
                    market_id: "M1".to_string(),
                    outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                    collateral_delta: -0.02,
                    outcome_delta: 0.03,
                }],
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
        };

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("valid mixed candidate should survive malformed direct branch");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Mixed);
    }

    #[test]
    fn translate_compare_result_keeps_valid_direct_when_mixed_is_malformed() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let compare = CompareResult {
            direct_only: certified_direct_result(vec![PredictionMarketTrade {
                market_id: "M1".to_string(),
                outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                collateral_delta: -0.02,
                outcome_delta: 0.03,
            }])
            .direct_only,
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan {
                    mint: 0.1,
                    merge: 0.1,
                },
            },
        };

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("valid direct candidate should survive malformed mixed branch");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Direct);
    }

    #[test]
    fn translate_compare_result_orders_buy_merge_sell_actions() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let compare = CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "uncertified".to_string(),
                mode: "direct_only".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary {
                    passed: false,
                }),
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                trades: vec![
                    PredictionMarketTrade {
                        market_id: "M1".to_string(),
                        outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                        collateral_delta: -0.03,
                        outcome_delta: 0.06,
                    },
                    PredictionMarketTrade {
                        market_id: "M2".to_string(),
                        outcome_id: "0x2222222222222222222222222222222222222222".to_string(),
                        collateral_delta: 0.02,
                        outcome_delta: -0.02,
                    },
                ],
                split_merge: super::super::protocol::SplitMergePlan {
                    mint: 0.0,
                    merge: 0.05,
                },
            },
        };

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("translation should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Mixed);
        assert!(matches!(
            candidates[0].actions.first(),
            Some(Action::Buy {
                market_name: "M1",
                ..
            })
        ));
        assert!(matches!(
            candidates[0].actions.get(1),
            Some(Action::Merge { .. })
        ));
        assert!(matches!(
            candidates[0].actions.get(2),
            Some(Action::Sell {
                market_name: "M2",
                ..
            })
        ));
    }

    #[test]
    fn translate_compare_result_nets_opposite_signed_trades_by_market() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let compare = certified_direct_result(vec![
            PredictionMarketTrade {
                market_id: "M1".to_string(),
                outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                collateral_delta: -0.03,
                outcome_delta: 0.05,
            },
            PredictionMarketTrade {
                market_id: "M1".to_string(),
                outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                collateral_delta: 0.01,
                outcome_delta: -0.02,
            },
        ]);

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("translation should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].actions.len(), 1);
        assert!(matches!(
            candidates[0].actions[0],
            Action::Buy {
                market_name: "M1",
                ..
            }
        ));
    }
}
