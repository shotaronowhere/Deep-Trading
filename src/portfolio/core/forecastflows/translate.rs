use std::collections::{HashMap, HashSet};
use std::fmt;

use uniswap_v3_math::tick_math::get_sqrt_ratio_at_tick;
#[cfg(any(test, feature = "benchmark_synthetic_fixtures"))]
use uniswap_v3_math::tick_math::get_tick_at_sqrt_ratio;

use crate::markets::MarketData;
use crate::pools::{FEE_PIPS, Slot0Result, normalize_market_name, sqrt_price_x96_to_price_outcome};

use super::protocol::{
    CompareResult, MarketSpecRequest, OutcomeSpecRequest, PredictionMarketProblemRequest,
    PredictionMarketSolveResult, PredictionMarketTrade, UniV3LiquidityBandRequest,
};
use super::{ForecastFlowsCandidateVariant, ForecastFlowsFamilyCandidate};
use crate::portfolio::Action;

use super::super::sim::{DUST, EPS, FEE_FACTOR};
use super::super::types::{BalanceMap, lookup_balance};

const REPLAY_AMOUNT_REL_TOL: f64 = 1e-6;
const REPLAY_AMOUNT_ABS_TOL: f64 = 1e-9;
const MAX_ROUTE_REPLAY_ROUNDS: usize = 256;
const MINT_CASH_ROUNDING_BUFFER: f64 = 1e-6;
const REPLAY_SELL_PROCEEDS_HAIRCUT_BPS: f64 = 1.0;

#[derive(Debug)]
pub(super) enum ForecastFlowsTranslationError {
    UnsupportedSnapshot(String),
    InvalidCertifiedResponse(String),
    InvalidReplayResponse(String),
}

impl fmt::Display for ForecastFlowsTranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSnapshot(message) => write!(f, "{message}"),
            Self::InvalidCertifiedResponse(message) => write!(f, "{message}"),
            Self::InvalidReplayResponse(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ForecastFlowsTranslationError {}

impl ForecastFlowsTranslationError {
    fn invalid_response(message: impl Into<String>) -> Self {
        Self::InvalidCertifiedResponse(message.into())
    }

    fn invalid_replay(message: impl Into<String>) -> Self {
        Self::InvalidReplayResponse(message.into())
    }

    fn drop_stage(&self) -> VariantDropStage {
        match self {
            Self::InvalidReplayResponse(_) => VariantDropStage::Replay,
            _ => VariantDropStage::Certified,
        }
    }
}

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
        let bands = build_univ3_liquidity_bands(slot0, market).ok_or_else(|| {
            ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
                "market {} does not have replayable contiguous liquidity ladder geometry",
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
            bands,
            fee_multiplier: 1.0 - (FEE_PIPS as f64 / 1_000_000.0),
        });
    }

    let collateral_balance = susds_balance.max(0.0);

    Ok(PredictionMarketProblemRequest {
        outcomes,
        collateral_balance,
        markets,
        split_bound: conservative_split_bound(collateral_balance, slot0_results),
    })
}

fn conservative_split_bound(
    collateral_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Option<f64> {
    let distinct_market_count = slot0_results
        .iter()
        .map(|(_, market)| market.market_id)
        .collect::<HashSet<_>>()
        .len();
    if distinct_market_count <= 1 {
        return None;
    }

    // Connected Seer families flatten multiple underlying markets into one
    // ForecastFlows split/merge edge, but the request omits connector / invalid
    // inventory. Letting the worker auto-bound from cash + all tradeable
    // holdings therefore overstates the feasible split/merge budget for the
    // flattened problem and can push the mixed solve into the non-finite
    // never-certified path. Cap the bound to spendable base collateral and
    // quantize down to a whole sUSDS so the worker starts from a stable,
    // conservative budget.
    let quantized = if collateral_balance >= 1.0 {
        collateral_balance.floor()
    } else {
        collateral_balance
    };
    Some(quantized.max(f64::EPSILON))
}

#[cfg(test)]
pub(super) fn translate_compare_result(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    compare: CompareResult,
) -> Result<Vec<ForecastFlowsFamilyCandidate>, ForecastFlowsTranslationError> {
    let report = translate_compare_result_report(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        compare,
    );
    if let Some(message) = report.all_certified_candidates_dropped_message() {
        return Err(ForecastFlowsTranslationError::invalid_response(message));
    }
    Ok(report.candidates)
}

pub(super) struct CompareTranslationReport {
    pub(super) candidates: Vec<ForecastFlowsFamilyCandidate>,
    pub(super) drop_reasons: Vec<VariantDropReason>,
    pub(super) replay_tolerance_clamp_used: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VariantDropStage {
    Certified,
    Replay,
}

impl VariantDropStage {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::Replay => "replay",
        }
    }
}

pub(super) struct VariantDropReason {
    pub(super) variant: ForecastFlowsCandidateVariant,
    pub(super) stage: VariantDropStage,
    pub(super) reason: String,
}

impl CompareTranslationReport {
    pub(super) fn first_non_replay_drop_reason(&self) -> Option<String> {
        self.drop_reasons
            .iter()
            .find(|drop| drop.stage == VariantDropStage::Certified)
            .map(|drop| format!("{}: {}", drop.variant.as_str(), drop.reason))
    }

    pub(super) fn first_replay_drop_reason(&self) -> Option<String> {
        self.drop_reasons
            .iter()
            .find(|drop| drop.stage == VariantDropStage::Replay)
            .map(|drop| format!("{}: {}", drop.variant.as_str(), drop.reason))
    }

    pub(super) fn all_certified_candidates_dropped_message(&self) -> Option<String> {
        if !self.candidates.is_empty() || self.drop_reasons.is_empty() {
            return None;
        }
        let joined = self
            .drop_reasons
            .iter()
            .map(|drop| {
                format!(
                    "{}({}): {}",
                    drop.variant.as_str(),
                    drop.stage.as_str(),
                    drop.reason
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "all certified ForecastFlows candidates were dropped during translation: {joined}"
        ))
    }
}

pub(super) fn translate_compare_result_report(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    compare: CompareResult,
) -> CompareTranslationReport {
    let mut candidates = Vec::new();
    let mut drop_reasons = Vec::new();
    let mut replay_tolerance_clamp_used = false;

    collect_variant_translation(
        &mut candidates,
        &mut drop_reasons,
        &mut replay_tolerance_clamp_used,
        balances,
        susds_balance,
        slot0_results,
        predictions,
        &compare.direct_only,
        ForecastFlowsCandidateVariant::Direct,
    );
    collect_variant_translation(
        &mut candidates,
        &mut drop_reasons,
        &mut replay_tolerance_clamp_used,
        balances,
        susds_balance,
        slot0_results,
        predictions,
        &compare.mixed_enabled,
        ForecastFlowsCandidateVariant::Mixed,
    );

    CompareTranslationReport {
        candidates,
        drop_reasons,
        replay_tolerance_clamp_used,
    }
}

fn collect_variant_translation(
    candidates: &mut Vec<ForecastFlowsFamilyCandidate>,
    drop_reasons: &mut Vec<VariantDropReason>,
    replay_tolerance_clamp_used: &mut bool,
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
        replay_tolerance_clamp_used,
    ) {
        Ok(Some(candidate)) => candidates.push(candidate),
        Ok(None) => {}
        Err(err) => {
            if result.is_certified() {
                drop_reasons.push(VariantDropReason {
                    variant,
                    stage: err.drop_stage(),
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
    _predictions: &HashMap<String, f64>,
    result: &PredictionMarketSolveResult,
    variant: ForecastFlowsCandidateVariant,
    replay_tolerance_clamp_used: &mut bool,
) -> Result<Option<ForecastFlowsFamilyCandidate>, ForecastFlowsTranslationError> {
    if !result.is_certified() {
        return Ok(None);
    }
    match variant {
        ForecastFlowsCandidateVariant::Direct if result.mode != "direct_only" => {
            return Err(ForecastFlowsTranslationError::invalid_response(format!(
                "direct candidate returned unexpected mode {}",
                result.mode
            )));
        }
        ForecastFlowsCandidateVariant::Mixed if result.mode != "mixed_enabled" => {
            return Err(ForecastFlowsTranslationError::invalid_response(format!(
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
        return Err(ForecastFlowsTranslationError::invalid_response(
            "mixed result cannot mint and merge in the same candidate".to_string(),
        ));
    }

    let mut sims = build_replay_markets(slot0_results)?;
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
    let (contract_1, contract_2) = action_contract_pair_for_replay(&sims);
    let mut actions = Vec::new();

    let mut ordered_route = ordered_trade_route(slot0_results, &net_trades);
    let mut merge_remaining = merge_amount;

    replay_direct_sells(
        &mut sims,
        &sim_idx_by_market,
        &mut sim_balances,
        &mut cash,
        &mut actions,
        &mut ordered_route,
        replay_tolerance_clamp_used,
    )?;
    replay_direct_merges(
        &mut sim_balances,
        &mut cash,
        &mut actions,
        &ordered_route,
        &mut merge_remaining,
        contract_1,
        contract_2,
        representative_market,
    )?;
    replay_mint_rounds(
        slot0_results,
        &mut sims,
        &sim_idx_by_market,
        &mut sim_balances,
        &mut cash,
        &mut actions,
        &mut ordered_route,
        mint_amount,
        contract_1,
        contract_2,
        representative_market,
        replay_tolerance_clamp_used,
    )?;
    replay_buy_merge_rounds(
        &mut sims,
        &sim_idx_by_market,
        &mut sim_balances,
        &mut cash,
        &mut actions,
        &mut ordered_route,
        &mut merge_remaining,
        contract_1,
        contract_2,
        representative_market,
    )?;
    replay_remaining_buys(
        &mut sims,
        &sim_idx_by_market,
        &mut sim_balances,
        &mut cash,
        &mut actions,
        &mut ordered_route,
    )?;

    Ok(Some(ForecastFlowsFamilyCandidate {
        actions,
        variant,
        estimated_execution_cost_susd: result.estimated_execution_cost,
        estimated_net_ev_susd: result.net_ev,
    }))
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

struct OrderedTradeRoute {
    market_name: &'static str,
    buy_remaining: f64,
    sell_remaining: f64,
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

fn ordered_trade_route(
    slot0_results: &[(Slot0Result, &'static MarketData)],
    net_trades: &NettedTrades,
) -> Vec<OrderedTradeRoute> {
    let buy_by_market = net_trades
        .buys
        .iter()
        .map(|trade| (trade.market_name, trade.amount))
        .collect::<HashMap<_, _>>();
    let sell_by_market = net_trades
        .sells
        .iter()
        .map(|trade| (trade.market_name, trade.amount))
        .collect::<HashMap<_, _>>();

    slot0_results
        .iter()
        .map(|(_, market)| OrderedTradeRoute {
            market_name: market.name,
            buy_remaining: buy_by_market.get(market.name).copied().unwrap_or(0.0),
            sell_remaining: sell_by_market.get(market.name).copied().unwrap_or(0.0),
        })
        .collect()
}

fn replay_direct_sells(
    sims: &mut [ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    ordered_route: &mut [OrderedTradeRoute],
    replay_tolerance_clamp_used: &mut bool,
) -> Result<(), ForecastFlowsTranslationError> {
    for route in ordered_route {
        let desired = route.sell_remaining;
        if desired <= DUST || amounts_match_within_replay_tolerance(desired, 0.0) {
            continue;
        }
        let idx = *sim_idx_by_market.get(route.market_name).ok_or_else(|| {
            ForecastFlowsTranslationError::invalid_response(format!(
                "missing sim for market {}",
                route.market_name
            ))
        })?;
        let available = sim_balances
            .get(route.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let feasible = desired
            .min(available)
            .min(sims[idx].max_sell_tokens().max(0.0));
        if feasible <= DUST || amounts_match_within_replay_tolerance(feasible, 0.0) {
            continue;
        }
        let requested = if amounts_match_within_replay_tolerance(desired, feasible) {
            desired
        } else {
            feasible
        };
        replay_sell(
            sims,
            sim_idx_by_market,
            sim_balances,
            cash,
            actions,
            route.market_name,
            requested,
            replay_tolerance_clamp_used,
        )?;
        route.sell_remaining = (route.sell_remaining - feasible).max(0.0);
    }
    Ok(())
}

fn replay_direct_merges(
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    ordered_route: &[OrderedTradeRoute],
    merge_remaining: &mut f64,
    contract_1: &'static str,
    contract_2: &'static str,
    representative_market: &'static str,
) -> Result<(), ForecastFlowsTranslationError> {
    for _ in 0..MAX_ROUTE_REPLAY_ROUNDS {
        if *merge_remaining <= DUST || amounts_match_within_replay_tolerance(*merge_remaining, 0.0)
        {
            return Ok(());
        }
        let direct_merge = ordered_route
            .iter()
            .map(|route| {
                sim_balances
                    .get(route.market_name)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0)
            })
            .fold(*merge_remaining, f64::min);
        if direct_merge <= DUST || amounts_match_within_replay_tolerance(direct_merge, 0.0) {
            return Ok(());
        }
        replay_merge(
            sim_balances,
            cash,
            actions,
            ordered_route.iter().map(|route| route.market_name),
            direct_merge,
            contract_1,
            contract_2,
            representative_market,
        )?;
        *merge_remaining = (*merge_remaining - direct_merge).max(0.0);
    }

    Err(ForecastFlowsTranslationError::invalid_replay(
        "local replay exceeded direct merge round limit".to_string(),
    ))
}

fn replay_mint_rounds(
    slot0_results: &[(Slot0Result, &'static MarketData)],
    sims: &mut [ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    ordered_route: &mut [OrderedTradeRoute],
    mint_amount: f64,
    contract_1: &'static str,
    contract_2: &'static str,
    representative_market: &'static str,
    replay_tolerance_clamp_used: &mut bool,
) -> Result<(), ForecastFlowsTranslationError> {
    let mut mint_remaining = mint_amount;
    for _ in 0..MAX_ROUTE_REPLAY_ROUNDS {
        if mint_remaining <= DUST
            || amounts_match_within_replay_tolerance(mint_remaining, 0.0)
            || *cash <= DUST
        {
            return Ok(());
        }
        let cash_budget = (*cash - MINT_CASH_ROUNDING_BUFFER).max(0.0);
        let round_amount = mint_remaining.min(cash_budget);
        if round_amount <= DUST || amounts_match_within_replay_tolerance(round_amount, 0.0) {
            return Ok(());
        }
        replay_mint(
            slot0_results,
            sim_balances,
            cash,
            actions,
            round_amount,
            contract_1,
            contract_2,
            representative_market,
        )?;
        mint_remaining = (mint_remaining - round_amount).max(0.0);

        for route in ordered_route.iter_mut() {
            let desired = route.sell_remaining.min(round_amount);
            if desired <= DUST || amounts_match_within_replay_tolerance(desired, 0.0) {
                continue;
            }
            let idx = *sim_idx_by_market.get(route.market_name).ok_or_else(|| {
                ForecastFlowsTranslationError::invalid_response(format!(
                    "missing sim for market {}",
                    route.market_name
                ))
            })?;
            let available = sim_balances
                .get(route.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let feasible = desired
                .min(available)
                .min(sims[idx].max_sell_tokens().max(0.0));
            if feasible <= DUST || amounts_match_within_replay_tolerance(feasible, 0.0) {
                continue;
            }
            let requested = if amounts_match_within_replay_tolerance(desired, feasible) {
                desired
            } else {
                feasible
            };
            replay_sell(
                sims,
                sim_idx_by_market,
                sim_balances,
                cash,
                actions,
                route.market_name,
                requested,
                replay_tolerance_clamp_used,
            )?;
            route.sell_remaining = (route.sell_remaining - feasible).max(0.0);
        }
    }

    Err(ForecastFlowsTranslationError::invalid_replay(
        "local replay exceeded mint round limit".to_string(),
    ))
}

fn replay_buy_merge_rounds(
    sims: &mut [ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    ordered_route: &mut [OrderedTradeRoute],
    merge_remaining: &mut f64,
    contract_1: &'static str,
    contract_2: &'static str,
    representative_market: &'static str,
) -> Result<(), ForecastFlowsTranslationError> {
    for _ in 0..MAX_ROUTE_REPLAY_ROUNDS {
        if *merge_remaining <= DUST || amounts_match_within_replay_tolerance(*merge_remaining, 0.0)
        {
            return Ok(());
        }
        let round_amount = affordable_merge_round(
            sims,
            sim_idx_by_market,
            sim_balances,
            ordered_route,
            *merge_remaining,
            *cash,
        )?;
        if round_amount <= DUST || amounts_match_within_replay_tolerance(round_amount, 0.0) {
            return Ok(());
        }

        for route in ordered_route.iter_mut() {
            let holding = sim_balances
                .get(route.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let shortfall = (round_amount - holding).max(0.0);
            if shortfall <= DUST || amounts_match_within_replay_tolerance(shortfall, 0.0) {
                continue;
            }
            replay_buy(
                sims,
                sim_idx_by_market,
                sim_balances,
                cash,
                actions,
                route.market_name,
                shortfall,
            )?;
            route.buy_remaining = (route.buy_remaining - shortfall).max(0.0);
        }

        replay_merge(
            sim_balances,
            cash,
            actions,
            ordered_route.iter().map(|route| route.market_name),
            round_amount,
            contract_1,
            contract_2,
            representative_market,
        )?;
        *merge_remaining = (*merge_remaining - round_amount).max(0.0);
    }

    Err(ForecastFlowsTranslationError::invalid_replay(
        "local replay exceeded buy-merge round limit".to_string(),
    ))
}

fn replay_remaining_buys(
    sims: &mut [ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    ordered_route: &mut [OrderedTradeRoute],
) -> Result<(), ForecastFlowsTranslationError> {
    for route in ordered_route {
        let desired = route.buy_remaining;
        if desired <= DUST || amounts_match_within_replay_tolerance(desired, 0.0) {
            continue;
        }
        let idx = *sim_idx_by_market.get(route.market_name).ok_or_else(|| {
            ForecastFlowsTranslationError::invalid_response(format!(
                "missing sim for market {}",
                route.market_name
            ))
        })?;
        let feasible = affordable_buy_amount(&sims[idx], desired, *cash);
        if feasible <= DUST || amounts_match_within_replay_tolerance(feasible, 0.0) {
            continue;
        }
        replay_buy(
            sims,
            sim_idx_by_market,
            sim_balances,
            cash,
            actions,
            route.market_name,
            feasible,
        )?;
        route.buy_remaining = (route.buy_remaining - feasible).max(0.0);
    }
    Ok(())
}

fn replay_mint(
    slot0_results: &[(Slot0Result, &'static MarketData)],
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    amount: f64,
    contract_1: &'static str,
    contract_2: &'static str,
    representative_market: &'static str,
) -> Result<(), ForecastFlowsTranslationError> {
    if *cash + EPS < amount {
        return Err(ForecastFlowsTranslationError::invalid_replay(
            "local replay rejected mint amount due to insufficient cash".to_string(),
        ));
    }
    *cash -= amount;
    for (_, market) in slot0_results {
        *sim_balances.entry(market.name).or_insert(0.0) += amount;
    }
    actions.push(Action::Mint {
        contract_1,
        contract_2,
        amount,
        target_market: representative_market,
    });
    Ok(())
}

fn replay_merge(
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    market_names: impl IntoIterator<Item = &'static str>,
    amount: f64,
    contract_1: &'static str,
    contract_2: &'static str,
    representative_market: &'static str,
) -> Result<(), ForecastFlowsTranslationError> {
    let market_names = market_names.into_iter().collect::<Vec<_>>();
    for market_name in &market_names {
        let holding = sim_balances
            .get(market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        if holding + EPS < amount {
            return Err(ForecastFlowsTranslationError::invalid_replay(format!(
                "local replay rejected merge amount due to insufficient holdings for {}",
                market_name
            )));
        }
    }
    for market_name in market_names {
        let holding = sim_balances.entry(market_name).or_insert(0.0);
        *holding = (*holding - amount).max(0.0);
    }
    *cash += amount;
    actions.push(Action::Merge {
        contract_1,
        contract_2,
        amount,
        source_market: representative_market,
    });
    Ok(())
}

fn affordable_buy_amount(sim: &ReplayMarketSim, desired: f64, cash: f64) -> f64 {
    let upper = desired.min(sim.max_buy_tokens().max(0.0));
    if upper <= DUST || amounts_match_within_replay_tolerance(upper, 0.0) {
        return 0.0;
    }
    if let Some((bought, cost, _)) = sim.buy_exact(upper)
        && bought + EPS >= upper
        && cost <= cash + EPS
    {
        return upper;
    }

    let mut lo = 0.0;
    let mut hi = upper;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        let affordable = sim
            .buy_exact(mid)
            .is_some_and(|(bought, cost, _)| bought + EPS >= mid && cost <= cash + EPS);
        if affordable {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Simulates a merge round's sequential buys using the same FP operations
/// as replay_buy. Returns true iff all buys are affordable with the given cash.
/// This ensures algorithmic symmetry: the binary search predicate and the
/// execution path agree at every floating-point bit.
fn simulate_merge_round_affordable(
    sims: &[ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &BalanceMap,
    ordered_route: &[OrderedTradeRoute],
    amount: f64,
    cash: f64,
) -> Result<bool, ForecastFlowsTranslationError> {
    let mut simulated_cash = cash;
    for route in ordered_route {
        let holding = sim_balances
            .get(route.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let shortfall = (amount - holding).max(0.0);
        if shortfall <= DUST || amounts_match_within_replay_tolerance(shortfall, 0.0) {
            continue;
        }
        let idx = *sim_idx_by_market.get(route.market_name).ok_or_else(|| {
            ForecastFlowsTranslationError::invalid_response(format!(
                "missing sim for market {}",
                route.market_name
            ))
        })?;
        let Some((bought, cost, _)) = sims[idx].buy_exact(shortfall) else {
            return Ok(false);
        };
        if bought + EPS < shortfall || !cost.is_finite() {
            return Ok(false);
        }
        // Identical to replay_buy's cash check: *cash + EPS < cost
        if simulated_cash + EPS < cost {
            return Ok(false);
        }
        // Identical to replay_buy's cash update: *cash -= cost
        simulated_cash -= cost;
    }
    Ok(true)
}

fn affordable_merge_round(
    sims: &[ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &BalanceMap,
    ordered_route: &[OrderedTradeRoute],
    merge_remaining: f64,
    cash: f64,
) -> Result<f64, ForecastFlowsTranslationError> {
    let mut upper = merge_remaining;
    for route in ordered_route {
        let idx = *sim_idx_by_market.get(route.market_name).ok_or_else(|| {
            ForecastFlowsTranslationError::invalid_response(format!(
                "missing sim for market {}",
                route.market_name
            ))
        })?;
        let holding = sim_balances
            .get(route.market_name)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        upper = upper
            .min(holding + route.buy_remaining)
            .min(holding + sims[idx].max_buy_tokens().max(0.0));
    }
    if upper <= DUST || amounts_match_within_replay_tolerance(upper, 0.0) {
        return Ok(0.0);
    }

    // Fast path: check if the full amount is affordable using sequential simulation
    if simulate_merge_round_affordable(
        sims,
        sim_idx_by_market,
        sim_balances,
        ordered_route,
        upper,
        cash,
    )? {
        return Ok(upper);
    }

    // Binary search using the same sequential FP operations as replay_buy
    let mut lo = 0.0;
    let mut hi = upper;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if simulate_merge_round_affordable(
            sims,
            sim_idx_by_market,
            sim_balances,
            ordered_route,
            mid,
            cash,
        )? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

fn net_worker_trades(
    market_catalog: &HashMap<&'static str, &'static str>,
    trades: &[PredictionMarketTrade],
) -> Result<NettedTrades, ForecastFlowsTranslationError> {
    let mut net_by_market: HashMap<&'static str, (f64, f64)> = HashMap::new();
    for trade in trades {
        if !trade.collateral_delta.is_finite() || !trade.outcome_delta.is_finite() {
            return Err(ForecastFlowsTranslationError::invalid_response(
                "worker returned a non-finite trade delta".to_string(),
            ));
        }
        let Some((&market_name, &outcome_id)) = market_catalog
            .iter()
            .find(|(market_name, _)| **market_name == trade.market_id)
        else {
            return Err(ForecastFlowsTranslationError::invalid_response(format!(
                "worker returned unknown market_id {}",
                trade.market_id
            )));
        };
        if outcome_id != trade.outcome_id {
            return Err(ForecastFlowsTranslationError::invalid_response(format!(
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
            return Err(ForecastFlowsTranslationError::invalid_response(format!(
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
    sims: &mut [ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    market_name: &'static str,
    amount: f64,
) -> Result<(), ForecastFlowsTranslationError> {
    if amount <= DUST || amounts_match_within_replay_tolerance(amount, 0.0) {
        return Ok(());
    }
    let idx = *sim_idx_by_market.get(market_name).ok_or_else(|| {
        ForecastFlowsTranslationError::invalid_response(format!(
            "missing sim for market {}",
            market_name
        ))
    })?;
    let (bought, cost, new_price) = sims[idx].buy_exact(amount).ok_or_else(|| {
        ForecastFlowsTranslationError::invalid_replay(format!(
            "local replay buy failed for market {}",
            market_name
        ))
    })?;
    if bought + EPS < amount || bought <= DUST || !cost.is_finite() || !new_price.is_finite() {
        return Err(ForecastFlowsTranslationError::invalid_replay(format!(
            "local replay buy was infeasible for market {}",
            market_name
        )));
    }
    if *cash + EPS < cost {
        return Err(ForecastFlowsTranslationError::invalid_replay(format!(
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
    sims: &mut [ReplayMarketSim],
    sim_idx_by_market: &HashMap<&'static str, usize>,
    sim_balances: &mut BalanceMap,
    cash: &mut f64,
    actions: &mut Vec<Action>,
    market_name: &'static str,
    amount: f64,
    replay_tolerance_clamp_used: &mut bool,
) -> Result<(), ForecastFlowsTranslationError> {
    if amount <= DUST || amounts_match_within_replay_tolerance(amount, 0.0) {
        return Ok(());
    }
    let available = sim_balances
        .get(market_name)
        .copied()
        .unwrap_or(0.0)
        .max(0.0);
    let requested_amount = if available + EPS < amount {
        if amounts_match_within_replay_tolerance(amount, available) {
            *replay_tolerance_clamp_used = true;
            available
        } else {
            return Err(ForecastFlowsTranslationError::invalid_replay(format!(
                "local replay rejected sell due to insufficient holdings for market {} (requested={}, available={})",
                market_name, amount, available
            )));
        }
    } else {
        amount
    };
    let idx = *sim_idx_by_market.get(market_name).ok_or_else(|| {
        ForecastFlowsTranslationError::invalid_response(format!(
            "missing sim for market {}",
            market_name
        ))
    })?;
    let (sold, proceeds, new_price) = sims[idx].sell_exact(requested_amount).ok_or_else(|| {
        ForecastFlowsTranslationError::invalid_replay(format!(
            "local replay sell failed for market {}",
            market_name
        ))
    })?;
    if sold + EPS < requested_amount {
        if amounts_match_within_replay_tolerance(sold, requested_amount) {
            *replay_tolerance_clamp_used = true;
        } else {
            return Err(ForecastFlowsTranslationError::invalid_replay(format!(
                "local replay sell was infeasible for market {} (requested={}, replayed={})",
                market_name, requested_amount, sold
            )));
        }
    }
    if sold <= DUST || !proceeds.is_finite() || !new_price.is_finite() {
        return Err(ForecastFlowsTranslationError::invalid_replay(format!(
            "local replay sell was infeasible for market {} (requested={}, replayed={})",
            market_name, requested_amount, sold
        )));
    }
    sims[idx].set_price(new_price);
    let conservative_proceeds = proceeds * (1.0 - REPLAY_SELL_PROCEEDS_HAIRCUT_BPS / 10_000.0);
    *cash += conservative_proceeds;
    let entry = sim_balances.entry(market_name).or_insert(0.0);
    *entry = (*entry - sold).max(0.0);
    actions.push(Action::Sell {
        market_name,
        amount: sold,
        proceeds: conservative_proceeds,
    });
    Ok(())
}

fn sanitize_nonnegative(value: f64, field: &str) -> Result<f64, ForecastFlowsTranslationError> {
    if !value.is_finite() {
        return Err(ForecastFlowsTranslationError::invalid_response(format!(
            "{field} is not finite"
        )));
    }
    if value < -EPS {
        return Err(ForecastFlowsTranslationError::invalid_response(format!(
            "{field} must be nonnegative"
        )));
    }
    Ok(value.max(0.0))
}

fn replay_amount_tolerance(lhs: f64, rhs: f64) -> f64 {
    REPLAY_AMOUNT_ABS_TOL.max(REPLAY_AMOUNT_REL_TOL * lhs.abs().max(rhs.abs()).max(1.0))
}

fn amounts_match_within_replay_tolerance(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= replay_amount_tolerance(lhs, rhs)
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

fn build_replay_markets(
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Result<Vec<ReplayMarketSim>, ForecastFlowsTranslationError> {
    let mut sims = Vec::with_capacity(slot0_results.len());
    for (slot0, market) in slot0_results {
        let sim = ReplayMarketSim::from_slot0(slot0, market).ok_or_else(|| {
            ForecastFlowsTranslationError::UnsupportedSnapshot(format!(
                "failed to build replay market state for {}",
                market.name
            ))
        })?;
        sims.push(sim);
    }
    Ok(sims)
}

fn action_contract_pair_for_replay(sims: &[ReplayMarketSim]) -> (&'static str, &'static str) {
    if sims.is_empty() {
        return ("", "");
    }
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

#[derive(Debug, Clone, Copy)]
struct DerivedLiquidityInterval {
    top_price: f64,
    bottom_price: f64,
    liquidity: u128,
}

#[derive(Debug, Clone)]
struct ReplayMarketSim {
    market_name: &'static str,
    market_id: &'static str,
    current_price: f64,
    intervals: Vec<DerivedLiquidityInterval>,
}

impl ReplayMarketSim {
    fn from_slot0(slot0: &Slot0Result, market: &'static MarketData) -> Option<Self> {
        let pool = market.pool.as_ref()?;
        let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
        let current_price = sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome)
            .and_then(|value| u128::try_from(value).ok())
            .map(|wad| wad as f64 / 1e18)?;
        let intervals = derive_contiguous_liquidity_intervals(slot0, market, current_price)?;
        Some(Self {
            market_name: market.name,
            market_id: market.market_id,
            current_price,
            intervals,
        })
    }

    fn set_price(&mut self, new_price: f64) {
        self.current_price = new_price;
    }

    fn max_sell_tokens(&self) -> f64 {
        let Some(start_idx) = self.interval_index_for_sell() else {
            return 0.0;
        };
        let mut price = self.current_price;
        let mut total = 0.0;
        for interval in self.intervals.iter().skip(start_idx) {
            let Some(tokens) = band_sell_capacity(price, interval) else {
                return 0.0;
            };
            total += tokens;
            price = interval.bottom_price;
        }
        total
    }

    fn max_buy_tokens(&self) -> f64 {
        let Some(start_idx) = self.interval_index_for_buy() else {
            return 0.0;
        };
        let mut price = self.current_price;
        let mut total = 0.0;
        for interval in self.intervals[..=start_idx].iter().rev() {
            let Some(tokens) = band_buy_capacity(price, interval) else {
                return 0.0;
            };
            total += tokens;
            price = interval.top_price;
        }
        total
    }

    fn sell_exact(&self, amount: f64) -> Option<(f64, f64, f64)> {
        if amount <= 0.0 {
            return Some((0.0, 0.0, self.current_price));
        }
        let start_idx = self.interval_index_for_sell()?;
        let mut remaining = amount;
        let mut sold = 0.0;
        let mut proceeds = 0.0;
        let mut price = self.current_price;

        for interval in self.intervals.iter().skip(start_idx) {
            let capacity = band_sell_capacity(price, interval)?;
            if capacity <= DUST {
                price = interval.bottom_price;
                continue;
            }
            if remaining <= capacity + EPS {
                let (actual, leg_proceeds, new_price) = band_sell_exact(price, remaining, interval)?;
                sold += actual;
                proceeds += leg_proceeds;
                return Some((sold, proceeds, new_price));
            }

            let (actual, leg_proceeds, new_price) = band_sell_exact(price, capacity, interval)?;
            sold += actual;
            proceeds += leg_proceeds;
            remaining = (remaining - actual).max(0.0);
            price = new_price;
            if remaining <= DUST {
                return Some((sold, proceeds, price));
            }
        }

        Some((sold, proceeds, price))
    }

    fn buy_exact(&self, amount: f64) -> Option<(f64, f64, f64)> {
        if amount <= 0.0 {
            return Some((0.0, 0.0, self.current_price));
        }
        let start_idx = self.interval_index_for_buy()?;
        let mut remaining = amount;
        let mut bought = 0.0;
        let mut cost = 0.0;
        let mut price = self.current_price;

        for interval in self.intervals[..=start_idx].iter().rev() {
            let capacity = band_buy_capacity(price, interval)?;
            if capacity <= DUST {
                price = interval.top_price;
                continue;
            }
            if remaining <= capacity + EPS {
                let (actual, leg_cost, new_price) = band_buy_exact(price, remaining, interval)?;
                bought += actual;
                cost += leg_cost;
                return Some((bought, cost, new_price));
            }

            let (actual, leg_cost, new_price) = band_buy_exact(price, capacity, interval)?;
            bought += actual;
            cost += leg_cost;
            remaining = (remaining - actual).max(0.0);
            price = new_price;
            if remaining <= DUST {
                return Some((bought, cost, price));
            }
        }

        Some((bought, cost, price))
    }

    fn interval_index_for_sell(&self) -> Option<usize> {
        for (idx, interval) in self.intervals.iter().enumerate() {
            if price_strictly_inside_interval(self.current_price, interval) {
                return Some(idx);
            }
        }
        for (idx, interval) in self.intervals.iter().enumerate() {
            if price_matches_boundary(self.current_price, interval.top_price) {
                return Some(idx);
            }
        }
        None
    }

    fn interval_index_for_buy(&self) -> Option<usize> {
        for (idx, interval) in self.intervals.iter().enumerate() {
            if price_strictly_inside_interval(self.current_price, interval) {
                return Some(idx);
            }
        }
        for (idx, interval) in self.intervals.iter().enumerate() {
            if price_matches_boundary(self.current_price, interval.bottom_price) {
                return Some(idx);
            }
        }
        None
    }
}

fn price_boundary_tolerance(lhs: f64, rhs: f64) -> f64 {
    1e-9 * (1.0 + lhs.abs().max(rhs.abs()).max(1.0))
}

fn price_matches_boundary(price: f64, boundary: f64) -> bool {
    (price - boundary).abs() <= price_boundary_tolerance(price, boundary)
}

fn price_strictly_inside_interval(price: f64, interval: &DerivedLiquidityInterval) -> bool {
    let top_tol = price_boundary_tolerance(price, interval.top_price);
    let bottom_tol = price_boundary_tolerance(price, interval.bottom_price);
    price <= interval.top_price + top_tol && price > interval.bottom_price + bottom_tol
}

fn band_liquidity_raw(interval: &DerivedLiquidityInterval) -> Option<f64> {
    if interval.liquidity == 0 {
        return None;
    }
    let liquidity = interval.liquidity as f64 / 1e18;
    if liquidity.is_finite() && liquidity > 0.0 {
        Some(liquidity)
    } else {
        None
    }
}

fn band_sell_capacity(current_price: f64, interval: &DerivedLiquidityInterval) -> Option<f64> {
    let liquidity = band_liquidity_raw(interval)?;
    if current_price <= interval.bottom_price {
        return Some(0.0);
    }
    Some(
        liquidity * (1.0 / interval.bottom_price.sqrt() - 1.0 / current_price.sqrt()) / FEE_FACTOR,
    )
}

fn band_buy_capacity(current_price: f64, interval: &DerivedLiquidityInterval) -> Option<f64> {
    let liquidity = band_liquidity_raw(interval)?;
    if current_price >= interval.top_price {
        return Some(0.0);
    }
    Some(liquidity * (1.0 / current_price.sqrt() - 1.0 / interval.top_price.sqrt()))
}

fn band_sell_exact(
    current_price: f64,
    amount: f64,
    interval: &DerivedLiquidityInterval,
) -> Option<(f64, f64, f64)> {
    let liquidity = band_liquidity_raw(interval)?;
    let kappa = FEE_FACTOR * current_price.sqrt() / liquidity;
    if kappa <= 0.0 {
        return None;
    }
    let actual = amount.max(0.0);
    let d = 1.0 + actual * kappa;
    if d <= 0.0 {
        return None;
    }
    let new_price = current_price / (d * d);
    let proceeds = current_price * actual * FEE_FACTOR / d;
    Some((actual, proceeds, new_price.max(interval.bottom_price)))
}

fn band_buy_exact(
    current_price: f64,
    amount: f64,
    interval: &DerivedLiquidityInterval,
) -> Option<(f64, f64, f64)> {
    let liquidity = band_liquidity_raw(interval)?;
    let lambda = current_price.sqrt() / liquidity;
    if lambda <= 0.0 {
        return None;
    }
    let actual = amount.max(0.0);
    let d = 1.0 - actual * lambda;
    if d <= 0.0 {
        return None;
    }
    let new_price = current_price / (d * d);
    let cost = actual * current_price / (FEE_FACTOR * d);
    Some((actual, cost, new_price.min(interval.top_price)))
}

fn build_univ3_liquidity_bands(
    slot0: &Slot0Result,
    market: &'static MarketData,
) -> Option<Vec<UniV3LiquidityBandRequest>> {
    let is_token1_outcome = market
        .pool
        .as_ref()?
        .token1
        .eq_ignore_ascii_case(market.outcome_token);
    let current_price = sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome)
        .and_then(|value| u128::try_from(value).ok())
        .map(|wad| wad as f64 / 1e18)?;
    let intervals = derive_contiguous_liquidity_intervals(slot0, market, current_price)?;
    let mut bands = intervals
        .iter()
        .map(|interval| UniV3LiquidityBandRequest {
            lower_price: interval.top_price,
            liquidity_l: interval.liquidity as f64 / 1e18,
        })
        .collect::<Vec<_>>();
    bands.push(UniV3LiquidityBandRequest {
        lower_price: intervals.last()?.bottom_price,
        liquidity_l: 0.0,
    });
    Some(bands)
}

fn derive_contiguous_liquidity_intervals(
    _slot0: &Slot0Result,
    market: &'static MarketData,
    current_price: f64,
) -> Option<Vec<DerivedLiquidityInterval>> {
    let primary = derive_contiguous_liquidity_intervals_primary(market, current_price);
    if let Some(intervals) = primary {
        return Some(intervals);
    }
    #[cfg(any(test, feature = "benchmark_synthetic_fixtures"))]
    {
        return fallback_single_tick_intervals(_slot0, market);
    }
    #[cfg(not(any(test, feature = "benchmark_synthetic_fixtures")))]
    None
}

fn derive_contiguous_liquidity_intervals_primary(
    market: &'static MarketData,
    current_price: f64,
) -> Option<Vec<DerivedLiquidityInterval>> {
    let pool = market.pool.as_ref()?;
    let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
    let mut collapsed_ticks = pool
        .ticks
        .iter()
        .map(|tick| (tick.tick_idx, tick.liquidity_net))
        .collect::<Vec<_>>();
    collapsed_ticks.sort_unstable_by_key(|(tick_idx, _)| *tick_idx);
    let mut deduped_ticks: Vec<(i32, i128)> = Vec::with_capacity(collapsed_ticks.len());
    for (tick_idx, liquidity_net) in collapsed_ticks {
        if let Some((last_tick_idx, last_liquidity_net)) = deduped_ticks.last_mut()
            && *last_tick_idx == tick_idx
        {
            *last_liquidity_net = last_liquidity_net.checked_add(liquidity_net)?;
        } else {
            deduped_ticks.push((tick_idx, liquidity_net));
        }
    }
    if deduped_ticks.len() < 2 {
        return None;
    }

    let mut active_liquidity = 0i128;
    let mut intervals = Vec::new();
    let mut seen_positive_interval = false;
    let mut saw_zero_gap_after_positive = false;
    let mut current_price_is_covered = false;

    for idx in 0..deduped_ticks.len().saturating_sub(1) {
        let (tick_lo, liquidity_net) = deduped_ticks[idx];
        let tick_hi = deduped_ticks[idx + 1].0;
        if tick_lo >= tick_hi {
            return None;
        }
        active_liquidity = active_liquidity.checked_add(liquidity_net)?;
        if active_liquidity < 0 {
            return None;
        }
        if active_liquidity == 0 {
            if seen_positive_interval {
                saw_zero_gap_after_positive = true;
            }
            continue;
        }
        if saw_zero_gap_after_positive {
            return None;
        }
        seen_positive_interval = true;
        let (top_price, bottom_price) = interval_price_bounds(is_token1_outcome, tick_lo, tick_hi)?;
        if interval_contains_price(current_price, bottom_price, top_price) {
            current_price_is_covered = true;
        }
        intervals.push(DerivedLiquidityInterval {
            top_price,
            bottom_price,
            liquidity: active_liquidity as u128,
        });
    }

    if intervals.is_empty() || !current_price_is_covered {
        return None;
    }

    intervals.sort_by(|left, right| right.top_price.total_cmp(&left.top_price));
    if !intervals_are_contiguous(&intervals) {
        return None;
    }
    Some(intervals)
}

fn interval_price_bounds(
    is_token1_outcome: bool,
    tick_lo: i32,
    tick_hi: i32,
) -> Option<(f64, f64)> {
    let sqrt_lo = get_sqrt_ratio_at_tick(tick_lo).ok()?;
    let sqrt_hi = get_sqrt_ratio_at_tick(tick_hi).ok()?;
    let price_lo = sqrt_price_x96_to_price_outcome(sqrt_lo, is_token1_outcome)
        .and_then(|value| u128::try_from(value).ok())
        .map(|wad| wad as f64 / 1e18)?;
    let price_hi = sqrt_price_x96_to_price_outcome(sqrt_hi, is_token1_outcome)
        .and_then(|value| u128::try_from(value).ok())
        .map(|wad| wad as f64 / 1e18)?;
    if !price_lo.is_finite() || !price_hi.is_finite() || price_lo <= 0.0 || price_hi <= 0.0 {
        return None;
    }
    Some((price_lo.max(price_hi), price_lo.min(price_hi)))
}

fn interval_contains_price(current_price: f64, bottom_price: f64, top_price: f64) -> bool {
    let tolerance = 1e-9 * (1.0 + current_price.abs().max(top_price.abs()));
    current_price + tolerance >= bottom_price && current_price <= top_price + tolerance
}

fn intervals_are_contiguous(intervals: &[DerivedLiquidityInterval]) -> bool {
    intervals.windows(2).all(|pair| {
        let upper = pair[0].bottom_price;
        let lower = pair[1].top_price;
        let tolerance = 1e-9 * (1.0 + upper.abs().max(lower.abs()));
        (upper - lower).abs() <= tolerance
    })
}

#[cfg(any(test, feature = "benchmark_synthetic_fixtures"))]
fn fallback_single_tick_intervals(
    slot0: &Slot0Result,
    market: &'static MarketData,
) -> Option<Vec<DerivedLiquidityInterval>> {
    const FALLBACK_PRICE_SPAN: f64 = 1.0e6;

    let pool = market.pool.as_ref()?;
    let liquidity = pool.liquidity.parse::<u128>().ok()?;
    let tick_lo = pool.ticks.iter().map(|tick| tick.tick_idx).min()?;
    let tick_hi = pool.ticks.iter().map(|tick| tick.tick_idx).max()?;
    let current_tick = get_tick_at_sqrt_ratio(slot0.sqrt_price_x96).ok()?;
    if liquidity == 0 || tick_lo >= tick_hi || current_tick < tick_lo || current_tick >= tick_hi {
        return None;
    }
    let is_token1_outcome = pool.token1.eq_ignore_ascii_case(market.outcome_token);
    let current_price = sqrt_price_x96_to_price_outcome(slot0.sqrt_price_x96, is_token1_outcome)
        .and_then(|value| u128::try_from(value).ok())
        .map(|wad| wad as f64 / 1e18)?;
    if !current_price.is_finite() || current_price <= 0.0 {
        return None;
    }
    let top_price = current_price * FALLBACK_PRICE_SPAN;
    let bottom_price = current_price / FALLBACK_PRICE_SPAN;
    if !top_price.is_finite() || !bottom_price.is_finite() || bottom_price <= 0.0 {
        return None;
    }
    Some(vec![DerivedLiquidityInterval {
        top_price,
        bottom_price,
        liquidity,
    }])
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

    fn mock_slot0_market_with_tick_ladder(
        name: &'static str,
        outcome_token: &'static str,
        price_fraction: f64,
        liquidity: u128,
        ticks: &[(i32, i128)],
    ) -> (Slot0Result, &'static MarketData) {
        let tick_slice = Box::leak(
            ticks
                .iter()
                .map(|(tick_idx, liquidity_net)| Tick {
                    tick_idx: *tick_idx,
                    liquidity_net: *liquidity_net,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
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
                ticks: tick_slice,
            }),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        });
        let current_tick = 75;
        let slot0 = Slot0Result {
            pool_id: Address::ZERO,
            sqrt_price_x96: get_sqrt_ratio_at_tick(current_tick)
                .expect("tick ratio should exist")
                .into(),
            tick: current_tick,
            observation_index: 0,
            observation_cardinality: 0,
            observation_cardinality_next: 0,
            fee_protocol: 0,
            unlocked: true,
        };
        let _ = price_fraction;
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

    fn two_underlying_market_fixture() -> (
        Vec<(Slot0Result, &'static MarketData)>,
        HashMap<&'static str, f64>,
        HashMap<String, f64>,
    ) {
        let (slot0_a, market_a) = mock_slot0_market_with_liquidity_and_ticks(
            "RootA",
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            0.3,
            1_000_000_000_000_000_000_000u128,
            -16_096,
            92_108,
        );
        let market_a = leak_market(MarketData {
            market_id: "0xroot-market",
            ..*market_a
        });
        let (slot0_b, market_b) = mock_slot0_market_with_liquidity_and_ticks(
            "ChildB",
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            0.2,
            1_000_000_000_000_000_000_000u128,
            -16_096,
            92_108,
        );
        let market_b = leak_market(MarketData {
            market_id: "0xchild-market",
            ..*market_b
        });
        let slot0_results = vec![(slot0_a, market_a), (slot0_b, market_b)];
        let balances = HashMap::from([("RootA", 2.0), ("ChildB", 3.0)]);
        let predictions =
            HashMap::from([("roota".to_string(), 0.55), ("childb".to_string(), 0.45)]);
        (slot0_results, balances, predictions)
    }

    fn certified_direct_result(trades: Vec<PredictionMarketTrade>) -> CompareResult {
        CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "direct_only".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades,
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "uncertified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary {
                    passed: false,
                }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            workspace_reused: false,
        }
    }

    #[test]
    fn build_problem_request_uses_contiguous_univ3_band_ladder_and_fee_mapping() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let problem = build_problem_request(&balances, 1.0, &slot0_results, &predictions, 2)
            .expect("problem request should build");

        assert_eq!(problem.outcomes.len(), 2);
        assert_eq!(problem.markets.len(), 2);
        assert_eq!(problem.split_bound, None);
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
        slot0_results[0].0.sqrt_price_x96 = get_sqrt_ratio_at_tick(100_000)
            .expect("tick ratio should exist")
            .into();
        let err = build_problem_request(&balances, 1.0, &slot0_results, &predictions, 2)
            .expect_err("non-derivable active range should be rejected");
        assert!(
            err.to_string()
                .contains("does not have replayable contiguous liquidity ladder geometry")
        );
    }

    #[test]
    fn build_problem_request_accepts_test_fixture_with_price_derived_single_range_geometry() {
        let (slot0_a, market_a) = mock_slot0_market_with_liquidity_and_ticks(
            "BenchA",
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            0.1,
            1_000_000_000_000_000_000_000u128,
            16_095,
            92_108,
        );
        let (slot0_b, market_b) = mock_slot0_market_with_liquidity_and_ticks(
            "BenchB",
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            0.2,
            1_000_000_000_000_000_000_000u128,
            16_095,
            92_108,
        );
        let slot0_results = vec![(slot0_a, market_a), (slot0_b, market_b)];
        let balances = HashMap::new();
        let predictions =
            HashMap::from([("bencha".to_string(), 0.12), ("benchb".to_string(), 0.25)]);

        let problem = build_problem_request(&balances, 1.0, &slot0_results, &predictions, 2)
            .expect("test fixture should derive single-range geometry from price");

        assert_eq!(problem.markets.len(), 2);
        let MarketSpecRequest::UniV3 { bands, .. } = &problem.markets[0];
        assert_eq!(bands.len(), 2);
        assert!(bands[0].liquidity_l > 0.0);
    }

    #[test]
    fn build_problem_request_sets_conservative_split_bound_for_multi_market_family() {
        let (slot0_results, balances, predictions) = two_underlying_market_fixture();
        let problem = build_problem_request(&balances, 10.7, &slot0_results, &predictions, 2)
            .expect("multi-market problem request should build");

        assert_eq!(problem.collateral_balance, 10.7);
        assert_eq!(problem.split_bound, Some(10.0));
    }

    #[test]
    fn build_problem_request_emits_multiple_contiguous_univ3_bands() {
        let liq_a = 1_000_000_000_000_000_000_000u128;
        let liq_b = 500_000_000_000_000_000_000u128;
        let (slot0_a, market_a) = mock_slot0_market_with_tick_ladder(
            "LadderA",
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            0.0,
            liq_a + liq_b,
            &[
                (-100, liq_a as i128),
                (50, liq_b as i128),
                (100, -(liq_b as i128)),
                (200, -(liq_a as i128)),
            ],
        );
        let (slot0_b, market_b) = mock_slot0_market_with_liquidity_and_ticks(
            "LadderB",
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            0.2,
            1_000_000_000_000_000_000_000u128,
            16_095,
            92_108,
        );
        let slot0_results = vec![(slot0_a, market_a), (slot0_b, market_b)];
        let balances = HashMap::new();
        let predictions =
            HashMap::from([("laddera".to_string(), 0.12), ("ladderb".to_string(), 0.25)]);

        let problem = build_problem_request(&balances, 1.0, &slot0_results, &predictions, 2)
            .expect("multi-band ladder should translate");

        let MarketSpecRequest::UniV3 { bands, .. } = &problem.markets[0];
        assert_eq!(bands.len(), 4);
        assert!(
            bands
                .windows(2)
                .all(|pair| pair[0].lower_price > pair[1].lower_price)
        );
        assert!((bands[0].liquidity_l - (liq_a as f64 / 1e18)).abs() < 1e-6);
        assert!((bands[1].liquidity_l - ((liq_a + liq_b) as f64 / 1e18)).abs() < 1e-6);
        assert!((bands[2].liquidity_l - (liq_a as f64 / 1e18)).abs() < 1e-6);
        assert_eq!(bands[3].liquidity_l, 0.0);
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
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan {
                    mint: 0.1,
                    merge: 0.1,
                },
            },
            workspace_reused: false,
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
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: vec![PredictionMarketTrade {
                    market_id: "M1".to_string(),
                    outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                    collateral_delta: -0.02,
                    outcome_delta: 0.03,
                }],
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            workspace_reused: false,
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
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan {
                    mint: 0.1,
                    merge: 0.1,
                },
            },
            workspace_reused: false,
        };

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("valid direct candidate should survive malformed mixed branch");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Direct);
    }

    #[test]
    fn translate_compare_result_stages_direct_merge_before_buy_merge_rounds() {
        let (slot0_results, _, predictions) = two_market_fixture();
        let balances = HashMap::from([("M1", 0.02), ("M2", 0.07)]);
        let compare = CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "uncertified".to_string(),
                mode: "direct_only".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary {
                    passed: false,
                }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: vec![PredictionMarketTrade {
                    market_id: "M1".to_string(),
                    outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                    collateral_delta: -0.03,
                    outcome_delta: 0.03,
                }],
                split_merge: super::super::protocol::SplitMergePlan {
                    mint: 0.0,
                    merge: 0.05,
                },
            },
            workspace_reused: false,
        };

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("translation should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Mixed);
        assert!(matches!(
            candidates[0].actions.first(),
            Some(Action::Merge { amount, .. }) if (amount - 0.02).abs() < 1e-9
        ));
        assert!(matches!(
            candidates[0].actions.get(1),
            Some(Action::Buy {
                market_name: "M1",
                ..
            })
        ));
        assert!(matches!(
            candidates[0].actions.get(2),
            Some(Action::Merge { amount, .. }) if (amount - 0.03).abs() < 1e-9
        ));
    }

    #[test]
    fn translate_compare_result_orders_sell_buy_merge_buy_for_partial_buy_merge_routes() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let compare = CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "uncertified".to_string(),
                mode: "direct_only".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary {
                    passed: false,
                }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: vec![
                    PredictionMarketTrade {
                        market_id: "M1".to_string(),
                        outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                        collateral_delta: -0.06,
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
            workspace_reused: false,
        };

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("translation should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Mixed);
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
        assert!(matches!(
            candidates[0].actions.get(2),
            Some(Action::Merge { amount, .. }) if (amount - 0.05).abs() < 1e-9
        ));
        assert!(matches!(
            candidates[0].actions.get(3),
            Some(Action::Buy {
                market_name: "M1",
                ..
            })
        ));
    }

    #[test]
    fn translate_compare_result_stages_mint_after_initial_sells() {
        let (slot0_results, _, predictions) = two_market_fixture();
        let balances = HashMap::from([("M1", 0.06), ("M2", 0.0)]);
        let compare = CompareResult {
            direct_only: PredictionMarketSolveResult {
                status: "uncertified".to_string(),
                mode: "direct_only".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary {
                    passed: false,
                }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: Vec::new(),
                split_merge: super::super::protocol::SplitMergePlan::default(),
            },
            mixed_enabled: PredictionMarketSolveResult {
                status: "certified".to_string(),
                mode: "mixed_enabled".to_string(),
                certificate: Some(super::super::protocol::SolveCertificateSummary { passed: true }),
                solver_time_sec: Some(0.001),
                estimated_execution_cost: None,
                net_ev: None,
                trades: vec![
                    PredictionMarketTrade {
                        market_id: "M1".to_string(),
                        outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
                        collateral_delta: 0.06,
                        outcome_delta: -0.06,
                    },
                    PredictionMarketTrade {
                        market_id: "M2".to_string(),
                        outcome_id: "0x2222222222222222222222222222222222222222".to_string(),
                        collateral_delta: 0.04,
                        outcome_delta: -0.04,
                    },
                ],
                split_merge: super::super::protocol::SplitMergePlan {
                    mint: 0.04,
                    merge: 0.0,
                },
            },
            workspace_reused: false,
        };

        let candidates =
            translate_compare_result(&balances, 1.0, &slot0_results, &predictions, compare)
                .expect("translation should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].variant, ForecastFlowsCandidateVariant::Mixed);
        assert!(matches!(
            candidates[0].actions.first(),
            Some(Action::Sell {
                market_name: "M1",
                ..
            })
        ));
        assert!(matches!(
            candidates[0].actions.get(1),
            Some(Action::Mint { amount, .. }) if (amount - 0.04).abs() < 1e-9
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

    #[test]
    fn translate_compare_result_clamps_tiny_sell_overshoot_to_available_holdings() {
        let (slot0_results, _, predictions) = two_market_fixture();
        let balances = HashMap::from([("M1", 1.0), ("M2", 0.0)]);
        let compare = certified_direct_result(vec![PredictionMarketTrade {
            market_id: "M1".to_string(),
            outcome_id: "0x1111111111111111111111111111111111111111".to_string(),
            collateral_delta: 0.01,
            outcome_delta: -1.0000003,
        }]);

        let report =
            translate_compare_result_report(&balances, 1.0, &slot0_results, &predictions, compare);
        assert!(report.replay_tolerance_clamp_used);
        let candidates = report.candidates;

        assert_eq!(candidates.len(), 1);
        assert!(matches!(
            candidates[0].actions[0],
            Action::Sell {
                market_name: "M1",
                amount,
                ..
            } if (amount - 1.0).abs() <= 1e-12
        ));
    }
}
