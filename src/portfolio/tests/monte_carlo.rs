use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use crate::execution::grouping::{
    ProfitabilityStepKind, group_execution_actions_by_profitability_step,
};

use super::super::rebalancer::rebalance;
use super::super::sim::{DUST, EPS, PoolSim, build_sims};
use super::{
    TestRng, assert_rebalance_action_invariants, build_rebalance_fuzz_case,
    build_slot0_results_for_markets, eligible_l1_markets_with_predictions, ev_from_state,
    replay_actions_to_market_state, replay_actions_to_state,
};

#[derive(Debug, Clone, Copy)]
struct MonteCarloConfig {
    trials: usize,
    max_group_steps: usize,
    seed: u64,
    start_trial_index: usize,
    convergence_every_group_steps: Option<usize>,
    require_family_coverage: bool,
}

impl MonteCarloConfig {
    fn smoke() -> Self {
        Self {
            trials: 24,
            max_group_steps: 1500,
            seed: 0x6D6F_6E74_655F_736Du64,
            start_trial_index: 0,
            convergence_every_group_steps: None,
            require_family_coverage: true,
        }
    }

    fn full_from_env() -> Self {
        let every = env_usize_allow_zero("MC_CONVERGENCE_EVERY_GROUP_STEPS", 0).filter(|v| *v > 0);
        Self {
            trials: env_usize("MC_TRIALS", 200000),
            max_group_steps: env_usize("MC_MAX_GROUP_STEPS", 1000000),
            seed: env_u64("MC_SEED", 0xC0DE_1BAD_5EED_u64),
            start_trial_index: env_usize_allow_zero("MC_START_TRIAL_INDEX", 0).unwrap_or(0),
            convergence_every_group_steps: every,
            require_family_coverage: env_bool("MC_REQUIRE_FAMILY_COVERAGE", true),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct FamilyCoverage {
    direct_trials: usize,
    mixed_trials: usize,
    arb_or_indirect_trials: usize,
}

#[derive(Debug, Clone)]
struct WorstTrial {
    trial_index: usize,
    scenario: &'static str,
    ev_before: f64,
    ev_after: f64,
    ev_delta: f64,
}

#[derive(Debug, Clone)]
struct MonteCarloStats {
    executed_trials: usize,
    total_group_steps: usize,
    max_ev_delta: f64,
    min_ev_delta: f64,
    best_trial: Option<WorstTrial>,
    worst_trial: Option<WorstTrial>,
    family_coverage: FamilyCoverage,
    step_kind_counts: BTreeMap<&'static str, usize>,
}

impl MonteCarloStats {
    fn new() -> Self {
        Self {
            executed_trials: 0,
            total_group_steps: 0,
            max_ev_delta: f64::NEG_INFINITY,
            min_ev_delta: f64::INFINITY,
            best_trial: None,
            worst_trial: None,
            family_coverage: FamilyCoverage::default(),
            step_kind_counts: BTreeMap::new(),
        }
    }

    fn record_step_kind(&mut self, kind: ProfitabilityStepKind) {
        let label = step_kind_label(kind);
        *self.step_kind_counts.entry(label).or_insert(0) += 1;
    }
}

#[derive(Debug, Clone, Copy)]
enum ScenarioTemplate {
    FullUnderpricedBaseline,
    DirectOnlyBaseline,
    FuzzFull,
    FuzzPartial,
}

#[derive(Debug)]
struct ScenarioCase {
    label: &'static str,
    slot0_results: Vec<(
        crate::pools::Slot0Result,
        &'static crate::markets::MarketData,
    )>,
    balances: HashMap<&'static str, f64>,
    susd_balance: f64,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_usize_allow_zero(key: &str, default: usize) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .or(Some(default))
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|raw| {
            let normalized = raw.trim().replace('_', "");
            if let Some(hex) = normalized
                .strip_prefix("0x")
                .or_else(|| normalized.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16).ok()
            } else {
                normalized.parse::<u64>().ok()
            }
        })
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|raw| {
            let t = raw.trim().to_ascii_lowercase();
            match t.as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            }
        })
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn pick_index(rng: &mut TestRng, upper_exclusive: usize) -> usize {
    debug_assert!(upper_exclusive > 0);
    let raw = (rng.next_f64() * (upper_exclusive as f64)) as usize;
    raw.min(upper_exclusive - 1)
}

fn step_kind_label(kind: ProfitabilityStepKind) -> &'static str {
    match kind {
        ProfitabilityStepKind::ArbMintSell => "arb_mint_sell",
        ProfitabilityStepKind::ArbBuyMerge => "arb_buy_merge",
        ProfitabilityStepKind::PureDirectBuy => "pure_direct_buy",
        ProfitabilityStepKind::PureDirectSell => "pure_direct_sell",
        ProfitabilityStepKind::PureDirectMerge => "pure_direct_merge",
        ProfitabilityStepKind::MixedDirectBuyMintSell => "mixed_direct_buy_mint_sell",
        ProfitabilityStepKind::MixedDirectSellBuyMerge => "mixed_direct_sell_buy_merge",
    }
}

fn scenario_template_for_trial(trial_index: usize) -> ScenarioTemplate {
    match trial_index {
        0 => ScenarioTemplate::FullUnderpricedBaseline,
        1 => ScenarioTemplate::DirectOnlyBaseline,
        _ if trial_index % 2 == 0 => ScenarioTemplate::FuzzFull,
        _ => ScenarioTemplate::FuzzPartial,
    }
}

fn build_full_underpriced_baseline() -> ScenarioCase {
    let markets = eligible_l1_markets_with_predictions();
    let multipliers = vec![0.5; markets.len()];
    ScenarioCase {
        label: "full_underpriced_baseline",
        slot0_results: build_slot0_results_for_markets(&markets, &multipliers),
        balances: HashMap::new(),
        susd_balance: 100.0,
    }
}

fn build_direct_only_baseline() -> ScenarioCase {
    let markets = eligible_l1_markets_with_predictions();
    assert!(
        markets.len() >= 2,
        "direct-only baseline requires at least two eligible pooled markets"
    );
    let selected = [markets[0], markets[1]];
    let mut balances = HashMap::new();
    balances.insert(selected[0].name, 8.0);
    balances.insert(selected[1].name, 0.5);

    ScenarioCase {
        label: "direct_only_baseline",
        slot0_results: build_slot0_results_for_markets(&selected, &[1.35, 0.55]),
        balances,
        susd_balance: 4.0,
    }
}

fn build_fuzz_case(rng: &mut TestRng, force_partial: bool) -> ScenarioCase {
    let (slot0_results, balances, susd_balance) = build_rebalance_fuzz_case(rng, force_partial);
    ScenarioCase {
        label: if force_partial {
            "fuzz_partial"
        } else {
            "fuzz_full"
        },
        slot0_results,
        balances,
        susd_balance,
    }
}

fn build_scenario_case(template: ScenarioTemplate, rng: &mut TestRng) -> ScenarioCase {
    match template {
        ScenarioTemplate::FullUnderpricedBaseline => build_full_underpriced_baseline(),
        ScenarioTemplate::DirectOnlyBaseline => build_direct_only_baseline(),
        ScenarioTemplate::FuzzFull => build_fuzz_case(rng, false),
        ScenarioTemplate::FuzzPartial => build_fuzz_case(rng, true),
    }
}

fn initial_holdings_from_case(case: &ScenarioCase) -> HashMap<&'static str, f64> {
    let mut holdings = HashMap::new();
    for (_, market) in &case.slot0_results {
        holdings.insert(
            market.name,
            case.balances
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0),
        );
    }
    holdings
}

fn balances_as_str_map<'a>(balances: &'a HashMap<&'static str, f64>) -> HashMap<&'a str, f64> {
    balances.iter().map(|(k, v)| (*k as &str, *v)).collect()
}

fn run_monte_carlo(config: MonteCarloConfig) -> MonteCarloStats {
    let mut rng = TestRng::new(config.seed);
    let mut stats = MonteCarloStats::new();
    let mut next_checkpoint_group_steps = config.convergence_every_group_steps;

    // If we start from a non-zero global trial index, advance RNG state exactly as
    // skipped fuzz trials would have consumed it in a full run from index 0.
    for skipped_trial_index in 0..config.start_trial_index {
        match scenario_template_for_trial(skipped_trial_index) {
            ScenarioTemplate::FuzzFull => {
                let _ = build_rebalance_fuzz_case(&mut rng, false);
            }
            ScenarioTemplate::FuzzPartial => {
                let _ = build_rebalance_fuzz_case(&mut rng, true);
            }
            ScenarioTemplate::FullUnderpricedBaseline | ScenarioTemplate::DirectOnlyBaseline => {}
        }
    }

    for local_trial_index in 0..config.trials {
        let trial_index = config.start_trial_index + local_trial_index;
        if stats.total_group_steps >= config.max_group_steps {
            break;
        }

        let template = scenario_template_for_trial(trial_index);
        let case = build_scenario_case(template, &mut rng);
        let balances = balances_as_str_map(&case.balances);
        let actions = rebalance(&balances, case.susd_balance, &case.slot0_results);
        assert_rebalance_action_invariants(
            &actions,
            &case.slot0_results,
            &balances,
            case.susd_balance,
        );
        let grouped = group_execution_actions_by_profitability_step(&actions)
            .expect("rebalance output should be profitability-step groupable");

        let has_direct = grouped.iter().any(|g| {
            matches!(
                g.kind,
                ProfitabilityStepKind::PureDirectBuy
                    | ProfitabilityStepKind::PureDirectSell
                    | ProfitabilityStepKind::PureDirectMerge
            )
        });
        let has_mixed = grouped.iter().any(|g| {
            matches!(
                g.kind,
                ProfitabilityStepKind::MixedDirectBuyMintSell
                    | ProfitabilityStepKind::MixedDirectSellBuyMerge
            )
        });
        let has_arb_or_indirect = grouped.iter().any(|g| {
            matches!(
                g.kind,
                ProfitabilityStepKind::ArbMintSell | ProfitabilityStepKind::ArbBuyMerge
            )
        });

        if has_direct {
            stats.family_coverage.direct_trials += 1;
        }
        if has_mixed {
            stats.family_coverage.mixed_trials += 1;
        }
        if has_arb_or_indirect {
            stats.family_coverage.arb_or_indirect_trials += 1;
        }

        for group in &grouped {
            stats.record_step_kind(group.kind);
        }
        stats.total_group_steps += grouped.len();
        stats.executed_trials += 1;

        let before_holdings = initial_holdings_from_case(&case);
        let ev_before = ev_from_state(&before_holdings, case.susd_balance);
        let (final_holdings, final_cash) =
            replay_actions_to_state(&actions, &case.slot0_results, &balances, case.susd_balance);
        let ev_after = ev_from_state(&final_holdings, final_cash);
        let ev_delta = ev_after - ev_before;
        let tol = 1e-8 * (1.0 + ev_before.abs().max(ev_after.abs()));

        assert!(
            ev_after + tol >= ev_before,
            "monte carlo EV regression at trial {} ({}) seed={} template={:?}: before={:.12}, after={:.12}, delta={:.12}, tol={:.12}, actions={}, groups={}",
            trial_index,
            case.label,
            config.seed,
            template,
            ev_before,
            ev_after,
            ev_delta,
            tol,
            actions.len(),
            grouped.len(),
        );

        if ev_delta > stats.max_ev_delta {
            stats.max_ev_delta = ev_delta;
            stats.best_trial = Some(WorstTrial {
                trial_index,
                scenario: case.label,
                ev_before,
                ev_after,
                ev_delta,
            });
        }
        if ev_delta < stats.min_ev_delta {
            stats.min_ev_delta = ev_delta;
            stats.worst_trial = Some(WorstTrial {
                trial_index,
                scenario: case.label,
                ev_before,
                ev_after,
                ev_delta,
            });
        }

        if let Some(step_interval) = config.convergence_every_group_steps {
            while let Some(next_threshold) = next_checkpoint_group_steps {
                if stats.total_group_steps < next_threshold {
                    break;
                }
                println!(
                    "[monte-carlo][checkpoint] trials={}, group_steps={}, max_delta={:.9}, min_delta={:.9}",
                    stats.executed_trials,
                    stats.total_group_steps,
                    stats.max_ev_delta,
                    stats.min_ev_delta
                );
                next_checkpoint_group_steps = next_threshold.checked_add(step_interval);
            }
        }
    }

    stats
}

fn print_summary(config: MonteCarloConfig, stats: &MonteCarloStats) {
    println!(
        "[monte-carlo] config: trials={}, max_group_steps={}, seed={}, start_trial_index={}, convergence_every_group_steps={}, require_family_coverage={}",
        config.trials,
        config.max_group_steps,
        config.seed,
        config.start_trial_index,
        config
            .convergence_every_group_steps
            .map(|v| v.to_string())
            .unwrap_or_else(|| "off".to_string()),
        config.require_family_coverage
    );
    println!(
        "[monte-carlo] executed: trials={}, group_steps={}",
        stats.executed_trials, stats.total_group_steps
    );
    println!(
        "[monte-carlo] ev deltas: max={:.9}, min={:.9}",
        stats.max_ev_delta, stats.min_ev_delta
    );
    if let Some(best) = &stats.best_trial {
        println!(
            "[monte-carlo] best trial: idx={}, scenario={}, before={:.9}, after={:.9}, delta={:.9}",
            best.trial_index, best.scenario, best.ev_before, best.ev_after, best.ev_delta
        );
    }
    if let Some(worst) = &stats.worst_trial {
        println!(
            "[monte-carlo] worst trial: idx={}, scenario={}, before={:.9}, after={:.9}, delta={:.9}",
            worst.trial_index, worst.scenario, worst.ev_before, worst.ev_after, worst.ev_delta
        );
    }
    println!(
        "[monte-carlo] family coverage: direct={}, mixed={}, arb_or_indirect={}",
        stats.family_coverage.direct_trials,
        stats.family_coverage.mixed_trials,
        stats.family_coverage.arb_or_indirect_trials
    );
    println!("[monte-carlo] step-kind counts:");
    for (kind, count) in &stats.step_kind_counts {
        println!("  - {}: {}", kind, count);
    }
}

#[derive(Debug, Clone, Copy)]
enum SearchGroupKind {
    DirectBuy,
    DirectSell,
    MintSell,
    BuyMerge,
}

impl SearchGroupKind {
    const COUNT: usize = 4;

    fn index(self) -> usize {
        match self {
            Self::DirectBuy => 0,
            Self::DirectSell => 1,
            Self::MintSell => 2,
            Self::BuyMerge => 3,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::DirectBuy => "direct_buy",
            Self::DirectSell => "direct_sell",
            Self::MintSell => "mint_sell",
            Self::BuyMerge => "buy_merge",
        }
    }
}

#[derive(Clone)]
struct SearchState {
    sims: Vec<PoolSim>,
    holdings: Vec<f64>,
    cash: f64,
    allow_indirect: bool,
}

impl SearchState {
    fn ev(&self) -> f64 {
        let holdings_ev: f64 = self
            .holdings
            .iter()
            .zip(self.sims.iter())
            .map(|(held, sim)| held.max(0.0) * sim.prediction)
            .sum();
        self.cash + holdings_ev
    }
}

#[derive(Debug, Clone)]
struct RandomSearchConfig {
    max_rollouts: usize,
    groups_per_rollout: usize,
    checkpoint_every_rollouts: usize,
    min_runtime_secs: f64,
    stale_checkpoints_for_convergence: usize,
    convergence_tol: f64,
    case_count: usize,
    search_seed: u64,
    assert_algo_not_worse: bool,
    algo_tolerance: f64,
}

impl RandomSearchConfig {
    fn from_env() -> Self {
        Self {
            max_rollouts: env_usize("MC_SEARCH_MAX_ROLLOUTS", 2_000_000),
            groups_per_rollout: env_usize("MC_SEARCH_GROUPS_PER_ROLLOUT", 8),
            checkpoint_every_rollouts: env_usize("MC_SEARCH_CHECKPOINT_EVERY", 10_000),
            min_runtime_secs: env_f64("MC_SEARCH_MIN_RUNTIME_SECS", 300.0).max(0.0),
            stale_checkpoints_for_convergence: env_usize("MC_SEARCH_STALE_CHECKPOINTS", 6),
            convergence_tol: env_f64("MC_SEARCH_CONVERGENCE_TOL", 1e-9).abs(),
            case_count: env_usize("MC_SEARCH_CASE_COUNT", 4).min(4),
            search_seed: env_u64("MC_SEARCH_SEED", 0xBAD5_EA12_34C0_FFEE),
            assert_algo_not_worse: env_bool("MC_SEARCH_ASSERT_ALGO_NOT_WORSE", false),
            algo_tolerance: env_f64("MC_SEARCH_ALGO_TOL", 1e-6).abs(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SearchCoverage {
    attempted: [usize; SearchGroupKind::COUNT],
    applied: [usize; SearchGroupKind::COUNT],
}

impl SearchCoverage {
    fn record_attempt(&mut self, kind: SearchGroupKind) {
        self.attempted[kind.index()] += 1;
    }

    fn record_apply(&mut self, kind: SearchGroupKind) {
        self.applied[kind.index()] += 1;
    }
}

#[derive(Debug, Clone)]
struct RandomSearchStats {
    rollouts_executed: usize,
    best_ev: f64,
    best_rollout: usize,
    converged: bool,
    elapsed_secs: f64,
    groups_attempted: usize,
    groups_applied: usize,
    coverage: SearchCoverage,
}

#[derive(Debug, Clone)]
struct LocalGradientSummary {
    eps: f64,
    max_direct_grad: f64,
    max_indirect_grad: f64,
    best_direct_label: String,
    best_indirect_label: String,
}

fn scenario_case_from_fuzz_seed(seed: u64, force_partial: bool, case_idx: usize) -> ScenarioCase {
    let mut rng = TestRng::new(seed);
    let mut built: Option<ScenarioCase> = None;
    for _ in 0..=case_idx {
        built = Some(build_fuzz_case(&mut rng, force_partial));
    }
    built.expect("fuzz case generation should produce a case")
}

fn algorithm_ev_for_case(case: &ScenarioCase) -> (f64, f64, f64) {
    let balances = balances_as_str_map(&case.balances);
    let actions = rebalance(&balances, case.susd_balance, &case.slot0_results);
    assert_rebalance_action_invariants(&actions, &case.slot0_results, &balances, case.susd_balance);

    let initial_holdings = initial_holdings_from_case(case);
    let ev_before = ev_from_state(&initial_holdings, case.susd_balance);
    let (final_holdings, final_cash) =
        replay_actions_to_state(&actions, &case.slot0_results, &balances, case.susd_balance);
    let ev_after = ev_from_state(&final_holdings, final_cash);
    (ev_before, ev_after, ev_after - ev_before)
}

fn initial_search_state(case: &ScenarioCase) -> SearchState {
    let predictions = crate::pools::prediction_map();
    let sims = build_sims(&case.slot0_results, &predictions)
        .expect("random search requires full prediction coverage");
    let full_count = eligible_l1_markets_with_predictions().len();
    let allow_indirect = case.slot0_results.len() == full_count && sims.len() == full_count;
    let holdings = sims
        .iter()
        .map(|sim| {
            case.balances
                .get(sim.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        })
        .collect();
    SearchState {
        sims,
        holdings,
        cash: case.susd_balance,
        allow_indirect,
    }
}

fn sample_search_group_kind(rng: &mut TestRng, allow_indirect: bool) -> SearchGroupKind {
    if allow_indirect {
        match pick_index(rng, SearchGroupKind::COUNT) {
            0 => SearchGroupKind::DirectBuy,
            1 => SearchGroupKind::DirectSell,
            2 => SearchGroupKind::MintSell,
            _ => SearchGroupKind::BuyMerge,
        }
    } else if pick_index(rng, 2) == 0 {
        SearchGroupKind::DirectBuy
    } else {
        SearchGroupKind::DirectSell
    }
}

fn random_fraction(rng: &mut TestRng) -> f64 {
    let t = rng.in_range(0.0, 1.0);
    match pick_index(rng, 3) {
        0 => t,
        1 => t * t,
        _ => 1.0 - (1.0 - t) * (1.0 - t),
    }
}

fn apply_direct_buy(state: &mut SearchState, rng: &mut TestRng) -> bool {
    if state.sims.is_empty() {
        return false;
    }
    let idx = pick_index(rng, state.sims.len());
    let cap = state.sims[idx].max_buy_tokens().max(0.0);
    if cap <= DUST {
        return false;
    }
    let request = cap * random_fraction(rng);
    if request <= DUST {
        return false;
    }
    let Some((bought, cost, new_price)) = state.sims[idx].buy_exact(request) else {
        return false;
    };
    if bought <= DUST || !cost.is_finite() || cost > state.cash + EPS {
        return false;
    }
    state.sims[idx].set_price(new_price);
    state.holdings[idx] += bought;
    state.cash -= cost;
    true
}

fn apply_direct_sell(state: &mut SearchState, rng: &mut TestRng) -> bool {
    let sellable: Vec<usize> = state
        .holdings
        .iter()
        .enumerate()
        .filter_map(|(i, held)| (*held > DUST).then_some(i))
        .collect();
    if sellable.is_empty() {
        return false;
    }
    let idx = sellable[pick_index(rng, sellable.len())];
    let request = state.holdings[idx].max(0.0) * random_fraction(rng);
    if request <= DUST {
        return false;
    }
    let Some((sold, proceeds, new_price)) = state.sims[idx].sell_exact(request) else {
        return false;
    };
    if sold <= DUST || !proceeds.is_finite() {
        return false;
    }
    state.sims[idx].set_price(new_price);
    state.holdings[idx] = (state.holdings[idx] - sold).max(0.0);
    state.cash += proceeds;
    true
}

fn apply_mint_sell(state: &mut SearchState, rng: &mut TestRng) -> bool {
    let n = state.sims.len();
    if n < 2 {
        return false;
    }
    let target_idx = pick_index(rng, n);

    let mut cap = f64::INFINITY;
    for i in 0..n {
        if i == target_idx {
            continue;
        }
        cap = cap.min(state.sims[i].max_sell_tokens().max(0.0));
    }
    if !cap.is_finite() || cap <= DUST {
        return false;
    }

    let amount = cap * random_fraction(rng);
    if amount <= DUST {
        return false;
    }

    for held in &mut state.holdings {
        *held += amount;
    }
    state.cash -= amount;

    for i in 0..n {
        if i == target_idx {
            continue;
        }
        let Some((sold, proceeds, new_price)) = state.sims[i].sell_exact(amount) else {
            return false;
        };
        if sold > DUST {
            state.sims[i].set_price(new_price);
            state.holdings[i] = (state.holdings[i] - sold).max(0.0);
            state.cash += proceeds;
        }
    }
    true
}

fn apply_buy_merge(state: &mut SearchState, rng: &mut TestRng) -> bool {
    let n = state.sims.len();
    if n < 2 {
        return false;
    }

    let sources: Vec<usize> = state
        .holdings
        .iter()
        .enumerate()
        .filter_map(|(i, held)| (*held > DUST).then_some(i))
        .collect();
    if sources.is_empty() {
        return false;
    }
    let source_idx = sources[pick_index(rng, sources.len())];

    let mut cap = state.holdings[source_idx].max(0.0);
    for i in 0..n {
        if i == source_idx {
            continue;
        }
        cap = cap.min(state.holdings[i].max(0.0) + state.sims[i].max_buy_tokens().max(0.0));
    }
    if !cap.is_finite() || cap <= DUST {
        return false;
    }

    let amount = cap * random_fraction(rng);
    if amount <= DUST {
        return false;
    }

    for i in 0..n {
        if i == source_idx {
            continue;
        }
        let shortfall = (amount - state.holdings[i].max(0.0)).max(0.0);
        if shortfall <= DUST {
            continue;
        }
        let Some((bought, cost, new_price)) = state.sims[i].buy_exact(shortfall) else {
            return false;
        };
        if bought + EPS < shortfall || !cost.is_finite() || cost < -EPS {
            return false;
        }
        state.sims[i].set_price(new_price);
        state.holdings[i] += bought;
        state.cash -= cost;
    }

    for held in &mut state.holdings {
        *held -= amount;
        if *held < 0.0 && *held > -1e-9 {
            *held = 0.0;
        }
    }
    state.cash += amount;
    true
}

fn state_is_valid(state: &SearchState) -> bool {
    if !state.cash.is_finite() || state.cash < -1e-9 {
        return false;
    }
    state
        .holdings
        .iter()
        .all(|held| held.is_finite() && *held >= -1e-9)
}

fn try_apply_random_group(
    state: &mut SearchState,
    kind: SearchGroupKind,
    rng: &mut TestRng,
) -> bool {
    let mut trial = state.clone();
    let applied = match kind {
        SearchGroupKind::DirectBuy => apply_direct_buy(&mut trial, rng),
        SearchGroupKind::DirectSell => apply_direct_sell(&mut trial, rng),
        SearchGroupKind::MintSell => apply_mint_sell(&mut trial, rng),
        SearchGroupKind::BuyMerge => apply_buy_merge(&mut trial, rng),
    };
    if !applied || !state_is_valid(&trial) {
        return false;
    }
    *state = trial;
    true
}

fn print_search_coverage(coverage: &SearchCoverage) {
    for kind in [
        SearchGroupKind::DirectBuy,
        SearchGroupKind::DirectSell,
        SearchGroupKind::MintSell,
        SearchGroupKind::BuyMerge,
    ] {
        let idx = kind.index();
        println!(
            "  - {}: attempted={}, applied={}",
            kind.label(),
            coverage.attempted[idx],
            coverage.applied[idx]
        );
    }
}

fn run_random_group_search(
    case_label: &str,
    initial: &SearchState,
    algo_ev: f64,
    config: &RandomSearchConfig,
    rng: &mut TestRng,
) -> RandomSearchStats {
    let mut coverage = SearchCoverage::default();
    let mut best_state = initial.clone();
    let mut best_ev = initial.ev();
    let mut best_rollout = 0usize;
    let mut rollouts_executed = 0usize;
    let mut groups_attempted = 0usize;
    let mut groups_applied = 0usize;
    let mut stale_checkpoints = 0usize;
    let mut checkpoint_best_ev = best_ev;
    let mut converged = false;
    let start = Instant::now();
    let checkpoint_every = config.checkpoint_every_rollouts.max(1);
    let max_rollouts = config.max_rollouts.max(1);
    let groups_per_rollout = config.groups_per_rollout.max(1);

    loop {
        let elapsed = start.elapsed().as_secs_f64();
        if converged && elapsed >= config.min_runtime_secs {
            break;
        }
        if rollouts_executed >= max_rollouts && elapsed >= config.min_runtime_secs {
            break;
        }

        rollouts_executed += 1;
        let mut state = if rng.in_range(0.0, 1.0) < 0.75 {
            best_state.clone()
        } else {
            initial.clone()
        };
        let group_count = 1 + pick_index(rng, groups_per_rollout);
        for _ in 0..group_count {
            let kind = sample_search_group_kind(rng, state.allow_indirect);
            coverage.record_attempt(kind);
            groups_attempted += 1;
            if try_apply_random_group(&mut state, kind, rng) {
                coverage.record_apply(kind);
                groups_applied += 1;
            }
        }

        let ev = state.ev();
        if ev > best_ev + config.convergence_tol {
            best_ev = ev;
            best_state = state;
            best_rollout = rollouts_executed;
        }

        if rollouts_executed % checkpoint_every == 0 {
            if best_ev > checkpoint_best_ev + config.convergence_tol {
                checkpoint_best_ev = best_ev;
                stale_checkpoints = 0;
            } else {
                stale_checkpoints += 1;
            }
            let gap_to_algo = algo_ev - best_ev;
            println!(
                "[mc-search][checkpoint] case={} rollouts={} best_ev={:.9} gap_to_algo={:.9} stale_checkpoints={} elapsed_s={:.1}",
                case_label,
                rollouts_executed,
                best_ev,
                gap_to_algo,
                stale_checkpoints,
                start.elapsed().as_secs_f64()
            );
            converged = stale_checkpoints >= config.stale_checkpoints_for_convergence;
        }
    }

    RandomSearchStats {
        rollouts_executed,
        best_ev,
        best_rollout,
        converged,
        elapsed_secs: start.elapsed().as_secs_f64(),
        groups_attempted,
        groups_applied,
        coverage,
    }
}

fn eval_direct_buy_gradient(state: &SearchState, idx: usize, eps: f64) -> Option<f64> {
    if idx >= state.sims.len() {
        return None;
    }
    let cap = state.sims[idx].max_buy_tokens().max(0.0);
    if cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST {
        return None;
    }
    let (bought, cost, _) = state.sims[idx].buy_exact(amount)?;
    if bought <= DUST || !cost.is_finite() || cost > state.cash + EPS {
        return None;
    }
    Some((state.sims[idx].prediction * bought - cost) / bought.max(DUST))
}

fn eval_direct_sell_gradient(state: &SearchState, idx: usize, eps: f64) -> Option<f64> {
    if idx >= state.sims.len() {
        return None;
    }
    let held = state.holdings[idx].max(0.0);
    let cap = held.min(state.sims[idx].max_sell_tokens().max(0.0));
    if cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST {
        return None;
    }
    let (sold, proceeds, _) = state.sims[idx].sell_exact(amount)?;
    if sold <= DUST || !proceeds.is_finite() {
        return None;
    }
    Some((proceeds - state.sims[idx].prediction * sold) / sold.max(DUST))
}

fn eval_mint_sell_gradient(state: &SearchState, target_idx: usize, eps: f64) -> Option<f64> {
    let n = state.sims.len();
    if n < 2 || target_idx >= n {
        return None;
    }

    let mut cap = f64::INFINITY;
    for i in 0..n {
        if i == target_idx {
            continue;
        }
        cap = cap.min(state.sims[i].max_sell_tokens().max(0.0));
    }
    if !cap.is_finite() || cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST || amount > state.cash + EPS {
        return None;
    }

    let base_ev = state.ev();
    let mut trial = state.clone();
    for held in &mut trial.holdings {
        *held += amount;
    }
    trial.cash -= amount;

    for i in 0..n {
        if i == target_idx {
            continue;
        }
        let (sold, proceeds, new_price) = trial.sims[i].sell_exact(amount)?;
        if sold <= DUST || !proceeds.is_finite() {
            return None;
        }
        trial.sims[i].set_price(new_price);
        trial.holdings[i] = (trial.holdings[i] - sold).max(0.0);
        trial.cash += proceeds;
    }
    if !state_is_valid(&trial) {
        return None;
    }
    Some((trial.ev() - base_ev) / amount.max(DUST))
}

fn eval_buy_merge_gradient(state: &SearchState, source_idx: usize, eps: f64) -> Option<f64> {
    let n = state.sims.len();
    if n < 2 || source_idx >= n {
        return None;
    }
    if state.holdings[source_idx] <= DUST {
        return None;
    }

    let mut cap = state.holdings[source_idx].max(0.0);
    for i in 0..n {
        if i == source_idx {
            continue;
        }
        cap = cap.min(state.holdings[i].max(0.0) + state.sims[i].max_buy_tokens().max(0.0));
    }
    if !cap.is_finite() || cap <= DUST {
        return None;
    }
    let amount = cap * eps;
    if amount <= DUST {
        return None;
    }

    let base_ev = state.ev();
    let mut trial = state.clone();
    for i in 0..n {
        if i == source_idx {
            continue;
        }
        let shortfall = (amount - trial.holdings[i].max(0.0)).max(0.0);
        if shortfall <= DUST {
            continue;
        }
        let (bought, cost, new_price) = trial.sims[i].buy_exact(shortfall)?;
        if bought + EPS < shortfall || !cost.is_finite() || cost > trial.cash + EPS {
            return None;
        }
        trial.sims[i].set_price(new_price);
        trial.holdings[i] += bought;
        trial.cash -= cost;
    }

    for held in &mut trial.holdings {
        *held -= amount;
        if *held < 0.0 && *held > -1e-9 {
            *held = 0.0;
        }
    }
    trial.cash += amount;
    if !state_is_valid(&trial) {
        return None;
    }
    Some((trial.ev() - base_ev) / amount.max(DUST))
}

fn estimate_local_gradients(state: &SearchState, eps: f64) -> LocalGradientSummary {
    let mut max_direct = f64::NEG_INFINITY;
    let mut max_indirect = f64::NEG_INFINITY;
    let mut best_direct = "none".to_string();
    let mut best_indirect = "none".to_string();

    for i in 0..state.sims.len() {
        if let Some(g) = eval_direct_buy_gradient(state, i, eps) {
            if g > max_direct {
                max_direct = g;
                best_direct = format!("direct_buy:{}", state.sims[i].market_name);
            }
        }
        if let Some(g) = eval_direct_sell_gradient(state, i, eps) {
            if g > max_direct {
                max_direct = g;
                best_direct = format!("direct_sell:{}", state.sims[i].market_name);
            }
        }
        if state.allow_indirect {
            if let Some(g) = eval_mint_sell_gradient(state, i, eps) {
                if g > max_indirect {
                    max_indirect = g;
                    best_indirect = format!("mint_sell_target:{}", state.sims[i].market_name);
                }
            }
            if let Some(g) = eval_buy_merge_gradient(state, i, eps) {
                if g > max_indirect {
                    max_indirect = g;
                    best_indirect = format!("buy_merge_source:{}", state.sims[i].market_name);
                }
            }
        }
    }

    LocalGradientSummary {
        eps,
        max_direct_grad: if max_direct.is_finite() {
            max_direct
        } else {
            0.0
        },
        max_indirect_grad: if max_indirect.is_finite() {
            max_indirect
        } else {
            0.0
        },
        best_direct_label: best_direct,
        best_indirect_label: best_indirect,
    }
}

fn state_after_algorithm_actions(case: &ScenarioCase, actions: &[super::Action]) -> SearchState {
    let balances = balances_as_str_map(&case.balances);
    let (holdings_after, cash_after) =
        replay_actions_to_state(actions, &case.slot0_results, &balances, case.susd_balance);
    let post_markets = replay_actions_to_market_state(actions, &case.slot0_results);
    let predictions = crate::pools::prediction_map();
    let sims = build_sims(&post_markets, &predictions)
        .expect("post-action gradient check requires full prediction coverage");
    let full_count = eligible_l1_markets_with_predictions().len();
    let allow_indirect = case.slot0_results.len() == full_count && sims.len() == full_count;
    let holdings = sims
        .iter()
        .map(|sim| {
            holdings_after
                .get(sim.market_name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        })
        .collect();
    SearchState {
        sims,
        holdings,
        cash: cash_after,
        allow_indirect,
    }
}

fn assert_coverage(stats: &MonteCarloStats) {
    assert!(
        stats.executed_trials > 0,
        "monte carlo run executed zero trials"
    );
    assert!(
        stats.family_coverage.direct_trials > 0,
        "expected at least one direct-only trial"
    );
    assert!(
        stats.family_coverage.mixed_trials > 0,
        "expected at least one mixed-route trial"
    );
    assert!(
        stats.family_coverage.arb_or_indirect_trials > 0,
        "expected at least one arb/indirect trial"
    );
}

#[test]
#[ignore = "Monte Carlo validation is opt-in; run explicitly"]
fn test_monte_carlo_ev_smoke_profitability_groups() {
    let config = MonteCarloConfig::smoke();
    let stats = run_monte_carlo(config);
    print_summary(config, &stats);
    if config.require_family_coverage {
        assert_coverage(&stats);
    }
}

#[test]
#[ignore = "long-running Monte Carlo validation; run in release mode"]
fn test_monte_carlo_ev_full_profitability_groups() {
    let config = MonteCarloConfig::full_from_env();
    let stats = run_monte_carlo(config);
    print_summary(config, &stats);
    if config.require_family_coverage {
        assert_coverage(&stats);
    }
}

#[test]
#[ignore = "independent random group-action search oracle; long-running stress test"]
fn test_random_group_search_vs_waterfall_complex_fuzz_cases() {
    let config = RandomSearchConfig::from_env();
    let cases = vec![
        (
            "fuzz_full_case_0",
            scenario_case_from_fuzz_seed(0xFEED_FACE_1234_4321u64, false, 0),
        ),
        (
            "fuzz_full_case_1",
            scenario_case_from_fuzz_seed(0xFEED_FACE_1234_4321u64, false, 1),
        ),
        (
            "fuzz_partial_case_0",
            scenario_case_from_fuzz_seed(0xABCD_1234_EF99_7788u64, true, 0),
        ),
        (
            "fuzz_partial_case_1",
            scenario_case_from_fuzz_seed(0xABCD_1234_EF99_7788u64, true, 1),
        ),
    ];

    println!(
        "[mc-search] config: max_rollouts={}, groups_per_rollout={}, checkpoint_every={}, min_runtime_secs={:.1}, stale_checkpoints_for_convergence={}, convergence_tol={:.3e}, case_count={}, search_seed={}, assert_algo_not_worse={}, algo_tolerance={:.3e}",
        config.max_rollouts,
        config.groups_per_rollout,
        config.checkpoint_every_rollouts,
        config.min_runtime_secs,
        config.stale_checkpoints_for_convergence,
        config.convergence_tol,
        config.case_count,
        config.search_seed,
        config.assert_algo_not_worse,
        config.algo_tolerance
    );

    let mut random_beats_algorithm: Vec<(String, f64)> = Vec::new();
    let case_count = config.case_count.max(1).min(cases.len());
    for (i, (label, case)) in cases.into_iter().take(case_count).enumerate() {
        let (ev_before, algo_ev_after, algo_gain) = algorithm_ev_for_case(&case);
        let initial = initial_search_state(&case);
        let init_ev = initial.ev();
        let init_ev_drift = (init_ev - ev_before).abs();
        assert!(
            init_ev_drift <= 1e-8 * (1.0 + ev_before.abs()),
            "search-state EV must match case baseline: label={}, init_ev={:.12}, ev_before={:.12}",
            label,
            init_ev,
            ev_before
        );

        let case_seed = config.search_seed
            ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ 0xD00D_F00D_ABCD_1234u64;
        let mut rng = TestRng::new(case_seed);
        let search_stats =
            run_random_group_search(label, &initial, algo_ev_after, &config, &mut rng);
        let search_gain = search_stats.best_ev - ev_before;
        let gap_to_algo = algo_ev_after - search_stats.best_ev;

        println!(
            "[mc-search] case={} algo_ev={:.9} algo_gain={:.9} random_best_ev={:.9} random_best_gain={:.9} gap_to_algo={:.9} best_rollout={} rollouts={} groups_attempted={} groups_applied={} converged={} elapsed_s={:.1}",
            label,
            algo_ev_after,
            algo_gain,
            search_stats.best_ev,
            search_gain,
            gap_to_algo,
            search_stats.best_rollout,
            search_stats.rollouts_executed,
            search_stats.groups_attempted,
            search_stats.groups_applied,
            search_stats.converged,
            search_stats.elapsed_secs
        );
        println!("[mc-search] case={} action coverage:", label);
        print_search_coverage(&search_stats.coverage);

        if search_stats.best_ev > algo_ev_after + config.algo_tolerance {
            random_beats_algorithm.push((label.to_string(), search_stats.best_ev - algo_ev_after));
        }
    }

    if config.assert_algo_not_worse {
        assert!(
            random_beats_algorithm.is_empty(),
            "independent random search found EV above algorithm output: {:?}",
            random_beats_algorithm
        );
    }
}

#[test]
#[ignore = "heuristic local-gradient diagnostic around waterfall output"]
fn test_waterfall_local_gradient_heuristic_complex_cases() {
    let eps = env_f64("MC_GRAD_EPS", 1e-4).clamp(1e-8, 1e-2);
    let case_count = env_usize("MC_GRAD_CASE_COUNT", 4).min(4).max(1);
    let assert_non_positive = env_bool("MC_GRAD_ASSERT_NON_POSITIVE", false);
    let tol = env_f64("MC_GRAD_TOL", 1e-6).abs();

    let cases = vec![
        (
            "fuzz_full_case_0",
            scenario_case_from_fuzz_seed(0xFEED_FACE_1234_4321u64, false, 0),
        ),
        (
            "fuzz_full_case_1",
            scenario_case_from_fuzz_seed(0xFEED_FACE_1234_4321u64, false, 1),
        ),
        (
            "fuzz_partial_case_0",
            scenario_case_from_fuzz_seed(0xABCD_1234_EF99_7788u64, true, 0),
        ),
        (
            "fuzz_partial_case_1",
            scenario_case_from_fuzz_seed(0xABCD_1234_EF99_7788u64, true, 1),
        ),
    ];

    println!(
        "[mc-grad] config: eps={:.3e}, case_count={}, assert_non_positive={}, tol={:.3e}",
        eps, case_count, assert_non_positive, tol
    );

    for (label, case) in cases.into_iter().take(case_count) {
        let balances = balances_as_str_map(&case.balances);
        let actions = rebalance(&balances, case.susd_balance, &case.slot0_results);
        let (ev_before, ev_after, gain) = algorithm_ev_for_case(&case);

        let before_state = initial_search_state(&case);
        let after_state = state_after_algorithm_actions(&case, &actions);

        let before_grad = estimate_local_gradients(&before_state, eps);
        let after_grad = estimate_local_gradients(&after_state, eps);

        println!(
            "[mc-grad] case={} ev_before={:.9} ev_after={:.9} gain={:.9}",
            label, ev_before, ev_after, gain
        );
        println!(
            "[mc-grad] case={} pre: max_direct_grad={:.9} ({}) max_indirect_grad={:.9} ({}) eps={:.3e}",
            label,
            before_grad.max_direct_grad,
            before_grad.best_direct_label,
            before_grad.max_indirect_grad,
            before_grad.best_indirect_label,
            before_grad.eps
        );
        println!(
            "[mc-grad] case={} post: max_direct_grad={:.9} ({}) max_indirect_grad={:.9} ({}) eps={:.3e}",
            label,
            after_grad.max_direct_grad,
            after_grad.best_direct_label,
            after_grad.max_indirect_grad,
            after_grad.best_indirect_label,
            after_grad.eps
        );

        if assert_non_positive {
            assert!(
                after_grad.max_direct_grad <= tol,
                "post-waterfall direct local gradient still positive: case={}, grad={:.12}, best={}",
                label,
                after_grad.max_direct_grad,
                after_grad.best_direct_label
            );
            assert!(
                after_grad.max_indirect_grad <= tol,
                "post-waterfall indirect local gradient still positive: case={}, grad={:.12}, best={}",
                label,
                after_grad.max_indirect_grad,
                after_grad.best_indirect_label
            );
        }
    }
}

#[test]
#[ignore = "diagnostic: second rebalance pass should be near-idempotent on complex cases"]
fn test_rebalance_second_pass_gain_complex_cases() {
    let case_count = env_usize("MC_SECOND_PASS_CASE_COUNT", 4).min(4).max(1);
    let rel_cap = env_f64("MC_SECOND_PASS_REL_CAP", 0.02).max(0.0);
    let abs_cap = env_f64("MC_SECOND_PASS_ABS_CAP", 1e-3).max(0.0);
    let assert_cap = env_bool("MC_SECOND_PASS_ASSERT", true);

    let cases = vec![
        (
            "fuzz_full_case_0",
            scenario_case_from_fuzz_seed(0xFEED_FACE_1234_4321u64, false, 0),
        ),
        (
            "fuzz_full_case_1",
            scenario_case_from_fuzz_seed(0xFEED_FACE_1234_4321u64, false, 1),
        ),
        (
            "fuzz_partial_case_0",
            scenario_case_from_fuzz_seed(0xABCD_1234_EF99_7788u64, true, 0),
        ),
        (
            "fuzz_partial_case_1",
            scenario_case_from_fuzz_seed(0xABCD_1234_EF99_7788u64, true, 1),
        ),
    ];

    println!(
        "[mc-second-pass] config: case_count={}, rel_cap={:.6}, abs_cap={:.6}, assert_cap={}",
        case_count, rel_cap, abs_cap, assert_cap
    );

    for (label, case) in cases.into_iter().take(case_count) {
        let balances0 = balances_as_str_map(&case.balances);
        let actions_first = rebalance(&balances0, case.susd_balance, &case.slot0_results);
        assert_rebalance_action_invariants(
            &actions_first,
            &case.slot0_results,
            &balances0,
            case.susd_balance,
        );

        let holdings0 = initial_holdings_from_case(&case);
        let ev_before = ev_from_state(&holdings0, case.susd_balance);
        let (holdings1, cash1) = replay_actions_to_state(
            &actions_first,
            &case.slot0_results,
            &balances0,
            case.susd_balance,
        );
        let ev_after_first = ev_from_state(&holdings1, cash1);
        let first_gain = (ev_after_first - ev_before).max(0.0);

        let slot0_after_first = replay_actions_to_market_state(&actions_first, &case.slot0_results);
        let balances1: HashMap<&str, f64> =
            holdings1.iter().map(|(k, v)| (*k as &str, *v)).collect();
        let actions_second = rebalance(&balances1, cash1, &slot0_after_first);
        assert_rebalance_action_invariants(&actions_second, &slot0_after_first, &balances1, cash1);
        let (holdings2, cash2) =
            replay_actions_to_state(&actions_second, &slot0_after_first, &balances1, cash1);
        let ev_after_second = ev_from_state(&holdings2, cash2);
        let second_gain = (ev_after_second - ev_after_first).max(0.0);

        let cap = abs_cap.max(rel_cap * (1.0 + first_gain));
        println!(
            "[mc-second-pass] case={} ev_before={:.9} ev_after_first={:.9} ev_after_second={:.9} first_gain={:.9} second_gain={:.9} cap={:.9} actions_first={} actions_second={}",
            label,
            ev_before,
            ev_after_first,
            ev_after_second,
            first_gain,
            second_gain,
            cap,
            actions_first.len(),
            actions_second.len()
        );

        if assert_cap {
            assert!(
                second_gain <= cap + 1e-9,
                "second pass gain too large: case={}, second_gain={:.12}, cap={:.12}, first_gain={:.12}",
                label,
                second_gain,
                cap,
                first_gain
            );
        }
    }
}
