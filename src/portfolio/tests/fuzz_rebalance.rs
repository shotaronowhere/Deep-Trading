use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::merge::{merge_sell_cap_with_inventory, optimal_sell_split_with_inventory};
use super::super::planning::{
    active_skip_indices, plan_active_routes, plan_is_budget_feasible, solve_prof,
};
use super::super::rebalancer::{
    RebalanceConfig, RebalanceEngine, RebalanceMode, rebalance,
    rebalance_with_config_and_diagnostics,
};
use super::super::sim::{PoolSim, Route, alt_price, profitability};
use super::super::solver::mint_cost_to_prof;
use super::super::waterfall::waterfall;
use super::{
    Action, TestRng, assert_rebalance_action_invariants,
    assert_strict_ev_gain_with_portfolio_trace, brute_force_best_split_with_inventory,
    build_rebalance_fuzz_case, build_slot0_results_for_markets, build_three_sims_with_preds,
    eligible_l1_markets_with_predictions, ev_from_state, mock_slot0_market,
    replay_actions_to_state,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct EvSnapshots {
    fuzz_full_l1_case_ev_after: [f64; 24],
    fuzz_partial_l1_case_ev_after: [f64; 24],
}

fn ev_snapshots() -> &'static EvSnapshots {
    static SNAPSHOTS: OnceLock<EvSnapshots> = OnceLock::new();
    SNAPSHOTS.get_or_init(|| {
        serde_json::from_str(include_str!("ev_snapshots.json"))
            .expect("ev snapshot fixture must be valid JSON with 24 full + 24 partial entries")
    })
}

fn ev_meets_floor(got: f64, floor: f64) -> (bool, f64) {
    // Keep a tiny tolerance for serialized f64 fixture comparisons.
    let tol = 1e-9 * (1.0 + got.abs().max(floor.abs()));
    (got + tol >= floor, tol)
}

fn collect_fuzz_ev_after(force_partial: bool) -> [f64; 24] {
    let seed = if force_partial {
        0xABCD_1234_EF99_7788u64
    } else {
        0xFEED_FACE_1234_4321u64
    };
    let mut out = [0.0_f64; 24];
    let mut rng = TestRng::new(seed);
    for case_idx in 0..24 {
        let (slot0_results, balances_static, susd_balance) =
            build_rebalance_fuzz_case(&mut rng, force_partial);
        if force_partial {
            assert!(
                slot0_results.len() < crate::predictions::PREDICTIONS_L1.len(),
                "partial fuzz case must disable mint/merge route availability"
            );
        }
        let balances: HashMap<&str, f64> = balances_static
            .iter()
            .map(|(k, v)| (*k as &str, *v))
            .collect();
        let actions = rebalance(&balances, susd_balance, &slot0_results);
        assert_rebalance_action_invariants(&actions, &slot0_results, &balances, susd_balance);
        if force_partial {
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
        }

        let mut holdings_before: HashMap<&'static str, f64> = HashMap::new();
        for (_, market) in &slot0_results {
            holdings_before.insert(
                market.name,
                balances.get(market.name).copied().unwrap_or(0.0).max(0.0),
            );
        }
        let ev_before = ev_from_state(&holdings_before, susd_balance);
        let (holdings_after, cash_after) =
            replay_actions_to_state(&actions, &slot0_results, &balances, susd_balance);
        let ev_after = ev_from_state(&holdings_after, cash_after);
        assert!(
            ev_after > ev_before,
            "expected EV strict improvement for fuzz snapshot case {} (force_partial={}): before={:.9}, after={:.9}",
            case_idx,
            force_partial,
            ev_before,
            ev_after
        );
        out[case_idx] = ev_after;
    }
    out
}

#[test]
#[ignore = "updates src/portfolio/tests/ev_snapshots.json; run explicitly"]
fn test_refresh_ev_snapshots_fixture() {
    let snapshots = EvSnapshots {
        fuzz_full_l1_case_ev_after: collect_fuzz_ev_after(false),
        fuzz_partial_l1_case_ev_after: collect_fuzz_ev_after(true),
    };
    let json = serde_json::to_string_pretty(&snapshots)
        .expect("ev snapshots should serialize to valid JSON");
    std::fs::write("src/portfolio/tests/ev_snapshots.json", format!("{json}\n"))
        .expect("failed to write ev snapshots fixture");
    println!("[snapshot-refresh] wrote src/portfolio/tests/ev_snapshots.json");
}

#[test]
#[ignore = "manual EV comparison report between incumbent and global candidate"]
fn test_compare_global_vs_incumbent_ev_across_rebalance_fixtures() {
    type GlobalOptimizer = super::super::global_solver::GlobalOptimizer;

    fn env_usize(key: &str) -> Option<usize> {
        std::env::var(key).ok()?.parse::<usize>().ok()
    }
    fn env_f64(key: &str) -> Option<f64> {
        std::env::var(key).ok()?.parse::<f64>().ok()
    }
    fn parse_optimizer_override(value: &str) -> Option<GlobalOptimizer> {
        match value.to_ascii_lowercase().as_str() {
            "dual" | "dual_decomposition" => Some(GlobalOptimizer::DualDecompositionPrototype),
            "dual_router" | "router" => Some(GlobalOptimizer::DualRouterV1),
            "lbfgsb" | "lbfgs" => Some(GlobalOptimizer::LbfgsbProjected),
            "newton" | "diagonal" => Some(GlobalOptimizer::DiagonalProjectedNewton),
            _ => None,
        }
    }
    #[derive(Debug, Clone, Copy, Default, Serialize)]
    struct ActionMix {
        buy_count: usize,
        sell_count: usize,
        mint_count: usize,
        merge_count: usize,
        flash_count: usize,
        repay_count: usize,
        buy_amount: f64,
        sell_amount: f64,
        mint_amount: f64,
        merge_amount: f64,
    }
    #[derive(Debug, Serialize)]
    struct CompareCaseRow<'a> {
        family: &'a str,
        case: &'a str,
        market_count: usize,
        optimizer: String,
        chosen_engine: String,
        candidate_valid: bool,
        invalid_reason: Option<String>,
        incumbent_ev: f64,
        candidate_ev: f64,
        delta_ev: f64,
        under_incumbent_bucket: Option<&'a str>,
        candidate_projected_grad_norm: f64,
        candidate_coupled_residual: f64,
        candidate_dual_residual_norm: f64,
        candidate_primal_restore_iters: usize,
        candidate_net_theta: f64,
        candidate_total_buy: f64,
        candidate_total_sell: f64,
        candidate_buy_sell_overlap: f64,
        candidate_replay_cash_delta: f64,
        candidate_replay_holdings_delta: f64,
        candidate_solver_iters: usize,
        candidate_line_search_trials: usize,
        candidate_line_search_accepts: usize,
        candidate_line_search_invalid_evals: usize,
        candidate_line_search_rescue_attempts: usize,
        candidate_line_search_rescue_accepts: usize,
        candidate_feasibility_repairs: usize,
        candidate_feasibility_hold_clamps: usize,
        candidate_feasibility_cash_scales: usize,
        candidate_active_dims: usize,
        candidate_curvature_skips: usize,
        incumbent_action_mix: ActionMix,
        candidate_action_mix: ActionMix,
    }
    fn action_mix(actions: &[Action]) -> ActionMix {
        let mut mix = ActionMix::default();
        for action in actions {
            match action {
                Action::Buy { amount, .. } => {
                    mix.buy_count += 1;
                    mix.buy_amount += amount.max(0.0);
                }
                Action::Sell { amount, .. } => {
                    mix.sell_count += 1;
                    mix.sell_amount += amount.max(0.0);
                }
                Action::Mint { amount, .. } => {
                    mix.mint_count += 1;
                    mix.mint_amount += amount.max(0.0);
                }
                Action::Merge { amount, .. } => {
                    mix.merge_count += 1;
                    mix.merge_amount += amount.max(0.0);
                }
                Action::FlashLoan { .. } => {
                    mix.flash_count += 1;
                }
                Action::RepayFlashLoan { .. } => {
                    mix.repay_count += 1;
                }
            }
        }
        mix
    }
    fn classify_under_incumbent_bucket(
        diag: &super::super::rebalancer::RebalanceDecisionDiagnostics,
        market_count: usize,
        delta: f64,
        tol: f64,
    ) -> Option<&'static str> {
        if delta > tol {
            return None;
        }
        if !diag.candidate_valid {
            return Some("invalid_candidate");
        }
        let overlap_tol = 1e-9
            * (1.0
                + diag
                    .candidate_total_buy
                    .abs()
                    .max(diag.candidate_total_sell.abs()));
        if diag.candidate_buy_sell_overlap > overlap_tol {
            return Some("overlap_churn");
        }
        if diag.candidate_net_theta.abs() > 1e-8 {
            return Some("mint_usage");
        }
        let expected_dims = 2 * market_count + 1;
        if diag.candidate_active_dims + 1 < expected_dims {
            return Some("boundary_saturation");
        }
        Some("other")
    }

    let optimizer_env = std::env::var("GLOBAL_SOLVER_OPTIMIZER").ok();
    let optimizer_override = optimizer_env
        .as_deref()
        .and_then(parse_optimizer_override);
    match (optimizer_env.as_deref(), optimizer_override) {
        (Some(raw), Some(parsed)) => {
            println!("[ev-compare] optimizer_override_raw={} parsed={:?}", raw, parsed);
        }
        (Some(raw), None) => {
            println!(
                "[ev-compare] optimizer_override_raw={} parsed=UNKNOWN (using default)",
                raw
            );
        }
        (None, _) => {}
    }

    let max_iters_override = env_usize("GLOBAL_SOLVER_MAX_ITERS");
    if let Some(value) = max_iters_override {
        println!("[ev-compare] max_iters_override={}", value);
    }
    let theta_l2_override = env_f64("GLOBAL_SOLVER_THETA_L2_REG");
    if let Some(value) = theta_l2_override {
        println!("[ev-compare] theta_l2_reg_override={:.12e}", value);
    }
    let line_search_trials_override = env_usize("GLOBAL_SOLVER_MAX_LINE_SEARCH_TRIALS");
    if let Some(value) = line_search_trials_override {
        println!("[ev-compare] max_line_search_trials_override={}", value);
    }
    let active_set_eps_override = env_f64("GLOBAL_SOLVER_ACTIVE_SET_EPS");
    if let Some(value) = active_set_eps_override {
        println!("[ev-compare] active_set_eps_override={:.12e}", value);
    }
    let dual_router_max_iters_override = env_usize("GLOBAL_SOLVER_DUAL_ROUTER_MAX_ITERS");
    if let Some(value) = dual_router_max_iters_override {
        println!("[ev-compare] dual_router_max_iters_override={}", value);
    }
    let dual_router_pg_tol_override = env_f64("GLOBAL_SOLVER_DUAL_ROUTER_PG_TOL");
    if let Some(value) = dual_router_pg_tol_override {
        println!("[ev-compare] dual_router_pg_tol_override={:.12e}", value);
    }
    let dual_router_lbfgs_history_override =
        env_usize("GLOBAL_SOLVER_DUAL_ROUTER_LBFGS_HISTORY");
    if let Some(value) = dual_router_lbfgs_history_override {
        println!("[ev-compare] dual_router_lbfgs_history_override={}", value);
    }
    let dual_router_primal_restore_iters_override =
        env_usize("GLOBAL_SOLVER_DUAL_ROUTER_PRIMAL_RESTORE_ITERS");
    if let Some(value) = dual_router_primal_restore_iters_override {
        println!(
            "[ev-compare] dual_router_primal_restore_iters_override={}",
            value
        );
    }
    let dual_router_primal_residual_tol_override =
        env_f64("GLOBAL_SOLVER_DUAL_ROUTER_PRIMAL_RESIDUAL_TOL");
    if let Some(value) = dual_router_primal_residual_tol_override {
        println!(
            "[ev-compare] dual_router_primal_residual_tol_override={:.12e}",
            value
        );
    }
    let dual_router_price_floor_override = env_f64("GLOBAL_SOLVER_DUAL_ROUTER_PRICE_FLOOR");
    if let Some(value) = dual_router_price_floor_override {
        println!("[ev-compare] dual_router_price_floor_override={:.12e}", value);
    }
    let buy_sell_churn_reg_override = env_f64("GLOBAL_SOLVER_BUY_SELL_CHURN_REG");
    if let Some(value) = buy_sell_churn_reg_override {
        println!("[ev-compare] buy_sell_churn_reg_override={:.12e}", value);
    }

    let optimizer_tag = optimizer_override
        .map(|value| format!("{value:?}").to_ascii_lowercase())
        .unwrap_or_else(|| "default".to_string());
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    let jsonl_path = format!("/tmp/global_ev_{}_{}.jsonl", optimizer_tag, now_ms);
    let jsonl_file = File::create(&jsonl_path)
        .expect("expected to create /tmp global EV telemetry output file");
    let mut jsonl_writer = BufWriter::new(jsonl_file);
    println!("[ev-compare] jsonl_output={}", jsonl_path);

    struct CompareStats {
        total_cases: usize,
        candidate_available: usize,
        candidate_better: usize,
        candidate_not_better: usize,
        candidate_valid: usize,
        candidate_invalid: usize,
        delta_sum: f64,
        best_delta: f64,
        worst_delta: f64,
        invalid_reason_counts: HashMap<String, usize>,
        under_incumbent_reason_counts: HashMap<String, usize>,
        family_deltas: HashMap<String, (usize, f64)>,
    }

    impl CompareStats {
        fn new() -> Self {
            Self {
                total_cases: 0,
                candidate_available: 0,
                candidate_better: 0,
                candidate_not_better: 0,
                candidate_valid: 0,
                candidate_invalid: 0,
                delta_sum: 0.0,
                best_delta: f64::NEG_INFINITY,
                worst_delta: f64::INFINITY,
                invalid_reason_counts: HashMap::new(),
                under_incumbent_reason_counts: HashMap::new(),
                family_deltas: HashMap::new(),
            }
        }

        fn record(
            &mut self,
            family: &str,
            inc_ev: f64,
            cand_ev: f64,
            candidate_valid: bool,
            invalid_reason: Option<String>,
            under_incumbent_bucket: Option<String>,
        ) {
            self.total_cases += 1;
            if candidate_valid {
                self.candidate_valid += 1;
            } else {
                self.candidate_invalid += 1;
                let key = invalid_reason.unwrap_or_else(|| "None".to_string());
                *self.invalid_reason_counts.entry(key).or_insert(0) += 1;
            }
            if !cand_ev.is_finite() || !inc_ev.is_finite() {
                return;
            }
            self.candidate_available += 1;
            let delta = cand_ev - inc_ev;
            let family_entry = self
                .family_deltas
                .entry(family.to_string())
                .or_insert((0usize, 0.0_f64));
            family_entry.0 += 1;
            family_entry.1 += delta;
            self.delta_sum += delta;
            self.best_delta = self.best_delta.max(delta);
            self.worst_delta = self.worst_delta.min(delta);
            let tol = 1e-9 * (1.0 + inc_ev.abs().max(cand_ev.abs()));
            if delta > tol {
                self.candidate_better += 1;
            } else {
                self.candidate_not_better += 1;
                let bucket = under_incumbent_bucket.unwrap_or_else(|| "other".to_string());
                *self
                    .under_incumbent_reason_counts
                    .entry(bucket)
                    .or_insert(0) += 1;
            }
        }
    }

    let mut stats = CompareStats::new();

    let mut run_case = |family: &str,
                        label: &str,
                        slot0_results: &[(
        crate::pools::Slot0Result,
        &'static crate::markets::MarketData,
    )],
                        balances: &HashMap<&str, f64>,
                        susd: f64| {
        let mut candidate_global_cfg = super::super::global_solver::GlobalSolveConfig::default();
        if let Some(value) = optimizer_override {
            candidate_global_cfg.optimizer = value;
        }
        if let Some(value) = max_iters_override {
            candidate_global_cfg.max_iters = value;
        }
        if let Some(value) = theta_l2_override {
            candidate_global_cfg.theta_l2_reg = value;
        }
        if let Some(value) = line_search_trials_override {
            candidate_global_cfg.max_line_search_trials = value;
        }
        if let Some(value) = active_set_eps_override {
            candidate_global_cfg.active_set_eps = value;
        }
        if let Some(value) = dual_router_max_iters_override {
            candidate_global_cfg.dual_router_max_iters = value;
        }
        if let Some(value) = dual_router_pg_tol_override {
            candidate_global_cfg.dual_router_pg_tol = value;
        }
        if let Some(value) = dual_router_lbfgs_history_override {
            candidate_global_cfg.dual_router_lbfgs_history = value;
        }
        if let Some(value) = dual_router_primal_restore_iters_override {
            candidate_global_cfg.dual_router_primal_restore_iters = value;
        }
        if let Some(value) = dual_router_primal_residual_tol_override {
            candidate_global_cfg.dual_router_primal_residual_tol = value;
        }
        if let Some(value) = dual_router_price_floor_override {
            candidate_global_cfg.dual_router_price_floor = value;
        }
        if let Some(value) = buy_sell_churn_reg_override {
            candidate_global_cfg.buy_sell_churn_reg = value;
        }

        let (inc_actions, inc_diag) = rebalance_with_config_and_diagnostics(
            balances,
            susd,
            slot0_results,
            RebalanceConfig {
                mode: RebalanceMode::Full,
                engine: RebalanceEngine::Incumbent,
                global: Default::default(),
            },
        );
        let (cand_actions, cand_diag) = rebalance_with_config_and_diagnostics(
            balances,
            susd,
            slot0_results,
            RebalanceConfig {
                mode: RebalanceMode::Full,
                engine: RebalanceEngine::GlobalCandidate,
                global: candidate_global_cfg,
            },
        );

        let inc_ev = inc_diag.incumbent_ev_after;
        let cand_ev = cand_diag.candidate_ev_after;
        let delta = cand_ev - inc_ev;
        let tol = 1e-9 * (1.0 + inc_ev.abs().max(cand_ev.abs()));
        let under_incumbent_bucket =
            classify_under_incumbent_bucket(&cand_diag, slot0_results.len(), delta, tol);
        let incumbent_mix = action_mix(&inc_actions);
        let candidate_mix = action_mix(&cand_actions);
        let ls_accept_ratio = if cand_diag.candidate_line_search_trials > 0 {
            cand_diag.candidate_line_search_accepts as f64
                / cand_diag.candidate_line_search_trials as f64
        } else {
            f64::NAN
        };
        println!(
            "[ev-compare] family={} case={} candidate_valid={} invalid_reason={:?} chosen={:?} optimizer={:?} incumbent_ev={:.12} candidate_ev={:.12} delta={:.12} under_incumbent_bucket={:?} pg_norm={:.9} coupled_residual={:.9} dual_residual={:.9} restore_iters={} net_theta={:.9} total_buy={:.9} total_sell={:.9} overlap={:.9} replay_cash_delta={:.9} replay_hold_delta={:.9} solver_iters={} ls_trials={} ls_accepts={} ls_invalid={} ls_rescue_attempts={} ls_rescue_accepts={} ls_accept_ratio={:.6} feasibility_repairs={} hold_clamps={} cash_scales={} active_dims={} curvature_skips={}",
            family,
            label,
            cand_diag.candidate_valid,
            cand_diag.candidate_invalid_reason,
            cand_diag.chosen_engine,
            cand_diag.candidate_optimizer,
            inc_ev,
            cand_ev,
            delta,
            under_incumbent_bucket,
            cand_diag.candidate_projected_grad_norm,
            cand_diag.candidate_coupled_residual,
            cand_diag.candidate_dual_residual_norm,
            cand_diag.candidate_primal_restore_iters,
            cand_diag.candidate_net_theta,
            cand_diag.candidate_total_buy,
            cand_diag.candidate_total_sell,
            cand_diag.candidate_buy_sell_overlap,
            cand_diag.candidate_replay_cash_delta,
            cand_diag.candidate_replay_holdings_delta,
            cand_diag.candidate_solver_iters,
            cand_diag.candidate_line_search_trials,
            cand_diag.candidate_line_search_accepts,
            cand_diag.candidate_line_search_invalid_evals,
            cand_diag.candidate_line_search_rescue_attempts,
            cand_diag.candidate_line_search_rescue_accepts,
            ls_accept_ratio,
            cand_diag.candidate_feasibility_repairs,
            cand_diag.candidate_feasibility_hold_clamps,
            cand_diag.candidate_feasibility_cash_scales,
            cand_diag.candidate_active_dims,
            cand_diag.candidate_curvature_skips
        );
        let row = CompareCaseRow {
            family,
            case: label,
            market_count: slot0_results.len(),
            optimizer: format!("{:?}", cand_diag.candidate_optimizer),
            chosen_engine: format!("{:?}", cand_diag.chosen_engine),
            candidate_valid: cand_diag.candidate_valid,
            invalid_reason: cand_diag
                .candidate_invalid_reason
                .map(|reason| format!("{reason:?}")),
            incumbent_ev: inc_ev,
            candidate_ev: cand_ev,
            delta_ev: delta,
            under_incumbent_bucket,
            candidate_projected_grad_norm: cand_diag.candidate_projected_grad_norm,
            candidate_coupled_residual: cand_diag.candidate_coupled_residual,
            candidate_dual_residual_norm: cand_diag.candidate_dual_residual_norm,
            candidate_primal_restore_iters: cand_diag.candidate_primal_restore_iters,
            candidate_net_theta: cand_diag.candidate_net_theta,
            candidate_total_buy: cand_diag.candidate_total_buy,
            candidate_total_sell: cand_diag.candidate_total_sell,
            candidate_buy_sell_overlap: cand_diag.candidate_buy_sell_overlap,
            candidate_replay_cash_delta: cand_diag.candidate_replay_cash_delta,
            candidate_replay_holdings_delta: cand_diag.candidate_replay_holdings_delta,
            candidate_solver_iters: cand_diag.candidate_solver_iters,
            candidate_line_search_trials: cand_diag.candidate_line_search_trials,
            candidate_line_search_accepts: cand_diag.candidate_line_search_accepts,
            candidate_line_search_invalid_evals: cand_diag.candidate_line_search_invalid_evals,
            candidate_line_search_rescue_attempts: cand_diag.candidate_line_search_rescue_attempts,
            candidate_line_search_rescue_accepts: cand_diag.candidate_line_search_rescue_accepts,
            candidate_feasibility_repairs: cand_diag.candidate_feasibility_repairs,
            candidate_feasibility_hold_clamps: cand_diag.candidate_feasibility_hold_clamps,
            candidate_feasibility_cash_scales: cand_diag.candidate_feasibility_cash_scales,
            candidate_active_dims: cand_diag.candidate_active_dims,
            candidate_curvature_skips: cand_diag.candidate_curvature_skips,
            incumbent_action_mix: incumbent_mix,
            candidate_action_mix: candidate_mix,
        };
        let row_json =
            serde_json::to_string(&row).expect("expected EV compare row to serialize into JSON");
        writeln!(jsonl_writer, "{row_json}")
            .expect("expected to append JSONL telemetry row for EV compare case");
        stats.record(
            family,
            inc_ev,
            cand_ev,
            cand_diag.candidate_valid,
            cand_diag
                .candidate_invalid_reason
                .map(|reason| format!("{reason:?}")),
            under_incumbent_bucket.map(str::to_string),
        );
    };

    // Full-L1 fuzz fixture family.
    {
        let mut rng = TestRng::new(0xFEED_FACE_1234_4321u64);
        for case_idx in 0..24 {
            let (slot0_results, balances_static, susd_balance) =
                build_rebalance_fuzz_case(&mut rng, false);
            let balances: HashMap<&str, f64> = balances_static
                .iter()
                .map(|(k, v)| (*k as &str, *v))
                .collect();
            let label = format!("fuzz_full_l1_case_{}", case_idx);
            run_case("fuzz_full_l1", &label, &slot0_results, &balances, susd_balance);
        }
    }

    // Partial-L1 fuzz fixture family.
    {
        let mut rng = TestRng::new(0xABCD_1234_EF99_7788u64);
        for case_idx in 0..24 {
            let (slot0_results, balances_static, susd_balance) =
                build_rebalance_fuzz_case(&mut rng, true);
            let balances: HashMap<&str, f64> = balances_static
                .iter()
                .map(|(k, v)| (*k as &str, *v))
                .collect();
            let label = format!("fuzz_partial_l1_case_{}", case_idx);
            run_case(
                "fuzz_partial_l1",
                &label,
                &slot0_results,
                &balances,
                susd_balance,
            );
        }
    }

    // Full-L1 regression snapshot A.
    {
        let markets = eligible_l1_markets_with_predictions();
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
        run_case(
            "regression_full_l1",
            "regression_full_l1_snapshot",
            &slot0_results,
            &balances,
            83.0,
        );
    }

    // Full-L1 regression snapshot B.
    {
        let markets = eligible_l1_markets_with_predictions();
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
        run_case(
            "regression_full_l1",
            "regression_full_l1_snapshot_variant_b",
            &slot0_results,
            &balances,
            41.0,
        );
    }

    let mean_delta = if stats.candidate_available > 0 {
        stats.delta_sum / (stats.candidate_available as f64)
    } else {
        f64::NAN
    };

    println!(
        "[ev-compare][summary] total_cases={} candidate_available={} candidate_valid={} candidate_invalid={} candidate_better={} candidate_not_better={} mean_delta={:.12} best_delta={:.12} worst_delta={:.12}",
        stats.total_cases,
        stats.candidate_available,
        stats.candidate_valid,
        stats.candidate_invalid,
        stats.candidate_better,
        stats.candidate_not_better,
        mean_delta,
        stats.best_delta,
        stats.worst_delta
    );

    let mut reason_pairs: Vec<_> = stats.invalid_reason_counts.iter().collect();
    reason_pairs.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    for (reason, count) in reason_pairs {
        println!(
            "[ev-compare][invalid-reason] reason={} count={}",
            reason, count
        );
    }

    let mut under_pairs: Vec<_> = stats.under_incumbent_reason_counts.iter().collect();
    under_pairs.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    for (reason, count) in under_pairs {
        println!(
            "[ev-compare][under-incumbent] reason={} count={}",
            reason, count
        );
    }

    let mut family_pairs: Vec<_> = stats.family_deltas.iter().collect();
    family_pairs.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    for (family, (count, delta_sum)) in family_pairs {
        let mean = if *count > 0 {
            *delta_sum / (*count as f64)
        } else {
            f64::NAN
        };
        println!(
            "[ev-compare][family] family={} cases={} mean_delta={:.12}",
            family, count, mean
        );
    }

    jsonl_writer
        .flush()
        .expect("expected to flush JSONL telemetry file");
    println!("[ev-compare] wrote_jsonl={}", jsonl_path);
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
                    saturated[i].set_price(p_new);
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
        let (cash_cost, value_cost, mint_amount, d_cost_d_pi) =
            mint_cost_to_prof(&sims, target_idx, target_prof, &HashSet::new(), price_sum)
                .expect("solver should return saturated mint solution when target is unreachable");

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
                    simulated[i].set_price(p_new);
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

        let sell_amount = rng.in_range(0.0, sim_balances.get("M1").copied().unwrap_or(0.0) + 2.5);
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
    for case_idx in 0..24 {
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
        let label = format!("fuzz_full_l1_case_{}", case_idx);
        let (_, ev_after, _) = assert_strict_ev_gain_with_portfolio_trace(
            &label,
            &actions_a,
            &slot0_results,
            &balances,
            susd_balance,
        );
        let expected_after = ev_snapshots().fuzz_full_l1_case_ev_after[case_idx];
        let (ok, tol) = ev_meets_floor(ev_after, expected_after);
        assert!(
            ok,
            "full-L1 fuzz EV regressed for case {}: got={:.12}, floor={:.12}, tol={:.12}",
            case_idx, ev_after, expected_after, tol
        );
    }
}

#[test]
fn test_fuzz_rebalance_end_to_end_partial_l1_invariants() {
    let mut rng = TestRng::new(0xABCD_1234_EF99_7788u64);
    for case_idx in 0..24 {
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
        let label = format!("fuzz_partial_l1_case_{}", case_idx);
        let (_, ev_after, _) = assert_strict_ev_gain_with_portfolio_trace(
            &label,
            &actions,
            &slot0_results,
            &balances,
            susd_balance,
        );
        let expected_after = ev_snapshots().fuzz_partial_l1_case_ev_after[case_idx];
        let (ok, tol) = ev_meets_floor(ev_after, expected_after);
        assert!(
            ok,
            "partial-L1 fuzz EV regressed for case {}: got={:.12}, floor={:.12}, tol={:.12}",
            case_idx, ev_after, expected_after, tol
        );
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

    let (ev_before, ev_after, _) = assert_strict_ev_gain_with_portfolio_trace(
        "regression_full_l1_snapshot",
        &actions_a,
        &slot0_results,
        &balances,
        budget,
    );
    let gain = ev_after - ev_before;

    const EXPECTED_EV_BEFORE: f64 = 83.329_134_223;
    const EXPECTED_EV_AFTER_FLOOR: f64 = 224.381_013_963;
    const EV_TOL: f64 = 3e-6;

    assert!(
        (ev_before - EXPECTED_EV_BEFORE).abs() <= EV_TOL,
        "ev_before drifted: got={:.9}, expected={:.9}, tol={:.9}",
        ev_before,
        EXPECTED_EV_BEFORE,
        EV_TOL
    );
    let (ok_after_floor, after_floor_tol) = ev_meets_floor(ev_after, EXPECTED_EV_AFTER_FLOOR);
    assert!(
        ok_after_floor,
        "ev_after regressed: got={:.9}, floor={:.9}, tol={:.9}",
        ev_after, EXPECTED_EV_AFTER_FLOOR, after_floor_tol
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

    let (ev_before, ev_after, _) = assert_strict_ev_gain_with_portfolio_trace(
        "regression_full_l1_snapshot_variant_b",
        &actions_a,
        &slot0_results,
        &balances,
        budget,
    );
    let gain = ev_after - ev_before;

    const EXPECTED_EV_BEFORE: f64 = 41.229_354_975;
    const EXPECTED_EV_AFTER_FLOOR: f64 = 45.865_172_947;
    const EV_TOL: f64 = 3e-6;

    assert!(
        (ev_before - EXPECTED_EV_BEFORE).abs() <= EV_TOL,
        "variant-B ev_before drifted: got={:.9}, expected={:.9}, tol={:.9}",
        ev_before,
        EXPECTED_EV_BEFORE,
        EV_TOL
    );
    let (ok_after_floor, after_floor_tol) = ev_meets_floor(ev_after, EXPECTED_EV_AFTER_FLOOR);
    assert!(
        ok_after_floor,
        "variant-B ev_after regressed: got={:.9}, floor={:.9}, tol={:.9}",
        ev_after, EXPECTED_EV_AFTER_FLOOR, after_floor_tol
    );
    assert!(
        gain > 0.0,
        "regression fixture should improve EV: before={:.9}, after={:.9}",
        ev_before,
        ev_after
    );
}
