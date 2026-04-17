mod client;
mod protocol;
mod translate;

use std::collections::HashMap;
use std::fmt;
#[cfg(test)]
use std::sync::MutexGuard;
use std::time::Duration;

use serde::Serialize;

use crate::markets::MarketData;
use crate::pools::Slot0Result;

use super::Action;
use super::rebalancer::PlannerCostConfig;

#[derive(Debug, Clone, Serialize)]
pub struct ForecastFlowsDoctorReport {
    pub worker_backend: String,
    pub worker_program: String,
    pub worker_version: Option<String>,
    pub worker_project_dir: Option<String>,
    pub worker_launch_args: Vec<String>,
    pub worker_runtime_detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub julia_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_repo_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_repo_rev: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_git_tree_sha1: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub julia_threads: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_plain_julia_escape_hatch: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sysimage_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sysimage_detail: Option<String>,
    pub live_solve_tuning: String,
    pub health_status: Option<String>,
    pub supported_commands: Vec<String>,
    pub supported_modes: Vec<String>,
    pub execution_model: Option<String>,
    pub health_duration_ms: Option<u128>,
    pub warmup_compare_duration_ms: Option<u128>,
    pub warmup_direct_status: Option<String>,
    pub warmup_mixed_status: Option<String>,
    pub warmup_direct_solver_time_ms: Option<u128>,
    pub warmup_mixed_solver_time_ms: Option<u128>,
    pub warmup_driver_overhead_ms: Option<u128>,
    pub representative_compare_duration_ms: Option<u128>,
    pub representative_direct_status: Option<String>,
    pub representative_mixed_status: Option<String>,
    pub representative_direct_solver_time_ms: Option<u128>,
    pub representative_mixed_solver_time_ms: Option<u128>,
    pub representative_driver_overhead_ms: Option<u128>,
    pub representative_fixture: String,
    pub stderr_tail: Vec<String>,
    pub failure: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForecastFlowsCandidateVariant {
    Direct,
    Mixed,
}

impl ForecastFlowsCandidateVariant {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ForecastFlowsFamilyCandidate {
    pub(super) actions: Vec<Action>,
    pub(super) variant: ForecastFlowsCandidateVariant,
    pub(super) estimated_execution_cost_susd: Option<f64>,
    pub(super) estimated_net_ev_susd: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ForecastFlowsTelemetry {
    pub(super) strategy: Option<String>,
    pub(super) worker_roundtrip_ms: Option<u128>,
    pub(super) driver_overhead_ms: Option<u128>,
    pub(super) translation_replay_ms: Option<u128>,
    pub(super) local_candidate_build_ms: Option<u128>,
    pub(super) local_step_prune_ms: Option<u128>,
    pub(super) local_route_prune_ms: Option<u128>,
    pub(super) workspace_reused: bool,
    pub(super) direct_status: Option<String>,
    pub(super) mixed_status: Option<String>,
    pub(super) direct_solver_time_ms: Option<u128>,
    pub(super) mixed_solver_time_ms: Option<u128>,
    pub(super) estimated_execution_cost_susd: Option<f64>,
    pub(super) estimated_net_ev_susd: Option<f64>,
    pub(super) validated_total_fee_susd: Option<f64>,
    pub(super) validated_net_ev_susd: Option<f64>,
    pub(super) fee_estimate_error_susd: Option<f64>,
    pub(super) validation_only: bool,
    pub(super) certified_drop_reason: Option<String>,
    pub(super) replay_drop_reason: Option<String>,
    pub(super) replay_tolerance_clamp_used: bool,
    pub(super) forecastflows_backend: Option<String>,
    pub(super) forecastflows_worker_version: Option<String>,
    pub(super) sysimage_status: Option<String>,
    pub(super) julia_threads: Option<String>,
    pub(super) solve_tuning: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct ForecastFlowsSolveReport {
    pub(super) candidates: Vec<ForecastFlowsFamilyCandidate>,
    pub(super) request_count: usize,
    pub(super) telemetry: ForecastFlowsTelemetry,
}

#[derive(Debug)]
pub(super) enum ForecastFlowsSolveError {
    Worker {
        message: String,
        request_count: usize,
        fallback_reason: &'static str,
        telemetry: ForecastFlowsTelemetry,
    },
    Translation {
        message: String,
        request_count: usize,
        telemetry: ForecastFlowsTelemetry,
    },
}

impl fmt::Display for ForecastFlowsSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Worker { message, .. } => write!(f, "{message}"),
            Self::Translation { message, .. } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ForecastFlowsSolveError {}

impl From<client::ForecastFlowsClientError> for ForecastFlowsSolveError {
    fn from(value: client::ForecastFlowsClientError) -> Self {
        Self::Worker {
            message: value.to_string(),
            request_count: 0,
            fallback_reason: value.fallback_reason(),
            telemetry: ForecastFlowsTelemetry::default(),
        }
    }
}

impl From<client::ForecastFlowsCompareFailure> for ForecastFlowsSolveError {
    fn from(value: client::ForecastFlowsCompareFailure) -> Self {
        let message = value.to_string();
        let request_count = value.request_count;
        let fallback_reason = value.fallback_reason();
        let telemetry = value
            .runtime_policy
            .map(|runtime_policy| ForecastFlowsTelemetry {
                forecastflows_backend: Some(runtime_policy.backend.as_str().to_string()),
                forecastflows_worker_version: runtime_policy.worker_version,
                sysimage_status: runtime_policy.sysimage_status,
                julia_threads: runtime_policy.julia_threads,
                solve_tuning: Some(runtime_policy.solve_tuning),
                ..ForecastFlowsTelemetry::default()
            })
            .unwrap_or_default();
        Self::Worker {
            message,
            request_count,
            fallback_reason,
            telemetry,
        }
    }
}

impl ForecastFlowsSolveError {
    pub(super) fn request_count(&self) -> usize {
        match self {
            Self::Worker { request_count, .. } | Self::Translation { request_count, .. } => {
                *request_count
            }
        }
    }

    pub(super) fn worker_available(&self) -> bool {
        matches!(
            self,
            Self::Translation { request_count, .. } | Self::Worker { request_count, .. }
                if *request_count > 0
        )
    }

    pub(super) fn fallback_reason(&self) -> &'static str {
        match self {
            Self::Worker {
                fallback_reason, ..
            } => fallback_reason,
            Self::Translation { request_count, .. } if *request_count == 0 => {
                "request_construction_error"
            }
            Self::Translation { .. } => "translation_error",
        }
    }

    pub(super) fn telemetry(&self) -> Option<&ForecastFlowsTelemetry> {
        match self {
            Self::Translation { telemetry, .. } | Self::Worker { telemetry, .. } => Some(telemetry),
        }
    }
}

fn solver_time_ms(solver_time_sec: Option<f64>) -> Option<u128> {
    let solver_time_sec = solver_time_sec?;
    if !solver_time_sec.is_finite() || solver_time_sec < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(solver_time_sec).as_millis())
}

pub(super) fn solve_family_candidates(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
    _cost_config: PlannerCostConfig,
) -> Result<ForecastFlowsSolveReport, ForecastFlowsSolveError> {
    let problem = translate::build_problem_request(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_outcome_count,
    )
    .map_err(|err| ForecastFlowsSolveError::Translation {
        message: err.to_string(),
        request_count: 0,
        telemetry: ForecastFlowsTelemetry::default(),
    })?;
    let compare_report = client::compare_prediction_market_families(problem)?;
    let direct_solver_time_ms = solver_time_ms(compare_report.compare.direct_only.solver_time_sec);
    let mixed_solver_time_ms = solver_time_ms(compare_report.compare.mixed_enabled.solver_time_sec);
    let mut telemetry = ForecastFlowsTelemetry {
        strategy: Some("rust_prune".to_string()),
        worker_roundtrip_ms: Some(compare_report.roundtrip.as_millis()),
        driver_overhead_ms: compare_report
            .roundtrip
            .as_millis()
            .checked_sub(direct_solver_time_ms.unwrap_or(0) + mixed_solver_time_ms.unwrap_or(0)),
        workspace_reused: compare_report.compare.workspace_reused,
        direct_status: Some(compare_report.compare.direct_only.status.clone()),
        mixed_status: Some(compare_report.compare.mixed_enabled.status.clone()),
        direct_solver_time_ms,
        mixed_solver_time_ms,
        forecastflows_backend: Some(compare_report.runtime_policy.backend.as_str().to_string()),
        forecastflows_worker_version: compare_report.runtime_policy.worker_version.clone(),
        sysimage_status: compare_report.runtime_policy.sysimage_status.clone(),
        julia_threads: compare_report.runtime_policy.julia_threads.clone(),
        solve_tuning: Some(compare_report.runtime_policy.solve_tuning.clone()),
        ..ForecastFlowsTelemetry::default()
    };
    let translation_started = std::time::Instant::now();
    let report = translate::translate_compare_result_report(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        compare_report.compare,
    );
    telemetry.translation_replay_ms = Some(translation_started.elapsed().as_millis());
    telemetry.certified_drop_reason = report.first_non_replay_drop_reason();
    telemetry.replay_drop_reason = report.first_replay_drop_reason();
    telemetry.replay_tolerance_clamp_used = report.replay_tolerance_clamp_used;
    for dropped in &report.drop_reasons {
        tracing::debug!(
            variant = dropped.variant.as_str(),
            stage = dropped.stage.as_str(),
            reason = %dropped.reason,
            "ForecastFlows certified candidate dropped during translation"
        );
    }
    if let Some(message) = report.all_certified_candidates_dropped_message() {
        return Err(ForecastFlowsSolveError::Translation {
            message,
            request_count: compare_report.request_count,
            telemetry,
        });
    }
    Ok(ForecastFlowsSolveReport {
        candidates: report.candidates,
        request_count: compare_report.request_count,
        telemetry,
    })
}

pub(super) fn warm_worker() -> Result<(), ForecastFlowsSolveError> {
    client::warm_worker().map_err(Into::into)
}

pub(super) fn shutdown_worker() -> Result<(), ForecastFlowsSolveError> {
    client::shutdown_worker().map_err(Into::into)
}

pub(super) fn doctor_report() -> Result<ForecastFlowsDoctorReport, ForecastFlowsSolveError> {
    client::doctor_report().map_err(Into::into)
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForecastFlowsBenchmarkSolveTuning {
    Baseline,
    LowLatency,
}

#[cfg(test)]
impl ForecastFlowsBenchmarkSolveTuning {
    fn as_client_tuning(self) -> client::ForecastFlowsSolveTuning {
        match self {
            Self::Baseline => client::ForecastFlowsSolveTuning::Baseline,
            Self::LowLatency => client::ForecastFlowsSolveTuning::LowLatency,
        }
    }
}

#[cfg(test)]
struct BenchmarkTestSessionGuard {
    _lock: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for BenchmarkTestSessionGuard {
    fn drop(&mut self) {
        let _ = shutdown_worker();
        client::reset_test_overrides();
    }
}

#[cfg(test)]
fn benchmark_test_session_guard() -> BenchmarkTestSessionGuard {
    let lock = client::forecastflows_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    client::reset_test_overrides();
    client::set_test_request_profile(client::ForecastFlowsRequestProfile::Benchmark);
    client::set_test_request_timeout(Duration::from_secs(protocol::WARMUP_REQUEST_TIMEOUT_SECS));
    BenchmarkTestSessionGuard { _lock: lock }
}

#[cfg(test)]
pub(super) fn with_benchmark_test_session_for_test<T>(
    configure: impl FnOnce(),
    run: impl FnOnce() -> Result<T, ForecastFlowsSolveError>,
) -> Result<T, ForecastFlowsSolveError> {
    let _guard = benchmark_test_session_guard();
    configure();
    run()
}

#[cfg(test)]
pub(super) fn with_live_benchmark_worker_for_test<T>(
    run: impl FnOnce() -> T,
) -> Result<T, ForecastFlowsSolveError> {
    with_benchmark_test_session_for_test(
        || {},
        || {
            warm_worker()?;
            Ok(run())
        },
    )
}

#[cfg(test)]
pub(super) fn with_live_benchmark_worker_and_tuning_for_test<T>(
    tuning: ForecastFlowsBenchmarkSolveTuning,
    run: impl FnOnce() -> T,
) -> Result<T, ForecastFlowsSolveError> {
    with_benchmark_test_session_for_test(
        || client::set_test_solve_tuning(tuning.as_client_tuning()),
        || {
            warm_worker()?;
            Ok(run())
        },
    )
}

#[cfg(test)]
pub(super) fn with_live_benchmark_worker_timeout_and_tuning_for_test<T>(
    tuning: ForecastFlowsBenchmarkSolveTuning,
    timeout: Duration,
    run: impl FnOnce() -> T,
) -> Result<T, ForecastFlowsSolveError> {
    with_benchmark_test_session_for_test(
        || {
            client::set_test_solve_tuning(tuning.as_client_tuning());
            client::set_test_request_timeout(timeout);
        },
        || {
            warm_worker()?;
            Ok(run())
        },
    )
}

#[cfg(test)]
pub(super) fn build_problem_request_band_counts_for_test(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
) -> Result<Vec<usize>, String> {
    let problem = translate::build_problem_request(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        expected_outcome_count,
    )
    .map_err(|err| err.to_string())?;
    Ok(problem
        .markets
        .iter()
        .map(|market| match market {
            protocol::MarketSpecRequest::UniV3 { bands, .. } => bands.len(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use alloy::primitives::{Address, U256};

    use super::{
        ForecastFlowsCandidateVariant, protocol, solve_family_candidates,
        with_benchmark_test_session_for_test,
    };
    use crate::markets::{MarketData, Pool, Tick};
    use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};
    use crate::portfolio::Action;

    use super::super::rebalancer::{
        RebalanceSolver, benchmark_planner_cost_config_for_test,
        rebalance_with_custom_predictions_and_solver_and_stats_for_test,
        rebalance_with_custom_predictions_and_solver_for_test,
    };
    use super::client::{
        forecastflows_test_lock, reset_test_overrides, set_test_circuit_breaker_config,
        set_test_request_timeout, set_test_solve_tuning, set_test_worker_command,
        test_request_profile_override, test_request_timeout_override, test_worker_harness_state,
    };

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
        let balances = HashMap::new();
        let predictions = HashMap::from([("m1".to_string(), 0.9), ("m2".to_string(), 0.1)]);
        (slot0_results, balances, predictions)
    }

    fn with_fake_worker<T>(
        scenario: &str,
        timeout: Option<Duration>,
        run: impl FnOnce() -> T,
    ) -> T {
        let _guard = forecastflows_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let script = format!(
            "{}/test/bin/fake_forecastflows_worker.sh",
            env!("CARGO_MANIFEST_DIR")
        );
        reset_test_overrides();
        set_test_worker_command(&["/bin/sh", &script, scenario]);
        if let Some(timeout) = timeout {
            set_test_request_timeout(timeout);
        }
        let result = run();
        reset_test_overrides();
        result
    }

    #[test]
    fn forecastflows_solver_uses_direct_candidate_from_worker() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let actions = with_fake_worker("healthy_direct", None, || {
            rebalance_with_custom_predictions_and_solver_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::Buy {
                market_name: "M1",
                ..
            }
        ));
    }

    #[test]
    fn forecastflows_default_live_path_uses_rust_prune() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let (_actions, stats) = with_fake_worker("healthy_direct", None, || {
            rebalance_with_custom_predictions_and_solver_and_stats_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(
            stats.forecastflows_telemetry.strategy.as_deref(),
            Some("rust_prune")
        );
        assert!(!stats.forecastflows_telemetry.validation_only);
        assert!(
            stats
                .forecastflows_telemetry
                .local_candidate_build_ms
                .is_some()
        );
        assert!(stats.forecastflows_telemetry.local_step_prune_ms.is_some());
        assert!(stats.forecastflows_telemetry.local_route_prune_ms.is_some());
    }

    #[test]
    fn forecastflows_solver_uses_mixed_candidate_from_worker() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let actions = with_fake_worker("healthy_mixed", None, || {
            rebalance_with_custom_predictions_and_solver_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(actions.len(), 3);
        assert!(matches!(actions[0], Action::Mint { .. }));
        assert!(matches!(
            actions[1],
            Action::Sell {
                market_name: "M2",
                ..
            }
        ));
        assert!(matches!(
            actions[2],
            Action::Buy {
                market_name: "M1",
                ..
            }
        ));
    }

    #[test]
    fn head_to_head_uses_local_plan_value_not_worker_reported_ev() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let native_actions = rebalance_with_custom_predictions_and_solver_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );
        let head_to_head_actions = with_fake_worker("no_op_huge_ev", None, || {
            rebalance_with_custom_predictions_and_solver_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::HeadToHead,
            )
        });

        assert!(
            !native_actions.is_empty(),
            "fixture should produce a non-empty native plan"
        );
        assert_eq!(head_to_head_actions, native_actions);
    }

    #[test]
    fn forecastflows_solver_fails_open_to_native_on_malformed_worker_output() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let native_actions = rebalance_with_custom_predictions_and_solver_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );
        let fallback_actions = with_fake_worker("malformed", None, || {
            rebalance_with_custom_predictions_and_solver_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(fallback_actions, native_actions);
    }

    #[test]
    fn forecastflows_solver_fails_open_to_native_on_timeout() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let native_actions = rebalance_with_custom_predictions_and_solver_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );
        let fallback_actions =
            with_fake_worker("timeout", Some(Duration::from_millis(100)), || {
                rebalance_with_custom_predictions_and_solver_for_test(
                    &balances,
                    1.0,
                    &slot0_results,
                    &predictions,
                    true,
                    RebalanceSolver::ForecastFlows,
                )
            });

        assert_eq!(fallback_actions, native_actions);
    }

    #[test]
    fn solve_family_candidates_rejects_invalid_mint_merge_shape() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let cost_config = benchmark_planner_cost_config_for_test();
        let err = with_fake_worker("invalid_mint_merge", None, || {
            solve_family_candidates(&balances, 1.0, &slot0_results, &predictions, 2, cost_config)
                .expect_err("invalid mint+merge shape should be rejected")
        });
        assert!(err.to_string().contains("cannot mint and merge"));
    }

    #[test]
    #[ignore = "opt-in live Julia worker smoke test"]
    fn live_worker_smoke_test() {
        let _guard = forecastflows_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_test_overrides();
        super::warm_worker().expect("worker warmup should succeed");

        let (slot0_results, balances, predictions) = two_market_fixture();
        let report = solve_family_candidates(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            2,
            benchmark_planner_cost_config_for_test(),
        )
        .expect("live worker should solve fixture");
        assert!(
            !report.candidates.is_empty(),
            "live worker should return at least one certified family candidate"
        );
    }

    #[test]
    fn forecastflows_solver_falls_back_on_uncertified_only_result() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let (native_actions, _) = rebalance_with_custom_predictions_and_solver_and_stats_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );
        let (fallback_actions, stats) = with_fake_worker("uncertified_only", None, || {
            rebalance_with_custom_predictions_and_solver_and_stats_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(fallback_actions, native_actions);
        assert_eq!(stats.forecastflows_requests, 1);
        assert!(stats.forecastflows_worker_available);
        assert_eq!(
            stats.forecastflows_fallback_reason,
            Some("no_certified_candidate")
        );
    }

    #[test]
    fn forecastflows_solver_fails_open_to_native_on_worker_error_response() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let (native_actions, _) = rebalance_with_custom_predictions_and_solver_and_stats_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );
        let (fallback_actions, stats) = with_fake_worker("worker_error", None, || {
            rebalance_with_custom_predictions_and_solver_and_stats_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(fallback_actions, native_actions);
        assert_eq!(stats.forecastflows_requests, 1);
        assert!(stats.forecastflows_worker_available);
        assert_eq!(
            stats.forecastflows_fallback_reason,
            Some("worker_error_response")
        );
        assert_eq!(
            stats
                .forecastflows_telemetry
                .forecastflows_backend
                .as_deref(),
            Some("julia_worker"),
        );
        assert!(
            stats
                .forecastflows_telemetry
                .sysimage_status
                .as_deref()
                .is_some_and(|value| !value.is_empty())
        );
    }

    #[test]
    fn forecastflows_solver_marks_pre_worker_request_construction_failure() {
        let (mut slot0_results, balances, predictions) = two_market_fixture();
        let liquidity = 1_000_000_000_000_000_000_000u128;
        let gap_ticks = Box::leak(Box::new([
            Tick {
                tick_idx: -16_096,
                liquidity_net: liquidity as i128,
            },
            Tick {
                tick_idx: 92_108,
                liquidity_net: -(liquidity as i128),
            },
            Tick {
                tick_idx: 100_000,
                liquidity_net: liquidity as i128,
            },
            Tick {
                tick_idx: 110_000,
                liquidity_net: -(liquidity as i128),
            },
        ]));
        let liquidity_str = Box::leak(liquidity.to_string().into_boxed_str());
        slot0_results[0].1 = leak_market(MarketData {
            name: "M1",
            market_id: "0xmarket-contract",
            outcome_token: "0x1111111111111111111111111111111111111111",
            pool: Some(Pool {
                token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
                token1: "0x1111111111111111111111111111111111111111",
                pool_id: "0xpool",
                liquidity: liquidity_str,
                ticks: gap_ticks,
            }),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        });
        let native_actions = rebalance_with_custom_predictions_and_solver_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );

        let (actions, stats) = rebalance_with_custom_predictions_and_solver_and_stats_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::ForecastFlows,
        );

        assert!(stats.forecastflows_requests == 0);
        assert!(!stats.forecastflows_worker_available);
        assert_eq!(
            stats.forecastflows_fallback_reason,
            Some("request_construction_error")
        );
        assert_eq!(actions, native_actions);
    }

    #[test]
    fn forecastflows_solver_records_winning_variant() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let (_, stats) = with_fake_worker("healthy_mixed", None, || {
            rebalance_with_custom_predictions_and_solver_and_stats_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(
            stats.forecastflows_winning_variant,
            Some(ForecastFlowsCandidateVariant::Mixed)
        );
    }

    #[test]
    fn forecastflows_solver_fails_open_to_native_on_closed_worker() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let native_actions = rebalance_with_custom_predictions_and_solver_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );
        let fallback_actions = with_fake_worker("closed", None, || {
            rebalance_with_custom_predictions_and_solver_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(fallback_actions, native_actions);
    }

    #[test]
    fn forecastflows_solver_enters_worker_cooldown_after_repeated_transport_failure() {
        let (slot0_results, balances, predictions) = two_market_fixture();
        let native_actions = rebalance_with_custom_predictions_and_solver_for_test(
            &balances,
            1.0,
            &slot0_results,
            &predictions,
            true,
            RebalanceSolver::Native,
        );
        let (fallback_actions, stats) = with_fake_worker("closed", None, || {
            set_test_circuit_breaker_config(1, Duration::from_secs(5), Duration::from_secs(5));
            let _ = rebalance_with_custom_predictions_and_solver_and_stats_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            );
            rebalance_with_custom_predictions_and_solver_and_stats_for_test(
                &balances,
                1.0,
                &slot0_results,
                &predictions,
                true,
                RebalanceSolver::ForecastFlows,
            )
        });

        assert_eq!(fallback_actions, native_actions);
        assert_eq!(stats.forecastflows_fallback_reason, Some("worker_cooldown"));
    }

    #[test]
    fn benchmark_session_helper_resets_timeout_override_and_worker_state() {
        let script = format!(
            "{}/test/bin/fake_forecastflows_worker.sh",
            env!("CARGO_MANIFEST_DIR")
        );

        with_benchmark_test_session_for_test(
            || set_test_worker_command(&["/bin/sh", &script, "healthy_direct"]),
            || {
                super::warm_worker().expect("healthy fake worker should warm");
                assert_eq!(
                    test_request_timeout_override(),
                    Some(Duration::from_secs(protocol::WARMUP_REQUEST_TIMEOUT_SECS))
                );
                assert_eq!(
                    test_request_profile_override(),
                    Some(super::client::ForecastFlowsRequestProfile::Benchmark)
                );
                let state = test_worker_harness_state();
                assert!(state.process_present);
                assert!(state.warm_complete);
                Ok(())
            },
        )
        .expect("benchmark session should succeed");

        assert_eq!(test_request_timeout_override(), None);
        assert_eq!(test_request_profile_override(), None);
        let state = test_worker_harness_state();
        assert!(!state.process_present);
        assert!(!state.warm_complete);
    }

    #[test]
    fn tuning_override_keeps_fake_worker_selection_behavior_stable() {
        let script = format!(
            "{}/test/bin/fake_forecastflows_worker.sh",
            env!("CARGO_MANIFEST_DIR")
        );
        let (slot0_results, balances, predictions) = two_market_fixture();
        let run_with_tuning = |tuning| {
            with_benchmark_test_session_for_test(
                || {
                    set_test_worker_command(&["/bin/sh", &script, "healthy_direct"]);
                    set_test_solve_tuning(tuning);
                },
                || {
                    Ok(
                        rebalance_with_custom_predictions_and_solver_and_stats_for_test(
                            &balances,
                            1.0,
                            &slot0_results,
                            &predictions,
                            true,
                            RebalanceSolver::ForecastFlows,
                        ),
                    )
                },
            )
            .expect("fake worker tuning session should succeed")
        };

        let baseline = run_with_tuning(super::client::ForecastFlowsSolveTuning::Baseline);
        let low_latency = run_with_tuning(super::client::ForecastFlowsSolveTuning::LowLatency);

        assert_eq!(baseline.0, low_latency.0);
        assert_eq!(baseline.1.forecastflows_fallback_reason, None);
        assert_eq!(low_latency.1.forecastflows_fallback_reason, None);
        assert_eq!(
            baseline.1.forecastflows_telemetry.direct_status.as_deref(),
            Some("certified")
        );
        assert_eq!(
            low_latency
                .1
                .forecastflows_telemetry
                .direct_status
                .as_deref(),
            Some("certified")
        );
        assert_eq!(
            baseline.1.forecastflows_telemetry.solve_tuning.as_deref(),
            Some("baseline")
        );
        assert_eq!(
            low_latency
                .1
                .forecastflows_telemetry
                .solve_tuning
                .as_deref(),
            Some("low_latency")
        );
    }
}
