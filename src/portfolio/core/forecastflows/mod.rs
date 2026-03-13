mod client;
mod protocol;
mod translate;

use std::collections::HashMap;
use std::fmt;

use crate::markets::MarketData;
use crate::pools::Slot0Result;

use super::Action;

#[derive(Debug, Clone)]
pub struct ForecastFlowsDoctorReport {
    pub julia_program: String,
    pub julia_version: Option<String>,
    pub project_dir: String,
    pub launch_args: Vec<String>,
    pub sysimage_status: String,
    pub sysimage_detail: String,
    pub health_status: Option<String>,
    pub supported_commands: Vec<String>,
    pub supported_modes: Vec<String>,
    pub execution_model: Option<String>,
    pub health_duration_ms: Option<u128>,
    pub warmup_compare_duration_ms: Option<u128>,
    pub representative_compare_duration_ms: Option<u128>,
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
}

#[derive(Debug, Clone)]
pub(super) struct ForecastFlowsSolveReport {
    pub(super) candidates: Vec<ForecastFlowsFamilyCandidate>,
    pub(super) request_count: usize,
}

#[derive(Debug)]
pub(super) enum ForecastFlowsSolveError {
    Worker {
        message: String,
        request_count: usize,
        fallback_reason: &'static str,
    },
    Translation {
        message: String,
        request_count: usize,
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
        }
    }
}

impl From<client::ForecastFlowsCompareFailure> for ForecastFlowsSolveError {
    fn from(value: client::ForecastFlowsCompareFailure) -> Self {
        Self::Worker {
            message: value.to_string(),
            request_count: value.request_count,
            fallback_reason: value.fallback_reason(),
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
        matches!(self, Self::Translation { request_count, .. } if *request_count > 0)
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
}

pub(super) fn solve_family_candidates(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    predictions: &HashMap<String, f64>,
    expected_outcome_count: usize,
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
    })?;
    let (compare, request_count) = client::compare_prediction_market_families(problem)?;
    let report = translate::translate_compare_result_report(
        balances,
        susds_balance,
        slot0_results,
        predictions,
        compare,
    )
    .map_err(|err| ForecastFlowsSolveError::Translation {
        message: err.to_string(),
        request_count,
    })?;
    for dropped in &report.certified_drop_reasons {
        tracing::debug!(
            variant = dropped.variant.as_str(),
            reason = %dropped.reason,
            "ForecastFlows certified candidate dropped during translation"
        );
    }
    Ok(ForecastFlowsSolveReport {
        candidates: report.candidates,
        request_count,
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
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use alloy::primitives::{Address, U256};

    use super::{ForecastFlowsCandidateVariant, solve_family_candidates};
    use crate::markets::{MarketData, Pool, Tick};
    use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};
    use crate::portfolio::Action;

    use super::super::rebalancer::{
        RebalanceSolver, rebalance_with_custom_predictions_and_solver_and_stats_for_test,
        rebalance_with_custom_predictions_and_solver_for_test,
    };
    use super::client::{
        forecastflows_test_lock, reset_test_overrides, set_test_circuit_breaker_config,
        set_test_request_timeout, set_test_worker_command,
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
        let err = with_fake_worker("invalid_mint_merge", None, || {
            solve_family_candidates(&balances, 1.0, &slot0_results, &predictions, 2)
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
        let report = solve_family_candidates(&balances, 1.0, &slot0_results, &predictions, 2)
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
        assert!(!stats.forecastflows_worker_available);
        assert_eq!(
            stats.forecastflows_fallback_reason,
            Some("worker_error_response")
        );
    }

    #[test]
    fn forecastflows_solver_marks_pre_worker_request_construction_failure() {
        let (mut slot0_results, balances, predictions) = two_market_fixture();
        slot0_results[0].0.tick = 100_000;
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
}
