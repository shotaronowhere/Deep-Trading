use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::ForecastFlowsDoctorReport;
use super::protocol::{
    CompareRequestEnvelope, CompareResult, DEFAULT_REQUEST_TIMEOUT_SECS, HealthResult,
    MarketSpecRequest, OutcomeSpecRequest, PROTOCOL_VERSION, PredictionMarketProblemRequest,
    SolveOptionsRequest, UniV3LiquidityBandRequest, WARMUP_REQUEST_TIMEOUT_SECS, WorkerEnvelope,
};

const STDERR_TAIL_LIMIT: usize = 200;
const STDERR_ERROR_CONTEXT_LIMIT: usize = 20;
const CIRCUIT_BREAKER_THRESHOLD: u32 = 3;
const CIRCUIT_BREAKER_BASE_BACKOFF_SECS: u64 = 60;
const CIRCUIT_BREAKER_MAX_BACKOFF_SECS: u64 = 300;
const LEGACY_WORKER_EXECUTION_MODEL: &str =
    "stateless NDJSON; one request at a time per worker process";
const CACHED_WORKER_EXECUTION_MODEL: &str = "cached NDJSON; one request at a time per worker process; compatible compare requests reuse a workspace";
const SERVE_PROTOCOL_EXECUTION_MODEL: &str = "NDJSON; one request at a time per worker process; serve_protocol reuses compatible compare workspaces; handle_protocol_json is stateless";
const REPRESENTATIVE_REPORT_SOURCE: &str =
    "embedded:test/fixtures/rebalancer_ab_live_l1_snapshot_report.json";
const REPRESENTATIVE_REPORT_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test/fixtures/rebalancer_ab_live_l1_snapshot_report.json"
));

#[derive(Debug)]
pub(super) enum ForecastFlowsClientError {
    Spawn(String),
    Io(String),
    Timeout(Duration),
    Closed(String),
    Protocol(String),
    Worker {
        code: String,
        message: String,
    },
    Cooldown {
        remaining: Duration,
        last_failure: Option<String>,
    },
}

impl fmt::Display for ForecastFlowsClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(message) => write!(f, "failed to start ForecastFlows worker: {message}"),
            Self::Io(message) => write!(f, "ForecastFlows worker I/O failed: {message}"),
            Self::Timeout(timeout) => write!(
                f,
                "ForecastFlows worker timed out after {:.3}s",
                timeout.as_secs_f64()
            ),
            Self::Closed(message) => {
                write!(f, "ForecastFlows worker closed unexpectedly: {message}")
            }
            Self::Protocol(message) => write!(f, "ForecastFlows worker protocol error: {message}"),
            Self::Worker { code, message } => {
                write!(f, "ForecastFlows worker returned {code}: {message}")
            }
            Self::Cooldown {
                remaining,
                last_failure,
            } => {
                write!(
                    f,
                    "ForecastFlows worker is in cooldown for {:.3}s",
                    remaining.as_secs_f64()
                )?;
                if let Some(last_failure) = last_failure {
                    write!(f, " after recent failure: {last_failure}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ForecastFlowsClientError {}

impl ForecastFlowsClientError {
    pub(super) fn fallback_reason(&self) -> &'static str {
        match self {
            Self::Spawn(_) => "worker_spawn_error",
            Self::Io(_) => "worker_io_error",
            Self::Timeout(_) => "worker_timeout",
            Self::Closed(_) => "worker_closed",
            Self::Protocol(_) => "worker_protocol_error",
            Self::Worker { .. } => "worker_error_response",
            Self::Cooldown { .. } => "worker_cooldown",
        }
    }

    fn counts_towards_circuit_breaker(&self) -> bool {
        matches!(
            self,
            Self::Spawn(_) | Self::Io(_) | Self::Timeout(_) | Self::Closed(_) | Self::Protocol(_)
        )
    }

    fn is_request_level_worker_failure(&self) -> bool {
        matches!(self, Self::Worker { .. })
    }

    fn with_context(self, context: Option<String>) -> Self {
        let Some(context) = context.filter(|context| !context.is_empty()) else {
            return self;
        };
        match self {
            Self::Spawn(message) => Self::Spawn(format!("{message}; {context}")),
            Self::Io(message) => Self::Io(format!("{message}; {context}")),
            Self::Closed(message) => Self::Closed(format!("{message}; {context}")),
            Self::Protocol(message) => Self::Protocol(format!("{message}; {context}")),
            Self::Worker { code, message } => Self::Worker {
                code,
                message: format!("{message}; {context}"),
            },
            other => other,
        }
    }
}

#[derive(Debug)]
pub(super) struct ForecastFlowsCompareFailure {
    pub(super) error: ForecastFlowsClientError,
    pub(super) request_count: usize,
    pub(super) runtime_policy: Option<ForecastFlowsRuntimePolicy>,
}

impl fmt::Display for ForecastFlowsCompareFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for ForecastFlowsCompareFailure {}

impl ForecastFlowsCompareFailure {
    pub(super) fn fallback_reason(&self) -> &'static str {
        self.error.fallback_reason()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForecastFlowsRequestProfile {
    Production,
    Benchmark,
    DoctorWarmup,
    DoctorRepresentative,
}

impl ForecastFlowsRequestProfile {
    fn request_timeout(self) -> Duration {
        match self {
            Self::Production => Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
            Self::Benchmark | Self::DoctorWarmup | Self::DoctorRepresentative => {
                Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS)
            }
        }
    }

    fn allows_plain_julia_without_escape_hatch(self) -> bool {
        !matches!(self, Self::Production)
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "production" => Some(Self::Production),
            "benchmark" => Some(Self::Benchmark),
            "doctor_warmup" => Some(Self::DoctorWarmup),
            "doctor_representative" => Some(Self::DoctorRepresentative),
            _ => None,
        }
    }
}

fn worker_execution_model_supported(execution_model: &str) -> bool {
    matches!(
        execution_model,
        LEGACY_WORKER_EXECUTION_MODEL
            | CACHED_WORKER_EXECUTION_MODEL
            | SERVE_PROTOCOL_EXECUTION_MODEL
    )
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForecastFlowsSolveTuning {
    Baseline,
    LowLatency,
}

impl ForecastFlowsSolveTuning {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::LowLatency => "low_latency",
        }
    }

    fn solve_options(self) -> SolveOptionsRequest {
        match self {
            Self::Baseline => SolveOptionsRequest {
                certify: true,
                throw_on_fail: false,
                max_doublings: 6,
                pgtol: 1e-6,
                max_iter: 10_000,
                max_fun: 20_000,
            },
            Self::LowLatency => SolveOptionsRequest {
                certify: true,
                throw_on_fail: false,
                max_doublings: 0,
                pgtol: 1e-6,
                max_iter: 2_500,
                max_fun: 5_000,
            },
        }
    }
}

#[derive(Debug)]
pub(super) struct ForecastFlowsCompareSuccess {
    pub(super) compare: CompareResult,
    pub(super) request_count: usize,
    pub(super) roundtrip: Duration,
    pub(super) runtime_policy: ForecastFlowsRuntimePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForecastFlowsBackend {
    JuliaWorker,
    RustWorker,
}

impl ForecastFlowsBackend {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::JuliaWorker => "julia_worker",
            Self::RustWorker => "rust_worker",
        }
    }
}

impl fmt::Display for ForecastFlowsBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub(super) struct ForecastFlowsRuntimePolicy {
    pub(super) backend: ForecastFlowsBackend,
    pub(super) worker_version: Option<String>,
    pub(super) solve_tuning: String,
    /// Julia-specific: one of "active", "disabled", "rejected". `None` when the
    /// backend does not use a Julia sysimage (e.g., the Rust worker).
    pub(super) sysimage_status: Option<String>,
    /// Julia-specific: value of `JULIA_NUM_THREADS`. `None` under the Rust
    /// worker which does not expose a thread configuration knob.
    pub(super) julia_threads: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ForecastFlowsWorkerConfig {
    julia_threads: OsString,
    allow_plain_julia_escape_hatch: bool,
}

#[derive(Debug, Clone)]
struct WorkerLaunchConfig {
    program: OsString,
    args: Vec<OsString>,
    current_dir: PathBuf,
    envs: Vec<(OsString, OsString)>,
    details: WorkerLaunchDetails,
}

#[derive(Debug, Clone)]
enum WorkerLaunchDetails {
    Julia(JuliaLaunchDetails),
    Rust(RustLaunchDetails),
}

#[derive(Debug, Clone)]
struct JuliaLaunchDetails {
    project_dir: PathBuf,
    julia_version: Option<String>,
    manifest_source: ForecastFlowsManifestSource,
    sysimage_status: SysimageStatus,
    worker_config: ForecastFlowsWorkerConfig,
}

#[derive(Debug, Clone)]
struct RustLaunchDetails {
    worker_bin: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ForecastFlowsManifestSource {
    repo_url: Option<String>,
    repo_rev: Option<String>,
    git_tree_sha1: Option<String>,
}

impl WorkerLaunchConfig {
    fn backend(&self) -> ForecastFlowsBackend {
        match self.details {
            WorkerLaunchDetails::Julia(_) => ForecastFlowsBackend::JuliaWorker,
            WorkerLaunchDetails::Rust(_) => ForecastFlowsBackend::RustWorker,
        }
    }

    fn julia(&self) -> Option<&JuliaLaunchDetails> {
        match &self.details {
            WorkerLaunchDetails::Julia(j) => Some(j),
            WorkerLaunchDetails::Rust(_) => None,
        }
    }

    fn program_display(&self) -> String {
        os_string_display(&self.program)
    }

    fn launch_args(&self) -> Vec<String> {
        self.args.iter().map(|arg| os_string_display(arg)).collect()
    }

    fn launch_envs(&self) -> Vec<(OsString, OsString)> {
        self.envs.clone()
    }

    fn runtime_detail(&self) -> String {
        match &self.details {
            WorkerLaunchDetails::Julia(j) => format!(
                "julia worker (sysimage {}; JULIA_NUM_THREADS={})",
                j.sysimage_status.detail(),
                os_string_display(&j.worker_config.julia_threads),
            ),
            WorkerLaunchDetails::Rust(r) => {
                format!("rust worker binary {}", r.worker_bin.display())
            }
        }
    }

    fn ensure_profile_support(
        &self,
        profile: ForecastFlowsRequestProfile,
    ) -> Result<(), ForecastFlowsClientError> {
        match &self.details {
            WorkerLaunchDetails::Julia(j) => j.ensure_profile_support(profile),
            // The Rust worker binary has no equivalent of the Julia sysimage
            // gate, so every request profile is supported as-is.
            WorkerLaunchDetails::Rust(_) => Ok(()),
        }
    }
}

impl JuliaLaunchDetails {
    fn ensure_profile_support(
        &self,
        profile: ForecastFlowsRequestProfile,
    ) -> Result<(), ForecastFlowsClientError> {
        if matches!(self.sysimage_status, SysimageStatus::Active(_))
            || self.worker_config.allow_plain_julia_escape_hatch
            || profile.allows_plain_julia_without_escape_hatch()
        {
            return Ok(());
        }
        Err(ForecastFlowsClientError::Spawn(format!(
            "ForecastFlows requires an active sysimage for {:?} requests; set FORECASTFLOWS_SYSIMAGE or override with FORECASTFLOWS_ALLOW_PLAIN_JULIA=1 ({})",
            profile,
            self.sysimage_status.detail()
        )))
    }

    fn julia_threads_display(&self) -> String {
        os_string_display(&self.worker_config.julia_threads)
    }
}

#[derive(Debug, Clone)]
enum SysimageStatus {
    Disabled,
    Active(PathBuf),
    Rejected { path: PathBuf, reason: String },
}

impl SysimageStatus {
    fn label(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Active(_) => "active",
            Self::Rejected { .. } => "rejected",
        }
    }

    fn detail(&self) -> String {
        match self {
            Self::Disabled => "no FORECASTFLOWS_SYSIMAGE configured".to_string(),
            Self::Active(path) => format!("using {}", path.display()),
            Self::Rejected { path, reason } => {
                format!("rejected {} ({reason})", path.display())
            }
        }
    }
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout_rx: Receiver<Result<String, String>>,
    stderr_tail: Arc<Mutex<StderrTail>>,
    launch_config: WorkerLaunchConfig,
    /// Populated after the first successful health probe from
    /// `HealthResult::package_version`. `None` if the worker response does not
    /// advertise a version (e.g., older Julia workers) or if the process has
    /// not yet completed its initial health handshake.
    worker_version: Option<String>,
}

#[derive(Default)]
struct StderrTail {
    lines: VecDeque<String>,
}

impl StderrTail {
    fn push(&mut self, line: String) {
        if self.lines.len() == STDERR_TAIL_LIMIT {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    fn snapshot(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }
}

#[derive(Default)]
struct WorkerState {
    process: Option<WorkerProcess>,
    next_request_id: u64,
    warm_complete: bool,
    respawn_scheduled: bool,
    launch_logged: bool,
    consecutive_worker_failures: u32,
    cooldown_until: Option<Instant>,
    last_failure: Option<String>,
    last_stderr_tail: Vec<String>,
}

#[cfg(test)]
impl WorkerState {
    fn harness_state(&self) -> TestWorkerHarnessState {
        TestWorkerHarnessState {
            process_present: self.process.is_some(),
            warm_complete: self.warm_complete,
            in_cooldown: WorkerService::cooldown_remaining_locked(self).is_some(),
            stderr_tail_len: self.last_stderr_tail.len(),
            last_failure: self.last_failure.clone(),
        }
    }
}

struct WorkerService {
    state: Mutex<WorkerState>,
}

impl WorkerService {
    fn new() -> Self {
        Self {
            state: Mutex::new(WorkerState::default()),
        }
    }
}

struct WorkerPool {
    slots: Mutex<Vec<Arc<WorkerService>>>,
    next_slot: AtomicUsize,
}

#[derive(Debug)]
struct HealthCheckOutcome {
    result: HealthResult,
    duration: Duration,
}

impl WorkerPool {
    fn global() -> &'static Self {
        static INSTANCE: OnceLock<WorkerPool> = OnceLock::new();
        INSTANCE.get_or_init(|| WorkerPool {
            slots: Mutex::new(Vec::new()),
            next_slot: AtomicUsize::new(0),
        })
    }

    fn warm_worker(
        &self,
        profile: ForecastFlowsRequestProfile,
    ) -> Result<(), ForecastFlowsClientError> {
        for slot in self.ensure_slots() {
            slot.warm_worker(profile)?;
        }
        Ok(())
    }

    fn compare_prediction_market_families(
        &self,
        problem: PredictionMarketProblemRequest,
        profile: ForecastFlowsRequestProfile,
    ) -> Result<ForecastFlowsCompareSuccess, ForecastFlowsCompareFailure> {
        let slots = self.ensure_slots();
        let next = self.next_slot.fetch_add(1, Ordering::Relaxed);
        let slot = Arc::clone(&slots[next % slots.len()]);
        slot.compare_prediction_market_families(problem, profile)
    }

    fn doctor_report(&self) -> Result<ForecastFlowsDoctorReport, ForecastFlowsClientError> {
        let slots = self.ensure_slots();
        Arc::clone(&slots[0]).doctor_report()
    }

    fn shutdown_worker(&self) -> Result<(), ForecastFlowsClientError> {
        for slot in self.current_slots() {
            slot.shutdown_worker()?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn test_harness_states(&self) -> Vec<TestWorkerHarnessState> {
        self.current_slots()
            .into_iter()
            .map(|slot| {
                let state = slot.state.lock().expect("worker service mutex");
                state.harness_state()
            })
            .collect()
    }

    fn ensure_slots(&self) -> Vec<Arc<WorkerService>> {
        let configured_size = configured_worker_pool_size();
        let mut slots_guard = self.slots.lock().expect("worker pool mutex");
        if slots_guard.len() == configured_size {
            return slots_guard.clone();
        }

        let old_slots = std::mem::take(&mut *slots_guard);
        let new_slots = (0..configured_size)
            .map(|_| Arc::new(WorkerService::new()))
            .collect::<Vec<_>>();
        *slots_guard = new_slots.clone();
        self.next_slot.store(0, Ordering::Relaxed);
        drop(slots_guard);

        for slot in old_slots {
            let _ = slot.shutdown_worker();
        }

        new_slots
    }

    fn current_slots(&self) -> Vec<Arc<WorkerService>> {
        self.slots.lock().expect("worker pool mutex").clone()
    }

    #[cfg(test)]
    fn first_slot(&self) -> Arc<WorkerService> {
        self.ensure_slots()
            .into_iter()
            .next()
            .expect("worker pool must expose at least one slot")
    }
}

impl WorkerService {
    fn warm_worker(
        &self,
        profile: ForecastFlowsRequestProfile,
    ) -> Result<(), ForecastFlowsClientError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ForecastFlowsClientError::Closed("worker mutex poisoned".to_string()))?;
        if let Some(err) = Self::cooldown_error_locked(&state) {
            return Err(err);
        }

        if let Err(err) = Self::ensure_process_locked(&mut state, profile) {
            let err = Self::record_and_reset_failure_locked(&mut state, err);
            return Err(err);
        }
        if state.warm_complete {
            return Ok(());
        }

        let request_id = Self::next_request_id(&mut state, "warmup");
        let warm_problem = tiny_compare_problem();
        let request = CompareRequestEnvelope {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id.clone(),
            command: "compare_prediction_market_families",
            problem: warm_problem,
            solve_options: solve_options_for_profile(profile),
        };
        let warm_result: Result<CompareResult, ForecastFlowsClientError> =
            Self::send_request_locked(
                &mut state,
                &request_id,
                "compare_prediction_market_families",
                &request,
                request_timeout(profile),
            );
        match warm_result {
            Ok(_) => {
                state.warm_complete = true;
                Self::note_success_locked(&mut state);
                Self::log_launch_details_locked(&mut state);
                Ok(())
            }
            Err(err) => {
                if err.is_request_level_worker_failure() {
                    return Err(err.with_context(Self::stderr_context_locked(&state)));
                }
                let err = Self::record_and_reset_failure_locked(&mut state, err);
                Err(err)
            }
        }
    }

    fn compare_prediction_market_families(
        self: &Arc<Self>,
        problem: PredictionMarketProblemRequest,
        profile: ForecastFlowsRequestProfile,
    ) -> Result<ForecastFlowsCompareSuccess, ForecastFlowsCompareFailure> {
        let mut request_count = 0usize;

        for attempt in 0..2 {
            let mut state = self.state.lock().map_err(|_| ForecastFlowsCompareFailure {
                error: ForecastFlowsClientError::Closed("worker mutex poisoned".to_string()),
                request_count,
                runtime_policy: None,
            })?;
            if let Some(err) = Self::cooldown_error_locked(&state) {
                return Err(ForecastFlowsCompareFailure {
                    error: err,
                    request_count,
                    runtime_policy: runtime_policy_from_state_locked(&state, profile),
                });
            }
            if let Err(err) = Self::ensure_process_locked(&mut state, profile) {
                if attempt == 1 {
                    let err = Self::record_and_reset_failure_locked(&mut state, err);
                    return Err(ForecastFlowsCompareFailure {
                        error: err,
                        request_count,
                        runtime_policy: runtime_policy_from_state_locked(&state, profile),
                    });
                }
                Self::reset_process_locked(&mut state);
                continue;
            }
            let request_id = Self::next_request_id(&mut state, "compare");
            let request = CompareRequestEnvelope {
                protocol_version: PROTOCOL_VERSION,
                request_id: request_id.clone(),
                command: "compare_prediction_market_families",
                problem: problem.clone(),
                solve_options: solve_options_for_profile(profile),
            };
            request_count += 1;
            let started = Instant::now();
            match Self::send_request_locked(
                &mut state,
                &request_id,
                "compare_prediction_market_families",
                &request,
                request_timeout(profile),
            ) {
                Ok(result) => {
                    Self::note_success_locked(&mut state);
                    Self::log_launch_details_locked(&mut state);
                    let runtime_policy = runtime_policy_from_state_locked(&state, profile)
                        .expect("spawned worker should have launch configuration");
                    return Ok(ForecastFlowsCompareSuccess {
                        compare: result,
                        request_count,
                        roundtrip: started.elapsed(),
                        runtime_policy,
                    });
                }
                Err(err) => {
                    if err.is_request_level_worker_failure() {
                        return Err(ForecastFlowsCompareFailure {
                            error: err.with_context(Self::stderr_context_locked(&state)),
                            request_count,
                            runtime_policy: runtime_policy_from_state_locked(&state, profile),
                        });
                    }
                    if attempt == 1 {
                        let err = Self::record_and_reset_failure_locked(&mut state, err);
                        self.schedule_background_respawn_locked(&mut state, profile);
                        return Err(ForecastFlowsCompareFailure {
                            error: err,
                            request_count,
                            runtime_policy: runtime_policy_from_state_locked(&state, profile),
                        });
                    }
                    Self::reset_process_locked(&mut state);
                }
            }
        }

        Err(ForecastFlowsCompareFailure {
            error: ForecastFlowsClientError::Closed(
                "worker retries exhausted without a final error".to_string(),
            ),
            request_count,
            runtime_policy: None,
        })
    }

    fn doctor_report(&self) -> Result<ForecastFlowsDoctorReport, ForecastFlowsClientError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ForecastFlowsClientError::Closed("worker mutex poisoned".to_string()))?;
        Self::reset_process_locked(&mut state);
        state.launch_logged = false;

        let launch = worker_launch_config(ForecastFlowsRequestProfile::DoctorWarmup)?;
        let julia = launch.julia().cloned();
        let mut report = ForecastFlowsDoctorReport {
            worker_backend: launch.backend().as_str().to_string(),
            worker_program: launch.program_display(),
            worker_version: None,
            worker_project_dir: julia.as_ref().map(|j| j.project_dir.display().to_string()),
            worker_launch_args: launch.launch_args(),
            worker_runtime_detail: launch.runtime_detail(),
            julia_version: julia.as_ref().and_then(|j| j.julia_version.clone()),
            manifest_repo_url: julia
                .as_ref()
                .and_then(|j| j.manifest_source.repo_url.clone()),
            manifest_repo_rev: julia
                .as_ref()
                .and_then(|j| j.manifest_source.repo_rev.clone()),
            manifest_git_tree_sha1: julia
                .as_ref()
                .and_then(|j| j.manifest_source.git_tree_sha1.clone()),
            julia_threads: julia
                .as_ref()
                .map(JuliaLaunchDetails::julia_threads_display),
            allow_plain_julia_escape_hatch: julia
                .as_ref()
                .map(|j| j.worker_config.allow_plain_julia_escape_hatch),
            sysimage_status: julia
                .as_ref()
                .map(|j| j.sysimage_status.label().to_string()),
            sysimage_detail: julia.as_ref().map(|j| j.sysimage_status.detail()),
            live_solve_tuning: solve_tuning_for_profile(ForecastFlowsRequestProfile::Production)
                .as_str()
                .to_string(),
            health_status: None,
            supported_commands: Vec::new(),
            supported_modes: Vec::new(),
            execution_model: None,
            health_duration_ms: None,
            warmup_compare_duration_ms: None,
            warmup_direct_status: None,
            warmup_mixed_status: None,
            warmup_direct_solver_time_ms: None,
            warmup_mixed_solver_time_ms: None,
            warmup_driver_overhead_ms: None,
            representative_compare_duration_ms: None,
            representative_direct_status: None,
            representative_mixed_status: None,
            representative_direct_solver_time_ms: None,
            representative_mixed_solver_time_ms: None,
            representative_driver_overhead_ms: None,
            representative_fixture: REPRESENTATIVE_REPORT_SOURCE.to_string(),
            stderr_tail: Self::stderr_tail_snapshot_locked(&state),
            failure: None,
        };

        let health = match Self::ensure_process_locked(
            &mut state,
            ForecastFlowsRequestProfile::DoctorWarmup,
        ) {
            Ok(Some(health)) => health,
            Ok(None) => {
                return Err(ForecastFlowsClientError::Closed(
                    "doctor expected fresh worker spawn".to_string(),
                ));
            }
            Err(err) => {
                return Ok(Self::doctor_failure_report_locked(&mut state, report, err));
            }
        };
        report.health_status = Some(health.result.status.clone());
        report
            .worker_version
            .clone_from(&health.result.package_version);
        report.supported_commands = health.result.supported_commands.clone();
        report.supported_modes = health.result.supported_modes.clone();
        report.execution_model = Some(health.result.execution_model.clone());
        report.health_duration_ms = Some(health.duration.as_millis());

        let warmup_request_id = Self::next_request_id(&mut state, "doctor-warmup");
        let warmup_request = CompareRequestEnvelope {
            protocol_version: PROTOCOL_VERSION,
            request_id: warmup_request_id.clone(),
            command: "compare_prediction_market_families",
            problem: tiny_compare_problem(),
            solve_options: solve_options_for_profile(ForecastFlowsRequestProfile::DoctorWarmup),
        };
        let warmup_started = Instant::now();
        let warmup_result: Result<CompareResult, ForecastFlowsClientError> =
            Self::send_request_locked(
                &mut state,
                &warmup_request_id,
                "compare_prediction_market_families",
                &warmup_request,
                request_timeout(ForecastFlowsRequestProfile::DoctorWarmup),
            );
        let warmup_result = match warmup_result {
            Ok(result) => result,
            Err(err) => {
                if err.is_request_level_worker_failure() {
                    report.stderr_tail = Self::stderr_tail_snapshot_locked(&state);
                    report.failure = Some(
                        err.with_context(Self::stderr_context_locked(&state))
                            .to_string(),
                    );
                    return Ok(report);
                }
                return Ok(Self::doctor_failure_report_locked(&mut state, report, err));
            }
        };
        report.warmup_compare_duration_ms = Some(warmup_started.elapsed().as_millis());
        set_doctor_compare_timing(
            &mut report.warmup_direct_status,
            &mut report.warmup_mixed_status,
            &mut report.warmup_direct_solver_time_ms,
            &mut report.warmup_mixed_solver_time_ms,
            &mut report.warmup_driver_overhead_ms,
            report.warmup_compare_duration_ms,
            &warmup_result,
        );
        state.warm_complete = true;

        let representative_problem = match representative_compare_problem() {
            Ok(problem) => problem,
            Err(err) => {
                report.stderr_tail = Self::stderr_tail_snapshot_locked(&state);
                report.failure = Some(err.to_string());
                return Ok(report);
            }
        };
        let representative_request_id = Self::next_request_id(&mut state, "doctor-representative");
        let representative_request = CompareRequestEnvelope {
            protocol_version: PROTOCOL_VERSION,
            request_id: representative_request_id.clone(),
            command: "compare_prediction_market_families",
            problem: representative_problem,
            solve_options: solve_options_for_profile(
                ForecastFlowsRequestProfile::DoctorRepresentative,
            ),
        };
        let representative_started = Instant::now();
        let representative_result: Result<CompareResult, ForecastFlowsClientError> =
            Self::send_request_locked(
                &mut state,
                &representative_request_id,
                "compare_prediction_market_families",
                &representative_request,
                request_timeout(ForecastFlowsRequestProfile::DoctorRepresentative),
            );
        let representative_result = match representative_result {
            Ok(result) => result,
            Err(err) => {
                if err.is_request_level_worker_failure() {
                    report.stderr_tail = Self::stderr_tail_snapshot_locked(&state);
                    report.failure = Some(
                        err.with_context(Self::stderr_context_locked(&state))
                            .to_string(),
                    );
                    return Ok(report);
                }
                return Ok(Self::doctor_failure_report_locked(&mut state, report, err));
            }
        };
        report.representative_compare_duration_ms =
            Some(representative_started.elapsed().as_millis());
        set_doctor_compare_timing(
            &mut report.representative_direct_status,
            &mut report.representative_mixed_status,
            &mut report.representative_direct_solver_time_ms,
            &mut report.representative_mixed_solver_time_ms,
            &mut report.representative_driver_overhead_ms,
            report.representative_compare_duration_ms,
            &representative_result,
        );

        Self::note_success_locked(&mut state);
        Self::log_launch_details_locked(&mut state);

        report.stderr_tail = Self::stderr_tail_snapshot_locked(&state);
        Ok(report)
    }

    fn shutdown_worker(&self) -> Result<(), ForecastFlowsClientError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ForecastFlowsClientError::Closed("worker mutex poisoned".to_string()))?;
        Self::reset_process_locked(&mut state);
        state.warm_complete = false;
        Ok(())
    }

    fn ensure_process_locked(
        state: &mut WorkerState,
        profile: ForecastFlowsRequestProfile,
    ) -> Result<Option<HealthCheckOutcome>, ForecastFlowsClientError> {
        if let Some(process) = state.process.as_ref() {
            if process
                .launch_config
                .ensure_profile_support(profile)
                .is_ok()
            {
                return Ok(None);
            }
            Self::reset_process_locked(state);
        }

        let process = Self::spawn_worker_process(profile)?;
        state.process = Some(process);
        state.warm_complete = false;
        let request_id = Self::next_request_id(state, "health");
        let request = json!({
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "command": "health",
        });
        let started = Instant::now();
        let result: HealthResult = match Self::send_request_locked(
            state,
            &request["request_id"].as_str().unwrap_or("health"),
            "health",
            &request,
            Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS),
        ) {
            Ok(result) => result,
            Err(err) => return Err(err),
        };
        if result.status != "ok" {
            return Err(ForecastFlowsClientError::Protocol(format!(
                "unexpected health status {}",
                result.status
            )));
        }
        if !result
            .supported_commands
            .iter()
            .any(|command| command == "compare_prediction_market_families")
        {
            return Err(ForecastFlowsClientError::Protocol(
                "worker health response does not advertise compare_prediction_market_families"
                    .to_string(),
            ));
        }
        if !result
            .supported_modes
            .iter()
            .any(|mode| mode == "mixed_enabled")
        {
            return Err(ForecastFlowsClientError::Protocol(
                "worker health response does not advertise mixed_enabled".to_string(),
            ));
        }
        if !worker_execution_model_supported(&result.execution_model) {
            return Err(ForecastFlowsClientError::Protocol(format!(
                "unexpected worker execution model {}",
                result.execution_model
            )));
        }

        if let Some(process) = state.process.as_mut() {
            process.worker_version.clone_from(&result.package_version);
        }

        Ok(Some(HealthCheckOutcome {
            result,
            duration: started.elapsed(),
        }))
    }

    fn send_request_locked<T: serde::de::DeserializeOwned, S: serde::Serialize>(
        state: &mut WorkerState,
        request_id: &str,
        expected_command: &str,
        request: &S,
        timeout: Duration,
    ) -> Result<T, ForecastFlowsClientError> {
        let process = state.process.as_mut().ok_or_else(|| {
            ForecastFlowsClientError::Closed("worker process unavailable".to_string())
        })?;
        let request_json = serde_json::to_string(request)
            .map_err(|err| ForecastFlowsClientError::Protocol(err.to_string()))?;
        process
            .stdin
            .write_all(request_json.as_bytes())
            .and_then(|_| process.stdin.write_all(b"\n"))
            .and_then(|_| process.stdin.flush())
            .map_err(|err| ForecastFlowsClientError::Io(err.to_string()))?;

        let response_line = match process.stdout_rx.recv_timeout(timeout) {
            Ok(Ok(line)) => line,
            Ok(Err(message)) => return Err(ForecastFlowsClientError::Closed(message)),
            Err(RecvTimeoutError::Timeout) => {
                return Err(ForecastFlowsClientError::Timeout(timeout));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(ForecastFlowsClientError::Closed(
                    "worker stdout reader disconnected".to_string(),
                ));
            }
        };

        let envelope: WorkerEnvelope<T> = serde_json::from_str(&response_line)
            .map_err(|err| ForecastFlowsClientError::Protocol(err.to_string()))?;
        if envelope.protocol_version != PROTOCOL_VERSION {
            return Err(ForecastFlowsClientError::Protocol(format!(
                "unexpected protocol version {}",
                envelope.protocol_version
            )));
        }
        if envelope.request_id.as_deref() != Some(request_id) {
            return Err(ForecastFlowsClientError::Protocol(format!(
                "request_id mismatch: expected {request_id}, got {:?}",
                envelope.request_id
            )));
        }
        if !envelope.ok {
            let error = envelope.error.ok_or_else(|| {
                ForecastFlowsClientError::Protocol(
                    "worker returned ok=false without error".to_string(),
                )
            })?;
            return Err(ForecastFlowsClientError::Worker {
                code: error.code,
                message: error.message,
            });
        }
        if envelope.command.as_deref() != Some(expected_command) {
            return Err(ForecastFlowsClientError::Protocol(format!(
                "unexpected command in response: {:?}",
                envelope.command
            )));
        }
        envelope.result.ok_or_else(|| {
            ForecastFlowsClientError::Protocol("worker returned ok=true without result".to_string())
        })
    }

    fn next_request_id(state: &mut WorkerState, prefix: &str) -> String {
        let request_id = format!("{prefix}-{}", state.next_request_id);
        state.next_request_id += 1;
        request_id
    }

    fn spawn_worker_process(
        profile: ForecastFlowsRequestProfile,
    ) -> Result<WorkerProcess, ForecastFlowsClientError> {
        let launch = worker_launch_config(profile)?;
        launch.ensure_profile_support(profile)?;
        if let WorkerLaunchDetails::Julia(julia) = &launch.details
            && let SysimageStatus::Rejected { reason, .. } = &julia.sysimage_status
        {
            tracing::warn!(reason = %reason, sysimage = %julia.sysimage_status.detail(), "ForecastFlows sysimage disabled; falling back to plain Julia");
        }
        let mut command = Command::new(&launch.program);
        command
            .args(&launch.args)
            .current_dir(&launch.current_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in launch.launch_envs() {
            command.env(key, value);
        }
        let mut child = command
            .spawn()
            .map_err(|err| ForecastFlowsClientError::Spawn(err.to_string()))?;
        let stdin = child.stdin.take().ok_or_else(|| {
            ForecastFlowsClientError::Spawn("worker stdin pipe unavailable".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ForecastFlowsClientError::Spawn("worker stdout pipe unavailable".to_string())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            ForecastFlowsClientError::Spawn("worker stderr pipe unavailable".to_string())
        })?;

        let (stdout_tx, stdout_rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if stdout_tx.send(Ok(line)).is_err() {
                            return;
                        }
                    }
                    Err(err) => {
                        let _ = stdout_tx.send(Err(err.to_string()));
                        return;
                    }
                }
            }
            let _ = stdout_tx.send(Err("worker stdout closed".to_string()));
        });

        let stderr_tail = Arc::new(Mutex::new(StderrTail::default()));
        let stderr_tail_clone = Arc::clone(&stderr_tail);
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if let Ok(mut tail) = stderr_tail_clone.lock() {
                            tail.push(line.clone());
                        }
                        tracing::debug!(target: "forecastflows_worker", "{line}");
                    }
                    Err(err) => {
                        tracing::debug!(
                            target: "forecastflows_worker",
                            error = %err,
                            "failed to read ForecastFlows worker stderr"
                        );
                        return;
                    }
                }
            }
        });

        Ok(WorkerProcess {
            child,
            stdin,
            stdout_rx,
            stderr_tail,
            launch_config: launch,
            worker_version: None,
        })
    }

    fn reset_process_locked(state: &mut WorkerState) {
        if let Some(process) = state.process.take() {
            state.last_stderr_tail = Self::stderr_tail_snapshot(&process.stderr_tail);
            let WorkerProcess {
                mut child,
                stdin,
                stdout_rx,
                stderr_tail,
                launch_config: _,
                worker_version: _,
            } = process;
            drop(stdin);
            drop(stdout_rx);
            drop(stderr_tail);
            let _ = child.kill();
            match child.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) => Self::spawn_child_reaper(child),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "failed to poll ForecastFlows worker exit status; reaping in background"
                    );
                    Self::spawn_child_reaper(child);
                }
            }
        }
        state.warm_complete = false;
    }

    fn schedule_background_respawn_locked(
        self: &Arc<Self>,
        state: &mut WorkerState,
        profile: ForecastFlowsRequestProfile,
    ) {
        if state.respawn_scheduled || Self::cooldown_remaining_locked(state).is_some() {
            return;
        }
        state.respawn_scheduled = true;
        let service = Arc::clone(self);
        thread::spawn(move || {
            if let Err(err) = service.warm_worker(profile) {
                tracing::warn!(error = %err, "ForecastFlows background worker respawn failed");
            }
            if let Ok(mut state) = service.state.lock() {
                state.respawn_scheduled = false;
            }
        });
    }

    fn cooldown_remaining_locked(state: &WorkerState) -> Option<Duration> {
        let cooldown_until = state.cooldown_until?;
        let now = Instant::now();
        (cooldown_until > now).then_some(cooldown_until.duration_since(now))
    }

    fn cooldown_error_locked(state: &WorkerState) -> Option<ForecastFlowsClientError> {
        Self::cooldown_remaining_locked(state).map(|remaining| ForecastFlowsClientError::Cooldown {
            remaining,
            last_failure: state.last_failure.clone(),
        })
    }

    fn note_success_locked(state: &mut WorkerState) {
        state.consecutive_worker_failures = 0;
        state.cooldown_until = None;
        state.last_failure = None;
    }

    fn record_and_reset_failure_locked(
        state: &mut WorkerState,
        err: ForecastFlowsClientError,
    ) -> ForecastFlowsClientError {
        let err = err.with_context(Self::stderr_context_locked(state));
        if err.counts_towards_circuit_breaker() {
            state.consecutive_worker_failures += 1;
            state.last_failure = Some(err.to_string());
            if state.consecutive_worker_failures >= circuit_breaker_threshold() {
                let over_threshold = state
                    .consecutive_worker_failures
                    .saturating_sub(circuit_breaker_threshold());
                let shift = over_threshold.min(4);
                let factor = 1u64 << shift;
                let backoff_secs = circuit_breaker_base_backoff()
                    .as_secs()
                    .saturating_mul(factor)
                    .min(circuit_breaker_max_backoff().as_secs());
                state.cooldown_until = Some(Instant::now() + Duration::from_secs(backoff_secs));
            }
        }
        Self::reset_process_locked(state);
        err
    }

    fn log_launch_details_locked(state: &mut WorkerState) {
        let Some(process) = state.process.as_ref() else {
            return;
        };
        if state.launch_logged {
            return;
        }
        match &process.launch_config.details {
            WorkerLaunchDetails::Julia(julia) => tracing::info!(
                backend = %ForecastFlowsBackend::JuliaWorker,
                worker_program = %process.launch_config.program_display(),
                worker_version = process.worker_version.as_deref().unwrap_or("unknown"),
                julia_version = julia.julia_version.as_deref().unwrap_or("unknown"),
                project_dir = %julia.project_dir.display(),
                julia_threads = %julia.julia_threads_display(),
                allow_plain_julia_escape_hatch = julia.worker_config.allow_plain_julia_escape_hatch,
                sysimage_status = julia.sysimage_status.label(),
                sysimage_detail = %julia.sysimage_status.detail(),
                "ForecastFlows worker launch configuration"
            ),
            WorkerLaunchDetails::Rust(rust) => tracing::info!(
                backend = %ForecastFlowsBackend::RustWorker,
                worker_program = %process.launch_config.program_display(),
                worker_version = process.worker_version.as_deref().unwrap_or("unknown"),
                worker_bin = %rust.worker_bin.display(),
                "ForecastFlows worker launch configuration"
            ),
        }
        state.launch_logged = true;
    }

    fn stderr_context_locked(state: &WorkerState) -> Option<String> {
        let stderr_tail = Self::stderr_tail_snapshot_locked(state);
        if stderr_tail.is_empty() {
            return None;
        }
        let start = stderr_tail.len().saturating_sub(STDERR_ERROR_CONTEXT_LIMIT);
        Some(format!(
            "worker stderr tail: {}",
            stderr_tail[start..].join(" | ")
        ))
    }

    fn stderr_tail_snapshot_locked(state: &WorkerState) -> Vec<String> {
        if let Some(process) = state.process.as_ref() {
            let snapshot = Self::stderr_tail_snapshot(&process.stderr_tail);
            if !snapshot.is_empty() {
                return snapshot;
            }
        }
        state.last_stderr_tail.clone()
    }

    fn stderr_tail_snapshot(stderr_tail: &Arc<Mutex<StderrTail>>) -> Vec<String> {
        stderr_tail
            .lock()
            .map(|tail| tail.snapshot())
            .unwrap_or_default()
    }

    fn spawn_child_reaper(mut child: Child) {
        thread::spawn(move || {
            if let Err(err) = child.wait() {
                tracing::warn!(error = %err, "failed to reap ForecastFlows worker process");
            }
        });
    }

    fn doctor_failure_report_locked(
        state: &mut WorkerState,
        mut report: ForecastFlowsDoctorReport,
        err: ForecastFlowsClientError,
    ) -> ForecastFlowsDoctorReport {
        let err = Self::record_and_reset_failure_locked(state, err);
        report.stderr_tail = Self::stderr_tail_snapshot_locked(state);
        report.failure = Some(err.to_string());
        report
    }
}

fn set_doctor_compare_timing(
    direct_status: &mut Option<String>,
    mixed_status: &mut Option<String>,
    direct_solver_time_ms: &mut Option<u128>,
    mixed_solver_time_ms: &mut Option<u128>,
    driver_overhead_ms: &mut Option<u128>,
    total_roundtrip_ms: Option<u128>,
    compare: &CompareResult,
) {
    *direct_status = Some(compare.direct_only.status.clone());
    *mixed_status = Some(compare.mixed_enabled.status.clone());
    *direct_solver_time_ms = solve_result_time_ms(compare.direct_only.solver_time_sec);
    *mixed_solver_time_ms = solve_result_time_ms(compare.mixed_enabled.solver_time_sec);
    *driver_overhead_ms = total_roundtrip_ms.and_then(|roundtrip_ms| {
        let solver_ms =
            (*direct_solver_time_ms).unwrap_or(0) + (*mixed_solver_time_ms).unwrap_or(0);
        roundtrip_ms.checked_sub(solver_ms)
    });
}

fn solve_result_time_ms(solver_time_sec: Option<f64>) -> Option<u128> {
    let solver_time_sec = solver_time_sec?;
    if !solver_time_sec.is_finite() || solver_time_sec < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(solver_time_sec).as_millis())
}

fn runtime_policy_from_state_locked(
    state: &WorkerState,
    profile: ForecastFlowsRequestProfile,
) -> Option<ForecastFlowsRuntimePolicy> {
    let process = state.process.as_ref()?;
    let launch = process.launch_config.clone();
    let worker_version = process.worker_version.clone();
    Some(runtime_policy_from_launch(&launch, worker_version, profile))
}

fn runtime_policy_from_launch(
    launch: &WorkerLaunchConfig,
    worker_version: Option<String>,
    profile: ForecastFlowsRequestProfile,
) -> ForecastFlowsRuntimePolicy {
    let solve_tuning = solve_tuning_for_profile(profile).as_str().to_string();
    match &launch.details {
        WorkerLaunchDetails::Julia(julia) => ForecastFlowsRuntimePolicy {
            backend: ForecastFlowsBackend::JuliaWorker,
            worker_version,
            solve_tuning,
            sysimage_status: Some(julia.sysimage_status.label().to_string()),
            julia_threads: Some(julia.julia_threads_display()),
        },
        WorkerLaunchDetails::Rust(_) => ForecastFlowsRuntimePolicy {
            backend: ForecastFlowsBackend::RustWorker,
            worker_version,
            solve_tuning,
            sysimage_status: None,
            julia_threads: None,
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SysimageMetadata {
    julia_major_minor: String,
    manifest_sha256: String,
    built_at_unix_secs: u64,
}

#[derive(Debug, Deserialize)]
struct RepresentativeSnapshotReport {
    predictions_wad: Vec<u128>,
    starting_prices_wad: Vec<u128>,
    liquidity: Vec<u128>,
    initial_holdings_wad: Vec<u128>,
    initial_cash_budget_wad: u128,
}

fn worker_launch_config(
    profile: ForecastFlowsRequestProfile,
) -> Result<WorkerLaunchConfig, ForecastFlowsClientError> {
    #[cfg(test)]
    if let Some(config) = test_worker_launch_config() {
        return Ok(config);
    }
    build_worker_launch_config_from_env(
        |key| std::env::var_os(key),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        std::env::current_dir().ok(),
        profile,
    )
}

fn build_worker_launch_config_from_env(
    read_env: impl Fn(&str) -> Option<OsString>,
    repo_root: PathBuf,
    current_dir: Option<PathBuf>,
    profile: ForecastFlowsRequestProfile,
) -> Result<WorkerLaunchConfig, ForecastFlowsClientError> {
    let backend = parse_backend_from_env(&read_env)?;
    let launch = match backend {
        ForecastFlowsBackend::JuliaWorker => {
            build_julia_worker_launch_config(&read_env, &repo_root, current_dir.as_deref())?
        }
        ForecastFlowsBackend::RustWorker => {
            build_rust_worker_launch_config(&read_env, current_dir.as_deref())?
        }
    };
    launch.ensure_profile_support(profile)?;
    Ok(launch)
}

fn parse_backend_from_env(
    read_env: &impl Fn(&str) -> Option<OsString>,
) -> Result<ForecastFlowsBackend, ForecastFlowsClientError> {
    let Some(value) = read_env("FORECASTFLOWS_BACKEND") else {
        return Ok(ForecastFlowsBackend::RustWorker);
    };
    match value.to_string_lossy().as_ref() {
        "" | "rust_worker" => Ok(ForecastFlowsBackend::RustWorker),
        "julia_worker" => Ok(ForecastFlowsBackend::JuliaWorker),
        other => Err(ForecastFlowsClientError::Spawn(format!(
            "FORECASTFLOWS_BACKEND must be \"julia_worker\" or \"rust_worker\" (got {other:?})"
        ))),
    }
}

fn build_julia_worker_launch_config(
    read_env: &impl Fn(&str) -> Option<OsString>,
    repo_root: &Path,
    current_dir: Option<&Path>,
) -> Result<WorkerLaunchConfig, ForecastFlowsClientError> {
    let project_dir = resolve_project_dir(read_env, repo_root, current_dir)?;
    let program = read_env("FORECASTFLOWS_JULIA_BIN").unwrap_or_else(|| OsString::from("julia"));
    let julia_version = probe_julia_version(program.as_os_str());
    let worker_config = ForecastFlowsWorkerConfig {
        julia_threads: read_env("FORECASTFLOWS_JULIA_THREADS")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| OsString::from("auto")),
        allow_plain_julia_escape_hatch: env_flag_is_true(
            read_env,
            "FORECASTFLOWS_ALLOW_PLAIN_JULIA",
        ),
    };
    let manifest_hash = manifest_sha256(&project_dir).ok_or_else(|| {
        ForecastFlowsClientError::Spawn(format!(
            "failed to hash ForecastFlows manifest {}",
            project_dir.join("Manifest.toml").display()
        ))
    })?;
    let manifest_source_value = manifest_source(&project_dir);
    let sysimage_status = read_env("FORECASTFLOWS_SYSIMAGE")
        .map(PathBuf::from)
        .map(|path| resolve_sysimage_status(&path, julia_version.as_deref(), Some(&manifest_hash)))
        .unwrap_or(SysimageStatus::Disabled);

    let mut args = vec![
        OsString::from("--startup-file=no"),
        project_flag_arg(&project_dir),
    ];
    if let SysimageStatus::Active(path) = &sysimage_status {
        args.push(OsString::from("-J"));
        args.push(path.as_os_str().to_os_string());
    }
    args.push(OsString::from("-e"));
    args.push(OsString::from(
        "using ForecastFlows; ForecastFlows.serve_protocol(stdin, stdout)",
    ));

    let current_dir = project_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| project_dir.clone());
    let envs = vec![(
        OsString::from("JULIA_NUM_THREADS"),
        worker_config.julia_threads.clone(),
    )];
    let julia_details = JuliaLaunchDetails {
        project_dir,
        julia_version,
        manifest_source: manifest_source_value,
        sysimage_status,
        worker_config,
    };

    Ok(WorkerLaunchConfig {
        program,
        args,
        current_dir,
        envs,
        details: WorkerLaunchDetails::Julia(julia_details),
    })
}

fn build_rust_worker_launch_config(
    read_env: &impl Fn(&str) -> Option<OsString>,
    current_dir: Option<&Path>,
) -> Result<WorkerLaunchConfig, ForecastFlowsClientError> {
    let program = read_env("FORECASTFLOWS_WORKER_BIN").ok_or_else(|| {
        ForecastFlowsClientError::Spawn(
            "rust_worker backend requires FORECASTFLOWS_WORKER_BIN \
             (absolute path to the forecast-flows-worker binary)"
                .to_string(),
        )
    })?;
    let worker_bin = PathBuf::from(&program);
    if !worker_bin.is_absolute() {
        return Err(ForecastFlowsClientError::Spawn(format!(
            "FORECASTFLOWS_WORKER_BIN {} must be an absolute path",
            worker_bin.display()
        )));
    }
    if !worker_bin.exists() {
        return Err(ForecastFlowsClientError::Spawn(format!(
            "FORECASTFLOWS_WORKER_BIN {} does not exist",
            worker_bin.display()
        )));
    }
    if !worker_bin.is_file() {
        return Err(ForecastFlowsClientError::Spawn(format!(
            "FORECASTFLOWS_WORKER_BIN {} is not a regular file",
            worker_bin.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = std::fs::metadata(&worker_bin).map_err(|err| {
            ForecastFlowsClientError::Spawn(format!(
                "failed to stat FORECASTFLOWS_WORKER_BIN {}: {}",
                worker_bin.display(),
                err
            ))
        })?;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(ForecastFlowsClientError::Spawn(format!(
                "FORECASTFLOWS_WORKER_BIN {} is not executable",
                worker_bin.display()
            )));
        }
    }
    let launch_dir = worker_bin
        .parent()
        .map(Path::to_path_buf)
        .or_else(|| current_dir.map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    Ok(WorkerLaunchConfig {
        program,
        args: Vec::new(),
        current_dir: launch_dir,
        envs: Vec::new(),
        details: WorkerLaunchDetails::Rust(RustLaunchDetails { worker_bin }),
    })
}

fn resolve_project_dir(
    read_env: &impl Fn(&str) -> Option<OsString>,
    repo_root: &Path,
    current_dir: Option<&Path>,
) -> Result<PathBuf, ForecastFlowsClientError> {
    let project_dir = match read_env("FORECASTFLOWS_PROJECT_DIR") {
        Some(path) => {
            let path = PathBuf::from(path);
            if path.is_relative() {
                let current_dir = current_dir.ok_or_else(|| {
                    ForecastFlowsClientError::Spawn(
                        "FORECASTFLOWS_PROJECT_DIR is relative but current_dir is unavailable"
                            .to_string(),
                    )
                })?;
                current_dir.join(path)
            } else {
                path
            }
        }
        None => repo_root.join("julia").join("forecastflows"),
    };
    if !project_dir.exists() {
        return Err(ForecastFlowsClientError::Spawn(format!(
            "ForecastFlows Julia project directory {} does not exist",
            project_dir.display()
        )));
    }
    if !project_dir.is_dir() {
        return Err(ForecastFlowsClientError::Spawn(format!(
            "ForecastFlows Julia project directory {} is not a directory",
            project_dir.display()
        )));
    }
    if !project_dir.join("Manifest.toml").is_file() {
        return Err(ForecastFlowsClientError::Spawn(format!(
            "ForecastFlows Julia project directory {} is missing Manifest.toml",
            project_dir.display()
        )));
    }
    Ok(project_dir)
}

fn env_flag_is_true(read_env: &impl Fn(&str) -> Option<OsString>, key: &str) -> bool {
    read_env(key).as_deref() == Some(OsStr::new("1"))
}

fn project_flag_arg(project_dir: &Path) -> OsString {
    let mut arg = OsString::from("--project=");
    arg.push(project_dir.as_os_str());
    arg
}

fn resolve_sysimage_status(
    path: &Path,
    julia_version: Option<&str>,
    manifest_hash: Option<&str>,
) -> SysimageStatus {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(err) => {
            return SysimageStatus::Rejected {
                path: path.to_path_buf(),
                reason: format!("failed to access sysimage {}: {err}", path.display()),
            };
        }
    };
    if !metadata.is_file() {
        return SysimageStatus::Rejected {
            path: path.to_path_buf(),
            reason: format!("sysimage path {} is not a regular file", path.display()),
        };
    }
    let metadata_path = sysimage_metadata_path(path);
    let metadata_raw = match fs::read_to_string(&metadata_path) {
        Ok(raw) => raw,
        Err(err) => {
            return SysimageStatus::Rejected {
                path: path.to_path_buf(),
                reason: format!("failed to read metadata {}: {err}", metadata_path.display()),
            };
        }
    };
    let metadata: SysimageMetadata = match serde_json::from_str(&metadata_raw) {
        Ok(metadata) => metadata,
        Err(err) => {
            return SysimageStatus::Rejected {
                path: path.to_path_buf(),
                reason: format!(
                    "failed to parse metadata {}: {err}",
                    metadata_path.display()
                ),
            };
        }
    };
    let Some(julia_major_minor) = julia_version.and_then(julia_major_minor) else {
        return SysimageStatus::Rejected {
            path: path.to_path_buf(),
            reason: "unable to determine Julia major/minor version".to_string(),
        };
    };
    if metadata.julia_major_minor != julia_major_minor {
        return SysimageStatus::Rejected {
            path: path.to_path_buf(),
            reason: format!(
                "metadata Julia major/minor {} does not match runtime {}",
                metadata.julia_major_minor, julia_major_minor
            ),
        };
    }
    let Some(manifest_hash) = manifest_hash else {
        return SysimageStatus::Rejected {
            path: path.to_path_buf(),
            reason: "unable to hash julia/forecastflows/Manifest.toml".to_string(),
        };
    };
    if metadata.manifest_sha256 != manifest_hash {
        return SysimageStatus::Rejected {
            path: path.to_path_buf(),
            reason: format!(
                "metadata manifest hash {} does not match current {}",
                metadata.manifest_sha256, manifest_hash
            ),
        };
    }
    SysimageStatus::Active(path.to_path_buf())
}

fn sysimage_metadata_path(path: &Path) -> PathBuf {
    match path.file_name() {
        Some(file_name) => path.with_file_name(format!("{}.json", file_name.to_string_lossy())),
        None => path.with_extension("json"),
    }
}

fn probe_julia_version(program: &OsStr) -> Option<String> {
    let output = Command::new(program).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        (!stderr.is_empty()).then_some(stderr)
    } else {
        Some(stdout)
    }
}

fn julia_major_minor(version: &str) -> Option<String> {
    let version_token = version
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))?;
    let mut parts = version_token.split('.');
    let major = parts.next()?;
    let minor = parts.next()?;
    Some(format!("{major}.{minor}"))
}

fn manifest_sha256(project_dir: &Path) -> Option<String> {
    let manifest = fs::read(project_dir.join("Manifest.toml")).ok()?;
    Some(format!("{:x}", Sha256::digest(manifest)))
}

fn manifest_source(project_dir: &Path) -> ForecastFlowsManifestSource {
    let manifest = match fs::read_to_string(project_dir.join("Manifest.toml")) {
        Ok(manifest) => manifest,
        Err(_) => return ForecastFlowsManifestSource::default(),
    };

    let mut source = ForecastFlowsManifestSource::default();
    let mut in_forecastflows_entry = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[[deps.") {
            in_forecastflows_entry = trimmed == "[[deps.ForecastFlows]]";
            continue;
        }
        if !in_forecastflows_entry {
            continue;
        }
        if source.repo_url.is_none() {
            source.repo_url = manifest_string_value(trimmed, "repo-url");
        }
        if source.repo_rev.is_none() {
            source.repo_rev = manifest_string_value(trimmed, "repo-rev");
        }
        if source.git_tree_sha1.is_none() {
            source.git_tree_sha1 = manifest_string_value(trimmed, "git-tree-sha1");
        }
    }

    source
}

fn manifest_string_value(line: &str, key: &str) -> Option<String> {
    let prefix = format!("{key} = \"");
    let remainder = line.strip_prefix(&prefix)?;
    remainder.strip_suffix('"').map(str::to_string)
}

fn solve_tuning_for_profile(profile: ForecastFlowsRequestProfile) -> ForecastFlowsSolveTuning {
    #[cfg(test)]
    if let Some(tuning) = test_solve_tuning() {
        return tuning;
    }
    match profile {
        ForecastFlowsRequestProfile::Benchmark => ForecastFlowsSolveTuning::Baseline,
        ForecastFlowsRequestProfile::Production
        | ForecastFlowsRequestProfile::DoctorWarmup
        | ForecastFlowsRequestProfile::DoctorRepresentative => ForecastFlowsSolveTuning::LowLatency,
    }
}

fn solve_options_for_profile(profile: ForecastFlowsRequestProfile) -> SolveOptionsRequest {
    solve_tuning_for_profile(profile).solve_options()
}

fn configured_worker_pool_size() -> usize {
    #[cfg(test)]
    if let Some(pool_size) = test_worker_pool_size() {
        return pool_size.max(1);
    }

    std::env::var("FORECASTFLOWS_POOL_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn tiny_compare_problem() -> PredictionMarketProblemRequest {
    PredictionMarketProblemRequest {
        outcomes: vec![
            OutcomeSpecRequest {
                outcome_id: "YES".to_string(),
                fair_value: 0.55,
                initial_holding: 0.0,
            },
            OutcomeSpecRequest {
                outcome_id: "NO".to_string(),
                fair_value: 0.45,
                initial_holding: 0.0,
            },
        ],
        collateral_balance: 1.0,
        markets: vec![
            MarketSpecRequest::UniV3 {
                market_id: "warm-u1".to_string(),
                outcome_id: "YES".to_string(),
                current_price: 0.5,
                bands: vec![
                    UniV3LiquidityBandRequest {
                        lower_price: 1.0,
                        liquidity_l: 10.0,
                    },
                    UniV3LiquidityBandRequest {
                        lower_price: 0.25,
                        liquidity_l: 0.0,
                    },
                ],
                fee_multiplier: 0.9999,
            },
            MarketSpecRequest::UniV3 {
                market_id: "warm-u2".to_string(),
                outcome_id: "NO".to_string(),
                current_price: 0.5,
                bands: vec![
                    UniV3LiquidityBandRequest {
                        lower_price: 1.0,
                        liquidity_l: 9.0,
                    },
                    UniV3LiquidityBandRequest {
                        lower_price: 0.25,
                        liquidity_l: 0.0,
                    },
                ],
                fee_multiplier: 0.9999,
            },
        ],
        split_bound: None,
    }
}

fn representative_compare_problem()
-> Result<PredictionMarketProblemRequest, ForecastFlowsClientError> {
    let report: RepresentativeSnapshotReport = serde_json::from_str(REPRESENTATIVE_REPORT_JSON)
        .map_err(|err| {
            ForecastFlowsClientError::Protocol(format!(
                "failed to parse representative ForecastFlows report fixture {REPRESENTATIVE_REPORT_SOURCE}: {err}"
            ))
        })?;
    let count = report.predictions_wad.len();
    if report.starting_prices_wad.len() != count
        || report.liquidity.len() != count
        || report.initial_holdings_wad.len() != count
    {
        return Err(ForecastFlowsClientError::Protocol(
            "representative ForecastFlows report fixture has mismatched vector lengths".to_string(),
        ));
    }

    let mut outcomes = Vec::with_capacity(count);
    let mut markets = Vec::with_capacity(count);
    for index in 0..count {
        let outcome_id = format!("REP-{index:03}");
        let market_id = format!("REP-{index:03}");
        let current_price = wad_to_f64(report.starting_prices_wad[index]);
        let fair_value = wad_to_f64(report.predictions_wad[index]);
        let initial_holding = wad_to_f64(report.initial_holdings_wad[index]);
        let liquidity_l = report.liquidity[index] as f64 / 1e18;
        let buy_limit_price = (current_price * 1.5)
            .clamp(current_price + 1e-6, 1.0)
            .max(current_price + 1e-6);
        let sell_limit_price = (current_price * 0.25).max(1e-6);
        outcomes.push(OutcomeSpecRequest {
            outcome_id: outcome_id.clone(),
            fair_value,
            initial_holding,
        });
        markets.push(MarketSpecRequest::UniV3 {
            market_id,
            outcome_id,
            current_price,
            bands: vec![
                UniV3LiquidityBandRequest {
                    lower_price: buy_limit_price,
                    liquidity_l,
                },
                UniV3LiquidityBandRequest {
                    lower_price: sell_limit_price,
                    liquidity_l: 0.0,
                },
            ],
            fee_multiplier: 0.9999,
        });
    }

    Ok(PredictionMarketProblemRequest {
        outcomes,
        collateral_balance: wad_to_f64(report.initial_cash_budget_wad),
        markets,
        split_bound: None,
    })
}

fn wad_to_f64(value: u128) -> f64 {
    value as f64 / 1e18
}

fn os_string_display(value: &OsStr) -> String {
    value.to_string_lossy().into_owned()
}

fn request_profile() -> ForecastFlowsRequestProfile {
    #[cfg(test)]
    if let Some(profile) = test_request_profile() {
        return profile;
    }
    if let Ok(value) = std::env::var("FORECASTFLOWS_REQUEST_PROFILE")
        && let Some(profile) = ForecastFlowsRequestProfile::parse(&value)
    {
        return profile;
    }
    ForecastFlowsRequestProfile::Production
}

fn request_timeout(profile: ForecastFlowsRequestProfile) -> Duration {
    #[cfg(test)]
    if let Some(timeout) = test_request_timeout() {
        return timeout;
    }
    profile.request_timeout()
}

fn circuit_breaker_threshold() -> u32 {
    #[cfg(test)]
    if let Some(value) = test_circuit_breaker_threshold() {
        return value;
    }
    CIRCUIT_BREAKER_THRESHOLD
}

fn circuit_breaker_base_backoff() -> Duration {
    #[cfg(test)]
    if let Some(value) = test_circuit_breaker_base_backoff() {
        return value;
    }
    Duration::from_secs(CIRCUIT_BREAKER_BASE_BACKOFF_SECS)
}

fn circuit_breaker_max_backoff() -> Duration {
    #[cfg(test)]
    if let Some(value) = test_circuit_breaker_max_backoff() {
        return value;
    }
    Duration::from_secs(CIRCUIT_BREAKER_MAX_BACKOFF_SECS)
}

pub(super) fn warm_worker() -> Result<(), ForecastFlowsClientError> {
    WorkerPool::global().warm_worker(request_profile())
}

pub(super) fn compare_prediction_market_families(
    problem: PredictionMarketProblemRequest,
) -> Result<ForecastFlowsCompareSuccess, ForecastFlowsCompareFailure> {
    WorkerPool::global().compare_prediction_market_families(problem, request_profile())
}

pub(super) fn doctor_report() -> Result<ForecastFlowsDoctorReport, ForecastFlowsClientError> {
    WorkerPool::global().doctor_report()
}

pub(super) fn shutdown_worker() -> Result<(), ForecastFlowsClientError> {
    WorkerPool::global().shutdown_worker()
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TestWorkerHarnessState {
    pub(super) process_present: bool,
    pub(super) warm_complete: bool,
    pub(super) in_cooldown: bool,
    pub(super) stderr_tail_len: usize,
    pub(super) last_failure: Option<String>,
}

#[cfg(test)]
#[derive(Default)]
struct TestOverrides {
    worker_command: Option<Vec<String>>,
    worker_backend: Option<ForecastFlowsBackend>,
    request_timeout: Option<Duration>,
    request_profile: Option<ForecastFlowsRequestProfile>,
    solve_tuning: Option<ForecastFlowsSolveTuning>,
    pool_size: Option<usize>,
    circuit_breaker_threshold: Option<u32>,
    circuit_breaker_base_backoff: Option<Duration>,
    circuit_breaker_max_backoff: Option<Duration>,
}

#[cfg(test)]
fn test_overrides() -> &'static Mutex<TestOverrides> {
    static INSTANCE: OnceLock<Mutex<TestOverrides>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(TestOverrides::default()))
}

#[cfg(test)]
pub(super) fn forecastflows_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
pub(super) fn set_test_worker_command(command: &[&str]) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.worker_command = Some(command.iter().map(|part| (*part).to_string()).collect());
}

#[cfg(test)]
pub(super) fn set_test_worker_backend(backend: ForecastFlowsBackend) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.worker_backend = Some(backend);
}

#[cfg(test)]
pub(super) fn set_test_request_timeout(timeout: Duration) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.request_timeout = Some(timeout);
}

#[cfg(test)]
pub(super) fn set_test_request_profile(profile: ForecastFlowsRequestProfile) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.request_profile = Some(profile);
}

#[cfg(test)]
pub(super) fn set_test_solve_tuning(tuning: ForecastFlowsSolveTuning) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.solve_tuning = Some(tuning);
}

#[cfg(test)]
pub(super) fn set_test_worker_pool_size(pool_size: usize) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.pool_size = Some(pool_size.max(1));
}

#[cfg(test)]
pub(super) fn set_test_circuit_breaker_config(
    threshold: u32,
    base_backoff: Duration,
    max_backoff: Duration,
) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.circuit_breaker_threshold = Some(threshold);
    overrides.circuit_breaker_base_backoff = Some(base_backoff);
    overrides.circuit_breaker_max_backoff = Some(max_backoff);
}

#[cfg(test)]
pub(super) fn reset_test_overrides() {
    {
        let mut overrides = test_overrides().lock().expect("test overrides mutex");
        *overrides = TestOverrides::default();
    }
    let pool = WorkerPool::global();
    for slot in pool.current_slots() {
        let mut state = slot.state.lock().expect("worker service mutex");
        WorkerService::reset_process_locked(&mut state);
        state.consecutive_worker_failures = 0;
        state.cooldown_until = None;
        state.last_failure = None;
        state.last_stderr_tail.clear();
        state.launch_logged = false;
    }
}

#[cfg(test)]
pub(super) fn test_request_timeout_override() -> Option<Duration> {
    test_request_timeout()
}

#[cfg(test)]
pub(super) fn test_request_profile_override() -> Option<ForecastFlowsRequestProfile> {
    test_request_profile()
}

#[cfg(test)]
pub(super) fn test_worker_harness_state() -> TestWorkerHarnessState {
    WorkerPool::global()
        .test_harness_states()
        .into_iter()
        .next()
        .unwrap_or(TestWorkerHarnessState {
            process_present: false,
            warm_complete: false,
            in_cooldown: false,
            stderr_tail_len: 0,
            last_failure: None,
        })
}

#[cfg(test)]
pub(super) fn test_worker_pool_harness_states() -> Vec<TestWorkerHarnessState> {
    WorkerPool::global().test_harness_states()
}

#[cfg(test)]
fn test_worker_launch_config() -> Option<WorkerLaunchConfig> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    let command = overrides.worker_command.as_ref()?;
    let (program, args) = command.split_first()?;
    let backend = overrides
        .worker_backend
        .unwrap_or(ForecastFlowsBackend::JuliaWorker);
    let details = match backend {
        ForecastFlowsBackend::JuliaWorker => WorkerLaunchDetails::Julia(JuliaLaunchDetails {
            project_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("julia")
                .join("forecastflows"),
            julia_version: None,
            manifest_source: ForecastFlowsManifestSource::default(),
            sysimage_status: SysimageStatus::Disabled,
            worker_config: ForecastFlowsWorkerConfig {
                julia_threads: OsString::from("auto"),
                allow_plain_julia_escape_hatch: true,
            },
        }),
        ForecastFlowsBackend::RustWorker => WorkerLaunchDetails::Rust(RustLaunchDetails {
            worker_bin: PathBuf::from(program),
        }),
    };
    Some(WorkerLaunchConfig {
        program: OsString::from(program),
        args: args.iter().map(OsString::from).collect(),
        current_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        envs: Vec::new(),
        details,
    })
}

#[cfg(test)]
fn test_request_timeout() -> Option<Duration> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.request_timeout
}

#[cfg(test)]
fn test_request_profile() -> Option<ForecastFlowsRequestProfile> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.request_profile
}

#[cfg(test)]
fn test_solve_tuning() -> Option<ForecastFlowsSolveTuning> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.solve_tuning
}

#[cfg(test)]
fn test_worker_pool_size() -> Option<usize> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.pool_size
}

#[cfg(test)]
fn test_circuit_breaker_threshold() -> Option<u32> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.circuit_breaker_threshold
}

#[cfg(test)]
fn test_circuit_breaker_base_backoff() -> Option<Duration> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.circuit_breaker_base_backoff
}

#[cfg(test)]
fn test_circuit_breaker_max_backoff() -> Option<Duration> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.circuit_breaker_max_backoff
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
        forecastflows_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("forecastflows-client-{name}-{suffix}"));
        fs::create_dir_all(path.join("julia").join("forecastflows")).expect("temp project dir");
        path
    }

    fn write_manifest(root: &Path, contents: &str) {
        fs::write(
            root.join("julia")
                .join("forecastflows")
                .join("Manifest.toml"),
            contents,
        )
        .expect("manifest write");
    }

    #[test]
    fn build_worker_launch_config_adds_startup_file_and_project_flags() {
        let _guard = lock_tests();
        let root = unique_temp_dir("launch-flags");
        write_manifest(&root, "manifest = true\n");

        let launch = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("julia_worker")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect("launch config should build");

        assert_eq!(launch.program, OsString::from("julia"));
        assert_eq!(launch.args[0], OsString::from("--startup-file=no"));
        assert_eq!(
            launch.args[1],
            project_flag_arg(&root.join("julia").join("forecastflows"))
        );
        assert_eq!(launch.current_dir, root.join("julia"));
        let julia = launch.julia().expect("julia details present");
        assert!(matches!(julia.sysimage_status, SysimageStatus::Disabled));
        assert_eq!(julia.julia_threads_display(), "auto");
        assert_eq!(
            launch.launch_envs(),
            vec![(OsString::from("JULIA_NUM_THREADS"), OsString::from("auto"))]
        );
    }

    #[test]
    fn request_profiles_map_to_expected_timeouts() {
        assert_eq!(
            ForecastFlowsRequestProfile::Production.request_timeout(),
            Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)
        );
        assert_eq!(
            ForecastFlowsRequestProfile::Benchmark.request_timeout(),
            Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS)
        );
        assert_eq!(
            ForecastFlowsRequestProfile::DoctorWarmup.request_timeout(),
            Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS)
        );
        assert_eq!(
            ForecastFlowsRequestProfile::DoctorRepresentative.request_timeout(),
            Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS)
        );
    }

    #[test]
    fn request_profile_parse_accepts_supported_env_values() {
        assert_eq!(
            ForecastFlowsRequestProfile::parse("production"),
            Some(ForecastFlowsRequestProfile::Production)
        );
        assert_eq!(
            ForecastFlowsRequestProfile::parse("benchmark"),
            Some(ForecastFlowsRequestProfile::Benchmark)
        );
        assert_eq!(
            ForecastFlowsRequestProfile::parse("doctor_warmup"),
            Some(ForecastFlowsRequestProfile::DoctorWarmup)
        );
        assert_eq!(
            ForecastFlowsRequestProfile::parse("doctor_representative"),
            Some(ForecastFlowsRequestProfile::DoctorRepresentative)
        );
        assert_eq!(ForecastFlowsRequestProfile::parse("wat"), None);
    }

    #[test]
    fn baseline_solve_tuning_uses_parity_limits() {
        let options = ForecastFlowsSolveTuning::Baseline.solve_options();
        assert!(options.certify);
        assert!(!options.throw_on_fail);
        assert_eq!(options.max_doublings, 6);
        assert_eq!(options.pgtol, 1e-6);
        assert_eq!(options.max_iter, 10_000);
        assert_eq!(options.max_fun, 20_000);
    }

    #[test]
    fn low_latency_solve_tuning_uses_expected_limits() {
        let options = ForecastFlowsSolveTuning::LowLatency.solve_options();
        assert!(options.certify);
        assert!(!options.throw_on_fail);
        assert_eq!(options.max_doublings, 0);
        assert_eq!(options.pgtol, 1e-6);
        assert_eq!(options.max_iter, 2_500);
        assert_eq!(options.max_fun, 5_000);
    }

    #[test]
    fn production_and_doctor_profiles_use_low_latency_tuning() {
        let _guard = lock_tests();
        assert_eq!(
            solve_tuning_for_profile(ForecastFlowsRequestProfile::Production),
            ForecastFlowsSolveTuning::LowLatency
        );
        assert_eq!(
            solve_tuning_for_profile(ForecastFlowsRequestProfile::DoctorWarmup),
            ForecastFlowsSolveTuning::LowLatency
        );
        assert_eq!(
            solve_tuning_for_profile(ForecastFlowsRequestProfile::DoctorRepresentative),
            ForecastFlowsSolveTuning::LowLatency
        );
        assert_eq!(
            solve_tuning_for_profile(ForecastFlowsRequestProfile::Benchmark),
            ForecastFlowsSolveTuning::Baseline
        );
    }

    #[test]
    fn manifest_source_extracts_forecastflows_git_metadata() {
        let _guard = lock_tests();
        let root = unique_temp_dir("manifest-source");
        write_manifest(
            &root,
            r#"
[[deps.ForecastFlows]]
git-tree-sha1 = "abc123"
repo-rev = "codex/ff-public-univ3-parity"
repo-url = "https://github.com/shotaronowhere/ForecastFlows.jl"
"#,
        );

        let source = manifest_source(&root.join("julia").join("forecastflows"));
        assert_eq!(
            source.repo_url.as_deref(),
            Some("https://github.com/shotaronowhere/ForecastFlows.jl")
        );
        assert_eq!(
            source.repo_rev.as_deref(),
            Some("codex/ff-public-univ3-parity")
        );
        assert_eq!(source.git_tree_sha1.as_deref(), Some("abc123"));
    }

    #[test]
    fn build_worker_launch_config_honors_julia_thread_override() {
        let _guard = lock_tests();
        let root = unique_temp_dir("thread-override");
        write_manifest(&root, "manifest = true\n");

        let launch = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("julia_worker")),
                "FORECASTFLOWS_JULIA_THREADS" => Some(OsString::from("8")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect("launch config should build with explicit Julia threads");

        assert_eq!(
            launch
                .julia()
                .expect("julia details present")
                .julia_threads_display(),
            "8"
        );
        assert_eq!(
            launch.launch_envs(),
            vec![(OsString::from("JULIA_NUM_THREADS"), OsString::from("8"))]
        );
    }

    #[test]
    fn production_profile_requires_sysimage_without_escape_hatch() {
        let _guard = lock_tests();
        let root = unique_temp_dir("strict-production");
        write_manifest(&root, "manifest = true\n");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("julia_worker")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Production,
        )
        .expect_err("production profile should reject plain Julia without an escape hatch");
        assert!(err.to_string().contains("FORECASTFLOWS_SYSIMAGE"));
        assert!(
            err.to_string()
                .contains("FORECASTFLOWS_ALLOW_PLAIN_JULIA=1")
        );

        let launch = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("julia_worker")),
                "FORECASTFLOWS_ALLOW_PLAIN_JULIA" => Some(OsString::from("1")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Production,
        )
        .expect("escape hatch should allow plain Julia for local production-style testing");
        let julia = launch.julia().expect("julia details present");
        assert!(matches!(julia.sysimage_status, SysimageStatus::Disabled));
        assert!(julia.worker_config.allow_plain_julia_escape_hatch);
    }

    #[test]
    fn build_worker_launch_config_uses_project_dir_override_for_project_and_manifest_hashing() {
        let _guard = lock_tests();
        let root = unique_temp_dir("project-override");
        write_manifest(&root, "default = true\n");
        let override_project = root.join("override-project");
        fs::create_dir_all(&override_project).expect("override project dir");
        fs::write(override_project.join("Manifest.toml"), "override = true\n")
            .expect("override manifest");
        let sysimage_path = root.join("forecastflows.dylib");
        fs::write(&sysimage_path, b"sysimage").expect("sysimage write");
        let fake_julia = root.join("fake-julia");
        fs::write(&fake_julia, b"#!/bin/sh\necho 'julia version 1.12.0'\n")
            .expect("fake julia write");
        mark_executable(&fake_julia);
        let metadata = SysimageMetadata {
            julia_major_minor: "1.12".to_string(),
            manifest_sha256: manifest_sha256(&override_project).expect("override hash"),
            built_at_unix_secs: 1,
        };
        fs::write(
            sysimage_metadata_path(&sysimage_path),
            serde_json::to_string(&metadata).expect("metadata json"),
        )
        .expect("metadata write");

        let launch = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("julia_worker")),
                "FORECASTFLOWS_JULIA_BIN" => Some(fake_julia.as_os_str().to_os_string()),
                "FORECASTFLOWS_PROJECT_DIR" => Some(OsString::from("override-project")),
                "FORECASTFLOWS_SYSIMAGE" => Some(sysimage_path.as_os_str().to_os_string()),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Production,
        )
        .expect("launch config should honor override project dir");

        let julia = launch.julia().expect("julia details present");
        assert_eq!(julia.project_dir, override_project);
        assert_eq!(
            launch.args[1],
            project_flag_arg(&root.join("override-project"))
        );
        assert_eq!(launch.current_dir, root);
        assert!(matches!(julia.sysimage_status, SysimageStatus::Active(_)));
    }

    #[test]
    fn build_worker_launch_config_rejects_missing_project_dir_override() {
        let _guard = lock_tests();
        let root = unique_temp_dir("missing-override");
        write_manifest(&root, "default = true\n");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("julia_worker")),
                "FORECASTFLOWS_PROJECT_DIR" => Some(OsString::from("missing-project")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect_err("missing override project dir must fail");
        assert!(
            err.to_string()
                .contains("ForecastFlows Julia project directory")
        );
    }

    #[test]
    fn parse_backend_from_env_returns_rust_worker_by_default() {
        let backend = parse_backend_from_env(&|_| None).expect("default backend parses");
        assert_eq!(backend, ForecastFlowsBackend::RustWorker);
    }

    #[test]
    fn parse_backend_from_env_treats_empty_value_as_rust_worker() {
        let backend = parse_backend_from_env(&|key| match key {
            "FORECASTFLOWS_BACKEND" => Some(OsString::new()),
            _ => None,
        })
        .expect("empty value parses");
        assert_eq!(backend, ForecastFlowsBackend::RustWorker);
    }

    #[test]
    fn parse_backend_from_env_accepts_explicit_julia_worker() {
        let backend = parse_backend_from_env(&|key| match key {
            "FORECASTFLOWS_BACKEND" => Some(OsString::from("julia_worker")),
            _ => None,
        })
        .expect("explicit julia_worker parses");
        assert_eq!(backend, ForecastFlowsBackend::JuliaWorker);
    }

    #[test]
    fn parse_backend_from_env_accepts_rust_worker() {
        let backend = parse_backend_from_env(&|key| match key {
            "FORECASTFLOWS_BACKEND" => Some(OsString::from("rust_worker")),
            _ => None,
        })
        .expect("rust_worker parses");
        assert_eq!(backend, ForecastFlowsBackend::RustWorker);
    }

    #[test]
    fn parse_backend_from_env_rejects_unknown_value() {
        let err = parse_backend_from_env(&|key| match key {
            "FORECASTFLOWS_BACKEND" => Some(OsString::from("go_worker")),
            _ => None,
        })
        .expect_err("unknown backend should fail");
        let message = err.to_string();
        assert!(message.contains("FORECASTFLOWS_BACKEND"));
        assert!(message.contains("julia_worker"));
        assert!(message.contains("rust_worker"));
    }

    #[test]
    fn build_rust_worker_launch_config_requires_worker_bin_env() {
        let _guard = lock_tests();
        let root = unique_temp_dir("rust-missing-env");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("rust_worker")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect_err("rust backend without FORECASTFLOWS_WORKER_BIN must fail");
        assert!(err.to_string().contains("FORECASTFLOWS_WORKER_BIN"));
    }

    #[test]
    fn build_rust_worker_launch_config_rejects_nonexistent_binary() {
        let _guard = lock_tests();
        let root = unique_temp_dir("rust-missing-binary");
        let bogus = root.join("does-not-exist");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("rust_worker")),
                "FORECASTFLOWS_WORKER_BIN" => Some(bogus.as_os_str().to_os_string()),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect_err("missing rust worker binary must fail");
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn build_rust_worker_launch_config_rejects_directory_binary() {
        let _guard = lock_tests();
        let root = unique_temp_dir("rust-dir-binary");
        let dir_path = root.join("worker-dir");
        fs::create_dir_all(&dir_path).expect("worker dir");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("rust_worker")),
                "FORECASTFLOWS_WORKER_BIN" => Some(dir_path.as_os_str().to_os_string()),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect_err("directory path for rust worker binary must fail");
        assert!(err.to_string().contains("not a regular file"));
    }

    #[cfg(unix)]
    fn mark_executable(path: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = fs::metadata(path).expect("stat worker bin").permissions();
        perm.set_mode(0o755);
        fs::set_permissions(path, perm).expect("chmod worker bin");
    }

    #[cfg(not(unix))]
    fn mark_executable(_path: &Path) {}

    #[test]
    fn build_rust_worker_launch_config_produces_direct_launch_without_julia_flags() {
        let _guard = lock_tests();
        let root = unique_temp_dir("rust-happy");
        let worker_bin = root.join("forecast-flows-worker");
        fs::write(&worker_bin, b"#!/bin/sh\nexit 0\n").expect("worker bin write");
        mark_executable(&worker_bin);

        let launch = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("rust_worker")),
                "FORECASTFLOWS_WORKER_BIN" => Some(worker_bin.as_os_str().to_os_string()),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Production,
        )
        .expect("rust worker launch should build under production profile without sysimage gate");

        assert_eq!(launch.backend(), ForecastFlowsBackend::RustWorker);
        assert!(matches!(launch.details, WorkerLaunchDetails::Rust(_)));
        assert_eq!(launch.program, worker_bin.as_os_str());
        assert!(launch.args.is_empty(), "rust worker takes no launch args");
        assert!(
            launch.launch_envs().is_empty(),
            "rust worker must not propagate JULIA_NUM_THREADS or other Julia env vars",
        );
        assert_eq!(launch.current_dir, root);
        assert!(launch.julia().is_none());
        assert!(launch.runtime_detail().starts_with("rust worker binary "));
    }

    #[test]
    fn build_rust_worker_launch_config_rejects_relative_path() {
        let _guard = lock_tests();
        let root = unique_temp_dir("rust-relative");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("rust_worker")),
                "FORECASTFLOWS_WORKER_BIN" => Some(OsString::from("forecast-flows-worker")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect_err("relative worker bin path must fail up front");
        assert!(
            err.to_string().contains("must be an absolute path"),
            "error should cite absolute-path requirement, got {err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn build_rust_worker_launch_config_rejects_non_executable_file() {
        use std::os::unix::fs::PermissionsExt;
        let _guard = lock_tests();
        let root = unique_temp_dir("rust-non-exec");
        let worker_bin = root.join("forecast-flows-worker");
        fs::write(&worker_bin, b"fake").expect("worker bin write");
        let mut perm = fs::metadata(&worker_bin)
            .expect("stat worker bin")
            .permissions();
        perm.set_mode(0o644);
        fs::set_permissions(&worker_bin, perm).expect("chmod worker bin");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_BACKEND" => Some(OsString::from("rust_worker")),
                "FORECASTFLOWS_WORKER_BIN" => Some(worker_bin.as_os_str().to_os_string()),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect_err("non-executable worker bin must fail up front");
        assert!(
            err.to_string().contains("not executable"),
            "error should cite executable-bit requirement, got {err}"
        );
    }

    #[test]
    fn build_worker_launch_config_default_uses_rust_backend() {
        let _guard = lock_tests();
        let root = unique_temp_dir("backend-default-rust");
        let worker_bin = root.join("forecast-flows-worker");
        fs::write(&worker_bin, b"#!/bin/sh\nexit 0\n").expect("worker bin write");
        mark_executable(&worker_bin);

        let launch = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_WORKER_BIN" => Some(worker_bin.as_os_str().to_os_string()),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
            ForecastFlowsRequestProfile::Benchmark,
        )
        .expect("default backend should build via Rust worker path");

        assert_eq!(launch.backend(), ForecastFlowsBackend::RustWorker);
        assert!(matches!(launch.details, WorkerLaunchDetails::Rust(_)));
    }

    #[test]
    fn representative_compare_problem_uses_embedded_full_l1_fixture() {
        let _guard = lock_tests();
        let problem = representative_compare_problem().expect("fixture should parse");
        assert_eq!(
            REPRESENTATIVE_REPORT_SOURCE,
            "embedded:test/fixtures/rebalancer_ab_live_l1_snapshot_report.json"
        );
        assert!(problem.outcomes.len() >= 90);
        assert_eq!(problem.outcomes.len(), problem.markets.len());
        assert!(problem.collateral_balance > 0.0);
    }

    #[test]
    fn resolve_sysimage_status_accepts_matching_metadata() {
        let _guard = lock_tests();
        let root = unique_temp_dir("sysimage-valid");
        write_manifest(&root, "manifest = true\n");
        let sysimage_path = root.join("forecastflows.dylib");
        fs::write(&sysimage_path, b"sysimage").expect("sysimage write");
        let metadata = SysimageMetadata {
            julia_major_minor: "1.12".to_string(),
            manifest_sha256: manifest_sha256(&root.join("julia").join("forecastflows"))
                .expect("hash"),
            built_at_unix_secs: 1,
        };
        fs::write(
            sysimage_metadata_path(&sysimage_path),
            serde_json::to_string(&metadata).expect("metadata json"),
        )
        .expect("metadata write");

        let status = resolve_sysimage_status(
            &sysimage_path,
            Some("julia version 1.12.5"),
            manifest_sha256(&root.join("julia").join("forecastflows")).as_deref(),
        );
        assert!(matches!(status, SysimageStatus::Active(_)));
    }

    #[test]
    fn resolve_sysimage_status_rejects_manifest_mismatch() {
        let _guard = lock_tests();
        let root = unique_temp_dir("sysimage-mismatch");
        write_manifest(&root, "manifest = true\n");
        let sysimage_path = root.join("forecastflows.dylib");
        fs::write(&sysimage_path, b"sysimage").expect("sysimage write");
        let metadata = SysimageMetadata {
            julia_major_minor: "1.12".to_string(),
            manifest_sha256: "deadbeef".to_string(),
            built_at_unix_secs: 1,
        };
        fs::write(
            sysimage_metadata_path(&sysimage_path),
            serde_json::to_string(&metadata).expect("metadata json"),
        )
        .expect("metadata write");

        let status = resolve_sysimage_status(
            &sysimage_path,
            Some("julia version 1.12.5"),
            manifest_sha256(&root.join("julia").join("forecastflows")).as_deref(),
        );
        match status {
            SysimageStatus::Rejected { reason, .. } => {
                assert!(reason.contains("manifest hash"));
            }
            other => panic!("expected rejected status, got {other:?}"),
        }
    }

    #[test]
    fn resolve_sysimage_status_rejects_missing_sysimage_file() {
        let _guard = lock_tests();
        let root = unique_temp_dir("sysimage-missing");
        write_manifest(&root, "manifest = true\n");
        let sysimage_path = root.join("forecastflows.dylib");
        let metadata = SysimageMetadata {
            julia_major_minor: "1.12".to_string(),
            manifest_sha256: manifest_sha256(&root.join("julia").join("forecastflows"))
                .expect("hash"),
            built_at_unix_secs: 1,
        };
        fs::write(
            sysimage_metadata_path(&sysimage_path),
            serde_json::to_string(&metadata).expect("metadata json"),
        )
        .expect("metadata write");

        let status = resolve_sysimage_status(
            &sysimage_path,
            Some("julia version 1.12.5"),
            manifest_sha256(&root.join("julia").join("forecastflows")).as_deref(),
        );
        match status {
            SysimageStatus::Rejected { reason, .. } => {
                assert!(reason.contains("failed to access sysimage"));
            }
            other => panic!("expected rejected status, got {other:?}"),
        }
    }

    #[test]
    fn circuit_breaker_enters_cooldown_after_consecutive_failures() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "closed",
        ]);
        set_test_circuit_breaker_config(1, Duration::from_secs(5), Duration::from_secs(5));

        let err = warm_worker().expect_err("worker should fail");
        assert_eq!(err.fallback_reason(), "worker_closed");
        let slot = WorkerPool::global().first_slot();
        let state = slot.state.lock().expect("state");
        assert_eq!(state.consecutive_worker_failures, 1);
        assert!(state.cooldown_until.is_some());
        drop(state);

        reset_test_overrides();
    }

    #[test]
    fn cooldown_resets_after_successful_compare() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "healthy_direct",
        ]);
        warm_worker().expect("healthy fake worker should warm");
        let compare = compare_prediction_market_families(tiny_compare_problem())
            .expect("compare should pass");
        assert_eq!(compare.request_count, 1);
        assert!(compare.roundtrip > Duration::ZERO);
        assert_eq!(compare.runtime_policy.solve_tuning, "low_latency");
        assert!(!compare.compare.workspace_reused);

        let slot = WorkerPool::global().first_slot();
        let state = slot.state.lock().expect("state");
        assert_eq!(state.consecutive_worker_failures, 0);
        assert!(state.cooldown_until.is_none());
        drop(state);

        reset_test_overrides();
    }

    #[test]
    fn compare_under_rust_backend_override_reports_rust_runtime_policy() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_backend(ForecastFlowsBackend::RustWorker);
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "healthy_direct",
        ]);

        warm_worker().expect("rust-backend fake worker should warm");
        let compare = compare_prediction_market_families(tiny_compare_problem())
            .expect("rust-backend compare should pass");
        assert_eq!(compare.request_count, 1);
        assert_eq!(
            compare.runtime_policy.backend,
            ForecastFlowsBackend::RustWorker
        );
        assert!(compare.runtime_policy.sysimage_status.is_none());
        assert!(compare.runtime_policy.julia_threads.is_none());

        reset_test_overrides();
    }

    #[test]
    fn cached_execution_model_reuses_workspace_on_second_compare() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "cached_reuse",
        ]);

        let first = compare_prediction_market_families(tiny_compare_problem())
            .expect("first cached compare should pass");
        let second = compare_prediction_market_families(tiny_compare_problem())
            .expect("second cached compare should pass");

        assert!(!first.compare.workspace_reused);
        assert!(second.compare.workspace_reused);

        reset_test_overrides();
    }

    #[test]
    fn failure_messages_include_worker_stderr_tail() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "stderr_closed",
        ]);

        let err = warm_worker().expect_err("worker should fail after writing stderr");
        assert!(err.to_string().contains("stderr line one"));
        assert!(err.to_string().contains("stderr line two"));

        reset_test_overrides();
    }

    #[test]
    fn worker_error_response_does_not_reset_or_trip_breaker() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "worker_error",
        ]);

        let err = compare_prediction_market_families(tiny_compare_problem())
            .expect_err("worker should return ok=false");
        assert_eq!(err.fallback_reason(), "worker_error_response");
        assert_eq!(err.request_count, 1);
        assert_eq!(
            err.runtime_policy
                .as_ref()
                .map(|policy| policy.solve_tuning.as_str()),
            Some("low_latency")
        );

        let slot = WorkerPool::global().first_slot();
        let mut state = slot.state.lock().expect("state");
        assert_eq!(state.consecutive_worker_failures, 0);
        assert!(state.cooldown_until.is_none());
        let process = state.process.as_mut().expect("worker should remain live");
        assert!(
            process.child.try_wait().expect("child status").is_none(),
            "worker process should still be running"
        );
        drop(state);

        reset_test_overrides();
    }

    #[test]
    fn warm_worker_keeps_worker_alive_on_request_level_worker_error() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "worker_error",
        ]);

        let err = warm_worker().expect_err("warmup compare should return ok=false");
        assert_eq!(err.fallback_reason(), "worker_error_response");

        let slot = WorkerPool::global().first_slot();
        let mut state = slot.state.lock().expect("state");
        assert_eq!(state.consecutive_worker_failures, 0);
        assert!(state.cooldown_until.is_none());
        assert!(!state.warm_complete);
        let process = state.process.as_mut().expect("worker should remain live");
        assert!(
            process.child.try_wait().expect("child status").is_none(),
            "worker process should still be running"
        );
        drop(state);

        reset_test_overrides();
    }

    #[test]
    fn doctor_report_returns_partial_diagnostics_on_worker_failure() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "stderr_closed",
        ]);

        let report = doctor_report().expect("doctor should return partial report");
        assert!(!report.worker_program.is_empty());
        assert_eq!(report.worker_backend, "julia_worker");
        assert!(report.worker_project_dir.is_some());
        assert!(report.sysimage_status.is_some());
        assert_eq!(report.representative_fixture, REPRESENTATIVE_REPORT_SOURCE);
        assert_eq!(report.health_status.as_deref(), Some("ok"));
        assert!(report.warmup_compare_duration_ms.is_none());
        assert!(report.representative_compare_duration_ms.is_none());
        assert!(report.failure.is_some());
        assert!(
            report
                .stderr_tail
                .iter()
                .any(|line| line.contains("stderr line one"))
        );
        assert!(
            report
                .stderr_tail
                .iter()
                .any(|line| line.contains("stderr line two"))
        );

        reset_test_overrides();
    }

    #[test]
    fn doctor_report_elides_julia_fields_under_rust_backend() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_backend(ForecastFlowsBackend::RustWorker);
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "healthy_direct",
        ]);

        let report = doctor_report().expect("doctor should pass under rust backend");
        assert_eq!(report.worker_backend, "rust_worker");
        assert!(report.worker_project_dir.is_none());
        assert!(report.julia_version.is_none());
        assert!(report.manifest_repo_url.is_none());
        assert!(report.manifest_repo_rev.is_none());
        assert!(report.manifest_git_tree_sha1.is_none());
        assert!(report.julia_threads.is_none());
        assert!(report.allow_plain_julia_escape_hatch.is_none());
        assert!(report.sysimage_status.is_none());
        assert!(report.sysimage_detail.is_none());
        assert_eq!(report.health_status.as_deref(), Some("ok"));
        assert!(
            report
                .worker_runtime_detail
                .starts_with("rust worker binary ")
        );
        assert!(report.warmup_compare_duration_ms.is_some());
        assert!(report.representative_compare_duration_ms.is_some());
        assert!(report.failure.is_none());

        reset_test_overrides();
    }

    #[test]
    fn doctor_report_keeps_worker_alive_on_request_level_worker_error() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "worker_error",
        ]);

        let report = doctor_report().expect("doctor should return partial report");
        assert_eq!(report.health_status.as_deref(), Some("ok"));
        assert!(report.warmup_compare_duration_ms.is_none());
        assert!(report.representative_compare_duration_ms.is_none());
        assert!(
            report
                .failure
                .as_deref()
                .is_some_and(|message| message.contains("synthetic worker error"))
        );

        let slot = WorkerPool::global().first_slot();
        let mut state = slot.state.lock().expect("state");
        assert_eq!(state.consecutive_worker_failures, 0);
        assert!(state.cooldown_until.is_none());
        assert!(!state.warm_complete);
        let process = state.process.as_mut().expect("worker should remain live");
        assert!(
            process.child.try_wait().expect("child status").is_none(),
            "worker process should still be running"
        );
        drop(state);

        reset_test_overrides();
    }

    #[test]
    fn shutdown_worker_clears_process_and_allows_fresh_warmup() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "healthy_direct",
        ]);

        warm_worker().expect("healthy fake worker should warm");
        shutdown_worker().expect("shutdown should succeed");
        shutdown_worker().expect("shutdown should be idempotent");

        {
            let slot = WorkerPool::global().first_slot();
            let state = slot.state.lock().expect("state");
            assert!(state.process.is_none());
            assert!(!state.warm_complete);
        }

        warm_worker().expect("worker should warm again after shutdown");

        let slot = WorkerPool::global().first_slot();
        let state = slot.state.lock().expect("state");
        assert!(state.process.is_some());
        assert!(state.warm_complete);
        drop(state);

        reset_test_overrides();
    }

    #[test]
    fn worker_pool_size_one_matches_singleton_behavior() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_pool_size(1);
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "healthy_direct",
        ]);

        warm_worker().expect("single pooled worker should warm");
        let compare = compare_prediction_market_families(tiny_compare_problem())
            .expect("single pooled compare should pass");
        assert_eq!(compare.request_count, 1);

        let states = test_worker_pool_harness_states();
        assert_eq!(states.len(), 1);
        assert!(states[0].process_present);
        assert!(states[0].warm_complete);
        assert!(!states[0].in_cooldown);

        reset_test_overrides();
    }

    #[test]
    fn worker_pool_keeps_cooldown_and_stderr_state_isolated_per_worker() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_pool_size(2);
        set_test_circuit_breaker_config(1, Duration::from_secs(60), Duration::from_secs(60));
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "stderr_closed",
        ]);

        let first_err = compare_prediction_market_families(tiny_compare_problem())
            .expect_err("first pool slot should fail closed after writing stderr");
        assert_eq!(first_err.fallback_reason(), "worker_closed");

        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "healthy_direct",
        ]);
        let second = compare_prediction_market_families(tiny_compare_problem())
            .expect("second pool slot should stay available");
        assert_eq!(second.request_count, 1);

        let states = test_worker_pool_harness_states();
        assert_eq!(states.len(), 2);
        assert!(states[0].in_cooldown);
        assert!(states[0].stderr_tail_len >= 2);
        assert!(
            states[0]
                .last_failure
                .as_deref()
                .is_some_and(|message| message.contains("stderr line one"))
        );
        assert!(states[1].process_present);
        assert!(!states[1].in_cooldown);
        assert_eq!(states[1].stderr_tail_len, 0);
        assert_eq!(states[1].last_failure, None);

        let third_err = compare_prediction_market_families(tiny_compare_problem())
            .expect_err("round-robin should revisit the cooled-down first slot");
        assert_eq!(third_err.fallback_reason(), "worker_cooldown");

        reset_test_overrides();
    }

    #[test]
    fn background_respawn_keeps_captured_request_profile() {
        let _guard = lock_tests();
        reset_test_overrides();
        set_test_worker_command(&[
            "/bin/sh",
            &format!(
                "{}/test/bin/fake_forecastflows_worker.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
            "benchmark_warmup_only",
        ]);

        let service = WorkerPool::global().first_slot();
        let mut overrides = test_overrides().lock().expect("test overrides mutex");
        overrides.request_profile = Some(ForecastFlowsRequestProfile::Benchmark);
        {
            let mut state = service.state.lock().expect("state");
            service.schedule_background_respawn_locked(
                &mut state,
                ForecastFlowsRequestProfile::Benchmark,
            );
            assert!(state.respawn_scheduled);
        }
        overrides.request_profile = None;
        drop(overrides);

        let started = std::time::Instant::now();
        loop {
            let state = service.state.lock().expect("state");
            if !state.respawn_scheduled {
                assert!(
                    state.warm_complete,
                    "background respawn should warm successfully with the captured benchmark profile"
                );
                break;
            }
            drop(state);
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "background respawn did not finish in time"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        reset_test_overrides();
    }
}
