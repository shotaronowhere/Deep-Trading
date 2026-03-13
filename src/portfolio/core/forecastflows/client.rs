use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
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

#[derive(Debug, Clone)]
struct WorkerLaunchConfig {
    program: OsString,
    args: Vec<OsString>,
    current_dir: PathBuf,
    project_dir: PathBuf,
    julia_version: Option<String>,
    sysimage_status: SysimageStatus,
}

impl WorkerLaunchConfig {
    fn sysimage_status_label(&self) -> &'static str {
        self.sysimage_status.label()
    }

    fn sysimage_status_detail(&self) -> String {
        self.sysimage_status.detail()
    }

    fn program_display(&self) -> String {
        os_string_display(&self.program)
    }

    fn launch_args(&self) -> Vec<String> {
        self.args.iter().map(|arg| os_string_display(arg)).collect()
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

struct WorkerService {
    state: Mutex<WorkerState>,
}

#[derive(Debug)]
struct HealthCheckOutcome {
    result: HealthResult,
    duration: Duration,
}

impl WorkerService {
    fn global() -> &'static Self {
        static INSTANCE: OnceLock<WorkerService> = OnceLock::new();
        INSTANCE.get_or_init(|| WorkerService {
            state: Mutex::new(WorkerState::default()),
        })
    }

    fn warm_worker(&self) -> Result<(), ForecastFlowsClientError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ForecastFlowsClientError::Closed("worker mutex poisoned".to_string()))?;
        if let Some(err) = Self::cooldown_error_locked(&state) {
            return Err(err);
        }

        if let Err(err) = Self::ensure_process_locked(&mut state) {
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
            solve_options: default_solve_options(),
        };
        let warm_result: Result<CompareResult, ForecastFlowsClientError> =
            Self::send_request_locked(
                &mut state,
                &request_id,
                "compare_prediction_market_families",
                &request,
                Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS),
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
        &self,
        problem: PredictionMarketProblemRequest,
    ) -> Result<(CompareResult, usize), ForecastFlowsCompareFailure> {
        let mut request_count = 0usize;

        for attempt in 0..2 {
            let mut state = self.state.lock().map_err(|_| ForecastFlowsCompareFailure {
                error: ForecastFlowsClientError::Closed("worker mutex poisoned".to_string()),
                request_count,
            })?;
            if let Some(err) = Self::cooldown_error_locked(&state) {
                return Err(ForecastFlowsCompareFailure {
                    error: err,
                    request_count,
                });
            }
            if let Err(err) = Self::ensure_process_locked(&mut state) {
                if attempt == 1 {
                    let err = Self::record_and_reset_failure_locked(&mut state, err);
                    return Err(ForecastFlowsCompareFailure {
                        error: err,
                        request_count,
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
                solve_options: default_solve_options(),
            };
            request_count += 1;
            match Self::send_request_locked(
                &mut state,
                &request_id,
                "compare_prediction_market_families",
                &request,
                request_timeout(),
            ) {
                Ok(result) => {
                    Self::note_success_locked(&mut state);
                    Self::log_launch_details_locked(&mut state);
                    return Ok((result, request_count));
                }
                Err(err) => {
                    if err.is_request_level_worker_failure() {
                        return Err(ForecastFlowsCompareFailure {
                            error: err.with_context(Self::stderr_context_locked(&state)),
                            request_count,
                        });
                    }
                    if attempt == 1 {
                        let err = Self::record_and_reset_failure_locked(&mut state, err);
                        Self::schedule_background_respawn_locked(&mut state);
                        return Err(ForecastFlowsCompareFailure {
                            error: err,
                            request_count,
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
        })
    }

    fn doctor_report(&self) -> Result<ForecastFlowsDoctorReport, ForecastFlowsClientError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ForecastFlowsClientError::Closed("worker mutex poisoned".to_string()))?;
        Self::reset_process_locked(&mut state);
        state.launch_logged = false;

        let launch = worker_launch_config()?;
        let mut report = ForecastFlowsDoctorReport {
            julia_program: launch.program_display(),
            julia_version: launch.julia_version.clone(),
            project_dir: launch.project_dir.display().to_string(),
            launch_args: launch.launch_args(),
            sysimage_status: launch.sysimage_status_label().to_string(),
            sysimage_detail: launch.sysimage_status_detail(),
            health_status: None,
            supported_commands: Vec::new(),
            supported_modes: Vec::new(),
            execution_model: None,
            health_duration_ms: None,
            warmup_compare_duration_ms: None,
            representative_compare_duration_ms: None,
            representative_fixture: REPRESENTATIVE_REPORT_SOURCE.to_string(),
            stderr_tail: Self::stderr_tail_snapshot_locked(&state),
            failure: None,
        };

        let health = match Self::ensure_process_locked(&mut state) {
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
            solve_options: default_solve_options(),
        };
        let warmup_started = Instant::now();
        let warmup_result: Result<CompareResult, ForecastFlowsClientError> =
            Self::send_request_locked(
                &mut state,
                &warmup_request_id,
                "compare_prediction_market_families",
                &warmup_request,
                Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS),
            );
        if let Err(err) = warmup_result {
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
        report.warmup_compare_duration_ms = Some(warmup_started.elapsed().as_millis());
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
            solve_options: default_solve_options(),
        };
        let representative_started = Instant::now();
        let representative_result: Result<CompareResult, ForecastFlowsClientError> =
            Self::send_request_locked(
                &mut state,
                &representative_request_id,
                "compare_prediction_market_families",
                &representative_request,
                Duration::from_secs(WARMUP_REQUEST_TIMEOUT_SECS),
            );
        if let Err(err) = representative_result {
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
        report.representative_compare_duration_ms =
            Some(representative_started.elapsed().as_millis());

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
    ) -> Result<Option<HealthCheckOutcome>, ForecastFlowsClientError> {
        if state.process.is_some() {
            return Ok(None);
        }

        let process = Self::spawn_worker_process()?;
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
        if result.execution_model != "stateless NDJSON; one request at a time per worker process" {
            return Err(ForecastFlowsClientError::Protocol(format!(
                "unexpected worker execution model {}",
                result.execution_model
            )));
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

    fn spawn_worker_process() -> Result<WorkerProcess, ForecastFlowsClientError> {
        let launch = worker_launch_config()?;
        if let SysimageStatus::Rejected { reason, .. } = &launch.sysimage_status {
            tracing::warn!(reason = %reason, sysimage = %launch.sysimage_status.detail(), "ForecastFlows sysimage disabled; falling back to plain Julia");
        }
        let mut command = Command::new(&launch.program);
        command
            .args(&launch.args)
            .current_dir(&launch.current_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
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

    fn schedule_background_respawn_locked(state: &mut WorkerState) {
        if state.respawn_scheduled || Self::cooldown_remaining_locked(state).is_some() {
            return;
        }
        state.respawn_scheduled = true;
        let service = WorkerService::global();
        thread::spawn(move || {
            if let Err(err) = service.warm_worker() {
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
        tracing::info!(
            julia_program = %process.launch_config.program_display(),
            julia_version = process.launch_config.julia_version.as_deref().unwrap_or("unknown"),
            project_dir = %process.launch_config.project_dir.display(),
            sysimage_status = process.launch_config.sysimage_status_label(),
            sysimage_detail = %process.launch_config.sysimage_status_detail(),
            "ForecastFlows worker launch configuration"
        );
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

fn worker_launch_config() -> Result<WorkerLaunchConfig, ForecastFlowsClientError> {
    #[cfg(test)]
    if let Some(config) = test_worker_launch_config() {
        return Ok(config);
    }
    build_worker_launch_config_from_env(
        |key| std::env::var_os(key),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        std::env::current_dir().ok(),
    )
}

fn build_worker_launch_config_from_env(
    read_env: impl Fn(&str) -> Option<OsString>,
    repo_root: PathBuf,
    current_dir: Option<PathBuf>,
) -> Result<WorkerLaunchConfig, ForecastFlowsClientError> {
    let project_dir = resolve_project_dir(&read_env, &repo_root, current_dir.as_deref())?;
    let program = read_env("FORECASTFLOWS_JULIA_BIN").unwrap_or_else(|| OsString::from("julia"));
    let julia_version = probe_julia_version(program.as_os_str());
    let manifest_hash = manifest_sha256(&project_dir).ok_or_else(|| {
        ForecastFlowsClientError::Spawn(format!(
            "failed to hash ForecastFlows manifest {}",
            project_dir.join("Manifest.toml").display()
        ))
    })?;
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

    Ok(WorkerLaunchConfig {
        program,
        args,
        current_dir: project_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| project_dir.clone()),
        project_dir,
        julia_version,
        sysimage_status,
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

fn default_solve_options() -> SolveOptionsRequest {
    SolveOptionsRequest {
        certify: true,
        throw_on_fail: false,
        max_doublings: 6,
        pgtol: 1e-8,
        max_iter: 5_000,
        max_fun: 10_000,
    }
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

fn request_timeout() -> Duration {
    #[cfg(test)]
    if let Some(timeout) = test_request_timeout() {
        return timeout;
    }
    Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)
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
    WorkerService::global().warm_worker()
}

pub(super) fn compare_prediction_market_families(
    problem: PredictionMarketProblemRequest,
) -> Result<(CompareResult, usize), ForecastFlowsCompareFailure> {
    WorkerService::global().compare_prediction_market_families(problem)
}

pub(super) fn doctor_report() -> Result<ForecastFlowsDoctorReport, ForecastFlowsClientError> {
    WorkerService::global().doctor_report()
}

pub(super) fn shutdown_worker() -> Result<(), ForecastFlowsClientError> {
    WorkerService::global().shutdown_worker()
}

#[cfg(test)]
#[derive(Default)]
struct TestOverrides {
    worker_command: Option<Vec<String>>,
    request_timeout: Option<Duration>,
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
pub(super) fn set_test_request_timeout(timeout: Duration) {
    let mut overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.request_timeout = Some(timeout);
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
    let mut state = WorkerService::global()
        .state
        .lock()
        .expect("worker service mutex");
    WorkerService::reset_process_locked(&mut state);
    state.consecutive_worker_failures = 0;
    state.cooldown_until = None;
    state.last_failure = None;
    state.last_stderr_tail.clear();
    state.launch_logged = false;
}

#[cfg(test)]
fn test_worker_launch_config() -> Option<WorkerLaunchConfig> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    let command = overrides.worker_command.as_ref()?;
    let (program, args) = command.split_first()?;
    Some(WorkerLaunchConfig {
        program: OsString::from(program),
        args: args.iter().map(OsString::from).collect(),
        current_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        project_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("julia")
            .join("forecastflows"),
        julia_version: None,
        sysimage_status: SysimageStatus::Disabled,
    })
}

#[cfg(test)]
fn test_request_timeout() -> Option<Duration> {
    let overrides = test_overrides().lock().expect("test overrides mutex");
    overrides.request_timeout
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

        let launch =
            build_worker_launch_config_from_env(|_| None, root.clone(), Some(root.clone()))
                .expect("launch config should build");

        assert_eq!(launch.program, OsString::from("julia"));
        assert_eq!(launch.args[0], OsString::from("--startup-file=no"));
        assert_eq!(
            launch.args[1],
            project_flag_arg(&root.join("julia").join("forecastflows"))
        );
        assert_eq!(launch.current_dir, root.join("julia"));
        assert!(matches!(launch.sysimage_status, SysimageStatus::Disabled));
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
                "FORECASTFLOWS_PROJECT_DIR" => Some(OsString::from("override-project")),
                "FORECASTFLOWS_SYSIMAGE" => Some(sysimage_path.as_os_str().to_os_string()),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
        )
        .expect("launch config should honor override project dir");

        assert_eq!(launch.project_dir, override_project);
        assert_eq!(
            launch.args[1],
            project_flag_arg(&root.join("override-project"))
        );
        assert_eq!(launch.current_dir, root);
        assert!(matches!(launch.sysimage_status, SysimageStatus::Active(_)));
    }

    #[test]
    fn build_worker_launch_config_rejects_missing_project_dir_override() {
        let _guard = lock_tests();
        let root = unique_temp_dir("missing-override");
        write_manifest(&root, "default = true\n");

        let err = build_worker_launch_config_from_env(
            |key| match key {
                "FORECASTFLOWS_PROJECT_DIR" => Some(OsString::from("missing-project")),
                _ => None,
            },
            root.clone(),
            Some(root.clone()),
        )
        .expect_err("missing override project dir must fail");
        assert!(
            err.to_string()
                .contains("ForecastFlows Julia project directory")
        );
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
        let state = WorkerService::global().state.lock().expect("state");
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
        let (_, request_count) = compare_prediction_market_families(tiny_compare_problem())
            .expect("compare should pass");
        assert_eq!(request_count, 1);

        let state = WorkerService::global().state.lock().expect("state");
        assert_eq!(state.consecutive_worker_failures, 0);
        assert!(state.cooldown_until.is_none());
        drop(state);

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

        let mut state = WorkerService::global().state.lock().expect("state");
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

        let mut state = WorkerService::global().state.lock().expect("state");
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
        assert!(!report.julia_program.is_empty());
        assert!(!report.project_dir.is_empty());
        assert!(!report.sysimage_status.is_empty());
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

        let mut state = WorkerService::global().state.lock().expect("state");
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
            let state = WorkerService::global().state.lock().expect("state");
            assert!(state.process.is_none());
            assert!(!state.warm_complete);
        }

        warm_worker().expect("worker should warm again after shutdown");

        let state = WorkerService::global().state.lock().expect("state");
        assert!(state.process.is_some());
        assert!(state.warm_complete);
        drop(state);

        reset_test_overrides();
    }
}
