use std::ffi::OsString;
use std::fmt::Display;
use std::process::ExitCode;

use deep_trading_bot::portfolio::{self, ForecastFlowsDoctorReport};

#[derive(Debug, Default, PartialEq, Eq)]
struct DoctorCliOptions {
    json: bool,
    require_production_ready: bool,
}

struct ForecastFlowsShutdownGuard;

impl Drop for ForecastFlowsShutdownGuard {
    fn drop(&mut self) {
        if let Err(err) = portfolio::shutdown_forecastflows_worker() {
            eprintln!("warning: failed to shut down ForecastFlows worker ({err})");
        }
    }
}

fn print_list(label: &str, values: &[String]) {
    if values.is_empty() {
        println!("{label}: none");
        return;
    }
    println!("{label}: {}", values.join(", "));
}

fn print_optional<T: Display>(label: &str, value: Option<T>, missing: &str) {
    match value {
        Some(value) => println!("{label}: {value}"),
        None => println!("{label}: {missing}"),
    }
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<DoctorCliOptions, String> {
    let mut options = DoctorCliOptions::default();
    for arg in args.into_iter().skip(1) {
        match arg.to_string_lossy().as_ref() {
            "--json" => options.json = true,
            "--require-production-ready" => options.require_production_ready = true,
            "--help" | "-h" => {
                return Err(
                    "usage: forecastflows_doctor [--json] [--require-production-ready]".to_string(),
                );
            }
            other => {
                return Err(format!(
                    "unknown argument {other}; usage: forecastflows_doctor [--json] [--require-production-ready]"
                ));
            }
        }
    }
    Ok(options)
}

fn production_ready_failure(report: &ForecastFlowsDoctorReport) -> Option<String> {
    match report.worker_backend.as_str() {
        // The Rust worker has no sysimage gate; any successful health probe is
        // sufficient to declare the backend production-ready.
        "rust_worker" => None,
        "julia_worker" => {
            if report.sysimage_status.as_deref() == Some("active") {
                None
            } else {
                Some(format!(
                    "production-ready ForecastFlows (julia_worker) requires an active sysimage ({})",
                    report.sysimage_detail.as_deref().unwrap_or("unknown")
                ))
            }
        }
        other => Some(format!("unknown ForecastFlows worker backend: {other}")),
    }
}

fn print_report(report: &ForecastFlowsDoctorReport) {
    println!("ForecastFlows doctor");
    println!("worker_backend: {}", report.worker_backend);
    println!("worker_program: {}", report.worker_program);
    print_optional(
        "worker_version",
        report.worker_version.as_deref(),
        "unknown",
    );
    print_optional(
        "worker_project_dir",
        report.worker_project_dir.as_deref(),
        "n/a",
    );
    println!(
        "worker_launch_args: {}",
        report.worker_launch_args.join(" ")
    );
    println!("worker_runtime_detail: {}", report.worker_runtime_detail);
    if report.worker_backend == "julia_worker" {
        print_optional("julia_version", report.julia_version.as_deref(), "n/a");
        print_optional(
            "manifest_repo_url",
            report.manifest_repo_url.as_deref(),
            "n/a",
        );
        print_optional(
            "manifest_repo_rev",
            report.manifest_repo_rev.as_deref(),
            "n/a",
        );
        print_optional(
            "manifest_git_tree_sha1",
            report.manifest_git_tree_sha1.as_deref(),
            "n/a",
        );
        print_optional("julia_threads", report.julia_threads.as_deref(), "n/a");
    }
    println!("live_solve_tuning: {}", report.live_solve_tuning);
    if report.worker_backend == "julia_worker" {
        print_optional(
            "allow_plain_julia_escape_hatch",
            report.allow_plain_julia_escape_hatch,
            "n/a",
        );
        print_optional("sysimage_status", report.sysimage_status.as_deref(), "n/a");
        print_optional("sysimage_detail", report.sysimage_detail.as_deref(), "n/a");
    }
    print_optional("health_status", report.health_status.as_deref(), "unknown");
    print_list("supported_commands", &report.supported_commands);
    print_list("supported_modes", &report.supported_modes);
    print_optional(
        "execution_model",
        report.execution_model.as_deref(),
        "unknown",
    );
    print_optional("health_duration_ms", report.health_duration_ms, "none");
    print_optional(
        "warmup_compare_duration_ms",
        report.warmup_compare_duration_ms,
        "none",
    );
    print_optional(
        "warmup_direct_status",
        report.warmup_direct_status.as_deref(),
        "none",
    );
    print_optional(
        "warmup_mixed_status",
        report.warmup_mixed_status.as_deref(),
        "none",
    );
    print_optional(
        "warmup_direct_solver_time_ms",
        report.warmup_direct_solver_time_ms,
        "none",
    );
    print_optional(
        "warmup_mixed_solver_time_ms",
        report.warmup_mixed_solver_time_ms,
        "none",
    );
    print_optional(
        "warmup_driver_overhead_ms",
        report.warmup_driver_overhead_ms,
        "none",
    );
    print_optional(
        "representative_compare_duration_ms",
        report.representative_compare_duration_ms,
        "none",
    );
    print_optional(
        "representative_direct_status",
        report.representative_direct_status.as_deref(),
        "none",
    );
    print_optional(
        "representative_mixed_status",
        report.representative_mixed_status.as_deref(),
        "none",
    );
    print_optional(
        "representative_direct_solver_time_ms",
        report.representative_direct_solver_time_ms,
        "none",
    );
    print_optional(
        "representative_mixed_solver_time_ms",
        report.representative_mixed_solver_time_ms,
        "none",
    );
    print_optional(
        "representative_driver_overhead_ms",
        report.representative_driver_overhead_ms,
        "none",
    );
    println!("representative_fixture: {}", report.representative_fixture);
    if report.stderr_tail.is_empty() {
        println!("stderr_tail: none");
    } else {
        println!("stderr_tail:");
        for line in &report.stderr_tail {
            println!("  {line}");
        }
    }
    if let Some(failure) = &report.failure {
        println!("failure: {failure}");
    }
}

fn main() -> ExitCode {
    dotenvy::dotenv().ok();
    let _forecastflows_shutdown_guard = ForecastFlowsShutdownGuard;
    let options = match parse_args(std::env::args_os()) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            return if message.starts_with("usage:") {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            };
        }
    };

    match portfolio::forecastflows_doctor_report() {
        Ok(report) => {
            if options.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report)
                        .expect("ForecastFlows doctor report should serialize")
                );
            } else {
                print_report(&report);
            }

            if report.failure.is_some() {
                ExitCode::FAILURE
            } else if options.require_production_ready {
                if let Some(message) = production_ready_failure(&report) {
                    if !options.json {
                        eprintln!("{message}");
                    }
                    ExitCode::FAILURE
                } else {
                    ExitCode::SUCCESS
                }
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DoctorCliOptions, parse_args, production_ready_failure};
    use deep_trading_bot::portfolio::ForecastFlowsDoctorReport;
    use std::ffi::OsString;

    fn empty_julia_report() -> ForecastFlowsDoctorReport {
        ForecastFlowsDoctorReport {
            worker_backend: "julia_worker".to_string(),
            worker_program: "julia".to_string(),
            worker_version: None,
            worker_project_dir: Some("/tmp/project".to_string()),
            worker_launch_args: vec!["julia".to_string()],
            worker_runtime_detail: "julia worker (sysimage disabled)".to_string(),
            julia_version: Some("1.12.5".to_string()),
            manifest_repo_url: Some(
                "https://github.com/shotaronowhere/ForecastFlows.jl".to_string(),
            ),
            manifest_repo_rev: Some("codex/ff-public-univ3-parity".to_string()),
            manifest_git_tree_sha1: Some("abc123".to_string()),
            julia_threads: Some("4".to_string()),
            live_solve_tuning: "low_latency".to_string(),
            allow_plain_julia_escape_hatch: Some(false),
            sysimage_status: Some("disabled".to_string()),
            sysimage_detail: Some("no FORECASTFLOWS_SYSIMAGE configured".to_string()),
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
            representative_fixture: "fixture".to_string(),
            stderr_tail: Vec::new(),
            failure: None,
        }
    }

    fn empty_rust_report() -> ForecastFlowsDoctorReport {
        ForecastFlowsDoctorReport {
            worker_backend: "rust_worker".to_string(),
            worker_program: "/opt/forecast-flows-worker".to_string(),
            worker_version: Some("2.0.0".to_string()),
            worker_project_dir: None,
            worker_launch_args: Vec::new(),
            worker_runtime_detail: "rust worker binary /opt/forecast-flows-worker".to_string(),
            julia_version: None,
            manifest_repo_url: None,
            manifest_repo_rev: None,
            manifest_git_tree_sha1: None,
            julia_threads: None,
            live_solve_tuning: "low_latency".to_string(),
            allow_plain_julia_escape_hatch: None,
            sysimage_status: None,
            sysimage_detail: None,
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
            representative_fixture: "fixture".to_string(),
            stderr_tail: Vec::new(),
            failure: None,
        }
    }

    #[test]
    fn parse_args_accepts_json_and_production_ready_flags() {
        let options = parse_args([
            OsString::from("forecastflows_doctor"),
            OsString::from("--json"),
            OsString::from("--require-production-ready"),
        ])
        .expect("flags should parse");
        assert_eq!(
            options,
            DoctorCliOptions {
                json: true,
                require_production_ready: true,
            }
        );
    }

    #[test]
    fn production_ready_failure_requires_active_sysimage_under_julia() {
        let mut report = empty_julia_report();
        assert!(production_ready_failure(&report).is_some());

        report.sysimage_status = Some("active".to_string());
        report.sysimage_detail = Some("using /tmp/forecastflows.so".to_string());
        assert_eq!(production_ready_failure(&report), None);
    }

    #[test]
    fn production_ready_accepts_rust_worker_without_sysimage() {
        let report = empty_rust_report();
        assert_eq!(production_ready_failure(&report), None);
    }
}
