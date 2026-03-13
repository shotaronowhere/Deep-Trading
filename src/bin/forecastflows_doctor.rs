use std::fmt::Display;
use std::process::ExitCode;

use deep_trading_bot::portfolio::{self, ForecastFlowsDoctorReport};

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

fn print_report(report: &ForecastFlowsDoctorReport) {
    println!("ForecastFlows doctor");
    println!("julia_program: {}", report.julia_program);
    println!(
        "julia_version: {}",
        report.julia_version.as_deref().unwrap_or("unknown")
    );
    println!("project_dir: {}", report.project_dir);
    println!("launch_args: {}", report.launch_args.join(" "));
    println!("sysimage_status: {}", report.sysimage_status);
    println!("sysimage_detail: {}", report.sysimage_detail);
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
        "representative_compare_duration_ms",
        report.representative_compare_duration_ms,
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

    match portfolio::forecastflows_doctor_report() {
        Ok(report) => {
            print_report(&report);
            if report.failure.is_some() {
                ExitCode::FAILURE
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
