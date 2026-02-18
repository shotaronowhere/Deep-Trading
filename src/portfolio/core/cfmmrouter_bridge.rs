#![allow(dead_code)]

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CfmmrouterFixture {
    pub prices: Vec<f64>,
    pub predictions: Vec<f64>,
    pub holdings: Vec<f64>,
    pub cash: f64,
    pub allow_complete_set: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CfmmrouterParityResult {
    pub status: String,
    pub message: Option<String>,
    pub objective: Option<f64>,
    pub buys: Vec<f64>,
    pub sells: Vec<f64>,
    pub theta: f64,
}

pub(crate) fn run_cfmmrouter_cli(
    fixture: &CfmmrouterFixture,
) -> Result<CfmmrouterParityResult, String> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script_path = repo_root.join("tools/cfmmrouter/route_fixture.jl");
    if !script_path.exists() {
        return Err(format!(
            "CFMMRouter fixture runner missing at {}",
            script_path.display()
        ));
    }

    let tmp_name = format!(
        "cfmmrouter_fixture_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_micros())
            .unwrap_or(0)
    );
    let fixture_path = std::env::temp_dir().join(tmp_name);
    let fixture_json =
        serde_json::to_vec_pretty(fixture).map_err(|err| format!("serialize fixture: {err}"))?;
    std::fs::write(&fixture_path, fixture_json)
        .map_err(|err| format!("write fixture {}: {err}", fixture_path.display()))?;

    let julia_bin = std::env::var("JULIA_BIN").unwrap_or_else(|_| "julia".to_string());
    let output = Command::new(&julia_bin)
        .arg(&script_path)
        .arg("--fixture")
        .arg(&fixture_path)
        .output()
        .map_err(|err| format!("spawn {julia_bin}: {err}"));

    let _ = std::fs::remove_file(&fixture_path);

    let output = output?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "CFMMRouter runner failed status={} stdout={} stderr={}",
            output.status, stdout, stderr
        ));
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| format!("CFMMRouter runner stdout utf8: {err}"))?;
    serde_json::from_str::<CfmmrouterParityResult>(stdout.trim())
        .map_err(|err| format!("parse runner json output: {err}; stdout={stdout}"))
}
