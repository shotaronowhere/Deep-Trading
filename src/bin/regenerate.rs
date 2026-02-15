//! Regenerates markets.rs and predictions.rs from API + RPC data.
//! Run: cargo run --bin regenerate

#[path = "regenerate/common.rs"]
mod common;
#[path = "regenerate/markets.rs"]
mod markets_codegen;
#[path = "regenerate/predictions.rs"]
mod predictions_codegen;

use std::env;
use std::path::Path;

use markets_codegen::{DEFAULT_OPTIMISM_RPC_URL, generate_markets_rs};
use predictions_codegen::generate_predictions_rs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let manifest_path = Path::new(&manifest_dir);

    let l1_predictions_csv = manifest_path.join("l1-predictions.csv");
    let l2_predictions_csv = manifest_path.join("l2-predictions.csv");
    let originality_predictions_csv = manifest_path.join("originality-predictions.csv");
    let markets_data_l1_json = manifest_path.join("markets_data_l1.json");
    let markets_data_l2_json = manifest_path.join("markets_data_l2.json");
    let markets_data_originality_json = manifest_path.join("markets_data_originality.json");
    let predictions_rs = manifest_path.join("src/predictions.rs");
    let markets_rs = manifest_path.join("src/markets.rs");

    if !l1_predictions_csv.exists() {
        return Err("l1-predictions.csv not found".into());
    }
    if !l2_predictions_csv.exists() {
        return Err("l2-predictions.csv not found".into());
    }
    if !originality_predictions_csv.exists() {
        return Err("originality-predictions.csv not found".into());
    }

    generate_predictions_rs(
        &l1_predictions_csv,
        &l2_predictions_csv,
        &originality_predictions_csv,
        &predictions_rs,
    )?;

    let rpc_url = env::var("RPC")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_OPTIMISM_RPC_URL.to_string());

    eprintln!("Generating markets.rs from JSON data using RPC {}", rpc_url);

    generate_markets_rs(
        &markets_data_l1_json,
        &markets_data_l2_json,
        &markets_data_originality_json,
        &markets_rs,
        &rpc_url,
    )
    .await?;

    eprintln!("Done. Generated files:");
    eprintln!("  {}", predictions_rs.display());
    eprintln!("  {}", markets_rs.display());

    Ok(())
}
