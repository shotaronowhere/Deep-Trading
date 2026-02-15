use serde::Deserialize;
use std::collections::HashSet;
use std::fmt::Write;
use std::fs;
use std::path::Path;

use super::common::rustfmt_generated_file;

#[derive(Debug, Deserialize)]
struct L1PredictionRecord {
    repo: String,
    #[serde(rename = "parent")]
    _parent: String,
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct L2PredictionRecord {
    dependency: String,
    repo: String,
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct OriginalityRecord {
    repo: String,
    originality: f64,
}

/// Format f64 to always include a decimal point (for valid Rust float literals)
fn format_f64(f: f64) -> String {
    let s = f.to_string();
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{}.0", s)
    }
}

pub(super) fn generate_predictions_rs(
    l1_csv_path: &Path,
    l2_csv_path: &Path,
    originality_csv_path: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse and validate L1 predictions
    let mut l1_predictions = Vec::new();
    let mut seen_l1_markets = HashSet::new();
    let mut reader = csv::Reader::from_path(l1_csv_path)?;
    for (i, result) in reader.deserialize().enumerate() {
        let record: L1PredictionRecord = result?;
        let row = i + 2; // 1-indexed, skip header
        let repo = record.repo.trim();
        let market = repo.strip_prefix("https://github.com/").unwrap_or(repo);
        if record.weight < 0.0 {
            panic!("l1-predictions.csv row {}: weight cannot be negative", row);
        }
        if !seen_l1_markets.insert(market.to_string()) {
            panic!(
                "l1-predictions.csv row {}: duplicate market '{}'",
                row, market
            );
        }
        l1_predictions.push((market.to_string(), record.weight));
    }

    // Parse and validate L2 predictions
    let mut l2_predictions = Vec::new();
    let mut seen_l2_pairs = HashSet::new();
    let mut reader = csv::Reader::from_path(l2_csv_path)?;
    for (i, result) in reader.deserialize().enumerate() {
        let record: L2PredictionRecord = result?;
        let row = i + 2;
        if record.weight < 0.0 {
            panic!("l2-predictions.csv row {}: weight cannot be negative", row);
        }
        let pair = (record.dependency.clone(), record.repo.clone());
        if !seen_l2_pairs.insert(pair) {
            panic!(
                "l2-predictions.csv row {}: duplicate dependency '{}' for repo '{}'",
                row, record.dependency, record.repo
            );
        }
        l2_predictions.push((record.dependency, record.repo, record.weight));
    }

    // Parse and validate originality predictions
    let mut originality_predictions = Vec::new();
    let mut seen_originality_markets = HashSet::new();
    let mut reader = csv::Reader::from_path(originality_csv_path)?;
    for (i, result) in reader.deserialize().enumerate() {
        let record: OriginalityRecord = result?;
        let row = i + 2;
        let repo = record.repo.trim();
        let market = repo.strip_prefix("https://github.com/").unwrap_or(repo);
        if record.originality < 0.0 || record.originality > 1.0 {
            panic!(
                "originality-predictions.csv row {}: originality must be in [0, 1], got {}",
                row, record.originality
            );
        }
        if !seen_originality_markets.insert(market.to_string()) {
            panic!(
                "originality-predictions.csv row {}: duplicate market '{}'",
                row, market
            );
        }
        originality_predictions.push((market.to_string(), record.originality));
    }

    let mut output = String::new();
    // Validate no duplicate normalized keys (matches runtime normalize_market_name behavior)
    let mut normalized_keys = HashSet::new();
    for (market, _) in &l1_predictions {
        let key = market
            .trim_end_matches("\\t")
            .trim_end_matches('\t')
            .to_lowercase();
        if !normalized_keys.insert(key.clone()) {
            return Err(format!("duplicate normalized L1 prediction key '{key}'").into());
        }
    }

    writeln!(
        &mut output,
        "// AUTO-GENERATED — do not edit manually.\n// Regenerate with: cargo run --bin regenerate\n"
    )?;

    // L1 Prediction struct and array
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct Prediction {{")?;
    writeln!(&mut output, "    pub market: &'static str,")?;
    writeln!(&mut output, "    pub prediction: f64,")?;
    writeln!(&mut output, "}}\n")?;

    writeln!(
        &mut output,
        "pub static PREDICTIONS_L1: [Prediction; {}] = [",
        l1_predictions.len()
    )?;
    for (market, weight) in &l1_predictions {
        writeln!(
            &mut output,
            "    Prediction {{ market: \"{}\", prediction: {} }},",
            market,
            format_f64(*weight)
        )?;
    }
    writeln!(&mut output, "];\n")?;

    // L2 Prediction struct and array
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct PredictionL2 {{")?;
    writeln!(&mut output, "    pub dependency: &'static str,")?;
    writeln!(&mut output, "    pub repo: &'static str,")?;
    writeln!(&mut output, "    pub prediction: f64,")?;
    writeln!(&mut output, "}}\n")?;

    writeln!(
        &mut output,
        "pub static PREDICTIONS_L2: [PredictionL2; {}] = [",
        l2_predictions.len()
    )?;
    for (dependency, repo, weight) in &l2_predictions {
        writeln!(
            &mut output,
            "    PredictionL2 {{ dependency: \"{}\", repo: \"{}\", prediction: {} }},",
            dependency,
            repo,
            format_f64(*weight)
        )?;
    }
    writeln!(&mut output, "];\n")?;

    // Originality struct and array
    writeln!(&mut output, "#[derive(Debug, Clone, Copy)]")?;
    writeln!(&mut output, "pub struct Originality {{")?;
    writeln!(&mut output, "    pub market: &'static str,")?;
    writeln!(&mut output, "    pub originality: f64,")?;
    writeln!(&mut output, "}}\n")?;

    writeln!(
        &mut output,
        "pub static ORIGINALITY: [Originality; {}] = [",
        originality_predictions.len()
    )?;
    for (market, originality) in &originality_predictions {
        writeln!(
            &mut output,
            "    Originality {{ market: \"{}\", originality: {} }},",
            market,
            format_f64(*originality)
        )?;
    }
    writeln!(&mut output, "];")?;

    fs::write(output_path, output)?;
    rustfmt_generated_file(output_path);
    eprintln!(
        "Generated {} with {} L1, {} L2, {} originality predictions",
        output_path.display(),
        l1_predictions.len(),
        l2_predictions.len(),
        originality_predictions.len()
    );

    Ok(())
}
