use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const GENERATED_FILES: &[&str] = &["src/markets.rs", "src/predictions.rs"];
const SKIP_STALENESS_ENV: &str = "DEEP_TRADING_SKIP_GENERATED_STALENESS_CHECK";
const GENERATOR_INPUTS: &[&str] = &[
    "l1-predictions.csv",
    "l2-predictions.csv",
    "originality-predictions.csv",
    "markets_data_l1.json",
    "markets_data_l2.json",
    "markets_data_originality.json",
    "src/bin/regenerate.rs",
    "src/bin/regenerate/common.rs",
    "src/bin/regenerate/markets.rs",
    "src/bin/regenerate/predictions.rs",
];

fn modified_time(path: &Path) -> SystemTime {
    fs::metadata(path)
        .unwrap_or_else(|err| panic!("missing required file '{}': {}", path.display(), err))
        .modified()
        .unwrap_or_else(|err| panic!("failed reading mtime for '{}': {}", path.display(), err))
}

fn newest(paths: &[PathBuf]) -> (&PathBuf, SystemTime) {
    paths
        .iter()
        .max_by_key(|path| modified_time(path))
        .map(|path| (path, modified_time(path)))
        .expect("path list must be non-empty")
}

fn oldest(paths: &[PathBuf]) -> (&PathBuf, SystemTime) {
    paths
        .iter()
        .min_by_key(|path| modified_time(path))
        .map(|path| (path, modified_time(path)))
        .expect("path list must be non-empty")
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| panic!("missing CARGO_MANIFEST_DIR: {}", err));
    let root = Path::new(&manifest_dir);

    let output_paths: Vec<PathBuf> = GENERATED_FILES.iter().map(|p| root.join(p)).collect();
    let input_paths: Vec<PathBuf> = GENERATOR_INPUTS.iter().map(|p| root.join(p)).collect();

    for path in GENERATED_FILES {
        println!("cargo::rerun-if-changed={path}");
    }
    for path in GENERATOR_INPUTS {
        println!("cargo::rerun-if-changed={path}");
    }

    for output in &output_paths {
        if !output.exists() {
            panic!("Generated files missing. Run: cargo run --bin regenerate");
        }
    }

    if std::env::var_os(SKIP_STALENESS_ENV).is_some() {
        return;
    }

    let (newest_input, newest_input_mtime) = newest(&input_paths);
    let (oldest_output, oldest_output_mtime) = oldest(&output_paths);

    if newest_input_mtime > oldest_output_mtime {
        panic!(
            "Generated files are stale: '{}' is newer than '{}'. Run: cargo run --bin regenerate (or set {}=1 only while regenerating)",
            newest_input.display(),
            oldest_output.display(),
            SKIP_STALENESS_ENV
        );
    }
}
