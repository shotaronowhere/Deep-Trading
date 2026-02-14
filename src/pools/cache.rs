use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::markets::MARKETS_L1;

/// Default cache staleness: 5 minutes.
const CACHE_MAX_AGE_SECS: u64 = 300;

/// Cached balance data, serialized to JSON.
#[derive(Debug, Serialize, Deserialize)]
pub struct BalanceCache {
    pub timestamp: u64,
    pub wallet: String,
    pub susds: f64,
    pub outcomes: HashMap<String, f64>,
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Save balance cache to a JSON file.
pub fn save_balance_cache(
    path: &Path,
    wallet: &str,
    susds: f64,
    outcomes: &HashMap<&str, f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let cache = BalanceCache {
        timestamp: now_secs(),
        wallet: wallet.to_string(),
        susds,
        outcomes: outcomes.iter().map(|(&k, &v)| (k.to_string(), v)).collect(),
    };
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &cache)?;
    Ok(())
}

/// Load balance cache if it exists, matches wallet, and is fresh enough.
/// Returns None if cache is missing, stale, or for a different wallet.
pub fn load_balance_cache(
    path: &Path,
    wallet: &str,
    max_age_secs: Option<u64>,
) -> Option<BalanceCache> {
    let file = std::fs::File::open(path).ok()?;
    let cache: BalanceCache = serde_json::from_reader(std::io::BufReader::new(file)).ok()?;
    if !cache.wallet.eq_ignore_ascii_case(wallet) {
        return None;
    }
    let max_age = max_age_secs.unwrap_or(CACHE_MAX_AGE_SECS);
    if now_secs().saturating_sub(cache.timestamp) > max_age {
        return None;
    }
    Some(cache)
}

/// Convert cached outcome balances (String keys) to static str keys by matching market names.
pub fn cache_to_balances(cache: &BalanceCache) -> HashMap<&'static str, f64> {
    let mut result = HashMap::new();
    for market in MARKETS_L1.iter() {
        if let Some(&val) = cache.outcomes.get(market.name) {
            result.insert(market.name, val);
        }
    }
    result
}
