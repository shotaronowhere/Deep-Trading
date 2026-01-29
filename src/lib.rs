use alloy::sol;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

pub mod markets;
pub mod weights;

sol! {
    #[sol(rpc)]
    contract Market {
        function wrappedOutcome(uint256 index) external view returns (address wrapped1155, bytes memory data);
        function outcomes(uint256) external view returns (string memory);
    }
}

// --- Structs for prepare() API response ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiResponse {
    markets_data: BTreeMap<String, ApiMarket>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiMarket {
    id: String,
    market_id: Option<String>,  // L1 has this, L2 doesn't
    pool: Option<Pool>,         // L1 uses single pool
    pools: Option<Vec<Pool>>,   // L2 uses pools array
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Pool {
    token0: String,
    token1: String,
    liquidity: String,
    ticks: Vec<Tick>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Tick {
    tick_idx: String,
    liquidity_net: String,
}

// --- Structs for output JSON ---

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputFile {
    markets_data: BTreeMap<String, OutputMarket>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputMarket {
    outcome_token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    market_id: Option<String>,
    pools: Vec<Pool>,
}

impl From<ApiResponse> for OutputFile {
    fn from(api: ApiResponse) -> Self {
        let markets_data = api
            .markets_data
            .into_iter()
            .map(|(name, market)| (name, market.into()))
            .collect();
        Self { markets_data }
    }
}

impl From<ApiMarket> for OutputMarket {
    fn from(api: ApiMarket) -> Self {
        // Collect all pools: L1 has single pool, L2 has pools array
        let pools: Vec<Pool> = api
            .pool
            .into_iter()
            .chain(api.pools.into_iter().flatten())
            .collect();

        Self {
            outcome_token: api.id,
            market_id: api.market_id,
            pools,
        }
    }
}

/// Fetches market data from the API and saves it to markets_data.json.
/// Run this to update the local data file, then rebuild to regenerate markets.rs.
pub async fn prepare(url: &str, file_name: &str) -> Result<(), Box<dyn Error>> {
    let response = reqwest::get(url).await?.error_for_status()?;
    let api_data: ApiResponse = response.json().await?;

    let output_data: OutputFile = api_data.into();
    let file = File::create(file_name)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &output_data)?;
    println!("Saved to {}", file_name);

    Ok(())
}

///

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prepare_l1() {
        let url: &str = "https://deep.seer.pm/.netlify/functions/get-l1-markets-data";
        let name = "markets_data_l1.json";
        let result = prepare(url, name).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_prepare_l2() {
        let url: &str = "https://deep.seer.pm/.netlify/functions/get-l2-markets-data";
        let name = "markets_data_l2.json";
        let result = prepare(url, name).await;
        assert!(result.is_ok());
    }
}
