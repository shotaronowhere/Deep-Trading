use serde::{Deserialize, Serialize};

pub(super) const PROTOCOL_VERSION: u64 = 2;
pub(super) const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 5;
pub(super) const WARMUP_REQUEST_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, Clone, Serialize)]
pub(super) struct OutcomeSpecRequest {
    pub(super) outcome_id: String,
    pub(super) fair_value: f64,
    pub(super) initial_holding: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct UniV3LiquidityBandRequest {
    pub(super) lower_price: f64,
    #[serde(rename = "liquidity_L")]
    pub(super) liquidity_l: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub(super) enum MarketSpecRequest {
    #[serde(rename = "univ3")]
    UniV3 {
        market_id: String,
        outcome_id: String,
        current_price: f64,
        bands: Vec<UniV3LiquidityBandRequest>,
        fee_multiplier: f64,
    },
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct PredictionMarketProblemRequest {
    pub(super) outcomes: Vec<OutcomeSpecRequest>,
    pub(super) collateral_balance: f64,
    pub(super) markets: Vec<MarketSpecRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) split_bound: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SolveOptionsRequest {
    pub(super) certify: bool,
    pub(super) throw_on_fail: bool,
    pub(super) max_doublings: u64,
    pub(super) pgtol: f64,
    pub(super) max_iter: u64,
    pub(super) max_fun: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct CompareRequestEnvelope {
    pub(super) protocol_version: u64,
    pub(super) request_id: String,
    pub(super) command: &'static str,
    pub(super) problem: PredictionMarketProblemRequest,
    pub(super) solve_options: SolveOptionsRequest,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct WorkerErrorPayload {
    pub(super) code: String,
    pub(super) message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct WorkerEnvelope<T> {
    pub(super) protocol_version: u64,
    pub(super) request_id: Option<String>,
    pub(super) ok: bool,
    pub(super) command: Option<String>,
    pub(super) result: Option<T>,
    pub(super) error: Option<WorkerErrorPayload>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct HealthResult {
    pub(super) status: String,
    pub(super) supported_commands: Vec<String>,
    pub(super) supported_modes: Vec<String>,
    pub(super) execution_model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct CompareResult {
    pub(super) direct_only: PredictionMarketSolveResult,
    pub(super) mixed_enabled: PredictionMarketSolveResult,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct SolveCertificateSummary {
    pub(super) passed: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(super) struct SplitMergePlan {
    pub(super) mint: f64,
    pub(super) merge: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct PredictionMarketTrade {
    pub(super) market_id: String,
    pub(super) outcome_id: String,
    pub(super) collateral_delta: f64,
    pub(super) outcome_delta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct PredictionMarketSolveResult {
    pub(super) status: String,
    pub(super) mode: String,
    pub(super) certificate: Option<SolveCertificateSummary>,
    #[serde(default)]
    pub(super) trades: Vec<PredictionMarketTrade>,
    #[serde(default)]
    pub(super) split_merge: SplitMergePlan,
}

impl PredictionMarketSolveResult {
    pub(super) fn is_certified(&self) -> bool {
        self.status == "certified"
            && self
                .certificate
                .as_ref()
                .is_some_and(|certificate| certificate.passed)
    }
}
