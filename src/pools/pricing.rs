use std::collections::HashMap;
use std::sync::OnceLock;

use alloy_primitives::U256;
use uniswap_v3_math::full_math::mul_div;

use crate::predictions::PREDICTIONS_L1;

/// Normalize a market name for prediction lookup:
/// trim trailing literal "\\t" or tab, then lowercase.
pub fn normalize_market_name(name: &str) -> String {
    name.trim_end_matches("\\t")
        .trim_end_matches('\t')
        .to_lowercase()
}

/// Returns a shared map from normalized market name (lowercase) to prediction value.
pub fn prediction_map() -> &'static HashMap<String, f64> {
    static PREDICTIONS_BY_MARKET: OnceLock<HashMap<String, f64>> = OnceLock::new();
    PREDICTIONS_BY_MARKET.get_or_init(|| {
        let mut by_market = HashMap::with_capacity(PREDICTIONS_L1.len());
        for p in PREDICTIONS_L1 {
            let key = normalize_market_name(p.market);
            let prev = by_market.insert(key.clone(), p.prediction);
            assert!(
                prev.is_none(),
                "duplicate normalized prediction key '{}' for market '{}' (previous value: {:?}, new value: {})",
                key,
                p.market,
                prev,
                p.prediction
            );
        }
        by_market
    })
}

/// Price scale factor: 10^18 (18 decimals of precision)
const PRICE_SCALE: U256 = U256::from_limbs([1_000_000_000_000_000_000u64, 0, 0, 0]);
const TWO_96: U256 = U256::from_limbs([0, 0x1_0000_0000, 0, 0]); // 2^96

/// Converts sqrtPriceX96 to price (token1/token0) scaled by 10^18.
/// Returns None on overflow (shouldn't happen with valid sqrtPriceX96).
pub fn sqrt_price_x96_to_price(sqrt_price_x96: U256) -> Option<U256> {
    // price = sqrtPriceX96² / 2^192
    // price_scaled = sqrtPriceX96² * 10^18 / 2^192
    //              = mul_div(sqrtPriceX96, sqrtPriceX96 * 10^18, 2^192)
    let scaled = sqrt_price_x96.checked_mul(PRICE_SCALE)?;
    mul_div(sqrt_price_x96, scaled, TWO_96 * TWO_96).ok()
}

/// Converts sqrtPriceX96 to inverse price (token0/token1) scaled by 10^18.
/// Returns None on overflow or division by zero.
pub fn sqrt_price_x96_to_inv_price(sqrt_price_x96: U256) -> Option<U256> {
    // inv_price = 2^192 / sqrtPriceX96²
    // inv_price_scaled = 2^192 * 10^18 / sqrtPriceX96²
    // Compute as: mul_div(2^96, 2^96, sqrtPriceX96) * 10^18 / sqrtPriceX96
    let intermediate = mul_div(TWO_96, TWO_96, sqrt_price_x96).ok()?;
    mul_div(intermediate, PRICE_SCALE, sqrt_price_x96).ok()
}

/// Returns price of outcome token in quote token, scaled by 10^18.
pub fn sqrt_price_x96_to_price_outcome(
    sqrt_price_x96: U256,
    is_token1_outcome: bool,
) -> Option<U256> {
    if is_token1_outcome {
        // outcome is token1, quote is token0 -> price = token0/token1 = 1/price
        sqrt_price_x96_to_inv_price(sqrt_price_x96)
    } else {
        // outcome is token0, quote is token1 -> price = token1/token0
        sqrt_price_x96_to_price(sqrt_price_x96)
    }
}

/// Converts U256 (assumed 18-decimal fixed point) to f64.
pub(crate) fn u256_to_f64(v: U256) -> f64 {
    f64::from(v) / 1e18
}

/// Converts a prediction probability to the corresponding sqrtPriceX96.
pub(crate) fn prediction_to_sqrt_price_x96(
    prediction: f64,
    is_token1_outcome: bool,
) -> Option<U256> {
    let prediction_scaled = U256::from((prediction * 1e18) as u128);
    if prediction_scaled.is_zero() {
        return None;
    }
    let two_192 = TWO_96 * TWO_96;
    if is_token1_outcome {
        // outcome_price = 2^192 * 1e18 / sqrtPriceX96^2
        // sqrtPriceX96 = sqrt(2^192 * 1e18 / prediction_scaled)
        let numerator = mul_div(two_192, PRICE_SCALE, prediction_scaled).ok()?;
        Some(numerator.root(2))
    } else {
        // outcome_price = sqrtPriceX96^2 * 1e18 / 2^192
        // sqrtPriceX96 = sqrt(prediction_scaled * 2^192 / 1e18)
        let numerator = mul_div(prediction_scaled, two_192, PRICE_SCALE).ok()?;
        Some(numerator.root(2))
    }
}
