use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use alloy::network::{Ethereum, ReceiptResponse};
use alloy::primitives::{Address, U256};
use alloy::providers::Provider;
use alloy::sol_types::SolCall;

use super::{
    BASE_COLLATERAL, BATCH_SWAP_ROUTER_ADDRESS, CTF_ROUTER_ADDRESS, IERC20, ITradeExecutor,
    MARKET_2_COLLATERAL, SWAP_ROUTER_ADDRESS,
};
use crate::markets::MARKETS_L1;

const MAX_APPROVE_AMOUNT: U256 = U256::MAX;
const DEFAULT_APPROVAL_CALLS_PER_TX: usize = 24;

#[derive(Debug, Clone)]
pub struct ApprovalRunSummary {
    pub checked_pairs: usize,
    pub missing_pairs: usize,
    pub sent_txs: usize,
}

#[derive(Debug)]
pub enum ApprovalError {
    InvalidOutcomeToken { market: &'static str, raw: String },
    Provider(String),
    FailedBatchExecute(String),
}

impl fmt::Display for ApprovalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutcomeToken { market, raw } => {
                write!(f, "invalid outcome token for market '{market}': {raw}")
            }
            Self::Provider(message) => write!(f, "provider error: {message}"),
            Self::FailedBatchExecute(message) => {
                write!(f, "approval batch execute failed: {message}")
            }
        }
    }
}

impl std::error::Error for ApprovalError {}

pub async fn ensure_executor_approvals<P>(
    provider: P,
    executor: Address,
) -> Result<ApprovalRunSummary, ApprovalError>
where
    P: Provider<Ethereum> + Clone,
{
    ensure_executor_approvals_with_chunk(provider, executor, DEFAULT_APPROVAL_CALLS_PER_TX).await
}

pub async fn ensure_executor_approvals_with_chunk<P>(
    provider: P,
    executor: Address,
    max_calls_per_tx: usize,
) -> Result<ApprovalRunSummary, ApprovalError>
where
    P: Provider<Ethereum> + Clone,
{
    let pairs = approval_pairs()?;
    let checked_pairs = pairs.len();

    let mut missing_calls = Vec::new();
    for (token, spender) in pairs {
        let token_contract = IERC20::new(token, provider.clone());
        let allowance = token_contract
            .allowance(executor, spender)
            .call()
            .await
            .map_err(|err| ApprovalError::Provider(err.to_string()))?;
        if allowance < MAX_APPROVE_AMOUNT {
            let calldata = IERC20::approveCall {
                spender,
                amount: MAX_APPROVE_AMOUNT,
            }
            .abi_encode();
            missing_calls.push(ITradeExecutor::Call {
                to: token,
                data: calldata.into(),
            });
        }
    }

    let missing_pairs = missing_calls.len();
    if missing_calls.is_empty() {
        return Ok(ApprovalRunSummary {
            checked_pairs,
            missing_pairs: 0,
            sent_txs: 0,
        });
    }

    let calls_per_tx = max_calls_per_tx.max(1);
    let executor_contract = ITradeExecutor::new(executor, provider.clone());
    let mut sent_txs = 0usize;
    for chunk in missing_calls.chunks(calls_per_tx) {
        let receipt = executor_contract
            .batchExecute(chunk.to_vec())
            .send()
            .await
            .map_err(|err| ApprovalError::Provider(err.to_string()))?
            .get_receipt()
            .await
            .map_err(|err| ApprovalError::Provider(err.to_string()))?;
        if !receipt.status() {
            return Err(ApprovalError::FailedBatchExecute(format!(
                "tx {} reverted",
                receipt.transaction_hash()
            )));
        }
        sent_txs += 1;
    }

    Ok(ApprovalRunSummary {
        checked_pairs,
        missing_pairs,
        sent_txs,
    })
}

fn approval_pairs() -> Result<Vec<(Address, Address)>, ApprovalError> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    // sUSDS -> CTF, SwapRouter, BatchSwapRouter
    for spender in [
        CTF_ROUTER_ADDRESS,
        SWAP_ROUTER_ADDRESS,
        BATCH_SWAP_ROUTER_ADDRESS,
    ] {
        push_pair(&mut out, &mut seen, BASE_COLLATERAL, spender);
    }

    // Market-2 collateral token -> CTF
    push_pair(&mut out, &mut seen, MARKET_2_COLLATERAL, CTF_ROUTER_ADDRESS);

    // Each outcome token -> CTF (merge/split burn/mint path).
    // Pooled outcomes additionally need SwapRouter + BatchSwapRouter approvals.
    for market in MARKETS_L1.iter() {
        if market.outcome_token.is_empty() {
            continue;
        }
        let token = Address::from_str(market.outcome_token).map_err(|_| {
            ApprovalError::InvalidOutcomeToken {
                market: market.name,
                raw: market.outcome_token.to_string(),
            }
        })?;
        push_pair(&mut out, &mut seen, token, CTF_ROUTER_ADDRESS);
        if market.pool.is_some() {
            push_pair(&mut out, &mut seen, token, SWAP_ROUTER_ADDRESS);
            push_pair(&mut out, &mut seen, token, BATCH_SWAP_ROUTER_ADDRESS);
        }
    }

    Ok(out)
}

fn push_pair(
    out: &mut Vec<(Address, Address)>,
    seen: &mut HashSet<(Address, Address)>,
    token: Address,
    spender: Address,
) {
    if seen.insert((token, spender)) {
        out.push((token, spender));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_pairs_include_ctf_for_all_outcomes() {
        let pairs = approval_pairs().expect("approval matrix should build");
        let set: HashSet<(Address, Address)> = pairs.into_iter().collect();

        for market in MARKETS_L1 {
            if market.outcome_token.is_empty() {
                continue;
            }
            let token =
                Address::from_str(market.outcome_token).expect("market outcome token must parse");
            assert!(
                set.contains(&(token, CTF_ROUTER_ADDRESS)),
                "missing CTF approval for market {}",
                market.name
            );
        }
    }

    #[test]
    fn swap_and_batch_approvals_remain_pooled_only() {
        let pairs = approval_pairs().expect("approval matrix should build");
        let set: HashSet<(Address, Address)> = pairs.into_iter().collect();

        for market in MARKETS_L1 {
            if market.outcome_token.is_empty() {
                continue;
            }
            let token =
                Address::from_str(market.outcome_token).expect("market outcome token must parse");
            let has_swap = set.contains(&(token, SWAP_ROUTER_ADDRESS));
            let has_batch = set.contains(&(token, BATCH_SWAP_ROUTER_ADDRESS));
            if market.pool.is_some() {
                assert!(has_swap, "pooled market {} missing swap approval", market.name);
                assert!(
                    has_batch,
                    "pooled market {} missing batch approval",
                    market.name
                );
            } else {
                assert!(
                    !has_swap,
                    "non-pooled market {} should not be swap-approved",
                    market.name
                );
                assert!(
                    !has_batch,
                    "non-pooled market {} should not be batch-approved",
                    market.name
                );
            }
        }
    }
}
