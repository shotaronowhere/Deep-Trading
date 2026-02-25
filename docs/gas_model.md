# Gas Model

## Swap Fees (Uniswap V3)

The market pools use a **100-pip (0.01%) fee tier**. This fee is baked into `FEE_FACTOR = 1 - 100/1_000_000 = 0.9999` in `PoolSim` (`src/portfolio/core/sim.rs`). All buy/sell math is fee-inclusive; no separate swap-fee deduction is needed.

## L2 Gas Estimates

| Group kind           | L2 gas (est.)              | Calldata (est.)      |
|----------------------|---------------------------|----------------------|
| DirectBuy            | 220,000                   | 434 bytes            |
| DirectSell           | 200,000                   | 434 bytes            |
| DirectMerge          | 150,000                   | 330 bytes            |
| MintSell (N legs)    | 550,000 + N × 170,000     | 310 + N × 224 bytes  |
| BuyMerge (N legs)    | 500,000 + N × 180,000     | 310 + N × 224 bytes  |

At typical Optimism gas prices (0.001–0.01 gwei, $3000/ETH):
- DirectBuy: ~$0.001–$0.01 L2
- MintSell (97 legs): ~$0.05–$0.50 L2

## L1 Data Fee

Estimated by probing the `GasPriceOracle.getL1Fee()` on-chain contract with two different payload sizes (256 and 512 bytes), measuring the marginal fee per byte (Fjord/Brotli model). Cache TTL: 60 seconds. Floor: $0.10 sUSD.

**Probe bytes:** A deterministic LCG (linear congruential generator, seed `0xDEADBEEF`) produces high-entropy non-repeating bytes for each probe payload. This is critical: under the Fjord upgrade, Optimism uses Brotli compression for L1 data. Uniform repeated bytes (e.g. `0x00`, `0x01`, or any single value) compress to near-zero regardless of length, collapsing the slope to ~0 and severely under-estimating L1 fees. Real calldata (addresses, swap amounts) has high entropy and compresses poorly — the LCG probe approximates this.

### Calldata constants

- `SWAP_BYTES = 224`: ABI payload for `exactInputSingle` excluding selector (7 × 32 = 224). The selector is part of `BATCH_CALL_BASE_BYTES`.
- `FLASH_ROUTE_EXTRA_BYTES = 160`: `BatchSwapRouter` flash route overhead. CTF `splitPosition`/`mergePositions` are **internal L2 calls** and do not appear in EOA calldata — no L1 data cost for them.

## Dual-Market CTF Operations

- **Mint**: `splitPosition(sUSDS, market1)` → `splitPosition(OTHER_REPOS_TOKEN, market2)`
- **Merge**: `mergePositions(OTHER_REPOS_TOKEN, market2)` → `mergePositions(sUSDS, market1)`

The second call uses the "Other repos" token (outcome `0x63a4...`) as collateral, not sUSDS. This token is excluded from trading and sim construction — it has `pool: None` in `MARKETS_L1` and must never appear as a swap leg.

## Minimum Trade Threshold (Waterfall)

Before admitting an outcome to the active set, `best_non_active()` checks:

```
remaining_budget × profitability >= gas_cost
```

Equivalently: `remaining_budget >= gas_cost / profitability`.

This ensures the maximum possible profit from the remaining budget (if all of it went to this outcome) at least covers the gas cost. The threshold uses worst-case gas (`MintSell` with `sims.len() - 1` sell legs) for the mint route.

Gas thresholds are computed once per rebalance call from `GasAssumptions` using conservative defaults (1 gwei L2, $3000/ETH). The `rebalance_with_mode` function (used by all tests) passes `0.0, 0.0` thresholds (no filtering). The `rebalance_with_gas` entry point (used by `main.rs`) passes real thresholds.

`main.rs` only fetches the L1 fee oracle when `REBALANCE_MODE=full` (the default). In `ArbOnly` mode the fetch is skipped entirely and `GasAssumptions::default()` is used — which is harmless since `ArbOnly` ignores gas thresholds regardless.

**Fail-closed:** If `estimate_min_gas_susd_for_group` returns non-finite (e.g. due to missing L1 fee data), the threshold is set to `f64::INFINITY`, blocking all trades rather than silently allowing them.

## Preconditions

All outcome tokens and sUSDS are infinite-approved to `BATCH_SWAP_ROUTER` and `CTF_ROUTER` prior to execution. No per-trade approval gas is incurred.
