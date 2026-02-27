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

## Runtime Thresholds and Gates

Thresholds are computed per route group kind:

- `DirectBuy`
- `MintSell`
- `DirectSell`
- `BuyMerge`
- `DirectMerge`

### Waterfall candidate admission

Before admitting a non-active entry to the waterfall active set:

```
remaining_budget × profitability >= gas_cost_for_route
```

Direct entries use `direct_buy`; mint entries use `mint_sell`.

### Execution-aligned edge gate

Phase-1 liquidation, phase-3 recycling, and waterfall step execution all use the same gate shape:

```
edge_susd > gas_susd + max(buffer_min_susd, buffer_frac * edge_susd) + EPS
```

Default runtime buffer settings:

- `buffer_frac = 0.20`
- `buffer_min_susd = 0.25`

Fail-closed semantics:

- non-finite/negative edge: reject
- non-finite/negative gas estimate: reject
- non-finite thresholds from estimator are sanitized to `f64::INFINITY`

### API and pricing assumptions

Gas-aware APIs:

- `rebalance_with_gas(...)` (compatibility wrapper)
- `rebalance_with_gas_pricing(..., gas_price_eth, eth_usd)` (explicit pricing)

Defaults for the wrapper:

- `gas_price_eth = 1e-9`
- `eth_usd = 3000.0`

Runtime override:

- `ETH_USD` env var in `main` and `execute` binaries (default `3000`)

`main.rs` fetches L1 fee oracle data only in `RebalanceMode::Full`; `ArbOnly` uses defaults and ignores waterfall gas thresholds.

## Preconditions

All outcome tokens and sUSDS are infinite-approved to `BATCH_SWAP_ROUTER` and `CTF_ROUTER` prior to execution. No per-trade approval gas is incurred.
