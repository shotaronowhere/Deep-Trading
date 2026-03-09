# Gas Model

## Swap Fees (Uniswap V3)

The market pools use a **100-pip (0.01%) fee tier**. This fee is baked into `FEE_FACTOR = 1 - 100/1_000_000 = 0.9999` in `PoolSim` (`src/portfolio/core/sim.rs`). All buy/sell math is fee-inclusive; no separate swap-fee deduction is needed.

## Two-Source Model

Current gas-aware diagnostics use two sources:

- L2 execution gas units: calibrated from Foundry `TradeExecutor.batchExecute(...)` gas profiles.
- L1 data fee: exact `GasPriceOracle.getL1Fee(txData)` on the unsigned `batchExecute` transaction bytes built from the existing execution path.

The planner now ranks candidates by estimated net EV:

- `estimated_net_ev = raw_ev - estimated_execution_cost`
- execution cost includes only Optimism L2 execution gas and Optimism L1 data fee
- swap / pool fees are already embedded in the raw EV math and are not subtracted again

Runtime ranking uses:

- calibrated L2 gas units
- exact local unsigned tx shape for the chosen execution program
- heuristic `l1_fee_per_byte_wei`
- one shared pricing snapshot per solver run

Execution-program pricing is now chunk-aware:

- `Strict` prices one tx per strict subgroup
- `Packed` prices one tx per packed chunk of consecutive strict subgroups
- each chunk is capped at `< 40_000_000` estimated L2 gas
- L1 data cost is charged once per chunk tx envelope, not once per subgroup inside the chunk

Discovery thresholds are intentionally lighter than final pricing:

- route admission uses incremental per-group gas
- final ranking uses packed-program net EV

This prevents the planner from baking the fragmented execution topology into the objective before packing has a chance to compress the plan.

Exact live `getL1Fee(txData)` quoting remains the diagnostic and benchmark truth path, not the planner hot-loop path.
The current heuristic fallback floor is `0.001` sUSD, recalibrated from exact live OP quotes on canonical small execution shapes.

Benchmark-mode deterministic modeling is stricter:

- benchmark tests and modeled benchmark snapshot helpers pin `l1_fee_per_byte_wei = 1,643,855.3414634147`
- benchmark tests set `l1_data_fee_floor_susd = 0.0`
- this avoids cache / RPC drift in net-EV benchmark selection while keeping runtime discovery conservative

## L2 Gas Estimates

| Group kind           | L2 gas (est.)              | Calldata (est.)      |
|----------------------|---------------------------|----------------------|
| DirectBuy            | 57,542                    | 434 bytes            |
| DirectSell           | 38,099                    | 434 bytes            |
| DirectMerge          | 21,502                    | 330 bytes            |
| MintSell (N legs)    | 17,783 + N × 50,649       | 310 + N × 224 bytes  |
| BuyMerge (N legs)    | 37,370 + N × 29,670       | 310 + N × 224 bytes  |

At typical Optimism gas prices (0.001–0.01 gwei, $3000/ETH):
- DirectBuy: ~$0.00017–$0.0017 L2
- MintSell (97 legs): ~$0.015–$0.15 L2

These constants were refreshed from the deterministic Foundry suite in `test/TradeExecutorGasProfile.t.sol`.

Measured profile points:

| Shape | Measured gas |
|---|---:|
| `DirectBuy` | `57,542` |
| `DirectSell` | `38,099` |
| `DirectMerge` | `21,502` |
| `MintSell(1)` | `67,754` |
| `MintSell(5)` | `272,425` |
| `MintSell(20)` | `1,029,936` |
| `MintSell(97)` | `4,930,833` |
| `BuyMerge(1)` | `68,055` |
| `BuyMerge(5)` | `186,066` |
| `BuyMerge(20)` | `629,103` |
| `BuyMerge(97)` | `2,915,715` |

## L1 Data Fee

There are now two paths:

- heuristic fallback: cached marginal `fee_per_byte` from `getL1Fee()` probes, used only when exact tx-data quoting is unavailable
- exact diagnostic path: `getL1Fee(unsigned_batchExecute_tx_bytes)` on the real unsigned calldata

Cache TTL for the heuristic fallback remains 60 seconds. The exact path does not use the byte-slope approximation.

**Probe bytes:** A deterministic LCG (linear congruential generator, seed `0xDEADBEEF`) produces high-entropy non-repeating bytes for each probe payload. This is critical: under the Fjord upgrade, Optimism uses Brotli compression for L1 data. Uniform repeated bytes (e.g. `0x00`, `0x01`, or any single value) compress to near-zero regardless of length, collapsing the slope to ~0 and severely under-estimating L1 fees. Real calldata (addresses, swap amounts) has high entropy and compresses poorly — the LCG probe approximates this.

### Calldata constants

- `SWAP_BYTES = 224`: ABI payload for `exactInputSingle` excluding selector (7 × 32 = 224). The selector is part of `BATCH_CALL_BASE_BYTES`.
- `FLASH_ROUTE_EXTRA_BYTES = 160`: `BatchSwapRouter` flash route overhead. CTF `splitPosition`/`mergePositions` are **internal L2 calls** and do not appear in EOA calldata — no L1 data cost for them.

### Live OP exact-fee example

Measured via `cargo test print_live_op_first_group_exact_gas_report -- --ignored --nocapture` on 2026-03-08:

- `eth_gasPrice = 1,006,543 wei`
- first `DirectBuy` subgroup unsigned tx bytes: `492`
- exact `getL1Fee(txData) = 808,776,828 wei`
- calibrated L2 gas units: `57,542`
- live `eth_estimateGas = 29,318`
- exact total fee on that subgroup: about `$0.00017618`

This is why the exact-fee path exists: the older conservative defaults significantly overstated live OP cost on simple first-group routes.

### Canonical small-shape calibration

Measured via `cargo test print_live_op_canonical_small_shape_l1_fee_floor_calibration -- --ignored --nocapture` on 2026-03-08:

| Shape | Exact total fee (sUSD) |
|---|---:|
| `DirectBuy` | `0.000176154997284` |
| `DirectSell` | `0.00011726016444` |
| `DirectMerge` | `0.000067578070464` |
| `MintSell(1)` | `0.000210493567404` |
| `BuyMerge(1)` | `0.000206453150568` |

The current fallback floor comes from `2 x max(small-shape exact total fee)`, rounded up to the nearest `0.001` sUSD, which yields `0.001`.

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
