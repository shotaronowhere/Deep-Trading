# Local Foundry E2E Benchmark Matrix

Status: current as of 2026-04-21.

## Purpose

The benchmark matrix is the executable, local, on-chain cross-solver comparison that sits on top
of the Local Foundry E2E harness. It produces a row per `(scenario, solver)` pair with measured
gas, calldata, raw EV, modeled fee, and realized net EV after actually executing the plan through
`TradeExecutor.batchExecute` against locally deployed contracts.

Each scenario runs every solver against the **same post-setup state** using
`vm.snapshotState` / `vm.revertToState`, so the lanes are directly comparable.

## Entry Points

Two Foundry tests, each writing to its own JSONL + Markdown artifact pair:

- `test_benchmark_matrix_single_market`
  - 98 outcomes on a single root market
  - output: [test/fixtures/local_foundry_e2e_benchmark_matrix_single_market.jsonl](../test/fixtures/local_foundry_e2e_benchmark_matrix_single_market.jsonl)
  - markdown: `test/fixtures/local_foundry_e2e_benchmark_matrix_single_market.md`
- `test_benchmark_matrix_connected`
  - 67 root + 32 child outcomes (L1-like connected topology)
  - output: [test/fixtures/local_foundry_e2e_benchmark_matrix_connected.jsonl](../test/fixtures/local_foundry_e2e_benchmark_matrix_connected.jsonl)
  - markdown: `test/fixtures/local_foundry_e2e_benchmark_matrix_connected.md`

## Solver Lanes

Each scenario emits five rows, one per solver. Lanes that cannot run in a given topology emit a
skip row with a machine-readable `skip_reason` so the matrix remains shape-stable.

| lane | source | single_market | connected |
|---|---|---|---|
| `offchain_waterfall` | Rust native waterfall solver via FFI fixture | run | run |
| `offchain_forecastflows` | Rust FF worker (`FORECASTFLOWS_WORKER_BIN`) via FFI fixture | run | run; the benchmark harness launches this lane with `FORECASTFLOWS_REQUEST_PROFILE=benchmark` so the worker uses baseline tuning instead of the production low-latency profile |
| `onchain_rebalance_exact` | `Rebalancer.rebalanceExact` | run (may skip `onchain_revert_full_range_tick_scan` against full-range synthetic pools) | skip `onchain_single_market_only` |
| `onchain_rebalance_arb_direct` | `Rebalancer.rebalance` | run | skip `onchain_single_market_only` |
| `onchain_rebalance_mixed_constant_l` | `RebalancerMixed.rebalanceMixedConstantL` | run | skip `onchain_single_market_only` |

## Skip Reasons

| reason | meaning |
|---|---|
| `forecastflows_worker_bin_unset` | `FORECASTFLOWS_WORKER_BIN` env var not provided; FF lane cannot be exercised |
| `no_certified_candidate` (or another `ForecastFlowsError::fallback_reason` token) | The fixture binary refuses silent fallback and exits with `forecastflows fallback=<reason>; ...` on stderr. The harness extracts `<reason>` and emits it verbatim as `skip_reason`, so downstream readers see the real underlying reason (e.g. `no_certified_candidate`, `worker_closed`, `worker_cooldown`) instead of a flattened label |
| `forecastflows_fixture_exit` | The fixture binary exited non-zero but stderr did not contain a `fallback=<reason>;` token. This is a neutral fallback for compile breaks, malformed JSON, or other unexpected failures surfaced by `vm.tryFfi` |
| `executor_batch_execute_failed` | The fixture returned a plan, but replaying one of its chunks through `TradeExecutor` still failed locally. This is reserved for true executor-path failures after FFI succeeded |
| `onchain_single_market_only` | On-chain `Rebalancer` / `RebalancerMixed` APIs accept one Seer market; connected topology is out of scope for these contracts (tracked in `docs/local_foundry_e2e_harness_next_steps.md` §2) |
| `onchain_revert_full_range_tick_scan` | On-chain tick scan reverts against the synthetic full-range `[MIN_TICK, MAX_TICK]` pools used in the benchmark; scenario-specific limitation of the benchmark pool construction |

## Row Schema

Each JSONL line is a flat object with these fields (all integer EV/fee fields are decimal strings
of WAD-scale values):

- `case_id`, `topology`, `solver`
- `pre_raw_ev_wad`, `post_raw_ev_wad` (measured from `TradeExecutor`'s portfolio)
- `l2_gas_used`, `calldata_bytes` (summed across all chunks for off-chain lanes)
- `modeled_fee_wad` = `(gas * GAS_PRICE_WEI + calldata * L1_FEE_PER_BYTE_WEI) * ETH_USD`
- `realized_net_ev_wad` = `post_raw_ev_wad - pre_raw_ev_wad - modeled_fee_wad`
- `action_count`, `chunk_count` (from fixture output; 0 for on-chain/skip rows)
- `expected_raw_ev_wad`, `estimated_fee_wad`, `estimated_net_ev_wad` (fixture-reported estimates for off-chain lanes; 0 for on-chain/skip rows)
- `skip_reason` (empty string for executed rows)

## State Isolation

`_benchmarkSolversOnScenario` snapshots state after funding, approvals, and solver deployment.
Each solver runs from that snapshot and the state is reverted before the next lane, so identical
balances/approvals are observable across all lanes for the same scenario.

## FFI Failure Handling

`_runFixture` asserts on non-zero exit (happy path for the waterfall lane).
`_tryRunFixture` wraps `vm.tryFfi` and returns `(ok, fixture, stderr)`. The off-chain runner
calls the try variant so the FF lane can degrade to a skip row instead of aborting the suite when
the FF worker returns uncertified for a topology it cannot solve today.

For the benchmark FF lane, the harness invokes the fixture binary through:

```bash
env FORECASTFLOWS_REQUEST_PROFILE=benchmark cargo run --release --quiet --bin local_foundry_e2e_fixture ...
```

That forces the fixture's ForecastFlows call onto the benchmark profile
(`baseline` solve tuning plus the warmup-style timeout) while leaving the
non-benchmark executable FF tests on the production low-latency profile.

When the FF lane fails, `_runOffchainLane` parses the captured `stderr` via
`_extractForecastflowsFallbackReason`, which extracts the `fallback=<reason>;` token emitted by
the fixture binary (`src/bin/local_foundry_e2e_fixture.rs`) and propagates `<reason>` as
`skip_reason`. If no token is present, the row falls back to `forecastflows_fixture_exit`.

If the FF fixture succeeds but the benchmark lane still cannot execute the
returned chunk list through `TradeExecutor`, the harness emits
`executor_batch_execute_failed` instead of crashing the whole matrix.

For the benchmark ForecastFlows lane specifically, the fixture also zeroes the
per-leg `sqrtPriceLimitX96` bounds and compiles the plan in strict chunking
mode. This keeps the deterministic local benchmark focused on executable
solver output and real realized EV instead of benchmark-only slippage-limit
tuning artifacts.

## Running the Benchmark

Use release builds for the fixture binary; the harness compiles the fixture with
`--release` and runs the production translator, matching the binary path used in
non-benchmark scenarios.

The benchmark scenarios (`bench_single_market_98`, `bench_connected_98`) deploy Uniswap V3
pools with ticks narrowed to ±100,000 around the active tick rather than the full usable
range. That keeps the WAD-scaled sqrt-price bounds within `u128`, so the production
translator's `interval_price_bounds` / `build_univ3_liquidity_bands` path accepts the
geometry without the `benchmark_synthetic_fixtures` fallback.

The benchmark tests pause Foundry gas metering around scenario setup,
snapshot/revert, and artifact writes, then resume it only around the actual
lane execution calls. Row gas numbers therefore continue to come from explicit
`gasleft()` measurement of the real execution path, while the top-level Forge
test no longer hits the whole-test gas ceiling before all five rows are written.

```bash
# Single-market benchmark (no FF worker required)
forge test --ffi --match-test test_benchmark_matrix_single_market -vv

# Both off-chain lanes, single market
FORECASTFLOWS_WORKER_BIN=/absolute/path/to/forecast-flows-worker \
  forge test --ffi --match-test test_benchmark_matrix_single_market -vv

# Connected topology
FORECASTFLOWS_WORKER_BIN=/absolute/path/to/forecast-flows-worker \
  forge test --ffi --match-test test_benchmark_matrix_connected -vv

# Direct connected FF fixture repro under the same benchmark profile
FORECASTFLOWS_REQUEST_PROFILE=benchmark \
FORECASTFLOWS_WORKER_BIN=/absolute/path/to/forecast-flows-worker \
  cargo run --release --quiet --bin local_foundry_e2e_fixture \
  test/fixtures/local_foundry_e2e_fixture_input_bench_connected_98_forecastflows.json
```

When `FORECASTFLOWS_WORKER_BIN` is not set the `offchain_forecastflows` lane is
emitted as a skip row rather than being omitted, so the output JSONL always
has a consistent five rows per scenario.

## Known Limitations

- The single-market 98-outcome benchmark now executes on the off-chain
  ForecastFlows lane, but it does so as 29 strict chunks with zeroed benchmark
  price limits. That is an intentional benchmark-harness choice, not a claim
  that the same 533-action plan would fit under one packed 40M-gas envelope.
- The on-chain exact solver can still revert on the benchmark pool geometry;
  this is emitted as `onchain_revert_full_range_tick_scan` rather than treated
  as a solver regression.
- The connected topology is currently only exercised on off-chain lanes because the deployed
  `Rebalancer` / `RebalancerMixed` APIs are single-market. A connected on-chain benchmark is
  tracked in `docs/local_foundry_e2e_harness_next_steps.md` §2.
- Connected FF requests send an explicit conservative `split_bound` based on
  base collateral because the flattened benchmark problem omits connector /
  invalid inventory. Without that override, the worker's auto-bound can drive
  the mixed solve into the non-finite never-certified path on this topology.
- The benchmark harness deliberately uses the `benchmark` ForecastFlows profile
  for this lane. The production `low_latency` profile can still downgrade the
  connected 67+32 request to uncertified after `max_doublings=0`; the
  hard-failing executable FF tests remain the place where production-profile
  behavior is exercised.

## Verification

```bash
# Fast single-market lane (no FF worker needed)
forge test --ffi --match-test test_benchmark_matrix_single_market -vv

# Full matrix with FF worker
FORECASTFLOWS_WORKER_BIN=/absolute/path/to/forecast-flows-worker \
  forge test --ffi --match-path test/LocalFoundryExecutableTxE2E.t.sol -vv
```

Both benchmark tests must pass and produce exactly five JSONL rows per scenario. Any lane that
cannot run must emit a skip row with one of the documented `skip_reason` values.
