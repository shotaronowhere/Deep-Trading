# Portfolio Rebalancing

Canonical off-chain algorithm behavior lives in `docs/waterfall.md`.

This document is the portfolio module reference: APIs, execution shapes, diagnostics, and test map.

## Scope

Primary implementation paths:

- `src/portfolio/core/rebalancer.rs`
- `src/portfolio/core/waterfall.rs`
- `src/portfolio/core/planning.rs`
- `src/portfolio/core/solver.rs`
- `src/portfolio/core/trading.rs`
- `src/portfolio/core/sim.rs`

## Public entry points

```rust
pub fn rebalance(...) -> Vec<Action>
pub fn rebalance_with_mode(..., mode: RebalanceMode) -> Vec<Action>
pub fn rebalance_with_gas(..., mode: RebalanceMode, gas: &GasAssumptions) -> Vec<Action>
pub fn rebalance_with_gas_pricing(
    ...,
    mode: RebalanceMode,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
) -> Vec<Action>
```

Mode summary:

- `RebalanceMode::Full`: prediction-driven rebalance flow.
- `RebalanceMode::ArbOnly`: complete-set arb only.

Arb-only behavior is specified in `docs/waterfall.md` under `Arb-Only Mode`.

## Action model

```rust
enum Action {
    Mint { contract_1, contract_2, amount, target_market },
    Merge { contract_1, contract_2, amount, source_market },
    Buy { market_name, amount, cost },
    Sell { market_name, amount, proceeds },
}
```

Route/action shapes emitted by planner:

- `DirectBuy`
- `DirectSell`
- `DirectMerge`
- `MintSell` (`Mint -> Sell+`)
- `BuyMerge` (`Buy+ -> Merge`)

No flash-loan actions are emitted.

## Runtime inputs

Planner inputs are built from live on-chain state and optional wallet balances:

1. `slot0` snapshots (`src/pools/rpc.rs`)
2. wallet balances if `WALLET` is set, otherwise synthetic starting cash
3. gas assumptions (`src/execution/gas.rs`)
4. mode (`full` or `arb_only`)

## Off-Chain Full Solver

Default full-mode planning is now operator-based:

- `R_exact(state)` is the exact online rebalance operator over the chosen discrete block:
  - first-frontier family in `{default, direct, mint}`
  - preserve subset over a capped churn universe
- `A(state)` is the existing arb-only operator on the current state.
- The runtime evaluates two bounded whole-plan families:
  - `Plain`: `R_exact`
  - `ArbPrimed`: positive `A`, then `R_exact`
- Each chosen action plan is then compiled into an execution program:
  - `Packed`: greedily pack consecutive strict subgroups into tx chunks under the `40_000_000` L2-gas cap
  - `Strict`: one tx per strict subgroup
- Candidate ranking is estimated net EV first, then raw EV, then fewer tx chunks, then fewer actions, then stable family/frontier/preserve ordering.

`R_exact` keeps the existing waterfall, recycle, polish, and cleanup logic unchanged. The exactness is only over the online discrete block around that continuous core.

Preserve-universe construction:

- start from the three no-preserve frontier seeds (`default`, forced `direct`, forced `mint`)
- extract sell-then-rebuy churn candidates from those seed action streams
- run one-step singleton-preserve probes from the same state to expand that churn universe once
- aggregate by max churn amount, then max sold amount, then stable market order
- cap the online preserve universe at `K = 4`
- enumerate every preserve subset across every frontier family from fresh state
- evaluate that grid in parallel, then reduce deterministically

Runtime pruning result:

- the tiny teacher-distilled preserve/frontier proposal layer remains available in test/diagnostic helpers, but it was removed from the default runtime path after gas-aware ablation
- the late arb-correction tail was also removed from the default runtime path
- both were non-regressive, but neither earned enough committed-benchmark EV or net-EV improvement to justify default-path complexity

Execution-program compilation:

- the runtime no longer prices the discovered trace as one transaction per strict subgroup
- exact no-arb candidates are no longer compacted only by deleting profitability steps
- instead the runtime compares:
  - `baseline_step_prune`
  - `target_delta` re-emission from the rich terminal holdings
  - `analytic_mixed` compact common-shift solver
  - `coupled_mixed` continuous mixed-frontier compiler
  - `direct_only` compact no-mint/no-merge guard
  - `noop`
- instead it compiles that trace into the cheaper of:
  - packed chunked execution
  - strict subgroup execution
- only consecutive strict subgroups are packed in v1; there is no reordering or algebraic fusion

Reference fallback:

- the old staged meta-solver path is still compiled as a reference implementation
- it is now opt-in via `REBALANCE_ENABLE_STAGED_FALLBACK=1`
- the default runtime hot path is the packed operator solver, not staged fallback selection
- the realistic heterogeneous 98-outcome benchmark now clears the on-chain net-EV references under the shared snapshot after `analytic_mixed` is enabled
- the remaining mixed-route favorable synthetic gap is now treated as a true compact target-discovery gap, not an execution-fragmentation gap
- `coupled_mixed` is now part of the default compiler set, but it did not displace the selected benchmark winners; that is the current stop signal for further online solver expansion

Compatibility note:

- `RebalanceFlags.enable_ev_guarded_greedy_churn_pruning` remains in the public API for compatibility, but it no longer changes default full-mode behavior

Operational diagnostics:

- exact rebalance tracing reports family label, exact-rebalance call count, candidate-evaluation count, preserve-universe size, chosen frontier family, chosen preserve-set size, EV, and action count
- final selection tracing reports chosen family, chosen frontier family, preserve-set size, EV/actions, estimated tx count, arb-operator evaluation count, and whether the arb-primed root was taken
- exact first-group gas tracing on the execution path now reports:
  - unsigned `batchExecute` tx bytes
  - live OP gas price
  - exact `getL1Fee(txData)`
  - calibrated L2 fee and net EV after gas
- ignored test helpers print:
  - machine-readable teacher snapshots for benchmark and seeded hard cases
  - benchmark-layer gas-aware ablation rows
  - seeded hard-case gas-aware ablation rows
  - per-family breakdown on the heterogeneous 98-outcome fixture

## Diagnostics and operator tools

- Execution-program packing and chunked submission: `docs/execution_program_packing.md`
- Execution submission and strict fail-closed gates: `docs/execution_submission.md`
- First-group preview diagnostics (`plan_preview` + runtime preview output): `docs/execution_submission.md`
- Cross-approach policy thresholds: `docs/rebalancer_approaches_playbook.md`
- Off-chain EV optimization memory and keep/cut decisions: `docs/offchain_ev_optimization_log.md`
- Central release-facing solver benchmark table: `docs/solver_benchmark_matrix.md`

## Balance fetching and local cache

Balance helpers live in `src/pools/cache.rs` and `src/pools/rpc.rs`.

- `fetch_balances(...)` fetches sUSD + outcome balances via multicall
- `save_balance_cache(...)` writes wallet-scoped cached snapshot
- `load_balance_cache(...)` reads cache with staleness/wallet checks

## Test map

Portfolio tests are split across:

- `src/portfolio/tests.rs` (fixtures + early deterministic coverage)
- `src/portfolio/tests/fuzz_rebalance.rs`
- `src/portfolio/tests/oracle.rs`
- `src/portfolio/tests/execution.rs`

Key classes covered:

- direct-only oracle parity and near-optimality
- mixed-route feasibility and ordering behavior
- EV non-regression fuzz checks
- execution/replay consistency invariants

## Performance

`test_rebalance_perf_full_l1` benchmarks full rebalance across the 98 tradeable L1 outcomes.

Current release measurements after the packed-execution change are not yet rerun in a stable release harness.

- default packed path: `n/a`
- staged-reference opt-in path (`REBALANCE_ENABLE_STAGED_FALLBACK=1`): `n/a`

See `docs/solver_benchmark_matrix.md` for the current benchmark-facing economics table. Runtime speed should be treated as pending until the release perf harness is rerun on the packed default path.

For the release-facing EV / gas / speed comparison matrix across all solver flavors, see `docs/solver_benchmark_matrix.md`.

Run with:

```bash
cargo test --release test_rebalance_perf_full_l1 -- --nocapture
```

## Related docs

- `docs/waterfall.md`: canonical off-chain algorithm spec
- `docs/model.md`: math derivations
- `docs/gas_model.md`: gas threshold model
- `docs/rebalancer.md`: on-chain direct solver
