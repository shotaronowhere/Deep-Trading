# Rebalance Test EV Trace Contract

This document describes the shared test instrumentation used by portfolio rebalancing stress tests.

## Shared helper

`src/portfolio/tests.rs` defines:

- `compute_ev_trace(...)`
- `assert_strict_ev_gain_with_portfolio_trace(...)`

`compute_ev_trace(...)` performs pure state/EV calculation.
`assert_strict_ev_gain_with_portfolio_trace(...)` handles the required logging and strict EV assertion.

Formatting internals now live in `src/portfolio/core/diagnostics.rs` and are called from thin wrappers in `src/portfolio/tests.rs`.
Runtime summary replay in diagnostics is fail-soft: if a snapshot cannot be mapped to sims, logging degrades to an "unavailable" message instead of panicking.
`replay_actions_to_portfolio_state(...)` now lives in diagnostics as a shared replay helper used by both tests and runtime output paths.

The diagnostics module is split into small reusable abstractions:

- `ActionSetKind` / `ActionSetSpec`: ordered, typed action-set definitions (`Buy`, `Sell`, `Flash Loan`, `Mint`, `Merge`, `Repay`) used to build mixed-step previews without ad-hoc per-branch vector wiring.
- `ActionSetView` (`Display`): trait-based renderer for action sets, covering both stage snapshots and homogeneous first/last previews.
- `GroupPreviewRow` (`Display`): trait-based renderer for the compact group table row (`group/kind/count/indices/badges`).

The helper is responsible for:

1. Printing the initial portfolio (`cash` and non-zero holdings).
2. Printing a trade summary. If a complete-set arb prefix is present (`complete_set_arb` marker), it prints separate `arb_phase`, `non_arb_phase`, and `total` summaries; otherwise it prints a single summary.
3. For large action streams (`>= 20` actions), printing a compact action-group preview: arb phase (if present), then first 2 post-arb groups, a skipped-group rollup line (`... skipped N groups (...)`), then last 2 post-arb groups. The preview uses stable columns (`group`, `kind`, `count`, `indices`, `badges`) with short kind labels (`Arb:MintSell`, `Arb:BuyMerge`, `Mixed:Buy+MintSell`, `Mixed:Sell+BuyMerge`) to preserve vertical scan alignment. Boundary badges are explicit about unknown replay coverage (`[BOUNDARY h/k known, u unknown]`) instead of silently showing `0/x`. Mixed groups print typed stage outlines (`Buy`, `Flash Loan`, `Mint`, `Sell`, `Merge`, `Repay`) and skip duplicate generic first/last rows; multi-item stage rows are split into compact `count` and `first/last` lines to avoid extra-wide log lines. Non-mixed groups keep concise first/last rows (`index + action type + normalized detail`). Flash subgroup output is also compacted: if many subgroups exist, it prints first 2, a skipped-subgroup rollup, and the last subgroup.
4. Printing market price deltas and market price sums. By default, deltas are compact and direction-balanced: it shows top movers split into `top up movers` and `top down movers` (with optional flat rows and redistribution when one side has fewer entries), plus an omitted-row rollup and total up/down/flat counts; set `REBALANCE_FULL_MARKET_DELTAS=1` to print the full per-market list. Delta lines remain color-coded by direction (green up / red down / gray flat) when stdout is a terminal.
5. Printing the final portfolio (`cash` and non-zero holdings). By default this is compacted to top 20 positions by `|units|` with an omitted-count rollup; set `REBALANCE_FULL_PORTFOLIO=1` to print all non-zero positions.
6. Computing and printing expected value before and after replay.
7. Enforcing a strict EV improvement assertion: `ev_after > ev_before`.

## Tests using the helper

The helper is wired into the stress/optimization rebalance tests:

- `src/portfolio/tests/fuzz_rebalance.rs`
  - `test_fuzz_rebalance_end_to_end_full_l1_invariants`
  - `test_fuzz_rebalance_end_to_end_partial_l1_invariants`
  - `test_rebalance_regression_full_l1_snapshot_invariants`
  - `test_rebalance_regression_full_l1_snapshot_variant_b_invariants`
- `src/portfolio/tests/execution.rs`
  - `test_rebalance_perf_full_l1`

## Verification

After integration, these tests were run with `--nocapture` to confirm:

- initial portfolio logs are emitted,
- trade summaries are emitted,
- final portfolio logs are emitted,
- EV before/after/gain is printed,
- strict EV improvement assertions hold.

## Live full-L1 optimization test

`src/portfolio/tests/execution.rs` now includes:

- `test_rebalance_optimization_full_l1_live_prices`

This is a live-network integration test that:

1. Fetches live `slot0` for all L1 pools.
2. Filters to tradeable L1 outcomes with predictions and asserts full coverage.
3. Runs `rebalance(...)` on the live snapshot with a fixed budget.
4. Verifies deterministic output for repeated runs on the same snapshot.
5. Verifies action-stream invariants.
6. Prints the shared execution summary (phase-aware trade summaries + compact action-group preview for large streams + colored per-market price deltas + price sums).
7. Prints EV before/after/gain and asserts expected value is non-decreasing.

RPC behavior:

- Uses `RPC` from environment first if set.
- Falls back to `https://optimism.drpc.org`.
- If neither endpoint is reachable, prints a skip message and returns without failing.

Run:

```bash
cargo test test_rebalance_optimization_full_l1_live_prices -- --nocapture
```

Optional override:

```bash
RPC=<your_rpc_url> cargo test test_rebalance_optimization_full_l1_live_prices -- --nocapture
```

## Live full-L1 arb-only test

`src/portfolio/tests/execution.rs` now includes:

- `test_rebalance_arb_only_full_l1_live_prices`

This is a live-network integration test that:

1. Fetches live `slot0` for all L1 pools.
2. Filters to tradeable L1 outcomes with predictions and asserts full coverage.
3. Runs `rebalance_with_mode(..., RebalanceMode::ArbOnly)` on the live snapshot with a fixed budget.
4. Prints action count and then the shared summary's compact action-group preview for large streams (instead of dumping every action).
5. Verifies deterministic output for repeated runs on the same snapshot.
6. Verifies action-stream invariants.
7. Prints the shared execution summary (phase-aware trade summaries + compact action-group preview for large streams + colored per-market price deltas + price sums).
8. Asserts expected value is non-decreasing.

RPC behavior:

- Uses `RPC` from environment first if set.
- Falls back to `https://optimism.drpc.org`.
- If neither endpoint is reachable, prints a skip message and returns without failing.

Run:

```bash
cargo test test_rebalance_arb_only_full_l1_live_prices -- --nocapture
```

Optional override:

```bash
RPC=<your_rpc_url> cargo test test_rebalance_arb_only_full_l1_live_prices -- --nocapture
```
