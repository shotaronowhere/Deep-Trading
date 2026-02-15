# Rebalance Test EV Trace Contract

This document describes the shared test instrumentation used by portfolio rebalancing stress tests.

## Shared helper

`src/portfolio/tests.rs` defines:

- `compute_ev_trace(...)`
- `assert_strict_ev_gain_with_portfolio_trace(...)`

`compute_ev_trace(...)` performs pure state/EV calculation.
`assert_strict_ev_gain_with_portfolio_trace(...)` handles the required logging and strict EV assertion.

The helper is responsible for:

1. Printing the initial portfolio (`cash` and non-zero holdings).
2. Printing a trade summary (counts and aggregate amounts for buy/sell/mint/merge/flash-loan actions).
3. Printing the final portfolio (`cash` and non-zero holdings).
4. Computing and printing expected value before and after replay.
5. Enforcing a strict EV improvement assertion: `ev_after > ev_before`.

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
3. Prints per-market prices before rebalance and their sum.
4. Runs `rebalance(...)` on the live snapshot with a fixed budget.
5. Verifies deterministic output for repeated runs on the same snapshot.
6. Verifies action-stream invariants.
7. Replays actions, prints per-market prices after rebalance and their sum, and prints EV before/after/gain.
8. Asserts expected value is non-decreasing.

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
3. Prints per-market prices before arb-only rebalance and their sum.
4. Runs `rebalance_with_mode(..., RebalanceMode::ArbOnly)` on the live snapshot with a fixed budget.
5. Prints the full action list.
6. Verifies deterministic output for repeated runs on the same snapshot.
7. Verifies action-stream invariants.
8. Replays actions, prints per-market prices after rebalance and their sum, and prints EV before/after/gain.
9. Asserts expected value is non-decreasing.

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
