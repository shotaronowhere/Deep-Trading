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
