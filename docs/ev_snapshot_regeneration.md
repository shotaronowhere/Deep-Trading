# EV Snapshot Regeneration

Use this procedure only after planner behavior changes are believed to be
correct. Do not refresh snapshots to hide a known bug.

## Snapshot File

- [src/portfolio/tests/ev_snapshots.json](../src/portfolio/tests/ev_snapshots.json)

## Refresh Command

```bash
cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_refresh_ev_snapshots_fixture' \
  -- --ignored --exact --nocapture --test-threads=1
```

This runs the repo's dedicated snapshot writer in
[src/portfolio/tests/fuzz_rebalance.rs](../src/portfolio/tests/fuzz_rebalance.rs)
and rewrites `src/portfolio/tests/ev_snapshots.json`.

## Required Verification

After refreshing, rerun both ignored regression consumers of the snapshot:

```bash
cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_fuzz_rebalance_ev_regression_fast_suite' \
  -- --ignored --exact --nocapture --test-threads=1

cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_fuzz_rebalance_end_to_end_full_l1_invariants' \
  -- --ignored --exact --nocapture --test-threads=1
```

## Review Checklist

- Confirm the snapshot diff is expected and tied to a known-good planner state.
- Confirm both ignored regression tests pass after the refresh.
- Include the snapshot refresh command and verification commands in the PR
  description so the change is auditable later.
