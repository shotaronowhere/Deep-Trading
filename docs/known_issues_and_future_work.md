# Known Issues and Future Work

Updated 2026-04-21 after the ForecastFlows E2E checklist follow-up work.

This file now separates resolved items from still-open follow-ups so future
solver work does not treat already-fixed regressions as active bugs.

## Resolved On 2026-04-21

### 1. `src/portfolio/tests/ev_snapshots.json` was stale

**Status.** Resolved.

**What changed.** After the planner oversell fix landed, the repo snapshot was
regenerated at [src/portfolio/tests/ev_snapshots.json](../src/portfolio/tests/ev_snapshots.json)
using the explicit refresh test in
[src/portfolio/tests/fuzz_rebalance.rs](../src/portfolio/tests/fuzz_rebalance.rs).

**Verification.**

```bash
cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_refresh_ev_snapshots_fixture' \
  -- --ignored --exact --nocapture --test-threads=1

cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_fuzz_rebalance_ev_regression_fast_suite' \
  -- --ignored --exact --nocapture --test-threads=1

cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_fuzz_rebalance_end_to_end_full_l1_invariants' \
  -- --ignored --exact --nocapture --test-threads=1
```

**Notes.** The refresh workflow is now documented in
[docs/ev_snapshot_regeneration.md](ev_snapshot_regeneration.md).

### 2. Planner over-sell bug in the monte-carlo full-profitability path

**Status.** Resolved.

**Root cause.** Two planner issues were contributing to non-replayable action
streams:

- baseline step pruning could drop action indices from a mutated action list,
  which let later prune passes orphan sub-actions
- the "consistent replay" helper was not actually rejecting infeasible action
  streams, so candidates that overspent cash or oversold holdings could still
  survive selection

**What changed.**

- [src/portfolio/core/rebalancer.rs](../src/portfolio/core/rebalancer.rs)
  now anchors step-prune filtering to the original consistent baseline action
  list and rejects infeasible buy/sell/mint/merge streams during consistent
  replay.
- Candidate builders now materialize `PlanResult`s from consistent replayed
  actions instead of mixing a replayed terminal state with stale raw actions.
- [src/portfolio/tests/monte_carlo.rs](../src/portfolio/tests/monte_carlo.rs)
  now includes deterministic regressions for:
  - the old `full_underpriced_baseline` replay path
  - the first previously failing fuzz-full trial and its seeded candidate
    layers

**Verification.**

```bash
cargo test --release test_full_underpriced_baseline_replays_without_sell_overshoot -- --nocapture

cargo test --release test_fuzz_full_trial_four_layers_replay_without_sell_overshoot -- --nocapture

MC_REQUIRE_FAMILY_COVERAGE=0 MC_TRIALS=20 \
  cargo test --release \
  'portfolio::core::tests::monte_carlo::test_monte_carlo_ev_full_profitability_groups' \
  -- --ignored --exact --nocapture --test-threads=1

MC_REQUIRE_FAMILY_COVERAGE=0 MC_START_TRIAL_INDEX=4 MC_TRIALS=20 \
  cargo test --release \
  'portfolio::core::tests::monte_carlo::test_monte_carlo_ev_full_profitability_groups' \
  -- --ignored --exact --nocapture --test-threads=1
```

**Why the default ignored test is not listed above.** The default
`test_monte_carlo_ev_full_profitability_groups` configuration uses
`MC_TRIALS=200000` and is a long-running soak. During this fix we verified the
original failure reproducer directly and then reran bounded sweeps around the
previously failing region after the fix.

## Still Open

### 1. `test_random_group_search_vs_waterfall_complex_fuzz_cases` runtime

**Status.** Still open, but not a correctness bug.

**Symptom.** The default ignored run takes about 20 minutes in release mode.

**Why it remains.** It is still the strongest independent
algorithm-vs-randomized-oracle cross-check we have for the hardest fuzz cases.

**Manual run.**

```bash
cargo test --release \
  'portfolio::core::tests::monte_carlo::test_random_group_search_vs_waterfall_complex_fuzz_cases' \
  -- --ignored --exact --nocapture --test-threads=1
```

**Faster exploratory run.**

```bash
RANDOM_SEARCH_MAX_ROLLOUTS=200 \
  RANDOM_SEARCH_MIN_RUNTIME_SECS=30 \
  cargo test --release \
  'portfolio::core::tests::monte_carlo::test_random_group_search_vs_waterfall_complex_fuzz_cases' \
  -- --ignored --exact --nocapture --test-threads=1
```

## Future Work

- **Structural-vs-replay near-tie auditor.** A small offline tool that scans
  candidate sets for structural-vs-replay near ties would let us size replay
  validation empirically instead of reacting after regressions.
- **Keep the snapshot refresh workflow current.** If the snapshot refresh test
  or output path changes, update
  [docs/ev_snapshot_regeneration.md](ev_snapshot_regeneration.md) in the same
  PR.
- **Foundry E2E coverage maintenance.** Keep the ForecastFlows executable test
  and benchmark matrix healthy as solver behavior changes so the harness does
  not silently regress again.
