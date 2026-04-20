# Known Issues and Future Work

Living index of pre-existing issues surfaced during the
[ForecastFlows replay-compile latency audit](ff_replay_compile_latency_report.md)
on 2026-04-20. None of these were introduced by the two-stage structural-first
ranking work; they were uncovered while running the full Category D ignored
assertion suite at the new `FINAL_REPLAY_VALIDATION_MAX_K = 16`.

Each entry lists: where it manifests, how it was reproduced, why we believe
it is pre-existing, and the proposed remediation.

## 1. Stale `src/portfolio/tests/ev_snapshots.json`

**Symptom.** Two ignored fuzz tests fail with case-0 EV mismatches well below
their snapshot floor tolerance:

- `test_fuzz_rebalance_ev_regression_fast_suite` — case 0 (full):
  got=327.66, expected=366.58, delta=-38.92, floor_tol=0.18.
- `test_fuzz_rebalance_end_to_end_full_l1_invariants` — same case-0 EV
  mismatch.

**Where.**

- Snapshot fixture: [src/portfolio/tests/ev_snapshots.json](../src/portfolio/tests/ev_snapshots.json)
- Assertion sites:
  [src/portfolio/tests/fuzz_rebalance.rs:578](../src/portfolio/tests/fuzz_rebalance.rs#L578)
  (fast_suite) and
  [src/portfolio/tests/fuzz_rebalance.rs:1239](../src/portfolio/tests/fuzz_rebalance.rs#L1239)
  (full_l1_invariants).

**Reproduction.**

```
cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_fuzz_rebalance_ev_regression_fast_suite' \
  -- --ignored --exact --nocapture --test-threads=1
```

**Why it is pre-existing.** The snapshot was last touched in commit `002d269`
on 2026-02-24 (~2 months before this audit). Several solver changes have
landed since. The failure also reproduces at `FINAL_REPLAY_VALIDATION_MAX_K =
1024`, confirming it is independent of the truncation knob.

**Proposed fix.** Regenerate the snapshot once any planner-side issues
(see #2 below) are resolved, so we do not encode known-bad behavior. Track
the regen as a follow-up PR; the snapshot's purpose is to flag *unintended*
EV drift, so it must be refreshed under a known-good planner state.

## 2. Planner over-sell bug in monte-carlo full-profitability path

**Symptom.** `test_monte_carlo_ev_full_profitability_groups` panics inside
`replay_actions_to_state`:

```
sell over-consumed holdings for argotorg/solidity: -716.7713258457056
```

**Where.**

- Assertion: [src/portfolio/tests.rs:1201](../src/portfolio/tests.rs#L1201)
  (`*bal >= -1e-6`)
- Test:
  [src/portfolio/tests/monte_carlo.rs](../src/portfolio/tests/monte_carlo.rs)
  — `test_monte_carlo_ev_full_profitability_groups`.

**Reproduction.**

```
cargo test --release \
  'portfolio::core::tests::monte_carlo::test_monte_carlo_ev_full_profitability_groups' \
  -- --ignored --exact --nocapture --test-threads=1
```

**Why it is pre-existing.** The same panic with the same -716.77 holdings
shortfall reproduces at `FINAL_REPLAY_VALIDATION_MAX_K = 1024`, so this is
not caused by the structural-first truncation. The planner is emitting a
sell action for ~716 more `argotorg/solidity` units than the simulated
holding contains.

**Likely cause.** The full-profitability group composition step is layering
sells from multiple sub-actions without de-duplicating against the running
holding balance. Needs a focused investigation in the group-action builder
that feeds this monte-carlo case.

**Proposed fix.** Reproduce in isolation, capture the offending action set,
trace which builder produced the over-sell, and fix the holding accounting.
Once fixed, regenerate the snapshot in #1.

## 3. `test_random_group_search_vs_waterfall_complex_fuzz_cases` runtime

**Status.** Not a bug — documented here so future maintainers do not mistake
its runtime for a regression.

**Symptom.** Test takes ~1201 s (~20 min) on release, exceeding the 5-minute
test wall-time policy. **Passes.**

**Where.** [src/portfolio/tests/monte_carlo.rs:1338-1429](../src/portfolio/tests/monte_carlo.rs#L1338-L1429).
Stays `#[ignore]`.

**Why it stays in the suite.** It is the most aggressive
algorithm-vs-randomized-oracle check we have for the rebalancer. It iterates
the four hardest fuzz cases and asserts the algorithm is never beaten by an
independent random group-action search. Knobs (`max_rollouts`,
`groups_per_rollout`, `min_runtime_secs`) are env-driven via
`RandomSearchConfig::from_env()` at
[src/portfolio/tests/monte_carlo.rs:660](../src/portfolio/tests/monte_carlo.rs#L660),
so CI/local runs can dial it down without code changes.

**When to run it.** Manually, before solver changes that could plausibly
affect the four hardest fuzz cases.

**Reproduction (default knobs, ~20 min).**

```
cargo test --release \
  'portfolio::core::tests::monte_carlo::test_random_group_search_vs_waterfall_complex_fuzz_cases' \
  -- --ignored --exact --nocapture --test-threads=1
```

**Faster manual run (env-tuned).**

```
RANDOM_SEARCH_MAX_ROLLOUTS=200 \
  RANDOM_SEARCH_MIN_RUNTIME_SECS=30 \
  cargo test --release \
  'portfolio::core::tests::monte_carlo::test_random_group_search_vs_waterfall_complex_fuzz_cases' \
  -- --ignored --exact --nocapture --test-threads=1
```

## Future work (not blocking)

- **Structural-vs-replay near-tie auditor.** The K=8 → K=16 fix was
  triggered by a 32_768-wei structural/replay disagreement on the
  heterogeneous case. A small offline tool that scans the candidate set for
  near-ties between structural ranking and replay-priced ranking would let
  us pick `FINAL_REPLAY_VALIDATION_MAX_K` empirically rather than bumping
  it reactively.
- **Snapshot regen workflow doc.** Once #2 is fixed and #1 is regenerated,
  document the regen procedure in this repo so the next planner change
  does not need to rediscover it.
- **Foundry E2E re-validation.** Phase 4 of the latency plan (Foundry E2E
  re-run post two-stage change) is still outstanding per
  [docs/ff_replay_compile_latency_report.md](ff_replay_compile_latency_report.md).
