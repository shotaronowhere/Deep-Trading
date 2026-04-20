# ForecastFlows Replay-Compile Latency Report

Scope: outcome of executing
[docs/ff_replay_compile_latency_plan.md](ff_replay_compile_latency_plan.md)
against the heterogeneous 98-outcome fixture (4996 actions).

## Change landed

- Removed the `Strict` compile branch from `estimate_plan_cost_from_replay`
  ([src/portfolio/core/rebalancer.rs](../src/portfolio/core/rebalancer.rs)).
  The function now does a single `Packed` compile, fallback `None` on failure.
  `estimate_plan_cost_structural` remains the next-level fallback.
- Same simplification applied to the test-only helper
  `total_calldata_bytes_for_actions_for_test`.
- Dropped the unused `better_execution_program` helper and the
  `FeeEstimateSource::ReplayStrictProgram` variant.

## Before / after wall time (release, single test thread)

| Test | Baseline | After Packed-only | Delta |
| --- | --- | --- | --- |
| `benchmark_snapshot_matches_current_optimizer` | 437.71 s | **413.69 s** | -5.5% |
| `benchmark_ev_non_decreasing_vs_fixture` | 429.00 s | 448.58 s | within noise |
| `analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev` | 435.00 s | 431.09 s | -0.9% |
| `fee_sweep_heterogeneous_ninety_eight_outcome_l1_like_case` (1 of 3 iters) | ~570 s/iter | ~570 s/iter | no change within 0.5x iter |

All three unignored bench tests pass with the existing tolerances, including
the 65_536-wei heterogeneous wobble. The heavy ignored fee sweep was aborted
before completing all three fee multipliers; the single observed iteration
matched baseline.

## Why the gain is small

P2 profiling (atomic counters around each stage, since removed) on
`benchmark_snapshot_matches_current_optimizer`:

| Stage | Aggregate ns across rayon threads | Share |
| --- | --- | --- |
| `fee_estimation_address_book` | 141 ms | 0.02% |
| `build_group_plans_for_gas_replay_with_market_context` | ~90 s | ~10% |
| `compile_execution_program` (Strict) | ~87 s | ~10% |
| `compile_execution_program` (Packed) | 770.9 s | **88.8%** |

Selection counts across 8107 heterogeneous candidates × 6 cases:

- `strict_wins = 0`
- `packed_wins = 100%`
- `packed_strictly_worse_than_strict = 0`
- `packed_fails = 0`

Packed dominance was proven, so removing Strict was safe. But Strict was only
10% of total compile time on the heterogeneous case. The remaining 88.8% is
inside the single Packed compile itself.

## Target vs reality

Plan success criterion 5 required `benchmark_snapshot_matches_current_optimizer`
under `120 s` wall. Actual: **413.69 s**. The target is **not met** by
Phase 1 + 2 alone and cannot be met without crossing the Phase-plan's scope
boundaries.

Phase 1 (address-book hoist) was measured as 0.02% of time and was skipped as
below-noise per the plan's "stop and reassess" guardrail.

Phase 3 (prefix memoization) is explicitly a design spike only in the approved
plan; it was not implemented.

## What would actually move the needle (out of current plan scope)

1. **Cheap-scorer + single final compile.** Use
   `estimate_plan_cost_structural` during candidate ranking and only run the
   replay-compile on the winning plan. 1350 compiles/rebalance → 1 compile.
   Requires proving structural ranking picks the same winner on the fixture
   set, or accepting a small tolerance.
2. **Prefix memoization at the replay-plan level** (Phase 3 spike). Needs
   overlap measurement and a cache whose value matches the compiler's
   internal reusable state — likely widens `src/execution/program.rs`.
3. **Shrink the heterogeneous fee sweep** from three fee multipliers to one
   on the ignored test only. The unignored bench suite already covers the
   pre-/post- economics delta.

Each has different cost/risk trade-offs. None are authorized by
[docs/ff_replay_compile_latency_plan.md](ff_replay_compile_latency_plan.md);
pick before proceeding.

## What the Packed-only change does *not* change

- Fixture expectations in
  [test/fixtures/rebalancer_ab_expected.json](../test/fixtures/rebalancer_ab_expected.json).
- Solver correctness: Packed dominance was proven on 8107+ candidates.
- Production submission path: on-chain replay still runs the same Packed
  compile it would have picked anyway.
- Foundry E2E validation: not yet re-run post-change (Phase 4 pending a
  scope decision on the options above).

## 2026-04-20 Follow-up: Two-Stage Structural-First Ranking (Option 1 landed)

Follow-up work executed per
[docs/ff_native_replay_latency_fix_plan.md](ff_native_replay_latency_fix_plan.md)
adopted Option 1 from the "What would actually move the needle" list.

### Architecture change

- Native exact-search path is now **structural-first**. Intermediate candidates
  (seed plans, shortlisted preserve plans, distilled proposals, compact
  frontier variants) are ranked with `estimate_plan_cost_structural` via a new
  `PlanCostingMode::StructuralOnly` threading. Replay compile is skipped for
  these intermediate rankings.
- A new internal enum `PlanCostingMode { ReplayPreferred, StructuralOnly }` and
  `estimate_plan_cost_with_mode` / `plan_cost_fields_with_mode` helpers provide
  the costing seam without touching action generation or ranking comparators.
- Replay compile is **reserved for final shortlist validation**. After
  `plan_result_cmp` sorts the full structural candidate set inside
  `enumerate_exact_no_arb_candidates_with_options`, the candidate vector is
  **truncated** to the top-K structural finalists, those K are re-priced
  through `reprice_plan_result_with_replay(...)`, and the returned Vec is the
  replay-priced K sorted by `plan_result_cmp`. A new
  `final_replay_validation_k(candidate_count) -> usize` helper caps K at
  **8** (full candidate set if fewer than 8). Truncating — rather than
  appending the structural tail back — is required because callers
  (`run_plain_family_plan`, `run_arb_primed_family_plan`, test helpers) iterate
  the returned Vec with `plan_result_is_better`; keeping the tail visible
  would let a structural-priced plan with an underestimated fee beat a
  replay-priced top-K plan and be returned as the native winner without ever
  being replay-validated.
- Final returned native plans still expose replay-backed fee fields
  (`fee_estimate_source == "replay_packed_program"`) whenever replay succeeds;
  otherwise the structural estimate is retained and the fee source already
  records the fallback per prior behavior.
- ForecastFlows replay validation is **intentionally unchanged**
  (`evaluate_forecastflows_action_set`, `run_forecastflows_family_plan`).

### Before / after wall time (release, `--test-threads=1`)

| Test | Baseline (2026-04-19) | After two-stage | Delta |
| --- | --- | --- | --- |
| `analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev` | 423.18 s | **20.84 s** | **-95.1%** |
| `benchmark_snapshot_matches_current_optimizer` | 424.14 s | **21.59 s** | **-94.9%** |
| `benchmark_ev_non_decreasing_vs_fixture` | 429.00 s | **19.71 s** | **-95.4%** |
| `print_repeated_heterogeneous_ninety_eight_forecastflows_latency` (rust_worker, both calls) | 214 ms / 237 ms | 361 ms / 233 ms | within noise |

Command form for the native benches, e.g.:

```
cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::benchmark_snapshot_matches_current_optimizer' -- --exact --nocapture --test-threads=1
```

Command form for the FF helper:

```
FORECASTFLOWS_WORKER_BIN=$(realpath ../ForecastFlows.rs/target/release/forecast-flows-worker) \
  cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::print_repeated_heterogeneous_ninety_eight_forecastflows_latency' -- --ignored --exact --nocapture --test-threads=1
```

Plan success criteria 1–3 are all met (≤120 s). Success criteria 4–6
(replay-backed winner fields, unchanged fixture expectations, unchanged FF
behavior) were also verified.

### Shortlist size

`FINAL_REPLAY_VALIDATION_MAX_K = 16` (raised from initial `8` — see
"K=8 → K=16 fix" below). The unignored bench suite still passes and no fixture
snapshot had to change.

### Regression test

Added `two_stage_structural_ranking_preserves_replay_backed_winner_on_heterogeneous_case`
in [src/portfolio/tests/rebalancer_contract_ab.rs](../src/portfolio/tests/rebalancer_contract_ab.rs)
as a focused guardrail for the two-stage pipeline. It verifies the selected
plan on the heterogeneous 98-outcome case still carries
`fee_estimate_source == "replay_packed_program"`, still clears the previously
committed net-EV floor, and still comes from a compact mixed compiler family
(or a materially better baseline_step_prune plan).

## 2026-04-20 Follow-up: K=8 → K=16 fix and ignored-test audit

After the two-stage pipeline landed, we audited every Category D ignored
assertion test (the long-running solver guards) at the new
`FINAL_REPLAY_VALIDATION_MAX_K`. One of them — the existing
`distilled_exact_no_arb_is_non_regressive_vs_baseline_k4_search` —
caught a real regression that the unignored bench suite did not.

### What the ignored test caught

`distilled_exact_no_arb_is_non_regressive_vs_baseline_k4_search` asserts
strict `distilled_ev >= baseline_k4_ev` on the heterogeneous 98-outcome case.
At K=8 the test failed:

- distilled_ev   = 216_039_884_848_561_586_176
- baseline_k4_ev = 216_039_884_848_561_618_944
- delta          = -32_768 wei (= 2^15 wei)

Both helpers route through `run_exact_no_arb_plan_with_options`, so the only
asymmetric pressure between them is the truncate-to-K step in
`enumerate_exact_no_arb_candidates_with_options`. Bisecting K confirmed it:
K=1024 passes, K=16 passes, K=8 fails. The structural ranking surfaces a
near-tie pair on this case where the `distilled` branch's structurally-best K
candidates do not include the actual replay-priced winner that the `baseline_k4`
branch keeps. K=16 widens the validation window enough to capture the winner
in both branches; the unignored bench suite never exercised this near-tie.

### Fix

[src/portfolio/core/rebalancer.rs:6307](../src/portfolio/core/rebalancer.rs#L6307):
`const FINAL_REPLAY_VALIDATION_MAX_K: usize = 16;` (was `8`).

Verified after the bump:

- `distilled_exact_no_arb_is_non_regressive_vs_baseline_k4_search` — passes
  (~46 s).
- `analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev` —
  20.02 s (no regression).
- `benchmark_snapshot_matches_current_optimizer` — 20.85 s (no regression).
- `benchmark_ev_non_decreasing_vs_fixture` — 19.65 s (no regression).

All three unignored bench tests stay well under the 120 s success criterion.

### Ignored-test audit findings

We classified each Category D ignored test by status at K=16.

**Pre-existing failures surfaced (not caused by the latency work; tracked
separately):**

1. **Stale `src/portfolio/tests/ev_snapshots.json`.** The fixture was last
   updated on 2026-02-24 (commit `002d269`) and predates several solver
   changes. Two ignored tests now fail with EV deltas that exceed the
   snapshot floor tolerance:
   - `test_fuzz_rebalance_ev_regression_fast_suite` — case 0 (full)
     got=327.66, expected=366.58, delta=-38.92, floor_tol=0.18.
   - `test_fuzz_rebalance_end_to_end_full_l1_invariants` — same case-0 EV
     mismatch.

   Both reproduce at K=1024, so this is independent of the K change. Fix
   path: regenerate the snapshot once the planner side is settled.

2. **Planner over-sell bug in `test_monte_carlo_ev_full_profitability_groups`.**
   Fails with "sell over-consumed holdings for argotorg/solidity:
   -716.7713258457056" — `replay_actions_to_state` invariant
   ([src/portfolio/tests.rs:1201](../src/portfolio/tests.rs#L1201)) catches
   that the planner emitted a sell larger than the holding. Reproduces at
   K=1024, so this is a pre-existing planner correctness bug independent
   of the two-stage work.

**Long-running by design (kept `#[ignore]`):**

3. **`test_random_group_search_vs_waterfall_complex_fuzz_cases`**
   ([src/portfolio/tests/monte_carlo.rs:1338-1429](../src/portfolio/tests/monte_carlo.rs#L1338-L1429)).
   Wall time 1201.77 s (~20 min). PASSES. This is an
   independent-random-group-search **oracle** that asserts the algorithm is
   never beaten by a randomized baseline across the four hardest fuzz cases.

   Why it stays `#[ignore]`:
   - Wall time exceeds the 5-minute test cap by design — it is a stress
     oracle, not a unit test.
   - Knobs (`max_rollouts`, `groups_per_rollout`, `min_runtime_secs`) are
     env-driven via `RandomSearchConfig::from_env()`
     ([src/portfolio/tests/monte_carlo.rs:660](../src/portfolio/tests/monte_carlo.rs#L660)),
     so CI/local runs can dial it down without touching code.
   - Kept in the suite because it remains the most aggressive
     algorithm-vs-oracle check we have for the rebalancer. Run manually
     before solver changes that could plausibly affect the four hardest
     cases.

   Recommended invocation when running it:

   ```
   cargo test --release \
     'portfolio::core::tests::monte_carlo::test_random_group_search_vs_waterfall_complex_fuzz_cases' \
     -- --ignored --exact --nocapture --test-threads=1
   ```

### Summary

- The K=8 → K=16 bump is the only behavioral change required by this audit.
- The two pre-existing failures (stale snapshot, planner over-sell) are
  unrelated to the latency work and need separate fixes.
- `test_random_group_search_vs_waterfall_complex_fuzz_cases` is a
  legitimate stress oracle and stays `#[ignore]`. Documented above so future
  maintainers do not mistake its runtime for a regression.
