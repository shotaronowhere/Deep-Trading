# ForecastFlows Native Replay Latency Fix Plan

Status: proposed implementation plan only. No code changes are described as
already landed in this document.

## Objective

Reduce native solver latency on the heterogeneous 98-outcome fixture without
changing the final selected plan economics or weakening replay-backed fee
validation for the plan we actually return.

The current problem is not the Rust ForecastFlows worker. The current problem
is that the native exact-search path uses replay compilation to score too many
intermediate candidates.

## Measured Baseline

Local release measurements from 2026-04-19:

| Target | Result |
| --- | --- |
| `analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev` | `423.18s` |
| `benchmark_snapshot_matches_current_optimizer` | `424.14s` |
| `print_repeated_heterogeneous_ninety_eight_forecastflows_latency` with Rust worker | `214ms`, then `237ms` |

Live process samples taken after the 3-minute mark in both native tests showed
the same hot stack:

1. `enumerate_exact_no_arb_candidates_with_options`
2. `run_no_arb_rebalance_plan_from_state`
3. `estimate_plan_cost_from_replay`
4. `compile_execution_program_unchecked_with_address_book`
5. `build_unsigned_batch_execute_tx_bytes`

This matches the earlier report in
[docs/ff_replay_compile_latency_report.md](ff_replay_compile_latency_report.md),
which found that packed replay compilation dominated wall time.

## Root Cause

The native exact candidate funnel repeatedly calls replay-based fee estimation
while it is still exploring and ranking a large candidate set.

The expensive path is:

1. Build candidate actions.
2. Build replay group plans.
3. Compile a packed execution program.
4. ABI-encode and RLP-encode the full batch transaction bytes.
5. Use those replay-priced economics to rank another intermediate candidate.

This is safe but too expensive when repeated many times across:

1. seed candidates
2. shortlisted preserve candidates
3. distilled proposal candidates
4. compact frontier variants
5. whole-fixture benchmark tests that call the full solver multiple times

## Chosen Fix

Use a two-stage costing pipeline in the native exact-search path:

1. Rank intermediate candidates with the existing structural fee estimate.
2. Replay-price only a small final shortlist before selecting the returned
   winner.

This is the smallest safe change because it reduces the number of replay
compiles dramatically without deleting the replay-based cost model from the
final decision.

## Success Criteria

1. `benchmark_snapshot_matches_current_optimizer` finishes under `120s` in
   `cargo test --release -- --exact --nocapture --test-threads=1`.
2. `benchmark_ev_non_decreasing_vs_fixture` finishes under `120s` under the
   same release settings.
3. `analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev`
   finishes under `120s`.
4. Final returned native plans still expose replay-backed fee fields when
   replay succeeds.
5. Existing snapshot expectations and heterogeneous wobble tolerances remain
   unchanged.
6. ForecastFlows replay validation behavior does not change in this pass.

## Non-Goals

1. Do not redesign execution program packing in this pass.
2. Do not optimize `build_unsigned_batch_execute_tx_bytes` first.
3. Do not change the ForecastFlows Rust worker path.
4. Do not change fixture expectations unless the user explicitly approves a
   correctness-level behavior change.
5. Do not add runtime configurability for shortlist sizes. Keep knobs internal.

## Files Most Likely To Change

1. [src/portfolio/core/rebalancer.rs](../src/portfolio/core/rebalancer.rs)
2. [src/portfolio/tests/rebalancer_contract_ab.rs](../src/portfolio/tests/rebalancer_contract_ab.rs)
3. [docs/ff_replay_compile_latency_report.md](ff_replay_compile_latency_report.md)

## Step-by-Step Implementation

### Step 1: Add an Internal Costing Mode

Goal: create a safe seam between cheap candidate ranking and expensive final
validation.

Primary functions:

1. `estimate_plan_cost`
2. `estimate_plan_cost_from_replay`
3. `estimate_plan_cost_structural`
4. `plan_cost_fields`

Implementation:

1. Add a small internal enum near the cost estimation helpers:
   `PlanCostingMode { ReplayPreferred, StructuralOnly }`.
2. Add `estimate_plan_cost_with_mode(...)`.
3. Add `plan_cost_fields_with_mode(...)`.
4. Keep the current `estimate_plan_cost(...)` behavior exactly as-is by having
   it call `estimate_plan_cost_with_mode(..., ReplayPreferred)`.
5. Keep the current `plan_cost_fields(...)` behavior exactly as-is by having it
   call `plan_cost_fields_with_mode(..., ReplayPreferred)`.

Why this step matters:

1. It lets us change ranking behavior in a targeted way.
2. It avoids rewriting every caller at once.
3. It preserves current behavior for unchanged paths.

Verify:

1. `cargo test --release plan_result_cmp -- --nocapture`
2. `cargo check`

If compilation fails widely, stop and simplify before touching candidate logic.

### Step 2: Add a Plan Builder That Uses Structural-Only Costing

Goal: build `PlanResult` values for candidate comparison without paying replay
compile cost yet.

Primary function:

1. `build_plan_result`

Implementation:

1. Introduce either:
   `build_plan_result_with_costing_mode(...)`
   or
   `build_plan_result_structural_only(...)`.
2. Keep the current `build_plan_result(...)` as the replay-preferred builder.
3. Make the structural-only builder reuse all existing fields:
   `actions`, `terminal_state`, `raw_ev`, `family`, `compiler_variant`,
   `selected_common_shift`, `selected_mixed_lambda`, and
   `selected_active_set_size`.
4. Only change how fee-related fields are populated.

Why this step matters:

1. Candidate construction stays readable.
2. The rest of the solver can still compare normal `PlanResult`s.

Verify:

1. Add or update a small unit test that confirms the structural-only builder
   preserves non-fee fields exactly.
2. Confirm the structural-only builder reports structural fee source instead of
   replay fee source.

### Step 3: Thread Structural-Only Costing Through the Native Exact Candidate Funnel

Goal: stop paying replay compile cost for every intermediate native candidate.

Primary functions:

1. `seed_no_arb_plan_from_state`
2. `run_no_arb_rebalance_plan_from_state`
3. `compact_raw_no_arb_plan_for_program_net_ev`
4. `compile_best_frontier_candidate_for_program_net_ev`
5. `compile_coupled_mixed_candidate_for_program_net_ev`
6. any helper those functions call that currently uses `build_plan_result(...)`

Implementation:

1. Change seed candidates to use structural-only costing.
2. Change full native exact candidates to use structural-only costing while they
   are still in the exploration funnel.
3. Change compact frontier-derived candidate builders to use structural-only
   costing while comparing alternatives.
4. Keep action generation identical. Do not alter solver search behavior yet.
5. Keep candidate ranking logic unchanged. It should still use
   `plan_result_is_better` and `plan_result_cmp`; only the fee inputs behind the
   intermediate `PlanResult`s should differ.

Why this step matters:

1. This is where the replay compile explosion currently happens.
2. Reducing replay usage here should remove the majority of the 7-minute cost.

Verify:

1. `cargo check`
2. Run one fast small-case test from `rebalancer_contract_ab` to confirm the
   native solver still returns a sensible plan.

Do not run the long heterogeneous tests yet. The final winner is not replay-
validated until Step 4.

### Step 4: Add a Replay Repricing Helper for Finalists

Goal: keep replay-backed economics on the plan we actually return.

Primary functions:

1. `run_exact_no_arb_plan_with_options`
2. `selected_plan_summary_from_plan_for_test`
3. fee helpers around `plan_cost_fields_from_estimate`

Implementation:

1. Add `reprice_plan_result_with_replay(...) -> PlanResult`.
2. It should:
   - take an existing `PlanResult`
   - call the replay-preferred cost path on `plan.actions`
   - update only the fee-related fields and fee source
   - preserve all other fields exactly
3. If replay pricing fails on a finalist, leave the structural estimate in
   place and record that via the fee source already used by existing code.

Why this step matters:

1. The final returned plan should still be grounded in replay economics.
2. This limits replay compile work to a much smaller set.

Verify:

1. Add a unit test for `reprice_plan_result_with_replay(...)` on a small replay-
   valid action set.
2. Confirm repricing changes fee fields but not actions or selected metadata.

### Step 5: Reprice Only a Small Final Shortlist

Goal: combine cheap search with correct final replay-based plan selection.

Primary function:

1. `run_exact_no_arb_plan_with_options`

Implementation:

1. Keep generating the full native candidate set structurally.
2. Sort candidates using the existing `plan_result_cmp`.
3. Add a helper such as `final_replay_validation_k(candidate_count) -> usize`.
4. Start with a conservative shortlist size of `8`.
5. Reprice only the top `8` structural candidates with
   `reprice_plan_result_with_replay(...)`.
6. Choose the final winner from that replay-priced shortlist using the existing
   comparison logic.
7. If there are fewer than `8` candidates, reprice them all.

Why this step matters:

1. It dramatically cuts replay compile count.
2. It still lets replay economics break close structural ties among finalists.

Verify:

1. Run the single heterogeneous native test:
   `cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev' -- --exact --nocapture --test-threads=1`
2. Confirm:
   - it passes
   - final plan still has replay-backed fee fields
   - runtime is materially lower than the `423.18s` baseline

If plan selection changes unexpectedly, do not broaden the code change. First
increase the shortlist to `16` and retest.

### Step 6: Leave ForecastFlows Replay Validation Alone

Goal: avoid mixing two separate latency problems.

Primary functions:

1. `evaluate_forecastflows_action_set`
2. `run_forecastflows_family_plan`

Implementation:

1. Do not switch ForecastFlows candidate validation to structural-only costing.
2. Do not remove its replay requirement.
3. Do not change telemetry semantics.

Why this step matters:

1. ForecastFlows is already fast enough locally with the Rust worker.
2. The benchmark issue is in native exact search, not this path.

Verify:

1. `FORECASTFLOWS_WORKER_BIN=$(realpath ../ForecastFlows.rs/target/release/forecast-flows-worker) cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::print_repeated_heterogeneous_ninety_eight_forecastflows_latency' -- --ignored --exact --nocapture --test-threads=1`
2. Confirm the helper still stays in the sub-second range and telemetry remains
   populated.

### Step 7: Add a Focused Regression Test For The New Two-Stage Ranking Rule

Goal: create a small, stable guardrail around the new architecture.

Recommended test coverage:

1. single heterogeneous case only
2. final winner still replay-priced
3. final selected `compiler_variant` matches current committed behavior
4. final action count matches current committed behavior
5. final `estimated_net_ev` matches within existing heterogeneous tolerance

Implementation:

1. Add a single-case test in `rebalancer_contract_ab.rs`.
2. Do not add another full-fixture loop.
3. Reuse existing fixture builders.

Why this step matters:

1. It protects the new ranking architecture.
2. It gives us a much cheaper correctness gate than rerunning every benchmark
   loop during development.

Verify:

1. Run the new test by exact name in release mode.

### Step 8: Run the Full Verification Loop

Run in this exact order:

1. `cargo check`
2. `cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev' -- --exact --nocapture --test-threads=1`
3. `cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::benchmark_snapshot_matches_current_optimizer' -- --exact --nocapture --test-threads=1`
4. `cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::benchmark_ev_non_decreasing_vs_fixture' -- --exact --nocapture --test-threads=1`
5. `FORECASTFLOWS_WORKER_BIN=$(realpath ../ForecastFlows.rs/target/release/forecast-flows-worker) cargo test --release 'portfolio::core::tests::rebalancer_contract_ab::print_repeated_heterogeneous_ninety_eight_forecastflows_latency' -- --ignored --exact --nocapture --test-threads=1`

Timing guardrails:

1. If any test passes `3m`, capture a `sample` before waiting longer.
2. If any test approaches `10m`, stop and inspect before continuing.
3. Never let a debugging run exceed `10m`.

Sampling command:

1. `sample <pid> 5 -mayDie`

What to look for in the sample:

1. fewer samples under `estimate_plan_cost_from_replay`
2. fewer samples under `compile_execution_program_unchecked_with_address_book`
3. fewer samples under `build_unsigned_batch_execute_tx_bytes`

### Step 9: Update Documentation

Goal: make the final architecture obvious to future readers.

Docs to update:

1. [docs/ff_replay_compile_latency_report.md](ff_replay_compile_latency_report.md)
2. optionally [docs/solver_benchmark_matrix.md](solver_benchmark_matrix.md) if
   benchmark timing tables are refreshed

Document:

1. native exact ranking is now structural-first
2. replay compile is reserved for final shortlist validation
3. final returned native plans still use replay-backed fee fields when replay
   succeeds
4. ForecastFlows replay validation was intentionally unchanged
5. before/after timings
6. chosen shortlist size

Verify:

1. confirm docs match the actual code paths
2. confirm timing numbers are labeled with date and command form

## Rollback Plan

If the structural-first shortlist approach changes the committed winner on the
heterogeneous case and increasing the shortlist from `8` to `16` does not fix
it:

1. revert only the structural-only candidate ranking changes
2. keep any harmless test helpers that were added if they still compile cleanly
3. document the mismatch clearly
4. stop before attempting a larger redesign such as replay-plan memoization

## If This Plan Is Not Enough

If the native benchmark is still too slow after the shortlist revalidation
change, the next option is:

1. replay-price only the single best structural winner

That is a riskier behavior change because replay is no longer allowed to
reorder the finalists. Do not implement that as part of the first pass.

The next bigger structural option after that is:

1. replay-plan or execution-program memoization inside the compiler path

That is a larger engineering task and should be a separate document if needed.
