# ForecastFlows And Known Issues PR Checklist

Status: all six PRs completed as of 2026-04-21.

## Execution Status

- [x] PR 1: Skip-reason propagation and doc sync
- [x] PR 2: Remove the benchmark-vs-production ForecastFlows mismatch
- [x] PR 3: Add a hard-failing ForecastFlows executable E2E test
- [x] PR 4: Triage and fix the connected 67+32 `no_certified_candidate` path
- [x] PR 5: Fix the planner over-sell bug
- [x] PR 6: Regenerate stale snapshots and document the refresh workflow

Snapshot refresh procedure:
- [docs/ev_snapshot_regeneration.md](ev_snapshot_regeneration.md)

## Purpose

This document turns the current ForecastFlows E2E review findings plus the
existing items in [docs/known_issues_and_future_work.md](known_issues_and_future_work.md)
into a PR-by-PR execution checklist.

The intended audience is a junior engineer working with local Rust + Foundry
tooling. The goal is to make each PR:

- small enough to review safely
- easy to verify locally
- explicit about when to stop and ask for help

## What This Plan Covers

This checklist addresses two groups of issues:

1. Existing tracked issues in
   [docs/known_issues_and_future_work.md](known_issues_and_future_work.md)
   including:
   - stale EV snapshots
   - planner over-sell in the monte-carlo profitability path
   - missing Foundry E2E re-validation
2. Additional proven harness issues from the ForecastFlows local E2E review:
   - no hard-failing ForecastFlows executable E2E test exists today
   - benchmark ForecastFlows coverage currently depends on a
     benchmark-only synthetic fallback path
   - benchmark skip reasons are flattened and docs have drifted

## Working Rules

Read this before starting:

- Do one PR at a time. Do not mix solver fixes, harness fixes, and snapshot
  updates in the same PR.
- Do not regenerate snapshots until the planner over-sell bug is fixed.
- When a step says "stop and escalate", do that instead of guessing.
- Prefer the smallest code change that proves or disproves the hypothesis.
- Keep notes in the PR description about:
  - what was reproduced
  - what changed
  - what commands were run
  - what still remains open

## Suggested PR Order

Land the PRs in this order:

1. PR 1: Skip-reason propagation and doc sync
2. PR 2: Remove the benchmark-vs-production ForecastFlows mismatch
3. PR 3: Add a hard-failing ForecastFlows executable E2E test
4. PR 4: Triage and fix the connected 67+32 `no_certified_candidate` path
5. PR 5: Fix the planner over-sell bug
6. PR 6: Regenerate stale snapshots and close out the known-issues doc

Do not reorder PR 5 and PR 6.

## PR 1: Skip-Reason Propagation And Doc Sync

### Goal

Make ForecastFlows benchmark failures observable and make the docs match the
actual behavior.

### Why This PR Goes First

Right now the harness collapses any ForecastFlows fixture failure into
`forecastflows_uncertified`, which makes later debugging harder. Fix the
visibility problem before touching solver behavior.

### Primary Files

- [test/LocalFoundryExecutableTxE2E.t.sol](../test/LocalFoundryExecutableTxE2E.t.sol)
- [docs/local_foundry_e2e_benchmark_matrix.md](local_foundry_e2e_benchmark_matrix.md)
- [docs/ff_fee_estimator_post_landing_followups.md](ff_fee_estimator_post_landing_followups.md)

### Implementation Checklist

- [ ] Read `_runOffchainLane`.
- [ ] Read `_tryRunFixture` and `_tryRunFixtureInner`.
- [ ] Keep the `stderr` string returned from `_tryRunFixture(...)`.
- [ ] Extract the `fallback=<reason>;` token if present.
- [ ] Emit that extracted token as `skip_reason`.
- [ ] If no token is present, emit a neutral fallback like
      `forecastflows_fixture_exit`.
- [ ] Update the benchmark-matrix doc so its skip-reason table matches the
      new code.
- [ ] Update the follow-ups doc so it describes what really landed.

### Suggested Implementation Notes

- Do not rewrite the benchmark harness structure in this PR.
- Do not change solver selection logic in this PR.
- Keep the parsing helper local to the test file unless reuse is obviously
  needed.

### Verification

Run:

```bash
FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  forge test --ffi --match-test test_benchmark_matrix_connected -vv
```

Then inspect:

- [test/fixtures/local_foundry_e2e_benchmark_matrix_connected.jsonl](../test/fixtures/local_foundry_e2e_benchmark_matrix_connected.jsonl)

### Done Criteria

- The connected ForecastFlows row reports the real reason such as
  `no_certified_candidate`.
- The docs no longer claim a hardcoded reason if the code no longer does that.
- No solver behavior changed.

### Stop And Escalate If

- Foundry `stderr` shape is inconsistent across runs and you cannot extract a
  stable token without risky string parsing.

## PR 2: Remove The Benchmark-Only ForecastFlows Mask

### Goal

Make the ForecastFlows benchmark prove the normal production translation path
rather than a benchmark-only fallback path.

### Proven Problem

The benchmark harness currently compiles the fixture with
`--features benchmark_synthetic_fixtures`, which enables the synthetic
single-band fallback in `translate.rs`. The production path fails on the
98-outcome benchmark ForecastFlows inputs with:

```text
market ... does not have replayable contiguous liquidity ladder geometry
```

### Primary Files

- [test/LocalFoundryExecutableTxE2E.t.sol](../test/LocalFoundryExecutableTxE2E.t.sol)
- [src/portfolio/core/forecastflows/translate.rs](../src/portfolio/core/forecastflows/translate.rs)
- [docs/local_foundry_e2e_benchmark_matrix.md](local_foundry_e2e_benchmark_matrix.md)

### Strategy

Do not broaden production semantics first. First try to make the benchmark pool
construction replayable by the production translator.

### Implementation Checklist

- [ ] Read `derive_contiguous_liquidity_intervals_primary`.
- [ ] Read `fallback_single_tick_intervals`.
- [ ] Understand what pool/tick geometry the production path accepts.
- [ ] Add a benchmark-only liquidity seeding helper in the Foundry harness that
      mints a finite ladder around the active tick instead of only one
      full-range `[MIN_TICK, MAX_TICK]` position.
- [ ] Use that helper only for benchmark scenarios at first.
- [ ] Keep existing non-benchmark scenario behavior unchanged in this PR.
- [ ] Re-run the single-market ForecastFlows fixture without
      `benchmark_synthetic_fixtures`.
- [ ] If the production path succeeds, remove the benchmark-only feature flag
      from `_tryRunFixtureInner`.
- [ ] Update benchmark docs to remove mention of the synthetic fallback if it
      is no longer required.

### Suggested Implementation Notes

- Start with `bench_single_market_98` only.
- Do not try to fix the connected case in the same PR.
- Keep liquidity construction as simple as possible:
  - a few adjacent initialized ranges
  - symmetric around the active tick
  - enough to satisfy contiguous-ladder expectations

### Verification

First run the production fixture directly:

```bash
FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  cargo run --release --quiet --bin local_foundry_e2e_fixture \
  test/fixtures/local_foundry_e2e_fixture_input_bench_single_market_98_forecastflows.json
```

Then run the benchmark:

```bash
FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  forge test --ffi --match-test test_benchmark_matrix_single_market -vv
```

### Done Criteria

- The single-market ForecastFlows benchmark succeeds via the normal production
  translator path.
- `_tryRunFixtureInner` no longer needs `--features benchmark_synthetic_fixtures`
  for that benchmark.
- Docs explain the new benchmark construction clearly.

### Stop And Escalate If

- Making the pools replayable requires a large redesign of the benchmark
  topology instead of a surgical liquidity-construction change.
- You cannot make the production translator accept the benchmark pools without
  changing core ForecastFlows semantics.

## PR 3: Add A Hard-Failing ForecastFlows Executable E2E Test

### Goal

Make at least one local-contract ForecastFlows execution path fail the suite if
it regresses.

### Proven Problem

The current positive scenario tests use `_fixtureInputJson`, which hardcodes
`solver="native"`. ForecastFlows only appears in the benchmark matrix, where
fixture failure is downgraded to a skip row.

### Primary Files

- [test/LocalFoundryExecutableTxE2E.t.sol](../test/LocalFoundryExecutableTxE2E.t.sol)

### Implementation Checklist

- [ ] Add a new helper or test path that calls `_fixtureInputJsonWithSolver`
      with `solver="forecastflows"`.
- [ ] Use `_runFixture`, not `_tryRunFixture`, so failure aborts the test.
- [ ] Start with the smallest stable topology that passes after PR 2.
- [ ] Reuse the same assertion structure already used in
      `_executeRustFixtureScenario`:
  - nonzero action count
  - nonzero chunk count
  - execution succeeds
  - realized net EV is positive
  - raw EV matches expected within tolerance
  - no helper/router stranding
- [ ] Keep the native tests in place; this is additive coverage.

### Suggested Test Shape

Start with one new test only, for example:

- single-market
- small outcome count
- no connected child routing
- no special seeded merge inventory

Add a connected ForecastFlows executable test later only after the connected
worker path is understood.

### Verification

```bash
FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  forge test --ffi --match-path test/LocalFoundryExecutableTxE2E.t.sol -vv
```

### Done Criteria

- At least one ForecastFlows executable E2E path is enforced by Forge.
- A ForecastFlows regression makes the test fail instead of silently becoming a
  skip row.

### Stop And Escalate If

- The only passing ForecastFlows path still depends on benchmark-only behavior
  from PR 2.

## PR 4: Triage And Fix The Connected 67+32 `no_certified_candidate` Path

### Goal

Determine whether the connected ForecastFlows failure is caused by:

- malformed `deep_trading` request construction
- translation/replay constraints in this repo
- or a worker-side limitation in `ForecastFlows.rs`

### Root Cause To Verify

The final PR should confirm or refute this concrete hypothesis:

- the flattened connected request needs a conservative explicit `split_bound`
  because connector / invalid inventory is not represented in the FF problem
- and the Local Foundry benchmark lane must run FF with the `benchmark`
  request profile (`baseline` tuning), because the production
  `low_latency` profile can downgrade the connected request to uncertified
  before any doubling is attempted

### Primary Files

- [src/bin/local_foundry_e2e_fixture.rs](../src/bin/local_foundry_e2e_fixture.rs)
- [src/portfolio/core/forecastflows/mod.rs](../src/portfolio/core/forecastflows/mod.rs)
- [src/portfolio/core/forecastflows/translate.rs](../src/portfolio/core/forecastflows/translate.rs)
- Possibly the sibling `ForecastFlows.rs` repo if the request is valid

### Implementation Checklist

- [ ] Add a temporary debug mode in `local_foundry_e2e_fixture.rs` behind an
      env var such as `LOCAL_FOUNDRY_E2E_DUMP_PROBLEM=1`.
- [ ] When enabled, dump the exact `PredictionMarketProblemRequest` sent to the
      worker.
- [ ] Capture one passing single-market 98-outcome request.
- [ ] Capture one failing connected 67+32 request.
- [ ] Compare:
  - outcome count
  - market count
  - fair values
  - current prices
  - liquidity band counts
  - whether child markets differ materially from root markets
- [ ] Record:
  - direct status
  - mixed status
  - certified drop reason
  - replay drop reason
- [ ] If the request is malformed, fix this repo.
- [ ] If the request is sane but still uncertified, file the issue against the
      worker repo with the dumped repro artifact.

### Important Rule

Do not change both the request shape and the worker in the same PR.

### Verification

```bash
FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  cargo run --release --quiet --bin local_foundry_e2e_fixture \
  test/fixtures/local_foundry_e2e_fixture_input_bench_connected_98_forecastflows.json
```

Then:

```bash
FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  forge test --ffi --match-test test_benchmark_matrix_connected -vv
```

### Done Criteria

One of these must be true:

- The connected ForecastFlows benchmark becomes executable, or
- there is a clear written root-cause note explaining why it remains
  `no_certified_candidate`, with ownership assigned to this repo or the worker
  repo.

### Stop And Escalate If

- The request looks valid but the worker behavior is opaque and you need solver
  internals from the sibling repo.

## PR 5: Fix The Planner Over-Sell Bug

### Goal

Fix the planner bug described in
[docs/known_issues_and_future_work.md](known_issues_and_future_work.md) where a
monte-carlo profitability test generates sells beyond the simulated holdings.

### Primary Files

- [src/portfolio/tests/monte_carlo.rs](../src/portfolio/tests/monte_carlo.rs)
- [src/portfolio/tests.rs](../src/portfolio/tests.rs)
- whichever planner/group-builder files feed that test

### Implementation Checklist

- [ ] Reproduce the exact failing ignored test.
- [ ] Add targeted instrumentation locally to identify the first action that
      pushes holdings negative.
- [ ] Trace which builder generated that sell action.
- [ ] Confirm whether multiple sub-actions are being layered without updating
      the running balance.
- [ ] Add a focused regression test if you can isolate the action builder.
- [ ] Fix only the holding-accounting bug in this PR.
- [ ] Remove any temporary debugging output before merging.

### Verification

```bash
cargo test --release \
  'portfolio::core::tests::monte_carlo::test_monte_carlo_ev_full_profitability_groups' \
  -- --ignored --exact --nocapture --test-threads=1
```

Recommended extra check:

```bash
cargo test --release \
  'portfolio::core::tests::monte_carlo::test_random_group_search_vs_waterfall_complex_fuzz_cases' \
  -- --ignored --exact --nocapture --test-threads=1
```

Use the faster env-tuned version if needed for iteration.

### Done Criteria

- The over-sell panic is fixed.
- The targeted monte-carlo reproduction is green.
- No snapshots were regenerated in this PR.

### Stop And Escalate If

- Fixing the over-sell requires a large planner refactor instead of a localized
  accounting fix.

## PR 6: Regenerate Stale Snapshots And Close Out The Doc

### Goal

Refresh stale snapshot fixtures only after the planner behavior is believed to
be correct.

### Primary Files

- [src/portfolio/tests/ev_snapshots.json](../src/portfolio/tests/ev_snapshots.json)
- [docs/known_issues_and_future_work.md](known_issues_and_future_work.md)
- any snapshot-regeneration helper docs you add

### Implementation Checklist

- [ ] Re-run the stale snapshot repro from the known-issues doc.
- [ ] Regenerate `ev_snapshots.json` using the approved repo workflow.
- [ ] Verify the two known failing ignored fuzz tests now pass.
- [ ] Add a short snapshot-regeneration procedure doc if one still does not
      exist.
- [ ] Update `known_issues_and_future_work.md`:
  - mark fixed items as resolved
  - add the ForecastFlows harness issues found in the E2E review
  - keep unresolved items open with current status

### Verification

```bash
cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_fuzz_rebalance_ev_regression_fast_suite' \
  -- --ignored --exact --nocapture --test-threads=1

cargo test --release \
  'portfolio::core::tests::fuzz_rebalance::test_fuzz_rebalance_end_to_end_full_l1_invariants' \
  -- --ignored --exact --nocapture --test-threads=1
```

### Done Criteria

- Snapshot tests pass with the new fixture.
- The doc reflects current reality rather than the pre-fix audit state.
- Snapshot regeneration happened after, not before, the planner fix.

## Final Repo-Wide Verification

After all PRs land, run the smallest realistic regression set:

```bash
cargo test --release --bin local_foundry_e2e_fixture

FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  forge test --ffi --match-path test/LocalFoundryExecutableTxE2E.t.sol -vv

FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  cargo test --release \
  'portfolio::core::tests::rebalancer_contract_ab::print_shared_op_snapshot_forecastflows_selected_rows_jsonl' \
  -- --ignored --exact --nocapture --test-threads=1
```

If time allows, also run the targeted known-issues reproductions from
[docs/known_issues_and_future_work.md](known_issues_and_future_work.md).

## Expected Outcomes If Everything Goes Well

At the end of this sequence, the repo should have:

- real skip reasons in benchmark artifacts
- no benchmark-only masking of the ForecastFlows single-market path
- at least one hard-failing ForecastFlows executable E2E test
- a clear root-cause explanation for the connected ForecastFlows status
- a fix for the planner over-sell bug
- regenerated snapshots based on a known-good planner state
- updated docs that match the code

## Ownership Notes

Use this rule when deciding whether to keep working locally or escalate:

- If the issue is in Foundry harness setup, fixture JSON construction, address
  book wiring, or translation/replay code, fix it in this repo.
- If the problem is a sane request returning no candidate from the worker, open
  a focused worker-side follow-up in `ForecastFlows.rs` with the dumped repro
  payload.
