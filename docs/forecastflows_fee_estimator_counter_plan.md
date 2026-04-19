# Counter-Plan: Fix ForecastFlows Fee Replay Without a Zeroed Synthetic Address Book

## Goal

Restore the single-market ForecastFlows benchmark path by fixing the fee replay
estimator's address resolution failure, while keeping the replayed execution
program aligned with the actual call topology that the fixture and real harness
compile.

This document is a counter-plan to
`/tmp/ff_fee_estimator_decoupling_plan.md`. The original plan correctly found
the immediate `UnknownMarket` failure, but its proposed synthetic address book
is not safe as written for this codebase.

## Confirmed Failure Path

The current failure is real and local:

1. `estimate_plan_cost_from_replay` in `src/portfolio/core/rebalancer.rs`
   builds replay plans, then calls `compile_execution_program_unchecked(...)`.
2. `compile_execution_program_unchecked` in
   `src/execution/program.rs` uses `ExecutionAddressBook::default()`.
3. `ExecutionAddressBook::default()` leaves `outcome_tokens` empty in
   `src/execution/tx_builder.rs`.
4. Replay actions for the synthetic benchmark use names like
   `bench_single_market_98_root_0`, which are not in `MARKETS_L1`.
5. `outcome_token_for_market` in `src/execution/tx_builder.rs` therefore
   returns `TxBuildError::UnknownMarket`.
6. `estimate_plan_cost_from_replay` returns `None`.
7. `evaluate_forecastflows_action_set` rejects the candidate and
   `run_forecastflows_family_plan` reports `fallback_reason =
   "no_replayable_candidate"`.

Important nuance: this happens on the first raw candidate already. The prune
loops also re-enter the same failing estimator, but they are not the earliest
blocker.

## Strong Disagreement With the Original Plan

### 1. The proposed synthetic book is not semantics-free

The original plan treats most address fields as transport-only placeholders.
That is not true in this codebase.

`market2` is used as a topology sentinel in `src/execution/tx_builder.rs`:

- `build_mint_sell_calls` emits the second split only when
  `address_book.market2 != Address::ZERO`
- `build_buy_merge_calls` emits the second merge only when
  `address_book.market2 != Address::ZERO`
- `build_direct_merge_calls` emits the second merge only when
  `address_book.market2 != Address::ZERO`

So a fee-estimation book that zeroes `market2` can change the number of emitted
calls, not just the byte values inside otherwise-identical calldata.

That means the original "all non-outcome fields are `Address::ZERO`" proposal
can undercount both:

- calldata length
- program shape
- estimated fee

### 2. The plan's "byte-exact" claim is false

The current modeled L1 fee path does not inspect calldata byte values. It only
prices `unsigned_tx_data.len()` through
`estimate_l1_data_fee_susd_for_tx_bytes_len` in `src/execution/gas.rs`.

So today:

- zero vs nonzero address bytes do not change the modeled L1 fee
- but changing addresses can still change encoded bytes and transaction length
- and `market2 == 0` can change whether extra calls exist at all

Separately, the planner path also uses synthetic fee inputs with:

- chain id `10`
- sender nonce `0`
- executor `Address::ZERO`

while the fixture binary compiles with:

- chain id `31337`
- real executor from fixture input

So the current planner replay path is already only an approximation of the
fixture-emitted transaction bytes. The right response is not to add more
zero-address placeholders and call the result exact.

### 3. Deleting the `#[cfg(test)]` fallback is fine only after the replay path is fixed

The existing test-only pseudo-address fallback in
`outcome_token_for_market` is not the right long-term API, but it should not be
deleted until both replay callers are moved off the default empty address book:

- `estimate_plan_cost_from_replay`
- `total_calldata_bytes_for_actions_for_test`

Otherwise we risk breaking tests that still rely on the old path.

## Correct Design Principles

The safe minimal design for this repo is:

1. Keep compiling a real `ExecutionProgramPlan` for replay scoring.
2. Build an explicit fee-estimation `ExecutionAddressBook`.
3. Populate outcome-token mappings from the replay snapshot itself, not from
   synthetic keccak fallbacks.
4. Preserve the real default topology addresses from
   `ExecutionAddressBook::default()`.
5. Do not zero `market2`, routers, collateral, or market addresses.
6. Do not claim byte-exactness unless the estimator also threads the real
   executor and fee inputs from the caller.

## Proposed Fix

### Step 1. Build a fee-estimation address book from `slot0_results`

Add a small helper near the replay cost estimator in
`src/portfolio/core/rebalancer.rs`:

```rust
fn fee_estimation_address_book(
    slot0_results: &[(Slot0Result, &'static crate::markets::MarketData)],
) -> ExecutionAddressBook
```

Behavior:

- start from `ExecutionAddressBook::default()`
- fill `outcome_tokens` from `slot0_results`
- use `market.name -> market.outcome_token`
- ignore snapshot entries whose `outcome_token` does not parse as an address
  only if the current code already treats that as impossible; otherwise fail
  closed

Why this is enough:

- the synthetic benchmark fixture already provides real deployed outcome-token
  addresses for each synthetic market name
- `slot0_results` carries those `MarketData` values all the way into the replay
  estimator
- we do not need a synthetic keccak mapping rule for the single-market fixture
  failure

### Step 2. Use the explicit address book in both replay-program compiles

In `estimate_plan_cost_from_replay`:

- replace both calls to `compile_execution_program_unchecked(...)`
- call `compile_execution_program_unchecked_with_address_book(...)` instead
- pass the fee-estimation book built from `slot0_results`

This is the actual fix for the benchmark blocker.

### Step 3. Mirror the same fix in the test-only calldata helper

`total_calldata_bytes_for_actions_for_test` in the same file duplicates the same
broken pattern. It should use the same fee-estimation address book and the same
`compile_execution_program_unchecked_with_address_book(...)` path.

Without this, the main estimator would be fixed but test-only replay-byte
diagnostics could still fail on synthetic market names.

### Step 4. Leave `tx_builder` mostly alone for the first patch

For the smallest safe patch, do not introduce:

- `synthetic_outcome_token`
- `synthetic_address_book_for_fee_estimate`
- action-name scanning helpers in `tx_builder`

Those are broader API changes than necessary.

The smallest repo-safe fix lives entirely in `rebalancer.rs`, plus imports.

### Step 5. Optional cleanup after the replay path is stable

After the estimator is fixed and tests are updated, we can choose one of two
cleanup directions for `outcome_token_for_market`:

1. Keep the `#[cfg(test)]` fallback temporarily, but make new replay callers
   stop depending on it.
2. Remove the fallback entirely once all synthetic test paths use explicit
   address books.

That cleanup should be a second step, not part of the unblock patch.

## Why This Is Smaller Than the Original Plan

The original plan proposes new helpers in `tx_builder.rs` and a semantic change
to address resolution policy.

The smaller fix is:

- no new tx-builder API
- no synthetic keccak address rule
- no zero-address placeholders
- no production behavior change in `outcome_token_for_market`
- only thread an already-available mapping into replay compilation

## Verification Plan

### Code-level verification

1. Run targeted replay-cost tests that hit `estimate_plan_cost_from_replay`.
2. Run `cargo test --release execution::tx_builder::tests -- --nocapture`.
3. Run `cargo test --release execution::program::tests -- --nocapture`.

### Benchmark verification

1. Build the fixture binary:

```bash
cargo build --release --bin local_foundry_e2e_fixture \
  --features benchmark_synthetic_fixtures
```

2. Run the single-market ForecastFlows fixture input.

Expected outcome:

- no `UnknownMarket`
- no `fallback_reason = "no_replayable_candidate"`
- nonzero `expected_raw_ev_wad`
- nonzero `estimated_total_fee_wad`

3. Run:

```bash
forge test --ffi --match-test test_benchmark_matrix_single_market -vv
```

Expected outcome:

- ForecastFlows single-market row no longer skips for replayability reasons
- connected-topology ForecastFlows row may still skip as `forecastflows_uncertified`
  because that is a separate worker-certification issue

## If Byte-Exactness Becomes a Requirement

If the goal is not just "unblock replay scoring" but "make replay fee modeling
match emitted fixture tx bytes as closely as possible", then the correct next
step is different:

1. Thread the real `ExecutionAddressBook` from the caller.
2. Thread the real executor address from the caller.
3. Thread the real `LiveOptimismFeeInputs` from the caller.
4. Optionally replace the length-only L1 fee estimate with the exact
   `getL1Fee(bytes payload)` path already implemented in `src/execution/gas.rs`.

That is a larger architectural change and should not be mixed into the unblock
patch.

## Out of Scope

- Connected-topology ForecastFlows certification behavior
- Changes to `translate.rs` replay logic
- Changes to the bounds conservative repricing fix
- Replacing the structural fallback architecture
- Full exact-fee alignment between planner replay and fixture-emitted tx bytes

## Recommended Implementation Order

1. Add `fee_estimation_address_book(slot0_results)` in
   `src/portfolio/core/rebalancer.rs`.
2. Switch `estimate_plan_cost_from_replay` to
   `compile_execution_program_unchecked_with_address_book`.
3. Switch `total_calldata_bytes_for_actions_for_test` to the same path.
4. Run targeted Rust tests.
5. Run the single-market fixture binary.
6. Run the Foundry single-market benchmark.
7. Only then decide whether to remove the `#[cfg(test)]` fallback from
   `tx_builder.rs`.

## Bottom Line

The original diagnosis of the immediate blocker is correct. The proposed
remediation is not.

The right fix is to feed replay compilation the outcome-token mapping it already
has in `slot0_results`, while preserving the real default execution topology.
That unblocks the single-market benchmark without introducing a zero-address
synthetic book that can silently change program shape and underprice fees.
