# ForecastFlows Fee Estimator Counter-Plan — Post-Landing Follow-ups

## What landed

1. **Counter-plan** (earlier in session):
   `estimate_plan_cost_from_replay` and `total_calldata_bytes_for_actions_for_test`
   now compile the replay program with an explicit fee-estimation `ExecutionAddressBook`
   built from `slot0_results`, instead of the empty default book. This unblocked the
   single-market `UnknownMarket` failure for synthetic benchmark names.

2. **Finding #1 — topology threading** (this session):
   `PlannerCostConfig` now carries `topology_market2` / `topology_market2_collateral`.
   `fee_estimation_address_book` overwrites `book.market2` / `market2_collateral`
   from the cost config. New public entry
   `rebalance_with_custom_predictions_and_solver_and_gas_pricing_and_flags_and_decision_with_topology`
   lets the fixture binary pass the real `address_book.market2` /
   `address_book.market2_collateral` through. Production defaults unchanged.

3. **Finding #2 — harness skip-reason propagation** (this session):
   `_runOffchainLane` now captures `stderr` from `_tryRunFixture` and emits the
   `fallback=<reason>;` token from the Rust fixture binary as `skip_reason`,
   replacing the hardcoded `forecastflows_uncertified`. Falls back to
   `forecastflows_fixture_exit` if no token matches.

## Verified green

- `cargo check --release --all-targets`
- `cargo test --release execution::tx_builder` (10 passed)
- `cargo test --release execution::program` (1 passed)
- `cargo test --release forecastflows` (78 passed, 6 ignored)
- `forge test --ffi --match-test test_benchmark_matrix_single_market -vv` (PASS)
- `forge test --ffi --match-test test_benchmark_matrix_connected -vv` (PASS,
  FF row now reports `skip_reason: "no_certified_candidate"` instead of the
  collapsed `forecastflows_uncertified`)

## Reproduced red locally

- `cargo test --release benchmark_snapshot_matches_current_optimizer -- --nocapture`
  reproduces the four snapshot mismatches listed below, including
  `heterogeneous_ninety_eight_outcome_l1_like_case mixed expected=150380245589644673024 actual=216039884848561586176`
  and `mixed_route_favorable_synthetic_case full_rebalance_only expected=100108205685259485184 actual=100104446119388053504`.
- `cargo test --release benchmark_ev_non_decreasing_vs_fixture -- --nocapture`
  reproduces the heterogeneous full-rebalance floor regression exactly:
  `expected_floor=150378870058707484672 actual=150368350977861419008`.
- `cargo test --release analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev -- --nocapture`
  reproduces the selected-plan failure with
  `family: "plain"`, `frontier_family: "direct"`,
  `compiler_variant: "baseline_step_prune"`, `raw_ev: 216.03988484856168`,
  `estimated_total_fee_susd: 0.8711960077387768`,
  `estimated_net_ev: 215.1686888408229`,
  `estimated_group_count: 4899`, `estimated_tx_count: 8`,
  `fee_estimate_source: "replay_packed_program"`,
  `action_count: 4996` (`4888` direct buys, `107` direct sells, `1` mint).

## Failing tests uncovered by the counter-plan landing

Running the non-ignored bench + net-ev suite surfaced three pre-existing
regressions that were masked while `estimate_plan_cost_from_replay` was returning
`None` on synthetic names. Now that replay scoring succeeds, the planner is
exercising more branches and the heterogeneous 98-outcome benchmark case drifts.

| Test | Location | Symptom |
| --- | --- | --- |
| `benchmark_snapshot_matches_current_optimizer` | [src/portfolio/core/../tests/rebalancer_contract_ab.rs:2381](src/portfolio/tests/rebalancer_contract_ab.rs#L2381) | `heterogeneous_ninety_eight_outcome_l1_like_case` mixed EV drifts up (150380...→216039...), `full_rebalance_only` drifts down ~10^10 wei, `actions` count explodes 94 → 4996. Also `mixed_route_favorable_synthetic_case` full_rebalance_only drifts. |
| `benchmark_ev_non_decreasing_vs_fixture` | [src/portfolio/core/../tests/rebalancer_contract_ab.rs:2417](src/portfolio/tests/rebalancer_contract_ab.rs#L2417) | `heterogeneous_ninety_eight_outcome_l1_like_case` `full_rebalance_only` floor regression: expected ≥150378870058707484672, actual 150368350977861419008 (−0.007%). |
| `analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev` | [src/portfolio/core/../tests/rebalancer_contract_ab.rs:4944](src/portfolio/tests/rebalancer_contract_ab.rs#L4944) | Realistic 98 selected plan is `plain` / `baseline_step_prune` with `action_count: 4996` (4888 direct_buy, 107 direct_sell, 1 mint) and 8 chunks — should remain `analytic_mixed` / `constant_l_mixed`, one packed tx, ≤197 actions. |

### Common thread

All three failures concentrate on `heterogeneous_ninety_eight_outcome_l1_like_case`
and point at the same symptom: the planner now prefers a wide, many-action
plain/direct-buy plan over the compact analytic-mixed family. This is consistent
with replay scoring being available for plans that previously failed replay and
fell back to structural estimates — the planner is now comparing more families
under a common fee basis, and its preference has changed for this specific case.
The red assertions are therefore best read as planner-wide compact-plan follow-ups
surfaced by the FF counter-plan landing, not as proof that the remaining bug is
inside the FF-specific candidate builder itself.

### Suspected causes (to investigate, not confirmed)

- **Packed-vs-strict tiebreaker bias.** Replay-scored compact mixed plans may now
  be losing to replay-scored wide plain plans because the `better_execution_program`
  ordering prefers lower `total_fee_susd`, and 4996 single-tick direct buys in
  eight packed chunks can amortize L1 fee per action below the analytic-mixed
  single-tx plan on this geometry.
- **Prune/compaction re-entry under the new estimator.**
  `baseline_step_prune_candidate_for_program_net_ev` and
  `route_group_prune_candidate_for_program_net_ev` rebuild and re-score subsets
  via `build_plan_result(...) -> plan_cost_fields(...) -> estimate_plan_cost(...)`
  on each drop attempt. ForecastFlows has analogous re-entry in
  `baseline_step_prune_forecastflows_candidate_for_program_net_ev` and
  `route_group_prune_forecastflows_candidate_for_program_net_ev`. When replay now
  succeeds for more subsets, these passes can follow a different search
  trajectory. Monotonicity assumptions under the new estimator need an audit.
- **Snapshot expectations predate the counter-plan.** The snapshot fixture
  encodes values that were produced under the old `None`-returning estimator.
  If the current plans are genuinely better under the real fee basis, the
  fixture itself is stale.

### What current code read already weakens

- **Address-book completeness is no longer the leading suspect for the
  in-process heterogeneous case.** `fee_estimation_address_book(...)` now threads
  `market2`, `market2_collateral`, and per-market `outcome_tokens` from
  `slot0_results`, which means the replay estimator and execution compiler are
  already aligned on the topology fields that caused the original fixture
  failure. That does not prove the planner is correct, but it does push the
  likely fault toward candidate ordering / re-scoring rather than a stale
  synthetic topology.

## TODO

### Immediate (this workstream)

- [ ] **Triage `benchmark_snapshot_matches_current_optimizer`**: determine
  whether the new `mixed = 216039…` number is a correct improvement over
  `150380…` (snapshot update) or a selection-bug symptom (fix, don't regenerate).
  Bisect by temporarily swapping the explicit `fee_estimation_address_book(...)`
  path back to the old default-book compile in both
  `estimate_plan_cost_from_replay(...)` and
  `total_calldata_bytes_for_actions_for_test(...)`.
- [ ] **Reproduce `analytic_mixed_selection_improves_realistic_heterogeneous_case_net_ev`
  at the planner level**: dump all candidate plans for
  `heterogeneous_ninety_eight_outcome_l1_like_case` with their `raw_ev`,
  `estimated_total_fee_susd`, `estimated_net_ev`, `compiler_variant`, and the
  `fee_estimate_source` (`replay_packed_program` vs
  `replay_strict_program` vs `structural_fallback`). Confirm whether the
  4996-action plain plan has higher `estimated_net_ev` than the analytic-mixed
  plan or whether a tiebreaker / ordering bug picks it anyway.
- [ ] **Audit prune/compaction monotonicity**:
  `baseline_step_prune_candidate_for_program_net_ev`,
  `route_group_prune_candidate_for_program_net_ev`, and the analogous
  ForecastFlows prune helpers. When the new estimator now returns `Some(...)`
  for more subsets, verify each pass still guards with `estimated_net_ev`
  non-decreasing.
- [ ] Decide: snapshot regeneration vs planner fix. If regenerate, update
  `test/fixtures/rebalancer_ab_expected.json` (the file loaded by
  `expected_from_fixture()`) with a commit that explicitly notes the
  counter-plan was the cause.

### Out of scope for this patch (per counter-plan doc §"If Byte-Exactness Becomes a Requirement")

- Threading the real executor / `LiveOptimismFeeInputs` / chain id into the
  planner replay path. Current replay fee estimate still uses
  `chain_id = 10`, `nonce = 0`, `executor = Address::ZERO`.
- Replacing the length-only L1 fee estimate with the exact `getL1Fee(bytes)`
  path from `src/execution/gas.rs`.
- Deleting the `#[cfg(test)]` pseudo-address fallback in
  `outcome_token_for_market` (tx_builder.rs:620-628). Safe to remove only
  after all synthetic test callers use an explicit address book.
- Connected-topology ForecastFlows certification (Julia-side worker change).
  Current connected FF row correctly reports `no_certified_candidate`.

### Related but separate

- ForecastFlows underperforms waterfall on single-market by 0.000436 sUSDS
  despite identical calldata (5444 bytes) and action count (14). FF uses
  +137k more L2 gas — action ordering appears suboptimal for warm/cold slot
  patterns. Treat as a solver-ordering follow-up, not a fee-estimator issue.
