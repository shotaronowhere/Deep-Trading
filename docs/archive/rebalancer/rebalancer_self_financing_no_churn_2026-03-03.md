# Self-Financing Mint Execution + Zero-Threshold No-Churn Preserve (2026-03-03)

## Scope

This change adds two runtime behaviors to the off-chain rebalancer:

1. Self-financing mint execution for bundle mint segments (no upfront over-mint vs current cash).
2. Zero-threshold churn avoidance for future-buy markets by preserving them from mint-sell legs.

Both changes are in the Rust off-chain optimizer/execution path.

## 1) Self-Financing Mint Execution

### Problem

Bundle mint execution previously debited `segment.mint_amount` in one shot, then added sell proceeds.
That could model a no-flash path as if it had temporary upfront cash that did not exist.

### New behavior

Bundle mint segments now execute in bounded rounds:

- each round mints at most current available cash,
- immediately sells planned non-frontier legs for that round,
- credits proceeds back to cash,
- repeats until the planned segment mint amount is fully satisfied or a hard feasibility stop is hit.

If the full segment amount cannot be completed, the segment is rolled back atomically.

### Code mapping

- `src/portfolio/core/trading.rs`
  - `ExecutionState::execute_bundle_step(...)` mint branch now uses segment-level atomic round execution.
  - Added:
    - `bundle_mint_round_liquidity_cap(...)`
    - `emit_mint_bundle_round_actions(...)`
    - `execute_mint_bundle_in_rounds(...)`

## 2) Zero-Threshold No-Churn Preserve

### Problem

Mixed mint routes can sell non-frontier legs that are bought later in the same rebalance.
That creates avoidable sell-then-buy fee churn.

### New behavior

No-churn policy is hard (`threshold = 0`) for preserved markets:

1. Run baseline full rebalance once.
2. Build preserve set from markets that:
   - had a direct `Buy` action, and
   - ended with net higher holdings than initial balances (filters transient buy-merge arb legs).
3. Re-run full rebalance with this preserve set.
4. In mint bundle planning, preserved outcomes are excluded from mint-sell leg candidates.

If preserve set is empty (or no mint actions in baseline), baseline actions are returned unchanged.

### Follow-up hardening (same day)

To avoid EV regressions from over-constraining preserve:

1. Preserve-set narrowing:
   - preserved market must be both sold in baseline and net-accumulated by the end.
   - this avoids preserving outcomes that were never part of sell/buy churn.
2. Preserve-aware mint scoring:
   - mint frontier profitability and mint/direct marginal route-switch checks now account for preserved non-sellable legs.
   - prevents selecting mint segments using stale "sell-all-legs" economics.
3. Two-pass EV guard:
   - baseline and preserve rerun are both evaluated with the same EV mark;
   - preserve candidates are now applied via greedy prune/add:
     - start from baseline (empty preserve),
     - try adding one churn candidate at a time,
     - accept only if EV strictly improves (within tolerance),
     - keep the best accepted plan.
   - this keeps no-churn as a selective optimization rather than a hard global constraint.

### Toggle behavior

Greedy EV-guarded churn pruning is now optional and **disabled by default**.

- Default solver path (`rebalance`, `rebalance_with_mode`, `rebalance_with_gas*`) runs without greedy churn pruning.
- Enable explicitly with `RebalanceFlags { enable_ev_guarded_greedy_churn_pruning: true }`.
- Runtime binaries also support env flag:
  - `REBALANCE_ENABLE_GREEDY_CHURN_PRUNING=1`

### Optimality caveat

The EV-guarded preserve selector is a greedy local search, not a global optimizer:

- It only accepts one-market preserve additions that strictly improve EV from the current plan.
- It can reject combinations that need a temporary local EV dip before a later larger gain.
- Therefore the guarantee is non-regression versus baseline under the same EV mark, not global-optimal churn elimination.

### Post-review hardening (2026-03-03)

Two implementation gaps from external review were resolved:

1. Preserve-aware mint scoring now includes preserved-token prediction value in mint profitability ranking.
2. If a mint step fails at execution time (atomic self-financing rollback), waterfall retries a direct-only fallback for the same frontier instead of terminating immediately.

### Code mapping

- `src/portfolio/core/rebalancer.rs`
  - `rebalance_full(...)` now uses a two-pass flow.
  - Added helpers:
    - `build_rebalance_context(...)`
    - `collect_mint_sell_preserve_candidates(...)`
    - `action_plan_expected_value(...)`
    - `run_rebalance_full_with_preserve(...)`
  - Added `RebalanceFlags` and flag-aware public entry points:
    - `rebalance_with_mode_and_flags(...)`
    - `rebalance_with_gas_and_flags(...)`
    - `rebalance_with_gas_pricing_and_flags(...)`
  - `RebalanceContext` now includes:
    - `mint_sell_preserve_markets: HashSet<&'static str>`
  - All waterfall calls in full flow now use preserve-aware entrypoint.

- `src/portfolio/core/waterfall.rs`
  - Added `waterfall_with_execution_gate_and_preserve(...)`.
  - Existing `waterfall_with_execution_gate(...)` is now a wrapper with `None` preserve set.
  - Frontier and planner calls now pass preserve indices.

- `src/portfolio/core/planning.rs`
  - `plan_bundle_step_with_scratch(...)` accepts preserve indices.
  - `build_mint_segment(...)` forwards preserve indices to solver.
  - mint/direct route-switch threshold now uses preserve-aware mint marginal cost.

- `src/portfolio/core/solver.rs`
  - Mint bundle sell-leg discovery and cost functions now skip preserved indices:
    - `mint_bundle_boundary_amount(...)`
    - `mint_bundle_cost_for_amount(...)`
    - `mint_bundle_cost_to_prof(...)`

- `src/portfolio/core/bundle.rs`
  - mint frontier profitability now accounts for preserved non-target outcomes.
  - `mint_bundle_marginal_cost_at_prof(...)` skips preserved sell indices.

- `src/main.rs`, `src/bin/execute.rs`, `src/bin/plan_preview.rs`
  - Parse `REBALANCE_ENABLE_GREEDY_CHURN_PRUNING` and thread `RebalanceFlags`.

## Safeguards

The no-churn policy is hard (no economic threshold), but mechanical safety remains:

- bounded rounds/iterations,
- dust cutoffs (`DUST`/`EPS`),
- fail-closed rollback when a planned mint segment cannot be fully executed.

## Tests Added

- `src/portfolio/core/waterfall.rs`
  - `waterfall_mint_path_never_mints_above_available_cash`
    - validates no-flash self-financing invariant at action level.

- `src/portfolio/tests/oracle.rs`
  - `test_rebalance_preserve_set_blocks_sell_buy_churn_for_preserved_markets`
    - verifies preserve set removes sell/buy overlap for preserved future-buy markets.
  - `sweep_preserve_no_churn_ev_impact` (`#[ignore]`)
    - randomized baseline-vs-preserve EV sweep with diagnostics:
      - EV delta distribution (`preserved - baseline`),
      - changed/overlap case counts,
      - worst/best cases with churn proceeds, mint totals, and action counts,
      - EV-guard simulation metrics (`guard_picks_preserve`, `guard_mean_delta`, `guard_regressions`).

## Verification Run

Validated locally with:

- `cargo fmt`
- `cargo test waterfall_ -- --nocapture`
- `cargo test test_rebalance_with_mode_full_matches_rebalance_default -- --nocapture`
- `cargo test test_rebalance_output_is_groupable -- --nocapture`
- `cargo test test_rebalance_mixed_groups_are_route_coupled -- --nocapture`
- `cargo test test_rebalance_preserve_set_blocks_sell_buy_churn_for_preserved_markets -- --nocapture`
- `PRESERVE_SWEEP_CASES=400 PRESERVE_SWEEP_SEED=20260303 cargo test sweep_preserve_no_churn_ev_impact -- --ignored --nocapture --test-threads=1`
- `PRESERVE_SWEEP_CASES=200 PRESERVE_SWEEP_SEED=20260303 PRESERVE_SWEEP_FORCE_PARTIAL=1 cargo test sweep_preserve_no_churn_ev_impact -- --ignored --nocapture --test-threads=1`
- `PRESERVE_SWEEP_CASES=200 PRESERVE_SWEEP_SEED=20260303 PRESERVE_SWEEP_FORCE_PARTIAL=0 cargo test sweep_preserve_no_churn_ev_impact -- --ignored --nocapture --test-threads=1`
