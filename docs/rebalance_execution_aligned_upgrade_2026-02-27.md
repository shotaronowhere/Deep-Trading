# Rebalance Execution-Aligned Upgrade (2026-02-27)

This note records the implementation status of the phased execution-alignment upgrade.

## Scope Delivered

Implemented in code:

1. Objective alignment with execution gates across emission points.
2. Full-mode two-sided complete-set arbitrage pre-pass.
3. Waterfall tail de-fragmentation via per-step sub-gas pruning.
4. Additive gas-aware API with explicit pricing inputs.
5. Experimental exact mixed-route solver behind runtime gate.
6. Runtime observability counters and summary logging.

Non-goal preservation:

- No contract ABI changes.
- No `Action` enum shape changes.
- No strict-step submission loop rewrite.

## API Changes

Added:

- `rebalance_with_gas_pricing(..., gas_price_eth, eth_usd) -> Vec<Action>`

Preserved compatibility:

- `rebalance_with_gas(...)` remains and wraps `rebalance_with_gas_pricing(...)` with defaults:
  - `gas_price_eth = 1e-9`
  - `eth_usd = 3000.0`

Runtime env support:

- `ETH_USD` (optional override, default `3000`) in:
  - `src/main.rs`
  - `src/bin/execute.rs`

## Implementation Map by Phase

### Phase 0 (Baseline Lock)

- Baseline doc recorded at:
  - `docs/rebalance_execution_alignment_baseline_2026-02-27.md`

### Phase 1 (Execution Gate Alignment)

Implemented in `src/portfolio/core/rebalancer.rs`:

- New internal threshold carrier:
  - `RouteGateThresholds { direct_buy, mint_sell, direct_sell, buy_merge, direct_merge, buffer_frac, buffer_min_susd }`
- New shared gate helper:
  - `passes_execution_gate(edge_susd, gas_susd, buffer_frac, buffer_min_susd)`
- Thresholds computed for all required route kinds via `estimate_min_gas_susd_for_group(...)`.
- Gate checks applied in:
  - phase-1 liquidation
  - phase-3 recycling
  - cleanup waterfall passes (through waterfall gate-aware entrypoint)
- Non-finite threshold sanitization to fail-closed route blocking.

### Phase 2 (Two-Sided Complete-Set Arb)

In full mode pre-pass:

- runtime gas-gated path uses `execute_two_sided_complete_set_arb()`.
- zero-threshold compatibility path keeps legacy `execute_complete_set_arb()` behavior.

### Phase 3 (Waterfall Tail De-Fragmentation)

Implemented in `src/portfolio/core/waterfall.rs`:

- Added `waterfall_with_execution_gate(...)`.
- Added `WaterfallGateStats`.
- Per-step approximate edge:
  - `approx_edge_susd = step.cost * max(current_prof, 0.0)`
- Sub-gas failing steps are pruned from active set and descent replans.
- Existing `waterfall(...)` kept as compatibility wrapper with zero gate/buffer.

### Phase 4 (Experimental Mixed Solver)

Implemented in `src/portfolio/core/planning.rs`:

- Runtime flag:
  - `REBALANCE_EXACT_MIXED_SOLVER=1`
- Experimental coupled mixed solver path with hard fallback to existing bisection on failure/non-convergence.
- Default behavior unchanged when flag is unset.

### Phase 5 (Observability / Safety Rails)

Implemented in `src/portfolio/core/rebalancer.rs`:

- Counters:
  - `skipped_by_gate_direct_buy`
  - `skipped_by_gate_mint_sell`
  - `skipped_by_gate_direct_sell`
  - `skipped_by_gate_buy_merge`
  - `skipped_by_gate_direct_merge`
  - `waterfall_steps_pruned_subgas`
  - `phase1_candidates_skipped_subgas`
  - `phase3_candidates_skipped_subgas`
- Per-run summary logging in rebalance flow.

Implemented in `src/bin/execute.rs`:

- Empty-plan diagnostics when actions are non-empty but strict plans are empty.

## Tests Added

### Oracle tests (`src/portfolio/tests/oracle.rs`)

- `phase1_skips_subgas_liquidation_runtime_thresholds`
- `phase3_skips_subgas_recycling_runtime_thresholds`
- `full_mode_two_sided_arb_executes_when_price_sum_above_one`

### Waterfall tests (`src/portfolio/core/waterfall.rs`)

- `waterfall_prunes_subgas_steps`
- `waterfall_subgas_prune_does_not_emit_nonfinite_or_overspend`

### Bounds integration (`src/execution/bounds.rs`)

- `runtime_actions_produce_nonempty_plan_prefix_under_realistic_gas`

### Performance regression (`src/portfolio/tests/execution.rs`)

- `test_rebalance_perf_full_l1_with_gas_pricing`

## Verification Run (Targeted)

Executed and passing after implementation:

- `cargo check`
- `cargo test skips_subgas -- --nocapture`
- `cargo test full_mode_two_sided_arb_executes_when_price_sum_above_one -- --nocapture`
- `cargo test waterfall_prunes_subgas_steps -- --nocapture`
- `cargo test waterfall_subgas_prune_does_not_emit_nonfinite_or_overspend -- --nocapture`
- `cargo test runtime_actions_produce_nonempty_plan_prefix_under_realistic_gas -- --nocapture`
- `cargo test test_rebalance_perf_full_l1_with_gas_pricing -- --nocapture`

## Code vs Docs Verification

Manual alignment check completed:

1. `docs/waterfall.md` updated for:
   - `rebalance_with_gas_pricing` API
   - two-sided full-mode pre-pass
   - execution-meaningful waterfall step pruning
   - experimental mixed solver gate
2. `docs/portfolio.md` updated for:
   - additive gas-aware API + `ETH_USD`
   - two-sided full-mode phase-0 behavior
   - route gate alignment semantics
3. `docs/gas_model.md` updated for:
   - expanded route thresholds (`DirectSell`, `BuyMerge`, `DirectMerge`)
   - shared runtime gate formula and buffer semantics
   - explicit pricing/default/env behavior

This note and the baseline doc together form the acceptance comparison record for phases 1-3.

## Post-Review Hardening (Gemini/Claude Follow-Up)

Additional safety fixes applied after external model review:

1. Waterfall sub-gas prune tombstoning:
   - pruned `(idx, route)` entries are no longer removed from `active_set` during the same descent.
   - this prevents immediate re-selection loops at the same profitability frontier.
2. Batch router ERC20 call hardening:
   - switched to optional-return-safe low-level ERC20 wrappers (`transferFrom`, `transfer`, `approve`).
   - approval path now uses force-approve semantics (retry via `approve(0)` then target amount).
3. Batch router ETH refund reception:
   - replaced placeholder `execute()` with `receive() external payable`.

Verification rerun after these fixes:
- `cargo check`
- EV regression suites (fast + full ignored + partial ignored)
- sub-gas oracle/waterfall tests
- full-mode two-sided arb oracle test
- `forge test` suites:
  - `test/BatchSwapRouter.t.sol`
  - `test/BatchSwapRouterBranches.t.sol`
  - `test/BatchSwapRouterFuzz.t.sol`
