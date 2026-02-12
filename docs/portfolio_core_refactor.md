# Portfolio Core Refactor Notes (2026-02-12)

## Scope
Implemented the high-impact refactors selected from review feedback, plus the clone-allocation optimization in profitability solving.

## Changes Implemented

### 1) Split merge/sell math out of trading orchestration
- Added `src/portfolio/core/merge.rs` (renamed from `merge_math.rs` in final cleanup).
- Moved merge/direct split math and merge execution helpers out of `trading.rs`.
- Kept `trading.rs` focused on execution orchestration and accounting.

### 2) Rebalancer phase flow moved into context methods
- Updated `src/portfolio/core/rebalancer.rs`.
- Converted phase helpers into `RebalanceContext` methods:
  - `run_phase1_sell_overpriced`
  - `run_phase3_recycling`
- This reduced long parameter lists and keeps phase logic colocated with context state.

### 3) Centralized numeric tolerances
- Added shared tolerances in `src/portfolio/core/sim.rs`:
  - `EPS = 1e-12`
  - `DUST = 1e-18`
- Replaced remaining hardcoded epsilon checks in rebalancer flow with `EPS`.

### 4) Clone optimization for affordability checks
- Updated planning and solver:
  - `src/portfolio/core/planning.rs`
  - `src/portfolio/core/solver.rs`
- Added `plan_active_routes_with_scratch(...)` and reused a scratch `Vec<PoolSim>` during bisection affordability checks.
- This avoids repeated fresh vector clone allocation on each probe.

### 5) Minor follow-up cleanup
- Removed unused import in `src/portfolio/core/solver.rs` introduced during refactor.

### 6) Non-functional readability/layout refactor (no behavior changes)
- Updated `src/portfolio/core/rebalancer.rs` to reduce repetition and improve structure:
  - Added module-level phase constants.
  - Extracted shared balance helpers (`held_total`, `held_legacy`).
  - Added `Phase3Trial` helper struct to group speculative state.
  - Split repeated execution snippets into context/trial methods.
  - Kept all phase decision logic and thresholds unchanged.

### 7) Removed planning/solver circular dependency
- Moved affordability solve (`solve_prof`) into `src/portfolio/core/planning.rs`.
- `src/portfolio/core/solver.rs` is now numerical mint-cost logic only.
- Updated `src/portfolio/core/waterfall.rs` to consume `solve_prof` from planning.

### 8) Moved action command type out of simulation module
- Added `src/portfolio/core/types.rs` and moved `Action` enum there.
- Re-exported from `src/portfolio/core/mod.rs`.
- `sim.rs` now focuses on pool simulation math/state only.

### 9) Standardized execution context usage
- Extended `ExecutionState` in `src/portfolio/core/trading.rs` with an optional balances handle.
- Added `ExecutionState::with_balances(...)` so sell execution no longer threads balances as a separate argument.
- Updated rebalancer and relevant tests to use the unified execution context pattern.

### 10) Extracted reusable test fixtures
- Added `src/portfolio/tests/fixtures.rs` with shared fixture builders:
  - `mock_slot0_market*`
  - `build_three_sims*`
  - pool/market leak helpers
- `src/portfolio/tests.rs` now imports fixture helpers instead of defining all of them inline.

### 11) Final structure cleanup from review pass
- Centralized `BalanceMap` in `src/portfolio/core/types.rs` and removed duplicate aliases from `rebalancer.rs` and `trading.rs`.
- Replaced test wildcard imports with explicit imports in:
  - `src/portfolio/tests.rs`
  - `src/portfolio/tests/execution.rs`
  - `src/portfolio/tests/oracle.rs`
  - `src/portfolio/tests/fuzz_rebalance.rs`
- Kept `use fixtures::*;` intentionally as a narrow, dedicated fixture surface.

### 12) Additional cleanup from follow-up review
- Consolidated balance-manipulation helpers into `src/portfolio/core/types.rs`:
  - `lookup_balance`
  - `subtract_balance`
  - `apply_actions_to_sim_balances`
- Updated imports/callers in:
  - `src/portfolio/core/merge.rs`
  - `src/portfolio/core/trading.rs`
  - `src/portfolio/core/rebalancer.rs`
- Replaced hardcoded bisection loop bound in `src/portfolio/core/planning.rs` with `SOLVE_PROF_ITERS`.
- Removed per-call `Vec` allocation in `action_contract_pair` (`src/portfolio/core/merge.rs`) while preserving output behavior (first two lexicographically smallest unique contract IDs).

### 13) Final deduplication pass from comprehensive review
- Deduplicated action replay logic in tests:
  - `replay_actions_to_ev` (`src/portfolio/tests.rs`) now delegates to `replay_actions_to_state` + `ev_from_state`.
- Deduplicated merge execution paths in core:
  - `execute_merge_sell` (`src/portfolio/core/merge.rs`, test helper) now wraps `execute_merge_sell_with_inventory` using an empty inventory map and `f64::INFINITY` keep-threshold.

### 14) Planning hot-path allocation removal
- Reworked `execution_order` in `src/portfolio/core/planning.rs` to return an iterator chain instead of allocating a temporary `Vec<(usize, Route)>` on each call.
- Preserved mint-first then direct execution semantics while removing per-probe heap allocation inside `solve_prof` affordability bisection.

## Verification
Ran after refactor:
- `cargo check --lib` ✅
- `cargo test --lib` ✅ (88 passed, 0 failed, 2 ignored)
- `cargo clippy --lib` ✅ (clean for the touched refactor paths)

## Code/Doc Consistency Check
- Module boundaries described above match code layout (`merge`, `trading`, `rebalancer`, `planning`, `solver`).
- Verification outcomes reflect current build/test status after these changes.
