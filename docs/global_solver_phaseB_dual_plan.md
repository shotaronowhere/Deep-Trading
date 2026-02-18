# Global Solver Phase B: Dual/Decomposition Prototype Plan

Date: 2026-02-18  
Scope: `src/portfolio/core/global_solver_dual.rs` behind config flag  
Status: Implementation plan (doc-first)

## Problem Context

Phase A + A.5 recovered solver validity and invariants:

- `candidate_valid=50/50`
- `candidate_invalid=0`
- overall `mean_delta=-19.934197985691`

Residual EV gap remains concentrated in full-L1:

- full-L1 `mean_delta=-38.334996121988` (26-case subset)
- incumbent still wins in every compared case (`candidate_better=0`)

Interpretation: safety and feasibility are good; economic optimality still lags incumbent in coupled full-L1 regimes.

## Phase B Goals

1. Add a structural alternative path (dual/decomposition prototype) without changing default runtime path.
2. Keep fail-closed semantics, projection contract, and replay checks unchanged.
3. Provide experimental mode that can be tuned/evaluated independently.

## Non-Goals

1. No default switch away from `LbfgsbProjected` in this phase.
2. No changes to action grammar (`FlashLoan -> Mint -> Sell+ -> Repay`, direct buys/sells, merge).
3. No change to validity taxonomy or replay tolerances.
4. No objective-function redesign in this phase.

## Proposed Prototype

Phase B introduces a configurable dual-style outer loop that treats the existing primal solve as a subproblem oracle.

At a high level:

1. Run a primal solve with `optimizer=LbfgsbProjected` and modified regularization (`theta_l2_reg + lambda`).
2. Use solved `theta` as a coupling signal.
3. Update dual variable `lambda` with damped projected steps.
4. Repeat for bounded outer iterations.
5. Select best valid candidate by replay EV.

This is explicitly a prototype decomposition path, not the final dual architecture.

## Config Additions

Add to `GlobalSolveConfig`:

- `dual_outer_iters: usize`
- `dual_lambda_step: f64`
- `dual_lambda_decay: f64`
- `dual_theta_tolerance: f64`

Add `GlobalOptimizer::DualDecompositionPrototype`.

Defaults are conservative and keep current behavior:

- default optimizer remains `LbfgsbProjected`
- dual fields are ignored unless `optimizer=DualDecompositionPrototype`

## Module and API Shape

New module:

- `src/portfolio/core/global_solver_dual.rs`

Primary entry:

- `build_global_candidate_plan_dual(...) -> Option<GlobalCandidatePlan>`

Integration:

- `build_global_candidate_plan(...)` in `global_solver.rs` dispatches by optimizer:
  - `DiagonalProjectedNewton` -> primal path
  - `LbfgsbProjected` -> primal path
  - `DualDecompositionPrototype` -> Phase B path

## Algorithm Details

For each outer dual iteration:

1. Build subproblem config:
   - force primal optimizer to `LbfgsbProjected`
   - set `theta_l2_reg = cfg.theta_l2_reg + lambda`
2. Solve primal candidate using current warm start.
3. Compute replay EV for selection.
4. If valid and EV improves, keep as best candidate.
5. Update dual variable:
   - `lambda <- max(0, lambda + step * theta)`
   - `step <- step * dual_lambda_decay`
6. Stop early when `|theta| <= dual_theta_tolerance`.

Fallback behavior:

- If no valid dual iterate exists, return best available primal attempt.
- If all attempts unavailable, return `None` and preserve existing incumbent fallback behavior.

## Acceptance Contract (Phase B Prototype)

Safety gates (must hold):

1. validity/invariant regressions are not allowed.
2. default path (`LbfgsbProjected`) metrics must be unchanged when dual optimizer is not selected.

Prototype performance target (dual mode only):

1. maintain `candidate_valid=50/50` on EV regression corpus.
2. improve EV vs current Phase A.5 reference:
   - overall mean delta better than `-19.934197985691`
   - full-L1 mean delta better than `-38.334996121988`

Stretch trigger for promoting beyond prototype:

- approach Phase B target corridor:
  - overall `mean_delta <= -5.0`
  - full-L1 `mean_delta <= -10.0`

## Rollout

1. Land behind optimizer flag only.
2. Keep `RebalanceConfig::new` default optimizer = `LbfgsbProjected`.
3. Evaluate dual mode in regression harness before any default change discussion.
