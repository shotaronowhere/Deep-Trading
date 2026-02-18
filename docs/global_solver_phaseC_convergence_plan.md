# Global Solver Phase C: Convergence Recovery Plan

Date: 2026-02-18  
Scope: `src/portfolio/core/global_solver.rs` (primal solver internals only)  
Status: Doc-first implementation spec

## Problem Statement

Current accepted baseline (Phase A.5 default path):

- `candidate_valid=50/50`
- `candidate_invalid=0`
- `mean_delta=-19.934197985691`
- full-L1 (`n=26`) `mean_delta=-38.334996121988`
- partial-L1 (`n=24`) `mean_delta=-0.000000004702`

The safety contract is stable, but EV remains materially below incumbent in full-L1.

## Convergence Question: Is this just iteration budget?

No. New controlled sweeps show EV is not improving by “just running longer”.

### Evidence (same 50-case EV corpus)

- Baseline (default): `mean_delta=-19.934197985691`
- `GLOBAL_SOLVER_MAX_ITERS=4096`: `mean_delta=-19.934197985691` (no change)
- `GLOBAL_SOLVER_MAX_LINE_SEARCH_TRIALS=256`: `mean_delta=-19.934197985691` (no change)
- `GLOBAL_SOLVER_THETA_L2_REG=1e-10`: `mean_delta=-19.934186317592` (noise-level change)
- `GLOBAL_SOLVER_THETA_L2_REG=0.0`: `mean_delta=-19.934189284719` (noise-level change)
- `GLOBAL_SOLVER_ACTIVE_SET_EPS=1e-6`: `mean_delta=-19.934201065368` (no improvement)
- `GLOBAL_SOLVER_ACTIVE_SET_EPS=1e-4`: `mean_delta=-19.934211883979` (slightly worse)

Observed pattern:

- final `pg_norm` remains around `1e-2` on many full-L1 cases (far above `pg_tol=1e-5`)
- solver stops with low/moderate outer iterations (`~11` mean full-L1), usually after one failed line search
- increasing iteration/line-search budget does not change the final point

Conclusion: this is a descent/recovery issue, not a simple budget issue.

## Root-Cause Hypothesis

The solver frequently exits when Armijo search fails from the current quasi-Newton direction.  
Current behavior is immediate break on failure, even when projected gradient remains high.  
This causes premature termination in non-stationary states.

## Phase C Goals

1. Improve numerical convergence quality without changing objective model or projection semantics.
2. Preserve fail-closed behavior and replay validity contract.
3. Keep implementation bounded and deterministic.

## Non-Goals

1. No projection contract changes.
2. No replay tolerance changes.
3. No action grammar changes.
4. No default switch to dual/decomposition path.

## Chosen Approach

Add a bounded line-search rescue path:

1. Keep current Armijo path unchanged as first attempt.
2. If Armijo fails:
   - clear L-BFGS history for this iteration,
   - try a projected-gradient rescue direction,
   - accept only if objective strictly decreases by a minimum threshold.
3. If rescue also fails after bounded retries, terminate as before.

This addresses premature termination while preserving monotone descent and bounded runtime.

## Detailed Algorithm Changes

In `run_projected_newton(...)`:

1. Keep main direction construction and Armijo loop as-is.
2. On main loop failure (`accepted=false`):
   - increment `line_search_rescue_attempts`,
   - clear `lbfgs_state` (drop stale curvature),
   - compute rescue direction from scaled negative gradient on active/free mask,
   - run bounded rescue backtracking loop.
3. Rescue acceptance rule:
   - feasibility remains required (same projection + feasibility repair path),
   - accept only when `eval_trial.f <= eval.f - rescue_min_decrease * (1 + |eval.f|)`.
4. On rescue accept:
   - update `(x, eval, objective_trace)` and continue outer iterations,
   - increment `line_search_rescue_accepts`.
5. On rescue failure:
   - keep current fail-closed termination behavior.

## API / Config Additions

Add to `GlobalSolveConfig`:

- `line_search_rescue_trials: usize` (default `16`)
- `line_search_rescue_min_decrease: f64` (default `1e-12`)

## Diagnostics (Additive)

Add to solve diagnostics structs:

- `line_search_rescue_attempts`
- `line_search_rescue_accepts`

Propagate into rebalancer diagnostics and EV compare print line.

## Test Plan

Add/extend tests in `src/portfolio/core/global_solver.rs`:

1. rescue path activates after Armijo failure.
2. rescue path is bounded and cannot loop indefinitely.
3. objective trace stays monotone non-increasing under accepted steps.

Keep existing solver and invariant tests unchanged.

## Acceptance Contract

Safety (must hold):

1. `candidate_valid=50/50`
2. `candidate_invalid=0`
3. invariant suites remain green

Convergence/perf targets (Phase C):

1. improve EV vs current default baseline:
   - overall `mean_delta < -19.934197985691` (strictly better)
   - full-L1 `mean_delta < -38.334996121988` (strictly better)
2. runtime of EV compare test remains <= `1.5x` accepted baseline

## Rollback Triggers

Rollback Phase C defaults if any occurs:

1. validity drops below `50/50`,
2. invariants regress,
3. runtime exceeds `1.5x`,
4. EV worsens beyond noise (overall or full-L1).

## Phase C Exit Criteria

If Phase C improves EV but still cannot produce `candidate_better > 0`, proceed to Phase D (structural constrained reparameterization for coupled sell/hold feasibility) behind flag.
