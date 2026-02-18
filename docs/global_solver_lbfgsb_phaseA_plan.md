# Global Solver L-BFGS-B Phase A Plan

Date: 2026-02-18  
Scope: `src/portfolio/core/global_solver.rs` + diagnostics/test plumbing only  
Status: Canonical implementation spec (doc-first)

## Problem Statement

Current accepted checkpoint (post stage 1-3):

- `total_cases=50`
- `candidate_valid=50`
- `candidate_invalid=0`
- `mean_delta=-22.407627450367`
- `best_delta=0.000000000000`
- `worst_delta=-106.491017066174`

Residual gap decomposition:

- Full-L1 subset (`n=26`): `mean_delta=-43.091591245551`
- Partial-L1 subset (`n=24`): `mean_delta=-0.000000005584`

Observed full-L1 telemetry suggests convergence/conditioning stress:

- Full-L1 `mean_solver_iters=93.462`
- Full-L1 `mean_hold_clamps=8087.731`
- Full-L1 `mean_cash_scales=164.654`

Conclusion for this phase: keep formulation/execution semantics fixed; improve optimizer curvature handling.

## Review Synthesis

Independent reviews: Claude Opus 4.6 + Gemini 3 Pro Preview.

Agreement points (adopted):

1. The current model is likely mathematically sound; primary issue is optimization convergence in full-L1.
2. Diagonal step (`p[i] = -grad[i] / hdiag[i]`) is too weak for strongly coupled `theta`/buy/sell geometry.
3. Active-set-aware bounded quasi-Newton is the right low-risk next step.
4. Dual/decomposition ideas from CFMMRouter are relevant but should be deferred behind Phase A.

Disagreement/overreach handling:

1. Review suggestion to jump immediately to full Hessian or full dual rewrite is deferred.
2. Review suggestions with aggressive parity gates in first milestone are relaxed to balanced improvement gates.
3. Theta regularization removal is not part of Phase A base scope; only considered in conditional Phase A.5.

Adopted recommendations:

- Bounded L-BFGS two-loop recursion over free variables.
- Active/free coordinate mask.
- Descent fallback safety.
- Preserve fail-closed validation and replay contract.

Rejected/deferred recommendations:

- Immediate dual solver as default path.
- Immediate full Hessian implementation.
- Immediate objective/regularization retune before curvature fix isolation.

## Chosen Approach

Phase A default is `L-BFGS-B + active set` because it:

1. Improves curvature use without changing problem formulation.
2. Preserves current action projection/replay validity semantics.
3. Allows explicit fallback to existing diagonal path.
4. Keeps implementation localized to solver internals.

Dual decomposition is deferred to Phase B because it is a structural redesign with higher rollout risk.

## Non-Goals

This phase must NOT change:

1. Objective definition and economic model semantics.
2. Projection-to-actions contract and ordering (`Mint/Sell/Buy/Merge`).
3. Invalidity taxonomy and fail-closed fallback behavior.
4. Replay checks/tolerances and incumbent fallback logic.
5. Warm-start behavior introduced in stage 3.

## Detailed Algorithm

### Configuration

Add to `GlobalSolveConfig`:

- `optimizer: GlobalOptimizer`
- `lbfgs_history: usize` (default `15`)
- `lbfgs_min_curvature: f64` (default `1e-12`)
- `active_set_eps: f64` (default `1e-9`)
- `max_line_search_trials: usize` (default `64`)

Add enum:

- `GlobalOptimizer::DiagonalProjectedNewton`
- `GlobalOptimizer::LbfgsbProjected`

### Iteration Flow (for `run_projected_newton`)

For each outer iteration:

1. Evaluate gradient and apply existing zero-trade clamp.
2. Build free mask:
   - Coordinate is free if not near-active bound using `active_set_eps`, or bound-side gradient indicates inward-descent freedom.
3. Direction selection:
   - If `DiagonalProjectedNewton`: existing rule.
   - If `LbfgsbProjected`:
     - Run L-BFGS two-loop recursion only on free coordinates.
     - Set direction on non-free coordinates to `0`.
4. Descent safeguard:
   - If direction non-finite or `dot(grad, p) >= 0`, fallback to diagonal-scaled negative gradient on free coordinates; non-free remain `0`.
5. Line search:
   - Replace hardcoded trial cap `64` with `cfg.max_line_search_trials`.
   - Keep current Armijo condition and invalid-trial feasibility-repair behavior unchanged.
6. On accepted step:
   - Compute `s = x_new - x_old`.
   - Compute `y = grad_new - grad_old`.
   - Curvature check: store pair only if
     - `s_dot_y > lbfgs_min_curvature * ||s|| * ||y||`.
   - If check fails, skip pair and increment `curvature_skips`.
7. Keep existing convergence check (`projected_grad_norm`) and coupled residual telemetry unchanged.

### L-BFGS State

Internal-only state:

- ring buffers of `s`, `y`, and `rho = 1/(s_dot_y)` with capacity `lbfgs_history`.
- `curvature_skips` counter.

No external behavior changes besides optimizer choice.

## Acceptance Contract

Verification commands:

1. `cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1`
2. `cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture`
3. `cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture`

Balanced Phase A gates (all required):

1. `candidate_valid=50/50`
2. Overall `mean_delta <= -14.0`
3. Full-L1 subset `mean_delta <= -25.0`
4. Full-L1 `mean_solver_iters <= 60`
5. Full-L1 `mean_hold_clamps <= 3000`
6. Invariant suites remain green
7. EV regression runtime <= `1.5x` current accepted baseline runtime

Rollback trigger:

- Any validity or invariant regression, or runtime explosion beyond gate.

## Risk Register

1. **Non-descent L-BFGS directions near boundaries**
   - Mitigation: explicit descent fallback to diagonal-scaled gradient.
2. **Curvature pair instability (`s_dot_y <= 0`)**
   - Mitigation: strict curvature threshold + pair skip counter.
3. **Over-aggressive free-mask**
   - Mitigation: conservative `active_set_eps` default and unit tests for bound-pinned coordinates.
4. **Runtime increase from history updates**
   - Mitigation: small default history (`15`), runtime gate in acceptance contract.
5. **Hidden behavior shift in projection/replay path**
   - Mitigation: keep projection/replay code untouched; enforce invariant and validity gates.

## Rollout Plan

1. Land with both optimizers available by config.
2. Default in `RebalanceConfig::new(...)` moves to `LbfgsbProjected`.
3. Keep one-line fallback to `DiagonalProjectedNewton` for rollback.
4. Ship only if acceptance contract passes.
5. If Phase A misses EV gates but passes safety gates, proceed to Phase A.5:
   - selective weakest-edge buy reduction in feasibility repair,
   - then controlled `theta_l2_reg` sweep (`1e-9`, `1e-10`, `0.0`).

## Implementation Checklist (Decision-Complete)

1. Add config/enum/public exports and defaults.
2. Implement internal `LbfgsState`.
3. Implement free-mask + two-loop recursion path.
4. Add descent fallback guard.
5. Switch line-search trial cap to config field.
6. Add additive diagnostics fields:
   - `candidate_optimizer`
   - `candidate_active_dims`
   - `candidate_curvature_skips`
7. Extend EV regression print line with new fields.
8. Add solver unit tests:
   - `lbfgs_direction_is_descent_or_falls_back`
   - `active_set_zeroes_bound_pinned_coordinates`
   - `curvature_guard_skips_bad_pairs`
   - `history_bounded_by_lbfgs_history`
9. Run full verification matrix and record results in docs.
