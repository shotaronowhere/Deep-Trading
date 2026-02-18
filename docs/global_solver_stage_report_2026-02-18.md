# Global Solver Stage Report (2026-02-18)

## Scope

- Path: `src/portfolio/core/global_solver.rs`, `src/portfolio/core/rebalancer.rs`, `src/portfolio/tests/fuzz_rebalance.rs`
- Goal: reduce EV regression while keeping fail-closed validity.
- Regression command:

```bash
cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
```

## Recovery Addendum (Late 2026-02-18)

This report’s historical sections remain accurate for earlier phases, but the latest recovery checkpoint materially improved EV.

Latest benchmark artifact:

- `/tmp/global_ev_default_1771415895866.jsonl`

Latest aggregate results:

- `candidate_valid=50/50`
- `candidate_invalid=0`
- overall `mean_delta=-0.422000933261`
- full-L1 overall (`n=26`, fuzz + regression): `mean_delta=-0.670778756323`
- full-L1 fuzz (`n=24`): `mean_delta=-0.801324117995`
- partial-L1 (`n=24`): `mean_delta=-0.152491624944`
- regression full-L1 snapshots (`n=2`): `mean_delta=0.895765583734`
- `candidate_better=3/50`

Cases where global candidate beat incumbent:

- `fuzz_full_l1_case_0` (`delta=+0.888289612926`)
- `fuzz_full_l1_case_18` (`delta=+1.387274542502`)
- `regression_full_l1_snapshot` (`delta=+1.856221833414`)

Key implementation changes that produced this recovery:

1. Candidate-only path retained in `RebalanceEngine::GlobalCandidate` (no incumbent EV mixing at selection).
2. Route-faithful post-solve refinement in `global_solver` now includes:
   - complete-set arb pre-pass,
   - merge-aware phase-1 liquidation via `ExecutionState::execute_optimal_sell`,
   - EV-guarded phase-3 recycle loop,
   - EV-guarded polish reoptimization loop,
   - bounded cleanup sweeps.
3. Candidate now keeps whichever action set has higher replay EV:
   - projected primal actions, or
   - independent route-refinement actions.

Safety verification after recovery update:

- `cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture` -> pass
- `cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants` -> pass

## Phase A (L-BFGS-B) + Phase A.5 Execution Record

### Implemented in this cycle

1. Added optimizer/config surface:
   - `GlobalOptimizer::{DiagonalProjectedNewton, LbfgsbProjected}`
   - `GlobalSolveConfig` additions:
     - `optimizer`
     - `lbfgs_history`
     - `lbfgs_min_curvature`
     - `active_set_eps`
     - `max_line_search_trials`
2. Added bounded L-BFGS state with curvature guard:
   - bounded `(s, y, rho)` history
   - curvature skip counter
3. Added active-set projected direction path:
   - free-mask from bound proximity + gradient sign
   - two-loop recursion on free coordinates
   - non-free coordinates forced to zero direction
4. Added descent safeguard:
   - fallback to diagonal-scaled negative gradient on free set when direction is non-descent/non-finite
5. Kept all non-goals unchanged:
   - objective model
   - projection contract
   - replay/fail-closed validity semantics
6. Added telemetry:
   - `candidate_optimizer`
   - `candidate_active_dims`
   - `candidate_curvature_skips`
   - EV regression print line includes these fields
7. Added tests:
   - `lbfgs_direction_is_descent_or_falls_back`
   - `active_set_zeroes_bound_pinned_coordinates`
   - `curvature_guard_skips_bad_pairs`
   - `history_bounded_by_lbfgs_history`

### Phase A gate check (before A.5)

From `/tmp/ev_compare_phaseA.log`:

- `total_cases=50`
- `candidate_valid=45`
- `candidate_invalid=5`
- `mean_delta=-27.497830759407`
- invalid reasons: `ProjectionUnavailable=5`
- full-L1 subset:
  - `mean_delta=-52.880443763401`
  - `mean_solver_iters=14.923077`
  - `mean_hold_clamps=5487.538462`

Gate result: failed on `candidate_valid` and full-L1 `mean_hold_clamps`.

### Phase A.5 change applied

Feasibility repair changed from uniform buy scaling to selective weakest-edge buy reduction:

- ranking key: local marginal edge quality (`prediction - d(cost)/d(buy)`)
- reduce weakest buys first to recover cash feasibility
- keep feasibility clamping contract unchanged

Then added tolerance-aware hold-clamp accounting for tiny numerical violations:

- clamp state always enforced
- `hold_clamps` counter increments only when violation exceeds `solver_budget_eps`-scaled tolerance

### Final verification after A.5

Regression log: `/tmp/ev_compare_phaseA5b.log`

- Overall:
  - `total_cases=50`
  - `candidate_valid=50`
  - `candidate_invalid=0`
  - `mean_delta=-19.934197985691`
- Full-L1 subset (`26` cases):
  - `full_l1_valid=26`
  - `full_l1_invalid=0`
  - `mean_delta=-38.334996121988`
  - `mean_solver_iters=11.115385`
  - `mean_hold_clamps=2450.192308`
- Runtime:
  - EV regression test wall-clock: `31.59s` (from test output), no runtime explosion

Invariant suites:

- `cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture` -> pass
- `cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture` -> pass

### Acceptance contract result

1. `candidate_valid=50/50` -> pass
2. overall `mean_delta <= -14.0` -> pass
3. full-L1 `mean_delta <= -25.0` -> pass
4. full-L1 `mean_solver_iters <= 60` -> pass
5. full-L1 `mean_hold_clamps <= 3000` -> pass
6. invariants green -> pass
7. runtime did not regress materially in observed runs -> pass

Conclusion: Phase A + conditional Phase A.5 accepted; theta sweep (`1e-9`, `1e-10`, `0.0`) not required.

## Baseline Before This Stage Set

- `total_cases=50`
- `candidate_valid=2`
- `candidate_invalid=48`
- `mean_delta=-42.910196918399`
- Invalid reasons:
  - `ProjectedGradientTooLarge=44`
  - `ProjectionUnavailable=4`

## Stage 1-3 Changes (Implemented)

## Stage 1: Barrier and Feasibility Retune

- Added `barrier_shift` to `GlobalSolveConfig` and objective barrier terms.
- Lowered defaults:
  - `barrier_mu_cash` -> `1e-8`
  - `barrier_mu_hold` -> `1e-8`
  - `barrier_shift` -> `1e-4`
- Reduced feasibility-repair cash target from the previous conservative target to:
  - `cash_target = max(10*solver_budget_eps, 10*DOMAIN_EPS)`
- Line search now tries trial objective first and only runs coupled-feasibility restore on invalid trial states.

## Stage 2: Coupled Residual Gating

- Added `coupled_feasibility_residual` metric to solve trace/result.
- Added `candidate_coupled_residual` to diagnostics and EV-compare output.
- Updated validity gate:
  - High projected gradient alone no longer auto-invalidates.
  - `ProjectedGradientTooLarge` now requires both high projected gradient and high coupled residual.

## Stage 3: Warm Start from Incumbent

- Added warm-start plumbing:
  - `warm_start_from_actions(actions, sims)` maps incumbent actions into solver variables `(buy, sell, theta)`.
  - `run_projected_newton(..., warm_start: Option<&[f64]>, ...)`
- Candidate build now receives `warm_start_actions: Option<&[Action]>`.
- If warm-started solve fails, solver retries from unseeded start.
- Rebalancer passes incumbent actions as warm start for:
  - `RebalanceEngine::GlobalCandidate`
  - `RebalanceEngine::AutoBestReplay`

## Verification After Stage 1-3

- EV regression:
  - `total_cases=50`
  - `candidate_valid=50`
  - `candidate_invalid=0`
  - `mean_delta=-22.407627450367`
  - `best_delta=0.000000000000`
  - `worst_delta=-106.491017066174`
- Invariants:
  - `cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture` -> pass
  - `cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture` -> pass

## Residual EV Gap Decomposition (From `/tmp/global_ev_latest.log`)

- Full-L1 subset (`24 fuzz full + 2 full snapshots`, `n=26`):
  - `mean_delta=-43.091591245551`
  - `best_delta=0.000000000000`
  - `worst_delta=-106.491017066174`
  - `mean_solver_iters=93.462`
  - `iter_cap_count=2`
  - `mean_feasibility_repairs=165.000`
  - `mean_hold_clamps=8087.731`
  - `mean_cash_scales=164.654`
  - `mean_pg_norm=0.014229655`, `max_pg_norm=0.018022652`
- Partial-L1 subset (`n=24`):
  - `mean_delta=-0.000000005584` (near parity)
  - `mean_solver_iters=6.167`
  - `mean_feasibility_repairs=172.500`
  - `mean_hold_clamps=0.000`
  - `mean_cash_scales=172.500`

## Interpretation

- The remaining EV deficit is concentrated in complete-set-enabled full-L1 cases, not in partial-L1.
- Coupled residual is consistently zero, so this is not currently a primal-feasibility validity failure.
- The solver is execution-faithful and valid, but still underperforms incumbent EV in full-L1 economic regimes.

## Thesis/Routing Context Applied

- Dual-decomposition framing and coupling constraint logic:
  - `docs/routing-algorithm.md:133`
  - `docs/routing-algorithm.md:137`
  - `docs/routing-algorithm.md:215`
  - `docs/routing-algorithm.md:226`
- Nondifferentiable/subgradient caveat and primal-feasibility restoration:
  - `docs/phd_thesis.md:1203`
  - `docs/phd_thesis.md:1230`
  - `docs/phd_thesis.md:1676`
- Boundary-optimal behavior and nonnegative subgradients for monotone utilities:
  - `docs/phd_thesis.md:353`
  - `docs/phd_thesis.md:382`
- Translation-invariance reference (relevant to complete-set gauge handling):
  - `docs/phd_thesis.md:1353`
  - `docs/phd_thesis.md:3783`
- On-chain realism caveat (stale trading sets / robust routing motivation):
  - `docs/routing-algorithm.md:404`

## Stage 4 Experiment (Rejected)

- Attempted change: coupled projection in each line-search trial to enforce sell/hold/theta coupling pre-evaluation.
- Full-run regression outcome:
  - `mean_delta=-45.841560299366`
  - `best_delta=0.000000000000`
  - `worst_delta=-173.950186182147`
  - runtime increased to ~156s (vs ~34s post stage 1-3)
- Observed per-case failure pattern:
  - Case 0: `delta=-106.383377`, `solver_iters=1024`, `ls_trials=10319`
  - Case 1: `delta=-98.522320`, `solver_iters=1024`, `ls_trials=10758`
  - Case 2: `delta=-92.092463`, `solver_iters=1024`, `ls_trials=10719`
- Pattern: near-universal iteration cap with large line-search churn and substantially worse EV.
- Decision: reverted immediately; not retained.

## Next Stages

1. Add complete-set-specific solve telemetry (theta trajectory and per-iteration hold-barrier activity) to isolate EV loss mechanism in full-L1.
2. Introduce incumbent-EV parity guard in diagnostics-only mode first (no behavior change), to quantify candidate-under-incumbent frequency by regime.
3. Implement full-L1 focused objective/regularization tuning only if telemetry identifies a stable failure mode (not by broad parameter sweeps).

## Phase B Prototype Update (2026-02-18)

Doc-first spec:

- `docs/global_solver_phaseB_dual_plan.md`

Code landed behind flag:

- new optimizer variant: `GlobalOptimizer::DualDecompositionPrototype`
- new module: `src/portfolio/core/global_solver_dual.rs`
- new config fields:
  - `dual_outer_iters`
  - `dual_lambda_step`
  - `dual_lambda_decay`
  - `dual_theta_tolerance`
- default optimizer remains unchanged (`LbfgsbProjected`)

Benchmark commands:

```bash
# default path (unchanged baseline check)
cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1

# dual prototype path (override in harness)
GLOBAL_SOLVER_OPTIMIZER=dual cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
```

Results:

- Default path (post Phase B code, no override):
  - `candidate_valid=50/50`
  - `mean_delta=-19.934197985691`
  - unchanged from accepted Phase A.5 baseline
- Dual prototype override:
  - overall:
    - `candidate_valid=50/50`
    - `mean_delta=-17.550977387382` (improved vs `-19.934197985691`)
  - full-L1 subset (`n=26`):
    - `mean_delta=-33.751879586778` (improved vs `-38.334996121988`)
    - `mean_solver_iters=13.807692`
    - `mean_hold_clamps=3349.153846`

Interpretation:

- Phase B prototype improves EV relative to Phase A.5 while preserving validity.
- It is still not converged economically (`candidate_better=0/50`, negative mean deltas).
- It is not ready for default promotion.

## Phase C Convergence Recovery Update (2026-02-18)

Doc-first spec:

- `docs/global_solver_phaseC_convergence_plan.md`

### Pre-implementation sweep evidence

These runs were used to answer whether "run longer" helps:

- Baseline: `mean_delta=-19.934197985691`
- `GLOBAL_SOLVER_MAX_ITERS=4096`: `mean_delta=-19.934197985691` (no change)
- `GLOBAL_SOLVER_MAX_LINE_SEARCH_TRIALS=256`: `mean_delta=-19.934197985691` (no change)
- `GLOBAL_SOLVER_THETA_L2_REG=1e-10`: `mean_delta=-19.934186317592` (noise-level)
- `GLOBAL_SOLVER_THETA_L2_REG=0.0`: `mean_delta=-19.934189284719` (noise-level)
- `GLOBAL_SOLVER_ACTIVE_SET_EPS=1e-6`: `mean_delta=-19.934201065368` (no improvement)
- `GLOBAL_SOLVER_ACTIVE_SET_EPS=1e-4`: `mean_delta=-19.934211883979` (worse)

Conclusion: EV was not iteration-budget-limited.

### Phase C implementation

- Added bounded rescue path after Armijo failure in `run_projected_newton(...)`:
  - clear L-BFGS history,
  - attempt projected-gradient rescue backtracking,
  - accept only on strict objective decrease.
- Added config fields:
  - `line_search_rescue_trials` (default `16`)
  - `line_search_rescue_min_decrease` (default `1e-12`)
- Added diagnostics:
  - `line_search_rescue_attempts`
  - `line_search_rescue_accepts`
- Propagated rescue telemetry into:
  - `GlobalSolveResult`
  - `RebalanceDecisionDiagnostics`
  - EV compare per-case print line.
- Added solver unit tests:
  - `line_search_rescue_recovers_from_over_strict_armijo`
  - `line_search_rescue_is_bounded_when_no_step_is_acceptable`

### Verification

- `cargo test portfolio::core::global_solver::tests:: -- --nocapture` -> pass (12/12)
- `cargo test test_global_candidate_dual_prototype_mode_runs_fail_closed -- --nocapture` -> pass
- `cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture` -> pass
- `cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture` -> pass

### EV regression result after Phase C (default optimizer)

- Command:
  - `cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1`
- Summary:
  - `candidate_valid=50/50`
  - `candidate_invalid=0`
  - `candidate_better=0/50`
  - `mean_delta=-18.176103599424` (improved vs `-19.934197985691`)
- Full-L1 (`n=26`):
  - `mean_delta=-34.954045379167` (improved vs `-38.334996121988`)
  - `mean_solver_iters=15.307692`
  - `mean_hold_clamps=4505.115385`
  - `mean_rescue_attempts=1.423077`
  - `mean_rescue_accepts=0.423077`
- Runtime:
  - EV compare test finished in `33.13s` (baseline was `34.50s`)

Post-change budget check:

- `GLOBAL_SOLVER_MAX_ITERS=4096` after Phase C produced identical summary:
  - `mean_delta=-18.176103599424`
  - full-L1 `mean_delta=-34.954045379167`

This confirms the residual gap is still not solved by simply running longer.

### Optional dual override after Phase C

- Command:
  - `GLOBAL_SOLVER_OPTIMIZER=dual cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1`
- Summary:
  - `candidate_valid=50/50`
  - `mean_delta=-18.151172743759`
  - full-L1 `mean_delta=-34.906101425965`

Interpretation:

- Phase C improved EV and preserved validity/invariants.
- The optimizer is still not economically converged against incumbent (`candidate_better=0/50`).
- Letting it run longer was not the fix; descent-path quality was the binding issue.
