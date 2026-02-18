# Global Solver EV Regression Progress Summary

Date: 2026-02-18
Scope: end-to-end progress across Stage 1-4, Phase A/A.5, Phase B/C, and Phase D implementation checkpoint.

## Current Status (Phase D recovery update)

Safety and invariants are stable:

- `candidate_valid=50/50`
- `candidate_invalid=0`
- full and partial invariant suites pass.

Default candidate EV has improved to near-parity:

- `mean_delta=-0.422000933261`
- full-L1 fuzz family (`n=24`): `mean_delta=-0.801324117995`
- partial-L1 fuzz family (`n=24`): `mean_delta=-0.152491624944`
- full-L1 regression snapshots (`n=2`): `mean_delta=0.895765583734`
- `candidate_better=3/50`

Dual override (`GLOBAL_SOLVER_OPTIMIZER=dual`) at this checkpoint:

- not re-profiled after this recovery update.

Pure solver-only checkpoint (`GLOBAL_SOLVER_ENABLE_ROUTE_REFINEMENT=false`):

- `candidate_valid=50/50`
- overall `mean_delta=-19.762219369469`
- full-L1 fuzz family: `mean_delta=-34.604352163150`
- partial-L1 family: `mean_delta=-3.304407858962`
- full-L1 regression snapshots: `mean_delta=-39.150363971376`
- `candidate_better=2/50`
- artifact: `/tmp/global_ev_default_pure_1771417465226.jsonl`

Dual-router override (`GLOBAL_SOLVER_OPTIMIZER=dual_router`) currently resolves mostly via primal fallback and matches default aggregate EV:

- not re-profiled after this recovery update.

## Phase D Work Landed

Implemented in code:

1. New optimizer mode and config surface
- added `GlobalOptimizer::DualRouterV1`
- added `dual_router_*` config fields
- added `buy_sell_churn_reg`

2. New dual-router module
- added `src/portfolio/core/global_solver_dual_router.rs`
- dual iterate + route-faithful primal + bounded primal restore
- action projection/replay gate reused unchanged

3. Safety fallback for availability
- if dual-router candidate construction returns `None`, dispatch falls back to stable primal L-BFGS candidate path.

4. Diagnostics and telemetry
- added candidate dual/primal-restore/theta/buy/sell/overlap diagnostics fields
- EV compare harness now writes per-case JSONL rows to `/tmp/global_ev_*.jsonl`
- added under-incumbent reason buckets and family-level summary lines

5. Optional CFMMRouter parity scaffolding
- added ignored parity test, Rust CLI bridge, and Julia fixture runner scaffold.

6. Route-faithful rebuild in default primal candidate
- strengthened route refinement path with complete-set arb pre-pass.
- replaced direct-only sell cleanup with phase-1 merge-aware liquidation (`ExecutionState::execute_optimal_sell`).
- added EV-guarded phase-3 recycling loop and EV-guarded polish reoptimization loop.
- preserved fail-closed replay/validity gates and candidate-only selection in `GlobalCandidate`.

## Progress Timeline (condensed)

### 0) Initial severe regression

- `candidate_valid=2/50`
- `candidate_invalid=48/50`
- `mean_delta=-42.910196918399`

### 1) Stage 1-3 stabilization

- `candidate_valid=50/50`
- `mean_delta=-22.407627450367`

### 2) Stage 4 attempt (rejected)

- `mean_delta=-45.841560299366`

### 3) Phase A/A.5 (L-BFGS-B + active set + selective repair)

- `candidate_valid=50/50`
- `mean_delta=-19.934197985691`

### 4) Phase B prototype (dual behind flag)

- dual override improved but still under incumbent.

### 5) Phase C convergence recovery

- `candidate_valid=50/50`
- `mean_delta=-18.176103599424`

### 6) Phase D checkpoint

- route-faithful + dual-router structure landed,
- JSONL and reason-bucket telemetry landed,
- safety preserved with dual-router fallback,
- EV parity gap remains concentrated in full-L1.

### 7) Phase D recovery update (route-faithful + recycling + polish)

- `candidate_valid=50/50`
- `candidate_invalid=0`
- `mean_delta=-0.422000933261`
- full-L1 fuzz family: `mean_delta=-0.801324117995`
- partial-L1 fuzz family: `mean_delta=-0.152491624944`
- full-L1 regression snapshots: `mean_delta=0.895765583734`
- `candidate_better=3/50`

## What Is Fixed

- Candidate validity recovered and remains stable at `50/50`.
- Replay consistency and invariant checks remain green.
- New telemetry now identifies under-incumbent mechanisms (`mint_usage`, `boundary_saturation`, etc.).

## What Still Fails

- Candidate EV still loses to incumbent on most cases (`candidate_better=3/50`), but aggregate parity gates are now met (`overall>-0.5`, `full-L1>-1.0`).
- Dual-router path still requires additional work before promotion beyond fallback-backed availability.

## Latest Commands and Artifacts

Commands executed:

```bash
cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
GLOBAL_SOLVER_OPTIMIZER=dual cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
GLOBAL_SOLVER_OPTIMIZER=dual_router cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture
cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture
```

JSONL telemetry outputs:

- `/tmp/global_ev_default_1771413324320.jsonl`
- `/tmp/global_ev_dualdecompositionprototype_1771413364881.jsonl`
- `/tmp/global_ev_dualrouterv1_1771413489081.jsonl`
- `/tmp/global_ev_default_1771415895866.jsonl`

## Canonical Detailed Docs

- `docs/global_solver_phaseD_router_plan.md`
- `docs/global_solver_stage_report_2026-02-18.md`
- `docs/global_solver_plan_history.md`
- `docs/global_solver_phaseB_dual_plan.md`
- `docs/global_solver_phaseC_convergence_plan.md`
