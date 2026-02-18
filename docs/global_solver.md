# Global Solver (Execution-Faithful Candidate)

## Overview

This document describes the global candidate optimization stack used by:

- `RebalanceEngine::GlobalCandidate`
- `RebalanceEngine::AutoBestReplay`

`RebalanceEngine::Incumbent` remains the rollout-safe default.

## Latest EV Checkpoint (2026-02-18)

Latest EV comparison run:

```bash
cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
```

From `/tmp/global_ev_default_1771415895866.jsonl`:

- `candidate_valid=50/50`
- `candidate_invalid=0`
- overall `mean_delta=-0.422000933261`
- full-L1 overall (`fuzz + regression`, `n=26`): `mean_delta=-0.670778756323`
- full-L1 fuzz (`n=24`): `mean_delta=-0.801324117995`
- partial-L1 fuzz (`n=24`): `mean_delta=-0.152491624944`
- full-L1 regression snapshots (`n=2`): `mean_delta=0.895765583734`
- `candidate_better=3/50`

Current wins over incumbent in this run:

- `fuzz_full_l1_case_0`: `delta=+0.888289612926`
- `fuzz_full_l1_case_18`: `delta=+1.387274542502`
- `regression_full_l1_snapshot`: `delta=+1.856221833414`

## Optimizer Modes

`GlobalOptimizer` modes:

- `DiagonalProjectedNewton`
- `LbfgsbProjected`
- `DualDecompositionPrototype`
- `DualRouterV1`

`DualRouterV1` currently runs the dual-router route-faithful path and falls back to stable primal L-BFGS candidate construction if dual-router candidate construction fails.

## Public Config Surface

`GlobalSolveConfig` includes baseline projected-solver controls plus:

- `dual_router_max_iters`
- `dual_router_pg_tol`
- `dual_router_lbfgs_history`
- `dual_router_primal_restore_iters`
- `dual_router_primal_residual_tol`
- `dual_router_price_floor`
- `buy_sell_churn_reg`

## Solve Result and Diagnostics

`GlobalSolveResult` additive fields:

- `dual_residual_norm`
- `primal_restore_iters`

`RebalanceDecisionDiagnostics` additive fields:

- `candidate_dual_residual_norm`
- `candidate_primal_restore_iters`
- `candidate_net_theta`
- `candidate_total_buy`
- `candidate_total_sell`
- `candidate_buy_sell_overlap`

Existing diagnostics and fail-closed invalid reasons remain unchanged.

## Core Objective and Variables

Baseline primal variableization:

- direct per-market buy/sell quantities
- complete-set net flow (`theta`)

Dual-router route-faithful variableization:

- `buy_i`
- `sell_ratio_i`
- `theta_plus`, `theta_minus`

with:

- `inventory_i = hold0_i + buy_i + theta_plus - theta_minus`
- `sell_i = sell_ratio_i * max(inventory_i, 0)`
- `hold_i_final = max(inventory_i, 0) - sell_i`

Objective remains EV-centric with barrier safeguards; overlap churn is penalized by `buy_sell_churn_reg`.

## Projection and Safety Contract

Projection contract is unchanged and strict:

1. mint bracket if `theta > DUST` with executable sells,
2. residual sells,
3. buys,
4. merge if `theta < -DUST`.

Hard requirements preserved:

- no projection-time sell injection,
- no projection-time theta normalization,
- no projection-time affordability resize,
- replay cash/holdings equality checks under `solver_budget_eps` tolerances.

Any projection/replay mismatch fails closed and falls back to incumbent selection logic.

## EV Compare Harness

`test_compare_global_vs_incumbent_ev_across_rebalance_fixtures` now supports:

- optimizer override:
  - `GLOBAL_SOLVER_OPTIMIZER=dual`
  - `GLOBAL_SOLVER_OPTIMIZER=dual_router`
- dual-router config overrides:
  - `GLOBAL_SOLVER_DUAL_ROUTER_MAX_ITERS`
  - `GLOBAL_SOLVER_DUAL_ROUTER_PG_TOL`
  - `GLOBAL_SOLVER_DUAL_ROUTER_LBFGS_HISTORY`
  - `GLOBAL_SOLVER_DUAL_ROUTER_PRIMAL_RESTORE_ITERS`
  - `GLOBAL_SOLVER_DUAL_ROUTER_PRIMAL_RESIDUAL_TOL`
  - `GLOBAL_SOLVER_DUAL_ROUTER_PRICE_FLOOR`
  - `GLOBAL_SOLVER_BUY_SELL_CHURN_REG`
  - `GLOBAL_SOLVER_ENABLE_ROUTE_REFINEMENT`

Pure solver-only benchmark (disable all route-refinement layers in primal candidate path):

```bash
GLOBAL_SOLVER_ENABLE_ROUTE_REFINEMENT=false cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
```

Per-case JSONL telemetry is emitted to `/tmp/global_ev_*.jsonl` and includes:

- family/case labels,
- action-mix stats,
- solver state metrics,
- under-incumbent reason buckets.

## Optional CFMMRouter Parity

Optional, ignored-by-default parity scaffolding exists:

- `src/portfolio/tests/cfmmrouter_parity.rs`
- `src/portfolio/core/cfmmrouter_bridge.rs`
- `tools/cfmmrouter/route_fixture.jl`

Enable with:

```bash
RUN_CFMMROUTER_PARITY=1 cargo test cfmmrouter_parity -- --ignored --nocapture
```

This path is intentionally optional and not part of required Rust-only CI gates.
