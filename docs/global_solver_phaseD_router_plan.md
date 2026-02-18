# Global Solver Phase D: Dual Router + Route-Faithful Primal

Date: 2026-02-18

## Scope

Phase D replaces repair-heavy coupled optimization with a route-faithful variableization and a price-dual outer loop, while preserving the existing fail-closed projection/replay contract.

Production constraints for this phase:

- Rust solver remains canonical execution path.
- Existing invalid-reason semantics and replay checks remain unchanged.
- Incumbent fallback safety stays enabled (`RebalanceEngine::AutoBestReplay`).

## Context from Thesis and Router References

The solver structure follows the decomposition template from the thesis dual chapters and Appendix D connection to prediction markets:

- optimize over shadow prices (`nu`) on token nodes,
- solve edge-local arbitrage subproblems independently,
- recover coupled primal flows with explicit restoration when ties/non-smooth faces appear.

For prediction-market style complete-set coupling (`sum prices ~ 1` with temporary arbitrage dislocations), the complete-set edge is modeled explicitly as a synthetic mint/merge subproblem in the dual iterate.

## High-Level Architecture

`GlobalOptimizer::DualRouterV1` pipeline:

1. Build `PoolSim` subproblems and holdings/cash initial state.
2. Run dual-router outer loop over shadow prices.
3. Build a route-faithful primal seed from dual flows.
4. Solve route-faithful primal with box-projected line search.
5. Run bounded primal-feasibility restoration.
6. Emit actions through existing projection/replay gate.
7. If dual-router construction fails, fallback to stable primal L-BFGS projected solver (same fail-closed gate).

## Route-Faithful Parameterization

Per market `i`:

- `buy_i >= 0`
- `sell_ratio_i in [0,1]`
- `theta_plus >= 0`, `theta_minus >= 0`

Derived state:

- `theta = theta_plus - theta_minus`
- `inventory_i = hold0_i + buy_i + theta`
- `sell_i = sell_ratio_i * max(inventory_i, 0)`
- `hold_i_final = max(inventory_i, 0) - sell_i`

Objective includes:

- terminal EV (`cash + sum pred_i * hold_i_final`),
- existing theta regularization,
- barrier terms for non-negative cash/hold,
- small buy/sell overlap penalty (`buy_sell_churn_reg`).

## Dual Router Iteration

Token shadow prices are iterated with projected updates (`price >= dual_router_price_floor`).

Per dual iterate:

- each pool solves local arbitrage from current local shadow price,
- complete-set synthetic subproblem chooses mint vs merge edge,
- gradient is net token imbalance plus stabilizing local regularization term.

Stopping:

- projected dual gradient norm <= `dual_router_pg_tol`, or
- max outer iterations (`dual_router_max_iters`).

## Primal Restoration

After primal solve, bounded restoration enforces executable feasibility:

- clamp buys/sells to pool and inventory caps,
- compute cash/hold residuals,
- if cash infeasible, scale buys via bisection,
- iterate up to `dual_router_primal_restore_iters`,
- target residual <= `dual_router_primal_residual_tol`.

If restoration cannot produce a valid candidate path, solver returns `None` and dispatch fallback uses stable L-BFGS primal.

## Diagnostics and Telemetry

Added `GlobalSolveConfig` controls:

- `dual_router_max_iters`
- `dual_router_pg_tol`
- `dual_router_lbfgs_history`
- `dual_router_primal_restore_iters`
- `dual_router_primal_residual_tol`
- `dual_router_price_floor`
- `buy_sell_churn_reg`

Added diagnostics fields:

- `candidate_dual_residual_norm`
- `candidate_primal_restore_iters`
- `candidate_net_theta`
- `candidate_total_buy`
- `candidate_total_sell`
- `candidate_buy_sell_overlap`

EV compare harness now emits per-case JSONL rows to `/tmp/global_ev_*.jsonl`, including:

- family/case labels,
- action mix (counts + volumes),
- solver metrics,
- under-incumbent reason buckets (`mint_usage`, `overlap_churn`, `boundary_saturation`, etc.).

## Optional CFMMRouter Parity Scaffolding

Optional parity assets added (ignored by default):

- `src/portfolio/tests/cfmmrouter_parity.rs`
- `src/portfolio/core/cfmmrouter_bridge.rs`
- `tools/cfmmrouter/route_fixture.jl`

Execution:

- set `RUN_CFMMROUTER_PARITY=1` to run parity harness,
- bridge calls Julia fixture runner via CLI (`JULIA_BIN` override supported).

Current Julia runner is a deterministic scaffold with placeholder API mapping and explicit status reporting; it is intentionally isolated from required CI paths.

## Current Risks and Failure Modes

- Dual-router inner implementation is still early-stage; fallback to primal is currently required for broad-case availability.
- Full-L1 EV gap remains concentrated in complete-set-coupled cases.
- Fallback keeps safety stable but does not yet deliver target EV lift.

## Next Engineering Steps

1. Replace heuristic dual update with stricter objective/gradient pair and line-searchable descent.
2. Tighten primal restoration residual minimization on tie faces.
3. Promote dual-router path only after meeting full-L1 EV and candidate-better gates.
4. Wire true CFMMRouter API mapping in Julia runner for external parity oracle.
