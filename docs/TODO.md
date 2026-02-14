# TODO

## Open

- [ ] Integrate gas/MEV-aware net-PnL objective (EV minus execution costs) once live gas-cost estimator wiring is available.
- [ ] Replace heuristic L1 data-fee model (estimated calldata bytes + cached two-point `getL1Fee(bytes)` marginal fee/byte estimate) with exact transaction-serialization-based `getL1Fee(txData)` once action->transaction calldata wiring is implemented.
- [ ] Enforce strict staleness/deadline gating at submission time (`planned_at_block` + `max_stale_blocks`, `deadline_secs`) once action->transaction execution loop is wired.

## Deferred

- [ ] L2 and Originality market support
- [ ] Decouple portfolio core from L1-global assumptions (`PREDICTIONS_L1`, full-L1 market universe) to reduce integration risk for L2/Originality.

## Done

- [x] **Dual-route waterfall**: treat (outcome, route) pairs as separate waterfall entries, Phase 3 exit valuation fix
- [x] **Mint-route coupling**: replaced pre-sorted `ranked` vec with per-step `best_non_active` scan from current pool state. Entries promoted above `current_prof` by mint perturbations are absorbed immediately. Eliminates stale ranking without added complexity.
- [x] **Stale skip set in `active.retain`**: prune loop now re-derives skip set after each removal until the set stabilizes.
- [x] **Arbitrage loop bound**: waterfall loop capped at `MAX_WATERFALL_ITERS` (1000).
- [x] **Fix `mint_cost_to_prof` bug**: corrected Newton RHS to `(1 - tp) - Σ_{skip, j≠target} P⁰_j`, added `rhs ≤ 0` guard. (8879ce0)
- [x] **Mint-before-direct execution ordering**: budget-exhaustion branch now executes mints first (aggregate M split equally), then directs using post-mint pool state. (8879ce0)
- [x] **Coupled (π, M) Newton solver**: `solve_prof` returns `(profitability, aggregate_mint_M)` using D*/S₀ partition, inner Newton on M, outer Newton on π. (8879ce0)
- [x] **Mixed-route exactness upgrade**: replaced fixed-binding `i*` approximation with simulation-backed bisection over profitability and budget-feasible route planning/execution (mint-first, no partial-step skips).
