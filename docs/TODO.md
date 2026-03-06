# TODO

## Open

- [ ] Integrate gas/MEV-aware net-PnL objective (EV minus execution costs) once live gas-cost estimator wiring is available.
- [ ] Replace heuristic L1 data-fee model (estimated calldata bytes + cached two-point `getL1Fee(bytes)` marginal fee/byte estimate) with exact transaction-serialization-based `getL1Fee(txData)` now that strict action->transaction calldata wiring is implemented.
- [ ] If we want to benchmark the Phase 0-arb-inclusive off-chain strategy directly, add a CTF-aware on-chain `rebalanceAndArb()` / `rebalanceAndArbExact()` fixture. The current apples-to-apples A/B benchmark now compares direct parity and full rebalance-only dominance against `Rebalancer.rebalance()`; the historical Phase 0 mismatch is documented separately in `docs/archive/rebalancer/rebalancer_ab_mixed_gap_investigation_2026-03-03.md`.
- [ ] Explore strict arb-at-end cyclic scheduling (`rebalance` with no internal arb, then `arb`, repeated to convergence) as an optional mode. Include anti-churn guards to avoid fee-negative buy/sell round-trips and benchmark against start-arb flow on both 4-outcome and 98-outcome suites.

## Done (On-Chain)

- [x] **On-chain Rebalancer**: `Rebalancer.sol` with closed-form waterfall allocation (ψ), atomic pool state reads, and per-pool `sqrtPriceLimitX96` slippage protection. Replaces the off-chain hybrid approach whose slippage tolerance problem was intractable.
- [x] **On-chain complete-set arb**: `arb()` function in `Rebalancer.sol` for mint-sell (sum > 1) and buy-merge (sum < 1) normalization.

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
- [x] **TradeExecutor strict action->transaction wiring**: deterministic subgroup calldata builder (`DirectBuy`, `DirectSell`, `MintSell`, `BuyMerge`, `DirectMerge`) now targets `TradeExecutor.batchExecute` with unequal-amount batch-router support.
- [x] **Strict submission fail-closed gates**: execute loop now enforces `planned_at_block` staleness and wall-clock deadline checks before submission, with dry-run default and explicit `EXECUTE_SUBMIT=1` live mode.
