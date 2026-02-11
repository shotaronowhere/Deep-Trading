# TODO

## Open

(none)

## Deferred

- [ ] L2 and Originality market support
- [ ] ~~Multi-mint exact solve~~ — the coupled (π, M) solver uses a single binding i\*; when |Q|>1 the remaining mint entries have residual alt-price mismatch. Full K-constraint solve deferred — error ~1-3% of budget, dwarfed by execution noise. See `improvements.md`

## Done

- [x] **Dual-route waterfall**: treat (outcome, route) pairs as separate waterfall entries, Phase 3 exit valuation fix
- [x] **Mint-route coupling**: replaced pre-sorted `ranked` vec with per-step `best_non_active` scan from current pool state. Entries promoted above `current_prof` by mint perturbations are absorbed immediately. Eliminates stale ranking without added complexity.
- [x] **Stale skip set in `active.retain`**: prune loop now re-derives skip set after each removal until the set stabilizes.
- [x] **Arbitrage loop bound**: waterfall loop capped at `MAX_WATERFALL_ITERS` (1000).
- [x] **Fix `mint_cost_to_prof` bug**: corrected Newton RHS to `(1 - tp) - Σ_{skip, j≠target} P⁰_j`, added `rhs ≤ 0` guard. (8879ce0)
- [x] **Mint-before-direct execution ordering**: budget-exhaustion branch now executes mints first (aggregate M split equally), then directs using post-mint pool state. (8879ce0)
- [x] **Coupled (π, M) Newton solver**: `solve_prof` returns `(profitability, aggregate_mint_M)` using D*/S₀ partition, inner Newton on M, outer Newton on π. (8879ce0)
