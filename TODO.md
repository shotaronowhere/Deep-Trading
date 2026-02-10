# TODO

## Open

- [ ] L2 and Originality market support

## Done

- [x] **Dual-route waterfall**: treat (outcome, route) pairs as separate waterfall entries, Phase 3 exit valuation fix
- [x] **Mint-route coupling**: replaced pre-sorted `ranked` vec with per-step `best_non_active` scan from current pool state. Entries promoted above `current_prof` by mint perturbations are absorbed immediately. Eliminates stale ranking without added complexity.
- [x] **Stale skip set in `active.retain`**: prune loop now re-derives skip set after each removal until the set stabilizes.
- [x] **Arbitrage loop bound**: waterfall loop capped at `MAX_WATERFALL_ITERS` (1000).
