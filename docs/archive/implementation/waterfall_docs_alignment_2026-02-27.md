# Waterfall Docs Alignment (2026-02-27)

## Scope

This note records the implementation-vs-doc review for the waterfall rebalancing path and related write-ups.

Code used as source of truth:

- `src/portfolio/core/rebalancer.rs`
- `src/portfolio/core/waterfall.rs`
- `src/portfolio/core/planning.rs`
- `src/portfolio/core/solver.rs`
- `src/portfolio/core/trading.rs`
- `src/main.rs`

## Differences Found and Resolved

1. Runtime gas handling in waterfall admission
- Implemented behavior: `best_non_active()` admits direct/mint entries only when `remaining_budget × profitability >= route_gas_threshold`.
- Doc update: added this explicitly in `docs/portfolio.md` and `docs/individual_rebalance.md` (with `rebalance_with_gas` vs `rebalance_with_mode` behavior).

2. Waterfall execution description overstated "recompute before execution"
- Implemented behavior: planning runs on scratch state with mint-first/direct ordering; direct legs execute planned `(cost, new_price)`; mint legs execute in bounded rounds with fail-closed rollback if planned amount cannot be fully satisfied.
- Doc update: replaced the inaccurate claim in `docs/portfolio.md`.

3. Terminal cleanup sequence was under-specified
- Implemented behavior: one mixed cleanup pass, optional recycle, up to four direct-only sweeps, then (if mint available) an extra mixed pass, optional recycle, and up to two more direct-only sweeps.
- Doc update: expanded Phase 5 description in `docs/portfolio.md` and `docs/individual_rebalance.md`.

4. Mint Newton equation RHS in model docs
- Implemented behavior: RHS is skip-adjusted: `(1 - P_target) - Σ_{j∈skip, j≠target} P⁰_j`.
- Doc update: made this the primary equation in `docs/model.md` and clarified empty-skip reduction.

5. Vitalik annotations contained stale implementation assumptions
- Stale assumptions included: no gas-aware gate, flash-loan actions, and missing execution submission path.
- Doc update: added a status note at top of `docs/vitalik_annotations.md` and updated the stale statements while preserving the opinionated analysis.

6. Runtime transport wording drift
- Implemented behavior: `main.rs` uses an HTTP JSON-RPC provider (`with_reqwest`), not WebSocket.
- Doc update: corrected wording in `docs/architecture.md` and `readme`.

## Notes

- `docs/vitalik_annotations.md` remains a commentary document, not a normative spec.
- `docs/vitalik_annotations.md` is currently gitignored in this repo (`.gitignore`), so updates there are local unless explicitly force-added.
- Canonical normative spec is now `docs/waterfall.md`.
- Supporting references for details remain `docs/portfolio.md`, `docs/model.md`, and `docs/gas_model.md`.
