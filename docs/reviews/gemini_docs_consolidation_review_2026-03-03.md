### Section 1: Top Findings

**High Severity (Conflicting / Redundant Truths)**
1. **`docs/portfolio.md` vs `docs/waterfall.md` collision:** `docs/README.md` correctly declares `docs/waterfall.md` as the definitive off-chain spec. However, `docs/portfolio.md` retains a massive "Algorithm" section that details the exact same 6-phase flow, route pricing, and budget exhaustion logic. This creates a split source of truth.
2. **`docs/individual_rebalance.md` vs `docs/model.md` collision:** Both documents explain the exact same theoretical foundation: the AMM price model, KKT conditions, and Newton's method for the mint route. `docs/individual_rebalance.md` reads like an explanatory blog post but overlaps almost entirely with the canonical `docs/model.md`.
3. **Misplaced historical research:** `docs/improvements.md` contains long-form historical mathematical exploration and verbatim AI review critique logs. It is sitting in the active docs root instead of `docs/archive/`.

**Medium Severity (Root Clutter & Discoverability)**
4. **Root directory clutter (micro-docs):** The root `docs/` folder is cluttered with small, single-feature notes or changelogs (`docs/arb_mode.md`, `docs/no_flash_alt_route_rounding.md`, `docs/plan_preview.md`, `docs/uniswap-v4-removal.md`). These obscure the primary architectural documents for new engineers.
5. **Empty runbook:** `docs/transactions.md` is completely empty (except for a heading) but is still linked in `docs/README.md` under "Operational runbooks".

**Low Severity (Fragmentation)**
6. **Fragmented component docs:** Documentation for the BatchSwapRouter is split awkwardly between `docs/batch_swap_router_api.md` and `docs/batchrouter-uniswap-v3-fixture-tests.md`.
7. **Redundant pointer doc:** `docs/slippage.md` is a 15-line file that only contains links to other canonical specs and an archive document. It serves no standalone purpose.
8. **Stale sprint plans:** `docs/plans/2026-02-24-gas-aware-rebalancing.md` is a completed implementation plan sitting outside the archive.

### Section 2: What Is Working Well

- **`docs/README.md` Precedence Map:** The index is excellent. The explicit numbering for precedence (e.g., "If documents conflict, follow this precedence...") is exactly what a mature documentation architecture needs.
- **Policy Isolation:** `docs/rebalancer_approaches_playbook.md` successfully centralizes the "why" and "when" (policy decisions, threshold values, and live-validation protocols), keeping the "how" clean in the technical specs.
- **On-Chain Specs:** `docs/rebalancer.md` and `docs/rebalancer_mixed.md` are highly cohesive, self-contained, and perfectly separated from the off-chain Rust logic.
- **Active Archiving:** The `docs/archive/` folder is actively used and well-organized, which has successfully removed massive blocks of legacy text (like the old slippage sprint doc) from the critical path.

### Section 3: Actionable Consolidation Plan

1. **Strip `docs/portfolio.md`:** Delete the entire "Algorithm" section (Phases 0-5) from `docs/portfolio.md`. Repurpose the file strictly as the Rust module overview, test map, and EV regression harness guide. `docs/waterfall.md` must be the sole spec for the off-chain flow.
2. **Merge Math/Theory:** Consolidate `docs/individual_rebalance.md` into `docs/model.md` to create a single mathematical derivation document. 
3. **Clean Up Micro-Docs (Merge):**
   - Merge `docs/arb_mode.md` into `docs/waterfall.md` (Arb-Only Mode is a subset of the off-chain flow).
   - Merge `docs/plan_preview.md` into `docs/execution_submission.md`.
   - Merge `docs/batch_swap_router_api.md` and `docs/batchrouter-uniswap-v3-fixture-tests.md` into a single `docs/batch_swap_router.md`.
4. **Archive Historical Context:** Move `docs/improvements.md`, `docs/uniswap-v4-removal.md`, `docs/no_flash_alt_route_rounding.md`, and the `docs/plans/` directory into `docs/archive/`.
5. **Delete Dead Files:** Delete `docs/transactions.md` (empty) and `docs/slippage.md` (redundant). Remove their references from `docs/README.md`.

### Section 4: Files To Keep As Canonical

- `docs/README.md`
- `docs/architecture.md`
- `docs/waterfall.md` (Expanded with `arb_mode.md`)
- `docs/rebalancer.md`
- `docs/rebalancer_mixed.md`
- `docs/rebalancer_approaches_playbook.md`
- `docs/rebalancer_policy_metrics_schema.md` (and associated csv/json)
- `docs/execution_submission.md` (Expanded with `plan_preview.md`)
- `docs/deployments.md`
- `docs/model.md` (Expanded with `individual_rebalance.md`)
- `docs/gas_model.md`
- `docs/portfolio.md` (Stripped of the algorithm flow)
- `docs/rebalancing_mechanism_design_review.md`
- `docs/monte_carlo_rebalance_validation.md`
- `docs/rebalance_test_ev_trace.md`
- `docs/TODO.md`
- `docs/batch_swap_router.md` (Newly consolidated from the two batch router files)

### Section 5: Files To Archive Further Or Merge

- **`docs/individual_rebalance.md`**: MERGE into `docs/model.md`. (Reason: redundant explanations of the AMM price model and math).
- **`docs/arb_mode.md`**: MERGE into `docs/waterfall.md`. (Reason: it is an execution mode of the waterfall off-chain logic).
- **`docs/plan_preview.md`**: MERGE into `docs/execution_submission.md`. (Reason: it's a CLI diagnostic tool directly related to execution).
- **`docs/batch_swap_router_api.md` & `docs/batchrouter-uniswap-v3-fixture-tests.md`**: MERGE into `docs/batch_swap_router.md`. (Reason: fragmentation of a single contract's documentation).
- **`docs/improvements.md`**: ARCHIVE. (Reason: historical AI reviews and theoretical mixed-route planning).
- **`docs/no_flash_alt_route_rounding.md`**: ARCHIVE. (Reason: historical changelog note about removing flash loans).
- **`docs/uniswap-v4-removal.md`**: ARCHIVE. (Reason: historical changelog note).
- **`docs/plans/2026-02-24-gas-aware-rebalancing.md`**: ARCHIVE. (Reason: completed historical sprint plan).
- **`docs/transactions.md`**: DELETE. (Reason: completely empty file).
- **`docs/slippage.md`**: DELETE. (Reason: provides no unique info, just links that `docs/README.md` already handles).

### Section 6: Residual Risks

- **Broken Cross-links:** Archiving, merging, and deleting over 10 files will break relative links within the remaining markdown files (e.g., `docs/README.md`, or `docs/portfolio.md` linking to `arb_mode.md`). A pass with a broken-link checker across the repo will be necessary immediately after the move.
- **Context Loss in `portfolio.md`:** Developers accustomed to reading `docs/portfolio.md` for the step-by-step waterfall logic might be disoriented when it is removed. We must ensure `docs/waterfall.md` is prominently linked at the very top of `docs/portfolio.md` to reroute them seamlessly.
