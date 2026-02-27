# Portfolio Rebalancing

Canonical algorithm spec: see `docs/waterfall.md`.

## Overview

`src/portfolio/core/mod.rs` (with `sim.rs`, `planning.rs`, `solver.rs`, `trading.rs`, `waterfall.rs`, `rebalancer.rs`) computes optimal rebalancing trades for L1 prediction markets using an analytical single-tick `PoolSim` model (`f64`, with explicit tick-boundary caps). Implements the waterfall allocation algorithm: deploy capital to the most profitable outcome, equalize profitability progressively with the next-best outcomes, then recycle lower-profitability inventory.

## Function: `rebalance`

```rust
pub fn rebalance(
    balances: &HashMap<&str, f64>,       // current outcome token balances
    susds_balance: f64,                   // available sUSD capital
    slot0_results: &[(Slot0Result, &'static MarketData)],  // pool state from fetch_all_slot0
) -> Vec<Action>
```

## Modes

The portfolio module now supports mode-based execution:

- `RebalanceMode::Full` (default): existing prediction-driven rebalancing flow.
- `RebalanceMode::ArbOnly`: complete-set arbitrage only (`sum(prices) < 1` buy-merge, `sum(prices) > 1` mint-sell).

See [Arb-Only Mode](./arb_mode.md) for API details, sizing equations, and fail-closed behavior.

## Gas-Aware APIs

Public gas-aware entry points:

- `rebalance_with_gas(...)`: compatibility wrapper with conservative defaults (`1e-9` ETH gas price and `$3000` ETH/USD).
- `rebalance_with_gas_pricing(..., gas_price_eth, eth_usd)`: explicit runtime pricing inputs.

Runtime binaries support optional `ETH_USD` override (default: `3000`):

- `src/main.rs`
- `src/bin/execute.rs`

## EV Regression Reporting

`test_fuzz_rebalance_ev_regression_fast_suite` now prints a per-case EV report for its full/partial fixtures:

- `ev_before`, `ev_after`, and `delta`
- snapshot target `expected_after` and realized `snapshot_delta`
- tolerated snapshot band (`floor_tol`, `ceiling_tol`)

It also prints compact summary lines for `full`, `partial`, and `combined` groups.

## Action Types

```rust
enum Action {
    Mint { contract_1, contract_2, amount, target_market },  // Mint across both L1 contracts
    Merge { contract_1, contract_2, amount, source_market }, // Buy others + merge to sell
    Buy { market_name, amount, cost },                       // Buy outcome tokens from pool
    Sell { market_name, amount, proceeds },                  // Sell outcome tokens for sUSD
}
```

Indirect routes are self-funded and execute in bounded rounds:
- Mint route: `Mint → Sell(others)` in repeated liquidity/cash-feasible chunks.
- Buy-merge route: `Buy(others) → Merge` in repeated liquidity/cash-feasible chunks.
- No flash-loan actions are emitted.

## Algorithm

Implemented full-mode flow:
1. Phase 0: complete-set arbitrage pre-pass.
   - Runtime gas-gated path (`rebalance_with_gas*`): two-sided (`buy-all -> merge` when `sum(prices) < 1`, `mint -> sell-all` when `sum(prices) > 1`).
   - Zero-threshold compatibility path (`rebalance` / `rebalance_with_mode`): legacy one-sided (`buy-all -> merge` only when `sum(prices) < 1`).
2. Phase 1: iterative sell-overpriced liquidation.
3. Phase 2: waterfall allocation.
4. Phase 3: legacy-inventory recycling (EV-guarded trial commits).
5. Phase 4: bounded polish re-optimization loop (commit only when EV improves).
6. Phase 5: terminal cleanup sweeps (mixed + direct-only bounded passes).

### Phase 1: Sell overpriced holdings
For each outcome where market_price > prediction and we hold tokens:
- Optimize a split between **direct sell** (sell into pool) and **merge sell** (buy all other outcomes + merge complete sets for 1 sUSD each) to maximize total proceeds for the fixed sell amount.
- The split is solved as a 1D concave optimization (`f'(m)=0`) over merge amount `m`, with boundary checks (`m=0`, `m=cap`) and bisection on the interior root when needed.
- Merge consumes existing complementary holdings first, and buys only any shortfall from pools.
- Merge capacity is limited by complementary inventory plus buy caps (`held_j + max_buy_tokens_j`) across non-source outcomes; direct handles the remainder.
- Merge route only available when `mint_available` (all pools present).
- Pool state (`price`) updated after each sell for subsequent calculations.

### Phase 2: Waterfall allocation

#### Route pricing

Each outcome has up to two acquisition routes, each with its own price and profitability:

- **Direct buy**: current pool price from `PoolSim::price()`. Profitability = `(prediction - direct_price) / direct_price`.
- **Mint** (when all outcomes have liquid pools): price = `1 - sum(other_outcome_prices)`. Profitability = `(prediction - mint_price) / mint_price`.

Route availability:
- `mint_available = sims.len() == PREDICTIONS_L1.len()` — true when every tradeable outcome (98) has a non-zero-liquidity pool
- When `mint_available`: mint route competes with direct; merge sell route competes with direct sell
- Partial prediction sets are not supported — `build_sims` returns a typed error if any outcome is missing a prediction, and `rebalance` fail-closes with no actions.
- Initialization fail-close reasons are emitted with structured logs (`non_finite_budget`, `sim_build_failed`, `no_eligible_sims`) for runtime diagnostics.
- Invalid pool state entries (zero liquidity, malformed liquidity, missing ticks, or out-of-range tick math) are dropped from sim construction with an info log.

#### Dual-route waterfall

The waterfall treats each (outcome, route) pair as a separate entry. The same outcome can appear twice — once for direct, once for mint — with different profitabilities. This ensures both price channels are fully exploited.

**Dynamic entry selection:** Instead of pre-sorting entries by profitability, the waterfall calls `best_non_active()` each iteration to find the highest-profitability non-active entry from **current** pool state. This handles mint-route coupling: when a mint sell perturbs other pools' prices, the next entry selection reflects the updated state automatically.

**Gas-aware admission filter (runtime path):** `best_non_active()` only admits an entry if `remaining_budget × profitability >= gas_threshold_for_route`. In production (`rebalance_with_gas`, used by `main.rs`), route thresholds come from `GasAssumptions`; in tests (`rebalance_with_mode`) thresholds are `0.0` so this filter is disabled.

**Execution-aligned route gates (runtime path):**
- Route thresholds now include `DirectSell`, `BuyMerge`, and `DirectMerge` in addition to `DirectBuy` and `MintSell`.
- Phase-1 and phase-3 liquidation candidates are filtered by the same execution gate predicate used for planning semantics (`edge > gas + profit_buffer`).
- Waterfall tail steps are pruned when their approximate execution edge is sub-gas for their route.
- Non-finite gas estimates fail closed for the affected route.

**Monotonicity guard:** If a mint pushes a non-active entry's profitability *above* `current_prof`, it is absorbed into the active set immediately (no cost step) before the next descent. This prevents the water level from moving upward, which would break `solve_prof`'s assumptions.

**Intra-step active-set boundary:** Planning applies boundary step-splitting to **mint** legs only. The split point is the earliest of: (1) a non-active route reaching `current_prof`, or (2) a non-active direct profitability crossing the active mint route's in-step profitability (`prof_non_active >= prof_active_mint`) before `current_prof` is reached. Direct legs are not intra-step truncated. This route-asymmetric policy preserves mint-side coupling control while avoiding EV leakage from fragmented direct execution. A boundary hit is a **step split within the same profitability group**, not a new level and not a terminal stop: after that partial mint leg, the loop promotes all outcomes that now meet `current_prof`, recomputes `skip`, re-ranks non-active outcomes from current pool state (next-highest only), and keeps building the current step until the next lower profitability level is reached.
If budget is exhausted immediately after such a split, `waterfall` returns the realized post-split marginal profitability (from the executed boundary leg), not the pre-split `current_prof`.
If budget is exhausted on a non-boundary partial step, `waterfall` now continues from `current_prof = achievable` instead of terminating immediately, allowing newly profitable entries (from pool-state changes) to join in subsequent iterations.

**Ranking example:** Outcome A has direct price 0.05 (prof = 1.0) and mint price 0.03 (prof = 2.33). Outcome B has direct prof = 0.6.

| Entry | Route | Prof |
|-------|-------|------|
| A | mint | 2.33 |
| A | direct | 1.0 |
| B | direct | 0.6 |

**Waterfall steps:**
1. Deploy to A_mint until its profitability drops to 1.0 (matches A_direct)
2. Deploy to both A_mint and A_direct until profitability drops to 0.6 (matches B_direct)
3. Deploy to A_mint, A_direct, and B_direct until budget exhausted or no profitability remains
4. When budget is insufficient for a full step, `solve_prof` finds the achievable profitability level (closed-form for all-direct, bisection for mixed)

**Route interactions:** The two routes for the same outcome affect different state. Direct buys move the outcome pool price. Mint sells push non-target pool prices down (increasing alt price). They are independent price channels, but both drain profitability for the same outcome from different directions.

#### Cost computation

Cost per step uses the same analytical `PoolSim` formulas used by execution:
- **Direct route**: `PoolSim::cost_to_price` to target price derived from target profitability (`prediction / (1 + target_prof)`), clamped to `buy_limit_price`.
- **Mint route**: Newton's method finds the mint amount where the alt price (1 - sum of post-sell prices) equals the target price. Warm-started with the linearized first step: m₀ = (P_target - alt(0)) / (2 × Σ Pⱼκⱼ), saving one iteration. Uses the analytical price sensitivity parameter κⱼ = (1-fee) × √priceⱼ × 1e18 / Lⱼ per pool. Each pool's sell volume is capped at `max_sell_tokens()` = (√(price/limit_price) - 1) / κ. The solver clamps mint sizing by the **minimum** cap across required sell legs so planned mint legs remain executable as full `Mint->Sell` rounds (no synthetic flash bracket and no unsold-leg dependency). If a requested profitability level is unreachable under that cap, the solver clamps to the capped executable solution instead of dropping the route. Active outcomes are excluded from sell simulation.
- The caller specifies the route per entry — `direct_cost_to_prof` or `mint_cost_to_prof` is called directly based on the entry's route, rather than picking the best route internally.
- Negative costs (arbitrage, where sell proceeds exceed mint cost) are valid and increase the budget.

#### Budget exhaustion solver (`solve_prof`)

When the total cost to bring all active entries to the next profitability level exceeds the remaining budget:

**All-direct case (closed form):** Within a single tick range, the cost to buy is `L_eff × (√target_price - √current_price)` where `L_eff = L / (1e18 × (1-fee))`. Setting the sum equal to budget and solving for target profitability π:
- A = Σᵢ L_eff_i × √pred_i, B = budget + Σᵢ L_eff_i × √price_i
- **π = (A/B)² - 1** — zero iterations, exact solution.

**Mixed routes (simulation-backed bisection):** When mint routes are in the active set, `solve_prof` does not solve a closed-form coupled `(π, M)` system. Instead, it tests affordability by planning the exact mint-first then direct execution sequence at candidate profitability levels and requiring running-budget feasibility.

The solver searches the interval `[target_prof, current_prof]` with bisection (up to 64 iterations) and returns the lowest feasible profitability level in that interval.

#### Execution safeguards

- Planning and execution share the same mint-first/direct ordering and pricing model. Direct legs execute using planned `(cost, new_price)` from the scratch simulation. Mint legs execute in cash/liquidity-feasible rounds and fail closed with step-level rollback if the planned amount cannot be fully satisfied.
- Waterfall route planning reuses a scratch simulation buffer across iterations to avoid repeated `Vec<PoolSim>` allocation/cloning in the hot loop.
- Per-step budget feasibility is enforced with a running-budget check over the planned mint-first/direct sequence. If execution cannot proceed, the waterfall stops at the current achieved level.
- **Prune loop:** when entries fail cost computation, the active set is pruned in a loop that re-derives the skip set after each removal, so remaining entries always see a consistent skip set. Mint routes do not get pruned solely because the requested target is unreachable under caps; they clamp to saturated executable size.
- **Iteration cap:** the waterfall loop is bounded by `MAX_WATERFALL_ITERS` (1000) to prevent infinite cycling from negative-cost arbitrage that grows the budget.
- **No-progress guard:** budget-partial continuation paths are bounded by a stalled-progress guard (`MAX_STALLED_CONTINUES = 4`) that exits if neither profitability nor budget changes meaningfully across repeated continues.
- `sim_balances` is updated after waterfall by processing all four action types: `Mint` adds `amount` to all outcomes (complete sets), `Merge` subtracts `amount` from all outcomes, `Sell` subtracts sold amount, `Buy` adds directly. This correctly tracks residual unsold tokens when partial sells hit tick boundaries.
- `skip` in mint sell legs uses the set of all active outcome **indices** (not route-specific). If outcome A is active via both direct and mint, its index appears once in `skip` — preventing sell-then-rebuy regardless of which route triggered the mint.

### Phase 3: Post-allocation liquidation
After the waterfall:
1. Check remaining **legacy** holdings' profitability against the last profitability level reached. Profitability is computed using the direct pool price (`sims[i].price()`).
2. For legacy holdings below that level (lowest first), sell only enough to raise profitability toward `last_bought_prof`, using the same direct/merge split optimizer as Phase 1.
3. If a sell leaves meaningful residual legacy and profitability remains clearly below the frontier, run one bounded follow-up sell escalation in the same trial iteration.
4. Reallocate recovered capital via another waterfall pass.
5. Commit the trial pass only if expected value is non-decreasing (EV guardrail).

### Phase 4: Polish Re-optimization
A bounded loop (`MAX_POLISH_PASSES`) reruns arb pre-pass (if mint-available), Phase 1, waterfall, and Phase 3 on a trial state and commits only EV-improving trials.

### Phase 5: Terminal Cleanup
A bounded cleanup sequence runs additional sell-overpriced + waterfall sweeps to reduce residual local profitable directions before returning actions:
1. one mixed-availability pass (`mint_available` as initialized),
2. optional full-inventory phase-3 recycle on that pass (when mint is available and actions were emitted),
3. up to four direct-only sweeps,
4. if mint is available: one extra mixed pass + optional recycle + up to two more direct-only sweeps.

## Two Market Contracts

L1 has 2 contracts (67 + 33 outcomes). Minting a complete set requires:
1. Mint on contract 1 (costs 1 sUSD) → produces 67 tokens including "other repos"
2. Use "other repos" token as collateral to mint on contract 2 → produces 33 tokens

When using the mint route, `Action::Mint` encodes both contracts (derived from distinct `market_id` values in the `sims` array), and sell actions are emitted for non-target, non-active outcomes across both contracts. `Action::Merge` similarly derives contracts from `sims`. During waterfall allocation, outcomes currently being acquired (the "active" set) are skipped in mint sell legs to avoid wasteful sell-then-rebuy cycles.

## Pool Simulation (`PoolSim`)

Each outcome's pool state is tracked mutably during the rebalance:
- `price`: updated after each simulated trade
- Swap direction determined by token ordering (`zero_for_one_buy`)
- Tick boundary clamping ensures trades stay within the pool's liquidity range

### Sell-side methods
- `kappa()`: κ = (1-f) × √price × 1e18 / L — price sensitivity for selling
- `max_sell_tokens()`: max sellable before tick boundary — (√(price/limit_price) - 1) / κ
- `sell_proceeds(m)`: proceeds = price × m × (1-f) / (1 + mκ)

### Buy-side methods
- `lambda()`: λ = √price × 1e18 / L — price sensitivity for buying (dual of κ)
- `max_buy_tokens()`: max buyable before tick boundary — (1 - √(price/buy_limit_price)) / λ
- `buy_exact(amount)`: returns (actual, cost, new_price). Cost = m × price / ((1-f)(1-mλ)). Price after buying: P/(1-mλ)²

## Balance Reading & Caching

`src/pools.rs` provides on-chain balance fetching and local caching:

```rust
use deep_trading_bot::pools::{fetch_balances, save_balance_cache, load_balance_cache, cache_to_balances};

// Fetch on-chain balances via Multicall3
let (susds, balances) = fetch_balances(provider, wallet_address).await?;

// Cache to disk
save_balance_cache(Path::new("balance_cache.json"), &wallet_hex, susds, &balances)?;

// Load from cache (returns None if stale/missing/wrong wallet)
if let Some(cache) = load_balance_cache(Path::new("balance_cache.json"), &wallet_hex, None) {
    let balances = cache_to_balances(&cache);
}
```

## Performance

`test_rebalance_perf_full_l1` benchmarks the full rebalance across all 98 tradeable L1 outcomes using real market/tick/liquidity data from `MARKETS_L1` with synthetic slot0 prices (50% of prediction, creating buy opportunities for every outcome).

| Mode | Per call (98 outcomes) |
|------|-----------------------|
| Release | ~3.6ms |

Run with: `cargo test --release test_rebalance_perf_full_l1 -- --nocapture`

## Oracle and Fuzz Validation

Portfolio tests are split across `src/portfolio/tests.rs` (shared fixtures + early deterministic checks), `src/portfolio/tests/fuzz_rebalance.rs`, `src/portfolio/tests/oracle.rs`, and `src/portfolio/tests/execution.rs`.

- `test_oracle_two_pool_direct_only_with_legacy_holdings_matches_grid_optimum`: hardcoded 2-market fixture with legacy inventory in an overpriced market and an underpriced alternative; verifies rebalance emits sell/reallocate flow and lands near oracle EV.
- `test_oracle_fuzz_two_pool_direct_only_with_legacy_holdings_not_worse_than_grid`: randomized 2-market fuzz over prices, budget, and initial holdings; checks rebalance EV is not materially below the oracle optimum while preserving action-stream invariants.
- `test_oracle_two_pool_closed_form_direct_waterfall_matches_kkt_target`: hardcoded 2-market direct-only fixture with analytic closed-form KKT target; asserts waterfall profitability and final prices match the derived optimum while exhausting budget.
- `test_oracle_two_pool_direct_only_legacy_self_funding_budget_zero_matches_grid`: hardcoded 2-market zero-cash legacy fixture; validates that Phase-1 liquidation can self-fund reallocation and remain near the direct-only oracle EV.
- `test_mint_first_order_can_make_zero_cash_plan_feasible`: sampled mixed-route adversarial search proving order-sensitive feasibility; verifies mint-first execution can make a zero-cash plan feasible while direct-first ordering is infeasible.
- `test_fuzz_rebalance_partial_direct_only_ev_non_decreasing`: partial-L1 direct-only fuzz property asserting `rebalance` does not reduce expected value across randomized prices, holdings, and budgets.
- `test_fuzz_rebalance_partial_no_legacy_holdings_emits_no_sells`: partial-L1 fuzz property asserting no `Sell`/`Merge` actions are emitted when initial legacy inventory is empty (guards against Phase-3 churn on newly bought positions).
- `test_fuzz_rebalance_end_to_end_full_l1_invariants`: deterministic seeded 24-case full-L1 corpus asserting determinism, action/accounting invariants, strict EV gain, and per-case EV floor checks against `src/portfolio/tests/ev_snapshots.json` with a tiny floating-point tolerance (`1e-9 * (1 + max(|got|, |floor|))`).
- `test_fuzz_rebalance_end_to_end_partial_l1_invariants`: deterministic seeded 24-case partial-L1 corpus asserting direct-only route constraints, invariants, strict EV gain, and per-case EV floor checks against snapshots with the same tiny floating-point tolerance.
- `test_rebalance_negative_budget_legacy_sells_self_fund_rebalance`: hardcoded debt-entry fixture (`susds_balance < 0`) showing Phase-1 liquidation can recover cash and avoid EV regression.
- `test_rebalance_handles_nan_and_infinite_budget_without_non_finite_actions`: defensive-input test that `NaN`/`Infinity` budgets fail closed (no actions emitted).
- `test_rebalance_non_finite_balances_fail_closed_to_zero_inventory`: defensive-input fixture ensuring `NaN`/`Infinity` holdings are sanitized to zero inventory, preventing invalid liquidation flow.
- `test_rebalance_zero_liquidity_outcome_disables_mint_merge_routes`: explicit full-L1 fixture with one forced zero-liquidity outcome; verifies mint/merge routes are disabled while direct buys continue for remaining underpriced pools.
- `test_phase3_near_tie_low_liquidity_avoids_ev_regression`: tiny-liquidity near-equal-profitability fixture guarding against Phase-3 churn causing net EV loss.
- `test_phase3_recycling_full_l1_with_mint_routes_reduces_low_prof_legacy`: full-L1 mint-enabled fixture with near-fair legacy bucket; verifies low-marginal legacy holdings are actually reduced while preserving EV.
- `test_phase3_full_l1_recycling_limits_tiny_legacy_sell_fragmentation`: phase-3 de-fragmentation check that bounded follow-up escalation does not devolve into tiny residual sell churn in a crafted legacy scenario.
- Full-L1 action-stream invariants include indirect-route shape checks (`Mint->Sell+`, `Buy+->Merge`) alongside accounting consistency.
- `test_waterfall_misnormalized_prediction_sums_remain_finite`: robustness test for miscalibrated belief vectors (`sum(pred) > 1` and `< 1`) to ensure waterfall state and actions remain finite.
- `test_phase1_merge_split_can_leave_source_pool_overpriced`: adversarial Phase-1 fixture showing that when the optimal sell split uses merge legs, the source pool can remain overpriced even when sell sizing is derived from `sell_to_price(prediction)` in direct-price space.
- `test_fuzz_phase1_sell_order_budget_stability`: sampled order-stability check for Phase-1 liquidation across adversarial 3-outcome fixtures (same holdings/prices, swapped sell order), guarding against unexpected order-sensitive budget recovery drift.
- `test_fuzz_plan_execute_cost_consistency_near_mint_caps`: seeded near-cap mixed-route search ensuring planned route costs remain consistent with executed budget deltas when mint legs operate close to sell-cap boundaries.
- `test_intra_step_boundary_rerank_improves_ev_vs_no_split_control`: seeded low-liquidity mixed-route search that compares current split+rerank waterfall execution against a no-intra-step-split control, asserting existence of a strict EV improvement case from boundary-aware active-set refresh.
- `test_plan_near_full_mint_boundary_does_not_split`: verifies near-full mint boundary intersections are treated as numerically non-meaningful and do not trigger split churn.
- `test_waterfall_boundary_mint_realized_profitability_is_monotone_non_increasing`: checks realized profitability along mint boundary splits remains monotone non-increasing.
- `test_waterfall_budget_partial_continue_can_improve_ev_vs_break_control`: control-vs-current fixture showing continuation after budget-partial execution can improve EV (or match) vs legacy break-after-partial behavior.
- `test_direct_closed_form_target_can_overshoot_tick_boundary`: stress test documenting that all-direct closed-form profitability targets can imply prices above tick boundaries, while execution planning clamps to feasible `buy_limit_price`.
- `test_oracle_phase3_recycling_two_pool_direct_only_matches_grid_optimum`: direct-only 2-market oracle fixture validating Phase-3-style legacy-capital recycling (sell low-profitability legacy inventory, reallocate into a higher-marginal-profitability market) against a grid-search baseline.
- `test_fuzz_pool_sim_kappa_lambda_finite_difference_accuracy`: finite-difference fuzz validation that `kappa`/`lambda` sensitivity parameters match observed small-step price derivatives and closed-form price update equations from `buy_exact`/`sell_exact`.
- `test_rebalance_regression_full_l1_snapshot_invariants`: deterministic full-L1 regression snapshot using real generated L1 market metadata (liquidity/ticks/tokens) with fixed synthetic prices + initial holdings; asserts deterministic replay, action-stream/accounting invariants, and strict EV floor against baseline.
- `test_rebalance_regression_full_l1_snapshot_variant_b_invariants`: second deterministic full-L1 snapshot in a distinct near-fair/mixed-price regime with different holdings and budget; asserts deterministic replay, invariants, and a separate strict EV floor to reduce overfitting to one market profile.
- `test_rebalance_phase1_clears_or_fairs_legacy_overpriced_source_full_l1`: full-L1 adversarial check that legacy overpriced source inventory is not left both held and overpriced after rebalance (legacy inventory must be exhausted or the source price must be brought to prediction/fair level).

EV regression gates are now two-layered:
- deterministic full/partial seeded fuzz corpora with per-case EV-after snapshots loaded from `src/portfolio/tests/ev_snapshots.json`;
- deterministic full-L1 regression fixtures with fixed EV-before/EV-after snapshots.

When intentionally changing optimizer economics, refresh `src/portfolio/tests/ev_snapshots.json` in the same PR and keep the EV floor workflow (`EV_new + tol >= EV_old` on the 51-label corpus, where `tol = 1e-9 * (1 + max(|EV_new|, |EV_old|))`) as an external release gate.

Phase-1 liquidation now runs iteratively per outcome (bounded) instead of a single sell attempt. This prevents one-shot merge-heavy splits from leaving residual legacy inventory in an overpriced source market.
Phase-3 liquidation/reallocation now has an EV guardrail: each iteration is dry-run on cloned state and committed only if expected value does not regress (within numerical tolerance).
After polish, rebalance now runs a terminal cleanup sequence (sell-overpriced + mixed sweep + bounded direct-only sweeps) to reduce residual local profitable directions before returning final actions.

For tests, each `rebalance(...)` run now reports a post-run local-gradient summary (`[rebalance][post-grad]`) and enforces:
- max direct local gradient `<= 1e-6`
- max indirect local gradient `<= 1e-6`

The local-gradient probe uses finite-difference step size `eps = 1e-6` over direct (`buy`, `sell`) and indirect (`mint_sell`, `buy_merge`) directions.

## Usage

```rust
use deep_trading_bot::pools::{fetch_all_slot0, fetch_balances};
use deep_trading_bot::portfolio::{rebalance, Action};

let slot0_results = fetch_all_slot0(provider.clone()).await?;
let (susds, balances) = fetch_balances(provider, wallet).await?;
let actions = rebalance(&balances, susds, &slot0_results);

for action in &actions {
    match action {
        Action::Mint { amount, target_market, .. } => {
            println!("MINT {} sets for {}", amount, target_market)
        }
        Action::Merge { amount, source_market, .. } => {
            println!("MERGE {} sets from {}", amount, source_market)
        }
        Action::Buy { market_name, amount, cost } => {
            println!("BUY {} {} (cost: {})", amount, market_name, cost)
        }
        Action::Sell { market_name, amount, proceeds } => {
            println!("SELL {} {} (proceeds: {})", amount, market_name, proceeds)
        }
    }
}
```
