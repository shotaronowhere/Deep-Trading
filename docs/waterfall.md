# Waterfall Rebalancing Algorithm (Canonical Spec)

This is the definitive implementation spec for the waterfall rebalancing algorithm.

Status: current as of 2026-03-22.
Source-of-truth code paths:

- `src/portfolio/core/rebalancer.rs`
- `src/portfolio/core/waterfall.rs`
- `src/portfolio/core/planning.rs`
- `src/portfolio/core/solver.rs`
- `src/portfolio/core/trading.rs`
- `src/main.rs`

## Public Entry Points

```rust
pub fn rebalance(...) -> Vec<Action>
pub fn rebalance_with_mode(..., mode: RebalanceMode) -> Vec<Action>
pub fn rebalance_with_solver_and_flags(
    ...,
    mode: RebalanceMode,
    solver: RebalanceSolver,
    flags: RebalanceFlags,
) -> Vec<Action>
pub fn rebalance_with_gas(..., mode: RebalanceMode, gas: &GasAssumptions) -> Vec<Action>
pub fn rebalance_with_gas_pricing(
    ...,
    mode: RebalanceMode,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
) -> Vec<Action>
pub fn rebalance_with_solver_and_gas_pricing_and_flags(
    ...,
    mode: RebalanceMode,
    solver: RebalanceSolver,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
    flags: RebalanceFlags,
) -> Vec<Action>
```

- `rebalance(...)` is equivalent to `rebalance_with_mode(..., RebalanceMode::Full)`.
- The legacy solver-less library wrappers stay native-only for deterministic compatibility.
- Operator-facing binaries (`src/main.rs`, `src/bin/plan_preview.rs`, `src/bin/execute.rs`) now default `REBALANCE_SOLVER=head_to_head`, which races native full-mode planning against ForecastFlows in `Full` mode only.
- `rebalance_with_mode(...)` uses conservative default pricing inputs, not zero gas thresholds.
- In zero-threshold compatibility mode, full-mode phase-0 uses legacy one-sided complete-set arb (`buy-all -> merge` only when `sum(prices) < 1`).
- `rebalance_with_gas_pricing(...)` computes per-route thresholds from runtime gas assumptions and also supplies the shared pricing snapshot used for net-EV ranking.
- `rebalance_with_gas(...)` is a compatibility wrapper over `rebalance_with_gas_pricing(...)` using conservative defaults (`gas_price_eth = 1e-9`, `eth_usd = 3000.0`).

## Action Model

Algorithm output actions are:

- `Action::Mint { ... }`
- `Action::Buy { ... }`
- `Action::Sell { ... }`
- `Action::Merge { ... }`

No flash-loan actions are emitted.

## Preconditions and Fail-Closed Rules

- Non-finite budget fails closed to no actions.
- `Full` mode requires valid predictions for all simulated outcomes (`build_sims`).
- `mint_available` is true only when simulated pool count equals expected L1 tradeable outcome count.
- `ArbOnly` requires a complete pooled L1 snapshot (exact expected names/count); otherwise fails closed to no actions.

## Full-Mode Solver

`RebalanceMode::Full` now keeps the waterfall as the continuous core and searches a smaller operator surface around it.

Inner problem:

- the existing waterfall, recycle, polish, and cleanup logic optimize a fixed route/frontier choice

Outer problem:

- choose first-frontier family
- choose preserve subset on a capped churn universe
- choose whether arb is applied before or after the exact rebalance operator

Current runtime behavior:

1. Build `R_exact(state)` from the current state:
   - evaluate three no-preserve frontier seeds from fresh state:
     - `default`
     - forced first frontier `direct`
     - forced first frontier `mint`
   - extract preserve candidates from sell-then-rebuy churn in those seed action streams
   - run one-step singleton-preserve probes to expand the churn universe once
   - aggregate by max churn amount, then max sold amount, then stable market order
   - cap the online preserve universe at `K = 4`
   - enumerate every `(frontier_family, preserve_subset)` pair from fresh state
   - for each exact no-arb candidate, keep the rich trace, replay it to terminal holdings, and compare exactly eight compact forms:
     - `baseline_step_prune` (single-pass reverse pruning by profitability step group, O(G×A))
     - `target_delta`
     - `analytic_mixed`
     - `constant_l_mixed`
     - `coupled_mixed`
     - `staged_constant_l_2`
     - `direct_only`
     - `noop`
   - score the resulting compact plans by estimated net EV and reduce deterministically
2. Evaluate the bounded whole-plan families:
   - `Plain`: `R_exact`
   - `ArbPrimed`: positive root `A`, then `R_exact`
   - `ForecastFlows`: worker-backed `compare_prediction_market_families` on the same initial state, translated into local `Action`s and replayed locally before it is allowed into the ranking set
3. In `HeadToHead`, rank the native winner vs the ForecastFlows winner with the existing `PlanResult` comparator:
   - Rust remains the source of truth for raw EV, estimated fees, tx count, and tie-break ordering
   - worker-reported EV and terminal balances are informational only and are not trusted for acceptance
   - invalid, uncertified, malformed, timed-out, or locally unreplayable ForecastFlows results fail open to the native path
4. If `RebalanceSolver::ForecastFlows` is requested explicitly, try the external family first and fall back to the native full solver only when no valid external candidate survives acceptance.
5. Compile the chosen action plan into an execution program:
   - `Strict`: one tx per strict subgroup
   - `Packed`: greedily pack consecutive strict subgroups into gas-capped tx chunks (`< 40_000_000` estimated L2 gas)
   - rank those execution programs by net EV and keep the better one
5. Execute the packed program by default; strict execution remains the safety fallback when a chunk cannot be safely assembled.

Important boundaries:

- The inner waterfall is still the continuous optimizer; the online exactness is only over the chosen discrete block around it.
- ForecastFlows is a third whole-plan family, not an inner branch of `R_exact`.
- ForecastFlows currently only participates for full L1 snapshots whose live tick/liquidity state can be translated into a contiguous multi-band `UniV3` ladder; partial/incomplete snapshots and `ArbOnly` stay native-only.
- If Rust cannot derive a contiguous liquidity ladder that covers the current price from live tick/liquidity state, ForecastFlows treats the snapshot as unsupported and fails open to the native path instead of approximating with coarse bounds.
- The current compact normal forms are target-holdings-based, not chronological-trace-based:
  - `target_delta` re-emits the rich terminal holdings as one common-shift action plus residual direct buys/sells when that raises net EV
  - `analytic_mixed` solves directly for a compact common-shift-plus-residual frontier target without matching the rich trace holdings exactly
  - `coupled_mixed` solves profitable direct prefixes as a continuous mixed frontier candidate and keeps the result only if it beats the other compact forms on net EV
  - `direct_only` is the mandatory no-mint/no-merge net-EV guard
- First-frontier-family forcing is bounded to the first Phase-2 frontier choice only. After that, the waterfall returns to its existing deterministic frontier logic.
- Execution is now optimized over packed tx chunks, not priced as one tx per replay subgroup.
- ForecastFlows output is accepted only after local Rust replay proves the translated actions are feasible and execution-group-compatible (`Mint -> Sell+`, `Buy+ -> Merge`, or direct-only).
- `direct_only` and `mixed_enabled` are admitted independently; one malformed certified ForecastFlows branch does not discard the other valid branch.
- ForecastFlows remains the gross-EV route generator only; Rust still owns replay, fee modeling, and final net-EV ranking across solver families.
- The staged meta-solver remains compiled only in `#[cfg(test)]` as a reference teacher; it is not part of the runtime objective.
- Release-facing parity claims are single-tick only; `crossing_light` and `crossing_heavy` synthetic cases remain validation-only scope tests.
- Legacy distilled preserve/frontier proposals remain in the default exact-no-arb path; the newer V2 proposal path is diagnostic-only behind `REBALANCE_ENABLE_DISTILLED_PROPOSAL_V2=1`.
- `RebalanceFlags.enable_ev_guarded_greedy_churn_pruning` remains for compatibility but does not change default full-mode behavior.

## Full-Mode Phase Flow

After the meta-solver chooses a candidate, that candidate executes six phases:

1. `Phase 0`: complete-set arbitrage pre-pass when mint routes are available.
   - runtime gas-gated path: two-sided (`sum(prices) < 1` buy-merge, `sum(prices) > 1` mint-sell)
   - zero-threshold compatibility path: legacy one-sided (`sum(prices) < 1` buy-merge only)
   - runtime path estimates phase-0 edge from a dry-run and applies the same execution gate before emitting actions, so sub-gas phase-0 steps are skipped up front.
2. `Phase 1`: iterative sell-overpriced liquidation.
3. `Phase 2`: waterfall allocation.
4. `Phase 3`: legacy-only recycling with EV-guarded trial commit.
5. `Phase 4`: bounded polish loop; commit trial only if EV improves.
6. `Phase 5`: terminal cleanup sweeps.

### Phase 1 (Sell Overpriced)

For each outcome, while price is above prediction and holdings remain:

- size a sell target toward prediction price,
- execute optimal direct/merge split via `execute_optimal_sell`,
- when merge is considered, gate by the merge subtype actually used (`DirectMerge` vs `BuyMerge`) instead of requiring both gates to pass,
- if merge-heavy execution reduced holdings without moving source price enough, force one full remainder attempt.

Bound: `MAX_PHASE1_ITERS = 128` per outcome.

### Phase 2 (Waterfall Core)

Waterfall state:

- active set over `(outcome_idx, route)` with route in `{Direct, Mint}`,
- current profitability frontier `current_prof`,
- remaining cash budget.

Admission/ranking:

- The runtime seeds each waterfall iteration from a bundle frontier over outcomes, not from a raw `(idx, route)` active set.
- Frontier membership is chosen from the best currently executable route per outcome:
  - direct candidates use `remaining_budget * direct_prof >= gas_direct_susd`
  - mint candidates use `remaining_budget * mint_prof >= gas_mint_susd`
- `bundle_frontier` computes both direct and mint frontiers independently, then selects whichever has higher `current_prof`. This prevents direct-first bias from suppressing more profitable mint routes.
- This means mint-only opportunities are still admitted when every direct route is currently unprofitable.

Execution-meaningful step gating:

- Before executing each planned step, approximate edge is `step.cost * max(current_prof, 0.0)`.
- Route-specific gate applies:
  - direct step uses `direct_buy` threshold,
  - mint step uses `mint_sell` threshold.
- The gate requires:
  - finite positive edge,
  - finite non-negative gas estimate,
  - `edge > gas + max(buffer_min_susd, buffer_frac * edge) + EPS`.
- If the first segment in a bundle step fails the route gate and it was mint-first, the runtime retries the same frontier as a direct-only descent before stopping.
- If no executable segment remains after that retry, waterfall ends.

Iteration structure:

1. Promote any non-active entries that are now above `current_prof` (monotonicity guard).
2. Set `target_prof` as the next best non-active profitability (or `0.0`).
3. Prune active entries that cannot be costed to `target_prof`, re-deriving skip set after each prune.
4. Plan active routes mint-first, then direct (`plan_active_routes_with_scratch`).
5. If full step is budget-feasible, execute plan.
6. If not feasible, solve for achievable profitability and execute partial step.

Boundary behavior:

- Mint steps may split at active-set boundaries (join conditions).
- Boundary splits continue the same profitability level with refreshed ranking and skip set.
- Budget-partial non-boundary steps continue from `current_prof = achievable`.

Profitability tracking (`last_prof`):

- `last_prof` records the lowest profitability level the waterfall actually reached, used as Phase 3 recycling threshold.
- Only updated when a step is fully affordable (`BundleStepPlan.fully_affordable == true`). Budget-capped partial steps don't actually reach their `final_prof`, so recording it would overstate how far the waterfall descended and set an incorrect recycling threshold.

Safety bounds:

- `MAX_WATERFALL_ITERS = 1000`
- stalled progress guard `MAX_STALLED_CONTINUES = 4`

### Waterfall Budget Solver (`solve_prof`)

All-direct active set:

- closed-form `pi = (A/B)^2 - 1` (clamped to `[prof_lo, prof_hi]`).

Mixed active set:

- simulation-backed bisection (up to 64 iterations) by default,
- optional experimental coupled mixed-route solver behind `REBALANCE_EXACT_MIXED_SOLVER=1`,
- affordability predicate uses planned mint-first/direct execution with running budget feasibility.
- hard fallback to bisection on non-convergence or non-finite intermediates.

### Mint Route Cost Solve

`mint_cost_to_prof()` solves mint amount by Newton on:

- `g(m) = sum_j P_j / (1 + min(m, cap_j) * kappa_j)^2`
- `rhs = (1 - target_price) - sum_{j in skip, j != target} P_j^0`

Key semantics:

- warm-start from linearized first step,
- per-leg saturation caps at tick boundaries,
- unreachable targets clamp to capped executable solution instead of dropping route.

### Execution Semantics

Direct route:

- executes with planned `(cost, new_price)`.

Mint route:

- executes in bounded liquidity/cash-feasible rounds,
- step-level atomic behavior: if planned amount cannot be fully satisfied, rollback interim state and fail that step.

Budget feasibility is always checked with running budget over planned order.

### Phase 3 (Legacy Recycling)

Scope:

- recycles only legacy inventory (`legacy_remaining`), not newly bought positions.

Trial loop:

1. collect legacy holdings with profitability below current phase-3 frontier,
2. sell toward frontier using optimal direct/merge split,
3. optional escalation for meaningful residual legacy still below frontier,
4. re-run waterfall with recovered budget,
5. commit trial only if EV does not regress (within tolerance).

Bounds:

- `MAX_PHASE3_ITERS = 8`

### Phase 4 (Polish)

Trial context from current state:

- optional arb pre-pass if the chosen phase-order variant enables phase-0 inside polish,
- phase 1,
- waterfall,
- phase 3 over current inventory.

Commit condition:

- commit only if EV strictly improves (within tolerance).

Bound:

- `MAX_POLISH_PASSES = 64`

### Phase 5 (Terminal Cleanup)

Implemented sequence:

1. one additional mixed-availability pass: phase 1 + waterfall.
2. if mint available and pass emitted actions: recycle once across full current inventory.
3. up to 4 direct-only sweeps: phase 1 + waterfall(`mint_available = false`), break on no-op.
4. if mint available:
   - one extra mixed pass + optional recycle on emitted actions,
   - then up to 2 more direct-only sweeps.

Goal: reduce residual local positive gradients before returning final actions.

## Arb-Only Mode

`RebalanceMode::ArbOnly` bypasses prediction-driven phases and runs only two-sided complete-set arb:

- if `sum(prices) < 1 - EPS`: buy-all then merge,
- if `sum(prices) > 1 + EPS`: mint then sell-all,
- else no action.

Requires full pooled L1 outcome set; partial snapshots fail closed.

Sizing semantics:

- buy-merge side solves largest feasible `m` such that marginal buy cost stays `<= 1`
- mint-sell side solves largest feasible `m` such that marginal sell proceeds stay `>= 1`
- both sides are bounded by pool liquidity caps and return fail-closed zero when not feasible

## Accounting and State Invariants

- `apply_actions_to_sim_balances` updates holdings for all four action types.
- Mint/merge are cross-contract complete-set operations (two L1 markets).
- Post-run tests assert replay consistency and local gradient bounds.

## On-Chain Counterpart

The algorithm above runs off-chain in Rust simulation. An on-chain implementation exists in `contracts/Rebalancer.sol` that computes the same waterfall allocation atomically using live pool state. The on-chain version uses a closed-form ψ = (C - budget×(1-fee)) / D with iterative pruning, avoiding the simulation-backed bisection needed for mixed routes off-chain. See [docs/rebalancer.md](rebalancer.md) for the on-chain spec and [docs/slippage.md](slippage.md) for the current slippage policy.

## Related Docs

- `docs/portfolio.md`: module overview and extensive test map.
- `docs/model.md`: math derivations and implementation map.
- `docs/gas_model.md`: gas threshold model details.
- `docs/rebalancer.md`: on-chain rebalancer algorithm and interface.
