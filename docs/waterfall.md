# Waterfall Rebalancing Algorithm (Canonical Spec)

This is the definitive implementation spec for the waterfall rebalancing algorithm.

Status: current as of 2026-02-27.  
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
pub fn rebalance_with_gas(..., mode: RebalanceMode, gas: &GasAssumptions) -> Vec<Action>
```

- `rebalance(...)` is equivalent to `rebalance_with_mode(..., RebalanceMode::Full)`.
- `rebalance_with_mode(...)` uses zero gas thresholds (`0.0, 0.0`) for route admission.
- `rebalance_with_gas(...)` computes route thresholds from `GasAssumptions` and threads them into waterfall admission.

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

## Full-Mode Phase Flow

`RebalanceMode::Full` executes six phases:

1. `Phase 0`: complete-set arbitrage pre-pass (`buy-all -> merge`) when mint routes are available.
2. `Phase 1`: iterative sell-overpriced liquidation.
3. `Phase 2`: waterfall allocation.
4. `Phase 3`: legacy-only recycling with EV-guarded trial commit.
5. `Phase 4`: bounded polish loop; commit trial only if EV improves.
6. `Phase 5`: terminal cleanup sweeps.

### Phase 1 (Sell Overpriced)

For each outcome, while price is above prediction and holdings remain:

- size a sell target toward prediction price,
- execute optimal direct/merge split via `execute_optimal_sell`,
- if merge-heavy execution reduced holdings without moving source price enough, force one full remainder attempt.

Bound: `MAX_PHASE1_ITERS = 128` per outcome.

### Phase 2 (Waterfall Core)

Waterfall state:

- active set over `(outcome_idx, route)` with route in `{Direct, Mint}`,
- current profitability frontier `current_prof`,
- remaining cash budget.

Admission/ranking:

- `best_non_active()` is rescanned from current pool state each iteration.
- Runtime gas gate for each candidate:
  - direct: `remaining_budget * prof >= gas_direct_susd`
  - mint: `remaining_budget * prof >= gas_mint_susd`
- outcomes can appear twice (direct and mint), with independent profitability ranking.

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

Safety bounds:

- `MAX_WATERFALL_ITERS = 1000`
- stalled progress guard `MAX_STALLED_CONTINUES = 4`

### Waterfall Budget Solver (`solve_prof`)

All-direct active set:

- closed-form `pi = (A/B)^2 - 1` (clamped to `[prof_lo, prof_hi]`).

Mixed active set:

- simulation-backed bisection (up to 64 iterations),
- affordability predicate uses planned mint-first/direct execution with running budget feasibility.

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

- optional arb pre-pass (if mint available),
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

## Accounting and State Invariants

- `apply_actions_to_sim_balances` updates holdings for all four action types.
- Mint/merge are cross-contract complete-set operations (two L1 markets).
- Post-run tests assert replay consistency and local gradient bounds.

## Related Docs

- `docs/portfolio.md`: module overview and extensive test map.
- `docs/model.md`: math derivations and implementation map.
- `docs/gas_model.md`: gas threshold model details.
- `docs/arb_mode.md`: arb-only API and equations.
