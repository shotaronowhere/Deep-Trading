# Arb-Only Mode

## API

`RebalanceMode` adds two execution modes for portfolio planning:

```rust
pub enum RebalanceMode {
    Full,
    ArbOnly,
}
```

Use:

```rust
pub fn rebalance_with_mode(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[(Slot0Result, &'static MarketData)],
    mode: RebalanceMode,
) -> Vec<Action>
```

`rebalance(...)` remains backward-compatible and is equivalent to `rebalance_with_mode(..., RebalanceMode::Full)`.

`RebalanceMode::Full` keeps the existing phase-0 behavior (buy-all then merge only). Two-sided
complete-set arb is isolated to `RebalanceMode::ArbOnly`.

## Two-Sided Complete-Set Arb Rule

`ArbOnly` ignores prediction-driven phases and executes only complete-set arbitrage on current pool prices.

Decision logic:
1. Compute `price_sum = Σ_i price_i` over all pooled outcomes.
2. If `price_sum < 1 - EPS`, run buy-all then merge (`Buy* -> Merge`).
3. If `price_sum > 1 + EPS`, run mint then sell-all (`Mint -> Sell*`).
4. Otherwise do nothing.

Only one side is executed per call.

## Sizing

### Buy-Merge (`price_sum < 1`)
Amount is solved by marginal buy-cost boundary:

- Marginal buy cost at amount `m`:
  - `d_buy(m) = Σ_i P_i / ((1-f) * (1 - m * λ_i)^2)`
- Solve for largest feasible `m` with `d_buy(m) <= 1`, capped by `min_i max_buy_tokens_i`.

### Mint-Sell (`price_sum > 1`)
Amount is solved by marginal sell-proceeds boundary:

- Marginal sell proceeds at amount `m`:
  - `d_sell(m) = Σ_i P_i * (1-f) / (1 + m * κ_i)^2`
- Solve for largest feasible `m` with `d_sell(m) >= 1`, capped by `min_i max_sell_tokens_i`.

Both solvers use bounded bisection and fail closed to `0` when the boundary is not profitable or not reachable.

## Full-Set Requirement (Fail Closed)

`ArbOnly` requires a complete pooled L1 snapshot. It emits no actions unless:

- the simulation set size matches the expected pooled L1 outcome count, and
- the simulated market-name set exactly matches the pooled L1 market-name set (no duplicates, no missing, no unexpected names).

This avoids subset arbitrage behavior and enforces strict market-completeness assumptions.

## Action Stream Shapes

### Buy-Merge Branch
`Buy* -> Merge` (executed in one or more cash/liquidity-feasible rounds)

### Mint-Sell Branch
`Mint -> Sell*` (executed in one or more cash/liquidity-feasible rounds)

Both branches update simulated pool state and only realize budget gains when computed realized profit is positive.
