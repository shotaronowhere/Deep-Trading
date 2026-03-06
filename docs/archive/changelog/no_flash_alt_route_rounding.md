# No-Flash Indirect Route Execution

## Summary

The rebalancing engine no longer emits `FlashLoan` / `RepayFlashLoan` actions.
Indirect routes are now executed as self-funded rounds:

- `Mint -> Sell*` for mint-sell routes
- `Buy* -> Merge` for buy-merge routes

Each round is capped by both current liquidity and currently available cash.

## Why

Flash-loan actions were unnecessary plumbing for local planning/replay and introduced
extra action-shape complexity. The execution model now uses only economically real actions.

## Round Execution Rules

### Mint-Sell

For target mint amount `M`, execution loops until `M` is exhausted or no progress is possible:

1. Compute per-round liquidity cap as the minimum sell capacity across required sell legs.
2. Compute per-round cash cap as current budget (`max(budget, 0)`).
3. Compute an affordable `round <= min(remaining, liquidity_cap, cash_cap)` such that
   net round cash usage (`round - sum(sell_proceeds)`) does not exceed current cash.
4. Emit one `Mint(round)` and its corresponding `Sell` legs.
5. Update budget by `-round + sum(sell_proceeds)`.

If a round cannot satisfy required sell legs, execution stops fail-closed for that step.
With zero cash, mint rounds do not start.

Mint step execution is transactional: if the planned mint amount cannot be fully
completed, all interim round-side effects are rolled back (budget, actions,
simulated prices, and simulated balances).

### Buy-Merge

For requested merge amount `M`, execution loops in rounds:

1. Compute merge liquidity cap from complementary inventory + pool buy caps.
2. Compute spendable cash as `max(budget, 0)`.
3. Solve an affordable round amount (bounded bisection) such that round buy cost is <= spendable cash.
4. Emit required `Buy` legs, then `Merge(round)`.
5. Update budget by `-sum(buy_costs) + round`.

Inventory-only rounds (zero buy cost) are still allowed even with zero spendable cash.

## Action Grouping Shape

Execution grouping no longer depends on flash-loan brackets.

- `MintSell`: `Mint -> Sell+`
- `BuyMerge`: `Buy+ -> Merge`
- Direct groups unchanged.

Unsupported action shapes still fail closed at grouping.
