# Fix: Stale Cost Accounting in Direct Buy Execution

## Problem

In `execute_bundle_step()` (trading.rs), the `BundleRouteKind::Direct` arm stored pre-planned costs from `segment.direct_member_plans` in `Action::Buy`. These costs are computed during planning via `cost_to_price()` in `solver.rs`.

In the normal waterfall flow (plan → immediately execute), planned costs equal live costs because both use the same `sims`. However, when pool state changes between planning and execution — as happens during **polish loops** and **recycling phases** — the stored costs become stale, inflating reported EV.

This was identified via the pathological 18-outcome test case where:
- Rust reported EV: 132.14 (from stored costs)
- Actual on-chain EV: 118.53 (from AMM replay)
- Julia convex solver EV: 127.58

## Fix

Minimal change: recompute `cost` from live pool state via `buy_exact()` before storing it in `Action::Buy`. All other execution logic (budget deduction, pool state updates via `set_price`, `sim_balances` tracking) remains unchanged.

```rust
let live_cost = self.sims[idx]
    .buy_exact(amount)
    .map(|(_, c, _)| c)
    .unwrap_or(_planned_cost);
self.actions.push(Action::Buy {
    market_name: self.sims[idx].market_name,
    amount,
    cost: live_cost,
});
```

`buy_exact(&self)` is a pure read-only function, so this adds no side effects.

## Scope

Audited all 9 production `Action::Buy` creation sites:
- **trading.rs:458** (Direct buy execution) — the only site using pre-planned costs. **Fixed.**
- All other sites already use live costs from `buy_exact()` or `find_profitable_complete_set_buy_round()`.

## Verification

- No new test failures introduced (3 pre-existing failures confirmed on unmodified code)
- Diagnostic logging confirmed live costs = planned costs in normal waterfall flow (no divergence in standard test cases)
- The fix is defensive: it has no effect when planning and execution share the same sim state, but correctly handles the case when they diverge
