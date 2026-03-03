# Small Bundle Mixed EV Gap Investigation (2026-03-03)

## Question

Why does the committed `small_bundle_mixed_case` benchmark show the off-chain "mixed" path below on-chain `Rebalancer.rebalanceExact()`?

- Off-chain mixed EV: `100096481776136945664`
- On-chain exact EV: `100132739273228663690`
- Gap: `-36257497091718026` wei (`~0.036%`)

## Reproduction

Commands run locally:

```bash
cargo test benchmark_snapshot_matches_current_optimizer -- --nocapture
forge test --match-test test_rebalancer_ab_benchmark -vv
```

The Rust snapshot test still matches the committed fixture.
The Foundry benchmark still logs the same gap for `small_bundle_mixed_case`.

## Finding

This is primarily an execution-model mismatch in the benchmark, not evidence that the on-chain direct-only solver is economically better than the off-chain mixed planner.

### 1. The Rust "mixed" snapshot runs Phase 0 buy-merge arb first

`src/portfolio/core/rebalancer.rs` calls `run_phase0_complete_set_arb()` before waterfall allocation inside `finish_rebalance_full_inner(...)`.

In the Rust A/B fixture:

- `rebalance_with_custom_predictions_for_test(..., true)` forces `mint_available = true`
- route gates are disabled (`RouteGateThresholds::disabled()`)
- `small_bundle_mixed_case` starts with `sum(prices) = 0.15 + 0.15 + 0.28 + 0.24 = 0.82`

With `sum(prices) < 1`, Phase 0 selects the buy-merge complete-set arb path (`execute_complete_set_arb()`).

### 2. The 7 committed "mixed" actions are exactly what Phase 0 predicts

The committed fixture says:

- `offchain_action_count = 7`

That matches the observed control flow:

1. Phase 0 buy-merge arb:
   - 4 `Buy` legs (one per outcome)
   - 1 `Merge`
2. Post-arb cleanup:
   - 2 direct `Buy` actions on the remaining underpriced outcomes

Total: `4 + 1 + 2 = 7`

This matches the snapshot exactly.

### 3. The Foundry side does not include that arb path

`test/RebalancerAB.t.sol` only calls:

- `rebalancer.rebalance(...)`
- `rebalancer.rebalanceExact(...)`

It does not call the contract's separate `arb()` function.

So the current comparison is:

- Rust mixed path = `phase0 arb + rebalance`
- Solidity exact path = `rebalanceExact` only

That is not an apples-to-apples comparison.

### 4. Why the EV ends up lower

In this fixture, the Phase 0 buy-merge arb is profitable in cash, but it raises all four pool prices before the rebalancer runs.

That matters because:

- the best direct alpha is concentrated in outcomes `A` and `B`
- Phase 0 also buys the already expensive `C` and `D` legs as part of the complete set
- after the merge, the remaining direct mispricing in `A` and `B` is materially smaller

A simple replay of the current formulas lands near the committed snapshot:

- after Phase 0 buy-merge: cash is about `100.019`
- after the remaining direct `A/B` cleanup: terminal EV is about `100.0964`

That lines up with the committed Rust mixed EV (`100.096481776...`).

By contrast, the on-chain benchmark path skips Phase 0 arb and spends its first-order budget directly on the `A/B` mispricing, which is why its terminal EV stays near `100.1327`.

## Conclusion

The current `small_bundle_mixed_case` anomaly is best explained by a benchmark mismatch:

- the Rust path includes pre-rebalance buy-merge arb
- the Solidity path does not

The observed gap does not isolate "mixed-route planning quality" vs `Rebalancer.rebalanceExact()`.

## Recommended Next Step

Use one of these comparisons instead:

1. Disable Phase 0 complete-set arb in the Rust snapshot path for this A/B test, then compare mixed routing vs `rebalanceExact()`.
2. Keep Phase 0 enabled, but compare the Rust path against the on-chain sequence `arb()` followed by `rebalanceExact()`.

Until that alignment is done, `small_bundle_mixed_case` should not be used as evidence that the on-chain solver beats the off-chain mixed planner.
