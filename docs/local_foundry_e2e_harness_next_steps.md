# Local Foundry E2E Harness Next Steps

Status: current as of 2026-04-17.

## Summary

The local Foundry executable E2E harness is complete and merged for local executable transaction validation.

It already covers:

- local Seer deployment
- local Uniswap V3 full-range pools
- `SwapRouter02`
- real `TradeExecutor.batchExecute(Call[])` execution
- Rust off-chain executable plans
- synthetic 98-outcome on-chain solver calls
- gas, calldata, raw EV, modeled fee, and realized net-EV accounting

This document tracks the remaining work that would make the harness broader or stricter. These items are follow-up implementation projects, not blockers for the current merged harness.

## 1. True Connected Child Mint/Merge Route Coverage

### Current State

The connected direct path is executable locally, including root/child markets and local outcome-token address-book overrides.

The current mint-sell and buy-merge route probes are root-only synthetic scenarios. They use scripted `Mint -> Sell+` and `Buy+ -> Merge` action selection, then still exercise the normal Rust execution grouping, packed-program compiler, local address book, transaction builder, `SwapRouter02`, Seer router, and `TradeExecutor.batchExecute`.

This split exists because the checked-in Seer child-market artifact currently blocks nested local child split/merge execution. The future work is to reproduce and fix that artifact/setup issue before replacing the scripted route probes.

### Why It Matters

The L1-like topology has two connected markets:

- the root market has a connector outcome
- the child market is collateralized by that connector
- root invalid, child invalid, and connector outcomes are not tradeable

Direct swaps are enough to prove local executable transaction generation for connected tradeable pools, but true complete-set mint/merge coverage should eventually prove the Seer split/merge path across the connected root and child markets as well.

### Implementation Plan

1. Add a focused Foundry repro test that only deploys a small connected root/child Seer topology and attempts:
   - root split from `DummyUSDC`
   - connector approval to the Seer router
   - child split from connector collateral
   - child merge back to connector collateral
   - root merge back to `DummyUSDC`
2. Capture the exact revert path and identify whether the blocker is:
   - stale or mismatched checked-in artifact bytecode
   - missing wrapped-token implementation code
   - incorrect `Wrapped1155Factory` wiring
   - incorrect child market parent outcome or parent market setup
3. If artifact mismatch is the cause, regenerate or replace only the required Seer/Wrapped1155 artifacts and document the source commit.
4. Update `_seedLiquidityInventory` so connected scenarios can seed:
   - root complete-set inventory
   - connector inventory
   - child complete-set inventory
   - required invalid-token inventory for merge paths
5. Convert `small_mint_sell` and `small_buy_merge` from root-only synthetic scenarios to connected root/child scenarios.
6. Remove `scripted_route_probe_actions` only after production solver output can reliably select the intended connected mint/merge route shapes. If production solver selection remains too scenario-sensitive, keep the scripted action path but restrict it to a clearly named deterministic route-probe test.

### Acceptance Criteria

- Connected mint-sell executes through real local Seer root and child split paths.
- Connected buy-merge executes through real local Seer child and root merge paths.
- Root invalid, child invalid, and connector balances are present only where needed for Seer complete-set execution and are not treated as tradeable outcomes.
- `TradeExecutor`, Seer router, `SwapRouter02`, mint helper, `Rebalancer`, and `RebalancerMixed` retain no unexpected collateral or outcome balances.
- `forge test --ffi --match-path test/LocalFoundryExecutableTxE2E.t.sol -vv` passes.

### Out Of Scope

- True 6-decimal USDC collateral support.
- Changing the production Rust solver objective just to force route selection.
- Adding a new on-chain connected-market solver interface; that is tracked separately below.

## 2. Connected Two-Market On-Chain Solver Benchmarking

### Current State

The current on-chain solver benchmark is intentionally synthetic and single-market. It deploys one root market with 98 tradeable outcomes and executes:

- `Rebalancer.rebalance`
- `Rebalancer.rebalanceExact`
- `RebalancerMixed.rebalanceMixedConstantL`

This is the correct current boundary because the deployed `Rebalancer` and `RebalancerMixed` APIs accept one Seer market. They cannot faithfully represent the connected two-market L1 topology without a new interface or new entrypoint.

### Why It Matters

The Rust/off-chain executable path already proves the realistic connected 98-tradeable topology. A connected on-chain benchmark would let us compare on-chain and off-chain solver net EV for the same market structure, including connector and child-market complete-set constraints.

### Implementation Plan

1. Design a new on-chain parameter shape that can represent:
   - root market
   - child market
   - connector collateral token
   - collateral token
   - per-outcome market ownership
   - per-outcome pool metadata
   - invalid and connector exclusions
2. Add a new solver entrypoint rather than changing existing single-market behavior. Keep `rebalance`, `rebalanceExact`, and `rebalanceMixedConstantL` backward compatible.
3. Build internal split/merge helpers for connected complete sets:
   - root split/merge uses base collateral
   - child split/merge uses connector collateral
   - invalid tokens are required for merge execution but excluded from tradeable arrays
4. Add a small connected on-chain smoke test before adding the 98-outcome benchmark.
5. Add a connected 98-outcome on-chain benchmark parallel to the Rust/off-chain connected case.
6. Compare raw EV, L2 gas, calldata bytes, modeled fee, and realized net EV against the existing Rust/off-chain executable plan for the same scenario family.

### Acceptance Criteria

- Small connected on-chain solver transaction executes through `TradeExecutor.batchExecute`.
- Connected 98-outcome on-chain solver transaction executes through `TradeExecutor.batchExecute`.
- Every chunk or single solver transaction stays below the `40_000_000` L2 gas cap.
- The benchmark reports whether the Rust/off-chain executable plan still has net EV at least as high as the comparable on-chain solver transaction after gas.
- Existing single-market on-chain solver tests continue to pass unchanged.

### Out Of Scope

- Replacing the existing single-market on-chain solver APIs.
- Changing Seer market semantics.
- Making connected on-chain support a dependency for the current Rust/off-chain executable proof.

## 3. Tighter 98-Outcome Net-EV Tolerance

### Current State

The first merged executable harness uses `LARGE_TOLERANCE_WAD = 2_500e18` for the 98-outcome cases. That tolerance was intentionally loose for the first local executable proof.

The harness already asserts:

- positive realized net EV for each executable Rust fixture scenario
- realized raw EV near Rust expected raw EV
- realized net EV near Rust estimated net EV with an additional fee-model allowance
- real measured calldata bytes and L2 gas from `TradeExecutor.batchExecute`

### Why It Matters

A wide tolerance is acceptable for the initial proof but can mask future drift in:

- Rust pool simulation versus local V3 execution
- route grouping and transaction compilation
- fee modeling
- scripted route probe estimates

The goal is to turn the 98-outcome case from a broad executable proof into a sharper regression test.

### Implementation Plan

1. Add temporary diagnostic logging or a local report helper that records, for repeated runs:
   - expected raw EV
   - realized raw EV
   - estimated fee
   - modeled realized fee
   - estimated net EV
   - realized net EV
   - absolute raw EV delta
   - absolute net EV delta
2. Run the full harness repeatedly on the same machine and capture the observed maximum deltas.
3. Split the single large tolerance into separate named tolerances:
   - raw EV simulation tolerance
   - fee-model tolerance
   - net EV tolerance
4. Set the new tolerances to observed maximum delta plus an explicit margin.
5. Keep the small-case tolerance separate and tight.

### Acceptance Criteria

- The 98-outcome connected Rust executable test passes consistently across repeated local runs.
- The tolerance is documented with the observed maximum delta and chosen margin.
- A material raw EV or net EV regression fails the test instead of being absorbed by `2_500e18`.
- Small connected tests keep their existing tighter tolerance unless measured data justifies a change.

### Out Of Scope

- Changing the solver objective.
- Adding live RPC dependencies to the local harness.
- Requiring deterministic gas across different Foundry or compiler versions.

## 4. Optional Release-Mode FFI Fixture Execution

### Current State

Foundry currently calls the Rust fixture with:

```bash
cargo run --quiet --bin local_foundry_e2e_fixture <input-json-path>
```

This is correct and keeps the local development path straightforward, but cold debug builds and debug execution may be slower than `--release`.

### Why It Matters

The 98-outcome harness is intended to be practical for local regression checks. If release mode materially reduces fixture runtime after the initial compile, it may make the executable harness easier to run frequently.

### Implementation Plan

1. Measure full harness runtime with the current debug command:
   - cold build
   - warm build
2. Measure full harness runtime with:
   - `cargo run --quiet --release --bin local_foundry_e2e_fixture <input-json-path>`
3. Compare output stability:
   - action count
   - chunk count
   - calldata bytes
   - expected raw EV
   - estimated net EV
4. If release mode is materially faster for warm runs and output remains stable, update the FFI command and docs.
5. If release mode mainly shifts cost to a slower cold compile, keep debug mode and document the reason.

### Acceptance Criteria

- The chosen mode is documented in the harness doc.
- `forge test --ffi --match-path test/LocalFoundryExecutableTxE2E.t.sol -vv` passes with the chosen mode.
- Fixture output remains stable enough to satisfy the existing EV and calldata assertions.

### Out Of Scope

- Introducing a separate build system.
- Checking compiled fixture binaries into the repo.
- Making release mode mandatory for contributors unless the measured runtime gain clearly justifies it.

## 5. Broad Rust Test Regressions Outside Harness Scope

### Current State

Focused harness checks pass. Broad `cargo test` has two deterministic failures in tests that were not changed by the local Foundry E2E harness:

- `execution::bounds::tests::skips_group_when_edge_below_gas_plus_buffer`
- `portfolio::core::tests::oracle::test_rebalance_zero_liquidity_outcome_disables_mint_merge_routes`

These failures should be tracked separately so the harness work stays scoped and the root cause is not hidden by unrelated documentation or E2E changes.

### Why It Matters

The harness can be complete while the broader repo still has solver regression tests that need attention. Leaving those failures undocumented makes future verification ambiguous: a failing broad `cargo test` should not be mistaken for a local executable harness failure.

### Implementation Plan

1. Create a separate branch for broad Rust regression cleanup.
2. Reproduce each failure independently with:
   - `cargo test execution::bounds::tests::skips_group_when_edge_below_gas_plus_buffer -- --exact`
   - `cargo test portfolio::core::tests::oracle::test_rebalance_zero_liquidity_outcome_disables_mint_merge_routes -- --exact`
3. For the bounds test, determine whether the strict gas gate threshold changed or the assertion is stale.
4. For the zero-liquidity test, determine whether the solver should still emit direct buys when one pooled outcome has zero liquidity, or whether the invariant should be updated.
5. Fix behavior or assertions in the smallest possible patch.
6. Run focused tests first, then a broader `cargo test` sweep if runtime is acceptable.

### Acceptance Criteria

- Both deterministic tests pass or are explicitly updated with a clear rationale.
- The fix does not change local Foundry E2E harness behavior unless it reveals a real integration bug.
- Any changed solver behavior is reflected in the relevant canonical solver docs.

### Out Of Scope

- Solving unrelated ignored, live-RPC, or long-running opt-in tests.
- Refactoring the solver while fixing stale assertions.
- Blocking the local Foundry executable harness on these independent failures.

## Verification

This next-steps document is documentation-only.

Recommended verification for changes to this file:

```bash
sed -n '1,320p' docs/local_foundry_e2e_harness_next_steps.md
rg -n "local_foundry_e2e_harness_next_steps" docs/README.md
git diff --check
```

No Foundry or Rust test is required for documentation-only updates.

If future implementation changes the harness code, rerun:

```bash
forge test --ffi --match-path test/LocalFoundryExecutableTxE2E.t.sol -vv
```

## Assumptions

- This document tracks future work; `docs/local_foundry_e2e_harness.md` remains the stable reference for the current merged harness.
- The current merged harness is complete for local executable validation.
- Future child mint/merge work should fix the Seer artifact/setup issue before replacing scripted route probes.
- Future connected on-chain solver work requires a new interface or entrypoint because the current on-chain solver APIs accept one Seer market.
- Broad `cargo test` regressions are independent cleanup work unless future diagnosis proves they affect executable harness behavior.
