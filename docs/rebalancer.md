# On-Chain Rebalancer

For cross-approach policy and the March 7, 2026 solver postmortem, see
[Rebalancer Approaches Playbook](rebalancer_approaches_playbook.md) and
[Rebalancer Solver Postmortem (2026-03-07)](rebalancer_solver_postmortem_2026-03-07.md).

`Rebalancer.sol` implements the direct on-chain portfolio rebalancer. The current
contract keeps only the routes that survived adversarial 98-market benchmarking:

1. `rebalance`: the simple constant-`L` multi-pass waterfall baseline.
2. `rebalanceExact`: the explicit tick-aware exact route.

The removed bounded/adaptive experiments are documented in the postmortem linked
above. They were not kept because the added on-chain planner complexity did not
produce a large enough EV gain on realistic crossing-heavy fixtures.

## Current Entrypoints

Direct routes:

- `rebalance(RebalanceParams)`
- `rebalanceExact(RebalanceParams, maxBisectionIterations, maxTickCrossingsPerPool)`

Arb + recycle routes:

- `rebalanceAndArb(RebalanceParams, market, maxArbRounds, maxRecycleRounds)`
- `rebalanceAndArbWithFloors(RebalanceParams, market, maxArbRounds, maxRecycleRounds, minArbProfitCollateral, minRecycleProfitCollateral)`
- `rebalanceAndArbExact(RebalanceParams, market, maxArbRounds, maxRecycleRounds, maxBisectionIterations, maxTickCrossingsPerPool)`
- `rebalanceAndArbExactWithFloors(RebalanceParams, market, maxArbRounds, maxRecycleRounds, maxBisectionIterations, maxTickCrossingsPerPool, minArbProfitCollateral, minRecycleProfitCollateral)`

The `*WithFloors` entrypoints keep the useful coarse churn controls added during
the solver exploration. The contract still owns all price-sensitive planning.
Off-chain callers only choose the route and the coarse collateral profit floors.

## Route Semantics

### `rebalance`: constant-`L` baseline

This is the production baseline again.

- Reads `slot0().sqrtPriceX96` and in-range `liquidity()` for every pool.
- Solves the closed-form `psi` frontier under the constant-`L` assumption.
- Sorts underpriced pools by current profitability and buys in that order.
- Re-reads the full basket each pass and runs at most `MAX_LEGACY_WATERFALL_PASSES = 6`.
- Uses one `exactInputSingle` per pool swap with `sqrtPriceLimitX96` as the slippage guard.

The constant-`L` approximation is exact inside the current tick and approximate
across initialized tick crossings. That approximation error affects allocation
quality, not swap safety. Uniswap still enforces the actual execution path.

### `rebalanceExact`: explicit exact route

This is the high-fidelity route.

- Reads `slot0().sqrtPriceX96`, `slot0().tick`, and in-range `liquidity()`.
- Builds exact per-pool multi-tick cost ladders by scanning initialized ticks.
- Stops extending a ladder once its prefix cost exceeds the live collateral
  budget, because deeper spend cannot affect the current solve.
- Solves `psi` by bisection over exact summed multi-tick costs.
- Still executes at most one swap per pool.
- Preserves explicit exact semantics: if tick scanning exceeds
  `maxTickCrossingsPerPool`, the call reverts.

The exact route is where complexity currently earns its keep. On realistic
crossing-heavy 98-market fixtures it materially improves EV and still fits under
the 40M gas ceiling.

## Execution Flow

### Phase 1: Pull and sell overpriced outcomes

The contract:

1. Pulls `collateralAmount` plus any configured outcome balances from the caller.
2. Reads pool prices.
3. Sells only outcomes that remain overpriced after fee drag.
4. Uses fee-neutral price limits to avoid selling the last profitable slice into
   a negative-edge region.

Unsold or underpriced holdings remain inside the contract for the waterfall and
possible recycle rounds.

### Phase 2: Complete-set arb pre-pass

If the caller provides a `market`, the contract can run complete-set arb before
the buy-side waterfall spends the collateral budget. This gives deterministic
`sum(price) != 1` profit first access to cash.

`minArbProfitCollateral` is a coarse caller-supplied floor. It avoids wasting
gas on tiny arb rounds without moving detailed planning off-chain.

### Phase 3: Read direct solver state

- `rebalance` reads `sqrtPriceX96` and `liquidity()`.
- `rebalanceExact` also reads `tick` and scans tick state only when building
  exact ladders.

### Phase 4: Solve the profitability frontier

Both routes solve for a common profitability frontier `pi = 1 / psi^2 - 1`.

For constant-`L`, the closed-form solver uses:

```text
psi = (C + budget * (1 - fee)) / D

C = sum(L_i * g(s_i))
D = sum(L_i * g(p_i))

g(x) = 2^96 / x   for token1 outcomes
g(x) = x / 2^96   for token0 outcomes
```

The active set is the maximal profitability prefix that still clears the implied
frontier.

For exact mode, the same frontier is solved against exact multi-tick cost
curves instead of the constant-`L` approximation.

### Phase 5: Execute buys

For each active pool:

- compute the target `sqrtPriceX96` implied by the solved frontier,
- submit one `exactInputSingle` with the full remaining collateral budget, and
- let Uniswap stop the swap at the pool-specific price limit.

This keeps solve and execute atomic while making the router do the exact
tick-walking execution.

### Phase 6: Recycle below-frontier holdings

After the forward waterfall, some held tokens may sit below the solved
profitability frontier. The recycle path:

1. recomputes the current frontier,
2. sells only below-frontier holdings that clear round-trip fee drag,
3. uses the frontier price as the sell limit, and
4. redeploys recovered collateral through the same route that was chosen for the
   entrypoint.

`minRecycleProfitCollateral` is the coarse caller-supplied gate for this phase.

### Phase 7: Return all holdings

All remaining collateral and outcome balances are returned to the caller before
the transaction ends.

## Current Route Policy

The current direct-route frontier is intentionally simple:

- default to `rebalance`,
- escalate to `rebalanceExact` only when the expected EV uplift is material and
  the exact route stays within the gas budget,
- do not reintroduce intermediate approximate routes without new evidence on
  realistic 98-market fixtures.

That rule is permanent until new benchmarks prove otherwise.

## Benchmarks

Measured on March 7, 2026 after removing the bounded/adaptive branches:

### Two-pool multi-tick fixture

- `rebalance`: `213,077` gas, EV `6,533,623,513,833,670,712,014`
- `rebalanceExact`: `559,645` gas, EV `6,538,044,206,827,416,201,383`
- exact EV uplift: about `0.0677%`

### Synthetic 98-outcome multi-tick fixture

- `rebalance`: `3,617,331` gas, EV `319,743,852,785,843,393,829,282`
- `rebalanceExact`: `38,909,824` gas, EV `320,463,674,792,566,460,803,522`
- exact EV uplift: about `0.2251%`

### Realistic seeded 98-outcome multi-tick fixture

- `rebalance`: `17,148,395` gas, EV `329,064,319,625,327,189,099,520,213`
- `rebalanceExact`: `36,827,551` gas, EV `425,223,125,100,456,027,087,034,824`
- exact EV uplift: about `29.22%`

The practical conclusion is that exact is worth the complexity only on the
state families where it buys a large EV jump. For the approximate baseline,
simple constant-`L` remains the correct default.

## Test Map

Primary coverage in `test/Rebalancer.t.sol`:

- `testRebalanceNoOpWhenPredictionInsideCurrentTick`
- `testExactSolverUsesLiquidityNetAcrossInitializedTick`
- `testRecycleFloorSkipsBelowThresholdHoldings`
- `testRebalanceAndArbWithFloorsSkipsSubThresholdArb`
- `testBenchmarkABMultiTickConstantLVsExact`
- `testBenchmarkABMultiTickSyntheticNinetyEightOutcomeConstantLVsExact`
- `testBenchmarkABMultiTickRealisticSeededNinetyEightOutcomeConstantLVsExact`

Cross-approach parity and mixed-route comparisons live in
`test/RebalancerAB.t.sol`.

## Contract Surface

```solidity
function rebalance(RebalanceParams calldata params)
    external
    returns (uint256 totalProceeds, uint256 totalSpent);

function rebalanceExact(
    RebalanceParams calldata params,
    uint256 maxBisectionIterations,
    uint256 maxTickCrossingsPerPool
) external returns (uint256 totalProceeds, uint256 totalSpent);

function rebalanceAndArb(
    RebalanceParams calldata params,
    address market,
    uint256 maxArbRounds,
    uint256 maxRecycleRounds
) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);

function rebalanceAndArbWithFloors(
    RebalanceParams calldata params,
    address market,
    uint256 maxArbRounds,
    uint256 maxRecycleRounds,
    uint256 minArbProfitCollateral,
    uint256 minRecycleProfitCollateral
) public returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);

function rebalanceAndArbExact(
    RebalanceParams calldata params,
    address market,
    uint256 maxArbRounds,
    uint256 maxRecycleRounds,
    uint256 maxBisectionIterations,
    uint256 maxTickCrossingsPerPool
) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);

function rebalanceAndArbExactWithFloors(
    RebalanceParams calldata params,
    address market,
    uint256 maxArbRounds,
    uint256 maxRecycleRounds,
    uint256 maxBisectionIterations,
    uint256 maxTickCrossingsPerPool,
    uint256 minArbProfitCollateral,
    uint256 minRecycleProfitCollateral
) public returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);
```
