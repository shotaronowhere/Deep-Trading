# RebalancerMixed Constant-L Solver Notes

This note documents the implemented constant-L mixed-route path in `contracts/RebalancerMixed.sol` (`rebalanceMixedConstantL`).

## Model Boundary

- uses current `slot0` and in-range `liquidity()` snapshots,
- assumes single-tick analytical behavior for the solve,
- executes as one mixed pass (`mint/sell non-active`) followed by direct frontier buys,
- fails closed to direct-only constant-L when mixed gates fail.

## Variables

- `pi` (WAD): frontier profitability scalar (`>= 0`)
- `rho = sqrt(1 + pi)` (WAD): frontier sqrt-ratio multiplier
- `M`: aggregate complete-set mint amount

## Set Construction

1. Build direct underpriced order using `_sortedUnderpricedByPriority`.
2. Compute direct prefix with `_computePsiFromSorted`.
3. Use that prefix as active set `A` for the mixed candidate.
4. Non-active set `N` is the complement of `A`.

## Frontier and Limits

Given `rho`:

- token1 frontier limit: `sqrtPred * rho`
- token0 frontier limit: `sqrtPred / rho`

(implemented via `_frontierLimit(..., rho, ONE_WAD)`).

## Inner Mint Solve

For a fixed `pi`:

1. Build non-active per-pool curves at this frontier.
2. For each non-active pool, sell amount is `min(M, cap_i)` where `cap_i` is frontier-reachable sell cap.
3. Evaluate `delta(M)` as total reduction in non-active outcome prices.
4. Bisection solve the minimum `M` that reaches target `delta(pi)`.

## Outer Solve

Bisection on `pi` with objective:

`direct_cost(pi) + mint_net_cost(M(pi)) <= budget`

- `direct_cost`: summed active direct costs to frontier
- `mint_net_cost`: `M - sell_proceeds`

The selected `pi` is the lowest feasible frontier within tolerance.

## Safety Gates

Before mixed execution:

- active set-size cap is currently disabled (`MAX_MIXED_ACTIVE = type(uint256).max`) for benchmark exploration,
- non-active sellability guard,
- inner/outer solve success,
- zero-mint rejection,
- residual consistency tolerance check (`MINT_RESIDUAL_TOL_WAD`).

Any failure emits `MixedSolveFallback(reasonCode)` and runs direct-only `_readAndBuyConstantL`.

## Execution Order

1. split `M` collateral into full set,
2. sell only freshly minted non-active legs to frontier,
3. merge residual complete sets,
4. execute direct buys to the same frontier with remaining collateral.

## Current Limitation

`rebalanceMixedConstantL` currently prioritizes fail-closed behavior over aggressive mixed execution. In adverse fixture states this can intentionally fall back to direct-only constant-L.
