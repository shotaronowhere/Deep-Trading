# On-Chain Rebalancer (AlgebraV1.9 / Swapr)

## Overview

`RebalancerAlgebra.sol` is a fork of `Rebalancer.sol` adapted for **AlgebraV1.9 (Swapr)** DEX on Gnosis chain. It targets scalar futarchy markets where each child market has UP/DOWN outcome tokens paired with an `underlyingToken` as collateral.

The solver logic, waterfall allocation, and arb mechanism are identical to the Uniswap V3 version. Only the DEX interface layer differs.

## Differences from Rebalancer.sol (Uniswap V3)

### Router interface

| Uniswap V3 (SwapRouter02) | AlgebraV1.9 (Swapr) |
|---|---|
| `uint24 fee` in ExactInputSingleParams | Removed -- no fee field |
| `sqrtPriceLimitX96` | `limitSqrtPrice` (same Q64.96 semantics) |
| No deadline field | `uint256 deadline` added (`block.timestamp`) |

### Pool state reads

| Uniswap V3 | AlgebraV1.9 |
|---|---|
| `slot0()` -> 7 returns | `globalState()` -> 7 returns (different fields) |
| Fee is a fixed constructor param | Fee is dynamic, read from `globalState()` 3rd return (`uint16`) |
| `tickBitmap(int16 wordPos)` | `tickTable(int16 wordPos)` |
| Bitmap uses compressed ticks: `tick / spacing` | Bitmap uses raw ticks (no spacing compression) |
| `ticks()` returns `liquidityNet` | `ticks()` returns `liquidityDelta` (same semantics) |

### Fee handling

- **Uniswap V3**: Single `uint24 fee` passed in `RebalanceParams`, same for all pools
- **AlgebraV1.9**: Per-pool dynamic fee read from `globalState().fee` at execution time

The `RebalanceParams` struct no longer has a `fee` field. All fee-dependent math uses exact per-pool fees:

- **Psi computation**: `C = sum(c_i * FEE_UNITS / (FEE_UNITS - fee_i))`, `D = sum(d_i * FEE_UNITS / (FEE_UNITS - fee_i))`, `psi = (C + budget) / D`
- **Arb thresholds**: `mintYield = sum(price_i * (1 - fee_i/1e6))`, `buyCost = sum(price_i / (1 - fee_i/1e6))`
- **No average fee approximation** -- each pool's fee is weighted individually

### Tick scanning

- `_tickBitmapPosition`: `wordPos = tick >> 8`, `bitPos = tick % 256` (raw tick, not compressed by spacing)
- `_nextInitializedTickBelow/Above`: Pass raw tick directly to bitmap lookup (no division/multiplication by spacing)
- `_checkedAddCrossings`: Counts 1 per bitmap row scan (= 1 SLOAD), not `delta / spacing`
- `_tickDistance`: Removed (was V3-specific)

### What does NOT change

- Tick spacing alignment (`_floorToSpacing` still valid)
- All pure math helpers (`_sqrt`, `_fullMul`, `_coefficientPair`, etc.)
- Waterfall allocation logic
- Complete-set arb mechanism
- Exact solver bisection
- MSB/LSB bit scanning

## Key Addresses

See [deployments.md](deployments.md#gnosis-chain-swapr--algebrav19-addresses) for Gnosis chain contract addresses.

## Known Considerations

**Dynamic fee staleness**: AlgebraV1.9 recomputes fees on the first swap in a new block. The fee read from `globalState()` before any swap may be slightly stale. This affects budget estimation precision but not execution safety, since every swap uses `limitSqrtPrice` to enforce price bounds. The impact is negligible for pools with gradual fee adaptation.

## Source of Truth

The contract targets **AlgebraV1.9** as deployed by Swapr on Gnosis chain:
- Repository: `https://github.com/cryptoalgebra/AlgebraV1.9`
- NOT `https://github.com/cryptoalgebra/Algebra/tree/v1.9/` (Algebra Integral -- different codebase)
