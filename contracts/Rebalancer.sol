// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IV3SwapRouter} from "./interfaces/IV3SwapRouter.sol";
import {IERC20} from "./interfaces/IERC20.sol";
import {ICTFRouter} from "./interfaces/ICTFRouter.sol";
import {IUniswapV3Pool} from "./interfaces/IUniswapV3Pool.sol";
import {FullMath} from "./libraries/FullMath.sol";
import {TickMath} from "./libraries/TickMath.sol";

/// @title On-chain portfolio rebalancer with closed-form waterfall allocation
/// @notice Sells overpriced outcomes, then buys underpriced outcomes with profitability-equalized allocation.
/// @dev Uses the closed-form ψ = (C + budget×(1-fee)) / D to compute target prices analytically.
///      All underpriced outcomes end at the same profitability level π = 1/ψ² - 1, which is optimal.
contract Rebalancer {
    error TransferFailed();
    error ApprovalFailed();
    error TickScanLimitExceeded();

    IV3SwapRouter public immutable router;
    ICTFRouter public immutable ctfRouter;
    uint256 constant Q96 = 1 << 96;
    uint256 constant FEE_UNITS = 1e6;
    uint256 constant MAX_LEGACY_WATERFALL_PASSES = 6;
    uint256 constant EXACT_COST_TOL = 1e16;
    uint256 constant PSI_WAD = 1e18;

    struct RebalanceParams {
        address[] tokens;
        address[] pools;
        bool[] isToken1;
        uint256[] balances;
        uint256 collateralAmount;
        uint160[] sqrtPredX96;
        address collateral;
        uint24 fee;
    }

    // Intermediate state passed between phases to avoid stack-too-deep
    struct PsiResult {
        uint256 num;
        uint256 den;
        bool buyAll;
    }

    // Flattened per-pool exact cost ladders used by the exact solver so bisection
    // can reuse one storage scan instead of rescanning ticks on every psi guess.
    struct ExactCostCurves {
        uint256[] offsets;
        uint256[] counts;
        uint256[] totalCosts;
        uint160[] segmentEnds;
        uint128[] segmentLiquidities;
        uint256[] segmentPrefixCosts;
    }

    constructor(address _router, address _ctfRouter) {
        router = IV3SwapRouter(_router);
        ctfRouter = ICTFRouter(_ctfRouter);
    }

    /// @notice Full rebalance: sell overpriced, waterfall-buy underpriced, return all.
    function rebalance(RebalanceParams calldata params) external returns (uint256 totalProceeds, uint256 totalSpent) {
        totalProceeds = _pullAndSell(params);
        totalSpent = _readAndBuy(params);
        _returnAll(params);
    }

    /// @notice Full rebalance using an exact tick-scanned cost solver.
    ///         Solves psi by bisection over exact per-pool multi-tick costs, then executes one swap per pool.
    function rebalanceExact(
        RebalanceParams calldata params,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool
    ) external returns (uint256 totalProceeds, uint256 totalSpent) {
        totalProceeds = _pullAndSell(params);
        totalSpent = _readAndBuyExact(params, maxBisectionIterations, maxTickCrossingsPerPool);
        _returnAll(params);
    }

    /// @notice Full rebalance + pre-buy arb + recycle.
    ///         1. Pull tokens, sell overpriced
    ///         2. Complete-set arb (before waterfall consumes cash)
    ///         3. Waterfall buy (constant-L)
    ///         4. Recycle: sell below-frontier holdings, re-waterfall
    ///         5. Return all holdings to caller
    function rebalanceAndArb(
        RebalanceParams calldata params,
        address market,
        uint256 maxArbRounds,
        uint256 maxRecycleRounds
    ) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit) {
        return rebalanceAndArbWithFloors(params, market, maxArbRounds, maxRecycleRounds, 0, 0);
    }

    /// @notice Full rebalance + arb + recycle with coarse collateral profit floors for churn-heavy rounds.
    function rebalanceAndArbWithFloors(
        RebalanceParams calldata params,
        address market,
        uint256 maxArbRounds,
        uint256 maxRecycleRounds,
        uint256 minArbProfitCollateral,
        uint256 minRecycleProfitCollateral
    ) public returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit) {
        totalProceeds = _pullAndSell(params);

        // Arb first so deterministic complete-set profit gets first access to cash.
        arbProfit = _arbLoopWithFloor(
            params.tokens,
            params.pools,
            params.isToken1,
            params.collateral,
            market,
            params.fee,
            maxArbRounds,
            minArbProfitCollateral
        );

        totalSpent = _readAndBuy(params);

        // Recycle: sell below-frontier holdings, then re-run the constant-L waterfall.
        (uint256 recycleProceeds, uint256 recycleSpent) =
            _recycleSellWithFloor(params, maxRecycleRounds, minRecycleProfitCollateral);
        totalProceeds += recycleProceeds;
        totalSpent += recycleSpent;

        _returnAll(params);
    }

    /// @notice Full rebalance + pre-buy arb + recycle using the exact tick-scanned solver.
    function rebalanceAndArbExact(
        RebalanceParams calldata params,
        address market,
        uint256 maxArbRounds,
        uint256 maxRecycleRounds,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool
    ) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit) {
        return rebalanceAndArbExactWithFloors(
            params, market, maxArbRounds, maxRecycleRounds, maxBisectionIterations, maxTickCrossingsPerPool, 0, 0
        );
    }

    /// @notice Full rebalance + arb + recycle using the explicit exact path and coarse profit floors.
    function rebalanceAndArbExactWithFloors(
        RebalanceParams calldata params,
        address market,
        uint256 maxArbRounds,
        uint256 maxRecycleRounds,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool,
        uint256 minArbProfitCollateral,
        uint256 minRecycleProfitCollateral
    ) public returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit) {
        totalProceeds = _pullAndSell(params);

        arbProfit = _arbLoopWithFloor(
            params.tokens,
            params.pools,
            params.isToken1,
            params.collateral,
            market,
            params.fee,
            maxArbRounds,
            minArbProfitCollateral
        );

        totalSpent = _readAndBuyExact(params, maxBisectionIterations, maxTickCrossingsPerPool);

        (uint256 recycleProceeds, uint256 recycleSpent) = _recycleSellExactWithFloor(
            params, maxRecycleRounds, maxBisectionIterations, maxTickCrossingsPerPool, minRecycleProfitCollateral
        );
        totalProceeds += recycleProceeds;
        totalSpent += recycleSpent;

        _returnAll(params);
    }

    // ──────────────────────────────────────────────
    // Buy path: read pools, compute ψ, buy
    // ──────────────────────────────────────────────

    function _readAndBuy(RebalanceParams calldata params) internal returns (uint256) {
        return _readAndBuyConstantL(params);
    }

    function _readAndBuyConstantL(RebalanceParams calldata params) internal returns (uint256 totalSpent) {
        for (uint256 pass = 0; pass < MAX_LEGACY_WATERFALL_PASSES;) {
            uint256 budgetBefore = IERC20(params.collateral).balanceOf(address(this));
            if (budgetBefore == 0) break;

            (uint160[] memory sqrtPrices, uint128[] memory liquidities) = _readPoolState(params);
            (uint256[] memory order, uint160[] memory limits, uint256 count) = _buildConstantLBuyPlan(
                sqrtPrices, liquidities, params.sqrtPredX96, params.isToken1, budgetBefore, params.fee
            );
            if (count == 0) break;

            uint256 spent = _executeBuyPlan(params, order, limits, count, budgetBefore);
            if (spent == 0) break;
            totalSpent += spent;
            if (spent == budgetBefore) break;

            unchecked {
                ++pass;
            }
        }
    }

    function _readAndBuyExact(
        RebalanceParams calldata params,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool
    ) internal returns (uint256) {
        uint256 budget = IERC20(params.collateral).balanceOf(address(this));
        if (budget == 0) return 0;

        (uint160[] memory sqrtPrices, int24[] memory ticks, uint128[] memory liquidities) = _readExactPoolState(params);
        (uint256[] memory order, uint256 count) =
            _sortedUnderpricedByPriority(sqrtPrices, liquidities, params.sqrtPredX96, params.isToken1);
        (ExactCostCurves memory curves, uint256 buyAllCost) =
            _buildExactCostCurves(params, sqrtPrices, ticks, liquidities, maxTickCrossingsPerPool, budget);

        PsiResult memory psi = _computePsiExact(params, sqrtPrices, curves, buyAllCost, budget, maxBisectionIterations);

        return _waterfallBuyOrdered(params, sqrtPrices, order, count, psi);
    }

    function _computePsiExact(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        ExactCostCurves memory curves,
        uint256 buyAllCost,
        uint256 budget,
        uint256 maxBisectionIterations
    ) internal pure returns (PsiResult memory psi) {
        PsiResult memory buyAllPsi = PsiResult({num: PSI_WAD, den: PSI_WAD, buyAll: true});
        if (buyAllCost <= budget) {
            return buyAllPsi;
        }

        uint256 lo = 1;
        uint256 hi = PSI_WAD;
        for (uint256 iter = 0; iter < maxBisectionIterations;) {
            if (lo == hi) break;
            uint256 mid = (lo + hi + 1) / 2;
            PsiResult memory midPsi = PsiResult({num: mid, den: PSI_WAD, buyAll: false});
            uint256 exactCost = _exactTotalCostFromCurves(params, sqrtPrices, curves, midPsi);

            if (exactCost <= budget) {
                if (budget - exactCost <= _exactCostTolerance(budget)) {
                    return midPsi;
                }
                lo = mid;
            } else {
                hi = mid - 1;
            }

            unchecked {
                ++iter;
            }
        }

        psi = PsiResult({num: lo, den: PSI_WAD, buyAll: false});
    }

    function _buildExactCostCurves(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        int24[] memory ticks,
        uint128[] memory liquidities,
        uint256 maxTickCrossingsPerPool,
        uint256 costCap
    ) internal view returns (ExactCostCurves memory curves, uint256 totalCost) {
        uint256 n = params.tokens.length;
        uint256 activeCount = 0;
        for (uint256 i = 0; i < n;) {
            if (params.isToken1[i] ? sqrtPrices[i] > params.sqrtPredX96[i] : sqrtPrices[i] < params.sqrtPredX96[i]) {
                activeCount++;
            }

            unchecked {
                ++i;
            }
        }

        uint256 capacity = activeCount * (maxTickCrossingsPerPool + 1);
        curves = ExactCostCurves({
            offsets: new uint256[](n),
            counts: new uint256[](n),
            totalCosts: new uint256[](n),
            segmentEnds: new uint160[](capacity),
            segmentLiquidities: new uint128[](capacity),
            segmentPrefixCosts: new uint256[](capacity)
        });

        uint256 cursor = 0;
        for (uint256 i = 0; i < n; i++) {
            if (params.isToken1[i] ? sqrtPrices[i] <= params.sqrtPredX96[i] : sqrtPrices[i] >= params.sqrtPredX96[i]) {
                continue;
            }

            curves.offsets[i] = cursor;
            IUniswapV3Pool v3Pool = IUniswapV3Pool(params.pools[i]);
            if (params.isToken1[i]) {
                (cursor, curves.totalCosts[i]) = _buildExactDescendingCurve(
                    curves,
                    cursor,
                    v3Pool,
                    sqrtPrices[i],
                    ticks[i],
                    liquidities[i],
                    params.sqrtPredX96[i],
                    params.fee,
                    maxTickCrossingsPerPool,
                    costCap
                );
            } else {
                (cursor, curves.totalCosts[i]) = _buildExactAscendingCurve(
                    curves,
                    cursor,
                    v3Pool,
                    sqrtPrices[i],
                    ticks[i],
                    liquidities[i],
                    params.sqrtPredX96[i],
                    params.fee,
                    maxTickCrossingsPerPool,
                    costCap
                );
            }

            curves.counts[i] = cursor - curves.offsets[i];
            totalCost += curves.totalCosts[i];
        }
    }

    function _buildExactCostCurvesForMask(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        int24[] memory ticks,
        uint128[] memory liquidities,
        bool[] memory exactMask,
        uint256 maxTickCrossingsPerPool,
        uint256 capacity,
        uint256 costCap
    ) internal view returns (ExactCostCurves memory curves, uint256 totalCost) {
        uint256 n = params.tokens.length;
        curves = ExactCostCurves({
            offsets: new uint256[](n),
            counts: new uint256[](n),
            totalCosts: new uint256[](n),
            segmentEnds: new uint160[](capacity),
            segmentLiquidities: new uint128[](capacity),
            segmentPrefixCosts: new uint256[](capacity)
        });

        uint256 cursor = 0;
        for (uint256 i = 0; i < n; i++) {
            if (!exactMask[i]) continue;

            curves.offsets[i] = cursor;
            IUniswapV3Pool v3Pool = IUniswapV3Pool(params.pools[i]);
            if (params.isToken1[i]) {
                (cursor, curves.totalCosts[i]) = _buildExactDescendingCurve(
                    curves,
                    cursor,
                    v3Pool,
                    sqrtPrices[i],
                    ticks[i],
                    liquidities[i],
                    params.sqrtPredX96[i],
                    params.fee,
                    maxTickCrossingsPerPool,
                    costCap
                );
            } else {
                (cursor, curves.totalCosts[i]) = _buildExactAscendingCurve(
                    curves,
                    cursor,
                    v3Pool,
                    sqrtPrices[i],
                    ticks[i],
                    liquidities[i],
                    params.sqrtPredX96[i],
                    params.fee,
                    maxTickCrossingsPerPool,
                    costCap
                );
            }

            curves.counts[i] = cursor - curves.offsets[i];
            totalCost += curves.totalCosts[i];
        }
    }

    function _exactTotalCostFromCurves(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        ExactCostCurves memory curves,
        PsiResult memory psi
    ) internal pure returns (uint256 totalCost) {
        uint256 n = params.tokens.length;
        for (uint256 i = 0; i < n; i++) {
            if (curves.counts[i] == 0) continue;

            uint160 limit = _buyLimit(sqrtPrices[i], params.sqrtPredX96[i], params.isToken1[i], psi);
            if (limit == 0) continue;

            totalCost += _exactCostFromCurve(params, sqrtPrices, curves, i, limit);
        }
    }

    function _exactCostFromCurve(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        ExactCostCurves memory curves,
        uint256 poolIndex,
        uint160 limit
    ) internal pure returns (uint256) {
        uint256 count = curves.counts[poolIndex];
        if (count == 0) return 0;

        uint256 offset = curves.offsets[poolIndex];
        uint160 segmentStart = sqrtPrices[poolIndex];
        bool isT1 = params.isToken1[poolIndex];

        for (uint256 j = 0; j < count;) {
            uint256 idx = offset + j;
            uint160 segmentEnd = curves.segmentEnds[idx];

            if (isT1 ? limit >= segmentEnd : limit <= segmentEnd) {
                uint256 segmentCost = isT1
                    ? _segmentCostToken1(segmentStart, limit, curves.segmentLiquidities[idx], params.fee)
                    : _segmentCostToken0(segmentStart, limit, curves.segmentLiquidities[idx], params.fee);
                return curves.segmentPrefixCosts[idx] + segmentCost;
            }

            segmentStart = segmentEnd;

            unchecked {
                ++j;
            }
        }

        return curves.totalCosts[poolIndex];
    }

    function _exactCostToLimit(
        address pool,
        uint160 sqrtPrice,
        int24 tick,
        uint128 liquidity,
        uint160 limit,
        bool isT1,
        uint24 fee,
        uint256 maxTickCrossingsPerPool
    ) internal view returns (uint256 cost) {
        if (limit == sqrtPrice) return 0;

        IUniswapV3Pool v3Pool = IUniswapV3Pool(pool);
        int24 spacing = v3Pool.tickSpacing();
        int24 minTick = _minUsableTick(spacing);
        int24 maxTick = _maxUsableTick(spacing);
        uint160 current = sqrtPrice;
        int24 lowerTick = _floorToSpacing(tick, spacing);
        if (lowerTick < minTick) lowerTick = minTick;
        int24 upperTick = lowerTick + spacing;
        if (upperTick > maxTick) upperTick = maxTick;
        uint256 crossings = 0;

        if (isT1) {
            uint160 segmentFloor = TickMath.getSqrtRatioAtTick(lowerTick);
            if (limit >= segmentFloor) {
                return _segmentCostToken1(current, limit, liquidity, fee);
            }

            cost += _segmentCostToken1(current, segmentFloor, liquidity, fee);
            if (maxTickCrossingsPerPool == 0) revert TickScanLimitExceeded();

            (, int128 lowerLiquidityNet,,,,,,) = v3Pool.ticks(lowerTick);
            liquidity = _applyLiquidityNetDescending(liquidity, lowerLiquidityNet);
            crossings = 1;
            current = segmentFloor;

            int24 lowerBoundaryTick = lowerTick;
            while (true) {
                if (lowerBoundaryTick == minTick) {
                    cost += _segmentCostToken1(current, limit, liquidity, fee);
                    break;
                }

                (int24 nextTick, bool initialized) =
                    _nextInitializedTickBelow(v3Pool, lowerBoundaryTick, spacing, minTick);
                uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
                if (limit >= nextSqrt) {
                    cost += _segmentCostToken1(current, limit, liquidity, fee);
                    break;
                }

                cost += _segmentCostToken1(current, nextSqrt, liquidity, fee);
                crossings =
                    _checkedAddCrossings(crossings, lowerBoundaryTick, nextTick, spacing, maxTickCrossingsPerPool);

                if (initialized) {
                    (, int128 nextLiquidityNet,,,,,,) = v3Pool.ticks(nextTick);
                    liquidity = _applyLiquidityNetDescending(liquidity, nextLiquidityNet);
                }

                current = nextSqrt;
                lowerBoundaryTick = nextTick;
            }

            return cost;
        }

        uint160 segmentCeiling = TickMath.getSqrtRatioAtTick(upperTick);
        if (limit <= segmentCeiling) {
            return _segmentCostToken0(current, limit, liquidity, fee);
        }

        cost += _segmentCostToken0(current, segmentCeiling, liquidity, fee);
        if (maxTickCrossingsPerPool == 0) revert TickScanLimitExceeded();

        (, int128 upperLiquidityNet,,,,,,) = v3Pool.ticks(upperTick);
        liquidity = _applyLiquidityNetAscending(liquidity, upperLiquidityNet);
        crossings = 1;
        current = segmentCeiling;

        int24 upperBoundaryTick = upperTick;
        while (true) {
            if (upperBoundaryTick == maxTick) {
                cost += _segmentCostToken0(current, limit, liquidity, fee);
                break;
            }

            (int24 nextTick, bool initialized) = _nextInitializedTickAbove(v3Pool, upperBoundaryTick, spacing, maxTick);
            uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
            if (limit <= nextSqrt) {
                cost += _segmentCostToken0(current, limit, liquidity, fee);
                break;
            }

            cost += _segmentCostToken0(current, nextSqrt, liquidity, fee);
            crossings = _checkedAddCrossings(crossings, upperBoundaryTick, nextTick, spacing, maxTickCrossingsPerPool);

            if (initialized) {
                (, int128 nextLiquidityNet,,,,,,) = v3Pool.ticks(nextTick);
                liquidity = _applyLiquidityNetAscending(liquidity, nextLiquidityNet);
            }

            current = nextSqrt;
            upperBoundaryTick = nextTick;
        }
    }

    function _buildExactDescendingCurve(
        ExactCostCurves memory curves,
        uint256 cursor,
        IUniswapV3Pool v3Pool,
        uint160 sqrtPrice,
        int24 tick,
        uint128 liquidity,
        uint160 limit,
        uint24 fee,
        uint256 maxTickCrossingsPerPool,
        uint256 costCap
    ) internal view returns (uint256 nextCursor, uint256 totalCost) {
        int24 spacing = v3Pool.tickSpacing();
        int24 minTick = _minUsableTick(spacing);
        int24 lowerTick = _floorToSpacing(tick, spacing);
        if (lowerTick < minTick) lowerTick = minTick;

        uint160 current = sqrtPrice;
        uint256 crossings = 0;
        nextCursor = cursor;
        bool capped;

        uint160 segmentFloor = TickMath.getSqrtRatioAtTick(lowerTick);
        if (limit >= segmentFloor) {
            (nextCursor, totalCost, capped) =
                _appendExactToken1Segment(curves, nextCursor, current, limit, liquidity, fee, totalCost, costCap);
            return (nextCursor, totalCost);
        }

        (nextCursor, totalCost, capped) =
            _appendExactToken1Segment(curves, nextCursor, current, segmentFloor, liquidity, fee, totalCost, costCap);
        if (capped) return (nextCursor, totalCost);
        if (maxTickCrossingsPerPool == 0) revert TickScanLimitExceeded();

        (, int128 lowerLiquidityNet,,,,,,) = v3Pool.ticks(lowerTick);
        liquidity = _applyLiquidityNetDescending(liquidity, lowerLiquidityNet);
        crossings = 1;
        current = segmentFloor;

        int24 lowerBoundaryTick = lowerTick;
        while (true) {
            if (lowerBoundaryTick == minTick) {
                (nextCursor, totalCost, capped) =
                    _appendExactToken1Segment(curves, nextCursor, current, limit, liquidity, fee, totalCost, costCap);
                break;
            }

            (int24 nextTick, bool initialized) = _nextInitializedTickBelow(v3Pool, lowerBoundaryTick, spacing, minTick);
            uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
            if (limit >= nextSqrt) {
                (nextCursor, totalCost, capped) =
                    _appendExactToken1Segment(curves, nextCursor, current, limit, liquidity, fee, totalCost, costCap);
                break;
            }

            (nextCursor, totalCost, capped) =
                _appendExactToken1Segment(curves, nextCursor, current, nextSqrt, liquidity, fee, totalCost, costCap);
            if (capped) break;
            crossings = _checkedAddCrossings(crossings, lowerBoundaryTick, nextTick, spacing, maxTickCrossingsPerPool);

            if (initialized) {
                (, int128 nextLiquidityNet,,,,,,) = v3Pool.ticks(nextTick);
                liquidity = _applyLiquidityNetDescending(liquidity, nextLiquidityNet);
            }

            current = nextSqrt;
            lowerBoundaryTick = nextTick;
        }
    }

    function _buildExactAscendingCurve(
        ExactCostCurves memory curves,
        uint256 cursor,
        IUniswapV3Pool v3Pool,
        uint160 sqrtPrice,
        int24 tick,
        uint128 liquidity,
        uint160 limit,
        uint24 fee,
        uint256 maxTickCrossingsPerPool,
        uint256 costCap
    ) internal view returns (uint256 nextCursor, uint256 totalCost) {
        int24 spacing = v3Pool.tickSpacing();
        int24 maxTick = _maxUsableTick(spacing);
        int24 upperTick = _floorToSpacing(tick, spacing) + spacing;
        if (upperTick > maxTick) upperTick = maxTick;

        uint160 current = sqrtPrice;
        uint256 crossings = 0;
        nextCursor = cursor;
        bool capped;

        uint160 segmentCeiling = TickMath.getSqrtRatioAtTick(upperTick);
        if (limit <= segmentCeiling) {
            (nextCursor, totalCost, capped) =
                _appendExactToken0Segment(curves, nextCursor, current, limit, liquidity, fee, totalCost, costCap);
            return (nextCursor, totalCost);
        }

        (nextCursor, totalCost, capped) =
            _appendExactToken0Segment(curves, nextCursor, current, segmentCeiling, liquidity, fee, totalCost, costCap);
        if (capped) return (nextCursor, totalCost);
        if (maxTickCrossingsPerPool == 0) revert TickScanLimitExceeded();

        (, int128 upperLiquidityNet,,,,,,) = v3Pool.ticks(upperTick);
        liquidity = _applyLiquidityNetAscending(liquidity, upperLiquidityNet);
        crossings = 1;
        current = segmentCeiling;

        int24 upperBoundaryTick = upperTick;
        while (true) {
            if (upperBoundaryTick == maxTick) {
                (nextCursor, totalCost, capped) =
                    _appendExactToken0Segment(curves, nextCursor, current, limit, liquidity, fee, totalCost, costCap);
                break;
            }

            (int24 nextTick, bool initialized) = _nextInitializedTickAbove(v3Pool, upperBoundaryTick, spacing, maxTick);
            uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
            if (limit <= nextSqrt) {
                (nextCursor, totalCost, capped) =
                    _appendExactToken0Segment(curves, nextCursor, current, limit, liquidity, fee, totalCost, costCap);
                break;
            }

            (nextCursor, totalCost, capped) =
                _appendExactToken0Segment(curves, nextCursor, current, nextSqrt, liquidity, fee, totalCost, costCap);
            if (capped) break;
            crossings = _checkedAddCrossings(crossings, upperBoundaryTick, nextTick, spacing, maxTickCrossingsPerPool);

            if (initialized) {
                (, int128 nextLiquidityNet,,,,,,) = v3Pool.ticks(nextTick);
                liquidity = _applyLiquidityNetAscending(liquidity, nextLiquidityNet);
            }

            current = nextSqrt;
            upperBoundaryTick = nextTick;
        }
    }

    function _appendExactCurveSegment(
        ExactCostCurves memory curves,
        uint256 cursor,
        uint160 segmentEnd,
        uint128 segmentLiquidity,
        uint256 prefixCost
    ) internal pure returns (uint256 nextCursor) {
        curves.segmentEnds[cursor] = segmentEnd;
        curves.segmentLiquidities[cursor] = segmentLiquidity;
        curves.segmentPrefixCosts[cursor] = prefixCost;
        nextCursor = cursor + 1;
    }

    function _appendExactToken1Segment(
        ExactCostCurves memory curves,
        uint256 cursor,
        uint160 start,
        uint160 end,
        uint128 liquidity,
        uint24 fee,
        uint256 prefixCost,
        uint256 costCap
    ) internal pure returns (uint256 nextCursor, uint256 nextTotalCost, bool capped) {
        nextCursor = _appendExactCurveSegment(curves, cursor, end, liquidity, prefixCost);
        nextTotalCost = prefixCost + _segmentCostToken1(start, end, liquidity, fee);
        if (nextTotalCost > costCap) {
            nextTotalCost = _exactCostCapExceeded(costCap);
            capped = true;
        }
    }

    function _appendExactToken0Segment(
        ExactCostCurves memory curves,
        uint256 cursor,
        uint160 start,
        uint160 end,
        uint128 liquidity,
        uint24 fee,
        uint256 prefixCost,
        uint256 costCap
    ) internal pure returns (uint256 nextCursor, uint256 nextTotalCost, bool capped) {
        nextCursor = _appendExactCurveSegment(curves, cursor, end, liquidity, prefixCost);
        nextTotalCost = prefixCost + _segmentCostToken0(start, end, liquidity, fee);
        if (nextTotalCost > costCap) {
            nextTotalCost = _exactCostCapExceeded(costCap);
            capped = true;
        }
    }

    function _nextInitializedTickBelow(IUniswapV3Pool pool, int24 currentTick, int24 spacing, int24 minTick)
        internal
        view
        returns (int24 nextTick, bool initialized)
    {
        int24 compressed = currentTick / spacing;
        (int24 nextCompressed, bool found) = _nextInitializedTickWithinOneWord(pool, compressed - 1, true);
        int24 candidate = int24(int256(nextCompressed) * int256(spacing));
        if (candidate < minTick) {
            candidate = minTick;
            found = false;
        }

        return (candidate, found);
    }

    function _nextInitializedTickAbove(IUniswapV3Pool pool, int24 currentTick, int24 spacing, int24 maxTick)
        internal
        view
        returns (int24 nextTick, bool initialized)
    {
        int24 compressed = currentTick / spacing;
        (int24 nextCompressed, bool found) = _nextInitializedTickWithinOneWord(pool, compressed, false);
        int24 candidate = int24(int256(nextCompressed) * int256(spacing));
        if (candidate > maxTick) {
            candidate = maxTick;
            found = false;
        }

        return (candidate, found);
    }

    function _nextInitializedTickWithinOneWord(IUniswapV3Pool pool, int24 compressed, bool lte)
        internal
        view
        returns (int24 nextCompressed, bool initialized)
    {
        if (lte) {
            (int16 lteWordPos, uint8 lteBitPos) = _tickBitmapPosition(compressed);
            uint256 lteMask = ((uint256(1) << lteBitPos) - 1) | (uint256(1) << lteBitPos);
            uint256 lteMasked = pool.tickBitmap(lteWordPos) & lteMask;

            initialized = lteMasked != 0;
            if (initialized) {
                nextCompressed = compressed - int24(uint24(lteBitPos - _mostSignificantBit(lteMasked)));
            } else {
                nextCompressed = compressed - int24(uint24(lteBitPos));
            }
            return (nextCompressed, initialized);
        }

        (int16 gtWordPos, uint8 gtBitPos) = _tickBitmapPosition(compressed + 1);
        uint256 gtMask = ~((uint256(1) << gtBitPos) - 1);
        uint256 gtMasked = pool.tickBitmap(gtWordPos) & gtMask;

        initialized = gtMasked != 0;
        if (initialized) {
            nextCompressed = compressed + 1 + int24(uint24(_leastSignificantBit(gtMasked) - gtBitPos));
        } else {
            nextCompressed = compressed + 1 + int24(uint24(type(uint8).max - gtBitPos));
        }
    }

    function _tickBitmapPosition(int24 compressed) internal pure returns (int16 wordPos, uint8 bitPos) {
        wordPos = int16(compressed >> 8);
        bitPos = uint8(uint24(compressed % 256));
    }

    function _checkedAddCrossings(
        uint256 currentCrossings,
        int24 fromTick,
        int24 toTick,
        int24 spacing,
        uint256 maxTickCrossingsPerPool
    ) internal pure returns (uint256 nextCrossings) {
        nextCrossings = currentCrossings + _tickDistance(fromTick, toTick, spacing);
        if (nextCrossings > maxTickCrossingsPerPool) revert TickScanLimitExceeded();
    }

    function _tickDistance(int24 fromTick, int24 toTick, int24 spacing) internal pure returns (uint256) {
        int256 delta = int256(fromTick) - int256(toTick);
        if (delta < 0) delta = -delta;
        return uint256(delta) / uint256(uint24(spacing));
    }

    function _minUsableTick(int24 spacing) internal pure returns (int24) {
        return (TickMath.MIN_TICK / spacing) * spacing;
    }

    function _maxUsableTick(int24 spacing) internal pure returns (int24) {
        return (TickMath.MAX_TICK / spacing) * spacing;
    }

    function _mostSignificantBit(uint256 x) internal pure returns (uint8 r) {
        assert(x > 0);

        if (x >= 1 << 128) {
            x >>= 128;
            r += 128;
        }
        if (x >= 1 << 64) {
            x >>= 64;
            r += 64;
        }
        if (x >= 1 << 32) {
            x >>= 32;
            r += 32;
        }
        if (x >= 1 << 16) {
            x >>= 16;
            r += 16;
        }
        if (x >= 1 << 8) {
            x >>= 8;
            r += 8;
        }
        if (x >= 1 << 4) {
            x >>= 4;
            r += 4;
        }
        if (x >= 1 << 2) {
            x >>= 2;
            r += 2;
        }
        if (x >= 1 << 1) {
            r += 1;
        }
    }

    function _leastSignificantBit(uint256 x) internal pure returns (uint8 r) {
        assert(x > 0);
        r = 255;

        if ((x & type(uint128).max) > 0) {
            r -= 128;
        } else {
            x >>= 128;
        }
        if ((x & type(uint64).max) > 0) {
            r -= 64;
        } else {
            x >>= 64;
        }
        if ((x & type(uint32).max) > 0) {
            r -= 32;
        } else {
            x >>= 32;
        }
        if ((x & type(uint16).max) > 0) {
            r -= 16;
        } else {
            x >>= 16;
        }
        if ((x & type(uint8).max) > 0) {
            r -= 8;
        } else {
            x >>= 8;
        }
        if ((x & 0xF) > 0) {
            r -= 4;
        } else {
            x >>= 4;
        }
        if ((x & 0x3) > 0) {
            r -= 2;
        } else {
            x >>= 2;
        }
        if ((x & 0x1) > 0) {
            r -= 1;
        }
    }

    function _segmentCostToken1(uint160 start, uint160 end, uint128 liquidity, uint24 fee)
        internal
        pure
        returns (uint256)
    {
        if (end >= start) return 0;

        uint256 noFee = FullMath.mulDiv(uint256(liquidity), Q96, uint256(end))
            - FullMath.mulDiv(uint256(liquidity), Q96, uint256(start));
        return _mulDivRoundingUp(noFee, FEE_UNITS, FEE_UNITS - uint256(fee));
    }

    function _segmentCostToken0(uint160 start, uint160 end, uint128 liquidity, uint24 fee)
        internal
        pure
        returns (uint256)
    {
        if (end <= start) return 0;

        uint256 noFee = FullMath.mulDiv(uint256(liquidity), uint256(end) - uint256(start), Q96);
        return _mulDivRoundingUp(noFee, FEE_UNITS, FEE_UNITS - uint256(fee));
    }

    function _applyLiquidityNetDescending(uint128 liquidity, int128 liquidityNet) internal pure returns (uint128) {
        if (liquidityNet >= 0) {
            return liquidity - uint128(uint128(liquidityNet));
        }
        return liquidity + uint128(uint128(-liquidityNet));
    }

    function _applyLiquidityNetAscending(uint128 liquidity, int128 liquidityNet) internal pure returns (uint128) {
        if (liquidityNet >= 0) {
            return liquidity + uint128(uint128(liquidityNet));
        }
        return liquidity - uint128(uint128(-liquidityNet));
    }

    function _mulDivRoundingUp(uint256 a, uint256 b, uint256 denominator) internal pure returns (uint256 result) {
        result = FullMath.mulDiv(a, b, denominator);
        if (mulmod(a, b, denominator) > 0) {
            result++;
        }
    }

    function _exactCostTolerance(uint256 budget) internal pure returns (uint256 tol) {
        tol = budget / 100;
        if (tol > EXACT_COST_TOL) {
            tol = EXACT_COST_TOL;
        }
    }

    function _exactCostCapExceeded(uint256 costCap) internal pure returns (uint256 cappedCost) {
        if (costCap == type(uint256).max) return costCap;
        cappedCost = costCap + 1;
    }

    // ──────────────────────────────────────────────
    // Phase 1: Sell overpriced outcomes
    // ──────────────────────────────────────────────

    /// @dev Pull all caller tokens into contract, sell overpriced ones down to the fee-neutral boundary.
    ///      Unsold tokens stay in contract for recycling.
    function _pullAndSell(RebalanceParams calldata p) internal returns (uint256 amountOut) {
        uint256 n = p.tokens.length;

        if (p.collateralAmount != 0) {
            _safeTransferFrom(p.collateral, msg.sender, address(this), p.collateralAmount);
        }

        // Pull all tokens in
        for (uint256 i = 0; i < n; i++) {
            if (p.balances[i] == 0) continue;
            _safeTransferFrom(p.tokens[i], msg.sender, address(this), p.balances[i]);
        }

        // Sell overpriced tokens: check direction before selling to avoid SPL reverts
        for (uint256 i = 0; i < n; i++) {
            uint256 bal = IERC20(p.tokens[i]).balanceOf(address(this));
            if (bal == 0) continue;

            (uint160 sqrtPrice,,,,,,) = IUniswapV3Pool(p.pools[i]).slot0();
            uint160 sellLimit = _sellLimit(p.sqrtPredX96[i], p.isToken1[i], p.fee);

            // Only sell when the current edge clears the fee drag. If not, hold.
            bool sellable = p.isToken1[i] ? sqrtPrice < sellLimit : sqrtPrice > sellLimit;
            if (!sellable) continue;

            _safeApprove(p.tokens[i], address(router), bal);
            amountOut += router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: p.tokens[i],
                    tokenOut: p.collateral,
                    fee: p.fee,
                    recipient: address(this),
                    amountIn: bal,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: sellLimit
                })
            );
        }
    }

    // ──────────────────────────────────────────────
    // Compute ψ via closed-form
    // ──────────────────────────────────────────────

    /// @dev ψ = (C + budget×(1-fee)) / D where
    ///      C = Σ L_i × g(s_i), D = Σ L_i × g(p_i) over underpriced outcomes.
    ///      g(x) = 2^96/x for token1 outcomes, x/2^96 for token0 outcomes.
    ///      g(x) = √(outcome_price) for both orderings.
    ///      Iteratively prunes outcomes whose profitability < π = 1/ψ² - 1.
    function _computePsi(
        uint160[] memory s,
        uint128[] memory L,
        uint160[] calldata p,
        bool[] calldata isT1,
        uint256 budget,
        uint24 fee
    ) internal pure returns (PsiResult memory result) {
        (uint256[] memory order, uint256 count) = _sortedUnderpricedByPriority(s, L, p, isT1);
        (result,) = _computePsiFromSorted(order, count, s, L, p, isT1, budget, fee);
    }

    function _buildConstantLBuyPlan(
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        uint160[] calldata sqrtPredX96,
        bool[] calldata isToken1,
        uint256 budget,
        uint24 fee
    ) internal pure returns (uint256[] memory order, uint160[] memory limits, uint256 count) {
        (order, count) = _sortedUnderpricedByPriority(sqrtPrices, liquidities, sqrtPredX96, isToken1);
        (PsiResult memory psi, uint256 activeCount) =
            _computePsiFromSorted(order, count, sqrtPrices, liquidities, sqrtPredX96, isToken1, budget, fee);

        limits = new uint160[](sqrtPrices.length);
        count = activeCount;
        for (uint256 k = 0; k < count;) {
            uint256 i = order[k];
            limits[i] = _buyLimit(sqrtPrices[i], sqrtPredX96[i], isToken1[i], psi);

            unchecked {
                ++k;
            }
        }
    }

    function _sortedUnderpricedByPriority(
        uint160[] memory s,
        uint128[] memory L,
        uint160[] calldata p,
        bool[] calldata isT1
    ) internal pure returns (uint256[] memory order, uint256 count) {
        uint256 n = s.length;
        order = new uint256[](n);

        for (uint256 i = 0; i < n; i++) {
            if (L[i] == 0) continue;

            bool underpriced = isT1[i] ? s[i] > p[i] : s[i] < p[i];
            if (!underpriced) continue;

            uint256 insertAt = count;
            while (insertAt > 0 && _hasHigherBuyPriority(s, p, isT1, i, order[insertAt - 1])) {
                order[insertAt] = order[insertAt - 1];
                insertAt--;
            }
            order[insertAt] = i;
            count++;
        }
    }

    function _computePsiFromSorted(
        uint256[] memory order,
        uint256 count,
        uint160[] memory s,
        uint128[] memory L,
        uint160[] calldata p,
        bool[] calldata isT1,
        uint256 budget,
        uint24 fee
    ) internal pure returns (PsiResult memory result, uint256 activeCount) {
        if (count == 0) {
            return (PsiResult({num: 1, den: 1, buyAll: false}), 0);
        }

        uint256 budgetAdj = budget * (FEE_UNITS - uint256(fee)) / FEE_UNITS;
        uint256 C = 0;
        uint256 D = 0;

        for (uint256 k = 0; k < count;) {
            uint256 i = order[k];
            (uint256 cContribution, uint256 dContribution) = _coefficientPair(s[i], L[i], p[i], isT1[i]);
            uint256 nextC = C + cContribution;
            uint256 nextD = D + dContribution;
            uint256 nextNum = nextC + budgetAdj;

            if (nextNum < nextD) {
                bool qualifies = isT1[i]
                    ? _mulGte(uint256(s[i]), nextNum, uint256(p[i]), nextD)
                    : _mulGte(uint256(p[i]), nextNum, uint256(s[i]), nextD);
                if (!qualifies) break;
            }

            C = nextC;
            D = nextD;
            activeCount = k + 1;

            unchecked {
                ++k;
            }
        }

        if (activeCount == 0) {
            return (PsiResult({num: 1, den: 1, buyAll: false}), 0);
        }

        uint256 num = C + budgetAdj;
        if (activeCount == count && num >= D) {
            return (PsiResult({num: 1, den: 1, buyAll: true}), activeCount);
        }

        result = PsiResult({num: num, den: D, buyAll: false});
    }

    function _coefficientPair(uint160 sqrtPrice, uint128 liquidity, uint160 sqrtPred, bool isT1)
        internal
        pure
        returns (uint256 cContribution, uint256 dContribution)
    {
        if (isT1) {
            cContribution = FullMath.mulDiv(uint256(liquidity), Q96, uint256(sqrtPrice));
            dContribution = FullMath.mulDiv(uint256(liquidity), Q96, uint256(sqrtPred));
        } else {
            cContribution = FullMath.mulDiv(uint256(liquidity), uint256(sqrtPrice), Q96);
            dContribution = FullMath.mulDiv(uint256(liquidity), uint256(sqrtPred), Q96);
        }
    }

    // ──────────────────────────────────────────────
    // Phase 4: Execute waterfall buys
    // ──────────────────────────────────────────────

    function _executeBuyPlan(
        RebalanceParams calldata p,
        uint256[] memory order,
        uint160[] memory limits,
        uint256 count,
        uint256 budgetBefore
    ) internal returns (uint256 totalSpent) {
        _safeApprove(p.collateral, address(router), budgetBefore);
        IERC20 collateralToken = IERC20(p.collateral);

        for (uint256 k = 0; k < count;) {
            uint256 remaining = collateralToken.balanceOf(address(this));
            if (remaining == 0) break;

            uint256 i = order[k];
            router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: p.collateral,
                    tokenOut: p.tokens[i],
                    fee: p.fee,
                    recipient: address(this),
                    amountIn: remaining,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: limits[i]
                })
            );

            unchecked {
                ++k;
            }
        }

        totalSpent = budgetBefore - collateralToken.balanceOf(address(this));
    }

    function _waterfallBuy(RebalanceParams calldata p, uint160[] memory sqrtPrices, PsiResult memory psi)
        internal
        returns (uint256 totalSpent)
    {
        uint256 budgetBefore = IERC20(p.collateral).balanceOf(address(this));
        if (budgetBefore == 0) return 0;
        if (psi.buyAll) {
            return _executeBuyAllToPrediction(p, sqrtPrices, budgetBefore);
        }

        (uint256[] memory order, uint160[] memory limits, uint256 count) =
            _buildBuyPlan(sqrtPrices, p.sqrtPredX96, p.isToken1, psi);
        if (count == 0) return 0;

        totalSpent = _executeBuyPlan(p, order, limits, count, budgetBefore);
    }

    function _waterfallBuyOrdered(
        RebalanceParams calldata p,
        uint160[] memory sqrtPrices,
        uint256[] memory order,
        uint256 count,
        PsiResult memory psi
    ) internal returns (uint256 totalSpent) {
        uint256 budgetBefore = IERC20(p.collateral).balanceOf(address(this));
        if (budgetBefore == 0 || count == 0) return 0;
        if (psi.buyAll) {
            return _executeBuyAllToPrediction(p, sqrtPrices, budgetBefore);
        }

        uint160[] memory limits = new uint160[](sqrtPrices.length);
        uint256 activeCount = 0;
        for (uint256 k = 0; k < count;) {
            uint256 i = order[k];
            uint160 limit = _buyLimit(sqrtPrices[i], p.sqrtPredX96[i], p.isToken1[i], psi);
            if (limit == 0) break;
            limits[i] = limit;
            activeCount++;

            unchecked {
                ++k;
            }
        }

        if (activeCount == 0) return 0;
        totalSpent = _executeBuyPlan(p, order, limits, activeCount, budgetBefore);
    }

    function _executeBuyAllToPrediction(RebalanceParams calldata p, uint160[] memory sqrtPrices, uint256 budgetBefore)
        internal
        returns (uint256 totalSpent)
    {
        _safeApprove(p.collateral, address(router), budgetBefore);
        IERC20 collateralToken = IERC20(p.collateral);
        uint256 n = p.tokens.length;

        for (uint256 i = 0; i < n; i++) {
            uint160 limit = p.isToken1[i]
                ? (sqrtPrices[i] > p.sqrtPredX96[i] ? p.sqrtPredX96[i] : 0)
                : (sqrtPrices[i] < p.sqrtPredX96[i] ? p.sqrtPredX96[i] : 0);
            if (limit == 0) continue;

            uint256 remaining = collateralToken.balanceOf(address(this));
            if (remaining == 0) break;

            router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: p.collateral,
                    tokenOut: p.tokens[i],
                    fee: p.fee,
                    recipient: address(this),
                    amountIn: remaining,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: limit
                })
            );
        }

        totalSpent = budgetBefore - collateralToken.balanceOf(address(this));
    }

    function _buildBuyPlan(
        uint160[] memory sqrtPrices,
        uint160[] calldata sqrtPredX96,
        bool[] calldata isToken1,
        PsiResult memory psi
    ) internal pure returns (uint256[] memory order, uint160[] memory limits, uint256 count) {
        uint256 n = sqrtPrices.length;
        order = new uint256[](n);
        limits = new uint160[](n);

        for (uint256 i = 0; i < n; i++) {
            uint160 limit = _buyLimit(sqrtPrices[i], sqrtPredX96[i], isToken1[i], psi);
            if (limit == 0) continue;

            limits[i] = limit;
            uint256 insertAt = count;
            while (insertAt > 0 && _hasHigherBuyPriority(sqrtPrices, sqrtPredX96, isToken1, i, order[insertAt - 1])) {
                order[insertAt] = order[insertAt - 1];
                insertAt--;
            }
            order[insertAt] = i;
            count++;
        }
    }

    function _hasHigherBuyPriority(
        uint160[] memory sqrtPrices,
        uint160[] calldata sqrtPredX96,
        bool[] calldata isToken1,
        uint256 lhs,
        uint256 rhs
    ) internal pure returns (bool) {
        uint256 lhsNum = isToken1[lhs] ? uint256(sqrtPrices[lhs]) : uint256(sqrtPredX96[lhs]);
        uint256 lhsDen = isToken1[lhs] ? uint256(sqrtPredX96[lhs]) : uint256(sqrtPrices[lhs]);
        uint256 rhsNum = isToken1[rhs] ? uint256(sqrtPrices[rhs]) : uint256(sqrtPredX96[rhs]);
        uint256 rhsDen = isToken1[rhs] ? uint256(sqrtPredX96[rhs]) : uint256(sqrtPrices[rhs]);
        return _mulCompare(lhsNum, rhsDen, rhsNum, lhsDen) > 0;
    }

    /// @dev Returns sqrtPriceLimitX96 for buying outcome i. Returns 0 to skip.
    ///      token1 buy: zeroForOne=true, sqrtPrice decreases → limit < current
    ///      token0 buy: zeroForOne=false, sqrtPrice increases → limit > current
    function _buyLimit(uint160 sqrtPrice, uint160 sqrtPred, bool isT1, PsiResult memory psi)
        internal
        pure
        returns (uint160)
    {
        if (psi.buyAll) {
            // Buy to prediction: token1 limit = sqrtPred < sqrtPrice, token0 limit = sqrtPred > sqrtPrice
            if (isT1) return sqrtPrice > sqrtPred ? sqrtPred : 0;
            else return sqrtPrice < sqrtPred ? sqrtPred : 0;
        }

        if (isT1) {
            // target = sqrtPred × D / (C + budgetAdj) = sqrtPred × den / num
            // For underpriced token1 (s > p): target < s (limit below current) ✓
            if (!_mulGte(uint256(sqrtPrice), psi.num, uint256(sqrtPred), psi.den)) return 0;
            uint160 target = uint160(FullMath.mulDiv(uint256(sqrtPred), psi.den, psi.num));
            return target < sqrtPrice ? target : 0;
        } else {
            // target = sqrtPred × (C + budgetAdj) / D = sqrtPred × num / den
            // For underpriced token0 (s < p): target > s (limit above current) ✓
            if (!_mulGte(uint256(sqrtPred), psi.num, uint256(sqrtPrice), psi.den)) return 0;
            uint160 target = uint160(FullMath.mulDiv(uint256(sqrtPred), psi.num, psi.den));
            return target > sqrtPrice ? target : 0;
        }
    }

    // ──────────────────────────────────────────────
    // Phase 5: Complete-set arbitrage
    // ──────────────────────────────────────────────

    /// @notice Arbitrage to normalize prices toward summing to 1.
    ///         If sum > 1: mint complete set for 1 collateral, sell all outcomes for > 1. Repeat.
    ///         If sum < 1: buy all outcomes for < 1 collateral, merge for 1. Repeat.
    /// @param tokens All outcome tokens for this market (must be complete set).
    /// @param collateral The collateral token.
    /// @param market The market contract (for split/merge via CTF router).
    /// @param fee Uniswap fee tier.
    /// @param maxRounds Cap on iteration to bound gas.
    /// @return profit Net collateral gained from arbitrage.
    function arb(
        address[] calldata tokens,
        address[] calldata pools,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint24 fee,
        uint256 maxRounds
    ) external returns (uint256 profit) {
        profit = _arbLoop(tokens, pools, isToken1, collateral, market, fee, maxRounds);

        // Send all holdings to caller (collateral profit + leftover outcome tokens)
        uint256 n = tokens.length;
        for (uint256 i = 0; i < n;) {
            uint256 bal = IERC20(tokens[i]).balanceOf(address(this));
            if (bal > 0) _safeTransfer(tokens[i], msg.sender, bal);

            unchecked {
                ++i;
            }
        }
        uint256 collateralHeld = IERC20(collateral).balanceOf(address(this));
        if (collateralHeld > 0) {
            _safeTransfer(collateral, msg.sender, collateralHeld);
        }
    }

    /// @dev Core arb loop, shared by arb() and rebalanceAndArb().
    function _arbLoop(
        address[] calldata tokens,
        address[] calldata pools,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint24 fee,
        uint256 maxRounds
    ) internal returns (uint256 profit) {
        return _arbLoopWithFloor(tokens, pools, isToken1, collateral, market, fee, maxRounds, 0);
    }

    function _arbLoopWithFloor(
        address[] calldata tokens,
        address[] calldata pools,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint24 fee,
        uint256 maxRounds,
        uint256 minProfitCollateral
    ) internal returns (uint256 profit) {
        for (uint256 r = 0; r < maxRounds;) {
            (uint160[] memory sqrtPrices, uint256 priceSum) = _readArbState(pools, isToken1);
            uint256 collateralBal = IERC20(collateral).balanceOf(address(this));
            if (_arbPotentialProfit(collateralBal, priceSum, fee) < minProfitCollateral) break;

            uint256 gained = _arbRoundWithState(tokens, isToken1, collateral, market, fee, sqrtPrices, priceSum);
            if (gained == 0) break;
            profit += gained;

            unchecked {
                ++r;
            }
        }
    }

    /// @dev One round of complete-set arb using price-limit mechanism.
    ///      Reads pool prices, computes equilibrium targets, executes with sqrtPriceLimitX96.
    ///      Fee-adjusted thresholds define a dead zone: no arb when sum is within [buyThreshold, mintThreshold].
    function _arbRound(
        address[] calldata tokens,
        address[] calldata pools,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint24 fee
    ) internal returns (uint256 gained) {
        (uint160[] memory sqrtPrices, uint256 priceSum) = _readArbState(pools, isToken1);
        return _arbRoundWithState(tokens, isToken1, collateral, market, fee, sqrtPrices, priceSum);
    }

    function _readArbState(address[] calldata pools, bool[] calldata isToken1)
        internal
        view
        returns (uint160[] memory sqrtPrices, uint256 priceSum)
    {
        uint256 n = pools.length;
        sqrtPrices = new uint160[](n);

        for (uint256 i = 0; i < n;) {
            (sqrtPrices[i],,,,,,) = IUniswapV3Pool(pools[i]).slot0();
            if (isToken1[i]) {
                priceSum += FullMath.mulDiv(
                    FullMath.mulDiv(1e18, Q96, uint256(sqrtPrices[i])), Q96, uint256(sqrtPrices[i])
                );
            } else {
                priceSum += FullMath.mulDiv(
                    FullMath.mulDiv(uint256(sqrtPrices[i]), uint256(sqrtPrices[i]), Q96), 1e18, Q96
                );
            }

            unchecked {
                ++i;
            }
        }
    }

    function _arbRoundWithState(
        address[] calldata tokens,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint24 fee,
        uint160[] memory sqrtPrices,
        uint256 priceSum
    ) internal returns (uint256 gained) {
        // Fee-adjusted thresholds: arb only profitable beyond fee drag
        // mintThreshold = 1e18 * 1e6 / (1e6 - fee)  [sum > this → mint-sell]
        // buyThreshold  = 1e18 * (1e6 - fee) / 1e6   [sum < this → buy-merge]
        uint256 feeComp = 1e6 - uint256(fee);
        uint256 mintThreshold = FullMath.mulDiv(1e18, 1e6, feeComp);
        uint256 buyThreshold = FullMath.mulDiv(1e18, feeComp, 1e6);

        if (priceSum > mintThreshold) {
            gained = _tryMintSell(tokens, isToken1, collateral, market, fee, sqrtPrices, priceSum);
        } else if (priceSum < buyThreshold) {
            gained = _tryBuyMerge(tokens, isToken1, collateral, market, fee, sqrtPrices, priceSum);
        }
    }

    function _arbPotentialProfit(uint256 collateralBalance, uint256 priceSum, uint24 fee)
        internal
        pure
        returns (uint256 potential)
    {
        if (collateralBalance == 0) return 0;

        uint256 feeComp = FEE_UNITS - uint256(fee);
        if (feeComp == 0) return 0;

        uint256 mintThreshold = FullMath.mulDiv(1e18, FEE_UNITS, feeComp);
        uint256 buyThreshold = FullMath.mulDiv(1e18, feeComp, FEE_UNITS);

        if (priceSum > mintThreshold) {
            uint256 effectivePriceSum = FullMath.mulDiv(priceSum, feeComp, FEE_UNITS);
            return FullMath.mulDiv(collateralBalance, effectivePriceSum - 1e18, 1e18);
        }

        if (priceSum < buyThreshold) {
            uint256 grossCost = FullMath.mulDiv(priceSum, FEE_UNITS, feeComp);
            return FullMath.mulDiv(collateralBalance, 1e18 - grossCost, 1e18);
        }

        return 0;
    }

    /// @dev Mint-sell arb with price limits.
    ///      Mint complete set, sell each outcome token down to equilibrium target price.
    ///      At equilibrium: Σ price_i × (1-fee) = 1, so each price scales by K = 1e6×1e18/(feeComp×priceSum).
    ///      Unsold tokens (complete sets) are merged back into collateral.
    function _tryMintSell(
        address[] calldata tokens,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint24 fee,
        uint160[] memory sqrtPrices,
        uint256 priceSum
    ) internal returns (uint256 gained) {
        uint256 collateralBal = IERC20(collateral).balanceOf(address(this));
        if (collateralBal == 0) return 0;
        uint256 n = tokens.length;

        // Mint: collateral → all outcome tokens
        _safeApprove(collateral, address(ctfRouter), collateralBal);
        ctfRouter.splitPosition(collateral, market, collateralBal);

        // Sell each outcome token with price limit targeting equilibrium.
        // K = 1e6×1e18 / (feeComp×priceSum) < 1 (prices scale down)
        // token1 sell (zeroForOne=false): sqrtPrice increases. target = s × √(feeComp×priceSum) / √(1e6×1e18)
        // token0 sell (zeroForOne=true):  sqrtPrice decreases. target = s × √(1e6×1e18) / √(feeComp×priceSum)
        uint256 feeComp = 1e6 - uint256(fee);
        uint256 sqrtA = _sqrt(feeComp * priceSum); // > sqrtB since feeComp*priceSum > 1e6*1e18
        uint256 sqrtB = _sqrt(1e6 * 1e18);

        for (uint256 i = 0; i < n; i++) {
            uint256 tokenBal = IERC20(tokens[i]).balanceOf(address(this));
            if (tokenBal == 0) continue;

            uint160 limit;
            if (isToken1[i]) {
                // Selling token1: zeroForOne=false, sqrtPrice goes UP, limit > current
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtA, sqrtB));
                if (limit <= sqrtPrices[i]) continue;
            } else {
                // Selling token0: zeroForOne=true, sqrtPrice goes DOWN, limit < current
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtB, sqrtA));
                if (limit >= sqrtPrices[i]) continue;
            }

            _safeApprove(tokens[i], address(router), tokenBal);
            router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: tokens[i],
                    tokenOut: collateral,
                    fee: fee,
                    recipient: address(this),
                    amountIn: tokenBal,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: limit
                })
            );
        }

        // Merge back unsold complete sets: find min outcome balance, merge that amount
        uint256 mergeAmount = type(uint256).max;
        for (uint256 i = 0; i < n;) {
            uint256 bal = IERC20(tokens[i]).balanceOf(address(this));
            if (bal < mergeAmount) mergeAmount = bal;

            unchecked {
                ++i;
            }
        }
        if (mergeAmount > 0) {
            for (uint256 i = 0; i < n;) {
                _safeApprove(tokens[i], address(ctfRouter), mergeAmount);

                unchecked {
                    ++i;
                }
            }
            ctfRouter.mergePositions(collateral, market, mergeAmount);
        }

        // Profit = final collateral - initial collateral
        uint256 collateralAfter = IERC20(collateral).balanceOf(address(this));
        if (collateralAfter > collateralBal) {
            gained = collateralAfter - collateralBal;
        }
    }

    /// @dev Buy-merge arb with price limits (waterfall pattern).
    ///      Buy each outcome with full remaining budget + price limit, merge minimum balance.
    ///      At equilibrium: Σ price_i / (1-fee) = 1, so each price scales by K = feeComp×1e18/(1e6×priceSum).
    function _tryBuyMerge(
        address[] calldata tokens,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint24 fee,
        uint160[] memory sqrtPrices,
        uint256 priceSum
    ) internal returns (uint256 gained) {
        uint256 collateralBal = IERC20(collateral).balanceOf(address(this));
        if (collateralBal == 0) return 0;
        uint256 n = tokens.length;

        // K = feeComp×1e18 / (1e6×priceSum) > 1 (prices scale up)
        // token1 buy (zeroForOne=true):  sqrtPrice decreases. target = s × √(1e6×priceSum) / √(feeComp×1e18)
        // token0 buy (zeroForOne=false): sqrtPrice increases. target = s × √(feeComp×1e18) / √(1e6×priceSum)
        uint256 feeComp = 1e6 - uint256(fee);
        uint256 sqrtC = _sqrt(1e6 * priceSum); // < sqrtD since 1e6*priceSum < feeComp*1e18
        uint256 sqrtD = _sqrt(feeComp * 1e18);

        _safeApprove(collateral, address(router), collateralBal);

        for (uint256 i = 0; i < n; i++) {
            uint256 remaining = IERC20(collateral).balanceOf(address(this));
            if (remaining == 0) break;

            uint160 limit;
            if (isToken1[i]) {
                // Buying token1: zeroForOne=true, sqrtPrice goes DOWN, limit < current
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtC, sqrtD));
                if (limit >= sqrtPrices[i]) continue;
            } else {
                // Buying token0: zeroForOne=false, sqrtPrice goes UP, limit > current
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtD, sqrtC));
                if (limit <= sqrtPrices[i]) continue;
            }

            router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: collateral,
                    tokenOut: tokens[i],
                    fee: fee,
                    recipient: address(this),
                    amountIn: remaining,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: limit
                })
            );
        }

        // Merge minimum balance into collateral
        uint256 mergeAmount = type(uint256).max;
        for (uint256 i = 0; i < n;) {
            uint256 bal = IERC20(tokens[i]).balanceOf(address(this));
            if (bal < mergeAmount) mergeAmount = bal;

            unchecked {
                ++i;
            }
        }

        if (mergeAmount > 0) {
            for (uint256 i = 0; i < n;) {
                _safeApprove(tokens[i], address(ctfRouter), mergeAmount);

                unchecked {
                    ++i;
                }
            }
            ctfRouter.mergePositions(collateral, market, mergeAmount);
        }

        // Profit = final collateral - initial collateral
        uint256 collateralAfter = IERC20(collateral).balanceOf(address(this));
        if (collateralAfter > collateralBal) {
            gained = collateralAfter - collateralBal;
        }
    }

    // ──────────────────────────────────────────────
    // Reverse waterfall: sell below-frontier holdings to frontier price
    // ──────────────────────────────────────────────

    /// @dev Reverse waterfall: sell held tokens below the profitability frontier,
    ///      each with a price limit at the frontier price (same target as _buyLimit).
    ///      Selling pushes profitability UP toward frontier — self-regulates like the
    ///      forward waterfall. Least profitable tokens sell the most, naturally ratcheting
    ///      up to meet higher-profitability holdings. No explicit sorting needed.
    ///      Recovered collateral is redeployed via forward waterfall.
    /// @param maxRounds Bounds iterations (reverse sell + forward waterfall = 1 round).
    function _recycleSell(RebalanceParams calldata p, uint256 maxRounds)
        internal
        returns (uint256 totalRecycled, uint256 totalRedeployed)
    {
        return _recycleSellWithFloor(p, maxRounds, 0);
    }

    function _recycleSellWithFloor(RebalanceParams calldata p, uint256 maxRounds, uint256 minRecycleProfitCollateral)
        internal
        returns (uint256 totalRecycled, uint256 totalRedeployed)
    {
        for (uint256 r = 0; r < maxRounds;) {
            uint256 n = p.tokens.length;
            (uint160[] memory sqrtPrices, uint128[] memory liquidities) = _readPoolState(p);

            uint256 budget = IERC20(p.collateral).balanceOf(address(this));
            PsiResult memory psi = _computePsi(sqrtPrices, liquidities, p.sqrtPredX96, p.isToken1, budget, p.fee);

            // If buyAll, every underpriced outcome is on the frontier — nothing to recycle
            if (psi.buyAll) break;
            if (psi.num == 0 && psi.den == 0) break; // no underpriced outcomes
            if (_recyclePotentialGain(p, sqrtPrices, psi) < minRecycleProfitCollateral) break;

            // Reverse waterfall: sell below-frontier holdings with frontier price limits.
            // Frontier target is the same price as the forward waterfall's buy target:
            //   token1: target = p × den/num  (above current s, selling pushes s UP → ✓)
            //   token0: target = p × num/den  (below current s, selling pushes s DOWN → ✓)
            uint256 sold = 0;
            for (uint256 i = 0; i < n; i++) {
                uint256 bal = IERC20(p.tokens[i]).balanceOf(address(this));
                if (bal == 0) continue;

                uint160 limit;
                if (p.isToken1[i]) {
                    // Below frontier: s × num < p × den
                    if (_mulGte(uint256(sqrtPrices[i]), psi.num, uint256(p.sqrtPredX96[i]), psi.den)) continue;
                    if (!_isRecycleWorthwhile(sqrtPrices[i], p.sqrtPredX96[i], true, psi, p.fee)) continue;
                    // Sell limit = p × den/num (above current, zeroForOne=false ✓)
                    limit = uint160(FullMath.mulDiv(uint256(p.sqrtPredX96[i]), psi.den, psi.num));
                } else {
                    // Below frontier: p × num < s × den
                    if (_mulGte(uint256(p.sqrtPredX96[i]), psi.num, uint256(sqrtPrices[i]), psi.den)) continue;
                    if (!_isRecycleWorthwhile(sqrtPrices[i], p.sqrtPredX96[i], false, psi, p.fee)) continue;
                    // Sell limit = p × num/den (below current, zeroForOne=true ✓)
                    limit = uint160(FullMath.mulDiv(uint256(p.sqrtPredX96[i]), psi.num, psi.den));
                }

                _safeApprove(p.tokens[i], address(router), bal);
                sold += router.exactInputSingle(
                    IV3SwapRouter.ExactInputSingleParams({
                        tokenIn: p.tokens[i],
                        tokenOut: p.collateral,
                        fee: p.fee,
                        recipient: address(this),
                        amountIn: bal,
                        amountOutMinimum: 0,
                        sqrtPriceLimitX96: limit
                    })
                );
            }

            if (sold == 0) break;
            totalRecycled += sold;

            // Forward waterfall: deploy recovered collateral into above-frontier outcomes.
            totalRedeployed += _readAndBuy(p);

            unchecked {
                ++r;
            }
        }
    }

    function _recycleSellExact(
        RebalanceParams calldata p,
        uint256 maxRounds,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool
    ) internal returns (uint256 totalRecycled, uint256 totalRedeployed) {
        return _recycleSellExactWithFloor(p, maxRounds, maxBisectionIterations, maxTickCrossingsPerPool, 0);
    }

    function _recycleSellExactWithFloor(
        RebalanceParams calldata p,
        uint256 maxRounds,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool,
        uint256 minRecycleProfitCollateral
    ) internal returns (uint256 totalRecycled, uint256 totalRedeployed) {
        for (uint256 r = 0; r < maxRounds;) {
            uint256 n = p.tokens.length;
            (uint160[] memory sqrtPrices, int24[] memory ticks, uint128[] memory liquidities) = _readExactPoolState(p);
            uint256 budget = IERC20(p.collateral).balanceOf(address(this));
            (ExactCostCurves memory curves, uint256 buyAllCost) =
                _buildExactCostCurves(p, sqrtPrices, ticks, liquidities, maxTickCrossingsPerPool, budget);
            PsiResult memory psi = _computePsiExact(p, sqrtPrices, curves, buyAllCost, budget, maxBisectionIterations);

            if (psi.buyAll) break;
            if (psi.num == 0 && psi.den == 0) break;
            if (_recyclePotentialGain(p, sqrtPrices, psi) < minRecycleProfitCollateral) break;

            uint256 sold = 0;
            for (uint256 i = 0; i < n; i++) {
                uint256 bal = IERC20(p.tokens[i]).balanceOf(address(this));
                if (bal == 0) continue;

                uint160 limit;
                if (p.isToken1[i]) {
                    if (_mulGte(uint256(sqrtPrices[i]), psi.num, uint256(p.sqrtPredX96[i]), psi.den)) continue;
                    if (!_isRecycleWorthwhile(sqrtPrices[i], p.sqrtPredX96[i], true, psi, p.fee)) continue;
                    limit = uint160(FullMath.mulDiv(uint256(p.sqrtPredX96[i]), psi.den, psi.num));
                } else {
                    if (_mulGte(uint256(p.sqrtPredX96[i]), psi.num, uint256(sqrtPrices[i]), psi.den)) continue;
                    if (!_isRecycleWorthwhile(sqrtPrices[i], p.sqrtPredX96[i], false, psi, p.fee)) continue;
                    limit = uint160(FullMath.mulDiv(uint256(p.sqrtPredX96[i]), psi.num, psi.den));
                }

                _safeApprove(p.tokens[i], address(router), bal);
                sold += router.exactInputSingle(
                    IV3SwapRouter.ExactInputSingleParams({
                        tokenIn: p.tokens[i],
                        tokenOut: p.collateral,
                        fee: p.fee,
                        recipient: address(this),
                        amountIn: bal,
                        amountOutMinimum: 0,
                        sqrtPriceLimitX96: limit
                    })
                );
            }

            if (sold == 0) break;
            totalRecycled += sold;
            totalRedeployed += _readAndBuyExact(p, maxBisectionIterations, maxTickCrossingsPerPool);

            unchecked {
                ++r;
            }
        }
    }

    function _recyclePotentialGain(RebalanceParams calldata p, uint160[] memory sqrtPrices, PsiResult memory psi)
        internal
        view
        returns (uint256 potentialGain)
    {
        uint256 n = p.tokens.length;
        for (uint256 i = 0; i < n;) {
            uint256 bal = IERC20(p.tokens[i]).balanceOf(address(this));
            if (bal == 0) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint160 frontier = _buyLimit(sqrtPrices[i], p.sqrtPredX96[i], p.isToken1[i], psi);
            if (frontier == 0) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint256 currentPrice = _priceFromSqrt(sqrtPrices[i], p.isToken1[i]);
            uint256 frontierPrice = _priceFromSqrt(frontier, p.isToken1[i]);
            if (currentPrice > frontierPrice) {
                potentialGain += FullMath.mulDiv(bal, currentPrice - frontierPrice, 1e18);
            }

            unchecked {
                ++i;
            }
        }
    }

    // ──────────────────────────────────────────────
    // Return all holdings to caller
    // ──────────────────────────────────────────────

    /// @dev Send all outcome token balances and remaining collateral to msg.sender.
    function _returnAll(RebalanceParams calldata p) internal {
        uint256 n = p.tokens.length;
        for (uint256 i = 0; i < n;) {
            uint256 bal = IERC20(p.tokens[i]).balanceOf(address(this));
            if (bal > 0) {
                _safeTransfer(p.tokens[i], msg.sender, bal);
            }

            unchecked {
                ++i;
            }
        }
        uint256 collateralBal = IERC20(p.collateral).balanceOf(address(this));
        if (collateralBal > 0) {
            _safeTransfer(p.collateral, msg.sender, collateralBal);
        }
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    /// @dev Babylonian integer square root. Returns floor(sqrt(x)).
    function _sqrt(uint256 x) internal pure returns (uint256 z) {
        if (x == 0) return 0;
        z = x;
        uint256 y = (z + 1) / 2;
        while (y < z) {
            z = y;
            y = (x / z + z) / 2;
        }
    }

    function _priceFromSqrt(uint160 sqrtPrice, bool isToken1) internal pure returns (uint256) {
        if (isToken1) {
            return FullMath.mulDiv(FullMath.mulDiv(1e18, Q96, uint256(sqrtPrice)), Q96, uint256(sqrtPrice));
        }

        return FullMath.mulDiv(FullMath.mulDiv(uint256(sqrtPrice), uint256(sqrtPrice), Q96), 1e18, Q96);
    }

    /// @dev Exact-input sells should stop at the fee-neutral fair-value boundary, not raw prediction.
    function _sellLimit(uint160 sqrtPred, bool isT1, uint24 fee) internal pure returns (uint160) {
        uint256 feeComp = FEE_UNITS - uint256(fee);
        uint256 sqrtFeeComp = _sqrt(feeComp * FEE_UNITS);
        if (sqrtFeeComp == 0) return sqrtPred;

        if (isT1) {
            return uint160(FullMath.mulDiv(uint256(sqrtPred), sqrtFeeComp, FEE_UNITS));
        }

        return uint160(FullMath.mulDiv(uint256(sqrtPred), FEE_UNITS, sqrtFeeComp));
    }

    /// @dev Require the frontier EV to beat the current holding by at least round-trip fee drag.
    ///      The comparison is done in sqrt-profitability space, so a single fee factor here
    ///      corresponds to two swap fees in price space.
    function _isRecycleWorthwhile(uint160 sqrtPrice, uint160 sqrtPred, bool isT1, PsiResult memory psi, uint24 fee)
        internal
        pure
        returns (bool)
    {
        uint256 feeComp = FEE_UNITS - uint256(fee);
        if (feeComp == 0 || psi.num == 0) return false;

        uint256 currentNum = (isT1 ? uint256(sqrtPrice) : uint256(sqrtPred)) * FEE_UNITS;
        uint256 currentDen = (isT1 ? uint256(sqrtPred) : uint256(sqrtPrice)) * feeComp;

        return _mulCompare(psi.den, currentDen, currentNum, psi.num) > 0;
    }

    function _readPoolState(RebalanceParams calldata params)
        internal
        view
        returns (uint160[] memory sqrtPrices, uint128[] memory liquidities)
    {
        uint256 n = params.tokens.length;
        sqrtPrices = new uint160[](n);
        liquidities = new uint128[](n);

        for (uint256 i = 0; i < n;) {
            (sqrtPrices[i],,,,,,) = IUniswapV3Pool(params.pools[i]).slot0();
            liquidities[i] = IUniswapV3Pool(params.pools[i]).liquidity();

            unchecked {
                ++i;
            }
        }
    }

    function _readExactPoolState(RebalanceParams calldata params)
        internal
        view
        returns (uint160[] memory sqrtPrices, int24[] memory ticks, uint128[] memory liquidities)
    {
        uint256 n = params.tokens.length;
        sqrtPrices = new uint160[](n);
        ticks = new int24[](n);
        liquidities = new uint128[](n);

        for (uint256 i = 0; i < n;) {
            (sqrtPrices[i], ticks[i],,,,,) = IUniswapV3Pool(params.pools[i]).slot0();
            liquidities[i] = IUniswapV3Pool(params.pools[i]).liquidity();

            unchecked {
                ++i;
            }
        }
    }

    function _floorToSpacing(int24 tick, int24 spacing) internal pure returns (int24) {
        int24 compressed = tick / spacing;
        if (tick < 0 && tick % spacing != 0) {
            compressed--;
        }
        return compressed * spacing;
    }

    function _safeTransfer(address token, address to, uint256 amount) internal {
        bool success = IERC20(token).transfer(to, amount);
        if (!success) revert TransferFailed();
    }

    function _safeTransferFrom(address token, address from, address to, uint256 amount) internal {
        bool success = IERC20(token).transferFrom(from, to, amount);
        if (!success) revert TransferFailed();
    }

    function _safeApprove(address token, address spender, uint256 amount) internal {
        if (amount == 0) return;
        if (IERC20(token).allowance(address(this), spender) >= amount) return;

        bool success = IERC20(token).approve(spender, type(uint256).max);
        if (!success) revert ApprovalFailed();
    }

    function _mulGte(uint256 a, uint256 b, uint256 c, uint256 d) internal pure returns (bool) {
        return _mulCompare(a, b, c, d) >= 0;
    }

    function _mulCompare(uint256 a, uint256 b, uint256 c, uint256 d) internal pure returns (int8) {
        (uint256 lhsHi, uint256 lhsLo) = _fullMul(a, b);
        (uint256 rhsHi, uint256 rhsLo) = _fullMul(c, d);

        if (lhsHi > rhsHi) return 1;
        if (lhsHi < rhsHi) return -1;
        if (lhsLo > rhsLo) return 1;
        if (lhsLo < rhsLo) return -1;
        return 0;
    }

    function _fullMul(uint256 x, uint256 y) internal pure returns (uint256 hi, uint256 lo) {
        assembly {
            let mm := mulmod(x, y, not(0))
            lo := mul(x, y)
            hi := sub(sub(mm, lo), lt(mm, lo))
        }
    }
}
