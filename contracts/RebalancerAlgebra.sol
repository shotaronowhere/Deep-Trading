// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IAlgebraSwapRouter} from "./interfaces/IAlgebraSwapRouter.sol";
import {IERC20} from "./interfaces/IERC20.sol";
import {ICTFRouter} from "./interfaces/ICTFRouter.sol";
import {IAlgebraPool} from "./interfaces/IAlgebraPool.sol";
import {FullMath} from "./libraries/FullMath.sol";
import {TickMath} from "./libraries/TickMath.sol";

/// @title On-chain portfolio rebalancer for AlgebraV1.9 (Swapr) DEX
/// @notice Fork of Rebalancer.sol adapted for AlgebraV1.9 pool interface and dynamic fees.
/// @dev Key differences from Uniswap V3 version:
///      - globalState() instead of slot0(), fee is per-pool and dynamic
///      - tickTable() instead of tickBitmap(), raw tick decomposition (no spacing compression)
///      - Router: no fee param, adds deadline, limitSqrtPrice instead of sqrtPriceLimitX96
contract RebalancerAlgebra {
    error TransferFailed();
    error ApprovalFailed();
    error TickScanLimitExceeded();

    IAlgebraSwapRouter public immutable router;
    ICTFRouter public immutable ctfRouter;
    uint256 constant Q96 = 1 << 96;
    uint256 constant FEE_UNITS = 1e6;
    uint256 constant MAX_WATERFALL_PASSES = 6;
    uint256 constant PSI_WAD = 1e18;

    struct RebalanceParams {
        address[] tokens;
        address[] pools;
        bool[] isToken1;
        uint256[] balances;
        uint256 collateralAmount;
        uint160[] sqrtPredX96;
        address collateral;
    }

    struct PsiResult {
        uint256 num;
        uint256 den;
        bool buyAll;
    }

    struct ExactCostCurves {
        uint256[] offsets;
        uint256[] counts;
        uint256[] totalCosts;
        uint16[] fees;
        uint160[] segmentEnds;
        uint128[] segmentLiquidities;
        uint256[] segmentPrefixCosts;
    }

    constructor(address _router, address _ctfRouter) {
        router = IAlgebraSwapRouter(_router);
        ctfRouter = ICTFRouter(_ctfRouter);
    }

    function rebalance(RebalanceParams calldata params) external returns (uint256 totalProceeds, uint256 totalSpent) {
        totalProceeds = _pullAndSell(params);
        totalSpent = _readAndBuy(params);
        _returnAll(params);
    }

    function rebalanceExact(
        RebalanceParams calldata params,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool
    ) external returns (uint256 totalProceeds, uint256 totalSpent) {
        totalProceeds = _pullAndSell(params);
        totalSpent = _readAndBuyExact(params, maxBisectionIterations, maxTickCrossingsPerPool);
        _returnAll(params);
    }

    function rebalanceAndArb(
        RebalanceParams calldata params,
        address market,
        uint256 maxArbRounds,
        uint256 maxRecycleRounds
    ) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit) {
        totalProceeds = _pullAndSell(params);
        arbProfit = _arbLoop(params.tokens, params.pools, params.isToken1, params.collateral, market, maxArbRounds);
        totalSpent = _readAndBuy(params);

        (uint256 recycleProceeds, uint256 recycleSpent) = _recycleSell(params, maxRecycleRounds);
        totalProceeds += recycleProceeds;
        totalSpent += recycleSpent;

        _returnAll(params);
    }

    function rebalanceAndArbExact(
        RebalanceParams calldata params,
        address market,
        uint256 maxArbRounds,
        uint256 maxRecycleRounds,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool
    ) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit) {
        totalProceeds = _pullAndSell(params);
        arbProfit = _arbLoop(params.tokens, params.pools, params.isToken1, params.collateral, market, maxArbRounds);
        totalSpent = _readAndBuyExact(params, maxBisectionIterations, maxTickCrossingsPerPool);

        (uint256 recycleProceeds, uint256 recycleSpent) =
            _recycleSellExact(params, maxRecycleRounds, maxBisectionIterations, maxTickCrossingsPerPool);
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

    function _readAndBuyConstantL(RebalanceParams calldata params) internal returns (uint256) {
        uint256 totalSpent = 0;

        for (uint256 pass = 0; pass < MAX_WATERFALL_PASSES;) {
            (uint256 spent, uint256 budgetBefore) = _waterfallPass(params);
            if (budgetBefore == 0 || spent == 0) break;
            totalSpent += spent;
            if (spent == budgetBefore) break;

            unchecked {
                ++pass;
            }
        }

        return totalSpent;
    }

    function _readAndBuyExact(
        RebalanceParams calldata params,
        uint256 maxBisectionIterations,
        uint256 maxTickCrossingsPerPool
    ) internal returns (uint256) {
        uint256 budget = IERC20(params.collateral).balanceOf(address(this));
        if (budget == 0) return 0;

        (uint160[] memory sqrtPrices, int24[] memory ticks, uint128[] memory liquidities, uint16[] memory fees) =
            _readExactPoolState(params);
        (ExactCostCurves memory curves, uint256 buyAllCost) =
            _buildExactCostCurves(params, sqrtPrices, ticks, liquidities, fees, maxTickCrossingsPerPool);

        PsiResult memory psi = _computePsiExact(params, sqrtPrices, curves, buyAllCost, budget, maxBisectionIterations);

        return _waterfallBuy(params, sqrtPrices, psi);
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
        uint16[] memory fees,
        uint256 maxTickCrossingsPerPool
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
            fees: fees,
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
            IAlgebraPool pool = IAlgebraPool(params.pools[i]);
            if (params.isToken1[i]) {
                (cursor, curves.totalCosts[i]) = _buildExactDescendingCurve(
                    curves,
                    cursor,
                    pool,
                    sqrtPrices[i],
                    ticks[i],
                    liquidities[i],
                    params.sqrtPredX96[i],
                    fees[i],
                    maxTickCrossingsPerPool
                );
            } else {
                (cursor, curves.totalCosts[i]) = _buildExactAscendingCurve(
                    curves,
                    cursor,
                    pool,
                    sqrtPrices[i],
                    ticks[i],
                    liquidities[i],
                    params.sqrtPredX96[i],
                    fees[i],
                    maxTickCrossingsPerPool
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
        uint16 fee = curves.fees[poolIndex];

        for (uint256 j = 0; j < count;) {
            uint256 idx = offset + j;
            uint160 segmentEnd = curves.segmentEnds[idx];

            if (isT1 ? limit >= segmentEnd : limit <= segmentEnd) {
                uint256 segmentCost = isT1
                    ? _segmentCostToken1(segmentStart, limit, curves.segmentLiquidities[idx], fee)
                    : _segmentCostToken0(segmentStart, limit, curves.segmentLiquidities[idx], fee);
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
        uint16 fee,
        uint256 maxTickCrossingsPerPool
    ) internal view returns (uint256 cost) {
        if (limit == sqrtPrice) return 0;

        IAlgebraPool algebraPool = IAlgebraPool(pool);
        int24 spacing = algebraPool.tickSpacing();
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

            (, int128 lowerLiquidityDelta,,,,,,) = algebraPool.ticks(lowerTick);
            liquidity = _applyLiquidityNetDescending(liquidity, lowerLiquidityDelta);
            crossings = 1;
            current = segmentFloor;

            int24 lowerBoundaryTick = lowerTick;
            while (true) {
                if (lowerBoundaryTick == minTick) {
                    cost += _segmentCostToken1(current, limit, liquidity, fee);
                    break;
                }

                (int24 nextTick, bool initialized) =
                    _nextInitializedTickBelow(algebraPool, lowerBoundaryTick, spacing, minTick);
                uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
                if (limit >= nextSqrt) {
                    cost += _segmentCostToken1(current, limit, liquidity, fee);
                    break;
                }

                cost += _segmentCostToken1(current, nextSqrt, liquidity, fee);
                crossings =
                    _checkedAddCrossings(crossings, maxTickCrossingsPerPool);

                if (initialized) {
                    (, int128 nextLiquidityDelta,,,,,,) = algebraPool.ticks(nextTick);
                    liquidity = _applyLiquidityNetDescending(liquidity, nextLiquidityDelta);
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

        (, int128 upperLiquidityDelta,,,,,,) = algebraPool.ticks(upperTick);
        liquidity = _applyLiquidityNetAscending(liquidity, upperLiquidityDelta);
        crossings = 1;
        current = segmentCeiling;

        int24 upperBoundaryTick = upperTick;
        while (true) {
            if (upperBoundaryTick == maxTick) {
                cost += _segmentCostToken0(current, limit, liquidity, fee);
                break;
            }

            (int24 nextTick, bool initialized) =
                _nextInitializedTickAbove(algebraPool, upperBoundaryTick, spacing, maxTick);
            uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
            if (limit <= nextSqrt) {
                cost += _segmentCostToken0(current, limit, liquidity, fee);
                break;
            }

            cost += _segmentCostToken0(current, nextSqrt, liquidity, fee);
            crossings = _checkedAddCrossings(crossings, maxTickCrossingsPerPool);

            if (initialized) {
                (, int128 nextLiquidityDelta,,,,,,) = algebraPool.ticks(nextTick);
                liquidity = _applyLiquidityNetAscending(liquidity, nextLiquidityDelta);
            }

            current = nextSqrt;
            upperBoundaryTick = nextTick;
        }
    }

    function _buildExactDescendingCurve(
        ExactCostCurves memory curves,
        uint256 cursor,
        IAlgebraPool pool,
        uint160 sqrtPrice,
        int24 tick,
        uint128 liquidity,
        uint160 limit,
        uint16 fee,
        uint256 maxTickCrossingsPerPool
    ) internal view returns (uint256 nextCursor, uint256 totalCost) {
        int24 spacing = pool.tickSpacing();
        int24 minTick = _minUsableTick(spacing);
        int24 lowerTick = _floorToSpacing(tick, spacing);
        if (lowerTick < minTick) lowerTick = minTick;

        uint160 current = sqrtPrice;
        uint256 crossings = 0;
        nextCursor = cursor;

        uint160 segmentFloor = TickMath.getSqrtRatioAtTick(lowerTick);
        if (limit >= segmentFloor) {
            nextCursor = _appendExactCurveSegment(curves, nextCursor, limit, liquidity, totalCost);
            totalCost += _segmentCostToken1(current, limit, liquidity, fee);
            return (nextCursor, totalCost);
        }

        nextCursor = _appendExactCurveSegment(curves, nextCursor, segmentFloor, liquidity, totalCost);
        totalCost += _segmentCostToken1(current, segmentFloor, liquidity, fee);
        if (maxTickCrossingsPerPool == 0) revert TickScanLimitExceeded();

        (, int128 lowerLiquidityDelta,,,,,,) = pool.ticks(lowerTick);
        liquidity = _applyLiquidityNetDescending(liquidity, lowerLiquidityDelta);
        crossings = 1;
        current = segmentFloor;

        int24 lowerBoundaryTick = lowerTick;
        while (true) {
            if (lowerBoundaryTick == minTick) {
                nextCursor = _appendExactCurveSegment(curves, nextCursor, limit, liquidity, totalCost);
                totalCost += _segmentCostToken1(current, limit, liquidity, fee);
                break;
            }

            (int24 nextTick, bool initialized) =
                _nextInitializedTickBelow(pool, lowerBoundaryTick, spacing, minTick);
            uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
            if (limit >= nextSqrt) {
                nextCursor = _appendExactCurveSegment(curves, nextCursor, limit, liquidity, totalCost);
                totalCost += _segmentCostToken1(current, limit, liquidity, fee);
                break;
            }

            nextCursor = _appendExactCurveSegment(curves, nextCursor, nextSqrt, liquidity, totalCost);
            totalCost += _segmentCostToken1(current, nextSqrt, liquidity, fee);
            crossings = _checkedAddCrossings(crossings, maxTickCrossingsPerPool);

            if (initialized) {
                (, int128 nextLiquidityDelta,,,,,,) = pool.ticks(nextTick);
                liquidity = _applyLiquidityNetDescending(liquidity, nextLiquidityDelta);
            }

            current = nextSqrt;
            lowerBoundaryTick = nextTick;
        }
    }

    function _buildExactAscendingCurve(
        ExactCostCurves memory curves,
        uint256 cursor,
        IAlgebraPool pool,
        uint160 sqrtPrice,
        int24 tick,
        uint128 liquidity,
        uint160 limit,
        uint16 fee,
        uint256 maxTickCrossingsPerPool
    ) internal view returns (uint256 nextCursor, uint256 totalCost) {
        int24 spacing = pool.tickSpacing();
        int24 maxTick = _maxUsableTick(spacing);
        int24 upperTick = _floorToSpacing(tick, spacing) + spacing;
        if (upperTick > maxTick) upperTick = maxTick;

        uint160 current = sqrtPrice;
        uint256 crossings = 0;
        nextCursor = cursor;

        uint160 segmentCeiling = TickMath.getSqrtRatioAtTick(upperTick);
        if (limit <= segmentCeiling) {
            nextCursor = _appendExactCurveSegment(curves, nextCursor, limit, liquidity, totalCost);
            totalCost += _segmentCostToken0(current, limit, liquidity, fee);
            return (nextCursor, totalCost);
        }

        nextCursor = _appendExactCurveSegment(curves, nextCursor, segmentCeiling, liquidity, totalCost);
        totalCost += _segmentCostToken0(current, segmentCeiling, liquidity, fee);
        if (maxTickCrossingsPerPool == 0) revert TickScanLimitExceeded();

        (, int128 upperLiquidityDelta,,,,,,) = pool.ticks(upperTick);
        liquidity = _applyLiquidityNetAscending(liquidity, upperLiquidityDelta);
        crossings = 1;
        current = segmentCeiling;

        int24 upperBoundaryTick = upperTick;
        while (true) {
            if (upperBoundaryTick == maxTick) {
                nextCursor = _appendExactCurveSegment(curves, nextCursor, limit, liquidity, totalCost);
                totalCost += _segmentCostToken0(current, limit, liquidity, fee);
                break;
            }

            (int24 nextTick, bool initialized) = _nextInitializedTickAbove(pool, upperBoundaryTick, spacing, maxTick);
            uint160 nextSqrt = TickMath.getSqrtRatioAtTick(nextTick);
            if (limit <= nextSqrt) {
                nextCursor = _appendExactCurveSegment(curves, nextCursor, limit, liquidity, totalCost);
                totalCost += _segmentCostToken0(current, limit, liquidity, fee);
                break;
            }

            nextCursor = _appendExactCurveSegment(curves, nextCursor, nextSqrt, liquidity, totalCost);
            totalCost += _segmentCostToken0(current, nextSqrt, liquidity, fee);
            crossings = _checkedAddCrossings(crossings, maxTickCrossingsPerPool);

            if (initialized) {
                (, int128 nextLiquidityDelta,,,,,,) = pool.ticks(nextTick);
                liquidity = _applyLiquidityNetAscending(liquidity, nextLiquidityDelta);
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

    // ──────────────────────────────────────────────
    // Tick scanning — AlgebraV1.9: raw tick bitmap (no spacing compression)
    // ──────────────────────────────────────────────

    /// @dev AlgebraV1.9 uses raw tick indices in tickTable (no division by spacing).
    ///      Ticks are still spacing-aligned for initialization, but bitmap lookup uses raw tick.
    function _nextInitializedTickBelow(IAlgebraPool pool, int24 currentTick, int24 /* spacing */, int24 minTick)
        internal
        view
        returns (int24 nextTick, bool initialized)
    {
        // AlgebraV1.9: pass raw tick - 1 to bitmap scan (not compressed by spacing)
        (int24 candidate, bool found) = _nextInitializedTickWithinOneWord(pool, currentTick - 1, true);
        if (candidate < minTick) {
            candidate = minTick;
            found = false;
        }

        return (candidate, found);
    }

    function _nextInitializedTickAbove(IAlgebraPool pool, int24 currentTick, int24 /* spacing */, int24 maxTick)
        internal
        view
        returns (int24 nextTick, bool initialized)
    {
        (int24 candidate, bool found) = _nextInitializedTickWithinOneWord(pool, currentTick, false);
        if (candidate > maxTick) {
            candidate = maxTick;
            found = false;
        }

        return (candidate, found);
    }

    /// @dev AlgebraV1.9 tickTable scan. Uses raw tick (not compressed by spacing).
    ///      tickTable(wordPos) where wordPos = tick >> 8, bitPos = tick & 0xFF.
    function _nextInitializedTickWithinOneWord(IAlgebraPool pool, int24 tick, bool lte)
        internal
        view
        returns (int24 nextTick, bool initialized)
    {
        if (lte) {
            (int16 lteWordPos, uint8 lteBitPos) = _tickBitmapPosition(tick);
            uint256 lteMask = ((uint256(1) << lteBitPos) - 1) | (uint256(1) << lteBitPos);
            uint256 lteMasked = pool.tickTable(lteWordPos) & lteMask;

            initialized = lteMasked != 0;
            if (initialized) {
                nextTick = tick - int24(uint24(lteBitPos - _mostSignificantBit(lteMasked)));
            } else {
                nextTick = tick - int24(uint24(lteBitPos));
            }
            return (nextTick, initialized);
        }

        (int16 gtWordPos, uint8 gtBitPos) = _tickBitmapPosition(tick + 1);
        uint256 gtMask = ~((uint256(1) << gtBitPos) - 1);
        uint256 gtMasked = pool.tickTable(gtWordPos) & gtMask;

        initialized = gtMasked != 0;
        if (initialized) {
            nextTick = (tick + 1) + int24(uint24(_leastSignificantBit(gtMasked) - gtBitPos));
        } else {
            nextTick = (tick + 1) + int24(uint24(type(uint8).max - gtBitPos));
        }
    }

    /// @dev AlgebraV1.9: raw tick decomposition (no spacing division).
    function _tickBitmapPosition(int24 tick) internal pure returns (int16 wordPos, uint8 bitPos) {
        wordPos = int16(tick >> 8);
        bitPos = uint8(uint24(tick % 256));
    }

    /// @dev AlgebraV1.9: count each bitmap row scan as 1 crossing (= 1 SLOAD).
    ///      Unlike V3's spacing-compressed bitmap, raw ticks mean one row = up to 256 ticks.
    ///      Counting per-row ensures buffer capacity (maxTickCrossingsPerPool + 1 segments)
    ///      stays synchronized with actual iteration count regardless of tickSpacing.
    function _checkedAddCrossings(
        uint256 currentCrossings,
        uint256 maxTickCrossingsPerPool
    ) internal pure returns (uint256 nextCrossings) {
        nextCrossings = currentCrossings + 1;
        if (nextCrossings > maxTickCrossingsPerPool) revert TickScanLimitExceeded();
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

    function _segmentCostToken1(uint160 start, uint160 end, uint128 liquidity, uint16 fee)
        internal
        pure
        returns (uint256)
    {
        if (end >= start) return 0;

        uint256 noFee = FullMath.mulDiv(uint256(liquidity), Q96, uint256(end))
            - FullMath.mulDiv(uint256(liquidity), Q96, uint256(start));
        return _mulDivRoundingUp(noFee, FEE_UNITS, FEE_UNITS - uint256(fee));
    }

    function _segmentCostToken0(uint160 start, uint160 end, uint128 liquidity, uint16 fee)
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

    // ──────────────────────────────────────────────
    // Waterfall pass: read pool state, solve ψ, execute
    // ──────────────────────────────────────────────

    function _waterfallPass(RebalanceParams calldata params) internal returns (uint256 spent, uint256 budgetBefore) {
        budgetBefore = IERC20(params.collateral).balanceOf(address(this));
        if (budgetBefore == 0) return (0, 0);

        (uint160[] memory sqrtPrices, uint128[] memory liquidities, uint16[] memory fees) = _readPoolState(params);
        (uint256[] memory order, uint160[] memory limits, uint256 count) = _buildConstantLBuyPlan(
            sqrtPrices, liquidities, params.sqrtPredX96, params.isToken1, budgetBefore, fees
        );
        if (count == 0) return (0, budgetBefore);

        spent = _executeBuyPlan(params, order, limits, count, budgetBefore);
    }

    // ──────────────────────────────────────────────
    // Phase 1: Sell overpriced outcomes
    // ──────────────────────────────────────────────

    function _pullAndSell(RebalanceParams calldata p) internal returns (uint256 amountOut) {
        uint256 n = p.tokens.length;

        if (p.collateralAmount != 0) {
            _safeTransferFrom(p.collateral, msg.sender, address(this), p.collateralAmount);
        }

        for (uint256 i = 0; i < n; i++) {
            if (p.balances[i] == 0) continue;
            _safeTransferFrom(p.tokens[i], msg.sender, address(this), p.balances[i]);
        }

        for (uint256 i = 0; i < n; i++) {
            uint256 bal = IERC20(p.tokens[i]).balanceOf(address(this));
            if (bal == 0) continue;

            (uint160 sqrtPrice,, uint16 poolFee,,,,) = IAlgebraPool(p.pools[i]).globalState();
            uint160 sellLimit = _sellLimit(p.sqrtPredX96[i], p.isToken1[i], poolFee);

            bool sellable = p.isToken1[i] ? sqrtPrice < sellLimit : sqrtPrice > sellLimit;
            if (!sellable) continue;

            _safeApprove(p.tokens[i], address(router), bal);
            amountOut += router.exactInputSingle(
                IAlgebraSwapRouter.ExactInputSingleParams({
                    tokenIn: p.tokens[i],
                    tokenOut: p.collateral,
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: bal,
                    amountOutMinimum: 0,
                    limitSqrtPrice: sellLimit
                })
            );
        }
    }

    // ──────────────────────────────────────────────
    // Compute ψ via closed-form (per-pool fees)
    // ──────────────────────────────────────────────

    function _computePsi(
        uint160[] memory s,
        uint128[] memory L,
        uint160[] calldata p,
        bool[] calldata isT1,
        uint256 budget,
        uint16[] memory fees
    ) internal pure returns (PsiResult memory result) {
        (uint256[] memory order, uint256 count) = _sortedUnderpricedByPriority(s, L, p, isT1);
        (result,) = _computePsiFromSorted(order, count, s, L, p, isT1, budget, fees);
    }

    function _buildConstantLBuyPlan(
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        uint160[] calldata sqrtPredX96,
        bool[] calldata isToken1,
        uint256 budget,
        uint16[] memory fees
    ) internal pure returns (uint256[] memory order, uint160[] memory limits, uint256 count) {
        (order, count) = _sortedUnderpricedByPriority(sqrtPrices, liquidities, sqrtPredX96, isToken1);
        (PsiResult memory psi, uint256 activeCount) =
            _computePsiFromSorted(order, count, sqrtPrices, liquidities, sqrtPredX96, isToken1, budget, fees);

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

    /// @dev ψ = (C + budget) / D with exact per-pool fee weighting.
    ///      C = Σ c_i × FEE_UNITS / (FEE_UNITS - fee_i)
    ///      D = Σ d_i × FEE_UNITS / (FEE_UNITS - fee_i)
    ///      where c_i = L_i × g(s_i), d_i = L_i × g(p_i).
    ///      Each coefficient is individually fee-adjusted, so no average fee approximation is needed.
    function _computePsiFromSorted(
        uint256[] memory order,
        uint256 count,
        uint160[] memory s,
        uint128[] memory L,
        uint160[] calldata p,
        bool[] calldata isT1,
        uint256 budget,
        uint16[] memory fees
    ) internal pure returns (PsiResult memory result, uint256 activeCount) {
        if (count == 0) {
            return (PsiResult({num: 1, den: 1, buyAll: false}), 0);
        }

        uint256 C = 0;
        uint256 D = 0;

        for (uint256 k = 0; k < count;) {
            uint256 i = order[k];
            (uint256 cRaw, uint256 dRaw) = _coefficientPair(s[i], L[i], p[i], isT1[i]);
            uint256 feeComp = FEE_UNITS - uint256(fees[i]);
            uint256 cContribution = cRaw * FEE_UNITS / feeComp;
            uint256 dContribution = dRaw * FEE_UNITS / feeComp;
            uint256 nextC = C + cContribution;
            uint256 nextD = D + dContribution;
            uint256 nextNum = nextC + budget;

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

        uint256 num = C + budget;
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
                IAlgebraSwapRouter.ExactInputSingleParams({
                    tokenIn: p.collateral,
                    tokenOut: p.tokens[i],
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: remaining,
                    amountOutMinimum: 0,
                    limitSqrtPrice: limits[i]
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
                IAlgebraSwapRouter.ExactInputSingleParams({
                    tokenIn: p.collateral,
                    tokenOut: p.tokens[i],
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: remaining,
                    amountOutMinimum: 0,
                    limitSqrtPrice: limit
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

    function _buyLimit(uint160 sqrtPrice, uint160 sqrtPred, bool isT1, PsiResult memory psi)
        internal
        pure
        returns (uint160)
    {
        if (psi.buyAll) {
            if (isT1) return sqrtPrice > sqrtPred ? sqrtPred : 0;
            else return sqrtPrice < sqrtPred ? sqrtPred : 0;
        }

        if (isT1) {
            if (!_mulGte(uint256(sqrtPrice), psi.num, uint256(sqrtPred), psi.den)) return 0;
            uint160 target = uint160(FullMath.mulDiv(uint256(sqrtPred), psi.den, psi.num));
            return target < sqrtPrice ? target : 0;
        } else {
            if (!_mulGte(uint256(sqrtPred), psi.num, uint256(sqrtPrice), psi.den)) return 0;
            uint160 target = uint160(FullMath.mulDiv(uint256(sqrtPred), psi.num, psi.den));
            return target > sqrtPrice ? target : 0;
        }
    }

    // ──────────────────────────────────────────────
    // Phase 5: Complete-set arbitrage (per-pool dynamic fees)
    // ──────────────────────────────────────────────

    function arb(
        address[] calldata tokens,
        address[] calldata pools,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint256 maxRounds
    ) external returns (uint256 profit) {
        profit = _arbLoop(tokens, pools, isToken1, collateral, market, maxRounds);

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

    function _arbLoop(
        address[] calldata tokens,
        address[] calldata pools,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint256 maxRounds
    ) internal returns (uint256 profit) {
        for (uint256 r = 0; r < maxRounds;) {
            uint256 gained = _arbRound(tokens, pools, isToken1, collateral, market);
            if (gained == 0) break;
            profit += gained;

            unchecked {
                ++r;
            }
        }
    }

    /// @dev One round of complete-set arb. Reads per-pool dynamic fees from globalState().
    function _arbRound(
        address[] calldata tokens,
        address[] calldata pools,
        bool[] calldata isToken1,
        address collateral,
        address market
    ) internal returns (uint256 gained) {
        uint256 n = pools.length;
        uint160[] memory sqrtPrices = new uint160[](n);
        uint16[] memory fees = new uint16[](n);
        // Compute exact per-pool fee-adjusted sums:
        // mintYield = Σ price_i × (1 - fee_i/1e6)   — net proceeds from selling all minted tokens
        // buyCost   = Σ price_i / (1 - fee_i/1e6)    — cost to buy one of each outcome token
        uint256 mintYield = 0;
        uint256 buyCost = 0;

        for (uint256 i = 0; i < n;) {
            (sqrtPrices[i],, fees[i],,,,) = IAlgebraPool(pools[i]).globalState();
            uint256 price;
            if (isToken1[i]) {
                price = FullMath.mulDiv(
                    FullMath.mulDiv(1e18, Q96, uint256(sqrtPrices[i])), Q96, uint256(sqrtPrices[i])
                );
            } else {
                price = FullMath.mulDiv(
                    FullMath.mulDiv(uint256(sqrtPrices[i]), uint256(sqrtPrices[i]), Q96), 1e18, Q96
                );
            }
            uint256 feeComp = FEE_UNITS - uint256(fees[i]);
            mintYield += price * feeComp / FEE_UNITS;
            buyCost += price * FEE_UNITS / feeComp;

            unchecked {
                ++i;
            }
        }

        if (mintYield > 1e18) {
            gained = _tryMintSell(tokens, isToken1, collateral, market, fees, sqrtPrices, mintYield);
        } else if (buyCost < 1e18) {
            gained = _tryBuyMerge(tokens, isToken1, collateral, market, fees, sqrtPrices, buyCost);
        }
    }

    function _tryMintSell(
        address[] calldata tokens,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint16[] memory /* fees */,
        uint160[] memory sqrtPrices,
        uint256 mintYield
    ) internal returns (uint256 gained) {
        uint256 collateralBal = IERC20(collateral).balanceOf(address(this));
        if (collateralBal == 0) return 0;
        uint256 n = tokens.length;

        _safeApprove(collateral, address(ctfRouter), collateralBal);
        ctfRouter.splitPosition(collateral, market, collateralBal);

        // Equilibrium: each price scales by K = 1e18/mintYield (< 1, prices go down).
        // mintYield already has per-pool fees baked in: Σ price_i × (1 - fee_i/1e6).
        // target_sqrtPrice = sqrtPrice × √K = sqrtPrice × √(1e18) / √(mintYield).
        // For token1 sell (sqrtPrice goes UP): limit = sqrtPrice × √(mintYield) / √(1e18) > sqrtPrice ✓
        // For token0 sell (sqrtPrice goes DOWN): limit = sqrtPrice × √(1e18) / √(mintYield) < sqrtPrice ✓
        uint256 sqrtA = _sqrt(mintYield);
        uint256 sqrtB = _sqrt(1e18);

        for (uint256 i = 0; i < n; i++) {
            uint256 tokenBal = IERC20(tokens[i]).balanceOf(address(this));
            if (tokenBal == 0) continue;

            uint160 limit;
            if (isToken1[i]) {
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtA, sqrtB));
                if (limit <= sqrtPrices[i]) continue;
            } else {
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtB, sqrtA));
                if (limit >= sqrtPrices[i]) continue;
            }

            _safeApprove(tokens[i], address(router), tokenBal);
            router.exactInputSingle(
                IAlgebraSwapRouter.ExactInputSingleParams({
                    tokenIn: tokens[i],
                    tokenOut: collateral,
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: tokenBal,
                    amountOutMinimum: 0,
                    limitSqrtPrice: limit
                })
            );
        }

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

        uint256 collateralAfter = IERC20(collateral).balanceOf(address(this));
        if (collateralAfter > collateralBal) {
            gained = collateralAfter - collateralBal;
        }
    }

    function _tryBuyMerge(
        address[] calldata tokens,
        bool[] calldata isToken1,
        address collateral,
        address market,
        uint16[] memory /* fees */,
        uint160[] memory sqrtPrices,
        uint256 buyCost
    ) internal returns (uint256 gained) {
        uint256 collateralBal = IERC20(collateral).balanceOf(address(this));
        if (collateralBal == 0) return 0;
        uint256 n = tokens.length;

        // Equilibrium: each price scales by K = 1e18/buyCost (> 1, prices go up).
        // buyCost already has per-pool fees baked in: Σ price_i / (1 - fee_i/1e6).
        // For token1 buy (sqrtPrice goes DOWN): limit = sqrtPrice × √(buyCost) / √(1e18) < sqrtPrice ✓
        // For token0 buy (sqrtPrice goes UP): limit = sqrtPrice × √(1e18) / √(buyCost) > sqrtPrice ✓
        uint256 sqrtC = _sqrt(buyCost);
        uint256 sqrtD = _sqrt(1e18);

        _safeApprove(collateral, address(router), collateralBal);

        for (uint256 i = 0; i < n; i++) {
            uint256 remaining = IERC20(collateral).balanceOf(address(this));
            if (remaining == 0) break;

            uint160 limit;
            if (isToken1[i]) {
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtC, sqrtD));
                if (limit >= sqrtPrices[i]) continue;
            } else {
                limit = uint160(FullMath.mulDiv(uint256(sqrtPrices[i]), sqrtD, sqrtC));
                if (limit <= sqrtPrices[i]) continue;
            }

            router.exactInputSingle(
                IAlgebraSwapRouter.ExactInputSingleParams({
                    tokenIn: collateral,
                    tokenOut: tokens[i],
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: remaining,
                    amountOutMinimum: 0,
                    limitSqrtPrice: limit
                })
            );
        }

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

        uint256 collateralAfter = IERC20(collateral).balanceOf(address(this));
        if (collateralAfter > collateralBal) {
            gained = collateralAfter - collateralBal;
        }
    }

    // ──────────────────────────────────────────────
    // Reverse waterfall: sell below-frontier holdings
    // ──────────────────────────────────────────────

    function _recycleSell(RebalanceParams calldata p, uint256 maxRounds)
        internal
        returns (uint256 totalRecycled, uint256 totalRedeployed)
    {
        for (uint256 r = 0; r < maxRounds;) {
            (uint160[] memory sqrtPrices, uint128[] memory liquidities, uint16[] memory fees) = _readPoolState(p);

            uint256 budget = IERC20(p.collateral).balanceOf(address(this));
            PsiResult memory psi = _computePsi(sqrtPrices, liquidities, p.sqrtPredX96, p.isToken1, budget, fees);

            if (psi.buyAll) break;
            if (psi.num == 0 && psi.den == 0) break;

            uint256 sold = _recycleSellRound(p, sqrtPrices, fees, psi);
            if (sold == 0) break;
            totalRecycled += sold;
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
        for (uint256 r = 0; r < maxRounds;) {
            (uint160[] memory sqrtPrices, int24[] memory ticks, uint128[] memory liquidities, uint16[] memory fees) =
                _readExactPoolState(p);
            (ExactCostCurves memory curves, uint256 buyAllCost) =
                _buildExactCostCurves(p, sqrtPrices, ticks, liquidities, fees, maxTickCrossingsPerPool);

            uint256 budget = IERC20(p.collateral).balanceOf(address(this));
            PsiResult memory psi = _computePsiExact(p, sqrtPrices, curves, buyAllCost, budget, maxBisectionIterations);

            if (psi.buyAll) break;
            if (psi.num == 0 && psi.den == 0) break;

            uint256 sold = _recycleSellRound(p, sqrtPrices, fees, psi);
            if (sold == 0) break;
            totalRecycled += sold;
            totalRedeployed += _readAndBuyExact(p, maxBisectionIterations, maxTickCrossingsPerPool);

            unchecked {
                ++r;
            }
        }
    }

    /// @dev Execute one round of reverse-waterfall sells for below-frontier holdings.
    function _recycleSellRound(
        RebalanceParams calldata p,
        uint160[] memory sqrtPrices,
        uint16[] memory fees,
        PsiResult memory psi
    ) internal returns (uint256 sold) {
        uint256 n = p.tokens.length;
        for (uint256 i = 0; i < n; i++) {
            uint256 bal = IERC20(p.tokens[i]).balanceOf(address(this));
            if (bal == 0) continue;

            uint160 limit;
            if (p.isToken1[i]) {
                if (_mulGte(uint256(sqrtPrices[i]), psi.num, uint256(p.sqrtPredX96[i]), psi.den)) continue;
                if (!_isRecycleWorthwhile(sqrtPrices[i], p.sqrtPredX96[i], true, psi, fees[i])) continue;
                limit = uint160(FullMath.mulDiv(uint256(p.sqrtPredX96[i]), psi.den, psi.num));
            } else {
                if (_mulGte(uint256(p.sqrtPredX96[i]), psi.num, uint256(sqrtPrices[i]), psi.den)) continue;
                if (!_isRecycleWorthwhile(sqrtPrices[i], p.sqrtPredX96[i], false, psi, fees[i])) continue;
                limit = uint160(FullMath.mulDiv(uint256(p.sqrtPredX96[i]), psi.num, psi.den));
            }

            _safeApprove(p.tokens[i], address(router), bal);
            sold += router.exactInputSingle(
                IAlgebraSwapRouter.ExactInputSingleParams({
                    tokenIn: p.tokens[i],
                    tokenOut: p.collateral,
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: bal,
                    amountOutMinimum: 0,
                    limitSqrtPrice: limit
                })
            );
        }
    }

    // ──────────────────────────────────────────────
    // Return all holdings to caller
    // ──────────────────────────────────────────────

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

    function _sqrt(uint256 x) internal pure returns (uint256 z) {
        if (x == 0) return 0;
        z = x;
        uint256 y = (z + 1) / 2;
        while (y < z) {
            z = y;
            y = (x / z + z) / 2;
        }
    }

    function _sellLimit(uint160 sqrtPred, bool isT1, uint16 fee) internal pure returns (uint160) {
        uint256 feeComp = FEE_UNITS - uint256(fee);
        uint256 sqrtFeeComp = _sqrt(feeComp * FEE_UNITS);
        if (sqrtFeeComp == 0) return sqrtPred;

        if (isT1) {
            return uint160(FullMath.mulDiv(uint256(sqrtPred), sqrtFeeComp, FEE_UNITS));
        }

        return uint160(FullMath.mulDiv(uint256(sqrtPred), FEE_UNITS, sqrtFeeComp));
    }

    function _isRecycleWorthwhile(uint160 sqrtPrice, uint160 sqrtPred, bool isT1, PsiResult memory psi, uint16 fee)
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

    /// @dev Read pool state from AlgebraV1.9 globalState() — returns per-pool dynamic fees.
    function _readPoolState(RebalanceParams calldata params)
        internal
        view
        returns (uint160[] memory sqrtPrices, uint128[] memory liquidities, uint16[] memory fees)
    {
        uint256 n = params.tokens.length;
        sqrtPrices = new uint160[](n);
        liquidities = new uint128[](n);
        fees = new uint16[](n);

        for (uint256 i = 0; i < n;) {
            (sqrtPrices[i],, fees[i],,,,) = IAlgebraPool(params.pools[i]).globalState();
            liquidities[i] = IAlgebraPool(params.pools[i]).liquidity();

            unchecked {
                ++i;
            }
        }
    }

    function _readExactPoolState(RebalanceParams calldata params)
        internal
        view
        returns (uint160[] memory sqrtPrices, int24[] memory ticks, uint128[] memory liquidities, uint16[] memory fees)
    {
        uint256 n = params.tokens.length;
        sqrtPrices = new uint160[](n);
        ticks = new int24[](n);
        liquidities = new uint128[](n);
        fees = new uint16[](n);

        for (uint256 i = 0; i < n;) {
            (sqrtPrices[i], ticks[i], fees[i],,,,) = IAlgebraPool(params.pools[i]).globalState();
            liquidities[i] = IAlgebraPool(params.pools[i]).liquidity();

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
