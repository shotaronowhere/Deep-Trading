// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Rebalancer} from "./Rebalancer.sol";
import {IERC20} from "./interfaces/IERC20.sol";
import {IV3SwapRouter} from "./interfaces/IV3SwapRouter.sol";
import {FullMath} from "./libraries/FullMath.sol";

/// @title Experimental greedy mixed-route rebalancer
/// @notice Keeps `Rebalancer` unchanged and adds a conservative direct-vs-mint stepper.
/// @dev This contract is intentionally heuristic. It does not port the full off-chain mixed-route
///      solver. Instead, it repeatedly:
///      1. Finds the highest direct-profitability outcome set (including exact ties)
///      2. Defines the next lower direct-profitability frontier
///      3. Chooses a route greedily: direct if basket sum <= 1, mint if basket sum > 1
///      4. Executes only a capped step, re-reads state, and repeats
contract RebalancerMixed is Rebalancer {
    uint256 internal constant ONE_WAD = 1e18;
    // Explicitly uncapped for benchmark exploration; mixed path still fail-closes
    // on other feasibility gates.
    uint256 internal constant MAX_MIXED_ACTIVE = type(uint256).max;
    uint256 internal constant COST_TOL_WAD = 1e6;
    uint256 internal constant MINT_RESIDUAL_TOL_WAD = 5e15;

    uint8 internal constant FB_ACTIVE_TOO_LARGE = 1;
    uint8 internal constant FB_NO_NON_ACTIVE = 2;
    uint8 internal constant FB_NO_SELLABLE = 3;
    uint8 internal constant FB_SOLVE_FAILED = 4;
    uint8 internal constant FB_RESIDUAL = 5;
    uint8 internal constant FB_ZERO_MINT = 6;
    uint8 internal constant FB_INVALID_PARAMS = 7;

    error ZeroStepBudget();
    error ZeroSteps();
    error InvalidIterations();
    error MixedSolveInfeasible();

    event MixedSolveFallback(uint8 reasonCode);

    struct NonActiveCurve {
        uint160 sqrtPrice;
        uint160 limit;
        uint128 liquidity;
        uint256 p0;
        uint256 cap;
        bool isToken1;
    }

    constructor(address _router, address _ctfRouter) Rebalancer(_router, _ctfRouter) {}

    /// @notice Experimental mixed-route rebalance.
    /// @param params Base rebalance inputs.
    /// @param market Market passed to the CTF router for split/merge operations.
    /// @param maxMixedSteps Hard cap on greedy iterations.
    /// @param maxStepCollateral Maximum collateral the loop may commit up front per step.
    /// @return totalProceeds Collateral realized from sell phase plus any net-collateral-positive mint steps.
    /// @return totalSpent Net collateral consumed by buy steps.
    function rebalanceMixed(
        RebalanceParams calldata params,
        address market,
        uint256 maxMixedSteps,
        uint256 maxStepCollateral
    ) external returns (uint256 totalProceeds, uint256 totalSpent) {
        if (maxMixedSteps == 0) revert ZeroSteps();
        if (maxStepCollateral == 0) revert ZeroStepBudget();

        totalProceeds = _pullAndSell(params);
        (uint256 extraProceeds, uint256 extraSpent) =
            _readAndBuyGreedyMixed(params, market, maxMixedSteps, maxStepCollateral);
        totalProceeds += extraProceeds;
        totalSpent += extraSpent;
        _returnAll(params);
    }

    /// @notice Constant-L mixed-route rebalance with deterministic 1D bisection solve.
    /// @dev Fail-closed: falls back to direct-only constant-L path when mixed solve/gates fail.
    function rebalanceMixedConstantL(
        RebalanceParams calldata params,
        address market,
        uint256 maxOuterIterations,
        uint256 maxInnerIterations,
        uint256 maxMintCollateral
    ) external returns (uint256 totalProceeds, uint256 totalSpent) {
        if (maxOuterIterations == 0 || maxInnerIterations == 0) revert InvalidIterations();

        totalProceeds = _pullAndSell(params);
        (uint256 mixedProceeds, uint256 mixedSpent, bool useFallback, uint8 fallbackReason) =
            _readAndBuyMixedConstantL(params, market, maxOuterIterations, maxInnerIterations, maxMintCollateral);

        if (useFallback) {
            emit MixedSolveFallback(fallbackReason);
            mixedSpent += _readAndBuyConstantL(params);
        }

        totalProceeds += mixedProceeds;
        totalSpent += mixedSpent;
        _returnAll(params);
    }

    function _readAndBuyMixedConstantL(
        RebalanceParams calldata params,
        address market,
        uint256 maxOuterIterations,
        uint256 maxInnerIterations,
        uint256 maxMintCollateral
    ) internal returns (uint256 extraProceeds, uint256 extraSpent, bool useFallback, uint8 fallbackReason) {
        uint256 budget = IERC20(params.collateral).balanceOf(address(this));
        if (budget == 0) return (0, 0, false, 0);

        (uint160[] memory sqrtPrices, uint128[] memory liquidities) = _readPoolState(params);
        (uint256[] memory order, uint256 count) =
            _sortedUnderpricedByPriority(sqrtPrices, liquidities, params.sqrtPredX96, params.isToken1);
        if (count == 0) return (0, 0, false, 0);

        (PsiResult memory directPsi, uint256 activeCount) =
            _computePsiFromSorted(order, count, sqrtPrices, liquidities, params.sqrtPredX96, params.isToken1, budget, params.fee);
        if (activeCount == 0) return (0, 0, false, 0);
        if (activeCount > MAX_MIXED_ACTIVE) return (0, 0, true, FB_ACTIVE_TOO_LARGE);

        bool[] memory active = new bool[](params.tokens.length);
        uint256 piStarWad = 0;
        uint256 s0Wad = 0;
        uint256 priceSum = 0;
        for (uint256 i = 0; i < params.tokens.length; i++) {
            uint256 p0 = _priceE18(sqrtPrices[i], params.isToken1[i]);
            priceSum += p0;
        }
        for (uint256 k = 0; k < activeCount; k++) {
            uint256 idx = order[k];
            active[idx] = true;
            piStarWad += _priceE18(params.sqrtPredX96[idx], params.isToken1[idx]);
        }
        uint256 activeUniverseCount = 0;
        for (uint256 i = 0; i < params.tokens.length; i++) {
            if (active[i]) {
                activeUniverseCount++;
            } else {
                s0Wad += _priceE18(sqrtPrices[i], params.isToken1[i]);
            }
        }
        if (activeUniverseCount == params.tokens.length) return (0, 0, true, FB_NO_NON_ACTIVE);

        if (piStarWad == 0) return (0, 0, true, FB_SOLVE_FAILED);

        uint256 piLoWad = activeCount < count ? _profitabilityWad(order[activeCount], params, sqrtPrices) : 0;
        uint256 directPsiPiWad = _piFromPsi(directPsi);
        uint256 topPiWad = _profitabilityWad(order[0], params, sqrtPrices);
        uint256 piHiWad = directPsiPiWad > topPiWad ? directPsiPiWad : topPiWad;
        if (piHiWad <= piLoWad) return (0, 0, true, FB_SOLVE_FAILED);

        uint256 iStar = order[0];
        uint256 bestDeficit = 0;
        uint256 onePlusHi = ONE_WAD + piHiWad;
        for (uint256 k = 0; k < activeCount; k++) {
            uint256 idx = order[k];
            uint256 target = FullMath.mulDiv(_priceE18(params.sqrtPredX96[idx], params.isToken1[idx]), ONE_WAD, onePlusHi);
            uint256 currentAlt = ONE_WAD > (priceSum - _priceE18(sqrtPrices[idx], params.isToken1[idx]))
                ? ONE_WAD - (priceSum - _priceE18(sqrtPrices[idx], params.isToken1[idx]))
                : 0;
            uint256 deficit = target > currentAlt ? target - currentAlt : 0;
            if (deficit > bestDeficit) {
                bestDeficit = deficit;
                iStar = idx;
            }
        }

        uint256 mintBudgetCap = maxMintCollateral == 0 ? budget : _min(budget, maxMintCollateral);
        if (mintBudgetCap == 0) return (0, 0, true, FB_INVALID_PARAMS);

        (bool ok, uint256 piWad, uint256 rhoWad, uint256 mintAmount, bool anySellable) = _solveMixedForActiveSet(
            params,
            sqrtPrices,
            liquidities,
            active,
            piLoWad,
            piHiWad,
            piStarWad,
            s0Wad,
            mintBudgetCap,
            budget,
            maxOuterIterations,
            maxInnerIterations
        );
        if (!anySellable) return (0, 0, true, FB_NO_SELLABLE);
        if (!ok) return (0, 0, true, FB_SOLVE_FAILED);
        if (mintAmount == 0) return (0, 0, true, FB_ZERO_MINT);

        if (!_residualWithinTolerance(params, sqrtPrices, liquidities, active, piWad, mintAmount, piStarWad, s0Wad, iStar)) {
            return (0, 0, true, FB_RESIDUAL);
        }

        uint256 collateralBefore = IERC20(params.collateral).balanceOf(address(this));
        bool minted = _executeMintFrontierStep(
            params,
            market,
            sqrtPrices,
            active,
            activeCount,
            rhoWad,
            ONE_WAD,
            mintAmount
        );
        if (!minted) return (0, 0, true, FB_SOLVE_FAILED);

        (uint160[] memory sqrtAfter,) = _readPoolState(params);
        PsiResult memory frontierPsi = PsiResult({num: ONE_WAD, den: rhoWad, buyAll: false});
        (uint256[] memory buyOrder, uint160[] memory limits, uint256 buyCount) =
            _buildBuyPlan(sqrtAfter, params.sqrtPredX96, params.isToken1, frontierPsi);
        if (buyCount > 0) {
            uint256 buyBudget = IERC20(params.collateral).balanceOf(address(this));
            _executeBuyPlan(params, buyOrder, limits, buyCount, buyBudget);
        }

        uint256 collateralAfter = IERC20(params.collateral).balanceOf(address(this));
        if (collateralAfter > collateralBefore) {
            extraProceeds = collateralAfter - collateralBefore;
        } else {
            extraSpent = collateralBefore - collateralAfter;
        }
    }

    function _solveMixedForActiveSet(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        bool[] memory active,
        uint256 piLoWad,
        uint256 piHiWad,
        uint256 piStarWad,
        uint256 s0Wad,
        uint256 mintBudgetCap,
        uint256 budget,
        uint256 maxOuterIterations,
        uint256 maxInnerIterations
    ) internal pure returns (bool ok, uint256 piWad, uint256 rhoWad, uint256 mintAmount, bool anySellable) {
        (bool okLo, uint256 costLo,, bool sellableAtLo) = _evaluateMixedCostAtPi(
            params,
            sqrtPrices,
            liquidities,
            active,
            piLoWad,
            piStarWad,
            s0Wad,
            mintBudgetCap,
            maxInnerIterations
        );
        anySellable = sellableAtLo;
        if (!okLo) return (false, 0, 0, 0, anySellable);
        if (costLo <= budget) {
            uint256 rhoLo = _frontierRhoFromPi(piLoWad);
            (bool okMint, uint256 mintLo) = _solveMintAmountForDelta(
                params,
                sqrtPrices,
                liquidities,
                active,
                piLoWad,
                piStarWad,
                s0Wad,
                mintBudgetCap,
                maxInnerIterations
            );
            if (!okMint) return (false, 0, 0, 0, anySellable);
            return (true, piLoWad, rhoLo, mintLo, anySellable);
        }

        (bool okHi, uint256 costHi, uint256 mintHi, bool sellableAtHi) = _evaluateMixedCostAtPi(
            params,
            sqrtPrices,
            liquidities,
            active,
            piHiWad,
            piStarWad,
            s0Wad,
            mintBudgetCap,
            maxInnerIterations
        );
        anySellable = anySellable || sellableAtHi;
        if (!okHi || costHi > budget) return (false, 0, 0, 0, anySellable);

        uint256 lo = piLoWad;
        uint256 hi = piHiWad;
        uint256 bestMint = mintHi;

        for (uint256 iter = 0; iter < maxOuterIterations;) {
            if (lo >= hi) break;
            uint256 mid = (lo + hi) / 2;
            (bool okMid, uint256 costMid, uint256 mintMid, bool sellableAtMid) = _evaluateMixedCostAtPi(
                params,
                sqrtPrices,
                liquidities,
                active,
                mid,
                piStarWad,
                s0Wad,
                mintBudgetCap,
                maxInnerIterations
            );
            anySellable = anySellable || sellableAtMid;
            if (!okMid) return (false, 0, 0, 0, anySellable);

            if (costMid > budget) {
                lo = mid + 1;
            } else {
                hi = mid;
                bestMint = mintMid;
            }

            unchecked {
                ++iter;
            }
        }

        (bool okFinal, uint256 costFinal, uint256 mintFinal, bool sellableAtFinal) = _evaluateMixedCostAtPi(
            params,
            sqrtPrices,
            liquidities,
            active,
            hi,
            piStarWad,
            s0Wad,
            mintBudgetCap,
            maxInnerIterations
        );
        anySellable = anySellable || sellableAtFinal;
        if (!okFinal) return (false, 0, 0, 0, anySellable);

        if (budget > costFinal && (budget - costFinal) > COST_TOL_WAD) {
            return (false, 0, 0, 0, anySellable);
        }

        piWad = hi;
        rhoWad = _frontierRhoFromPi(piWad);
        mintAmount = mintFinal > 0 ? mintFinal : bestMint;
        ok = true;
    }

    function _evaluateMixedCostAtPi(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        bool[] memory active,
        uint256 piWad,
        uint256 piStarWad,
        uint256 s0Wad,
        uint256 mintBudgetCap,
        uint256 maxInnerIterations
    ) internal pure returns (bool ok, uint256 totalCost, uint256 mintAmount, bool anySellable) {
        uint256 onePlusPi = ONE_WAD + piWad;
        uint256 piTargetWad = FullMath.mulDiv(piStarWad, ONE_WAD, onePlusPi);
        uint256 base = ONE_WAD > s0Wad ? ONE_WAD - s0Wad : 0;
        uint256 delta = piTargetWad > base ? piTargetWad - base : 0;
        uint256 rhoWad = _frontierRhoFromPi(piWad);

        (NonActiveCurve[] memory curves, uint256 maxCap, bool sellable) =
            _buildNonActiveCurves(params, sqrtPrices, liquidities, active, rhoWad);
        anySellable = sellable;
        if (delta > 0 && !sellable) return (false, 0, 0, false);

        uint256 mintCap = _min(mintBudgetCap, maxCap);
        if (delta > 0 && mintCap == 0) return (false, 0, 0, anySellable);

        (bool okMint, uint256 mintSolved) =
            _solveMintAmountForDeltaFromCurves(curves, delta, mintCap, params.fee, maxInnerIterations);
        if (!okMint) return (false, 0, 0, anySellable);
        mintAmount = mintSolved;

        (uint256 deltaSolved, uint256 proceeds,) = _deltaAndProceedsForMint(curves, mintAmount, params.fee);
        if (deltaSolved + 1 < delta) return (false, 0, 0, anySellable);

        uint256 mintCost = mintAmount > proceeds ? mintAmount - proceeds : 0;
        uint256 directCost = _directCostAtRho(params, sqrtPrices, liquidities, active, rhoWad);
        totalCost = directCost + mintCost;
        ok = true;
    }

    function _buildNonActiveCurves(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        bool[] memory active,
        uint256 rhoWad
    ) internal pure returns (NonActiveCurve[] memory curves, uint256 maxCap, bool anySellable) {
        uint256 n = params.tokens.length;
        uint256 nonActiveCount = 0;
        for (uint256 i = 0; i < n;) {
            if (!active[i]) nonActiveCount++;
            unchecked {
                ++i;
            }
        }

        curves = new NonActiveCurve[](nonActiveCount);
        uint256 cursor = 0;
        for (uint256 i = 0; i < n;) {
            if (active[i]) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint160 limit = _frontierLimit(params.sqrtPredX96[i], params.isToken1[i], rhoWad, ONE_WAD);
            uint256 cap = _effectiveSellCapToLimit(
                sqrtPrices[i], limit, params.isToken1[i], liquidities[i], params.fee
            );
            if (cap > 0) {
                anySellable = true;
                if (cap > maxCap) maxCap = cap;
            }

            curves[cursor] = NonActiveCurve({
                sqrtPrice: sqrtPrices[i],
                limit: limit,
                liquidity: liquidities[i],
                p0: _priceE18(sqrtPrices[i], params.isToken1[i]),
                cap: cap,
                isToken1: params.isToken1[i]
            });
            cursor++;

            unchecked {
                ++i;
            }
        }
    }

    function _solveMintAmountForDelta(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        bool[] memory active,
        uint256 piWad,
        uint256 piStarWad,
        uint256 s0Wad,
        uint256 mintBudgetCap,
        uint256 maxInnerIterations
    ) internal pure returns (bool ok, uint256 mintAmount) {
        uint256 onePlusPi = ONE_WAD + piWad;
        uint256 piTargetWad = FullMath.mulDiv(piStarWad, ONE_WAD, onePlusPi);
        uint256 base = ONE_WAD > s0Wad ? ONE_WAD - s0Wad : 0;
        uint256 delta = piTargetWad > base ? piTargetWad - base : 0;
        uint256 rhoWad = _frontierRhoFromPi(piWad);

        (NonActiveCurve[] memory curves, uint256 maxCap,) =
            _buildNonActiveCurves(params, sqrtPrices, liquidities, active, rhoWad);
        uint256 mintCap = _min(mintBudgetCap, maxCap);
        return _solveMintAmountForDeltaFromCurves(curves, delta, mintCap, params.fee, maxInnerIterations);
    }

    function _solveMintAmountForDeltaFromCurves(
        NonActiveCurve[] memory curves,
        uint256 deltaTarget,
        uint256 mintCap,
        uint24 fee,
        uint256 maxInnerIterations
    ) internal pure returns (bool ok, uint256 mintAmount) {
        if (deltaTarget == 0) return (true, 0);
        if (mintCap == 0) return (false, 0);

        (uint256 deltaHi,,) = _deltaAndProceedsForMint(curves, mintCap, fee);
        if (deltaHi + 1 < deltaTarget) return (false, 0);

        uint256 lo = 0;
        uint256 hi = mintCap;
        for (uint256 iter = 0; iter < maxInnerIterations;) {
            if (lo >= hi) break;
            uint256 mid = (lo + hi) / 2;
            (uint256 deltaMid,,) = _deltaAndProceedsForMint(curves, mid, fee);
            if (deltaMid + 1 < deltaTarget) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
            unchecked {
                ++iter;
            }
        }

        mintAmount = hi;
        ok = true;
    }

    function _deltaAndProceedsForMint(NonActiveCurve[] memory curves, uint256 mintAmount, uint24 fee)
        internal
        pure
        returns (uint256 deltaWad, uint256 proceeds, uint256 sumNewPrices)
    {
        uint256 len = curves.length;
        for (uint256 i = 0; i < len;) {
            NonActiveCurve memory c = curves[i];
            uint256 sold = mintAmount < c.cap ? mintAmount : c.cap;
            if (sold == 0) {
                sumNewPrices += c.p0;
                unchecked {
                    ++i;
                }
                continue;
            }

            (uint256 newPriceWad, uint256 outCollateral) = _simulateSellOnCurve(c, sold, fee);
            if (c.p0 > newPriceWad) {
                deltaWad += c.p0 - newPriceWad;
            }
            proceeds += outCollateral;
            sumNewPrices += newPriceWad;

            unchecked {
                ++i;
            }
        }
    }

    function _simulateSellOnCurve(NonActiveCurve memory c, uint256 sellAmount, uint24 fee)
        internal
        pure
        returns (uint256 newPriceWad, uint256 outCollateral)
    {
        if (sellAmount == 0 || c.liquidity == 0) {
            return (c.p0, 0);
        }

        uint256 feeComp = FEE_UNITS - uint256(fee);
        uint256 effectiveIn = FullMath.mulDiv(sellAmount, feeComp, FEE_UNITS);
        if (effectiveIn == 0) return (c.p0, 0);

        uint256 sqrtNew;
        if (c.isToken1) {
            uint256 deltaS = FullMath.mulDiv(effectiveIn, Q96, uint256(c.liquidity));
            sqrtNew = uint256(c.sqrtPrice) + deltaS;
            if (sqrtNew > uint256(c.limit)) sqrtNew = uint256(c.limit);

            outCollateral = FullMath.mulDiv(uint256(c.liquidity), Q96, uint256(c.sqrtPrice))
                - FullMath.mulDiv(uint256(c.liquidity), Q96, sqrtNew);
        } else {
            uint256 lq = uint256(c.liquidity) * Q96;
            uint256 den = lq + effectiveIn * uint256(c.sqrtPrice);
            sqrtNew = FullMath.mulDiv(uint256(c.sqrtPrice), lq, den);
            if (sqrtNew < uint256(c.limit)) sqrtNew = uint256(c.limit);

            outCollateral = FullMath.mulDiv(uint256(c.liquidity), uint256(c.sqrtPrice) - sqrtNew, Q96);
        }

        newPriceWad = _priceE18(uint160(sqrtNew), c.isToken1);
    }

    function _directCostAtRho(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        bool[] memory active,
        uint256 rhoWad
    ) internal pure returns (uint256 totalCost) {
        uint256 n = params.tokens.length;
        for (uint256 i = 0; i < n;) {
            if (!active[i] || liquidities[i] == 0) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint160 limit = _frontierLimit(params.sqrtPredX96[i], params.isToken1[i], rhoWad, ONE_WAD);
            if (params.isToken1[i]) {
                if (limit < sqrtPrices[i]) {
                    totalCost += _segmentCostToken1(sqrtPrices[i], limit, liquidities[i], params.fee);
                }
            } else if (limit > sqrtPrices[i]) {
                totalCost += _segmentCostToken0(sqrtPrices[i], limit, liquidities[i], params.fee);
            }

            unchecked {
                ++i;
            }
        }
    }

    function _residualWithinTolerance(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        uint128[] memory liquidities,
        bool[] memory active,
        uint256 piWad,
        uint256 mintAmount,
        uint256 piStarWad,
        uint256 s0Wad,
        uint256 iStar
    ) internal pure returns (bool) {
        if (mintAmount == 0) return true;
        if (!active[iStar]) return false;

        uint256 rhoWad = _frontierRhoFromPi(piWad);
        (NonActiveCurve[] memory curves,, bool anySellable) =
            _buildNonActiveCurves(params, sqrtPrices, liquidities, active, rhoWad);
        if (!anySellable) return false;

        (, uint256 proceeds, uint256 sumNewPrices) = _deltaAndProceedsForMint(curves, mintAmount, params.fee);
        if (proceeds > mintAmount + mintAmount) return false;

        uint256 onePlusPi = ONE_WAD + piWad;
        uint256 piTargetWad = FullMath.mulDiv(piStarWad, ONE_WAD, onePlusPi);
        uint256 base = ONE_WAD > s0Wad ? ONE_WAD - s0Wad : 0;
        uint256 solvedDelta = piTargetWad > base ? piTargetWad - base : 0;
        uint256 reconstructedDelta = s0Wad > sumNewPrices ? s0Wad - sumNewPrices : 0;
        return _absDiff(solvedDelta, reconstructedDelta) <= MINT_RESIDUAL_TOL_WAD;
    }

    function _effectiveSellCapToLimit(uint160 sqrtPrice, uint160 limit, bool isToken1, uint128 liquidity, uint24 fee)
        internal
        pure
        returns (uint256)
    {
        if (liquidity == 0) return 0;
        if (isToken1) {
            if (limit <= sqrtPrice) return 0;
            return _segmentCostToken0(sqrtPrice, limit, liquidity, fee);
        }
        if (limit >= sqrtPrice) return 0;
        return _segmentCostToken1(sqrtPrice, limit, liquidity, fee);
    }

    function _frontierRhoFromPi(uint256 piWad) internal pure returns (uint256) {
        return _sqrt((ONE_WAD + piWad) * ONE_WAD);
    }

    function _profitabilityWad(uint256 idx, RebalanceParams calldata params, uint160[] memory sqrtPrices)
        internal
        pure
        returns (uint256)
    {
        (bool underpriced, uint256 num, uint256 den) =
            _directProfitRatio(params.sqrtPredX96[idx], sqrtPrices[idx], params.isToken1[idx]);
        if (!underpriced || den == 0) return 0;

        uint256 ratioWad = FullMath.mulDiv(num, ONE_WAD, den);
        uint256 sq = FullMath.mulDiv(ratioWad, ratioWad, ONE_WAD);
        return sq > ONE_WAD ? sq - ONE_WAD : 0;
    }

    function _piFromPsi(PsiResult memory psi) internal pure returns (uint256) {
        if (psi.buyAll || psi.num == 0) return 0;
        uint256 invPsiWad = FullMath.mulDiv(psi.den, ONE_WAD, psi.num);
        uint256 sq = FullMath.mulDiv(invPsiWad, invPsiWad, ONE_WAD);
        return sq > ONE_WAD ? sq - ONE_WAD : 0;
    }

    function _absDiff(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a - b : b - a;
    }

    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    function _readAndBuyGreedyMixed(
        RebalanceParams calldata params,
        address market,
        uint256 maxMixedSteps,
        uint256 maxStepCollateral
    ) internal returns (uint256 totalProceeds, uint256 totalSpent) {
        uint256 n = params.tokens.length;
        bool[] memory active = new bool[](n);

        for (uint256 step = 0; step < maxMixedSteps;) {
            (bool didWork, uint256 stepProceeds, uint256 stepSpent) =
                _executeGreedyMixedStep(params, market, maxStepCollateral, active);
            if (!didWork) break;

            totalProceeds += stepProceeds;
            totalSpent += stepSpent;

            unchecked {
                ++step;
            }
        }
    }

    function _executeGreedyMixedStep(
        RebalanceParams calldata params,
        address market,
        uint256 maxStepCollateral,
        bool[] memory active
    ) internal returns (bool didWork, uint256 stepProceeds, uint256 stepSpent) {
        uint256 budget = IERC20(params.collateral).balanceOf(address(this));
        if (budget == 0) return (false, 0, 0);

        (uint160[] memory sqrtPrices,) = _readPoolState(params);
        (bool foundBest, uint256 bestIdx, uint256 activeCount) =
            _markTopDirectProfitability(params, sqrtPrices, active);
        if (!foundBest) return (false, 0, 0);

        (uint256 frontierNum, uint256 frontierDen) = _nextFrontierRatio(params, sqrtPrices, active, bestIdx);
        uint256 stepBudget = budget < maxStepCollateral ? budget : maxStepCollateral;
        uint256 collateralBefore = budget;

        if (_useMintForActiveBundle(params, sqrtPrices, active, frontierNum, frontierDen)) {
            didWork =
                _executeMintFrontierStep(params, market, sqrtPrices, active, activeCount, frontierNum, frontierDen, stepBudget);
        } else {
            didWork = _executeDirectFrontierStep(params, sqrtPrices, active, frontierNum, frontierDen, stepBudget);
        }
        if (!didWork) return (false, 0, 0);

        uint256 collateralAfter = IERC20(params.collateral).balanceOf(address(this));
        if (collateralAfter > collateralBefore) {
            stepProceeds = collateralAfter - collateralBefore;
        } else {
            stepSpent = collateralBefore - collateralAfter;
        }
    }

    function _markTopDirectProfitability(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        bool[] memory active
    ) internal pure returns (bool foundBest, uint256 bestIdx, uint256 activeCount) {
        uint256 n = params.tokens.length;

        for (uint256 i = 0; i < n;) {
            active[i] = false;
            unchecked {
                ++i;
            }
        }

        uint256 bestNum = 0;
        uint256 bestDen = 1;

        for (uint256 i = 0; i < n;) {
            (bool underpriced, uint256 num, uint256 den) = _directProfitRatio(params.sqrtPredX96[i], sqrtPrices[i], params.isToken1[i]);
            if (!underpriced) {
                unchecked {
                    ++i;
                }
                continue;
            }

            if (!foundBest || _mulCompare(num, bestDen, bestNum, den) > 0) {
                foundBest = true;
                bestIdx = i;
                bestNum = num;
                bestDen = den;
            }

            unchecked {
                ++i;
            }
        }

        if (!foundBest) {
            return (false, 0, 0);
        }

        for (uint256 i = 0; i < n;) {
            (bool underpriced, uint256 num, uint256 den) = _directProfitRatio(params.sqrtPredX96[i], sqrtPrices[i], params.isToken1[i]);
            if (underpriced && _mulCompare(num, bestDen, bestNum, den) == 0) {
                active[i] = true;
                activeCount++;
            }

            unchecked {
                ++i;
            }
        }
    }

    function _nextFrontierRatio(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        bool[] memory active,
        uint256 bestIdx
    ) internal pure returns (uint256 frontierNum, uint256 frontierDen) {
        frontierNum = 1;
        frontierDen = 1;

        (bool bestUnderpriced, uint256 bestNum, uint256 bestDen) =
            _directProfitRatio(params.sqrtPredX96[bestIdx], sqrtPrices[bestIdx], params.isToken1[bestIdx]);
        if (!bestUnderpriced) {
            return (frontierNum, frontierDen);
        }

        bool foundLower = false;
        uint256 n = params.tokens.length;
        for (uint256 i = 0; i < n;) {
            if (active[i]) {
                unchecked {
                    ++i;
                }
                continue;
            }

            (bool underpriced, uint256 num, uint256 den) =
                _directProfitRatio(params.sqrtPredX96[i], sqrtPrices[i], params.isToken1[i]);
            if (!underpriced || _mulCompare(num, bestDen, bestNum, den) >= 0) {
                unchecked {
                    ++i;
                }
                continue;
            }

            if (!foundLower || _mulCompare(num, frontierDen, frontierNum, den) > 0) {
                foundLower = true;
                frontierNum = num;
                frontierDen = den;
            }

            unchecked {
                ++i;
            }
        }
    }

    function _executeDirectFrontierStep(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        bool[] memory active,
        uint256 frontierNum,
        uint256 frontierDen,
        uint256 stepBudget
    ) internal returns (bool didWork) {
        if (stepBudget == 0) return false;

        uint256 spent = 0;
        uint256 n = params.tokens.length;
        _safeApprove(params.collateral, address(router), stepBudget);

        for (uint256 i = 0; i < n;) {
            if (!active[i] || spent >= stepBudget) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint160 limit = _frontierLimit(params.sqrtPredX96[i], params.isToken1[i], frontierNum, frontierDen);
            if (!_isDirectBuyLimitValid(params.isToken1[i], sqrtPrices[i], limit)) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint256 amountIn = stepBudget - spent;
            uint256 before = IERC20(params.collateral).balanceOf(address(this));
            router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: params.collateral,
                    tokenOut: params.tokens[i],
                    fee: params.fee,
                    recipient: address(this),
                    amountIn: amountIn,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: limit
                })
            );
            uint256 afterBal = IERC20(params.collateral).balanceOf(address(this));
            if (before > afterBal) {
                spent += before - afterBal;
                didWork = true;
            }

            unchecked {
                ++i;
            }
        }
    }

    function _executeMintFrontierStep(
        RebalanceParams calldata params,
        address market,
        uint160[] memory sqrtPrices,
        bool[] memory active,
        uint256 activeCount,
        uint256 frontierNum,
        uint256 frontierDen,
        uint256 stepBudget
    ) internal returns (bool didWork) {
        if (stepBudget == 0 || activeCount == 0) return false;

        uint256 n = params.tokens.length;
        uint256[] memory preBalances = new uint256[](n);
        for (uint256 i = 0; i < n;) {
            preBalances[i] = IERC20(params.tokens[i]).balanceOf(address(this));
            unchecked {
                ++i;
            }
        }

        _safeApprove(params.collateral, address(ctfRouter), stepBudget);
        ctfRouter.splitPosition(params.collateral, market, stepBudget);

        bool soldAny = false;
        for (uint256 i = 0; i < n;) {
            if (active[i]) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint160 limit = _frontierLimit(params.sqrtPredX96[i], params.isToken1[i], frontierNum, frontierDen);
            if (!_isDirectSellLimitValid(params.isToken1[i], sqrtPrices[i], limit)) {
                unchecked {
                    ++i;
                }
                continue;
            }

            uint256 mintedDelta = IERC20(params.tokens[i]).balanceOf(address(this)) - preBalances[i];
            if (mintedDelta == 0) {
                unchecked {
                    ++i;
                }
                continue;
            }

            _safeApprove(params.tokens[i], address(router), mintedDelta);
            router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: params.tokens[i],
                    tokenOut: params.collateral,
                    fee: params.fee,
                    recipient: address(this),
                    amountIn: mintedDelta,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: limit
                })
            );
            soldAny = true;

            unchecked {
                ++i;
            }
        }

        uint256 mergeAmount = type(uint256).max;
        for (uint256 i = 0; i < n;) {
            uint256 current = IERC20(params.tokens[i]).balanceOf(address(this));
            uint256 delta = current > preBalances[i] ? current - preBalances[i] : 0;
            if (delta < mergeAmount) mergeAmount = delta;

            unchecked {
                ++i;
            }
        }

        if (mergeAmount == 0 || mergeAmount == type(uint256).max) {
            return soldAny;
        }

        for (uint256 i = 0; i < n;) {
            _safeApprove(params.tokens[i], address(ctfRouter), mergeAmount);
            unchecked {
                ++i;
            }
        }
        ctfRouter.mergePositions(params.collateral, market, mergeAmount);

        return soldAny || mergeAmount < stepBudget;
    }

    function _frontierLimit(uint160 sqrtPred, bool isToken1, uint256 frontierNum, uint256 frontierDen)
        internal
        pure
        returns (uint160)
    {
        if (isToken1) {
            return uint160(FullMath.mulDiv(uint256(sqrtPred), frontierNum, frontierDen));
        }
        return uint160(FullMath.mulDiv(uint256(sqrtPred), frontierDen, frontierNum));
    }

    function _useMintForActiveBundle(
        RebalanceParams calldata params,
        uint160[] memory sqrtPrices,
        bool[] memory active,
        uint256 frontierNum,
        uint256 frontierDen
    ) internal pure returns (bool) {
        uint256 n = params.tokens.length;
        uint256 directBundlePrice = 0;
        uint256 sellableOtherPrice = 0;

        for (uint256 i = 0; i < n;) {
            uint256 price = _priceE18(sqrtPrices[i], params.isToken1[i]);
            if (active[i]) {
                directBundlePrice += price;
            } else {
                uint160 limit = _frontierLimit(params.sqrtPredX96[i], params.isToken1[i], frontierNum, frontierDen);
                if (_isDirectSellLimitValid(params.isToken1[i], sqrtPrices[i], limit)) {
                    sellableOtherPrice += price;
                }
            }

            unchecked {
                ++i;
            }
        }

        if (directBundlePrice == 0) return false;

        uint256 altBundlePrice = sellableOtherPrice >= ONE_WAD ? 0 : ONE_WAD - sellableOtherPrice;
        return altBundlePrice < directBundlePrice;
    }

    function _priceE18(uint160 sqrtPrice, bool isToken1) internal pure returns (uint256) {
        if (isToken1) {
            return FullMath.mulDiv(
                FullMath.mulDiv(1e18, Q96, uint256(sqrtPrice)),
                Q96,
                uint256(sqrtPrice)
            );
        }

        return FullMath.mulDiv(
            FullMath.mulDiv(uint256(sqrtPrice), uint256(sqrtPrice), Q96),
            1e18,
            Q96
        );
    }

    function _directProfitRatio(uint160 sqrtPred, uint160 sqrtPrice, bool isToken1)
        internal
        pure
        returns (bool underpriced, uint256 num, uint256 den)
    {
        if (isToken1) {
            num = uint256(sqrtPrice);
            den = uint256(sqrtPred);
        } else {
            num = uint256(sqrtPred);
            den = uint256(sqrtPrice);
        }

        underpriced = num > den;
    }

    function _isDirectBuyLimitValid(bool isToken1, uint160 current, uint160 limit) internal pure returns (bool) {
        if (limit == 0) return false;
        return isToken1 ? limit < current : limit > current;
    }

    function _isDirectSellLimitValid(bool isToken1, uint160 current, uint160 limit) internal pure returns (bool) {
        if (limit == 0) return false;
        return isToken1 ? limit > current : limit < current;
    }
}
