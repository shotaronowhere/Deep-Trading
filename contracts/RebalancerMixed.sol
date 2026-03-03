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

    error ZeroStepBudget();
    error ZeroSteps();

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
