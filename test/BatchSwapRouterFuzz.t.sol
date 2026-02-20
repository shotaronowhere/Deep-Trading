// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {BatchSwapRouter} from "../contracts/BatchSwapRouter.sol";
import {IBatchSwapRouter} from "../contracts/interfaces/IBatchSwapRouter.sol";
import {MockERC20, MockV3SwapRouter} from "./utils/BatchSwapRouterMocks.sol";

contract BatchSwapRouterFuzzTest is Test {
    MockERC20 internal tokenIn;
    MockERC20 internal tokenInAlt;
    MockERC20 internal tokenOut;
    MockERC20 internal tokenOutAlt;
    MockV3SwapRouter internal router;
    BatchSwapRouter internal batch;

    function setUp() public {
        tokenIn = new MockERC20();
        tokenInAlt = new MockERC20();
        tokenOut = new MockERC20();
        tokenOutAlt = new MockERC20();
        router = new MockV3SwapRouter();
        batch = new BatchSwapRouter(address(router));

        tokenIn.mint(address(this), 1_000_000);
        tokenInAlt.mint(address(this), 1_000_000);
        tokenIn.approve(address(batch), type(uint256).max);
        tokenInAlt.approve(address(batch), type(uint256).max);
    }

    function testFuzzExactInputAggregatesAcrossSwaps(
        uint8 rawCount,
        uint96 rawAmountInPerSwap,
        uint96 rawOutPerSwap
    ) public {
        uint256 count = bound(uint256(rawCount), 1, 8);
        uint256 amountInPerSwap = bound(uint256(rawAmountInPerSwap), 1, 10_000);
        uint256 outPerSwap = bound(uint256(rawOutPerSwap), 0, 20_000);
        uint256 expectedOut = outPerSwap * count;

        router.setExactInputSingleReturn(outPerSwap);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](count);
        for (uint256 i = 0; i < count; i++) {
            swaps[i] = IBatchSwapRouter.SwapParam({
                token: i % 2 == 0 ? tokenIn : tokenInAlt,
                fee: 500,
                sqrtPriceLimitX96: 0
            });
        }

        uint256 tokenInBefore = tokenIn.balanceOf(address(this));
        uint256 tokenInAltBefore = tokenInAlt.balanceOf(address(this));

        uint256 amountOut = batch.exactInput(tokenOut, amountInPerSwap, expectedOut, swaps);

        uint256 tokenInSwapCount = (count + 1) / 2;
        uint256 tokenInAltSwapCount = count - tokenInSwapCount;
        assertEq(amountOut, expectedOut);
        assertEq(tokenOut.balanceOf(address(this)), expectedOut);
        assertEq(tokenInBefore - tokenIn.balanceOf(address(this)), amountInPerSwap * tokenInSwapCount);
        assertEq(tokenInAltBefore - tokenInAlt.balanceOf(address(this)), amountInPerSwap * tokenInAltSwapCount);
    }

    function testFuzzExactInputRevertsWhenMinimumTooHigh(
        uint8 rawCount,
        uint96 rawAmountInPerSwap,
        uint96 rawOutPerSwap,
        uint96 rawExtraMinimum
    ) public {
        uint256 count = bound(uint256(rawCount), 1, 8);
        uint256 amountInPerSwap = bound(uint256(rawAmountInPerSwap), 1, 10_000);
        uint256 outPerSwap = bound(uint256(rawOutPerSwap), 0, 20_000);
        uint256 extraMinimum = bound(uint256(rawExtraMinimum), 1, 20_000);
        uint256 minimumTooHigh = (outPerSwap * count) + extraMinimum;

        router.setExactInputSingleReturn(outPerSwap);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](count);
        for (uint256 i = 0; i < count; i++) {
            swaps[i] = IBatchSwapRouter.SwapParam({
                token: i % 2 == 0 ? tokenIn : tokenInAlt,
                fee: 500,
                sqrtPriceLimitX96: 0
            });
        }

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactInput(tokenOut, amountInPerSwap, minimumTooHigh, swaps);
    }

    function testFuzzExactOutputTracksRemainingBudgetAndRefunds(
        uint8 rawCount,
        uint96 rawAmountOutPerSwap,
        uint96 rawCostPerSwap,
        uint96 rawSlack
    ) public {
        uint256 count = bound(uint256(rawCount), 1, 8);
        uint256 amountOutPerSwap = bound(uint256(rawAmountOutPerSwap), 1, 10_000);
        uint256 costPerSwap = bound(uint256(rawCostPerSwap), 0, 2_000);
        uint256 slack = bound(uint256(rawSlack), 0, 2_000);
        uint256 totalCost = costPerSwap * count;
        uint256 amountInTotalMax = totalCost + slack;

        router.setExactOutputSingleReturn(costPerSwap);
        router.setEnforceAmountInMaximumOnExactOutput(true);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](count);
        for (uint256 i = 0; i < count; i++) {
            swaps[i] = IBatchSwapRouter.SwapParam({
                token: i % 2 == 0 ? tokenOut : tokenOutAlt,
                fee: 500,
                sqrtPriceLimitX96: 0
            });
        }

        uint256 tokenInBefore = tokenIn.balanceOf(address(this));
        uint256 tokenOutBefore = tokenOut.balanceOf(address(this));
        uint256 tokenOutAltBefore = tokenOutAlt.balanceOf(address(this));

        uint256 amountIn = batch.exactOutput(tokenIn, amountOutPerSwap, amountInTotalMax, swaps);

        uint256 tokenOutSwapCount = (count + 1) / 2;
        uint256 tokenOutAltSwapCount = count - tokenOutSwapCount;
        assertEq(amountIn, totalCost);
        assertEq(tokenInBefore - tokenIn.balanceOf(address(this)), totalCost);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
        assertEq(tokenOut.balanceOf(address(this)) - tokenOutBefore, amountOutPerSwap * tokenOutSwapCount);
        assertEq(tokenOutAlt.balanceOf(address(this)) - tokenOutAltBefore, amountOutPerSwap * tokenOutAltSwapCount);
        assertEq(router.exactOutputCalls(), count);

        for (uint256 i = 0; i < count; i++) {
            assertEq(router.exactOutputAmountInMaximumHistory(i), amountInTotalMax - (costPerSwap * i));
        }
    }

    function testFuzzExactOutputRevertsWhenBudgetBelowTotalCost(
        uint8 rawCount,
        uint96 rawCostPerSwap,
        uint96 rawDeficit
    ) public {
        uint256 count = bound(uint256(rawCount), 1, 8);
        uint256 costPerSwap = bound(uint256(rawCostPerSwap), 1, 2_000);
        uint256 totalCost = costPerSwap * count;
        uint256 deficit = bound(uint256(rawDeficit), 1, totalCost);
        uint256 amountInTotalMax = totalCost - deficit;

        router.setExactOutputSingleReturn(costPerSwap);
        router.setEnforceAmountInMaximumOnExactOutput(true);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](count);
        for (uint256 i = 0; i < count; i++) {
            swaps[i] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});
        }

        vm.expectRevert(MockV3SwapRouter.AmountInMaximumExceeded.selector);
        batch.exactOutput(tokenIn, 1, amountInTotalMax, swaps);
    }
}
