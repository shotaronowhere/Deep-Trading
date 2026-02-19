// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {BatchRouter, IV3SwapRouter} from "../contracts/BatchRouter.sol";
import {UniswapV3Fixture} from "./utils/UniswapV3Fixture.sol";

contract BatchRouterTest is Test, UniswapV3Fixture {
    function setUp() public {
        _deployFixture();
    }

    function testDirectAdapterSwapSmoke() public {
        uint256 amountIn = 1 ether;
        tokenIn.approve(address(v3Adapter), amountIn);

        IV3SwapRouter.ExactInputSingleParams memory params = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountIn: amountIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        uint256 outBefore = tokenOut.balanceOf(address(this));
        uint256 amountOut = v3Adapter.exactInputSingle(params);

        assertEq(tokenOut.balanceOf(address(this)) - outBefore, amountOut);
        assertGt(amountOut, 0);
    }

    function testSellSingleSwap() public {
        uint256 amountIn = 2 ether;
        tokenIn.transfer(address(batchRouter), amountIn);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountIn: amountIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        uint256 outBefore = tokenOut.balanceOf(address(this));
        uint256 amountOut = batchRouter.sell(swaps, 0);

        assertEq(tokenOut.balanceOf(address(this)) - outBefore, amountOut);
        assertEq(tokenIn.balanceOf(address(batchRouter)), 0);
        assertGt(amountOut, 0);
    }

    function testSellMultiSwapAggregatesAmountOut() public {
        uint256 firstIn = 1 ether;
        uint256 secondIn = 3 ether;
        tokenIn.transfer(address(batchRouter), firstIn + secondIn);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](2);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountIn: firstIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });
        swaps[1] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_MEDIUM,
            recipient: address(this),
            amountIn: secondIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        uint256 outBefore = tokenOut.balanceOf(address(this));
        uint256 amountOut = batchRouter.sell(swaps, 0);

        assertEq(tokenOut.balanceOf(address(this)) - outBefore, amountOut);
        assertEq(tokenIn.balanceOf(address(batchRouter)), 0);
        assertGt(amountOut, 0);
    }

    function testSellRevertsWhenAggregateMinNotMet() public {
        uint256 amountIn = 1 ether;
        tokenIn.transfer(address(batchRouter), amountIn);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountIn: amountIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        vm.expectRevert(BatchRouter.SlippageExceeded.selector);
        batchRouter.sell(swaps, type(uint256).max);
    }

    function testSellRevertsWhenPerSwapAmountOutMinimumNotMet() public {
        uint256 amountIn = 1 ether;
        tokenIn.transfer(address(batchRouter), amountIn);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountIn: amountIn,
            amountOutMinimum: amountIn,
            sqrtPriceLimitX96: 0
        });

        vm.expectRevert(bytes("Too little received"));
        batchRouter.sell(swaps, 0);
    }

    function testBuySingleSwapRefundsUnusedInput() public {
        uint256 maxIn = 100 ether;
        uint256 amountOutTarget = 0.01 ether;

        uint256 tokenInBefore = tokenIn.balanceOf(address(this));
        uint256 tokenOutBefore = tokenOut.balanceOf(address(this));

        tokenIn.transfer(address(batchRouter), maxIn);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountOut: amountOutTarget,
            amountInMaximum: maxIn,
            sqrtPriceLimitX96: 0
        });

        uint256 spent = batchRouter.buy(swaps, maxIn);

        uint256 tokenInAfter = tokenIn.balanceOf(address(this));
        uint256 tokenOutAfter = tokenOut.balanceOf(address(this));

        assertEq(tokenInBefore - tokenInAfter, spent);
        assertEq(tokenOutAfter - tokenOutBefore, amountOutTarget);
        assertEq(tokenIn.balanceOf(address(batchRouter)), 0);
        assertGt(maxIn, spent);
    }

    function testBuyMultiSwapAggregatesAmountInAndRefunds() public {
        uint256 maxIn = 500 ether;

        uint256 tokenInBefore = tokenIn.balanceOf(address(this));
        uint256 tokenOutBefore = tokenOut.balanceOf(address(this));

        tokenIn.transfer(address(batchRouter), maxIn);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](2);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountOut: 0.01 ether,
            amountInMaximum: 250 ether,
            sqrtPriceLimitX96: 0
        });
        swaps[1] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_MEDIUM,
            recipient: address(this),
            amountOut: 0.01 ether,
            amountInMaximum: 250 ether,
            sqrtPriceLimitX96: 0
        });

        uint256 spent = batchRouter.buy(swaps, maxIn);

        uint256 tokenInAfter = tokenIn.balanceOf(address(this));
        uint256 tokenOutAfter = tokenOut.balanceOf(address(this));

        assertEq(tokenInBefore - tokenInAfter, spent);
        assertEq(tokenOutAfter - tokenOutBefore, 0.02 ether);
        assertEq(tokenIn.balanceOf(address(batchRouter)), 0);
        assertGt(maxIn, spent);
    }

    function testBuyRevertsForEmptySwapArray() public {
        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](0);

        vm.expectRevert(stdError.indexOOBError);
        batchRouter.buy(swaps, 0);
    }

    function testSellEmptyArrayReturnsZeroWhenMinZero() public {
        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](0);

        uint256 amountOut = batchRouter.sell(swaps, 0);
        assertEq(amountOut, 0);
    }

    function testSellEmptyArrayRevertsWhenMinPositive() public {
        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](0);

        vm.expectRevert(BatchRouter.SlippageExceeded.selector);
        batchRouter.sell(swaps, 1);
    }

    function testSellSupportsDifferentTokenOutPerSwapByContractDesign() public {
        uint256 amountInFirst = 1 ether;
        uint256 amountInSecond = 1 ether;
        tokenIn.transfer(address(batchRouter), amountInFirst + amountInSecond);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](2);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: FEE_LOW,
            recipient: address(this),
            amountIn: amountInFirst,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });
        swaps[1] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOutAlt),
            fee: FEE_LOW,
            recipient: address(this),
            amountIn: amountInSecond,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        uint256 outBefore = tokenOut.balanceOf(address(this));
        uint256 outAltBefore = tokenOutAlt.balanceOf(address(this));

        uint256 amountOut = batchRouter.sell(swaps, 0);

        uint256 outDelta = tokenOut.balanceOf(address(this)) - outBefore;
        uint256 outAltDelta = tokenOutAlt.balanceOf(address(this)) - outAltBefore;

        assertEq(outDelta + outAltDelta, amountOut);
        assertGt(outDelta, 0);
        assertGt(outAltDelta, 0);
    }
}
