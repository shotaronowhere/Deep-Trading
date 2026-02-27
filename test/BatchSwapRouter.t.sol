// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {BatchSwapRouter} from "../contracts/BatchSwapRouter.sol";
import {MockERC20, MockV3SwapRouter} from "./utils/BatchSwapRouterMocks.sol";

contract BatchSwapRouterTest is Test {
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

    function testConstructorSetsImmutableRouter() public {
        assertEq(address(batch.router()), address(router));
    }

    function testExactInputSingleSwapSuccess() public {
        router.setExactInputSingleReturn(15);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        uint256 out = batch.exactInput(tokens, 10, address(tokenOut), 0, 500, 77);

        assertEq(out, 15);
        assertEq(tokenOut.balanceOf(address(this)), 15);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
        assertEq(router.exactInputCalls(), 1);
        (address lastTokenIn, address lastTokenOut, uint24 fee, address recipient, uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96) =
            router.lastExactInput();
        assertEq(lastTokenIn, address(tokenIn));
        assertEq(lastTokenOut, address(tokenOut));
        assertEq(fee, 500);
        assertEq(recipient, address(this));
        assertEq(amountIn, 10);
        assertEq(amountOutMinimum, 0);
        assertEq(sqrtPriceLimitX96, 77);
    }

    function testExactInputMultiSwapAggregatesOutput() public {
        router.setExactInputSingleReturn(6);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenIn);
        tokens[1] = address(tokenIn);

        uint256 out = batch.exactInput(tokens, 10, address(tokenOut), 12, 500, 0);

        assertEq(out, 12);
        assertEq(tokenOut.balanceOf(address(this)), 12);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 20);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
        assertEq(router.exactInputCalls(), 2);
    }

    function testExactInputSupportsDifferentInputTokensPerSwap() public {
        router.setExactInputSingleReturn(5);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenIn);
        tokens[1] = address(tokenInAlt);

        uint256 out = batch.exactInput(tokens, 10, address(tokenOut), 10, 500, 0);

        assertEq(out, 10);
        assertEq(tokenOut.balanceOf(address(this)), 10);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 10);
        assertEq(tokenInAlt.balanceOf(address(this)), 1_000_000 - 10);
        assertEq(tokenIn.allowance(address(batch), address(router)), 0);
        assertEq(tokenInAlt.allowance(address(batch), address(router)), 0);
    }

    function testExactInputEmptyArrayReturnsZeroAtMinZero() public {
        address[] memory tokens = new address[](0);

        uint256 out = batch.exactInput(tokens, 10, address(tokenOut), 0, 500, 0);

        assertEq(out, 0);
        assertEq(router.exactInputCalls(), 0);
    }

    function testExactOutputSingleSwapSuccessWithRefund() public {
        router.setExactOutputSingleReturn(7);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        uint256 inSpent = batch.exactOutput(tokens, 4, address(tokenIn), 10, 500, 123);

        assertEq(inSpent, 7);
        assertEq(tokenOut.balanceOf(address(this)), 4);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 7);
        assertEq(router.exactOutputCalls(), 1);
        (address lastTokenIn, address lastTokenOut, uint24 fee, address recipient, uint256 outAmount, uint256 amountInMaximum, uint160 sqrtPriceLimitX96) =
            router.lastExactOutput();
        assertEq(lastTokenIn, address(tokenIn));
        assertEq(lastTokenOut, address(tokenOut));
        assertEq(fee, 500);
        assertEq(recipient, address(this));
        assertEq(outAmount, 4);
        assertEq(amountInMaximum, 10);
        assertEq(sqrtPriceLimitX96, 123);
    }

    function testExactOutputMultiSwapAggregatesInputAndTransfersEachToken() public {
        router.setExactOutputSingleReturn(3);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOutAlt);

        uint256 inSpent = batch.exactOutput(tokens, 5, address(tokenIn), 10, 500, 0);

        assertEq(inSpent, 6);
        assertEq(tokenOut.balanceOf(address(this)), 5);
        assertEq(tokenOutAlt.balanceOf(address(this)), 5);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 6);
        assertEq(router.exactOutputCalls(), 2);
    }

    function testExactOutputThreeSwapTightBudgetTracksRemainingMaximum() public {
        router.setExactOutputSingleReturn(3);
        router.setEnforceAmountInMaximumOnExactOutput(true);

        address[] memory tokens = new address[](3);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOutAlt);
        tokens[2] = address(tokenOut);

        uint256 inSpent = batch.exactOutput(tokens, 1, address(tokenIn), 9, 500, 0);

        assertEq(inSpent, 9);
        assertEq(router.exactOutputCalls(), 3);
        assertEq(router.exactOutputAmountInMaximumHistory(0), 9);
        assertEq(router.exactOutputAmountInMaximumHistory(1), 6);
        assertEq(router.exactOutputAmountInMaximumHistory(2), 3);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 9);
    }

    function testExactOutputTwoSwapCanSpendExactlyTotalMax() public {
        router.setExactOutputSingleReturn(5);
        router.setEnforceAmountInMaximumOnExactOutput(true);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOutAlt);

        uint256 inSpent = batch.exactOutput(tokens, 2, address(tokenIn), 10, 500, 0);

        assertEq(inSpent, 10);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 10);
        assertEq(router.exactOutputAmountInMaximumHistory(0), 10);
        assertEq(router.exactOutputAmountInMaximumHistory(1), 5);
    }

    function testExactOutputZeroRemainingSkipsRefundTransfer() public {
        router.setExactOutputSingleReturn(10);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        uint256 before = tokenIn.balanceOf(address(this));
        uint256 inSpent = batch.exactOutput(tokens, 1, address(tokenIn), 10, 500, 0);

        assertEq(inSpent, 10);
        assertEq(tokenIn.balanceOf(address(this)), before - 10);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
    }

    function testExactInputSupportsUnequalAmountsArray() public {
        router.setExactInputSingleReturn(5);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenIn);
        tokens[1] = address(tokenInAlt);

        uint256[] memory amountsIn = new uint256[](2);
        amountsIn[0] = 3;
        amountsIn[1] = 11;

        uint256 out = batch.exactInput(tokens, amountsIn, address(tokenOut), 10, 500, 0);

        assertEq(out, 10);
        assertEq(tokenOut.balanceOf(address(this)), 10);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 3);
        assertEq(tokenInAlt.balanceOf(address(this)), 1_000_000 - 11);
        assertEq(router.exactInputCalls(), 2);
    }

    function testExactOutputSupportsUnequalAmountsArray() public {
        router.setExactOutputSingleReturn(4);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOutAlt);

        uint256[] memory amountsOut = new uint256[](2);
        amountsOut[0] = 2;
        amountsOut[1] = 7;

        uint256 inSpent = batch.exactOutput(tokens, amountsOut, address(tokenIn), 9, 500, 0);

        assertEq(inSpent, 8);
        assertEq(tokenOut.balanceOf(address(this)), 2);
        assertEq(tokenOutAlt.balanceOf(address(this)), 7);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 8);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
        assertEq(router.exactOutputCalls(), 2);
    }
}
