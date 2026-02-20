// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {BatchSwapRouter} from "../contracts/BatchSwapRouter.sol";
import {IBatchSwapRouter} from "../contracts/interfaces/IBatchSwapRouter.sol";
import {MockERC20, MockV3SwapRouter} from "./utils/BatchSwapRouterMocks.sol";

contract BatchSwapRouterTest is Test {
    MockERC20 internal tokenIn;
    MockERC20 internal tokenOut;
    MockERC20 internal tokenOutAlt;
    MockV3SwapRouter internal router;
    BatchSwapRouter internal batch;

    function setUp() public {
        tokenIn = new MockERC20();
        tokenOut = new MockERC20();
        tokenOutAlt = new MockERC20();
        router = new MockV3SwapRouter();
        batch = new BatchSwapRouter(address(router));

        tokenIn.mint(address(this), 1_000_000);
        tokenIn.approve(address(batch), type(uint256).max);
    }

    function testConstructorSetsImmutableRouter() public {
        assertEq(address(batch.router()), address(router));
    }

    function testExactInputSingleSwapSuccess() public {
        router.setExactInputSingleReturn(15);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenIn, fee: 500, sqrtPriceLimitX96: 77});

        uint256 out = batch.exactInput(tokenOut, 10, 0, swaps);

        assertEq(out, 15);
        assertEq(tokenOut.balanceOf(address(this)), 15);
        assertEq(router.exactInputCalls(), 1);
        (, , , , uint256 amountIn, , uint160 sqrtPriceLimitX96) = router.lastExactInput();
        assertEq(amountIn, 10);
        assertEq(sqrtPriceLimitX96, 77);
    }

    function testExactInputMultiSwapAggregatesOutput() public {
        router.setExactInputSingleReturn(6);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](2);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenIn, fee: 500, sqrtPriceLimitX96: 0});
        swaps[1] = IBatchSwapRouter.SwapParam({token: tokenIn, fee: 3_000, sqrtPriceLimitX96: 0});

        uint256 out = batch.exactInput(tokenOut, 10, 12, swaps);

        assertEq(out, 12);
        assertEq(tokenOut.balanceOf(address(this)), 12);
        assertEq(router.exactInputCalls(), 2);
    }

    function testExactInputEmptyArrayReturnsZeroAtMinZero() public {
        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](0);

        uint256 out = batch.exactInput(tokenOut, 10, 0, swaps);

        assertEq(out, 0);
        assertEq(router.exactInputCalls(), 0);
    }

    function testExactOutputSingleSwapSuccessWithRefund() public {
        router.setExactOutputSingleReturn(7);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 123});

        uint256 inSpent = batch.exactOutput(tokenIn, 4, 10, swaps);

        assertEq(inSpent, 7);
        assertEq(tokenOut.balanceOf(address(this)), 4);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 7);
        assertEq(router.exactOutputCalls(), 1);
        (, , , , , uint256 amountInMaximum, uint160 sqrtPriceLimitX96) = router.lastExactOutput();
        assertEq(amountInMaximum, 0);
        assertEq(sqrtPriceLimitX96, 123);
    }

    function testExactOutputMultiSwapAggregatesInputAndTransfersEachToken() public {
        router.setExactOutputSingleReturn(3);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](2);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});
        swaps[1] = IBatchSwapRouter.SwapParam({token: tokenOutAlt, fee: 3_000, sqrtPriceLimitX96: 0});

        uint256 inSpent = batch.exactOutput(tokenIn, 5, 10, swaps);

        assertEq(inSpent, 6);
        assertEq(tokenOut.balanceOf(address(this)), 5);
        assertEq(tokenOutAlt.balanceOf(address(this)), 5);
        assertEq(tokenIn.balanceOf(address(this)), 1_000_000 - 6);
        assertEq(router.exactOutputCalls(), 2);
    }

    function testExactOutputZeroRemainingSkipsRefundTransfer() public {
        router.setExactOutputSingleReturn(10);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});

        uint256 before = tokenIn.balanceOf(address(this));
        uint256 inSpent = batch.exactOutput(tokenIn, 1, 10, swaps);

        assertEq(inSpent, 10);
        assertEq(tokenIn.balanceOf(address(this)), before - 10);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
    }
}
