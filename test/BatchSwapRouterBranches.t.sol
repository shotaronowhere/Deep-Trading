// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {BatchSwapRouter} from "../contracts/BatchSwapRouter.sol";
import {IBatchSwapRouter} from "../contracts/interfaces/IBatchSwapRouter.sol";
import {MockERC20, MockV3SwapRouter} from "./utils/BatchSwapRouterMocks.sol";

contract BatchSwapRouterBranchTest is Test {
    MockERC20 private tokenIn;
    MockERC20 private tokenOut;
    MockV3SwapRouter private router;
    BatchSwapRouter private batch;

    function setUp() public {
        tokenIn = new MockERC20();
        tokenOut = new MockERC20();
        router = new MockV3SwapRouter();
        batch = new BatchSwapRouter(address(router));

        tokenIn.mint(address(this), 1_000);
        tokenIn.approve(address(batch), type(uint256).max);
    }

    function testExactInputRevertsOnTransferFromFailed() public {
        tokenIn.setTransferFromResult(false);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenIn, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactInput(tokenOut, 1, 0, swaps);
    }

    function testExactInputRevertsOnApprovalFailed() public {
        tokenIn.setApproveResult(false);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenIn, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.exactInput(tokenOut, 1, 0, swaps);
    }

    function testExactInputRevertsOnAggregateSlippage() public {
        router.setExactInputSingleReturn(3);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenIn, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactInput(tokenOut, 1, 4, swaps);
    }

    function testExactInputRevertsOnFinalTransferFailed() public {
        router.setExactInputSingleReturn(2);
        tokenOut.setTransferResult(false);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenIn, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactInput(tokenOut, 1, 0, swaps);
    }

    function testExactInputEmptyArrayRevertsForPositiveMin() public {
        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](0);

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactInput(tokenOut, 1, 1, swaps);
    }

    function testExactOutputRevertsOnInitialTransferFromFailed() public {
        tokenIn.setTransferFromResult(false);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactOutput(tokenIn, 1, 10, swaps);
    }

    function testExactOutputRevertsOnApprovalFailed() public {
        tokenIn.setApproveResult(false);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.exactOutput(tokenIn, 1, 10, swaps);
    }

    function testExactOutputRevertsOnTokenOutTransferFailed() public {
        router.setExactOutputSingleReturn(2);
        tokenOut.setTransferResult(false);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactOutput(tokenIn, 1, 10, swaps);
    }

    function testExactOutputRevertsWhenAggregateSpentExceedsMax() public {
        router.setExactOutputSingleReturn(6);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](2);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});
        swaps[1] = swaps[0];

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactOutput(tokenIn, 1, 10, swaps);
    }

    function testExactOutputRevertsWhenRefundTransferFails() public {
        router.setExactOutputSingleReturn(5);
        tokenIn.setTransferResult(false);

        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](1);
        swaps[0] = IBatchSwapRouter.SwapParam({token: tokenOut, fee: 500, sqrtPriceLimitX96: 0});

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactOutput(tokenIn, 1, 10, swaps);
    }

    function testExactOutputEmptyArrayRefundsAllInput() public {
        IBatchSwapRouter.SwapParam[] memory swaps = new IBatchSwapRouter.SwapParam[](0);

        uint256 before = tokenIn.balanceOf(address(this));
        uint256 spent = batch.exactOutput(tokenIn, 1, 10, swaps);

        assertEq(spent, 0);
        assertEq(tokenIn.balanceOf(address(this)), before);
    }
}
