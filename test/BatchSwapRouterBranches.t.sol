// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {BatchSwapRouter} from "../contracts/BatchSwapRouter.sol";
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

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactInput(tokenOut, 1, 0, 500, 0, tokens);
    }

    function testExactInputRevertsOnApprovalFailed() public {
        tokenIn.setApproveResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.exactInput(tokenOut, 1, 0, 500, 0, tokens);
    }

    function testExactInputRevertsOnAggregateSlippage() public {
        router.setExactInputSingleReturn(3);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactInput(tokenOut, 1, 4, 500, 0, tokens);
    }

    function testExactInputIgnoresTokenOutTransferFlag() public {
        router.setExactInputSingleReturn(2);
        tokenOut.setTransferResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        uint256 amountOut = batch.exactInput(tokenOut, 1, 0, 500, 0, tokens);
        assertEq(amountOut, 2);
        assertEq(tokenOut.balanceOf(address(this)), 2);
    }

    function testExactInputEmptyArrayRevertsForPositiveMin() public {
        address[] memory tokens = new address[](0);

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactInput(tokenOut, 1, 1, 500, 0, tokens);
    }

    function testExactOutputRevertsOnInitialTransferFromFailed() public {
        tokenIn.setTransferFromResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactOutput(tokenIn, 1, 10, 500, 0, tokens);
    }

    function testExactOutputRevertsOnApprovalFailed() public {
        tokenIn.setApproveResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.exactOutput(tokenIn, 1, 10, 500, 0, tokens);
    }

    function testExactOutputIgnoresTokenOutTransferFlag() public {
        router.setExactOutputSingleReturn(2);
        tokenOut.setTransferResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        uint256 amountIn = batch.exactOutput(tokenIn, 1, 10, 500, 0, tokens);
        assertEq(amountIn, 2);
        assertEq(tokenOut.balanceOf(address(this)), 1);
    }

    function testExactOutputRevertsWhenAggregateSpentExceedsMax() public {
        router.setExactOutputSingleReturn(6);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactOutput(tokenIn, 1, 10, 500, 0, tokens);
    }

    function testExactOutputRevertsWhenPerSwapRemainingMaximumExceeded() public {
        router.setExactOutputSingleReturn(6);
        router.setEnforceAmountInMaximumOnExactOutput(true);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOut);

        vm.expectRevert(MockV3SwapRouter.AmountInMaximumExceeded.selector);
        batch.exactOutput(tokenIn, 1, 10, 500, 0, tokens);
    }

    function testExactOutputRevertsWhenRefundTransferFails() public {
        router.setExactOutputSingleReturn(5);
        tokenIn.setTransferResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactOutput(tokenIn, 1, 10, 500, 0, tokens);
    }

    function testExactOutputEmptyArrayRefundsAllInput() public {
        address[] memory tokens = new address[](0);

        uint256 before = tokenIn.balanceOf(address(this));
        uint256 spent = batch.exactOutput(tokenIn, 1, 10, 500, 0, tokens);

        assertEq(spent, 0);
        assertEq(tokenIn.balanceOf(address(this)), before);
    }
}
