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
        batch.exactInput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenOut), 0, 500);
    }

    function testExactInputRevertsOnApprovalFailed() public {
        tokenIn.setApproveResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.exactInput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenOut), 0, 500);
    }

    function testExactInputRevertsOnAggregateSlippage() public {
        router.setExactInputSingleReturn(3);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactInput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenOut), 4, 500);
    }

    function testExactInputBatchRouterDoesNotDeliverTokenOut() public {
        router.setExactInputSingleReturn(2);
        router.setMintTokenOutOnExactInput(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenIn);

        uint256 amountOut =
            batch.exactInput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenOut), 0, 500);
        assertEq(amountOut, 2);
        assertEq(tokenOut.balanceOf(address(this)), 0); // delivery is the router's responsibility
    }

    function testExactInputEmptyArrayRevertsForPositiveMin() public {
        address[] memory tokens = new address[](0);

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactInput(tokens, _filledUint256Array(0, 1), _filledUint160Array(0, 0), address(tokenOut), 1, 500);
    }

    function testExactOutputRevertsOnInitialTransferFromFailed() public {
        tokenIn.setTransferFromResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactOutput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenIn), 10, 500);
    }

    function testExactOutputRevertsOnApprovalFailed() public {
        tokenIn.setApproveResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.exactOutput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenIn), 10, 500);
    }

    function testExactOutputBatchRouterDoesNotDeliverTokenOut() public {
        router.setExactOutputSingleReturn(2);
        router.setMintTokenOutOnExactOutput(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        uint256 amountIn =
            batch.exactOutput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenIn), 10, 500);
        assertEq(amountIn, 2);
        assertEq(tokenOut.balanceOf(address(this)), 0); // delivery is the router's responsibility
    }

    function testExactOutputRevertsWhenAggregateSpentExceedsMax() public {
        router.setExactOutputSingleReturn(6);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.exactOutput(tokens, _filledUint256Array(2, 1), _filledUint160Array(2, 0), address(tokenIn), 10, 500);
    }

    function testExactOutputRevertsWhenPerSwapRemainingMaximumExceeded() public {
        router.setExactOutputSingleReturn(6);
        router.setEnforceAmountInMaximumOnExactOutput(true);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOut);

        vm.expectRevert(MockV3SwapRouter.AmountInMaximumExceeded.selector);
        batch.exactOutput(tokens, _filledUint256Array(2, 1), _filledUint160Array(2, 0), address(tokenIn), 10, 500);
    }

    function testExactOutputRevertsWhenRefundTransferFails() public {
        router.setExactOutputSingleReturn(5);
        tokenIn.setTransferResult(false);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.exactOutput(tokens, _filledUint256Array(1, 1), _filledUint160Array(1, 0), address(tokenIn), 10, 500);
    }

    function testExactOutputEmptyArrayRefundsAllInput() public {
        address[] memory tokens = new address[](0);

        uint256 before = tokenIn.balanceOf(address(this));
        uint256 spent =
            batch.exactOutput(tokens, _filledUint256Array(0, 1), _filledUint160Array(0, 0), address(tokenIn), 10, 500);

        assertEq(spent, 0);
        assertEq(tokenIn.balanceOf(address(this)), before);
        assertEq(tokenIn.balanceOf(address(batch)), 0);
        assertEq(router.exactOutputCalls(), 0);
    }

    function testExactInputArrayAmountsRevertsOnLengthMismatch() public {
        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenIn);
        tokens[1] = address(tokenIn);

        uint256[] memory amountsIn = new uint256[](1);
        amountsIn[0] = 1;

        vm.expectRevert(BatchSwapRouter.InvalidArrayLength.selector);
        batch.exactInput(tokens, amountsIn, _filledUint160Array(2, 0), address(tokenOut), 0, 500);
    }

    function testExactOutputArrayAmountsRevertsOnLengthMismatch() public {
        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOut);

        uint256[] memory amountsOut = new uint256[](1);
        amountsOut[0] = 1;

        vm.expectRevert(BatchSwapRouter.InvalidArrayLength.selector);
        batch.exactOutput(tokens, amountsOut, _filledUint160Array(2, 0), address(tokenIn), 10, 500);
    }

    function testWaterfallBuyRevertsOnLengthMismatch() public {
        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenOut);
        tokens[1] = address(tokenOut);

        vm.expectRevert(BatchSwapRouter.InvalidArrayLength.selector);
        batch.waterfallBuy(tokens, _filledUint160Array(1, 0), address(tokenIn), 10, 500);
    }

    function _filledUint256Array(uint256 length, uint256 value) internal pure returns (uint256[] memory values) {
        values = new uint256[](length);
        for (uint256 i = 0; i < length; i++) {
            values[i] = value;
        }
    }

    function _filledUint160Array(uint256 length, uint160 value) internal pure returns (uint160[] memory values) {
        values = new uint160[](length);
        for (uint256 i = 0; i < length; i++) {
            values[i] = value;
        }
    }
}
