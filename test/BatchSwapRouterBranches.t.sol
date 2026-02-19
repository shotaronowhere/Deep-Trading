// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {BatchSwapRouter, IV3SwapRouter, IERC20} from "../contracts/BatchSwapRouter.sol";

interface IERC20Extended is IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function mint(address to, uint256 amount) external;
}

contract ToggleERC20 is IERC20Extended {
    mapping(address => uint256) public override balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    bool public approveResult = true;
    bool public transferResult = true;

    function setApproveResult(bool value) external {
        approveResult = value;
    }

    function setTransferResult(bool value) external {
        transferResult = value;
    }

    function mint(address to, uint256 amount) external override {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external override returns (bool) {
        allowance[msg.sender][spender] = amount;
        return approveResult;
    }

    function transfer(address to, uint256 amount) external override returns (bool) {
        if (!transferResult) {
            return false;
        }
        require(balanceOf[msg.sender] >= amount, "insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external override returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        require(allowed >= amount, "insufficient allowance");
        require(balanceOf[from] >= amount, "insufficient balance");
        allowance[from][msg.sender] = allowed - amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract MockV3SwapRouterForBranches is IV3SwapRouter {
    uint256 public exactInputSingleReturn;
    uint256 public exactOutputSingleReturn;
    bool public pullTokenInOnExactOutput;

    function setExactInputSingleReturn(uint256 value) external {
        exactInputSingleReturn = value;
    }

    function setExactOutputSingleReturn(uint256 value) external {
        exactOutputSingleReturn = value;
    }

    function setPullTokenInOnExactOutput(bool value) external {
        pullTokenInOnExactOutput = value;
    }

    function exactInputSingle(ExactInputSingleParams calldata) external payable override returns (uint256 amountOut) {
        amountOut = exactInputSingleReturn;
    }

    function exactOutputSingle(ExactOutputSingleParams calldata params)
        external
        payable
        override
        returns (uint256 amountIn)
    {
        if (pullTokenInOnExactOutput && exactOutputSingleReturn > 0) {
            IERC20Extended(params.tokenIn).transferFrom(msg.sender, address(this), exactOutputSingleReturn);
        }
        amountIn = exactOutputSingleReturn;
    }
}

contract BatchSwapRouterBranchTest is Test {
    ToggleERC20 private token;
    ToggleERC20 private tokenOut;
    MockV3SwapRouterForBranches private router;
    BatchSwapRouter private batch;

    function setUp() public {
        token = new ToggleERC20();
        tokenOut = new ToggleERC20();
        router = new MockV3SwapRouterForBranches();
        batch = new BatchSwapRouter(address(router));
    }

    function testSellRevertsOnApprovalFailed() public {
        token.setApproveResult(false);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(token),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountIn: 1,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.sell(swaps, 0);
    }

    function testBuyRevertsOnApprovalFailed() public {
        token.setApproveResult(false);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(token),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 1,
            amountInMaximum: 10,
            sqrtPriceLimitX96: 0
        });

        vm.expectRevert(BatchSwapRouter.ApprovalFailed.selector);
        batch.buy(swaps, 10);
    }

    // This mock-only test covers BatchSwapRouter's defense-in-depth aggregate max check.
    // With a standard router/token flow, per-swap amountInMaximum usually prevents this path.
    function testBuyRevertsWhenAggregateAmountExceedsMax_DefenseInDepthMockPath() public {
        router.setExactOutputSingleReturn(7);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](2);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(token),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 1,
            amountInMaximum: 10,
            sqrtPriceLimitX96: 0
        });
        swaps[1] = swaps[0];

        vm.expectRevert(BatchSwapRouter.SlippageExceeded.selector);
        batch.buy(swaps, 10);
    }

    function testBuySkipsRefundWhenRemainingIsZero() public {
        uint256 spent = 10;
        token.mint(address(batch), spent);
        router.setExactOutputSingleReturn(spent);
        router.setPullTokenInOnExactOutput(true);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(token),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 1,
            amountInMaximum: spent,
            sqrtPriceLimitX96: 0
        });

        uint256 amount = batch.buy(swaps, spent);
        assertEq(amount, spent);
        assertEq(token.balanceOf(address(batch)), 0);
        assertEq(token.balanceOf(address(this)), 0);
    }

    function testBuyRevertsOnTransferFailedRefund() public {
        token.mint(address(batch), 10);
        token.setTransferResult(false);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(token),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 1,
            amountInMaximum: 10,
            sqrtPriceLimitX96: 0
        });

        vm.expectRevert(BatchSwapRouter.TransferFailed.selector);
        batch.buy(swaps, 10);
    }
}
