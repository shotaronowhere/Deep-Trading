// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "../../contracts/interfaces/IERC20.sol";
import {IV3SwapRouter} from "../../contracts/interfaces/IV3SwapRouter.sol";

interface IMintableToken {
    function mint(address to, uint256 amount) external;
}

contract MockERC20 is IERC20, IMintableToken {
    mapping(address => uint256) public override balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    bool public approveResult = true;
    bool public transferResult = true;
    bool public transferFromResult = true;

    function setApproveResult(bool value) external {
        approveResult = value;
    }

    function setTransferResult(bool value) external {
        transferResult = value;
    }

    function setTransferFromResult(bool value) external {
        transferFromResult = value;
    }

    function mint(address to, uint256 amount) external override {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external override returns (bool) {
        allowance[msg.sender][spender] = amount;
        return approveResult;
    }

    function transfer(address to, uint256 amount) external override returns (bool) {
        if (!transferResult || balanceOf[msg.sender] < amount) {
            return false;
        }

        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external override returns (bool) {
        if (!transferFromResult) {
            return false;
        }

        uint256 allowed = allowance[from][msg.sender];
        if (allowed < amount || balanceOf[from] < amount) {
            return false;
        }

        allowance[from][msg.sender] = allowed - amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract MockV3SwapRouter is IV3SwapRouter {
    error AmountInMaximumExceeded();

    uint256 public exactInputSingleReturn;
    uint256 public exactOutputSingleReturn;

    bool public mintTokenOutOnExactInput = true;
    bool public mintTokenOutOnExactOutput = true;
    bool public pullTokenInOnExactOutput = true;
    bool public enforceAmountInMaximumOnExactOutput;

    uint256 public exactInputCalls;
    uint256 public exactOutputCalls;
    uint256[] public exactOutputAmountInMaximumHistory;

    ExactInputSingleParams public lastExactInput;
    ExactOutputSingleParams public lastExactOutput;

    function setExactInputSingleReturn(uint256 value) external {
        exactInputSingleReturn = value;
    }

    function setExactOutputSingleReturn(uint256 value) external {
        exactOutputSingleReturn = value;
    }

    function setMintTokenOutOnExactInput(bool value) external {
        mintTokenOutOnExactInput = value;
    }

    function setMintTokenOutOnExactOutput(bool value) external {
        mintTokenOutOnExactOutput = value;
    }

    function setPullTokenInOnExactOutput(bool value) external {
        pullTokenInOnExactOutput = value;
    }

    function setEnforceAmountInMaximumOnExactOutput(bool value) external {
        enforceAmountInMaximumOnExactOutput = value;
    }

    function exactInputSingle(ExactInputSingleParams calldata params) external payable override returns (uint256 amountOut) {
        exactInputCalls += 1;
        lastExactInput = params;
        amountOut = exactInputSingleReturn;

        if (mintTokenOutOnExactInput && amountOut > 0) {
            IMintableToken(params.tokenOut).mint(params.recipient, amountOut);
        }
    }

    function exactOutputSingle(ExactOutputSingleParams calldata params)
        external
        payable
        override
        returns (uint256 amountIn)
    {
        exactOutputCalls += 1;
        lastExactOutput = params;
        amountIn = exactOutputSingleReturn;
        exactOutputAmountInMaximumHistory.push(params.amountInMaximum);

        if (enforceAmountInMaximumOnExactOutput && amountIn > params.amountInMaximum) {
            revert AmountInMaximumExceeded();
        }

        if (pullTokenInOnExactOutput && amountIn > 0) {
            IERC20(params.tokenIn).transferFrom(msg.sender, address(this), amountIn);
        }

        if (mintTokenOutOnExactOutput && params.amountOut > 0) {
            IMintableToken(params.tokenOut).mint(params.recipient, params.amountOut);
        }
    }
}
