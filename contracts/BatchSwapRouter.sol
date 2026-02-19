// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IV3SwapRouter} from "./interfaces/IV3SwapRouter.sol";
import {IERC20} from "./interfaces/IERC20.sol";

/// @title Batch Swap Router for Uniswap V3
/// @notice Batches Uniswap V3 swaps with aggregate slippage protection.
contract BatchSwapRouter {
    error SlippageExceeded();
    error ApprovalFailed();
    error TransferFailed();

    IV3SwapRouter public immutable router;

    constructor(address _router) {
        router = IV3SwapRouter(_router);
    }

    /// @notice Batch sells equal amounts of tokenIns for the same tokenOut.
    /// @param swaps Array of ExactInputSingleParams for each swap.
    /// @param min Minimum total amount of output tokens to receive.
    /// @return amount Total amount of output tokens received.
    /// @dev Caller must transfer tokenIn to this contract before calling.
    /// @dev To enforce only group slippage protection, set each swap's amountOutMinimum to zero.
    /// @dev tokenOut is assumed to be the same for all swaps in swaps.
    /// @dev tokenOut is sent to recipient specified in each swaps[i].
    function sell(
        IV3SwapRouter.ExactInputSingleParams[] calldata swaps,
        uint256 min
    ) external returns (uint256 amount) {
        for (uint i = 0; i < swaps.length; i++) {
            IERC20 tokenIn = IERC20(swaps[i].tokenIn);
            bool success = tokenIn.approve(address(router), swaps[i].amountIn);
            require(success, ApprovalFailed());
            // TRUSTED: external call to Uniswap V3 swap router
            amount += router.exactInputSingle(swaps[i]);
        }
        require(amount >= min, SlippageExceeded());
    }

    /// @notice Batch buys equal amounts of tokenOuts for the same tokenIn.
    /// @param swaps Array of ExactOutputSingleParams for each swap.
    /// @param max Maximum total amount of input tokens to spend.
    /// @return amount Total amount of input tokens spent.
    /// @dev Caller must transfer tokenInMax to this contract before calling.
    /// @dev To enforce only group slippage protection, set each swap's amountOutMinimum to zero.
    /// @dev tokenIn is assumed to be the same for all swaps in swaps.
    /// @dev tokenOut is sent to recipient specified in each swaps[i].
    function buy(
        IV3SwapRouter.ExactOutputSingleParams[] calldata swaps,
        uint256 max
    ) external returns (uint256 amount) {
        // tokenIn is same for all swaps, so approve once
        IERC20 tokenIn = IERC20(swaps[0].tokenIn);
        bool success = tokenIn.approve(address(router), max);
        require(success, ApprovalFailed());
        for (uint i = 0; i < swaps.length; i++) {
            // TRUSTED: external call to Uniswap V3 swap router
            amount += router.exactOutputSingle(swaps[i]);
        }
        require(amount <= max, SlippageExceeded());
        // Refund unused tokenIn to caller
        uint256 remaining = tokenIn.balanceOf(address(this));
        if (remaining > 0) {
            success = tokenIn.transfer(msg.sender, remaining);
            require(success, TransferFailed());
        }
    }
}
