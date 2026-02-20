// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IV3SwapRouter} from "./interfaces/IV3SwapRouter.sol";
import {IERC20} from "./interfaces/IERC20.sol";
import {IBatchSwapRouter} from "./interfaces/IBatchSwapRouter.sol";

/// @title Batch Swap Router for Uniswap V3
/// @notice Batches Uniswap V3 swaps with aggregate slippage protection.
contract BatchSwapRouter is IBatchSwapRouter {
    error SlippageExceeded();
    error ApprovalFailed();
    error TransferFailed();

    IV3SwapRouter public immutable router;

    constructor(address _router) {
        router = IV3SwapRouter(_router);
    }

    /// @param swaps Array of ExactInputSingleParams for each swap.
    /// @param amountOutMin Minimum total amount of output tokens to receive.
    /// @return amountOut Total amount of output tokens received.
    /// @dev Caller must approve tokenIn to this contract before calling.
    function exactInput(
        IERC20 tokenOut,
        uint256 amountIn,
        uint256 amountOutMin,
        SwapParam[] calldata swaps
    ) external returns (uint256 amountOut) {
        for (uint i = 0; i < swaps.length; i++) {
            IERC20 tokenIn = swaps[i].token;
            bool success = tokenIn.transferFrom(msg.sender, address(this), amountIn);
            require(success, TransferFailed());
            success = tokenIn.approve(address(router), amountIn);
            require(success, ApprovalFailed());
            amountOut += router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: address(tokenIn),
                    tokenOut: address(tokenOut), // same tokenOut for all swaps
                    fee: swaps[i].fee,
                    recipient: msg.sender,
                    amountIn: amountIn, // same amountIn for all swaps
                    amountOutMinimum: 0, // No psqrtPriceLimitX96er-swap slippage protection; aggregate checked at the end
                    sqrtPriceLimitX96: swaps[i].sqrtPriceLimitX96
                })
            );
        }
        require(amountOut >= amountOutMin, SlippageExceeded());
    }

    /// @notice Batch buys equal amounts of tokenOuts for the same tokenIn.
    /// @param swaps Array of ExactOutputSingleParams for each swap.
    /// @param amountInMax Maximum total amount of input tokens to spend.
    /// @return amountIn Total amount of input tokens spent.
    /// @dev Caller must approve amountInMax to this contract before calling.
    function exactOutput(
        IERC20 tokenIn,
        uint256 amountOut,
        uint256 amountInMax,
        SwapParam[] calldata swaps
    ) external returns (uint256 amountIn) {
        bool success = tokenIn.transferFrom(msg.sender, address(this), amountInMax);
        require(success, TransferFailed());

        // tokenIn is same for all swaps, so approve once
        success = tokenIn.approve(address(router), amountInMax);
        require(success, ApprovalFailed());

        for (uint i = 0; i < swaps.length; i++) {
            amountIn += router.exactOutputSingle(
                IV3SwapRouter.ExactOutputSingleParams({
                    tokenIn: address(tokenIn),
                    tokenOut: address(swaps[i].token), // same tokenOut for all swaps
                    fee: swaps[i].fee,
                    recipient: msg.sender,
                    amountOut: amountOut, // same amountOut for all swaps
                    amountInMaximum: 0, // No per-swap slippage protection; aggregate checked at the end
                    sqrtPriceLimitX96: swaps[i].sqrtPriceLimitX96
                })
            );
        }
        require(amountIn <= amountInMax, SlippageExceeded());
        // Refund unused tokenIn to caller
        uint256 remaining = tokenIn.balanceOf(address(this));
        if (remaining > 0) {
            success = tokenIn.transfer(msg.sender, remaining);
            require(success, TransferFailed());
        }
    }
}
