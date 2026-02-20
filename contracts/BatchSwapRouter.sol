// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IV3SwapRouter} from "./interfaces/IV3SwapRouter.sol";
import {IERC20} from "./interfaces/IERC20.sol";
import {IBatchSwapRouter} from "./interfaces/IBatchSwapRouter.sol";

/// @title Batch Swap Router for Uniswap V3
/// @notice Batches Uniswap V3 swaps with aggregate slippage protection.
/// @dev Any tokens accidentally sent to this contract may be drained by anyone, so only approve the exact amount needed for swaps and never send tokens directly to this contract.
contract BatchSwapRouter is IBatchSwapRouter {
    error SlippageExceeded();
    error ApprovalFailed();
    error TransferFailed();

    IV3SwapRouter public immutable router;

    constructor(address _router) {
        router = IV3SwapRouter(_router);
    }

    /// @notice Batch sells equal amounts of tokenIns for tokenOut.
    /// @dev No individual slippage tolerance per swap; only aggregate slippage protection for the total amountOut. Caller must approve total amountIn for all swaps to this contract before calling.
    /// @dev Caller must approve tokenIn to this contract before calling.
    /// @param tokenOut The token to receive from all swaps.
    /// @param amountIn Amount of input tokens to swap for each swap.
    /// @param amountOutTotalMinimum Minimum total amount of output tokens to receive.
    /// @param swaps Array of ExactInputSingleParams for each swap.
    /// @return amountOut Total amount of output tokens received.
    function exactInput(
        IERC20 tokenOut,
        uint256 amountIn,
        uint256 amountOutTotalMinimum,
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
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: swaps[i].sqrtPriceLimitX96
                })
            );
        }
        require(amountOut >= amountOutTotalMinimum, SlippageExceeded());
    }

    /// @notice Batch buys equal amounts of tokenOuts for the same tokenIn.
    /// @dev Caller must approve amountInTotalMax to this contract before calling.
    /// @param tokenIn The token to spend for all swaps.
    /// @param amountOut Amount of output tokens to receive for each swap.
    /// @param amountInTotalMax Maximum total amount of input tokens to spend.
    /// @param swaps Array of ExactOutputSingleParams for each swap.
    /// @return amountIn Total amount of input tokens spent.
    function exactOutput(
        IERC20 tokenIn,
        uint256 amountOut,
        uint256 amountInTotalMax,
        SwapParam[] calldata swaps
    ) external returns (uint256 amountIn) {
        bool success = tokenIn.transferFrom(msg.sender, address(this), amountInTotalMax);
        require(success, TransferFailed());

        // tokenIn is same for all swaps, so approve once
        success = tokenIn.approve(address(router), amountInTotalMax);
        require(success, ApprovalFailed());


        for (uint i = 0; i < swaps.length; i++) {
            uint256 amountInRemaining = amountInTotalMax - amountIn; // remaining max for this swap
            amountIn += router.exactOutputSingle(
                IV3SwapRouter.ExactOutputSingleParams({
                    tokenIn: address(tokenIn),
                    tokenOut: address(swaps[i].token), // same tokenOut for all swaps
                    fee: swaps[i].fee,
                    recipient: msg.sender,
                    amountOut: amountOut, // same amountOut for all swaps
                    amountInMaximum: amountInRemaining, // per-swap ceiling uses remaining max; aggregate checked at the end
                    sqrtPriceLimitX96: swaps[i].sqrtPriceLimitX96
                })
            );
        }
        require(amountIn <= amountInTotalMax, SlippageExceeded());
        // Refund unused tokenIn to caller
        uint256 remaining = tokenIn.balanceOf(address(this));
        if (remaining > 0) {
            success = tokenIn.transfer(msg.sender, remaining);
            require(success, TransferFailed());
        }
    }
}
