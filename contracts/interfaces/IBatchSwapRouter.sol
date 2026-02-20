// SPDX-License-Identifier: MIT
//https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Router.sol
pragma solidity ^0.8.24;

import {IERC20} from "./IERC20.sol";

interface IBatchSwapRouter {
    struct SwapParam {
        IERC20 token;
        uint24 fee;
        uint160 sqrtPriceLimitX96;
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
    ) external returns (uint256 amountOut);

    /// @notice Batch buys equal amounts of tokenOuts for the same tokenIn.
    /// @param swaps Array of ExactOutputSingleParams for each swap.
    /// @param amountInMax Maximum total amount of input tokens to spend.
    /// @return amountIn Total amount of input tokens spent.
    /// @dev Caller must approve tokenInMax to this contract before calling.
    function exactOutput(
        IERC20 tokenIn,
        uint256 amountOut,
        uint256 amountInMax,
        SwapParam[] calldata swaps
    ) external returns (uint256 amountIn);
}