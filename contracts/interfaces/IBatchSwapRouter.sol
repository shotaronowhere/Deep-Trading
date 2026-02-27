// SPDX-License-Identifier: MIT
//https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Router.sol
pragma solidity ^0.8.24;

import {IERC20} from "./IERC20.sol";

interface IBatchSwapRouter {
    /// @notice Batch sells equal amounts of tokenIns for tokenOut.
    /// @dev No individual slippage tolerance per swap; only aggregate slippage protection for the total amountOut. Caller must approve total amountIn for all swaps to this contract before calling.
    /// @dev Caller must approve tokenIn to this contract before calling.
    /// @param _tokenIns Array of tokens for each swap.
    /// @param _amountIn Amount of input tokens to swap for each swap.
    /// @param _tokenOut The token to receive from all swaps.
    /// @param _amountOutTotalMinimum Minimum total amount of output tokens to receive.
    /// @param _fee The Uniswap fee tier to use for all swaps.
    /// @param _sqrtPriceLimitX96 The sqrt price limit for all swaps; 0 for no limit.
    /// @return amountOut Total amount of output tokens received.
    function exactInput(
        address[] memory _tokenIns,
        uint256 _amountIn,
        address _tokenOut,
        uint256 _amountOutTotalMinimum,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountOut);

    /// @notice Batch sells equal amounts of tokenIns for tokenOut.
    /// @dev No individual slippage tolerance per swap; only aggregate slippage protection for the total amountOut. Caller must approve total amountIn for all swaps to this contract before calling.
    /// @dev Caller must approve tokenIn to this contract before calling.
    /// @param _tokenIns Array of tokens for each swap.
    /// @param _amountIn Amount of input tokens to swap for each swap.
    /// @param _tokenOut The token to receive from all swaps.
    /// @param _amountOutTotalMinimum Minimum total amount of output tokens to receive.
    /// @param _fee The Uniswap fee tier to use for all swaps.
    /// @param _sqrtPriceLimitX96 The sqrt price limit for all swaps; 0 for no limit.
    /// @return amountOut Total amount of output tokens received.
    function exactInput(
        address[] memory _tokenIns,
        uint256[] memory _amountIn,
        address _tokenOut,
        uint256 _amountOutTotalMinimum,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountOut);

    /// @notice Batch buys equal amounts of tokenOuts for the same tokenIn.
    /// @dev Caller must approve amountInTotalMax to this contract before calling.
    /// @param _tokenOuts Array of tokens for each swap.
    /// @param _amountOut Amount of output tokens to receive for each swap.
    /// @param _tokenIn The token to spend for all swaps.
    /// @param _amountInTotalMax Maximum total amount of input tokens to spend.
    /// @param _fee The Uniswap fee tier to use for all swaps.
    /// @param _sqrtPriceLimitX96 The sqrt price limit for all swaps; 0 for no limit.
    /// @return amountIn Total amount of input tokens spent.
    function exactOutput(
        address[] memory _tokenOuts,
        uint256 _amountOut,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountIn);

    /// @notice Batch buys equal amounts of tokenOuts for the same tokenIn.
    /// @dev Caller must approve amountInTotalMax to this contract before calling.
    /// @param _tokenOuts Array of tokens for each swap.
    /// @param _amountOut Amount of output tokens to receive for each swap.
    /// @param _tokenIn The token to spend for all swaps.
    /// @param _amountInTotalMax Maximum total amount of input tokens to spend.
    /// @param _fee The Uniswap fee tier to use for all swaps.
    /// @param _sqrtPriceLimitX96 The sqrt price limit for all swaps; 0 for no limit.
    /// @return amountIn Total amount of input tokens spent.
    function exactOutput(
        address[] memory _tokenOuts,
        uint256[] memory _amountOut,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountIn);
}