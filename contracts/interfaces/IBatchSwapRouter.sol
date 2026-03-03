// SPDX-License-Identifier: MIT
//https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Router.sol
pragma solidity ^0.8.24;

import {IERC20} from "./IERC20.sol";

interface IBatchSwapRouter {
    /// @notice Batch sells tokenIns for tokenOut using per-leg input amounts and per-leg price limits.
    /// @dev Unexecuted input from a price-limited partial fill is refunded to the caller after each leg.
    /// @dev Caller must approve each tokenIn amount to this contract before calling.
    /// @param _tokenIns Array of input tokens to sell.
    /// @param _amountIns Amount of each input token to sell.
    /// @param _sqrtPriceLimitsX96 Per-leg sqrt price limit; swap stops when the limit is hit.
    /// @param _tokenOut The token to receive from all swaps.
    /// @param _amountOutTotalMinimum Minimum total amount of output tokens to receive.
    /// @param _fee The Uniswap fee tier to use for all swaps.
    /// @return amountOut Total amount of output tokens received.
    function exactInput(
        address[] memory _tokenIns,
        uint256[] memory _amountIns,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenOut,
        uint256 _amountOutTotalMinimum,
        uint24 _fee
    ) external returns (uint256 amountOut);

    /// @notice Batch buys tokenOuts for the same tokenIn using per-leg output targets and price limits.
    /// @dev Caller must approve amountInTotalMax to this contract before calling.
    /// @param _tokenOuts Array of tokens for each swap.
    /// @param _amountOuts Amount of output tokens to receive for each swap.
    /// @param _sqrtPriceLimitsX96 Per-leg sqrt price limit; swap stops when the limit is hit.
    /// @param _tokenIn The token to spend for all swaps.
    /// @param _amountInTotalMax Maximum total amount of input tokens to spend.
    /// @param _fee The Uniswap fee tier to use for all swaps.
    /// @return amountIn Total amount of input tokens spent.
    function exactOutput(
        address[] memory _tokenOuts,
        uint256[] memory _amountOuts,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee
    ) external returns (uint256 amountIn);

    /// @notice Waterfall buy: spends tokenIn budget to buy a basket of tokenOuts, each up to a per-token price limit.
    /// @dev Off-chain code computes final target prices via waterfall algorithm; pools are independent so sequential
    ///      execution produces the same result as interleaved. Unspent budget is refunded.
    /// @dev Caller must approve _amountIn of _tokenIn to this contract before calling.
    /// @param _tokenOuts Array of output tokens to buy, sorted by profitability (most profitable first).
    /// @param _sqrtPriceLimitsX96 Per-token target price; swap stops when limit is hit.
    /// @param _tokenIn The token to spend.
    /// @param _amountIn Total amount of tokenIn to spend.
    /// @param _fee The Uniswap fee tier to use for all swaps.
    /// @return amountInSpent Total amount of tokenIn consumed across all swaps.
    function waterfallBuy(
        address[] memory _tokenOuts,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenIn,
        uint256 _amountIn,
        uint24 _fee
    ) external returns (uint256 amountInSpent);
}
