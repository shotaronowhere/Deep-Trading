// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
}

// https://github.com/Uniswap/swap-router-contracts/blob/70bc2e40dfca294c1cea9bf67a4036732ee54303/contracts/interfaces/IV3SwapRouter.sol
interface IV3SwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }

    /// @notice Swaps `amountIn` of one token for as much as possible of another token
    /// @dev Setting `amountIn` to 0 will cause the contract to look up its own balance,
    /// and swap the entire amount, enabling contracts to send tokens before calling this function.
    /// @param params The parameters necessary for the swap, encoded as `ExactInputSingleParams` in calldata
    /// @return amountOut The amount of the received token
    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);

    struct ExactOutputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 amountOut;
        uint256 amountInMaximum;
        uint160 sqrtPriceLimitX96;
    }

    /// @notice Swaps as little as possible of one token for `amountOut` of another token
    /// that may remain in the router after the swap.
    /// @param params The parameters necessary for the swap, encoded as `ExactOutputSingleParams` in calldata
    /// @return amountIn The amount of the input token
    function exactOutputSingle(ExactOutputSingleParams calldata params) external payable returns (uint256 amountIn);
} 

/// @title Batch Router for Uniswap V3
/// @notice Batches Uniswap V3 swaps with aggregate slippage protection.
contract BatchRouter {
    error SlippageExceeded();

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
            tokenIn.approve(address(router), swaps[i].amountIn);
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
    /// @dev To enforce only group slippage protection, all swap params must be zero.
    /// @dev tokenIn is assumed to be the same for all swaps in swaps.
    /// @dev tokenOut is sent to recipient specified in each swaps[i].
    function buy(
        IV3SwapRouter.ExactOutputSingleParams[] calldata swaps,
        uint256 max
    ) external returns (uint256 amount) {
        // tokenIn is same for all swaps, so approve once
        IERC20 tokenIn = IERC20(swaps[0].tokenIn);
        tokenIn.approve(address(router), max);
        for (uint i = 0; i < swaps.length; i++) {
            // TRUSTED: external call to Uniswap V3 swap router
            amount += router.exactOutputSingle(swaps[i]);
        }
        require(amount <= max, SlippageExceeded());
        // Refund unused tokenIn to caller
        uint256 remaining = tokenIn.balanceOf(address(this));
        if (remaining > 0) tokenIn.transfer(msg.sender, remaining);
    }
}
