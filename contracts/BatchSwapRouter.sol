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
    error InvalidArrayLength();

    IV3SwapRouter public immutable router;

    constructor(address _router) {
        router = IV3SwapRouter(_router);
    }

    // see IBatchSwapRouter for function docs
    function exactInput(
        address[] memory _tokenIns,
        uint256 _amountIn,
        address _tokenOut,
        uint256 _amountOutTotalMinimum,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountOut) {
        uint256[] memory amountsIn = new uint256[](_tokenIns.length);
        for (uint i = 0; i < _tokenIns.length; i++) {
            amountsIn[i] = _amountIn;
        }
        return exactInput(_tokenIns, amountsIn, _tokenOut, _amountOutTotalMinimum, _fee, _sqrtPriceLimitX96);
    }

    // see IBatchSwapRouter for function docs
    function exactInput(
        address[] memory _tokenIns,
        uint256[] memory _amountIn,
        address _tokenOut,
        uint256 _amountOutTotalMinimum,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) public returns (uint256 amountOut) {
        if (_tokenIns.length != _amountIn.length) {
            revert InvalidArrayLength();
        }
        for (uint i = 0; i < _tokenIns.length; i++) {
            bool success = IERC20(_tokenIns[i]).transferFrom(msg.sender, address(this), _amountIn[i]);
            if (!success) {
                revert TransferFailed();
            }
            success = IERC20(_tokenIns[i]).approve(address(router), _amountIn[i]);
            if (!success) {
                revert ApprovalFailed();
            }
            amountOut += router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: _tokenIns[i],
                    tokenOut: address(_tokenOut), // same tokenOut for all swaps
                    fee: _fee,
                    recipient: msg.sender,
                    amountIn: _amountIn[i],
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: _sqrtPriceLimitX96
                })
            );
        }
        if (amountOut < _amountOutTotalMinimum) {
            revert SlippageExceeded();
        }
    }

    // see IBatchSwapRouter for function docs
    function exactOutput(
        address[] memory _tokenOuts,
        uint256 _amountOut,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountIn) {
        uint256[] memory amountsOut = new uint256[](_tokenOuts.length);
        for (uint i = 0; i < _tokenOuts.length; i++) {
            amountsOut[i] = _amountOut;
        }
        return exactOutput(_tokenOuts, amountsOut, _tokenIn, _amountInTotalMax, _fee, _sqrtPriceLimitX96);
    }

    // see IBatchSwapRouter for function docs
    function exactOutput(
        address[] memory _tokenOuts,
        uint256[] memory _amountOut,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) public returns (uint256 amountIn) {
        if (_tokenOuts.length != _amountOut.length) {
            revert InvalidArrayLength();
        }
        bool success = IERC20(_tokenIn).transferFrom(msg.sender, address(this), _amountInTotalMax);
        if (!success) {
            revert TransferFailed();
        }

        // tokenIn is same for all swaps, so approve once
        success = IERC20(_tokenIn).approve(address(router), _amountInTotalMax);
        if (!success) {
            revert ApprovalFailed();
        }


        for (uint i = 0; i < _tokenOuts.length; i++) {
            uint256 amountInRemaining = _amountInTotalMax - amountIn; // remaining max for this swap
            amountIn += router.exactOutputSingle(
                IV3SwapRouter.ExactOutputSingleParams({
                    tokenIn: address(_tokenIn),
                    tokenOut: _tokenOuts[i], // same tokenOut for all swaps
                    fee: _fee,
                    recipient: msg.sender,
                    amountOut: _amountOut[i], // same amountOut for all swaps
                    amountInMaximum: amountInRemaining, // per-swap ceiling uses remaining max; aggregate checked at the end
                    sqrtPriceLimitX96: _sqrtPriceLimitX96
                })
            );
        }
        if (amountIn > _amountInTotalMax) {
            revert SlippageExceeded();
        }
        // Refund unused tokenIn to caller
        uint256 remaining = IERC20(_tokenIn).balanceOf(address(this));
        if (remaining > 0) {
            success = IERC20(_tokenIn).transfer(msg.sender, remaining);
            if (!success) {
                revert TransferFailed();
            }
        }
    }
}
