// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IV3SwapRouter} from "./interfaces/IV3SwapRouter.sol";
import {IERC20} from "./interfaces/IERC20.sol";
import {IBatchSwapRouter} from "./interfaces/IBatchSwapRouter.sol";
import {ICTFRouter} from "./interfaces/ICTFRouter.sol";

/// @title Batch Swap Router for Uniswap V3
/// @notice Batches Uniswap V3 swaps with aggregate slippage protection.
/// @dev Any tokens accidentally sent to this contract may be drained by anyone, so only approve the exact amount needed for swaps and never send tokens directly to this contract.
contract BatchSwapRouter is IBatchSwapRouter {
    error SlippageExceeded();
    error ApprovalFailed();
    error TransferFailed();
    error InvalidArrayLength();

    IV3SwapRouter public immutable router;
    ICTFRouter public immutable ctfRouter;

    constructor(address _router, address _ctfRouter) {
        router = IV3SwapRouter(_router);
        ctfRouter = ICTFRouter(_ctfRouter);
    }
/*
    function rebalance(
        address[] memory _tokenIns,
        uint256[] memory _amountIns,
        address[] memory _tokenOuts,
        uint256[] memory _amountOuts,
        address _tokenCollateral,
        uint256 _amountOutTotalMinimum,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external {
        // (1) sell overpriced outcome
        uint256 amountOutIntermediate = _exactInput(_tokenIns, _amountIns, _tokenCollateral, _fee, _sqrtPriceLimitX96);
        // (2)
        uint256[] memory amountsIn = new uint256[](1);
        amountsIn[0] = amountOutIntermediate;
        address[] memory tokenIns = new address[](1);
        tokenIns[0] = _tokenOut;

        uint256 amountOut = _exactInput(tokenIns, amountsIn, _tokenOut, _fee, _sqrtPriceLimitX96);
        if (amountOut < _amountOutTotalMinimum) {
            revert SlippageExceeded();
        }
    }*/

    // batch sell outcome for collateral
    // see IBatchSwapRouter for function docs
    function exactInput(
        address[] memory _tokenIns,
        uint256[] memory _amountIns,
        address _tokenOut,
        uint256 _amountOutTotalMinimum,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountOut) {
        amountOut = _exactInput(_tokenIns, _amountIns, _tokenOut, _fee, _sqrtPriceLimitX96);
        if (amountOut < _amountOutTotalMinimum) {
            revert SlippageExceeded();
        }
    }

    // see IBatchSwapRouter for function docs
    function _exactInput(
        address[] memory _tokenIns,
        uint256[] memory _amountIn,
        address _tokenOut,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) internal returns (uint256 amountOut) {
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
    }

    // batch buy outcome with collateral
    // see IBatchSwapRouter for function docs
    function exactOutput(
        address[] memory _tokenOuts,
        uint256[] memory _amountOuts,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) external returns (uint256 amountIn) {
        amountIn = _exactOutput(_tokenOuts, _amountOuts, _tokenIn, _amountInTotalMax, _fee, _sqrtPriceLimitX96);
        if (amountIn > _amountInTotalMax) {
            revert SlippageExceeded();
        }
    }

    // see IBatchSwapRouter for function docs
    function _exactOutput(
        address[] memory _tokenOuts,
        uint256[] memory _amountOuts,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee,
        uint160 _sqrtPriceLimitX96
    ) internal returns (uint256 amountIn) {
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
                    amountOut: _amountOuts[i], // same amountOut for all swaps
                    amountInMaximum: amountInRemaining, // per-swap ceiling uses remaining max; aggregate checked at the end
                    sqrtPriceLimitX96: _sqrtPriceLimitX96
                })
            );
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