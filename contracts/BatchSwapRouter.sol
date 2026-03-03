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
    error InvalidArrayLength();
    error ApprovalFailed();
    error TransferFailed();

    IV3SwapRouter public immutable router;

    constructor(address _router) {
        router = IV3SwapRouter(_router);
    }

    function _callOptionalBool(address token, bytes memory data) private returns (bool) {
        (bool success, bytes memory returndata) = token.call(data);
        if (!success) {
            return false;
        }
        if (returndata.length == 0) {
            return true;
        }
        if (returndata.length != 32) {
            return false;
        }
        return abi.decode(returndata, (bool));
    }

    function _safeTransferFrom(address token, address from, address to, uint256 amount) private {
        bool success = _callOptionalBool(token, abi.encodeWithSelector(IERC20.transferFrom.selector, from, to, amount));
        if (!success) {
            revert TransferFailed();
        }
    }

    function _safeTransfer(address token, address to, uint256 amount) private {
        bool success = _callOptionalBool(token, abi.encodeWithSelector(IERC20.transfer.selector, to, amount));
        if (!success) {
            revert TransferFailed();
        }
    }

    function _forceApprove(address token, address spender, uint256 amount) private {
        bool success = _callOptionalBool(token, abi.encodeWithSelector(IERC20.approve.selector, spender, amount));
        if (success) {
            return;
        }

        bool resetOk = _callOptionalBool(token, abi.encodeWithSelector(IERC20.approve.selector, spender, 0));
        if (!resetOk) {
            revert ApprovalFailed();
        }
        bool approveOk = _callOptionalBool(token, abi.encodeWithSelector(IERC20.approve.selector, spender, amount));
        if (!approveOk) {
            revert ApprovalFailed();
        }
    }

    // batch sell outcome for collateral
    // see IBatchSwapRouter for function docs
    function exactInput(
        address[] memory _tokenIns,
        uint256[] memory _amountIns,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenOut,
        uint256 _amountOutTotalMinimum,
        uint24 _fee
    ) external returns (uint256 amountOut) {
        if (_tokenIns.length != _amountIns.length || _tokenIns.length != _sqrtPriceLimitsX96.length) {
            revert InvalidArrayLength();
        }
        amountOut = _exactInput(_tokenIns, _amountIns, _sqrtPriceLimitsX96, _tokenOut, _fee);
        if (amountOut < _amountOutTotalMinimum) {
            revert SlippageExceeded();
        }
    }

    // batch sell with per-token price limits, refund unexecuted input
    // see IBatchSwapRouter for function docs
    function _exactInput(
        address[] memory _tokenIns,
        uint256[] memory _amountIns,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenOut,
        uint24 _fee
    ) internal returns (uint256 amountOut) {
        for (uint256 i = 0; i < _tokenIns.length; i++) {
            uint256 balanceBefore = IERC20(_tokenIns[i]).balanceOf(address(this));
            _safeTransferFrom(_tokenIns[i], msg.sender, address(this), _amountIns[i]);
            _forceApprove(_tokenIns[i], address(router), _amountIns[i]);
            amountOut += router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: _tokenIns[i],
                    tokenOut: _tokenOut,
                    fee: _fee,
                    recipient: msg.sender,
                    amountIn: _amountIns[i],
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: _sqrtPriceLimitsX96[i]
                })
            );
            // Refund unspent input (partial fill from price limit)
            uint256 balanceAfter = IERC20(_tokenIns[i]).balanceOf(address(this));
            if (balanceAfter > balanceBefore) {
                _safeTransfer(_tokenIns[i], msg.sender, balanceAfter - balanceBefore);
            }
        }
    }

    // batch buy outcome with collateral
    // see IBatchSwapRouter for function docs
    function exactOutput(
        address[] memory _tokenOuts,
        uint256[] memory _amountOuts,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee
    ) external returns (uint256 amountIn) {
        if (_tokenOuts.length != _amountOuts.length || _tokenOuts.length != _sqrtPriceLimitsX96.length) {
            revert InvalidArrayLength();
        }
        amountIn = _exactOutput(_tokenOuts, _amountOuts, _sqrtPriceLimitsX96, _tokenIn, _amountInTotalMax, _fee);
        if (amountIn > _amountInTotalMax) {
            revert SlippageExceeded();
        }
    }

    // waterfall buy: spend tokenIn budget across tokenOuts, each up to its price limit
    // see IBatchSwapRouter for function docs
    function waterfallBuy(
        address[] memory _tokenOuts,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenIn,
        uint256 _amountIn,
        uint24 _fee
    ) external returns (uint256 amountInSpent) {
        if (_tokenOuts.length != _sqrtPriceLimitsX96.length) {
            revert InvalidArrayLength();
        }
        uint256 balanceBefore = IERC20(_tokenIn).balanceOf(address(this));
        _safeTransferFrom(_tokenIn, msg.sender, address(this), _amountIn);
        _forceApprove(_tokenIn, address(router), _amountIn);

        for (uint256 i = 0; i < _tokenOuts.length; i++) {
            uint256 remaining = IERC20(_tokenIn).balanceOf(address(this));
            if (remaining == 0) break;

            router.exactInputSingle(
                IV3SwapRouter.ExactInputSingleParams({
                    tokenIn: _tokenIn,
                    tokenOut: _tokenOuts[i],
                    fee: _fee,
                    recipient: msg.sender,
                    amountIn: remaining,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: _sqrtPriceLimitsX96[i]
                })
            );
        }

        // Refund unspent budget
        uint256 refund = IERC20(_tokenIn).balanceOf(address(this));
        uint256 callerUnspent = refund > balanceBefore ? refund - balanceBefore : 0;
        if (callerUnspent > 0) {
            _safeTransfer(_tokenIn, msg.sender, callerUnspent);
        }
        if (callerUnspent > _amountIn) {
            callerUnspent = _amountIn;
        }
        amountInSpent = _amountIn - callerUnspent;
    }

    // see IBatchSwapRouter for function docs
    function _exactOutput(
        address[] memory _tokenOuts,
        uint256[] memory _amountOuts,
        uint160[] memory _sqrtPriceLimitsX96,
        address _tokenIn,
        uint256 _amountInTotalMax,
        uint24 _fee
    ) internal returns (uint256 amountIn) {
        uint256 balanceBefore = IERC20(_tokenIn).balanceOf(address(this));
        _safeTransferFrom(_tokenIn, msg.sender, address(this), _amountInTotalMax);

        // tokenIn is same for all swaps, so approve once
        _forceApprove(_tokenIn, address(router), _amountInTotalMax);

        for (uint256 i = 0; i < _tokenOuts.length; i++) {
            uint256 amountInRemaining = _amountInTotalMax - amountIn; // remaining max for this swap
            amountIn += router.exactOutputSingle(
                IV3SwapRouter.ExactOutputSingleParams({
                    tokenIn: address(_tokenIn),
                    tokenOut: _tokenOuts[i], // same tokenOut for all swaps
                    fee: _fee,
                    recipient: msg.sender,
                    amountOut: _amountOuts[i], // same amountOut for all swaps
                    amountInMaximum: amountInRemaining, // per-swap ceiling uses remaining max; aggregate checked at the end
                    sqrtPriceLimitX96: _sqrtPriceLimitsX96[i]
                })
            );
        }
        // Refund unused tokenIn to caller
        uint256 remaining = IERC20(_tokenIn).balanceOf(address(this));
        uint256 callerRefund = remaining > balanceBefore ? remaining - balanceBefore : 0;
        if (callerRefund > 0) {
            _safeTransfer(_tokenIn, msg.sender, callerRefund);
        }
    }
}
