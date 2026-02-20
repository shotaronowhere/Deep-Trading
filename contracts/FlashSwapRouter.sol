// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IUnlockCallback} from "lib/v4-core/src/interfaces/callback/IUnlockCallback.sol";
import {IERC20Minimal} from "lib/v4-core/src/interfaces/external/IERC20Minimal.sol";
import {IPoolManager} from "lib/v4-core/src/interfaces/IPoolManager.sol";
import {IBatchSwapRouter} from "./interfaces/IBatchSwapRouter.sol";
import {ICTFRouter} from "./interfaces/ICTFRouter.sol";
import {Currency} from "lib/v4-core/src/libraries/CurrencyDelta.sol";

// https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Market.sol
interface IMarket {
    function wrappedOutcome(uint256 index) external view returns (IERC20 wrapped1155, bytes memory data);
    // doesn't include the invalid outcome
    function numOutcomes() external view returns (uint256);
}

contract FlashSwapRouter is IUnlockCallback {
    IPoolManager internal immutable pool;
    ICTFRouter internal immutable ctfRouter;
    IPSM3 internal immutable psm;
    IBatchSwapRouter internal immutable batchSwapRouter;

    constructor(IPoolManager _pool, ICTFRouter _ctfRouter, IPSM3 _psm, IBatchSwapRouter _batchSwapRouter) {
        pool = _pool;
        ctfRouter = _ctfRouter;
        psm = _psm;
        batchSwapRouter = _batchSwapRouter;
    }

    function initiate(bytes calldata data) public {
        pool.unlock(data);
    }

    /// @notice Callback to handle the flashloan..
    /// @param data The encoded token address.
    function unlockCallback(bytes calldata data) external returns (bytes memory) {
        (
            address market,
            address tokenLoan,
            address tokenSwap,
            uint256 amountLoan,
            uint256 amountSwap,
            uint256 amountOutMin,
            IBatchSwapRouter.SwapParam[] memory swapParams
        ) = abi.decode(data, (address, address, uint256, uint256, uint256, IBatchSwapRouter.SwapParam[]));

        // loan
        pool.take(Currency.wrap(tokenLoan), address(this), amountLoan);

        // convert to tokenSwap via PSM
        IERC20Minimal(tokenLoan).approve(address(psm), amountLoan);
        psm.swapExactIn(tokenLoan, tokenSwap, amountLoan, amountSwap, address(this),0);

        // logic
        
        // mint + sell
        ctfRouter.splitPosition(tokenLoan, market, amountSwap);
        batchSwapRouter.exactInput(tokenSwap, amountSwap, amountOutMin, swapParams);

        uint256 numOutcomes = IMarket(market).numOutcomes() + 1; // include invalid outcome
        for (uint256 i = 0; i < numOutcomes; i++) {
            (IERC20 wrapped1155, bytes memory data) = IMarket(market).wrappedOutcome(i);
            uint256 balance = wrapped1155.balanceOf(address(this));
            if (balance > 0) {
                wrapped1155.transfer(msg.sender, balance);
            }
        }

        // repay
        pool.sync(Currency.wrap(tokenLoan));
        IERC20Minimal(tokenLoan).transfer(address(pool), amountLoan);
        pool.settle(Currency.wrap(tokenLoan));
    }
}