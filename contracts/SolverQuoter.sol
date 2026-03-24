// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "./interfaces/IERC20.sol";

/// @title SolverQuoter — revert-encoded on-chain solver preview
/// @notice Designed for eth_call only. Always reverts with encoded results.
/// @dev Called inside TradeExecutor.batchExecute: executor first transfers all
///      tokens + collateral to this contract, then calls quote(). The quoter
///      approves the solver, calls it, reads post-rebalance balances, and
///      reverts with abi-encoded (bool success, bytes returnData, uint256 postCash,
///      uint256[] postBalances). The entire batchExecute reverts so no state changes.
contract SolverQuoter {
    /// @notice Run a solver and revert with the result.
    /// @param solver       Address of the Rebalancer / RebalancerMixed contract
    /// @param solverCall   ABI-encoded function call (e.g. rebalanceExact(params,...))
    /// @param tokens       Outcome token addresses (same order as RebalanceParams.tokens)
    /// @param collateral   Collateral token (sUSDS)
    function quote(
        address solver,
        bytes calldata solverCall,
        address[] calldata tokens,
        address collateral
    ) external {
        // Approve solver for all tokens this contract holds
        uint256 n = tokens.length;
        for (uint256 i = 0; i < n; i++) {
            uint256 bal = IERC20(tokens[i]).balanceOf(address(this));
            if (bal > 0) {
                IERC20(tokens[i]).approve(solver, bal);
            }
        }
        uint256 collBal = IERC20(collateral).balanceOf(address(this));
        if (collBal > 0) {
            IERC20(collateral).approve(solver, collBal);
        }

        // Call the solver
        (bool success, bytes memory returnData) = solver.call(solverCall);

        // Read post-rebalance balances (solver returns tokens to msg.sender = this)
        uint256 postCash = IERC20(collateral).balanceOf(address(this));
        uint256[] memory postBalances = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            postBalances[i] = IERC20(tokens[i]).balanceOf(address(this));
        }

        // Revert with encoded results — caller parses from revert data
        bytes memory result = abi.encode(success, returnData, postCash, postBalances);
        assembly {
            revert(add(result, 0x20), mload(result))
        }
    }
}
