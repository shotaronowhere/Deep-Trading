// SPDX-License-Identifier: MIT
// https://github.com/seer-pm/ai-prediction-markets/blob/1b58498d284e890ea11f003c8206342c95b9382a/src/contracts/TradeExecutor.sol
pragma solidity ^0.8.24;

contract TradeExecutor {
    struct Call {
        address to;
        bytes data;
    }

    struct ValueCall {
        address to;
        uint256 value;
        bytes data;
    }
    
    /// @dev Contract owner
    address public immutable owner;

    
    address public permitted; /// @dev wallet address which is temporary allowed to call functions.
	uint public expire; /// @dev wallet address which is temporary allowed to call functions.

    error CallFailed();
    
    /// @dev Modifier to restrict access to owner only
    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    /// @dev Modifier to restrict access to owner or wallet
    modifier onlyAuthorized() {
        require(msg.sender==owner || (msg.sender==permitted && block.timestamp<expire),  "Caller is not authorized");
        _;
    }

    /// @dev Constructor.
    /// @param _owner Immutable owner of the contract.
    constructor(
        address _owner
    ) {
        owner = _owner;
    }

    /// @dev set wallet to call the contract functions. Only callable by owner.
	/// @param _permitted an address temporarily allowed to control the trade executor.
	/// @param _expire the time when this permission expire.
    function setTemporaryPermission(address _permitted, uint _expire) external onlyOwner {
        permitted = _permitted;
		expire = _expire;
    }
    
    /// @dev Execute calls in a single transaction. Only callable by the owner or current wallet.
    /// @param calls Array of calls to execute
    function batchExecute(Call[] calldata calls) external onlyAuthorized {
        for (uint i = 0; i < calls.length; i++) {
            (bool success, bytes memory returndata) = calls[i].to.call(calls[i].data);
            _revertIfFailed(success, returndata);
        }
    }

    /// @dev Execute calls with value in a single transaction. Only callable by the owner or current wallet.
    /// @param calls Array of calls to execute
    function batchValueExecute(ValueCall[] calldata calls) external payable onlyAuthorized {
        for (uint i = 0; i < calls.length; i++) {
            (bool success, bytes memory returndata) = calls[i].to.call{value: calls[i].value}(calls[i].data);
            _revertIfFailed(success, returndata);
        }
    }

    /// @dev Receive ETH.
    receive() external payable { }

    function _revertIfFailed(bool success, bytes memory returndata) private pure {
        if (success) {
            return;
        }
        if (returndata.length > 0) {
            assembly {
                revert(add(returndata, 0x20), mload(returndata))
            }
        }
        revert CallFailed();
    }
}
