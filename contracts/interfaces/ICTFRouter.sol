// SPDX-License-Identifier: MIT
//https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Router.sol
pragma solidity ^0.8.24;

interface ICTFRouter {
    function splitPosition(address collateralToken, address market, uint256 amount) external;
    function mergePositions(address collateralToken, address market, uint256 amount) external;
}