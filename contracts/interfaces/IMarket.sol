// https://github.com/seer-pm/demo/blob/ed0a98c70ce13a0764ec5405126a90ebb7f6c94d/contracts/src/Market.sol
interface IMarket {
    function wrappedOutcome(uint256 index) external view returns (IERC20 wrapped1155, bytes memory data);
    // doesn't include the invalid outcome
    function numOutcomes() external view returns (uint256);
    function conditionalTokensParams() external view returns (bytes32 ,bytes32 ,uint256 parentOutcome,address parentMarket,bytes32 );
}