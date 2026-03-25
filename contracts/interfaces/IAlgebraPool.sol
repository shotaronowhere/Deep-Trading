// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity ^0.8.24;

/// @title AlgebraV1.9 (Swapr) pool interface
/// @dev Source: https://github.com/cryptoalgebra/AlgebraV1.9/blob/main/src/core/contracts/interfaces/pool/IAlgebraPoolState.sol
interface IAlgebraPool {
    function globalState()
        external
        view
        returns (
            uint160 price,
            int24 tick,
            uint16 fee,
            uint16 timepointIndex,
            uint8 communityFeeToken0,
            uint8 communityFeeToken1,
            bool unlocked
        );

    function liquidity() external view returns (uint128);

    function tickSpacing() external view returns (int24);

    function tickTable(int16 wordPosition) external view returns (uint256);

    function ticks(int24 tick)
        external
        view
        returns (
            uint128 liquidityTotal,
            int128 liquidityDelta,
            uint256 outerFeeGrowth0Token,
            uint256 outerFeeGrowth1Token,
            int56 outerTickCumulative,
            uint160 outerSecondsPerLiquidity,
            uint32 outerSecondsSpent,
            bool initialized
        );

    function token0() external view returns (address);
}
