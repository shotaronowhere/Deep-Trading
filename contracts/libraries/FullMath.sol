// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FullMath
/// @notice Full-precision multiply-then-divide using 512-bit intermediates.
/// @dev Based on Uniswap V3 FullMath (MIT), adapted for Solidity 0.8.x.
library FullMath {
    /// @notice Calculates floor(a × b ÷ denominator) with full precision.
    ///         Reverts if result overflows uint256 or denominator is zero.
    function mulDiv(uint256 a, uint256 b, uint256 denominator) internal pure returns (uint256 result) {
        // 512-bit multiply [prod1, prod0] = a * b.
        uint256 prod0;
        uint256 prod1;
        assembly {
            let mm := mulmod(a, b, not(0))
            prod0 := mul(a, b)
            prod1 := sub(sub(mm, prod0), lt(mm, prod0))
        }

        // Short-circuit: no overflow.
        if (prod1 == 0) {
            require(denominator > 0);
            assembly {
                result := div(prod0, denominator)
            }
            return result;
        }

        // Overflow guard.
        require(denominator > prod1);

        // Subtract remainder to make [prod1, prod0] divisible by denominator.
        uint256 remainder;
        assembly {
            remainder := mulmod(a, b, denominator)
            prod1 := sub(prod1, gt(remainder, prod0))
            prod0 := sub(prod0, remainder)
        }

        // Factor powers of two out of denominator.
        uint256 twos = denominator & (~denominator + 1);
        assembly {
            denominator := div(denominator, twos)
            prod0 := div(prod0, twos)
        }

        // Shift prod1 bits into prod0.
        assembly {
            prod0 := or(prod0, mul(prod1, add(div(sub(0, twos), twos), 1)))
        }

        // Modular inverse via Newton-Raphson (6 iterations for 256-bit precision).
        unchecked {
            uint256 inv = (3 * denominator) ^ 2;
            inv *= 2 - denominator * inv;
            inv *= 2 - denominator * inv;
            inv *= 2 - denominator * inv;
            inv *= 2 - denominator * inv;
            inv *= 2 - denominator * inv;
            inv *= 2 - denominator * inv;
            result = prod0 * inv;
        }
    }
}
