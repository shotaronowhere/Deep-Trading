// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IV3SwapRouter} from "./interfaces/IV3SwapRouter.sol";
import {IERC20} from "./interfaces/IERC20.sol";
import {IUniswapV3Pool} from "./interfaces/IUniswapV3Pool.sol";
import {FullMath} from "./libraries/FullMath.sol";

/// @title InOutRouter
/// @notice Allocates a single collateral budget across Uniswap V3 pools such that
///         outcome token prices satisfy constant ratios: price_i = r_i × price_0.
///
///         Uses the V3 analytical cost formula (valid for BOTH token orderings):
///
///             cost_i = L_eff_i × (√q_target_i − √q_current_i)
///
///         where q is the outcome token price in collateral terms and
///         L_eff = L / (1 − fee).  The formula collapses for both orderings because
///         amount0_delta = L×(1/√P_lo − 1/√P_hi), amount1_delta = L×(√P_hi − √P_lo),
///         and √P maps to 1/√q or √q depending on which token is collateral.
///
///         Given ratio constraints q_i = r_i × q_0, all targets are determined by
///         a single free variable √q_0.  Within one liquidity range the budget
///         equation is linear in √q_0 and has closed-form solution:
///
///             √q_0 = (B + C) × 2^96 / A
///
///         where A = Σ L_eff_i × √r_i / 2^96,  C = Σ L_eff_i × √q_current_i / 2^96.
///
///         Multi-tick liquidity changes are handled by iterating: compute single-tick
///         targets → swap (the router handles tick crossing) → re-read state → repeat.
contract InOutRouter {
    error TransferFailed();
    error ApprovalFailed();
    error LengthMismatch();

    IV3SwapRouter public immutable router;

    uint256 internal constant Q96 = 1 << 96;
    uint256 internal constant MAX_ITER = 5;

    constructor(address _router) {
        router = IV3SwapRouter(_router);
    }

    struct SwapParams {
        /// @dev Uniswap V3 pool addresses (one per outcome token).
        ///      Must be sorted by activationX96 ascending.
        address[] pools;
        /// @dev Collateral token to spend (e.g. SUSDS).
        address tokenIn;
        /// @dev Total collateral budget pulled from caller.
        uint256 amountIn;
        /// @dev √r_i in Q96 format, defining target price ratios: price_i = r_i × price_0.
        ///      sqrtRatiosX96[0] must equal 2^96 (since r_0 = 1).
        uint160[] sqrtRatiosX96;
        /// @dev Ratchet activation thresholds: pool i only participates in the
        ///      allocation when the solved √q_0 ≥ activationX96[i].
        ///      Must be sorted ascending.  activationX96[0] must be 0 (pool 0 always active).
        ///      Once activated, a pool stays active (ratchet — forward only).
        uint160[] activationX96;
    }

    /// @dev Cached immutable info for each pool.
    struct PoolInfo {
        address pool;
        address tokenOut;
        uint24 fee;
        bool zeroForOne;
    }

    /// @notice Buys outcome tokens across pools, consuming the full budget while
    ///         maintaining constant price ratios.
    function swapToRatio(SwapParams calldata p)
        external
        returns (uint256[] memory amountsOut, uint256 amountInUsed)
    {
        uint256 n = p.pools.length;
        if (n != p.sqrtRatiosX96.length || n != p.activationX96.length) revert LengthMismatch();

        if (!IERC20(p.tokenIn).transferFrom(msg.sender, address(this), p.amountIn))
            revert TransferFailed();
        if (!IERC20(p.tokenIn).approve(address(router), p.amountIn))
            revert ApprovalFailed();

        amountsOut = new uint256[](n);
        PoolInfo[] memory info = _cachePoolInfo(p.pools, p.tokenIn, n);

        // Iterative allocation.  Each pass:
        //   1. Read current √q and L_eff for every pool.
        //   2. Closed-form solve for target √q_0 (single-tick approximation).
        //   3. Convert to pool sqrtPrice limits.
        //   4. Swap via router (it handles tick crossing internally).
        //   5. Remaining budget feeds the next iteration.
        // Convergence: first pass consumes most budget; subsequent passes mop up
        // residual from tick-crossing liquidity differences.
        for (uint256 iter = 0; iter < MAX_ITER; iter++) {
            uint256 budget = IERC20(p.tokenIn).balanceOf(address(this));
            if (budget == 0) break;

            uint160[] memory sqrtPriceLimits = _computeLimits(
                p.sqrtRatiosX96, p.activationX96, info, budget, n
            );

            uint256 consumed;
            for (uint256 i = 0; i < n; i++) {
                if (sqrtPriceLimits[i] == 0) continue;
                uint256 out;
                uint256 spent;
                (out, spent) = _swap(p.tokenIn, info[i], sqrtPriceLimits[i]);
                amountsOut[i] += out;
                consumed += spent;
            }

            amountInUsed += consumed;
            if (consumed == 0) break;
        }

        // TODO: global slippage constraint — e.g.:
        //   require(amountInUsed >= minSpend && amountInUsed <= maxSpend);
        //   or verify final pool prices lie within caller-supplied bounds.

        uint256 leftover = IERC20(p.tokenIn).balanceOf(address(this));
        if (leftover > 0) {
            if (!IERC20(p.tokenIn).transfer(msg.sender, leftover)) revert TransferFailed();
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Internal helpers
    // ──────────────────────────────────────────────────────────────────────

    /// @dev Read and cache immutable pool info (token ordering, fee, tokenOut).
    function _cachePoolInfo(address[] calldata pools, address tokenIn, uint256 n)
        internal
        view
        returns (PoolInfo[] memory info)
    {
        info = new PoolInfo[](n);
        for (uint256 i = 0; i < n; i++) {
            IUniswapV3Pool pool = IUniswapV3Pool(pools[i]);
            address t0 = pool.token0();
            bool z = (t0 == tokenIn);
            info[i] = PoolInfo({
                pool:       pools[i],
                tokenOut:   z ? pool.token1() : t0,
                fee:        pool.fee(),
                zeroForOne: z
            });
        }
    }

    /// @dev Read pool state, solve for targets, return sqrtPriceLimits for the router.
    ///      Entries are 0 for pools that should be skipped (not yet activated / above target / no liquidity).
    function _computeLimits(
        uint160[] calldata sqrtRatiosX96,
        uint160[] calldata activationX96,
        PoolInfo[] memory info,
        uint256 budget,
        uint256 n
    ) internal view returns (uint160[] memory sqrtPriceLimits) {
        sqrtPriceLimits = new uint160[](n);

        // Read current pool state.
        uint160[] memory sqrtQX96 = new uint160[](n);
        uint256[] memory Leff     = new uint256[](n);
        _readPoolState(info, n, sqrtQX96, Leff);

        // Solve for target √q_0 with ratchet activation.
        (uint160 sqrtQ0Target, bool[] memory active) =
            _solveTarget(sqrtRatiosX96, activationX96, sqrtQX96, Leff, budget);

        if (sqrtQ0Target == 0) return sqrtPriceLimits;

        // Convert target √q per pool → pool sqrtPrice limit.
        for (uint256 i = 0; i < n; i++) {
            if (!active[i]) continue;

            uint160 sqrtQTgt = uint160(FullMath.mulDiv(
                uint256(sqrtRatiosX96[i]), uint256(sqrtQ0Target), Q96
            ));
            if (sqrtQTgt <= sqrtQX96[i]) continue;

            // zeroForOne: sqrtPriceLimit = 2^192 / sqrtQTgt  (price decreases)
            // else:       sqrtPriceLimit = sqrtQTgt           (price increases)
            sqrtPriceLimits[i] = info[i].zeroForOne
                ? uint160(FullMath.mulDiv(Q96, Q96, sqrtQTgt))
                : sqrtQTgt;
        }
    }

    /// @dev Read sqrtQ (outcome-price sqrt, Q96) and L_eff from each pool.
    function _readPoolState(
        PoolInfo[] memory info,
        uint256 n,
        uint160[] memory sqrtQX96,
        uint256[] memory Leff
    ) internal view {
        for (uint256 i = 0; i < n; i++) {
            IUniswapV3Pool pool = IUniswapV3Pool(info[i].pool);
            (uint160 sqrtPriceX96,,,,,,) = pool.slot0();
            uint128 L = pool.liquidity();

            Leff[i] = uint256(L) * 1e6 / (1e6 - uint256(info[i].fee));

            // zeroForOne (collateral = token0): sqrtP = 1/√q × 2^96  →  sqrtQ = 2^192 / sqrtP
            // else       (collateral = token1): sqrtP = √q × 2^96    →  sqrtQ = sqrtP
            sqrtQX96[i] = info[i].zeroForOne
                ? uint160(FullMath.mulDiv(Q96, Q96, sqrtPriceX96))
                : sqrtPriceX96;
        }
    }

    /// @dev Closed-form target √q_0 with ratchet activation.
    ///
    ///      Outer loop (ratchet): pools activate in threshold order.  Pool k is
    ///      included only when the solved √q_0 ≥ activationX96[k].  Once activated
    ///      a pool stays in the candidate set (ratchet — forward only).
    ///
    ///      Inner loop (exclusion): among activated pools, those whose ratio-implied
    ///      target ≤ current price are excluded (no point buying an overpriced pool).
    ///
    ///      Returns 0 if no active liquidity.
    function _solveTarget(
        uint160[] calldata sqrtRatiosX96,
        uint160[] calldata activationX96,
        uint160[] memory sqrtQX96,
        uint256[] memory Leff,
        uint256 budget
    ) internal pure returns (uint160 sqrtQ0Target, bool[] memory active) {
        uint256 n = sqrtQX96.length;
        active = new bool[](n);

        // Outer loop: activate pools in ratchet order.
        for (uint256 k = 0; k < n; k++) {
            // Reset active set: all pools 0..k are candidates.
            // (Re-evaluate because adding pool k may change which earlier pools are overpriced.)
            for (uint256 i = 0; i <= k; i++) active[i] = true;
            for (uint256 i = k + 1; i < n; i++) active[i] = false;

            // Inner loop: exclude overpriced pools until stable.
            sqrtQ0Target = 0;
            for (uint256 round = 0; round <= k + 1; round++) {
                uint256 A;
                uint256 C;
                for (uint256 i = 0; i <= k; i++) {
                    if (!active[i]) continue;
                    A += FullMath.mulDiv(Leff[i], uint256(sqrtRatiosX96[i]), Q96);
                    C += FullMath.mulDiv(Leff[i], uint256(sqrtQX96[i]), Q96);
                }
                if (A == 0) return (0, active);

                sqrtQ0Target = uint160(FullMath.mulDiv(budget + C, Q96, A));

                bool stable = true;
                for (uint256 i = 0; i <= k; i++) {
                    if (!active[i]) continue;
                    uint160 tgt = uint160(FullMath.mulDiv(
                        uint256(sqrtRatiosX96[i]), uint256(sqrtQ0Target), Q96
                    ));
                    if (tgt <= sqrtQX96[i]) {
                        active[i] = false;
                        stable = false;
                    }
                }
                if (stable) break;
            }

            // Check if √q_0 reaches the next pool's activation threshold.
            if (k + 1 < n && sqrtQ0Target < activationX96[k + 1]) break;
        }
    }

    /// @dev Execute a single swap, returning (amountOut, amountInSpent).
    function _swap(
        address tokenIn,
        PoolInfo memory pi,
        uint160 sqrtPriceLimit
    ) internal returns (uint256 amountOut, uint256 spent) {
        uint256 remaining = IERC20(tokenIn).balanceOf(address(this));
        if (remaining == 0) return (0, 0);

        uint256 balBefore = remaining;
        amountOut = router.exactInputSingle(
            IV3SwapRouter.ExactInputSingleParams({
                tokenIn:           tokenIn,
                tokenOut:          pi.tokenOut,
                fee:               pi.fee,
                recipient:         msg.sender,
                amountIn:          remaining,
                amountOutMinimum:  0, // TODO: global slippage
                sqrtPriceLimitX96: sqrtPriceLimit
            })
        );
        spent = balBefore - IERC20(tokenIn).balanceOf(address(this));
    }
}
