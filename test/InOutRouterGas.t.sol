// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {InOutRouter} from "../contracts/InOutRouter.sol";
import {IERC20} from "../contracts/interfaces/IERC20.sol";
import {IV3SwapRouter} from "../contracts/interfaces/IV3SwapRouter.sol";

// ── Mocks ────────────────────────────────────────────────────────────────

contract GasMockERC20 {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        if (allowance[from][msg.sender] < amount) return false;
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

/// @dev Mock router that consumes a fixed fraction of amountIn per swap.
///      Simulates partial fills (as if sqrtPriceLimit stopped the swap early).
contract GasMockRouter is IV3SwapRouter {
    uint256 public consumeAmount;

    function setConsumeAmount(uint256 amount) external {
        consumeAmount = amount;
    }

    function exactInputSingle(ExactInputSingleParams calldata params)
        external
        payable
        override
        returns (uint256 amountOut)
    {
        uint256 pull = consumeAmount < params.amountIn ? consumeAmount : params.amountIn;
        if (pull > 0) {
            IERC20(params.tokenIn).transferFrom(msg.sender, address(this), pull);
        }
        amountOut = pull; // 1:1 for simplicity
    }

    function exactOutputSingle(ExactOutputSingleParams calldata)
        external
        payable
        override
        returns (uint256)
    {
        revert("not used");
    }
}

/// @dev Mock V3 pool returning configurable state.
contract GasMockPool {
    address public immutable token0;
    address public immutable token1;
    uint24  public immutable fee;
    uint160 public sqrtPriceX96;
    uint128 public liq;

    constructor(address _t0, address _t1, uint24 _fee, uint160 _sqrtPrice, uint128 _liq) {
        token0 = _t0;
        token1 = _t1;
        fee = _fee;
        sqrtPriceX96 = _sqrtPrice;
        liq = _liq;
    }

    function slot0() external view returns (
        uint160, int24, uint16, uint16, uint16, uint8, bool
    ) {
        return (sqrtPriceX96, 0, 0, 0, 0, 0, true);
    }

    function liquidity() external view returns (uint128) {
        return liq;
    }
}

// ── Test ─────────────────────────────────────────────────────────────────

contract InOutRouterGasTest is Test {
    GasMockERC20  internal tokenIn;
    GasMockRouter internal mockRouter;
    InOutRouter   internal inOutRouter;

    // Realistic V3 values:
    //   outcome price q ≈ 0.01  →  √q = 0.1
    //   zeroForOne (tokenIn = token0):  sqrtPriceX96 = (1/√q) × 2^96 = 10 × 2^96
    uint160 constant SQRT_PRICE = uint160(10) * (1 << 96);
    uint128 constant LIQUIDITY  = 1e18;
    uint24  constant FEE        = 3000; // 0.3%
    uint256 constant Q96        = 1 << 96;

    function setUp() public {
        tokenIn    = new GasMockERC20();
        mockRouter = new GasMockRouter();
        inOutRouter = new InOutRouter(address(mockRouter));
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// @dev Deploy n mock pools where tokenIn is always token0 (zeroForOne).
    function _createPools(uint256 n)
        internal
        returns (address[] memory pools, uint160[] memory sqrtRatios, uint160[] memory activations)
    {
        pools       = new address[](n);
        sqrtRatios  = new uint160[](n);
        activations = new uint160[](n);

        for (uint256 i = 0; i < n; i++) {
            // Each pool has a unique "outcome" token address.
            // Use address > tokenIn so tokenIn is token0.
            address outcome = address(uint160(address(tokenIn)) + uint160(i) + 1);

            pools[i] = address(new GasMockPool(
                address(tokenIn), outcome, FEE, SQRT_PRICE, LIQUIDITY
            ));
            sqrtRatios[i]  = uint160(Q96); // r_i = 1 (equal weighting)
            activations[i] = 0;            // all active immediately
        }
    }

    /// @dev Mint budget, approve, and call swapToRatio. Returns gas used.
    function _runSwap(uint256 n, uint256 budget)
        internal
        returns (uint256 gasUsed, uint256 amountInUsed)
    {
        (address[] memory pools, uint160[] memory sqrtRatios, uint160[] memory activations) = _createPools(n);

        // Each swap consumes budget/n so one iteration fully drains.
        mockRouter.setConsumeAmount(budget / n);

        tokenIn.mint(address(this), budget);
        tokenIn.approve(address(inOutRouter), budget);

        InOutRouter.SwapParams memory p = InOutRouter.SwapParams({
            pools:          pools,
            tokenIn:        address(tokenIn),
            amountIn:       budget,
            sqrtRatiosX96:  sqrtRatios,
            activationX96:  activations
        });

        uint256 gasBefore = gasleft();
        (, amountInUsed) = inOutRouter.swapToRatio(p);
        gasUsed = gasBefore - gasleft();
    }

    // ── Gas benchmarks ───────────────────────────────────────────────────

    function test_gas_1pool() public {
        (uint256 gas, uint256 used) = _runSwap(1, 1e18);
        emit log_named_uint("gas_1_pool", gas);
        emit log_named_uint("amountInUsed", used);
        assertGt(used, 0);
    }

    function test_gas_5pools() public {
        (uint256 gas, uint256 used) = _runSwap(5, 5e18);
        emit log_named_uint("gas_5_pools", gas);
        emit log_named_uint("per_pool", gas / 5);
        assertGt(used, 0);
    }

    function test_gas_10pools() public {
        (uint256 gas, uint256 used) = _runSwap(10, 10e18);
        emit log_named_uint("gas_10_pools", gas);
        emit log_named_uint("per_pool", gas / 10);
        assertGt(used, 0);
    }

    function test_gas_50pools() public {
        (uint256 gas, uint256 used) = _runSwap(50, 50e18);
        emit log_named_uint("gas_50_pools", gas);
        emit log_named_uint("per_pool", gas / 50);
        assertGt(used, 0);
    }

    function test_gas_98pools() public {
        (uint256 gas, uint256 used) = _runSwap(98, 98e18);
        emit log_named_uint("gas_98_pools", gas);
        emit log_named_uint("per_pool", gas / 98);
        assertGt(used, 0);
    }

    function test_gas_100pools() public {
        (uint256 gas, uint256 used) = _runSwap(100, 100e18);
        emit log_named_uint("gas_100_pools", gas);
        emit log_named_uint("per_pool", gas / 100);
        assertGt(used, 0);
    }

    /// @dev Measures gas for 2 iterations (mock consumes 60% per swap on first pass).
    function test_gas_100pools_multiIteration() public {
        uint256 n = 100;
        uint256 budget = 100e18;
        (address[] memory pools, uint160[] memory sqrtRatios, uint160[] memory activations) = _createPools(n);

        // Consume 60% of budget/n per swap → ~60% consumed in iter 1, remainder in iter 2+.
        mockRouter.setConsumeAmount((budget * 60) / (n * 100));

        tokenIn.mint(address(this), budget);
        tokenIn.approve(address(inOutRouter), budget);

        InOutRouter.SwapParams memory p = InOutRouter.SwapParams({
            pools:          pools,
            tokenIn:        address(tokenIn),
            amountIn:       budget,
            sqrtRatiosX96:  sqrtRatios,
            activationX96:  activations
        });

        uint256 gasBefore = gasleft();
        (, uint256 used) = inOutRouter.swapToRatio(p);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("gas_100_multi_iter", gasUsed);
        emit log_named_uint("per_pool", gasUsed / n);
        emit log_named_uint("amountInUsed", used);
        assertGt(used, 0);
    }

    /// @dev Measures gas with ratchet activation: pools activate progressively.
    ///      Thresholds are set so ~half the pools activate given the solved √q_0.
    function test_gas_100pools_ratchet() public {
        uint256 n = 100;
        uint256 budget = 100e18;
        (address[] memory pools, uint160[] memory sqrtRatios, uint160[] memory activations) = _createPools(n);

        // Set progressive activation thresholds.
        // Pool 0 must have activation = 0. Remaining pools get increasing thresholds.
        // sqrtQ current ≈ Q96/10 (since sqrtPrice = 10*Q96, zeroForOne → sqrtQ = Q96²/sqrtP = Q96/10).
        // Target √q_0 will be > current, so set thresholds that some pools are below it.
        uint160 sqrtQCurrent = uint160(Q96 / 10);
        for (uint256 i = 1; i < n; i++) {
            // Linear ramp: pool i activates at sqrtQCurrent + i * step
            // Use a step such that roughly pool 0..49 activate and 50..99 don't.
            activations[i] = sqrtQCurrent + uint160(i * (Q96 / 200));
        }

        mockRouter.setConsumeAmount(budget / n);

        tokenIn.mint(address(this), budget);
        tokenIn.approve(address(inOutRouter), budget);

        InOutRouter.SwapParams memory p = InOutRouter.SwapParams({
            pools:          pools,
            tokenIn:        address(tokenIn),
            amountIn:       budget,
            sqrtRatiosX96:  sqrtRatios,
            activationX96:  activations
        });

        uint256 gasBefore = gasleft();
        (, uint256 used) = inOutRouter.swapToRatio(p);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("gas_100_ratchet", gasUsed);
        emit log_named_uint("per_pool", gasUsed / n);
        emit log_named_uint("amountInUsed", used);
        // Some pools may not activate, but at least some should swap.
        assertGt(used, 0);
    }
}
