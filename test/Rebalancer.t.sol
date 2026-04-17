// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";

import {Rebalancer} from "../contracts/Rebalancer.sol";
import {IERC20} from "../contracts/interfaces/IERC20.sol";
import {IV3SwapRouter} from "../contracts/interfaces/IV3SwapRouter.sol";
import {FullMath} from "../contracts/libraries/FullMath.sol";
import {TickMath} from "../contracts/libraries/TickMath.sol";

contract MockERC20 {
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
        if (balanceOf[msg.sender] < amount) return false;
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        if (balanceOf[from] < amount) return false;
        uint256 allowed = allowance[from][msg.sender];
        if (allowed < amount) return false;

        allowance[from][msg.sender] = allowed - amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract MockPool {
    struct TickData {
        int128 liquidityNet;
        bool initialized;
    }

    uint160 internal currentSqrtPriceX96;
    int24 internal currentTick;
    int24 internal currentTickSpacing;
    uint128 internal currentLiquidity;
    mapping(int24 => TickData) internal tickData;
    mapping(int16 => uint256) internal bitmaps;

    constructor(uint160 sqrtPriceX96_, uint128 liquidity_) {
        currentSqrtPriceX96 = sqrtPriceX96_;
        currentLiquidity = liquidity_;
        currentTickSpacing = 100000;
    }

    function slot0() external view returns (uint160, int24, uint16, uint16, uint16, uint8, bool) {
        return (currentSqrtPriceX96, currentTick, 0, 0, 0, 0, false);
    }

    function liquidity() external view returns (uint128) {
        return currentLiquidity;
    }

    function tickSpacing() external view returns (int24) {
        return currentTickSpacing;
    }

    function tickBitmap(int16 wordPosition) external view returns (uint256) {
        return bitmaps[wordPosition];
    }

    function ticks(int24 tick_)
        external
        view
        returns (uint128, int128, uint256, uint256, int56, uint160, uint32, bool)
    {
        TickData memory data = tickData[tick_];
        return (0, data.liquidityNet, 0, 0, 0, 0, 0, data.initialized);
    }

    function setSqrtPrice(uint160 nextSqrtPriceX96) external {
        currentSqrtPriceX96 = nextSqrtPriceX96;
    }

    function setTick(int24 nextTick) external {
        currentTick = nextTick;
    }

    function setTickSpacing(int24 nextTickSpacing) external {
        currentTickSpacing = nextTickSpacing;
    }

    function setLiquidity(uint128 nextLiquidity) external {
        currentLiquidity = nextLiquidity;
    }

    function setTickData(int24 tick_, int128 liquidityNet_) external {
        tickData[tick_] = TickData({liquidityNet: liquidityNet_, initialized: true});

        int24 compressed = tick_ / currentTickSpacing;
        int16 wordPos = int16(compressed >> 8);
        uint8 bitPos = uint8(uint24(compressed % 256));
        bitmaps[wordPos] |= uint256(1) << bitPos;
    }
}

contract MockRouter {
    struct Config {
        uint256 fixedSpend;
        uint160 halfSpendThresholdX96;
        address poolToMove;
        uint160 nextSqrtPriceX96;
    }

    mapping(address => Config) internal configs;

    function configure(
        address tokenOut,
        uint256 fixedSpend,
        uint160 halfSpendThresholdX96,
        address poolToMove,
        uint160 nextSqrtPriceX96
    ) external {
        configs[tokenOut] = Config({
            fixedSpend: fixedSpend,
            halfSpendThresholdX96: halfSpendThresholdX96,
            poolToMove: poolToMove,
            nextSqrtPriceX96: nextSqrtPriceX96
        });
    }

    function exactInputSingle(IV3SwapRouter.ExactInputSingleParams calldata params)
        external
        payable
        returns (uint256 amountOut)
    {
        Config memory config = configs[params.tokenOut];

        uint256 spend = config.fixedSpend;
        if (spend == 0) {
            spend = params.amountIn;
            if (config.halfSpendThresholdX96 != 0 && params.sqrtPriceLimitX96 < config.halfSpendThresholdX96) {
                spend = params.amountIn / 2;
            }
        }
        if (spend > params.amountIn) {
            spend = params.amountIn;
        }

        bool success = MockERC20(params.tokenIn).transferFrom(msg.sender, address(this), spend);
        require(success, "transferFrom failed");

        MockERC20(params.tokenOut).mint(params.recipient, spend);

        if (config.poolToMove != address(0) && config.nextSqrtPriceX96 != 0) {
            MockPool(config.poolToMove).setSqrtPrice(config.nextSqrtPriceX96);
        }

        return spend;
    }

    function exactOutputSingle(IV3SwapRouter.ExactOutputSingleParams calldata)
        external
        payable
        returns (uint256 amountIn)
    {
        return amountIn;
    }
}

contract DynamicBuyRouter {
    uint256 internal constant Q96 = 1 << 96;
    uint256 internal constant FEE_UNITS = 1e6;

    struct Config {
        address pool;
        uint24 fee;
        bool enabled;
    }

    mapping(address => Config) internal configs;

    function configure(address tokenOut, address pool, uint24 fee) external {
        configs[tokenOut] = Config({pool: pool, fee: fee, enabled: true});
    }

    function exactInputSingle(IV3SwapRouter.ExactInputSingleParams calldata params)
        external
        payable
        returns (uint256 amountOut)
    {
        Config memory config = configs[params.tokenOut];
        require(config.enabled, "router not configured");

        MockPool pool = MockPool(config.pool);
        (uint160 sqrtPriceX96, int24 tick,,,,,) = pool.slot0();
        require(params.sqrtPriceLimitX96 > sqrtPriceX96, "buy-only benchmark");

        uint128 liquidity = pool.liquidity();
        int24 spacing = pool.tickSpacing();
        int24 lowerTick = _floorToSpacing(tick, spacing);
        uint256 remaining = params.amountIn;
        uint256 spent = 0;

        while (remaining > 0 && sqrtPriceX96 < params.sqrtPriceLimitX96 && liquidity > 0) {
            int24 upperTick = lowerTick + spacing;
            uint160 segmentCeiling = TickMath.getSqrtRatioAtTick(upperTick);
            uint160 segmentEnd = params.sqrtPriceLimitX96 < segmentCeiling ? params.sqrtPriceLimitX96 : segmentCeiling;
            uint256 segmentCost = _segmentCostToken0(sqrtPriceX96, segmentEnd, liquidity, config.fee);

            if (segmentCost <= remaining) {
                spent += segmentCost;
                remaining -= segmentCost;
                amountOut += _segmentOutToken0(sqrtPriceX96, segmentEnd, liquidity);
                sqrtPriceX96 = segmentEnd;

                if (segmentEnd == segmentCeiling && segmentEnd < params.sqrtPriceLimitX96) {
                    (, int128 liquidityNet,,,,,,) = pool.ticks(upperTick);
                    liquidity = _applyLiquidityNetAscending(liquidity, liquidityNet);
                    lowerTick = upperTick;
                    tick = upperTick;
                } else {
                    tick = lowerTick;
                }
            } else {
                uint160 partialEnd = _solvePartialEndToken0(sqrtPriceX96, segmentEnd, liquidity, config.fee, remaining);
                if (partialEnd == sqrtPriceX96) break;

                spent += remaining;
                amountOut += _segmentOutToken0(sqrtPriceX96, partialEnd, liquidity);
                sqrtPriceX96 = partialEnd;
                remaining = 0;
                tick = lowerTick;
            }
        }

        bool success = MockERC20(params.tokenIn).transferFrom(msg.sender, address(this), spent);
        require(success, "transferFrom failed");

        MockERC20(params.tokenOut).mint(params.recipient, amountOut);
        pool.setSqrtPrice(sqrtPriceX96);
        pool.setTick(tick);
        pool.setLiquidity(liquidity);
    }

    function exactOutputSingle(IV3SwapRouter.ExactOutputSingleParams calldata)
        external
        payable
        returns (uint256 amountIn)
    {
        return amountIn;
    }

    function _solvePartialEndToken0(uint160 start, uint160 end, uint128 liquidity, uint24 fee, uint256 budget)
        internal
        pure
        returns (uint160 result)
    {
        uint160 lo = start;
        uint160 hi = end;

        for (uint256 i = 0; i < 48; i++) {
            if (lo == hi) break;
            uint160 mid = uint160((uint256(lo) + uint256(hi) + 1) / 2);
            uint256 cost = _segmentCostToken0(start, mid, liquidity, fee);
            if (cost <= budget) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        result = lo;
    }

    function _segmentCostToken0(uint160 start, uint160 end, uint128 liquidity, uint24 fee)
        internal
        pure
        returns (uint256)
    {
        if (end <= start) return 0;

        uint256 noFee = FullMath.mulDiv(uint256(liquidity), uint256(end) - uint256(start), Q96);
        if (fee == 0) return noFee;

        return _mulDivRoundingUp(noFee, FEE_UNITS, FEE_UNITS - uint256(fee));
    }

    function _segmentOutToken0(uint160 start, uint160 end, uint128 liquidity) internal pure returns (uint256) {
        if (end <= start) return 0;

        return FullMath.mulDiv(uint256(liquidity), Q96, uint256(start))
            - FullMath.mulDiv(uint256(liquidity), Q96, uint256(end));
    }

    function _applyLiquidityNetAscending(uint128 liquidity, int128 liquidityNet) internal pure returns (uint128) {
        if (liquidityNet >= 0) {
            return liquidity + uint128(uint128(liquidityNet));
        }
        return liquidity - uint128(uint128(-liquidityNet));
    }

    function _floorToSpacing(int24 tick, int24 spacing) internal pure returns (int24) {
        int24 compressed = tick / spacing;
        if (tick < 0 && tick % spacing != 0) {
            compressed--;
        }
        return compressed * spacing;
    }

    function _mulDivRoundingUp(uint256 a, uint256 b, uint256 denominator) internal pure returns (uint256 result) {
        result = FullMath.mulDiv(a, b, denominator);
        if (mulmod(a, b, denominator) > 0) {
            result++;
        }
    }
}

contract MockCTFRouter {
    function splitPosition(address, address, uint256) external {}
    function mergePositions(address, address, uint256) external {}
}

contract RebalancerHarness is Rebalancer {
    constructor() Rebalancer(address(1), address(2)) {}

    function buildBuyOrder(
        uint160[] calldata sqrtPrices,
        uint160[] calldata sqrtPredX96,
        bool[] calldata isToken1,
        uint256 num,
        uint256 den,
        bool buyAll
    ) external pure returns (uint256[] memory order) {
        uint160[] memory current = sqrtPrices;
        PsiResult memory psi = PsiResult({num: num, den: den, buyAll: buyAll});
        (uint256[] memory fullOrder,, uint256 count) = _buildBuyPlan(current, sqrtPredX96, isToken1, psi);

        order = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            order[i] = fullOrder[i];
        }
    }

    function compareProducts(uint256 a, uint256 b, uint256 c, uint256 d) external pure returns (int8) {
        return _mulCompare(a, b, c, d);
    }

    function sellLimit(uint160 sqrtPred, bool isToken1, uint24 fee) external pure returns (uint160) {
        return _sellLimit(sqrtPred, isToken1, fee);
    }

    function recycleWorthwhile(uint160 sqrtPrice, uint160 sqrtPred, bool isToken1, uint256 num, uint256 den, uint24 fee)
        external
        pure
        returns (bool)
    {
        PsiResult memory psi = PsiResult({num: num, den: den, buyAll: false});
        return _isRecycleWorthwhile(sqrtPrice, sqrtPred, isToken1, psi, fee);
    }

    function exactCost(
        address pool,
        uint160 sqrtPrice,
        int24 tick,
        uint128 liquidity,
        uint160 limit,
        bool isToken1,
        uint24 fee,
        uint256 maxTickCrossingsPerPool
    ) external view returns (uint256) {
        return _exactCostToLimit(pool, sqrtPrice, tick, liquidity, limit, isToken1, fee, maxTickCrossingsPerPool);
    }

    function recyclePotentialGain(
        RebalanceParams calldata params,
        uint160[] calldata sqrtPrices,
        uint256 num,
        uint256 den
    ) external view returns (uint256) {
        uint160[] memory current = sqrtPrices;
        PsiResult memory psi = PsiResult({num: num, den: den, buyAll: false});
        return _recyclePotentialGain(params, current, psi);
    }

    function recycleWithFloor(RebalanceParams calldata params, uint256 maxRounds, uint256 minRecycleProfitCollateral)
        external
        returns (uint256 totalRecycled, uint256 totalRedeployed)
    {
        return _recycleSellWithFloor(params, maxRounds, minRecycleProfitCollateral);
    }
}

contract RebalancerBuyAllHarness is Rebalancer {
    constructor(address router_, address ctfRouter_) Rebalancer(router_, ctfRouter_) {}

    function executeExactBuyAllFast(RebalanceParams calldata params, uint160[] calldata sqrtPrices)
        external
        returns (uint256)
    {
        uint160[] memory current = sqrtPrices;
        PsiResult memory psi = PsiResult({num: 1, den: 1, buyAll: true});
        return _waterfallBuy(params, current, psi);
    }

    function executeExactBuyAllLegacy(RebalanceParams calldata params, uint160[] calldata sqrtPrices)
        external
        returns (uint256)
    {
        uint160[] memory current = sqrtPrices;
        PsiResult memory psi = PsiResult({num: 1, den: 1, buyAll: true});
        uint256 budgetBefore = IERC20(params.collateral).balanceOf(address(this));
        if (budgetBefore == 0) return 0;

        (uint256[] memory order, uint160[] memory limits, uint256 count) =
            _buildBuyPlan(current, params.sqrtPredX96, params.isToken1, psi);
        if (count == 0) return 0;

        return _executeBuyPlan(params, order, limits, count, budgetBefore);
    }
}

contract RebalancerTest is Test {
    uint160 internal constant TEST_Q96 = uint160(1 << 96);
    uint256 internal constant TEST_FEE_UNITS = 1e6;

    RebalancerHarness internal harness;

    function setUp() public {
        harness = new RebalancerHarness();
    }

    function _sqrt(uint256 x) internal pure returns (uint256 z) {
        if (x == 0) return 0;
        z = x;
        uint256 y = (z + 1) / 2;
        while (y < z) {
            z = y;
            y = (x / z + z) / 2;
        }
    }

    function _expectedSellLimit(uint160 sqrtPred, bool isToken1, uint24 fee) internal pure returns (uint160) {
        uint256 feeComp = TEST_FEE_UNITS - uint256(fee);
        uint256 sqrtFeeComp = _sqrt(feeComp * TEST_FEE_UNITS);
        if (sqrtFeeComp == 0) return sqrtPred;

        if (isToken1) {
            return uint160(FullMath.mulDiv(uint256(sqrtPred), sqrtFeeComp, TEST_FEE_UNITS));
        }

        return uint160(FullMath.mulDiv(uint256(sqrtPred), TEST_FEE_UNITS, sqrtFeeComp));
    }

    function _fundCallerCollateral(MockERC20 collateral, address spender, uint256 amount) internal {
        if (amount == 0) return;
        collateral.mint(address(this), amount);
        collateral.approve(spender, amount);
    }

    function _buildUniformConstantLParams(uint256 outcomeCount, uint160 sqrtPred)
        internal
        returns (MockERC20 collateral, Rebalancer rebalancer, Rebalancer.RebalanceParams memory params)
    {
        collateral = new MockERC20();
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        rebalancer = new Rebalancer(address(router), address(ctfRouter));

        address[] memory tokens = new address[](outcomeCount);
        address[] memory pools = new address[](outcomeCount);
        bool[] memory isToken1 = new bool[](outcomeCount);
        uint256[] memory balances = new uint256[](outcomeCount);
        uint160[] memory sqrtPredX96 = new uint160[](outcomeCount);

        for (uint256 i = 0; i < outcomeCount; i++) {
            MockERC20 token = new MockERC20();
            MockPool pool = new MockPool(TEST_Q96, 100);

            tokens[i] = address(token);
            pools[i] = address(pool);
            sqrtPredX96[i] = sqrtPred;

            router.configure(address(token), 1, 0, address(0), 0);
        }

        _fundCallerCollateral(collateral, address(rebalancer), outcomeCount);

        params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: outcomeCount,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });
    }

    function _buildUniformExactParams(uint256 outcomeCount, int24 targetTick, bool addInitializedTick)
        internal
        returns (
            MockERC20 collateral,
            Rebalancer rebalancer,
            Rebalancer.RebalanceParams memory params,
            uint256 maxTickCrossingsPerPool
        )
    {
        collateral = new MockERC20();
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        rebalancer = new Rebalancer(address(router), address(ctfRouter));

        address[] memory tokens = new address[](outcomeCount);
        address[] memory pools = new address[](outcomeCount);
        bool[] memory isToken1 = new bool[](outcomeCount);
        uint256[] memory balances = new uint256[](outcomeCount);
        uint160[] memory sqrtPredX96 = new uint160[](outcomeCount);
        uint160 target = TickMath.getSqrtRatioAtTick(targetTick);

        for (uint256 i = 0; i < outcomeCount; i++) {
            MockERC20 token = new MockERC20();
            MockPool pool = new MockPool(TEST_Q96, 1_000_000);
            pool.setTick(0);

            if (addInitializedTick) {
                pool.setTickSpacing(10);
                pool.setTickData(10, int128(uint128(1_000_000)));
                maxTickCrossingsPerPool = 8;
            } else {
                maxTickCrossingsPerPool = 8;
            }

            tokens[i] = address(token);
            pools[i] = address(pool);
            sqrtPredX96[i] = target;

            router.configure(address(token), 1, 0, address(0), 0);
        }

        _fundCallerCollateral(collateral, address(rebalancer), outcomeCount);

        params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: outcomeCount,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });
    }

    function _buildBuyAllFastPathScenario(uint256 outcomeCount)
        internal
        returns (
            RebalancerBuyAllHarness fastHarness,
            RebalancerBuyAllHarness legacyHarness,
            Rebalancer.RebalanceParams memory params,
            uint160[] memory sqrtPrices
        )
    {
        MockERC20 collateral = new MockERC20();
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        fastHarness = new RebalancerBuyAllHarness(address(router), address(ctfRouter));
        legacyHarness = new RebalancerBuyAllHarness(address(router), address(ctfRouter));

        address[] memory tokens = new address[](outcomeCount);
        address[] memory pools = new address[](outcomeCount);
        bool[] memory isToken1 = new bool[](outcomeCount);
        uint256[] memory balances = new uint256[](outcomeCount);
        uint160[] memory sqrtPredX96 = new uint160[](outcomeCount);
        sqrtPrices = new uint160[](outcomeCount);

        for (uint256 i = 0; i < outcomeCount; i++) {
            MockERC20 token = new MockERC20();
            MockPool pool = new MockPool(TEST_Q96, 100);

            tokens[i] = address(token);
            pools[i] = address(pool);
            sqrtPredX96[i] = uint160(uint256(TEST_Q96) * 2);
            sqrtPrices[i] = TEST_Q96;

            router.configure(address(token), 1, 0, address(0), 0);
        }

        collateral.mint(address(fastHarness), outcomeCount);
        collateral.mint(address(legacyHarness), outcomeCount);

        params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 0,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });
    }

    function _runConstantLGasScenario(uint256 outcomeCount) internal {
        vm.pauseGasMetering();
        (MockERC20 collateral, Rebalancer rebalancer, Rebalancer.RebalanceParams memory params) =
            _buildUniformConstantLParams(outcomeCount, uint160(uint256(TEST_Q96) * 2));

        vm.resumeGasMetering();
        rebalancer.rebalance(params);
        vm.pauseGasMetering();

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function _runExactGasScenario(uint256 outcomeCount, bool addInitializedTick) internal {
        vm.pauseGasMetering();
        (
            MockERC20 collateral,
            Rebalancer rebalancer,
            Rebalancer.RebalanceParams memory params,
            uint256 maxTickCrossingsPerPool
        ) = _buildUniformExactParams(outcomeCount, 20, addInitializedTick);

        vm.resumeGasMetering();
        rebalancer.rebalanceExact(params, 24, maxTickCrossingsPerPool);
        vm.pauseGasMetering();

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function _segmentCostToken0Benchmark(uint160 start, uint160 end, uint128 liquidity)
        internal
        pure
        returns (uint256)
    {
        if (end <= start) return 0;
        return FullMath.mulDiv(uint256(liquidity), uint256(end) - uint256(start), uint256(TEST_Q96));
    }

    function _token0PriceAt(uint160 sqrtPriceX96) internal pure returns (uint256) {
        return FullMath.mulDiv(
            FullMath.mulDiv(uint256(sqrtPriceX96), uint256(sqrtPriceX96), uint256(TEST_Q96)), 1e18, uint256(TEST_Q96)
        );
    }

    function _evWithTwoTokens(MockERC20 collateral, MockERC20 tokenA, MockERC20 tokenB, uint160 sqrtPredX96)
        internal
        view
        returns (uint256)
    {
        uint256 price = _token0PriceAt(sqrtPredX96);
        uint256 tokenValueA = FullMath.mulDiv(tokenA.balanceOf(address(this)), price, 1e18);
        uint256 tokenValueB = FullMath.mulDiv(tokenB.balanceOf(address(this)), price, 1e18);
        return collateral.balanceOf(address(this)) + tokenValueA + tokenValueB;
    }

    function _evFromParams(MockERC20 collateral, Rebalancer.RebalanceParams memory params, uint160 sqrtPredX96)
        internal
        view
        returns (uint256 ev)
    {
        uint256 price = _token0PriceAt(sqrtPredX96);
        ev = collateral.balanceOf(address(this));

        for (uint256 i = 0; i < params.tokens.length; i++) {
            ev += FullMath.mulDiv(MockERC20(params.tokens[i]).balanceOf(address(this)), price, 1e18);
        }
    }

    function _countTouchedPools(Rebalancer.RebalanceParams memory params) internal view returns (uint256 count) {
        for (uint256 i = 0; i < params.tokens.length; i++) {
            if (MockERC20(params.tokens[i]).balanceOf(address(this)) > 0) {
                count++;
            }
        }
    }

    function _buildABMultiTickTwoPoolScenario()
        internal
        returns (
            MockERC20 collateral,
            MockERC20 tokenA,
            MockERC20 tokenB,
            Rebalancer rebalancer,
            Rebalancer.RebalanceParams memory params,
            uint160 sqrtPredX96,
            uint256 actualTotalCost,
            uint256 estimatedTotalCost
        )
    {
        collateral = new MockERC20();
        tokenA = new MockERC20();
        tokenB = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 1_000_000 ether);
        MockPool poolB = new MockPool(TEST_Q96, 1_000_000 ether);
        DynamicBuyRouter router = new DynamicBuyRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        rebalancer = new Rebalancer(address(router), address(ctfRouter));

        poolA.setTick(0);
        poolB.setTick(0);
        poolA.setTickSpacing(10);
        poolB.setTickSpacing(10);

        uint128 runningLiquidity = 1_000_000 ether;
        for (int24 tickBoundary = 10; tickBoundary <= 60; tickBoundary += 10) {
            uint128 nextLiquidity = runningLiquidity + 1_000_000 ether;
            int128 liquidityNet = int128(uint128(nextLiquidity - runningLiquidity));
            poolA.setTickData(tickBoundary, liquidityNet);
            runningLiquidity = nextLiquidity;
        }

        sqrtPredX96 = TickMath.getSqrtRatioAtTick(65);
        uint256 singleEstimatedCost = _segmentCostToken0Benchmark(TEST_Q96, sqrtPredX96, 1_000_000 ether);
        estimatedTotalCost = singleEstimatedCost * 2;
        actualTotalCost = harness.exactCost(address(poolA), TEST_Q96, 0, 1_000_000 ether, sqrtPredX96, false, 0, 256)
            + harness.exactCost(address(poolB), TEST_Q96, 0, 1_000_000 ether, sqrtPredX96, false, 0, 256);

        uint256 budget = estimatedTotalCost;
        _fundCallerCollateral(collateral, address(rebalancer), budget);
        router.configure(address(tokenA), address(poolA), 0);
        router.configure(address(tokenB), address(poolB), 0);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);

        address[] memory pools = new address[](2);
        pools[0] = address(poolA);
        pools[1] = address(poolB);

        bool[] memory isToken1 = new bool[](2);
        uint256[] memory balances = new uint256[](2);
        uint160[] memory sqrtPreds = new uint160[](2);
        sqrtPreds[0] = sqrtPredX96;
        sqrtPreds[1] = sqrtPredX96;

        params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: budget,
            sqrtPredX96: sqrtPreds,
            collateral: address(collateral),
            fee: 0
        });
    }

    function _buildABMultiTickSyntheticNinetyEightOutcomeScenario()
        internal
        returns (
            MockERC20 collateral,
            Rebalancer rebalancer,
            Rebalancer.RebalanceParams memory params,
            uint160 sqrtPredX96,
            uint256 actualTotalCost,
            uint256 estimatedTotalCost
        )
    {
        uint256 outcomeCount = 98;
        uint128 baseLiquidity = 1_000_000 ether;
        DynamicBuyRouter router = new DynamicBuyRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        rebalancer = new Rebalancer(address(router), address(ctfRouter));
        collateral = new MockERC20();

        address[] memory tokens = new address[](outcomeCount);
        address[] memory pools = new address[](outcomeCount);
        bool[] memory isToken1 = new bool[](outcomeCount);
        uint256[] memory balances = new uint256[](outcomeCount);
        uint160[] memory sqrtPreds = new uint160[](outcomeCount);

        sqrtPredX96 = TickMath.getSqrtRatioAtTick(65);
        uint256 singleEstimatedCost = _segmentCostToken0Benchmark(TEST_Q96, sqrtPredX96, baseLiquidity);
        uint256 singleActualCost = 0;

        for (uint256 i = 0; i < outcomeCount; i++) {
            MockERC20 token = new MockERC20();
            MockPool pool = new MockPool(TEST_Q96, baseLiquidity);

            pool.setTick(0);
            pool.setTickSpacing(10);

            uint128 runningLiquidity = baseLiquidity;
            for (int24 tickBoundary = 10; tickBoundary <= 60; tickBoundary += 10) {
                uint128 nextLiquidity = runningLiquidity + baseLiquidity;
                int128 liquidityNet = int128(uint128(nextLiquidity - runningLiquidity));
                pool.setTickData(tickBoundary, liquidityNet);
                runningLiquidity = nextLiquidity;
            }

            if (i == 0) {
                singleActualCost =
                    harness.exactCost(address(pool), TEST_Q96, 0, baseLiquidity, sqrtPredX96, false, 0, 256);
            }

            router.configure(address(token), address(pool), 0);
            tokens[i] = address(token);
            pools[i] = address(pool);
            sqrtPreds[i] = sqrtPredX96;
        }

        estimatedTotalCost = singleEstimatedCost * outcomeCount;
        actualTotalCost = singleActualCost * outcomeCount;

        _fundCallerCollateral(collateral, address(rebalancer), estimatedTotalCost);

        params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: estimatedTotalCost,
            sqrtPredX96: sqrtPreds,
            collateral: address(collateral),
            fee: 0
        });
    }

    function _buildABMultiTickRealisticSeededNinetyEightOutcomeScenario()
        internal
        returns (
            MockERC20 collateral,
            Rebalancer rebalancer,
            Rebalancer.RebalanceParams memory params,
            uint160 sqrtPredX96,
            uint256 actualTotalCost,
            uint256 estimatedTotalCost
        )
    {
        uint256 outcomeCount = 98;
        uint128 baseLiquidity = 1_000_000 ether;
        DynamicBuyRouter router = new DynamicBuyRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        rebalancer = new Rebalancer(address(router), address(ctfRouter));
        collateral = new MockERC20();

        address[] memory tokens = new address[](outcomeCount);
        address[] memory pools = new address[](outcomeCount);
        bool[] memory isToken1 = new bool[](outcomeCount);
        uint256[] memory balances = new uint256[](outcomeCount);
        uint160[] memory sqrtPreds = new uint160[](outcomeCount);

        // Seeded from the dominant positive-side L1 ladder shape:
        // spot-adjacent sentinel near 512, main range boundary near 16095, far outer sentinel near 92108.
        // We snap onto a 512 grid so the benchmark remains tractable on-chain.
        int24 spacing = 512;
        int24 bandStart = 512;
        int24 bandEnd = 16384;
        int24 outerSentinel = 92160;
        sqrtPredX96 = TickMath.getSqrtRatioAtTick(17408);
        uint256 singleEstimatedCost = _segmentCostToken0Benchmark(TEST_Q96, sqrtPredX96, baseLiquidity);
        uint256 singleActualCost = 0;

        for (uint256 i = 0; i < outcomeCount; i++) {
            MockERC20 token = new MockERC20();
            MockPool pool = new MockPool(TEST_Q96, baseLiquidity);

            pool.setTick(0);
            pool.setTickSpacing(spacing);
            pool.setTickData(bandStart, int128(uint128((baseLiquidity * 3) / 4)));
            pool.setTickData(bandEnd, -int128(uint128((baseLiquidity * 3) / 4)));
            pool.setTickData(outerSentinel, 0);

            if (i == 0) {
                singleActualCost =
                    harness.exactCost(address(pool), TEST_Q96, 0, baseLiquidity, sqrtPredX96, false, 0, 64);
            }

            router.configure(address(token), address(pool), 0);
            tokens[i] = address(token);
            pools[i] = address(pool);
            sqrtPreds[i] = sqrtPredX96;
        }

        estimatedTotalCost = singleEstimatedCost * outcomeCount;
        actualTotalCost = singleActualCost * outcomeCount;

        _fundCallerCollateral(collateral, address(rebalancer), estimatedTotalCost);

        params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: estimatedTotalCost,
            sqrtPredX96: sqrtPreds,
            collateral: address(collateral),
            fee: 0
        });
    }

    function testBuildBuyOrderSortsToken0CandidatesByCurrentProfitability() public {
        uint160[] memory sqrtPrices = new uint160[](3);
        sqrtPrices[0] = 80;
        sqrtPrices[1] = 100;
        sqrtPrices[2] = 50;

        uint160[] memory sqrtPredX96 = new uint160[](3);
        sqrtPredX96[0] = 120;
        sqrtPredX96[1] = 200;
        sqrtPredX96[2] = 90;

        bool[] memory isToken1 = new bool[](3);

        uint256[] memory order = harness.buildBuyOrder(sqrtPrices, sqrtPredX96, isToken1, 1, 1, true);

        assertEq(order.length, 3);
        assertEq(order[0], 1);
        assertEq(order[1], 2);
        assertEq(order[2], 0);
    }

    function testBuildBuyOrderSortsMixedPoolOrientations() public {
        uint160[] memory sqrtPrices = new uint160[](2);
        sqrtPrices[0] = 50;
        sqrtPrices[1] = 300;

        uint160[] memory sqrtPredX96 = new uint160[](2);
        sqrtPredX96[0] = 100;
        sqrtPredX96[1] = 100;

        bool[] memory isToken1 = new bool[](2);
        isToken1[1] = true;

        uint256[] memory order = harness.buildBuyOrder(sqrtPrices, sqrtPredX96, isToken1, 1, 1, true);

        assertEq(order.length, 2);
        assertEq(order[0], 1);
        assertEq(order[1], 0);
    }

    function testCompareProductsHandlesLargeOperandsWithoutOverflow() public {
        uint256 a = uint256(1) << 200;
        uint256 b = uint256(1) << 100;
        uint256 c = uint256(1) << 199;
        uint256 d = uint256(1) << 100;
        uint256 e = uint256(1) << 201;
        uint256 f = uint256(1) << 99;

        assertEq(int256(harness.compareProducts(a, b, c, d)), 1);
        assertEq(int256(harness.compareProducts(c, d, a, b)), -1);
        assertEq(int256(harness.compareProducts(a, b, e, f)), 0);
    }

    function testSellLimitUsesFeeNeutralBoundaryForToken1() public {
        assertEq(harness.sellLimit(1000, true, 100), _expectedSellLimit(1000, true, 100));
    }

    function testSellLimitUsesFeeNeutralBoundaryForToken0() public {
        assertEq(harness.sellLimit(1000, false, 100), _expectedSellLimit(1000, false, 100));
    }

    function testSellLimitKeepsPrecisionForFiveBpsPools() public {
        uint160 sqrtPred = 1_000_000;
        uint24 fee = 500;

        uint160 token1Limit = harness.sellLimit(sqrtPred, true, fee);
        uint160 token0Limit = harness.sellLimit(sqrtPred, false, fee);
        uint160 coarseToken1Limit = uint160(FullMath.mulDiv(uint256(sqrtPred), 999, 1000));
        uint160 coarseToken0Limit = uint160(FullMath.mulDiv(uint256(sqrtPred), 1000, 999));

        assertEq(token1Limit, _expectedSellLimit(sqrtPred, true, fee));
        assertEq(token0Limit, _expectedSellLimit(sqrtPred, false, fee));
        assertGt(token1Limit, coarseToken1Limit);
        assertLt(token0Limit, coarseToken0Limit);
    }

    function testRecycleWorthwhileRejectsThinToken1Edge() public {
        bool worthwhile = harness.recycleWorthwhile(1005, 1000, true, 1000, 1010, 5_000);
        assertFalse(worthwhile);
    }

    function testRecycleWorthwhileAcceptsToken1EdgeBeyondRoundTripFees() public {
        bool worthwhile = harness.recycleWorthwhile(1000, 1000, true, 1000, 1010, 5_000);
        assertTrue(worthwhile);
    }

    function testRecycleFloorSkipsBelowThresholdHoldings() public {
        RebalancerHarness recycleHarness = new RebalancerHarness();
        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockERC20 tokenB = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 100);
        MockPool poolB = new MockPool(uint160(uint256(TEST_Q96) * 14 / 10), 100);

        collateral.mint(address(recycleHarness), 10);
        tokenB.mint(address(recycleHarness), 10);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);

        address[] memory pools = new address[](2);
        pools[0] = address(poolA);
        pools[1] = address(poolB);

        bool[] memory isToken1 = new bool[](2);
        uint256[] memory balances = new uint256[](2);
        uint160[] memory sqrtPredX96 = new uint160[](2);
        sqrtPredX96[0] = uint160(uint256(TEST_Q96) * 2);
        sqrtPredX96[1] = uint160(uint256(TEST_Q96) * 15 / 10);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 0,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        (uint256 skippedRecycled, uint256 skippedRedeployed) =
            recycleHarness.recycleWithFloor(params, 1, type(uint256).max);
        assertEq(skippedRecycled, 0);
        assertEq(skippedRedeployed, 0);

        vm.expectRevert();
        recycleHarness.recycleWithFloor(params, 1, 0);
    }

    function testConstantLSolverExecutesStaleSecondLegWithinPass() public {
        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockERC20 tokenB = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 100);
        MockPool poolB = new MockPool(TEST_Q96, 100);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        router.configure(address(tokenA), 80, 0, address(poolA), uint160(uint256(TEST_Q96) * 2));
        router.configure(address(tokenB), 0, uint160(uint256(TEST_Q96) * 5 / 4), address(0), 0);

        _fundCallerCollateral(collateral, address(rebalancer), 100);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);

        address[] memory pools = new address[](2);
        pools[0] = address(poolA);
        pools[1] = address(poolB);

        bool[] memory isToken1 = new bool[](2);
        uint256[] memory balances = new uint256[](2);

        uint160[] memory sqrtPredX96 = new uint160[](2);
        sqrtPredX96[0] = uint160(uint256(TEST_Q96) * 2);
        sqrtPredX96[1] = uint160(uint256(TEST_Q96) * 3 / 2);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 100,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        rebalancer.rebalance(params);

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function testExactSolverSmokeExecutesThirdEntryPoint() public {
        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 100);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        router.configure(address(tokenA), 10, 0, address(0), 0);

        _fundCallerCollateral(collateral, address(rebalancer), 100);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenA);

        address[] memory pools = new address[](1);
        pools[0] = address(poolA);

        bool[] memory isToken1 = new bool[](1);
        uint256[] memory balances = new uint256[](1);

        uint160[] memory sqrtPredX96 = new uint160[](1);
        sqrtPredX96[0] = uint160(uint256(TEST_Q96) * 2);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 100,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        rebalancer.rebalanceExact(params, 16, 0);

        assertEq(collateral.balanceOf(address(this)), 90);
    }

    function testExactSolverSmokeExecutesArbEntryPointWithFloors() public {
        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 100);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        router.configure(address(tokenA), 10, 0, address(0), 0);
        _fundCallerCollateral(collateral, address(rebalancer), 100);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenA);

        address[] memory pools = new address[](1);
        pools[0] = address(poolA);

        bool[] memory isToken1 = new bool[](1);
        uint256[] memory balances = new uint256[](1);
        uint160[] memory sqrtPredX96 = new uint160[](1);
        sqrtPredX96[0] = uint160(uint256(TEST_Q96) * 2);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 100,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        rebalancer.rebalanceAndArbExactWithFloors(params, address(0), 0, 0, 24, 8, 0, 0);

        assertEq(collateral.balanceOf(address(this)), 90);
    }

    function testRebalanceAndArbWithFloorsSkipsSubThresholdArb() public {
        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockERC20 tokenB = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 100);
        MockPool poolB = new MockPool(TEST_Q96, 100);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        _fundCallerCollateral(collateral, address(rebalancer), 100);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);

        address[] memory pools = new address[](2);
        pools[0] = address(poolA);
        pools[1] = address(poolB);

        bool[] memory isToken1 = new bool[](2);
        uint256[] memory balances = new uint256[](2);
        uint160[] memory sqrtPredX96 = new uint160[](2);
        sqrtPredX96[0] = TEST_Q96;
        sqrtPredX96[1] = TEST_Q96;

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 100,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        rebalancer.rebalanceAndArbWithFloors(params, address(0), 1, 0, 101, 0);

        assertEq(collateral.balanceOf(address(this)), 100);
        assertEq(tokenA.balanceOf(address(this)), 0);
        assertEq(tokenB.balanceOf(address(this)), 0);
    }

    function testRebalanceNoOpWhenPredictionInsideCurrentTick() public {
        MockERC20 collateral = new MockERC20();
        MockERC20 token = new MockERC20();
        MockPool pool = new MockPool(TEST_Q96, 100);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        pool.setTick(0);
        pool.setTickSpacing(10);
        router.configure(address(token), 1, 0, address(0), 0);
        _fundCallerCollateral(collateral, address(rebalancer), 1);

        address[] memory tokens = new address[](1);
        tokens[0] = address(token);
        address[] memory pools = new address[](1);
        pools[0] = address(pool);
        bool[] memory isToken1 = new bool[](1);
        uint256[] memory balances = new uint256[](1);
        uint160[] memory sqrtPredX96 = new uint160[](1);
        sqrtPredX96[0] = TickMath.getSqrtRatioAtTick(5);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 1,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        rebalancer.rebalance(params);

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function testExactSolverUsesLiquidityNetAcrossInitializedTick() public {
        uint128 liquidity = 1_000_000;
        uint160 fullTarget = TickMath.getSqrtRatioAtTick(20);
        uint160 conservativeThreshold = TickMath.getSqrtRatioAtTick(19);

        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, liquidity);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        poolA.setTick(0);
        poolA.setTickSpacing(10);

        uint256 costWithoutBoundary = harness.exactCost(address(poolA), TEST_Q96, 0, liquidity, fullTarget, false, 0, 8);

        poolA.setTickData(10, int128(uint128(liquidity)));

        uint256 costWithBoundary = harness.exactCost(address(poolA), TEST_Q96, 0, liquidity, fullTarget, false, 0, 8);
        uint256 budget = costWithoutBoundary + 1;

        assertGt(costWithBoundary, costWithoutBoundary);
        assertLt(costWithoutBoundary, budget);
        assertGt(costWithBoundary, budget);

        router.configure(address(tokenA), 0, conservativeThreshold, address(0), 0);
        _fundCallerCollateral(collateral, address(rebalancer), budget);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenA);

        address[] memory pools = new address[](1);
        pools[0] = address(poolA);

        bool[] memory isToken1 = new bool[](1);
        uint256[] memory balances = new uint256[](1);

        uint160[] memory sqrtPredX96 = new uint160[](1);
        sqrtPredX96[0] = fullTarget;

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: budget,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        rebalancer.rebalanceExact(params, 24, 8);

        assertEq(collateral.balanceOf(address(this)), budget - (budget / 2));
    }

    function testGasProfileConstantLTwoOutcomeSingleTick() public {
        vm.pauseGasMetering();

        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockERC20 tokenB = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 100);
        MockPool poolB = new MockPool(TEST_Q96, 100);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        router.configure(address(tokenA), 80, 0, address(poolA), uint160(uint256(TEST_Q96) * 2));
        router.configure(address(tokenB), 20, 0, address(0), 0);
        _fundCallerCollateral(collateral, address(rebalancer), 100);

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);

        address[] memory pools = new address[](2);
        pools[0] = address(poolA);
        pools[1] = address(poolB);

        bool[] memory isToken1 = new bool[](2);
        uint256[] memory balances = new uint256[](2);

        uint160[] memory sqrtPredX96 = new uint160[](2);
        sqrtPredX96[0] = uint160(uint256(TEST_Q96) * 2);
        sqrtPredX96[1] = uint160(uint256(TEST_Q96) * 3 / 2);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 100,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        vm.resumeGasMetering();
        rebalancer.rebalance(params);
        vm.pauseGasMetering();

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function testGasProfileExactSingleTickCostSolve() public {
        vm.pauseGasMetering();

        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 1_000_000);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        poolA.setTick(0);
        router.configure(address(tokenA), 1, 0, address(0), 0);
        _fundCallerCollateral(collateral, address(rebalancer), 1);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenA);

        address[] memory pools = new address[](1);
        pools[0] = address(poolA);

        bool[] memory isToken1 = new bool[](1);
        uint256[] memory balances = new uint256[](1);

        uint160[] memory sqrtPredX96 = new uint160[](1);
        sqrtPredX96[0] = TickMath.getSqrtRatioAtTick(20);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 1,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        vm.resumeGasMetering();
        rebalancer.rebalanceExact(params, 24, 8);
        vm.pauseGasMetering();

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function testGasExactBuyAllFastPathBeatsLegacySortedPlan() public {
        vm.pauseGasMetering();
        (
            RebalancerBuyAllHarness fastHarness,
            RebalancerBuyAllHarness legacyHarness,
            Rebalancer.RebalanceParams memory params,
            uint160[] memory sqrtPrices
        ) = _buildBuyAllFastPathScenario(98);

        vm.resumeGasMetering();
        uint256 gasStart = gasleft();
        uint256 fastSpent = fastHarness.executeExactBuyAllFast(params, sqrtPrices);
        uint256 fastGas = gasStart - gasleft();

        gasStart = gasleft();
        uint256 legacySpent = legacyHarness.executeExactBuyAllLegacy(params, sqrtPrices);
        uint256 legacyGas = gasStart - gasleft();
        vm.pauseGasMetering();

        emit log_named_uint("exact_buy_all_fast_gas", fastGas);
        emit log_named_uint("exact_buy_all_legacy_gas", legacyGas);

        assertEq(fastSpent, legacySpent);
        assertEq(fastSpent, 98);
        assertLt(fastGas, legacyGas);
    }

    function testGasProfileExactInitializedTickCrossing() public {
        vm.pauseGasMetering();

        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 1_000_000);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        poolA.setTick(0);
        poolA.setTickSpacing(10);
        poolA.setTickData(10, int128(uint128(1_000_000)));
        router.configure(address(tokenA), 1, 0, address(0), 0);
        _fundCallerCollateral(collateral, address(rebalancer), 1);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenA);

        address[] memory pools = new address[](1);
        pools[0] = address(poolA);

        bool[] memory isToken1 = new bool[](1);
        uint256[] memory balances = new uint256[](1);

        uint160[] memory sqrtPredX96 = new uint160[](1);
        sqrtPredX96[0] = TickMath.getSqrtRatioAtTick(20);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 1,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        vm.resumeGasMetering();
        rebalancer.rebalanceExact(params, 24, 8);
        vm.pauseGasMetering();

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function testGasProfileExactSparseBitmapJump() public {
        vm.pauseGasMetering();

        MockERC20 collateral = new MockERC20();
        MockERC20 tokenA = new MockERC20();
        MockPool poolA = new MockPool(TEST_Q96, 1_000_000);
        MockRouter router = new MockRouter();
        MockCTFRouter ctfRouter = new MockCTFRouter();
        Rebalancer rebalancer = new Rebalancer(address(router), address(ctfRouter));

        poolA.setTick(0);
        poolA.setTickSpacing(10);
        poolA.setTickData(3000, int128(uint128(1_000_000)));
        router.configure(address(tokenA), 1, 0, address(0), 0);
        _fundCallerCollateral(collateral, address(rebalancer), 1);

        address[] memory tokens = new address[](1);
        tokens[0] = address(tokenA);

        address[] memory pools = new address[](1);
        pools[0] = address(poolA);

        bool[] memory isToken1 = new bool[](1);
        uint256[] memory balances = new uint256[](1);

        uint160[] memory sqrtPredX96 = new uint160[](1);
        sqrtPredX96[0] = TickMath.getSqrtRatioAtTick(3010);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 1,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        vm.resumeGasMetering();
        rebalancer.rebalanceExact(params, 24, 512);
        vm.pauseGasMetering();

        assertEq(collateral.balanceOf(address(this)), 0);
    }

    function testGasProfileConstantLEightOutcomeSingleTick() public {
        _runConstantLGasScenario(8);
    }

    function testGasProfileConstantLSixteenOutcomeSingleTick() public {
        _runConstantLGasScenario(16);
    }

    function testGasProfileConstantLThirtyTwoOutcomeSingleTick() public {
        _runConstantLGasScenario(32);
    }

    function testGasProfileConstantLNinetyEightOutcomeSingleTick() public {
        _runConstantLGasScenario(98);
    }

    function testGasProfileExactEightOutcomeSingleTick() public {
        _runExactGasScenario(8, false);
    }

    function testGasProfileExactSixteenOutcomeSingleTick() public {
        _runExactGasScenario(16, false);
    }

    function testGasProfileExactThirtyTwoOutcomeSingleTick() public {
        _runExactGasScenario(32, false);
    }

    function testGasProfileExactNinetyEightOutcomeSingleTick() public {
        _runExactGasScenario(98, false);
    }

    function testBenchmarkABMultiTickConstantLVsExact() public {
        (
            MockERC20 constantCollateral,
            MockERC20 constantTokenA,
            MockERC20 constantTokenB,
            Rebalancer constantRebalancer,
            Rebalancer.RebalanceParams memory constantParams,
            uint160 sqrtPredX96,
            uint256 actualTotalCost,
            uint256 estimatedTotalCost
        ) = _buildABMultiTickTwoPoolScenario();

        uint256 gasStart = gasleft();
        constantRebalancer.rebalance(constantParams);
        uint256 constantGas = gasStart - gasleft();
        uint256 constantEv = _evWithTwoTokens(constantCollateral, constantTokenA, constantTokenB, sqrtPredX96);
        uint256 constantTouched = _countTouchedPools(constantParams);

        (
            MockERC20 exactCollateral,
            MockERC20 exactTokenA,
            MockERC20 exactTokenB,
            Rebalancer exactRebalancer,
            Rebalancer.RebalanceParams memory exactParams,,,
        ) = _buildABMultiTickTwoPoolScenario();

        gasStart = gasleft();
        exactRebalancer.rebalanceExact(exactParams, 32, 256);
        uint256 exactGas = gasStart - gasleft();
        uint256 exactEv = _evWithTwoTokens(exactCollateral, exactTokenA, exactTokenB, sqrtPredX96);
        uint256 exactTouched = _countTouchedPools(exactParams);

        emit log_named_uint("estimated_total_cost", estimatedTotalCost);
        emit log_named_uint("actual_total_cost", actualTotalCost);
        emit log_named_uint("constant_gas", constantGas);
        emit log_named_uint("exact_gas", exactGas);
        emit log_named_uint("constant_touched_pools", constantTouched);
        emit log_named_uint("exact_touched_pools", exactTouched);
        emit log_named_uint("constant_ev", constantEv);
        emit log_named_uint("exact_ev", exactEv);
        emit log_named_int("ev_delta", int256(exactEv) - int256(constantEv));

        assertGt(actualTotalCost, estimatedTotalCost);
        assertGt(exactGas, constantGas);
        assertGt(exactEv, constantEv);
    }

    function testBenchmarkABMultiTickSyntheticNinetyEightOutcomeConstantLVsExact() public {
        (
            MockERC20 constantCollateral,
            Rebalancer constantRebalancer,
            Rebalancer.RebalanceParams memory constantParams,
            uint160 sqrtPredX96,
            uint256 actualTotalCost,
            uint256 estimatedTotalCost
        ) = _buildABMultiTickSyntheticNinetyEightOutcomeScenario();

        uint256 gasStart = gasleft();
        constantRebalancer.rebalance(constantParams);
        uint256 constantGas = gasStart - gasleft();
        uint256 constantEv = _evFromParams(constantCollateral, constantParams, sqrtPredX96);
        uint256 constantTouched = _countTouchedPools(constantParams);

        (MockERC20 exactCollateral, Rebalancer exactRebalancer, Rebalancer.RebalanceParams memory exactParams,,,) =
            _buildABMultiTickSyntheticNinetyEightOutcomeScenario();

        gasStart = gasleft();
        exactRebalancer.rebalanceExact(exactParams, 32, 256);
        uint256 exactGas = gasStart - gasleft();
        uint256 exactEv = _evFromParams(exactCollateral, exactParams, sqrtPredX96);
        uint256 exactTouched = _countTouchedPools(exactParams);

        emit log_named_uint("estimated_total_cost", estimatedTotalCost);
        emit log_named_uint("actual_total_cost", actualTotalCost);
        emit log_named_uint("constant_gas", constantGas);
        emit log_named_uint("exact_gas", exactGas);
        emit log_named_uint("constant_touched_pools", constantTouched);
        emit log_named_uint("exact_touched_pools", exactTouched);
        emit log_named_uint("constant_ev", constantEv);
        emit log_named_uint("exact_ev", exactEv);
        emit log_named_int("ev_delta", int256(exactEv) - int256(constantEv));

        assertGt(actualTotalCost, estimatedTotalCost);
        assertGt(exactGas, constantGas);
        assertGt(exactEv, constantEv);
    }

    function testBenchmarkABMultiTickRealisticSeededNinetyEightOutcomeConstantLVsExact() public {
        (
            MockERC20 constantCollateral,
            Rebalancer constantRebalancer,
            Rebalancer.RebalanceParams memory constantParams,
            uint160 sqrtPredX96,
            uint256 actualTotalCost,
            uint256 estimatedTotalCost
        ) = _buildABMultiTickRealisticSeededNinetyEightOutcomeScenario();

        uint256 gasStart = gasleft();
        constantRebalancer.rebalance(constantParams);
        uint256 constantGas = gasStart - gasleft();
        uint256 constantEv = _evFromParams(constantCollateral, constantParams, sqrtPredX96);
        uint256 constantTouched = _countTouchedPools(constantParams);

        (MockERC20 exactCollateral, Rebalancer exactRebalancer, Rebalancer.RebalanceParams memory exactParams,,,) =
            _buildABMultiTickRealisticSeededNinetyEightOutcomeScenario();

        gasStart = gasleft();
        exactRebalancer.rebalanceExact(exactParams, 32, 64);
        uint256 exactGas = gasStart - gasleft();
        uint256 exactEv = _evFromParams(exactCollateral, exactParams, sqrtPredX96);
        uint256 exactTouched = _countTouchedPools(exactParams);

        emit log_named_uint("estimated_total_cost", estimatedTotalCost);
        emit log_named_uint("actual_total_cost", actualTotalCost);
        emit log_named_uint("constant_gas", constantGas);
        emit log_named_uint("exact_gas", exactGas);
        emit log_named_uint("constant_touched_pools", constantTouched);
        emit log_named_uint("exact_touched_pools", exactTouched);
        emit log_named_uint("constant_ev", constantEv);
        emit log_named_uint("exact_ev", exactEv);
        emit log_named_int("ev_delta", int256(exactEv) - int256(constantEv));

        assertGt(actualTotalCost, estimatedTotalCost);
        assertGt(exactGas, constantGas);
        assertGt(exactEv, constantEv);
    }
}
