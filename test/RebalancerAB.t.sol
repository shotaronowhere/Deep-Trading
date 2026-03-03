// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "forge-std/StdJson.sol";

import {Rebalancer} from "../contracts/Rebalancer.sol";
import {IV3SwapRouter} from "../contracts/interfaces/IV3SwapRouter.sol";
import {FullMath} from "../contracts/libraries/FullMath.sol";

contract BenchmarkERC20 {
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

contract BenchmarkPool {
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
}

contract BenchmarkCTFRouter {
    function splitPosition(address, address, uint256) external {}
    function mergePositions(address, address, uint256) external {}
}

contract BenchmarkRouter {
    uint256 internal constant Q96 = 1 << 96;
    uint256 internal constant FEE_UNITS = 1e6;

    struct Config {
        address pool;
        bool isToken1;
        bool enabled;
    }

    mapping(address => Config) internal configs;

    function configure(address outcomeToken, address pool, bool isToken1) external {
        configs[outcomeToken] = Config({pool: pool, isToken1: isToken1, enabled: true});
    }

    function exactInputSingle(IV3SwapRouter.ExactInputSingleParams calldata params) external payable returns (uint256 amountOut) {
        address outcomeToken = configs[params.tokenOut].enabled ? params.tokenOut : params.tokenIn;
        Config memory config = configs[outcomeToken];
        require(config.enabled && config.isToken1, "unsupported benchmark route");

        if (outcomeToken == params.tokenOut) {
            return _buyToken1WithToken0(params, BenchmarkPool(config.pool));
        }
        return _sellToken1ForToken0(params, BenchmarkPool(config.pool));
    }

    function exactOutputSingle(IV3SwapRouter.ExactOutputSingleParams calldata) external pure returns (uint256) {
        revert("unused");
    }

    function _buyToken1WithToken0(
        IV3SwapRouter.ExactInputSingleParams calldata params,
        BenchmarkPool pool
    ) internal returns (uint256 amountOut) {
        (uint160 start,,,,,,) = pool.slot0();
        uint128 liquidity = pool.liquidity();
        uint160 end;
        uint256 actualIn;

        unchecked {
            uint256 feeComp = FEE_UNITS - uint256(params.fee);
            uint256 effectiveIn = FullMath.mulDiv(params.amountIn, feeComp, FEE_UNITS);
            uint256 lq96 = uint256(liquidity) * Q96;

            uint160 endFromInput = start;
            if (effectiveIn > 0) {
                uint256 denom = lq96 + effectiveIn * uint256(start);
                endFromInput = uint160(FullMath.mulDiv(lq96, uint256(start), denom));
            }

            uint160 target = params.sqrtPriceLimitX96;
            end = endFromInput < target ? target : endFromInput;
            if (end > start) end = start;
            uint256 priceDelta = uint256(start) > uint256(end) ? uint256(start) - uint256(end) : 0;
            uint256 noFeeIn = priceDelta == 0
                ? 0
                : _mulDivRoundingUp(lq96, priceDelta, uint256(start) * uint256(end));
            actualIn = feeComp == 0 ? 0 : _mulDivRoundingUp(noFeeIn, FEE_UNITS, feeComp);
            if (actualIn > params.amountIn) actualIn = params.amountIn;
            amountOut = priceDelta == 0 ? 0 : FullMath.mulDiv(uint256(liquidity), priceDelta, Q96);
        }

        require(BenchmarkERC20(params.tokenIn).transferFrom(msg.sender, address(this), actualIn), "buy transferFrom failed");
        BenchmarkERC20(params.tokenOut).mint(params.recipient, amountOut);
        pool.setSqrtPrice(end);
    }

    function _sellToken1ForToken0(
        IV3SwapRouter.ExactInputSingleParams calldata params,
        BenchmarkPool pool
    ) internal returns (uint256 amountOut) {
        (uint160 start,,,,,,) = pool.slot0();
        uint128 liquidity = pool.liquidity();
        uint160 end;
        uint256 actualIn;

        unchecked {
            uint256 feeComp = FEE_UNITS - uint256(params.fee);
            uint256 effectiveIn = FullMath.mulDiv(params.amountIn, feeComp, FEE_UNITS);

            uint160 endFromInput = start;
            if (effectiveIn > 0) {
                endFromInput = uint160(uint256(start) + FullMath.mulDiv(effectiveIn, Q96, uint256(liquidity)));
            }

            uint160 target = params.sqrtPriceLimitX96;
            end = endFromInput > target ? target : endFromInput;
            if (end < start) end = start;
            uint256 priceDelta = uint256(end) > uint256(start) ? uint256(end) - uint256(start) : 0;
            uint256 noFeeIn = priceDelta == 0 ? 0 : FullMath.mulDiv(uint256(liquidity), priceDelta, Q96);
            actualIn = feeComp == 0 ? 0 : _mulDivRoundingUp(noFeeIn, FEE_UNITS, feeComp);
            if (actualIn > params.amountIn) actualIn = params.amountIn;
            uint256 lq96 = uint256(liquidity) * Q96;
            amountOut = priceDelta == 0 ? 0 : FullMath.mulDiv(lq96, priceDelta, uint256(end) * uint256(start));
        }

        require(BenchmarkERC20(params.tokenIn).transferFrom(msg.sender, address(this), actualIn), "sell transferFrom failed");
        BenchmarkERC20(params.tokenOut).mint(params.recipient, amountOut);
        pool.setSqrtPrice(end);
    }

    function _mulDivRoundingUp(uint256 a, uint256 b, uint256 denominator) internal pure returns (uint256 result) {
        if (a == 0 || b == 0) return 0;
        result = FullMath.mulDiv(a, b, denominator);
        if (mulmod(a, b, denominator) > 0) {
            unchecked {
                result++;
            }
        }
    }
}

contract RebalancerABTest is Test {
    using stdJson for string;

    uint256 internal constant WAD = 1e18;
    struct Scenario {
        BenchmarkERC20 collateral;
        Rebalancer rebalancer;
        Rebalancer.RebalanceParams params;
        BenchmarkERC20[] tokens;
        uint256[] predictionsWad;
    }

    struct ScenarioArrays {
        address[] tokens;
        address[] pools;
        bool[] isToken1;
        uint256[] balances;
        uint160[] sqrtPredX96;
    }

    function test_rebalancer_ab_benchmark() external {
        string memory failures = "";

        failures = _appendFailure(failures, _checkDirectParity("two_pool_single_tick_direct_only"));
        failures = _appendFailure(failures, _checkDirectParity("ninety_eight_outcome_multitick_direct_only"));
        failures = _appendFailure(failures, _checkMixedGap("small_bundle_mixed_case"));
        failures = _appendFailure(failures, _checkDirectParity("legacy_holdings_direct_only_case"));

        assertEq(bytes(failures).length, 0, failures);
    }

    function _checkDirectParity(string memory caseId) internal returns (string memory failure) {
        (uint256 offchainDirect,,,) = _expectedRow(caseId);
        (uint256 constantEv, uint256 exactEv) = _runCase(caseId);
        uint256 tol = _evParityTol(offchainDirect, exactEv);

        emit log_string(caseId);
        emit log_named_uint("offchain_direct_ev", offchainDirect);
        emit log_named_uint("onchain_constant_ev", constantEv);
        emit log_named_uint("onchain_exact_ev", exactEv);
        emit log_named_int("exact_minus_offchain_direct", int256(exactEv) - int256(offchainDirect));

        if (_absDiff(exactEv, offchainDirect) > tol) {
            return string.concat(caseId, ": onchain exact vs offchain direct mismatch");
        }
        if (exactEv + tol < constantEv) {
            return string.concat(caseId, ": exact path underperformed constant path");
        }

        return "";
    }

    function _checkMixedGap(string memory caseId) internal returns (string memory failure) {
        (, uint256 offchainMixed,, uint256 expectedExact) = _expectedRow(caseId);
        (uint256 constantEv, uint256 exactEv) = _runCase(caseId);
        uint256 tol = _evGapTol(exactEv, expectedExact);

        emit log_string(caseId);
        emit log_named_uint("offchain_mixed_ev", offchainMixed);
        emit log_named_uint("onchain_constant_ev", constantEv);
        emit log_named_uint("onchain_exact_ev", exactEv);
        emit log_named_int("offchain_mixed_minus_onchain_exact", int256(offchainMixed) - int256(exactEv));

        if (_absDiff(exactEv, expectedExact) > tol) {
            return string.concat(caseId, ": onchain exact EV drifted from committed anomaly snapshot");
        }

        return "";
    }

    function _runCase(string memory caseId) internal returns (uint256 constantEv, uint256 exactEv) {
        Scenario memory constantScenario = _buildCase(caseId);
        constantScenario.rebalancer.rebalance(constantScenario.params);
        constantEv = _portfolioEvWad(
            constantScenario.collateral,
            constantScenario.tokens,
            constantScenario.predictionsWad
        );

        Scenario memory exactScenario = _buildCase(caseId);
        exactScenario.rebalancer.rebalanceExact(exactScenario.params, 24, 4);
        exactEv = _portfolioEvWad(
            exactScenario.collateral,
            exactScenario.tokens,
            exactScenario.predictionsWad
        );
    }

    function _buildCase(string memory caseId) internal returns (Scenario memory scenario) {
        string memory json = _casesJson();
        string memory prefix = string.concat(".", caseId);
        uint256 uniformCount = json.readUint(string.concat(prefix, ".uniform_count"));
        uint256 cashWad = json.readUint(string.concat(prefix, ".initial_cash_budget_wad"));
        uint24 feeTier = uint24(json.readUint(string.concat(prefix, ".fee_tier")));

        if (uniformCount > 0) {
            uint256 uniformPriceBps = json.readUint(string.concat(prefix, ".uniform_price_bps"));
            uint256 uniformLiquidity = json.readUint(string.concat(prefix, ".uniform_liquidity"));
            return _buildUniformScenario(uniformCount, uniformPriceBps, uniformLiquidity, cashWad, feeTier);
        }

        uint256[] memory pricesWad = json.readUintArray(string.concat(prefix, ".starting_prices_wad"));
        uint256[] memory predsWad = json.readUintArray(string.concat(prefix, ".predictions_wad"));
        uint256[] memory holdingsWad = json.readUintArray(string.concat(prefix, ".initial_holdings_wad"));
        uint256[] memory liquidities = json.readUintArray(string.concat(prefix, ".liquidity"));
        bool[] memory isToken1 = json.readBoolArray(string.concat(prefix, ".is_token1"));

        return _buildScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, feeTier);
    }

    function _buildScenario(
        uint256[] memory pricesWad,
        uint256[] memory predsWad,
        uint256[] memory holdingsWad,
        uint256[] memory liquidities,
        bool[] memory isToken1,
        uint256 cashWad,
        uint24 feeTier
    ) internal returns (Scenario memory scenario) {
        uint256 n = pricesWad.length;
        require(
            n == predsWad.length &&
            n == holdingsWad.length &&
            n == liquidities.length &&
            n == isToken1.length,
            "length mismatch"
        );

        scenario.collateral = new BenchmarkERC20();
        BenchmarkRouter router = new BenchmarkRouter();
        scenario.rebalancer = new Rebalancer(address(router), address(new BenchmarkCTFRouter()));
        scenario.tokens = new BenchmarkERC20[](n);
        scenario.predictionsWad = predsWad;

        ScenarioArrays memory arrays = _allocScenarioArrays(n);

        for (uint256 i = 0; i < n; i++) {
            BenchmarkERC20 token = new BenchmarkERC20();
            require(isToken1[i], "benchmark harness supports token1 pools only");
            BenchmarkPool pool = new BenchmarkPool(
                _priceWadToSqrtX96Token1(pricesWad[i]),
                uint128(liquidities[i])
            );
            pool.setTick(0);

            scenario.tokens[i] = token;
            arrays.tokens[i] = address(token);
            arrays.pools[i] = address(pool);
            arrays.isToken1[i] = isToken1[i];
            arrays.balances[i] = holdingsWad[i];
            arrays.sqrtPredX96[i] = _priceWadToSqrtX96Token1(predsWad[i]);

            router.configure(address(token), address(pool), isToken1[i]);

            if (holdingsWad[i] > 0) {
                token.mint(address(this), holdingsWad[i]);
                token.approve(address(scenario.rebalancer), holdingsWad[i]);
            }
        }

        if (cashWad > 0) {
            scenario.collateral.mint(address(this), cashWad);
            scenario.collateral.approve(address(scenario.rebalancer), cashWad);
        }

        scenario.params = Rebalancer.RebalanceParams({
            tokens: arrays.tokens,
            pools: arrays.pools,
            isToken1: arrays.isToken1,
            balances: arrays.balances,
            collateralAmount: cashWad,
            sqrtPredX96: arrays.sqrtPredX96,
            collateral: address(scenario.collateral),
            fee: feeTier
        });
    }

    function _buildUniformScenario(
        uint256 count,
        uint256 priceBps,
        uint256 liquidityPerPool,
        uint256 cashWad,
        uint24 feeTier
    ) internal returns (Scenario memory) {
        uint256[] memory pricesWad = new uint256[](count);
        uint256[] memory predsWad = new uint256[](count);
        uint256[] memory holdingsWad = new uint256[](count);
        uint256[] memory liquidities = new uint256[](count);
        bool[] memory isToken1 = new bool[](count);
        uint256 basePred = WAD / count;
        uint256 remainder = WAD - basePred * count;

        for (uint256 i = 0; i < count; i++) {
            uint256 pred = basePred;
            if (i == 0) pred += remainder;
            predsWad[i] = pred;
            pricesWad[i] = pred * priceBps / 10_000;
            holdingsWad[i] = 0;
            liquidities[i] = liquidityPerPool;
            isToken1[i] = true;
        }

        return _buildScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, feeTier);
    }

    function _allocScenarioArrays(uint256 n) internal pure returns (ScenarioArrays memory arrays) {
        arrays.tokens = new address[](n);
        arrays.pools = new address[](n);
        arrays.isToken1 = new bool[](n);
        arrays.balances = new uint256[](n);
        arrays.sqrtPredX96 = new uint160[](n);
    }

    function _portfolioEvWad(
        BenchmarkERC20 collateral,
        BenchmarkERC20[] memory tokens,
        uint256[] memory predictionsWad
    ) internal view returns (uint256 evWad) {
        evWad = collateral.balanceOf(address(this));
        for (uint256 i = 0; i < tokens.length; i++) {
            evWad += FullMath.mulDiv(tokens[i].balanceOf(address(this)), predictionsWad[i], WAD);
        }
    }

    function _expectedRow(string memory caseId)
        internal
        view
        returns (uint256 offchainDirect, uint256 offchainMixed, uint256 offchainActions, uint256 expectedExact)
    {
        string memory json = _expectedJson();
        string memory prefix = string.concat(".", caseId);
        offchainDirect = json.readUint(string.concat(prefix, ".offchain_direct_ev_wei"));
        offchainMixed = json.readUint(string.concat(prefix, ".offchain_mixed_ev_wei"));
        offchainActions = json.readUint(string.concat(prefix, ".offchain_action_count"));
        expectedExact = json.readUint(string.concat(prefix, ".expected_onchain_exact_ev_wei"));
    }

    function _evParityTol(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 maxValue = a > b ? a : b;
        return 5e12 * (1 + maxValue / WAD);
    }

    function _evGapTol(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 maxValue = a > b ? a : b;
        return 5e13 * (1 + maxValue / WAD);
    }

    function _absDiff(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a - b : b - a;
    }

    function _appendFailure(string memory failures, string memory next) internal pure returns (string memory) {
        if (bytes(next).length == 0) return failures;
        if (bytes(failures).length == 0) return next;
        return string.concat(failures, "\n", next);
    }

    function _casesJson() internal view returns (string memory) {
        return vm.readFile(string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_cases.json"));
    }

    function _expectedJson() internal view returns (string memory) {
        return vm.readFile(string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_expected.json"));
    }

    function _priceWadToSqrtX96Token1(uint256 priceWad) internal pure returns (uint160) {
        require(priceWad > 0, "price=0");
        uint256 two192 = uint256(1) << 192;
        uint256 scaled = FullMath.mulDiv(two192, WAD, priceWad);
        return uint160(_sqrt(scaled));
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

}
