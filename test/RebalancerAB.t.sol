// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "forge-std/StdJson.sol";

import {Rebalancer} from "../contracts/Rebalancer.sol";
import {RebalancerMixed} from "../contracts/RebalancerMixed.sol";
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

    function burnFrom(address from, uint256 amount) external returns (bool) {
        if (balanceOf[from] < amount) return false;
        uint256 allowed = allowance[from][msg.sender];
        if (allowed < amount) return false;
        allowance[from][msg.sender] = allowed - amount;
        balanceOf[from] -= amount;
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
    address[] internal tokens;

    function setTokens(address[] memory tokens_) external {
        delete tokens;
        for (uint256 i = 0; i < tokens_.length; i++) {
            tokens.push(tokens_[i]);
        }
    }

    function splitPosition(address collateralToken, address, uint256 amount) external {
        require(BenchmarkERC20(collateralToken).transferFrom(msg.sender, address(this), amount), "split pull failed");
        for (uint256 i = 0; i < tokens.length; i++) {
            BenchmarkERC20(tokens[i]).mint(msg.sender, amount);
        }
    }

    function mergePositions(address collateralToken, address, uint256 amount) external {
        for (uint256 i = 0; i < tokens.length; i++) {
            require(BenchmarkERC20(tokens[i]).burnFrom(msg.sender, amount), "merge burn failed");
        }
        require(BenchmarkERC20(collateralToken).transfer(msg.sender, amount), "merge send failed");
    }
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

    function exactInputSingle(IV3SwapRouter.ExactInputSingleParams calldata params)
        external
        payable
        returns (uint256 amountOut)
    {
        address outcomeToken = configs[params.tokenOut].enabled ? params.tokenOut : params.tokenIn;
        Config memory config = configs[outcomeToken];
        require(config.enabled, "unsupported benchmark route");

        if (outcomeToken == params.tokenOut) {
            if (config.isToken1) {
                return _buyToken1WithToken0(params, BenchmarkPool(config.pool));
            }
            return _buyToken0WithToken1(params, BenchmarkPool(config.pool));
        }
        if (config.isToken1) {
            return _sellToken1ForToken0(params, BenchmarkPool(config.pool));
        }
        return _sellToken0ForToken1(params, BenchmarkPool(config.pool));
    }

    function exactOutputSingle(IV3SwapRouter.ExactOutputSingleParams calldata) external pure returns (uint256) {
        revert("unused");
    }

    function _buyToken1WithToken0(IV3SwapRouter.ExactInputSingleParams calldata params, BenchmarkPool pool)
        internal
        returns (uint256 amountOut)
    {
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
            uint256 noFeeIn = priceDelta == 0 ? 0 : _mulDivRoundingUp(lq96, priceDelta, uint256(start) * uint256(end));
            actualIn = feeComp == 0 ? 0 : _mulDivRoundingUp(noFeeIn, FEE_UNITS, feeComp);
            if (actualIn > params.amountIn) actualIn = params.amountIn;
            amountOut = priceDelta == 0 ? 0 : FullMath.mulDiv(uint256(liquidity), priceDelta, Q96);
        }

        require(
            BenchmarkERC20(params.tokenIn).transferFrom(msg.sender, address(this), actualIn), "buy transferFrom failed"
        );
        BenchmarkERC20(params.tokenOut).mint(params.recipient, amountOut);
        pool.setSqrtPrice(end);
    }

    function _sellToken1ForToken0(IV3SwapRouter.ExactInputSingleParams calldata params, BenchmarkPool pool)
        internal
        returns (uint256 amountOut)
    {
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

        require(
            BenchmarkERC20(params.tokenIn).transferFrom(msg.sender, address(this), actualIn), "sell transferFrom failed"
        );
        BenchmarkERC20(params.tokenOut).mint(params.recipient, amountOut);
        pool.setSqrtPrice(end);
    }

    function _buyToken0WithToken1(IV3SwapRouter.ExactInputSingleParams calldata params, BenchmarkPool pool)
        internal
        returns (uint256 amountOut)
    {
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

        require(
            BenchmarkERC20(params.tokenIn).transferFrom(msg.sender, address(this), actualIn), "buy transferFrom failed"
        );
        BenchmarkERC20(params.tokenOut).mint(params.recipient, amountOut);
        pool.setSqrtPrice(end);
    }

    function _sellToken0ForToken1(IV3SwapRouter.ExactInputSingleParams calldata params, BenchmarkPool pool)
        internal
        returns (uint256 amountOut)
    {
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
            uint256 noFeeIn = priceDelta == 0 ? 0 : _mulDivRoundingUp(lq96, priceDelta, uint256(start) * uint256(end));
            actualIn = feeComp == 0 ? 0 : _mulDivRoundingUp(noFeeIn, FEE_UNITS, feeComp);
            if (actualIn > params.amountIn) actualIn = params.amountIn;
            amountOut = priceDelta == 0 ? 0 : FullMath.mulDiv(uint256(liquidity), priceDelta, Q96);
        }

        require(
            BenchmarkERC20(params.tokenIn).transferFrom(msg.sender, address(this), actualIn), "sell transferFrom failed"
        );
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
    uint256 internal constant BENCHMARK_MAX_OUTER_ITERS = 24;
    uint256 internal constant BENCHMARK_MAX_INNER_ITERS = 24;
    address internal constant BENCHMARK_MARKET = address(0xBEEF);

    struct Scenario {
        BenchmarkERC20 collateral;
        Rebalancer rebalancer;
        Rebalancer.RebalanceParams params;
        BenchmarkERC20[] tokens;
        uint256[] predictionsWad;
    }

    struct MixedScenario {
        BenchmarkERC20 collateral;
        RebalancerMixed rebalancerMixed;
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

    struct MixedComparisonResult {
        uint256 rebalancerEv;
        uint256 mixedEv;
        uint256 rebalancerGas;
        uint256 mixedGas;
        int256 evDiff;
        int256 gasDiff;
        int256 evDiffBps;
        int256 gasDiffBps;
    }

    struct OnchainBenchmarkCallArtifact {
        address target;
        bytes calldataBytes;
        uint256 gasUnits;
        uint256 rawEvWei;
    }

    function test_rebalancer_ab_benchmark() external {
        string memory failures = "";

        failures = _appendFailure(failures, _checkDirectParity("two_pool_single_tick_direct_only"));
        failures = _appendFailure(failures, _checkDirectParity("ninety_eight_outcome_multitick_direct_only"));
        failures = _appendFailure(failures, _checkFullDominance("small_bundle_mixed_case"));
        failures = _appendFailure(failures, _checkFullDominance("mixed_route_favorable_synthetic_case"));
        failures = _appendFailure(failures, _checkFullDominance("heterogeneous_ninety_eight_outcome_l1_like_case"));
        failures = _appendFailure(failures, _checkDirectParity("legacy_holdings_direct_only_case"));

        assertEq(bytes(failures).length, 0, failures);
    }

    function test_rebalancer_ab_live_l1_snapshot_report() external {
        string memory json = _liveReportJson();
        string memory caseId = json.readString(".case_id");
        uint256 offchainDirect = json.readUint(".offchain_direct_ev_wei");
        uint256 offchainFullRebalanceOnly = json.readUint(".offchain_full_rebalance_only_ev_wei");
        uint256[] memory pricesWad = json.readUintArray(".starting_prices_wad");
        uint256[] memory predsWad = json.readUintArray(".predictions_wad");
        uint256[] memory holdingsWad = json.readUintArray(".initial_holdings_wad");
        uint256[] memory liquidities = json.readUintArray(".liquidity");
        bool[] memory isToken1 = json.readBoolArray(".is_token1");
        uint24 feeTier = uint24(json.readUint(".fee_tier"));
        uint256 cashWad = json.readUint(".initial_cash_budget_wad");

        Scenario memory constantScenario =
            _buildScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, feeTier);
        constantScenario.rebalancer.rebalance(constantScenario.params);
        uint256 constantEv =
            _portfolioEvWad(constantScenario.collateral, constantScenario.tokens, constantScenario.predictionsWad);

        Scenario memory exactScenario =
            _buildScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, feeTier);
        exactScenario.rebalancer.rebalanceExact(exactScenario.params, 24, 4);
        uint256 exactEv = _portfolioEvWad(exactScenario.collateral, exactScenario.tokens, exactScenario.predictionsWad);

        uint256 directTol = _evParityTol(offchainDirect, constantEv);
        uint256 fullTol = _evParityTol(offchainFullRebalanceOnly, constantEv);

        emit log_string(caseId);
        emit log_named_uint("offchain_direct_ev", offchainDirect);
        emit log_named_uint("offchain_full_rebalance_only_ev", offchainFullRebalanceOnly);
        emit log_named_uint("onchain_constant_ev", constantEv);
        emit log_named_uint("onchain_exact_ev", exactEv);
        emit log_named_int("constant_minus_offchain_direct", int256(constantEv) - int256(offchainDirect));
        emit log_named_int(
            "offchain_full_minus_onchain_constant", int256(offchainFullRebalanceOnly) - int256(constantEv)
        );

        assertTrue(
            offchainDirect + directTol >= constantEv,
            string.concat(caseId, ": onchain constant exceeded offchain direct")
        );
        assertTrue(
            _absDiff(constantEv, offchainDirect) <= directTol,
            string.concat(caseId, ": onchain constant vs offchain direct mismatch")
        );
        assertTrue(
            offchainFullRebalanceOnly + fullTol >= constantEv,
            string.concat(caseId, ": offchain full rebalance-only underperformed onchain constant")
        );
        assertTrue(exactEv + fullTol >= constantEv, string.concat(caseId, ": exact path underperformed constant path"));
    }

    function test_rebalancer_vs_mixed_apples_to_apples_report() external {
        string[] memory caseIds = new string[](6);
        caseIds[0] = "two_pool_single_tick_direct_only";
        caseIds[1] = "ninety_eight_outcome_multitick_direct_only";
        caseIds[2] = "small_bundle_mixed_case";
        caseIds[3] = "mixed_route_favorable_synthetic_case";
        caseIds[4] = "heterogeneous_ninety_eight_outcome_l1_like_case";
        caseIds[5] = "legacy_holdings_direct_only_case";

        for (uint256 i = 0; i < caseIds.length; i++) {
            MixedComparisonResult memory r = _runRebalancerVsMixedCase(caseIds[i]);
            emit log_string(caseIds[i]);
            emit log_named_uint("rebalancer_ev_wei", r.rebalancerEv);
            emit log_named_uint("rebalancer_mixed_ev_wei", r.mixedEv);
            emit log_named_int("ev_diff_wei_mixed_minus_rebalancer", r.evDiff);
            emit log_named_int("ev_diff_bps_mixed_minus_rebalancer", r.evDiffBps);
            emit log_named_uint("rebalancer_gas", r.rebalancerGas);
            emit log_named_uint("rebalancer_mixed_gas", r.mixedGas);
            emit log_named_int("gas_diff_mixed_minus_rebalancer", r.gasDiff);
            emit log_named_int("gas_diff_bps_mixed_minus_rebalancer", r.gasDiffBps);
        }
    }

    function test_write_rebalancer_ab_onchain_call_report() external {
        string[] memory caseIds = new string[](6);
        caseIds[0] = "two_pool_single_tick_direct_only";
        caseIds[1] = "ninety_eight_outcome_multitick_direct_only";
        caseIds[2] = "small_bundle_mixed_case";
        caseIds[3] = "mixed_route_favorable_synthetic_case";
        caseIds[4] = "heterogeneous_ninety_eight_outcome_l1_like_case";
        caseIds[5] = "legacy_holdings_direct_only_case";

        string memory rootKey = "rebalancer_ab_onchain_call_report";
        string memory rootJson = "";
        for (uint256 i = 0; i < caseIds.length; i++) {
            string memory caseId = caseIds[i];
            OnchainBenchmarkCallArtifact memory exact = _buildExactArtifact(caseId);
            OnchainBenchmarkCallArtifact memory mixed = _buildMixedArtifact(caseId);

            string memory exactKey = string.concat(caseId, "_exact");
            string memory exactJson = vm.serializeAddress(exactKey, "target", exact.target);
            exactJson = vm.serializeBytes(exactKey, "calldata", exact.calldataBytes);
            exactJson = vm.serializeUint(exactKey, "gas_units", exact.gasUnits);
            exactJson = vm.serializeString(exactKey, "raw_ev_wei", vm.toString(exact.rawEvWei));

            string memory mixedKey = string.concat(caseId, "_mixed");
            string memory mixedJson = vm.serializeAddress(mixedKey, "target", mixed.target);
            mixedJson = vm.serializeBytes(mixedKey, "calldata", mixed.calldataBytes);
            mixedJson = vm.serializeUint(mixedKey, "gas_units", mixed.gasUnits);
            mixedJson = vm.serializeString(mixedKey, "raw_ev_wei", vm.toString(mixed.rawEvWei));

            string memory caseKey = string.concat(caseId, "_case");
            string memory caseJson = vm.serializeString(caseKey, "case_id", caseId);
            caseJson = vm.serializeString(caseKey, "exact", exactJson);
            caseJson = vm.serializeString(caseKey, "mixed", mixedJson);
            rootJson = vm.serializeString(rootKey, caseId, caseJson);
        }

        vm.writeJson(rootJson, _onchainCallReportPath());
    }

    function test_write_rebalancer_ab_stress_onchain_call_report() external {
        string[] memory caseIds = _stressCaseIds();
        string memory rootKey = "rebalancer_ab_stress_onchain_call_report";
        string memory rootJson = "";

        for (uint256 i = 0; i < caseIds.length; i++) {
            string memory caseId = caseIds[i];
            OnchainBenchmarkCallArtifact memory exact = _buildExactStressArtifact(caseId);
            OnchainBenchmarkCallArtifact memory mixed = _buildMixedStressArtifact(caseId);

            string memory exactKey = string.concat(caseId, "_exact");
            string memory exactJson = vm.serializeAddress(exactKey, "target", exact.target);
            exactJson = vm.serializeBytes(exactKey, "calldata", exact.calldataBytes);
            exactJson = vm.serializeUint(exactKey, "gas_units", exact.gasUnits);
            exactJson = vm.serializeString(exactKey, "raw_ev_wei", vm.toString(exact.rawEvWei));

            string memory mixedKey = string.concat(caseId, "_mixed");
            string memory mixedJson = vm.serializeAddress(mixedKey, "target", mixed.target);
            mixedJson = vm.serializeBytes(mixedKey, "calldata", mixed.calldataBytes);
            mixedJson = vm.serializeUint(mixedKey, "gas_units", mixed.gasUnits);
            mixedJson = vm.serializeString(mixedKey, "raw_ev_wei", vm.toString(mixed.rawEvWei));

            string memory caseKey = string.concat(caseId, "_case");
            string memory caseJson = vm.serializeString(caseKey, "case_id", caseId);
            caseJson = vm.serializeString(caseKey, "exact", exactJson);
            caseJson = vm.serializeString(caseKey, "mixed", mixedJson);
            rootJson = vm.serializeString(rootKey, caseId, caseJson);
        }

        vm.writeJson(rootJson, _stressOnchainCallReportPath());
    }

    function _checkDirectParity(string memory caseId) internal returns (string memory failure) {
        (uint256 offchainDirect,,,,) = _expectedRow(caseId);
        (uint256 constantEv, uint256 exactEv) = _runCase(caseId);
        uint256 tol = _evParityTol(offchainDirect, constantEv);

        emit log_string(caseId);
        emit log_named_uint("offchain_direct_ev", offchainDirect);
        emit log_named_uint("onchain_constant_ev", constantEv);
        emit log_named_uint("onchain_exact_ev", exactEv);
        emit log_named_int("constant_minus_offchain_direct", int256(constantEv) - int256(offchainDirect));

        if (offchainDirect + tol < constantEv) {
            return string.concat(caseId, ": onchain constant exceeded offchain direct");
        }
        if (_absDiff(constantEv, offchainDirect) > tol) {
            return string.concat(caseId, ": onchain constant vs offchain direct mismatch");
        }
        if (exactEv + tol < constantEv) {
            return string.concat(caseId, ": exact path underperformed constant path");
        }

        return "";
    }

    function _checkFullDominance(string memory caseId) internal returns (string memory failure) {
        (uint256 offchainDirect,, uint256 offchainFullRebalanceOnly,,) = _expectedRow(caseId);
        (uint256 constantEv, uint256 exactEv) = _runCase(caseId);
        uint256 directTol = _evParityTol(offchainDirect, constantEv);
        uint256 tol = _evParityTol(offchainFullRebalanceOnly, constantEv);

        emit log_string(caseId);
        emit log_named_uint("offchain_direct_ev", offchainDirect);
        emit log_named_uint("offchain_full_rebalance_only_ev", offchainFullRebalanceOnly);
        emit log_named_uint("onchain_constant_ev", constantEv);
        emit log_named_uint("onchain_exact_ev", exactEv);
        emit log_named_int("constant_minus_offchain_direct", int256(constantEv) - int256(offchainDirect));
        emit log_named_int(
            "offchain_full_minus_onchain_constant", int256(offchainFullRebalanceOnly) - int256(constantEv)
        );

        if (offchainDirect + directTol < constantEv) {
            return string.concat(caseId, ": onchain constant exceeded offchain direct");
        }
        if (_absDiff(constantEv, offchainDirect) > directTol) {
            return string.concat(caseId, ": onchain constant vs offchain direct mismatch");
        }
        if (offchainFullRebalanceOnly + tol < constantEv) {
            return string.concat(caseId, ": offchain full rebalance-only underperformed onchain constant");
        }
        if (exactEv + tol < constantEv) {
            return string.concat(caseId, ": exact path underperformed constant path");
        }

        return "";
    }

    function _runCase(string memory caseId) internal returns (uint256 constantEv, uint256 exactEv) {
        Scenario memory constantScenario = _buildCase(caseId);
        constantScenario.rebalancer.rebalance(constantScenario.params);
        constantEv =
            _portfolioEvWad(constantScenario.collateral, constantScenario.tokens, constantScenario.predictionsWad);

        Scenario memory exactScenario = _buildCase(caseId);
        exactScenario.rebalancer.rebalanceExact(exactScenario.params, 24, 4);
        exactEv = _portfolioEvWad(exactScenario.collateral, exactScenario.tokens, exactScenario.predictionsWad);
    }

    function _runRebalancerVsMixedCase(string memory caseId) internal returns (MixedComparisonResult memory r) {
        Scenario memory constantScenario = _buildCase(caseId);
        uint256 gasStartConstant = gasleft();
        constantScenario.rebalancer.rebalance(constantScenario.params);
        r.rebalancerGas = gasStartConstant - gasleft();
        r.rebalancerEv =
            _portfolioEvWad(constantScenario.collateral, constantScenario.tokens, constantScenario.predictionsWad);

        MixedScenario memory mixedScenario = _buildMixedCase(caseId);
        uint256 gasStartMixed = gasleft();
        mixedScenario.rebalancerMixed
            .rebalanceMixedConstantL(
                mixedScenario.params, BENCHMARK_MARKET, BENCHMARK_MAX_OUTER_ITERS, BENCHMARK_MAX_INNER_ITERS, 0
            );
        r.mixedGas = gasStartMixed - gasleft();
        r.mixedEv = _portfolioEvWad(mixedScenario.collateral, mixedScenario.tokens, mixedScenario.predictionsWad);

        r.evDiff = _signedDiff(r.mixedEv, r.rebalancerEv);
        r.gasDiff = _signedDiff(r.mixedGas, r.rebalancerGas);
        r.evDiffBps = _signedBpsDiff(r.mixedEv, r.rebalancerEv);
        r.gasDiffBps = _signedBpsDiff(r.mixedGas, r.rebalancerGas);
    }

    function _buildExactArtifact(string memory caseId) internal returns (OnchainBenchmarkCallArtifact memory artifact) {
        Scenario memory exactScenario = _buildCase(caseId);
        artifact.target = address(exactScenario.rebalancer);
        artifact.calldataBytes = abi.encodeCall(exactScenario.rebalancer.rebalanceExact, (exactScenario.params, 24, 4));
        uint256 gasStart = gasleft();
        exactScenario.rebalancer.rebalanceExact(exactScenario.params, 24, 4);
        artifact.gasUnits = gasStart - gasleft();
        artifact.rawEvWei =
            _portfolioEvWad(exactScenario.collateral, exactScenario.tokens, exactScenario.predictionsWad);
    }

    function _buildMixedArtifact(string memory caseId) internal returns (OnchainBenchmarkCallArtifact memory artifact) {
        MixedScenario memory mixedScenario = _buildMixedCase(caseId);
        artifact.target = address(mixedScenario.rebalancerMixed);
        artifact.calldataBytes = abi.encodeCall(
            mixedScenario.rebalancerMixed.rebalanceMixedConstantL,
            (mixedScenario.params, BENCHMARK_MARKET, BENCHMARK_MAX_OUTER_ITERS, BENCHMARK_MAX_INNER_ITERS, 0)
        );
        uint256 gasStart = gasleft();
        mixedScenario.rebalancerMixed
            .rebalanceMixedConstantL(
                mixedScenario.params, BENCHMARK_MARKET, BENCHMARK_MAX_OUTER_ITERS, BENCHMARK_MAX_INNER_ITERS, 0
            );
        artifact.gasUnits = gasStart - gasleft();
        artifact.rawEvWei =
            _portfolioEvWad(mixedScenario.collateral, mixedScenario.tokens, mixedScenario.predictionsWad);
    }

    function _buildExactStressArtifact(string memory caseId)
        internal
        returns (OnchainBenchmarkCallArtifact memory artifact)
    {
        Scenario memory exactScenario = _buildStressCase(caseId);
        artifact.target = address(exactScenario.rebalancer);
        artifact.calldataBytes = abi.encodeCall(exactScenario.rebalancer.rebalanceExact, (exactScenario.params, 24, 4));
        uint256 gasStart = gasleft();
        exactScenario.rebalancer.rebalanceExact(exactScenario.params, 24, 4);
        artifact.gasUnits = gasStart - gasleft();
        artifact.rawEvWei =
            _portfolioEvWad(exactScenario.collateral, exactScenario.tokens, exactScenario.predictionsWad);
    }

    function _buildMixedStressArtifact(string memory caseId)
        internal
        returns (OnchainBenchmarkCallArtifact memory artifact)
    {
        MixedScenario memory mixedScenario = _buildMixedStressCase(caseId);
        artifact.target = address(mixedScenario.rebalancerMixed);
        artifact.calldataBytes = abi.encodeCall(
            mixedScenario.rebalancerMixed.rebalanceMixedConstantL,
            (mixedScenario.params, BENCHMARK_MARKET, BENCHMARK_MAX_OUTER_ITERS, BENCHMARK_MAX_INNER_ITERS, 0)
        );
        uint256 gasStart = gasleft();
        mixedScenario.rebalancerMixed
            .rebalanceMixedConstantL(
                mixedScenario.params, BENCHMARK_MARKET, BENCHMARK_MAX_OUTER_ITERS, BENCHMARK_MAX_INNER_ITERS, 0
            );
        artifact.gasUnits = gasStart - gasleft();
        artifact.rawEvWei =
            _portfolioEvWad(mixedScenario.collateral, mixedScenario.tokens, mixedScenario.predictionsWad);
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

    function _buildMixedCase(string memory caseId) internal returns (MixedScenario memory scenario) {
        string memory json = _casesJson();
        string memory prefix = string.concat(".", caseId);
        uint256 uniformCount = json.readUint(string.concat(prefix, ".uniform_count"));
        uint256 cashWad = json.readUint(string.concat(prefix, ".initial_cash_budget_wad"));
        uint24 feeTier = uint24(json.readUint(string.concat(prefix, ".fee_tier")));

        if (uniformCount > 0) {
            uint256 uniformPriceBps = json.readUint(string.concat(prefix, ".uniform_price_bps"));
            uint256 uniformLiquidity = json.readUint(string.concat(prefix, ".uniform_liquidity"));
            return _buildMixedUniformScenario(uniformCount, uniformPriceBps, uniformLiquidity, cashWad, feeTier);
        }

        uint256[] memory pricesWad = json.readUintArray(string.concat(prefix, ".starting_prices_wad"));
        uint256[] memory predsWad = json.readUintArray(string.concat(prefix, ".predictions_wad"));
        uint256[] memory holdingsWad = json.readUintArray(string.concat(prefix, ".initial_holdings_wad"));
        uint256[] memory liquidities = json.readUintArray(string.concat(prefix, ".liquidity"));
        bool[] memory isToken1 = json.readBoolArray(string.concat(prefix, ".is_token1"));

        return _buildMixedScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, feeTier);
    }

    function _buildStressCase(string memory caseId) internal returns (Scenario memory scenario) {
        if (_stressCaseIdMatches(caseId, "preserve_frontier_inactive_sell_case")) {
            uint256[] memory pricesWad = new uint256[](5);
            uint256[] memory predsWad = new uint256[](5);
            uint256[] memory holdingsWad = new uint256[](5);
            uint256[] memory liquidities = new uint256[](5);
            bool[] memory isToken1 = new bool[](5);

            pricesWad[0] = 13e16;
            pricesWad[1] = 16e16;
            pricesWad[2] = 34e16;
            pricesWad[3] = 25e16;
            pricesWad[4] = 12e16;
            predsWad[0] = 27e16;
            predsWad[1] = 24e16;
            predsWad[2] = 18e16;
            predsWad[3] = 16e16;
            predsWad[4] = 15e16;
            holdingsWad[2] = 7e18;
            holdingsWad[3] = 4e18;
            liquidities[0] = 2_400_000_000_000_000_000_000;
            liquidities[1] = 2_300_000_000_000_000_000_000;
            liquidities[2] = 3_100_000_000_000_000_000_000;
            liquidities[3] = 2_700_000_000_000_000_000_000;
            liquidities[4] = 2_500_000_000_000_000_000_000;
            for (uint256 i = 0; i < 5; i++) {
                isToken1[i] = true;
            }
            return _buildScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, 18e18, 100);
        }
        if (_stressCaseIdMatches(caseId, "mint_dominant_case")) {
            uint256[] memory pricesWad = new uint256[](3);
            uint256[] memory predsWad = new uint256[](3);
            uint256[] memory holdingsWad = new uint256[](3);
            uint256[] memory liquidities = new uint256[](3);
            bool[] memory isToken1 = new bool[](3);

            pricesWad[0] = 84e16;
            pricesWad[1] = 8e16;
            pricesWad[2] = 8e16;
            predsWad[0] = 36e16;
            predsWad[1] = 32e16;
            predsWad[2] = 32e16;
            liquidities[0] = 2_800_000_000_000_000_000_000;
            liquidities[1] = 2_800_000_000_000_000_000_000;
            liquidities[2] = 2_800_000_000_000_000_000_000;
            for (uint256 i = 0; i < 3; i++) {
                isToken1[i] = true;
            }
            return _buildScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, 30e18, 100);
        }
        if (_stressCaseIdMatches(caseId, "boundary_profitable_16_full_single_tick")) {
            return _buildBoundaryStressScenario(16, 100e18);
        }
        if (_stressCaseIdMatches(caseId, "boundary_profitable_20_tiny_single_tick")) {
            return _buildBoundaryStressScenario(20, 5e18);
        }
        revert("unknown stress case");
    }

    function _buildMixedStressCase(string memory caseId) internal returns (MixedScenario memory scenario) {
        if (_stressCaseIdMatches(caseId, "preserve_frontier_inactive_sell_case")) {
            uint256[] memory pricesWad = new uint256[](5);
            uint256[] memory predsWad = new uint256[](5);
            uint256[] memory holdingsWad = new uint256[](5);
            uint256[] memory liquidities = new uint256[](5);
            bool[] memory isToken1 = new bool[](5);

            pricesWad[0] = 13e16;
            pricesWad[1] = 16e16;
            pricesWad[2] = 34e16;
            pricesWad[3] = 25e16;
            pricesWad[4] = 12e16;
            predsWad[0] = 27e16;
            predsWad[1] = 24e16;
            predsWad[2] = 18e16;
            predsWad[3] = 16e16;
            predsWad[4] = 15e16;
            holdingsWad[2] = 7e18;
            holdingsWad[3] = 4e18;
            liquidities[0] = 2_400_000_000_000_000_000_000;
            liquidities[1] = 2_300_000_000_000_000_000_000;
            liquidities[2] = 3_100_000_000_000_000_000_000;
            liquidities[3] = 2_700_000_000_000_000_000_000;
            liquidities[4] = 2_500_000_000_000_000_000_000;
            for (uint256 i = 0; i < 5; i++) {
                isToken1[i] = true;
            }
            return _buildMixedScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, 18e18, 100);
        }
        if (_stressCaseIdMatches(caseId, "mint_dominant_case")) {
            uint256[] memory pricesWad = new uint256[](3);
            uint256[] memory predsWad = new uint256[](3);
            uint256[] memory holdingsWad = new uint256[](3);
            uint256[] memory liquidities = new uint256[](3);
            bool[] memory isToken1 = new bool[](3);

            pricesWad[0] = 84e16;
            pricesWad[1] = 8e16;
            pricesWad[2] = 8e16;
            predsWad[0] = 36e16;
            predsWad[1] = 32e16;
            predsWad[2] = 32e16;
            liquidities[0] = 2_800_000_000_000_000_000_000;
            liquidities[1] = 2_800_000_000_000_000_000_000;
            liquidities[2] = 2_800_000_000_000_000_000_000;
            for (uint256 i = 0; i < 3; i++) {
                isToken1[i] = true;
            }
            return _buildMixedScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, 30e18, 100);
        }
        if (_stressCaseIdMatches(caseId, "boundary_profitable_16_full_single_tick")) {
            return _buildMixedBoundaryStressScenario(16, 100e18);
        }
        if (_stressCaseIdMatches(caseId, "boundary_profitable_20_tiny_single_tick")) {
            return _buildMixedBoundaryStressScenario(20, 5e18);
        }
        revert("unknown stress case");
    }

    function _buildBoundaryStressScenario(uint256 profitableCount, uint256 cashWad) internal returns (Scenario memory) {
        uint256 outcomeCount = profitableCount + 2;
        uint256[] memory pricesWad = new uint256[](outcomeCount);
        uint256[] memory predsWad = new uint256[](outcomeCount);
        uint256[] memory holdingsWad = new uint256[](outcomeCount);
        uint256[] memory liquidities = new uint256[](outcomeCount);
        bool[] memory isToken1 = new bool[](outcomeCount);
        uint256 basePred = WAD / outcomeCount;
        uint256 remainder = WAD - basePred * outcomeCount;

        for (uint256 i = 0; i < outcomeCount; i++) {
            uint256 pred = basePred;
            if (i == 0) pred += remainder;
            predsWad[i] = pred;
            uint256 priceBps = i < profitableCount ? 7600 + 300 * (i % 4) : 10300 + 200 * ((i - profitableCount) % 3);
            pricesWad[i] = pred * priceBps / 10_000;
            liquidities[i] = 3_500_000_000_000_000_000_000 + i * 50_000_000_000_000_000;
            isToken1[i] = true;
        }

        return _buildScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, 100);
    }

    function _buildMixedBoundaryStressScenario(uint256 profitableCount, uint256 cashWad)
        internal
        returns (MixedScenario memory)
    {
        uint256 outcomeCount = profitableCount + 2;
        uint256[] memory pricesWad = new uint256[](outcomeCount);
        uint256[] memory predsWad = new uint256[](outcomeCount);
        uint256[] memory holdingsWad = new uint256[](outcomeCount);
        uint256[] memory liquidities = new uint256[](outcomeCount);
        bool[] memory isToken1 = new bool[](outcomeCount);
        uint256 basePred = WAD / outcomeCount;
        uint256 remainder = WAD - basePred * outcomeCount;

        for (uint256 i = 0; i < outcomeCount; i++) {
            uint256 pred = basePred;
            if (i == 0) pred += remainder;
            predsWad[i] = pred;
            uint256 priceBps = i < profitableCount ? 7600 + 300 * (i % 4) : 10300 + 200 * ((i - profitableCount) % 3);
            pricesWad[i] = pred * priceBps / 10_000;
            liquidities[i] = 3_500_000_000_000_000_000_000 + i * 50_000_000_000_000_000;
            isToken1[i] = true;
        }

        return _buildMixedScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, 100);
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
            n == predsWad.length && n == holdingsWad.length && n == liquidities.length && n == isToken1.length,
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
            BenchmarkPool pool =
                new BenchmarkPool(_priceWadToSqrtX96(pricesWad[i], isToken1[i]), uint128(liquidities[i]));
            pool.setTick(0);

            scenario.tokens[i] = token;
            arrays.tokens[i] = address(token);
            arrays.pools[i] = address(pool);
            arrays.isToken1[i] = isToken1[i];
            arrays.balances[i] = holdingsWad[i];
            arrays.sqrtPredX96[i] = _priceWadToSqrtX96(predsWad[i], isToken1[i]);

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

    function _buildMixedScenario(
        uint256[] memory pricesWad,
        uint256[] memory predsWad,
        uint256[] memory holdingsWad,
        uint256[] memory liquidities,
        bool[] memory isToken1,
        uint256 cashWad,
        uint24 feeTier
    ) internal returns (MixedScenario memory scenario) {
        uint256 n = pricesWad.length;
        require(
            n == predsWad.length && n == holdingsWad.length && n == liquidities.length && n == isToken1.length,
            "length mismatch"
        );

        scenario.collateral = new BenchmarkERC20();
        BenchmarkRouter router = new BenchmarkRouter();
        BenchmarkCTFRouter ctfRouter = new BenchmarkCTFRouter();
        scenario.rebalancerMixed = new RebalancerMixed(address(router), address(ctfRouter));
        scenario.tokens = new BenchmarkERC20[](n);
        scenario.predictionsWad = predsWad;

        ScenarioArrays memory arrays = _allocScenarioArrays(n);

        for (uint256 i = 0; i < n; i++) {
            BenchmarkERC20 token = new BenchmarkERC20();
            BenchmarkPool pool =
                new BenchmarkPool(_priceWadToSqrtX96(pricesWad[i], isToken1[i]), uint128(liquidities[i]));
            pool.setTick(0);

            scenario.tokens[i] = token;
            arrays.tokens[i] = address(token);
            arrays.pools[i] = address(pool);
            arrays.isToken1[i] = isToken1[i];
            arrays.balances[i] = holdingsWad[i];
            arrays.sqrtPredX96[i] = _priceWadToSqrtX96(predsWad[i], isToken1[i]);

            router.configure(address(token), address(pool), isToken1[i]);

            if (holdingsWad[i] > 0) {
                token.mint(address(this), holdingsWad[i]);
                token.approve(address(scenario.rebalancerMixed), holdingsWad[i]);
            }
        }
        ctfRouter.setTokens(arrays.tokens);

        if (cashWad > 0) {
            scenario.collateral.mint(address(this), cashWad);
            scenario.collateral.approve(address(scenario.rebalancerMixed), cashWad);
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

    function _buildMixedUniformScenario(
        uint256 count,
        uint256 priceBps,
        uint256 liquidityPerPool,
        uint256 cashWad,
        uint24 feeTier
    ) internal returns (MixedScenario memory) {
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

        return _buildMixedScenario(pricesWad, predsWad, holdingsWad, liquidities, isToken1, cashWad, feeTier);
    }

    function _allocScenarioArrays(uint256 n) internal pure returns (ScenarioArrays memory arrays) {
        arrays.tokens = new address[](n);
        arrays.pools = new address[](n);
        arrays.isToken1 = new bool[](n);
        arrays.balances = new uint256[](n);
        arrays.sqrtPredX96 = new uint160[](n);
    }

    function _portfolioEvWad(BenchmarkERC20 collateral, BenchmarkERC20[] memory tokens, uint256[] memory predictionsWad)
        internal
        view
        returns (uint256 evWad)
    {
        evWad = collateral.balanceOf(address(this));
        for (uint256 i = 0; i < tokens.length; i++) {
            evWad += FullMath.mulDiv(tokens[i].balanceOf(address(this)), predictionsWad[i], WAD);
        }
    }

    function _expectedRow(string memory caseId)
        internal
        view
        returns (
            uint256 offchainDirect,
            uint256 offchainMixed,
            uint256 offchainFullRebalanceOnly,
            uint256 offchainActions,
            uint256 expectedExact
        )
    {
        string memory json = _expectedJson();
        string memory prefix = string.concat(".", caseId);
        offchainDirect = json.readUint(string.concat(prefix, ".offchain_direct_ev_wei"));
        offchainMixed = json.readUint(string.concat(prefix, ".offchain_mixed_ev_wei"));
        offchainFullRebalanceOnly = json.readUint(string.concat(prefix, ".offchain_full_rebalance_only_ev_wei"));
        offchainActions = json.readUint(string.concat(prefix, ".offchain_action_count"));
        expectedExact = json.readUint(string.concat(prefix, ".expected_onchain_exact_ev_wei"));
    }

    function _evParityTol(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 maxValue = a > b ? a : b;
        return 5e12 * (1 + maxValue / WAD);
    }

    function _absDiff(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a - b : b - a;
    }

    function _signedDiff(uint256 a, uint256 b) internal pure returns (int256) {
        uint256 diff = _absDiff(a, b);
        int256 signed = int256(diff);
        return a >= b ? signed : -signed;
    }

    function _signedBpsDiff(uint256 nextValue, uint256 baseValue) internal pure returns (int256) {
        if (baseValue == 0) return 0;
        uint256 numerator = _absDiff(nextValue, baseValue);
        uint256 bps = FullMath.mulDiv(numerator, 10_000, baseValue);
        int256 signed = int256(bps);
        return nextValue >= baseValue ? signed : -signed;
    }

    function _appendFailure(string memory failures, string memory next) internal pure returns (string memory) {
        if (bytes(next).length == 0) return failures;
        if (bytes(failures).length == 0) return next;
        return string.concat(failures, "\n", next);
    }

    function _casesJson() internal view returns (string memory) {
        return vm.readFile(string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_cases.json"));
    }

    function _stressCaseIds() internal pure returns (string[] memory caseIds) {
        caseIds = new string[](4);
        caseIds[0] = "preserve_frontier_inactive_sell_case";
        caseIds[1] = "mint_dominant_case";
        caseIds[2] = "boundary_profitable_16_full_single_tick";
        caseIds[3] = "boundary_profitable_20_tiny_single_tick";
    }

    function _stressCaseIdMatches(string memory left, string memory right) internal pure returns (bool) {
        return keccak256(bytes(left)) == keccak256(bytes(right));
    }

    function _expectedJson() internal view returns (string memory) {
        return vm.readFile(string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_expected.json"));
    }

    function _liveReportJson() internal view returns (string memory) {
        return vm.readFile(string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_live_l1_snapshot_report.json"));
    }

    function _onchainCallReportPath() internal view returns (string memory) {
        return string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_onchain_call_report.json");
    }

    function _stressOnchainCallReportPath() internal view returns (string memory) {
        return string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_stress_onchain_call_report.json");
    }

    function _priceWadToSqrtX96(uint256 priceWad, bool isToken1Outcome) internal pure returns (uint160) {
        require(priceWad > 0, "price=0");
        uint256 two192 = uint256(1) << 192;
        uint256 scaled =
            isToken1Outcome ? FullMath.mulDiv(two192, WAD, priceWad) : FullMath.mulDiv(two192, priceWad, WAD);
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
