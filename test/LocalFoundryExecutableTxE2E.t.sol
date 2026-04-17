// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";

import {Rebalancer} from "../contracts/Rebalancer.sol";
import {RebalancerMixed} from "../contracts/RebalancerMixed.sol";
import {TradeExecutor} from "../contracts/TradeExecutor.sol";
import {IERC20} from "../contracts/interfaces/IERC20.sol";
import {ICTFRouter} from "../contracts/interfaces/ICTFRouter.sol";
import {IMarket} from "../contracts/interfaces/IMarket.sol";
import {IV3SwapRouter} from "../contracts/interfaces/IV3SwapRouter.sol";
import {FullMath} from "../contracts/libraries/FullMath.sol";
import {TickMath} from "../contracts/libraries/TickMath.sol";

interface ILocalUniswapV3Factory {
    function createPool(address tokenA, address tokenB, uint24 fee) external returns (address pool);
    function enableFeeAmount(uint24 fee, int24 tickSpacing) external;
    function feeAmountTickSpacing(uint24 fee) external view returns (int24);
}

interface ILocalUniswapV3Pool {
    function initialize(uint160 sqrtPriceX96) external;
    function mint(address recipient, int24 tickLower, int24 tickUpper, uint128 amount, bytes calldata data)
        external
        returns (uint256 amount0, uint256 amount1);
    function slot0() external view returns (uint160, int24, uint16, uint16, uint16, uint8, bool);
    function liquidity() external view returns (uint128);
    function token0() external view returns (address);
    function token1() external view returns (address);
    function tickSpacing() external view returns (int24);
}

interface ILocalMarketFactory {
    struct CreateMarketParams {
        string marketName;
        string[] outcomes;
        string questionStart;
        string questionEnd;
        string outcomeType;
        uint256 parentOutcome;
        address parentMarket;
        string category;
        string lang;
        uint256 lowerBound;
        uint256 upperBound;
        uint256 minBond;
        uint32 openingTime;
        string[] tokenNames;
    }

    function createCategoricalMarket(CreateMarketParams calldata params) external returns (address);
}

contract DummyUSDC {
    string public constant name = "DummyUSDC";
    string public constant symbol = "USDC";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "USDC: insufficient");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "USDC: insufficient");
        uint256 allowed = allowance[from][msg.sender];
        if (allowed != type(uint256).max) {
            require(allowed >= amount, "USDC: allowance");
            allowance[from][msg.sender] = allowed - amount;
        }
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract LocalMintHelper {
    function mintLiquidity(address pool, int24 tickLower, int24 tickUpper, uint128 amount) external {
        ILocalUniswapV3Pool(pool).mint(address(this), tickLower, tickUpper, amount, "");
    }

    function uniswapV3MintCallback(uint256 amount0, uint256 amount1, bytes calldata) external {
        if (amount0 > 0) IERC20(ILocalUniswapV3Pool(msg.sender).token0()).transfer(msg.sender, amount0);
        if (amount1 > 0) IERC20(ILocalUniswapV3Pool(msg.sender).token1()).transfer(msg.sender, amount1);
    }

    function sweep(address token, address to) external {
        uint256 balance = IERC20(token).balanceOf(address(this));
        if (balance > 0) IERC20(token).transfer(to, balance);
    }
}

contract LocalFoundryExecutableTxE2E is Test {
    uint256 constant WAD = 1e18;
    uint24 constant FEE = 100;
    uint256 constant GAS_PRICE_WEI = 1_002_325;
    uint256 constant ETH_USD = 3000;
    uint256 constant ETH_USD_WAD = 3000e18;
    uint256 constant L1_FEE_PER_BYTE_WEI = 1_643_855;
    uint256 constant MAX_PACKED_TX_L2_GAS = 40_000_000;
    uint256 constant SMALL_TOLERANCE_WAD = 25e18;
    uint256 constant LARGE_TOLERANCE_WAD = 2_500e18;
    uint128 constant DEFAULT_LIQUIDITY = 1_000_000e18;
    uint256 constant LP_SEED = 1_000_000_000e18;
    uint256 constant MERGE_SEED = 100_000e18;

    struct TradeablePool {
        string name;
        address market;
        uint256 outcomeIndex;
        address token;
        address pool;
        bool isToken1;
        int24 tick;
        int24 tickLower;
        int24 tickUpper;
        uint256 priceWad;
        uint256 predictionWad;
        uint256 initialBalanceWad;
        uint128 liquidity;
    }

    struct ConnectedScenario {
        string id;
        address rootMarket;
        address childMarket;
        address connector;
        address rootInvalid;
        address childInvalid;
        TradeablePool[] tradeables;
    }

    struct LocalFixtureCall {
        address to;
        bytes data;
    }

    struct LocalFixtureChunk {
        LocalFixtureCall[] calls;
        uint256 estimatedL2Gas;
        uint256 estimatedCalldataBytes;
        uint256 estimatedTotalFeeWad;
    }

    struct LocalFixtureResult {
        LocalFixtureChunk[] chunks;
        uint256 actionCount;
        uint256 preRawEvWad;
        uint256 expectedRawEvWad;
        uint256 estimatedTotalFeeWad;
        uint256 estimatedNetEvWad;
    }

    DummyUSDC collateral;
    address conditionalTokens;
    address wrapped1155Factory;
    address realitio;
    address seerRouter;
    address marketFactory;
    ILocalUniswapV3Factory uniFactory;
    address swapRouter02;
    address weth9;
    LocalMintHelper mintHelper;

    function setUp() public {
        collateral = new DummyUSDC();
        conditionalTokens = _deployFromArtifact("test/fixtures/ConditionalTokens.json");
        realitio = _deployFromArtifact("test/fixtures/RealityETH_v3_0.json");
        wrapped1155Factory = _deployFromArtifact("test/fixtures/Wrapped1155Factory.json");
        address realityProxy =
            _deployFromArtifactWithArgs("test/fixtures/SeerRealityProxy.json", abi.encode(conditionalTokens, realitio));
        address marketImpl = _deployFromArtifact("test/fixtures/SeerMarket.json");
        marketFactory = _deployFromArtifactWithArgs(
            "test/fixtures/SeerMarketFactory.json",
            abi.encode(
                marketImpl,
                address(1),
                realitio,
                wrapped1155Factory,
                conditionalTokens,
                address(collateral),
                realityProxy,
                uint32(1 days)
            )
        );
        seerRouter = _deployFromArtifactWithArgs(
            "test/fixtures/SeerRouter.json", abi.encode(conditionalTokens, wrapped1155Factory)
        );

        weth9 = _deployFromArtifact("test/fixtures/WETH9.json");
        uniFactory = ILocalUniswapV3Factory(_deployFromArtifact("test/fixtures/UniswapV3Factory.json"));
        uniFactory.enableFeeAmount(FEE, 1);
        swapRouter02 = _deployFromArtifactWithArgs(
            "lib/swap-router-contracts/artifacts/contracts/SwapRouter02.sol/SwapRouter02.json",
            abi.encode(address(0), address(uniFactory), address(0), weth9)
        );
        mintHelper = new LocalMintHelper();
    }

    function test_small_connected_direct_buy_sell_executable_rust_plan() external {
        uint256[] memory prices = _smallArray(0.1e18, 0.45e18, 0.25e18, 0.2e18);
        uint256[] memory predictions = _smallArray(0.3e18, 0.2e18, 0.25e18, 0.25e18);
        uint256[] memory holdings = _smallArray(0, 25e18, 0, 0);
        ConnectedScenario memory scenario =
            _deployConnectedScenario("small_direct", 3, 2, prices, predictions, holdings);
        _executeRustFixtureScenario(scenario, 1_000e18, false, false, SMALL_TOLERANCE_WAD);
    }

    function test_small_connected_mint_sell_executable_rust_plan() external {
        uint256[] memory prices = _smallArray(0.8e18, 0.78e18, 0.76e18, 0.74e18);
        uint256[] memory predictions = _smallArray(0.25e18, 0.25e18, 0.25e18, 0.25e18);
        uint256[] memory holdings = _smallArray(0, 0, 0, 0);
        ConnectedScenario memory scenario =
            _deployConnectedScenario("small_mint_sell", 4, 0, prices, predictions, holdings);
        _executeRustFixtureScenario(scenario, 500e18, true, false, SMALL_TOLERANCE_WAD);
    }

    function test_small_connected_buy_merge_with_seeded_invalid_inventory() external {
        uint256[] memory prices = _smallArray(0.15e18, 0.14e18, 0.16e18, 0.15e18);
        uint256[] memory predictions = _smallArray(0.25e18, 0.25e18, 0.25e18, 0.25e18);
        uint256[] memory holdings = _smallArray(0, 0, 0, 0);
        ConnectedScenario memory scenario =
            _deployConnectedScenario("small_buy_merge", 4, 0, prices, predictions, holdings);
        _executeRustFixtureScenario(scenario, 10_000e18, true, true, SMALL_TOLERANCE_WAD);
    }

    function test_ninety_eight_outcome_connected_l1_like_executable_rust_plan() external {
        uint256 n = 98;
        uint256[] memory prices = new uint256[](n);
        uint256[] memory predictions = new uint256[](n);
        uint256[] memory holdings = new uint256[](n);
        uint256 basePrediction = WAD / n;
        for (uint256 i = 0; i < n; i++) {
            predictions[i] = basePrediction;
            prices[i] = basePrediction;
            if (i % 11 == 0) prices[i] = basePrediction * 55 / 100;
            if (i % 13 == 0) prices[i] = basePrediction * 165 / 100;
            if (i % 17 == 0) holdings[i] = 20e18;
        }
        predictions[0] += WAD - basePrediction * n;

        ConnectedScenario memory scenario =
            _deployConnectedScenario("l1_like_98", 67, 32, prices, predictions, holdings);
        _executeRustFixtureScenario(scenario, 25_000e18, false, false, LARGE_TOLERANCE_WAD);
    }

    function test_synthetic_ninety_eight_onchain_solver_calls_execute_through_trade_executor() external {
        uint256 n = 98;
        uint256[] memory prices = new uint256[](n);
        uint256[] memory predictions = new uint256[](n);
        uint256[] memory holdings = new uint256[](n);
        uint256 basePrediction = WAD / n;
        for (uint256 i = 0; i < n; i++) {
            predictions[i] = basePrediction;
            prices[i] = i % 7 == 0 ? basePrediction * 60 / 100 : basePrediction * 120 / 100;
            holdings[i] = i % 9 == 0 ? 10e18 : 0;
        }
        predictions[0] += WAD - basePrediction * n;

        ConnectedScenario memory scenario =
            _deployConnectedScenario("synthetic_onchain_98", 98, 0, prices, predictions, holdings);
        _executeOnchainSolverScenario(scenario, 20_000e18, LARGE_TOLERANCE_WAD);
    }

    function _deployConnectedScenario(
        string memory id,
        uint256 rootNamedOutcomes,
        uint256 childNamedOutcomes,
        uint256[] memory prices,
        uint256[] memory predictions,
        uint256[] memory holdings
    ) internal returns (ConnectedScenario memory scenario) {
        scenario.id = id;
        bool hasChild = childNamedOutcomes > 0;
        scenario.rootMarket = _createMarket(string.concat(id, "_root"), rootNamedOutcomes, 0, address(0));
        uint256 connectorIndex = hasChild ? rootNamedOutcomes - 1 : type(uint256).max;
        if (hasChild) {
            scenario.connector = _wrappedOutcome(scenario.rootMarket, connectorIndex);
            scenario.childMarket =
                _createMarket(string.concat(id, "_child"), childNamedOutcomes, connectorIndex, scenario.rootMarket);
            scenario.childInvalid = _wrappedOutcome(scenario.childMarket, childNamedOutcomes);
        }
        scenario.rootInvalid = _wrappedOutcome(scenario.rootMarket, rootNamedOutcomes);

        uint256 rootTradeable = hasChild ? rootNamedOutcomes - 1 : rootNamedOutcomes;
        uint256 childTradeable = hasChild ? childNamedOutcomes : 0;
        uint256 tradeableCount = rootTradeable + childTradeable;
        require(prices.length == tradeableCount, "prices length");
        require(predictions.length == tradeableCount, "predictions length");
        scenario.tradeables = new TradeablePool[](tradeableCount);

        _seedLiquidityInventory(scenario.rootMarket, scenario.childMarket, rootNamedOutcomes, childNamedOutcomes);

        uint256 cursor = 0;
        for (uint256 i = 0; i < rootTradeable; i++) {
            scenario.tradeables[cursor] = _createTradeablePool(
                string.concat(id, "_root_", vm.toString(i)),
                scenario.rootMarket,
                i,
                prices[cursor],
                predictions[cursor],
                holdings.length > cursor ? holdings[cursor] : 0
            );
            cursor++;
        }
        for (uint256 i = 0; i < childTradeable; i++) {
            scenario.tradeables[cursor] = _createTradeablePool(
                string.concat(id, "_child_", vm.toString(i)),
                scenario.childMarket,
                i,
                prices[cursor],
                predictions[cursor],
                holdings.length > cursor ? holdings[cursor] : 0
            );
            cursor++;
        }
        _sweepMintHelper(scenario.tradeables);
    }

    function _createTradeablePool(
        string memory name,
        address market,
        uint256 outcomeIndex,
        uint256 priceWad,
        uint256 predictionWad,
        uint256 initialBalanceWad
    ) internal returns (TradeablePool memory tradeable) {
        address token = _wrappedOutcome(market, outcomeIndex);
        bool isToken1 = address(collateral) < token;
        address token0 = isToken1 ? address(collateral) : token;
        address token1 = isToken1 ? token : address(collateral);
        uint160 sqrtStart = _priceToSqrtX96(priceWad, isToken1);
        address pool = uniFactory.createPool(token0, token1, FEE);
        ILocalUniswapV3Pool(pool).initialize(sqrtStart);
        int24 spacing = ILocalUniswapV3Pool(pool).tickSpacing();
        int24 tickLower = _minUsableTick(spacing);
        int24 tickUpper = _maxUsableTick(spacing);
        uint256 localBalance = IERC20(token).balanceOf(address(this));
        if (localBalance > 0) IERC20(token).transfer(address(mintHelper), localBalance);
        if (IERC20(token).balanceOf(address(mintHelper)) == 0) deal(token, address(mintHelper), LP_SEED);
        mintHelper.mintLiquidity(pool, tickLower, tickUpper, DEFAULT_LIQUIDITY);
        (uint160 actualSqrt, int24 tick,,,,,) = ILocalUniswapV3Pool(pool).slot0();
        assertEq(uint256(actualSqrt), uint256(sqrtStart), "pool price orientation");
        assertGt(ILocalUniswapV3Pool(pool).liquidity(), 0, "pool liquidity");
        tradeable = TradeablePool({
            name: name,
            market: market,
            outcomeIndex: outcomeIndex,
            token: token,
            pool: pool,
            isToken1: isToken1,
            tick: tick,
            tickLower: tickLower,
            tickUpper: tickUpper,
            priceWad: priceWad,
            predictionWad: predictionWad,
            initialBalanceWad: initialBalanceWad,
            liquidity: DEFAULT_LIQUIDITY
        });
    }

    function _executeRustFixtureScenario(
        ConnectedScenario memory scenario,
        uint256 startingCashWad,
        bool forceMintAvailable,
        bool seedMergeInventory,
        uint256 toleranceWad
    ) internal {
        TradeExecutor executor = new TradeExecutor(address(this));
        _fundExecutor(scenario, executor, startingCashWad, seedMergeInventory);
        _approveExecutor(executor, scenario.tradeables, scenario.connector);
        if (seedMergeInventory) {
            _approveExecutorToken(executor, scenario.rootInvalid, seerRouter);
            if (scenario.childInvalid != address(0)) {
                _approveExecutorToken(executor, scenario.childInvalid, seerRouter);
            }
        }

        string memory json = _fixtureInputJson(scenario, address(executor), startingCashWad, forceMintAvailable);
        LocalFixtureResult memory fixture = _runFixture(scenario.id, json);
        assertGt(fixture.actionCount, 0, "fixture actions");
        assertGt(fixture.chunks.length, 0, "fixture chunks");

        uint256 preRaw = _portfolioRawEv(address(executor), scenario.tradeables);
        assertApproxEqAbs(preRaw, fixture.preRawEvWad, 2, "pre raw ev");

        uint256 totalGas;
        uint256 totalCalldata;
        for (uint256 i = 0; i < fixture.chunks.length; i++) {
            TradeExecutor.Call[] memory calls = _toExecutorCalls(fixture.chunks[i].calls);
            uint256 gasBefore = gasleft();
            executor.batchExecute(calls);
            uint256 gasUsed = gasBefore - gasleft();
            uint256 calldataBytes = abi.encodeCall(TradeExecutor.batchExecute, (calls)).length;
            assertLt(gasUsed, MAX_PACKED_TX_L2_GAS, "packed gas cap");
            totalGas += gasUsed;
            totalCalldata += calldataBytes;
            emit log_named_uint("chunk_gas", gasUsed);
            emit log_named_uint("chunk_calldata_bytes", calldataBytes);
        }

        uint256 postRaw = _portfolioRawEv(address(executor), scenario.tradeables);
        uint256 fee = (totalGas * GAS_PRICE_WEI + totalCalldata * L1_FEE_PER_BYTE_WEI) * ETH_USD;
        int256 realizedNet = int256(postRaw) - int256(preRaw) - int256(fee);
        emit log_named_uint("pre_raw_ev_wad", preRaw);
        emit log_named_uint("post_raw_ev_wad", postRaw);
        emit log_named_uint("l2_gas_used", totalGas);
        emit log_named_uint("calldata_bytes", totalCalldata);
        emit log_named_uint("modeled_fee_wad", fee);
        emit log_named_int("realized_net_ev_wad", realizedNet);

        assertGt(realizedNet, 0, "realized net ev");
        assertApproxEqAbs(postRaw, fixture.expectedRawEvWad, toleranceWad, "raw ev");
        assertApproxEqAbs(_positiveInt(realizedNet), fixture.estimatedNetEvWad, toleranceWad + fee / 20, "net ev");
        _assertNoHelperStranding(scenario);
    }

    function _executeOnchainSolverScenario(
        ConnectedScenario memory scenario,
        uint256 startingCashWad,
        uint256 toleranceWad
    ) internal {
        TradeExecutor executor = new TradeExecutor(address(this));
        _fundExecutor(scenario, executor, startingCashWad, false);
        _approveExecutor(executor, scenario.tradeables, address(0));

        Rebalancer rebalancer = new Rebalancer(swapRouter02, seerRouter);
        RebalancerMixed mixed = new RebalancerMixed(swapRouter02, seerRouter);
        _approveExecutorForSolver(executor, scenario.tradeables, address(rebalancer));
        _approveExecutorForSolver(executor, scenario.tradeables, address(mixed));

        Rebalancer.RebalanceParams memory params = _rebalanceParams(address(executor), scenario, startingCashWad);
        uint256 preRaw = _portfolioRawEv(address(executor), scenario.tradeables);

        _executeSolverCall(executor, address(rebalancer), abi.encodeCall(rebalancer.rebalance, (params)));
        uint256 afterRebalance = _portfolioRawEv(address(executor), scenario.tradeables);
        assertGt(afterRebalance + toleranceWad, preRaw, "rebalance raw ev");

        params = _rebalanceParams(address(executor), scenario, IERC20(address(collateral)).balanceOf(address(executor)));
        _executeSolverCall(executor, address(rebalancer), abi.encodeCall(rebalancer.rebalanceExact, (params, 24, 2)));

        params = _rebalanceParams(address(executor), scenario, IERC20(address(collateral)).balanceOf(address(executor)));
        _executeSolverCall(
            executor,
            address(mixed),
            abi.encodeCall(mixed.rebalanceMixedConstantL, (params, scenario.rootMarket, 24, 24, 0))
        );
        _assertNoTokenStranding(address(rebalancer), scenario.tradeables);
        _assertNoTokenStranding(address(mixed), scenario.tradeables);
        _assertNoHelperStranding(scenario);
    }

    function _executeSolverCall(TradeExecutor executor, address target, bytes memory data) internal {
        TradeExecutor.Call[] memory calls = new TradeExecutor.Call[](1);
        calls[0] = TradeExecutor.Call({to: target, data: data});
        uint256 gasBefore = gasleft();
        executor.batchExecute(calls);
        uint256 gasUsed = gasBefore - gasleft();
        assertLt(gasUsed, MAX_PACKED_TX_L2_GAS, "solver gas cap");
        emit log_named_uint("solver_gas", gasUsed);
        emit log_named_uint("solver_calldata_bytes", abi.encodeCall(TradeExecutor.batchExecute, (calls)).length);
    }

    function _fundExecutor(
        ConnectedScenario memory scenario,
        TradeExecutor executor,
        uint256 startingCashWad,
        bool seedMergeInventory
    ) internal {
        collateral.mint(address(executor), startingCashWad);
        uint256 rootSplit = seedMergeInventory ? MERGE_SEED * 2 : MERGE_SEED;
        collateral.mint(address(this), rootSplit);
        collateral.approve(seerRouter, type(uint256).max);
        ICTFRouter(seerRouter).splitPosition(address(collateral), scenario.rootMarket, rootSplit);
        if (seedMergeInventory && scenario.childMarket != address(0)) {
            IERC20(scenario.connector).approve(seerRouter, type(uint256).max);
            ICTFRouter(seerRouter).splitPosition(scenario.connector, scenario.childMarket, MERGE_SEED);
        }
        for (uint256 i = 0; i < scenario.tradeables.length; i++) {
            uint256 holding = scenario.tradeables[i].initialBalanceWad;
            if (holding > 0) {
                uint256 balance = IERC20(scenario.tradeables[i].token).balanceOf(address(this));
                if (balance >= holding) {
                    IERC20(scenario.tradeables[i].token).transfer(address(executor), holding);
                } else {
                    deal(scenario.tradeables[i].token, address(executor), holding);
                }
            }
        }
        if (seedMergeInventory) {
            if (scenario.connector != address(0)) IERC20(scenario.connector).transfer(address(executor), MERGE_SEED);
            IERC20(scenario.rootInvalid).transfer(address(executor), MERGE_SEED);
            if (scenario.childInvalid != address(0)) {
                IERC20(scenario.childInvalid).transfer(address(executor), MERGE_SEED);
            }
        }
    }

    function _approveExecutor(TradeExecutor executor, TradeablePool[] memory tradeables, address connector) internal {
        uint256 len = 2 + tradeables.length * 2 + (connector == address(0) ? 0 : 1);
        TradeExecutor.Call[] memory calls = new TradeExecutor.Call[](len);
        uint256 cursor;
        calls[cursor++] = _approveCall(address(collateral), swapRouter02);
        calls[cursor++] = _approveCall(address(collateral), seerRouter);
        if (connector != address(0)) calls[cursor++] = _approveCall(connector, seerRouter);
        for (uint256 i = 0; i < tradeables.length; i++) {
            calls[cursor++] = _approveCall(tradeables[i].token, swapRouter02);
            calls[cursor++] = _approveCall(tradeables[i].token, seerRouter);
        }
        executor.batchExecute(calls);
    }

    function _approveExecutorForSolver(TradeExecutor executor, TradeablePool[] memory tradeables, address solver)
        internal
    {
        TradeExecutor.Call[] memory calls = new TradeExecutor.Call[](1 + tradeables.length);
        calls[0] = _approveCall(address(collateral), solver);
        for (uint256 i = 0; i < tradeables.length; i++) {
            calls[i + 1] = _approveCall(tradeables[i].token, solver);
        }
        executor.batchExecute(calls);
    }

    function _approveExecutorToken(TradeExecutor executor, address token, address spender) internal {
        TradeExecutor.Call[] memory calls = new TradeExecutor.Call[](1);
        calls[0] = _approveCall(token, spender);
        executor.batchExecute(calls);
    }

    function _approveCall(address token, address spender) internal pure returns (TradeExecutor.Call memory) {
        return TradeExecutor.Call({to: token, data: abi.encodeCall(IERC20.approve, (spender, type(uint256).max))});
    }

    function _runFixture(string memory scenarioId, string memory inputJson)
        internal
        returns (LocalFixtureResult memory fixture)
    {
        string memory path = string.concat(
            vm.projectRoot(), "/test/fixtures/local_foundry_e2e_fixture_input_", scenarioId, ".json"
        );
        vm.writeFile(path, inputJson);
        string[] memory cmd = new string[](6);
        cmd[0] = "cargo";
        cmd[1] = "run";
        cmd[2] = "--quiet";
        cmd[3] = "--bin";
        cmd[4] = "local_foundry_e2e_fixture";
        cmd[5] = path;
        bytes memory output = vm.ffi(cmd);
        bytes memory payload = vm.parseJsonBytes(string(output), ".abi");
        fixture = abi.decode(payload, (LocalFixtureResult));
    }

    function _fixtureInputJson(
        ConnectedScenario memory scenario,
        address executor,
        uint256 startingCashWad,
        bool forceMintAvailable
    ) internal view returns (string memory) {
        return string.concat(
            "{",
            '"scenario_id":"',
            scenario.id,
            '",',
            '"solver":"native",',
            '"executor":"',
            vm.toString(executor),
            '",',
            '"starting_cash_wad":"',
            vm.toString(startingCashWad),
            '",',
            '"current_block":1,',
            '"max_stale_blocks":2,',
            '"gas_price_wei":',
            vm.toString(GAS_PRICE_WEI),
            ",",
            '"eth_usd_wad":"',
            vm.toString(ETH_USD_WAD),
            '",',
            '"l1_fee_per_byte_wei":',
            vm.toString(L1_FEE_PER_BYTE_WEI),
            ",",
            '"force_mint_available":',
            forceMintAvailable ? "true" : "false",
            ",",
            '"address_book":',
            _addressBookJson(scenario),
            ",",
            '"markets":',
            _marketsJson(scenario.tradeables),
            "}"
        );
    }

    function _addressBookJson(ConnectedScenario memory scenario) internal view returns (string memory) {
        return string.concat(
            "{",
            '"collateral":"',
            vm.toString(address(collateral)),
            '",',
            '"seer_router":"',
            vm.toString(seerRouter),
            '",',
            '"swap_router02":"',
            vm.toString(swapRouter02),
            '",',
            '"market1":"',
            vm.toString(scenario.rootMarket),
            '",',
            '"market2":"',
            vm.toString(scenario.childMarket),
            '",',
            '"market2_collateral_connector":"',
            vm.toString(scenario.connector),
            '"}'
        );
    }

    function _marketsJson(TradeablePool[] memory tradeables) internal view returns (string memory json) {
        json = "[";
        for (uint256 i = 0; i < tradeables.length; i++) {
            if (i > 0) json = string.concat(json, ",");
            json = string.concat(json, _marketJson(tradeables[i]));
        }
        json = string.concat(json, "]");
    }

    function _marketJson(TradeablePool memory tradeable) internal view returns (string memory) {
        string memory head = string.concat(
            "{",
            '"name":"',
            tradeable.name,
            '",',
            '"market_id":"',
            vm.toString(tradeable.market),
            '",',
            '"outcome_token":"',
            vm.toString(tradeable.token),
            '",',
            '"pool":"',
            vm.toString(tradeable.pool),
            '",',
            '"token0":"',
            vm.toString(ILocalUniswapV3Pool(tradeable.pool).token0()),
            '",',
            '"token1":"',
            vm.toString(ILocalUniswapV3Pool(tradeable.pool).token1()),
            '",',
            '"quote_token":"',
            vm.toString(address(collateral)),
            '",',
            '"sqrt_price_x96":"',
            vm.toString(uint256(_slot0Sqrt(tradeable.pool))),
            '",',
            '"tick":',
            vm.toString(int256(tradeable.tick)),
            ",",
            '"tick_lower":',
            vm.toString(int256(tradeable.tickLower)),
            ",",
            '"tick_upper":',
            vm.toString(int256(tradeable.tickUpper)),
            ",",
            '"liquidity":"',
            vm.toString(uint256(tradeable.liquidity)),
            '"'
        );
        string memory tail = string.concat(
            ",",
            '"price_wad":"',
            vm.toString(tradeable.priceWad),
            '",',
            '"prediction_wad":"',
            vm.toString(tradeable.predictionWad),
            '",',
            '"initial_balance_wad":"',
            vm.toString(tradeable.initialBalanceWad),
            '"}'
        );
        return string.concat(head, tail);
    }

    function _rebalanceParams(address executor, ConnectedScenario memory scenario, uint256 collateralAmount)
        internal
        view
        returns (Rebalancer.RebalanceParams memory params)
    {
        uint256 n = scenario.tradeables.length;
        address[] memory tokens = new address[](n);
        address[] memory pools = new address[](n);
        bool[] memory isToken1 = new bool[](n);
        uint256[] memory balances = new uint256[](n);
        uint160[] memory sqrtPredX96 = new uint160[](n);
        for (uint256 i = 0; i < n; i++) {
            tokens[i] = scenario.tradeables[i].token;
            pools[i] = scenario.tradeables[i].pool;
            isToken1[i] = scenario.tradeables[i].isToken1;
            balances[i] = IERC20(tokens[i]).balanceOf(executor);
            sqrtPredX96[i] = _priceToSqrtX96(scenario.tradeables[i].predictionWad, isToken1[i]);
        }
        params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: collateralAmount,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: FEE
        });
    }

    function _toExecutorCalls(LocalFixtureCall[] memory fixtureCalls)
        internal
        pure
        returns (TradeExecutor.Call[] memory calls)
    {
        calls = new TradeExecutor.Call[](fixtureCalls.length);
        for (uint256 i = 0; i < fixtureCalls.length; i++) {
            calls[i] = TradeExecutor.Call({to: fixtureCalls[i].to, data: fixtureCalls[i].data});
        }
    }

    function _portfolioRawEv(address account, TradeablePool[] memory tradeables) internal view returns (uint256 ev) {
        ev = IERC20(address(collateral)).balanceOf(account);
        for (uint256 i = 0; i < tradeables.length; i++) {
            ev += FullMath.mulDiv(IERC20(tradeables[i].token).balanceOf(account), tradeables[i].predictionWad, WAD);
        }
    }

    function _assertNoHelperStranding(ConnectedScenario memory scenario) internal view {
        assertEq(IERC20(address(collateral)).balanceOf(address(mintHelper)), 0, "mintHelper collateral");
        assertEq(IERC20(address(collateral)).balanceOf(seerRouter), 0, "seerRouter collateral");
        assertEq(IERC20(address(collateral)).balanceOf(swapRouter02), 0, "swapRouter collateral");
        for (uint256 i = 0; i < scenario.tradeables.length; i++) {
            assertEq(IERC20(scenario.tradeables[i].token).balanceOf(seerRouter), 0, "seerRouter outcome");
            assertEq(IERC20(scenario.tradeables[i].token).balanceOf(swapRouter02), 0, "swapRouter outcome");
        }
    }

    function _assertNoTokenStranding(address target, TradeablePool[] memory tradeables) internal view {
        assertEq(IERC20(address(collateral)).balanceOf(target), 0, "stranded collateral");
        for (uint256 i = 0; i < tradeables.length; i++) {
            assertEq(IERC20(tradeables[i].token).balanceOf(target), 0, "stranded outcome");
        }
    }

    function _sweepMintHelper(TradeablePool[] memory tradeables) internal {
        mintHelper.sweep(address(collateral), address(this));
        for (uint256 i = 0; i < tradeables.length; i++) {
            mintHelper.sweep(tradeables[i].token, address(this));
        }
    }

    function _seedLiquidityInventory(
        address rootMarket,
        address childMarket,
        uint256 rootNamedOutcomes,
        uint256 childNamedOutcomes
    ) internal {
        collateral.mint(address(this), LP_SEED * 2);
        collateral.mint(address(mintHelper), LP_SEED * 2);
        collateral.approve(seerRouter, type(uint256).max);
        ICTFRouter(seerRouter).splitPosition(address(collateral), rootMarket, LP_SEED);
        // The checked-in Seer child artifact does not support nested local split/unwrap here;
        // child LP inventory is supplied directly in _createTradeablePool.
        rootNamedOutcomes;
        childMarket;
        childNamedOutcomes;
    }

    function _createMarket(string memory label, uint256 namedOutcomes, uint256 parentOutcome, address parentMarket)
        internal
        returns (address)
    {
        string[] memory outcomes = new string[](namedOutcomes);
        string[] memory tokenNames = new string[](namedOutcomes);
        for (uint256 i = 0; i < namedOutcomes; i++) {
            outcomes[i] = string.concat(label, "_outcome_", vm.toString(i));
            tokenNames[i] = string.concat(label, "_T", vm.toString(i));
        }
        return ILocalMarketFactory(marketFactory)
            .createCategoricalMarket(
                ILocalMarketFactory.CreateMarketParams({
                marketName: label,
                outcomes: outcomes,
                questionStart: "",
                questionEnd: "",
                outcomeType: "",
                parentOutcome: parentOutcome,
                parentMarket: parentMarket,
                category: "test",
                lang: "en_US",
                lowerBound: 0,
                upperBound: 0,
                minBond: 1 ether,
                openingTime: uint32(block.timestamp + 1),
                tokenNames: tokenNames
            })
            );
    }

    function _wrappedOutcome(address market, uint256 index) internal view returns (address token) {
        (IERC20 wrapped,) = IMarket(market).wrappedOutcome(index);
        token = address(wrapped);
    }

    function _slot0Sqrt(address pool) internal view returns (uint160 sqrtPriceX96) {
        (sqrtPriceX96,,,,,,) = ILocalUniswapV3Pool(pool).slot0();
    }

    function _smallArray(uint256 a, uint256 b, uint256 c, uint256 d) internal pure returns (uint256[] memory out) {
        out = new uint256[](4);
        out[0] = a;
        out[1] = b;
        out[2] = c;
        out[3] = d;
    }

    function _priceToSqrtX96(uint256 priceWad, bool outcomeIsToken1) internal pure returns (uint160) {
        uint256 sqrtP = _sqrt(priceWad);
        if (outcomeIsToken1) {
            return uint160((79228162514264337593543950336 * 1e9) / sqrtP);
        }
        return uint160((sqrtP * 79228162514264337593543950336) / 1e9);
    }

    function _sqrt(uint256 x) internal pure returns (uint256 y) {
        if (x == 0) return 0;
        y = x;
        uint256 z = (x + 1) / 2;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }

    function _minUsableTick(int24 spacing) internal pure returns (int24) {
        return (TickMath.MIN_TICK / spacing) * spacing;
    }

    function _maxUsableTick(int24 spacing) internal pure returns (int24) {
        return (TickMath.MAX_TICK / spacing) * spacing;
    }

    function _positiveInt(int256 value) internal pure returns (uint256) {
        return value > 0 ? uint256(value) : 0;
    }

    function _deployFromArtifact(string memory path) internal returns (address deployed) {
        bytes memory bytecode = vm.getCode(path);
        assembly {
            deployed := create(0, add(bytecode, 0x20), mload(bytecode))
        }
        require(deployed != address(0), string.concat("Deploy failed: ", path));
    }

    function _deployFromArtifactWithArgs(string memory path, bytes memory args) internal returns (address deployed) {
        bytes memory bytecode = abi.encodePacked(vm.getCode(path), args);
        assembly {
            deployed := create(0, add(bytecode, 0x20), mload(bytecode))
        }
        require(deployed != address(0), string.concat("Deploy+args failed: ", path));
    }
}
