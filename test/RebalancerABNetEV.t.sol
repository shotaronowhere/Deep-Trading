// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "forge-std/StdJson.sol";

import {Rebalancer} from "../contracts/Rebalancer.sol";
import {RebalancerMixed} from "../contracts/RebalancerMixed.sol";
import {IERC20} from "../contracts/interfaces/IERC20.sol";
import {ICTFRouter} from "../contracts/interfaces/ICTFRouter.sol";
import {IMarket} from "../contracts/interfaces/IMarket.sol";
import {FullMath} from "../contracts/libraries/FullMath.sol";
import {TickMath} from "../contracts/libraries/TickMath.sol";

interface IUniswapV3Factory {
    function createPool(address tokenA, address tokenB, uint24 fee) external returns (address pool);
    function feeAmountTickSpacing(uint24 fee) external view returns (int24);
    function enableFeeAmount(uint24 fee, int24 tickSpacing) external;
}

interface IUniswapV3Pool {
    function initialize(uint160 sqrtPriceX96) external;
    function mint(address recipient, int24 tickLower, int24 tickUpper, uint128 amount, bytes calldata data)
        external
        returns (uint256 amount0, uint256 amount1);
    function slot0() external view returns (uint160, int24, uint16, uint16, uint16, uint8, bool);
    function liquidity() external view returns (uint128);
    function token0() external view returns (address);
    function token1() external view returns (address);
}

interface IMarketFactory {
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

contract NetEVSUSD {
    string public constant name = "NetEV sUSD";
    string public constant symbol = "sUSD";
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
        require(balanceOf[msg.sender] >= amount, "sUSD: insufficient");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "sUSD: insufficient");
        uint256 allowed = allowance[from][msg.sender];
        if (allowed != type(uint256).max) {
            require(allowed >= amount, "sUSD: allowance");
            allowance[from][msg.sender] = allowed - amount;
        }
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract NetEVMintHelper {
    function mintLiquidity(address pool, int24 tickLower, int24 tickUpper, uint128 amount) external {
        IUniswapV3Pool(pool).mint(address(this), tickLower, tickUpper, amount, "");
    }

    function uniswapV3MintCallback(uint256 amount0, uint256 amount1, bytes calldata) external {
        if (amount0 > 0) IERC20(IUniswapV3Pool(msg.sender).token0()).transfer(msg.sender, amount0);
        if (amount1 > 0) IERC20(IUniswapV3Pool(msg.sender).token1()).transfer(msg.sender, amount1);
    }
}

contract RebalancerABNetEV is Test {
    using stdJson for string;

    uint256 constant WAD = 1e18;
    // OP benchmark snapshot pricing
    uint256 constant GAS_PRICE_WEI = 1_002_325;
    uint256 constant ETH_USD = 3000;
    uint256 constant L1_FEE_PER_BYTE_WEI = 1_643_855;
    uint256 constant INVALID_PRICE_WAD = 1e15; // 0.001 sUSD

    NetEVSUSD sUSD;
    address conditionalTokens;
    address wrapped1155Factory;
    address realitio;
    IUniswapV3Factory uniFactory;
    address swapRouter02;
    address weth9;
    NetEVMintHelper mintHelper;
    address seerRouter;
    address marketFactoryAddr;

    function setUp() public {
        sUSD = new NetEVSUSD();

        conditionalTokens = _deployFromArtifact("test/fixtures/ConditionalTokens.json");
        realitio = _deployFromArtifact("test/fixtures/RealityETH_v3_0.json");
        wrapped1155Factory = _deployFromArtifact("test/fixtures/Wrapped1155Factory.json");

        address realityProxy =
            _deployFromArtifactWithArgs("test/fixtures/SeerRealityProxy.json", abi.encode(conditionalTokens, realitio));
        address marketImpl = _deployFromArtifact("test/fixtures/SeerMarket.json");
        marketFactoryAddr = _deployFromArtifactWithArgs(
            "test/fixtures/SeerMarketFactory.json",
            abi.encode(
                marketImpl,
                address(1),
                realitio,
                wrapped1155Factory,
                conditionalTokens,
                address(sUSD),
                realityProxy,
                uint32(1 days)
            )
        );
        seerRouter = _deployFromArtifactWithArgs(
            "test/fixtures/SeerRouter.json", abi.encode(conditionalTokens, wrapped1155Factory)
        );

        weth9 = _deployFromArtifact("test/fixtures/WETH9.json");
        uniFactory = IUniswapV3Factory(_deployFromArtifact("test/fixtures/UniswapV3Factory.json"));
        uniFactory.enableFeeAmount(100, 1);
        swapRouter02 = _deployFromArtifactWithArgs(
            "lib/swap-router-contracts/artifacts/contracts/SwapRouter02.sol/SwapRouter02.json",
            abi.encode(address(0), address(uniFactory), address(0), weth9)
        );
        mintHelper = new NetEVMintHelper();
    }

    // ────── Test functions ──────

    function test_two_pool_single_tick_direct_only() external {
        _runJsonCase("two_pool_single_tick_direct_only");
    }

    function test_small_bundle_mixed_case() external {
        _runJsonCase("small_bundle_mixed_case");
    }

    function test_legacy_holdings_direct_only_case() external {
        _runJsonCase("legacy_holdings_direct_only_case");
    }

    function test_mixed_route_favorable_synthetic_case() external {
        _runJsonCase("mixed_route_favorable_synthetic_case");
    }

    function test_ninety_eight_outcome_multitick_direct_only() external {
        _runJsonCase("ninety_eight_outcome_multitick_direct_only");
    }

    function test_heterogeneous_ninety_eight_outcome_l1_like_case() external {
        _runJsonCase("heterogeneous_ninety_eight_outcome_l1_like_case");
    }

    // ────── Orchestrator ──────

    function _runJsonCase(string memory caseId) internal {
        string memory json = vm.readFile(string.concat(vm.projectRoot(), "/test/fixtures/rebalancer_ab_cases.json"));
        string memory prefix = string.concat(".", caseId);

        uint256 uniformCount = json.readUint(string.concat(prefix, ".uniform_count"));
        uint256 cashWad = json.readUint(string.concat(prefix, ".initial_cash_budget_wad"));
        uint24 feeTier = uint24(json.readUint(string.concat(prefix, ".fee_tier")));

        uint256[] memory predsWad;
        uint256[] memory pricesWad;
        uint256[] memory liquidities;
        uint256[] memory holdingsWad;

        if (uniformCount > 0) {
            uint256 uniformPriceBps = json.readUint(string.concat(prefix, ".uniform_price_bps"));
            uint256 uniformLiquidity = json.readUint(string.concat(prefix, ".uniform_liquidity"));

            predsWad = new uint256[](uniformCount);
            pricesWad = new uint256[](uniformCount);
            liquidities = new uint256[](uniformCount);
            holdingsWad = new uint256[](uniformCount);

            uint256 basePred = WAD / uniformCount;
            uint256 remainder = WAD - basePred * uniformCount;

            for (uint256 i = 0; i < uniformCount; i++) {
                uint256 pred = basePred;
                if (i == 0) pred += remainder;
                predsWad[i] = pred;
                pricesWad[i] = pred * uniformPriceBps / 10_000;
                liquidities[i] = uniformLiquidity;
            }
        } else {
            predsWad = json.readUintArray(string.concat(prefix, ".predictions_wad"));
            pricesWad = json.readUintArray(string.concat(prefix, ".starting_prices_wad"));
            liquidities = json.readUintArray(string.concat(prefix, ".liquidity"));
            holdingsWad = json.readUintArray(string.concat(prefix, ".initial_holdings_wad"));
        }

        _setupAndRun(caseId, predsWad, pricesWad, liquidities, holdingsWad, cashWad, feeTier);
    }

    function _setupAndRun(
        string memory caseId,
        uint256[] memory predsWad,
        uint256[] memory pricesWad,
        uint256[] memory liquidities,
        uint256[] memory holdingsWad,
        uint256 cashWad,
        uint24 feeTier
    ) internal {
        uint256 n = predsWad.length;
        uint256 total = n + 1; // +1 for invalid outcome

        // ── 1. Create Seer market ──
        string[] memory outcomes = new string[](n);
        string[] memory tokenNames = new string[](n);
        for (uint256 i = 0; i < n; i++) {
            outcomes[i] = vm.toString(i);
            tokenNames[i] = string.concat("T", vm.toString(i));
        }
        address market = IMarketFactory(marketFactoryAddr).createCategoricalMarket(
            IMarketFactory.CreateMarketParams({
                marketName: caseId,
                outcomes: outcomes,
                questionStart: "",
                questionEnd: "",
                outcomeType: "",
                parentOutcome: 0,
                parentMarket: address(0),
                category: "test",
                lang: "en_US",
                lowerBound: 0,
                upperBound: 0,
                minBond: 1 ether,
                openingTime: uint32(block.timestamp + 1),
                tokenNames: tokenNames
            })
        );

        // ── 2. Get outcome token addresses ──
        address[] memory tokens = new address[](total);
        for (uint256 i = 0; i < total; i++) {
            (IERC20 w,) = IMarket(market).wrappedOutcome(i);
            tokens[i] = address(w);
        }

        // ── 3. Determine token ordering ──
        bool[] memory isToken1 = new bool[](total);
        for (uint256 i = 0; i < total; i++) {
            isToken1[i] = address(sUSD) < tokens[i];
        }

        // ── 4. Split sUSD to get outcome tokens for LP ──
        uint256 maxLiq;
        for (uint256 i = 0; i < n; i++) {
            if (liquidities[i] > maxLiq) maxLiq = liquidities[i];
        }
        uint256 splitAmount = maxLiq * 300;
        if (splitAmount < 50_000e18) splitAmount = 50_000e18;

        sUSD.mint(address(this), splitAmount);
        sUSD.approve(seerRouter, splitAmount);
        ICTFRouter(seerRouter).splitPosition(address(sUSD), market, splitAmount);

        // ── 5. Transfer all tokens to mintHelper for LP ──
        for (uint256 i = 0; i < total; i++) {
            IERC20(tokens[i]).transfer(address(mintHelper), IERC20(tokens[i]).balanceOf(address(this)));
        }
        // Also give mintHelper sUSD for pool token0 side
        sUSD.mint(address(mintHelper), splitAmount);

        // ── 6. Create pools and seed liquidity (per-pool tick ranges) ──
        (address[] memory poolAddrs, uint256 maxTickCrossings) =
            _createPools(tokens, isToken1, predsWad, pricesWad, liquidities, n, feeTier);

        // ── 7. Set exact actor balances ──
        for (uint256 i = 0; i < total; i++) {
            uint256 holding = (i < n && i < holdingsWad.length) ? holdingsWad[i] : 0;
            deal(tokens[i], address(this), holding);
        }
        // sUSD: address(this) currently has 0 (all transferred/consumed). Mint fresh.
        sUSD.mint(address(this), cashWad);

        // ── 8. Build predictions (N+1: invalid gets same as price → no trade) ──
        uint256[] memory fullPreds = new uint256[](total);
        for (uint256 i = 0; i < n; i++) {
            fullPreds[i] = predsWad[i];
        }
        fullPreds[n] = INVALID_PRICE_WAD; // prediction = price → at fair value

        // ── 9. Deploy solvers ──
        Rebalancer rebal = new Rebalancer(swapRouter02, seerRouter);
        RebalancerMixed rebalMixed = new RebalancerMixed(swapRouter02, seerRouter);

        // ── 10. Approve ──
        sUSD.approve(address(rebal), type(uint256).max);
        sUSD.approve(address(rebalMixed), type(uint256).max);
        for (uint256 i = 0; i < total; i++) {
            IERC20(tokens[i]).approve(address(rebal), type(uint256).max);
            IERC20(tokens[i]).approve(address(rebalMixed), type(uint256).max);
        }

        // ── 11. Build params ──
        uint256[] memory balances = new uint256[](total);
        uint160[] memory sqrtPredX96 = new uint160[](total);
        for (uint256 i = 0; i < total; i++) {
            balances[i] = IERC20(tokens[i]).balanceOf(address(this));
            sqrtPredX96[i] = _priceToSqrtX96(fullPreds[i], isToken1[i]);
        }

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: poolAddrs,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: cashWad,
            sqrtPredX96: sqrtPredX96,
            collateral: address(sUSD),
            fee: feeTier
        });

        // ── 12. Encode calldatas for byte measurement ──
        uint256 cdLen1 = abi.encodeCall(rebal.rebalance, (params)).length;
        uint256 cdLen2 = abi.encodeCall(rebal.rebalanceExact, (params, 24, maxTickCrossings)).length;
        uint256 cdLen3 = abi.encodeCall(rebalMixed.rebalanceMixedConstantL, (params, market, 24, 24, 0)).length;

        // ── 13. Snapshot → run each solver → revert ──
        emit log_string(string.concat("=== ", caseId, " ==="));

        uint256 snap = vm.snapshot();

        // rebalance (constant-L direct)
        uint256 g = gasleft();
        rebal.rebalance(params);
        _logSolver("rebalance", _portfolioEv(tokens, fullPreds), g - gasleft(), cdLen1);

        vm.revertTo(snap);
        snap = vm.snapshot();

        // rebalanceExact (try/catch: may OOG for large pool counts)
        g = gasleft();
        try rebal.rebalanceExact(params, 24, maxTickCrossings) {
            _logSolver("rebalanceExact", _portfolioEv(tokens, fullPreds), g - gasleft(), cdLen2);
        } catch {
            emit log_named_string("solver", "rebalanceExact");
            emit log_string("  FAILED (MemoryOOG or TickScanLimit)");
        }

        vm.revertTo(snap);

        // rebalanceMixedConstantL
        g = gasleft();
        try rebalMixed.rebalanceMixedConstantL(params, market, 24, 24, 0) {
            _logSolver("mixedConstantL", _portfolioEv(tokens, fullPreds), g - gasleft(), cdLen3);
        } catch {
            emit log_named_string("solver", "mixedConstantL");
            emit log_string("  FAILED");
        }
    }

    // ────── Helpers ──────

    function _createPools(
        address[] memory tokens,
        bool[] memory isToken1,
        uint256[] memory predsWad,
        uint256[] memory pricesWad,
        uint256[] memory liquidities,
        uint256 n,
        uint24 feeTier
    ) internal returns (address[] memory poolAddrs, uint256 maxTickCrossings) {
        uint256 total = tokens.length;
        poolAddrs = new address[](total);
        for (uint256 i = 0; i < total; i++) {
            uint256 price = i < n ? pricesWad[i] : INVALID_PRICE_WAD;
            uint256 pred = i < n ? predsWad[i] : INVALID_PRICE_WAD;
            uint128 liq = uint128(i < n ? liquidities[i] : WAD);
            (poolAddrs[i], maxTickCrossings) =
                _createOnePool(tokens[i], isToken1[i], price, pred, liq, feeTier, maxTickCrossings);
        }
        maxTickCrossings += 500;
    }

    function _createOnePool(
        address outcome,
        bool isT1,
        uint256 price,
        uint256 pred,
        uint128 liq,
        uint24 feeTier,
        uint256 curMax
    ) internal returns (address pool, uint256 newMax) {
        address token0 = isT1 ? address(sUSD) : outcome;
        address token1 = isT1 ? outcome : address(sUSD);

        uint160 sqrtStart = _priceToSqrtX96(price, isT1);
        pool = uniFactory.createPool(token0, token1, feeTier);
        IUniswapV3Pool(pool).initialize(sqrtStart);

        int24 tickStart = _getTickAtSqrtRatio(sqrtStart);
        int24 tickPred = _getTickAtSqrtRatio(_priceToSqrtX96(pred, isT1));
        int24 lo = tickStart < tickPred ? tickStart : tickPred;
        int24 hi = tickStart > tickPred ? tickStart : tickPred;
        int24 spacing = uniFactory.feeAmountTickSpacing(feeTier);
        int24 tickLower = _floorTick(lo - 500, spacing);
        int24 tickUpper = _ceilTick(hi + 500, spacing);

        mintHelper.mintLiquidity(pool, tickLower, tickUpper, liq);

        uint256 span = uint256(uint24(tickUpper - tickLower));
        newMax = span > curMax ? span : curMax;
    }

    function _portfolioEv(address[] memory tokens, uint256[] memory preds) internal view returns (uint256 ev) {
        ev = sUSD.balanceOf(address(this));
        for (uint256 i = 0; i < tokens.length; i++) {
            ev += FullMath.mulDiv(IERC20(tokens[i]).balanceOf(address(this)), preds[i], WAD);
        }
    }

    function _logSolver(string memory name, uint256 rawEv, uint256 gasUsed, uint256 cdLen) internal {
        // fee in sUSD wei: (gas_cost_eth_wei + data_cost_eth_wei) * ETH_USD
        // 1 ETH_wei = ETH_USD sUSD_wei, so no /1e18 needed
        uint256 fee = (gasUsed * GAS_PRICE_WEI + cdLen * L1_FEE_PER_BYTE_WEI) * ETH_USD;
        uint256 netEv = rawEv > fee ? rawEv - fee : 0;

        emit log_named_string("solver", name);
        emit log_named_uint("raw_ev_wei", rawEv);
        emit log_named_uint("gas_used", gasUsed);
        emit log_named_uint("calldata_bytes", cdLen);
        emit log_named_uint("fee_susd_wei", fee);
        emit log_named_uint("net_ev_wei", netEv);
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

    function _priceToSqrtX96(uint256 priceWad, bool outcomeIsToken1) internal pure returns (uint160) {
        uint256 sqrtP = _sqrt(priceWad);
        if (outcomeIsToken1) {
            return uint160((79228162514264337593543950336 * 1e9) / sqrtP);
        } else {
            return uint160((sqrtP * 79228162514264337593543950336) / 1e9);
        }
    }

    function _getTickAtSqrtRatio(uint160 sqrtPriceX96) internal pure returns (int24) {
        int24 lo = TickMath.MIN_TICK;
        int24 hi = TickMath.MAX_TICK;
        while (lo < hi) {
            int24 mid = lo + (hi - lo + 1) / 2;
            if (TickMath.getSqrtRatioAtTick(mid) <= sqrtPriceX96) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    function _floorTick(int24 tick, int24 spacing) internal pure returns (int24) {
        int24 r = tick / spacing;
        if (tick < 0 && tick % spacing != 0) r -= 1;
        return r * spacing;
    }

    function _ceilTick(int24 tick, int24 spacing) internal pure returns (int24) {
        int24 r = tick / spacing;
        if (tick > 0 && tick % spacing != 0) r += 1;
        return r * spacing;
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
}
