// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";

import {Rebalancer} from "../contracts/Rebalancer.sol";
import {RebalancerMixed} from "../contracts/RebalancerMixed.sol";
import {IV3SwapRouter} from "../contracts/interfaces/IV3SwapRouter.sol";

contract MixedMockERC20 {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function burnFrom(address from, uint256 amount) external returns (bool) {
        if (balanceOf[from] < amount) return false;

        uint256 allowed = allowance[from][msg.sender];
        if (allowed < amount) return false;

        allowance[from][msg.sender] = allowed - amount;
        balanceOf[from] -= amount;
        return true;
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

contract MixedMockPool {
    uint160 internal currentSqrtPriceX96;

    constructor(uint160 sqrtPriceX96_) {
        currentSqrtPriceX96 = sqrtPriceX96_;
    }

    function slot0() external view returns (uint160, int24, uint16, uint16, uint16, uint8, bool) {
        return (currentSqrtPriceX96, 0, 0, 0, 0, 0, false);
    }

    function liquidity() external pure returns (uint128) {
        return 1;
    }
}

contract MixedMockRouter {
    struct Config {
        uint256 fixedSpend;
    }

    mapping(address => Config) internal configs;

    function configure(address tokenOut, uint256 fixedSpend) external {
        configs[tokenOut] = Config({fixedSpend: fixedSpend});
    }

    function exactInputSingle(IV3SwapRouter.ExactInputSingleParams calldata params) external payable returns (uint256 amountOut) {
        Config memory config = configs[params.tokenOut];
        uint256 spend = config.fixedSpend == 0 ? params.amountIn : config.fixedSpend;
        if (spend > params.amountIn) spend = params.amountIn;

        bool success = MixedMockERC20(params.tokenIn).transferFrom(msg.sender, address(this), spend);
        require(success, "transferFrom failed");

        MixedMockERC20(params.tokenOut).mint(params.recipient, spend);
        return spend;
    }

    function exactOutputSingle(IV3SwapRouter.ExactOutputSingleParams calldata) external payable returns (uint256 amountIn) {
        return amountIn;
    }
}

contract MixedMockCTFRouter {
    address[] internal tokens;
    uint256 public splitCount;
    uint256 public mergeCount;

    function setTokens(address[] memory tokens_) external {
        delete tokens;
        for (uint256 i = 0; i < tokens_.length; i++) {
            tokens.push(tokens_[i]);
        }
    }

    function splitPosition(address collateralToken, address, uint256 amount) external {
        bool success = MixedMockERC20(collateralToken).transferFrom(msg.sender, address(this), amount);
        require(success, "collateral pull failed");

        for (uint256 i = 0; i < tokens.length; i++) {
            MixedMockERC20(tokens[i]).mint(msg.sender, amount);
        }

        splitCount++;
    }

    function mergePositions(address collateralToken, address, uint256 amount) external {
        for (uint256 i = 0; i < tokens.length; i++) {
            bool success = MixedMockERC20(tokens[i]).burnFrom(msg.sender, amount);
            require(success, "burn failed");
        }

        bool collateralSent = MixedMockERC20(collateralToken).transfer(msg.sender, amount);
        require(collateralSent, "collateral send failed");
        mergeCount++;
    }
}

contract RebalancerMixedTest is Test {
    uint160 internal constant TEST_Q96 = uint160(1 << 96);

    function testRebalanceMixedUsesDirectStepWhenPriceSumIsBelowOne() public {
        MixedMockERC20 collateral = new MixedMockERC20();
        MixedMockERC20 tokenA = new MixedMockERC20();
        MixedMockERC20 tokenB = new MixedMockERC20();
        MixedMockERC20 tokenC = new MixedMockERC20();

        MixedMockPool poolA = new MixedMockPool(TEST_Q96 / 4);
        MixedMockPool poolB = new MixedMockPool(TEST_Q96 / 4);
        MixedMockPool poolC = new MixedMockPool(TEST_Q96 / 2);

        MixedMockRouter router = new MixedMockRouter();
        MixedMockCTFRouter ctfRouter = new MixedMockCTFRouter();
        RebalancerMixed rebalancer = new RebalancerMixed(address(router), address(ctfRouter));

        address[] memory tokens = new address[](3);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);
        tokens[2] = address(tokenC);
        ctfRouter.setTokens(tokens);

        router.configure(address(tokenA), 2);
        router.configure(address(tokenB), 2);

        address[] memory pools = new address[](3);
        pools[0] = address(poolA);
        pools[1] = address(poolB);
        pools[2] = address(poolC);

        bool[] memory isToken1 = new bool[](3);
        uint256[] memory balances = new uint256[](3);
        uint160[] memory sqrtPredX96 = new uint160[](3);
        sqrtPredX96[0] = TEST_Q96 / 2;
        sqrtPredX96[1] = TEST_Q96 / 2;
        sqrtPredX96[2] = (TEST_Q96 * 5) / 8;

        collateral.mint(address(this), 5);
        collateral.approve(address(rebalancer), 5);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 5,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        (uint256 proceeds, uint256 spent) = rebalancer.rebalanceMixed(params, address(0xBEEF), 1, 5);

        assertEq(proceeds, 0);
        assertEq(spent, 4);
        assertEq(ctfRouter.splitCount(), 0);
        assertEq(ctfRouter.mergeCount(), 0);
        assertEq(collateral.balanceOf(address(this)), 1);
        assertEq(tokenA.balanceOf(address(this)), 2);
        assertEq(tokenB.balanceOf(address(this)), 2);
        assertEq(tokenC.balanceOf(address(this)), 0);
    }

    function testRebalanceMixedCanStillUseDirectWhenTotalPriceIsAboveOne() public {
        MixedMockERC20 collateral = new MixedMockERC20();
        MixedMockERC20 tokenA = new MixedMockERC20();
        MixedMockERC20 tokenB = new MixedMockERC20();
        MixedMockERC20 tokenC = new MixedMockERC20();

        MixedMockPool poolA = new MixedMockPool((TEST_Q96 * 2) / 3);
        MixedMockPool poolB = new MixedMockPool((TEST_Q96 * 2) / 3);
        MixedMockPool poolC = new MixedMockPool((TEST_Q96 * 3) / 5);

        MixedMockRouter router = new MixedMockRouter();
        MixedMockCTFRouter ctfRouter = new MixedMockCTFRouter();
        RebalancerMixed rebalancer = new RebalancerMixed(address(router), address(ctfRouter));

        address[] memory tokens = new address[](3);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);
        tokens[2] = address(tokenC);
        ctfRouter.setTokens(tokens);

        router.configure(address(tokenA), 1);
        router.configure(address(tokenB), 1);

        address[] memory pools = new address[](3);
        pools[0] = address(poolA);
        pools[1] = address(poolB);
        pools[2] = address(poolC);

        bool[] memory isToken1 = new bool[](3);
        uint256[] memory balances = new uint256[](3);
        uint160[] memory sqrtPredX96 = new uint160[](3);
        sqrtPredX96[0] = (TEST_Q96 * 4) / 5;
        sqrtPredX96[1] = (TEST_Q96 * 4) / 5;
        sqrtPredX96[2] = (TEST_Q96 * 7) / 10;

        collateral.mint(address(this), 3);
        collateral.approve(address(rebalancer), 3);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 3,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        (uint256 proceeds, uint256 spent) = rebalancer.rebalanceMixed(params, address(0xBEEF), 1, 3);

        assertEq(proceeds, 0);
        assertEq(spent, 2);
        assertEq(ctfRouter.splitCount(), 0);
        assertEq(ctfRouter.mergeCount(), 0);
        assertEq(collateral.balanceOf(address(this)), 1);
        assertEq(tokenA.balanceOf(address(this)), 1);
        assertEq(tokenB.balanceOf(address(this)), 1);
        assertEq(tokenC.balanceOf(address(this)), 0);
    }

    function testRebalanceMixedUsesMintStepWhenPriceSumIsAboveOne() public {
        MixedMockERC20 collateral = new MixedMockERC20();
        MixedMockERC20 tokenA = new MixedMockERC20();
        MixedMockERC20 tokenB = new MixedMockERC20();
        MixedMockERC20 tokenC = new MixedMockERC20();
        MixedMockERC20 tokenD = new MixedMockERC20();

        MixedMockPool poolA = new MixedMockPool((TEST_Q96 * 3) / 4);
        MixedMockPool poolB = new MixedMockPool((TEST_Q96 * 3) / 4);
        MixedMockPool poolC = new MixedMockPool(TEST_Q96 / 2);
        MixedMockPool poolD = new MixedMockPool(TEST_Q96 / 2);

        MixedMockRouter router = new MixedMockRouter();
        MixedMockCTFRouter ctfRouter = new MixedMockCTFRouter();
        RebalancerMixed rebalancer = new RebalancerMixed(address(router), address(ctfRouter));

        address[] memory tokens = new address[](4);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);
        tokens[2] = address(tokenC);
        tokens[3] = address(tokenD);
        ctfRouter.setTokens(tokens);

        router.configure(address(collateral), 5);

        address[] memory pools = new address[](4);
        pools[0] = address(poolA);
        pools[1] = address(poolB);
        pools[2] = address(poolC);
        pools[3] = address(poolD);

        bool[] memory isToken1 = new bool[](4);
        uint256[] memory balances = new uint256[](4);
        uint160[] memory sqrtPredX96 = new uint160[](4);
        sqrtPredX96[0] = (TEST_Q96 * 7) / 8;
        sqrtPredX96[1] = (TEST_Q96 * 7) / 8;
        sqrtPredX96[2] = (TEST_Q96 * 9) / 16;
        sqrtPredX96[3] = (TEST_Q96 * 17) / 32;

        collateral.mint(address(this), 5);
        collateral.approve(address(rebalancer), 5);

        Rebalancer.RebalanceParams memory params = Rebalancer.RebalanceParams({
            tokens: tokens,
            pools: pools,
            isToken1: isToken1,
            balances: balances,
            collateralAmount: 5,
            sqrtPredX96: sqrtPredX96,
            collateral: address(collateral),
            fee: 0
        });

        (uint256 proceeds, uint256 spent) = rebalancer.rebalanceMixed(params, address(0xBEEF), 1, 5);

        assertEq(proceeds, 0);
        assertEq(spent, 0);
        assertEq(ctfRouter.splitCount(), 1);
        assertEq(ctfRouter.mergeCount(), 0);
        assertEq(collateral.balanceOf(address(this)), 5);
        assertEq(tokenA.balanceOf(address(this)), 5);
        assertEq(tokenB.balanceOf(address(this)), 5);
        assertEq(tokenC.balanceOf(address(this)), 5);
        assertEq(tokenD.balanceOf(address(this)), 0);
    }
}
