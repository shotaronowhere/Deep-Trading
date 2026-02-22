// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";

import {BatchSwapRouter} from "../contracts/BatchSwapRouter.sol";
import {MockERC20} from "./utils/BatchSwapRouterMocks.sol";

interface INonfungiblePositionManagerLike {
    struct MintParams {
        address token0;
        address token1;
        uint24 fee;
        int24 tickLower;
        int24 tickUpper;
        uint256 amount0Desired;
        uint256 amount1Desired;
        uint256 amount0Min;
        uint256 amount1Min;
        address recipient;
        uint256 deadline;
    }

    function createAndInitializePoolIfNecessary(address token0, address token1, uint24 fee, uint160 sqrtPriceX96)
        external
        payable
        returns (address pool);

    function mint(MintParams calldata params)
        external
        payable
        returns (uint256 tokenId, uint128 liquidity, uint256 amount0, uint256 amount1);
}

contract BatchSwapRouterUniswapIntegrationTest is Test {
    uint24 internal constant FEE = 3_000;
    uint160 internal constant SQRT_PRICE_1_1 = 79_228_162_514_264_337_593_543_950_336;
    int24 internal constant MIN_TICK = -887_220;
    int24 internal constant MAX_TICK = 887_220;

    string internal constant V3_FACTORY_ARTIFACT_PRIMARY =
        "node_modules/@uniswap/v3-core/artifacts/contracts/UniswapV3Factory.sol/UniswapV3Factory.json";
    string internal constant V3_FACTORY_ARTIFACT_FALLBACK =
        "lib/swap-router-contracts/node_modules/@uniswap/v3-core/artifacts/contracts/UniswapV3Factory.sol/UniswapV3Factory.json";
    string internal constant NPM_ARTIFACT_PRIMARY =
        "node_modules/@uniswap/v3-periphery/artifacts/contracts/NonfungiblePositionManager.sol/NonfungiblePositionManager.json";
    string internal constant NPM_ARTIFACT_FALLBACK =
        "lib/swap-router-contracts/node_modules/@uniswap/v3-periphery/artifacts/contracts/NonfungiblePositionManager.sol/NonfungiblePositionManager.json";
    string internal constant ROUTER02_ARTIFACT_PRIMARY =
        "lib/swap-router-contracts/artifacts/contracts/SwapRouter02.sol/SwapRouter02.json";
    string internal constant ROUTER02_ARTIFACT_FALLBACK =
        "node_modules/@uniswap/swap-router-contracts/artifacts/contracts/SwapRouter02.sol/SwapRouter02.json";

    MockERC20 internal stable;
    MockERC20 internal tokenA;
    MockERC20 internal tokenB;

    INonfungiblePositionManagerLike internal positionManager;
    BatchSwapRouter internal batch;

    function setUp() public {
        stable = new MockERC20();
        tokenA = new MockERC20();
        tokenB = new MockERC20();

        uint256 initialBalance = 20_000_000 ether;
        stable.mint(address(this), initialBalance);
        tokenA.mint(address(this), initialBalance);
        tokenB.mint(address(this), initialBalance);

        string memory v3FactoryArtifact = _resolveArtifact(V3_FACTORY_ARTIFACT_PRIMARY, V3_FACTORY_ARTIFACT_FALLBACK);
        address factory = vm.deployCode(v3FactoryArtifact);
        address weth9 = address(0xBEEF);
        address descriptor = address(0);

        string memory npmArtifact = _resolveArtifact(NPM_ARTIFACT_PRIMARY, NPM_ARTIFACT_FALLBACK);
        address npm = vm.deployCode(npmArtifact, abi.encode(factory, weth9, descriptor));
        positionManager = INonfungiblePositionManagerLike(npm);

        string memory routerArtifact = _resolveArtifact(ROUTER02_ARTIFACT_PRIMARY, ROUTER02_ARTIFACT_FALLBACK);
        address router02 = vm.deployCode(routerArtifact, abi.encode(address(0), factory, npm, weth9));
        batch = new BatchSwapRouter(router02);

        stable.approve(address(positionManager), type(uint256).max);
        tokenA.approve(address(positionManager), type(uint256).max);
        tokenB.approve(address(positionManager), type(uint256).max);

        stable.approve(address(batch), type(uint256).max);
        tokenA.approve(address(batch), type(uint256).max);
        tokenB.approve(address(batch), type(uint256).max);

        _createPoolAndSeedLiquidity(tokenA, stable, 5_000_000 ether, 5_000_000 ether);
        _createPoolAndSeedLiquidity(tokenB, stable, 5_000_000 ether, 5_000_000 ether);
    }

    function testExactInputAgainstRealSwapRouter02AcrossTwoPools() public {
        uint256 amountInPerSwap = 1_000 ether;

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);

        uint256 stableBefore = stable.balanceOf(address(this));
        uint256 tokenABefore = tokenA.balanceOf(address(this));
        uint256 tokenBBefore = tokenB.balanceOf(address(this));

        uint256 amountOut = batch.exactInput(stable, amountInPerSwap, 1, FEE, 0, tokens);

        assertGt(amountOut, 0);
        assertEq(stable.balanceOf(address(this)) - stableBefore, amountOut);
        assertEq(tokenABefore - tokenA.balanceOf(address(this)), amountInPerSwap);
        assertEq(tokenBBefore - tokenB.balanceOf(address(this)), amountInPerSwap);
        assertEq(tokenA.balanceOf(address(batch)), 0);
        assertEq(tokenB.balanceOf(address(batch)), 0);
    }

    function testExactOutputAgainstRealSwapRouter02AcrossTwoPools() public {
        uint256 amountOutPerSwap = 100 ether;
        uint256 amountInMax = 5_000 ether;

        address[] memory tokens = new address[](2);
        tokens[0] = address(tokenA);
        tokens[1] = address(tokenB);

        uint256 stableBefore = stable.balanceOf(address(this));
        uint256 tokenABefore = tokenA.balanceOf(address(this));
        uint256 tokenBBefore = tokenB.balanceOf(address(this));

        uint256 amountIn = batch.exactOutput(stable, amountOutPerSwap, amountInMax, FEE, 0, tokens);

        assertGt(amountIn, 0);
        assertLe(amountIn, amountInMax);
        assertEq(stableBefore - stable.balanceOf(address(this)), amountIn);
        assertEq(tokenA.balanceOf(address(this)) - tokenABefore, amountOutPerSwap);
        assertEq(tokenB.balanceOf(address(this)) - tokenBBefore, amountOutPerSwap);
        assertEq(stable.balanceOf(address(batch)), 0);
    }

    function _createPoolAndSeedLiquidity(
        MockERC20 tokenX,
        MockERC20 tokenY,
        uint256 amountX,
        uint256 amountY
    ) internal {
        (address token0, address token1) = address(tokenX) < address(tokenY)
            ? (address(tokenX), address(tokenY))
            : (address(tokenY), address(tokenX));

        positionManager.createAndInitializePoolIfNecessary(token0, token1, FEE, SQRT_PRICE_1_1);

        (uint256 amount0Desired, uint256 amount1Desired) = address(tokenX) < address(tokenY)
            ? (amountX, amountY)
            : (amountY, amountX);

        positionManager.mint(
            INonfungiblePositionManagerLike.MintParams({
                token0: token0,
                token1: token1,
                fee: FEE,
                tickLower: MIN_TICK,
                tickUpper: MAX_TICK,
                amount0Desired: amount0Desired,
                amount1Desired: amount1Desired,
                amount0Min: 0,
                amount1Min: 0,
                recipient: address(this),
                deadline: block.timestamp
            })
        );
    }

    function _resolveArtifact(string memory primaryPath, string memory fallbackPath) internal view returns (string memory) {
        if (vm.exists(primaryPath)) {
            return primaryPath;
        }
        if (vm.exists(fallbackPath)) {
            return fallbackPath;
        }
        revert("artifact not found");
    }
}
