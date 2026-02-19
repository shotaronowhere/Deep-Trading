// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "forge-std/StdJson.sol";
import {BatchRouter, IV3SwapRouter, IERC20} from "../../contracts/BatchRouter.sol";

interface IERC20Ext is IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IUniswapV3FactoryLike {
    function createPool(address tokenA, address tokenB, uint24 fee) external returns (address pool);
}

interface IUniswapV3PoolLike {
    function initialize(uint160 sqrtPriceX96) external;

    function mint(address recipient, int24 tickLower, int24 tickUpper, uint128 amount, bytes calldata data)
        external
        returns (uint256 amount0, uint256 amount1);
}

interface IPeripherySwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }

    struct ExactOutputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountOut;
        uint256 amountInMaximum;
        uint160 sqrtPriceLimitX96;
    }

    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);

    function exactOutputSingle(ExactOutputSingleParams calldata params) external payable returns (uint256 amountIn);
}

contract MintableERC20 {
    string public name;
    string public symbol;
    uint8 public immutable decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 amount);
    event Approval(address indexed owner, address indexed spender, uint256 amount);

    constructor(string memory _name, string memory _symbol) {
        name = _name;
        symbol = _symbol;
    }

    function mint(address to, uint256 amount) external {
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        if (allowed != type(uint256).max) {
            allowance[from][msg.sender] = allowed - amount;
        }

        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
        return true;
    }
}

contract DummyWETH9 is MintableERC20 {
    constructor() MintableERC20("Wrapped Ether", "WETH") {}

    function deposit() external payable {
        balanceOf[msg.sender] += msg.value;
        totalSupply += msg.value;
        emit Transfer(address(0), msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        balanceOf[msg.sender] -= amount;
        totalSupply -= amount;
        emit Transfer(msg.sender, address(0), amount);
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok, "WETH_WITHDRAW_FAILED");
    }
}

/// @dev Bridges BatchRouter IV3SwapRouter interface to v3-periphery ISwapRouter (adds deadline).
contract V3PeripheryRouterAdapter is IV3SwapRouter {
    IPeripherySwapRouter public immutable swapRouter;

    constructor(address _swapRouter) {
        swapRouter = IPeripherySwapRouter(_swapRouter);
    }

    function exactInputSingle(ExactInputSingleParams calldata params)
        external
        payable
        override
        returns (uint256 amountOut)
    {
        require(
            IERC20Ext(params.tokenIn).transferFrom(msg.sender, address(this), params.amountIn),
            "TRANSFER_IN_FAILED"
        );
        require(IERC20(params.tokenIn).approve(address(swapRouter), params.amountIn), "APPROVE_FAILED");

        IPeripherySwapRouter.ExactInputSingleParams memory peripheryParams = IPeripherySwapRouter.ExactInputSingleParams({
            tokenIn: params.tokenIn,
            tokenOut: params.tokenOut,
            fee: params.fee,
            recipient: params.recipient,
            deadline: block.timestamp,
            amountIn: params.amountIn,
            amountOutMinimum: params.amountOutMinimum,
            sqrtPriceLimitX96: params.sqrtPriceLimitX96
        });

        amountOut = swapRouter.exactInputSingle(peripheryParams);
    }

    function exactOutputSingle(ExactOutputSingleParams calldata params)
        external
        payable
        override
        returns (uint256 amountIn)
    {
        uint256 balanceBefore = IERC20(params.tokenIn).balanceOf(address(this));
        require(
            IERC20Ext(params.tokenIn).transferFrom(msg.sender, address(this), params.amountInMaximum),
            "TRANSFER_IN_MAX_FAILED"
        );
        require(IERC20(params.tokenIn).approve(address(swapRouter), params.amountInMaximum), "APPROVE_FAILED");

        IPeripherySwapRouter.ExactOutputSingleParams memory peripheryParams = IPeripherySwapRouter.ExactOutputSingleParams({
            tokenIn: params.tokenIn,
            tokenOut: params.tokenOut,
            fee: params.fee,
            recipient: params.recipient,
            deadline: block.timestamp,
            amountOut: params.amountOut,
            amountInMaximum: params.amountInMaximum,
            sqrtPriceLimitX96: params.sqrtPriceLimitX96
        });

        amountIn = swapRouter.exactOutputSingle(peripheryParams);

        uint256 balanceAfter = IERC20(params.tokenIn).balanceOf(address(this));
        uint256 refund = balanceAfter - balanceBefore;
        if (refund > 0) {
            require(IERC20(params.tokenIn).transfer(msg.sender, refund), "REFUND_FAILED");
        }
    }
}

abstract contract UniswapV3Fixture is Test {
    using stdJson for string;

    uint24 internal constant FEE_LOW = 500;
    uint24 internal constant FEE_MEDIUM = 3000;
    uint160 internal constant SQRT_PRICE_1_1 = 79228162514264337593543950336;
    int24 internal constant TICK_LOWER = -600;
    int24 internal constant TICK_UPPER = 600;
    uint128 internal constant LIQUIDITY = 1_000_000_000_000_000_000;

    string internal constant V3_FACTORY_ARTIFACT =
        "node_modules/@uniswap/v3-core/artifacts/contracts/UniswapV3Factory.sol/UniswapV3Factory.json";
    string internal constant V3_SWAP_ROUTER_ARTIFACT =
        "node_modules/@uniswap/v3-periphery/artifacts/contracts/SwapRouter.sol/SwapRouter.json";

    MintableERC20 internal tokenIn;
    MintableERC20 internal tokenOut;
    MintableERC20 internal tokenOutAlt;

    BatchRouter internal batchRouter;
    V3PeripheryRouterAdapter internal v3Adapter;

    address internal v3Factory;
    address internal v3PeripheryRouter;

    address internal poolInOutLow;
    address internal poolInOutMedium;
    address internal poolInOutAlt;

    function _deployFixture() internal {
        v3Factory = _deployArtifact(V3_FACTORY_ARTIFACT, bytes(""));
        address weth = address(new DummyWETH9());
        v3PeripheryRouter = _deployArtifact(V3_SWAP_ROUTER_ARTIFACT, abi.encode(v3Factory, weth));
        v3Adapter = new V3PeripheryRouterAdapter(v3PeripheryRouter);

        tokenIn = new MintableERC20("Token In", "TIN");
        tokenOut = new MintableERC20("Token Out", "TOUT");
        tokenOutAlt = new MintableERC20("Token Out Alt", "TOALT");

        tokenIn.mint(address(this), 1_000_000 ether);
        tokenOut.mint(address(this), 1_000_000 ether);
        tokenOutAlt.mint(address(this), 1_000_000 ether);

        poolInOutLow = _createPoolAndSeedLiquidity(address(tokenIn), address(tokenOut), FEE_LOW);
        poolInOutMedium = _createPoolAndSeedLiquidity(address(tokenIn), address(tokenOut), FEE_MEDIUM);
        poolInOutAlt = _createPoolAndSeedLiquidity(address(tokenIn), address(tokenOutAlt), FEE_LOW);

        batchRouter = new BatchRouter(address(v3Adapter));
    }

    function _deployArtifact(string memory path, bytes memory constructorArgs) internal returns (address deployed) {
        string memory json = vm.readFile(path);
        bytes memory creationCode = json.readBytes(".bytecode");
        bytes memory initCode = abi.encodePacked(creationCode, constructorArgs);

        assembly {
            deployed := create(0, add(initCode, 0x20), mload(initCode))
        }
        require(deployed != address(0), "ARTIFACT_DEPLOY_FAILED");
    }

    function _createPoolAndSeedLiquidity(address tokenA, address tokenB, uint24 fee) internal returns (address pool) {
        pool = IUniswapV3FactoryLike(v3Factory).createPool(tokenA, tokenB, fee);
        IUniswapV3PoolLike(pool).initialize(SQRT_PRICE_1_1);

        (address token0, address token1) = tokenA < tokenB ? (tokenA, tokenB) : (tokenB, tokenA);
        IUniswapV3PoolLike(pool).mint(address(this), TICK_LOWER, TICK_UPPER, LIQUIDITY, abi.encode(token0, token1));
    }

    function uniswapV3MintCallback(uint256 amount0Owed, uint256 amount1Owed, bytes calldata data) external {
        (address token0, address token1) = abi.decode(data, (address, address));

        if (amount0Owed > 0) {
            require(IERC20(token0).transfer(msg.sender, amount0Owed), "PAY_TOKEN0_FAILED");
        }
        if (amount1Owed > 0) {
            require(IERC20(token1).transfer(msg.sender, amount1Owed), "PAY_TOKEN1_FAILED");
        }
    }
}
