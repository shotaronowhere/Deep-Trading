// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../contracts/BatchRouter.sol";

interface IERC20Extended is IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract MockERC20 is IERC20Extended {
    string public name;
    string public symbol;
    uint8 public decimals;

    mapping(address => uint256) public override balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
    }

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external override returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external override returns (bool) {
        require(balanceOf[msg.sender] >= amount, "insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external override returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        require(allowed >= amount, "insufficient allowance");
        require(balanceOf[from] >= amount, "insufficient balance");
        allowance[from][msg.sender] = allowed - amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract MockV3SwapRouter is IV3SwapRouter {
    uint256 public inputRateNum = 1;
    uint256 public inputRateDen = 1;
    uint256 public outputRateNum = 1;
    uint256 public outputRateDen = 1;

    function setInputRate(uint256 num, uint256 den) external {
        require(den > 0, "zero den");
        inputRateNum = num;
        inputRateDen = den;
    }

    function setOutputRate(uint256 num, uint256 den) external {
        require(den > 0, "zero den");
        outputRateNum = num;
        outputRateDen = den;
    }

    function exactInputSingle(
        ExactInputSingleParams calldata params
    ) external payable override returns (uint256 amountOut) {
        amountOut = (params.amountIn * inputRateNum) / inputRateDen;
        IERC20Extended(params.tokenIn).transferFrom(msg.sender, address(this), params.amountIn);
        IERC20(params.tokenOut).transfer(params.recipient, amountOut);
    }

    function exactOutputSingle(
        ExactOutputSingleParams calldata params
    ) external payable override returns (uint256 amountIn) {
        amountIn = (params.amountOut * outputRateNum) / outputRateDen;
        require(amountIn <= params.amountInMaximum, "router max in");
        IERC20Extended(params.tokenIn).transferFrom(msg.sender, address(this), amountIn);
        IERC20(params.tokenOut).transfer(params.recipient, params.amountOut);
    }
}

contract BatchRouterTest {
    MockERC20 private tokenIn;
    MockERC20 private tokenOut;
    MockERC20 private otherOut;
    MockV3SwapRouter private router;
    BatchRouter private batch;

    constructor() {
        tokenIn = new MockERC20("Token In", "TIN", 18);
        tokenOut = new MockERC20("Token Out", "TOUT", 18);
        otherOut = new MockERC20("Other Out", "OOUT", 18);
        router = new MockV3SwapRouter();
        batch = new BatchRouter(address(router));

        tokenIn.mint(address(this), 10_000 ether);
        tokenOut.mint(address(router), 10_000 ether);
        otherOut.mint(address(router), 10_000 ether);
    }

    function testSellSweepsWhenRecipientIsRouter() external {
        uint256 amountIn = 10 ether;
        tokenIn.transfer(address(batch), amountIn);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(batch),
            amountIn: amountIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        uint256 beforeOut = tokenOut.balanceOf(address(this));
        uint256 out = batch.sell(swaps, amountIn);
        uint256 afterOut = tokenOut.balanceOf(address(this));

        require(out == amountIn, "unexpected out");
        require(afterOut == beforeOut + amountIn, "sweep failed");
        require(tokenOut.balanceOf(address(batch)) == 0, "router retained output");
    }

    function testSellMinViolationReverts() external {
        uint256 amountIn = 10 ether;
        tokenIn.transfer(address(batch), amountIn);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(batch),
            amountIn: amountIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        try batch.sell(swaps, amountIn + 1) returns (uint256) {
            revert("expected sell slippage revert");
        } catch {
        }
    }

    function testBuyEmptyReturnsZero() external {
        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](0);
        uint256 spent = batch.buy(swaps, 0);
        require(spent == 0, "empty buy should return zero");
    }

    function testBuyMaxViolationReverts() external {
        router.setOutputRate(2, 1);
        tokenIn.transfer(address(batch), 100 ether);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 10 ether,
            amountInMaximum: 100 ether,
            sqrtPriceLimitX96: 0
        });

        try batch.buy(swaps, 9 ether) returns (uint256) {
            revert("expected buy slippage revert");
        } catch {
        }
    }

    function testBuyRefundsUnusedTokenIn() external {
        tokenIn.transfer(address(batch), 100 ether);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 10 ether,
            amountInMaximum: 100 ether,
            sqrtPriceLimitX96: 0
        });

        uint256 before = tokenIn.balanceOf(address(this));
        uint256 spent = batch.buy(swaps, 100 ether);
        uint256 afterBalance = tokenIn.balanceOf(address(this));

        require(spent == 10 ether, "unexpected amount in");
        require(afterBalance == before + 90 ether, "unused tokenIn was not refunded");
        require(tokenIn.balanceOf(address(batch)) == 0, "router retained tokenIn");
    }

    function testSellRejectsMixedTokenOut() external {
        tokenIn.transfer(address(batch), 20 ether);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](2);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountIn: 10 ether,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });
        swaps[1] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(otherOut),
            fee: 500,
            recipient: address(this),
            amountIn: 10 ether,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        try batch.sell(swaps, 0) returns (uint256) {
            revert("expected invalid config revert");
        } catch {
        }
    }

    function testSellRejectsMixedRecipient() external {
        tokenIn.transfer(address(batch), 20 ether);

        IV3SwapRouter.ExactInputSingleParams[] memory swaps = new IV3SwapRouter.ExactInputSingleParams[](2);
        swaps[0] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountIn: 10 ether,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });
        swaps[1] = IV3SwapRouter.ExactInputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(batch),
            amountIn: 10 ether,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        try batch.sell(swaps, 0) returns (uint256) {
            revert("expected invalid config revert");
        } catch {
        }
    }

    function testBuyRejectsMixedTokenIn() external {
        MockERC20 anotherIn = new MockERC20("Another In", "AIN", 18);
        anotherIn.mint(address(this), 100 ether);

        tokenIn.transfer(address(batch), 100 ether);
        anotherIn.transfer(address(batch), 100 ether);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](2);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 10 ether,
            amountInMaximum: 100 ether,
            sqrtPriceLimitX96: 0
        });
        swaps[1] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(anotherIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(this),
            amountOut: 5 ether,
            amountInMaximum: 100 ether,
            sqrtPriceLimitX96: 0
        });

        try batch.buy(swaps, 100 ether) returns (uint256) {
            revert("expected invalid config revert");
        } catch {
        }
    }

    function testBuyRejectsInvalidRecipient() external {
        tokenIn.transfer(address(batch), 100 ether);

        IV3SwapRouter.ExactOutputSingleParams[] memory swaps = new IV3SwapRouter.ExactOutputSingleParams[](1);
        swaps[0] = IV3SwapRouter.ExactOutputSingleParams({
            tokenIn: address(tokenIn),
            tokenOut: address(tokenOut),
            fee: 500,
            recipient: address(router),
            amountOut: 10 ether,
            amountInMaximum: 100 ether,
            sqrtPriceLimitX96: 0
        });

        try batch.buy(swaps, 100 ether) returns (uint256) {
            revert("expected invalid config revert");
        } catch {
        }
    }
}
