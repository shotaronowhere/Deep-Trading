// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";

import {TradeExecutor} from "../contracts/TradeExecutor.sol";
import {IV3SwapRouter} from "../contracts/interfaces/IV3SwapRouter.sol";

interface IProfileCTFRouter {
    function splitPosition(address collateralToken, address market, uint256 amount) external;
    function mergePositions(address collateralToken, address market, uint256 amount) external;
}

contract GasProfileERC20 {
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

contract GasProfileSwapRouter {
    function exactInputSingle(IV3SwapRouter.ExactInputSingleParams calldata params)
        external
        payable
        returns (uint256 amountOut)
    {
        bool success = GasProfileERC20(params.tokenIn).transferFrom(msg.sender, address(this), params.amountIn);
        require(success, "transferFrom failed");
        GasProfileERC20(params.tokenOut).mint(params.recipient, params.amountIn);
        return params.amountIn;
    }

    function exactOutputSingle(IV3SwapRouter.ExactOutputSingleParams calldata params)
        external
        payable
        returns (uint256 amountIn)
    {
        bool success =
            GasProfileERC20(params.tokenIn).transferFrom(msg.sender, address(this), params.amountInMaximum);
        require(success, "transferFrom failed");
        GasProfileERC20(params.tokenOut).mint(params.recipient, params.amountOut);
        return params.amountInMaximum;
    }
}

contract GasProfileCTFRouter {
    address[] internal tokens;

    function setTokens(address[] memory tokens_) external {
        delete tokens;
        for (uint256 i = 0; i < tokens_.length; i++) {
            tokens.push(tokens_[i]);
        }
    }

    function splitPosition(address collateralToken, address, uint256 amount) external {
        bool success = GasProfileERC20(collateralToken).transferFrom(msg.sender, address(this), amount);
        require(success, "collateral pull failed");

        for (uint256 i = 0; i < tokens.length; i++) {
            GasProfileERC20(tokens[i]).mint(msg.sender, amount);
        }
    }

    function mergePositions(address collateralToken, address, uint256 amount) external {
        for (uint256 i = 0; i < tokens.length; i++) {
            bool success = GasProfileERC20(tokens[i]).burnFrom(msg.sender, amount);
            require(success, "burn failed");
        }

        bool collateralSent = GasProfileERC20(collateralToken).transfer(msg.sender, amount);
        require(collateralSent, "collateral send failed");
    }
}

contract TradeExecutorGasProfileTest is Test {
    uint24 internal constant FEE = 100;
    uint160 internal constant Q96 = uint160(1 << 96);
    uint256 internal constant APPROVAL_AMOUNT = 1_000_000 ether;
    uint256 internal constant CALL_AMOUNT = 1 ether;

    TradeExecutor internal executor;
    GasProfileSwapRouter internal router;
    GasProfileCTFRouter internal ctfRouter;
    GasProfileERC20 internal collateral;

    function setUp() public {
        executor = new TradeExecutor(address(this));
        router = new GasProfileSwapRouter();
        ctfRouter = new GasProfileCTFRouter();
        collateral = new GasProfileERC20();
        collateral.mint(address(executor), APPROVAL_AMOUNT);
        collateral.mint(address(ctfRouter), APPROVAL_AMOUNT * 2);
    }

    function testGasProfileDirectBuy() public {
        GasProfileERC20 token = _newOutcomeToken();
        _configureTokens(singleton(address(token)));
        _approveExecutor(singleton(address(collateral)), address(router));

        uint256 gasUsed = _measure(_buildDirectBuyCalls(token));
        emit log_named_uint("trade_executor_direct_buy_gas", gasUsed);
    }

    function testGasProfileDirectSell() public {
        GasProfileERC20 token = _newOutcomeToken();
        token.mint(address(executor), APPROVAL_AMOUNT);
        _configureTokens(singleton(address(token)));
        _approveExecutor(singleton(address(token)), address(router));

        uint256 gasUsed = _measure(_buildDirectSellCalls(token));
        emit log_named_uint("trade_executor_direct_sell_gas", gasUsed);
    }

    function testGasProfileDirectMerge() public {
        address[] memory tokens = _newOutcomeTokens(2);
        _configureTokens(tokens);
        _mintOutcomesToExecutor(tokens, APPROVAL_AMOUNT);
        _approveExecutor(tokens, address(ctfRouter));

        uint256 gasUsed = _measure(_buildDirectMergeCalls());
        emit log_named_uint("trade_executor_direct_merge_gas", gasUsed);
    }

    function testGasProfileMintSellOneLeg() public {
        _profileMintSell(1);
    }

    function testGasProfileMintSellFiveLegs() public {
        _profileMintSell(5);
    }

    function testGasProfileMintSellTwentyLegs() public {
        _profileMintSell(20);
    }

    function testGasProfileMintSellNinetySevenLegs() public {
        _profileMintSell(97);
    }

    function testGasProfileBuyMergeOneLeg() public {
        _profileBuyMerge(1);
    }

    function testGasProfileBuyMergeFiveLegs() public {
        _profileBuyMerge(5);
    }

    function testGasProfileBuyMergeTwentyLegs() public {
        _profileBuyMerge(20);
    }

    function testGasProfileBuyMergeNinetySevenLegs() public {
        _profileBuyMerge(97);
    }

    function _profileMintSell(uint256 legs) internal {
        address[] memory tokens = _newOutcomeTokens(legs);
        _configureTokens(tokens);
        _approveExecutor(singleton(address(collateral)), address(ctfRouter));
        _approveExecutor(tokens, address(router));

        uint256 gasUsed = _measure(_buildMintSellCalls(tokens));
        emit log_named_uint(_mintSellLabel(legs), gasUsed);
    }

    function _profileBuyMerge(uint256 legs) internal {
        address[] memory tokens = _newOutcomeTokens(legs);
        _configureTokens(tokens);
        _approveExecutor(singleton(address(collateral)), address(router));
        _approveExecutor(tokens, address(ctfRouter));

        uint256 gasUsed = _measure(_buildBuyMergeCalls(tokens));
        emit log_named_uint(_buyMergeLabel(legs), gasUsed);
    }

    function _configureTokens(address[] memory tokens) internal {
        ctfRouter.setTokens(tokens);
    }

    function _measure(TradeExecutor.Call[] memory calls) internal returns (uint256 gasUsed) {
        vm.resumeGasMetering();
        uint256 gasStart = gasleft();
        executor.batchExecute(calls);
        gasUsed = gasStart - gasleft();
        vm.pauseGasMetering();
    }

    function _approveExecutor(address[] memory tokens, address spender) internal {
        vm.pauseGasMetering();
        TradeExecutor.Call[] memory calls = new TradeExecutor.Call[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            calls[i] = TradeExecutor.Call({
                to: tokens[i],
                data: abi.encodeWithSignature("approve(address,uint256)", spender, APPROVAL_AMOUNT)
            });
        }
        executor.batchExecute(calls);
    }

    function _buildDirectBuyCalls(GasProfileERC20 token)
        internal
        view
        returns (TradeExecutor.Call[] memory calls)
    {
        calls = new TradeExecutor.Call[](1);
        calls[0] = TradeExecutor.Call({
            to: address(router),
            data: abi.encodeCall(
                IV3SwapRouter.exactOutputSingle,
                (
                    IV3SwapRouter.ExactOutputSingleParams({
                        tokenIn: address(collateral),
                        tokenOut: address(token),
                        fee: FEE,
                        recipient: address(executor),
                        amountOut: CALL_AMOUNT,
                        amountInMaximum: CALL_AMOUNT,
                        sqrtPriceLimitX96: Q96
                    })
                )
            )
        });
    }

    function _buildDirectSellCalls(GasProfileERC20 token)
        internal
        view
        returns (TradeExecutor.Call[] memory calls)
    {
        calls = new TradeExecutor.Call[](1);
        calls[0] = TradeExecutor.Call({
            to: address(router),
            data: abi.encodeCall(
                IV3SwapRouter.exactInputSingle,
                (
                    IV3SwapRouter.ExactInputSingleParams({
                        tokenIn: address(token),
                        tokenOut: address(collateral),
                        fee: FEE,
                        recipient: address(executor),
                        amountIn: CALL_AMOUNT,
                        amountOutMinimum: CALL_AMOUNT,
                        sqrtPriceLimitX96: Q96
                    })
                )
            )
        });
    }

    function _buildDirectMergeCalls() internal view returns (TradeExecutor.Call[] memory calls) {
        calls = new TradeExecutor.Call[](1);
        calls[0] = TradeExecutor.Call({
            to: address(ctfRouter),
            data: abi.encodeCall(
                IProfileCTFRouter.mergePositions, (address(collateral), address(0xBEEF), CALL_AMOUNT)
            )
        });
    }

    function _buildMintSellCalls(address[] memory tokens)
        internal
        view
        returns (TradeExecutor.Call[] memory calls)
    {
        calls = new TradeExecutor.Call[](tokens.length + 1);
        calls[0] = TradeExecutor.Call({
            to: address(ctfRouter),
            data: abi.encodeCall(
                IProfileCTFRouter.splitPosition, (address(collateral), address(0xBEEF), CALL_AMOUNT)
            )
        });
        for (uint256 i = 0; i < tokens.length; i++) {
            calls[i + 1] = TradeExecutor.Call({
                to: address(router),
                data: abi.encodeCall(
                    IV3SwapRouter.exactInputSingle,
                    (
                        IV3SwapRouter.ExactInputSingleParams({
                            tokenIn: tokens[i],
                            tokenOut: address(collateral),
                            fee: FEE,
                            recipient: address(executor),
                            amountIn: CALL_AMOUNT,
                            amountOutMinimum: CALL_AMOUNT,
                            sqrtPriceLimitX96: Q96
                        })
                    )
                )
            });
        }
    }

    function _buildBuyMergeCalls(address[] memory tokens)
        internal
        view
        returns (TradeExecutor.Call[] memory calls)
    {
        calls = new TradeExecutor.Call[](tokens.length + 1);
        for (uint256 i = 0; i < tokens.length; i++) {
            calls[i] = TradeExecutor.Call({
                to: address(router),
                data: abi.encodeCall(
                    IV3SwapRouter.exactOutputSingle,
                    (
                        IV3SwapRouter.ExactOutputSingleParams({
                            tokenIn: address(collateral),
                            tokenOut: tokens[i],
                            fee: FEE,
                            recipient: address(executor),
                            amountOut: CALL_AMOUNT,
                            amountInMaximum: CALL_AMOUNT,
                            sqrtPriceLimitX96: Q96
                        })
                    )
                )
            });
        }
        calls[tokens.length] = TradeExecutor.Call({
            to: address(ctfRouter),
            data: abi.encodeCall(
                IProfileCTFRouter.mergePositions, (address(collateral), address(0xBEEF), CALL_AMOUNT)
            )
        });
    }

    function _newOutcomeToken() internal returns (GasProfileERC20 token) {
        token = new GasProfileERC20();
    }

    function _newOutcomeTokens(uint256 count) internal returns (address[] memory tokens) {
        tokens = new address[](count);
        for (uint256 i = 0; i < count; i++) {
            tokens[i] = address(_newOutcomeToken());
        }
    }

    function _mintOutcomesToExecutor(address[] memory tokens, uint256 amount) internal {
        for (uint256 i = 0; i < tokens.length; i++) {
            GasProfileERC20(tokens[i]).mint(address(executor), amount);
        }
    }

    function singleton(address value) internal pure returns (address[] memory values) {
        values = new address[](1);
        values[0] = value;
    }

    function _mintSellLabel(uint256 legs) internal pure returns (string memory) {
        if (legs == 1) return "trade_executor_mint_sell_1_gas";
        if (legs == 5) return "trade_executor_mint_sell_5_gas";
        if (legs == 20) return "trade_executor_mint_sell_20_gas";
        if (legs == 97) return "trade_executor_mint_sell_97_gas";
        return "trade_executor_mint_sell_unknown_gas";
    }

    function _buyMergeLabel(uint256 legs) internal pure returns (string memory) {
        if (legs == 1) return "trade_executor_buy_merge_1_gas";
        if (legs == 5) return "trade_executor_buy_merge_5_gas";
        if (legs == 20) return "trade_executor_buy_merge_20_gas";
        if (legs == 97) return "trade_executor_buy_merge_97_gas";
        return "trade_executor_buy_merge_unknown_gas";
    }
}
