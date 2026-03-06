# BatchSwapRouter Test Coverage

## Goal
Keep `contracts/BatchSwapRouter.sol` fully covered after the API update to `exactInput` and `exactOutput`.

## API under test
- `exactInput(address[] tokenIns, uint256 amountIn, address tokenOut, uint256 amountOutTotalMinimum, uint24 fee, uint160 sqrtPriceLimitX96)`
- `exactInput(address[] tokenIns, uint256[] amountIn, address tokenOut, uint256 amountOutTotalMinimum, uint24 fee, uint160 sqrtPriceLimitX96)`
- `exactOutput(address[] tokenOuts, uint256 amountOut, address tokenIn, uint256 amountInTotalMax, uint24 fee, uint160 sqrtPriceLimitX96)`
- `exactOutput(address[] tokenOuts, uint256[] amountOut, address tokenIn, uint256 amountInTotalMax, uint24 fee, uint160 sqrtPriceLimitX96)`
- `address[]` selects per-swap tokens, and `uint256[]` overloads preserve unequal per-leg amounts.

## Test strategy
The suite now has two layers:
- deterministic unit tests for explicit branch control
- real Uniswap V3 integration tests that deploy contracts from published artifacts (including `SwapRouter02`)
- bounded fuzz tests for aggregate accounting, per-swap budget progression, and revert boundaries

- Shared mocks: `/Users/shotaro/proj/deep_trading/test/utils/BatchSwapRouterMocks.sol`
- Success-path tests: `/Users/shotaro/proj/deep_trading/test/BatchSwapRouter.t.sol`
- Branch/revert tests: `/Users/shotaro/proj/deep_trading/test/BatchSwapRouterBranches.t.sol`
- Fuzz tests: `/Users/shotaro/proj/deep_trading/test/BatchSwapRouterFuzz.t.sol`
- Real-router integration tests: `/Users/shotaro/proj/deep_trading/test/BatchSwapRouterUniswapIntegration.t.sol`

The router mock preserves key Uniswap-style semantics relevant to this contract:
- `exactInputSingle` mints output tokens to `params.recipient`, matching router delivery directly to the user.
- `exactOutputSingle` can pull `tokenIn` from the caller (`BatchSwapRouter`) via allowance and mints each output token to `params.recipient`.
- `exactOutputSingle` can optionally enforce `amountIn <= params.amountInMaximum` and records per-call `amountInMaximum` history for tight-budget assertions.

The integration fixture deploys from JSON artifacts resolved in this order:
- `@uniswap/v3-core` `UniswapV3Factory`
- `@uniswap/v3-periphery` `NonfungiblePositionManager`
- local `lib/swap-router-contracts` `SwapRouter02` artifact (compiled from commit `70bc2e40dfca294c1cea9bf67a4036732ee54303`)

For `UniswapV3Factory` and `NonfungiblePositionManager`, tests first try root `node_modules`, then fallback to `lib/swap-router-contracts/node_modules` so fixture setup is reproducible across local/CI layouts.

## Covered behavior matrix
`exactInput` coverage:
- constructor immutability readback (`router`)
- single swap success
- multi swap aggregate output
- multi swap with distinct input tokens and per-token router approvals
- empty swaps with `_amountOutTotalMinimum == 0`
- unequal per-swap input amount array success
- revert on `tokenIn.transferFrom` failure
- revert on `tokenIn.approve` failure
- revert on aggregate slippage (`amountOut < _amountOutTotalMinimum`)
- revert on token/amount array length mismatch
- success path remains valid even when output token `transfer` is disabled in mock (router sends directly to recipient)
- empty swaps with positive `_amountOutTotalMinimum` revert
- fuzzed aggregation invariants over bounded swap counts, amounts, and slippage thresholds

`exactOutput` coverage:
- single swap success with refund
- multi swap aggregate input and per-swap output mints to recipient
- three-swap tight-budget regression that validates remaining per-swap maxima (`totalMax`, `totalMax-spent1`, `totalMax-spent1-spent2`)
- boundary case where aggregate spend equals `amountInTotalMax`
- zero-remaining path (no refund transfer)
- unequal per-swap output amount array success
- revert on initial `tokenIn.transferFrom` failure
- revert on `tokenIn.approve` failure
- success path remains valid even when output token `transfer` is disabled in mock (no output transfer by `BatchSwapRouter`)
- revert on aggregate slippage (`amountIn > _amountInTotalMax`)
- revert when strict per-swap remaining maximum is exceeded
- revert on token/amount array length mismatch
- revert on refund transfer failure
- empty swaps path returning full refund and `amountIn == 0`
- fuzzed per-swap budget progression checks (`amountInMaximum` history), exact-spend/refund invariants, and bounded over-budget revert cases

`exactOutput` implementation note:
- per-swap `amountInMaximum` is computed from remaining budget (`amountInTotalMax - amountIn`) each iteration so multi-swap exact-output calls enforce both per-swap and aggregate ceilings.

## Verification commands

```bash
forge test -q
forge test --match-path test/BatchSwapRouterFuzz.t.sol
forge coverage --report summary
```

Current result:
- `/Users/shotaro/proj/deep_trading/contracts/BatchSwapRouter.sol`: 100% lines, 100% statements, 100% branches, 100% functions.
