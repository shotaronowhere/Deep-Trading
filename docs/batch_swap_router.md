# BatchSwapRouter

This is the canonical doc for `contracts/BatchSwapRouter.sol`: external API plus test/coverage status.

## Contract

- Solidity implementation: `contracts/BatchSwapRouter.sol`
- Interface: `contracts/interfaces/IBatchSwapRouter.sol`

## External API

### exactInput

`exactInput(address[] tokenIns, uint256[] amountIns, uint160[] sqrtPriceLimitsX96, address tokenOut, uint256 amountOutTotalMinimum, uint24 fee)`

- Sells multiple input tokens into one output token.
- `tokenIns`, `amountIns`, and `sqrtPriceLimitsX96` must have equal lengths.
- Per-leg partial fills are bounded by `sqrtPriceLimitsX96` and unswapped caller-owned remainders are refunded.
- Reverts with `SlippageExceeded()` if aggregate output `< amountOutTotalMinimum`.

### exactOutput

`exactOutput(address[] tokenOuts, uint256[] amountOuts, uint160[] sqrtPriceLimitsX96, address tokenIn, uint256 amountInTotalMax, uint24 fee)`

- Buys multiple output tokens using one shared input budget.
- `tokenOuts`, `amountOuts`, and `sqrtPriceLimitsX96` must have equal lengths.
- Unused input is refunded after the batch.
- Reverts with `SlippageExceeded()` if aggregate input `> amountInTotalMax`.

### waterfallBuy

`waterfallBuy(address[] tokenOuts, uint160[] sqrtPriceLimitsX96, address tokenIn, uint256 amountIn, uint24 fee)`

- Spends a shared input budget sequentially across output tokens.
- Each leg consumes current remaining budget and stops at its own price limit.
- Unspent input is refunded after execution.

## Behavior and invariants

- Array length mismatch reverts with `InvalidArrayLength()`.
- Router balance handling is caller-delta scoped; pre-existing router balances are not swept.
- Per-leg `sqrtPriceLimitX96` is required for every leg (`0` is explicit no-limit semantics).
- Constructor takes one argument: `(address router)`.

## Tests and coverage

Coverage suites:

- Success-path tests: `test/BatchSwapRouter.t.sol`
- Branch/revert tests: `test/BatchSwapRouterBranches.t.sol`
- Fuzz tests: `test/BatchSwapRouterFuzz.t.sol`
- Real-router integration tests: `test/BatchSwapRouterUniswapIntegration.t.sol`
- Shared mocks: `test/utils/BatchSwapRouterMocks.sol`

Covered behavior includes:

- constructor/router immutability checks
- single/multi leg `exactInput` and `exactOutput`
- unequal per-leg amount arrays
- aggregate slippage bound enforcement
- per-swap remaining-budget enforcement (`exactOutput`)
- array mismatch reverts
- transfer/approve failure reverts
- empty-swap edge cases
- bounded fuzz invariants for aggregate accounting and budget progression

Current reported result in committed notes:

- `BatchSwapRouter.sol`: 100% lines/statements/branches/functions

## Verification commands

```bash
forge test -q
forge test --match-path test/BatchSwapRouterFuzz.t.sol
forge coverage --report summary
```
