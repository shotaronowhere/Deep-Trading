# BatchSwapRouter Test Coverage

## Goal
Keep `contracts/BatchSwapRouter.sol` fully covered after the API update to `exactInput` and `exactOutput`.

## Test strategy
The suite now uses deterministic mocks instead of the previous local Uniswap fixture so every revert/success branch can be triggered directly.

- Shared mocks: `/Users/shotaro/proj/deep_trading/test/utils/BatchSwapRouterMocks.sol`
- Success-path tests: `/Users/shotaro/proj/deep_trading/test/BatchSwapRouter.t.sol`
- Branch/revert tests: `/Users/shotaro/proj/deep_trading/test/BatchSwapRouterBranches.t.sol`

The router mock preserves key Uniswap-style semantics relevant to this contract:
- `exactInputSingle` can mint output tokens to the caller (`BatchSwapRouter`), matching the downstream transfer expectation.
- `exactOutputSingle` can pull `tokenIn` from the caller via allowance and mint output tokens for per-swap transfers.

## Covered behavior matrix
`exactInput` coverage:
- constructor immutability readback (`router`)
- single swap success
- multi swap aggregate output
- empty swaps with `amountOutMin == 0`
- revert on `tokenIn.transferFrom` failure
- revert on `tokenIn.approve` failure
- revert on aggregate slippage (`amountOut < amountOutMin`)
- revert on final `tokenOut.transfer` failure
- empty swaps with positive `amountOutMin` revert

`exactOutput` coverage:
- single swap success with refund
- multi swap aggregate input and per-swap output transfers
- zero-remaining path (no refund transfer)
- revert on initial `tokenIn.transferFrom` failure
- revert on `tokenIn.approve` failure
- revert on per-swap output-token transfer failure
- revert on aggregate slippage (`amountIn > amountInMax`)
- revert on refund transfer failure
- empty swaps path returning full refund and `amountIn == 0`

## Verification commands

```bash
forge test -q
forge coverage --report summary
```

Current result:
- `/Users/shotaro/proj/deep_trading/contracts/BatchSwapRouter.sol`: 100% lines, 100% statements, 100% branches, 100% functions.
