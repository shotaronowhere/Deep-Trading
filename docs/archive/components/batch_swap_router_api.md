# Batch Swap Router API

This document describes the current `IBatchSwapRouter` ABI implemented by [`BatchSwapRouter.sol`](/Users/shotaro/proj/deep_trading/contracts/BatchSwapRouter.sol).

## External functions

`exactInput(address[] tokenIns, uint256[] amountIns, uint160[] sqrtPriceLimitsX96, address tokenOut, uint256 amountOutTotalMinimum, uint24 fee)`

- Sells multiple input tokens into one output token.
- Each leg uses its own input amount and `sqrtPriceLimitX96`.
- `tokenIns`, `amountIns`, and `sqrtPriceLimitsX96` must be the same length or the call reverts with `InvalidArrayLength()`.
- Any unswapped remainder from a price-limited partial fill is refunded after that leg.
- The refund is limited to the caller-owned delta from that leg; pre-existing router balances are not swept out.
- Reverts with `SlippageExceeded()` if the aggregate output is below `amountOutTotalMinimum`.

`exactOutput(address[] tokenOuts, uint256[] amountOuts, uint160[] sqrtPriceLimitsX96, address tokenIn, uint256 amountInTotalMax, uint24 fee)`

- Buys multiple output tokens using one shared input token budget.
- Each leg uses its own desired output amount and `sqrtPriceLimitX96`.
- `tokenOuts`, `amountOuts`, and `sqrtPriceLimitsX96` must be the same length or the call reverts with `InvalidArrayLength()`.
- Unused caller input is refunded after the batch completes.
- Reverts with `SlippageExceeded()` if aggregate input exceeds `amountInTotalMax`.

`waterfallBuy(address[] tokenOuts, uint160[] sqrtPriceLimitsX96, address tokenIn, uint256 amountIn, uint24 fee)`

- Spends a shared `tokenIn` budget sequentially across `tokenOuts`.
- Each leg uses the current remaining balance as input and stops at its own `sqrtPriceLimitX96`.
- `tokenOuts` and `sqrtPriceLimitsX96` must be the same length or the call reverts with `InvalidArrayLength()`.
- Any unspent budget is refunded at the end.
- Pre-existing router inventory is left untouched; `amountInSpent` reports only the consumed portion of the caller-provided `amountIn`.

## Notes

- The ABI is intentionally array-based and uses per-leg price limits; the older scalar overloads are no longer part of the interface.
- The batch-sell implementation helper (`_exactInput`) is internal only; callers must use the aggregate-checked `exactInput(...)` entry point.
- Off-chain callers must supply a `sqrtPriceLimitsX96` entry for every leg, even when they want the legacy "no limit" behavior (`0` for each leg).
- `BatchSwapRouter` has a single constructor argument: `(address router)`.
