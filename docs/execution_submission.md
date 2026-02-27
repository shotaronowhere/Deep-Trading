# TradeExecutor Submission Runbook

## Scope

Execution is TradeExecutor-only and strict-mode only:

- one subgroup per transaction
- replan after every submitted transaction
- fail closed on stale/deadline/shape/bounds/revert

`src/main.rs` remains planning/diagnostics.
Live submission is in `src/bin/execute.rs`.

## Environment

- `PRIVATE_KEY` (required): signer key used for owner-controlled executor calls.
- `RPC` (optional, default `https://optimism.drpc.org`): JSON-RPC endpoint.
- `EXECUTE_SUBMIT` (optional, default `0`): `1/true/yes/on` submits transactions.
- `EXECUTION_MAX_STEPS` (optional, default `32`): strict loop cap.
- `EXECUTION_MAX_STALE_BLOCKS` (optional, default `2`): stale-plan guard.
- `EXECUTION_DEADLINE_SECS` (optional, default `20`): wall-clock submit deadline guard.
- `REBALANCE_MODE` (optional): `full` (default) or `arb_only`.

## Execution flow

1. Parse runtime config.
2. Build wallet provider from `PRIVATE_KEY` and `RPC`.
3. Resolve TradeExecutor using cache (`cache/trade_executor.json`):
   - live mode (`EXECUTE_SUBMIT=1`): reuse if valid, otherwise deploy from `out/TradeExecutor.sol/TradeExecutor.json` and cache
   - dry-run mode (`EXECUTE_SUBMIT=0`): read cache only; never deploy
4. Preprocess approvals from executor context:
   - live mode only (skipped in dry-run)
   - read on-chain allowances `(executor, spender)`
   - queue only insufficient approvals
   - execute approvals via `TradeExecutor.batchExecute` in chunks
5. Strict execution loop:
   - fetch slot0 + executor balances
   - run `rebalance_with_gas`
   - build plannable groups and stamp `planned_at_block`
   - select first subgroup only
   - derive bounds with staleness check
   - enforce deadline window
   - build deterministic `TradeExecutor::Call[]`
   - dry-run log calls or submit `batchExecute`
   - wait receipt, then replan

## Calldata builder mapping

- `DirectBuy` -> one `BatchSwapRouter.exactOutput(address[],uint256[],...)`
- `DirectSell` -> one `BatchSwapRouter.exactInput(address[],uint256[],...)`
- `MintSell` -> `splitPosition(market1)` -> `splitPosition(market2)` -> batch `exactInput`
- `BuyMerge` -> batch `exactOutput` -> `mergePositions(market2)` -> `mergePositions(market1)`
- `DirectMerge` -> `mergePositions(market2)` -> `mergePositions(market1)`

Unequal per-leg amount arrays are preserved for `BuyMerge` and `MintSell`.

## Dry-run vs live

- Dry-run default (`EXECUTE_SUBMIT=0`): logs deterministic call targets/calldata and stops after the first subgroup.
- Dry-run requires a valid cached executor entry and will not deploy or send approval transactions.
- Live mode (`EXECUTE_SUBMIT=1`): submits one subgroup tx per loop iteration and replans after receipt.

## Expected fail-closed behavior

Execution aborts the current run on:

- stale or unstamped plan
- deadline exceeded pre-submit
- missing/mismatched bounds
- unsupported action shape
- call-build error
- reverted receipt
