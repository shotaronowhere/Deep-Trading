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
- `EXECUTE_SUBMIT` (optional, default `0`): submission mode.
  - `0/false/no/off` (default): dry-run — log calldata and stop after first chunk.
  - `1/true/yes/on`: live — submit transactions automatically.
  - `confirm/interactive/prompt`: interactive — print trade summary and wait for Enter before each submission.
- `EXECUTION_MAX_STEPS` (optional, default `32`): strict loop cap.
- `EXECUTION_MAX_STALE_BLOCKS` (optional, default `2`): stale-plan guard.
- `EXECUTION_DEADLINE_SECS` (optional, default `20`): wall-clock submit deadline guard.
- `REBALANCE_MODE` (optional): `full` (default) or `arb_only`.
- `NET_EV_THRESHOLD_SUSD` (optional, default `0.0`): minimum chunk net EV (edge minus gas, in sUSD) required to submit a transaction. If the estimated net EV of the first execution chunk is below this threshold, submission is skipped for that cycle.

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
   - compute chunk net EV; skip if below `NET_EV_THRESHOLD_SUSD`
   - dry-run: log calls and print trade summary; confirm: print summary and wait for Enter
   - submit `batchExecute`; on send/receipt error: log warning, invalidate snapshot, continue
   - on revert: log warning, invalidate snapshot cache, continue
   - on success: log receipt, run post-tx verification (re-fetch balances, compute realized gain, log per-token deltas)
   - invalidate snapshot cache after any tx attempt, then replan

## Calldata builder mapping

- `DirectBuy` -> one `BatchSwapRouter.exactOutput(address[],uint256[],...)`
- `DirectSell` -> one `BatchSwapRouter.exactInput(address[],uint256[],...)`
- `MintSell` -> `splitPosition(market1)` -> `splitPosition(market2)` -> batch `exactInput`
- `BuyMerge` -> batch `exactOutput` -> `mergePositions(market2)` -> `mergePositions(market1)`
- `DirectMerge` -> `mergePositions(market2)` -> `mergePositions(market1)`

Unequal per-leg amount arrays are preserved for `BuyMerge` and `MintSell`.

## Submission modes

- **Dry-run** (`EXECUTE_SUBMIT=0`, default): logs deterministic call targets/calldata, prints trade summary, and stops after the first subgroup. Requires a valid cached executor entry and will not deploy or send approval transactions.
- **Live** (`EXECUTE_SUBMIT=1`): submits one subgroup tx per loop iteration and replans after receipt. Deploys executor if needed and preprocesses approvals.
- **Confirm** (`EXECUTE_SUBMIT=confirm`): same as live, but prints a trade summary (tokens bought/sold, quantities, net EV, estimated gas) and waits for the user to press Enter before each submission. Ctrl-C aborts.

## On-chain solver preview

Before submitting (in both dry-run and confirm modes), the executor runs all three on-chain solver variants via `eth_call` and presents a comparison table:

| Solver | Method |
|--------|--------|
| Exact | `Rebalancer.rebalanceExact(params, maxBisection, maxTickCrossings)` |
| Arb | `Rebalancer.rebalanceAndArb(params, market, maxArbRounds, maxRecycleRounds)` |
| Mixed | `RebalancerMixed.rebalanceMixedConstantL(params, market, maxOuter, maxInner, maxMintCollateral)` |

Each solver call is wrapped in a `SolverQuoter.quote()` that reverts with `abi.encode(success, returnData, postCash, postBalances)`. The quoter bytecode is injected via `eth_call` state overrides (no deployment needed). The Rebalancer and RebalancerMixed contracts require constructor args and are deployed and cached alongside TradeExecutor.

Raw EV is computed as `postCash + Σ(prediction × postBalance)` using post-rebalance token balances from the quoter revert data. The comparison table shows each solver's raw EV, delta from pre-rebalance EV, and success/failure status.

The per-leg summary also shows `price=X.XXXX pred=X.XXXX` for each token to help assess edge quality.

Implementation: `src/execution/onchain_preview.rs`.

## Plan preview diagnostic

`cargo run --bin plan_preview` prints the first live executable subgroup without submitting transactions.

It shares the same preview formatter as runtime output (`src/execution/preview.rs`) so line format and leg summaries match `deep_trading_bot`.

Inputs used by the preview path:

1. `.env` load (if present)
2. live `slot0` snapshots from `RPC`
3. wallet balances when `WALLET` is set, otherwise synthetic `STARTING_SUSD`
4. current rebalance mode/gas assumptions
5. conservative execution repricing assumptions

## Expected fail-closed behavior

Execution aborts the current run on:

- stale or unstamped plan
- deadline exceeded pre-submit
- missing/mismatched bounds
- unsupported action shape
- call-build error

Execution continues the loop (does not abort) on:

- reverted receipt (logs warning, invalidates snapshot)
- RPC send error (logs warning, invalidates snapshot)
- receipt retrieval failure (logs warning, invalidates snapshot)
