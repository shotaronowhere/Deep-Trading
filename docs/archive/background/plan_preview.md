# Plan Preview Diagnostic

`cargo run --bin plan_preview` prints the first live executable group without deploying or reusing a `TradeExecutor`.

Its leg output now matches `deep_trading_bot`: all legs when there are 4 or fewer, otherwise the first 2, an omitted-count line, and the last 2.

That preview text is now built in shared library code (`execution::preview`) before either binary prints it, so the exact line format is covered by unit tests without needing to run the binaries.

`cargo run --bin deep_trading_bot` now prints the same first execution-group preview as part of its normal runtime summary. Its leg output is compact: all legs when there are 4 or fewer, otherwise the first 2, an omitted-count line, and the last 2. The dedicated preview binary is mainly useful when you want the planning output without the broader rebalance report.

It uses the same off-chain inputs as the runtime planner:

1. Loads `.env` if present.
2. Fetches live `slot0` snapshots from `RPC`.
3. Uses `WALLET` balances when set, otherwise uses synthetic cash from `STARTING_SUSD`.
4. Builds actions with the current rebalance mode and gas assumptions.
5. Applies conservative execution repricing (`EXECUTION_QUOTE_LATENCY_BLOCKS`, `EXECUTION_ADVERSE_MOVE_BPS_PER_BLOCK`).
6. Prints the first stamped `ExecutionGroupPlan`, including per-leg `sqrtPriceLimitX96` and derived batch bounds.

This is intended for local verification of the execution planner when `EXECUTE_SUBMIT=0` is blocked because `cache/trade_executor.json` does not exist yet.
