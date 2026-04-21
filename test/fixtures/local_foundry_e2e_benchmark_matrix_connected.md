# Local Foundry E2E Benchmark Matrix

All columns are measured from real local execution:
- `gas` / `calldata_bytes` come from Foundry `gasleft()` and abi-encoded `batchExecute` bytes
- `net_ev = (post_raw_ev - pre_raw_ev) - modeled_fee`
- `modeled_fee = (gas * GAS_PRICE_WEI + calldata * L1_FEE_PER_BYTE_WEI) * ETH_USD` in WAD

| case | topology | solver | pre_raw_ev | post_raw_ev | gas | calldata | modeled_fee (WAD) | realized_net_ev (WAD) | actions | chunks | skip |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bench_connected_98 | connected | offchain_waterfall | 25001224489795918367760 | 43631417972876141478558 | 815352 | 3140 | 2467228194300000 | 18630191015852028810798 | 8 | 1 | - |
| bench_connected_98 | connected | offchain_forecastflows | 25001224489795918367760 | 25001357004194972082765 | 371316 | 2712 | 1129912333380000 | 131384486720335005 | 287 | 6 | - |
| bench_connected_98 | connected | onchain_rebalance_exact | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | onchain_single_market_only |
| bench_connected_98 | connected | onchain_rebalance_arb_direct | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | onchain_single_market_only |
| bench_connected_98 | connected | onchain_rebalance_mixed_constant_l | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | onchain_single_market_only |
