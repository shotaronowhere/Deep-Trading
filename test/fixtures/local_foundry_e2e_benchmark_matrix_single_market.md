# Local Foundry E2E Benchmark Matrix

All columns are measured from real local execution:
- `gas` / `calldata_bytes` come from Foundry `gasleft()` and abi-encoded `batchExecute` bytes
- `net_ev = (post_raw_ev - pre_raw_ev) - modeled_fee`
- `modeled_fee = (gas * GAS_PRICE_WEI + calldata * L1_FEE_PER_BYTE_WEI) * ETH_USD` in WAD

| case | topology | solver | pre_raw_ev | post_raw_ev | gas | calldata | modeled_fee (WAD) | realized_net_ev (WAD) | actions | chunks | skip |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bench_single_market_98 | single_market | offchain_waterfall | 20001122448979591836930 | 32733572707518792215540 | 1429068 | 5444 | 4324019189160000 | 12732445934520011218610 | 14 | 1 | - |
| bench_single_market_98 | single_market | offchain_forecastflows | 20001122448979591836930 | 53578717199243545880955 | 90865118 | 205876 | 274244429073990000 | 33577320505834880054025 | 533 | 29 | - |
| bench_single_market_98 | single_market | onchain_rebalance_exact | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | onchain_revert_full_range_tick_scan |
| bench_single_market_98 | single_market | onchain_rebalance_arb_direct | 20001122448979591836930 | 32734425448423159202039 | 3835193 | 16356 | 11612990148315000 | 12733291386453419050109 | 0 | 1 | - |
| bench_single_market_98 | single_market | onchain_rebalance_mixed_constant_l | 20001122448979591836930 | 32734425448423159202039 | 5177801 | 16484 | 15650810079435000 | 12733287348633487930109 | 0 | 1 | - |
