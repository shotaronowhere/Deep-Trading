# Local Foundry E2E Benchmark Matrix

All columns are measured from real local execution:
- `gas` / `calldata_bytes` come from Foundry `gasleft()` and abi-encoded `batchExecute` bytes
- `net_ev = (post_raw_ev - pre_raw_ev) - modeled_fee`
- `modeled_fee = (gas * GAS_PRICE_WEI + calldata * L1_FEE_PER_BYTE_WEI) * ETH_USD` in WAD

| case | topology | solver | pre_raw_ev | post_raw_ev | gas | calldata | modeled_fee (WAD) | realized_net_ev (WAD) | actions | chunks | skip |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bench_single_market_98 | single_market | offchain_waterfall | 20001122448979591836930 | 32733572707518830050771 | 1442325 | 5444 | 4363882656735000 | 12732445894656581478841 | 14 | 1 | - |
| bench_single_market_98 | single_market | offchain_forecastflows | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | forecastflows_worker_bin_unset |
| bench_single_market_98 | single_market | onchain_rebalance_exact | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | onchain_revert_full_range_tick_scan |
| bench_single_market_98 | single_market | onchain_rebalance_arb_direct | 20001122448979591836930 | 32734425448423159202039 | 4286213 | 16356 | 12969196012815000 | 12733290030247554550109 | 0 | 1 | - |
| bench_single_market_98 | single_market | onchain_rebalance_mixed_constant_l | 20001122448979591836930 | 32734425448423159202039 | 5636398 | 16484 | 17029799793510000 | 12733285969643773855109 | 0 | 1 | - |
