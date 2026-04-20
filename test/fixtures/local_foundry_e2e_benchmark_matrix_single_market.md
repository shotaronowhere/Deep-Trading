# Local Foundry E2E Benchmark Matrix

All columns are measured from real local execution:
- `gas` / `calldata_bytes` come from Foundry `gasleft()` and abi-encoded `batchExecute` bytes
- `net_ev = (post_raw_ev - pre_raw_ev) - modeled_fee`
- `modeled_fee = (gas * GAS_PRICE_WEI + calldata * L1_FEE_PER_BYTE_WEI) * ETH_USD` in WAD

| case | topology | solver | pre_raw_ev | post_raw_ev | gas | calldata | modeled_fee (WAD) | realized_net_ev (WAD) | actions | chunks | skip |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bench_single_market_98 | single_market | offchain_waterfall | 20001122448979591836930 | 32733572707518792215540 | 1429082 | 5444 | 4324061286810000 | 12732445934477913568610 | 14 | 1 | - |
| bench_single_market_98 | single_market | offchain_forecastflows | 20001122448979591836930 | 32734425448416993505857 | 2282307 | 8900 | 6906731019825000 | 12733296092706381843927 | 23 | 1 | - |
| bench_single_market_98 | single_market | onchain_rebalance_exact | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | onchain_revert_full_range_tick_scan |
| bench_single_market_98 | single_market | onchain_rebalance_arb_direct | 20001122448979591836930 | 32734425448423159202039 | 4700359 | 16356 | 14214522681165000 | 12733288784920886200109 | 0 | 1 | - |
| bench_single_market_98 | single_market | onchain_rebalance_mixed_constant_l | 20001122448979591836930 | 32734425448423159202039 | 6053780 | 16484 | 18284857032960000 | 12733284714586534405109 | 0 | 1 | - |
