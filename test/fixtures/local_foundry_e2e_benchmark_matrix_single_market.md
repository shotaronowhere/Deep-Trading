# Local Foundry E2E Benchmark Matrix

All columns are measured from real local execution:
- `gas` / `calldata_bytes` come from Foundry `gasleft()` and abi-encoded `batchExecute` bytes
- `net_ev = (post_raw_ev - pre_raw_ev) - modeled_fee`
- `modeled_fee = (gas * GAS_PRICE_WEI + calldata * L1_FEE_PER_BYTE_WEI) * ETH_USD` in WAD

| case | topology | solver | pre_raw_ev | post_raw_ev | gas | calldata | modeled_fee (WAD) | realized_net_ev (WAD) | actions | chunks | skip |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bench_single_market_98 | single_market | offchain_waterfall | 20001122448979591836930 | 32733572707518792215540 | 1428837 | 5444 | 4323324577935000 | 12732445935214622443610 | 14 | 1 | - |
| bench_single_market_98 | single_market | offchain_forecastflows | 20001122448979591836930 | 32733572684978285204532 | 1566376 | 5444 | 4736900912460000 | 12732445499097780907602 | 14 | 1 | - |
| bench_single_market_98 | single_market | onchain_rebalance_exact | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | onchain_revert_full_range_tick_scan |
| bench_single_market_98 | single_market | onchain_rebalance_arb_direct | 20001122448979591836930 | 32734425448423159202039 | 4697741 | 16356 | 14206650420615000 | 12733288792793146750109 | 0 | 1 | - |
| bench_single_market_98 | single_market | onchain_rebalance_mixed_constant_l | 20001122448979591836930 | 32734425448423159202039 | 6051141 | 16484 | 18276921625935000 | 12733284722521941430109 | 0 | 1 | - |
