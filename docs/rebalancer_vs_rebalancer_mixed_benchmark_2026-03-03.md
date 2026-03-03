# Rebalancer vs RebalancerMixed Apples-to-Apples Benchmark (2026-03-03)

## Scope

On-chain comparison between:

- `Rebalancer.rebalance(...)`
- `RebalancerMixed.rebalanceMixedConstantL(..., 24, 24, 0)`

using identical fixtures, initial portfolios, AMM simulation, fee tier, and EV marking.

This run uses the uncapped mixed-active setting (`MAX_MIXED_ACTIVE = type(uint256).max`).

## Harness

- Test: `test/RebalancerAB.t.sol::test_rebalancer_vs_mixed_apples_to_apples_report`
- Fixtures: `test/fixtures/rebalancer_ab_cases.json`
- EV mark:
  - `EV = collateral + sum_i(prediction_i * token_i_balance)`
  - all values in 18-decimal token units (same base as `wad`/`wei`)

## Command

```bash
forge test --match-test test_rebalancer_vs_mixed_apples_to_apples_report -vv
```

## Results

| Case | Rebalancer EV | RebalancerMixed EV | EV Diff (Mixed - Rebalancer) | EV Diff % | Rebalancer Gas | RebalancerMixed Gas | Gas Diff | Gas Diff % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `two_pool_single_tick_direct_only` | 100.102675011503391898 | 100.102675011503391898 | 0 | 0.000000% | 246,698 | 271,250 | 24,552 | 9.9514% |
| `ninety_eight_outcome_multitick_direct_only` | 98.123254863636424997 | 98.123254863636424997 | 0 | 0.000000% | 7,346,469 | 8,459,928 | 1,113,459 | 15.1556% |
| `small_bundle_mixed_case` | 100.132867689650403452 | 100.148477979531421487 | 0.015610289881018035 | 0.015590% | 271,605 | 968,691 | 697,086 | 256.6546% |
| `mixed_route_favorable_synthetic_case` | 100.021703842635411426 | 100.108519024112198000 | 0.086815181476786574 | 0.086796% | 212,458 | 1,123,559 | 911,101 | 428.8314% |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | 150.258288614947875485 | 150.380322266504052605 | 0.122033651556177120 | 0.081212% | 5,057,031 | 22,441,994 | 17,384,963 | 343.7720% |
| `legacy_holdings_direct_only_case` | 38.863181798833603563 | 38.863181796980015678 | -0.000000001853587885 | -0.000000% | 297,900 | 630,254 | 332,354 | 111.5646% |

## Interpretation

- Mixed route now improves EV on the mixed-favorable fixtures, with exact parity on pure direct-only fixtures.
- Traces still show `MixedSolveFallback(reasonCode: 2)` on direct-only style cases where there is no non-active universe to sell into.
- Gas is materially higher for mixed attempts/execution in this harness (roughly +10% to +429% depending on case).
