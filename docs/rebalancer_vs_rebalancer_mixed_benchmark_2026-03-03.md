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
| `two_pool_single_tick_direct_only` | 100.102675011503391898 | 100.102675011503391898 | 0 | 0.000000% | 246,698 | 264,840 | 18,142 | 7.3539% |
| `ninety_eight_outcome_multitick_direct_only` | 98.123254863636424997 | 98.123254863636424997 | 0 | 0.000000% | 7,346,469 | 8,201,177 | 854,708 | 11.6342% |
| `small_bundle_mixed_case` | 100.132867689650403452 | 100.132867689650403452 | 0 | 0.000000% | 271,605 | 296,739 | 25,134 | 9.2539% |
| `mixed_route_favorable_synthetic_case` | 100.021703842635411426 | 100.021703842635411426 | 0 | 0.000000% | 212,458 | 234,502 | 22,044 | 10.3758% |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | 150.258288614947875485 | 150.258288614947875485 | 0 | 0.000000% | 5,057,031 | 6,241,827 | 1,184,796 | 23.4287% |
| `legacy_holdings_direct_only_case` | 38.863181798833603563 | 38.863181798833603563 | 0 | 0.000000% | 297,900 | 312,705 | 14,805 | 4.9698% |

## Interpretation

- In this fixture set, terminal EV is identical across all tested cases.
- Traces show `MixedSolveFallback(reasonCode: 2)` on every case (`FB_NO_NON_ACTIVE`), so mixed path degrades to direct-only behavior.
- `RebalancerMixed` currently pays additional overhead (roughly 5% to 23%) for mixed-solve/fallback checks.
