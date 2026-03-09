# Solver Benchmark Matrix

Last updated: 2026-03-09

## Purpose

This is the central release-facing solver comparison document.

It answers, in one place:

- which solver flavors exist
- which off-chain path ships by default
- what the realistic 98-outcome L1-like case looks like under the current net-EV objective
- how on-chain and off-chain compare under the same Optimism fee snapshot
- what the current speed tradeoff is

## Sources

- committed fixture EVs: `test/fixtures/rebalancer_ab_expected.json`
- on-chain calldata artifact:
  - `forge test --match-test test_write_rebalancer_ab_onchain_call_report -vv`
  - output: `test/fixtures/rebalancer_ab_onchain_call_report.json`
- shared-snapshot on-chain pricing under the benchmark OP snapshot:
  - `cargo test print_shared_op_snapshot_onchain_benchmark_rows_jsonl -- --ignored --nocapture`
- shared-snapshot off-chain selected-plan pricing under the benchmark OP snapshot:
  - `cargo test print_shared_op_snapshot_offchain_selected_rows_jsonl -- --ignored --nocapture`
  - note: this uses the current solver-selected modeled fee estimate from the planner summary
  - current selected benchmark cases are still using the structural packed fallback estimator
- release speed checks:
  - `cargo test --release portfolio::core::tests::execution::test_rebalance_perf_full_l1 -- --exact --nocapture`
  - `cargo test --release portfolio::core::tests::execution::test_rebalance_perf_full_l1_with_gas_pricing -- --exact --nocapture`
  - `REBALANCE_ENABLE_STAGED_FALLBACK=1 cargo test --release portfolio::core::tests::execution::test_rebalance_perf_full_l1 -- --exact --nocapture`
  - `REBALANCE_ENABLE_STAGED_FALLBACK=1 cargo test --release portfolio::core::tests::execution::test_rebalance_perf_full_l1_with_gas_pricing -- --exact --nocapture`

## Conventions

- `raw EV = cash + sum(prediction * holdings)`
- `net EV = raw EV - execution_cost`
- execution cost includes only:
  - Optimism L2 execution gas
  - Optimism L1 data fee
- swap / pool fees are already embedded in the raw EV math and are not subtracted again

Shared benchmark OP pricing snapshot used for the comparison rows below:

- `chain_id = 10`
- `gas_price_wei = 1,002,325`
- `eth_usd = 3000`
- `l1_fee_per_byte_wei = 1,643,855.3414634147`
- `l1_data_fee_floor_susd = 0.0`
- benchmark-mode modeled rows use this fixed 2026-03-08 OP snapshot rather than a live RPC lookup

Fee-source notes:

- off-chain shared-snapshot rows use:
  - the current solver-selected planner-side fee estimate
  - packed-program tx counting when replay hydration succeeds
  - otherwise the structural packed fallback estimator over grouped actions
- on-chain shared-snapshot rows use:
  - Foundry-measured transaction gas units
  - exact unsigned tx bytes for the benchmark call
  - the fixed benchmark `l1_fee_per_byte_wei` snapshot applied to that tx length

Availability notes:

- `n/a` means the metric is not available or not measured in a trustworthy way for that row
- literal `0` means the selected plan genuinely no-ops or the measured value is exactly zero under the shared-snapshot model

## Solver Flavors

| Flavor | What it is | Status | Canonical doc |
|---|---|---|---|
| Off-chain default full solver | Operator-based `R_exact` plus `Plain` / `ArbPrimed`, with rich-trace compaction chosen from `baseline_step_prune`, `target_delta`, `analytic_mixed`, `constant_l_mixed`, `coupled_mixed`, `staged_constant_l_2`, `direct_only`, or `noop`, then compiled into the better of packed-vs-strict execution programs and ranked by estimated net EV | Default v1 | `docs/waterfall.md` |
| Off-chain direct-only baseline | Off-chain direct-route-only path under the same net-EV objective | Benchmark baseline | `docs/portfolio.md` |
| Off-chain full-rebalance-only | Raw benchmark row without the full mixed fallback behavior | Benchmark row only | `docs/portfolio.md` |
| On-chain exact direct | `Rebalancer.rebalanceExact(...)` atomic direct solve | Production reference | `docs/rebalancer.md` |
| On-chain mixed constant-`L` | `RebalancerMixed.rebalanceMixedConstantL(...)` atomic mixed-route solve | Experimental on-chain reference | `docs/rebalancer_mixed.md` |
| Off-chain staged reference enabled | Same packed solver with `REBALANCE_ENABLE_STAGED_FALLBACK=1` reference comparison | Rollout / validation only | `docs/portfolio.md` |

## Central 98-Outcome L1-Like Case

Case id: `heterogeneous_ninety_eight_outcome_l1_like_case`

| Flavor | Raw EV | Total fee | Net EV | Actions | Groups / tx | Fee source | Note |
|---|---:|---:|---:|---:|---:|---|---|
| Off-chain default full solver | `150.380245589644673024` | `$0.015059843882464171` | `150.3651857457622` | `94` | `1` | replay packed program estimate | `constant_l_mixed` compiler selected; `Plain` family; one `Mint`, 32 direct buys, 61 direct sells; active set `32` |
| Off-chain direct-only baseline | `150.258105490229428224` | `$0.006413207231552927` | `150.25169228299788` | `39` | `1` | replay packed program estimate | `direct_only` compiler selected; compact exact-direct-like frontier program |
| On-chain exact direct | `150.258288614947875485` | `$0.013109266442788025` | `150.24517934850508` | `n/a` | `1` | modeled L1 + Foundry gas | Off-chain direct baseline now slightly beats this row on modeled net EV |
| On-chain mixed constant-`L` | `150.380322266504052605` | `$0.0678848029642546` | `150.3124374635398` | `n/a` | `1` | modeled L1 + Foundry gas | Off-chain default now beats this row on modeled net EV |
| Off-chain full-rebalance-only | `150.378870058707484672` | `n/a` | `n/a` | `n/a` | `n/a` | raw fixture only | Raw benchmark reference only; not a shipped runtime flavor |

Interpretation:

- the packed execution-program compiler fixed the old fragmentation failure mode; the native `constant_l_mixed` compiler also closes the former compact mixed-route miss on the small synthetic cases
- the shipped off-chain default still beats both on-chain exact and on-chain mixed on modeled net EV for the realistic 98-outcome case
- the remaining uncertainty is now teacher-side: how much large-`n` or multi-stage headroom is left above the current `K=1` compact solve

## Selected Compiler Summary

| Case | Off-chain default compiler | Off-chain default family | Selected active set | Key shape |
|---|---|---|---|---|
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `constant_l_mixed` | `plain` | `32` | `1` mint + `61` sells + `32` buys in `1` packed tx |
| `mixed_route_favorable_synthetic_case` | `constant_l_mixed` | `plain` | `1` | `1` mint + `3` sells + `1` buy; matches on-chain mixed raw EV in `1` packed tx |
| `small_bundle_mixed_case` | `constant_l_mixed` | `plain` | `2` | `1` mint + `2` sells + `2` buys; converges to the on-chain mixed constant-`L` state in `1` packed tx |

## Committed Fixture Raw-EV Matrix

This is still the canonical raw benchmark table from the committed fixture.

| Case | Off-chain direct | Off-chain full-rebalance-only | Off-chain default mixed | On-chain exact direct | On-chain mixed constant-`L` | Off-chain default action count |
|---|---:|---:|---:|---:|---:|---:|
| `two_pool_single_tick_direct_only` | `100.102675011503407104` | `100.102675011503407104` | `100.102675011503407104` | `100.102675011503391898` | `100.102675011503391898` | `2` |
| `ninety_eight_outcome_multitick_direct_only` | `98.123254863635890176` | `98.123254863635890176` | `98.123254863635890176` | `98.123254863636424997` | `98.123254863636424997` | `98` |
| `small_bundle_mixed_case` | `100.132867689650405376` | `100.132867689650405376` | `100.148477979531444224` | `100.132867689650403452` | `100.148477979531421487` | `5` |
| `legacy_holdings_direct_only_case` | `38.863181796980015104` | `38.863181796980015104` | `38.863181796980015104` | `38.863181798833603563` | `38.863181796980015678` | `2` |
| `mixed_route_favorable_synthetic_case` | `100.021703842635415552` | `100.108205685259485184` | `100.108519024112205824` | `100.021703842635411426` | `100.108519024112198000` | `5` |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `150.258105490229428224` | `150.378870058707484672` | `150.380245589644673024` | `150.258288614947875485` | `150.380322266504052605` | `94` |

Raw-EV interpretation:

- direct-only rows remain at parity to dust, which is expected
- `small_bundle_mixed_case` now collapses to the compact constant-`L` state-space solution instead of preserving the old 13-action trace
- `mixed_route_favorable_synthetic_case` no longer regresses on raw EV; the compact solver now lands at on-chain mixed parity with `5` actions
- the heterogeneous 98-outcome default row now accepts a slightly lower-raw but lower-fee `constant_l_mixed` program, improving net EV while cutting the packed action count from `98` to `94`

## Shared-Snapshot Net-EV Matrix

These rows price both off-chain and on-chain under the same benchmark OP snapshot:

- same `gas_price_wei`
- same `eth_usd`
- same cached `l1_fee_per_byte_wei`
- off-chain uses packed-vs-strict execution-program pricing over the same replay plan
- on-chain uses the exact benchmark call payload from the Foundry artifact

| Case | Flavor | Raw EV | Total fee | Net EV | Actions | Groups / tx | Fee source |
|---|---|---:|---:|---:|---:|---:|---|
| `two_pool_single_tick_direct_only` | `offchain_direct` | `100.102675011503407104` | `$0.0003503796943033902` | `100.10232463180911` | `2` | `1` | replay packed program estimate |
| `two_pool_single_tick_direct_only` | `offchain_default` | `100.102675011503407104` | `$0.0003503796943033902` | `100.10232463180911` | `2` | `1` | replay packed program estimate |
| `two_pool_single_tick_direct_only` | `onchain_exact` | `100.102675011503391898` | `$0.0007473665477533902` | `100.10192764495564` | `n/a` | `1` | modeled L1 + Foundry gas |
| `two_pool_single_tick_direct_only` | `onchain_mixed` | `100.102675011503391898` | `$0.0008202825723789512` | `100.10185472893102` | `n/a` | `1` | modeled L1 + Foundry gas |
| `ninety_eight_outcome_multitick_direct_only` | `offchain_direct` | `98.123254863635890176` | `$0.01714280306742651` | `98.10611206056846` | `98` | `1` | replay packed program estimate |
| `ninety_eight_outcome_multitick_direct_only` | `offchain_default` | `98.123254863635890176` | `$0.01714280306742651` | `98.10611206056846` | `98` | `1` | replay packed program estimate |
| `ninety_eight_outcome_multitick_direct_only` | `onchain_exact` | `98.123254863636424997` | `$0.021731487606613024` | `98.10152337602982` | `n/a` | `1` | modeled L1 + Foundry gas |
| `ninety_eight_outcome_multitick_direct_only` | `onchain_mixed` | `98.123254863636424997` | `$0.025573340082288586` | `98.09768152355414` | `n/a` | `1` | modeled L1 + Foundry gas |
| `small_bundle_mixed_case` | `offchain_direct` | `100.132867689650405376` | `$0.0003503796943033902` | `100.13251730995611` | `2` | `1` | replay packed program estimate |
| `small_bundle_mixed_case` | `offchain_default` | `100.148477979531444224` | `$0.0007147656887896098` | `100.14776321384267` | `5` | `1` | replay packed program estimate |
| `small_bundle_mixed_case` | `onchain_exact` | `100.132867689650403452` | `$0.0008153807545311951` | `100.13205230889588` | `n/a` | `1` | modeled L1 + Foundry gas |
| `small_bundle_mixed_case` | `onchain_mixed` | `100.148477979531421487` | `$0.002919048324481756` | `100.14555893120693` | `n/a` | `1` | modeled L1 + Foundry gas |
| `legacy_holdings_direct_only_case` | `offchain_direct` | `38.863181796980015104` | `$0.0002919150793783902` | `38.862889881900635` | `2` | `1` | replay packed program estimate |
| `legacy_holdings_direct_only_case` | `offchain_default` | `38.863181796980015104` | `$0.0002919150793783902` | `38.862889881900635` | `2` | `1` | replay packed program estimate |
| `legacy_holdings_direct_only_case` | `onchain_exact` | `38.863181798833603563` | `$0.0008969364912283903` | `38.86228486234237` | `n/a` | `1` | modeled L1 + Foundry gas |
| `legacy_holdings_direct_only_case` | `onchain_mixed` | `38.863181796980015678` | `$0.0018997986252789513` | `38.861281998354734` | `n/a` | `1` | modeled L1 + Foundry gas |
| `mixed_route_favorable_synthetic_case` | `offchain_direct` | `100.021703842635415552` | `$0.000175453685934` | `100.02152838894948` | `1` | `1` | replay packed program estimate |
| `mixed_route_favorable_synthetic_case` | `offchain_default` | `100.108519024112205824` | `$0.0006940386101146098` | `100.10782498550209` | `5` | `1` | replay packed program estimate |
| `mixed_route_favorable_synthetic_case` | `onchain_exact` | `100.021703842635411426` | `$0.0006322108724061952` | `100.02107163176301` | `n/a` | `1` | modeled L1 + Foundry gas |
| `mixed_route_favorable_synthetic_case` | `onchain_mixed` | `100.108519024112198000` | `$0.003384732528781756` | `100.10513429158343` | `n/a` | `1` | modeled L1 + Foundry gas |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `offchain_direct` | `150.258105490229428224` | `$0.006413207231552927` | `150.25169228299788` | `39` | `1` | replay packed program estimate |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `offchain_default` | `150.380245589644673024` | `$0.015059843882464171` | `150.3651857457622` | `94` | `1` | replay packed program estimate |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `onchain_exact` | `150.258288614947875485` | `$0.013109266442788025` | `150.24517934850508` | `n/a` | `1` | modeled L1 + Foundry gas |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `onchain_mixed` | `150.380322266504052605` | `$0.0678848029642546` | `150.3124374635398` | `n/a` | `1` | modeled L1 + Foundry gas |

## Release Speed Matrix

Release perf was rerun after the teacher/memoization pass. These numbers are release-only measurements of the full 98-outcome path and are not used in benchmark selection.

| Runtime mode | `test_rebalance_perf_full_l1` | `test_rebalance_perf_full_l1_with_gas_pricing` | Meaning |
|---|---:|---:|---|
| Default packed path | `11.167825412s` | `4.829752766s` | Current shipped hot path after packing |
| Packed path + staged reference (`REBALANCE_ENABLE_STAGED_FALLBACK=1`) | `12.359121054s` | `4.840933383s` | Rollout / validation path only |

Interpretation:

- the gas-aware packed path is materially faster than the ungated full-L1 path
- enabling the staged reference still adds measurable overhead on the raw full-L1 perf path, but almost none on the gas-priced path
- the structural economics table above and the speed table here now reflect the same post-packing solver generation

## Gas Calibration Snapshot

Canonical small-shape exact-fee calibration on 2026-03-08:

| Shape | Exact total fee |
|---|---:|
| `DirectBuy` | `$0.000176154997284` |
| `DirectSell` | `$0.000117260164440` |
| `DirectMerge` | `$0.000067578070464` |
| `MintSell(1)` | `$0.000210493567404` |
| `BuyMerge(1)` | `$0.000206453150568` |

Current fallback calibration decision:

- `l1_data_fee_floor_susd = 0.001`

## Release Takeaways

- the off-chain solver is now optimizing net EV, not paper EV
- pool fees are not double-counted; only OP L2 execution gas and OP L1 data fees are subtracted
- execution-program packing materially improves off-chain economics by collapsing many strict subgroups into a single tx chunk when feasible
- on the 98-outcome direct-only case, packed off-chain now beats on-chain exact on net EV
- the realistic heterogeneous 98-outcome case now clears both on-chain exact and on-chain mixed on modeled net EV with `constant_l_mixed`
- the mixed-route favorable synthetic case no longer trails on-chain mixed; it now lands at on-chain mixed raw parity with a 5-action compact plan
- teacher diagnostics now exist for `K=1`:
  - exact small oracle
  - exact medium oracle
  - large-case best-known comparison
- runtime `K=2` remains heuristic; the exact `K=2` teacher is currently a validation tool, not a shipped optimization layer
- raw-EV benchmark history still matters for understanding search behavior, but `net EV` is now the primary economic metric for the shipped off-chain solver
