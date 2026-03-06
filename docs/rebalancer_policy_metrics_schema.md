# Rebalancer Policy Metrics Schema

This schema defines the dashboard row produced by:

- `scripts/rebalancer_policy_metrics_from_logs.sh`

It is the canonical machine-readable payload for policy-A and policy-B threshold checks in [rebalancer_approaches_playbook.md](rebalancer_approaches_playbook.md).

For mechanism-level context and rationale behind many policy levers, see [rebalancing_mechanism_design_review.md](rebalancing_mechanism_design_review.md).

## Files

- CSV template: [rebalancer_policy_metrics_template.csv](rebalancer_policy_metrics_template.csv)
- JSON template: [rebalancer_policy_metrics_template.json](rebalancer_policy_metrics_template.json)

## Column definitions

1. `date_utc`: run date (`YYYY-MM-DD`)
2. `block_ref`: optional block number or block tag
3. `policy_a_case`: case/test identifier selected for policy A
4. `policy_a_constant_ev_wei`: constant-mode EV in wei
5. `policy_a_exact_ev_wei`: exact-mode EV in wei
6. `policy_a_ev_gain_susd`: `(exact_ev - constant_ev) / 1e18`
7. `policy_a_constant_gas`: constant-mode gas units (if present)
8. `policy_a_exact_gas`: exact-mode gas units (if present)
9. `policy_a_extra_gas_susd`: `(exact_gas - constant_gas) * gas_price * ETHUSD`
10. `policy_a_hurdle_susd`: `max(0.10, 3 * policy_a_extra_gas_susd)`
11. `policy_a_hurdle_met`: `policy_a_ev_gain_susd >= policy_a_hurdle_susd`
12. `policy_a_trigger_now`: same as `policy_a_hurdle_met` for point-in-time trigger
13. `policy_b_case_count`: number of mixed-vs-direct cases included in aggregates
14. `policy_b_median_ev_gain_susd`: median of per-case `(mixed_ev - direct_ev)/1e18`
15. `policy_b_p10_ev_gain_susd`: p10 of per-case EV gains
16. `policy_b_median_extra_gas_susd`: median of per-case gas deltas in sUSD
17. `policy_b_hurdle_susd`: `max(0.20, 5 * policy_b_median_extra_gas_susd)`
18. `policy_b_hurdle_met`: `policy_b_median_ev_gain_susd >= policy_b_hurdle_susd`
19. `policy_b_negative_tail_breach`: `policy_b_p10_ev_gain_susd < -0.05`
20. `fallback_total`: total parsed `MixedSolveFallback(reasonCode)` events
21. `fallback_rate_pct`: `fallback_total / attempt_count * 100`
22. `fallback_dominant_reason`: reason code with highest count
23. `fallback_dominant_share_pct`: dominant reason share in percent
24. `fallback_reliability_met`: `fallback_rate_pct <= 20`
25. `fallback_dominance_met`: `fallback_dominant_share_pct <= 60`
26. `policy_b_enable_now`: `policy_b_hurdle_met && !policy_b_negative_tail_breach && fallback_reliability_met && fallback_dominance_met`
27. `gas_price_gwei`: gas assumption used for conversion
28. `eth_usd`: ETHUSD assumption used for conversion

## Expected log sources

Policy A (constant vs exact):

- preferred: `forge test --match-test testBenchmarkABMultiTickSyntheticNinetyEightOutcomeConstantLVsExact -vv`
- also supported (EV-only): `forge test --match-test test_rebalancer_ab_live_l1_snapshot_report -vv`

Policy B (mixed vs direct):

- `forge test --match-test test_rebalancer_vs_mixed_apples_to_apples_report -vv`

Fallback telemetry for policy-B gating:

- any log containing `MixedSolveFallback(<reasonCode>)`

## Example command

```bash
scripts/rebalancer_policy_metrics_from_logs.sh \
  --policy-a-log /tmp/ab_constant_exact.log \
  --policy-b-log /tmp/ab_mixed.log \
  --fallback-log /tmp/ab_mixed_trace.log \
  --policy-a-case testBenchmarkABMultiTickSyntheticNinetyEightOutcomeConstantLVsExact \
  --date 2026-03-03 \
  --block 136002137 \
  --gas-price-gwei 1.0 \
  --eth-usd 3000.0 \
  --out-json /tmp/rebalancer_policy_metrics_latest.json \
  --append-csv docs/rebalancer_policy_metrics.csv
```

## Notes

- If policy-A gas fields are missing in the supplied log, EV fields are still computed but gas-gated policy-A fields remain blank.
- `--fallback-log` is required to compute policy-B fallback gates. Without it, fallback-rate fields remain blank and `policy_b_enable_now` stays `false`.
- By default, policy-B aggregation excludes case names containing `direct_only`, and fallback counting is filtered to that same included case set. Use `--include-direct-only-policy-b-cases` to override.
