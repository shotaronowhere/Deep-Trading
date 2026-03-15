# Rebalancer A/B Benchmark

Status note:

- This is a historical native-vs-on-chain benchmark document.
- The current release-facing cross-solver benchmark matrix, including ForecastFlows rows and the
  local ForecastFlows latency lane, lives in `docs/solver_benchmark_matrix.md`.

This benchmark is the reference frame for comparing the off-chain Rust planner against the on-chain `Rebalancer.sol` contract on the same frozen pool snapshot.

## Goal

Measure two distinct effects separately:

1. **Direct parity**
   - Compare off-chain direct-only rebalance against on-chain constant-`L` `rebalance(...)`.
   - These should be economically very close on the same frozen snapshot.

2. **Full-solver dominance**
   - Compare the richer off-chain rebalance logic (without the standalone Phase 0 complete-set arb pre-pass) against on-chain constant-`L` `rebalance(...)`.
   - Because the off-chain route set is a superset, this path should be EV-greater-than-or-equal up to tolerance.

## Benchmark Rules

For every case:

1. Freeze one identical pool snapshot.
2. Run the Rust planner on that snapshot in two modes:
   - direct-only
   - full rebalance-only (same rebalance flow, but without the standalone Phase 0 complete-set arb pre-pass)
3. Reset to the same snapshot.
4. Run `Rebalancer.rebalance(...)`.
5. Reset again.
6. Run `Rebalancer.rebalanceExact(...)`.
7. Mark all terminal portfolios with the same EV function.

Case geometry for both the Rust and Foundry harnesses is sourced from [rebalancer_ab_cases.json](/Users/shotaro/proj/deep_trading/test/fixtures/rebalancer_ab_cases.json). The 98-outcome synthetic case uses the parametric `uniform_*` fields in that fixture instead of code-local hardcoding.

Recommended EV mark:

`EV = cash + sum(prediction_i * holdings_i)`

Optional reporting:

- `EV_net = EV - gas_in_susd`
- conservative liquidation floor using executable sell bids

Current committed dominance fixtures:

- `small_bundle_mixed_case`: aligned apples-to-apples 4-outcome mixed-route sanity case
- `mixed_route_favorable_synthetic_case`: frozen from the 4-outcome randomized dominance sweep
- `heterogeneous_ninety_eight_outcome_l1_like_case`: frozen from the 98-outcome randomized dominance sweep

## Interpretation

Important caveats:

1. On-chain solving removes stale-plan mismatch, but it does **not** remove your own price impact.
2. The direct parity check is a modeling sanity test, not a proof of exact equality. Small differences are expected from float vs integer/router rounding.
3. The full-solver comparison is a dominance test. A lower off-chain EV on a frozen snapshot means the richer off-chain policy made an EV-reducing choice on that state.
4. A higher off-chain EV on a frozen snapshot does **not** mean the off-chain path is always better live; it only measures objective quality on that state.

This benchmark is meant to make those tradeoffs visible, not to claim that one path universally dominates.

## Current Harness Notes

1. The direct-only parity checks use a small relative tolerance (`5e-6`) because the Rust replay model and the Solidity fixture router round differently.
2. The current hardcoded fee assumption is `1 bp` across the off-chain planner, the benchmark fixture, and the execution submission path, so both sides are evaluated under the same fee model in this benchmark.
3. The direct-only parity checks are measured against on-chain constant-`L` `rebalance(...)`, which is the apples-to-apples objective match. The Foundry harness fails if on-chain constant materially exceeds the off-chain direct EV or if the gap grows beyond tolerance.
4. The full-solver comparison also uses on-chain constant-`L` `rebalance(...)`, but only against the off-chain rebalance flow without the standalone Phase 0 complete-set arb pre-pass. That removes the prior benchmark mismatch while keeping the richer mint/merge/recycle rebalancer logic in scope.
5. `rebalanceExact()` remains in the harness as a sanity reference. On the current single-tick synthetic fixtures it matches `rebalance(...)`; if it materially underperforms constant-`L`, the Foundry test fails.
6. The legacy Phase 0-arb-inclusive off-chain snapshot is still stored in the Rust fixture (`offchain_mixed_ev_wei`) for historical context, but it is not used for the apples-to-apples Foundry dominance check.
7. The ignored Rust sweep helper accepts `AB_SWEEP_CASES`, `AB_SWEEP_SEED`, and `AB_SWEEP_OUTCOMES`, so the same dominance search can probe both small synthetic bundles and 98-outcome states before promoting a case into the committed fixture set.
8. For an explicit live L1 comparison, run the ignored Rust helper `write_live_l1_single_tick_benchmark_report` first. It fetches the current live 98-outcome L1 `slot0` snapshot, collapses each pool to its current single active tick band using the embedded tick ladder, and writes a transient report to [rebalancer_ab_live_l1_snapshot_report.json](/Users/shotaro/proj/deep_trading/test/fixtures/rebalancer_ab_live_l1_snapshot_report.json). Then run the ignored Foundry test `test_rebalancer_ab_live_l1_snapshot_report` to compare the on-chain constant/exact solvers against the same frozen report.
9. The arb-timing hypothesis helper now supports the strict interpretation "arb at end only": repeat `rebalance-only -> arb-only` cycles until convergence, instead of arbing before discretionary rebalancing. Use `print_phase0_arb_start_vs_end_cyclic_hypothesis` for committed fixtures and `sweep_phase0_arb_start_vs_end_cyclic_hypothesis` for randomized families.
10. The arb-timing helpers include a held-inventory sanity check: for materially held outcomes (`> 1e-6` units), terminal held-outcome profitability spread must stay below `1e-3`.

## Current Finding

On the current committed benchmark fixtures:

1. In all current direct-only parity fixtures, off-chain direct and on-chain constant-`L` `rebalance(...)` now match within tiny rounding tolerance under the aligned `1 bp` fee model.
2. In the aligned `small_bundle_mixed_case`, the off-chain full rebalance-only EV still equals the off-chain direct EV and matches on-chain constant-`L` `rebalance(...)` within `1924` wei.
3. In `mixed_route_favorable_synthetic_case`, the off-chain full rebalance-only EV exceeds both off-chain direct and on-chain constant by about `8.65e16` wei.
4. In `heterogeneous_ninety_eight_outcome_l1_like_case`, the off-chain full rebalance-only EV exceeds on-chain constant by about `1.2058e17` wei, while off-chain direct remains within the same parity tolerance band as on-chain constant.
5. The ignored randomized sweep now makes the route-value claim measurable instead of anecdotal: with the current seeded search family, 4-outcome sweeps found full-solver improvements in `480 / 500` sampled states, and 98-outcome sweeps found improvements in `100 / 100` sampled states.
6. The previously observed lower `offchain_mixed_ev` on `small_bundle_mixed_case` came from including the standalone Phase 0 buy-merge arb pre-pass in the Rust path while comparing it against plain `rebalance(...)`. That historical value is still recorded, but it is no longer treated as an apples-to-apples benchmark result.
7. Under the strict end-only cyclic interpretation (`rebalance-only -> arb-only` repeated), the committed fixtures still show one material improvement case (`small_bundle_mixed_case`) and otherwise ties. In randomized 4-outcome sweeps the result is mixed (both wins and losses), while in 98-outcome sweeps the deltas are mostly ties with tiny improvements.

The supported benchmark conclusion is now:

- direct-only is a parity sanity check
- full rebalance-only is a weak-dominance check
- the committed fixture set now includes cases where the richer off-chain solver is measurably better on EV
