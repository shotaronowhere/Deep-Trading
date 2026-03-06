# Rebalancer Approaches Playbook (Canonical)

## Purpose

This is the canonical strategy document for choosing and improving rebalancer approaches in this repo.

Companion deep-dive:

- [rebalancing_mechanism_design_review.md](rebalancing_mechanism_design_review.md) provides a longer-form mechanism-design critique and hypothesis set.
- This playbook remains the operational source of truth for policy thresholds and execution protocol.

It unifies:

1. Off-chain planning and execution behavior (`src/portfolio/core/*`, `src/execution/*`)
2. On-chain direct solver behavior (`contracts/Rebalancer.sol`)
3. On-chain mixed solver behavior (`contracts/RebalancerMixed.sol`)
4. Slippage, staleness, and churn controls
5. Production strategy selection policy with concrete thresholds
6. Live-L1 validation protocol for calibrating and revising thresholds

## Current implementation map

## Off-chain planner + strict executor

Core files:

- `src/portfolio/core/rebalancer.rs`
- `src/portfolio/core/waterfall.rs`
- `src/portfolio/core/planning.rs`
- `src/portfolio/core/bundle.rs`
- `src/portfolio/core/solver.rs`
- `src/execution/bounds.rs`
- `src/execution/grouping.rs`
- `src/execution/tx_builder.rs`
- `src/bin/execute.rs`

Properties:

- multi-phase optimization (arb, sell, waterfall, recycle, polish, cleanup)
- richer route surface (`DirectBuy`, `DirectSell`, `MintSell`, `BuyMerge`, `DirectMerge`)
- strict execution unit is one subgroup per transaction, then replan
- conservative short-horizon repricing and per-leg `sqrtPriceLimitX96`
- stale/deadline/bounds/shape failures are fail-closed

## On-chain direct solver (`Rebalancer.sol`)

Entrypoints:

- `rebalance` (constant-`L`)
- `rebalanceExact` (tick-aware exact)
- `rebalanceAndArb` and `rebalanceAndArbExact`

Properties:

- atomic solve + execute inside one transaction
- per-pool price limits for buys/sells
- bounded constant-`L` refinement loop (`MAX_WATERFALL_PASSES = 6`)
- exact mode solves on tick-scanned cost curves

## On-chain mixed solver (`RebalancerMixed.sol`)

Entrypoints:

- `rebalanceMixed` (legacy heuristic)
- `rebalanceMixedConstantL` (mixed attempt + fail-closed direct fallback)

Properties:

- deterministic fallback with `MixedSolveFallback(reasonCode)`
- currently useful as conditional EV extractor, not default path
- material gas premium in benchmark fixtures

## Approach strengths and weaknesses

| Approach | Strengths | Weaknesses | Best Use |
|---|---|---|---|
| Off-chain full solver + strict executor | Richest objective surface, can outperform direct-only on frozen states, already has conservative execution controls | Inclusion-latency exposure, more moving parts, local heuristics still present | Lower contention windows, large mixed-route opportunities |
| On-chain constant-`L` (`rebalance`) | Lowest operational complexity, atomic state consistency, robust baseline for single-tick-heavy markets | Approximation across initialized tick crossings | Default production mode |
| On-chain exact (`rebalanceExact`) | Best direct-route fidelity under multi-tick complexity | Higher planning gas | Crossing-heavy states where extra fidelity pays for itself |
| On-chain mixed (`rebalanceMixedConstantL`) | Captures mixed-route EV in favorable states, deterministic failover | High gas overhead, feasibility/fallback sensitivity | Conditional use only when uplift clears risk-adjusted gas hurdle |

## Route mechanics and practical implications

## Direct route

- Stable and predictable.
- Closest apples-to-apples parity between off-chain direct and on-chain constant-`L` in committed fixtures.
- Best first-line route in high-competition conditions.

## Mixed route

- Can add EV when non-active complements are sellable at favorable frontier economics.
- Value is state-dependent and can be outweighed by gas.
- Must be gated by incremental EV-net, not enabled blindly.

## Complete-set arb timing

- Phase-0 positioning changes downstream portfolio shape.
- Benchmarks must compare identical strategy sequences (for example, `rebalance` vs `rebalance`, or `arb+rebalance` vs `arb+rebalance`).
- Sequence mismatch can create false conclusions.

Historical benchmark context:

- `docs/archive/rebalancer/rebalancer_ab.md`
- `docs/archive/rebalancer/rebalancer_ab_mixed_gap_investigation_2026-03-03.md`

## Slippage, staleness, and churn

## Slippage and staleness

- Historical static-tolerance hybrid is intentionally deprecated (legacy write-up is archived at `docs/archive/slippage_guard_sprint_spec_legacy.md`).
- Current off-chain execution now uses:
  - strict subgroup receding horizon
  - conservative quote widening from short-horizon adverse move assumptions
  - explicit per-leg price limits
  - stale/deadline fail-closed submission checks
- On-chain solver remains strongest against staleness because solve and execute are atomic.

## Churn

- Off-chain has optional EV-guarded greedy preserve selection (`RebalanceFlags.enable_ev_guarded_greedy_churn_pruning`).
- This is a local non-regressive selector, not a global optimum solver.
- Keep optional and measurable.

## Production strategy policy (v1 thresholds)

These thresholds are concrete starting values. They are not permanent constants; they are designed to be validated and tuned by the experiment protocol below.

## Policy A: On-chain constant-`L` vs on-chain exact

Default:

- Use `Rebalancer.rebalance` (constant-`L`).

Escalate to `rebalanceExact` only when both conditions hold:

1. Expected uplift hurdle:
   - `EV_gain_exact_vs_constant_susd >= max(0.10, 3.0 * extra_gas_exact_vs_constant_susd)`
2. Consistency hurdle over window:
   - In the latest 20 live-L1 snapshots, at least 6 satisfy condition 1.

Demote back to constant-`L` when:

1. Median `EV_gain_exact_vs_constant_susd < 2.0 * extra_gas_exact_vs_constant_susd` over latest 20 snapshots.
2. Any hard negative outlier appears below `-0.02 sUSD` in that window.

## Policy B: On-chain mixed activation

Default:

- Do not run mixed on-chain by default.

Enable mixed only when all conditions hold:

1. Uplift hurdle:
   - `EV_gain_mixed_vs_direct_susd >= max(0.20, 5.0 * extra_gas_mixed_vs_direct_susd)`
2. Reliability hurdle:
   - `MixedSolveFallback` rate <= 20% over the latest 100 attempts.
3. Adverse-selection hurdle:
   - If one fallback reason exceeds 60% of fallbacks, disable mixed until root-caused.

Disable mixed immediately when either condition holds:

1. Rolling median `EV_gain_mixed_vs_direct_susd <= 0`.
2. Rolling p10 `EV_gain_mixed_vs_direct_susd < -0.05 sUSD`.

## Policy C: Off-chain execution vs on-chain execution

Default:

- Prefer on-chain direct solver during competitive hours.

Permit off-chain strict execution when all conditions hold:

1. Conservative move regime:
   - `execution_quote_latency_blocks * execution_adverse_move_bps_per_block <= 20 bps`
2. Margin regime:
   - first strict subgroup `guaranteed_profit_floor_susd >= 2.0 * estimated_gas_total_susd`
3. Freshness regime:
   - no stale-plan aborts in last 50 execution attempts

Otherwise, route to on-chain atomic solving.

## Minimal live-L1 threshold validation protocol

Goal:

- Validate and tune policies A, B, and C on real 98-outcome L1 state snapshots.

Cadence:

- Daily for one week to bootstrap thresholds
- Then every 3 days, plus after any major solver/execution change

## Step 1: Capture a fresh live-L1 frozen report

Run:

```bash
cargo test write_live_l1_single_tick_benchmark_report -- --ignored --nocapture --test-threads=1
```

Expected artifact:

- `test/fixtures/rebalancer_ab_live_l1_snapshot_report.json`

## Step 2: Compare on-chain constant vs exact on the same frozen report

Run:

```bash
forge test --match-test test_rebalancer_ab_live_l1_snapshot_report -vv
```

Collect:

- `EV_constant`
- `EV_exact`
- `Gas_constant`
- `Gas_exact`
- Any touched-pool or solve diagnostics printed by harness

Compute:

- `EV_gain_exact_vs_constant_susd = EV_exact - EV_constant`
- `extra_gas_exact_vs_constant_susd = (Gas_exact - Gas_constant) * gas_price * ETHUSD`

## Step 3: Compare on-chain direct vs on-chain mixed

Run:

```bash
forge test --match-test test_rebalancer_vs_mixed_apples_to_apples_report -vv
```

Collect per case:

- `EV_direct` (`Rebalancer.rebalance`)
- `EV_mixed` (`RebalancerMixed.rebalanceMixedConstantL`)
- `Gas_direct`
- `Gas_mixed`
- fallback reason traces

Compute:

- `EV_gain_mixed_vs_direct_susd = EV_mixed - EV_direct`
- `extra_gas_mixed_vs_direct_susd = (Gas_mixed - Gas_direct) * gas_price * ETHUSD`
- fallback-rate and dominant-reason metrics

## Step 4: Validate off-chain robustness and sequence sensitivity

Run:

```bash
cargo test benchmark_snapshot_matches_current_optimizer -- --nocapture
cargo test print_phase0_arb_start_vs_end_cyclic_hypothesis -- --ignored --nocapture --test-threads=1
```

Optional wider sweep:

```bash
cargo test sweep_phase0_arb_start_vs_end_cyclic_hypothesis -- --ignored --nocapture --test-threads=1
```

Purpose:

- confirm route-value and arb-timing assumptions used in policy C
- avoid sequence-mismatch conclusions

## Step 5: Update rolling threshold dashboard

Schema and templates:

- [rebalancer_policy_metrics_schema.md](rebalancer_policy_metrics_schema.md)
- [rebalancer_policy_metrics_template.csv](rebalancer_policy_metrics_template.csv)
- [rebalancer_policy_metrics_template.json](rebalancer_policy_metrics_template.json)

Automation script:

- `scripts/rebalancer_policy_metrics_from_logs.sh`

Example:

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

`--fallback-log` is required for a positive Policy-B enable/disable decision. Without it, the script still reports EV/gas aggregates but leaves fallback-rate fields empty and keeps `policy_b_enable_now=false`.

For each run, append one row with:

1. Date and block reference
2. `EV_gain_exact_vs_constant_susd`
3. `extra_gas_exact_vs_constant_susd`
4. `EV_gain_mixed_vs_direct_susd`
5. `extra_gas_mixed_vs_direct_susd`
6. mixed fallback rate
7. stale-plan abort count from off-chain strict runtime logs

Decision updates:

1. Apply policy A/B/C exactly.
2. If policies flap for 3 consecutive runs, widen hysteresis multipliers by +0.5 on gas multiples.
3. If realized post-trade EV consistently trails predicted floors, raise margin multipliers by 25%.

## Initial recommendations

1. Keep on-chain constant-`L` as production default now.
2. Enable exact only when policy A triggers on live snapshots.
3. Keep on-chain mixed disabled by default; enable only under policy B.
4. Use off-chain strict execution as conditional mode under policy C, not as universal default.

## References

- `docs/README.md`
- `docs/rebalancer.md`
- `docs/rebalancer_mixed.md`
- `docs/slippage.md`
- `docs/archive/rebalancer/rebalancer_ab.md`
- `docs/archive/rebalancer/rebalancer_mixed_constant_l_solver.md`
- `docs/archive/rebalancer/rebalancer_vs_rebalancer_mixed_benchmark_2026-03-03.md`
- `docs/archive/rebalancer/rebalancer_ab_mixed_gap_investigation_2026-03-03.md`
