# Global Solver EV Regression

Date: 2026-02-18  
Scope: `RebalanceEngine::GlobalCandidate` vs incumbent EV on rebalance fixtures.

## Regression Command

```bash
cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
```

## Latest Recovery Checkpoint (Phase D route-faithful rebuild)

Latest artifact:

- `/tmp/global_ev_default_1771415895866.jsonl`

Latest results:

- `total_cases=50`
- `candidate_valid=50`
- `candidate_invalid=0`
- `mean_delta=-0.422000933261`
- `best_delta=1.856221833414`
- `worst_delta=-3.207255161607`
- `candidate_better=3/50`

Family decomposition:

- full-L1 overall (`n=26`, fuzz + regression): `mean_delta=-0.670778756323`
- full-L1 fuzz (`n=24`): `mean_delta=-0.801324117995`
- full-L1 regression snapshots (`n=2`): `mean_delta=0.895765583734`
- partial-L1 (`n=24`): `mean_delta=-0.152491624944`

Cases where candidate beat incumbent:

- `fuzz_full_l1_case_0`: `delta=+0.888289612926`
- `fuzz_full_l1_case_18`: `delta=+1.387274542502`
- `regression_full_l1_snapshot`: `delta=+1.856221833414`

## Pure Solver-Only Benchmark (route refinement disabled)

Command:

```bash
GLOBAL_SOLVER_ENABLE_ROUTE_REFINEMENT=false cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
```

Artifact:

- `/tmp/global_ev_default_pure_1771417465226.jsonl`

Results:

- `candidate_valid=50/50`
- overall `mean_delta=-19.762219369469`
- full-L1 fuzz (`n=24`): `mean_delta=-34.604352163150`
- partial-L1 (`n=24`): `mean_delta=-3.304407858962`
- full-L1 regression snapshots (`n=2`): `mean_delta=-39.150363971376`
- `candidate_better=2/50`

Interpretation: the raw projected solver remains far below incumbent EV; most recovered EV comes from the route-refinement layers.

## Stage Checkpoint Summary

Before stage 1-3:

- `total_cases=50`
- `candidate_valid=2`
- `candidate_invalid=48`
- `mean_delta=-42.910196918399`
- Invalid reasons:
  - `ProjectedGradientTooLarge=44`
  - `ProjectionUnavailable=4`

After stage 1-3:

- `total_cases=50`
- `candidate_valid=50`
- `candidate_invalid=0`
- `mean_delta=-22.407627450367`
- `best_delta=0.000000000000`
- `worst_delta=-106.491017066174`

## Where the Gap Remains

From parsed case-level output (`/tmp/global_ev_latest.log`):

- Full-L1 subset (`n=26`): `mean_delta=-43.091591245551`
- Partial-L1 subset (`n=24`): `mean_delta=-0.000000005584`

Interpretation:

- Validity/convergence gating is no longer the blocker.
- Residual EV underperformance is concentrated in full-L1 complete-set regimes.

## Verification Loop

- `cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1` (pass)
- `cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture` (pass)
- `cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture` (pass)

## Detailed Analysis

See `docs/global_solver_stage_report_2026-02-18.md` for:

- stage-by-stage implementation log
- full vs partial decomposition
- thesis/routing context mapping
- rejected stage-4 experiment and next-stage plan
