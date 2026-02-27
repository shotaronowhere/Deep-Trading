# Rebalance Execution Alignment Baseline (2026-02-27)

This document captures pre-change baselines for the execution-aligned waterfall upgrade.

## Environment

- Date: 2026-02-27
- Branch: `master`
- Profile: `--release`

## 1) Local Gradient Heuristic

Command:

```bash
cargo test --release test_waterfall_local_gradient_heuristic_complex_cases -- --ignored --nocapture --test-threads=1
```

Key outcomes:

- `fuzz_full_case_0`: `ev_before=198.834559875`, `ev_after=366.578577341`, `gain=167.744017466`
- `fuzz_full_case_1`: `ev_before=151.070724989`, `ev_after=313.042562008`, `gain=161.971837020`
- `fuzz_partial_case_0`: `ev_before=137.540917680`, `ev_after=234.146224136`, `gain=96.605306456`
- `fuzz_partial_case_1`: `ev_before=244.853693791`, `ev_after=357.770533502`, `gain=112.916839712`

Post-grad snapshot examples:

- `max_direct_grad=-0.000303289`, `max_indirect_grad=0.000000000`
- `max_direct_grad=-0.001522715`, `max_indirect_grad=0.000000000`
- `max_direct_grad=-0.001198158`, `max_indirect_grad=0.000000000`
- `max_direct_grad=-0.000117417`, `max_indirect_grad=0.000000000`

## 2) Second-Pass Idempotence

Command:

```bash
cargo test --release test_rebalance_second_pass_gain_complex_cases -- --ignored --nocapture --test-threads=1
```

Key outcomes:

- `fuzz_full_case_0`: `first_gain=167.744017466`, `second_gain=0.000000000`, `actions_first=70939`, `actions_second=0`
- `fuzz_full_case_1`: `first_gain=161.971837020`, `second_gain=0.000000000`, `actions_first=52332`, `actions_second=0`
- `fuzz_partial_case_0`: `first_gain=96.605306456`, `second_gain=0.000000000`, `actions_first=200`, `actions_second=0`
- `fuzz_partial_case_1`: `first_gain=112.916839712`, `second_gain=0.000000000`, `actions_first=129`, `actions_second=0`

## 3) Random Search vs Waterfall (5s constrained run)

Command:

```bash
MC_SEARCH_MAX_ROLLOUTS=50000 MC_SEARCH_MIN_RUNTIME_SECS=5 MC_SEARCH_CHECKPOINT_EVERY=5000 cargo test --release test_random_group_search_vs_waterfall_complex_fuzz_cases -- --ignored --nocapture --test-threads=1
```

Key outcomes:

- `fuzz_full_case_0`: `algo_ev=366.578577341`, `random_best_ev=366.764322244`, `gap_to_algo=-0.185744903`
- `fuzz_full_case_1`: `algo_ev=313.042562008`, `random_best_ev=311.911053391`, `gap_to_algo=1.131508618`
- `fuzz_partial_case_0`: `algo_ev=234.146224136`, `random_best_ev=234.005706204`, `gap_to_algo=0.140517932`
- `fuzz_partial_case_1`: `algo_ev=357.770533502`, `random_best_ev=357.699337731`, `gap_to_algo=0.071195771`

## 4) Full-L1 Performance

Command:

```bash
cargo test --release test_rebalance_perf_full_l1 -- --nocapture
```

Key outcomes:

- `Total: 1.69090725s for 10 iterations`
- `Per call: 169.090725ms`
- `Actions: 40712 total (8996 buys, 30589 sells, 1089 mints, 38 merges)`
- `EV before=100.000000000, after=1075.074419919, gain=975.074419919`
