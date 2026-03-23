# ForecastFlows Pathological Benchmarks

Status: stress-only diagnostic coverage, not part of the release-facing solver matrix.

Source-of-truth code path:

- `/Users/shotaro/proj/deep_trading/src/portfolio/tests/rebalancer_contract_ab.rs`

## Why This Exists

The committed release-facing benchmark matrix in
[`docs/solver_benchmark_matrix.md`](/Users/shotaro/proj/deep_trading/docs/solver_benchmark_matrix.md)
already shows ForecastFlows beating the native waterfall path on every shared-snapshot committed
case, but most of those wins are effectively dust.

The current committed deltas versus native waterfall are:

- `two_pool_single_tick_direct_only`: `+0.00000000353536`
- `ninety_eight_outcome_multitick_direct_only`: `+0.00809839419698`
- `small_bundle_mixed_case`: `+0.00000000496113`
- `legacy_holdings_direct_only_case`: `+0.00000000353164`
- `mixed_route_favorable_synthetic_case`: `+0.00000002188504`
- `heterogeneous_ninety_eight_outcome_l1_like_case`: `+0.00023457826520`

That benchmark is good release evidence, but it is not a strong stress harness for the two places
where ForecastFlows should plausibly shine:

- large active-set search beyond the native runtime-capped `constant_l_mixed` seed/local-search window
- true multi-band mixed chronology where the worker sees an explicit `UniV3` ladder instead of a
  single-range simplification

## Added Stress Cases

The pathological stress suite intentionally stays out of:

- `test/fixtures/rebalancer_ab_cases.json`
- `test/fixtures/rebalancer_ab_expected.json`
- the release-facing matrix in
  [`docs/solver_benchmark_matrix.md`](/Users/shotaro/proj/deep_trading/docs/solver_benchmark_matrix.md)

The suite adds two generated Rust-only cases:

### `forecastflows_large_nonprefix_active_set_case`

- single-tick shape
- more than `16` directly profitable outcomes
- top direct-profitability outcomes are deliberately deep and expensive
- lower-ranked profitable outcomes are shallower and cheaper
- intended stress:
  - native `constant_l_mixed` runtime cap at `16` profitable outcomes
  - prefix/singleton seed bias when the best active set is not a direct-profitability prefix

Target assertion:

- ForecastFlows net EV beats native by at least `0.001` sUSD under the benchmark OP snapshot

### `forecastflows_multiband_mixed_chronology_case`

- true multi-band tick ladders
- asymmetric overvalued sell legs plus undervalued buy legs
- buy legs require meaningful in-ladder movement rather than a single active band
- intended stress:
  - chronology across mixed sell/buy routing
  - ForecastFlows request construction over real multi-band ladders

Target assertion:

- ForecastFlows net EV beats native by at least `0.0005` sUSD under the benchmark OP snapshot
- every stressed market sent to ForecastFlows uses `band_count > 2`

## How To Run

Compile-only sanity pass:

```bash
cargo test forecastflows_pathological -- --nocapture
```

Gated live-worker assertion:

```bash
FORECASTFLOWS_BENCHMARK_ASSERT=1 cargo test \
  forecastflows_pathological_stress_cases_show_material_net_ev_gap_when_enabled \
  -- --nocapture --test-threads=1
```

Machine-readable pathological rows:

```bash
cargo test print_forecastflows_pathological_rows_jsonl \
  -- --ignored --nocapture --test-threads=1
```

When the solver-heavy large-active-set case is materially faster in optimized mode on your machine,
the same printer is also useful under `--release`:

```bash
cargo test --release print_forecastflows_pathological_rows_jsonl \
  -- --ignored --nocapture --test-threads=1
```

## Output Contract

`print_forecastflows_pathological_rows_jsonl` emits one JSON row per case with:

- case id and scenario metadata
- profitable count and profitable bucket
- band counts used in the ForecastFlows request
- native raw EV / fee / net EV
- ForecastFlows raw EV / fee / net EV
- raw-EV gap and net-EV gap
- native and ForecastFlows action / tx counts
- ForecastFlows family / compiler variant
- ForecastFlows worker request count, availability, fallback reason, and latency metadata

## Verification Notes

The code and docs are aligned around these names:

- test: `forecastflows_pathological_stress_cases_show_material_net_ev_gap_when_enabled`
- printer: `print_forecastflows_pathological_rows_jsonl`
- cases:
  - `forecastflows_large_nonprefix_active_set_case`
  - `forecastflows_multiband_mixed_chronology_case`

If the printer is rerun and the case constants are tuned further, update this document with the new
measured native-vs-ForecastFlows deltas and keep the assertion floors in sync with the committed
case definitions.
