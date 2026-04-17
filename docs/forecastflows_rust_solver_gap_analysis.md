# ForecastFlows Rust Solver Gap Analysis

Status: reassessed on 2026-04-16 after switching the default backend to
`rust_worker`.

## Conclusion

The earlier headline claim that ForecastFlows underperforms the current native
solver by about 30% on `heterogeneous_ninety_eight_outcome_l1_like_case` is no
longer supported by the current benchmark harness.

Current Rust-worker selected rows show ForecastFlows matching or slightly
beating the release-facing native rows on all committed shared-snapshot cases.
The main 98-outcome L1-like case is:

| Case | ForecastFlows net EV | Release native net EV | Delta |
|---|---:|---:|---:|
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `150.36542032405256` | `150.3651857457622` | `+0.00023457829036` |

The old `215.17` native net-EV figure should be treated as stale and not used
as evidence of ForecastFlows solver underperformance.

## Verified Current Rust-Worker Rows

Command:

```bash
FORECASTFLOWS_WORKER_BIN=/Users/shotaro/proj/ForecastFlows.rs/target/release/forecast-flows-worker \
  cargo test 'portfolio::core::tests::rebalancer_contract_ab::print_shared_op_snapshot_forecastflows_selected_rows_jsonl' \
  -- --ignored --exact --nocapture --test-threads=1
```

Result summary:

| Case | ForecastFlows net EV | Variant | Actions | Backend |
|---|---:|---|---:|---|
| `two_pool_single_tick_direct_only` | `100.10232463534447` | direct | 2 | `rust_worker` |
| `ninety_eight_outcome_multitick_direct_only` | `98.11421045306142` | mixed | 99 | `rust_worker` |
| `small_bundle_mixed_case` | `100.14776321881526` | mixed | 5 | `rust_worker` |
| `legacy_holdings_direct_only_case` | `38.86288988543227` | direct | 2 | `rust_worker` |
| `mixed_route_favorable_synthetic_case` | `100.10782500742624` | mixed | 5 | `rust_worker` |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `150.36542032405256` | mixed | 92 | `rust_worker` |

These rows agree with the release-facing matrix in
`docs/solver_benchmark_matrix.md`: the current committed shared-snapshot lane is
not showing material ForecastFlows underperformance.

## Theoretical Foundation

ForecastFlows is the right global solver for the static convex routing problem
it is given:

- each outcome pool is a convex AMM edge between collateral and one outcome
- the complete-set facility is a linear split/merge hyperedge
- the objective is linear terminal portfolio value
- the solver searches direct-only and split/merge-enabled families, then
  certifies the duality gap

Under those assumptions, sequential price impact is not a reason for a
waterfall trace to beat a certified ForecastFlows solution. For a deterministic
CFMM edge, the cost of moving a pool from one reserve state to another is a path
integral; splitting a monotone trade into many smaller trades does not create
extra economic value. The split/merge operation is linear and fee-free, so
multiple gross mints and merges collapse to one net split/merge flow unless
some extra execution constraint is being imposed.

Therefore, if a waterfall plan appears to beat a certified ForecastFlows plan by
a large margin on the same snapshot, the first hypothesis should be:

1. the two plans are not being evaluated under the same feasible action
   semantics;
2. the native trace is using a richer or different action set than the
   ForecastFlows net-flow formulation;
3. the benchmark accounting is inconsistent; or
4. the ForecastFlows branch is uncertified or dropped during local translation.

It should not be explained as "ForecastFlows cannot model sequential
price-impact feedback" unless a specific non-convex, path-dependent execution
constraint has first been identified.

## Semantic Difference: ForecastFlows vs Waterfall Actions

ForecastFlows emits net flows:

- one net AMM trade per market, after worker-side netting
- one net split/merge amount, either mint or merge
- Rust translation then chooses an executable chronology:
  direct sells, direct merges, mint-and-sell rounds, buy-merge rounds, and
  remaining direct buys
- translation rejects dual mint+merge candidates and rejects merge/sell replay
  when balances are insufficient

The native waterfall emits a procedural trace:

- phase-0 complete-set arb
- phase-1 sells
- phase-2 waterfall allocation
- phase-3 recycling
- bounded polish passes
- cleanup sweeps

That trace can contain thousands of actions. The trace is not a convex global
solve; it is a heuristic execution search plus later compaction and net-EV
ranking.

The important semantic mismatch is balance feasibility. ForecastFlows replay is
strict about merge inventory. Native balance updates use `subtract_balance`,
which clamps negative balances to zero. Native EV summaries also use
`state_snapshot_expected_value`, which clamps holdings at zero before valuing
the terminal portfolio. Thus an over-merge or over-sell can be made invisible in
native summary EV, while a signed action replay still exposes the deficit.

## Pathological Large-Nonprefix Case

The synthetic `forecastflows_large_nonprefix_active_set_case` still prints a
negative ForecastFlows-vs-native `net_ev_gap_susd` in
`print_forecastflows_pathological_rows_jsonl`, but the row is internally
inconsistent:

| Metric | Value |
|---|---:|
| Native action-replay raw EV | `127.64659349423643` |
| Native reported fee | `0.23405446619938053` |
| Native signed-replay net EV | `127.41253902803705` |
| Native summary net EV | `128.8536672860593` |
| ForecastFlows net EV | `127.56570001109891` |

The native summary net EV is larger than the independently replayed native raw
EV after subtracting a positive fee, which is impossible on a consistent basis.
The difference is explained by negative holdings in the native action replay:
the summary path clamps negative holdings to zero, while the replay used for
`native_raw_ev_susd` keeps signed balances.

On the signed action-replay basis, ForecastFlows is ahead of native by roughly
`0.15316` sUSD:

```text
127.56570001109891 - (127.64659349423643 - 0.23405446619938053)
= 0.15316098306186
```

So this case is not clean evidence that ForecastFlows is economically worse.
It is evidence that the native trace/accounting path can produce or tolerate an
infeasible over-merge/negative-balance trace that ForecastFlows correctly
refuses to emulate.

## Pathological Multiband Case

`print_forecastflows_pathological_rows_jsonl` currently panics before producing
the multiband chronology row:

```text
price 0.06195247868562458 outside tick bounds [0.16531376479919885, 0.9999000099990001]
for forecastflows_multiband_mixed_chronology_case_0
```

This is not a ForecastFlows certification result. The synthetic case's starting
price is outside its constructed tick ladder, so the fixture itself must be
fixed before it can be used to diagnose mixed-mode convergence.

## Why ForecastFlows Does Not Always "Find the Global Optimum"

The precise statement is:

ForecastFlows finds the global optimum of its certified convex net-flow
formulation.

It is not a global optimizer over every possible driver behavior if the driver
allows behavior outside that formulation, for example:

- native traces that over-merge or over-sell and then clamp negative balances;
- exact transaction packing or gas discontinuities not included in the worker
  objective;
- unsupported or malformed multi-band fixtures;
- uncertified mixed solves that are intentionally dropped; or
- executable rounding and integer constraints that appear after the continuous
  solve.

The current investigation did not find a valid committed benchmark where a
certified Rust ForecastFlows solution is beaten by a feasible native plan under
the same economic accounting.

## Recommended Next Steps

1. Make native replay fail closed on negative balances for `Sell` and `Merge`,
   or make summary EV use signed holdings consistently. Do not compare
   ForecastFlows against clamped negative-balance native summaries.
2. Fix `forecastflows_multiband_mixed_chronology_case` so every starting price
   lies inside the generated ladder.
3. Rerun the pathological benchmark assertions only after the accounting and
   fixture issues are fixed.
4. Do not implement iterative ForecastFlows or native-polish-after-FF based on
   the stale `215.17` gap. If a strict replay benchmark later shows a real
   native edge, compare terminal state, net pool deltas, split/merge amount, and
   signed balances before changing the ForecastFlows formulation.
