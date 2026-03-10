# Off-Chain EV Optimization Log

Last updated: 2026-03-09

## Purpose

This document is the permanent memory for off-chain EV-optimization work:

- what ideas were explored
- what benchmark movement they produced
- which complexity layers are worth keeping
- which ideas are teacher-only or rejected for online use

Use this before adding any new off-chain search dimension.

Important chronology note:

- earlier sections in this log describe the raw-EV-first era of the solver
- the current runtime comparator is now estimated net EV, not raw EV
- the canonical current economics table is `docs/solver_benchmark_matrix.md`

Release-facing benchmark summary:

- `docs/solver_benchmark_matrix.md`

## Current status: K=1 certified, teacher-first era

Latest accepted solver state:

- the packed runtime now compacts rich exact no-arb traces by comparing:
  - `baseline_step_prune`
  - `target_delta`
  - `analytic_mixed`
  - `constant_l_mixed`
  - `coupled_mixed`
  - `staged_constant_l_2`
  - `direct_only`
  - `noop`
- `constant_l_mixed` is the baseline compact mixed compiler:
  - exact sequential-cash accounting for `Mint -> inactive Sell -> active Buy`
  - explicit mixed certificates
  - native active-set search in state space, not trace pruning
- `K=1` teacher coverage now exists:
  - exact oracle for `n <= 12`
  - exact medium oracle for `n <= 13`
  - deterministic large-case `best_known` comparison without the runtime profitable-universe caps
- `K=2` now has a teacher-only exact oracle on small mixed cases (`n <= 8`)
- benchmark-mode fee modeling is pinned to a fixed OP snapshot for deterministic net-EV comparisons

Latest benchmark outcome:

- `heterogeneous_ninety_eight_outcome_l1_like_case`
  - off-chain default raw EV: `150.380245589644673024`
  - modeled net EV: `150.3651857457622`
  - selected compiler: `constant_l_mixed`
  - actions: `94`
  - tx count: `1`
  - off-chain direct baseline raw EV: `150.258105490229428224`
  - off-chain direct baseline net EV: `150.25169228299788`
- `mixed_route_favorable_synthetic_case`
  - off-chain default raw EV: `100.108519024112205824`
  - modeled net EV: `100.10782498550209`
  - selected compiler: `constant_l_mixed`
  - actions: `5`
  - tx count: `1`
  - off-chain direct baseline raw EV: `100.021703842635415552`
  - off-chain direct baseline net EV: `100.02152838894948`

Interpretation:

- the synthetic compact mixed miss is closed
- the realistic 98-outcome case still beats both on-chain exact and on-chain mixed on modeled net EV
- direct-profitability prefixes were **not** the full K=1 story; the small-case oracle found non-prefix optima, so runtime K=1 was broadened
- large-`n` K=1 and all runtime `K=2` claims remain heuristic until teacher evidence says otherwise

## Current teacher-first rule

Current conclusion:

- do not add new runtime mixed-search dimensions without teacher evidence
- if a claim concerns single-stage optimality, settle it against the exact `K=1` teachers first
- if a claim concerns staged optimality, settle it against the `K=2` teacher before changing the online staged search
- treat large-`n` best-known comparisons as diagnostics, not proofs
- keep `docs/solver_benchmark_matrix.md` as the release-facing economics source of truth

Archived note:

- the earlier “near saturation after `coupled_mixed`” conclusion belongs to the pre-`constant_l_mixed` / pre-oracle era and should not be used to block new work

## 2026-03-10: gated proposal V2 and prefix-distance diagnostics

Current state:

- the default exact-no-arb path still keeps the legacy distilled preserve/frontier proposals
- `REBALANCE_ENABLE_DISTILLED_PROPOSAL_V2=1` adds a second bounded proposal family on top of the legacy proposals; it is additive and non-default in this milestone
- the bounded V2 family is intentionally tiny:
  - at most 3 preserve sets
  - at most 6 extra exact-no-arb proposal tasks
- the new `K=1` teacher diagnostic does not change runtime search; it classifies the oracle-best active mask relative to the best direct-profitability prefix seed as `prefix`, `one_move`, `two_moves`, or `beyond_two_moves`

Promotion gate:

- keep the ungated runtime behavior unchanged
- require non-regression on direct-only committed cases
- require non-regression on `mixed_route_favorable_synthetic_case`
- require non-regression on `heterogeneous_ninety_eight_outcome_l1_like_case`
- only after that should V2 be considered for default promotion or legacy-proposal replacement

Ignored diagnostics added for this phase:

- `cargo test print_constant_l_prefix_distance_histogram -- --ignored --nocapture`
- `cargo test print_heterogeneous_ninety_eight_proposal_v2_breakdown -- --ignored --nocapture`
- `cargo test print_shared_op_snapshot_offchain_selected_rows_with_proposal_v2_jsonl -- --ignored --nocapture --test-threads=1`

## Historical raw-EV benchmark frontier

Sources:

- current fixture: `test/fixtures/rebalancer_ab_expected.json`
- pre-operator legacy snapshot: commit `093e4b9`, same fixture path
- archived on-chain mixed comparison: `docs/archive/rebalancer/rebalancer_vs_rebalancer_mixed_benchmark_2026-03-03.md`

Historical note:

- the comparison tables in this section are preserved from earlier raw-EV-first milestones
- they are no longer the canonical current economics view after packed execution and analytic-mixed compaction
- the current release-facing net-EV comparison now lives in `docs/solver_benchmark_matrix.md`

### Raw-EV-era off-chain mixed vs legacy snapshot (`093e4b9`)

| Case | EV delta | EV delta % | Action delta |
|---|---:|---:|---:|
| `two_pool_single_tick_direct_only` | `0` | `0.000000%` | `0` |
| `ninety_eight_outcome_multitick_direct_only` | `0` | `0.000000%` | `0` |
| `small_bundle_mixed_case` | `+0.051996203394482176` | `+0.051946%` | `+2` |
| `legacy_holdings_direct_only_case` | `0` | `0.000000%` | `0` |
| `mixed_route_favorable_synthetic_case` | `+0.000291089000448` | `+0.000291%` | `-7` |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `+0.001444798828052480` | `+0.000961%` | `-3895` |

Interpretation:

- The meaningful legacy-to-current EV win is `small_bundle_mixed_case`.
- The heterogeneous 98-outcome case also improved, and it did so while materially reducing action count.
- Direct-only rows did not move, which is correct: off-chain mixed complexity should not perturb direct-only parity.

### Raw-EV-era off-chain mixed vs current on-chain exact

| Case | EV delta (off-chain mixed - on-chain exact) | EV delta % |
|---|---:|---:|
| `two_pool_single_tick_direct_only` | `+0.000000000000015206` | `+0.000000%` |
| `ninety_eight_outcome_multitick_direct_only` | `-0.000000000000534821` | `-0.000000%` |
| `small_bundle_mixed_case` | `+0.015610289881024388` | `+0.015590%` |
| `legacy_holdings_direct_only_case` | `-0.000000001853588459` | `-0.000000%` |
| `mixed_route_favorable_synthetic_case` | `+0.086792931624521758` | `+0.086774%` |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `+0.122026242587694435` | `+0.081211%` |

Interpretation:

- Off-chain mixed is already saturated on direct-only cases: parity up to dust.
- Off-chain mixed still materially beats on-chain exact direct on the genuinely mixed cases.

### Raw-EV-era off-chain mixed vs archived on-chain mixed constant-`L`

| Case | EV delta (off-chain mixed - on-chain mixed) |
|---|---:|
| `two_pool_single_tick_direct_only` | `+0.000000000000015206` |
| `ninety_eight_outcome_multitick_direct_only` | `-0.000000000000534821` |
| `small_bundle_mixed_case` | `+0.000000000000006353` |
| `mixed_route_favorable_synthetic_case` | `-0.000022249852264816` |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `-0.000007408968482685` |
| `legacy_holdings_direct_only_case` | `-0.000000000000000574` |

Interpretation:

- Current off-chain mixed is effectively at parity with the archived on-chain mixed benchmark.
- Remaining benchmark headroom versus on-chain mixed is now dust, not a large frontier gap.

## Runtime measurements

The pre-packing release measurements are now historical only.

- current packed default-path release perf has not been rerun yet
- use `docs/solver_benchmark_matrix.md` as the source of truth
- until the packed release harness is rerun, runtime values should be treated as `n/a` rather than inferred from the older staged-default numbers

## Gas-aware evaluation status

Implemented in this phase:

- exact unsigned `batchExecute` tx-byte construction from the real execution path
- live OP fee input queries:
  - `eth_gasPrice`
  - `eth_chainId`
  - `eth_getTransactionCount`
  - `GasPriceOracle.getL1Fee(txData)`
- exact first-group live report:
  - `cargo test print_live_op_first_group_exact_gas_report -- --ignored --nocapture`
- deterministic Foundry gas-profile suite:
  - `forge test --match-contract TradeExecutorGasProfileTest -vv`

Measured live OP first-group example on 2026-03-08:

- group kind: `DirectBuy`
- unsigned tx bytes: `492`
- exact L1 fee: `808,776,828 wei`
- gas price: `1,006,543 wei`
- exact total fee: about `$0.00017618`
- live `eth_estimateGas`: `29,318`

Implication:

- the old fallback L2 gas constants were materially too conservative
- the fallback L2 units were refreshed from the new Foundry suite
- gas is now measured exactly in diagnostics, and the planner now ranks by estimated net EV

Current state:

- benchmark-layer gas ablation now reconstructs nonzero strict subgroup plans on the committed benchmark fixture
- canonical small-shape calibration supports `l1_data_fee_floor_susd = 0.001`
- seeded hard-case gas replay is still incomplete:
  - many seeded cases currently emit zero replayable groups and only skip reasons
  - benchmark-layer net-EV pruning is trustworthy
  - seeded hard-case net-EV pruning is still partial

Interpretation:

- The current default full solver is too heavy to justify chasing tiny residual EV gains with more online search.
- Gas-aware pruning changes the runtime regime completely; online complexity should be judged under realistic gated conditions, not only the zero-threshold stress case.

## Gas-aware pruning result

Benchmark-layer ablation on 2026-03-08:

- `heterogeneous_ninety_eight_outcome_l1_like_case`
  - `r_exact_baseline_k4`, `distilled_proposals`, and `operator_only` all tie at:
    - raw EV `150.37985133577143`
    - net EV `35.543188032428546`
    - `100657` actions
    - `97301` replayable groups
  - `staged_fallback` improves to:
    - raw EV `150.38031485753552`
    - net EV `149.00171039448665`
    - `2594` actions
    - `966` replayable groups
- `small_bundle_mixed_case`
  - `operator_only` improves raw EV and net EV materially versus the exact no-arb baseline
  - `staged_fallback` still adds a smaller but positive raw-EV and net-EV increment while reducing actions from `11` to `9`
- `mixed_route_favorable_synthetic_case`
  - all measured layers tie exactly
- direct-only benchmark rows
  - all measured layers tie exactly

Seeded hard-case ablation on 2026-03-08:

- `distilled_proposals` help one seeded full case modestly
- `operator_only` does not improve on that distilled result
- `staged_fallback` is materially worse on that seeded full case
- replayable subgroup counts are still zero on the seeded hard-case suite, so those net-EV rows are only partial signals today

Default-path keep/cut decisions:

- Keep:
  - mixed-route capability
  - `R_exact` over `{default, direct, mint}`
  - `Plain` and `ArbPrimed` whole-plan families
  - legacy distilled preserve/frontier proposals inside `R_exact`
  - staged fallback, as the last retained complexity layer pending the net-EV-era release benchmark refresh
- Gate behind env only:
  - bounded preserve/frontier proposal V2 via `REBALANCE_ENABLE_DISTILLED_PROPOSAL_V2=1`
- Remove from default path:
  - late arb-correction tail
- Do not add:
  - larger preserve enumeration
  - deeper frontier branching
  - any new online search dimension

## What the latest teacher/oracle proved

Ignored diagnostic:

- `cargo test print_heterogeneous_ninety_eight_exact_preserve_oracle_breakdown -- --ignored --nocapture`

Result on the hard 98-outcome case:

- flat exact `K = 4` preserve search underperforms staged fallback
- flat exact `K = 8` preserve search improves EV, but still underperforms staged fallback
- staged-action churn seeding also underperforms staged fallback
- replaying the staged solver's actual chosen preserve set and first-frontier choice through the flat no-arb operator recovers the staged EV up to a tiny rounding wobble

Conclusion:

- the continuous waterfall is not the bottleneck
- the remaining gap is discrete choice recovery
- the next online win is to recover the right preserve/frontier choice cheaply, not to add more generic search depth

## Teacher distillation phase

Implemented in the current code:

- offline teacher snapshots from the staged winner on benchmark and seeded hard cases
- machine-readable ignored test printers for benchmark and seeded hard-case rows
- a tiny online distilled preserve/frontier proposal heuristic inside `R_exact`

What the phase proved:

- the new teacher harness is useful and permanent
- the distilled online proposals are non-regressive versus the old flat `K = 4` exact no-arb path
- the fallback-removal gate does not pass yet:
  - the heterogeneous 98-outcome operator-only gap remains far above the `65_536`-wei target
  - the staged fallback still protects the committed mixed-EV frontier
- this means the old teacher-distillation line of work was not enough to replace the staged fallback under the pre-packing execution topology

Engineering decision:

- replace fragmented subgroup pricing with packed execution-program compilation
- make the packed operator solver the default hot path
- keep staged reference only as an opt-in rollout comparison via `REBALANCE_ENABLE_STAGED_FALLBACK=1`
- keep the legacy distilled proposals in the default path and add any new proposal family only behind an explicit gate
- remove the late arb-correction tail from the default path
- do not add another online search dimension on top of this phase
- if EV work resumes, it should be teacher-to-student proposal distillation only
- otherwise shift effort to speed/simplicity work and keep only the optimization paths that clearly earn their cost

Follow-up diagnostic:

- `cargo test print_heterogeneous_ninety_eight_variant_proposal_breakdown -- --ignored --nocapture`

Result:

- extracting an exact preserve-set proposal independently from each static phase-order variant converges to the same 4-name preserve set
- replaying that proposal through `exact_no_arb` recovers most of the residual heterogeneous-case gap
- but it still trails the staged choice by about `0.000011913546661888` EV while materially increasing runtime when wired into the online solver

Conclusion:

- this is a useful teacher signal
- it is not worth keeping in the online default path until it can be distilled into a much cheaper proposal heuristic

## Release status (v1)

- v1 default path is now the packed execution-program solver
- reason:
  - packing fixed the main fragmentation failure mode without adding another online search dimension
  - the staged reference remains available only as an opt-in rollout comparison via `REBALANCE_ENABLE_STAGED_FALLBACK=1`
- v1 default path keeps:
  - mixed-route capability
  - `R_exact` over `{default, direct, mint}`
  - `Plain` and `ArbPrimed`
  - legacy distilled preserve/frontier proposals
  - packed-vs-strict execution-program compilation
  - net-EV ranking with gas-aware gating
- v1 default path prunes:
  - late arb-correction tail
  - default staged-fallback selection
  - any further online search expansion
- v1 optional gated additions:
  - bounded preserve/frontier proposal V2 via `REBALANCE_ENABLE_DISTILLED_PROPOSAL_V2=1`
- explicitly deferred post-release:
  - packed-path release perf rerun
  - any teacher-distilled preserve/frontier codebook beyond offline diagnostics
  - any additional route/search improvement needed to close the residual on-chain gap on heterogeneous and mixed-route favorable cases
- the canonical release-facing solver comparison table now lives in `docs/solver_benchmark_matrix.md`

## Explored ideas and status

### Keep online

| Idea | Status | Why |
|---|---|---|
| Mixed-route capability | Keep | Real EV uplift on the mixed cases |
| `R_exact` over `{default, direct, mint}` frontier families | Keep | Cheap exactness over the right continuous core |
| Small preserve-set search | Keep | Preserve choice is a real EV lever |
| Legacy distilled preserve/frontier proposals | Keep | Cheap discrete-choice hints inside `R_exact` and still part of the ungated default path |
| `Plain` and `ArbPrimed` operator families | Keep | They are the simplified operator-level default search |

### Keep only as temporary safety net

| Idea | Status | Why |
|---|---|---|
| Staged fallback | Temporary | Still preserves the committed EV frontier on the hard heterogeneous case, but is too expensive to keep forever |

### Teacher-only / reject for online default

| Idea | Status | Why |
|---|---|---|
| Larger online `2^K` preserve enumeration (`K > 4`) | Reject | Improves EV, but runtime cost is too high and still does not fully close the gap |
| Positive root-arb seed for preserve discovery | Reject | No measurable benefit on the hard case |
| Staged-action churn seeding | Reject | Still misses the staged winner |
| Promoting any new preserve/frontier proposal family to the default path without teacher parity | Reject | Keep legacy proposals; leave V2 gated until the hard cases clear |
| Late arb-correction tail in the default path | Reject | Removed after gas-aware pruning; no committed-benchmark lift beyond the operator core |
| Online per-variant exact-subset preserve proposals | Reject | Recovers most of the residual hard-case gap, but runtime cost is too high for the remaining dust-sized EV |
| Preserve local search / pair-add / pairwise probes | Reject | Added complexity without moving the hard case |
| Deeper generic frontier branching | Defer | No strong evidence that it is the binding constraint now |

## Ordering and sequencing learnings

- No static ordering is globally dominant.
- Direct-only cases do not justify mixed or arb-heavy search.
- The hard heterogeneous case's staged winner uses `arb_first`.
- The small bundle mixed case benefited from cyclic late-arb correction.

Implication:

- ordering should be state-triggered
- do not hardcode a single global phase order as the "ultimate" solver

## External review synthesis

External reviews from Gemini and Claude on 2026-03-08 agreed on the main conclusion:

- online EV is near saturation
- the waterfall is not the bottleneck
- the remaining gap is discrete choice recovery, mainly preserve/frontier choice
- more online brute force is the wrong direction
- the only credible remaining idea class is teacher-driven proposal generation followed by simplification

Critique of those reviews:

- Gemini was directionally strongest on the engineering conclusion: stop adding online search layers and distill the remaining preserve/frontier signal into a cheap runtime heuristic.
- Claude was right that the remaining structure is low-dimensional, but too optimistic that a flat small exact online loop is already sufficient. The heterogeneous hard case still shows path-dependent preserve discovery.
- Claude's suggestion to switch the main comparator to net-EV/gas-adjusted scoring is reasonable as an experimental metric, but not as the default objective while the benchmark target remains raw EV.

Decision:

- treat online EV optimization as effectively exhausted except for teacher-distilled preserve/frontier proposals
- do not add new online search dimensions such as larger preserve enumeration, more phase variants, or deeper branching
- the next worthwhile online change must be a tiny deterministic proposal heuristic learned from staged winners
- after that proposal heuristic reaches parity on the hard cases, remove the staged fallback and shift focus to speed, robustness, and gas-aware execution quality
- until then, default-path work is pruning and execution-quality work, not new solver search

## Best next ideas if EV work resumes

Only resume online EV work from this list:

1. Teacher-driven preserve-set proposal heuristic.
   - Learn a few whole preserve-set proposals from staged winners.
   - Evaluate those proposals directly inside `R_exact`.

2. Variant-aware preserve proposal generation.
   - Use `arb_first` and `no_arb` seed plans to propose preserve sets.
   - Do not recompute full per-variant exact subsets online unless evidence forces it.

3. Remove staged fallback after proposal parity.
   - Once the proposal heuristic matches staged on the hard cases, delete the fallback and remeasure release performance.

## Current stop condition status

The final simplification stop condition is not reached yet:

- benchmark-layer gas replay works on the committed benchmark fixture
- the default solver still has the staged fallback in the hot path
- the staged fallback still passes the keep rule on committed mixed cases
- fallback removal would materially improve runtime, but it would also give up meaningful benchmark EV/net-EV on the current committed mixed fixture

So the correct status is:

- EV-search expansion: stopped
- complexity pruning: mostly done
- final fallback removal: deferred until a cheaper preserve/frontier proposal heuristic can recover the same committed mixed frontier

## Stop conditions

Do not add another online search layer unless at least one of these is true:

- it improves committed mixed EV by a meaningful amount, not dust
- it materially reduces action count at equal EV
- it materially reduces runtime while preserving EV

Practical stop rule:

- do not accept roughly `1 s` of release runtime to buy a sub-basis-point EV gain
- if a new idea only moves EV by dust on the benchmark fixture, prefer simplification over retention
