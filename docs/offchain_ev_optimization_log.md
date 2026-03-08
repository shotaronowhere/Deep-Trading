# Off-Chain EV Optimization Log

Last updated: 2026-03-08

## Purpose

This document is the permanent memory for off-chain EV-optimization work:

- what ideas were explored
- what benchmark movement they produced
- which complexity layers are worth keeping
- which ideas are teacher-only or rejected for online use

Use this before adding any new off-chain search dimension.

## Current benchmark frontier

Sources:

- current fixture: `test/fixtures/rebalancer_ab_expected.json`
- pre-operator legacy snapshot: commit `093e4b9`, same fixture path
- archived on-chain mixed comparison: `docs/archive/rebalancer/rebalancer_vs_rebalancer_mixed_benchmark_2026-03-03.md`

### Current off-chain mixed vs legacy snapshot (`093e4b9`)

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

### Current off-chain mixed vs current on-chain exact

| Case | EV delta (off-chain mixed - on-chain exact) | EV delta % |
|---|---:|---:|
| `two_pool_single_tick_direct_only` | `+0.000000000000015206` | `+0.000000%` |
| `ninety_eight_outcome_multitick_direct_only` | `-0.000000000000534821` | `-0.000000%` |
| `small_bundle_mixed_case` | `+0.015610289881024388` | `+0.015590%` |
| `legacy_holdings_direct_only_case` | `-0.000000001853588459` | `-0.000000%` |
| `mixed_route_favorable_synthetic_case` | `+0.086792931624521758` | `+0.086774%` |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `+0.122026242587661667` | `+0.081211%` |

Interpretation:

- Off-chain mixed is already saturated on direct-only cases: parity up to dust.
- Off-chain mixed still materially beats on-chain exact direct on the genuinely mixed cases.

### Current off-chain mixed vs archived on-chain mixed constant-`L`

| Case | EV delta (off-chain mixed - on-chain mixed) |
|---|---:|
| `two_pool_single_tick_direct_only` | `+0.000000000000015206` |
| `ninety_eight_outcome_multitick_direct_only` | `-0.000000000000534821` |
| `small_bundle_mixed_case` | `+0.000000000000006353` |
| `mixed_route_favorable_synthetic_case` | `-0.000022249852264816` |
| `heterogeneous_ninety_eight_outcome_l1_like_case` | `-0.000007408968515453` |
| `legacy_holdings_direct_only_case` | `-0.000000000000000574` |

Interpretation:

- Current off-chain mixed is effectively at parity with the archived on-chain mixed benchmark.
- Remaining benchmark headroom versus on-chain mixed is now dust, not a large frontier gap.

## Runtime measurements

Measured on 2026-03-08 with current code:

- `cargo test --release test_rebalance_perf_full_l1 -- --nocapture`
  - full stress case: about `993 ms/call`
- `cargo test --release test_rebalance_perf_full_l1_with_gas_pricing -- --nocapture`
  - gas-gated stress case: about `34.7 ms/call`

Interpretation:

- The current default full solver is too heavy to justify chasing tiny residual EV gains with more online search.
- Gas-aware pruning changes the runtime regime completely; online complexity should be judged under realistic gated conditions, not only the zero-threshold stress case.

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
- this means the plan succeeded in turning the last online EV idea into code, and the answer was negative: the idea is not yet strong enough to replace the fallback

Engineering decision:

- keep the staged fallback enabled
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

## Explored ideas and status

### Keep online

| Idea | Status | Why |
|---|---|---|
| Mixed-route capability | Keep | Real EV uplift on the mixed cases |
| `R_exact` over `{default, direct, mint}` frontier families | Keep | Cheap exactness over the right continuous core |
| Small preserve-set search | Keep | Preserve choice is a real EV lever |
| Tiny distilled preserve/frontier proposals | Keep, but only with fallback | Non-regressive versus flat `K = 4`, but not yet sufficient to replace staged dominance guard |
| Conditional cyclic late-arb | Keep | Closed the small-bundle late-arb gap and reduced action count there |

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

## Stop conditions

Do not add another online search layer unless at least one of these is true:

- it improves committed mixed EV by a meaningful amount, not dust
- it materially reduces action count at equal EV
- it materially reduces runtime while preserving EV

Practical stop rule:

- do not accept roughly `1 s` of release runtime to buy a sub-basis-point EV gain
- if a new idea only moves EV by dust on the benchmark fixture, prefer simplification over retention
