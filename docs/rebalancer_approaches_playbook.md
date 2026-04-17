# Rebalancer Approaches Playbook

This is the canonical strategy-selection document for rebalancer approaches in
this repo.

Companion references:

- [rebalancer.md](rebalancer.md): current on-chain direct solver behavior.
- [rebalancer_solver_postmortem_2026-03-07.md](rebalancer_solver_postmortem_2026-03-07.md): why the bounded/adaptive direct routes were removed.
- [rebalancer_mixed.md](rebalancer_mixed.md): current mixed-route experiment.
- [rebalancing_mechanism_design_review.md](rebalancing_mechanism_design_review.md): broader mechanism-design critique and research ideas.

## Current implementation map

### Off-chain planner + strict executor

Core Rust files:

- `src/portfolio/core/rebalancer.rs`
- `src/portfolio/core/waterfall.rs`
- `src/portfolio/core/planning.rs`
- `src/execution/bounds.rs`
- `src/execution/grouping.rs`
- `src/execution/tx_builder.rs`

Properties:

- rich objective surface
- strict submission bounds and stale-plan failure
- subgroup execution with replanning between transactions
- still exposed to latency and inclusion risk

### On-chain direct solver

Entrypoints:

- `rebalance`
- `rebalanceExact`
- `rebalanceAndArb`
- `rebalanceAndArbWithFloors`
- `rebalanceAndArbExact`
- `rebalanceAndArbExactWithFloors`

Properties:

- atomic solve + execute
- per-pool price limits on every swap
- default direct route is simple multi-pass constant-`L`
- exact route keeps fully explicit multi-tick semantics
- caller-supplied collateral floors remain as the coarse churn gate for arb and recycle

### On-chain mixed solver

Entrypoints:

- `rebalanceMixed`
- `rebalanceMixedConstantL`

Properties:

- still experimental
- deterministic fallback via `MixedSolveFallback(reasonCode)`
- materially higher gas in committed benchmarks

## Design rules from the March 7, 2026 cleanup

These are the rules that came out of the bounded/adaptive solver experiment and
should be treated as hard-earned design constraints:

1. Do not add an intermediate on-chain direct solver unless it materially beats
   `rebalance` on realistic 98-market crossing fixtures.
2. Fractions of a percent of EV are not enough to justify a much more complex
   on-chain planner.
3. If `rebalanceExact` already fits under the gas cap on the relevant state
   family, optimize exact before inventing a new approximate route.
4. No new solver should ship without a benchmark table for:
   - a deep-crossing two-pool fixture,
   - the synthetic 98-outcome crossing fixture,
   - the realistic seeded 98-outcome crossing fixture.
5. A new approximate route must not introduce catastrophic EV regressions on any
   realistic crossing benchmark, even if its median gas number looks good.

The detailed evidence is recorded in
[rebalancer_solver_postmortem_2026-03-07.md](rebalancer_solver_postmortem_2026-03-07.md).

## Current frontier

| Approach | Strengths | Weaknesses | Current role |
|---|---|---|---|
| Off-chain strict executor | Richest route surface, can exploit mixed opportunities, good when submission risk is low | Sensitive to staleness, more moving parts, needs conservative bounds | Conditional use when latency/slippage regime is acceptable |
| On-chain direct baseline (`rebalance`) | Simple, atomic, predictable, low enough gas for 98-market production use | Constant-`L` is only an approximation across tick crossings | Default production direct route |
| On-chain exact (`rebalanceExact`) | Best on-chain EV fidelity, exact multi-tick pricing, still under 40M gas on current 98-market realistic benchmark | Highest planning gas, explicit revert-on-cap behavior | High-value direct route and benchmark ceiling |
| On-chain mixed (`rebalanceMixedConstantL`) | Can capture mixed-route EV in favorable states | High gas premium, fallback sensitivity, state-dependent value | Experimental only |

## Route policy

### Direct on-chain policy

Default:

- Use `rebalance`.

Escalate to `rebalanceExact` only when both conditions hold:

1. The benchmarked state family shows a material EV gain over `rebalance`.
2. `rebalanceExact` remains below the 40M gas limit on that state family.

Practical interpretation:

- Basis-point or low-tenths-of-a-percent gains do not justify a new direct route.
- Large gains on realistic crossing-heavy states do justify exact.
- The current data supports a two-route direct frontier, not a three- or four-route one.

### Mixed-route policy

Default:

- Do not run mixed on-chain by default.

Enable mixed only when:

1. its incremental EV over direct is clearly positive after gas, and
2. fallback behavior is stable across recent attempts.

### Off-chain execution policy

Default:

- Prefer on-chain direct execution in competitive or adversarial windows.

Permit off-chain strict execution only when:

1. quote latency is short enough that conservative bounds still leave profit,
2. the first executable subgroup has meaningful gas-adjusted margin, and
3. recent stale-plan aborts are rare.

## Validation protocol for new solver ideas

Every new direct-solver idea must answer all of these questions before it is
kept:

1. Does it beat `rebalance` on realistic seeded 98-outcome multi-tick fixtures?
2. Is the gain large enough to justify the added planner complexity?
3. Does it stay under the 40M gas cap?
4. Does it avoid severe EV regressions on any benchmark state?
5. Does it beat simply putting the effort into `rebalanceExact` instead?

Required output for any new solver proposal:

- gas and EV for the deep-crossing two-pool fixture
- gas and EV for synthetic 98-outcome multi-tick
- gas and EV for realistic seeded 98-outcome multi-tick
- a one-paragraph complexity argument explaining why the added code is worth it

If the measured gain is only a small fraction of a percent, reject the new route
unless it also reduces code size or meaningfully simplifies operations.

## Current benchmark takeaway

After the cleanup, the surviving on-chain direct frontier is:

- `rebalance`: simple constant-`L` baseline
- `rebalanceExact`: expensive but genuinely higher-fidelity route

That is the clean mechanism-design answer the benchmarks gave us. The middle
layer of bounded/adaptive heuristics did not pay rent.
