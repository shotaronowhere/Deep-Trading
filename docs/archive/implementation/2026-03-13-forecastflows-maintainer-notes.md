# ForecastFlows Maintainer Notes

Date: 2026-03-13

This note is written as an upstream-ready handoff from the Rust driver integration in
`deep_trading`. It is meant to be copied into upstream issues with minimal editing.

Historical note: this document captures the 2026-03-13 upstream handoff state. The current
`deep_trading` architecture decision about net-EV ownership is documented separately in
`docs/archive/implementation/2026-03-14-forecastflows-net-ev-boundary.md`.

## What Works Well In ForecastFlows

ForecastFlows looks structurally solid as a solver core.

- The prediction-market facade has a clear typed boundary.
- Certification is exposed explicitly instead of being hidden in logs.
- `PredictionMarketWorkspace` already exists for reuse-heavy solve loops.
- The package has meaningful protocol and benchmark coverage.
- The public `UniV3MarketSpec` shape is a good, stable driver contract.

The main concerns below are about the serving boundary and latency model, not the math core.

## Solver-Side Vs Driver-Side

What is primarily solver-side:

- The stateless worker model currently serves one request at a time per process.
- `compare_prediction_market_families` runs two independent solves instead of reusing a workspace.
- The protocol does not yet expose enough branch-level diagnostics for driver-side attribution.
- Trade outputs are numerically close to executable, but the executable rounding contract is implicit.

What is primarily driver-side:

- We were initially underusing the `UniV3` facade by collapsing markets to one active band.
- We still own replay, gas modeling, net-EV ranking, and fail-open behavior locally.
- Our latency profile depends heavily on process launch policy, Julia threading, and sysimage discipline.
- We do not want to push our `deep_trading`-specific execution compiler or gas replay model into ForecastFlows.jl.

## Concrete Requests For ForecastFlows Maintainers

Highest-value upstream requests:

1. Add a workspace-aware compare path.
   A `compare_prediction_market_families!` or equivalent sessionful entrypoint should reuse normalization, buffers, and seeds across the direct and mixed branches.

2. Expose richer worker diagnostics.
   The NDJSON compare response should make branch-level timing and termination metadata easy to consume without reverse-engineering logs. Today `solver_time_sec` is useful, but we would benefit from more explicit per-branch diagnostics.

3. Define an executable rounding contract for trades.
   The Rust driver still needs tolerance-based acceptance around tiny sell overshoots and dust. It would be better if the solver or facade snapped outputs to a documented executable tolerance.

4. Consider a stateful worker protocol in addition to stateless NDJSON.
   The current protocol is safe and simple, but operationally expensive for production latency. A sessionful mode would make workspace reuse and preallocation much easier.

## Repro Context From Our EV Benchmark Suite

Our committed net-EV benchmark lane runs explicit ForecastFlows against the same Rust replay, fee,
and ranking rules used for native plans.

Relevant workloads:

- `heterogeneous_ninety_eight_outcome_l1_like_case`
- `ninety_eight_outcome_multitick_direct_only`
- smaller single-tick and mixed-route synthetic fixtures

The current driver now records:

- worker roundtrip time
- per-branch `status`
- per-branch `solver_time_sec`
- whether Rust dropped a certified branch at response validation or replay time
- whether replay tolerance clamping was needed

This makes it much easier to separate solver latency from driver overhead on our side.

What we verified locally in `deep_trading` after instrumenting the Rust side more deeply:

- the `PredictionMarketProblem` requests we send for the committed single-tick EV fixtures match the intended fair values, current prices, and hard price boundaries from the fixture data
- Rust mixed-route replay now mirrors the staged `replay_desired_route` semantics from the Julia benchmark harness rather than bulk-minting or bulk-merging before all trades
- even after that replay fix, the public `compare_prediction_market_families` path still returns economically weak certified candidates on several benchmark cases:
  - `two_pool_single_tick_direct_only` returns a certified direct no-op
  - `small_bundle_mixed_case` returns a certified mixed route that mints and sells but does not buy, losing EV versus the native Rust waterfall path
  - `mixed_route_favorable_synthetic_case` shows the same mint-and-sell-only pattern

The remaining net-EV polishing on our side now happens in Rust by replaying and step-pruning
ForecastFlows routes under the same local execution-cost model used for native plans. We are not
asking ForecastFlows.jl to absorb that `deep_trading`-specific gas simulator.

## Suggested Upstream API Additions

- `compare_prediction_market_families!(workspace, problem; ...)`
- A worker/session API that can hold a `PredictionMarketWorkspace` across requests
- Explicit per-branch diagnostics in compare responses, for example:
  - termination reason
  - certify status/message
  - workspace reuse indicator
  - branch-level latency breakdown
- A documented trade rounding policy for executable outputs

## Bottom Line

ForecastFlows does not look brittle as a solver core. The most important next steps are boundary,
latency, and service-shape improvements that let downstream drivers use more of the solver
efficiently and with less implicit numerical guesswork. In `deep_trading`, we now treat
ForecastFlows as the gross-EV route generator and keep execution-cost scoring local. If a future
upstream extension is desirable, the clean direction would be a generic sparsity or activation
penalty hint rather than a `deep_trading`-specific gas model.
