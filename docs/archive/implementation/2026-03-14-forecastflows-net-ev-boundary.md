# ForecastFlows Net-EV Boundary

Date: 2026-03-14

Superseded by `docs/archive/implementation/2026-03-14-forecastflows-execution-gas-cutover.md`.

## Decision

Keep the current architecture:

- ForecastFlows.jl is the route generator.
- `deep_trading` is the exact execution-cost and final net-EV oracle.

We are not planning to make ForecastFlows.jl own the full `deep_trading` gas model, tx-packing
model, or chunked submission economics.

## Why

The current split is clean and benchmark-winning:

- the upstream public `UniV3` facade now matches the intended outcome-price contract
- the Rust translator now sends contiguous liquidity ladders instead of a single active band
- the Rust side replays ForecastFlows candidates through the same packed-program fee model used for native plans
- local net-EV polish can prune marginal profitability steps and coupled route groups without any extra Julia calls

On the committed local EV benchmark suite, this is enough for ForecastFlows to match or beat the
native waterfall benchmark under the shared execution-cost model.

## What The Upstream Gas Model Is

ForecastFlows.jl exposes `PredictionMarketFixedGasModel` as a simple edge-activation penalty model.

That model is useful as a generic sparsity hint, but it is not an exact execution compiler:

- one fixed cost per active market edge
- one fixed cost for using split/merge in mixed mode
- no packed transaction chunking
- no calldata byte model
- no shared tx-envelope savings across many actions
- no `deep_trading`-specific route replay semantics

That is acceptable for a reusable solver library. It is not sufficient to replace the exact
execution-cost oracle in `deep_trading`.

## Non-Goal

We are not porting the `deep_trading` execution compiler into ForecastFlows.jl.

That would duplicate chain-specific logic across Rust and Julia, increase maintenance cost, and
push the solver library toward `deep_trading`-specific behavior.

## Current Boundary

ForecastFlows.jl should continue to own:

- gross-EV route search
- certification
- stable market/problem abstractions
- optional light activation-penalty solving

`deep_trading` should continue to own:

- exact pool replay
- execution-group validation
- tx packing under the `< 40_000_000` L2 gas cap
- calldata and L1 data-fee modeling
- final net-EV comparison across solver families

## When To Revisit

Do not deepen the Julia-side gas model unless one of these becomes true:

1. Multiple downstream integrators want a richer generic activation-penalty interface.
2. The current Rust-side net-EV polish stops being enough on real benchmark or production cases.
3. We decide not to pursue a Rust rewrite and want the Julia library itself to become the
   production net-EV optimizer.

If a Rust rewrite becomes a real project, further Julia-side exact net-EV work should be postponed.
In that world, the clean target is a single Rust stack that owns both route optimization and exact
execution economics.
