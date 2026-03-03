# Rebalancer A/B Benchmark

This benchmark is the reference frame for comparing the off-chain Rust planner against the on-chain `Rebalancer.sol` contract on the same frozen pool snapshot.

## Goal

Measure two distinct effects separately:

1. **Stale-plan mismatch removal**
   - On-chain solving reads live state inside the transaction.
   - Off-chain solving reads a snapshot, then submits later.

2. **Objective gap**
   - `Rebalancer.sol` solves the direct-only waterfall objective.
   - The Rust planner can solve a richer mixed objective on the buy side (bundle-frontier direct vs mint).

## Benchmark Rules

For every case:

1. Freeze one identical pool snapshot.
2. Run the Rust planner on that snapshot.
3. Reset to the same snapshot.
4. Run `Rebalancer.rebalance(...)`.
5. Reset again.
6. Run `Rebalancer.rebalanceExact(...)`.
7. Mark all terminal portfolios with the same EV function.

Case geometry for both the Rust and Foundry harnesses is sourced from [rebalancer_ab_cases.json](/Users/shotaro/proj/deep_trading/test/fixtures/rebalancer_ab_cases.json). The 98-outcome synthetic case uses the parametric `uniform_*` fields in that fixture instead of code-local hardcoding.

Recommended EV mark:

`EV = cash + sum(prediction_i * holdings_i)`

Optional reporting:

- `EV_net = EV - gas_in_susd`
- conservative liquidation floor using executable sell bids

## Interpretation

Important caveats:

1. On-chain solving removes stale-plan mismatch, but it does **not** remove your own price impact.
2. On-chain solving still inherits the contract objective. If the contract does not model the mixed bundle route, it cannot recover that EV.
3. A higher off-chain EV on a frozen snapshot does **not** mean the off-chain path is always better live; it only measures objective quality on that state.
4. A higher on-chain EV in live trading may come from lower staleness, not from a better optimizer.

This benchmark is meant to make those tradeoffs visible, not to claim that one path universally dominates.

## Current Harness Notes

1. The direct-only parity checks use a small relative tolerance (`5e-6`) because the Rust replay model and the Solidity fixture router round differently.
2. The mixed-case fixture does not assert that the off-chain mixed path must dominate. The current experimental mixed planner can underperform the direct-only on-chain objective on some frozen fixtures, and the benchmark is intended to surface that rather than mask it.
3. The mixed-case Foundry check now pins the committed on-chain exact EV snapshot within a looser tolerance, so the known anomaly cannot silently drift without review.

## Current Finding

On the current committed synthetic fixtures:

1. Off-chain direct and on-chain `rebalanceExact()` are economically very close in the direct-only parity cases (all within a few parts per million of EV).
2. The current `small_bundle_mixed_case` shows the opposite of the original hypothesis: the off-chain mixed planner underperforms the on-chain solver on that frozen state.
3. In that fixture, `offchain_mixed_ev = 100096481776136945664` and `onchain_exact_ev = 100132739273228663690`, so the off-chain mixed path is lower by `36257497091718026` wei of EV (about `0.036%`).

This is a benchmark result, not a general theorem. It means the current experimental mixed planner and fixture assumptions still need review before we treat the mixed route as an EV improvement.
