# Off-Chain Mixed State Solver

## Summary

The off-chain mixed compiler now has a native state-space path instead of relying on trace pruning alone:

- `constant_l_mixed` is the baseline compact solver.
- `staged_constant_l_2` remains the bounded runtime reference extension, but teacher work is now guiding whether it should survive.
- compact mixed candidates carry internal certificates with their active set, `pi`, mint amount, budget usage, delta residuals, and modeled economics.
- teacher-only `K=1` and `K=2` oracles now live on top of the same fixed-active Rust solver rather than a second mathematical model.

This is an off-chain solver family, not an on-chain fallback. The Rust planner solves the mixed frontier directly and competes under the normal off-chain net-EV comparator.

## `K=1`: `constant_l_mixed`

The `K=1` solver now has two layers.

1. A fixed-active-set equilibrium solve:
   - solve the frontier profitability `pi` with a continuous outer bisection
   - for each trial `pi`, solve the mint amount `m` from the inactive-leg price-delta equation
   - score feasibility with the exact sequential cash identity for `Mint -> inactive Sell -> active Buy`:
     - required starting cash = `max(m, direct_cost + m - sell_proceeds)`
2. A runtime active-set search over that fixed-active oracle:
   - direct-profitability prefixes
   - all profitable singletons
   - for small profitable universes, analytic/direct-only support seeds
   - add/drop/one-swap local search from the best seed

For the chosen active set, the solver synthesizes the compact program:
   - `Mint(m)` if `m > 0`
   - direct `Sell`s on non-active outcomes up to the frontier target
   - direct `Buy`s on active outcomes up to the same frontier target

The emitted plan is scored like every other off-chain candidate: estimated fee, estimated net EV, action count, then stable tie-breaks.

The fixed-active solve is memoized per search pass by:

- solver-state fingerprint
- budget cap
- active mask

This keeps runtime and teacher searches on the same solution cache rather than recomputing identical `K=1` subproblems.

## K=1 Oracle And Certificates

In test mode there are now two exact `K=1` teachers:

- `K1Oracle` for `n <= 12`
- `K1MediumOracle` for `n <= 13`

Both exhaustively evaluate active subsets under the same constant-`L` equations and the same fee-aware comparator used by runtime.

This has two uses:

1. certify that the Rust `K=1` solver matches the model-global optimum on small cases
2. detect when direct-profitability prefixes are not sufficient and force the runtime search to broaden its active-set seeds

For larger universes there is also a teacher-only `best_known` comparison that reruns the same `K=1` search without the runtime profitable-universe caps. It is diagnostic only: not an oracle and not an upper bound.

Each compact mixed candidate carries a certificate with:

- `active_mask`
- `active_set_size`
- `pi`
- `mint_amount`
- `direct_cost`
- `sell_proceeds`
- `mint_net_cost`
- `budget_used`
- `budget_residual`
- `delta_target`
- `delta_realized`
- `raw_ev`
- `estimated_fee_susd`
- `estimated_net_ev`

Teacher diagnostics now report:

- `runtime_k1_gap_net_ev`
- `runtime_k1_gap_raw_ev`
- `oracle_best_is_direct_prefix`
- `oracle_best_active_set_size`

## `K=2`: `staged_constant_l_2`

The current runtime staged extension still only runs when all of these hold:

- `constant_l_mixed` produced a feasible candidate
- the rich trace still exceeds the best compact candidate on raw EV by more than `1e-6` sUSD
- the rich trace contains more than one profitability step

Stage 1 uses a deterministic spend-cap grid over the starting cash:

- `{0, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1}`

For each fraction:

1. Solve a capped `constant_l_mixed` stage on the starting state.
2. Apply that stage to simulated state.
3. Solve a second full-budget `constant_l_mixed` stage on the residual state.

One refinement pass is then run around the best interior fraction by inserting the two neighboring midpoints. No random search and no generic optimizer are used.

This runtime `K=2` path should now be treated as provisional. The new teacher-first rule is:

- use exact `K=1` teachers to certify the single-stage frontier first
- use the `K=2` teacher to determine whether a staged redesign is justified
- do not add more runtime mixed-search dimensions without teacher evidence

The teacher-only `K2Oracle` is exact only on small mixed cases (`n <= 8`). It enumerates stage-1 candidates from the exact `K=1` teacher family, applies each to simulated state, solves exact `K=1` again on the residual state, and compares the concatenated two-stage plan under the same fee-aware net-EV comparator.

## Why This Is Not A Fallback

The compiler does not call Solidity or defer to an on-chain oracle. It owns the same one-stage equilibrium off-chain, exposes its certificate (`active_mask`, `pi`, `mint_amount`, budget usage, delta residuals), and then extends that one-stage solve with a bounded staged composition when the richer off-chain trajectory still contains extra raw EV.

That gives the off-chain planner a clean convergence target:

- when the one-stage constant-`L` model is best, the off-chain result should match the on-chain mixed solution
- when chronology still matters, the off-chain solver can move beyond the one-stage limit in a controlled staged way

The current remaining limitation is not the `K=1` state solve itself. It is that:

- large-`n` active-set search is still capped online
- large-`n` teacher comparisons are best-known, not exact
- runtime `K=2` is heuristic until the teacher proves it is worth keeping
