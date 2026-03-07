# Portfolio Rebalancing

Canonical off-chain algorithm behavior lives in `docs/waterfall.md`.

This document is the portfolio module reference: APIs, execution shapes, diagnostics, and test map.

## Scope

Primary implementation paths:

- `src/portfolio/core/rebalancer.rs`
- `src/portfolio/core/waterfall.rs`
- `src/portfolio/core/planning.rs`
- `src/portfolio/core/solver.rs`
- `src/portfolio/core/trading.rs`
- `src/portfolio/core/sim.rs`

## Public entry points

```rust
pub fn rebalance(...) -> Vec<Action>
pub fn rebalance_with_mode(..., mode: RebalanceMode) -> Vec<Action>
pub fn rebalance_with_gas(..., mode: RebalanceMode, gas: &GasAssumptions) -> Vec<Action>
pub fn rebalance_with_gas_pricing(
    ...,
    mode: RebalanceMode,
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
) -> Vec<Action>
```

Mode summary:

- `RebalanceMode::Full`: prediction-driven rebalance flow.
- `RebalanceMode::ArbOnly`: complete-set arb only.

Arb-only behavior is specified in `docs/waterfall.md` under `Arb-Only Mode`.

## Action model

```rust
enum Action {
    Mint { contract_1, contract_2, amount, target_market },
    Merge { contract_1, contract_2, amount, source_market },
    Buy { market_name, amount, cost },
    Sell { market_name, amount, proceeds },
}
```

Route/action shapes emitted by planner:

- `DirectBuy`
- `DirectSell`
- `DirectMerge`
- `MintSell` (`Mint -> Sell+`)
- `BuyMerge` (`Buy+ -> Merge`)

No flash-loan actions are emitted.

## Runtime inputs

Planner inputs are built from live on-chain state and optional wallet balances:

1. `slot0` snapshots (`src/pools/rpc.rs`)
2. wallet balances if `WALLET` is set, otherwise synthetic starting cash
3. gas assumptions (`src/execution/gas.rs`)
4. mode (`full` or `arb_only`)

## Off-Chain Full Solver

Default full-mode planning is now operator-based:

- `R_exact(state)` is the exact online rebalance operator over the chosen discrete block:
  - first-frontier family in `{default, direct, mint}`
  - preserve subset over a capped churn universe
- `A(state)` is the existing arb-only operator on the current state.
- The runtime evaluates two bounded whole-plan families:
  - `Plain`: `R_exact`, then optional late `A -> R_exact`
  - `ArbPrimed`: positive `A`, then `R_exact`, then optional late `A -> R_exact`
- Candidate ranking is raw EV first, then fewer actions, then stable family/frontier/preserve ordering.

`R_exact` keeps the existing waterfall, recycle, polish, and cleanup logic unchanged. The exactness is only over the online discrete block around that continuous core.

Preserve-universe construction:

- start from the three no-preserve frontier seeds (`default`, forced `direct`, forced `mint`)
- extract sell-then-rebuy churn candidates from those seed action streams
- run one-step singleton-preserve probes from the same state to expand that churn universe once
- aggregate by max churn amount, then max sold amount, then stable market order
- cap the online preserve universe at `K = 4`
- enumerate every preserve subset across every frontier family from fresh state
- evaluate that grid in parallel, then reduce deterministically

Practical dominance fallback:

- the old staged meta-solver path is still compiled as a reference implementation
- the default runtime compares the new operator-based winner against that staged-reference plan under the same raw-EV comparator
- whichever whole-plan result is better is returned
- this keeps the new solver live while preserving the previously committed EV frontier on cases the flat exact operator does not yet subsume

Compatibility note:

- `RebalanceFlags.enable_ev_guarded_greedy_churn_pruning` remains in the public API for compatibility, but it no longer changes default full-mode behavior

Operational diagnostics:

- exact rebalance tracing reports family label, exact-rebalance call count, candidate-evaluation count, preserve-universe size, chosen frontier family, chosen preserve-set size, EV, and action count
- final selection tracing reports chosen family, chosen frontier family, preserve-set size, EV/actions, arb-operator evaluation count, arb-correction count, and whether the arb-primed root was taken
- an ignored test helper prints per-family breakdown on the heterogeneous 98-outcome fixture for targeted debugging

## Diagnostics and operator tools

- Execution submission and strict fail-closed gates: `docs/execution_submission.md`
- First-group preview diagnostics (`plan_preview` + runtime preview output): `docs/execution_submission.md`
- Cross-approach policy thresholds: `docs/rebalancer_approaches_playbook.md`
- Off-chain EV optimization memory and keep/cut decisions: `docs/offchain_ev_optimization_log.md`

## Balance fetching and local cache

Balance helpers live in `src/pools/cache.rs` and `src/pools/rpc.rs`.

- `fetch_balances(...)` fetches sUSD + outcome balances via multicall
- `save_balance_cache(...)` writes wallet-scoped cached snapshot
- `load_balance_cache(...)` reads cache with staleness/wallet checks

## Test map

Portfolio tests are split across:

- `src/portfolio/tests.rs` (fixtures + early deterministic coverage)
- `src/portfolio/tests/fuzz_rebalance.rs`
- `src/portfolio/tests/oracle.rs`
- `src/portfolio/tests/execution.rs`

Key classes covered:

- direct-only oracle parity and near-optimality
- mixed-route feasibility and ordering behavior
- EV non-regression fuzz checks
- execution/replay consistency invariants

## Performance

`test_rebalance_perf_full_l1` benchmarks full rebalance across the 98 tradeable L1 outcomes.

The current exact-family plus staged-fallback path is materially heavier than the previous bounded meta-solver. The release benchmark should be rerun after any further solver simplification or after removing the staged fallback; the older single-digit-millisecond figure is no longer a reliable description of the default path.

Run with:

```bash
cargo test --release test_rebalance_perf_full_l1 -- --nocapture
```

## Related docs

- `docs/waterfall.md`: canonical off-chain algorithm spec
- `docs/model.md`: math derivations
- `docs/gas_model.md`: gas threshold model
- `docs/rebalancer.md`: on-chain direct solver
