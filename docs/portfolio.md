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

## Diagnostics and operator tools

- Execution submission and strict fail-closed gates: `docs/execution_submission.md`
- First-group preview diagnostics (`plan_preview` + runtime preview output): `docs/execution_submission.md`
- Cross-approach policy thresholds: `docs/rebalancer_approaches_playbook.md`

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

Typical release benchmark figure from the committed harness:

- ~3.6ms per call (98 outcomes)

Run with:

```bash
cargo test --release test_rebalance_perf_full_l1 -- --nocapture
```

## Related docs

- `docs/waterfall.md`: canonical off-chain algorithm spec
- `docs/model.md`: math derivations
- `docs/gas_model.md`: gas threshold model
- `docs/rebalancer.md`: on-chain direct solver
