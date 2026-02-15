# Architecture

## Overview

Rust prediction market portfolio balancing bot for Seer PM on Optimism. Uses compile-time code generation to embed market data.

## Data Flow

```
API (deep.seer.pm)
       |
       v
  lib.rs tests (cargo test test_prepare_*)
       |
       v
  JSON files (markets_data_*.json)
       |
       v
  cargo run --bin regenerate
       |
       v
  Generated Rust (src/markets.rs, src/predictions.rs)
       |
       v
  build.rs staleness guard (compile time)
       |
       v
  Runtime binary (main.rs)
```

## Market Types

| Type | Struct | Pools | Quote Token | Key Fields |
|------|--------|-------|-------------|------------|
| L1 | `MarketData` | Single `Option<Pool>` | Hardcoded | `outcome_token` |
| L2 | `MarketDataL2` | Multiple `&[Pool]` | From `parentWrappedOutcome()` | `outcome_token` |
| Originality | `MarketDataOriginality` | Two `&[Pool]` (up/down) | Hardcoded | `up_token`, `down_token` |

## Key Files

- **build.rs**: Validates generated files exist and are not stale versus CSV/JSON inputs; fails fast with a regenerate instruction when stale/missing.
- **src/bin/regenerate.rs**: Orchestration entrypoint for manual codegen (`cargo run --bin regenerate`)
- **src/bin/regenerate/predictions.rs**: CSV parsing/validation and `predictions.rs` generation
- **src/bin/regenerate/markets.rs**: JSON parsing, multicall address resolution, and `markets.rs` generation
- **src/bin/regenerate/common.rs**: Shared codegen helper(s) (e.g., formatting generated outputs)
- **src/lib.rs**: API client, fetches market data from Seer PM endpoints
- **src/main.rs**: WebSocket connection to Optimism RPC
- **src/markets.rs**: Generated static market data arrays
- **src/predictions.rs**: Generated prediction weights from CSVs
- **src/pools.rs**: Facade/re-exports for pool utilities
- **src/pools/pricing.rs**: Price conversions, prediction lookup map, and shared numeric helpers
- **src/pools/swap.rs**: Uniswap V3 swap-step simulation helpers
- **src/pools/analytics.rs**: Profitability/depth analytics
- **src/pools/rpc.rs**: On-chain pool + balance multicall queries
- **src/pools/cache.rs**: Local balance cache serialization and staleness checks
- **src/execution/bounds.rs**: Group planning, strict gating, and execution-plan orchestration
- **src/execution/edge.rs**: Group cashflow/EV edge derivation helpers
- **src/execution/batch_bounds.rs**: Aggregate `sell(min)` / `buy(max)` bound derivation + plan stamping
- **src/execution/gas.rs**: L2/L1 gas estimate model + cached Optimism L1 fee-per-byte hydration
- **src/execution/grouping.rs**: Action grouping and flash-bracket shape validation
- **src/portfolio/mod.rs**: Portfolio module entrypoint exporting `Action` and `rebalance`
- **src/portfolio/core/mod.rs**: Portfolio core aggregation module
- **src/portfolio/core/sim.rs**: Pool simulation primitives and route-agnostic math helpers
- **src/portfolio/core/planning.rs**: Pure route planning and cost modeling helpers
- **src/portfolio/core/solver.rs**: Numerical solvers (mint Newton solve and budget-exhaustion profitability solve)
- **src/portfolio/core/trading.rs**: Trade/plan execution, merge/mint helpers, and inventory accounting (`ExecutionState` centralizes mutable execution state and execution methods)
- **src/portfolio/core/waterfall.rs**: Waterfall allocation strategy and active-set profitability equalization loop
- **src/portfolio/core/rebalancer.rs**: Rebalance phase orchestration and phase-specific inventory/budget flows (`RebalanceContext` handles setup/validation)
- **src/portfolio/tests.rs**: Portfolio test root (shared fixtures + early deterministic tests)
- **src/portfolio/tests/fuzz_rebalance.rs**: Fuzz and full/partial rebalance regression tests
- **src/portfolio/tests/oracle.rs**: Oracle parity, phase behavior, and invariants tests
- **src/portfolio/tests/execution.rs**: Merge/sell execution, perf, and integration test helpers

## Pool Analytics (`src/pools.rs`)

### Price Conversion
Converts Uniswap V3 `sqrtPriceX96` to outcome token prices (18-decimal fixed point). Handles both token orderings via `is_token1_outcome`.

### Swap Simulation
`simulate_swap` / `simulate_buy` wrap `uniswap_v3_math::compute_swap_step` to simulate trades within a single tick range. Supports exact-input and exact-output.

### Profitability Analysis
`profitability()` compares prediction probabilities against current market prices for all L1 markets. For each market where prediction > market price:
- **Liquidity check**: Skips depth calculation if pool has zero liquidity
- **Tick depth**: Max outcome tokens buyable before hitting the tick boundary, and their quote token cost
- **Breakeven depth**: Max outcome tokens buyable before the price moves to match the prediction (profitability = 0), and their cost. Uses `prediction_to_sqrt_price_x96` to convert the prediction probability to a target sqrtPriceX96, clamped to the tick boundary.

### Multicall Batching
`fetch_all_slot0` batches `slot0()` calls across all pools using Multicall3, with configurable batch size (200). `fetch_balances` batches ERC20 `balanceOf` calls for sUSD + all 100 outcome tokens (including non-pooled outcomes). Balance caching via `save/load_balance_cache` persists to JSON with wallet validation and 5-minute staleness.

### Alternative Pricing
`price_long_simple_alt` computes the implied long price for an outcome by summing prices of all other outcomes: `1 - sum(others)`.

## External Dependencies

- RPC: Optimism (configured via `RPC` env var)
- APIs:
  - `https://deep.seer.pm/.netlify/functions/get-l1-markets-data`
  - `https://deep.seer.pm/.netlify/functions/get-l2-markets-data`
  - `https://deep.seer.pm/.netlify/functions/get-originality-markets-data`

## Build Process

1. Run `cargo run --bin regenerate`
2. Parse prediction CSVs and regenerate `src/predictions.rs`
3. Read JSON market data files
4. Batch `getPool()` calls via Multicall3 to resolve Uniswap V3 pool addresses
5. For L2: batch `parentWrappedOutcome()` to resolve quote tokens
6. Generate static Rust arrays in `src/markets.rs`
7. During compile, `build.rs` verifies generated outputs are present and newer than generator inputs; if not, the build fails closed.

## Constants

- `MULTICALL_BATCH_SIZE`: 16000 (sub-calls per multicall)
- `FEE_TIER`: 100 (0.01% Uniswap fee tier)
- `L1_QUOTE_TOKEN`: `0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0`
