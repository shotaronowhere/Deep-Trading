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
  build.rs (compile time)
       |
       v
  Generated Rust (src/markets.rs, src/predictions.rs)
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

- **build.rs**: Fetches pool addresses via Multicall3, generates `markets.rs` and `predictions.rs`
- **src/lib.rs**: API client, fetches market data from Seer PM endpoints
- **src/main.rs**: WebSocket connection to Optimism RPC
- **src/markets.rs**: Generated static market data arrays
- **src/predictions.rs**: Generated prediction weights from CSVs
- **src/pools.rs**: On-chain pool queries, trading analytics, balance fetching, and caching
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
`price_alt` computes the implied long price for an outcome by summing prices of all other outcomes: `1 - sum(others)`.

## External Dependencies

- RPC: Optimism (configured via `RPC` env var)
- APIs:
  - `https://deep.seer.pm/.netlify/functions/get-l1-markets-data`
  - `https://deep.seer.pm/.netlify/functions/get-l2-markets-data`
  - `https://deep.seer.pm/.netlify/functions/get-originality-markets-data`

## Build Process

1. Read JSON market data files
2. Batch `getPool()` calls via Multicall3 to resolve Uniswap V3 pool addresses
3. For L2: batch `parentWrappedOutcome()` to resolve quote tokens
4. Generate static Rust arrays with all market/pool data
5. Parse prediction CSVs and generate prediction arrays

## Constants

- `MULTICALL_BATCH_SIZE`: 16000 (sub-calls per multicall)
- `FEE_TIER`: 100 (0.01% Uniswap fee tier)
- `L1_QUOTE_TOKEN`: `0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0`
