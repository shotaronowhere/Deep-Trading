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
