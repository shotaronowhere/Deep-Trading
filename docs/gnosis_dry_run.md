# Gnosis Chain Dry-Run (Movie Futarchy Markets)

## Overview

Dry-run system that polls Gnosis chain blocks, static-calls RebalancerAlgebra on-chain solver variants via SolverQuoter, and reports which solver produces the best net-EV portfolio (after gas fee deduction).

## Markets

15 scalar movie futarchy markets from "Session 1 - Movies Experiment". Each movie has an `upToken` and `downToken` paired against a per-movie `underlyingToken` (wrapped ERC1155 outcome token from the parent market) in Swapr (AlgebraV1.9) pools.

**Key distinction**: Pools are `outcomeToken/underlyingToken` pairs, NOT `outcomeToken/sDAI`. Each movie has its own `underlyingToken`, so solver calls are grouped per-movie (2 pools each: up + down).

### Prediction mapping
- `upToken` price = `score / 100` (from `data/movie_predictions.csv`)
- `downToken` price = `1 - score / 100`

## Architecture

### Files

| File | Purpose |
|------|---------|
| `src/gnosis.rs` | Static market data: 15 movies with tokens, underlying_token, scores, CREATE2 pool derivation |
| `src/execution/gnosis_preview.rs` | Algebra solver preview: SolverQuoter eth_call with state overrides, RebalanceParams builder |
| `src/bin/gnosis_preview.rs` | Binary: block polling loop, pool discovery, balance fetching, per-movie solver comparison |

### Key constants (Gnosis chain)

| Constant | Address |
|----------|---------|
| sDAI | `0xaf204776c7245bF4147c2612BF6e5972Ee483701` |
| GnosisRouter (CTF) | `0xeC9048b59b3467415b1a38F63416407eA0c70fB8` |
| Swapr Router | `0xfFB643E73f280B97809A8b41f7232AB401a04EE1` |
| Pool Deployer | `0xC1b576AC6Ec749d5Ace1787bF9Ec6340908ddB47` |
| Parent Market | `0x6f7ae2815e7e13c14a6560f4b382ae78e7b1493e` |

### Pool address derivation

CREATE2: `keccak256(0xff ++ poolDeployer ++ keccak256(abi.encode(token0, token1)) ++ initCodeHash)[12..]`

Tokens are sorted (token0 < token1) before encoding.

## Usage

```bash
# Dry-run (read-only, requires cached solver address)
cargo run --release --bin gnosis_preview

# Live mode (deploys solver if needed, sends approval txs)
EXECUTE_SUBMIT=Live cargo run --release --bin gnosis_preview
```

### Required `.env`

```
RPC_GNOSIS=<gnosis chain RPC>
PRIVATE_KEY=<signer private key>
TRADE_EXECUTOR=<deployed TradeExecutor address>
```

### Optional `.env`

```
EXECUTE_SUBMIT=Live    # enables solver deployment + approval txs
RUST_LOG=info          # tracing filter
```

## Solver variants

1. **rebalanceExact** — pure waterfall rebalance across pools
2. **rebalanceAndArbExact** — waterfall + complete-set arbitrage via parent market

Both are called via SolverQuoter (state override at `0x...DeAdBeef`) which reverts with `(bool success, bytes returnData, uint256 postCash, uint256[] postBalances)`.

## Flow

1. Discover active pools (verify CREATE2 addresses have on-chain code)
2. Resolve RebalancerAlgebra solver (deploy or read from cache)
3. Pre-check infinite approvals (live mode only)
4. Block polling loop:
   - Fetch all token balances (outcome + underlying tokens)
   - Group pools by collateral (underlying_token) — one solver call per movie
   - Static-call both solver variants per movie
   - Report comparison table with raw EV, EV delta, net EV delta (after gas), status
