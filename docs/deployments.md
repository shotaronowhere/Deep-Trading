# Deployments

## Optimism L1 execution addresses (hardcoded)

- `CTF_ROUTER_ADDRESS`: `0x179d8F8c811B8C759c33809dbc6c5ceDc62D05DD`
- `SWAP_ROUTER_ADDRESS`: `0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45`
- `BATCH_SWAP_ROUTER_ADDRESS`: `0x4081136d23FEeCD324a420A54635e007F51fd94a`
- `MARKET_1_ADDRESS`: `0x3220a208aaf4d2ceecde5a2e21ec0c9145f40ba6`
- `MARKET_2_ADDRESS`: `0xfea47428981f70110c64dd678889826c3627245b`
- `MARKET_2_COLLATERAL`: `0x63a4f76ef5846f68d069054c271465b7118e8ed9`

## TradeExecutor deployment

TradeExecutor is resolved at runtime and cached in:

- `cache/trade_executor.json`

Cache key is `(chain_id, owner)` and each entry stores a deployed executor address.

On startup, execute flow:

1. Reads cached executor for `(chain_id, owner)`.
2. Validates non-empty bytecode and `owner()` match.
3. Redeploys from `out/TradeExecutor.sol/TradeExecutor.json` if invalid or missing.
4. Persists refreshed cache entry.

## Note

`BATCH_SWAP_ROUTER_ADDRESS` is currently hardcoded in Rust. If router bytecode is redeployed, update
the constant in `src/execution/mod.rs` and this document together.
