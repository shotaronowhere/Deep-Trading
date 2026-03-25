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

## Gnosis chain (Swapr / AlgebraV1.9) addresses

Used by `RebalancerAlgebra.sol` for scalar futarchy markets.

- Swapr AlgebraV1.9 Router: `0xfFB643E73f280B97809A8b41f7232AB401a04EE1`
- Algebra Pool Deployer (CREATE2): `0xC1b576AC6Ec749d5Ace1787bF9Ec6340908ddB47`
- Pool Init Code Hash: `0xbce37a54eab2fcd71913a0d40723e04238970e7fc1159bfd58ad5b79531697e7`
- GnosisRouter (CTF): `0xeC9048b59b3467415b1a38F63416407eA0c70fB8`
- sDAI (collateral): `0xaf204776c7245bF4147c2612BF6e5972Ee483701`

Pool addresses are derived via CREATE2 from the Pool Deployer + init code hash + `keccak256(token0, token1)`.
All Swapr futarchy pools use `tickSpacing = 60`.

## Note

`BATCH_SWAP_ROUTER_ADDRESS` is currently hardcoded in Rust. If router bytecode is redeployed, update
the constant in `src/execution/mod.rs` and this document together.
