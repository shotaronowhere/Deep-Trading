# End-to-End Real Contract Test Harness Plan

Plan for deploying real Seer PM + Uniswap V3 contracts locally to end-to-end test on-chain solvers (`Rebalancer.sol`, `RebalancerMixed.sol`), then later wire up the off-chain waterfall solver.

## Assumptions

- Start with a root categorical market, not a child market.
- Use base `Router`, not `GnosisRouter` or `MainnetRouter`, because the harness collateral is a local dummy ERC20.
- Keep all harness-only Solidity inside `test/RebalancerRealE2E.t.sol`; only dependency vendoring/remappings and this doc should live elsewhere.
- **INVALID_RESULT token handling**: The Rebalancer does NOT special-case the invalid token — it treats all entries in `tokens[]` uniformly. Critically, `splitPosition` produces `numOutcomes() + 1` tokens (including invalid), and `mergePositions` requires ALL tokens including invalid. The Rebalancer's `_tryMintSell` and `_tryBuyMerge` only approve tokens in `tokens[]` for the ctfRouter before calling `mergePositions` ([Rebalancer.sol:1329-1330](contracts/Rebalancer.sol#L1329-L1330), [1411-1412](contracts/Rebalancer.sol#L1411-L1412)). If the invalid token is excluded from `tokens[]`, the Seer Router's `transferFrom` on the invalid token will revert with insufficient allowance. Therefore the harness must:
  - **Include the invalid token in `RebalanceParams.tokens[]`, `pools[]`, `isToken1[]`, and `sqrtPredX96[]`** with its own Uniswap pool. This is the only way arb functions work without contract modifications.
  - Give the invalid token a low prediction probability (e.g., `0.05` sUSD) and seed its pool with liquidity.
  - `_returnAll` ([Rebalancer.sol:1567-1583](contracts/Rebalancer.sol#L1567-L1583)) iterates `p.tokens[]` and returns collateral — since invalid is now in `tokens[]`, it will be returned to the caller.
- **Single fee tier per call**: `RebalanceParams.fee` is a single `uint24` ([Rebalancer.sol:35](contracts/Rebalancer.sol#L35)) used for ALL swaps. All pools in one `rebalance()` / `rebalanceAndArb()` call must share the same fee tier. Test different fee tiers in separate test cases, not within one call.

## Key Decisions

- **Seer deployment order** (factory-based path): `ConditionalTokens -> RealityETH_v3_0 -> RealityProxy -> Wrapped1155Factory -> Market implementation -> MarketFactory -> Router`.
- `ConditionalRouter`, `GnosisRouter`, `MainnetRouter`, and futarchy contracts are **out of scope** for phase 1.
- **Recommend pinning Seer under `lib/`** at a fixed commit instead of copying minimal files. Copying is only a fallback if pragma/import conflicts become unmanageable.
- **Minimal practical Uniswap stack**: `WETH9 -> UniswapV3Factory -> SwapRouter02`. For liquidity, prefer a tiny local mint-callback helper declared inside the test file; `NonfungiblePositionManager` is optional, not required.
- **Must use SwapRouter02** (from `swap-router-contracts`), not the v3-periphery SwapRouter. Our `IV3SwapRouter.sol` matches SwapRouter02's ABI (no `deadline` in `ExactInputSingleParams` struct). Deploying the wrong router causes silent ABI misalignment. SwapRouter02 constructor: `(address _factoryV2, address factoryV3, address _positionManager, address _WETH9)`. For a V3-only harness, pass `address(0)` for `_factoryV2` and `_positionManager`.
- **Must deploy 0.7.6 contracts from pre-compiled JSON artifacts, not source**. Uniswap v3-core is `pragma solidity =0.7.6` (pinned). The test file is `^0.8.24`. You cannot `import` across Solidity major versions. `foundry.toml` has `via_ir = true`, and attempting to compile 0.7.6 source with `via_ir` will crash with Yul translation errors. Do NOT use `deployCode("SwapRouter02.sol", ...)` — instead use pre-compiled artifacts:
  ```solidity
  // Example: deploy from JSON artifact
  bytes memory bytecode = vm.getCode("lib/swap-router-contracts/artifacts/contracts/SwapRouter02.sol/SwapRouter02.json");
  address router;
  assembly { router := create(0, add(bytecode, 0x20), mload(bytecode)) }
  ```
  The `fs_permissions` in `foundry.toml` already grant read access to `./lib/swap-router-contracts/artifacts`. For v3-core artifacts (UniswapV3Factory, WETH9), either add an `fs_permissions` entry for `./out` (if compiling locally) or for the specific artifact directory under `lib/`. You may also need to add `{ access = "read", path = "./lib/v3-core" }` if loading from v3-core test artifacts. Only 0.8.x contracts (Seer, our contracts) can be imported directly.
- **Available fee tiers** (Optimism defaults): 100/1, 500/10, 3000/60, 10000/200. Fee tier 100/1 is **NOT** auto-enabled by the UniswapV3Factory constructor — it must be explicitly enabled via `factory.enableFeeAmount(100, 1)`. The factory constructor only enables 500/10, 3000/60, and 10000/200. Since `RebalanceParams.fee` is a single `uint24`, test different fee tiers in **separate test cases** (e.g., one test with all pools at fee 10000, another with all pools at fee 500). Existing mock tests use tickSpacing 100000 — real tick traversal behavior will differ significantly.

## Implementation Plan

### Step 1: Lock the Runtime Path

Read `contracts/Rebalancer.sol`, `contracts/RebalancerMixed.sol`, current mock tests, and `docs/architecture.md` to lock the exact runtime path:
- Constructor args
- Whether `rebalance()` uses split, merge, swap, or all three
- Whether it loops `numOutcomes()` or expects an extra invalid token

**Verify**: Write down the exact preconditions the real harness must satisfy.

### Step 2: Vendor External Sources

Add under `lib/` and pin commits:
- `seer-pm/demo` (contains Router, Market, MarketFactory, RealityProxy)
- Gnosis `conditional-tokens-contracts`
- Gnosis `1155-to-20` (Wrapped1155Factory)
- The RealityETH source used by Seer

Do NOT cherry-pick files into `contracts/`. Preserve original import layout and add remappings to `foundry.toml`.

**Verify**: `Router`, `Market`, `MarketFactory`, `Wrapped1155Factory`, `RealityProxy`, `ConditionalTokens`, and the Uniswap artifacts compile together. Also verify multi-solc compatibility (Uniswap v3 core/periphery and Seer may not share the same pragma range).

### Step 3: Create the Test Harness

Create `test/RebalancerRealE2E.t.sol` as the single self-contained harness. All local-only helpers in that file:
- Dummy `sUSD` ERC20 (must be 18 decimals — Rebalancer math hardcodes `1e18` and wrapped outcome tokens are 18 decimals)
- Optional Uniswap V3 mint-callback helper
- Price encoding helpers
- Deployment helpers

**Verify**: The test file is the only new Solidity test surface.

### Step 4: Deploy Seer Root-Market Stack

Deploy inside the test `setUp()` in this order:
1. Dummy `sUSD` collateral
2. `ConditionalTokens`
3. `RealityETH_v3_0`
4. `RealityProxy(conditionalTokens, realitio)`
5. `Wrapped1155Factory`
6. `Market` implementation (template)
7. `MarketFactory(marketImpl, arbitrator, realitio, wrapped1155Factory, conditionalTokens, sUSD, realityProxy, questionTimeout)`
8. Base `Router(conditionalTokens, wrapped1155Factory)`

Use `address(0)` for arbitrator unless source inspection shows otherwise. Verify whether the vendored source also needs a separate `Wrapped1155` implementation from `1155-to-20`.

**Verify**: Nonzero address and correct immutable wiring after each deploy.

### Step 5: Create a Small Categorical Market

Create through `MarketFactory.createCategoricalMarket(CreateMarketParams)` with **3 named outcomes**. Seer creates 4 wrapped ERC20s total (3 tradeable + 1 invalid). All wrapped outcome tokens have 18 decimals (Gnosis 1155-to-20 standard).

```solidity
// CreateMarketParams fields:
//   marketName, outcomes (["A","B","C"]), category, lang,
//   lowerBound (0), upperBound (0),  // categorical → both 0
//   minBond, openingTime, tokenNames
```

Set `openingTime = uint32(block.timestamp)` or `uint32(block.timestamp + 1)`. RealityETH v3 requires `opening_ts >= block.timestamp` — passing a past timestamp will revert.

**Verify**:
- `market.numOutcomes() == 3`
- `wrappedOutcome(0..2)` return distinct tradeable ERC20s
- The invalid wrapped token (index 3) is identified explicitly and recorded
- Verify `address(0)` is accepted for arbitrator — if MarketFactory rejects it, use a dummy EOA

### Step 6: Deploy Uniswap V3 Stack

1. `WETH9` — deploy from pre-compiled JSON artifact via `vm.getCode()`
2. `UniswapV3Factory` — deploy from pre-compiled JSON artifact (pragma =0.7.6, cannot compile with `via_ir = true`)
3. Enable fee tier 100/1: `factory.enableFeeAmount(100, 1)` (not auto-enabled; test contract is factory owner since it deployed it)
4. `SwapRouter02` — deploy from pre-compiled JSON artifact with constructor args:
   ```
   _factoryV2:      address(0)    // unused, V3-only harness
   factoryV3:       factory       // UniswapV3Factory address
   _positionManager: address(0)   // unused, no NFT LP
   _WETH9:          weth9         // WETH9 address
   ```

**Verify**:
- `factory.feeAmountTickSpacing(100) == 1`
- `factory.feeAmountTickSpacing(500) == 10`
- `factory.feeAmountTickSpacing(3000) == 60`
- `factory.feeAmountTickSpacing(10000) == 200`
- SwapRouter02 address is nonzero

### Step 7: Bootstrap Outcome Tokens via Split

Before creating pools or adding liquidity, obtain outcome tokens by splitting collateral through the Seer Router:

1. Mint sufficient sUSD to the LP actor (e.g., `1000e18`)
2. Approve Seer `Router` to spend sUSD
3. Call `router.splitPosition(sUSD, market, splitAmount)` — this produces equal amounts of ALL 4 outcome tokens (3 tradeable + 1 invalid)

**Verify**:
- LP actor holds `splitAmount` of each of the 4 outcome tokens (including invalid)
- LP actor's sUSD balance decreased by `splitAmount`

### Step 8: Create Pools, Initialize Prices, Add Liquidity

**Pool creation** — for ALL 4 outcome tokens (3 tradeable + 1 invalid), all at the **same fee tier** (since `RebalanceParams.fee` is a single `uint24`):

1. Sort `token0` / `token1` by address: `token0 = min(sUSD, outcome)`, `token1 = max(sUSD, outcome)`
2. Record `isToken1[i] = (address(sUSD) < address(outcomeToken[i]))` — the outcome IS token1 when sUSD sorts lower
3. Create pool: `factory.createPool(token0, token1, fee)`
4. Initialize with correct sqrtPriceX96:

```
Price encoding (Uniswap price = token1/token0):

If sUSD is token0 (outcome is token1):
  pool_price = outcome_per_sUSD = 1 / outcome_price_in_sUSD
  sqrtPriceX96 = sqrt(1 / outcome_price) × 2^96

If sUSD is token1 (outcome is token0):
  pool_price = sUSD_per_outcome = outcome_price_in_sUSD
  sqrtPriceX96 = sqrt(outcome_price) × 2^96
```

Default fee tier: 10000 (tickSpacing 200). Separate test cases exercise other fee tiers (500, 3000, 100).

**Price selection matters for arb coverage.** The arb thresholds ([Rebalancer.sol:1247-1257](contracts/Rebalancer.sol#L1247-L1257)) are:
- Mint-sell: `priceSum > 1e18 × 1e6 / (1e6 - fee)` (prices inflated → profitable to mint and sell)
- Buy-merge: `priceSum < 1e18 × (1e6 - fee) / 1e6` (prices deflated → profitable to buy all and merge)
- Dead zone: neither threshold met → no arb

For fee=10000: mint threshold ≈ 1.0101, buy threshold ≈ 0.99. Prices summing to exactly 1.0 trigger NO arb. Arb runs **after** `_pullAndSell()` ([Rebalancer.sol:92](contracts/Rebalancer.sol#L92)), so it's the **post-sell** pool state that determines which branch executes. Use different initial price sets for different scenarios:
- **Mint-sell scenario**: seed prices summing > 1.01 (e.g., `0.50 / 0.35 / 0.25 / 0.06 = 1.16`)
- **Buy-merge scenario**: seed prices summing < 0.99 (e.g., `0.40 / 0.28 / 0.18 / 0.04 = 0.90`)
- **Balanced scenario** (no arb, waterfall only): prices summing to 1.0 (e.g., `0.45 / 0.30 / 0.20 / 0.05 = 1.0`)

**Liquidity provision** — via direct `pool.mint()` with a local callback helper:

```solidity
contract MintHelper {
    function mint(address pool, int24 tickLower, int24 tickUpper, uint128 amount) external {
        IUniswapV3Pool(pool).mint(address(this), tickLower, tickUpper, amount, "");
    }
    function uniswapV3MintCallback(uint256 amount0, uint256 amount1, bytes calldata) external {
        IERC20(IUniswapV3Pool(msg.sender).token0()).transfer(msg.sender, amount0);
        IERC20(IUniswapV3Pool(msg.sender).token1()).transfer(msg.sender, amount1);
    }
}
```

Transfer both sUSD and outcome tokens to MintHelper before calling `mint()`. Use one wide tick range per pool for simple debugging. Tick bounds must be multiples of the pool's tickSpacing.

**Note:** The MintHelper uses `mint()` and `token0()`/`token1()` on the pool, but the project's [IUniswapV3Pool](contracts/interfaces/IUniswapV3Pool.sol) only exposes `slot0`, `liquidity`, `tickSpacing`, `tickBitmap`, `ticks`, and `token0()`. The MintHelper must use a test-local extended interface (importing from `@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol` or defining the additional selectors inline).

**Probe swaps** — before running probe swaps, the probe actor must approve `SwapRouter02` to spend both sUSD and each outcome token. Without this, a missing-approval revert looks identical to a pool-setup bug.

**Token ordering assertion** — after all pools are deployed, assert that at least one pool has `isToken1 == true` and at least one has `isToken1 == false`. Address ordering is deterministic from CREATE, so if all outcomes sort the same way relative to sUSD, the `isToken1 == false` code path is silently untested. If needed, deploy sUSD with a chosen salt via `CREATE2` to force both orderings.

**Verify**:
- `pool.liquidity() > 0` for each pool
- `slot0.sqrtPriceX96` matches intended price orientation
- Probe swaps succeed in both directions for each pool (with approvals)
- Both `isToken1 == true` and `isToken1 == false` cases exist across the pool set

### Step 9: Deploy Real Rebalancer and Set Up Approvals

Deploy: `Rebalancer(address(swapRouter02), address(seerRouter))`

**Approval chain** (the caller must approve Rebalancer; Rebalancer internally approves SwapRouter and Seer Router):
1. Caller approves Rebalancer to spend sUSD (`collateral.approve(rebalancer, type(uint256).max)`)
2. Caller approves Rebalancer to spend ALL 4 outcome tokens (including invalid)

**Fund the actor** with a skewed starting inventory (not a balanced basket):
- Some sUSD as liquid capital
- Uneven outcome token amounts to force both buys and sells
- Include invalid token in the inventory

Populate `RebalanceParams` with all 4 outcomes:
- `tokens[0..3]`: all 4 wrapped outcome addresses (including invalid)
- `pools[0..3]`: corresponding Uniswap pools
- `isToken1[0..3]`: token ordering flags
- `sqrtPredX96[0..3]`: prediction prices (invalid gets low prediction, e.g. 0.05)
- `fee`: single fee tier matching all pools

Snapshot balances for: caller, rebalancer, Seer router, wrapped1155 factory, conditional tokens, every pool.

**Verify**:
- All allowances are set for the full call path (all 4 outcome tokens + collateral)
- Balances match expected starting inventory

### Step 10: First End-to-End Scenario

Target `rebalanceAndArb()` (not plain `rebalance()`) to exercise the full Seer integration path including split and merge:

- Start from skewed holdings (from Step 9) so at least one outcome requires buying and another requires selling
- Path exercises real `splitPosition` (mint-sell arb) and at least one real `exactInputSingle`
- Include buy-merge cycle: Rebalancer buys underpriced tokens (including invalid), merges complete set for collateral profit
- Invalid token is treated as a regular outcome — its pool participates in arb cycles naturally

**Verify**:
- Real pool `Swap` events are emitted (including on the invalid token's pool during arb)
- Rebalancer balances move toward target allocation (predicted prices)
- Token flows through Seer Router and Wrapped1155Factory are real, not mocked
- `_returnAll` returns all 4 outcome tokens + collateral back to the caller (no tokens stranded)
- No unexpected balances remain on the Rebalancer except known dust

### Step 11: Documentation

**Implementation notes** (completed):

- **Deployment order**: DummySUSD → ConditionalTokens (bytecode) → RealityETH_v3_0 (bytecode) → Wrapped1155Factory (Seer's modified version, bytecode) → RealityProxy (Seer, bytecode) → Market impl (bytecode) → MarketFactory (bytecode) → Seer Router (bytecode) → UniswapV3Factory (bytecode) → WETH9 (bytecode) → SwapRouter02 (bytecode) → MintHelper → Rebalancer
- **Pragma conflict resolution**: Seer contracts are `pragma solidity 0.8.20` (pinned), our contracts are `^0.8.24`. Cannot import directly. All Seer contracts (Router, Market, MarketFactory, RealityProxy) are compiled separately in an isolated Foundry project and deployed from pre-compiled JSON artifacts via `vm.getCode()`. Same approach for ConditionalTokens (^0.5.1), Wrapped1155Factory (>=0.6.0), RealityETH (0.8.6), and UniswapV3Factory (=0.7.6).
- **Wrapped1155Factory version**: The npm `1155-to-20@1.0.2` package has a 2-param `requireWrapped1155(multiToken, tokenId)`. Seer uses a modified 3-param version `requireWrapped1155(multiToken, tokenId, data)` from `lib/seer-pm-demo/contracts/src/interaction/1155-to-20/Wrapped1155Factory.sol`. Must compile and use the Seer version.
- **Price convention**: `priceWad` = cost in sUSD per outcome token (WAD scale). If outcomeIsToken1: `sqrtPriceX96 = 2^96 * 1e9 / sqrt(priceWad)`. If outcome is token0: `sqrtPriceX96 = sqrt(priceWad) * 2^96 / 1e9`.
- **Token ordering**: Both `isToken1=true` and `isToken1=false` orderings are confirmed present across the 4 outcome pools (scenario 5 verified).
- **Invalid token**: Included in `tokens[]` with its own pool and prediction (0.05 sUSD). Works correctly through split, merge, swap, and arb cycles.
- **Base Router used**: Our Rebalancer uses `ICTFRouter` interface matching the base `Router(IConditionalTokens, IWrapped1155Factory)` — no GnosisRouter or MainnetRouter needed since collateral is a plain ERC20.
- **Arb scenarios**: 4a (mint-sell, fee=500, sum=1.16) and 4b (buy-merge, fee=3000, sum=0.90) both produce real arb profit (~618-622 sUSD).

**Fixture artifacts** in `test/fixtures/`:
- `ConditionalTokens.json` — from npm `@gnosis.pm/conditional-tokens-contracts` (solc 0.5.10)
- `Wrapped1155Factory.json` — compiled from Seer's modified source (solc 0.7.6)
- `RealityETH_v3_0.json` — from npm `@reality.eth/contracts` (solc 0.8.6)
- `SeerRouter.json`, `SeerMarket.json`, `SeerMarketFactory.json`, `SeerRealityProxy.json` — compiled from `lib/seer-pm-demo` (solc 0.8.20)
- `UniswapV3Factory.json` — from npm `@uniswap/v3-core` (solc 0.7.6)
- `WETH9.json` — from `lib/swap-router-contracts/test/contracts/`

## Test Matrix

All tests include all 4 outcomes (3 tradeable + 1 invalid) in `tokens[]`/`pools[]`. Different fee tiers are tested in **separate test cases** (not mixed within one call).

| # | Scenario | Entry Point | Fee | Description |
|---|----------|-------------|-----|-------------|
| 1 | Seer smoke | direct calls | n/a | `splitPosition` then `mergePositions` round-trip on all 4 outcomes, no Uniswap |
| 2 | Uniswap smoke | `swapRouter.exactInputSingle` | 10000 | Single outcome pool buy + sell against real V3 core, both token orderings |
| 3 | Root-market rebalance | `rebalance()` | 10000 | 4 outcomes, swap-only (no Seer split/merge — `rebalance()` never calls ctfRouter), sell overpriced + waterfall buy |
| 4a | Root-market arb (mint-sell) | `rebalanceAndArb()` | 500 | Seed pool prices summing 1.16 (> mintThreshold) so `_arbRound` takes mint-sell branch |
| 4b | Root-market arb (buy-merge) | `rebalanceAndArb()` | 3000 | Seed pool prices summing 0.90 (< buyThreshold) so `_arbRound` takes buy-merge branch |
| 5 | Token-ordering | `rebalance()` | 10000 | At least one pool where sUSD is `token0` and one where it is `token1` — verify `isToken1[]` correctness |
| 6 | Fee-tier 500 | `rebalance()` | 500 | All pools at fee 500 (tickSpacing 10) — exercises finer tick traversal with same 4-outcome setup |
| 7 | Fee-tier 100 | `rebalance()` | 100 | All pools at fee 100 (tickSpacing 1) — near-tick-granularity traversal |
| 8 | Low-liquidity | `rebalance()` | 10000 | Narrower ranges or smaller liquidity: revert/no-op/slippage behavior |
| 9 | Child market (later) | TBD | TBD | Conditional child market where parent wrapped outcome is collateral |

## Dependencies to Vendor

| Library | Purpose | Source | Status |
|---------|---------|--------|--------|
| `seer-pm/demo` | Router, Market, MarketFactory, RealityProxy | https://github.com/seer-pm/demo | done (submodule + artifacts) |
| `conditional-tokens-contracts` | ConditionalTokens (Gnosis CTF) | npm `@gnosis.pm/conditional-tokens-contracts` | done (bytecode fixture) |
| `1155-to-20` | Wrapped1155Factory (Seer's modified version) | `lib/seer-pm-demo/contracts/src/interaction/1155-to-20/` | done (compiled artifact) |
| RealityETH | RealityETH_v3_0 (used by Seer) | npm `@reality.eth/contracts` | done (bytecode fixture) |
| `v3-core` | UniswapV3Factory, pools | https://github.com/Uniswap/v3-core | already in lib/ |
| `v3-periphery` | Periphery base contracts | https://github.com/Uniswap/v3-periphery | already in lib/ |
| `swap-router-contracts` | SwapRouter02 | https://github.com/Uniswap/swap-router-contracts | already in lib/ |
| `v3-periphery-foundry` | Foundry-compatible periphery | https://github.com/gakonst/v3-periphery-foundry | already in lib/ |

## Sources

- Seer create market: https://seer-3.gitbook.io/seer-documentation/developers/interact-with-seer/create-a-market
- Seer router: https://seer-3.gitbook.io/seer-documentation/developers/contracts/core/router
- Seer market: https://seer-3.gitbook.io/seer-documentation/developers/contracts/core/market
- Seer market example: https://seer-3.gitbook.io/seer-documentation/developers/interact-with-seer/market-example
- Seer RealityProxy: https://seer-3.gitbook.io/seer-documentation/developers/contracts/core/realityproxy
- Uniswap v3 periphery: https://github.com/Uniswap/v3-periphery
- Gnosis Conditional Tokens: https://github.com/gnosis/conditional-tokens-contracts
- Gnosis 1155-to-20: https://github.com/gnosis/1155-to-20
