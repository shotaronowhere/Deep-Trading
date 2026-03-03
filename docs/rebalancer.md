# On-Chain Rebalancer

## Overview

`Rebalancer.sol` implements the full portfolio rebalancing logic on-chain using a closed-form waterfall allocation. Given a belief vector (predictions as `sqrtPriceX96` values), it:

1. **Pulls** the caller's explicit collateral budget plus any configured outcome balances into the contract
2. **Sells** overpriced outcomes down to the fee-neutral fair-value boundary
3. **Arbs** complete-set dislocations before the waterfall consumes cash
4. **Buys** underpriced outcomes with profitability-equalized allocation
5. **Recycles** below-frontier holdings only when the frontier edge clears round-trip fees
6. **Returns** all holdings to caller

All in a single atomic transaction with built-in slippage protection via Uniswap V3 price limits. The contract holds all tokens internally during execution, enabling recycling of low-profitability holdings.

## Solver Modes

The contract supports two buy-side solver modes:

1. **Constant-L**
   - Uses the current in-range `liquidity()` from each pool once per waterfall pass.
   - Sorts underpriced outcomes once by current profitability, solves the closed-form `psi` frontier over that sorted prefix, and then executes that same ordered buy plan.
   - This is the default, cheaper, simpler path.
2. **TickAware Exact**
   - Computes exact per-pool cost-to-target by scanning the tick bitmap, jumping to the next initialized boundary (or bitmap word edge), and applying each crossed segment's `liquidityNet`.
   - Solves the global `psi` frontier by bisection on the exact summed multi-tick cost function, reusing each cached per-pool segment ladder during the solve, then executes one swap per pool to the solved target.
   - This is the highest-fidelity path for maximizing EV, and the most expensive path.

Both modes use the same execution primitive: one `exactInputSingle` per actual pool swap with a computed `sqrtPriceLimitX96`. The difference is how the contract computes the frontier.

### Consequences of the Constant-L Assumption

The `psi` derivation is exact only while a pool remains inside its current tick range, because Uniswap V3 liquidity is piecewise constant between initialized ticks.

When a swap crosses one or more initialized ticks:

- execution remains correct and protected, because Uniswap itself walks the ticks internally and enforces the price limit;
- the **Constant-L** solver can become allocation-suboptimal, because it priced the move using only the starting in-range liquidity;
- the loss is an optimization loss (budget allocation / residual cash), not a slippage-safety loss.

In practice this means:

- **Constant-L** is exact inside the current tick and approximate across tick crossings;
- **TickAware Exact** uses the same direct-only waterfall objective, but replaces the constant-`L` cost approximation with exact multi-segment Uniswap V3 costs over the scanned tick path.

### Exact Mode Overhead

The exact solver does not submit one swap per tick. It still executes one swap per pool.

The extra cost is entirely in the planning math:

- read `tickSpacing()` for each active pool,
- read `tickBitmap(word)` to skip empty stretches quickly,
- read `ticks(boundary)` only when a crossed boundary is initialized,
- build a per-pool in-memory segment ladder to prediction,
- reuse that cached ladder inside the bounded `psi` bisection loop instead of rescanning storage on every guess.

The current implementation counts actual `tickSpacing()` boundaries crossed for `maxTickCrossingsPerPool`, but it no longer reads every empty boundary individually. It jumps by bitmap word when nothing is initialized, which preserves the exact same cost curve while materially reducing planning gas in sparse liquidity regions.

## Algorithm

### Phase 1: Pull and Sell

1. Pull `collateralAmount` of collateral plus any nonzero outcome balances into the contract (via `transferFrom`)
2. Read each pool's sqrtPrice to determine direction
3. For outcomes that are still overpriced after fees (token1: `s < p × √(1-fee)`, token0: `s > p / √(1-fee)`):
   - Swap outcome → collateral with `sqrtPriceLimitX96` at the fee-neutral fair-value boundary
   - Uniswap stops the swap before the final slice turns negative EV
   - The contract computes `√(1-fee)` in fixed point as `sqrt((FEE_UNITS - fee) * FEE_UNITS) / FEE_UNITS`, so 5 bps pools keep the intended precision instead of collapsing to a coarse `999/1000` sqrt factor
4. Unsold and underpriced tokens remain in contract for recycling

### Phase 2: Complete-Set Arb Pre-Pass

If the caller provided a complete set for `market`, the contract runs complete-set arb before the first buy pass. This gives deterministic `sum(price) != 1` profit first access to collateral, instead of letting the waterfall consume the budget first.

### Phase 3: Read Pool State

Read `sqrtPriceX96` and `liquidity` from each pool's `slot0()`. These are the post-sell prices.

### Phase 4: Compute ψ (Closed-Form)

At waterfall equilibrium, all active (underpriced) outcomes share the same profitability π = 1/ψ² - 1, where:

```
ψ = (C + budget × (1 - fee)) / D

C = Σ L_i × g(sqrtPrice_i)     (current state, underpriced outcomes only)
D = Σ L_i × g(sqrtPred_i)      (prediction state, underpriced outcomes only)

g(x) = 2^96 / x   for token1 outcomes  (= √outcome_price)
g(x) = x / 2^96   for token0 outcomes  (= √outcome_price)
```

For underpriced outcomes: C < D (since g(current) < g(prediction)). With budget > 0, ψ ∈ (0, 1] typically.

**Sorted prefix solve**: Sort underpriced outcomes by current profitability, then walk that sorted list once while accumulating `C` and `D`. The active set is the maximal profitability prefix whose marginal outcome still clears the implied frontier. This produces the same closed-form frontier as iterative pruning, but avoids repeated full-basket scans in the common constant-`L` path.

**Pruning condition** (no division needed):
- token1: `s_i × (C + budgetAdj) ≥ p_i × D`
- token0: `p_i × (C + budgetAdj) ≥ s_i × D`

### Phase 5: Execute Buys

Sort all buyable outcomes on-chain by current profitability before executing swaps:
- token1 outcomes rank by `sqrtPrice / sqrtPred`
- token0 outcomes rank by `sqrtPred / sqrtPrice`

The contract reuses that same profitability ordering from the `psi` solve; it does not sort the basket a second time before execution.

For each active outcome, compute target `sqrtPriceX96`:
- token1: `target = sqrtPred × D / (C + budgetAdj)` (sqrtPrice decreases, outcome price goes up)
- token0: `target = sqrtPred × (C + budgetAdj) / D` (sqrtPrice increases, outcome price goes up)

Swap collateral → outcome with full remaining budget and computed price limit. Each pool consumes only what it needs to reach its target; the rest flows to subsequent pools. Sorting ensures that if the single-tick liquidity approximation underestimates the true cost-to-target, the highest-edge outcomes consume the scarce budget first.

Approvals are also lazy and sticky: the contract grants `type(uint256).max` allowance only when the current allowance is insufficient, instead of rewriting exact allowances every pass.

### Phase 6: Bounded Waterfall Refinement

If budget remains after a buy pass (due to single-tick L approximation or favorable price movements), the contract re-reads pool state, recomputes ψ from updated prices, and repeats the buy pass. This continues until a pass spends nothing, collateral is exhausted, or the fixed `MAX_WATERFALL_PASSES` cap is reached. The loop is still bounded, but no longer assumes that two passes are always enough.

In **TickAware Exact** mode, the contract solves the exact frontier up front by bisection on the scanned multi-tick cost curve, so it does not need repeated buy passes unless it later enters a recycle round.

When exact mode determines `buyAll`, it also skips building the normal sorted buy plan and simply executes every underpriced pool directly to prediction. Since the exact scan already proved the basket can fully clear to prediction, ordering is irrelevant in that special case.

### Phase 7: Reverse Waterfall (Recycle Below-Frontier Holdings)

After the forward waterfall, some held tokens may have profitability below the frontier π. These are less profitable than the outcomes we just bought. The reverse waterfall sells them with **frontier price limits** — the same target price as the forward waterfall's buy limit — to recover capital for redeployment.

The mechanism mirrors the forward waterfall:
- **Forward**: buy most profitable first, profitability drops to frontier. Price limits self-regulate.
- **Reverse**: sell least profitable first, profitability rises to frontier. Price limits self-regulate.

No explicit sorting needed — each pool independently sells to its frontier price. The least profitable token sells the most (furthest from frontier), naturally ratcheting up to meet higher-profitability holdings.

For each recycle round:
1. Read pool state, compute ψ with current collateral budget
2. For each held token below the frontier, compute sell limit = frontier price:
   - token1: `limit = p × den/num` (above current, selling pushes sqrtPrice UP → profitability rises)
   - token0: `limit = p × num/den` (below current, selling pushes sqrtPrice DOWN → profitability rises)
3. Recycle only if the frontier profitability beats the current holding by at least round-trip fee drag
4. Sell each qualifying below-frontier token with full balance + frontier limit (Uniswap stops at frontier)
5. Forward waterfall: deploy recovered collateral into above-frontier outcomes using the same bounded refinement loop
6. Repeat until no below-frontier tokens remain or `maxRecycleRounds` hit

The sell limit is the **same formula** as `_buyLimit` — the frontier price is the equilibrium from both directions. This ensures:
- Never sell past the frontier (no round-trip overshoot)
- Avoid fee-churn on thin edge differences
- Self-regulating capital allocation across pools

### Phase 8: Return All

Send all outcome token balances and remaining collateral to caller.

### Full Rebalance Flow (with arb)

When called as `rebalanceAndArb()`, the full sequence is:

1. Pull tokens, sell only where the edge remains positive after fees (Phase 1)
2. Run complete-set arbitrage before the waterfall spends the budget (Phase 2)
3. Waterfall buy with bounded iterative refinement (Phases 3-6)
4. Recycle only materially lower-edge holdings, then re-run the same bounded waterfall (Phase 7)
5. Return all holdings to caller (Phase 8)

## Why Sequential Execution Mostly Works

Pools are independent Uniswap V3 pools. Buying outcome_1 does not change the price in outcome_2's pool, so if every active pool can reach its target then the execution order does not affect the final prices. In practice, the ψ calculation uses only the current in-range liquidity, so multi-tick paths can make the true cost-to-target higher than the single-tick estimate. In those cases the order matters because later pools may be budget-constrained, so the contract sorts buyable outcomes on-chain by current profitability to allocate the budget to the best opportunities first.

## Multi-Tick Robustness

The ψ formula uses `liquidity()` — the in-range liquidity at the current tick. In Uniswap V3, liquidity changes at tick boundaries. This makes ψ an approximation when swaps cross ticks. However, **the mechanism is robust to arbitrary liquidity profiles** for the following reasons:

1. **Execution is exact regardless of L**: Each `exactInputSingle` call passes the full remaining budget as `amountIn` with the target as `sqrtPriceLimitX96`. Uniswap internally walks through every tick boundary, adjusting L at each crossing, and stops exactly at the limit. The cost-to-target is computed by Uniswap's own math, not by our ψ formula.

2. **Profitability equalization is exact for all pools that reach their targets**: For any pool that reaches `s_target = p_i / ψ`, its profitability is `(p_i / s_target)² - 1 = ψ² - 1`, independent of L. The liquidity profile only affects the cost to get there, not the profitability at the target.

3. **L only affects budget prediction**: The single-tick L determines whether ψ correctly predicts total cost = budget. If L is overestimated (actual liquidity thinner), later pools may not reach their targets. If underestimated, surplus budget is refunded. Both cases are safe.

4. **Budget self-regulation**: Since each pool gets the full remaining budget (not a pre-allocated share), the sequential execution naturally adapts. Pools that cost less than predicted leave more for subsequent pools.

5. **Bounded refinement**: After each waterfall pass, if budget remains, another pass re-reads pool state, recomputes ψ from updated post-swap prices, and allocates the surplus. The loop is bounded by an explicit cap.

For typical rebalancing where budget is small relative to total pool depth (budgetAdj << C), the L errors in C and D largely cancel in the ratio, making ψ accurate even with the single-tick approximation.

## Slippage Protection

- **Per-outcome**: Price limits prevent any individual swap from overshooting
- **Budget excess**: If prices moved favorably, later bounded refinement passes deploy the surplus
- **Budget shortfall**: If prices moved unfavorably, later pools don't fully reach their targets; profitability is slightly unequal but no overpayment occurs
- **EV tolerance**: The caller controls aggressiveness via predictions; tighter predictions = less buying
- **Fee-aware liquidation**: Overpriced inventory is only sold while the marginal execution still clears swap fees
- **Fee-aware recycling**: Reverse-waterfall churn is skipped when the frontier edge does not beat round-trip fee drag

## Gas Estimate (Optimism)

- Pool reads: ~250k gas (98 external SLOADs)
- ψ computation: ~100k gas (iterative computation, pure)
- Swap execution: ~150k gas per active pool swap
- Total: ~16M gas for 98 outcomes (fits in block)

## Gas Profiling Tests

`test/Rebalancer.t.sol` includes dedicated `testGasProfile*` scenarios for comparing solver overhead under controlled liquidity shapes.

These tests use `vm.pauseGasMetering()` during setup and again during assertions, so the reported per-test gas from `forge test` is dominated by the rebalance call itself rather than mock deployment noise.

The contract also applies a small set of low-risk micro-optimizations in hot paths: cached array lengths, cached collateral ERC20 handles inside repeated `balanceOf` loops, and `unchecked` increments only in bounded loops that do not use `continue`.

Current scenarios cover:

- `testGasProfileConstantLTwoOutcomeSingleTick`: baseline two-outcome constant-`L` waterfall in a simple single-tick shape
- `testGasProfileExactSingleTickCostSolve`: exact solver where the target remains inside the current tick
- `testGasExactBuyAllFastPathBeatsLegacySortedPlan`: direct 98-outcome benchmark for the exact-mode `buyAll` fast path, comparing the new unsorted execute-to-prediction path against the legacy sorted-plan path under the same state
- `testGasProfileExactInitializedTickCrossing`: exact solver crossing one initialized boundary and applying `liquidityNet`
- `testGasProfileExactSparseBitmapJump`: exact solver with a distant initialized boundary so the planner must skip sparse empty space through the bitmap
- `testGasProfile{ConstantL,Exact}{Eight,Sixteen,ThirtyTwo,NinetyEight}OutcomeSingleTick`: scaling matrix for 8/16/32/98 underpriced pools in a uniform single-tick shape
- `testBenchmarkABMultiTickConstantLVsExact`: two-outcome A/B benchmark with identical starting prices and predictions, but one pool has materially deeper liquidity after several crossed initialized ticks. `rebalance` and `rebalanceExact` run against the same direct-buy execution oracle, and the test logs call gas, terminal EV, and how many pools were actually touched before budget exhaustion.
- `testBenchmarkABMultiTickSyntheticNinetyEightOutcomeConstantLVsExact`: 98-outcome extension of the same synthetic deep-after-crossing shape. Every pool starts with the same current liquidity, but future initialized ticks add more liquidity, so `ConstantL` underestimates the true cost to reach prediction across the basket. The benchmark also logs the pool-touch count so you can see how early `ConstantL` stops.
- `testBenchmarkABMultiTickRealisticSeededNinetyEightOutcomeConstantLVsExact`: 98-outcome benchmark seeded from the dominant embedded L1 ladder shape (spot-adjacent sentinel near `512`, main boundary near `16095`, far sentinel near `92108`), snapped to a coarser on-chain-friendly grid. It adds a modest in-path liquidity band at those real-L1-style boundaries so the benchmark stays economically meaningful while still reflecting the production ladder structure, and logs the touched-pool count for the same reason.

This gives a clean way to compare:

- approximation-path overhead versus exact-path overhead
- single-tick exact planning versus initialized multi-tick planning
- dense local crossing versus sparse bitmap jumping
- how planner and execution overhead scale with outcome count under the same simple liquidity shape
- direct solver A/B economics under a real multi-tick mispricing shape, not just planner gas
- full-basket A/B economics for both intentionally adversarial synthetic ladders and a more production-shaped L1-seeded ladder

## Interface

```solidity
struct RebalanceParams {
    address[] tokens;       // outcome token addresses
    address[] pools;        // corresponding Uniswap V3 pools
    bool[] isToken1;        // true if outcome is token1 in pool
    uint256[] balances;     // amount of each outcome to sell
    uint256 collateralAmount; // collateral to pull from caller for the buy budget
    uint160[] sqrtPredX96;  // predictions as sqrtPriceX96
    address collateral;     // e.g., sUSD
    uint24 fee;             // e.g., 100 (0.01%)
}

function rebalance(RebalanceParams calldata params)
    external
    returns (uint256 totalProceeds, uint256 totalSpent);

function rebalanceExact(
    RebalanceParams calldata params,
    uint256 maxBisectionIterations,
    uint256 maxTickCrossingsPerPool
) external returns (uint256 totalProceeds, uint256 totalSpent);

function rebalanceAndArb(
    RebalanceParams calldata params,
    address market,
    uint256 maxArbRounds,
    uint256 maxRecycleRounds
) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);

function rebalanceAndArbExact(
    RebalanceParams calldata params,
    address market,
    uint256 maxArbRounds,
    uint256 maxRecycleRounds,
    uint256 maxBisectionIterations,
    uint256 maxTickCrossingsPerPool
) external returns (uint256 totalProceeds, uint256 totalSpent, uint256 arbProfit);
```

`totalProceeds` includes collateral realized from the initial sell phase and any recycle-round sells. `totalSpent`
includes the initial waterfall passes / exact solve buys and recycle-round redeployments.

Callers must approve both:

- every nonzero `balances[i]` outcome amount they want the contract to sell
- `collateralAmount` of `collateral` so the rebalance can fund buys atomically instead of relying on pre-funded contract balance

## Complete-Set Arbitrage

The `arb()` function normalizes complete-set prices toward summing to 1 using the same price-limit self-regulation as the waterfall. `rebalanceAndArb()` runs this logic as a pre-pass before the first buy loop.

### Fee-Adjusted Thresholds

Arb is only profitable beyond the fee drag dead zone:
- `mintThreshold = 1e18 × 1e6 / (1e6 - fee)` — mint-sell when sum > this
- `buyThreshold = 1e18 × (1e6 - fee) / 1e6` — buy-merge when sum < this

For fee = 100 (0.01%): dead zone is [0.9999e18, 1.0001e18]. No arb within this band.

### Mint-Sell (sum > mintThreshold)

Uses price limits to fully consume the opportunity in one pass:

1. Read all pool sqrtPrices, compute priceSum
2. Compute equilibrium scaling factor K = 1e6×1e18 / (feeComp × priceSum), where K < 1
3. Mint complete set with all available collateral
4. Sell each outcome token with `sqrtPriceLimitX96` targeting the equilibrium price:
   - token1: `limit = sqrtPrice × √(feeComp × priceSum) / √(1e6 × 1e18)` (above current, selling pushes sqrtPrice up)
   - token0: `limit = sqrtPrice × √(1e6 × 1e18) / √(feeComp × priceSum)` (below current, selling pushes sqrtPrice down)
5. Uniswap stops each sell at the equilibrium — unsold tokens remain
6. Merge unsold complete sets back into collateral
7. Leftover incomplete sets stay in contract

### Buy-Merge (sum < buyThreshold)

Same self-regulating pattern as the waterfall:

1. Compute K = feeComp×1e18 / (1e6 × priceSum), where K > 1
2. Buy each outcome with full remaining budget + price limit:
   - token1: `limit = sqrtPrice × √(1e6 × priceSum) / √(feeComp × 1e18)` (below current, buying pushes sqrtPrice down)
   - token0: `limit = sqrtPrice × √(feeComp × 1e18) / √(1e6 × priceSum)` (above current, buying pushes sqrtPrice up)
3. Each pool consumes only what it needs; the rest flows to subsequent pools
4. Merge minimum outcome balance into collateral
5. Leftover incomplete sets stay in contract

### Why One-Pass Works

Each sell/buy uses the full token balance or budget with a price limit. The pool's own math determines exactly how much to consume. No budget pre-allocation or equal splits needed — the price limit self-regulates capital allocation across pools, just as in the waterfall.

Iterates up to `maxRounds` (typically 2-3 rounds capture rounding residuals).

```solidity
function arb(
    address[] calldata tokens,
    address[] calldata pools,
    bool[] calldata isToken1,
    address collateral,
    address market,
    uint24 fee,
    uint256 maxRounds
) external returns (uint256 profit);
```

## Dependencies

- `IV3SwapRouter` — Uniswap V3 SwapRouter02
- `ICTFRouter` — Conditional Token Framework (split/merge)
- `IUniswapV3Pool` — pool `slot0()` and `liquidity()` reads
- `FullMath` — 512-bit `mulDiv` from Uniswap V3 core

## Mathematical Derivation

For outcome i with sqrtPrice s_i and sqrtPrediction p_i:

**Outcome price** (in collateral):
- token1: `price_i = Q96² / s_i²`  (outcome = token1, collateral = token0)
- token0: `price_i = s_i² / Q96²`  (outcome = token0, collateral = token1)

**g(x)** maps sqrtPriceX96 to √(outcome_price):
- token1: `g(x) = Q96 / x` — underpriced when `s > p`, buying decreases sqrtPrice
- token0: `g(x) = x / Q96` — underpriced when `s < p`, buying increases sqrtPrice

**Profitability** (how much the market underprices):
- token1: `prof_i = (s_i / p_i)² - 1`
- token0: `prof_i = (p_i / s_i)² - 1`

**Cost to buy** from s to target s_t (single tick range):
- token1: `cost = L × (Q96/s_t - Q96/s) / (1-fee)`  (s_t < s, buying decreases sqrtPrice)
- token0: `cost = L × (s_t - s) / (Q96 × (1-fee))`   (s_t > s, buying increases sqrtPrice)

**At equilibrium** (all active outcomes at profitability π = 1/ψ² - 1):
- token1: `s_t = p_i × D / (C + budget×(1-fee))`  (target below current)
- token0: `s_t = p_i × (C + budget×(1-fee)) / D`   (target above current)

Summing costs = budget gives: `ψ = (C + budget×(1-fee)) / D`

No square roots needed — ψ is computed as a simple ratio.

**Swap direction conventions** (Uniswap V3 sqrtPriceX96 = √(token1/token0) × 2^96):
- Buying token1 (zeroForOne=true): sqrtPrice decreases, limit must be below current
- Buying token0 (zeroForOne=false): sqrtPrice increases, limit must be above current
- Selling token1 (zeroForOne=false): sqrtPrice increases, limit must be above current
- Selling token0 (zeroForOne=true): sqrtPrice decreases, limit must be below current
