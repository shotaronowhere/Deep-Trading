# Optimal Portfolio Rebalancing on Prediction Market AMMs

*How to allocate a budget across 98 prediction market outcomes when every purchase moves the price against you.*

## The Setup

You hold beliefs — probability estimates — over 98 mutually exclusive outcomes. Each outcome trades on a Uniswap V3 pool against a stablecoin (sUSD). You have a budget B in sUSD, possibly some existing outcome token holdings, and you want to maximize the expected value of your portfolio.

The objective is clean:

```
max  Σᵢ πᵢ hᵢ + cash
s.t. total cost ≤ B
     hᵢ ≥ 0
```

where πᵢ is your prediction for outcome i and hᵢ is your holdings. If AMM prices were fixed, this would be a linear program — you'd dump everything into the single outcome with the best prediction-to-price ratio. But AMM prices aren't fixed. Buying pushes the price up. Your cost function is nonlinear and convex. This changes the problem structure entirely.

## The AMM Price Model

Uniswap V3 concentrates liquidity L within a tick range. Within that range, it behaves as a constant-product AMM. We need three facts.

**Fact 1: Cost to move price.** Buying outcome tokens to push the price from P₀ to P₁ costs

```
cost = L_eff × (√P₁ - √P₀)
```

where L_eff = L / (10¹⁸ × (1 − f)) and f = 0.0001 is the fee. This holds regardless of whether the outcome token is token0 or token1 in the pool — the two orderings produce the same formula after simplification.

**Fact 2: Price impact of selling.** Selling m tokens into a pool at price P₀ moves the price to

```
P(m) = P₀ / (1 + mκ)²
```

where κ = (1 − f) × √P₀ × 10¹⁸ / L is the price sensitivity parameter. The sell proceeds are

```
proceeds(m) = P₀ m (1 − f) / (1 + mκ)
```

**Fact 3: Tick boundary cap.** Each pool has finite liquidity range. The maximum sellable before exhausting the tick:

```
M_cap = (√(P₀ / P_limit) − 1) / κ
```

Beyond this, the pool is saturated. All computations use min(m, M_cap) per pool.

## Profitability and the KKT Condition

Define the profitability of outcome i at price P:

```
prof(P) = (πᵢ − P) / P
```

and the inverse — the price at which profitability equals some level π*:

```
P_target = πᵢ / (1 + π*)
```

The key structural result: **at the constrained optimum, every purchased outcome has the same marginal profitability.** This is the KKT condition. If outcome A had higher marginal profitability than outcome B, you could improve your portfolio by shifting a dollar from B to A. At the optimum, no such improvement exists.

This means the solution has a threshold π* such that:
- Outcomes with profitability > π* at current prices are bought until their profitability drops to π*
- Outcomes with profitability ≤ π* are not bought

The problem reduces to finding π*.

## The Waterfall Algorithm

The algorithm pours capital like water filling containers of decreasing height.

**Step 1.** Find the outcome with highest profitability. Call it active.

**Step 2.** Deploy capital to the active outcome. As you buy, its price rises and profitability falls.

**Step 3.** When the active outcome's profitability drops to meet the next-best outcome, add that outcome to the active set.

**Step 4.** Deploy capital to all active outcomes simultaneously, maintaining equal profitability across them. Repeat from Step 3.

**Step 5.** When the budget is insufficient to reach the next profitability level, solve for the achievable π* and execute.

This is exact, not a heuristic. Each step either completes a full descent (active outcomes reach the next level and a new outcome joins) or exhausts the budget (triggering the budget solver in Step 5).

## Two Acquisition Routes

Each outcome can be acquired two ways:

**Direct buy:** Swap sUSD for outcome tokens in the pool. Price = current pool price. Cost governed by Fact 1.

**Mint route:** Pay 1 sUSD to mint one token of every outcome (a "complete set"), then sell all outcomes except the target. Net cost = 1 − Σⱼ≠ᵢ proceeds(selling jth token). The effective price is the alt price:

```
alt_price(i) = 1 − Σⱼ≠ᵢ Pⱼ
```

which rises as you sell non-target tokens and push their prices down.

The waterfall treats (outcome, route) pairs as separate entries. The same outcome can appear twice — direct at one profitability, mint at another. Both are exploited independently.

**Route interaction.** Direct buys move the target pool's price. Mint sells move non-target pools' prices. These are independent price channels draining profitability from different directions.

## Closed-Form Budget Solver (All-Direct Case)

When every active outcome uses direct buy, Step 5 has an exact solution. The budget constraint is:

```
Σᵢ L_eff_i × (√(πᵢ/(1 + π*)) − √Pᵢ) = B
```

Factor out 1/√(1 + π*):

```
(Σᵢ L_eff_i × √πᵢ) / √(1 + π*) = B + Σᵢ L_eff_i × √Pᵢ
```

Define A = Σᵢ L_eff_i √πᵢ and B' = B + Σᵢ L_eff_i √Pᵢ. Then:

```
π* = (A / B')² − 1
```

No iteration. One line of arithmetic. This works because the cost function — a sum of √P terms — has the right algebraic structure to invert cleanly.

## Newton's Method for the Mint Route

The mint route is harder. To find the mint amount m that achieves a target alt price P_target, we solve:

```
g(m) = Σⱼ Pⱼ / (1 + min(m, M_cap_j) × κⱼ)² = 1 − P_target
```

This is a sum of decreasing functions (each pool's price drops as you sell into it), so g is monotone decreasing, and Newton converges reliably.

The derivative:

```
g'(m) = Σⱼ (−2 Pⱼ κⱼ) / (1 + mκⱼ)³     [only for unsaturated pools]
```

Saturated pools (m ≥ M_cap_j) contribute zero derivative — their price is pinned at P_limit.

**Warm start.** Linearizing at m = 0: g(0) = Σ Pⱼ, g'(0) = −2Σ Pⱼκⱼ. The first Newton step gives m₀ = (P_target − alt(0)) / (2Σ Pⱼκⱼ), which is typically close enough that 2–3 more iterations suffice.

The mint route returns two cost metrics:

```
cash_cost(m)  = m − Σⱼ≠ᵢ Pⱼ × mⱼ × (1−f) / (1 + mⱼ × κⱼ)
value_cost(m) = cash_cost(m) − Σⱼ∈capped predⱼ × (m − mⱼ)
```

where mⱼ = min(m, M_cap_j). `cash_cost` is the actual sUSD spent (used for budget feasibility). `value_cost` subtracts the expected value of unsold tokens from saturated pools, reflecting the true economic cost of the position.

The derivative d(cash_cost)/dm = 1 − (1−f) × Σⱼ∈uncapped Pⱼ(m). Both costs and the cash derivative with respect to π* (via the implicit function theorem, dm/dπ = P_target / ((1 + π*) × g'(m))) are computed in a single pass. The budget solver uses cash_cost to ensure feasibility.

## Mixed-Route Budget Solver

When the active set contains both direct and mint entries, the budget equation couples them:

```
Σᵢ∈direct cost_direct_i(π*) + Σⱼ∈mint cost_mint_j(π*) = B
```

Direct costs have closed-form derivatives: d(cost)/dπ = −L_eff √pred / (2(1 + π*)^(3/2)).

Mint costs have implicit derivatives, computed via the chain rule through the inner Newton solve for m.

The outer solver runs Newton on π* using these analytical gradients. An inner Newton solves for the mint amount M at each outer step. This nested structure — outer Newton on π*, inner Newton on M — converges in about 15 × 8 = 120 function evaluations, which is fast enough (the whole rebalance runs in 3.6ms for 98 outcomes).

**Coupling simplification.** All non-active pools see the same aggregate sell volume M from minting, regardless of which target outcome the mint is for. This collapses the problem to two scalar unknowns (π*, M) rather than one per mint entry.

## Two Sell Routes

When reducing exposure to an outcome (Phases 1 and 3), there are two routes:

**Direct sell:** Sell outcome tokens into the pool. Proceeds governed by Fact 2. Moves the pool price down.

**Merge route:** Buy one token of every *other* outcome, then merge (burn) complete sets to recover 1 sUSD per set. The effective sell price is:

```
merge_price(i) = 1 − Σⱼ≠ᵢ buy_cost(j, 1) / 1
```

At marginal (m→0): merge_price ≈ 1 − Σⱼ≠ᵢ Pⱼ/(1−f), which is worse than direct by the fee spread. But for high-price outcomes (P > ~0.5), buying the cheap complementary set and merging dominates direct selling.

**Buying m tokens** of outcome j at price P₀ₐ uses the dual sensitivity parameter:

```
λⱼ = √Pⱼ × 10¹⁸ / L
```

Price after buying m: P(m) = P₀ / (1 − mλ)². Cost: m × P₀ / ((1−f)(1−mλ)). Max buyable: M_buy_cap = (1 − √(P₀/P_buy_limit)) / λ.

The merge route is only available when all pools are present (no partial-pool handling). When selling, we optimize a direct/merge split for each token amount and execute the proceeds-maximizing mixture.

## The Full Rebalance: Three Phases

**Phase 1: Sell overpriced holdings.** For any outcome where the market price exceeds your prediction and you hold tokens, sell until price = prediction or you exhaust holdings. For each sell amount, optimize a **mixture** of direct sells and merge sells (buy all others + merge), because marginal route attractiveness shifts as each route is partially consumed; execute the proceeds-maximizing split. This is unambiguously correct — holding overpriced assets is negative expected value — and the proceeds fund Phase 2.

**Phase 2: Waterfall allocation.** Deploy the budget (initial sUSD + Phase 1 proceeds) via the waterfall. Returns the final profitability level π_last.

**Phase 3: Liquidation and reallocation.** After the waterfall, scan existing holdings. Any holding with profitability below π_last is capital trapped in a suboptimal position. Sell the worst holdings first (only enough to raise their profitability to π_last — full liquidation would cause round-trip losses), optimizing a direct/merge split for each sell. Recover the capital and run a second waterfall pass.

The three phases ensure: (1) no capital sits in overpriced positions, (2) available capital goes to the best opportunities, (3) legacy holdings that lag the portfolio's marginal standard are recycled.

## Why This Works

The waterfall is not a greedy heuristic that happens to do well. It is the exact solution to the constrained optimization, derived from the KKT conditions. The equal-marginal-profitability condition is both necessary and sufficient for optimality (the cost function is convex, the objective is linear, strong duality holds).

The closed-form solver for all-direct and the nested Newton for mixed routes are not approximations — they find the exact π* (up to floating-point precision and the single-tick-range assumption). The only genuine approximation is that we model each pool as having a single tick range of liquidity. For pools with liquidity concentrated in one range (which these are), this is exact.

The algorithm handles edge cases structurally:
- **Zero-liquidity pools**: excluded at construction (no PoolSim created), mint route disabled
- **Tick boundary saturation**: per-pool sell caps enforced in Newton, derivative zeroed for saturated pools
- **Negative mint costs** (arbitrage): executed normally, budget increases
- **Active-set changes during execution**: costs recomputed right before each trade, waterfall breaks early if recomputed costs diverge from estimates

Performance: 3.6ms per call for 98 outcomes on release build. The dominant cost is the waterfall loop (at most 1000 iterations, each doing O(n) work across outcomes). In practice, convergence happens in 10–20 iterations.
