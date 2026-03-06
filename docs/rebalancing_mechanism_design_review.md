# Rebalancing Mechanism Design Review

Author perspective: Dan Robinson (senior DeFi mechanism designer)

Actionable production policy derived from this analysis is maintained in `docs/rebalancer_approaches_playbook.md`.

## 1. Architecture Overview

Three solver families exist, each with distinct tradeoffs:

```
                        ┌─────────────────────────────┐
                        │  Off-chain Rust Solver       │
                        │  (6-phase, mixed routes,     │
                        │   EV guards, rich heuristic) │
                        └─────────┬───────────────────┘
                                  │ plans actions
                                  ▼
                        ┌─────────────────────────────┐
                        │  Strict Executor             │
                        │  (receding-horizon subgroup  │
                        │   execution, per-leg limits) │
                        └─────────────────────────────┘

        ┌────────────────────────┐     ┌──────────────────────────┐
        │  On-chain Constant-L   │     │  On-chain Mixed          │
        │  (Rebalancer.sol)      │     │  (RebalancerMixed.sol)   │
        │  closed-form ψ, O(n)   │     │  bisection on π, nested  │
        │  atomic solve+execute  │     │  mint bisection, fallback│
        └────────────────────────┘     └──────────────────────────┘
```

All three share the same mathematical foundation: equalize marginal profitability across all purchased outcomes (KKT condition for constrained EV maximization). They differ in route richness, staleness exposure, gas cost, and optimality guarantees.

---

## 2. Off-chain Solver

### 2.1 Six-phase pipeline

| Phase | Purpose | Bound |
|-------|---------|-------|
| 0 | Complete-set arbitrage (buy-merge if `sum(P) < 1`, mint-sell if `sum(P) > 1`) | Gas-gated dry-run |
| 1 | Sell overpriced holdings (`P > prediction`) | 128 iters/outcome |
| 2 | Waterfall allocation (core capital deployment) | 1000 iters |
| 3 | Legacy inventory recycling with EV guard | 8 iters |
| 4 | Polish loop (re-run phases 1-3, commit only on EV improvement) | 64 passes |
| 5 | Terminal cleanup sweeps | 4+2 direct + 1 mixed |

### 2.2 Waterfall algorithm

The waterfall equalizes profitability `π = (prediction - price) / price` across all active outcomes:

1. **Frontier discovery**: find the set of outcomes tied at highest profitability (within `ACTIVE_FRONTIER_REL_TOL = 1e-9`)
2. **Route selection**: for each frontier member, pick cheapest route (direct vs mint). Mint-first ordering (mints free budget, directs consume it)
3. **Step planning**: deploy capital to bring frontier down to next-best profitability level
4. **Iterate**: promote newly-tied outcomes, repeat

**Closed-form solution** (all-direct active set):

```
A = Σᵢ L_eff_i × √pred_i
B' = budget + Σᵢ L_eff_i × √price_i
π = (A/B')² - 1
```

No iteration required. This is exact when all active outcomes use direct buy and pools stay within one tick range. This is the key insight — the waterfall is analytically solvable for the constant-L single-tick case.

**Mixed active sets**: simulation-backed bisection (up to 64 iterations). Each probe evaluates the full route execution sequence with mint-first ordering, so solver/executor drift is minimized.

### 2.3 Mixed route cost solving

**Direct route**: `cost = L_eff × (√P_target - √P_current)`. Closed-form.

**Mint route**: mint `m` complete sets, sell all non-target outcomes, keep target.

Net cost: `m - Σⱼ proceeds_j(m)` where `proceeds_j(m) = P_j × m × (1-f) / (1 + m × κ_j)`

Finding the right `m` for a target profitability uses Newton's method on:

```
g(m) = Σⱼ P_j / (1 + min(m, M_j) × κ_j)²
```

where `M_j` is the sell cap for pool `j` (tick boundary). Warm-started from linearized first step. Converges in 2-3 iterations typically.

**Route switching**: when both direct and mint are available, the planner bisects on the switch point where `direct_marginal_cost = mint_marginal_cost`, executing the cheaper route on each segment.

### 2.4 EV churn pruning

**Mechanism**: phases 3 and 4 use trial-commit-or-reject:

```
ev_before = budget + Σ holdings_i × pred_i
... execute trial liquidation + reallocation ...
ev_after = budget + Σ holdings_i × pred_i
accept iff ev_after ≥ ev_before - tol × (1 + |ev_before| + |ev_after|)
```

where `tol = 1e-10` (relative).

**What it prevents**: round-trip fee drag from selling and rebuying the same assets. Each round-trip costs ~0.02% (two 0.01% swaps). The guard ensures churn only happens when EV improvement exceeds this drag.

**Critical limitation**: this is **greedy local search**, not global optimization. The preserve-set is built by adding outcomes one at a time and accepting each if EV improves. It can get trapped in local optima where the globally-best preserve set requires simultaneously adding/removing multiple outcomes.

### 2.5 Known limitations

1. **Local optimality only**: the waterfall is greedy — it always buys the currently most profitable outcome. This is locally optimal (KKT), but the interaction between mint routes and direct routes can create situations where a different active set decomposition yields higher total EV.

2. **Concavity error in mint proceeds**: treating each non-target pool's proceeds independently overestimates total proceeds (Jensen's inequality). The mint route cost is underestimated by O(ΔP) where ΔP is the aggregate price movement.

3. **Heuristic phase ordering**: the sequence arb → sell → waterfall → recycle → polish is reasonable but not provably optimal. Different orderings can yield different terminal EV (the benchmark gap investigation confirmed this — Phase 0 arb before waterfall can reduce available alpha).

4. **Single-tick assumption**: the f64 sim model (`sim.rs`) is exact within one tick range but does not model tick crossings. Multi-tick scenarios are approximated.

---

## 3. On-chain Constant-L Solver (`Rebalancer.sol`)

### 3.1 Closed-form ψ

The on-chain solver computes ψ (psi) directly:

```
ψ = (C + budget × (1-f)) / D

where:
  C = Σᵢ L_eff_i × g(sqrtPrice_i)
  D = Σᵢ L_eff_i × g(sqrtPred_i)
  g(x) = 2⁹⁶/x  (token1 outcome)
  g(x) = x/2⁹⁶  (token0 outcome)

Profitability: π = 1/ψ² - 1
```

This is the on-chain equivalent of the off-chain closed-form `π = (A/B')² - 1`. They produce identical results on identical inputs (verified to within 5e-6 relative tolerance in A/B benchmarks).

### 3.2 Sorted prefix solve

The key efficiency trick: sort outcomes by buy priority (profitability ratio), then walk the sorted list once. Once an outcome is unprofitable at the current ψ level, all subsequent outcomes are also unprofitable. This avoids combinatorial active-set enumeration.

**Complexity**: O(n log n) for sorting + O(n) for prefix walk. For 98 outcomes, this is negligible.

### 3.3 Recycling (reverse waterfall)

After the forward waterfall, check if any held outcomes are below the frontier:

```
for each held outcome:
  if profitability < frontier AND recycle is fee-worthwhile:
    sell toward frontier (price-limited)
    redeploy recovered collateral via forward waterfall
```

**Fee-worthwhile check**: only recycle if the frontier EV beats current holding EV by more than round-trip swap fees. This prevents pointless churning.

Bounded by `maxRecycleRounds` (caller-controlled) and `MAX_WATERFALL_PASSES = 6`.

### 3.4 When constant-L breaks down

The constant-L assumption is exact when pool price stays within one tick range during the swap. It breaks when:

- **Tick crossings occur**: Uniswap V3 internally handles tick transitions, so execution is still safe (no funds lost). But the *allocation* becomes suboptimal — the solver computed costs assuming constant L, while actual costs use different L values across tick boundaries.
- **Large price movements**: when `budget / L` is large relative to tick width, multiple ticks are crossed and constant-L allocation error grows.
- **Heterogeneous liquidity**: if some pools have deep liquidity (stay in tick) and others have shallow liquidity (cross ticks), the relative allocation between them is wrong.

**Practical impact for current L1 market**: most pools are single-tick with concentrated liquidity. Constant-L is a reasonable approximation. The error is in allocation optimality, not safety.

### 3.5 Strengths

- **Atomic solve+execute**: eliminates staleness. Pool state is read and acted upon in the same transaction.
- **Deterministic**: closed-form ψ always succeeds (no convergence failure).
- **Gas efficient**: ~7.3M gas for 98 outcomes. Affordable on Optimism.
- **Fee-aware**: all boundaries account for the 0.01% Uniswap fee.

---

## 4. On-chain Exact Solver

### 4.1 Tick-aware cost curves

The exact variant (`rebalanceExact`) builds per-pool cost ladders:

```
ExactCostCurves {
  segmentEnds[]:        sqrtPrice boundaries at each tick
  segmentLiquidities[]: L value in each tick range
  segmentPrefixCosts[]: accumulated cost up to each tick
}
```

This is constructed by scanning the tick bitmap for each pool, walking from current price to prediction price.

### 4.2 Bisection on ψ

Instead of the closed-form ψ (which assumes constant L), the exact solver bisects:

```
find lowest ψ such that Σᵢ exact_cost_i(ψ) ≤ budget
```

where `exact_cost_i(ψ)` is computed from the multi-segment cost curve. This is O(log(ψ_range) × n × avg_segments).

### 4.3 When to use

- When crossing risk is significant (price movement > tick width)
- When pools have heterogeneous tick structures
- When allocation precision matters more than gas savings

**Gas**: ~8-12M for 98 outcomes (vs ~7.3M for constant-L). The overhead is in tick bitmap scanning and segment construction, not in the bisection itself.

---

## 5. On-chain Mixed Solver (`RebalancerMixed.sol`)

### 5.1 Algorithm

Outer bisection on profitability π, with nested inner bisection on mint amount:

```
Outer: find lowest π ∈ [π_lo, π_hi] such that total_cost(π) ≤ budget
Inner: find mint amount M such that delta(M) ≈ target_delta(π)
```

where `delta(M) = Σⱼ (P_j^0 - P_j(M))` is the total price reduction across non-active outcomes from selling minted tokens.

### 5.2 Non-active curve simulation

For each non-active outcome, the solver pre-computes:

```
NonActiveCurve {
  sqrtPrice, limit, liquidity, p0, cap, isToken1
}
```

The `cap` is the maximum sellable amount before hitting the frontier price limit. At each candidate π, the solver simulates selling minted tokens across all non-active pools and sums proceeds.

### 5.3 Safety gates and fallback

Seven distinct failure codes, all leading to direct-only fallback:

| Code | Reason | Meaning |
|------|--------|---------|
| 1 | Active set too large | Combinatorial explosion risk |
| 2 | No non-active universe | Nothing to sell minted tokens into |
| 3 | Non-active not sellable | All pools at or below frontier |
| 4 | Solve failed | Bisection or math failure |
| 5 | Residual tolerance exceeded | Planned vs actual mint diverge |
| 6 | Zero mint solved | Mint amount is negligible |
| 7 | Invalid params | Zero iteration counts |

Codes 1-6 emit `MixedSolveFallback(code)` and run direct-only constant-L. Code 7 reverts. This fail-closed design is correct — better to execute a slightly suboptimal direct-only plan than to revert the entire transaction.

### 5.4 Assessment

The mixed solver is well-engineered for safety but fundamentally limited by its on-chain gas cost. The nested bisection (24 outer × 24 inner = up to 576 evaluations) is expensive in EVM, and the per-evaluation cost includes multiple sqrt/mul/div operations across all non-active pools.

---

## 6. Mixed Route Value Analysis

### 6.1 Benchmark data

From committed on-chain A/B benchmarks:

| Case | EV Uplift (mixed vs direct) | Gas Overhead |
|------|----------------------------|--------------|
| Direct-only fixtures | 0.000% | +10-15% |
| `small_bundle_mixed_case` | +0.0156% | +257% |
| `mixed_route_favorable_synthetic` | +0.0868% | +429% |
| `heterogeneous_98_outcome_l1_like` | +0.0812% | +344% |

Off-chain full solver vs on-chain constant-L (randomized sweeps):
- 4-outcome: improvements in 480/500 cases (96%)
- 98-outcome: improvements in 100/100 cases (100%)

### 6.2 When mixed routes add value

Mixed routes help when:

1. `sum(prices) < 1` significantly — minting is cheap relative to direct buying
2. Some outcomes have thin direct liquidity but the collective non-active set is deep
3. The target outcome is expensive to buy directly but cheap to acquire via mint-and-sell-others

### 6.3 Why gas often dominates

For a 98-outcome market on Optimism:
- Constant-L gas: ~7.3M × gas_price
- Mixed gas: ~22.4M × gas_price
- Gas delta: ~15.1M units

At 0.01 gwei on Optimism, 15.1M gas ≈ 0.000151 ETH ≈ $0.45 at $3000/ETH.

The EV uplift on a $150 portfolio is +$0.12 (0.08%). So:

**Net benefit = $0.12 - $0.45 = -$0.33**

Mixed route is **unprofitable** at current gas costs for this portfolio size. Break-even portfolio size: ~$560 for the 0.08% case. For the 0.016% case: ~$2,800.

### 6.4 Can mixed routes be improved?

**Yes, but the leverage is limited.** Three concrete approaches:

**A. Top-K non-active selection**: instead of selling into all 97 non-active pools, select the top K by marginal proceeds contribution. Most of the mint value comes from the 5-10 most liquid non-active pools. Selling into 10 instead of 97 pools cuts gas by ~85% while preserving ~90% of proceeds.

**B. Batch mint-sell with proceeds threshold**: only attempt mixed when estimated proceeds exceed a threshold. Pre-screen with a cheap estimate: `sum(top_K_prices) × (1-f)` vs `mint_cost`. Skip if the spread is too thin.

**C. Hybrid execution**: use the off-chain solver to determine IF mixed is worthwhile (cheap to compute off-chain), then execute the direct-only on-chain path with a one-shot mint-sell prepended if the off-chain analysis says it's worth it. This separates the decision (off-chain, rich) from the execution (on-chain, simple).

### 6.5 Honest assessment

Mixed routes are theoretically sound but practically marginal for the current market structure (98 outcomes, single-tick concentrated liquidity, low mispricing). The complexity and gas cost outweigh the EV uplift in most realistic scenarios. The right approach is:

1. Default to direct-only on-chain constant-L
2. Gate mixed behind `estimated_uplift > gas_cost + risk_margin`
3. Invest engineering effort elsewhere (prediction quality, execution speed, capital efficiency)

---

## 7. Slippage and Staleness

### 7.1 Historical approach (abandoned)

Static hybrid: compute off-chain, submit on-chain with aggregate slippage bounds. Correctly abandoned because:

- Static tolerance can't adapt to varying liquidity/volatility
- Plan-to-inclusion latency creates unpredictable drift
- Replan loops add latency and RPC cost

### 7.2 On-chain solver (strongest staleness profile)

Solve and execute share one state transition. No staleness by construction. Per-pool `sqrtPriceLimitX96` prevents overshoot on individual legs. Budget excess is handled by bounded refinement passes; budget shortfall results in proportionally less buying (not overpayment).

This is the right answer for competitive environments. The only "slippage" is MEV extraction between tx submission and inclusion, which is a function of mempool visibility, not solver design.

### 7.3 Off-chain strict executor (current)

Receding-horizon execution with per-leg bounds:

```
edge = proceeds - cost  (or amount × prediction - cost for buys)
gas_total = gas_l2 + gas_l1
profit_buffer = max(buffer_min, buffer_frac × edge)
slippage_budget = edge - gas_total - profit_buffer
skip if slippage_budget ≤ 0
```

Per-leg allocation: `alloc_i = slippage_budget × (notional_i / total_notional)`

This is reasonable but fundamentally inferior to on-chain solving because it can't adapt to state changes between tx submission and inclusion. The conservative buffers (20% of edge + $0.25 minimum) eat into profitability.

### 7.4 Assessment

For competitive markets with other bots, on-chain solving dominates. Off-chain execution is only preferable when:
- Richer route analysis is needed (mixed routes, arb sequencing)
- Market is calm enough that plan-to-inclusion drift is small
- Portfolio is large enough that the off-chain EV advantage exceeds the staleness cost

---

## 8. Churn Pruning

### 8.1 Current mechanism

EV-guarded greedy preserve selection:

1. Start with empty preserve set
2. For each candidate outcome (sorted by profitability):
   - Trial: add to preserve set, run full rebalance
   - Accept if `EV_after ≥ EV_before - relative_tol`
3. Return best preserve set found

**Why it's greedy-local**: adding outcome A might be unprofitable alone, but adding A+B together might be profitable. The greedy search never tries adding A because it fails the individual EV check. Similarly, removing a previously-added outcome is never attempted.

### 8.2 How to get closer to global optimum

**Approach 1: 2-opt/3-opt neighborhood search**

After the greedy pass, try swapping pairs/triples:
- Remove outcome X from preserve set, add outcome Y
- If EV improves, accept the swap
- Repeat until no improving swap exists

This is standard local search for combinatorial optimization. Complexity: O(n² × rebalance_cost) per improvement round, bounded by convergence.

**Approach 2: Branch-and-bound with bounding**

Use the greedy solution as an initial lower bound. Branch on preserve-set membership decisions. Prune branches where the upper bound (best possible EV with remaining outcomes) is below the current best.

The bounding function: assume all remaining outcomes are preservable at zero cost. This overestimates EV (upper bound). If even this overestimate doesn't beat the incumbent, prune.

**Practical**: for 98 outcomes, full branch-and-bound is intractable (2^98 subsets). But with good bounding and the greedy warm-start, the search tree is dramatically pruned. Most outcomes have clear preserve/don't-preserve signals; only the borderline 5-10 outcomes need combinatorial exploration.

**Approach 3: Robust objective**

Instead of optimizing point EV, optimize `EV_net = EV - gas - slippage_risk_penalty`:

```
slippage_risk = Σᵢ |action_i| × adverse_move_i × √(latency_blocks)
```

This naturally penalizes excessive churn because each action incurs gas and slippage risk. The preserve set that maximizes `EV_net` is implicitly churn-aware without needing a separate pruning pass.

### 8.3 Recommendation

Implement 2-opt neighborhood search as the next step. It's simple, bounded, and catches the most common greedy failures (pairwise interactions). Save branch-and-bound for later if 2-opt proves insufficient. The robust objective is the right long-term direction but requires calibrating the slippage risk model, which is a separate effort.

---

## 9. Comparison Matrix

| Dimension | On-chain Constant-L | On-chain Exact | On-chain Mixed | Off-chain Full |
|-----------|--------------------:|---------------:|---------------:|---------------:|
| EV (direct-only) | baseline | baseline | baseline (fallback) | baseline |
| EV (mixed-favorable) | baseline | baseline | +0.02-0.09% | +0.08-0.12% |
| Gas (98 outcomes) | ~7.3M | ~8-12M | ~22.4M | N/A |
| Staleness | none | none | none | plan-to-inclusion drift |
| Tick precision | single-tick approx | exact multi-tick | single-tick approx | single-tick approx |
| Route richness | direct only | direct only | direct + mint | direct + mint + merge + arb |
| Failure mode | always succeeds | revert on tick scan overflow | fail-closed to direct | fail-closed to no actions |
| Recycling | yes (fee-gated) | yes (fee-gated) | no | yes (EV-guarded) |
| Optimality | KKT-optimal (single tick) | KKT-optimal (multi-tick) | heuristic (bisection) | heuristic (greedy waterfall) |

### When to use each

**On-chain constant-L** (default for production):
- Competitive markets with bot contention
- Single-tick-dominant pools (current L1 market)
- Gas-sensitive execution
- Maximum reliability needed

**On-chain exact** (escalation):
- Pools with multiple active tick ranges
- Large price movements expected (budget >> tick depth)
- Allocation precision justifies +1-5M gas

**On-chain mixed** (conditional):
- Estimated mixed EV uplift > gas overhead + risk margin
- `sum(prices)` significantly deviates from 1.0
- Portfolio large enough for EV uplift to matter in absolute terms

**Off-chain full solver** (research/calm markets):
- Calm markets with predictable inclusion timing
- Need to explore arb sequencing or complex route combinations
- Portfolio analysis and strategy development

---

## 10. Improvement Proposals

### 10.1 High priority (near-term, concrete)

**1. Adaptive constant-L vs exact switch** (on-chain)

Before calling `rebalance()`, compute a cheap crossing-risk indicator per pool:

```
crossing_risk_i = budget_share_i / (L_i × tick_width_i)
```

If `max(crossing_risk) > threshold`, use `rebalanceExact()`. Otherwise use `rebalance()`. This is a single `view` call followed by one `rebalance` variant.

**2. Top-K non-active selection** (on-chain mixed)

Instead of selling into all non-active pools, sort by `P_j × L_j` (liquidity-weighted price = proceeds proxy) and sell into top K only. For K=10 on a 98-outcome market:
- Gas: ~7M instead of ~22M (comparable to direct-only)
- Proceeds: ~90% of full complement (the tail 88 pools contribute minimal proceeds)
- Implementation: add a `maxNonActiveSells` parameter to `rebalanceMixedConstantL`

**3. 2-opt churn pruning** (off-chain)

After greedy preserve selection, run pairwise swap trials:
```
for each (i, j) where i ∈ preserve, j ∉ preserve:
  trial: swap i ↔ j in preserve set
  if EV improves: accept, restart
```

Bounded by O(n² × rebalance_cost) per improvement round. Expected: 1-3 rounds before convergence.

### 10.2 Medium priority

**4. Robust objective function** (off-chain)

Replace `max EV` with:
```
max EV_net = EV - Σ (gas_per_action × gas_price) - risk_penalty × Σ |ΔP_i| × √(latency)
```

This unifies gas awareness, slippage risk, and churn avoidance into one objective. The current system has these as separate guards (gas gates, slippage budgets, EV guards), which can conflict.

**5. Arb timing policy optimization** (off-chain)

The Phase 0 arb vs Phase 2 waterfall ordering matters (see `small_bundle_mixed_case` gap investigation). Instead of hardcoding arb-first:
- Evaluate `arb→waterfall` EV and `waterfall→arb` EV
- Pick the higher one
- Cost: one extra rebalance call (fast in f64 simulation)

**6. Off-chain mixed route confidence scoring** (off-chain)

Before executing a mixed subgroup, score it with replay robustness:
- Perturb prices by ±1-2% (adverse scenario)
- If the mixed subgroup EV drops below direct-only EV under perturbation, demote to direct-only
- This filters out brittle mixed plans that only work on exact current prices

### 10.3 Lower priority (longer-term)

**7. Multi-tick off-chain sim** (off-chain)

Extend `PoolSim` to model tick crossings in f64. Use the tick structure from build-time data to construct piecewise-linear cost curves. This improves off-chain/on-chain fidelity and enables the off-chain solver to detect when constant-L is a poor approximation.

**8. Branch-and-bound preserve search** (off-chain)

For borderline outcomes (profitability near frontier), use bounded combinatorial search:
- Greedy solution as initial lower bound
- Branch on preserve/don't-preserve for the top-K most ambiguous outcomes
- Prune branches via upper-bound check (assume all remaining outcomes preservable)
- Practically: K ≤ 10-15 outcomes makes this tractable

**9. On-chain gas profiling and auto-gating** (on-chain)

Track `MixedSolveFallback` reason distribution in production telemetry. If a reason code (e.g., code 3: non-active not sellable) dominates, auto-disable mixed mode for that market shape. This is runtime adaptation, not static configuration.

---

## 11. Open Questions

1. **Is the off-chain solver still needed for production?** If on-chain constant-L is the default and mixed is conditional, the off-chain solver's role shrinks to research and strategy development. The engineering effort to maintain two parallel solvers may not be justified unless the off-chain solver's route richness produces material live EV improvements (not just frozen-snapshot improvements).

2. **Multi-tick generalization timeline**: the current single-tick assumption works for the L1 market's concentrated liquidity. When (not if) pools span multiple ticks, the constant-L approximation degrades. The exact solver handles this but at higher gas cost. How much gas headroom exists on Optimism before this becomes a constraint?

3. **MEV protection**: none of the solvers address MEV extraction between tx submission and inclusion. For competitive markets, this may be the dominant source of execution cost. Flashbots Protect, private mempools, or MEV-aware tx ordering could be more impactful than solver optimization.

4. **Capital efficiency**: the current approach deploys all available capital. For prediction markets with binary outcomes, the kelly criterion or fractional kelly might produce better risk-adjusted returns than full capital deployment. This is a portfolio management question, not a solver question, but it interacts with the solver through the budget parameter.
