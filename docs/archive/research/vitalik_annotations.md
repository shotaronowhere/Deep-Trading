# Annotations on "Optimal Portfolio Rebalancing on Prediction Market AMMs"

*Some thoughts on Tao's write-up, from the perspective of someone who thinks a lot about mechanism design and the on-chain execution environment these trades actually live in.*

## Status Note (2026-02-27)

This document is an annotations/opinion note on an earlier implementation snapshot. Several critiques below have since been implemented:

- Runtime planning now uses `rebalance_with_gas` (see `src/main.rs`), and waterfall route admission is gas-aware via `best_non_active()` break-even gating (`remaining_budget × profitability >= gas_threshold`).
- The action model is now `Mint`, `Buy`, `Sell`, `Merge` only; flash-loan actions were removed.
- Submission wiring exists in `src/bin/execute.rs` (strict subgroup execution with replan-after-receipt), documented in `docs/execution_submission.md`.
- Full-mode rebalancing is now a six-phase flow (Phase 0-5), including EV-guarded polish and terminal cleanup sweeps.

## The Math is Clean. The Reality is Messier.

Terry's exposition of the waterfall algorithm is excellent — the reduction to a single scalar π* via KKT conditions is elegant, and the closed-form (A/B')² − 1 for the all-direct case is the kind of result that makes you feel like the problem was designed to have a nice answer. But I want to push on the gap between the mathematical model and what actually happens when you try to execute this on a blockchain.

## Minting Is Not a Free Lunch

The mint route — pay 1 sUSD, get one of every outcome, sell 97, keep the one you want — looks like a clever arbitrage. And it is. But the write-up treats minting as a costless facility with the only friction being AMM sell impact. On-chain, minting a complete set on Seer PM is two contract calls (one per market contract), plus 97 separate swap transactions to sell the non-target tokens. On Optimism, each swap is roughly 150k gas. That's ~14.5M gas just for the sells, which at current L2 gas prices is maybe $2-5. For a $100 portfolio, this overhead is 2-5% — the same order of magnitude as the profitability the algorithm is trying to capture.

The optimization should have a gas cost term. Something like:

```
net_cost_mint(m) = m − Σ proceeds_j(m) + gas_cost × n_swaps
```

Without this, the optimizer will recommend mint routes that are mathematically profitable but economically negative after execution costs. This is the classic MEV trap: the theoretical edge exists, but it gets eaten by the infrastructure layer.

**Practical fix:** batch the sells into a multicall. One transaction, one base fee, 97 swap calls as sub-calls. This drops the gas overhead by ~90%. The code already uses Multicall3 for reads — extending it to writes is natural.

**Current status:** Gas is now modeled as a route-level minimum trade threshold in runtime planning (`docs/gas_model.md`). This is a conservative break-even gate, not a full objective-term gas integration.

## The Single-Tick-Range Assumption Deserves More Scrutiny

Terry acknowledges this as "the only genuine approximation" and then moves on. I think it deserves more attention because it determines when the algorithm fails, not just when it's slightly imprecise.

Uniswap V3 pools for prediction market outcomes typically have liquidity concentrated in a single range — this is true. But the *width* of that range matters enormously. A pool with liquidity from tick 16095 to 92108 (as in the test fixtures) covers a price range of roughly [0.0001, 0.2]. If your target price is 0.3, you've crossed the tick boundary and the pool has zero liquidity to offer. The algorithm correctly caps sells at the tick boundary, but it treats this as an edge case rather than a structural feature.

For prediction markets specifically, the interesting regime is when outcomes are cheap (P < 0.05) and you think they should be less cheap (π ~ 0.1-0.3). These are exactly the trades with the highest profitability — and exactly the trades most likely to exhaust the tick range. The algorithm handles this by excluding the outcome from further consideration, but the right response might be to provide liquidity (open a new position in a higher tick range) rather than give up.

This connects to a broader point about prediction market microstructure: the liquidity providers and the informed traders are solving complementary problems. The LP needs to know where to put liquidity. The informed trader needs liquidity to exist where they want to trade. In mature markets, LPs observe where informed flow concentrates and move liquidity accordingly. This rebalancer, by optimizing trade execution, is implicitly signaling where liquidity is most needed.

## The Profitability Metric Implicitly Assumes Risk Neutrality

The profitability prof(P) = (π − P) / P is the expected return assuming your prediction π is correct. The waterfall equalizes this across all purchased outcomes. But this is optimal only under risk neutrality — you don't care about variance, only expected value.

For a prediction market with 98 mutually exclusive outcomes, at most one pays out. If you hold tokens across 20 outcomes, 19 of those positions go to zero. The expected value is positive, but the variance is enormous. A Kelly criterion approach would suggest sizing positions based on edge/odds, which coincides with risk-neutral optimization when the portfolio is small relative to your bankroll, but diverges when it's not.

The write-up doesn't mention bankroll management. For a bot with, say, $10,000 in capital, betting $100 per rebalance cycle is ~1% — well within Kelly bounds for typical edges. But the algorithm doesn't enforce this. If the edge is large (prices at 50% of predictions, as in the performance test), the algorithm will happily deploy the entire budget, and the six-phase structure can liquidate/recycle inventory to deploy more.

This isn't wrong — it's just a choice. Risk-neutral optimization is the right default for a market-making bot that rebalances frequently and can tolerate drawdowns. But it should be explicit.

## The Phase 3 Liquidation Is Doing Something Subtle

Phase 3 — sell held outcomes below π_last, reallocate upward — is described as recycling capital from suboptimal positions. What it's actually doing is more interesting: it's implementing a continuous-time portfolio rebalancing strategy in discrete steps.

Consider: you hold outcome X from a previous rebalance at profitability 0.5. The current rebalance's waterfall reaches π_last = 0.3. Phase 3 sells X down to profitability 0.3, then reallocates that capital through the waterfall. But the waterfall might push π_last down further (more capital = lower achievable profitability). In the limit of infinitely frequent rebalancing, this converges to a fixed point where all held positions have equal profitability.

The design choice to sell only down to π_last rather than fully liquidating is important. Full liquidation of a large position incurs massive price impact (the P/(1+mκ)² term), and then rebuying the same outcome through the waterfall incurs impact again. The round-trip cost is proportional to κ² × position_size², which can exceed the profitability gain from reallocation. Selling partially avoids this, at the cost of potentially leaving some capital in suboptimal positions.

This is a reasonable heuristic. The exact optimal would solve a joint problem — Phase 2 and Phase 3 simultaneously, accounting for the capital freed by liquidation affecting the waterfall's π*. But that's a fixed-point problem, and the iterative approach (waterfall → liquidate → waterfall) converges fast in practice.

## On the Route-Selection Abstraction

The treatment of (outcome, route) as separate entries in the waterfall is clean but hides a subtle non-convexity. The mint route for outcome i depends on all other pools' prices. When you mint for outcome A and sell into pools B, C, D, ..., you're changing the mint-route profitability for outcomes B, C, D. Two mint entries in the active set interfere with each other.

The code handles this through the skip set — active outcomes are excluded from each other's sell legs. But this means the mint cost for outcome A is computed assuming you *don't* sell into outcome B's pool (because B is also active). The actual cost would be different if B were not active. This is a valid approximation when the active set is small (2-3 entries), but could diverge when many outcomes are simultaneously active via mint.

Terry's write-up addresses this as "coupling simplification: all non-active pools see the same aggregate sell volume M." This is right — the collapse to scalar M is the key insight that makes the mixed solver tractable. But it's worth noting that this works because prediction markets have many outcomes (98) and few are simultaneously active, so the non-active set is large and dominates the sum. In a market with, say, 5 outcomes, the skip set could be a significant fraction and the approximation would be worse.

## What I'd Want to See Next

Three things:

**1. On-chain execution plan.** The algorithm outputs `Mint`/`Buy`/`Sell`/`Merge` actions. The current execution path submits strict subgroup transactions and replans after each receipt (see `docs/execution_submission.md`). Remaining question: should scheduling be single-subgroup strict mode only, or should some environments allow larger atomic bundles?

**2. MEV protection.** On Optimism, the sequencer is centralized (for now), which means there's no public mempool and no traditional sandwich risk. But the sequencer can still reorder transactions. If the rebalancer submits a large buy, a sophisticated sequencer could front-run it. Private mempools, commit-reveal, or simply using a trusted sequencer endpoint would mitigate this.

**3. Multi-block strategy.** The algorithm solves a single-period problem: given current prices, find optimal trades. But if you're going to rebalance every block (or every N blocks), the optimal single-period trade is not the same as the optimal trade in a multi-period strategy. Specifically, patient execution — splitting a large trade across multiple blocks — reduces price impact quadratically (impact scales with size², so two half-trades cost half as much as one full trade). The 3.6ms solve time leaves plenty of room for a multi-block planner that optimizes the execution schedule.

## Bottom Line

The mathematical framework here is genuinely excellent. The reduction to a scalar via KKT, the closed-form for the all-direct case, the nested Newton with analytical gradients — this is how you do applied optimization. The code matches the math, the tests verify the invariants, and the performance is more than adequate.

The gap is between the mathematical model and the on-chain execution environment. Gas costs, atomicity constraints, MEV, multi-block execution, slippage tolerance — these aren't mathematical complications, they're engineering constraints that change the optimal strategy. The algorithm answers "what should I trade?" The harder question is "how should I trade it?" and "when should I trade it?" Those are the next problems to solve.
