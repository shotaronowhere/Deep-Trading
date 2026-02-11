# Portfolio Rebalancing

## Overview

`src/portfolio.rs` computes optimal rebalancing trades for L1 prediction markets using exact swap simulation via Uniswap V3 math. Implements the waterfall allocation algorithm: deploy capital to the most profitable outcome, equalize profitability progressively with the next-best outcomes, then liquidate underperforming holdings and reallocate.

## Function: `rebalance`

```rust
pub fn rebalance(
    balances: &HashMap<&str, f64>,       // current outcome token balances
    susds_balance: f64,                   // available sUSD capital
    slot0_results: &[(Slot0Result, &'static MarketData)],  // pool state from fetch_all_slot0
) -> Vec<Action>
```

## Action Types

```rust
enum Action {
    Mint { contract_1, contract_2, amount, target_market },  // Mint across both L1 contracts
    Merge { contract_1, contract_2, amount, source_market }, // Buy others + merge to sell
    Buy { market_name, amount, cost },                       // Buy outcome tokens from pool
    Sell { market_name, amount, proceeds },                  // Sell outcome tokens for sUSD
    FlashLoan { amount },                                    // Borrow sUSD to fund minting
    RepayFlashLoan { amount },                               // Repay after selling minted tokens
}
```

Mint routes are wrapped in flash loans: `FlashLoan → Mint → Sell(others) → RepayFlashLoan`. Merge routes are similarly wrapped: `FlashLoan → Buy(all others) → Merge → RepayFlashLoan`. Both decouple the upfront capital requirement from the portfolio's liquid budget.

## Algorithm

### Phase 1: Sell overpriced holdings
For each outcome where market_price > prediction and we hold tokens:
- Optimize a split between **direct sell** (sell into pool) and **merge sell** (buy all other outcomes + merge complete sets for 1 sUSD each) to maximize total proceeds for the fixed sell amount.
- The split is solved as a 1D concave optimization (`f'(m)=0`) over merge amount `m`, with boundary checks (`m=0`, `m=cap`) and bisection on the interior root when needed.
- Merge consumes existing complementary holdings first, and buys only any shortfall from pools.
- Merge capacity is limited by complementary inventory plus buy caps (`held_j + max_buy_tokens_j`) across non-source outcomes; direct handles the remainder.
- Merge route only available when `mint_available` (all pools present).
- Pool state (`sqrt_price_x96`) updated after each sell for subsequent calculations.

### Phase 2: Waterfall allocation

#### Route pricing

Each outcome has up to two acquisition routes, each with its own price and profitability:

- **Direct buy**: current pool price from `sqrt_price_x96`. Profitability = `(prediction - direct_price) / direct_price`.
- **Mint** (when all outcomes have liquid pools): price = `1 - sum(other_outcome_prices)`. Profitability = `(prediction - mint_price) / mint_price`.

Route availability:
- `mint_available = sims.len() == PREDICTIONS_L1.len()` — true when every tradeable outcome (98) has a non-zero-liquidity pool
- When `mint_available`: mint route competes with direct; merge sell route competes with direct sell
- Partial prediction sets are not supported — `build_sims` panics if any outcome is missing a prediction

#### Dual-route waterfall

The waterfall treats each (outcome, route) pair as a separate entry. The same outcome can appear twice — once for direct, once for mint — with different profitabilities. This ensures both price channels are fully exploited.

**Dynamic entry selection:** Instead of pre-sorting entries by profitability, the waterfall calls `best_non_active()` each iteration to find the highest-profitability non-active entry from **current** pool state. This handles mint-route coupling: when a mint sell perturbs other pools' prices, the next entry selection reflects the updated state automatically.

**Monotonicity guard:** If a mint pushes a non-active entry's profitability *above* `current_prof`, it is absorbed into the active set immediately (no cost step) before the next descent. This prevents the water level from moving upward, which would break `solve_prof`'s assumptions.

**Ranking example:** Outcome A has direct price 0.05 (prof = 1.0) and mint price 0.03 (prof = 2.33). Outcome B has direct prof = 0.6.

| Entry | Route | Prof |
|-------|-------|------|
| A | mint | 2.33 |
| A | direct | 1.0 |
| B | direct | 0.6 |

**Waterfall steps:**
1. Deploy to A_mint until its profitability drops to 1.0 (matches A_direct)
2. Deploy to both A_mint and A_direct until profitability drops to 0.6 (matches B_direct)
3. Deploy to A_mint, A_direct, and B_direct until budget exhausted or no profitability remains
4. When budget is insufficient for a full step, `solve_prof` finds the achievable profitability level (closed-form for all-direct, Newton for mixed)

**Route interactions:** The two routes for the same outcome affect different state. Direct buys move the outcome's pool `sqrt_price_x96`. Mint sells push non-target pool prices down (increasing alt price). They are independent price channels, but both drain profitability for the same outcome from different directions.

#### Cost computation

Cost per step uses exact swap simulation:
- **Direct route**: `compute_swap_step` to target price derived from target profitability. Target price = `prediction / (1 + target_prof)`.
- **Mint route**: Newton's method finds the mint amount where the alt price (1 - sum of post-sell prices) equals the target price. Warm-started with the linearized first step: m₀ = (P_target - alt(0)) / (2 × Σ Pⱼκⱼ), saving one iteration. Uses the analytical price sensitivity parameter κⱼ = (1-fee) × √priceⱼ × 1e18 / Lⱼ per pool. Each pool's sell volume is capped at `max_sell_tokens()` = (√(price/limit_price) - 1) / κ — the max tokens sellable before hitting the tick boundary. In the Newton solver, each pool's price contribution uses min(m, capⱼ), and the derivative is zero for saturated pools. Net cost uses capped proceeds: proceedsⱼ(m) = priceⱼ × min(m, capⱼ) × (1-fee) / (1 + min(m, capⱼ) × κⱼ). This ensures the analytical estimate matches the tick-bounded execution in `emit_mint_actions`. Active outcomes are excluded from sell simulation.
- The caller specifies the route per entry — `direct_cost_to_prof` or `mint_cost_to_prof` is called directly based on the entry's route, rather than picking the best route internally.
- Negative costs (arbitrage, where sell proceeds exceed mint cost) are valid and increase the budget.

#### Budget exhaustion solver (`solve_prof`)

When the total cost to bring all active entries to the next profitability level exceeds the remaining budget:

**All-direct case (closed form):** Within a single tick range, the cost to buy is `L_eff × (√target_price - √current_price)` where `L_eff = L / (1e18 × (1-fee))`. Setting the sum equal to budget and solving for target profitability π:
- A = Σᵢ L_eff_i × √pred_i, B = budget + Σᵢ L_eff_i × √price_i
- **π = (A/B)² - 1** — zero iterations, exact solution.

**Mixed routes (coupled (π, M) Newton):** When mint routes are in the active set, skip semantics collapse inter-mint coupling to a scalar: all non-active pools see the same aggregate sell volume M = Σ mᵢ, and active pool prices are unperturbed by mints. This reduces the problem to 2 unknowns (π, M).

The solver partitions outcomes into D\* (direct-active ∪ binding mint target i\*), Q' (mint-active-only, pool price frozen), and N (non-active, sold into by mints). Key equations:
- **Alt-price constraint** (defines M for given π): `ΔG(M) = δ(π)`, where `δ(π) = Π*/(1+π) - (1-S₀)`, `Π* = Σ_{D*} predⱼ`, `S₀ = Σ_{j∉D*} P⁰ⱼ`
- **Budget constraint**: `Σ_{D} d_j(π) + C_mint(M(π)) = B`

Outer Newton on π (up to 15 iterations), inner Newton on M (2-3 iterations per outer step). Returns `(profitability, aggregate_M)`. Waterfall executes mints first (M split equally among entries), then directs using post-mint pool state. See `improvements.md` for full derivation and review history.

#### Execution safeguards

- Each outcome's cost is recomputed right before execution (not from stale precomputed values), since mint actions mutate other pools' state.
- Per-action budget guard skips actions where cost exceeds remaining budget. Negative costs (arbitrage) are executed — they increase the budget and update pool states. If any outcome is skipped during a step (recomputed cost diverged from estimate), the waterfall breaks early and `last_prof` is set to `current_prof` (the level actually achieved), not the target. This prevents Phase 3 from using a stale profitability threshold.
- **Prune loop:** when entries fail cost computation (e.g. tick boundary hit), the active set is pruned in a loop that re-derives the skip set after each removal, so remaining entries always see a consistent skip set.
- **Iteration cap:** the waterfall loop is bounded by `MAX_WATERFALL_ITERS` (1000) to prevent infinite cycling from negative-cost arbitrage that grows the budget.
- `sim_balances` is updated after waterfall by processing all four action types: `Mint` adds `amount` to all outcomes (complete sets), `Merge` subtracts `amount` from all outcomes, `Sell` subtracts sold amount, `Buy` adds directly. This correctly tracks residual unsold tokens when partial sells hit tick boundaries.
- `skip` in mint sell legs uses the set of all active outcome **indices** (not route-specific). If outcome A is active via both direct and mint, its index appears once in `skip` — preventing sell-then-rebuy regardless of which route triggered the mint.

### Phase 3: Post-allocation liquidation
After the waterfall:
1. Check all held outcomes' profitability against the last profitability level reached. Profitability is computed using the **direct pool price** (`sims[i].price()`), not `best_route`, because this step is a valuation threshold based on immediate pool-exit value.
2. For holdings with profitability below that level (lowest first), sell only enough to raise profitability to match `last_bought_prof` (avoids round-trip churn from full liquidation). For each such sale, use the same direct/merge split optimizer as Phase 1.
3. Reallocate recovered capital via another waterfall pass

## Two Market Contracts

L1 has 2 contracts (67 + 33 outcomes). Minting a complete set requires:
1. Mint on contract 1 (costs 1 sUSD) → produces 67 tokens including "other repos"
2. Use "other repos" token as collateral to mint on contract 2 → produces 33 tokens

When using the mint route, `Action::Mint` encodes both contracts (derived from distinct `market_id` values in the `sims` array), and sell actions are emitted for non-target, non-active outcomes across both contracts. `Action::Merge` similarly derives contracts from `sims`. During waterfall allocation, outcomes currently being acquired (the "active" set) are skipped in mint sell legs to avoid wasteful sell-then-rebuy cycles.

## Pool Simulation (`PoolSim`)

Each outcome's pool state is tracked mutably during the rebalance:
- `sqrt_price_x96`: updated after each simulated trade
- Swap direction determined by token ordering (`zero_for_one_buy`)
- Tick boundary clamping ensures trades stay within the pool's liquidity range

### Sell-side methods
- `kappa()`: κ = (1-f) × √price × 1e18 / L — price sensitivity for selling
- `max_sell_tokens()`: max sellable before tick boundary — (√(price/limit_price) - 1) / κ
- `sell_proceeds(m)`: proceeds = price × m × (1-f) / (1 + mκ)

### Buy-side methods
- `lambda()`: λ = √price × 1e18 / L — price sensitivity for buying (dual of κ)
- `max_buy_tokens()`: max buyable before tick boundary — (1 - √(price/buy_limit_price)) / λ
- `buy_exact(amount)`: returns (actual, cost, new_price). Cost = m × price / ((1-f)(1-mλ)). Price after buying: P/(1-mλ)²

## Balance Reading & Caching

`src/pools.rs` provides on-chain balance fetching and local caching:

```rust
use deep_trading_bot::pools::{fetch_balances, save_balance_cache, load_balance_cache, cache_to_balances};

// Fetch on-chain balances via Multicall3
let (susds, balances) = fetch_balances(provider, wallet_address).await?;

// Cache to disk
save_balance_cache(Path::new("balance_cache.json"), &wallet_hex, susds, &balances)?;

// Load from cache (returns None if stale/missing/wrong wallet)
if let Some(cache) = load_balance_cache(Path::new("balance_cache.json"), &wallet_hex, None) {
    let balances = cache_to_balances(&cache);
}
```

## Performance

`test_rebalance_perf_full_l1` benchmarks the full rebalance across all 98 tradeable L1 outcomes using real market/tick/liquidity data from `MARKETS_L1` with synthetic slot0 prices (50% of prediction, creating buy opportunities for every outcome).

| Mode | Per call (98 outcomes) |
|------|-----------------------|
| Release | ~3.6ms |

Run with: `cargo test --release test_rebalance_perf_full_l1 -- --nocapture`

## Usage

```rust
use deep_trading_bot::pools::{fetch_all_slot0, fetch_balances};
use deep_trading_bot::portfolio::{rebalance, Action};

let slot0_results = fetch_all_slot0(provider.clone()).await?;
let (susds, balances) = fetch_balances(provider, wallet).await?;
let actions = rebalance(&balances, susds, &slot0_results);

for action in &actions {
    match action {
        Action::Mint { amount, target_market, .. } => {
            println!("MINT {} sets for {}", amount, target_market)
        }
        Action::Merge { amount, source_market, .. } => {
            println!("MERGE {} sets from {}", amount, source_market)
        }
        Action::Buy { market_name, amount, cost } => {
            println!("BUY {} {} (cost: {})", amount, market_name, cost)
        }
        Action::Sell { market_name, amount, proceeds } => {
            println!("SELL {} {} (proceeds: {})", amount, market_name, proceeds)
        }
    }
}
```
