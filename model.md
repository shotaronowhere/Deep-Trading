# Mathematical Model

## Optimization Problem

Maximize expected portfolio value subject to a budget constraint:

```
max  Σᵢ πᵢ × hᵢ
s.t. Σᵢ cᵢ(hᵢ) ≤ B
     hᵢ ≥ 0
```

where πᵢ is the prediction (probability) for outcome i, hᵢ is holdings acquired, cᵢ(hᵢ) is the cost to acquire hᵢ tokens, and B is the sUSD budget.

The cost function cᵢ is nonlinear due to AMM price impact. The solution structure is a **waterfall**: equalize marginal profitability across all purchased outcomes. This is the KKT condition for the constrained optimum — at the solution, the marginal expected return per unit cost is equal across all active positions.

## AMM Price Model (Single Tick Range)

Uniswap V3 within one tick range is a constant-product AMM with concentrated liquidity L. Define:

- ρ = sqrtPriceX96 / 2⁹⁶ (normalized sqrt price)
- For token1 = outcome: outcome price P = 1/ρ²
- For token0 = outcome: outcome price P = ρ²

The token deltas for a price move are:

```
Δtoken0 = L × (1/ρ_lower - 1/ρ_upper)
Δtoken1 = L × (ρ_upper - ρ_lower)
```

### Universal Cost Formula

For **both** token orderings, buying outcome tokens to move price from P₀ to P₁ > P₀ costs:

```
cost = L × (√P₁ - √P₀) / (1 - f)
```

in raw token units (divide by 10¹⁸ for f64). This holds because:
- When outcome = token1: quote input = L × (1/ρ₁ - 1/ρ₀) and 1/ρ = √P
- When outcome = token0: quote input = L × (ρ₁ - ρ₀) and ρ = √P

Both collapse to L × (√P₁ - √P₀). Fee f (= 0.0001 for 1bp) is taken from input.

Define effective liquidity: **L_eff = L / (10¹⁸ × (1-f))**, then cost in f64 = L_eff × (√P₁ - √P₀).

### Price Sensitivity (κ)

When selling m tokens of outcome into a pool, the new price is:

```
P(m) = P₀ / (1 + m × κ)²
```

where **κ = (1-f) × √P₀ × 10¹⁸ / L**.

Derivation: selling m tokens (with fee taken from input) moves ρ by m_eff/L where m_eff = m × (1-f) × 10¹⁸. For token1 outcome, new ρ₁ = ρ₀ + m_eff/L, so new √P = 1/ρ₁ = √P₀ × L/(L + m_eff × √P₀) = √P₀/(1 + m×κ). Squaring gives P(m). The token0 case yields the same via the analogous formula.

Sell proceeds:

```
proceeds(m) = P₀ × m × (1-f) / (1 + m × κ)
```

### Tick Boundary Cap

Each pool has a finite tick range. The max sellable tokens before hitting the tick boundary:

```
M = (√(P₀ / P_limit) - 1) / κ
```

where P_limit is the outcome price at `sqrt_price_sell_limit`. When selling m > M tokens, the pool saturates: price pins at P_limit and proceeds cap at proceeds(M). This is used in mint cost estimation — each non-target pool's contribution is evaluated at min(m, Mⱼ).

## Closed-Form Budget Allocation (All-Direct)

When all active outcomes use the direct buy route, the waterfall's budget-exhaustion step has an exact solution.

**Problem**: find profitability level π such that

```
Σᵢ L_eff_i × (√(πᵢ/(1+π)) - √Pᵢ) = B
```

where πᵢ is prediction, Pᵢ is current price, B is budget.

**Solution**: define

```
A = Σᵢ L_eff_i × √πᵢ
B' = B + Σᵢ L_eff_i × √Pᵢ
```

Then A/√(1+π) = B', giving:

```
π = (A/B')² - 1
```

No iteration required.

## Newton's Method: Mint Route

### Mint Amount for Target Profitability

Minting m complete sets and selling all non-target outcomes yields exposure to outcome i at net cost:

```
net_cost(m) = m - Σⱼ≠ᵢ proceeds_j(m)
            = m × (1 - Σⱼ≠ᵢ Pⱼ(1-f) / (1 + m×κⱼ))
```

The effective acquisition price (alt price after selling) is:

```
alt(m) = 1 - Σⱼ≠ᵢ Pⱼ / (1 + m×κⱼ)²
```

To find m such that alt(m) = P_target, solve g(m) = rhs where:

```
g(m) = Σⱼ Pⱼ / (1 + min(m, Mⱼ)×κⱼ)²
rhs  = 1 - P_target
```

Newton step: m ← m - (g(m) - rhs) / g'(m), with:

```
g'(m) = Σⱼ { -2×Pⱼ×κⱼ/(1+m×κⱼ)³  if m < Mⱼ,  0  if m ≥ Mⱼ }
```

Each pool's price contribution freezes at P_limit once m exceeds that pool's sell cap Mⱼ (derivative = 0). Warm-started with the linearized first Newton step: m₀ = (P_target - alt(0)) / (2 × Σ Pⱼκⱼ). Converges reliably in 2-3 iterations.

**Fixed**: `rhs = (1 - P_target) - Σ_{j∈skip, j≠target} P⁰_j` correctly accounts for skip pool prices frozen at their spot values.

### Mixed-Route Budget Solver

When the active set contains both direct and mint routes, Newton's method solves for π:

```
total_cost(π) = Σᵢ∈direct L_eff_i × (√(πᵢ/(1+π)) - √Pᵢ) + Σⱼ∈mint mint_cost_j(π) = B
```

Both routes have analytical derivatives:

```
d(cost_direct)/dπ = -L_eff × √pred / (2 × (1+π)^(3/2))
```

For mint routes, the derivative uses the implicit function theorem on the constraint g(m) = 1 - P_target:

```
dm/dπ = P_target / ((1+π) × g'(m))
d(net_cost)/dm = 1 - (1-f) × Σⱼ∈uncapped Pⱼ(m)
d(net_cost)/dπ = d(net_cost)/dm × dm/dπ
```

Both derivatives are computed in the same pass as the cost (no finite differences).

**Fixed**: `solve_prof` now uses a coupled (π, M) Newton solver. All non-active pools see aggregate sell volume M (not individual m_i). Inner Newton solves ΔG(M) = δ(π) for M at each outer step. Returns (profitability, aggregate_M). Waterfall executes mints first (M/|Q| per entry), then directs.

**Known approximation**: The binding mint target `i*` (the mint entry whose alt-price constraint is tightest) is selected once at `prof_hi` and held fixed throughout Newton iterations. If the active set contains 2+ mint entries with sufficiently different prediction/spot gaps, the true binding constraint could switch as π moves, causing slightly inaccurate profitability. In practice this is rare (waterfall adds one entry at a time) and the execute-then-check loop prevents budget overshoot.

## Profitability and Target Price

Profitability of outcome i at price P:

```
prof(P) = (πᵢ - P) / P
```

Target price for a given profitability level π:

```
P_target = πᵢ / (1 + π)
```

The waterfall equalizes prof across all active outcomes. At the optimum, every purchased outcome has the same marginal profitability — spending one more unit on any of them yields equal expected return.

## Implementation Map

| Math | Code | Method |
|------|------|--------|
| L_eff | `PoolSim::l_eff()` | L / (10¹⁸ × (1-f)) |
| κ | `PoolSim::kappa()` | (1-f) × √P × 10¹⁸ / L |
| M (sell cap) | `PoolSim::max_sell_tokens()` | (√(P/P_limit) - 1) / κ |
| π = (A/B')² - 1 | `solve_prof` (direct branch) | Closed form |
| g(m) = rhs | `mint_cost_to_prof` | Newton, ≤8 iter, tick-capped, warm-started |
| total_cost(π) = B | `solve_prof` (mixed branch) | Newton, ≤8 iter, analytical gradients |
| cost(P₀→P₁) | `PoolSim::cost_to_price` | `compute_swap_step` (U256) |
| proceeds(m) | analytical in `mint_cost_to_prof` | P×min(m,M)×(1-f)/(1+min(m,M)×κ) |

The f64 analytical math drives the optimization (choosing what to trade). The U256 `compute_swap_step` drives execution (computing exact trade amounts with full precision). The two agree within f64 rounding for single-tick-range pools.

## Invariants

- **Full prediction coverage**: every tradeable outcome must have a prediction. `build_sims` panics on mismatch. This ensures the mint route's cost model (1 - Σ proceeds) accounts for all minted tokens — no unsold residuals or untracked holdings.
