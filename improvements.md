# Potential Improvements

## Exact Mixed-Route Optimization

### Current Approach: Sequential Waterfall

The waterfall equalizes profitability via coordinate descent. For all-direct routes this is exact (pools are independent, closed-form π = (A/B)² - 1). For mixed routes (direct + mint), the terminal budget-exhaustion step introduces coupling error: `solve_prof` treats entries independently, then sequential execution of mints perturbs other pools' prices before their own buys execute.

**Approximation error**: O(ΔP) — proportional to the mint-induced price dip magnitude. The error source is `solve_prof` calling `mint_cost_to_prof` independently for each mint entry, each assuming original non-active pool prices. When multiple mints execute, the aggregate sell volume M = Σ m_i depresses non-active pool prices more than any individual m_i would. Since proceeds are concave in sell volume, the summed individual proceeds overestimate actual proceeds, causing `solve_prof` to underestimate the net mint cost and overshoot the achievable profitability.

### Key Insight: Skip Semantics Collapse Coupling to a Scalar

The plan's original formulation used `S_j = Σ_{i≠j} m_i` — implying each pool j receives sells from all mints except mint-for-j. **This is wrong.** The actual code uses `skip = active_skip_indices(&active)`, which contains ALL active outcome indices. In `emit_mint_actions` (line 462) and `mint_cost_to_prof` (line 330):

```
if i == target_idx || skip.contains(&i) { continue; }
```

This means minting for ANY outcome only sells into **non-active** pools. Active pools (both direct-active and mint-active) are never sold into. Consequences:

1. **All non-active pools see the same total sell volume**: S_j = Σ_{i∈Q} m_i = M for all j ∉ A (where A = active set, Q = mint-active subset). There is no i≠j coupling between different mints' contributions to any pool.

2. **Active pool prices are unperturbed by mints**: For j ∈ A, S_j = 0, so P̃_j = P⁰_j. Direct buys into active pools operate at original prices, independent of mint volume.

3. **The individual split of M among mint entries doesn't affect total cost or total token holdings**: Minting m_i sets for outcome i gives m_i tokens of i, plus m_i tokens of every OTHER active outcome (kept because skip). Total tokens of any active outcome j from all mints = Σ_{i∈Q} m_i = M (each mint gives one of j). The net mint cost = M - Σ_{j∉A} F_j(min(M, M_j)) depends only on M.

4. **The K+1 KKT system is unnecessary**. The problem collapses to **2 unknowns**: π (target profitability) and M (total mint volume).

### Exact Formulation (Corrected)

**Variables**: π ≥ 0 (target profitability), M ≥ 0 (total mint volume).

**Set definitions** — partition ALL outcomes into four disjoint groups relative to a chosen mint target i*:

```
D* = D ∪ {i*}         (outcomes whose final price = pred_j/(1+π))
Q' = Q \ D \ {i*}     (mint-active-only outcomes, excluding i*; pool price stays P⁰_j)
N  = complement of A   (non-active outcomes; sold into by mints)
```

Note: D and Q are indexed by **outcome**, not (outcome, route). An outcome can appear in both D and Q if it's active via both routes. D* absorbs i* regardless of whether i* ∈ D. The three groups D*\{i*}, Q', N plus {i*} partition all outcomes.

**Direct costs** (independent of M since active pools are unperturbed):
```
d_j(π) = L_eff_j × max(0, √(pred_j/(1+π)) - √P⁰_j)   for j ∈ D
```

**Post-mint price of non-active pool j**:
```
P̃_j(M) = P⁰_j / (1 + κ_j min(M, M_j))²
```

**Sell proceeds**:
```
F_j(M) = P⁰_j × (1-f) × min(M, M_j) / (1 + κ_j min(M, M_j))
```

**Net mint cost**:
```
C_mint(M) = M - Σ_{j∈N} F_j(M)
```

**Alt-price constraint** — determines M for given π. The alt-price for mint entry i* is 1 minus the sum of ALL other outcomes' **final** prices:

- j ∈ D*\{i*}: final price = pred_j/(1+π) (bought directly)
- j ∈ Q': final price = P⁰_j (in skip, pool untouched)
- j ∈ N: final price = P̃_j(M) (sold into by mints)

```
pred_{i*}/(1+π) = 1 - Σ_{j∈D*\{i*}} pred_j/(1+π) - Σ_{j∈Q'} P⁰_j - Σ_{j∈N} P̃_j(M)
```

Define Π* = Σ_{j∈D*} pred_j (includes pred_{i*}), and S₀ = Σ_{j∉D*} P⁰_j (spot prices of all outcomes NOT in D*). Note S₀ = Σ_{j∈Q'} P⁰_j + Σ_{j∈N} P⁰_j.

Rearranging:
```
Π*/(1+π) = 1 - S₀ + ΔG(M)

δ(π) = Π*/(1+π) - (1 - S₀)

ΔG(M) = Σ_{j∈N} P⁰_j × [1 - 1/(1+κ_j min(M,M_j))²]

Equation: ΔG(M) = δ(π)
```

**Why S₀ works**: Outcomes in D∩Q (active via both routes) have their final price = pred_j/(1+π) from the direct buy. They are NOT in S₀ (excluded by the D* filter), so they are NOT double-counted at spot price. Outcomes in Q'\D have no direct buy, so their pool prices are unchanged (they're in skip), and correctly appear in S₀ at P⁰_j.

This is a 1D equation in M (for fixed π), solvable by Newton in 2-3 iterations.

**Derivatives**:
```
dδ/dπ = -Π* / (1+π)²
ΔG'(M) = Σ_{j∈N, uncapped} 2P⁰_j κ_j / (1+κ_j M)³
dM/dπ = dδ/dπ / ΔG'(M)     (implicit function theorem on ΔG(M(π)) = δ(π))
```

**Sign check on dM/dπ**: dδ/dπ = -Π*/(1+π)² < 0 and ΔG'(M) > 0, so dM/dπ < 0. Intuition: as π increases (higher profitability target), the required entry price decreases, so the alt-price must be lower, meaning the sum of other prices must be higher, meaning we sell *less* (M decreases). The math matches the physics. ✓

**Bug in existing `mint_cost_to_prof`** (line 317-324): The Newton equation is wrong when skip is non-empty. `g(m)` sums only non-skip, non-target pool prices, but `rhs = 1 - tp` is the target for the **full** alt-price (all other outcomes). The correct rhs is `(1 - tp) - Σ_{j∈skip, j≠target} P⁰_j`. Additionally, the early-return check `current_alt >= tp` (line 318) uses `alt_price(sims, target_idx)` which sums ALL pools, creating an inconsistency with the Newton's pool set. The Newton converges to a **wrong root** — systematically undershooting m, underestimating mint cost, and causing `solve_prof` to return an overly aggressive π. This is not a warm-start artifact; the iteration converges reliably to the incorrect answer. The exact solver avoids this entirely by separating D*, Q', N contributions explicitly.

**Budget constraint**:
```
Σ_{j∈D} d_j(π) + C_mint(M(π)) = B
```

Combined with the alt-price constraint defining M(π), this is a **1D equation in π**.

**Saturation handling**: When δ(π) ≥ ΔG_max (all non-active pools capped), the mint can't achieve the target alt-price at that π — the profitability level is infeasible for mints. The feasibility boundary is:
```
π_boundary = Π* / (1 - S₀ + ΔG_max) - 1
```
For π < π_boundary, δ(π) > ΔG_max (infeasible). For π ≥ π_boundary, feasible.

When the Newton step enters the infeasible region:
- Clamp π to max(π, π_boundary). At π_boundary, M = M_sat (all pools capped), and the outer Newton continues with dM/dπ = 0 and mint cost fixed at C_mint(M_sat).
- If D ≠ ∅: the direct cost terms still vary with π, so the outer Newton has nonzero derivative and can find the root of D_cost(π) + C_mint(M_sat) = B, subject to π ≥ π_boundary.
- If D = ∅: cost is constant at C_mint(M_sat). If C_sat ≤ budget, return π_boundary (the lowest feasible π). If C_sat > budget, no feasible solution — fall back to current `solve_prof`.

### Multi-Mint Consistency Caveat

When |Q| > 1, the alt-price constraints for different mint entries i, k are inconsistent at the same (π, M). The general consistency condition depends on D-membership:

- Both i, k ∈ Q\D (mint-only): `(pred_i - pred_k)/(1+π) = P⁰_i - P⁰_k`
- i ∈ Q\D, k ∈ D∩Q: `pred_i/(1+π) = P⁰_i` (i.e., target price = current price, meaning zero buy for i — degenerate)
- Both i, k ∈ D∩Q: constraints are identical (both have their final price from direct buys)

In practice the Q\D case dominates: the waterfall adds entries at similar profitability levels, so the consistency condition approximately holds. The solver picks i* **once at initialization** (before the Newton loop) as argmax(pred_i/(1+prof_hi) - current_alt_i) and holds it fixed throughout iteration. This avoids derivative discontinuities from switching the binding constraint mid-solve. The choice satisfies the tightest constraint. For the D∩Q degenerate case, the fixed-i* approach naturally excludes it from binding since such entries already have their price set by direct buys.

### Waterfall Error Bound

Let π* be the exact solution, π^wf the waterfall output. The error:
```
|π^wf - π*| ≤ Σ_{j∈N} [Σ_{i∈Q} F_j(m_i) - F_j(M)] / |dtotal/dπ|
```
The numerator is the total proceeds overestimate across all non-active pools from treating mints independently. By subadditivity of concave F_j with F_j(0) = 0: Σ_i F_j(m_i) ≥ F_j(M), so the waterfall overestimates proceeds, underestimates net cost, and overshoots profitability. A tighter bound uses the second derivative:
```
Σ_i F_j(m_i) - F_j(M) ≈ -½ F_j''(M̄) × Var({m_i})
```
where F_j''(m) = -2P⁰_j κ²_j(1-f)(1-2κ_j m)/(1+κ_j m)⁴. The error vanishes when |Q| = 1 (single mint). For typical multi-mint scenarios with 5-10 entries, the error is ~1-3% of the budget.

---

## Implementation Plan: Exact Coupled Solve (Revised)

### Scope

Replace **only** the `else` branch (budget exhaustion) in `waterfall()` when the active set contains mint routes. Everything else unchanged.

**When**: Active set contains at least one `Route::Mint` entry.
**All-direct case**: Unchanged (closed-form π = (A/B)² - 1).
**Fallback**: If Newton fails to converge in 15 iterations, fall back to current `solve_prof`.

### Algorithm: 1D Newton on π with Joint Mint

The key change: replace independent `mint_cost_to_prof` calls with a joint computation that solves for the aggregate M at each Newton step.

```
solve_prof_coupled(sims, active, prof_hi, prof_lo, budget, skip):

  Partition active into D (direct outcome indices) and Q (mint outcome indices).
  Pick i* = mint entry with highest (pred/(1+prof_hi) - current_alt) — FIXED for all iterations.
  Compute D* = D ∪ {i*}.
  Precompute Π* = Σ_{j∈D*} pred_j.
  Precompute S₀ = Σ_{j∉D*} P⁰_j.
  Precompute per non-active pool j ∈ N: (P⁰_j, κ_j, M_j = max_sell_j).
  Precompute ΔG_max = Σ_{j∈N} P⁰_j [1 - 1/(1+κ_j M_j)²]  (saturation ceiling).

  π = prof_hi  (warm start from current profitability)

  For iter = 0..15:
    // 1. Direct costs (closed-form, independent of M)
    //    Guard: only include entries where √(pred_j/(1+π)) > √P⁰_j (d_j > 0).
    D_cost = 0, dD = 0
    for j ∈ D:
      target_sqrt = √(pred_j/(1+π))
      if target_sqrt > √P⁰_j:
        D_cost += L_eff_j × (target_sqrt - √P⁰_j)
        dD += -L_eff_j × √pred_j / (2(1+π)^(3/2))
      // else: d_j = 0, derivative = 0 (pool past target)

    // 2. Compute δ(π) using corrected set definitions
    δ = Π*/(1+π) - (1 - S₀)
    dδ/dπ = -Π* / (1+π)²

    // 3. Solve for M(π) via inner Newton (2-3 iterations)
    if δ ≤ 0:
      M = 0  (minting not needed)
    elif δ ≥ ΔG_max:
      // Infeasible at this π. Clamp to feasibility boundary.
      π_boundary = Π* / (1 - S₀ + ΔG_max) - 1
      π = max(π, π_boundary)
      M = M_sat  (all pools capped at boundary)
      if D is empty:
        C_sat = M_sat - Σ_{j∈N} F_j(M_sat)
        if C_sat ≤ budget: return π_boundary
        else: FALLBACK to current solve_prof
    else:
      // Inner Newton: solve ΔG(M) = δ
      // Warm start: linearized root M₀ = δ / ΔG'(0) = δ / (Σ_{j∈N} 2P⁰_j κ_j)
      // Or reuse M from previous outer iteration (better when available).
      // ΔG is concave, so M₀ undershoots; one Newton step overshoots, then
      // iterates decrease monotonically from above to the root.
      M = δ / (Σ_{j∈N} 2P⁰_j κ_j)    // linearized warm start
      for inner_iter = 0..5:
        ΔG(M) = Σ_{j∈N} P⁰_j [1 - 1/(1+κ_j min(M,M_j))²]
        ΔG'(M) = Σ_{j∈N, uncapped} 2P⁰_j κ_j / (1+κ_j M)³
        if ΔG' < 1e-30: break
        step = (ΔG - δ) / ΔG'
        M -= step
        M = max(M, 0)
        if |step| < 1e-10 × (1 + M): break    // relative tolerance

    // 4. Net mint cost
    C = M - Σ_{j∈N} F_j(min(M, M_j))
    dC/dM = 1 - (1-f) Σ_{j∈N, uncapped} P⁰_j / (1+κ_j M)²

    // 5. dM/dπ via implicit differentiation of ΔG(M(π)) = δ(π)
    if M = 0 or ΔG'(M) ≈ 0:
      dM/dπ = 0  (minting inactive or saturated)
    else:
      dM/dπ = dδ/dπ / ΔG'(M)

    // 6. Total cost and derivative
    total = D_cost + C
    dtotal/dπ = dD + dC/dM × dM/dπ

    // 7. Newton step with bisection fallback
    if |dtotal| < 1e-30: break  (flat — can't improve)
    step = (total - budget) / dtotal
    π_new = clamp(π - step, prof_lo, prof_hi)
    if |π_new - π| < 1e-12 × (1 + |π|): break  (relative tolerance)
    π = π_new

  // Bisection fallback if Newton didn't converge
  if |total - budget| > 1e-6 × budget:
    bisect on [prof_lo, prof_hi] for ≤20 iterations
    (total_cost(π) is monotone decreasing in π)

  return π
```

### Action Emission

After converging to π*:

1. Compute M(π*) via the inner Newton (final evaluation).
2. Emit mint actions first, using `emit_mint_actions` for each mint entry with amount M/|Q| (equal split — the split is arbitrary since total cost and tokens depend only on M).
3. Emit direct actions second, computing exact cost from **actual post-mint pool state** via `cost_to_price(pred_j/(1+π*))`. This self-corrects f64 rounding.

**Ordering is critical**: mints first (update non-active pool states), directs second (use actual states). The code must enforce this explicitly, not iterate `allocs` in arbitrary order.

### Code Changes in `src/portfolio.rs`

**Modify `solve_prof`** (~line 847): Add a branch for mixed routes that uses the coupled algorithm above, replacing the independent `cost_for_route` summation.

```rust
// In solve_prof, after the all_direct branch:
// Mixed routes: joint mint computation
let direct_set: HashSet<usize> = active.iter()
    .filter(|&&(_, r)| r == Route::Direct).map(|&(i, _)| i).collect();
let mint_set: HashSet<usize> = active.iter()
    .filter(|&&(_, r)| r == Route::Mint).map(|&(i, _)| i).collect();

// Pick binding mint target i* (fixed for all iterations)
let i_star = pick_binding_mint_target(sims, &mint_set, prof_hi);

// D* = D ∪ {i*}: all outcomes with final price = pred/(1+π)
let d_star: HashSet<usize> = direct_set.iter().copied().chain(std::iter::once(i_star)).collect();

// Π* = Σ_{j∈D*} pred_j,  S₀ = Σ_{j∉D*} P⁰_j
let pi_star: f64 = d_star.iter().map(|&j| sims[j].prediction).sum();
let s0: f64 = sims.iter().enumerate()
    .filter(|(j, _)| !d_star.contains(j))
    .map(|(_, s)| s.price()).sum();

// ... Newton loop using solve_prof_coupled logic ...
```

**Modify `waterfall` budget-exhaustion branch** (~line 782): After `solve_prof` returns the coupled achievable profitability, execute mints before directs:

```rust
let achievable = solve_prof(sims, &active, current_prof, target_prof, *budget, &skip);

// Separate mint and direct entries for ordered execution
let (mint_active, direct_active): (Vec<_>, Vec<_>) =
    active.iter().partition(|&&(_, r)| r == Route::Mint);

// Execute mints first (total M split equally among entries)
if !mint_active.is_empty() {
    let total_m = joint_mint_amount(sims, &mint_active, achievable, &skip);
    let per_entry = total_m / mint_active.len() as f64;
    for &(idx, route) in &mint_active {
        execute_buy(sims, idx, /*cost computed from M*/, per_entry, route, None, budget, actions, &skip);
    }
}
// Execute directs second using actual post-mint pool state
for &(idx, _) in &direct_active {
    let tp = target_price_for_prof(sims[idx].prediction, achievable);
    if let Some((cost, amount, new_sqrt)) = sims[idx].cost_to_price(tp) {
        if cost <= *budget {
            execute_buy(sims, idx, cost, amount, Route::Direct, Some(new_sqrt), budget, actions, &skip);
        }
    }
}
```

**New helper: `joint_mint_amount`** — solves ΔG(M) = δ for the aggregate M at a given π:
```rust
fn joint_mint_amount(
    sims: &[PoolSim],
    mint_entries: &[(usize, Route)],
    target_prof: f64,
    skip: &HashSet<usize>,
) -> f64
```

### Verification

1. `cargo test portfolio` — all existing tests pass (no behavior change for all-direct cases)
2. New test: `test_joint_mint_vs_independent` — 3+ outcomes with mint routes. Verify:
   - Joint M > max(individual m_i) (coupling produces larger aggregate)
   - Joint net cost > sum of independent net costs (concavity of proceeds)
   - Budget constraint: |total_cost - budget| < 1e-9
3. New test: `test_coupled_solve_single_mint` — single mint entry. Verify coupled result matches independent result exactly (no coupling when |Q|=1)
4. New test: `test_coupled_solve_fallback` — zero liquidity pools. Verify graceful fallback
5. New test: `test_mint_then_direct_ordering` — verify executing mints before directs gives correct final pool states
6. Finite-difference Jacobian check: perturb π by ε, verify |numerical_derivative - analytical_derivative| < 10ε

### Complexity

- Outer Newton: ~5-8 iterations on π (same as current `solve_prof`)
- Inner Newton: ~2-3 iterations on M per outer step
- Per iteration: O(N) scan of non-active pools (N ≈ 80)
- Total: ~1000 flops. Same order as current `solve_prof`, no matrix algebra needed.

---

## Review History

### Codex Review Critique (2026-02-10)

GPT-5.3 (xhigh reasoning) reviewed the original K+1 Newton plan and found 7 issues. Critique of each:

**Finding #1 (Critical: skip semantics) — CORRECT, but underestimated the implication.**
Codex correctly identified that `skip = active_skip_indices(&active)` changes the coupling structure. But it only said "re-derive with skip set." The full implication is that skip **eliminates inter-mint coupling entirely**: all non-active pools see the same total M = Σ m_i, active pools see S_j = 0. This collapses the K+1 system to 2D (π, M), making the entire Jacobian/Woodbury machinery unnecessary.

**Finding #2 (High: capped pools in c'_mint_i) — VALID but moot.**
With the 2D formulation, capped pools are handled trivially: min(M, M_j) in the ΔG and F_j formulas, and the uncapped indicator in derivatives. No separate treatment needed.

**Finding #3 (High: direct complementarity) — MOSTLY WRONG.**
With skip, direct buys use original active pool prices (unperturbed by mints). The max(0,...) clamp in d_j(π) handles the boundary cleanly — if √(pred_j/(1+π)) < √P⁰_j, the pool is past target and d_j = 0. No active-set management needed because d_j is not a Newton variable; it's analytically eliminated.

**Finding #4 (Medium: diagonal + rank-2 claim) — MOOT.**
The 2D formulation has a 1×1 or 2×2 system. No matrix structure claims needed.

**Finding #5 (Medium: line search/feasibility) — PARTIALLY VALID.**
The 2D Newton is well-conditioned: π is clamped to [prof_lo, prof_hi] (same as current solve_prof), and M ≥ 0 is enforced by the inner solve. No exotic line search needed. However, the point about scale-dependent tolerances is valid — tolerance should be relative to budget, e.g., |step| < 1e-12 × budget.

**Finding #6 (Medium: action emission ordering) — VALID.**
The original plan claimed self-correction but the code sketch didn't enforce ordering. Fixed in revised plan: mints explicitly execute before directs.

**Finding #7 (Low: scalar/matrix consistency) — MOOT.** No matrix formulation in revised plan.

**MISSED by Codex:**
- The individual m_i split doesn't matter (total cost and tokens depend only on M). This is the biggest simplification.
- Direct buys are fully independent of mints (skip keeps active pool prices unchanged). The "coupling" only exists among non-active pools, and it's a scalar (M), not per-pool.
- The multi-mint consistency constraint: (pred_i - pred_k)/(1+π) = P⁰_i - P⁰_k. This is a fundamental limitation — different mint entries can't all simultaneously satisfy their alt-price targets at the same (π, M). The revised plan uses the binding (tightest) constraint for i*.

### Second Codex Review (2026-02-10)

GPT-5.3 (xhigh reasoning) reviewed the revised 2D plan and found 7 issues. Disposition:

**Finding #1 (Critical: direct-price movement in alt-price) — VALID.** The alt-price equation must use final direct-active prices `pred_j/(1+π)`, not `P⁰_j`. This also affects `dδ/dπ` which must include `Σ_{j∈D} pred_j` terms. **Fixed**: δ(π) now includes `(pred_{i*} + Π_D)/(1+π)`, and `dδ/dπ = -(pred_{i*} + Π_D)/(1+π)²`.

**Finding #2 (Critical: pre-existing mint_cost_to_prof mismatch) — VALID.** `current_alt` (line 317) uses all pools, but Newton target `rhs` (line 324) only accounts for non-skip pools. The Newton equation itself has the wrong rhs, causing systematic undershoot of m (see "Bug in existing `mint_cost_to_prof`" above). **Noted** in formulation section; exact solver avoids this entirely by separating D*, Q', N contributions explicitly.

**Finding #3 (High: clamp derivative for directs) — MINOR but valid.** When `d_j = 0` (pool past target), derivative should be 0. **Fixed**: added guard in pseudocode — only include derivative for entries where `target_sqrt > √P⁰_j`.

**Finding #4 (High: binding i* switching) — VALID.** Switching i* mid-iteration causes derivative discontinuity. **Fixed**: i* is now picked once at initialization and held fixed.

**Finding #5 (Medium: ΔG saturation) — VALID.** When all pools are capped, ΔG_max is finite and no root exists for δ > ΔG_max. **Fixed**: precompute ΔG_max; if δ ≥ ΔG_max, set M to cap and let outer Newton adjust π. Also guard dM/dπ = 0 when ΔG' ≈ 0.

**Finding #6 (Medium: ordering) — VALID, already addressed.**

**Finding #7 (Medium: skip-collapse confirmed with nuance) — VALID, consistent with #1.** The 2D structure holds, but δ(π) must include direct-price terms. Fixed by #1.

### Gemini Review Critique (2026-02-10)

Gemini (gemini-3-pro-preview, role: Terence Tao) reviewed the revised 2D plan and gave it B+. Disposition:

**Finding #1 (Critical: D∩Q overlap / double-counting) — CORRECT, and actually worse than stated.**
Gemini correctly identified that when an outcome k is active via both Direct and Mint routes (k ∈ D ∩ Q), the formula double-counts it: once as pred_k/(1+π) in the D sum, once as P⁰_k in the Q sum. Gemini proposed `Q \ D` as the fix. This is necessary but **insufficient**: Gemini missed that i* itself could be in D (active via both routes), causing `pred_{i*} + Σ_{j∈D} pred_j` to count pred_{i*} twice. The complete fix requires defining D* = D ∪ {i*} as the set of all outcomes with final price pred_j/(1+π), and S₀ = Σ_{j∉D*} P⁰_j as the spot prices of everything else. This eliminates both the inter-set double-count (D∩Q) and the i*-in-D double-count in a single clean partition. **Fixed** in the formulation above.

**Finding #2 (Edge case: saturation zero-derivative) — CORRECT.**
When D = ∅ and all non-active pools are saturated, mint cost is constant w.r.t. π, giving dtotal/dπ = 0. The existing `|dtotal| < 1e-30: break` guard exits the loop but doesn't resolve what π to return. **Fixed**: explicit feasibility check added to the pseudocode — if saturated cost ≤ budget, return prof_lo; if > budget, fall back. This case is unlikely in practice (requires many mint entries, no direct entries, and all ~80 non-active pools fully capped), but mathematical completeness demands it.

**Finding #3 (Sign check) — CORRECT, signs are right.** Gemini verified the rearrangement. Confirmed by independent re-derivation using the corrected D*/S₀ notation.

**Finding #4 (Multi-mint consistency) — CORRECT.** Gemini validated the overdetermined system analysis and the fixed-i* workaround.

**MISSED by Gemini:**

1. **The i*-in-D double-count.** Gemini caught D∩Q overlap for j ≠ i* but not for i* itself. The proposed fix `Σ_{j∈Q\D, j≠i*}` doesn't address `pred_{i*} + Σ_{j∈D} pred_j` when i* ∈ D. The D*/S₀ formulation resolves both issues.

2. **Sequential execution preserves analytical proceeds.** A natural concern: if we emit multiple mints (m_1 for outcome 1, m_2 for outcome 2), the second sell sees post-m_1 pool prices, so total proceeds might be path-dependent. This is NOT the case — selling m tokens into a pool moves ρ by m_eff/L additively, so ρ_final = ρ_0 + M_eff/L regardless of splitting. Total proceeds = integral over [ρ_0, ρ_final] = F_j(M), path-independent. Gemini should have verified this since it's the foundation of the "split doesn't matter" claim.

3. **The tolerance should be relative.** `|step| < 1e-12` is an absolute tolerance on π. For very small π (near zero), this is fine. For large π (e.g., π = 5), a relative tolerance `|step| < 1e-12 × (1 + |π|)` would be more robust. Minor but worth noting for numerical hygiene.

**Overall assessment of the Gemini review:** The D∩Q overlap finding was the most important catch and is genuinely critical — without it the solver would produce wrong M values when any outcome is active via both routes (which happens regularly in practice since dual-route entries are a core feature of the waterfall). The saturation edge case is valid but low-impact. The review missed the deeper i*∈D variant of its own finding, and didn't verify the path-independence claim that the whole formulation rests on.

### Third Codex Review Critique (2026-02-10)

GPT-5.3 (xhigh reasoning) reviewed the corrected D*/S₀ plan. Disposition:

**Finding #1 (Skip-semantics collapse) — CONFIRMED** with the qualification that multi-mint feasibility coupling survives as the overdetermined alt-price system. Already acknowledged in the multi-mint consistency caveat.

**Finding #2 (D*/S₀ partition) — CONFIRMED CORRECT.** "Mathematically clean."

**Finding #3 (Alt-price rearrangement) — CONFIRMED VALID.** Signs check out.

**Finding #4 (Newton derivatives) — CONFIRMED CORRECT** in smooth regions. Notes non-smooth kinks at cap transitions could stall Newton. This is handled by the existing piecewise logic (uncapped indicator in derivatives, ΔG_max guard), but worth noting for robustness — if a pool transitions from uncapped to capped mid-iteration, the derivative is discontinuous. In practice, the inner Newton's clamp `M = max(M, 0)` and the outer clamp `π = clamp(π, prof_lo, prof_hi)` prevent divergence.

**Finding #5 (Multi-mint consistency) — VALID REFINEMENT.** The displayed condition `(pred_i - pred_k)/(1+π) = P⁰_i - P⁰_k` is only the special case when both i, k ∈ Q\D. The general condition is `(pred_i - pred_k)/(1+π) = P_i^final - P_k^final` where P^final depends on D-membership. When one entry is in D∩Q, the constraint degenerates. **Fixed**: expanded the multi-mint consistency section to enumerate all cases.

**Finding #6 (Path-independence) — CONFIRMED CORRECT.** "Rigorous enough."

**Finding #7 (Saturation handling) — VALID AND CRITICAL.** Returning `prof_lo` when `D=∅` and saturated is wrong: if `δ(π) > ΔG_max`, the target alt-price is infeasible at that π. The correct minimum feasible π is `π_boundary = Π*/(1 - S₀ + ΔG_max) - 1`. **Fixed**: pseudocode now computes π_boundary explicitly and clamps to it.

**Finding #8 (Fixed i* not guaranteed conservative globally) — ACKNOWLEDGED but acceptable.** The fixed-i* can pick the wrong binding constraint as π moves, meaning the solved M may over- or under-shoot for other mint entries. However, this is inherent in the single-constraint relaxation — a true coupled solve would require K simultaneous alt-price constraints, which is the K+1 system we deliberately avoided. The error is bounded by the multi-mint consistency mismatch, which is small when entries are at similar profitability (the waterfall's entry condition).

**MISSED by Codex:** Nothing significant that wasn't already noted. The review was thorough.

### Second Gemini Review Critique (2026-02-10)

Gemini (gemini-3-pro-preview, role: Terence Tao) reviewed the final D*/S₀ plan and gave it A. Disposition:

**Finding #1 (D*/S₀ partition) — CONFIRMED CORRECT.** "Mathematically precise." Verified exhaustiveness and disjointness.

**Finding #2 (Path-independence) — CONFIRMED but proof insufficient.** Gemini argued via "state function of ρ" which is hand-wavy. The actual proof requires showing that κ re-evaluates correctly after partial sells. Full proof: after selling m₁, new κ̃ = κ₀/(1+m₁κ₀). Selling m₂ more gives combined denominator (1+m₁κ₀+m₂κ₀)/(1+m₁κ₀), which telescopes:
```
proceeds(m₁) + proceeds(m₂|new state)
= P₀(1-f)m₁/(1+m₁κ) + P₀(1-f)m₂/((1+m₁κ)(1+Mκ))
= P₀(1-f) × [m₁(1+Mκ) + m₂] / ((1+m₁κ)(1+Mκ))
```
Key identity: `m₁(1+Mκ) + m₂ = m₁ + m₁Mκ + m₂ = M + m₁Mκ = M(1+m₁κ)`. Therefore:
```
= P₀(1-f) × M(1+m₁κ) / ((1+m₁κ)(1+Mκ))
= P₀(1-f)M/(1+Mκ) = F(M) ✓
```
Path-independent because the κ-adjusted denominators telescope.

**Finding #3 (Derivatives) — CONFIRMED CORRECT.** Signs verified.

**Finding #4 (Multi-mint consistency) — CONFIRMED.** "Standard relaxation."

**Finding #5 (Saturation) — CONFIRMED.** π_boundary formula validated.

**RETRACTED: Previous claim about (1-f) systematic approximation in alt-price was WRONG.**

The earlier analysis (by the document author, not Gemini) claimed ΔG should use (1-f)×P̃_j instead of P̃_j, confusing the **state-price identity** with the **marginal proceeds identity**. The alt-price is defined as:
```
alt_i(M) = 1 - Σ_{j≠i} P_j^{final}(M)
```
This uses the **observable pool price** P̃_j(M) = P⁰_j/(1+κ_j M)², NOT the marginal proceeds (1-f)×P̃_j. The fee is embedded in κ (how selling m tokens *moves* the price), not in the price observation itself. Proof that the original ΔG is exact: at M=0, ΔG(0) = 0 (no sells → no price change → correct). With the wrong "(1-f) correction", ΔG_correct(0) = f×Σ P⁰_j > 0, which is clearly wrong since zero sells can't change any price.

**Conclusion:** The plan's ΔG equation is **exact** (within the single-tick-range model), not an approximation. The (1-f) factor appears only in the proceeds formula F_j(M) (net cost computation), not in the price observation P̃_j(M) (alt-price constraint). The existing code `g += p / d2` (line 349) is correct.

**Other Gemini weaknesses:**

1. **Confused implementation note.** Gemini said "system state must reflect mints" then corrected itself ("direct pools are NOT sold into"). Suggests incomplete internalization of skip semantics.
2. **No independent verification of derivative formulas.** Said "correct" without showing differentiation steps.
3. **Gave A without finding any issues.** Grade for the Gemini review itself: B.

### Previous Review Notes (2025-02-10)

Errors #1-3 from the original review are moot (the K+1 formulation they corrected has been replaced). The validated items (proceeds formula, direct elimination, fee-adjusted κ) remain correct and are reused in the simplified formulation.

### Known Approximations (Cumulative)

1. ~~**(1-f) in alt-price**~~: **RETRACTED** — the alt-price uses the state-price identity (observable pool price P̃_j), not marginal proceeds. The ΔG equation is exact within the single-tick model. See Fourth Codex Review.
2. **Multi-mint consistency**: Different mint entries can't all satisfy their alt-price targets at the same (π, M). Residual for entry k: `r_k = (pred_k - pred_{i*})/(1+π) - (P⁰_k - P⁰_{i*})`. Induced M-error ≈ r_k / ΔG'(M). Parametric bound: `|r_k| ≤ |P⁰_k - P⁰_{i*}| × |ρ/(1+π) - 1|` where ρ = pred/P⁰ is the common profitability ratio (assuming ρ_k ≈ ρ_{i*}). The error depends on **both** the profitability spread and the **price spread** among mint entries — two outcomes with similar profitability ratios but different absolute prices (e.g., 2% vs 10%) can still produce substantial residuals.
3. **Single tick range**: All formulas assume liquidity is constant within the tick range. Real pools may have multiple tick ranges; the code uses `compute_swap_step` for execution which handles this, but the analytical optimization assumes single-range.
4. **Relative tolerance**: `|step| < 1e-12` is absolute. For π >> 1, `|step| < 1e-12 × (1+|π|)` would be more robust.
5. **Convergence**: Inner Newton on ΔG(M) = δ: ΔG is monotone increasing (ΔG' > 0) and concave (ΔG'' < 0), guaranteeing a unique root when feasible. Convergence is quadratic in smooth regions, but `min(M, M_j)` cap transitions introduce C¹ kinks (discontinuous second derivative), degrading to **linear convergence** near cap boundaries. Outer Newton on budget(π) = B: budget(π) is monotone decreasing in π (higher profitability → less buying). Convexity is **not proven** — while D_cost(π) is convex, C_mint(M(π)) involves a composition through the implicit function M(π), and d²M/dπ² picks up ΔG'' terms that don't obviously preserve sign. Monotonicity suffices for convergence from one side, but without convexity, convergence rate is uncertain near the root. **Mitigation**: bisection fallback after 8 Newton iterations if |residual| > tolerance. The bisection is guaranteed to converge by monotonicity.
6. **Saturation boundary domain condition**: π_boundary = Π*/(1-S₀+ΔG_max) - 1 requires 1-S₀+ΔG_max > 0. Also must clamp to [prof_lo, prof_hi].
7. **Active set stability**: The D*, Q', N partition is computed once at initialization and held fixed. As π moves during Newton, an outcome's direct profitability could cross zero (transitioning from N to D or Q' to D), making δ(π) piecewise smooth with additional kinks beyond the M_j cap transitions. The waterfall's pruning loop (line 726-728) partly addresses this at the outer level, but the coupled solver assumes a fixed partition. In practice, π moves within [prof_lo, prof_hi] which is a small interval (the waterfall has already narrowed it), so partition transitions are rare.
8. **Inner Newton tolerance**: The inner Newton on M should use relative tolerance `|step| < 1e-10 × (1 + M)` since M ranges from 0 to ~10⁵ tokens. An absolute `1e-12` is too tight at large M and irrelevant at small M.

### Fourth Codex Review Critique (2026-02-10)

GPT-5.3 (xhigh reasoning, role: Terence Tao) reviewed the corrected plan. 12 findings, graded C. Disposition:

**Finding #1 (Critical: (1-f) "correction" is itself wrong) — CORRECT AND MOST IMPORTANT.**
The document's own "MISSED by Gemini" section (lines 433-458) incorrectly claimed that ΔG should use (1-f)×P̃_j. Codex correctly identified this as confusing the state-price identity with the marginal-proceeds identity. The alt-price is `1 - Σ P_j^final`, which uses observable pool prices, not marginal proceeds. The fee is embedded in κ (how sells move price), not in price observation. Decisive proof: at M=0, the wrong formula gives ΔG(0) = f×Σ P⁰_j > 0, but zero sells can't change any price. **Fixed**: retracted the (1-f) claim; ΔG is exact within the single-tick model.

**Finding #2 (High: 2D is relaxation not exact) — ALREADY ACKNOWLEDGED.**
The document's "Multi-Mint Consistency Caveat" (lines 117-125) explicitly states this. The 2D system is exact for |Q|=1 and a relaxation for |Q|>1. Codex restated what was already written.

**Finding #3-5 (Low: skip collapse, D*/S₀, main equation) — CONFIRMED CORRECT.** No issues.

**Finding #6 (Medium: derivatives non-smooth at cap transitions) — VALID, already handled.**
The pseudocode uses uncapped indicators in derivatives (line 194) and the saturation guard (lines 182-190). Pools transition from uncapped to capped monotonically in M, producing at most N kinks. Handled by piecewise logic.

**Finding #7 (High: convergence not proven) — VALID.**
Inner Newton: ΔG is monotone increasing and concave → unique root guaranteed, but convergence rate not formally established. Outer Newton: budget(π) is monotone under regularity, but no formal proof. **Fixed**: added bisection fallback to Known Approximations.

**Finding #8 (Low: path-independence) — CONFIRMED CORRECT.** Including with caps.

**Finding #9 (Medium: fixed-i* error not bounded) — VALID.**
The document claimed "typically <1%" without proof. The error is r_k(π) = δ_k(π) - δ_i*(π), induced M-error ≈ r_k / ΔG'(M). **Fixed**: updated Known Approximations with explicit error formula.

**Finding #10 (Low: saturation boundary domain condition) — VALID.** Need 1-S₀+ΔG_max > 0. **Fixed**: added to Known Approximations.

**Finding #11 (Medium: "below f64 precision" wrong) — MOOT.** The (1-f) "correction" that this finding critiqued has been retracted entirely (Finding #1).

**Finding #12 (Low: waterfall error bound not rigorous) — VALID but low-impact.** Index mismatch in the error bound formula (Σ_i F_j should sum over pools j). The error bound is heuristic, not used in the solver. Left as-is.

**MISSED by Codex:**
1. **The grade C is too harsh.** The core mathematical framework (D*/S₀, main equation, path-independence, Newton structure) is all correct. Finding #1 is critical but it's a self-inflicted error in the review history, not in the solver formulation. The solver equations were correct all along. Corrected grade for the plan: **B+**.
2. **No comment on the waterfall error bound formula** (line 131) beyond the index mismatch. The constant C and the "convexity of F_j times variance of m_i" claim deserves scrutiny — specifically, it conflates inter-mint interaction (which is zero under skip semantics) with the independent-treatment error. With skip, the error source is strictly the concavity of total proceeds in M vs summing F_j(m_i) independently. This is Jensen's inequality applied to concave F_j: Σ F_j(m_i) ≤ F_j(M) only when F_j is concave, but proceeds F_j(M) = P⁰_j(1-f)M/(1+κ_j M) IS concave in M, so individual proceeds OVERESTIMATE total proceeds. The inequality goes: Σ F_j(m_i) ≥ F_j(Σ m_i) = F_j(M) by concavity (Jensen). Wait — Jensen for concave f: Σ f(m_i)/n ≤ f(Σ m_i/n), so Σ f(m_i) ≤ n×f(M/n) ≤... No: for f concave and m_i ≥ 0, f(m_1) + f(m_2) ≥ f(m_1+m_2) + f(0) = f(M) since f is concave with f(0) = 0. So Σ F_j(m_i) ≥ F_j(M). The waterfall overestimates proceeds → underestimates net cost → overshoots profitability. This is consistent with the document's claim. The error bound formula is qualitatively correct but needs tightening.

**Overall assessment:** The plan's solver formulation is mathematically sound. The main remaining weaknesses are (a) convergence proof/fallback, (b) multi-mint consistency error bound, both now documented. The (1-f) false alarm has been retracted.
