# Gas-Aware Rebalancing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the rebalancing algorithm gas-cost-aware so it never opens positions whose expected profit is smaller than the gas cost to execute them, and fix a critical oracle bug that causes the L1 data fee to be measured as near-zero.

**Architecture:** Four changes layered bottom-up: (1) fix the Fjord oracle probe to use high-entropy bytes instead of uniform `0x01` bytes, (2) write empirical calldata size tests to confirm existing constants, (3) add a per-route gas cost helper that flows into `waterfall()` as a minimum-trade threshold, (4) document the model. No changes to the analytical math inside `PoolSim` — gas is an external overhead applied at the scheduling layer.

**Tech Stack:** Rust, Alloy `sol!` macro for ABI encoding, existing `GasAssumptions` / `waterfall()` / `best_non_active()` in `src/execution/gas.rs` and `src/portfolio/core/waterfall.rs`.

---

## Background: current gap

The rebalancing waterfall (`waterfall.rs`) uses raw profitability `(prediction - price) / price` to rank outcomes and allocate budget. Gas costs only enter the picture in `execution/bounds.rs` — a post-hoc filter that drops action groups whose edge doesn't clear gas + buffer. This means the waterfall may allocate budget to thin opportunities that the post-hoc filter will then drop, leaving capital suboptimally placed.

### Fjord compression bug (critical)

`gas.rs:20`: `const L1_FEE_SAMPLE_FILL_BYTE: u8 = 1;`

The oracle probe fills payloads with `vec![0x01; N]`. Under the **Fjord** upgrade, Optimism uses Brotli compression for L1 data. Uniform `0x01` bytes compress to near-zero (a single compressed chunk), so `getL1Fee` returns nearly zero wei — the marginal slope between 256 and 512 bytes collapses to ~0. This makes `l1_fee_per_byte_wei` ≈ 0 and causes `estimate_group_l1_data_fee_susd` to fall back to only the `l1_data_fee_floor_susd` floor, dramatically under-pricing the L1 component of gas for large transactions.

Real calldata (Uniswap swap amounts, addresses) has high entropy and compresses poorly. The fix is to probe with **non-uniform** bytes. Note: changing the fill byte to another constant (e.g. `0xAB`) does **not** fix this — Brotli compresses any single repeated byte to the same near-zero size regardless of which byte it is. The fix is a deterministic non-repeating sequence.

### Minimum trade threshold

The correct check is: for outcome `i` at profitability `p_i`, the minimum budget allocation that breaks even on gas is `gas_cost / p_i`. If the remaining budget is below this threshold, opening a position in outcome `i` will not generate enough profit to cover gas.

In code: `remaining_budget × prof >= gas_cost` i.e. `remaining_budget >= gas_cost / prof`.

This is the check to add to `best_non_active()`. It is conservative (uses full remaining budget) but prevents clearly gas-losing micro-allocations without requiring an expensive solve.

### Key contracts (Optimism)

```
BASE_COLLATERAL (sUSDS): 0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0
BATCH_SWAP_ROUTER:       0x4081136d23FEeCD324a420A54635e007F51fd94a
CTF_ROUTER:              0x179d8F8c811B8C759c33809dbc6c5ceDc62D05DD
V3_SWAP_ROUTER:          0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45
```

### L1 market CTF call ordering

- **Mint (split):** `splitPosition(sUSDS, market1_0x3220)` → `splitPosition(other_repos_token_0x63a4, market2_0xfea4)`
- **Merge:** `mergePositions(other_repos_token_0x63a4, market2_0xfea4)` → `mergePositions(sUSDS, market1_0x3220)`

The second `splitPosition` / `mergePositions` uses the "Other repos" ERC20 token (from market1) as collateral for market2 — **not** sUSDS. "Other repos" outcome (`0x63a4...`) is the synthetic connector between the two markets and is **never** a swap leg.

### Note on calldata constants

- `SWAP_BYTES = 224`: This is the ABI payload for `exactInputSingle` **excluding the 4-byte selector** (7 params × 32 bytes). The selector is included in `BATCH_CALL_BASE_BYTES`. So 224 is correct.
- `FLASH_ROUTE_EXTRA_BYTES = 160`: This accounts for the `BatchSwapRouter` flash-route overhead. CTF `splitPosition`/`mergePositions` calls happen as **internal L2 calls** — they do not appear in the EOA transaction calldata and therefore have **zero L1 data fee**. So 160 is correct for its purpose (BatchSwapRouter overhead), not a CTF encoding size.

These constants do not need to be changed. The empirical tests below confirm this.

---

## Task 1 — Fix Fjord compression bug in L1 fee oracle

The root cause: `fetch_optimism_l1_fee_wei_for_payload_len` in `gas.rs:359` probes with
`Bytes::from(vec![L1_FEE_SAMPLE_FILL_BYTE; payload_len])`. Any **single repeated byte** compresses
to ~11 bytes under Brotli, regardless of which byte it is — `0x01`, `0xAB`, `0x00`, same result.
The fix must produce a **non-uniform** byte sequence.

**Files:**
- Modify: `src/execution/gas.rs` (lines ~18-21 and the probe construction at line ~359)

**Step 1: Write a failing unit test**

Add to the `#[cfg(test)]` module at the bottom of `gas.rs`:

```rust
#[test]
fn probe_bytes_are_non_uniform() {
    // Regression guard: the L1 fee probe must NOT be a single repeated byte.
    // Under Fjord Brotli, ANY uniform vec![b; N] compresses to ~11 bytes regardless
    // of which byte `b` is. This causes the marginal fee slope to collapse to ~0.
    // The probe must have enough byte diversity to simulate real calldata entropy.
    let small = make_l1_fee_probe_bytes(L1_FEE_SLOPE_SAMPLE_SMALL_BYTES);
    let large = make_l1_fee_probe_bytes(L1_FEE_SLOPE_SAMPLE_LARGE_BYTES);
    // At least 4 distinct byte values
    let distinct_small: std::collections::HashSet<u8> = small.iter().copied().collect();
    let distinct_large: std::collections::HashSet<u8> = large.iter().copied().collect();
    assert!(
        distinct_small.len() >= 4,
        "probe bytes have only {} distinct values (need ≥4 for realistic entropy)",
        distinct_small.len()
    );
    assert_eq!(small.len(), L1_FEE_SLOPE_SAMPLE_SMALL_BYTES);
    assert_eq!(large.len(), L1_FEE_SLOPE_SAMPLE_LARGE_BYTES);
    let _ = distinct_large; // large inherits same generator
}
```

**Step 2: Extract probe byte generation into a testable function**

In `gas.rs`, extract the probe generation into a pure function:

```rust
/// Build a non-uniform probe payload that approximates real calldata byte entropy.
/// Uses a deterministic LCG so the result is reproducible and non-trivially compressible.
fn make_l1_fee_probe_bytes(len: usize) -> Vec<u8> {
    let mut state: u32 = 0xDEAD_BEEF;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        })
        .collect()
}
```

Replace the inline `vec![L1_FEE_SAMPLE_FILL_BYTE; payload_len]` at line ~359 with:

```rust
let call_data = getL1FeeCall {
    payload: Bytes::from(make_l1_fee_probe_bytes(payload_len)),
}
.abi_encode();
```

Remove the now-unused `L1_FEE_SAMPLE_FILL_BYTE` constant.

**Step 3: Run tests**

```bash
cargo test probe_bytes_are_non_uniform uniform_fill -- --nocapture
cargo test
```

Expected: `probe_bytes_are_non_uniform` passes. All other tests pass.

**Step 4: Commit**

```bash
git add src/execution/gas.rs
git commit -m "fix: use non-uniform LCG probe bytes for Fjord L1 fee slope measurement"
```

---

## Task 2 — Write empirical calldata byte-count verification tests

These tests confirm that the constants `SWAP_BYTES=224` and `BATCH_CALL_BASE_BYTES` are ABI-grounded. They are documentation and regression guards, **not** bug fixes — the existing constants are expected to be correct.

**Files:**
- Modify: `src/execution/gas.rs` (add to `#[cfg(test)] mod tests`)

**Step 1: Add the ABI-encoding tests**

```rust
#[test]
fn exact_input_single_payload_without_selector_is_224_bytes() {
    // Verify that 7 EVM word fields = 7 × 32 = 224 bytes.
    // SWAP_BYTES = 224 excludes the 4-byte selector (which lives in BATCH_CALL_BASE_BYTES).
    use alloy::sol;
    use alloy::sol_types::SolCall;

    sol! {
        struct ExactInputSingleParams {
            address tokenIn;
            address tokenOut;
            uint24 fee;
            address recipient;
            uint256 amountIn;
            uint256 amountOutMinimum;
            uint160 sqrtPriceLimitX96;
        }
        function exactInputSingle(ExactInputSingleParams params) external returns (uint256);
    }

    let call = exactInputSingleCall {
        params: ExactInputSingleParams {
            tokenIn: alloy::primitives::Address::ZERO,
            tokenOut: alloy::primitives::Address::ZERO,
            fee: 100,
            recipient: alloy::primitives::Address::ZERO,
            amountIn: alloy::primitives::U256::ZERO,
            amountOutMinimum: alloy::primitives::U256::ZERO,
            sqrtPriceLimitX96: alloy::primitives::U256::ZERO.into(),
        },
    };
    let encoded = call.abi_encode();
    // selector(4) + 7 × 32 = 228 total; payload without selector = 224
    assert_eq!(
        encoded.len() - 4,
        224,
        "exactInputSingle payload (no selector) must be 224 bytes; got {}",
        encoded.len() - 4
    );
}

#[test]
fn split_position_full_call_is_100_bytes() {
    // Verify CTF splitPosition: selector(4) + 3 × address/uint256 words = 100 bytes.
    // Internal L2 calls (splitPosition is called by BatchSwapRouter internally) do NOT
    // contribute to L1 calldata — only the outer EOA tx bytes matter for L1 data fee.
    use alloy::sol;
    use alloy::sol_types::SolCall;

    sol! {
        function splitPosition(address collateralToken, address conditionId, uint256 amount) external;
    }

    let call = splitPositionCall {
        collateralToken: alloy::primitives::Address::ZERO,
        conditionId: alloy::primitives::Address::ZERO,
        amount: alloy::primitives::U256::ZERO,
    };
    let encoded = call.abi_encode();
    assert_eq!(encoded.len(), 100, "splitPosition ABI encoding must be 100 bytes");
}

#[test]
fn direct_buy_calldata_estimate_is_434_bytes() {
    // DirectBuy: TX_ENVELOPE(110) + BATCH_CALL_BASE(100) + SWAP_BYTES(224) = 434
    let estimate = estimate_group_calldata_bytes(GroupKind::DirectBuy, 0, 0);
    assert_eq!(estimate, 434, "DirectBuy calldata estimate should be 434 bytes");
}
```

**Step 2: Run tests**

```bash
cargo test exact_input_single_payload -- --nocapture
cargo test split_position_full -- --nocapture
cargo test direct_buy_calldata -- --nocapture
```

Expected: all pass. If any assertion fails, **do not** update the assertion to match the code — the assertions are grounded in the ABI spec. A failure means the calldata model is wrong and must be investigated before proceeding.

**Step 3: Commit**

```bash
git add src/execution/gas.rs
git commit -m "test: ABI-grounded calldata byte count assertions for SWAP_BYTES and CTF calls"
```

---

## Task 3 — Add `estimate_min_gas_susd_for_group` helper in `gas.rs`

This gives the waterfall a single call to get the estimated total gas cost (L2 + L1 data fee floor) for a given route kind, using cached L1 fee-per-byte. Returns the gas cost floor even when `l1_fee_per_byte_wei = 0` (uses `l1_data_fee_floor_susd` as minimum).

**Files:**
- Modify: `src/execution/gas.rs`

**Step 1: Add the function**

```rust
/// Estimated total gas cost in sUSD for a single trade group of the given kind.
///
/// Uses the L1 data fee floor when no L1 fee-per-byte is set.
/// `sell_legs` / `buy_legs` are the number of swap legs in the group.
pub fn estimate_min_gas_susd_for_group(
    assumptions: &GasAssumptions,
    kind: GroupKind,
    buy_legs: usize,
    sell_legs: usize,
    gas_price_eth: f64,
    eth_usd: f64,
) -> f64 {
    let l2_units = estimate_group_l2_gas_units(assumptions, kind, buy_legs, sell_legs);
    let gas_l2 = estimate_l2_gas_susd(l2_units, gas_price_eth, eth_usd);
    if !gas_l2.is_finite() || gas_l2 < 0.0 {
        return f64::INFINITY;
    }

    let gas_l1 = if assumptions.l1_fee_per_byte_wei > 0.0 {
        estimate_group_l1_data_fee_susd(assumptions, kind, buy_legs, sell_legs, eth_usd)
    } else {
        assumptions.l1_data_fee_floor_susd
    };
    if !gas_l1.is_finite() || gas_l1 < 0.0 {
        return f64::INFINITY;
    }
    gas_l2 + gas_l1
}
```

**Step 2: Write tests**

```rust
#[test]
fn min_gas_susd_direct_buy_at_floor() {
    let gas = GasAssumptions {
        l1_data_fee_floor_susd: 0.10,
        l1_fee_per_byte_wei: 0.0, // unknown — fall back to floor
        ..GasAssumptions::default()
    };
    // L2: 220_000 gas × 5e-10 ETH/gas × 3000 $/ETH = 0.33 sUSD
    // L1: floor = 0.10 sUSD
    // Total: 0.43 sUSD
    let cost = estimate_min_gas_susd_for_group(
        &gas, GroupKind::DirectBuy, 0, 0, 5e-10, 3000.0,
    );
    assert!((cost - 0.43).abs() < 1e-9, "expected ~0.43, got {cost}");
}

#[test]
fn min_gas_susd_is_finite_for_97_leg_mint_sell() {
    let gas = GasAssumptions {
        l1_data_fee_floor_susd: 0.10,
        l1_fee_per_byte_wei: 0.0,
        ..GasAssumptions::default()
    };
    // L2: (550_000 + 97 × 170_000) = 17_040_000 gas
    // At 5e-10 ETH/gas × 3000 $/ETH = ~25.56 sUSD L2 alone (high gas price scenario)
    // At realistic Optimism prices (~0.001 gwei = 1e-12), L2 ~ $0.05
    let cost = estimate_min_gas_susd_for_group(
        &gas, GroupKind::MintSell, 0, 97, 5e-10, 3000.0,
    );
    assert!(cost.is_finite(), "97-leg MintSell gas estimate must be finite");
    assert!(cost > 0.0, "97-leg MintSell gas must be positive");
    // Sanity: L2 component should dominate floor at 5e-10 gas price
    assert!(cost > 1.0, "97-leg MintSell at 5e-10 gas price should be > $1: {cost}");
}
```

**Step 3: Run tests**

```bash
cargo test min_gas_susd -- --nocapture
```

**Step 4: Commit**

```bash
git add src/execution/gas.rs
git commit -m "feat: add estimate_min_gas_susd_for_group helper"
```

---

## Task 4 — Thread a gas cost threshold into the waterfall

**Files:**
- Modify: `src/portfolio/core/waterfall.rs`

**Design:** Pass two `f64` scalars — `gas_direct_susd` and `gas_mint_susd` — into `waterfall()` and `best_non_active()`. Before admitting a candidate to the active set, check that the remaining budget is sufficient to cover the break-even minimum allocation: `remaining_budget × prof >= gas_cost`, equivalently `remaining_budget >= gas_cost / prof`. This prevents adding outcomes whose maximum possible profit (if all remaining budget went to them) cannot cover the gas cost.

**Step 1: Write failing tests**

Add a `#[cfg(test)]` module to `src/portfolio/core/waterfall.rs`. Since the test needs `PoolSim`, use the same construction pattern as `src/portfolio/tests/fixtures.rs` — inline a minimal helper rather than importing across modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::markets::{MarketData, Pool, Tick, MARKETS_L1};
    use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};
    use alloy::primitives::{Address, U256};

    /// Minimal PoolSim for waterfall tests: price ~= price_frac, prediction = pred.
    /// Uses Box::leak to produce 'static references (test process memory, not freed).
    fn make_sim(
        name: &'static str,
        token: &'static str,
        price_frac: f64,
        pred: f64,
    ) -> PoolSim {
        let liq_str: &'static str = Box::leak("1000000000000000000000".to_string().into_boxed_str());
        let ticks: &'static [Tick] = Box::leak(Box::new([
            Tick { tick_idx: 1, liquidity_net: 1_000_000_000_000_000_000_000 },
            Tick { tick_idx: 92108, liquidity_net: -1_000_000_000_000_000_000_000 },
        ]));
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            token0: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
            token1: token,
            pool_id: "0x0000000000000000000000000000000000000001",
            liquidity: liq_str,
            ticks,
        }));
        let sqrt = prediction_to_sqrt_price_x96(price_frac, true)
            .unwrap_or(U256::from(1u128 << 96));
        let market: &'static MarketData = Box::leak(Box::new(MarketData {
            name,
            market_id: MARKETS_L1[0].market_id,
            outcome_token: token,
            pool: Some(*pool),
            quote_token: "0xb5B2dc7fd34C249F4be7fB1fCea07950784229e0",
        }));
        let slot0 = Slot0Result {
            pool_id: Address::ZERO,
            sqrt_price_x96: sqrt,
            tick: 0,
            observation_index: 0,
            observation_cardinality: 0,
            observation_cardinality_next: 0,
            fee_protocol: 0,
            unlocked: true,
        };
        PoolSim::from_slot0(&slot0, market, pred).unwrap()
    }

    #[test]
    fn waterfall_skips_outcome_when_budget_below_break_even() {
        // profitability ≈ (0.0101 - 0.01) / 0.01 = 1% = 0.01
        // gas_direct = $0.50, break-even min budget = 0.50 / 0.01 = $50
        // budget = $1 < $50 → outcome must be skipped → zero actions
        let mut sims = vec![make_sim(
            "m1", "0x1111111111111111111111111111111111111111", 0.01, 0.0101,
        )];
        let mut budget = 1.0_f64;
        let mut actions = vec![];

        let last_prof = waterfall(
            &mut sims, &mut budget, &mut actions, false,
            0.50, // gas_direct_susd: $0.50
            2.00, // gas_mint_susd: $2.00
        );
        assert!(
            actions.is_empty(),
            "budget $1 at 1% profitability cannot cover $0.50 gas; got {} actions, last_prof={last_prof}",
            actions.len()
        );
        assert!(
            (budget - 1.0).abs() < 1e-9,
            "budget must be unchanged when all trades skipped; got {budget}"
        );
    }

    #[test]
    fn waterfall_executes_when_budget_above_break_even() {
        // Same outcome, budget = $100; break-even = $50; $100 > $50 → should trade
        let mut sims = vec![make_sim(
            "m2", "0x2222222222222222222222222222222222222222", 0.01, 0.0101,
        )];
        let mut budget = 100.0_f64;
        let mut actions = vec![];

        let _last_prof = waterfall(
            &mut sims, &mut budget, &mut actions, false, 0.50, 2.00,
        );
        assert!(
            !actions.is_empty(),
            "budget $100 at 1% profitability must exceed $0.50 gas break-even; got 0 actions"
        );
    }

    #[test]
    fn waterfall_with_zero_gas_thresholds_behaves_as_before() {
        // gas_direct=0, gas_mint=0 → no filtering, same as old signature
        let mut sims = vec![make_sim(
            "m3", "0x3333333333333333333333333333333333333333", 0.01, 0.0101,
        )];
        let mut budget = 1.0_f64;
        let mut actions = vec![];

        let _last_prof = waterfall(
            &mut sims, &mut budget, &mut actions, false, 0.0, 0.0,
        );
        assert!(
            !actions.is_empty(),
            "with zero gas thresholds, any positive profitability should produce actions"
        );
    }
}
```

**Step 2: Run failing tests** (compile error expected — wrong signature)

```bash
cargo test waterfall_skips_outcome -- --nocapture
cargo test waterfall_executes_when -- --nocapture
cargo test waterfall_with_zero -- --nocapture
```

Expected: compile error because `waterfall` takes 4 params, not 6.

**Step 3: Update `best_non_active` signature**

In `waterfall.rs`, add `remaining_budget`, `gas_direct_susd`, `gas_mint_susd` parameters and the break-even guard:

```rust
pub(super) fn best_non_active(
    sims: &[PoolSim],
    active_set: &HashSet<(usize, Route)>,
    mint_available: bool,
    price_sum: f64,
    remaining_budget: f64,      // ← new
    gas_direct_susd: f64,       // ← new
    gas_mint_susd: f64,         // ← new
) -> Option<(usize, Route, f64)> {
    let mut best: Option<(usize, Route, f64)> = None;
    for (i, sim) in sims.iter().enumerate() {
        if !active_set.contains(&(i, Route::Direct)) {
            let prof = profitability(sim.prediction, sim.price());
            if prof > 0.0
                && remaining_budget * prof >= gas_direct_susd
                && best.is_none_or(|b| prof > b.2)
            {
                best = Some((i, Route::Direct, prof));
            }
        }
        if mint_available && !active_set.contains(&(i, Route::Mint)) {
            let mp = alt_price(sims, i, price_sum);
            let prof = profitability(sim.prediction, mp);
            if prof > 0.0
                && remaining_budget * prof >= gas_mint_susd
                && best.is_none_or(|b| prof > b.2)
            {
                best = Some((i, Route::Mint, prof));
            }
        }
    }
    best
}
```

**Step 4: Update `waterfall` signature and all `best_non_active` call sites**

```rust
pub(super) fn waterfall(
    sims: &mut [PoolSim],
    budget: &mut f64,
    actions: &mut Vec<Action>,
    mint_available: bool,
    gas_direct_susd: f64,   // ← new
    gas_mint_susd: f64,     // ← new
) -> f64 {
```

Search for every `best_non_active(` call inside `waterfall.rs` and add `, *budget, gas_direct_susd, gas_mint_susd` before the closing `)`. There are approximately 3 call sites:

```bash
grep -n "best_non_active(" src/portfolio/core/waterfall.rs
```

Update each one. The signature is: `(sims, &active_set, mint_available, price_sum, *budget, gas_direct_susd, gas_mint_susd)`.

**Step 5: Fix all call sites — full migration surface**

The signature change ripples across four files. Run these greps first to count before editing:

```bash
grep -rn "waterfall(" src/portfolio/
grep -rn "best_non_active(" src/portfolio/
```

Expected counts (from codebase at plan-writing time): **29 `waterfall(` calls**, **15 `best_non_active(` calls**.

Files that need updating:

| File | `waterfall(` | `best_non_active(` |
|---|---|---|
| `src/portfolio/core/rebalancer.rs` | ~7 | 0 |
| `src/portfolio/tests/oracle.rs` | ~16 | ~10 |
| `src/portfolio/tests.rs` | ~4 | 0 |
| `src/portfolio/fuzz_rebalance.rs` | ~1 | 0 |
| `src/portfolio/core/waterfall.rs` (internal) | ~5 | ~5 |

For every `waterfall(sims, budget, actions, mint_available)` call (outside of `waterfall.rs` itself), add `, 0.0, 0.0` before the closing `)`:

```
waterfall(sims, budget, actions, mint_available, 0.0, 0.0)
```

The `best_non_active(` calls inside `waterfall.rs` were already updated in Step 4. The external calls in `oracle.rs` and `tests.rs` call `waterfall()` not `best_non_active()` directly, so only `waterfall` signature matters for those.

**Step 6: Run all tests**

```bash
cargo test
```

Expected: all pass.

**Step 7: Commit**

```bash
git add src/portfolio/core/waterfall.rs src/portfolio/core/rebalancer.rs \
        src/portfolio/tests/oracle.rs src/portfolio/tests.rs \
        src/portfolio/fuzz_rebalance.rs
git commit -m "feat: gas-aware minimum trade threshold in waterfall best_non_active"
```

---

## Task 5 — Compute and pass real gas thresholds from rebalancer

**Files:**
- Modify: `src/portfolio/core/rebalancer.rs`

Replace the `0.0, 0.0` placeholders from Task 4 with real gas threshold values computed from `GasAssumptions`.

**Step 1: Add gas threshold computation**

Near the top of `rebalancer.rs`, add:

```rust
use crate::execution::gas::{GasAssumptions, estimate_min_gas_susd_for_group};
use crate::execution::GroupKind;

fn compute_gas_thresholds(
    gas: &GasAssumptions,
    gas_price_eth: f64,
    eth_usd: f64,
    mint_sell_legs: usize,  // pass sims.len() - 1 as conservative upper bound
) -> (f64, f64) {
    let direct = estimate_min_gas_susd_for_group(
        gas, GroupKind::DirectBuy, 0, 0, gas_price_eth, eth_usd,
    );
    let mint_sell = estimate_min_gas_susd_for_group(
        gas, GroupKind::MintSell, 0, mint_sell_legs, gas_price_eth, eth_usd,
    );
    (
        if direct.is_finite() { direct } else { f64::INFINITY },
        if mint_sell.is_finite() { mint_sell } else { f64::INFINITY },
    )
}
```

> **Fail-closed, not fail-open.** If `estimate_min_gas_susd_for_group` returns `NaN` or ±∞ (e.g. due to a zero gas price or missing L1 fee data), we must block all trades, not silently allow them. `INFINITY` passed as a threshold means `budget × prof >= INFINITY` is never true, so no trade is taken — safe. `0.0` would disable the filter entirely — dangerous.

Note: `mint_sell_legs = sims.len().saturating_sub(1)` is the worst-case number of sell legs (every outcome except the target). This ensures the threshold is never under-estimated. Do **not** hardcode `97` — the number of tradeable outcomes changes as markets evolve.

**Step 2: Add a new gas-aware wrapper — do NOT modify `rebalance_with_mode` signature**

`rebalance_with_mode` is called from `main.rs` and 8+ test sites. Adding a parameter to it is API-breaking and forces mass edits. Instead, add a thin wrapper with a new name:

```rust
/// Default gas price assumptions for minimum-trade-threshold computation.
/// Conservative: 1 gwei L2, $3000/ETH.
const THRESHOLD_GAS_PRICE_ETH: f64 = 1e-9;
const THRESHOLD_ETH_USD: f64 = 3000.0;

/// Gas-aware entry point used by `main.rs`.
/// Computes per-route thresholds and threads them through `waterfall`.
/// The existing `rebalance_with_mode` keeps its current signature and is used
/// by all tests unchanged (it internally calls `waterfall` with `0.0, 0.0`).
pub fn rebalance_with_gas(
    balances: &HashMap<&str, f64>,
    susds_balance: f64,
    slot0_results: &[Slot0Result],
    mode: RebalanceMode,
    gas: &GasAssumptions,
) -> Vec<Action> {
    // Compute thresholds once per call — same block assumption.
    let n_sims = slot0_results.len();
    let (gas_direct, gas_mint) = compute_gas_thresholds(
        gas,
        THRESHOLD_GAS_PRICE_ETH,
        THRESHOLD_ETH_USD,
        n_sims.saturating_sub(1),
    );
    rebalance_with_mode_and_thresholds(balances, susds_balance, slot0_results, mode, gas_direct, gas_mint)
}
```

Extract the inner logic of `rebalance_with_mode` into a private `rebalance_with_mode_and_thresholds(... gas_direct: f64, gas_mint: f64)` that takes explicit thresholds. `rebalance_with_mode` becomes a thin call to it with `0.0, 0.0`. `rebalance_with_gas` calls it with real thresholds.

This means **zero changes** to existing test call sites.

**Step 3: Wire thresholds into each `waterfall()` call**

Inside `rebalance_with_mode_and_thresholds`, pass `gas_direct, gas_mint` to every `waterfall()` call. There are ~7 call sites in `rebalancer.rs`. Use the same pair for all — they don't change within a single rebalance.

**Step 4: Update `main.rs` call site**

Replace the call to `rebalance_with_mode(...)` in `main.rs` with `rebalance_with_gas(..., &gas_assumptions)` once `GasAssumptions` is hydrated (after `default_gas_assumptions_with_optimism_l1_fee(rpc_url).await`).

**Step 5: Run all tests**

```bash
cargo test
```

Expected: all pass — existing tests still call `rebalance_with_mode` unchanged.

**Step 6: Commit**

```bash
git add src/portfolio/core/rebalancer.rs src/main.rs
git commit -m "feat: pass real gas thresholds into waterfall from GasAssumptions"
```

---

## Task 6 — Verify "Other repos" is excluded from sell legs

**Files:**
- Add tests to `src/portfolio/tests.rs` (inline — no new files)

**Step 1: Write the tests**

```rust
#[test]
fn other_repos_outcome_has_no_pool() {
    // "Other repos" is intentionally present in MARKETS_L1 as the synthetic connector
    // between market1 and market2 (it appears with pool: None at markets.rs:1303).
    // The invariant is that it must NEVER have a pool — it must never be tradeable.
    const OTHER_REPOS: &str = "0x63a4f76ef5846f68d069054c271465b7118e8ed9";
    let tradeable = crate::markets::MARKETS_L1
        .iter()
        .any(|m| m.pool.is_some() && m.outcome_token.eq_ignore_ascii_case(OTHER_REPOS));
    assert!(
        !tradeable,
        "Other repos token {} must never have a pool (must not be tradeable)",
        OTHER_REPOS
    );
}

#[test]
fn l1_tradeable_outcome_count_matches_predictions() {
    // 66 from market1 (67 - Other repos connector) + 32 from market2 (33 - invalid has no pool)
    // Use PREDICTIONS_L1.len() as the source-of-truth count — not a hardcoded literal.
    let count = crate::markets::MARKETS_L1
        .iter()
        .filter(|m| m.pool.is_some())
        .count();
    assert_eq!(
        count,
        crate::predictions::PREDICTIONS_L1.len(),
        "MARKETS_L1 pools-present count must equal PREDICTIONS_L1 len; got {count}"
    );
}
```

**Step 2: Run**

```bash
cargo test other_repos_outcome_has_no_pool -- --nocapture
cargo test l1_tradeable_outcome_count_matches -- --nocapture
```

Expected: pass. If `l1_tradeable_outcome_count_is_98` fails with a different count, update the comment in the test but do **not** change the assertion — investigate why the count differs.

**Step 3: Commit (tests only)**

```bash
git add src/portfolio/tests.rs
git commit -m "test: verify Other repos excluded from sims and 98 tradeable outcomes"
```

---

## Task 7 — Update gas model documentation

**Files:**
- Create: `docs/gas_model.md`

**Step 1: Write the file**

```markdown
# Gas Model

## Swap Fees (Uniswap V3)

The market pools use a **100-pip (0.01%) fee tier**. This fee is baked into `FEE_FACTOR = 0.9999` in `PoolSim` (`src/portfolio/core/sim.rs`). All buy/sell math is fee-inclusive; no separate swap-fee deduction is needed.

## L2 Gas Estimates

| Group kind           | L2 gas (est.)              | Calldata (est.) |
|----------------------|---------------------------|-----------------|
| DirectBuy            | 220,000                   | 434 bytes       |
| DirectSell           | 200,000                   | 434 bytes       |
| DirectMerge          | 150,000                   | 330 bytes       |
| MintSell (N legs)    | 550,000 + N × 170,000     | 310 + N × 224 bytes |
| BuyMerge (N legs)    | 500,000 + N × 180,000     | 310 + N × 224 bytes |

At typical Optimism gas prices (0.001–0.01 gwei, $3000/ETH):
- DirectBuy: ~$0.001–$0.01 L2
- MintSell (97 legs): ~$0.05–$0.50 L2

## L1 Data Fee

Estimated by probing the `GasPriceOracle.getL1Fee()` on-chain contract with two different payload sizes (256 and 512 bytes), measuring the marginal fee per byte (Fjord/Brotli model). Cache TTL: 60 seconds. Floor: $0.10 sUSD.

**Probe byte:** `0xAB` — chosen for high entropy to approximate real calldata. Do not use `0x00` or `0x01`; uniform bytes compress to near-zero under Brotli, causing the slope measurement to return ~0.

### Calldata constants

- `SWAP_BYTES = 224`: ABI payload for `exactInputSingle` excluding selector (7 × 32 = 224). The selector is part of `BATCH_CALL_BASE_BYTES`.
- `FLASH_ROUTE_EXTRA_BYTES = 160`: `BatchSwapRouter` flash route overhead. CTF `splitPosition`/`mergePositions` are **internal L2 calls** and do not appear in EOA calldata — no L1 data cost for them.

## Dual-Market CTF Operations

- **Mint**: `splitPosition(sUSDS, market1)` → `splitPosition(OTHER_REPOS_TOKEN, market2)`
- **Merge**: `mergePositions(OTHER_REPOS_TOKEN, market2)` → `mergePositions(sUSDS, market1)`

The second call uses the "Other repos" token (outcome 0x63a4...) as collateral, not sUSDS. This token is excluded from trading and sim construction.

## Minimum Trade Threshold (Waterfall)

Before admitting an outcome to the active set, `best_non_active()` checks:

```
remaining_budget × profitability >= gas_cost
```

Equivalently: `remaining_budget >= gas_cost / profitability`.

This ensures the maximum possible profit from the remaining budget (if all of it went to this outcome) at least covers the gas cost. The threshold uses worst-case gas (`MintSell` with `sims.len() - 1` sell legs) for the mint route.

## Preconditions

All outcome tokens and sUSDS are infinite-approved to `BATCH_SWAP_ROUTER` and `CTF_ROUTER` prior to execution. No per-trade approval gas is incurred.
```

**Step 2: Commit**

```bash
git add docs/gas_model.md
git commit -m "docs: gas model — Fjord oracle, calldata bytes, L1 fee, swap fees, min-trade threshold"
```

---

## Verification Checklist

After all tasks:

```bash
cargo test        # All unit tests pass
cargo clippy      # No new warnings
cargo fmt --check # Formatted
```

Spot-check at runtime with a small budget (e.g., $1) and `gas_assumptions = GasAssumptions::default()` — the waterfall should skip all outcomes when no outcome's `budget × profitability` clears the gas threshold at that scale. With `gas_assumptions = None` (or zero thresholds), behavior is identical to the pre-gas-awareness code.
