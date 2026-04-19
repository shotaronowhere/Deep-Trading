# Fix: `test_benchmark_matrix_single_market` emits `forecastflows_uncertified`

**Scope:** **single-market benchmark only.** The connected-topology FF skip is a separate, pre-existing worker-certification limitation documented in [docs/local_foundry_e2e_benchmark_matrix.md:104-109](../docs/local_foundry_e2e_benchmark_matrix.md#L104-L109) and is **out of scope** for this change.

**Status:** Replacement remediation plan. This version removes temporary production diagnostics and replaces them with a test-first proof using existing replay APIs.

**Owner handoff:** any engineer; no solver-family context needed.

**Estimated effort:** one focused `bounds.rs` patch, two targeted tests, and benchmark verification.

---

## 1. Observed failure (single-market only)

The FF row in
[test/fixtures/local_foundry_e2e_benchmark_matrix_single_market.jsonl](../test/fixtures/local_foundry_e2e_benchmark_matrix_single_market.jsonl)
emits `skip_reason: "forecastflows_uncertified"` with all numeric fields
zero. The telemetry on that case resolves to `fallback=no_replayable_candidate`,
meaning the FF worker produced candidates but none survived the Rust-side replay gate.

The connected row also emits `forecastflows_uncertified`, but that remains a
separate worker-certification issue on the 67+32 topology. This doc does
**not** attempt to fix connected-topology certification.

## 2. Reproduction

The benchmark runs through Foundry + FFI, **not** a standalone cargo bin:

```bash
FORECASTFLOWS_WORKER_BIN=/absolute/path/to/forecast-flows-worker \
  forge test --ffi --match-test test_benchmark_matrix_single_market -vv
```

For local iteration against the fixture binary:

```bash
cargo build --release --bin local_foundry_e2e_fixture \
  --features benchmark_synthetic_fixtures
RUST_LOG=warn ./target/release/local_foundry_e2e_fixture \
  test/fixtures/local_foundry_e2e_fixture_input_bench_single_market_98_forecastflows.json
```

Expected current failure: the fixture binary hard-fails with the replay-only
gate and reports `fallback=no_replayable_candidate`.

## 3. Mechanical hypothesis

The single-market harness creates pools with full-range liquidity:

- [test/LocalFoundryExecutableTxE2E.t.sol:751-790](../test/LocalFoundryExecutableTxE2E.t.sol#L751-L790)
  creates each pool with:
  - `tickLower = _minUsableTick(spacing)`
  - `tickUpper = _maxUsableTick(spacing)`
- [test/LocalFoundryExecutableTxE2E.t.sol:1320-1325](../test/LocalFoundryExecutableTxE2E.t.sol#L1320-L1325)
  resolves those to `MIN_TICK` / `MAX_TICK`

At those extremes, [sqrt_price_x96_to_price_outcome](../src/pools/pricing.rs#L62-L74)
is arithmetically correct but saturates one direction:

- At `MIN_SQRT_RATIO = 4295128739`
  - direct price = `floor(sqrt^2 * 1e18 / 2^192) = 0`
  - inverse price =
    `340256786698763678858396856460488307819979090561317864144`
  - after `u256_to_f64`, those become `0.0` and `3.4025678669876368e38`
- At `MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342`
  - direct price =
    `340256786836388094070642339899681172762184831912720469415`
  - inverse price = `0`
  - after `u256_to_f64`, those become `3.402567868363881e38` and `0.0`

So a full-range pool legitimately yields one boundary at `0.0` and the
opposite boundary at `~3.4e38`.

That interacts badly with
[`market_price_state`](../src/execution/bounds.rs#L355-L408), which currently
rejects any pool whose `buy_limit_price <= 0.0 || sell_limit_price <= 0.0`.
For a full-range pool, that means valid boundary state is treated as invalid,
which can make
[`conservative_sqrt_price_limit`](../src/execution/bounds.rs#L419-L469)
return `None`, which then makes
[`apply_market_context_to_plan`](../src/execution/bounds.rs#L473-L543)
return `false`, which then becomes `GasReplaySkipReason::ConservativeRepricingFailed`
in
[`build_group_plans_for_gas_replay_with_market_context`](../src/execution/bounds.rs#L583-L668).

That replay skip is then collapsed into `None` by
[`estimate_plan_cost_from_replay`](../src/portfolio/core/rebalancer.rs#L907-L934),
which is why ForecastFlows ends up as `no_replayable_candidate`.

## 4. Step 1 — Prove the failure path with tests, not temporary logging

Do **not** add temporary `warn!` calls to `rebalancer.rs`. The code already
preserves the relevant signal in
`build_group_plans_for_gas_replay_with_market_context(...).skipped_groups[*].reason`,
so use that directly.

Add two tests in [`src/execution/bounds.rs`](../src/execution/bounds.rs),
alongside the existing market-context tests.

### 4.1. Add a small full-range single-market test helper

Create a private test helper that builds one synthetic market and one
`Slot0Result` matching the committed single-market benchmark shape. Do **not**
parse JSON fixtures at test runtime.

Use these exact values:

- `market.name = "full_range_overflow_case"`
- `outcome_token == token0`
- `quote_token == token1`
- therefore `is_token1_outcome == false`
- `tick_lower = -887272`
- `tick_upper = 887272`
- `liquidity = "1000000000000000000000000"`
- `slot0.sqrt_price_x96 = "8767124396362831555064374262"`
- `slot0.tick = -44029`

This orientation is important: it is the one where the huge full-range buy
limit later overflows `outcome_price_to_sqrt_limit_x96`.

Suggested helper shape:

```rust
fn full_range_overflow_slot0_result() -> Vec<(Slot0Result, &'static MarketData)>
```

Implementation notes:

- Build the `MarketData` inline the same way nearby tests build synthetic
  `MarketData` / `Pool` values.
- Use exactly two ticks in the pool:
  - lower tick with positive `liquidity_net`
  - upper tick with negative `liquidity_net`
- Reuse the existing test style in `bounds.rs`; do not add a new fixture file.

### 4.2. Add a mechanical regression for the orientation-dependent conversion

Add a small test that directly encodes the review finding:

- `outcome_price_to_sqrt_limit_x96(3.4025678669876368e38, false)` returns `None`
- `outcome_price_to_sqrt_limit_x96(3.4025678669876368e38, true)` returns
  `Some(U160::from(4295128739u64))`

Purpose: make it impossible for future edits or docs to overstate the overflow
as orientation-independent.

### 4.3. Add the main replay-builder regression

Write the real regression against
[`build_group_plans_for_gas_replay_with_market_context`](../src/execution/bounds.rs#L583-L668),
not `estimate_plan_cost_from_replay`.

Structure:

- actions:
  - a single `Action::Buy`
  - `market_name = "full_range_overflow_case"`
  - choose an `amount` large enough to drive the buy path onto the boundary
    fallback branch
  - use a positive finite `cost`
- gas assumptions:
  - use the existing `test_gas_assumptions()`
- conservative config:
  - use the same values as nearby tests:
    - `quote_latency_blocks: 1`
    - `adverse_move_bps_per_block: 15`

Implementation hint:

- Start with a buy amount in the low-thousands or ten-thousands range, since
  the synthetic full-range pool uses `1_000_000e18` raw liquidity and the test
  needs to push the conservative buy-side limit onto the fallback path.
- If the first amount does not hit the fallback branch, increase the amount
  until the test proves the branch you want. Keep the final chosen value fixed
  in the committed test.

Assertions to keep in the committed test:

- `replay.plans.len() == 1`
- `replay.skipped_groups.is_empty()`
- the only leg has `sqrt_price_limit_x96.is_some()`

Add one stronger assertion to confirm the fallback chose the tick-derived
boundary:

- compute `expected = get_sqrt_ratio_at_tick(887272).unwrap()`
- convert it to `U160` using the same checked conversion method the production
  code uses
- assert the leg's `sqrt_price_limit_x96 == Some(expected)`

Do **not** add any test that "flips a limit negative" in a copied
`MarketPriceState`. That path is unreachable through the production helper and
should not be part of this plan.

## 5. Step 2 — Minimal fix in `market_price_state`

Make the smallest possible change in
[`src/execution/bounds.rs`](../src/execution/bounds.rs).

Current behavior rejects full-range boundary prices because of:

- `buy_limit_price <= 0.0`
- `sell_limit_price <= 0.0`

Change that gate so boundary prices are rejected **only** when they are:

- `NaN`, or
- negative

Keep the `current_price > 0.0` requirement unchanged.

Required intent:

- `sell_limit_price == 0.0` is valid
- `buy_limit_price ~= 3.4e38` is valid
- negative values remain invalid

Concretely, the new validity check should read like:

```rust
if buy_limit_price.is_nan()
    || sell_limit_price.is_nan()
    || buy_limit_price < 0.0
    || sell_limit_price < 0.0
{
    return None;
}
```

Do not rework any other part of `market_price_state`.

## 6. Step 3 — Handle the downstream boundary case in `conservative_sqrt_price_limit`

Step 5 alone is not enough. The huge finite buy limit can still reach
[`outcome_price_to_sqrt_limit_x96`](../src/execution/bounds.rs#L326-L344),
which rejects it when `is_token1_outcome == false`.

### 6.1. Extend `MarketPriceState`

Add two private fields:

- `sqrt_buy_limit: U256`
- `sqrt_sell_limit: U256`

Populate them from the already-computed `sqrt_buy_limit` / `sqrt_sell_limit`
locals in `market_price_state`.

This is a private struct extension only; no external interface changes.

The struct should become:

```rust
struct MarketPriceState {
    is_token1_outcome: bool,
    current_price: f64,
    buy_limit_price: f64,
    sell_limit_price: f64,
    liquidity_raw: f64,
    sqrt_buy_limit: U256,
    sqrt_sell_limit: U256,
}
```

### 6.2. Keep the sell-side clamp logic

Do not rewrite the existing sell-side arithmetic. The current
`.max(state.sell_limit_price)` behavior is correct for a legitimate zero lower
boundary.

Add:

```rust
debug_assert!(state.sell_limit_price >= 0.0);
```

near the top of `conservative_sqrt_price_limit`.

### 6.3. Add a final conversion fallback

Leave the current terminal-price and adverse-price arithmetic intact.

Only replace the final line:

- first call `outcome_price_to_sqrt_limit_x96(adverse_price, state.is_token1_outcome)`
- if it returns `Some(limit)`, use it unchanged
- if it returns `None`, fall back to the precomputed tick boundary:
  - `LegKind::Buy => state.sqrt_buy_limit`
  - `LegKind::Sell => state.sqrt_sell_limit`

This fallback is the correct behavior for the full-range "unbounded in that
direction" sentinel.

The intended shape is:

```rust
match outcome_price_to_sqrt_limit_x96(adverse_price, state.is_token1_outcome) {
    Some(limit) => Some(limit),
    None => {
        let boundary = match leg.kind {
            LegKind::Buy => state.sqrt_buy_limit,
            LegKind::Sell => state.sqrt_sell_limit,
        };
        Some(U160::uint_try_from(boundary)
            .expect("tick-derived sqrt boundary must fit in U160"))
    }
}
```

### 6.4. Use a checked `U256 -> U160` conversion

Do **not** write `U160::from(state.sqrt_buy_limit)` in production code or in
the doc.

Use the checked Uint conversion helper that the `alloy` / `ruint` types support:

- import `alloy::primitives::ruint::UintTryFrom`
- convert with `U160::uint_try_from(...)`
- `.expect("tick-derived sqrt boundary must fit in U160")`

The expectation is valid here because Uniswap V3's extreme sqrt ratios are both
below `2^160`:

- `MIN_SQRT_RATIO = 4295128739`
- `MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342`
- `2^160 = 1461501637330902918203684832716283019655932542976`

So `MAX_SQRT_RATIO < 2^160` by a comfortable margin.

## 7. Files touched

Expected file set:

- [src/execution/bounds.rs](../src/execution/bounds.rs)
  - relax boundary gate
  - extend `MarketPriceState`
  - add checked sqrt-boundary fallback
  - add the new tests
- [docs/ff_benchmark_conservative_repricing_fix.md](../docs/ff_benchmark_conservative_repricing_fix.md)
  - replace the old plan with this one
- [test/fixtures/local_foundry_e2e_benchmark_matrix_single_market.jsonl](../test/fixtures/local_foundry_e2e_benchmark_matrix_single_market.jsonl)
  - regenerate after benchmark verification

Do **not** touch:

- [src/pools/pricing.rs](../src/pools/pricing.rs)
- [src/portfolio/core/rebalancer.rs](../src/portfolio/core/rebalancer.rs)
- [src/bin/local_foundry_e2e_fixture.rs](../src/bin/local_foundry_e2e_fixture.rs)
- anything related to connected-topology FF certification

## 8. Verification

Run checks in this order.

### 8.1. Bounds tests

The repo currently has a generated-file staleness guard, so use the existing
bypass env var during verification unless you are already regenerating those
files for another task.

```bash
DEEP_TRADING_SKIP_GENERATED_STALENESS_CHECK=1 \
  cargo test --release bounds -- --test-threads=1
```

### 8.2. ForecastFlows tests

```bash
DEEP_TRADING_SKIP_GENERATED_STALENESS_CHECK=1 \
  cargo test --release forecastflows
```

### 8.3. Real benchmark entrypoint

```bash
FORECASTFLOWS_WORKER_BIN=/absolute/path/to/forecast-flows-worker \
  forge test --ffi --match-test test_benchmark_matrix_single_market -vv
```

## 9. Success criteria

All of the following must be true:

- the new orientation-dependent conversion test passes
- the new replay-builder regression passes
- `cargo test --release bounds -- --test-threads=1` passes
- `cargo test --release forecastflows` passes
- `test_benchmark_matrix_single_market` passes
- the regenerated single-market JSONL ForecastFlows row has:
  - empty `skip_reason`
  - non-zero execution numerics
  - non-zero `action_count`
- connected-topology ForecastFlows status is unchanged or incidentally improved,
  but no connected-topology fix is required for this PR

## 10. Explicit assumptions

- This remediation is intentionally limited to the full-range single-market
  boundary issue
- The replay-only gate in `evaluate_forecastflows_action_set` is correct and
  stays unchanged
- The truncation in `pricing.rs` is a real boundary signal and stays unchanged
- The checked tick-boundary fallback is expected to trigger only on the
  overflowing orientation (`is_token1_outcome == false`)
