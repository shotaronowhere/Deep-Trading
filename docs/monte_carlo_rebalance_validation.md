# Monte Carlo Rebalance EV Validation

## Objective

Validate that rebalance actions do not reduce expected value under a broad set of sampled
portfolio states, while reporting coverage across profitability-step route families.

The pass condition is strict EV non-decrease per trial:

- `EV_after >= EV_before - tol`
- `tol = 1e-8 * (1 + max(|EV_before|, |EV_after|))`

Flash loans are treated as liquidity plumbing only and are included via replayed cash neutrality
(borrow and repay), with no explicit fee model in this validation.

Replay accounting note:

- `Mint(amount)` consumes `amount` cash (collateral outlay).
- `Merge(amount)` returns `amount` cash.
- `FlashLoan`/`RepayFlashLoan` are modeled as temporary cash bracket entries.

The stress summary reports EV delta extremes (best/worst), not averages:

- `max` EV delta: strongest improvement found for the current seed/run
- `min` EV delta: tightest margin to the non-decrease boundary

Optional convergence checkpoints can be emitted during long runs:

- `[monte-carlo][checkpoint] trials=..., group_steps=..., max_delta=..., min_delta=...`

## Test Entry Points

Both Monte Carlo tests are ignored by default (opt-in).

- Smoke test (fast, debug-safe):
  - `cargo test test_monte_carlo_ev_smoke_profitability_groups -- --ignored --nocapture --test-threads=1`
- Full test (long-running; intended for `--release`):
  - `cargo test --release test_monte_carlo_ev_full_profitability_groups -- --ignored --nocapture --test-threads=1`

## Scenario Mix

The Monte Carlo runner executes deterministic baselines first, then prioritizes fuzzing:

1. `full_underpriced_baseline` (run once at trial 0)
   - Full eligible L1 market set.
   - Price multiplier `0.5x` prediction for every market.
   - Exercises arb and mixed route behavior.
2. `direct_only_baseline` (run once at trial 1)
   - Two-market partial snapshot (disables mint/merge availability).
   - Overpriced legacy holding plus underpriced alternative.
   - Exercises direct route behavior.
3. `fuzz_full` (remaining even trials)
   - Random full snapshot via existing `build_rebalance_fuzz_case(..., false)`.
4. `fuzz_partial` (remaining odd trials)
   - Random partial snapshot via existing `build_rebalance_fuzz_case(..., true)`.

## Stopping Criteria

The full run stops when either bound is reached:

- `MC_TRIALS` trials (default `200000`)
- `MC_MAX_GROUP_STEPS` cumulative profitability-step groups (default `1000000`)

This caps runtime while honoring group-step bounded validation.

## Configuration

Environment variables for the full ignored test:

- `MC_TRIALS` (default `200000`)
- `MC_MAX_GROUP_STEPS` (default `1000000`)
- `MC_SEED` (default `0xC0DE1BAD5EED`; accepts decimal or `0x`-prefixed hex)
- `MC_START_TRIAL_INDEX` (default `0`; useful to target specific scenario templates)
- `MC_CONVERGENCE_EVERY_GROUP_STEPS` (default `off`; set e.g. `100000` for checkpoint output)
- `MC_REQUIRE_FAMILY_COVERAGE` (default `true`; set `false` for targeted single-scenario runs)

Smoke test uses fixed built-in settings for fast local feedback.

## Coverage Requirements

The run fails unless at least one trial is observed for each family:

- direct-only
- mixed-route
- arb/indirect

It also prints per-step-kind counts for:

- `arb_mint_sell`
- `arb_buy_merge`
- `pure_direct_buy`
- `pure_direct_sell`
- `pure_direct_merge`
- `mixed_direct_buy_mint_sell`
- `mixed_direct_sell_buy_merge`

## Runtime Expectations

Measured on this machine before adding this harness:

- Full underpriced (98 outcomes): about `37ms/call` in release
- Full near-fair worst case (98 outcomes): about `470ms/call` in release
- Partial underpriced (64 outcomes): about `0.5ms/call` in release

On this machine, the full default stress run is in the ~5 minute class, while smoke remains fast.

## Independent Random Search Oracle

To battle-test waterfall against a true Monte Carlo baseline, we also provide an ignored test
that does not call `rebalance` for its search decisions:

- `cargo test --release test_random_group_search_vs_waterfall_complex_fuzz_cases -- --ignored --nocapture --test-threads=1`

What it does:

- Builds fixed complex fuzz cases (full + partial).
- Computes algorithm EV once for comparison.
- Runs independent random rollout search over group kinds:
  - `direct_buy`
  - `direct_sell`
  - `mint_sell`
  - `buy_merge`
- Enforces route feasibility by scenario:
  - partial snapshots: direct-only random moves (`direct_buy`, `direct_sell`)
  - full snapshots: direct + indirect random moves
- Uses randomized action sizes per group and random group sequences.
- Reports best EV found, gap to algorithm, convergence checkpoints, and per-kind coverage.

This is intentionally brute-force/stochastic and can underperform waterfall if search budget is
too small.

### Random Search Env Knobs

- `MC_SEARCH_MAX_ROLLOUTS` (default `2000000`)
- `MC_SEARCH_GROUPS_PER_ROLLOUT` (default `8`)
- `MC_SEARCH_CHECKPOINT_EVERY` (default `10000`)
- `MC_SEARCH_MIN_RUNTIME_SECS` (default `300`)
- `MC_SEARCH_STALE_CHECKPOINTS` (default `6`)
- `MC_SEARCH_CONVERGENCE_TOL` (default `1e-9`)
- `MC_SEARCH_CASE_COUNT` (default `4`, max `4`)
- `MC_SEARCH_SEED` (default `0xBAD5EA1234C0FFEE`)
- `MC_SEARCH_ASSERT_ALGO_NOT_WORSE` (default `false`)
- `MC_SEARCH_ALGO_TOL` (default `1e-6`)

`MC_SEARCH_MIN_RUNTIME_SECS` is a floor for stress runs; when set, the loop continues until this
runtime is reached even if `MC_SEARCH_MAX_ROLLOUTS` was already crossed.

### Convergence Output

Each checkpoint prints:

- `best_ev` (max EV seen so far)
- `gap_to_algo` (`algorithm_ev - best_ev`)
- `stale_checkpoints` (consecutive checkpoints without material best-EV improvement)

A run is marked converged when stale checkpoints reach `MC_SEARCH_STALE_CHECKPOINTS`.

## Local Gradient Heuristic (Waterfall Output)

Ignored diagnostic test:

- `cargo test --release test_waterfall_local_gradient_heuristic_complex_cases -- --ignored --nocapture --test-threads=1`

Purpose:

- Replay waterfall output to post-trade portfolio + market state.
- Estimate finite-difference local gradients at that state for:
  - direct directions (`direct_buy`, `direct_sell`)
  - indirect directions (`mint_sell`, `buy_merge`)
- Compare pre-rebalance vs post-rebalance gradient magnitudes.

Configuration:

- `MC_GRAD_EPS` (default `1e-4`)
- `MC_GRAD_CASE_COUNT` (default `4`, max `4`)
- `MC_GRAD_ASSERT_NON_POSITIVE` (default `false`)
- `MC_GRAD_TOL` (default `1e-6`)

Interpretation:

- Positive post-gradient means a local profitable direction still exists at that epsilon scale.
- This is a heuristic diagnostic, not a formal optimality proof.

## Second-Pass Idempotence Check

Ignored diagnostic test:

- `cargo test --release test_rebalance_second_pass_gain_complex_cases -- --ignored --nocapture --test-threads=1`

Purpose:

- Run rebalance on fixed complex cases, replay resulting market/portfolio state, run rebalance again.
- Assert second-pass EV gain is near-zero relative to first-pass gain.

Configuration:

- `MC_SECOND_PASS_CASE_COUNT` (default `4`, max `4`)
- `MC_SECOND_PASS_REL_CAP` (default `0.02`)
- `MC_SECOND_PASS_ABS_CAP` (default `1e-3`)
- `MC_SECOND_PASS_ASSERT` (default `true`)
