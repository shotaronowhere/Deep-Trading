# Global Solver Plan History

Date: 2026-02-18
Owner: portfolio/global solver path

## 1. Context and Baseline

Baseline regression command:

```bash
cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
```

Baseline metrics recorded before this repair cycle:

- `total_cases=50`
- `candidate_valid=true` in `2/50`
- `mean_delta=-40.050089682532`

Dominant failure class:

- Solve-to-execution mismatch, where optimized continuous state and emitted action stream diverged.

Post-revision run (current implementation state):

- `total_cases=50`
- `candidate_valid=true` in `2/50`
- `mean_delta=-42.910196918399`
- invalid reasons:
  - `ProjectedGradientTooLarge=44`
  - `ProjectionUnavailable=4`

## 2. Original Plan (Frozen Snapshot)

This section is an immutable historical record.
Text below is copied verbatim from `/tmp/gemini_global_solver_plan_review_prompt.txt`.

```text
# Global Solver Repair Plan (Execution-Faithful, Single-Tick First)

## Summary
Baseline on 2026-02-18 from
cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1:
- total_cases=50
- candidate_valid=true only 2/50
- mean_delta=-40.050089682532

The fix will target the dominant failure mode: solved continuous state does not map to executable actions.
Core direction: keep production in Rust, keep single-tick scope for this cycle, and redesign the global candidate so optimization variables are directly execution-faithful (no ad hoc projection artifacts).

## Public API / Interface Changes
1. Add GlobalCandidateInvalidReason enum (public) with explicit reasons:
- ProjectionUnavailable
- ReplayCashMismatch
- ReplayHoldingsMismatch
- ProjectedGradientTooLarge
- NonFiniteSolveState
- NoExecutableMintSellShape
2. Extend RebalanceDecisionDiagnostics in src/portfolio/core/rebalancer.rs with additive fields:
- candidate_invalid_reason: Option<GlobalCandidateInvalidReason>
- candidate_projected_grad_norm: f64
- candidate_replay_cash_delta: f64
- candidate_replay_holdings_delta: f64
3. Extend GlobalSolveConfig in src/portfolio/core/global_solver.rs with additive solver controls:
- optimizer: GlobalOptimizer (ProjectedNewton, Lbfgsb)
- trade_l2_reg: f64 (tiny conditioning regularizer)

## Implementation Plan
1. Instrument failure reasons before behavior changes.
- Update src/portfolio/core/global_solver.rs so validity is reasoned, not boolean-only.
- Update src/portfolio/tests/fuzz_rebalance.rs to print reason histograms in EV-compare output.
- Verify: the 50-case report includes per-reason counts.

2. Replace (u_i, m, r) with execution-faithful variables.
- In src/portfolio/core/global_solver.rs, reformulate to:
  - buy_i >= 0
  - sell_i >= 0
  - theta (net complete-set shift; theta>0 mint, theta<0 merge)
- Derived state:
  - hold_i_final = hold_i_initial + buy_i - sell_i + theta
  - cash_final = cash_initial - sum_i buy_cost_i(buy_i) + sum_i sell_proceeds_i(sell_i) - theta
- Enforce feasibility in objective domain with barriers on cash_final and hold_i_final.
- Remove separate m/r degree of freedom (this aligns with prediction-market translation-invariance from D.2.5 and removes gauge drift).

3. Make objective/evaluator smooth and local-edge decomposed.
- Keep existing single-tick formulas (PoolSim::buy_exact/sell_exact primitives) as local edge models.
- Add analytic derivatives for buy-cost and sell-proceeds branches in separate helpers.
- Add tiny trade_l2_reg to stabilize near-flat regions and discourage pathological buy/sell churn on same market.
- Add a no-trade active-set clamp around zero based on local marginal bid/ask band (CFMMRouter-style no-trade condition).

4. Replace projection logic with deterministic executable mapping.
- Build actions directly from solved variables in fixed order:
  1. mint bracket if theta > DUST and executable sell flow exists,
  2. direct sells (residual),
  3. direct buys,
  4. merge if theta < -DUST.
- Remove artificial sell injection (MIN_MINT_BRACKET_SELL) and related fail paths.
- Keep strict execution shape unchanged (FlashLoan -> Mint -> Sell+ -> RepayFlashLoan) by construction.
- If theta>0 but no sell flow, normalize theta -> 0 before action emission (do not fabricate trades).

5. Eliminate affordability-clamp mismatch for candidate path.
- In global-candidate emission path, do not silently resize solved buys.
- Any inability to execute solved amount is a hard invalid reason (ProjectionUnavailable) instead of hidden bisection shrink.
- This preserves optimizer/execution equivalence contract.

6. Add CFMMRouter-style subproblem interface for future-proofing.
- Add trait in global solver module:
  - local-edge evaluation
  - local arbitrage solve at reference price
- Implement for single-tick PoolSim now and for complete-set synthetic edge.
- Use this interface for diagnostics/parity now and dual decomposition later.

7. Add offline parity harness (ignored test).
- Add ignored test file under src/portfolio/tests/ that:
  - runs reduced fixtures (small-N),
  - optionally invokes CFMMRouter reference flow if CFMMROUTER_JULIA_CMD is set,
  - compares no-arb residual, net-flow conservation, and EV.
- No runtime Julia dependency; test is offline/optional only.

8. Update docs as part of implementation.
- Update docs/global_solver.md.
- Update docs/global_solver_ev_regression.md.
- Add one durable implementation note (new markdown) describing final formulation, validity reasons, and acceptance gates.

## Test Cases and Acceptance Criteria
1. Existing regression gate:
- cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1
- Must satisfy:
  - candidate_valid >= 95% (at least 48/50)
  - mean_delta >= -1e-6

2. New solver unit tests in src/portfolio/core/global_solver.rs:
- feasibility from zero holdings/cash edge cases,
- no fabricated mint-sell leg injection,
- replay equivalence for cash/holdings within tolerance,
- projected-gradient termination sanity.

3. Invariant suites:
- cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture
- cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture
- Must remain green with no new action-shape violations.

4. Optional parity suite:
- ignored test with environment-gated Julia command.
- Must report bounded EV and flow residual deltas on shared fixtures.

## Assumptions and Defaults
1. Production implementation remains Rust-native.
2. Scope for this cycle is single-tick correctness only; multi-tick stays separate.
3. Objective remains pure EV (no gas/MEV netting in this cycle).
4. Additive public API changes are allowed.
5. Rust solver crates are allowed; default path should stay deterministic and fail-closed.
6. RebalanceEngine::Incumbent remains default until acceptance targets are met.
```

## 3. External Review Summary (Gemini)

Severity-grouped outcomes and adoption decisions for this cycle:

`fix_now`

- Remove projection-time `theta -> 0` mutation when mint flow is infeasible.
  - Decision: Adopted.
- Add explicit budget epsilon contract for exact action emission.
  - Decision: Adopted.
- Handle non-smooth no-trade region near zero buy/sell by explicit band clamp.
  - Decision: Adopted.

`worth_fixing_soon`

- Emit explicit invalid reason when mint exists but no executable sell flow exists.
  - Decision: Adopted as `MintWithoutSellFlow`.
- Add richer invalidity telemetry to EV regression harness.
  - Decision: Adopted.

`defer`

- CFMMRouter parity harness and deeper dual-decomposition interface.
  - Decision: Deferred until post-fix cycle.

`rejected_or_adjusted`

- Recommendation to always execute mint even without sell legs.
  - Decision: Rejected in current architecture because grouping/execution shape is strict `Mint -> Sell+` for flash brackets.

## 4. Critique of Review

The review suggestion to always execute mint without sell flow conflicts with the current execution grammar and planner constraints.

- `src/execution/grouping.rs` defines `MintSell` bracket shape as `FlashLoan -> Mint -> Sell+ -> RepayFlashLoan`.
- Under current architecture, mint-without-sell is not a valid flash bracket and cannot be auto-converted into executable flow.
- Therefore this cycle keeps fail-closed semantics via explicit invalid reason (`MintWithoutSellFlow`) rather than silent shape coercion.

## 5. Modified Plan (Authoritative)

This section is the authoritative revised plan used for implementation.

```text
# Global Solver Plan Record + Implementation Plan

## Summary
This plan does two things in one pass:
1. Permanently documents both the original and revised solver plans (with rationale and diffs).
2. Implements the revised plan in the Rust codebase, with regression gates tied to current EV-failure metrics.

Baseline to record (from 2026-02-18):
- Command: `cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1`
- `total_cases=50`
- `candidate_valid=true` in 2/50
- `mean_delta=-40.050089682532`

## Phase 1: Documentation (Thorough Record)

### 1. Create new canonical plan-history document
Create `/Users/shotaro/proj/deep_trading/docs/global_solver_plan_history.md` with these sections:

1. `Context and Baseline`
- Current regression metrics and command.
- Known dominant failure class: solve-to-execution mismatch.

2. `Original Plan (Frozen Snapshot)`
- Include the full original plan content exactly as previously proposed (execution-faithful reformulation, telemetry, projection replacement, parity harness, etc.).
- Mark this section as immutable historical record.

3. `External Review Summary (Gemini)`
- Include exact review outcomes grouped by severity.
- Include adoption decision per item:
- Adopted: remove `theta -> 0` projection mutation, budget epsilon contract, zero-trade non-smooth handling.
- Rejected/adjusted: “always execute mint even without sells” (conflicts with strict bracket rules).

4. `Critique of Review`
- Explain why strict `MintSell` shape in `/Users/shotaro/proj/deep_trading/src/execution/grouping.rs` requires `Sell+`.
- Explain that in current architecture, mint without sell flow is invalid, not auto-executable.

5. `Modified Plan (Authoritative)`
- Full revised plan text.
- Explicit “what changed from original” matrix:
- Removed: projection-time `theta` normalization.
- Added: `MintWithoutSellFlow` invalid reason.
- Added: `solver_budget_eps`, `zero_trade_band_eps`, `theta_l2_reg`.
- Deferred: CFMMRouter parity harness to post-fix cycle.

6. `Acceptance Contract`
- `candidate_valid >= 95%` on 50-case regression corpus.
- `mean_delta >= -1e-6`.
- No regression in invariants tests.

7. `Rollout and Fallback`
- Keep `RebalanceEngine::Incumbent` as default.
- Candidate remains fail-closed.

### 2. Update solver docs to reflect revised, not original, behavior
1. Update `/Users/shotaro/proj/deep_trading/docs/global_solver.md`.
- Replace any mention of projection-time sell injection and `theta`-like mutation behavior.
- Document new variable model and validity-reasoned gating.
- Document budget epsilon contract and zero-trade clamp.

2. Update `/Users/shotaro/proj/deep_trading/docs/global_solver_ev_regression.md`.
- Add section `Post-Revision Invalidity Taxonomy`.
- Add expected telemetry fields and histogram interpretation.

## Phase 2: Implementation (Revised Plan)

### 1. Add validity reason model and diagnostics plumbing
1. In `/Users/shotaro/proj/deep_trading/src/portfolio/core/global_solver.rs`:
- Add `pub enum GlobalCandidateInvalidReason`:
- `ProjectionUnavailable`
- `ReplayCashMismatch`
- `ReplayHoldingsMismatch`
- `ProjectedGradientTooLarge`
- `NonFiniteSolveState`
- `MintWithoutSellFlow`
- `BudgetEpsilonViolation`

2. Extend `GlobalCandidatePlan` to carry:
- `invalid_reason: Option<GlobalCandidateInvalidReason>`

3. In `/Users/shotaro/proj/deep_trading/src/portfolio/core/rebalancer.rs`:
- Extend `RebalanceDecisionDiagnostics` with additive fields:
- `candidate_invalid_reason: Option<GlobalCandidateInvalidReason>`
- `candidate_projected_grad_norm: f64`
- `candidate_replay_cash_delta: f64`
- `candidate_replay_holdings_delta: f64`
- Populate these for all engine branches without breaking incumbent defaults.

### 2. Reformulate solver variables and objective
In `/Users/shotaro/proj/deep_trading/src/portfolio/core/global_solver.rs`:

1. Replace decision vector `(u_i, m, r)` with:
- `buy_i >= 0` for each market
- `sell_i >= 0` for each market
- `theta` (net complete-set shift)

2. Derived state:
- `hold_i_final = hold_i_initial + buy_i - sell_i + theta`
- `cash_final = cash_initial - Σ buy_cost_i(buy_i) + Σ sell_proceeds_i(sell_i) - theta`

3. Keep interior feasibility barriers on:
- `cash_final > 0`
- `hold_i_final > 0`

4. Add objective regularization:
- `-theta_l2_reg * theta^2`

5. Add config fields (with defaults):
- `theta_l2_reg`
- `solver_budget_eps`
- `zero_trade_band_eps`

### 3. Non-smooth zero-trade handling
In `/Users/shotaro/proj/deep_trading/src/portfolio/core/global_solver.rs`:

1. Introduce explicit marginal checks for buy/sell at near-zero trade.
2. If local gradient lies in no-trade spread band (within `zero_trade_band_eps`), pin trade coordinate to zero in that iteration/update.
3. Keep projected gradient convergence criteria, but apply after clamp.

### 4. Make projection strictly faithful (no hidden mutation)
In `/Users/shotaro/proj/deep_trading/src/portfolio/core/global_solver.rs` and `/Users/shotaro/proj/deep_trading/src/portfolio/core/trading.rs`:

1. Remove projection-time artificial sell injection (`MIN_MINT_BRACKET_SELL` flow).
2. Remove any behavior equivalent to forcing solved `theta` to zero.
3. Candidate emission rules:
- If `theta > DUST`, emit mint bracket only if executable `Sell+` exists.
- If no executable sell exists, fail with `MintWithoutSellFlow`.
- If direct buy amount cannot be executed exactly within `solver_budget_eps`, fail with `BudgetEpsilonViolation`.
4. Do not use affordability resize/bisection in candidate emission path.

### 5. Candidate validity gate becomes reasoned, not opaque
In `/Users/shotaro/proj/deep_trading/src/portfolio/core/global_solver.rs`:

1. Replace boolean-only validity condition with reason selection in priority order.
2. Preserve fail-closed behavior:
- Candidate invalid/unavailable falls back to incumbent.
3. Continue replay cash/hold checks, now mapped to explicit reason fields.

## Phase 3: Tests and Verification

### 1. Extend EV regression test reporting
In `/Users/shotaro/proj/deep_trading/src/portfolio/tests/fuzz_rebalance.rs`:

1. Add counters for each invalid reason.
2. Print summary histogram in `test_compare_global_vs_incumbent_ev_across_rebalance_fixtures`.

### 2. Add/adjust global solver unit tests
In `/Users/shotaro/proj/deep_trading/src/portfolio/core/global_solver.rs` test module:

1. `theta` not silently normalized when mint flow infeasible.
2. Infeasible mint-without-sell returns `MintWithoutSellFlow`.
3. Budget epsilon boundary produces `BudgetEpsilonViolation` only beyond tolerance.
4. Zero-trade clamp behavior near branch point.
5. Replay equivalence for valid candidates.

### 3. Run verification loop
Run in this order:

1. `cargo test test_compare_global_vs_incumbent_ev_across_rebalance_fixtures -- --ignored --nocapture --test-threads=1`
2. `cargo test test_fuzz_rebalance_end_to_end_full_l1_invariants -- --nocapture`
3. `cargo test test_fuzz_rebalance_end_to_end_partial_l1_invariants -- --nocapture`
4. `cargo test` (full suite if feasible)

Record in doc:
- validity rate
- mean delta
- reason histogram
- any residual failing scenarios

## Acceptance Criteria
1. `candidate_valid >= 95%` (48/50+) on EV comparison corpus.
2. `mean_delta >= -1e-6`.
3. No regression in action-shape/invariant tests.
4. Updated docs reflect actual code behavior and include original vs revised plan history.

## Assumptions and Defaults
1. Keep solver production path Rust-native.
2. Keep single-tick scope for this cycle.
3. Keep objective as pure EV for this cycle.
4. Keep `RebalanceEngine::Incumbent` default until acceptance metrics hold.
5. Defer offline CFMMRouter parity harness until after core regression fix lands.
```

### What Changed from Original

| Area | Original Plan | Modified Plan |
| --- | --- | --- |
| Theta infeasible handling | Allowed projection-time `theta -> 0` normalization | Removed hidden normalization; fail with explicit invalid reason path |
| Mint feasibility reason | `NoExecutableMintSellShape` placeholder | Concrete `MintWithoutSellFlow` in code and telemetry |
| Emission budget contract | Implicit/partial | Explicit `solver_budget_eps` contract with `BudgetEpsilonViolation` |
| Zero-trade non-smoothness | Mentioned with trade regularizer concept | Explicit `zero_trade_band_eps` clamp in iterative updates |
| Regularization | Proposed `trade_l2_reg` | Implemented `theta_l2_reg` |
| Parity harness | Planned in-cycle | Deferred to post-fix cycle |

## 6. Acceptance Contract

- `candidate_valid >= 95%` on the 50-case EV comparison corpus.
- `mean_delta >= -1e-6` on the same corpus.
- No regression in invariant/action-shape suites.

Current status: not yet met (`candidate_valid=2/50`, `mean_delta=-42.910196918399`).

## 7. Rollout and Fallback

- `RebalanceEngine::Incumbent` remains default.
- Global candidate remains fail-closed.
- `AutoBestReplay` and `GlobalCandidate` continue to fallback to incumbent when candidate is invalid/unavailable.

## 8. Stage Progress Update (2026-02-18)

### Implemented Stage Set (1-3)

- Stage 1: barrier/feasibility retune
  - `barrier_mu_cash=1e-8`, `barrier_mu_hold=1e-8`, `barrier_shift=1e-4`
  - reduced repair cash target
  - line-search only repairs invalid trials
- Stage 2: coupled residual gating
  - added `coupled_feasibility_residual` to solve result + diagnostics
  - high projected gradient only invalidates when coupled residual is also high
- Stage 3: incumbent warm start
  - seed solver from incumbent actions (`buy/sell/theta` map)
  - fallback to unseeded solve if warm-started solve fails

### Result After Stage 1-3

- `total_cases=50`
- `candidate_valid=50`
- `candidate_invalid=0`
- `mean_delta=-22.407627450367`
- `best_delta=0.000000000000`
- `worst_delta=-106.491017066174`

### Residual Gap Localization

From parsed case-level output:

- Full-L1 subset (`n=26`): `mean_delta=-43.091591245551`
- Partial-L1 subset (`n=24`): `mean_delta=-0.000000005584`

Interpretation: residual EV gap is concentrated in complete-set/full-L1 regimes, not in partial-L1 direct-only regimes.

### Stage 4 Attempt (Rejected)

- Attempted coupled pre-projection in each line-search trial.
- Outcome: severe regression (`mean_delta=-45.841560299366`, `worst_delta=-173.950186182147`, runtime ~156s).
- Case-level pattern: near-universal `solver_iters=1024` with `ls_trials~10k`.
- Decision: reverted; not adopted.

## 9. Phase A Doc Pointer (2026-02-18)

- Canonical Phase A implementation spec (doc-first): `docs/global_solver_lbfgsb_phaseA_plan.md`
- Scope locked to bounded L-BFGS-B + active-set solver upgrade with unchanged execution/replay semantics.

## 10. Phase A Implementation Result (2026-02-18)

- Detailed execution record and gate outcomes: `docs/global_solver_stage_report_2026-02-18.md`
- Final accepted status:
  - `candidate_valid=50/50`
  - `mean_delta=-19.934197985691`
  - full-L1 `mean_delta=-38.334996121988`
  - full-L1 `mean_solver_iters=11.115385`
  - full-L1 `mean_hold_clamps=2450.192308`

## 11. Phase B Doc Pointer (2026-02-18)

- Canonical Phase B spec (doc-first): `docs/global_solver_phaseB_dual_plan.md`
- Scope: dual/decomposition prototype behind config flag; default runtime path unchanged.

## 12. Phase B Prototype Result (2026-02-18)

- Implemented behind `GlobalOptimizer::DualDecompositionPrototype` (non-default).
- Default-path EV regression remained unchanged:
  - `candidate_valid=50/50`
  - `mean_delta=-19.934197985691`
- Dual override (`GLOBAL_SOLVER_OPTIMIZER=dual`) result:
  - `candidate_valid=50/50`
  - `mean_delta=-17.550977387382`
  - full-L1 `mean_delta=-33.751879586778`
- Status: promising but still below incumbent EV; retained as experimental path.

## 13. Phase C Doc Pointer (2026-02-18)

- Canonical Phase C spec (doc-first): `docs/global_solver_phaseC_convergence_plan.md`
- Key finding before implementation:
  - increasing solver budget (`max_iters=4096`, `max_line_search_trials=256`) does not improve EV;
  - planned fix targets line-search failure recovery, not raw iteration count.

## 14. Phase C Implementation Result (2026-02-18)

- Added bounded post-Armijo rescue path (projected-gradient fallback with strict-decrease acceptance).
- Added config knobs:
  - `line_search_rescue_trials` (default `16`)
  - `line_search_rescue_min_decrease` (default `1e-12`)
- Added telemetry:
  - `line_search_rescue_attempts`
  - `line_search_rescue_accepts`

Validation:

- Solver/unit and invariant tests remained green.
- EV regression (default optimizer):
  - `candidate_valid=50/50`
  - `mean_delta=-18.176103599424` (improved from `-19.934197985691`)
  - full-L1 `mean_delta=-34.954045379167` (improved from `-38.334996121988`)
  - `candidate_better=0/50` (still no incumbent outperformance)
  - post-change `GLOBAL_SOLVER_MAX_ITERS=4096` produced identical EV summary (no extra gain from longer run)

Status:

- Convergence quality improved, but incumbent EV parity remains unsolved.
- Next phase should target structural coupled-constraint conditioning (not just more iterations).

## 15. Progress Summary Pointer (2026-02-18)

- Consolidated executive summary:
  - `docs/global_solver_ev_progress_summary_2026-02-18.md`
