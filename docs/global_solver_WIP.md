# Global Solver WIP Findings

Date: 2026-02-17  
External reviewer: Gemini CLI (`gemini-3-pro-preview`)

## Context Passed to Review

- Core implementation:
  - `src/portfolio/core/global_solver.rs`
  - `src/portfolio/core/rebalancer.rs`
  - `src/portfolio/core/sim.rs`
  - `src/portfolio/core/planning.rs`
  - `src/portfolio/core/trading.rs`
  - `src/portfolio/core/diagnostics.rs`
- Tests/docs:
  - `src/portfolio/tests.rs`
  - `src/portfolio/tests/fuzz_rebalance.rs`
  - `docs/global_solver.md`
  - `docs/portfolio.md`
  - `docs/architecture.md`
  - `docs/TODO.md`
- Reference material:
  - [An Optimal Routing Algorithm for Constant Function Market Makers](https://angeris.github.io/papers/routing-algorithm.pdf)
  - [Diamandis PhD thesis, section D.2.5](https://dspace.mit.edu/bitstream/handle/1721.1/158483/diamandis-tdiamand-phd-eecs-2024-thesis.pdf)

## Findings

### 1) `fix_now`: false mint feasible-region cap

- Location: `src/portfolio/core/global_solver.rs` (`build_bounds`, around `m_cap` construction).
- Previous behavior: `m_cap = min_i (sell_cap_i + hold_i_initial + buy_cap_i)`.
- Problem: one shallow pool can force `m_cap` near zero, globally suppressing valid strategies where minting is followed by selective selling and/or holding inventory.
- Impact: candidate solver can lose EV versus incumbent even when higher-EV mint routes exist.
- Status (2026-02-17): implemented. `m_cap` now uses a loose finite global cap in code (`cash + Σ(buy_cap + sell_cap + hold_pos)`) instead of shallowest-pool minimum coupling.

### 2) `worth_fixing_soon`: barrier weights likely too high for boundary-near optimum

- Location: `src/portfolio/core/global_solver.rs` and config defaults in `src/portfolio/core/rebalancer.rs`.
- Previous defaults: `barrier_mu_cash = 1e-4`, `barrier_mu_hold = 1e-6`.
- Problem: barriers can keep solution away from budget/holding boundaries that incumbent may exploit, leaving EV unused.
- Impact: persistent EV gap even when feasibility model is otherwise correct.
- Status (2026-02-17): implemented baseline retune. Defaults lowered to `barrier_mu_cash = 1e-7`, `barrier_mu_hold = 1e-7`.

### 3) `defer`: single-tick bounds/cost for per-outcome trades

- Location: `src/portfolio/core/global_solver.rs` + `PoolSim` local formulas.
- Current behavior: `u_i` bounds and derivatives are single-tick local depth only.
- Problem: cannot represent cumulative depth/cost across crossed ticks in richer Uniswap V3 liquidity shapes.
- Impact: candidate remains conservative for multi-tick pools until piecewise integrated modeling is added.

## Patch Direction (Decision Log)

### A. Mint bound correction (implemented)

- Decouple `m_cap` from shallowest `sell_cap_i`.
- Keep `u_i` box bounds on direct trades.
- Use a loose finite `m_cap` guardrail (budget/replay validity remain primary safety checks), then keep:
  - `r_cap = min_i (hold_i_initial + buy_cap_i + m_cap)`.

### B. Barrier tuning pass (implemented baseline)

- Lower `barrier_mu_cash` and `barrier_mu_hold` defaults and verify convergence/KKT residual behavior.
- If needed, add continuation-style barrier decay while preserving fail-closed candidate validation.

### C. Multi-tick upgrade (later phase)

- Replace single-tick local cost/derivative with piecewise integrated multi-tick primitives.
- Replace local tick caps with cumulative reachable depth bounds.
- Keep `AutoBestReplay` adjudication unchanged.

## Verification Criteria After A+B (executed)

- Candidate validity still passes replay/state consistency checks.
- No action invariant regressions.
- EV comparison on rebalance corpus improves or is non-inferior vs incumbent.
- Projected gradient norm and monotone line-search tests remain green.
