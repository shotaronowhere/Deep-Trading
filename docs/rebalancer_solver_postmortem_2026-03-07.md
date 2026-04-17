# Rebalancer Solver Postmortem (2026-03-07)

## Purpose

This note records the direct-solver experiment that added bounded first-boundary
clipping and the adaptive exact-subset route, and why both were removed.

The goal is to avoid repeating the same expensive detour.

## What we tried

We explored two new on-chain direct solvers on top of the existing constant-`L`
and exact routes:

1. A bounded constant-`L` route that clipped planning to the first tick
   boundary and limited itself to two buy passes.
2. An adaptive hybrid route that exact-scanned only a capped subset of
   cross-risk pools and used constant-`L` for the rest.

The motivation was reasonable:

- keep slippage-sensitive planning on-chain,
- handle liquidity-bound multi-tick states better than plain constant-`L`,
- stay comfortably below the 40M gas ceiling on the 98-market case.

## What happened

The bounded route failed the important benchmark bar.

- On realistic crossing-heavy 98-market fixtures it saved gas by refusing to
  spend into profitable depth.
- That was not a real improvement. It was just a worse approximation with
  better gas.

The adaptive route recovered some EV, but not enough to justify becoming a
permanent third direct route.

- On the realistic 98-market fixture it matched legacy EV while costing more gas.
- On the synthetic 98-market fixture it improved EV only slightly.

The exact route was the only complex route that clearly paid for itself, because
it still fit under the 40M gas cap and materially improved EV on the realistic
crossing-heavy benchmark.

## Last benchmark table before removal

These were the last measured results from the experimental branch before the
bounded/adaptive routes were deleted.

### Synthetic 98-outcome multi-tick fixture

| Route | Gas | EV |
|---|---:|---:|
| legacy constant-`L` | `3,621,921` | `319,743,852,785,843,393,829,282` |
| bounded constant-`L` | `8,466,185` | `319,297,075,833,328,704,894,202` |
| adaptive hybrid | `11,231,890` | `319,982,144,275,727,205,264,339` |
| exact | `38,960,737` | `320,463,674,792,566,460,803,522` |

Takeaway:

- bounded was worse than legacy on both EV and gas
- adaptive beat legacy by only about `0.0745%` EV

### Realistic seeded 98-outcome multi-tick fixture

| Route | Gas | EV |
|---|---:|---:|
| legacy constant-`L` | `17,152,999` | `329,064,319,625,327,189,099,520,213` |
| bounded constant-`L` | `8,456,777` | `147,581,063,961,692,811,347,758,042` |
| adaptive hybrid | `20,342,735` | `329,064,319,625,327,189,099,520,213` |
| exact | `36,842,078` | `425,223,125,100,456,027,087,034,824` |

Takeaway:

- bounded was catastrophically worse on EV
- adaptive matched legacy EV but cost more gas
- exact improved EV by roughly `29%` while staying below the gas ceiling

## What survived

The surviving direct frontier is:

1. `rebalance`: simple multi-pass constant-`L`
2. `rebalanceExact`: explicit exact multi-tick solver

Useful changes that were kept:

- caller-supplied coarse profit floors for arb and recycle
- budget-capped exact ladder construction

Useless changes that were removed:

- first-boundary clipped bounded route
- adaptive exact-subset route
- the extra direct-route API surface that only existed to support those paths

## Design rules going forward

These are the rules this experiment earned:

1. Simpler is better unless EV moves materially on realistic 98-market states.
2. Fractions of a percent of EV are not enough to justify a much more complex
   on-chain planner.
3. If exact already fits under the gas cap, prefer optimizing exact over adding
   another approximate route.
4. Never ship a new direct solver without benchmark results for:
   - deep-crossing two-pool,
   - synthetic 98-outcome multi-tick,
   - realistic seeded 98-outcome multi-tick.
5. A new approximate route must beat `rebalance` on realistic crossing-heavy
   fixtures without introducing a severe regression on any benchmark state.

## Current state after cleanup

Measured again on March 7, 2026 after removing the experimental routes:

### Two-pool multi-tick fixture

- `rebalance`: `213,077` gas, EV `6,533,623,513,833,670,712,014`
- `rebalanceExact`: `559,645` gas, EV `6,538,044,206,827,416,201,383`

### Synthetic 98-outcome multi-tick fixture

- `rebalance`: `3,617,331` gas, EV `319,743,852,785,843,393,829,282`
- `rebalanceExact`: `38,909,824` gas, EV `320,463,674,792,566,460,803,522`

### Realistic seeded 98-outcome multi-tick fixture

- `rebalance`: `17,148,395` gas, EV `329,064,319,625,327,189,099,520,213`
- `rebalanceExact`: `36,827,551` gas, EV `425,223,125,100,456,027,087,034,824`

This is the on-chain direct frontier until new evidence says otherwise.
