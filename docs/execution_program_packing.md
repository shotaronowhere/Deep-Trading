# Execution Program Packing

Status: current as of 2026-03-09.

## Purpose

The off-chain solver no longer treats the discovered micro-action trace as the literal execution program.

Runtime behavior is split into two layers:

1. `Discovery`
   - the existing off-chain planner finds a profitable target adjustment
   - profitability search, route selection, and waterfall allocation are unchanged at this layer
   - the rich discovery trace is then compacted by comparing:
     - `baseline_step_prune`
     - `target_delta` re-emission from the rich terminal holdings
     - `analytic_mixed` compact common-shift solver
     - `coupled_mixed` continuous mixed-frontier solver
     - `direct_only` compact no-mint/no-merge guard
     - `noop`
2. `Compilation`
   - the discovered strict subgroup plan is compiled into a gas-capped execution program
   - consecutive strict subgroups are greedily packed into transaction chunks

This changes the optimization target from:

- `micro-action trace -> one subgroup tx each`

to:

- `micro-action trace -> compact action program -> packed execution program -> net-EV ranking`

## Core Objects

### Strict subgroup

The strict subgroup is still the safety and planning primitive:

- it has a single route-kind shape
- it has one conservative bound set
- it preserves profitability-step boundaries
- it remains the fallback submission unit if packing fails

### Packed transaction chunk

A packed chunk is:

- an ordered list of **consecutive** strict subgroup plans
- executed as one `TradeExecutor.batchExecute(Call[])` transaction
- priced as one tx envelope / one L1 data payload

Packing constraints:

- preserve subgroup order exactly
- do not reorder non-consecutive groups
- do not algebraically fuse route families
- estimated summed L2 gas must stay below `40_000_000`
- concatenated unsigned `batchExecute(Call[])` tx bytes must remain locally buildable

### Execution program

For any discovered action plan, the runtime compares two execution programs:

- `Strict`
  - one tx per strict subgroup
- `Packed`
  - greedily pack consecutive strict subgroups into the fewest tx chunks under the gas cap

The cheaper valid program becomes the economic representation of that candidate.

## Ranking

Candidate ranking is now based on **packed-program net EV**:

- `estimated_net_ev = raw_ev - estimated_execution_cost`

Execution cost includes only:

- Optimism L2 execution gas
- Optimism L1 data fee

Pool fees are not subtracted again; they are already embedded in the swap math.

Comparator order:

1. higher estimated net EV
2. higher raw EV within net-EV tolerance
3. fewer tx chunks
4. fewer actions
5. stable deterministic solver ordering

## Discovery vs compilation

The route-gating logic during discovery is intentionally lighter than final execution pricing.

Discovery uses incremental gas thresholds:

- per-group L2 execution gas
- incremental L1 data cost without charging a full tx envelope to every subgroup

Final execution scoring uses packed-program pricing:

- exact local unsigned tx shape for each chunk
- shared `l1_fee_per_byte_wei` snapshot
- one tx envelope per chunk, not per subgroup

This avoids baking the fragmented execution topology into the planner.

## Submission behavior

Runtime submission now targets the packed execution program by default:

1. build strict subgroup plans as before
2. compile them into packed tx chunks
3. submit one `batchExecute(Call[])` transaction for the first chunk
4. replan only after that chunk confirms

If a packed chunk cannot be safely assembled with current bounds:

- runtime falls back to strict submission for that step
- it does **not** fall back to the legacy staged solver

## Current rollout status

- packed execution is the default runtime target
- strict execution remains the safety fallback
- the legacy staged solver is kept only as an optional reference path during rollout
- the off-chain benchmark matrix should be interpreted in terms of packed tx chunks, not strict subgroup count

## Design boundary

This is a deliberate v1 implementation:

- packing only consecutive strict subgroups
- no subgroup reordering
- no algebraic action fusion
- compact action synthesis currently uses four canonical compact forms:
  - `target_delta` normal form (`Mint`/`Merge` common shift plus residual direct buys/sells)
  - `analytic_mixed` frontier-normal form (one compact common shift plus residual direct trades)
  - `coupled_mixed` profitable-prefix mixed frontier program (continuous `pi` / `m`, then one compact mint-plus-residual plan)
  - `direct_only` compact direct frontier program

If residual off-chain underperformance remains after packing, the remaining gap should be treated as a search / route / target-state problem rather than a pure execution-fragmentation problem.
