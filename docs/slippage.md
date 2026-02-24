# Slippage Guard Sprint Spec (L1/Optimism)

## 1. Scope

We add strict execution guardrails for stale pool states between planning and inclusion.

### In scope
1. Group-level slippage budget enforcement for one dependency-coupled action group per tx.
2. Per-leg bound derivation in Rust plus canonical aggregate basket bounds for execution.
3. Small Solidity batch router contract enforcing aggregate slippage bounds (`sell(min)`, `buy(max)`).
4. Rust-side grouping, bound derivation, and L2 gas plus heuristic L1 data-fee estimate.
5. Foundry test suite for contract and guard logic.
6. Replan loop: execute one group, refresh state, recompute next group.

### Out of scope
1. Full on-chain optimizer.
2. Exact tx-bytes L1 data fee accounting (deferred; TODO added).
3. MEV-specific infrastructure (private orderflow, builder integration).
4. Multi-user batch auction/clearing.

## 2. Execution Model

### 2.1 Group unit
A strict subgroup is one dependency-coupled execution unit:
1. Direct buy group: one `Buy`.
2. Direct sell group: one `Sell`.
3. Direct merge group: one `Merge` (inventory merge, no pool buys).
4. Mint-sell group: `Mint -> Sell*`.
5. Buy-merge group: `Buy* -> Merge`.

Profitability-step grouping builds ordered step blocks from the strict stream:
1. Pure direct (`DirectBuy`, `DirectSell`, `DirectMerge`)
2. Pure arb (`MintSell`, `BuyMerge`)
3. Mixed `DirectBuy + MintSell`
4. Mixed `DirectSell + BuyMerge`

Each step stores its ordered strict subgroups so execution can preserve waterfall-equality semantics while still deriving per-subgroup basket bounds.
Strict mode submits one strict subgroup per tx.
Unsupported action streams outside these shapes fail closed at grouping time (planner error, no submission).
Route ordering is strict and validated exactly:
`MintSell` must be `Mint` followed by one-or-more `Sell` legs, and `BuyMerge` must be one-or-more `Buy` legs followed by `Merge`.

### 2.2 Profitability invariants
For group `g`:
1. `edge_plan_susd = planned_proceeds_susd - planned_cost_susd`
2. `gas_l2_susd = l2_gas_units * l2_gas_price_eth * eth_usd_assumed`
3. `estimated_calldata_bytes = f(group_kind, buy_legs, sell_legs)`
4. `l1_fee_per_byte_wei = cached marginal slope from getL1Fee(bytes)` using two non-zero payload sizes (256 and 512 bytes), refreshed on cache expiry
5. `gas_l1_susd = max(l1_data_fee_floor_susd_tmp, estimated_calldata_bytes * l1_fee_per_byte_wei * eth_usd_assumed / 1e18)`
6. `gas_total_susd = gas_l2_susd + gas_l1_susd`
7. `profit_buffer_susd = max(buffer_min_susd, buffer_frac * edge_plan_susd)`
8. `slippage_budget_susd = edge_plan_susd - gas_total_susd - profit_buffer_susd`

Note:
1. For `DirectBuy`, `edge_plan_susd` is expected-value delta (`amount * prediction - planned_cost`), not route-local cashflow.
2. Default planning derives `DirectBuy` edge from the L1 prediction table by normalized market name.
3. Missing prediction for a buy maps to non-positive edge, so that buy group is skipped.
4. When a `DirectBuy` prediction is missing, planner logs the skip reason with raw + normalized market name for operator visibility.
5. `DirectMerge` has no DEX price-risk legs, so planner sets `slippage_budget_susd = 0` and gates on `edge_plan_susd > gas_total_susd + profit_buffer_susd`.

Gate:
1. For DEX-leg groups, if `slippage_budget_susd <= 0`, skip group.
2. For `DirectMerge`, require positive post-buffer margin (`edge_plan_susd - gas_total_susd - profit_buffer_susd > 0`).
3. On-chain aggregate basket bounds derived from `slippage_budget_susd` (`sell(min)` / `buy(max)`) must hold.

This guarantees positive net expected value after gas and worst-case tolerated slippage.

### 2.3 Adverse slippage accounting
1. Buy leg adverse: `max(0, actual_cost_susd - planned_cost_susd)`
2. Sell leg adverse: `max(0, planned_proceeds_susd - actual_proceeds_susd)`
3. Group adverse: sum of all leg adverse in the group.

This accounting is used off-chain to derive aggregate batch bounds; on-chain enforcement is aggregate `buy(max)` / `sell(min)`.

On-chain requirement:
1. Sell baskets: `realized_total_out_susd >= min_total_out_susd`
2. Buy baskets: `realized_total_in_susd <= max_total_in_susd`

## 3. Per-Leg Bounds

### 3.1 Allocation from group budget
For each leg `i`, define adverse notional:
1. Buy leg: `notional_i = planned_cost_i`
2. Sell leg: `notional_i = planned_proceeds_i`

Let `W = sum(notional_i)` and `alloc_i = slippage_budget_susd * (notional_i / W)`.

Per-leg bounds:
1. Buy leg max cost: `max_cost_i = planned_cost_i + alloc_i`
2. Sell leg min proceeds: `min_proceeds_i = max(0, planned_proceeds_i - alloc_i)`
3. `DirectMerge` has no DEX legs; it carries an empty leg list and no per-leg slippage allocation.

### 3.2 Composite routes
1. Mint-sell route:
   - Off-chain: derive sell-leg notionals/allocations from planner state.
   - On-chain: enforce aggregate `sell(min_total_out_susd)` only.
   - `min_total_out_susd = max(0, sum(planned_sell_proceeds) - slippage_budget_susd)`.

2. Buy-merge route:
   - Off-chain: derive buy-leg notionals/allocations from planner state.
   - On-chain: enforce aggregate `buy(max_total_in_susd)` only.
   - `max_total_in_susd = sum(planned_buy_costs) + slippage_budget_susd`.

3. Direct merge route:
   - No DEX swap legs.
   - No `BatchRouter.buy/sell` call; execute merge path directly.

## 4. Solidity Contract (Small, Single-Purpose)

## 4.1 Contract responsibilities
`BatchRouter`:
1. Executes one precomputed basket of swaps atomically.
2. `sell(...)` loops `exactInputSingle` swaps and enforces aggregate `min` total output.
3. `buy(...)` loops `exactOutputSingle` swaps and enforces aggregate `max` total input.
4. Enforces basket-shape invariants (consistent token direction and recipient policy per basket).
5. Leaves route selection and profitability gating off-chain in Rust planner.

## 4.2 Contract interface
```solidity
function sell(ExactInputSingleParams[] calldata swaps, uint256 min)
    external
    returns (uint256 totalOut);

function buy(ExactOutputSingleParams[] calldata swaps, uint256 max)
    external
    returns (uint256 totalIn);
```

Recipient policy:
1. `swaps[i].recipient` must be either `msg.sender` or router `address(this)`.
2. Recipient must be identical across all swaps in the basket.

## 4.3 Rust-to-batch bound mapping
For a computed `ExecutionGroupPlan`:
1. Sell-dominant groups (`DirectSell`, `MintSell`) use:
   - `planned_total_out_susd = sum(sell_leg.planned_quote_susd)`
   - `min_total_out_susd = max(0, planned_total_out_susd - slippage_budget_susd)`
2. Buy-dominant groups (`DirectBuy`, `BuyMerge`) use:
   - `planned_total_in_susd = sum(buy_leg.planned_quote_susd)`
   - `max_total_in_susd = planned_total_in_susd + slippage_budget_susd`
3. `DirectMerge` has no DEX legs and bypasses `BatchRouter.buy/sell`.

Token-unit conversion:
1. `BatchQuoteBounds` convert to token units with explicit quote-token decimals (`to_token_bounds(decimals)`), not a hardcoded 1e18 scale.
2. Directional rounding is fixed to preserve safety:
   - sell minimums floor
   - buy maximums ceil

## 5. Foundry Setup

1. Add Foundry project files at repo root (`foundry.toml`, `contracts/`, `test/`).
2. Keep contract tiny and dependency-light.
3. Add tests:
   - aggregate sell minimum violation reverts
   - aggregate buy maximum violation reverts
   - successful sell basket respects `min`
   - successful buy basket respects `max`

## 6. Rust Integration

### 6.1 New execution planning types
Add execution-specific structs (no changes to optimizer math):
1. `ExecutionLegPlan`
2. `ExecutionGroupPlan`
3. `BatchQuoteBounds`
4. `GasAssumptions`
5. `ExecutionMode` (`Strict`, `Aggressive`)

### 6.2 Group extraction
From existing `Action` stream in `/src/portfolio/core/types.rs`:
1. Direct groups from standalone `Buy`/`Sell`.
2. Standalone `Merge` grouped as `DirectMerge`.
3. Mint-sell and buy-merge groups from contiguous route patterns (`Mint->Sell+`, `Buy+->Merge`).
4. Profitability-step grouping merges strict groups into ordered step blocks with explicit step kind metadata and embedded strict subgroup list.
5. Planner emits one `ExecutionGroupPlan` per strict subgroup and stamps each plan with:
   - `profitability_step_index`
   - `step_subgroup_index`
   - `step_subgroup_count`
6. `DirectBuy` edge defaults to prediction EV delta (`amount * prediction - cost`) via normalized market-name lookup.
7. `derive_batch_quote_bounds(plan, current_block, max_stale_blocks)` maps planned groups into canonical `buy(max)` / `sell(min)` inputs and rejects stale/unstamped plans.
8. `derive_batch_quote_bounds_unchecked(plan)` remains available for non-execution diagnostics.
9. Debug builds assert non-negative finite economics on planner-consumed action fields (`amount`, `cost`, `proceeds`) to catch upstream numeric corruption early.
10. Planner computes cashflow and DEX-leg summaries from a single action pass per group, reducing duplicate iteration and keeping extraction logic aligned.

### 6.3 Gas estimation (v1)
Hardcoded L2 gas units:
1. Direct buy group: `220_000`
2. Direct sell group: `200_000`
3. Direct merge group: `150_000`
4. Mint-sell group: `550_000 + 170_000 * sell_legs`
5. Buy-merge group: `500_000 + 180_000 * buy_legs`

Convert to sUSD:
1. `gas_l2_susd = units * gas_price_eth * eth_usd_assumed`
2. `gas_l1_susd = max(l1_data_fee_floor_susd_tmp, estimated_calldata_bytes * l1_fee_per_byte_wei * eth_usd_assumed / 1e18)`
3. `gas_total_susd = gas_l2_susd + gas_l1_susd`

Inputs:
1. `gas_price_eth` from RPC (`eth_gasPrice` or EIP-1559 field).
2. `eth_usd_assumed` from config/env.
3. temporary `l1_data_fee_floor_susd_tmp` (minimum per group).
4. cached `l1_fee_per_byte_wei` from a two-point `getL1Fee(bytes)` marginal slope sample (256/512 non-zero bytes) with periodic refresh (default TTL 60s).
5. heuristic `estimated_calldata_bytes` from group kind + leg counts.
6. execution path should call `build_group_plans_with_default_edges_and_l1_hydration` for planning passes; it is cache-aware and refreshes on expiry.
7. if hydration fails (RPC unavailable or malformed response), the helper returns a typed planning error (`GroupPlanningError::L1FeeHydration`) and execution fail-closes.
8. cache refresh is deduplicated with a single in-flight fetch lock so concurrent planning does not stampede RPC on cache expiry.
9. planner snapshots one effective `l1_fee_per_byte_wei` value once per planning pass and reuses it for all groups in that pass, so group ordering compares a consistent gas baseline.
10. `eth_call` L1-fee parsing rejects oversized integer payloads (>32 bytes) fail-closed instead of panicking.
11. non-hydrated diagnostic planning paths still fail closed when no usable fee-per-byte value is available (`gas_total_susd = +inf` -> skipped).
12. L1-fee RPC calls use bounded HTTP timeouts so hydration cannot hang indefinitely on stalled endpoints.
13. explicit caller-provided `l1_fee_per_byte_wei` (positive finite) overrides cache so backtests/replays remain deterministic.
14. cached L1-fee-per-byte values are keyed by RPC endpoint; endpoint-agnostic fallback only uses cache when there is exactly one fresh entry, otherwise it fails closed.

### 6.4 Ordering
Plan order is preserved as:
1. `profitability_step_index` (waterfall order)
2. `step_subgroup_index` (strict subgroup order inside each step)
3. Planning is prefix-safe: if any strict subgroup in a step is unplannable, the entire step is dropped and planning stops before later steps (no partial-step execution).

Each subgroup still gates on:
1. `guaranteed_profit_floor_susd = edge_plan_susd - gas_total_susd - slippage_budget_susd`
2. for DEX-leg groups, by construction this equals `profit_buffer_susd`
3. for `DirectMerge`, this is residual post-buffer margin (`edge_plan_susd - gas_total_susd - profit_buffer_susd`)

Execution should replan after each submitted subgroup (or after each full step when atomic multi-subgroup submission is supported).

## 7. Modes

### Strict mode (default for rollout)
1. `groups_per_tx = 1`
2. `buffer_frac = 0.20`
3. `buffer_min_susd = 0.25`
4. `max_stale_blocks = 2`
5. `deadline_secs = 20`
6. `l1_data_fee_floor_susd_tmp = 0.10`

Implementation status:
1. Planner carries `planned_at_block` metadata plus `is_plan_stale(...)`, plus stamping helpers (`stamp_plan_with_block`, `stamp_plans_with_block`).
2. Batch-bound derivation enforces staleness (`derive_batch_quote_bounds(..., current_block, max_stale_blocks)`), fail-closing stale or unstamped plans.
3. Execution-loop wall-clock `deadline_secs` enforcement remains TODO until action->transaction submission wiring is finalized.

### Aggressive mode (opt-in later)
1. `groups_per_tx = 2`
2. tighter per-leg allocations only after measured stability
3. enabled only after strict mode metrics reach target thresholds

## 8. Acceptance Criteria

1. Groups with non-positive slippage budget are not submitted.
2. Any aggregate basket bound breach reverts atomically.
3. Realized adverse slippage is logged per group.
4. Strict mode remains profitable net of modeled gas (`gas_total_susd`) in replay/simulation tests.
5. Existing optimizer behavior remains unchanged when executor path is disabled.

## 9. Risks

1. Heuristic calldata-byte model plus two-point marginal `getL1Fee(bytes)` estimate can misprice specific transactions.
2. Router call encoding errors can misclassify leg quote deltas.
3. Too-tight bounds can reduce fill rate substantially.

Mitigation:
1. conservative buffer settings
2. Foundry fork tests for real-call delta accounting
3. staged rollout with telemetry-based tuning

## 10. Telemetry

Log per group:
1. `group_kind`
2. `edge_plan_susd`
3. `gas_l2_susd`
4. `gas_total_susd`
5. `slippage_budget_susd`
6. `realized_adverse_susd`
7. `realized_net_pnl_susd`
8. `result` (`executed`, `skipped`, `reverted`)
9. `revert_reason` when available
10. `skip_reason` when `result=skipped` (e.g., `non_positive_edge`, `missing_prediction`, `stale_plan`)

## 11. References

1. [Optimism Transaction Fee Estimates](https://docs.optimism.io/app-developers/guides/transactions/estimates)
2. [Optimism Fee Components and Cost Estimation](https://docs.optimism.io/concepts/transactions/fees#estimating-the-transaction-cost)
