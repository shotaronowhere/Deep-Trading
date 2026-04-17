# Local Foundry Executable E2E Harness

Status: current as of 2026-04-17.

## Purpose

`test/LocalFoundryExecutableTxE2E.t.sol` is the executable local proof that the solver stack can emit transactions that run against real local contracts.

It deploys:

- local Seer market contracts from the checked-in test artifacts
- local Uniswap V3 core pools plus `SwapRouter02`
- local `TradeExecutor`
- local `Rebalancer` / `RebalancerMixed`
- dummy 18-decimal `DummyUSDC` collateral

The Rust fixture binary, `src/bin/local_foundry_e2e_fixture.rs`, receives the Foundry-deployed address book through `forge test --ffi`, runs the production solver/compiler path, and returns ABI-encoded `TradeExecutor.Call[]` chunks. Foundry decodes those chunks and executes them through `TradeExecutor.batchExecute`.

## Contract Sources

The harness follows these external contract boundaries:

- Seer demo contracts: `ConditionalTokens`, `RealityETH_v3_0`, `Wrapped1155Factory`, `RealityProxy`, `Market`, `MarketFactory`, and `Router`.
- Uniswap V3 core: `UniswapV3Factory` and per-pool core contracts.
- Uniswap Universal Router package: deploy `SwapRouter02`, not `V2SwapRouter`.

`V2SwapRouter` is an abstract base in `swap-router-contracts`; this harness deploys `SwapRouter02(address(0), factoryV3, address(0), WETH9)` and only uses V3 exact-input/exact-output calls.

## Deployment Order

The Foundry `setUp()` order is:

1. Deploy `DummyUSDC` with `18` decimals.
2. Deploy Seer/Gnosis artifacts: `ConditionalTokens`, `RealityETH_v3_0`, `Wrapped1155Factory`, `RealityProxy`, `Market`, `MarketFactory`, and `Router`.
3. Deploy `WETH9`, `UniswapV3Factory`, enable fee tier `100` with tick spacing `1`, and deploy `SwapRouter02`.
4. Deploy the local V3 mint helper.
5. Each test deploys its own `TradeExecutor` portfolio account.

The 18-decimal collateral choice is deliberate: the current Rust solver and Solidity rebalancers use WAD-scale collateral math.

## Market Topologies

### Small Connected Direct Case

`test_small_connected_direct_buy_sell_executable_rust_plan` deploys a connected root/child topology:

- root market: 3 named outcomes, with the final root outcome used as child collateral connector
- child market: 2 named outcomes
- tradeable outcomes: 2 root normal outcomes + 2 child normal outcomes
- excluded outcomes: root invalid, child invalid, and connector

This case exercises a real connected address book with production Rust solver output compiled into executable local `TradeExecutor` calls.

### Small Route Probes

`test_small_connected_mint_sell_executable_rust_plan` and `test_small_connected_buy_merge_with_seeded_invalid_inventory` are deterministic route probes:

- mint-sell: root-only 4-tradeable synthetic market, scripted `Mint -> Sell+` actions
- buy-merge: root-only 4-tradeable synthetic market, scripted `Buy+ -> Merge` actions with invalid inventory seeded

These probes still use the normal Rust execution grouping, packed-program compiler, local address book, transaction builder, `SwapRouter02`, Seer router, and `TradeExecutor.batchExecute`. The scripted action selection is intentional because the checked-in Seer child-market artifact currently blocks nested local child split/merge execution. The connected direct case and the 98-outcome case remain production-solver cases.

### Connected 98-Outcome L1-Like Case

`test_ninety_eight_outcome_connected_l1_like_executable_rust_plan` models the L1 market shape:

- root market: 67 named outcomes, with one connector outcome
- child market: 32 named outcomes
- tradeable pools: 66 root normal outcomes + 32 child normal outcomes = 98
- excluded outcomes: root invalid, child invalid, and connector

This case deploys 98 real local Uniswap V3 pools and validates a packed Rust/off-chain execution plan through the real `TradeExecutor`.

### Synthetic 98-Outcome On-Chain Solver Case

`test_synthetic_ninety_eight_onchain_solver_calls_execute_through_trade_executor` deploys one synthetic root market with 98 tradeable outcomes and executes:

- `Rebalancer.rebalance`
- `Rebalancer.rebalanceExact`
- `RebalancerMixed.rebalanceMixedConstantL`

Each solver call is ABI-encoded into a one-call `TradeExecutor.batchExecute` transaction. This is single-market because the current on-chain solver interface accepts one Seer market per call and cannot faithfully represent the connected two-market topology without a contract interface change.

## Pool And Liquidity Model

For each tradeable outcome:

1. Create one `DummyUSDC / outcome` Uniswap V3 pool at fee tier `100`.
2. Sort `token0` and `token1` by address.
3. Store whether the outcome is token1.
4. Initialize `sqrtPriceX96` from the scenario `priceWad` using the same orientation convention as the existing rebalancer tests.
5. Mint exactly one full-range liquidity position:
   - `tickLower = min usable tick`
   - `tickUpper = max usable tick`
6. Assert:
   - pool liquidity is nonzero
   - `slot0.sqrtPriceX96` matches the intended start price
   - the Rust fixture receives the real pool address, tick, tick bounds, and liquidity

The Rust fixture represents the one full-range position as two initialized ticks, which matches the local pool setup.

## Local Address Book

`src/execution/tx_builder.rs` now exposes `ExecutionAddressBook`.

Fields:

- `collateral`
- `seer_router`
- `swap_router`
- `market1`
- `market2`
- `market2_collateral`
- `outcome_tokens`

The default address book preserves the existing live Optimism constants. Local tests call `build_trade_executor_calls_with_address_book` through the execution-program helpers so generated calldata points at Foundry-deployed contracts instead of live addresses.

For synthetic single-market cases, `market2 == address(0)` disables second-market split/merge legs.

## Fixture Flow

Foundry writes a scenario-specific JSON input under `test/fixtures/local_foundry_e2e_fixture_input_<scenario>.json`, then calls:

```bash
cargo run --quiet --bin local_foundry_e2e_fixture <input-json-path>
```

The fixture binary:

1. Parses the Foundry address book and market descriptors.
2. Builds local `MarketData`, `Pool`, `Tick`, and `Slot0Result` values.
3. Runs the selected solver path with local predictions and balances.
4. Builds strict group plans with local pool context.
5. Compiles a packed execution program with the local address book.
6. Emits JSON containing human-readable actions/chunks plus an ABI-encoded `LocalFixtureResult`.

Foundry decodes the ABI payload into:

- `actionCount`
- packed chunks
- `preRawEvWad`
- `expectedRawEvWad`
- `estimatedTotalFeeWad`
- `estimatedNetEvWad`

## EV And Gas Accounting

Foundry treats the `TradeExecutor` as the portfolio account.

Before execution:

```text
preRawEV = collateralBalance + sum(prediction_i * outcomeBalance_i)
```

For each packed chunk:

1. Decode `TradeExecutor.Call[]`.
2. Execute `executor.batchExecute(calls)`.
3. Measure L2 gas with `gasleft()` around the call.
4. Compute calldata bytes from the real ABI encoding:

```solidity
abi.encodeCall(TradeExecutor.batchExecute, (calls)).length
```

After execution:

```text
postRawEV = collateralBalance + sum(prediction_i * outcomeBalance_i)
modeledFee = l2GasUsed * gasPriceWei * ETH_USD
           + calldataBytes * l1FeePerByteWei * ETH_USD
realizedNetEV = postRawEV - preRawEV - modeledFee
```

All values are WAD-scaled collateral units in assertions and logs.

## Acceptance Checks

The harness asserts:

- every emitted transaction executes successfully
- each chunk stays below the `40_000_000` L2 gas cap
- realized raw EV matches the Rust expected raw EV within the scenario tolerance
- realized net EV matches the Rust estimated net EV within tolerance plus a small fee-model allowance
- helper/router/rebalancer contracts do not keep unexpected collateral or outcome balances
- the full Foundry file passes under concurrent Forge scheduling by using scenario-specific FFI input paths

## Commands

Baseline checks:

```bash
forge test --match-path test/RebalancerRealE2E.t.sol -vv
forge test --match-path test/RebalancerABNetEV.t.sol -vv
```

New executable harness:

```bash
forge test --ffi --match-path test/LocalFoundryExecutableTxE2E.t.sol -vv
```

Rust checks:

```bash
cargo check --bin local_foundry_e2e_fixture
cargo test address_book_overrides_live_addresses_for_local_fixture_calls
cargo test --bin local_foundry_e2e_fixture
```

## Current Limitations

- `DummyUSDC` is intentionally 18 decimals; 6-decimal collateral support remains out of scope.
- Connected child direct swaps are executable locally, but local nested child split/merge through the checked-in Seer artifact is not currently used for route probes.
- On-chain solver benchmarking is single-market synthetic because the deployed on-chain solver interface accepts one Seer market.
- The fixture binary is a local harness tool. Live defaults remain unchanged in the normal transaction builder path.
