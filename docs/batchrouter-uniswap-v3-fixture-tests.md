# BatchRouter Uniswap V3 Fixture Tests

## Goal
Add deterministic, local integration-style test coverage for `BatchRouter.sol` using deployed Uniswap V3 core/periphery contracts, while keeping this repository self-contained.

## Dependency strategy
This repo uses local Foundry `lib/` dependencies (not a git submodule to another test repository):

- `uniswap/v3-core@v1.0.0-rc.2`
- `uniswap/v3-periphery@v1.3.0`
- `OpenZeppelin/openzeppelin-contracts@v4.9.6`
- `foundry-rs/forge-std@v1.9.6`

Remappings are configured in `/Users/shotaro/proj/deep_trading/foundry.toml`.

## Why an adapter is required
`BatchRouter` uses an `IV3SwapRouter` interface (from `swap-router-contracts`) whose structs do not include a `deadline` field.

Uniswap v3-periphery `ISwapRouter` includes `deadline` in `exactInputSingle` and `exactOutputSingle` params.

To test `BatchRouter` against real V3 pools/router without changing production code, the test harness includes:

- `V3PeripheryRouterAdapter` in `/Users/shotaro/proj/deep_trading/test/utils/UniswapV3Fixture.sol`

This adapter:

- receives calls in `BatchRouter` ABI format
- maps params to periphery `ISwapRouter` format with `deadline = block.timestamp`
- executes real periphery swaps against local deployed pools

## Fixture architecture
Implemented in `/Users/shotaro/proj/deep_trading/test/utils/UniswapV3Fixture.sol`.

- Deploys `UniswapV3Factory` and `SwapRouter`
- Deploys three mintable ERC20 test tokens (`tokenIn`, `tokenOut`, `tokenOutAlt`)
- Creates and initializes pools:
  - `tokenIn/tokenOut` at 500 and 3000 fee tiers
  - `tokenIn/tokenOutAlt` at 500 fee tier
- Seeds liquidity by calling pool `mint` and paying owed amounts via `uniswapV3MintCallback`
- Deploys `BatchRouter` pointing to the adapter

## Covered scenarios
Implemented in `/Users/shotaro/proj/deep_trading/test/BatchRouter.t.sol`:

- direct adapter smoke swap
- `sell` single-swap success path
- `sell` multi-swap aggregate amount-out accounting
- `sell` aggregate minimum slippage revert (`SlippageExceeded`)
- `buy` single-swap success with refund of unused `tokenIn`
- `buy` multi-swap aggregate amount-in accounting and refund behavior
- `buy` empty swap array revert behavior
- `sell` behavior when tokenOut differs across swaps (current contract design allows this)

Additional branch-completion scenarios are implemented in
`/Users/shotaro/proj/deep_trading/test/BatchRouterBranches.t.sol`:

- `sell` approval failure (`ApprovalFailed`)
- `buy` approval failure (`ApprovalFailed`)
- `buy` aggregate spend exceeds `max` (`SlippageExceeded`)
- `buy` refund transfer failure (`TransferFailed`)
- `buy` no-refund path when remaining input balance is zero

## Not covered
- Mainnet-fork behavior
- `buy` aggregate `SlippageExceeded` path under a fully standard router/token stack (allowance mechanics generally cap spend to `max` before final check)
- Any production code changes beyond compiler pragma normalization for local toolchain compatibility

## Run commands

```bash
forge test --match-contract BatchRouterTest -vv
forge test --match-contract BatchRouterBranchTest -vv
```

Or run all solidity tests:

```bash
forge test -vv
```

Coverage for BatchRouter only:

```bash
forge coverage --match-contract 'BatchRouter(Test|BranchTest)' --report summary
```

Expected BatchRouter coverage target from the current suite:

- `/Users/shotaro/proj/deep_trading/contracts/BatchRouter.sol`: 100% lines, statements, branches, and functions
