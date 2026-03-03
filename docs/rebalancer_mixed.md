# Experimental On-Chain Mixed Rebalancer

`contracts/RebalancerMixed.sol` is an experimental sibling to `contracts/Rebalancer.sol`. It does not modify the original contract.

## Goal

Provide a conservative on-chain approximation of a mixed direct-vs-mint buy path without porting the full off-chain mixed-route solver.

## Entry Point

```solidity
function rebalanceMixed(
    RebalanceParams calldata params,
    address market,
    uint256 maxMixedSteps,
    uint256 maxStepCollateral
) external returns (uint256 totalProceeds, uint256 totalSpent);
```

- `maxMixedSteps` bounds the greedy loop.
- `maxStepCollateral` caps how much collateral a single step may commit up front.

## Step Algorithm

Each loop iteration:

1. Reads current pool prices.
2. Finds the highest direct-profitability outcome set, including exact ratio ties.
3. Uses the next lower direct-profitability outcome as the frontier. If none exists, the frontier is prediction (`profitability = 0`).
4. Chooses a route heuristically:
   - compute direct bundle cost as the sum of the current top-set token prices;
   - compute alternative mint cost as `1 - sum(sellable non-top token prices)`, where only legs that can still be sold to the current frontier count;
   - execute the cheaper route for that step.
5. Re-reads pool state and repeats until no effective step is found, budget is exhausted, or `maxMixedSteps` is reached.

## Direct Step

- Buys only the current top-profitability set.
- Uses the shared frontier as the per-token `sqrtPriceLimitX96`.
- Spends at most `maxStepCollateral` across the step.

This is a bounded greedy approximation of a direct waterfall descent across the current top set.

## Mint Step

- Splits at most `maxStepCollateral` collateral into a full complete set using `market`.
- Keeps the freshly minted tokens that are currently in the top-profitability set.
- Sells only the newly minted non-top tokens, each only down to the shared frontier.
- Merges only freshly residual complete sets after those sells.

This is the experimental tie-aware approximation of “mint and sell the rest”:

- when several tokens share the current top profitability, the contract keeps all of them in the same mint step;
- the sell limits on the other tokens stop them at the same frontier so they are not pushed arbitrarily far past the current active boundary;
- non-top tokens already sitting at that boundary do not contribute to the alternative buy credit for that step.

## Important Limits

- This is intentionally heuristic. It is not the full mixed-route waterfall from the Rust planner.
- Route choice is greedy, not globally optimized.
- It only supports the same single `market` split/merge interface used by `ICTFRouter`.
- For connected multi-market trees, it still relies on the caller-side `market` abstraction and does not introduce a new cross-market mint interface.
- Tie grouping uses exact on-chain profitability-ratio equality, so near-ties are not merged unless the ratios match exactly in integer math.

## Verification Notes

- `test/RebalancerMixed.t.sol` covers:
  - a direct-preferred low-total-price branch,
  - a direct-preferred high-total-price branch where the frontier blocks the mint credit,
  - a mint-preferred branch where the alternative buy route is genuinely cheaper.
- The tests intentionally keep the router and CTF mocks simple; they verify branch shape and accounting, not execution optimality.
