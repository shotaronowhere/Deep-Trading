# Slippage Policy (Current)

This document defines the current slippage/staleness posture used by the repo.

## Current policy

1. On-chain direct solving (`Rebalancer.sol`) is the default in competitive conditions because solve and execute occur atomically on live state.
2. Off-chain strict execution is conditional and must pass conservative margin, stale-plan, and bounded adverse-move gates.
3. Per-leg `sqrtPriceLimitX96` bounds are mandatory for DEX legs; stale or unbound plans fail closed.
4. Mixed routing is opt-in behind EV-net and reliability thresholds from the playbook.

## Canonical references

- Strategy and thresholds: `docs/rebalancer_approaches_playbook.md`
- On-chain algorithm details: `docs/rebalancer.md`
- Off-chain waterfall behavior: `docs/waterfall.md`

## Historical context

The original long-form hybrid slippage sprint spec is preserved at:

- `docs/archive/slippage_guard_sprint_spec_legacy.md`

It is retained for historical rationale only.
