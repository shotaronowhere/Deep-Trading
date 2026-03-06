# Documentation Map (Single Source of Truth)

This file is the canonical index for project documentation.

If documents conflict, follow this precedence:

1. `docs/waterfall.md` for off-chain Rust rebalance algorithm behavior.
2. `docs/rebalancer.md` for on-chain `Rebalancer.sol` behavior.
3. `docs/rebalancer_approaches_playbook.md` for execution-policy selection and threshold protocol.
4. `docs/architecture.md` for repository/module topology.

## Canonical specs

- `docs/waterfall.md`: definitive off-chain waterfall/rebalance spec.
- `docs/rebalancer.md`: definitive on-chain direct solver spec.
- `docs/rebalancer_mixed.md`: on-chain mixed solver spec (experimental).
- `docs/rebalancer_approaches_playbook.md`: cross-approach policy, thresholds, and validation cadence.
- `docs/rebalancer_policy_metrics_schema.md`: machine-readable threshold dashboard schema.
- `docs/rebalancer_policy_metrics_template.csv`: metrics row template (CSV).
- `docs/rebalancer_policy_metrics_template.json`: metrics row template (JSON).
- `docs/model.md`: mathematical derivations used by the off-chain solver.
- `docs/architecture.md`: system architecture and source map.

## Operational runbooks

- `docs/execution_submission.md`: strict execution submission gates and operator steps.
- `docs/deployments.md`: contract addresses/constants.
- `docs/batch_swap_router.md`: BatchSwapRouter API + test coverage map.
- `docs/slippage.md`: current slippage/staleness policy summary.
- `docs/TODO.md`: prioritized open items and completion log.

## Module and validation docs

- `docs/portfolio.md`: portfolio module surface, diagnostics, and test map.
- `docs/gas_model.md`: gas threshold assumptions.
- `docs/rebalance_test_ev_trace.md`: EV trace instrumentation.
- `docs/monte_carlo_rebalance_validation.md`: Monte Carlo EV validation harness.

## Mechanism and design context

- `docs/rebalancing_mechanism_design_review.md`: mechanism-design critique and improvement hypotheses.

## Research and future design

- `docs/batch.md`: multi-agent batch-clearing design exploration (research, not implemented in runtime path).

## Research and historical notes

Historical/dated notes are archived under `docs/archive/`.

- Rebalancer experiments: `docs/archive/rebalancer/`
- Implementation migration notes: `docs/archive/implementation/`
- Research and background notes: `docs/archive/research/` and `docs/archive/background/`
- Legacy slippage sprint spec: `docs/archive/slippage_guard_sprint_spec_legacy.md`

## Contribution rule

- New behavioral or policy changes must update exactly one canonical spec in this index.
- Dated experiment write-ups should go to `docs/archive/` and link back to the canonical spec they informed.
