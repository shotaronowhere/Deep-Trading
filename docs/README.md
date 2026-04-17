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
- `docs/rebalancer_algebra.md`: on-chain AlgebraV1.9 (Swapr/Gnosis) solver spec.
- `docs/rebalancer_mixed.md`: on-chain mixed solver spec (experimental).
- `docs/rebalancer_approaches_playbook.md`: cross-approach policy, thresholds, and validation cadence.
- `docs/rebalancer_policy_metrics_schema.md`: machine-readable threshold dashboard schema.
- `docs/rebalancer_policy_metrics_template.csv`: metrics row template (CSV).
- `docs/rebalancer_policy_metrics_template.json`: metrics row template (JSON).
- `docs/model.md`: mathematical derivations used by the off-chain solver.
- `docs/architecture.md`: system architecture and source map.
- `docs/forecastflows_worker.md`: ForecastFlows worker boundary, supported backends (`julia_worker`, `rust_worker`), Julia sysimage workflow, doctor tool, replay acceptance, and fallback policy.

## Operational runbooks

- `docs/execution_program_packing.md`: packed execution-program compilation and chunked submission model.
- `docs/local_foundry_e2e_harness.md`: local Foundry executable-transaction harness for Seer + Uniswap V3 + solver output validation.
- `docs/local_foundry_e2e_harness_next_steps.md`: planned follow-up work for connected child mint/merge, connected on-chain solver support, tolerance tightening, release-mode FFI, and unrelated Rust regressions.
- `docs/execution_submission.md`: strict execution submission gates and operator steps.
- `docs/deployments.md`: contract addresses/constants.
- `docs/batch_swap_router.md`: BatchSwapRouter API + test coverage map.
- `docs/slippage.md`: current slippage/staleness policy summary.
- `docs/TODO.md`: prioritized open items and completion log.
- `docs/forecastflows_rust_worker_integration_plan.md`: implementation plan for migrating the ForecastFlows backend from Julia worker to Rust worker.

## Module and validation docs

- `docs/portfolio.md`: portfolio module surface, diagnostics, and test map.
- `docs/solver_benchmark_matrix.md`: central release-facing EV / gas / speed comparison across solver flavors.
- `docs/gas_model.md`: gas threshold assumptions.
- `docs/forecastflows_worker.md`: external solver worker lifecycle (Julia and Rust backends), doctor workflow, and translation contract.
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
