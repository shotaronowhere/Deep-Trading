# ForecastFlows Rust Worker Integration Plan

Status: proposed implementation plan

## Goal

Replace the current Julia-backed ForecastFlows worker with the Rust port in `../ForecastFlows.rs` using the cleanest and easiest implementation path first.

This plan intentionally keeps the existing `deep_trading` ownership split:

- `ForecastFlows` remains an external gross-route generator.
- `deep_trading` keeps ownership of:
  - live snapshot translation
  - replay acceptance
  - execution-group validation
  - fee modeling
  - local pruning
  - final net-EV ranking
  - fail-open fallback to the native planner

## Chosen Approach

The first implementation should swap the external worker backend only.

- Keep the current stdio NDJSON worker boundary.
- Add explicit support for a Rust worker binary backend.
- Leave the Julia worker backend in place as a fallback during rollout.
- Do not embed `ForecastFlows.rs` as a library in the first pass.
- Do not move `deep_trading` execution logic into `ForecastFlows.rs`.

This is the cleanest first implementation because it changes only one major variable:

- solver backend: Julia worker -> Rust worker

It does not also change the process boundary, replay contract, or local net-EV ownership at the same time.

## Why This Approach First

This project is performance-sensitive, but the first optimization target should be the largest uncertainty, not the most invasive integration style.

Reasons to start with the Rust worker backend instead of an in-process library integration:

1. It preserves the current operational safety model.
2. It keeps timeout, restart, cooldown, and fail-open behavior intact.
3. It isolates backend swap risk from integration-boundary risk.
4. It keeps benchmarking honest by changing one major variable at a time.
5. It reuses the current translation, replay, pruning, and ranking code paths unchanged.

The current client stack already records the timing split that we need to make the next decision:

- worker roundtrip
- derived driver overhead
- local translation/replay time
- local candidate build time
- local prune time

That timing split should tell us whether a later in-process integration is worth the extra complexity and operational risk.

## Assumptions

- `../ForecastFlows.rs` remains a sibling repository and is built independently.
- The Rust port is sufficiently protocol-compatible to act as a worker backend.
- The first production-grade Rust integration will use a prebuilt `forecast-flows-worker` binary, not `cargo run`.
- `deep_trading` still does not rely on worker-provided `estimated_execution_cost` or `net_ev` for acceptance or ranking.
- The existing `ForecastFlows` replay and plan acceptance contract remains correct and should not be changed in this migration.

## Non-Goals

- No solver-math rewrite.
- No protocol redesign.
- No direct library integration in the first pass.
- No attempt to collapse `deep_trading` and `ForecastFlows.rs` into one crate graph.
- No movement of `deep_trading` fee or execution compiler logic into `ForecastFlows.rs`.
- No removal of the Julia backend during the initial implementation commit.

## Current Load-Bearing Boundaries

The migration should preserve the behavior of these boundaries:

- `src/portfolio/core/forecastflows/translate.rs`
  - snapshot -> problem mapping
  - worker response -> local `Action` translation
  - local replay acceptance
- `src/portfolio/core/forecastflows/mod.rs`
  - ForecastFlows telemetry assembly
  - solve-family candidate intake
- `src/portfolio/core/rebalancer.rs`
  - ForecastFlows vs native ranking
  - final plan selection
- `src/portfolio/core/forecastflows/client.rs`
  - worker spawn
  - health
  - warmup
  - request/response handling
  - cooldown/backoff
  - doctor reporting

The only intentional behavior change in the first implementation is:

- when configured, the worker process is the Rust binary instead of the Julia runtime

## Implementation Phases

### Phase 1: Upstream Protocol Compatibility Pass in `ForecastFlows.rs`

Objective:
Make sure the Rust worker emits the fields that `deep_trading` already expects from the worker path.

Scope in `../ForecastFlows.rs`:

- Update `crates/forecast-flows-pm/src/protocol.rs`.
- Update `crates/forecast-flows-pm/src/result.rs`.
- Update `crates/forecast-flows-pm/src/worker.rs` if needed.

Required changes:

1. Add `workspace_reused` to the Rust worker's compare result wire format.
2. Populate `workspace_reused` truthfully from the cached compare workspace path.
3. Keep `estimated_execution_cost` and `net_ev` as `None` for now.
4. Keep protocol version and command semantics unchanged.

Verification:

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --release -- --test-threads=1`
- specifically confirm:
  - `compare_matches_fixture`
  - `worker_binary_replays_health_compare_and_invalid_request`

Acceptance criteria:

- The Rust worker remains protocol-v2 compatible.
- The compare path remains green under the existing parity harness.
- The worker now reports workspace reuse on the wire.

### Phase 2: Add an Explicit Backend Abstraction in `deep_trading`

Objective:
Separate the shared client orchestration logic from backend-specific process launch details.

Primary file:

- `src/portfolio/core/forecastflows/client.rs`

Introduce:

- `ForecastFlowsBackend`
  - `JuliaWorker`
  - `RustWorker`

Suggested env var:

- `FORECASTFLOWS_BACKEND`

Suggested accepted values:

- `julia_worker`
- `rust_worker`

Default for the first implementation commit:

- `julia_worker`

Required refactor shape:

1. Keep the worker pool, request IDs, health checks, retries, cooldowns, and request send/receive logic shared.
2. Move only the backend-specific launch config into backend branches.
3. Keep `compare_prediction_market_families`, `warm_worker`, `shutdown_worker`, and `doctor_report` public behavior unchanged where possible.

Acceptance criteria:

- No change to solver-family selection behavior.
- No change to fallback policy.
- No change to translation or replay behavior.
- Backend choice only changes how the external worker process is launched.

### Phase 3: Implement the Rust Worker Launch Path

Objective:
Allow `deep_trading` to spawn the Rust worker directly as an external process.

Primary file:

- `src/portfolio/core/forecastflows/client.rs`

Suggested env var:

- `FORECASTFLOWS_WORKER_BIN`

Expected value:

- absolute path to the prebuilt `forecast-flows-worker` executable

Rust worker launch behavior:

- spawn the worker binary directly
- no Julia runtime
- no `--project`
- no sysimage handling
- no Manifest hashing
- no `JULIA_NUM_THREADS`

Julia worker launch behavior:

- preserve the current implementation exactly

Important implementation constraint:

- do not use a wrapper script around `FORECASTFLOWS_JULIA_BIN` to fake a Rust backend
- do not use `cargo run` in the runtime path

Acceptance criteria:

- `warm_forecastflows_worker()` works with the Rust backend
- health checks succeed
- compare requests succeed
- cooldown/backoff logic still works for the Rust backend
- the Julia backend still works unchanged

### Phase 4: Make Doctor and Runtime Metadata Backend-Neutral

Objective:
Stop hard-coding Julia-specific assumptions into diagnostics and readiness checks.

Primary files:

- `src/portfolio/core/forecastflows/mod.rs`
- `src/portfolio/core/forecastflows/client.rs`
- `src/bin/forecastflows_doctor.rs`
- `src/portfolio/core/forecastflows/protocol.rs`

Problems to fix:

1. doctor/report structs are currently Julia-shaped
2. production-readiness is currently defined as "active sysimage"
3. telemetry currently assumes Julia-specific runtime metadata

Recommended report shape:

- `worker_backend`
- `worker_program`
- `worker_version`
- `worker_project_dir`
- `worker_launch_args`
- `worker_runtime_detail`
- `health_status`
- `supported_commands`
- `supported_modes`
- `execution_model`
- `warmup_compare_duration_ms`
- `representative_compare_duration_ms`
- optional backend-specific optimization fields

Recommended rule:

- Julia-specific fields remain optional and are only populated under the Julia backend
- Rust worker mode should not emit fake sysimage or Julia thread values

Protocol parsing update:

- allow deserializing optional health fields already emitted by the Rust worker such as:
  - `package`
  - `package_version`

Acceptance criteria:

- `forecastflows_doctor` output is truthful under both backends
- the Rust backend is not incorrectly classified as non-production-ready only because it has no sysimage
- JSON and human-readable doctor output remain stable and understandable

### Phase 5: Keep Telemetry Compatible but Truthful

Objective:
Preserve the existing benchmark and production timing model while adding backend-awareness.

Primary files:

- `src/portfolio/core/forecastflows/mod.rs`
- `src/portfolio/core/rebalancer.rs`
- `src/portfolio/tests/rebalancer_contract_ab.rs`

Recommended telemetry additions:

- `forecastflows_backend`
- `forecastflows_worker_version`

Existing timings to preserve:

- `worker_roundtrip_ms`
- `driver_overhead_ms`
- `translation_replay_ms`
- `local_candidate_build_ms`
- `local_step_prune_ms`
- `local_route_prune_ms`
- `direct_solver_time_ms`
- `mixed_solver_time_ms`

Rules:

- keep Julia-only fields optional
- under Rust worker mode, unset values should be `None`
- do not invent synthetic Julia metadata

Acceptance criteria:

- current benchmark row generation still works
- existing timing fields remain comparable across backends
- backend-specific metadata is visible in benchmark output

### Phase 6: Update Tests Before Default Flip

Objective:
Make the new backend safe to ship before changing defaults.

Test work in `deep_trading`:

1. Add backend parsing tests.
2. Add launch config tests for both backends.
3. Add doctor tests for both backends.
4. Add worker client tests for Rust backend happy path and failure path.
5. Relax assertions that currently hard-require Julia-only metadata.
6. Keep assertions that matter for both backends:
   - direct status present
   - mixed status present
   - at least one solver timing present
   - backend label present
   - fail-open behavior preserved

Acceptance criteria:

- both backends pass targeted tests
- the benchmark-facing tests remain meaningful for both backends

### Phase 7: Update Permanent Docs

Objective:
Document the integration and rollout as a first-class supported path.

Files to update:

- `docs/forecastflows_worker.md`
- `docs/architecture.md`
- `docs/README.md`

This plan doc should remain the migration implementation guide.

Required doc content:

- supported backends
- new env vars
- Rust worker build instructions
- rollout procedure
- rollback procedure
- backend-specific doctor/readiness expectations

Acceptance criteria:

- a new engineer can discover the doc from `docs/README.md`
- the Rust worker launch path is documented without reading code

### Phase 8: Rollout and Default Flip

Objective:
Ship the Rust worker backend with a low-risk rollout sequence.

Status: completed on 2026-04-16. The default backend is now `rust_worker`; set `FORECASTFLOWS_BACKEND=julia_worker` only for explicit legacy rollback.

Completed rollout order:

1. Land support for both backends while Julia was still the default.
2. Run parity and benchmark checks on the committed shared-snapshot cases.
3. Run `forecastflows_doctor` under both backends.
4. Flip the default to `rust_worker`. (Done.)
5. Keep `FORECASTFLOWS_BACKEND=julia_worker` as an immediate rollback switch.

Acceptance criteria:

- Rust worker path is stable on real benchmark cases
- no increase in translation or replay rejection rate
- no increase in worker-failure fallback rate
- validated net-EV remains aligned with Julia within agreed tolerance

## Verification Matrix

### `ForecastFlows.rs`

Run:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release -- --test-threads=1
```

### `deep_trading`

Run targeted tests and manual checks for both backends.

Core checks:

```bash
cargo test forecastflows -- --nocapture
cargo run --bin forecastflows_doctor -- --json
```

Also run the shared-snapshot ForecastFlows benchmark helpers under:

- `FORECASTFLOWS_BACKEND=julia_worker`
- `FORECASTFLOWS_BACKEND=rust_worker`

The same benchmark case set should be used for both backends.

## Completion Criteria

This migration is complete when all of the following are true:

1. `deep_trading` can run ForecastFlows using either Julia or Rust worker backends.
2. The Rust worker path preserves the existing replay/ranking contract.
3. Doctor and telemetry output are backend-aware and truthful.
4. The Rust worker passes the same protocol/benchmark verification gates as Julia.
5. The default backend is Rust with a one-env-var rollback to Julia.

## Risks and Mitigations

### Risk: Rust worker protocol is close, but not identical

Mitigation:

- close the `workspace_reused` gap before touching `deep_trading`
- keep protocol version and command handling unchanged

### Risk: Diagnostics become misleading after backend abstraction

Mitigation:

- make doctor output backend-neutral
- keep backend-specific fields optional

### Risk: Rollout regressions are hard to diagnose

Mitigation:

- keep Julia backend live during rollout
- add explicit backend labels to telemetry and benchmark rows

### Risk: Rust worker is fast, but local replay/prune still dominates

Mitigation:

- do not prematurely switch to library integration
- use the timing split already exposed by `deep_trading` to decide the next step

## Future TODO: Latency Profiling

This is future work and not part of the first implementation.

Objective:

- determine whether the worker boundary remains a material latency cost after the Rust worker swap

Recommended profiling work:

1. Add a backend label to every ForecastFlows benchmark JSON row.
2. Measure separately:
   - cold start
   - first warm request
   - second compatible request
   - incompatible-layout request
3. Record:
   - worker roundtrip
   - driver overhead
   - translation replay
   - local candidate build
   - local step prune
   - local route prune
   - direct solver time
   - mixed solver time
4. Run repeated measurements on the large cases first:
   - `heterogeneous_ninety_eight_outcome_l1_like_case`
   - `ninety_eight_outcome_multitick_direct_only`
   - ForecastFlows pathological stress cases
5. Capture repeated samples and compare at least p50 and p95, not just single runs.

Decision rule for deeper integration:

- only pursue a more invasive design if the worker/process/protocol overhead is still a meaningful part of total warm-solve latency after the Julia -> Rust worker swap

## Future TODO: Alternate High-Performance Implementation

This is future work and not part of the first implementation.

Alternate approach:

- add a `RustLib` backend that calls `forecast-flows-pm` directly in-process

Why this is deferred:

- it removes process isolation
- it weakens our ability to kill or timeout a stuck solve
- it increases blast radius if the solver or C-port misbehaves
- it changes two large variables at once if done before measuring the Rust worker path

If pursued later, the implementation should:

1. Keep the current worker path alive as fallback during development.
2. Add a new adapter module in `deep_trading` that constructs `forecast_flows_pm::PredictionMarketProblem` directly.
3. Use `PredictionMarketWorkspace` in-process to preserve warm-start reuse.
4. Return the same local compare/result shape the worker path uses today.
5. Reuse the existing downstream translation, replay, prune, and ranking pipeline unchanged.
6. Prove a material latency win over the Rust worker path before becoming the default.

Explicit non-goal for that future work:

- even with an in-process solver backend, `deep_trading` should still keep ownership of replay, fee modeling, execution grouping, and final net-EV ranking
