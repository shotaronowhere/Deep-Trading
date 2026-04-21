# ForecastFlows Worker Integration

Status: current as of 2026-04-16.
Source-of-truth code paths:

- `src/portfolio/core/forecastflows/client.rs`
- `src/portfolio/core/forecastflows/translate.rs`
- `src/portfolio/core/rebalancer.rs`

## Worker Backends

Two interchangeable worker backends speak the same NDJSON stdio protocol (version `2`). The driver selects one at startup via `FORECASTFLOWS_BACKEND`:

- `rust_worker` (default): a standalone `forecast-flows-worker` binary built from the Rust port at `../ForecastFlows.rs/crates/forecast-flows-worker`. Requires `FORECASTFLOWS_WORKER_BIN` to point at an existing regular file. The driver launches it directly with no `--project`, sysimage flag, or `JULIA_NUM_THREADS` env, because the Rust worker has no equivalent runtime knobs.
- `julia_worker`: the original `ForecastFlows.jl` Julia process described in the section below. This is now an explicit legacy rollback path, selected only with `FORECASTFLOWS_BACKEND=julia_worker`.

Rebalance, replay, pruning, translation, ranking, and fallback policy are backend-agnostic. Doctor and telemetry output tag each run with `worker_backend` and `worker_version` (worker `/health.package_version`, when the backend reports one). Julia-specific fields (`julia_version`, `sysimage_status`, `sysimage_detail`, `julia_threads`, `manifest_repo_url/rev/git_tree_sha1`, `allow_plain_julia_escape_hatch`) are reported only when `worker_backend = julia_worker`.

Build the Rust worker:

```bash
cargo build --release --manifest-path ../ForecastFlows.rs/crates/forecast-flows-worker/Cargo.toml
export FORECASTFLOWS_WORKER_BIN=$(realpath ../ForecastFlows.rs/target/release/forecast-flows-worker)
```

Runtime policy: the Rust worker is the default backend. Operators should set a valid `FORECASTFLOWS_WORKER_BIN`; no `FORECASTFLOWS_BACKEND` value is needed for the normal Rust path. Set `FORECASTFLOWS_BACKEND=julia_worker` only for an explicit legacy rollback. The Julia worker and sysimage tooling remain in-tree as a fallback.

## Purpose

ForecastFlows is integrated as an external whole-plan solver family for `RebalanceMode::Full`.
The driver now uses a single live path:

- `rust_prune`: ForecastFlows route generation plus Rust replay, exact-fee pruning,
  chunking, and final net-EV ranking

ForecastFlows still does not own exact tx grouping, tx packing, or submission.
Rust still owns:

- pool-state collection
- portfolio replay
- execution-group validation
- exact fee estimation, chunking, and final net-EV ranking
- fail-open behavior

## Julia Environment

- Repo-local project: `julia/forecastflows/`
- Dependency pin: `https://github.com/shotaronowhere/ForecastFlows.jl` on branch `codex/ff-public-univ3-parity`
- Worker command:

```bash
julia --startup-file=no --project=julia/forecastflows -e 'using ForecastFlows; ForecastFlows.serve_protocol(stdin, stdout)'
```

When `FORECASTFLOWS_BACKEND=julia_worker` is set, the runtime uses the repo-local `julia/forecastflows` project under `CARGO_MANIFEST_DIR`, so it does not depend on the shell cwd.
The Julia binary may be overridden with `FORECASTFLOWS_JULIA_BIN`.
The Julia project directory may be overridden with `FORECASTFLOWS_PROJECT_DIR`; when relative, it is resolved against the shell cwd.
Worker threads are set explicitly with `FORECASTFLOWS_JULIA_THREADS` and default to `auto`.
Worker pool size is controlled by `FORECASTFLOWS_POOL_SIZE` and defaults to `1`.

Local upstream iteration note:

- If you need to test an unmerged local `ForecastFlows.jl` checkout, use a local uncommitted Julia override such as `Pkg.develop(path="/abs/path/to/ForecastFlows.jl")` inside `julia/forecastflows/`.
- Do not commit absolute-path `Manifest.toml` pins for that workflow.

Optional cold-start optimization:

- Build script: `julia --project=julia/forecastflows julia/forecastflows/build_sysimage.jl`
- Runtime opt-in: `FORECASTFLOWS_SYSIMAGE=/abs/path/to/forecastflows_sysimage.{so,dylib,dll}`
- The local sysimage workload now precompiles the actual served path: `health`, direct compare protocol handling, and representative `UniV3` compare requests on both tiny and full-L1-like fixtures.
- The runtime only uses the sysimage when the configured path exists as a regular file and its sibling metadata file matches the current Julia major/minor version and the current `julia/forecastflows/Manifest.toml` hash.
- The sibling metadata file is written by the build script and currently records `julia_major_minor`, `manifest_sha256`, and `built_at_unix_secs`.
- Production-style solves now require an active sysimage unless `FORECASTFLOWS_ALLOW_PLAIN_JULIA=1` is set explicitly.
- Benchmark and doctor lanes still allow plain Julia automatically.
- Keep sysimage validation local for now. A typical production-like local run sets `FORECASTFLOWS_SYSIMAGE` plus a fixed `FORECASTFLOWS_JULIA_THREADS` value before running `forecastflows_doctor` or the explicit ForecastFlows benchmark tests.

## Runtime Profiles

- Production ForecastFlows requests use the `low_latency` solve tuning.
- `forecastflows_doctor` warmup and representative compare probes now use the same `low_latency` profile so operator timings match the live path.
- The explicit benchmark selected-row and latency helpers still use the `baseline` tuning unless you run the dedicated tuning helper.
- The dedicated tuning lane is:

```bash
FORECASTFLOWS_SYSIMAGE=/abs/path/to/forecastflows_sysimage.{so,dylib,dll} \
  cargo test print_shared_op_snapshot_forecastflows_tuning_rows_jsonl -- --ignored --nocapture
```

## Known Upstream Constraints / Requests

The dated maintainer handoff lives at
`docs/archive/implementation/2026-03-13-forecastflows-maintainer-notes.md`.

Short version:

- the solver core looks sound
- the main remaining upstream asks are richer branch diagnostics and a clearer executable rounding contract
- the biggest remaining driver-side latency work is on our Rust boundary, not the solver math
- the net-EV boundary decision lives at `docs/archive/implementation/2026-03-14-forecastflows-net-ev-boundary.md`

## Deferred Pending ForecastFlows Upstream

The following remain intentionally deferred on the `deep_trading` side while upstream feedback is pending:

- richer branch-level diagnostics directly from the worker protocol
- an explicit executable rounding contract for trade outputs
- multi-session or batching protocol work beyond the current cached NDJSON worker reuse

We will not fork the protocol locally to simulate these features. The dated maintainer handoff above remains the source of truth for upstream requests.

## Net-EV Ownership

Current benchmark rows live in `docs/solver_benchmark_matrix.md`.

- Julia solves without a gas model.
- Rust replays the returned candidate, runs the local exact-fee prune loop,
  compiles strict and packed programs, chunks under the `< 40_000_000` gas
  cap, and chooses the winner by validated net EV.
- `forecastflows_strategy` remains in telemetry for continuity and is currently
  always `rust_prune` on the live path.

## Process Model

- `FORECASTFLOWS_POOL_SIZE` long-lived worker processes per Rust process
- One request at a time per worker
- NDJSON over stdin/stdout, protocol version `2`
- Each worker slot keeps one cached `PredictionMarketWorkspace` and reuses it when repeated compare requests keep the same compatible market/outcome layout
- Driver-side supervision in Rust

Lifecycle:

1. Spawn the configured worker (Julia or Rust) with piped stdin/stdout/stderr.
2. On first use, send `health`.
3. For operator-facing binaries that request ForecastFlows, eagerly run `warm_forecastflows_worker()` at startup so every configured pool slot is prewarmed.
4. Warmup sends one tiny synthetic `compare_prediction_market_families` request on a `UniV3` fixture so the production code path is fully primed before live planning (Julia JIT warmup or Rust solver allocator priming).
5. Steady-state compare requests retry once on worker-level spawn, health, transport, or protocol failure before the external candidate is abandoned for that solve.
6. Repeated startup/transport failures trip a circuit breaker: after 3 consecutive failures, the worker enters cooldown for 60s, then exponentially backs off to a 5 minute cap until a compare succeeds.
7. Operator binaries and `forecastflows_doctor` call `shutdown_forecastflows_worker()` on exit as a best-effort cleanup hook so short-lived runs do not orphan worker processes.

Timeouts:

- steady-state request timeout: `5s`
- first post-spawn `health` timeout: `30s`
- startup warmup compare timeout: `30s`

Failure policy:

- timeout
- malformed JSON
- broken pipe / closed stdout
- worker `ok=false`
- local schema mismatch
- no certified external candidate
- certified candidate that fails local replay / grouping / program compilation

Transport/protocol failures kill the current worker process, clear the external candidate for that solve, and schedule one background warm respawn attempt when the breaker is not in cooldown. A protocol-valid `ok=false` response is treated as a request-level worker failure across startup warmup, doctor probes, and steady-state compare: Rust records the error, keeps the current worker alive, and does not retry, reset, or trip cooldown. Candidate-quality failures still clear the external candidate for that solve. In all cases the rebalance run fails open to the native path.

Diagnostics:

- Rust keeps the last 200 worker `stderr` lines in memory.
- Spawn / health / timeout / protocol errors include the recent `stderr` tail.
- Compare telemetry now records `forecastflows_strategy`, `workspace_reused`,
  local `candidate_build` timing, worker-estimated execution cost / net EV,
  Rust-validated total fee / net EV, and a `validation_only` flag.
- `validation_only` is currently `false` on the live path.
- `local_step_prune_ms` and `local_route_prune_ms` are populated for the live
  path because Rust always owns the exact-fee prune stage.
- `cargo run --bin forecastflows_doctor` prints the selected `worker_backend`, worker program path, reported worker version, launch args, live solve tuning, health summary, warmup timing, representative full-L1 compare timing, and solver-vs-driver timing splits when available, plus recent `stderr`. When running under `julia_worker`, it also prints Julia version, manifest repo URL / rev / git-tree, Julia thread config, sysimage status/detail, and the plain-Julia escape-hatch flag.
- `cargo run --bin forecastflows_doctor -- --json` emits the same report as machine-readable JSON.
- `cargo run --bin forecastflows_doctor -- --json --require-production-ready` exits nonzero unless the current production-style runtime is production-ready. Under `julia_worker` that requires an active sysimage; the `rust_worker` backend is production-ready on any successful health probe.
- If the doctor hits a worker or compare failure after launch, it still prints the partial report and exits nonzero with a final `failure:` line.

## Rolling Execute Runtime

`src/bin/execute.rs` is now the rolling live runtime for `RebalanceMode::Full`.

- It polls the latest block every `EXECUTION_BLOCK_POLL_MS` milliseconds and defaults to `250`.
- A planning cycle starts only when the latest block advances.
- The same ForecastFlows worker pool stays warm for the full process lifetime.
- Before solving, the runtime hashes the effective planning snapshot from current balances plus current `slot0`; if the snapshot is unchanged, it skips the solve for that block.
- Transient ForecastFlows transport failures do not poison that snapshot cache. On an unchanged market, `execute` retries the external solve on the next block instead of pinning the native fallback result until balances move.
- If blocks advance while a planning cycle is still busy, `execute` drops the stale backlog and replans only for the newest block once idle. It logs the skipped-block count instead of queueing every missed block.
- When `REBALANCE_SOLVER=forecastflows`, `execute` can run a non-blocking native audit every `FORECASTFLOWS_NATIVE_AUDIT_INTERVAL_BLOCKS` blocks; the default is `12` and `0` disables audits.
- Native audits rerun the native planner on the same snapshot as the live ForecastFlows cycle, compare the two plans with the ordinary Rust comparator, and log winner / net-EV delta / tx-count delta / action-count delta without blocking the live loop.
- Native audit task panics or join failures are logged and ignored so best-effort telemetry cannot stop the submission loop.

## Request Mapping

Current scope is the L1 replay model only for runtime translation. The doctor/sysimage representative timing probe uses an embedded copy of `test/fixtures/rebalancer_ab_live_l1_snapshot_report.json` for a full-L1-like benchmark workload without adding a runtime repo-path dependency.

Problem construction:

- `market_id = market.name`
- `outcome_id = market.outcome_token`
- `fair_value` from the Rust prediction map
- `initial_holding` from current Rust balances
- `collateral_balance` from current sUSD cash
- `split_bound` is omitted for single-market families so ForecastFlows uses its
  documented auto-bound / doubling behavior
- connected multi-market families send an explicit conservative `split_bound`
  equal to whole-sUSDS-truncated base collateral because the flattened request
  omits connector / invalid inventory and the worker's `cash + holdings`
  auto-bound can overstate feasible split / merge budget for that shape

Live compare request mapping is:

- Compare requests send only the translated problem and solve options.
- Execution-gas-aware Julia request shaping is no longer part of the live
  protocol. Exact executable economics remain a Rust-owned validation step.
- The Local Foundry benchmark harness launches the fixture binary with
  `FORECASTFLOWS_REQUEST_PROFILE=benchmark`, so benchmark rows use baseline
  ForecastFlows tuning and the longer benchmark timeout. Production executables
  still default to the low-latency production profile.

Markets are encoded as contiguous multi-band `UniV3` venues with a trailing zero-liquidity terminal band:

- `current_price` from current `slot0`
- one positive-liquidity band per contiguous active liquidity interval
- one zero-liquidity terminal band at the exhausted boundary
- `liquidity_L = active_liquidity_interval / 1e18`
- `fee_multiplier = 1 - FEE_PIPS / 1_000_000`

If Rust cannot derive a contiguous liquidity ladder that covers the current price from the live tick
and cumulative liquidity state, the snapshot is treated as unsupported for ForecastFlows and the
solve fails open to the native path. Rust does not approximate unsupported live geometry from
global min/max tick bounds. The only synthetic fallback is test-only support for benchmark fixtures
whose `slot0.tick` values are intentionally fake.

## Response Acceptance

The worker returns abstract trades plus aggregate `mint` / `merge` amounts.
Rust accepts a ForecastFlows candidate only if all of the following hold:

- result status is `certified`
- certificate is present and `passed = true`
- every `market_id` maps to a known Rust market
- every `outcome_id` matches the expected market outcome token
- every trade delta is finite
- the netted trade signs map cleanly to local buy or sell semantics
- `mint` and `merge` are not both positive above dust
- the translated action stream is locally replayable with current Rust `PoolSim`s
- the translated action stream compiles into supported execution-group shapes

Worker-reported EV, holdings, and final collateral are not used for acceptance.
Worker-reported `estimated_execution_cost` and `net_ev` are kept only as
diagnostics; Rust still validates the exact executable economics locally.
`direct_only` and `mixed_enabled` are translated independently, so one malformed certified branch
does not discard the other valid branch.

## Translation Rules

Trades are netted by `market_id` first.

Sign mapping:

- `collateral_delta < 0` and `outcome_delta > 0` => local `Buy`
- `collateral_delta > 0` and `outcome_delta < 0` => local `Sell`

Ordering rules:

- `direct_only`: initial `Sell*`, then final `Buy*`
- `mixed_enabled`: initial direct `Sell*`, then any direct `Merge*` rounds from existing holdings
- `mixed_enabled` with `mint > 0`: repeated `Mint` rounds, each followed by per-market `Sell*` replays capped by the minted round amount
- `mixed_enabled` with remaining `merge > 0`: repeated buy-to-shortfall `Buy*` rounds followed by `Merge`
- any leftover direct `Buy*` intents are replayed last

If both branches are uncertified, Rust records "no certified external candidate" and falls open to
native. If all certified branches fail translation, Rust records a translation failure and still
falls open to native.

The representative complete-set anchor market is the lexicographically smallest market name in the snapshot, matching the existing Rust compilers.

## Ranking

ForecastFlows candidates are converted into ordinary Rust `PlanResult`s with:

- `SolverFamily::ForecastFlows`
- `PlanCompilerVariant::ForecastFlowsDirect` or `PlanCompilerVariant::ForecastFlowsMixed`

Ranking remains the existing Rust comparator:

1. estimated net EV
2. estimated tx count
3. action count
4. stable family / variant tie-breaks

`HeadToHead` therefore compares native and ForecastFlows on the same local economics.

Explicit `RebalanceSolver::ForecastFlows` now means:

- try ForecastFlows first
- if there is a valid external candidate, use it
- otherwise fall back to the native planner
