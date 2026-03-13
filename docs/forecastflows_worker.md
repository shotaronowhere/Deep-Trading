# ForecastFlows Worker Integration

Status: current as of 2026-03-13.
Source-of-truth code paths:

- `src/portfolio/core/forecastflows/client.rs`
- `src/portfolio/core/forecastflows/translate.rs`
- `src/portfolio/core/rebalancer.rs`

## Purpose

ForecastFlows is integrated as an external whole-plan solver family for `RebalanceMode::Full`.
It does not own gas modeling, tx grouping, tx packing, or submission. Rust still owns:

- pool-state collection
- portfolio replay
- execution-group validation
- fee estimation and net-EV ranking
- fail-open behavior

## Julia Environment

- Repo-local project: `julia/forecastflows/`
- Dependency pin: `https://github.com/shotaronowhere/ForecastFlows.jl` on branch `forecast`
- Worker command:

```bash
julia --startup-file=no --project=julia/forecastflows -e 'using ForecastFlows; ForecastFlows.serve_protocol(stdin, stdout)'
```

By default the runtime uses the repo-local `julia/forecastflows` project under `CARGO_MANIFEST_DIR`, so it does not depend on the shell cwd.
The Julia binary may be overridden with `FORECASTFLOWS_JULIA_BIN`.
The Julia project directory may be overridden with `FORECASTFLOWS_PROJECT_DIR`; when relative, it is resolved against the shell cwd.

Optional cold-start optimization:

- Build script: `julia --project=julia/forecastflows julia/forecastflows/build_sysimage.jl`
- Runtime opt-in: `FORECASTFLOWS_SYSIMAGE=/abs/path/to/forecastflows_sysimage.{so,dylib,dll}`
- The runtime only uses the sysimage when the configured path exists as a regular file and its sibling metadata file matches the current Julia major/minor version and the current `julia/forecastflows/Manifest.toml` hash; otherwise it warns and falls back to plain Julia automatically.

## Process Model

- One long-lived worker process per Rust process
- One request at a time per worker
- NDJSON over stdin/stdout, protocol version `2`
- Driver-side supervision in Rust

Lifecycle:

1. Spawn Julia with piped stdin/stdout/stderr.
2. On first use, send `health`.
3. For operator-facing binaries that request ForecastFlows, eagerly run `warm_forecastflows_worker()` at startup.
4. Warmup sends one tiny synthetic `compare_prediction_market_families` request on a `UniV3` fixture so the production code path gets JIT-compiled before live planning.
5. Steady-state compare requests retry once on worker-level spawn, health, transport, or protocol failure before the external candidate is abandoned for that solve.
6. Repeated startup/transport failures trip a circuit breaker: after 3 consecutive failures, the worker enters cooldown for 60s, then exponentially backs off to a 5 minute cap until a compare succeeds.
7. Operator binaries and `forecastflows_doctor` call `shutdown_forecastflows_worker()` on exit as a best-effort cleanup hook so short-lived runs do not orphan Julia worker processes.

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
- `cargo run --bin forecastflows_doctor` prints Julia path/version, launch args, sysimage status, health summary, warmup timing, representative full-L1 compare timing, and recent `stderr`.
- If the doctor hits a worker or compare failure after launch, it still prints the partial report and exits nonzero with a final `failure:` line.

## Request Mapping

Current scope is the L1 single-range replay model only for runtime translation. The doctor/sysimage representative timing probe uses an embedded copy of `test/fixtures/rebalancer_ab_live_l1_snapshot_report.json` for a full-L1-like benchmark workload without adding a runtime repo-path dependency.

Problem construction:

- `market_id = market.name`
- `outcome_id = market.outcome_token`
- `fair_value` from the Rust prediction map
- `initial_holding` from current Rust balances
- `collateral_balance` from current sUSD cash
- `split_bound` omitted so ForecastFlows uses its documented auto-bound / doubling behavior

Markets are encoded as one-band `UniV3` venues with a trailing zero-liquidity terminal band:

- `current_price` from current `slot0`
- positive-liquidity band at the active buy boundary
- zero-liquidity terminal band at the exhausted sell boundary
- `liquidity_L = active_liquidity / 1e18`
- `fee_multiplier = 1 - FEE_PIPS / 1_000_000`

If Rust cannot derive the current active in-range liquidity band from the live tick and cumulative
liquidity state, the snapshot is treated as unsupported for ForecastFlows and the solve fails open
to the native path. Rust does not approximate unsupported geometry from global min/max tick bounds.

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
`direct_only` and `mixed_enabled` are translated independently, so one malformed certified branch
does not discard the other valid branch.

## Translation Rules

Trades are netted by `market_id` first.

Sign mapping:

- `collateral_delta < 0` and `outcome_delta > 0` => local `Buy`
- `collateral_delta > 0` and `outcome_delta < 0` => local `Sell`

Ordering rules:

- `direct_only`: `Sell*`, then `Buy*`
- `mixed_enabled` with `mint > 0`: `Mint`, then `Sell*`, then `Buy*`
- `mixed_enabled` with `merge > 0`: `Buy*`, then `Merge`, then `Sell*`

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
