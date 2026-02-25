# TMP Fix Plan 06 - Performance, Resource Efficiency, and Soak Stability

Date: 2026-02-24  
Status: COMPLETE (implemented 2026-02-24)
Priority: P2 (efficiency + long-session reliability)

## 1) Problem Cluster

This plan addresses performance and long-run stability concerns surfaced during audits:
- Expensive health checks on hot path (`/health`) and process-spawn overhead
- Potential task/process leakage risk during repeated playback/listen cycles
- Queue/backpressure behavior under sustained load
- Long-session memory/resource drift not yet validated by soak-style checks

Goal: ensure stable behavior across long classroom-style sessions and repeated operational actions.

## 2) Intended Runtime Contract (Post-Fix)

1. Core real-time flow remains responsive under sustained input.
2. No unbounded growth in queues/tasks/memory over long sessions.
3. Repeated start/stop/restart/playback actions leave clean resource state.
4. Operational endpoints do not consume disproportionate resources.
5. Performance thresholds are measurable and regression-tested.

## 3) Proposed Implementation Steps (One-by-One)

### Step 1 - Define measurable runtime SLO-style targets
- Targets (initial):
  - Queue drain time bounds after stop/disconnect
  - Maximum steady queue depth under nominal input
  - Health endpoint median/95th response time budget
  - No leaked async tasks after N repeated lifecycle operations
- Risk:
  - Unrealistic thresholds create flaky checks; set pragmatic baselines first.

### Step 2 - Add lightweight runtime metrics hooks
- Scope:
  - queue depth snapshots
  - in-flight translation count
  - playback active state transitions
  - cleanup completion markers
- Keep metrics internal/log-based initially.
- Risk:
  - Logging noise; keep sampling bounded.

### Step 3 - Optimize/contain expensive operational checks
- Scope:
  - `/health` expensive dependencies
  - repeated subprocess/network calls
- Strategy:
  - cached readiness windows or split endpoints (as in Plan 04)
- Risk:
  - stale status windows; choose short TTL and expose timestamp.

### Step 4 - Add resource cleanup assertions in code paths
- Scope:
  - websocket disconnect cleanup
  - playback cancel/finish cleanup
  - pipeline shutdown/restart cycle
- Add explicit “final state invariants” checks (no lingering task refs, flags reset).
- Risk:
  - false positives if invariants too strict; tune per mode.

### Step 5 - Build soak and repetition test harness
- Add repeat-loop integration tests for:
  - start/stop listening cycles
  - playback start/stop cycles
  - reconnect cycles
  - restart cycles
- Risk:
  - runtime length; mark as integration/slow and run separately in CI.

### Step 6 - Baseline and enforce non-regression
- Record before/after metrics in docs/artifacts.
- Add acceptance thresholds with guard bands (2-3x expected where timing-sensitive).
- Risk:
  - environment variance; avoid brittle absolute timing where not needed.

## 4) Test Plan (Must Pass Before Merge)

## A) New/Updated Tests

1. Repeated lifecycle stability tests
- Add integration tests running N cycles:
  - websocket connect -> start listening -> stop listening -> disconnect
  - playback start -> stop -> repeat
- Assert no leaked tasks and stable terminal state each cycle.

2. Queue/backpressure tests
- Stress queue with controlled input bursts.
- Assert bounded queue growth and eventual drain.

3. Health endpoint overhead tests
- Validate lightweight health path stays below agreed latency budget.
- Validate expensive checks are cached/split as designed.

4. Long-session memory/resource checks (smoke-level)
- Simulate sustained operation for fixed duration.
- Assert no monotonic uncontrolled growth in tracked resource counters.

## B) Regression Suite

- `python -m pytest habla/tests --ignore=habla/tests/benchmark -q`
- `python -m pytest -m integration habla/tests -q`
- targeted repeated-run suites for websocket/playback/orchestrator

## C) Benchmark/Soak Runs (Separate Lane)

1. Existing benchmark suite where applicable:
- `python -m pytest habla/tests/benchmark -v`

2. Soak scenario script/test:
- fixed-duration run with periodic metric snapshots.

3. Compare against baseline report and flag regressions.

## 5) Rollout Notes

1. Land instrumentation before optimizations to capture baseline.
2. Keep soak tests optional locally but required in pre-release/CI nightly lane.
3. Document thresholds and rationale in test status docs.

## 6) Exit Criteria

1. Repeated lifecycle tests show no leaked task/process state.
2. Queue and cleanup invariants hold under stress/repetition.
3. Health endpoint overhead reduced/controlled by design.
4. Soak run completes without stability regressions against baseline.
