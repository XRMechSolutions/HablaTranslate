# TMP Fix Plan 01 - Lifecycle and State Management

Date: 2026-02-24  
Status: COMPLETE (implemented 2026-02-24)
Priority: P0 (stability/reliability)

## 1) Problem Cluster

This plan addresses lifecycle/state issues identified in audit findings:
- A2: WebSocket disconnect path does not explicitly stop active playback task
- B1: Session ownership drift across pipeline restarts
- B2: Console `quit` uses `os._exit(0)` and bypasses normal teardown
- B3: WebSocket setup failure before receive loop can leak active-session state
- C2: Playback state can remain stale on early error path

Goal: ensure clean ownership and cleanup of session/pipeline/playback resources across connect, restart, disconnect, stop, and quit paths.

## 2) Intended Behavioral Contract (Post-Fix)

1. A websocket session is bound to one concrete pipeline instance for its lifetime.
2. Disconnect always runs cleanup and closes the same pipeline session that was created on connect.
3. Active playback is explicitly stopped/cancelled during disconnect/cleanup.
4. Startup/setup exceptions do not leave `_active_session` or background tasks leaked.
5. Playback service never retains stale `active_recording_id` on failed starts.
6. Console quit uses graceful app shutdown flow (DB close, manager stop, monitor stop), not forced process kill.

## 3) Proposed Implementation Steps (One-by-One)

### Step 1 - Bind session to concrete pipeline instance
- File: `habla/server/routes/websocket.py`
- Change:
  - Replace dynamic `ClientSession.pipeline` property usage with fixed `self.pipeline` assigned at session creation.
  - Keep `_current_pipeline` global for new connections only.
- Risk:
  - Existing tests may assume property behavior; update tests to reflect per-session binding.

### Step 2 - Move full websocket setup into guarded try/finally
- File: `habla/server/routes/websocket.py`
- Change:
  - Wrap from `ClientSession(...)` creation through receive loop in one `try/finally`.
  - Ensure `_active_session = None` in finally.
  - Ensure `cleanup()` and `close_session()` run if setup partially succeeded.
- Risk:
  - Double-cleanup; protect with idempotent guards.

### Step 3 - Explicit playback cancellation in cleanup/disconnect
- Files:
  - `habla/server/routes/websocket.py`
  - `habla/server/routes/api.py` (for service access if needed)
  - `habla/server/services/playback.py` (if helper needed)
- Change:
  - On session cleanup/disconnect, call playback stop when active for that session.
  - Ensure decode task/process cleanup is deterministic for playback mode.
- Risk:
  - Race with user-initiated `/api/playback/stop`; make stop idempotent.

### Step 4 - Fix stale playback state on early error
- File: `habla/server/services/playback.py`
- Change:
  - Assign `_active_recording_id` only after start preflight checks pass.
  - Or clear `_active_recording_id` before each error return.
- Risk:
  - Minimal; mostly state consistency.

### Step 5 - Replace forced console quit with graceful shutdown path
- File: `habla/server/main.py`
- Change:
  - Remove `os._exit(0)` path.
  - Trigger standard shutdown flow (same teardown branch used by lifespan exit).
  - If direct signal is needed, raise/propagate controlled cancellation.
- Risk:
  - Behavior change for interactive console; verify command still exits reliably.

### Step 6 - Harden cleanup idempotency
- Files:
  - `habla/server/routes/websocket.py`
  - `habla/server/services/playback.py`
  - Possibly `habla/server/pipeline/orchestrator.py`
- Change:
  - Ensure repeated stop/cleanup calls are safe and do not throw.
- Risk:
  - None if guarded carefully.

## 4) Test Plan (Must Pass Before Merge)

## A) New/Updated Unit & Integration Tests

1. WebSocket session binding
- `tests/routes/test_websocket.py`
- Case: restart global pipeline mid-session; existing session continues using original pipeline instance for close/cleanup.

2. WebSocket setup failure cleanup
- `tests/routes/test_websocket.py`
- Case: exception in `initialize()` or `create_session()`.
- Assert: `_active_session` reset, no leaked tasks, cleanup attempted safely.

3. Disconnect with active playback
- `tests/routes/test_websocket.py` + `tests/services/test_playback.py`
- Case: playback active, then websocket disconnect.
- Assert: playback stop/cancel invoked, decode task stopped, no active playback afterwards.

4. Playback early error state
- `tests/services/test_playback.py`
- Case: full mode, missing `raw_stream.webm`.
- Assert: start returns error and `_active_recording_id` remains `None`.

5. Console quit graceful path
- `tests` for `main` lifecycle behavior (mock-based)
- Assert: quit path invokes normal shutdown components (`pipeline.shutdown`, `lmstudio_manager.stop_monitor/stop`, `close_db`) without forced exit.

6. Idempotent cleanup
- Add tests calling cleanup/stop twice.
- Assert: no exception, stable final state.

## B) Regression Suite

Run all current unit tests:
- `python -m pytest habla/tests --ignore=habla/tests/benchmark -q`

Run targeted suites:
- `python -m pytest habla/tests/routes/test_websocket.py -q`
- `python -m pytest habla/tests/services/test_playback.py -q`
- `python -m pytest habla/tests/routes/test_playback_api.py -q`
- `python -m pytest habla/tests/pipeline/test_orchestrator.py -q`

## C) Runtime Smoke Checklist

1. Start server, connect one client, start/stop listening.
2. Start playback, disconnect client mid-playback.
3. Reconnect client, verify no stale active session/playback.
4. Trigger console `restart`, ensure new connections work.
5. Trigger console `quit`, verify graceful shutdown logs and no orphan LM Studio process.

## 5) Rollout Notes

1. Land in small commits per step above (avoid one large patch).
2. Keep behavior flags/logs during transition to diagnose cleanup timing.
3. After fix merge, remove temporary diagnostic logs if noisy.

## 6) Exit Criteria

1. All tests in section 4 pass.
2. No leaked tasks/processes observed in disconnect/restart flows.
3. No forced-exit path remains for console quit.
4. Session close/persistence maps to the same pipeline/session that created it.
