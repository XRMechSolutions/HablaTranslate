# Test Audit Review (2026-02-24)

Scope: Audit of implemented tests against `.dev/docs/Testing-Standards.md`.

## Findings

### 1) High - Missing error-path coverage hid real 500 behavior on uninitialized pipeline
- Status: ISSUE
- Impact: System endpoints can raise unhandled `AttributeError` when `_pipeline` is `None`, returning server error instead of controlled API response.
- Evidence:
  - `habla/server/routes/api.py:164` (`set_direction`)
  - `habla/server/routes/api.py:176` (`set_mode`)
  - `habla/server/routes/api.py:201` (`rename_speaker`)
  - `habla/tests/routes/test_api.py:382` (`TestSystemRoutes` lacks no-pipeline tests for these three routes)
- Follow-up:
  - Add no-pipeline tests for `/api/system/direction`, `/api/system/mode`, `/api/system/speakers/{speaker_id}`.
  - Decide and enforce expected status (`503` recommended for consistency with `/api/system/asr/language`).

### 2) Medium - Async mock setup causes unawaited coroutine warnings
- Status: ISSUE
- Impact: Warning noise can hide real async/resource issues and reduce trust in test signals.
- Evidence:
  - `habla/tests/pipeline/test_vad_buffer.py:413`
  - `habla/tests/pipeline/test_vad_buffer.py:414`
  - `habla/tests/pipeline/test_vad_buffer.py:459`
  - Full unit run shows warnings: `coroutine 'AsyncMockMixin._execute_mock_call' was never awaited`.
- Follow-up:
  - Replace `AsyncMock` with `MagicMock` for sync `stdin.write()`/`stdin.close()` paths.
  - Keep `AsyncMock` only for truly awaited methods (for example, `wait_closed`, `wait`, async readers).

### 3) Medium - Overly broad exception assertion in FK test
- Status: ISSUE
- Impact: Test may pass for wrong reasons and miss regression details.
- Evidence:
  - `habla/tests/db/test_database.py:394` uses `pytest.raises(Exception)`.
- Follow-up:
  - Assert specific DB error type/message for foreign-key failure.
  - Include message check (for example, matching `FOREIGN KEY constraint failed`).

### 4) Medium - Database concurrency behavior not yet validated
- Status: ISSUE
- Impact: WAL configuration is asserted, but concurrent read/write behavior is not proven by tests.
- Evidence:
  - `habla/tests/db/test_database.py` includes WAL pragma check only; no concurrency scenario test.
- Follow-up:
  - Add integration test with concurrent operations (for example, writer task + reader task, or parallel writes with busy timeout expectations).

## Checklist Snapshot

- Coverage: ISSUE (missing no-pipeline system-route error-path tests).
- Assertion Quality: ISSUE (broad `pytest.raises(Exception)` in DB FK test).
- Async Correctness: ISSUE (unawaited coroutine warnings from mock setup).
- Robustness / Anti-Trivial: ISSUE (error-path gaps permitted runtime 500 path).
- Naming/Organization: PASS for reviewed files.

## Priority Order

1. Fix system-route no-pipeline behavior and add tests.
2. Remove async mock warning sources in VAD/WebSocket-related tests.
3. Tighten DB FK exception assertion.
4. Add DB concurrency integration test.

## Linked Fix Plans

- Lifecycle/state management remediation plan:
  - [TMP-FIX-PLAN-01-Lifecycle-State-Management.md](/C:/Users/clint/HablaTranslate/.dev/docs/TMP-FIX-PLAN-01-Lifecycle-State-Management.md)
  - Primary mapping: A2, B1, B2, B3, C2
  - **Status: COMPLETE** - All 6 steps implemented, 636 tests pass.
- API contract and validation remediation plan:
  - [TMP-FIX-PLAN-02-API-Contract-Consistency.md](/C:/Users/clint/HablaTranslate/.dev/docs/TMP-FIX-PLAN-02-API-Contract-Consistency.md)
  - Primary mapping: A1, A3, C1, D1, D2 (and API aspects of A4)
  - **Status: COMPLETE** - All 5 steps implemented, 636 tests pass.
- Error handling and graceful degradation remediation plan:
  - [TMP-FIX-PLAN-03-Error-Handling-and-Degradation.md](/C:/Users/clint/HablaTranslate/.dev/docs/TMP-FIX-PLAN-03-Error-Handling-and-Degradation.md)
  - Primary mapping: A4, D2, D3, plus test hardening items from initial findings (2, 3, 4)
  - **Status: NOT STARTED**
- Operational hardening and safety remediation plan:
  - [TMP-FIX-PLAN-04-Operational-Hardening-and-Safety.md](/C:/Users/clint/HablaTranslate/.dev/docs/TMP-FIX-PLAN-04-Operational-Hardening-and-Safety.md)
  - Primary mapping: C3, C4, D4 (and operational validation complements for B2)
  - **Status: NOT STARTED**
- Code organization and documentation remediation plan:
  - [TMP-FIX-PLAN-05-Code-Organization-and-Documentation.md](/C:/Users/clint/HablaTranslate/.dev/docs/TMP-FIX-PLAN-05-Code-Organization-and-Documentation.md)
  - Primary mapping: cross-cutting maintainability debt discovered during A-D findings and fix rollout
  - **Status: COMPLETE** - api.py split into modules (implemented in separate session).
- Performance/resource efficiency and soak-stability remediation plan:
  - [TMP-FIX-PLAN-06-Performance-and-Soak-Stability.md](/C:/Users/clint/HablaTranslate/.dev/docs/TMP-FIX-PLAN-06-Performance-and-Soak-Stability.md)
  - Primary mapping: operational performance concerns related to C4 plus lifecycle cleanup/load behavior from A2/B1-B3/C2
  - **Status: COMPLETE** - All 5 steps implemented, 8 soak tests added, 636 tests pass.

---

## Code Audit Addendum (2026-02-24)

Scope: Runtime code audit across API routes, websocket lifecycle, and playback/session handling.

### A1) High - System endpoints can throw unhandled 500 when pipeline is not initialized
- Status: RESOLVED (Plan 02, Step 1)
- Impact: Calls to system management endpoints may crash with `AttributeError` instead of returning controlled API status.
- Evidence:
  - `habla/server/routes/api.py:163-168` (`set_direction` calls `_pipeline.set_direction(...)` with no `_pipeline` guard)
  - `habla/server/routes/api.py:175-180` (`set_mode` calls `_pipeline.set_mode(...)` with no `_pipeline` guard)
  - `habla/server/routes/api.py:200-207` (`rename_speaker` accesses `_pipeline.speaker_tracker...` with no `_pipeline` guard)
  - Contrast: `habla/server/routes/api.py:187-192` (`set_asr_language` correctly returns 503 when pipeline missing)
- Follow-up:
  - Add guard to all three endpoints: `if not _pipeline: raise HTTPException(503, "Pipeline not initialized")`.
  - Add API tests for no-pipeline behavior to prevent regressions.

### A2) High - WebSocket disconnect path does not explicitly stop active playback task
- Status: RESOLVED (Plan 01, Step 3)
- Impact: Playback may continue processing after client disconnect, and session shutdown can race with ongoing playback/decode work.
- Evidence:
  - `habla/server/routes/websocket.py:565-569` disconnect/finally path only runs `session.cleanup()` + `session.pipeline.close_session()`
  - `habla/server/routes/websocket.py:400-404` `cleanup()` stops microphone mode only (`if self.listening`), not active playback task
  - `habla/server/services/playback.py:145-153` playback runs as background task (`self._task = asyncio.create_task(...)`)
  - `habla/server/services/playback.py:226` starts websocket session decode task in playback mode
- Follow-up:
  - Ensure disconnect triggers playback cancellation (`PlaybackService.stop_playback()`), or extend `ClientSession.cleanup()` to stop playback-mode decode/stream resources.
  - Add integration test: disconnect during active playback must cancel playback and release decoder process.

### A3) Medium - Mutable default argument in session save route
- Status: RESOLVED (Plan 02, Step 5)
- Impact: Shared mutable defaults are a known Python footgun and can cause cross-request state leakage when mutation is introduced.
- Evidence:
  - `habla/server/routes/api.py:326` uses `body: dict = {}`
- Follow-up:
  - Change signature to `body: dict | None = None` and normalize with `body = body or {}`.

### A4) Medium - Invalid WebSocket JSON payloads are silently ignored
- Status: ISSUE
- Impact: Client receives no feedback for malformed control messages; this complicates diagnosis and violates informative-boundary error handling goals.
- Evidence:
  - `habla/server/routes/websocket.py:487-490` catches `json.JSONDecodeError` and `continue`s without error response
- Follow-up:
  - Send an explicit websocket error message for invalid JSON payloads (or structured protocol error counter/metric).
  - Add route test asserting error response for malformed JSON text frames.

---

## Code Audit Addendum 2 (2026-02-24)

Scope: Additional runtime/lifecycle audit pass to identify issues not already listed above.

### B1) High - Session ownership can drift across pipeline restarts (wrong pipeline handles close/persistence)
- Status: RESOLVED (Plan 01, Step 1)
- Impact: After a restart while a client is connected, the session object can start using a new pipeline instance without creating a new DB session on that pipeline. This risks mis-attribution/loss of persistence and incorrect close behavior.
- Evidence:
  - `habla/server/routes/websocket.py:84-87` `ClientSession.pipeline` returns module-global `_current_pipeline` dynamically.
  - `habla/server/routes/websocket.py:439` creates DB session once at connect (`await session.pipeline.create_session()`).
  - `habla/server/main.py:169-171` restart path swaps global pipeline reference via `set_ws_pipeline(pipeline)`.
  - `habla/server/routes/websocket.py:568` disconnect closes whichever pipeline is currently global (`await session.pipeline.close_session()`), not necessarily the one that created the session.
- Follow-up:
  - Bind each `ClientSession` to a concrete pipeline instance at connect time (do not dynamically swap per property lookup), or enforce controlled handoff that creates/closes sessions correctly on restart.
  - Add integration test: restart during active websocket session preserves session lifecycle correctness.

### B2) High - Console `quit` uses `os._exit(0)` and bypasses normal app shutdown path
- Status: RESOLVED (Plan 01, Step 5)
- Impact: Forced process exit skips FastAPI lifespan teardown logic that normally stops LM Studio monitor/process and closes DB cleanly; this can leave external processes/resources dangling.
- Evidence:
  - `habla/server/main.py:183-190` console `quit` branch calls `os._exit(0)` after pipeline shutdown.
  - `habla/server/main.py:197-205` normal teardown performs `pipeline.shutdown()`, `lmstudio_manager.stop_monitor()`, `lmstudio_manager.stop()`, and `close_db()`.
- Follow-up:
  - Replace hard exit with coordinated shutdown signal/path that executes full lifespan teardown.
  - Add operational test or runbook check ensuring console quit stops LM Studio and closes DB cleanly.

### B3) Medium - WebSocket setup failure before message loop can leak active-session state
- Status: RESOLVED (Plan 01, Step 2)
- Impact: If setup fails before entering the main receive loop, cleanup/finalization block is not reached, leaving `_active_session` set and potentially blocking recovery behavior.
- Evidence:
  - `habla/server/routes/websocket.py:430-440` creates/sets `_active_session`, initializes session, and calls `create_session()` before entering `try` block.
  - `habla/server/routes/websocket.py:476-569` `try/finally` cleanup only covers the receive loop onward.
- Follow-up:
  - Wrap full endpoint setup (from session creation onward) in one `try/finally` that always resets `_active_session` and runs cleanup.
  - Add test simulating exception during `initialize()` or `create_session()` and assert cleanup/state reset.

---

## Code Audit Addendum 3 (2026-02-24)

Scope: Additional API/config/playback/health checks for issues not previously recorded.

### C1) Medium - Unbounded/negative pagination and limit query params in REST endpoints
- Status: RESOLVED (Plan 02, Step 4)
- Impact: Negative values or very large limits can trigger inefficient queries and oversized responses (resource abuse risk, accidental heavy load).
- Evidence:
  - `habla/server/routes/api.py:67` `list_vocab(limit: int = 50, offset: int = 0, ...)` has no bounds.
  - `habla/server/routes/api.py:72` `vocab_due(limit: int = 20)` has no bounds.
  - `habla/server/routes/api.py:82` `search_vocab(..., limit: int = 20)` has no bounds.
  - `habla/server/routes/api.py:255` `list_sessions(limit: int = 20, offset: int = 0)` has no bounds.
  - `habla/server/routes/api.py:294` `get_session_exchanges(..., limit: int = 100, offset: int = 0)` has no bounds.
- Follow-up:
  - Use FastAPI `Query` constraints (for example, `ge=0`, `le=<max>` for limit/offset) and consistent max page size policy.
  - Add API tests for negative and oversized values returning validation errors.

### C2) Medium - Playback state can become stale on early error path
- Status: RESOLVED (Plan 01, Step 4)
- Impact: `PlaybackService` may report/retain a stale `_active_recording_id` even when playback never started, which can confuse observability and downstream control logic.
- Evidence:
  - `habla/server/services/playback.py:142` sets `self._active_recording_id = recording_id` before validating raw stream availability.
  - `habla/server/services/playback.py:150-151` returns error when `raw_stream.webm` missing, without clearing `_active_recording_id`.
- Follow-up:
  - Move `_active_recording_id` assignment after all preflight checks pass, or clear it on all early returns.
  - Add unit test for full-mode start with missing raw stream asserting no active recording id is retained.

### C3) Medium - Custom `DB_PATH` parent directory is not created before DB init
- Status: ISSUE
- Impact: When `DB_PATH` points outside `data/` to a non-existent directory, startup can fail during SQLite connect.
- Evidence:
  - `habla/server/config.py:151-152` allows overriding `config.db_path` via env.
  - `habla/server/config.py:169-173` creates `data_dir` and recording dirs, but not `config.db_path.parent`.
  - `habla/server/db/database.py:15` directly calls `aiosqlite.connect(str(db_path))`.
- Follow-up:
  - Ensure `config.db_path.parent.mkdir(parents=True, exist_ok=True)` before DB initialization.
  - Add config/startup test for custom `DB_PATH` in a new directory.

### C4) Low - `/health` endpoint leaks partial OpenAI key fingerprint and runs expensive checks per request
- Status: RESOLVED (Plan 06, Steps 1-2)
- Impact: Operational endpoint exposes partial secret fingerprint and performs process/network heavy checks on every hit, increasing information exposure and load under frequent probing.
- Evidence:
  - `habla/server/services/health.py:171-172` includes `Key set (<masked>)` in response message.
  - `habla/server/main.py:256-262` exposes runtime health publicly via `/health`.
  - `habla/server/services/health.py:236-237` runtime check runs `ffmpeg -version` subprocess each request.
- Follow-up:
  - Remove key fingerprint from public health output (use simple configured/not-configured status).
  - Cache expensive health checks briefly or split liveness/readiness endpoints (lightweight liveness, detailed internal readiness).

---

## Code Audit Addendum 4 (2026-02-24)

Scope: Additional service/API audit for new issues not listed in prior addendums.

### D1) Medium - Vocab review endpoint returns HTTP 200 for missing vocab IDs
- Status: RESOLVED (Plan 02, Step 2)
- Impact: API contract ambiguity; clients may treat failed reviews as success unless they inspect payload manually.
- Evidence:
  - `habla/server/services/vocab.py:97-99` returns `{"error": "not found"}` when item is missing.
  - `habla/server/routes/api.py:91-94` returns service result directly without mapping missing item to 404.
  - Runtime check: `POST /api/vocab/999/review` returns `200 {"error":"not found"}`.
- Follow-up:
  - Raise `HTTPException(404, "Vocab item not found")` in route (or raise in service and map consistently).
  - Add test asserting 404 for missing `vocab_id` in review endpoint.

### D2) Medium - Malformed FTS query can crash vocab search path (unhandled 500)
- Status: RESOLVED (Plan 02, Step 3)
- Impact: User-provided search text can trigger SQLite `OperationalError`, causing request failure instead of controlled 4xx response.
- Evidence:
  - `habla/server/services/vocab.py:145-154` executes `WHERE vocab_fts MATCH ?` with raw query and no error handling.
  - `habla/server/routes/api.py:81-83` forwards raw `q` directly to service.
  - Runtime check with `q='"'` raises `OperationalError: unterminated string`.
- Follow-up:
  - Catch SQLite FTS parse errors and return `HTTPException(400, "Invalid search query")` or sanitize/escape query syntax.
  - Add tests for malformed FTS inputs.

### D3) Medium - Invalid idiom JSON file can abort pipeline startup
- Status: ISSUE
- Impact: A single malformed idiom file can fail startup instead of degrading gracefully by skipping bad file.
- Evidence:
  - `habla/server/services/idiom_scanner.py:40-41` performs `json.load` with no `JSONDecodeError` handling.
  - `habla/server/pipeline/orchestrator.py:89-90` calls `load_from_json` in startup loop without per-file guard.
- Follow-up:
  - Add JSON parse error handling in `load_from_json` (log + skip bad file).
  - Add startup test with one malformed idiom file to confirm graceful continuation.

### D4) Low - Recording cleanup can target non-session directories by broad glob pattern
- Status: ISSUE
- Impact: Auto-cleanup may delete directories under recording output that are not recorder session folders, if names include underscores.
- Evidence:
  - `habla/server/services/audio_recorder.py:154` collects candidates via `output_dir.glob(\"*_*\")` (broad match).
  - `habla/server/services/audio_recorder.py:159-165` deletes matched directories when over retention limit.
- Follow-up:
  - Restrict cleanup target pattern to recorder-owned naming convention (`<session_id>_<timestamp>`), and verify directory markers before deletion.
  - Add cleanup safety tests ensuring unrelated underscore directories are not removed.
