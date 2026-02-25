# TMP Fix Plan 05 - Code Organization and Documentation

Date: 2026-02-24
Status: Steps 2-3 complete (route split + docstrings), Steps 1/4-6 pending
Priority: P2 (maintainability + future velocity)

## 1) Problem Cluster

This plan addresses structural debt from continual feature additions:
- Mixed responsibilities in large route/service modules
- Inconsistent API behavior documentation vs implementation reality
- Sparse operational runbooks for restart/shutdown/recovery
- Missing explicit architecture boundaries and ownership notes

Goal: improve readability, maintainability, onboarding speed, and confidence during future changes.

## 2) Intended Maintainability Contract (Post-Fix)

1. Modules have clear responsibility boundaries and minimal cross-cutting side effects.
2. Public endpoints and key service methods have accurate contract docs.
3. Operational behavior (startup, restart, playback, shutdown, recovery) is documented as procedures.
4. Devs can trace request/flow paths quickly from docs to code.
5. Refactors ship with regression coverage to preserve behavior.

## 3) Proposed Implementation Steps (One-by-One)

### Step 1 - Define target module boundaries
- Scope:
  - `routes/api.py` (split by domain responsibility)
  - websocket/playback orchestration boundary
  - service layer responsibilities (DB logic vs route mapping)
- Output:
  - short boundary map in docs (what belongs where, what must not cross).
- Risk:
  - Over-splitting too early; keep pragmatic and incremental.

### Step 2 - Refactor route composition incrementally -- DONE (2026-02-24)
- Completed split into domain modules:
  - `routes/_state.py` — shared mutable state (`_pipeline`, `_lmstudio_manager`, `_playback_service`) and setters. Uses `TYPE_CHECKING` to avoid circular imports.
  - `routes/api_vocab.py` (114 lines) — vocab CRUD, review, search, export
  - `routes/api_system.py` (131 lines) — status, direction, mode, ASR, speakers, recording
  - `routes/api_sessions.py` (177 lines) — session history CRUD and export
  - `routes/api_idioms.py` (136 lines) — idiom pattern CRUD with auto-regex generation
  - `routes/api_llm.py` (231 lines) — LLM provider management + LM Studio routes
  - `routes/api_playback.py` (117 lines) — recording playback routes
  - `routes/api.py` (16 lines) — thin re-export aggregator for backward compatibility
- All importers migrated to new module paths:
  - `server/main.py` — imports directly from domain modules and `_state`
  - `tests/routes/test_api.py` — imports and patch targets updated
  - `tests/routes/test_playback_api.py` — imports and patch targets updated
- Verification: 628/628 tests pass, zero regressions (71s runtime).

### Step 3 - Normalize docstrings and contract notes -- DONE (2026-02-24)
- Route endpoints: added HTTP status codes and error conditions to all 27 endpoints across 6 route files.
  - Previously 12/27 endpoints lacked any docstring; now 27/27 have contract docs.
  - Every docstring now includes relevant status codes (200, 400, 404, 409, 502, 503).
  - Side effects documented where routes mutate state (DB commits, pipeline config, scanner reload).
- Service classes: added error contract and side-effect documentation to class docstrings.
  - `VocabService`: error contract (record_review returns dict, search raises ValueError, delete returns bool).
  - `IdiomScanner`: pattern loading sources, silent skip on bad regex, scan performance (<10ms).
  - `SpeakerTracker`: in-memory-only nature, None returns for unknown IDs, state save/restore.
- Pipeline/lifecycle: added ownership and lifecycle docstrings.
  - `PipelineOrchestrator`: full lifecycle (create/startup/shutdown/restart), ownership of sub-services, error contract (create_session returns 0 on error, errors via callback).
  - `Translator`: retry/fallback strategy, provider switching, degraded mode, cost tracking.
  - `ClientSession`: per-connection lifecycle, owned resources, cleanup contract.
- Verification: 637/637 tests pass, zero regressions.

### Step 4 - Add architecture and flow docs
- New/updated docs:
  - request/response flow maps for websocket and playback
  - session lifecycle states and transitions
  - dependency map for external components (Ollama/LM Studio/ffmpeg/HF token)
- Risk:
  - Documentation overhead; keep concise and source-linked.

### Step 5 - Add operational runbook sections
- Include:
  - startup diagnostics
  - degraded mode expectations
  - safe restart procedure
  - safe shutdown/quit procedure
  - “what to check first” incident checklist
- Risk:
  - Stale runbooks; version/date stamp each section.

### Step 6 - Enforce documentation+tests gate for refactors
- Policy:
  - Any route/service refactor PR must include:
    - updated module docs
    - updated tests for moved logic
    - zero behavior regression on existing suite
- Risk:
  - Slightly slower PR velocity; improved reliability offsets this.

## 4) Test/Validation Plan (Must Pass Before Merge)

## A) Refactor Safety Tests

1. Baseline snapshot before refactor
- Run full unit suite and record pass count/runtime.

2. Per-domain route tests post-split
- Ensure all existing route tests still pass with new module paths/imports.
- Add smoke tests for router registration coverage.

3. Contract parity checks
- Validate status codes and response schemas unchanged unless intentionally updated.

## B) Regression Suite

- `python -m pytest habla/tests --ignore=habla/tests/benchmark -q`
- targeted:
  - `python -m pytest habla/tests/routes -q`
  - `python -m pytest habla/tests/services -q`
  - `python -m pytest habla/tests/pipeline -q`

## C) Documentation Verification Checklist

1. Every public route group has status code/error behavior documented.
2. Lifecycle diagrams match actual code paths.
3. Runbook steps tested at least once against current implementation.
4. Docs include last-updated date.

## 5) Rollout Notes

1. Do not combine behavioral fixes and structural refactors in the same patch unless necessary.
2. Land refactors in small vertical slices (one domain at a time).
3. Keep temporary compatibility imports until tests and callers are fully migrated.

## 6) Exit Criteria

1. Route/service modules are split into clear ownership domains or documented with equivalent clarity.
2. Updated architecture + runbook docs exist and are source-linked.
3. Full regression suite remains green through refactor sequence.
4. No unresolved TODO migration shims remain for refactor scope.
