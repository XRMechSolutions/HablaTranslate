# TMP Fix Plan 03 - Error Handling and Graceful Degradation

Date: 2026-02-24  
Status: Draft (implementation not started)  
Priority: P1 (reliability + diagnosability)

## 1) Problem Cluster

This plan addresses error-path robustness and graceful degradation:
- A4: malformed websocket JSON is silently ignored
- D2: malformed vocab FTS queries can raise unhandled runtime errors
- D3: malformed idiom JSON file can break startup path
- Test-side robustness gaps:
  - broad exception assertions (`pytest.raises(Exception)`)
  - async warning-producing mocks hiding correctness signals

Goal: ensure invalid input and dependency issues produce controlled responses, keep startup resilient, and improve observability without breaking normal throughput.

## 2) Intended Behavioral Contract (Post-Fix)

1. Invalid client payloads always produce explicit protocol/API errors (no silent drops).
2. Malformed data sources (e.g., one bad idiom file) degrade gracefully and do not abort whole startup.
3. Search/input parser errors map to stable client-safe error responses.
4. Test suite catches specific failure modes and runs warning-clean for async behavior.

## 3) Proposed Implementation Steps (One-by-One)

### Step 1 - WebSocket invalid JSON handling
- File: `habla/server/routes/websocket.py`
- Change:
  - On `JSONDecodeError`, send websocket error message (`type=error`) with clear reason.
  - Keep loop alive unless policy dictates disconnect for repeated violations.
- Risk:
  - Increased error message volume if noisy clients spam bad frames; consider simple rate limiting.

### Step 2 - Harden vocab FTS query handling
- Files:
  - `habla/server/services/vocab.py`
  - `habla/server/routes/api.py` (mapping to HTTP status if needed)
- Change:
  - Catch sqlite/FTS parse exceptions from `MATCH`.
  - Return controlled `400 Invalid search query` (or equivalent consistent error).
- Risk:
  - Must avoid masking legitimate DB failures; only map known query-parse errors to 400.

### Step 3 - Make idiom loading resilient to malformed JSON
- Files:
  - `habla/server/services/idiom_scanner.py`
  - optional guard in `habla/server/pipeline/orchestrator.py` startup loop
- Change:
  - Catch `json.JSONDecodeError` and log filename/context.
  - Skip bad file and continue loading remaining idiom files.
- Risk:
  - Silent coverage loss if many files are bad; include explicit warning counters in logs/startup summary.

### Step 4 - Tighten failure logging context
- Files:
  - `habla/server/services/vocab.py`
  - `habla/server/services/idiom_scanner.py`
  - `habla/server/routes/websocket.py`
- Change:
  - Include actionable context (input class, endpoint/msg type, file path) in warnings/errors.
  - Avoid leaking sensitive payload data.
- Risk:
  - Over-logging; keep bounded/structured.

### Step 5 - Eliminate known test warning debt and broad exception assertions
- Files:
  - `habla/tests/pipeline/test_vad_buffer.py`
  - `habla/tests/db/test_database.py`
  - related tests for websocket malformed-json and vocab search errors
- Change:
  - Replace `pytest.raises(Exception)` with specific exception type/message where appropriate.
  - Replace incorrect `AsyncMock` usage for sync methods with `MagicMock`.
- Risk:
  - Existing tests may need mock refactors; keep changes scoped and deterministic.

## 4) Test Plan (Must Pass Before Merge)

## A) New/Updated Tests

1. WebSocket malformed JSON contract
- `tests/routes/test_websocket.py`
- Send invalid JSON text frame; assert error frame is sent and connection remains functional.

2. Vocab FTS malformed query handling
- `tests/routes/test_api.py` and/or `tests/services/test_vocab.py`
- Input malformed FTS query (e.g., unmatched quote).
- Assert controlled 400 response (or agreed contract) and no unhandled exception.

3. Idiom loader resilience
- `tests/services/test_idiom_scanner.py` or orchestrator startup tests
- Include one malformed idiom file among valid files.
- Assert startup/load continues and valid files still populate patterns.

4. DB FK assertion specificity
- `tests/db/test_database.py`
- Assert specific integrity exception semantics, not generic `Exception`.

5. Async warning cleanup verification
- `tests/pipeline/test_vad_buffer.py`
- Confirm no unawaited coroutine warnings from decoder mocks.

## B) Regression Suite

- `python -m pytest habla/tests/routes/test_websocket.py -q`
- `python -m pytest habla/tests/routes/test_api.py -q`
- `python -m pytest habla/tests/services/test_vocab.py -q`
- `python -m pytest habla/tests/services/test_idiom_scanner.py -q`
- `python -m pytest habla/tests/db/test_database.py -q`
- `python -m pytest habla/tests --ignore=habla/tests/benchmark -q`

## C) Warning Gate

Run targeted suite with warnings elevated (or equivalent CI flag) to ensure async mock cleanup holds:
- `python -m pytest habla/tests/pipeline/test_vad_buffer.py -W error::RuntimeWarning -q`

## 5) Rollout Notes

1. Implement by failure domain:
  - websocket protocol errors
  - vocab/FTS errors
  - idiom-loader resiliency
  - test hardening
2. Keep external behavior changes explicit in changelog/tests.
3. Preserve degraded-mode philosophy: partial functionality should continue wherever safe.

## 6) Exit Criteria

1. No silent-drop behavior for malformed websocket JSON.
2. Malformed FTS input no longer causes unhandled exception path.
3. Malformed idiom JSON file no longer aborts startup loading sequence.
4. Test suite has no known async unawaited warnings and uses specific exception assertions in critical paths.
