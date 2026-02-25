# TMP Fix Plan 02 - API Contract Consistency and Input Validation

Date: 2026-02-24  
Status: COMPLETE (implemented 2026-02-24)
Priority: P1 (correctness + client reliability)

## 1) Problem Cluster

This plan addresses API contract and validation findings:
- A1: Missing pipeline guards on system endpoints (`/direction`, `/mode`, `/speakers/{id}`)
- A3: Mutable default arg in `save_session`
- C1: Unbounded pagination/limit params
- D1: `POST /api/vocab/{id}/review` returns 200 with error payload on missing ID
- D2: Vocab FTS malformed queries can raise unhandled 500
- A4 (API-adjacent): malformed websocket JSON silently ignored (covered in Plan 01 runtime handling; listed for consistency expectations)

Goal: enforce consistent, predictable API responses and validation semantics.

## 2) Intended API Contract (Post-Fix)

1. Missing pipeline dependency returns `503` with stable message.
2. Missing resources return `404`, never `200` with `{"error": ...}` payload.
3. Invalid input/query syntax returns `400`/`422` with clear message.
4. Pagination/limit query params are bounded and non-negative.
5. Route signatures avoid mutable defaults.
6. Error responses are machine-consistent across related endpoints.

## 3) Proposed Implementation Steps (One-by-One)

### Step 1 - Add missing `_pipeline` guards
- File: `habla/server/routes/api.py`
- Endpoints:
  - `POST /api/system/direction`
  - `POST /api/system/mode`
  - `PUT /api/system/speakers/{speaker_id}`
- Change:
  - Early `if not _pipeline: raise HTTPException(503, "Pipeline not initialized")`
- Risk:
  - Existing clients depending on current 500 behavior (unlikely/undesired).

### Step 2 - Normalize `review_vocab` not-found behavior
- Files:
  - `habla/server/services/vocab.py`
  - `habla/server/routes/api.py`
- Change:
  - Route maps missing item to 404 (or service raises domain exception mapped by route).
  - Remove 200-with-error payload pattern.
- Risk:
  - Client may currently parse `{"error":"not found"}`; communicate API change.

### Step 3 - Handle malformed FTS query safely
- Files:
  - `habla/server/services/vocab.py`
  - `habla/server/routes/api.py` (if route-level mapping needed)
- Change:
  - Catch SQLite FTS parse errors and return controlled `HTTPException(400, "Invalid search query")`.
  - Optionally sanitize or quote query syntax before MATCH.
- Risk:
  - Over-sanitization can reduce search flexibility; preserve valid MATCH syntax.

### Step 4 - Add query bounds for list/search endpoints
- File: `habla/server/routes/api.py`
- Change:
  - Use constrained query params (FastAPI `Query`) for `limit`/`offset`:
    - `offset >= 0`
    - `1 <= limit <= MAX_LIMIT` (per endpoint policy)
- Candidate endpoints:
  - `/api/vocab`
  - `/api/vocab/due`
  - `/api/vocab/search`
  - `/api/sessions`
  - `/api/sessions/{id}/exchanges`
  - `/api/idioms`
- Risk:
  - Strict limits may require small frontend adjustments.

### Step 5 - Remove mutable default in `save_session`
- File: `habla/server/routes/api.py`
- Change:
  - `body: dict | None = None`, then `body = body or {}`
- Risk:
  - Minimal; behavior preserved.

### Step 6 - Standardize error payload/messages
- File: `habla/server/routes/api.py`
- Change:
  - Ensure similar endpoints use consistent detail text for same failure class.
  - Document response codes in route docstrings/comments for future maintainers.
- Risk:
  - Keep compatibility where possible; avoid unnecessary message churn.

## 4) Test Plan (Must Pass Before Merge)

## A) New/Updated API Tests

1. Pipeline guard tests
- `tests/routes/test_api.py`
- Add:
  - `/api/system/direction` with no pipeline -> `503`
  - `/api/system/mode` with no pipeline -> `503`
  - `/api/system/speakers/{id}` with no pipeline -> `503`

2. Vocab review not-found test
- `tests/routes/test_api.py`
- Assert missing `vocab_id` returns `404` (not `200` error object).

3. Vocab search malformed FTS test
- `tests/routes/test_api.py` or `tests/services/test_vocab.py`
- Input `q='"'` (or equivalent malformed MATCH)
- Assert controlled error response (preferred `400`) and no unhandled exception.

4. Query bounds tests
- `tests/routes/test_api.py`
- For each bounded endpoint:
  - negative offset -> validation failure
  - zero/negative/too-large limit -> validation failure
  - valid edge values pass

5. Save session default body test
- `tests/routes/test_api.py`
- Call without JSON body; assert stable behavior and no shared-state side effects.

## B) Regression Suite

- `python -m pytest habla/tests/routes/test_api.py -q`
- `python -m pytest habla/tests/services/test_vocab.py -q`
- `python -m pytest habla/tests --ignore=habla/tests/benchmark -q`

## C) Contract Verification Checklist

1. Verify endpoint status codes in frontend/API consumers:
  - system control endpoints
  - vocab review endpoint
  - search endpoint
2. Confirm no route now emits `200` for resource-not-found.
3. Confirm validation failures return predictable schema (`detail` structure).

## 5) Rollout Notes

1. Implement in small patches:
  - Guards first
  - Not-found normalization
  - FTS handling
  - Query bounds
  - Signature cleanup
2. Update API docs/README snippets if status codes change.
3. Notify frontend assumptions if any tests reveal dependency on old behavior.

## 6) Exit Criteria

1. API tests cover all newly enforced status codes.
2. No unhandled exceptions for malformed search inputs.
3. All bounded endpoints reject invalid limit/offset values.
4. Full unit suite remains green after contract changes.
