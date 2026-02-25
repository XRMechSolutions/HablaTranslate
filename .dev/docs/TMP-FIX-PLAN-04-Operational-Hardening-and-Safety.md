# TMP Fix Plan 04 - Operational Hardening and Safety

Date: 2026-02-24  
Status: Draft (implementation not started)  
Priority: P2 (operations, safety, long-run maintainability)

## 1) Problem Cluster

This plan covers operational/safety findings that affect deployment behavior:
- C3: custom `DB_PATH` parent directory may not exist
- C4: `/health` endpoint exposes partial key fingerprint and runs expensive checks per request
- D4: recording cleanup glob may target non-session directories
- B2-related operational aspect: shutdown path consistency expectations (already addressed in Plan 01; validated here operationally)

Goal: strengthen runtime safety and operational predictability in production and troubleshooting scenarios.

## 2) Intended Operational Contract (Post-Fix)

1. Configured DB path is always creatable before DB initialization.
2. Public health endpoint is lightweight and non-sensitive.
3. Detailed/expensive checks are either cached or moved behind internal readiness path.
4. Recording retention cleanup only deletes recorder-owned session directories.
5. Shutdown/runbook behavior remains deterministic across console/system stop paths.

## 3) Proposed Implementation Steps (One-by-One)

### Step 1 - Ensure DB path parent exists
- File: `habla/server/config.py`
- Change:
  - Create `config.db_path.parent` if missing before `init_db`.
  - Log resolved DB path at startup for diagnostics.
- Risk:
  - Minimal; ensure permission errors surface clearly.

### Step 2 - Split health semantics (liveness vs readiness) or add caching
- Files:
  - `habla/server/main.py`
  - `habla/server/services/health.py`
- Change (recommended):
  - Keep `/health` lightweight (process alive + minimal app state).
  - Add `/health/ready` (or internal detailed endpoint) for expensive checks.
  - If keeping one endpoint, cache expensive checks briefly (e.g., 5-15s TTL).
- Risk:
  - Monitoring tooling may need endpoint update.

### Step 3 - Remove sensitive key fingerprint from health output
- File: `habla/server/services/health.py`
- Change:
  - Replace masked key string with non-sensitive status (`configured`/`missing`).
- Risk:
  - None; improves security posture.

### Step 4 - Restrict recorder cleanup to owned session folders
- File: `habla/server/services/audio_recorder.py`
- Change:
  - Replace broad `glob("*_*")` cleanup candidate selection with strict pattern/marker validation:
    - Directory name matches session format expected from recorder
    - Optional marker check (e.g., has `metadata.json` with expected schema)
  - Skip unknown directories explicitly.
- Risk:
  - If strict filter is wrong, old valid recordings may be skipped from retention cleanup.

### Step 5 - Add operational diagnostics for retention and health checks
- Files:
  - `habla/server/services/audio_recorder.py`
  - `habla/server/services/health.py`
- Change:
  - Log count of candidate directories and retained/deleted IDs.
  - Log health cache hit/miss (if caching implemented).
- Risk:
  - Keep logs concise to avoid noise.

## 4) Test Plan (Must Pass Before Merge)

## A) New/Updated Tests

1. DB path parent creation
- `tests` config/startup tests
- Set `DB_PATH` to nested non-existent directory.
- Assert startup creates parent and initializes DB successfully.

2. Health endpoint sensitivity/perf
- `tests/services/test_health.py` + route tests
- Assert no API key fingerprint appears in payload.
- Assert lightweight endpoint does not spawn ffmpeg per call (if split/cached design is adopted).

3. Recording cleanup safety
- `tests/services/test_audio_recorder.py`
- Seed output dir with:
  - valid recorder sessions
  - unrelated underscore directories
- Assert cleanup only removes valid owned sessions.

4. Retention behavior correctness
- `tests/services/test_audio_recorder.py`
- Verify deletion order oldest-first for owned sessions only.

## B) Regression Suite

- `python -m pytest habla/tests/services/test_audio_recorder.py -q`
- `python -m pytest habla/tests/services/test_health.py -q`
- `python -m pytest habla/tests --ignore=habla/tests/benchmark -q`

## C) Operational Smoke Checklist

1. Start server with custom `DB_PATH` nested directory.
2. Hit `/health` repeatedly; verify response latency stable and no sensitive data.
3. Verify readiness endpoint (or cached detailed path) reports dependency status.
4. Create mixed directories under recordings and run cleanup path; verify only owned sessions removed.

## 5) Rollout Notes

1. Coordinate health endpoint changes with deployment/monitoring configs.
2. Keep backward compatibility temporarily if external probes depend on existing path.
3. Document retention folder ownership rules in developer docs.

## 6) Exit Criteria

1. Custom `DB_PATH` startup path is robust.
2. Health output contains no key fingerprint and avoids heavy per-request work.
3. Recorder cleanup cannot delete non-owned directories.
4. Full regression suite passes with no operational behavior regressions.
