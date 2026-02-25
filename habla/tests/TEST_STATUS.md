# Habla Test Suite Status

**Last Updated:** 2026-02-25

## Summary

| Run Mode | Tests | Runtime | Command |
|---|---:|---:|---|
| **Default (fast)** | 642 passed, 26 deselected | ~35s | `python -m pytest` |
| **Slow only** | 14 passed | ~32s | `python -m pytest -m slow` |
| **Benchmark only** | 9 (require live services) | varies | `python -m pytest -m benchmark` |
| **All tests** | 659 passed, 9 skipped | ~85s | `python -m pytest -m ""` |

- **Total Tests:** 668 (659 pass + 9 skipped benchmark)
- **Passing:** 659/659 (100%)
- **Default run skips:** slow (14 tests) and benchmark (9 tests) via `pytest.ini` marker filter

---

## Test Tiers

Tests are split into three tiers via pytest markers configured in `pytest.ini`:

| Tier | Marker | Count | What | Default |
|---|---|---:|---|---|
| Fast | _(unmarked)_ | 642 | Unit tests, mocked dependencies | Runs |
| Slow | `@pytest.mark.slow` | 14 | Queue drain (30s timeout), soak stability | Skipped |
| Benchmark | `@pytest.mark.benchmark` | 9 | Live GPU/service timing tests | Skipped |

The default `addopts` in `pytest.ini` includes `-m "not slow and not benchmark"` so that `python -m pytest` runs only the fast tier.

---

## Covered Components

| Source File | Test File | Tests | Status |
|---|---|---:|---|
| `db/database.py` | `tests/db/test_database.py` | 37 | Covered |
| `pipeline/orchestrator.py` | `tests/pipeline/test_orchestrator.py` | 102 | Covered |
| `pipeline/translator.py` | `tests/pipeline/test_translator.py` | 49 | Covered |
| `pipeline/vad_buffer.py` | `tests/pipeline/test_vad_buffer.py` | 32 | Covered |
| `routes/api.py` | `tests/routes/test_api.py` | 76 | Covered |
| `routes/websocket.py` | `tests/routes/test_websocket.py` | 107 | Covered |
| `routes/playback endpoints` | `tests/routes/test_playback_api.py` | 12 | Covered |
| `services/audio_recorder.py` | `tests/services/test_audio_recorder.py` | 28 | Covered |
| `services/health.py` | `tests/services/test_health.py` | 47 | Covered |
| `services/idiom_scanner.py` | `tests/services/test_idiom_scanner.py` | 23 | Covered |
| `services/idiom merge logic` | `tests/services/test_idiom_merger.py` | 14 | Covered |
| `services/lmstudio_manager.py` | `tests/services/test_lmstudio_manager.py` | 26 | Covered |
| `services/playback.py` | `tests/services/test_playback.py` | 27 | Covered |
| `services/speaker_tracker.py` | `tests/services/test_speaker_tracker.py` | 32 | Covered |
| `services/vocab.py` | `tests/services/test_vocab.py` | 27 | Covered |
| _(soak stability)_ | `tests/test_soak_stability.py` | 20 | Covered (slow) |
| _(benchmarks)_ | `tests/benchmark/` | 9 | Covered (benchmark) |

---

## Recent Changes

**2026-02-25 — Phase 2.5 server resilience tests**
- Added 3 tests in `test_translator.py`: rate limiter enforcement, no-delay, metric tracking
- Added 4 tests in `test_health.py`: health monitor down/recovery/stable/no-session
- Added 2 tests in `test_api.py`: timing header present, /health excluded from timing
- Added 3 tests in `test_websocket.py`: heartbeat stale close, active no-close, cleanup cancel

**2026-02-25 — Cost persistence tests + test tier restructuring**
- Added 4 tests in `test_database.py`: cost columns exist, defaults, round-trip, aggregation
- Added 3 tests in `test_translator.py`: session cost snapshot, field exclusion, DB round-trip
- Marked `TestShutdown` in `test_orchestrator.py` as `@pytest.mark.slow` (6 tests, 30s queue drain)
- Marked `test_soak_stability.py` as `pytestmark = pytest.mark.slow` (20 tests)
- Updated `pytest.ini` addopts to skip slow and benchmark by default
- Default run dropped from ~80s to ~32s

---

## Coverage Gaps

No Priority 1/2/3 gaps remain. Potential future additions (lower priority):
- Focused tests for custom logic in `models/schemas.py` only if/when custom validators/computed behavior is added.
- Prompt template branching tests in `models/prompts.py` only if prompt generation logic becomes more conditional.

---

## Current Notes

- Full unit run currently reports **3 runtime warnings** from async mocks in:
  - `tests/pipeline/test_vad_buffer.py` (stdin write/close mock await behavior)
  - `tests/routes/test_websocket.py` (async mock not awaited)
- These warnings do not fail tests today, but should be cleaned up to keep the suite warning-free.

---

## Running Tests

```bash
# Fast tests only (recommended default — 32s)
python -m pytest

# Include slow tests (queue drain, soak stability — adds ~30s)
python -m pytest -m slow

# Benchmark tests only (require live GPU + services)
python -m pytest -m benchmark

# ALL tests (clear marker filter)
python -m pytest -m ""

# Stop on first failure
python -m pytest -x

# Verbose output
python -m pytest -v
```
