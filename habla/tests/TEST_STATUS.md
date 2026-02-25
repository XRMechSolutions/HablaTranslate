# Habla Test Suite Status

**Last Updated:** 2026-02-25

## Summary

- **Total Tests:** 648 (639 passed, 9 skipped benchmark)
- **Passing:** 639/639 (100%)
- **Runtime:** ~80s
- **Command:** `python -m pytest habla/tests --ignore=habla/tests/benchmark -q`

---

## Covered Components

| Source File | Test File | Tests | Status |
|---|---|---:|---|
| `db/database.py` | `tests/db/test_database.py` | 33 | Covered |
| `pipeline/orchestrator.py` | `tests/pipeline/test_orchestrator.py` | 102 | Covered |
| `pipeline/translator.py` | `tests/pipeline/test_translator.py` | 43 | Covered |
| `pipeline/vad_buffer.py` | `tests/pipeline/test_vad_buffer.py` | 32 | Covered |
| `routes/api.py` | `tests/routes/test_api.py` | 74 | Covered |
| `routes/websocket.py` | `tests/routes/test_websocket.py` | 104 | Covered |
| `routes/playback endpoints` | `tests/routes/test_playback_api.py` | 12 | Covered |
| `services/audio_recorder.py` | `tests/services/test_audio_recorder.py` | 28 | Covered |
| `services/health.py` | `tests/services/test_health.py` | 43 | Covered |
| `services/idiom_scanner.py` | `tests/services/test_idiom_scanner.py` | 23 | Covered |
| `services/idiom merge logic` | `tests/services/test_idiom_merger.py` | 14 | Covered |
| `services/lmstudio_manager.py` | `tests/services/test_lmstudio_manager.py` | 26 | Covered |
| `services/playback.py` | `tests/services/test_playback.py` | 27 | Covered |
| `services/speaker_tracker.py` | `tests/services/test_speaker_tracker.py` | 32 | Covered |
| `services/vocab.py` | `tests/services/test_vocab.py` | 27 | Covered |

---

## Coverage Gaps

No Priority 1/2/3 gaps remain from the prior plan. Current suite includes coverage for:
- `routes/api.py`
- `db/database.py`
- `services/health.py`
- `services/lmstudio_manager.py`
- `services/audio_recorder.py`

Potential future additions (lower priority):
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
# Unit tests (recommended default)
python -m pytest habla/tests --ignore=habla/tests/benchmark -q

# Verbose unit run
python -m pytest habla/tests --ignore=habla/tests/benchmark -v

# Benchmark tests (require live services)
python -m pytest habla/tests/benchmark -v

# Stop on first failure
python -m pytest habla/tests --ignore=habla/tests/benchmark -x
```
