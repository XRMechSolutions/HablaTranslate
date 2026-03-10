# Core Services

Standalone business logic modules that don't fit neatly into the pipeline or route layers. Currently covers health monitoring and vocabulary management.

## Health Service (`services/health.py`, 310L)

### Startup Checks

`run_startup_checks(config)` validates external dependencies before the pipeline loads:

| Check | What It Does | Failure Mode |
|-------|-------------|--------------|
| `check_ollama()` | HTTP GET to Ollama `/api/tags`, verify model pulled | Degraded (LLM unavailable) |
| `check_ffmpeg()` | Subprocess `ffmpeg -version` | Degraded (no audio decode) |
| `check_database()` | Query vocab table | Critical |
| `check_lmstudio()` | HTTP health check + loaded model count | Degraded |
| `check_openai_key()` | API key presence | Info only |
| `check_anthropic_key()` | API key presence | Info only |

Post-pipeline checks verify WhisperX and Pyannote loaded correctly. All checks allow degraded mode — the server starts even if some components are down.

### Runtime Checks

`run_runtime_checks()` provides live health status with a 15-second cache to reduce overhead on frequent `/health` requests.

### LLM Health Monitor

`run_llm_health_monitor()` runs as a background task, checking the active LLM provider every 60 seconds:
- On failure: notify client via WebSocket `error` message
- On recovery: notify client via `status` message
- For LM Studio: can trigger `ensure_running()` for auto-recovery

### Status Aggregation

Individual checks roll up to system health: DOWN > DEGRADED > OK. Each check reports name, status, message, and latency.

## Vocab Service (`services/vocab.py`, 327L)

### SM-2 Spaced Repetition

Full implementation of the SM-2 algorithm with extensions:

- **Ease factor**: Bounded [1.3, 5.0] (prevents infinite growth)
- **Interval scaling**: quality 0-2 resets to 1 day, quality 3-5 extends interval
- **Lapse tracking**: Counts how many times a card was forgotten after being learned
- **Mature threshold**: Cards with interval > 21 days are considered mature

### Review Sessions

`get_review_session()` plans balanced study sessions:
- 70% due for review
- 20% new items (never reviewed)
- 10% struggling items (high lapse count or low ease factor)

### Duplicate Handling

When saving a vocab item that already exists (case-insensitive match on term), the service increments `times_encountered` instead of creating a duplicate. This tracks word frequency naturally.

### Search

FTS5 full-text search across term, meaning, literal, and category columns. Raises ValueError on malformed FTS syntax (handled by route layer).

### Export

- **Anki TSV**: Front (term), Back (meaning + literal + category), Tags (region)
- **JSON**: Full vocab item array

## Design Rationale

**Why health checks allow degraded mode?** A classroom session shouldn't fail to start because Ollama is temporarily unreachable. The server starts in text-only mode and recovers when services come back.

**Why SM-2 specifically?** It's the proven algorithm used by Anki, which the project exports to. Using the same algorithm means SRS intervals will be consistent whether studying in Habla or in Anki.

**Why encounter counting instead of dedup?** If a student hears "por cierto" five times in a session, that frequency data is valuable for prioritizing review. Silently deduplicating would lose that signal.

## Planned Direction

- **Vocab review streaks** (7.7): Daily streak tracking, review goals, gamification.
- **Pronunciation tracking**: Store pronunciation scores alongside SRS data, integrate poor pronunciation as lower quality ratings.
