# REST API

Domain-split REST endpoints across 9 route modules, mounted via a single `api.py` aggregator. All routes use async handlers with `aiosqlite` for database access.

## Route Modules

### `/api/vocab` — Vocabulary & Spaced Repetition (`api_vocab.py`, 185L)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/` | Create vocab item (or bump encounter count if duplicate) |
| GET | `/` | List vocab with optional `category` filter |
| GET | `/due` | Items due for spaced repetition review |
| GET | `/review-session` | Balanced session: 70% due, 20% new, 10% struggling |
| GET | `/stats` | Count totals, due items, by-category breakdown |
| GET | `/search` | FTS5 full-text search across terms and meanings |
| GET | `/{id}` | Single vocab item |
| POST | `/{id}/review` | Record SM-2 review (quality 0-5) |
| PATCH | `/{id}` | Update vocab fields |
| DELETE | `/{id}` | Delete vocab item |
| GET | `/export/anki` | Export as TSV for Anki import |
| GET | `/export/json` | Export as JSON array |

### `/api/system` — Pipeline Control (`api_system.py`, 257L)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/status` | Full system status (direction, mode, speakers, queue depth, LLM provider) |
| POST | `/direction` | Set translation direction (es_to_en / en_to_es / auto) |
| GET | `/languages` | Supported languages for client UI |
| POST | `/mode` | Set mode (conversation / classroom) |
| POST | `/asr/language` | Toggle ASR auto-language detection |
| PUT | `/speakers/{id}` | Rename speaker + optional role hint |
| POST | `/recording` | Toggle recording enabled/disabled |
| GET | `/metrics` | Comprehensive metrics (confidence, corrections, idioms, pipeline, translator, processing) |

### `/api/sessions` — Session History (`api_sessions.py`, 259L)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | List sessions (most recent first) |
| GET | `/search` | Search across session exchange text |
| GET | `/{id}` | Session detail with speakers and exchange count |
| GET | `/{id}/summary` | LLM-generated study summary |
| POST | `/{id}/summary/regenerate` | Regenerate study summary |
| GET | `/{id}/exchanges` | Exchanges in chronological order |
| POST | `/{id}/save` | Mark session saved with optional notes |
| GET | `/{id}/export` | Export transcript as plain text |
| GET | `/{id}/exchanges/{eid}/audio` | WAV clip for an exchange |
| DELETE | `/{id}` | Delete session and all associated data (cascading) |

### `/api/llm` — LLM Provider Management (`api_llm.py`, 352L)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/current` | Active provider, model, metrics, costs |
| GET | `/providers` | Probe all configured providers with connection status |
| GET | `/models` | List models for a specific provider |
| POST | `/select` | Switch provider and model at runtime |
| GET | `/costs` | Cloud cost tracking (session + all-time) |
| GET | `/apikeys` | Masked API key status |
| POST | `/apikey` | Validate and store API key |
| GET | `/pricing` | Per-model pricing table with tier labels |

### `/api/lmstudio` — LM Studio Process Control (`api_llm.py`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/status` | Running state and loaded models |
| POST | `/restart` | Restart LM Studio and reload models |
| GET | `/models` | Proxy LM Studio's model list |

### `/api/idioms` — Idiom Pattern Management (`api_idioms.py`, 144L)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | List user-contributed patterns |
| POST | `/` | Create pattern from saved phrase (auto-generates regex) |
| DELETE | `/{id}` | Delete pattern |

### `/api/corrections` & `/api/recordings` — Whisper Fine-Tuning (`api_corrections.py`, 233L)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/recordings/{id}/ground-truth` | Load ground truth for correction |
| PUT | `/recordings/{id}/ground-truth` | Save corrected ground truth (atomic write) |
| GET | `/corrections/failed-segments` | ASR failures from quality_metrics |
| GET | `/corrections/stats` | Aggregate correction progress |

### `/api/playback` & Recording Audio (`api_playback.py`, 129L)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/recordings` | List saved recording sessions |
| GET | `/recordings/{id}` | Metadata and ground truth for a recording |
| GET | `/recordings/{id}/audio` | Stream raw WebM audio |
| GET | `/recordings/{id}/audio/{filename}` | Stream individual segment WAV |
| POST | `/playback/start` | Start playback through live pipeline |
| POST | `/playback/stop` | Stop active playback |

### `/api/bookmarks` — Conversation Bookmarks (`api_bookmarks.py`, 109L)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/exchanges/{id}/bookmark` | Toggle bookmark with optional note |
| GET | `/bookmarks/recent` | Recent bookmarks across all sessions |
| GET | `/sessions/{id}/bookmarks` | Bookmarks for a specific session |

## Shared State

`routes/_state.py` holds module-level globals set during app lifespan startup:
- `_pipeline` — PipelineOrchestrator instance
- `_lmstudio_manager` — LM Studio manager (if active)
- `_playback_service` — PlaybackService instance
- `_recording_config` — Recording configuration

Routes import from `_state.py` to access these without circular dependencies.

## Error Response Conventions

| Code | Meaning |
|------|---------|
| 400 | Validation error, invalid input |
| 404 | Resource not found |
| 409 | Conflict (duplicate) |
| 422 | Unprocessable input data |
| 502 | Provider unreachable |
| 503 | Pipeline not initialized |

## Design Rationale

**Why domain-split routes?** The original monolithic `api.py` grew to 600+ lines covering vocab, sessions, system control, and LLM config. Splitting into domain modules (api_vocab, api_sessions, api_llm, etc.) keeps each file focused and independently navigable.

**Why `_state.py` globals?** FastAPI's dependency injection works for request-scoped objects, but the pipeline and services are singleton lifespan objects. Module-level globals with setter functions avoid circular imports and keep the lifespan setup clean.

## Planned Direction

- **REST API reference doc** (Phase 9.1): Consolidated documentation of every endpoint with request/response schemas, error codes, and example curl commands.
- **Multi-language API** (7.9): Update `POST /api/system/direction` to accept `source_lang` + `target_lang` instead of hardcoded pair names.
- **Conversation export** (7.8): Add `POST /api/translate/text` for text/URL import with paragraph-by-paragraph translation.
