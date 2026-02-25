# Habla — Implementation Task List

Comprehensive roadmap from current state to fully functional app. Tasks are ordered by priority within each phase. Check off items as completed.

## Related Documents

Detailed plans, standards, and audit results that support the tasks below.

| Document | Status | Contents |
|----------|--------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current | Full file tree, line counts, module dependency graph, tech table |
| [.dev/docs/Audit-Checklist.md](.dev/docs/Audit-Checklist.md) | Reference | Code audit checklist (read before any audit) |
| [.dev/docs/Testing-Standards.md](.dev/docs/Testing-Standards.md) | Reference | Test quality bar, naming conventions, anti-trivial-pass rules |
| [.dev/docs/Test-Audit-Review-2026-02-24.md](.dev/docs/Test-Audit-Review-2026-02-24.md) | Complete | Audit of test suite against testing standards |
| [.dev/docs/TMP-FIX-PLAN-01-Lifecycle-State-Management.md](.dev/docs/TMP-FIX-PLAN-01-Lifecycle-State-Management.md) | Complete | Lifecycle & state management fixes |
| [.dev/docs/TMP-FIX-PLAN-02-API-Contract-Consistency.md](.dev/docs/TMP-FIX-PLAN-02-API-Contract-Consistency.md) | Complete | API contract consistency & input validation |
| [.dev/docs/TMP-FIX-PLAN-03-Error-Handling-and-Degradation.md](.dev/docs/TMP-FIX-PLAN-03-Error-Handling-and-Degradation.md) | Not started | Error handling & graceful degradation (P1) |
| [.dev/docs/TMP-FIX-PLAN-04-Operational-Hardening-and-Safety.md](.dev/docs/TMP-FIX-PLAN-04-Operational-Hardening-and-Safety.md) | Not started | Operational hardening & safety (P2) |
| [.dev/docs/TMP-FIX-PLAN-05-Code-Organization-and-Documentation.md](.dev/docs/TMP-FIX-PLAN-05-Code-Organization-and-Documentation.md) | Partial | Route split + docstrings done; other steps pending (P2) |
| [.dev/docs/TMP-FIX-PLAN-06-Performance-and-Soak-Stability.md](.dev/docs/TMP-FIX-PLAN-06-Performance-and-Soak-Stability.md) | Complete | Performance & soak stability fixes |
| [habla/tests/TEST_STATUS.md](habla/tests/TEST_STATUS.md) | Current | Test suite status: 668 tests, 659 passing, 9 skipped benchmarks |
| [habla/tests/EDGE_CASES.md](habla/tests/EDGE_CASES.md) | Reference | Idiom detection edge case catalog |
| [docs/AUDIO_TUNING_GUIDE.md](docs/AUDIO_TUNING_GUIDE.md) | Reference | Audio parameter tuning guide |
| [docs/WHISPER_FINE_TUNING_PLAN.md](docs/WHISPER_FINE_TUNING_PLAN.md) | Reference | WhisperX fine-tuning plan |
| [.dev/docs/PLAN-Phase1-Remaining.md](.dev/docs/PLAN-Phase1-Remaining.md) | Complete | Idiom client wiring + OpenAI cost persistence |

## Reusable Code References

Other projects on this machine with patterns to draw from:

| Project | Path | Reusable Patterns |
|---------|------|-------------------|
| **GeneralizedServiceChatbot** | `C:\Users\clint\GeneralizedServiceChatbot` | Multi-provider LLM switching (OpenAI/Claude/LM Studio), settings UI, health checking |
| — LMStudioMonitorService | `GeneralizedServiceChatbot\LMStudioMonitorService\` | .NET Windows service: process mgmt, model loading (CLI + API), health endpoint on :5000 |
| — LMStudioClient.kt | `GeneralizedServiceChatbot\app\...\LMStudioClient.kt` | Model discovery (`/v1/models`), health checks, auto-restart, `fetchAvailableModels()` |
| — LMStudioHealthChecker.kt | `GeneralizedServiceChatbot\app\...\LMStudioHealthChecker.kt` | Health check with model verification, restart/load requests |
| — SettingsActivity.kt | `GeneralizedServiceChatbot\app\...\SettingsActivity.kt` | Provider dropdown + dynamic model spinner, `loadAvailableModels()` |
| **StoryTimeLanguageKMM** | `C:\Users\clint\AndroidStudioProjects\StoryTimeLanguageKMM` | GPT-5 Nano/Mini Responses API, fallback chains, cost optimization |
| — LlmClient.kt | `shared\...\llm\LlmClient.kt` | `executeWithFallback()` — tries models in order, routes by `apiType` (`"responses"` vs `"chat"`) |
| — Gpt5StreamingHandler.kt | `shared\...\domain\Gpt5StreamingHandler.kt` | GPT-5 SSE streaming: `response.output_text.delta` events |
| — stringResourceTranslator.py | (project root) | Python GPT-5 Nano translation with per-model cost tracking |
| **StockMaster** | `C:\Clint file drop\StockMaster` | Server resilience patterns (already ported to Phase 2.5) |
| — circuit_breaker_manager.py | `ExecutionEngine\core\circuit_breaker_manager.py` | Circuit breaker with 7 types, cooldown, emergency stop |
| — logger_config.py | `DataGatheringTools\monitoring\logger_config.py` | Structured JSON logging, rotating files, metrics collection |
| — system_monitor.py | `control_center\services\system_monitor.py` | CPU/memory/disk monitoring with thresholds and alerts |
| **ConjugationGameKMM** | `C:\Users\clint\AndroidStudioProjects\ConjugationGameKMM` | Spanish verb conjugation game — potential vocab/drill patterns |
| **story-server** | `C:\Users\clint\story-server` | Node.js backend for StoryTime — server-side LLM orchestration |

---

## Phase 1: Core Feature Gaps (Highest Impact)

These are missing features that block primary use cases.

> **Phase 1 complete.** Implementation details: [PLAN-Phase1-Remaining.md](.dev/docs/PLAN-Phase1-Remaining.md).

### 1.1 Build Vocab Review Page (`vocab.html`)
- [x] Create `habla/client/vocab.html` with mobile-first dark theme matching `index.html`
- [x] Due-for-review view — fetch `/api/vocab/due`, display flashcard-style review UI
- [x] SM-2 review flow — quality rating buttons (0-5), POST to `/api/vocab/{id}/review`
- [x] Browse all vocab — paginated list from `/api/vocab/`, filter by category (idiom, correction, phrase)
- [x] Search vocab — full-text search via `/api/vocab/search`
- [x] Stats dashboard — total items, due count, by-category breakdown from `/api/vocab/stats`
- [x] Export buttons — Anki TSV (`/api/vocab/export/anki`) and JSON (`/api/vocab/export/json`)
- [x] Delete individual items with confirmation
- [x] Navigation back to main translator (`/`)
- [x] Fix vocab button in `index.html` (currently opens `/vocab` in new tab — should work seamlessly)

### 1.2 Persist Exchanges to Database
- [x] Create session record in `sessions` table on WebSocket connect
- [x] Save each completed exchange to `exchanges` table in orchestrator (after translation completes)
- [x] Persist speaker profiles to `speakers` table (currently in-memory only, lost on restart)
- [x] Persist topic summary to session record
- [x] Add session close timestamp on WebSocket disconnect
- [x] Expose session history via REST endpoint (`GET /api/sessions/`, `GET /api/sessions/{id}/exchanges`)

### 1.3 Idiom Feedback Loop
- [x] Wire client `saveIdiom()` in `ui.js` to also `POST /api/idioms` (fire-and-forget after vocab save)
- [x] Backend: generate regex pattern from idiom phrase (`api_idioms.py:_generate_pattern`)
- [x] Insert generated pattern into `idiom_patterns` DB table
- [x] Reload idiom scanner patterns after new pattern is added (or on next startup)
- [x] Add REST endpoint to manage user-contributed idiom patterns (`GET/POST/DELETE /api/idioms/`)
- [x] Prevent duplicate patterns (check existing DB + JSON patterns before insert)

### 1.4 Vocab FTS Sync
- [x] Add SQLite triggers to keep `vocab_fts` virtual table in sync on INSERT/UPDATE/DELETE to `vocab`
- [x] FTS rebuild on startup to index any pre-trigger data
- [x] Verify full-text search returns newly added items immediately

### 1.5 Multi-Provider LLM Support (Ollama + LM Studio + OpenAI GPT-5 Nano)

Allow runtime switching between LLM providers and models from the client UI.

> **Reference projects for reusable code:**
>
> | Project | Path | Relevant Code |
> |---------|------|---------------|
> | GeneralizedServiceChatbot | `C:\Users\clint\GeneralizedServiceChatbot` | Provider switching, LM Studio management |
> | — LMStudioClient.kt | `app\src\main\java\com\xrmech\customizablechatbot\LMStudioClient.kt` | Model discovery (`/v1/models`), health checks, auto-restart |
> | — LMStudioHealthChecker.kt | `app\src\main\java\com\xrmech\customizablechatbot\LMStudioHealthChecker.kt` | Health check with model verification, restart/load requests |
> | — LMStudioMonitorService | `LMStudioMonitorService\` (.NET 8 Windows service) | Process management, model loading via CLI + API, health endpoint on :5000 |
> | — SettingsManager.kt | `app\src\main\java\com\xrmech\customizablechatbot\SettingsManager.kt` | Provider/model persistence pattern |
> | — SettingsActivity.kt | `app\src\main\java\com\xrmech\customizablechatbot\SettingsActivity.kt` | Settings UI with provider dropdown + dynamic model spinner |
> | StoryTimeLanguageKMM | `C:\Users\clint\AndroidStudioProjects\StoryTimeLanguageKMM` | GPT-5 Nano/Mini API integration |
> | — LlmClient.kt | `shared\src\commonMain\kotlin\com\xrmech\storytimelanguage\llm\LlmClient.kt` | Three-provider fallback chain, Responses API vs Chat API routing |
> | — Gpt5StreamingHandler.kt | `shared\src\commonMain\kotlin\com\xrmech\storytimelanguage\domain\Gpt5StreamingHandler.kt` | GPT-5 streaming SSE parsing |
> | — stringResourceTranslator.py | (root) | Python GPT-5 Nano translation with cost tracking |

#### Backend — Provider Abstraction

Three provider types, each with different API formats:

| Provider | Endpoint | Request Format | Response Format |
|----------|----------|---------------|-----------------|
| **Ollama** (local) | `POST {url}/api/generate` | `prompt` + `system` fields | `response.response` |
| **LM Studio** (local) | `POST {url}/v1/chat/completions` | OpenAI `messages` array | `choices[0].message.content` |
| **OpenAI GPT-5** (cloud) | `POST https://api.openai.com/v1/responses` | `input` string, `reasoning.effort`, `text.verbosity` | `output[].content[].text` (where `type == "output_text"`) |

- [x] Rename `TranslatorConfig` to support multiple providers; add `provider` field (`"ollama"`, `"lmstudio"`, or `"openai"`)
- [x] Add env var overrides: `LMSTUDIO_URL`, `LMSTUDIO_MODEL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `LLM_PROVIDER`
- [x] Abstract `_call_ollama()` into provider-aware `_call_llm()` that routes to the correct API
- [x] Handle response parsing per provider (Ollama, LM Studio, OpenAI Responses API)
- [x] For OpenAI: use `max_output_tokens`, `reasoning.effort: "minimal"`, `text.verbosity: "low"`, `text.format.type: "json_object"`
- [x] Add model fallback chain (cloud→local only: if OpenAI fails, try Ollama then LM Studio; local providers never fall back to cloud to avoid unexpected costs)
- [x] Update `_quick_translate` in orchestrator to go through `translator._call_llm()` instead of direct Ollama call

#### Backend — Model Discovery API
- [x] `GET /api/llm/providers` — Return available providers with connection status and models
- [x] `GET /api/llm/models?provider=ollama` — List models for a specific provider
- [x] `POST /api/llm/select` — Switch provider and model at runtime (no server restart)
- [x] `GET /api/llm/current` — Return current provider, model, and metrics

#### Frontend — Settings Panel in Client
- [x] Add gear/settings button to header bar (next to mode toggle)
- [x] Build settings modal with provider dropdown, model dropdown, status indicator, refresh button, metrics display
- [x] On provider/model change, POST to `/api/llm/select`
- [x] Persist last-used provider/model selection in `localStorage`
- [x] On page load, restore selection from `localStorage` or fall back to server default

#### OpenAI API Cost Tracking

> **Reference:** `StoryTimeLanguageKMM\stringResourceTranslator.py` — per-model cost tracking with running totals

- [x] Track input/output token counts from OpenAI responses (`usage.input_tokens`, `usage.output_tokens`)
- [x] Calculate cost per request using model pricing (`gpt-5-nano`: $0.05/1M, `gpt-5-mini`: $0.25/1M, `gpt-4o-mini`: $2.50/1M)
- [x] Maintain running session cost total in translator metrics
- [x] Show current session cost in settings panel (only when OpenAI provider is active)
- [x] Add `GET /api/llm/costs` endpoint — session cost, all-time cost, breakdown by model
- [x] Local providers (Ollama, LM Studio) show "Free (local)" instead of cost
- [x] Persist cumulative cost to DB (per-session on close, all-time loaded from DB on startup)

#### Health Check Integration
- [x] Update `health.py` to check active LLM provider (Ollama, LM Studio, OpenAI key check)
- [x] Include current LLM provider/model in `/health` response
- [x] Include current LLM provider/model in `/api/system/status` response

---

## Phase 2: Client Robustness

Polish the web client for reliable real-world use over Tailscale on mobile.

### 2.1 WebSocket Reconnection Hardening
- [x] Add exponential backoff to reconnect (3s → 6s → 12s → max 60s, reset on success)
- [x] Cap max reconnect attempts (20), then show "Connection lost" banner with manual retry button
- [x] Handle `listening_started` / `listening_stopped` server messages to sync client state
- [x] Clear orphaned partial transcript cards on reconnect
- [x] Queue pending text inputs during disconnection, send on reconnect

### 2.2 Error Feedback in UI
- [x] Replace `console.error` calls with visible toast notifications (auto-dismiss after 5s)
- [x] Show server `error` message type in UI (currently logged to console only)
- [x] Show user-facing error when vocab save fails (revert button state)
- [x] Show "text-only mode" notice if server reports ASR unavailable (via `status` message `pipeline_ready`)
- [x] Show "Ollama unavailable" notice if translation times out (via server error messages)

### 2.3 PWA Support
- [x] Create `manifest.json` (app name, icons, theme color, standalone display)
- [x] Create basic service worker for app shell caching (HTML/CSS/JS — not API responses)
- [x] Register service worker in `index.html`
- [x] Add app icons (192px, 512px) for Android home screen install
- [ ] Test "Add to Home Screen" flow on Chrome Android

### 2.4 Audio Compatibility
- [x] Detect Safari / iOS and show "unsupported browser" message (no WebM/Opus or MediaRecorder)
- [x] Add fallback codec negotiation for browsers without Opus support
- [x] Verify mic works over Tailscale HTTPS (getUserMedia requires secure context)

---

## Phase 2.5: Server Resilience (from StockMaster patterns)

Operational patterns ported from StockMaster for production stability.

### 2.5.1 Startup Health Checks
- [x] Health check service (`server/services/health.py`) — validates Ollama, ffmpeg, DB, HF_TOKEN at startup
- [x] `GET /health` endpoint for Docker and uptime monitoring
- [x] Post-pipeline checks verify WhisperX and diarization loaded correctly
- [x] Degraded mode: server starts even if some components are DOWN (logs warnings)

### 2.5.2 Retry with Exponential Backoff
- [x] Translator `_call_ollama()` retries transient errors (timeouts, connection errors, 5xx)
- [x] Exponential backoff: 1s, 2s, 4s (max 15s), up to 3 retries for translation, 1 for topic summary
- [x] Error categorization: TEMPORARY (retry) vs PERMANENT (fail fast — model not found, 404)
- [x] Translator metrics tracking (requests, successes, retries, failures, timeouts, latency)

### 2.5.3 Docker Hardening
- [x] Non-root user in Dockerfile
- [x] `HEALTHCHECK` directive with 120s start-period (model loading)
- [x] `restart: unless-stopped` on both services
- [x] Memory limits (8GB Habla, 6GB Ollama)
- [x] Ollama pinned to version 0.5 (was `latest`)
- [x] `depends_on` with `condition: service_healthy` — Habla waits for Ollama to be ready
- [x] Ollama health check on `/api/tags` (15s interval, 5 retries)

### 2.5.4 Graceful Shutdown
- [x] Signal handlers for SIGINT/SIGTERM with logging
- [x] Pipeline drains processing queue (up to 30s) before shutting down
- [x] Session state saved to `data/last_session.json` (topic, context, speakers, metrics)
- [x] Worker task properly cancelled and awaited

### 2.5.5 Structured Logging
- [x] Console handler (human-readable, respects `LOG_LEVEL` env var)
- [x] Rotating file handler (`data/habla.log`, 10MB x 5 backups, DEBUG level)
- [x] Error-only file (`data/habla_errors.log`, 5MB x 3 backups)
- [x] File/line info in log files for debugging

### 2.5.6 Ollama Rate Limiting
- [x] Add minimum-interval rate limiter to `_call_provider()` (prevent flooding when partials fire rapidly)
- [x] Track `rate_limited` count in translator metrics
- [x] Configurable via `rate_limit_interval` (default 0.5s) and `RATE_LIMIT_INTERVAL` env var

### 2.5.7 Runtime LLM Health Monitoring
- [x] Periodic background health check (every 60s) for active LLM provider
- [x] If provider goes down mid-session, notify connected client via WebSocket `error` message
- [x] Auto-detect recovery and send `status` message to client

### 2.5.8 WebSocket Heartbeat Tuning
- [x] Server tracks `_last_activity_time` on every received message
- [x] Configurable via `WebSocketConfig` (ping_interval_seconds, missed_pings_threshold) and env vars
- [x] Auto-close connections with no activity for `interval * threshold` seconds (default 90s)

### 2.5.9 Request Timing Middleware
- [x] FastAPI middleware logs requests exceeding 5s (configurable via `SLOW_REQUEST_THRESHOLD`)
- [x] `X-Request-Duration-Ms` header on all HTTP responses (except /health and /ws)
- [x] `/health` endpoint includes `translator_metrics` (avg latency, request counts, failures)

### 2.5.10 Session State Resume on Restart
- [x] On startup, restore from `data/last_session.json` (topic, context, speakers, metrics)
- [x] Restore `topic_summary` and `recent_exchanges` so the LLM retains context across restarts
- [x] Restore speaker names/roles so returning speakers keep their labels

---

## Phase 3: Data Integrity & Backend Polish

Fix backend gaps that cause silent data loss or incorrect behavior.

### 3.1 Enhanced SRS Algorithm (from ConjugationGameKMM)

> **Reference:** `C:\Users\clint\AndroidStudioProjects\ConjugationGameKMM` — `SrsAlgorithm`, `SrsModels`, `SrsScheduler`

- [ ] Cap ease factor upper bound at 5.0 (currently unbounded — can grow infinitely)
- [ ] Add lapse tracking — count how many times a card was forgotten after being learned (ConjugationGameKMM tracks this)
- [ ] Add mature card detection — cards with interval > N days get different handling
- [ ] Smart session planning from `SrsScheduler`: compose review sessions as 70% due, 20% new, 10% struggling
- [ ] Add `GET /api/vocab/review-session` endpoint that returns a planned session mix (not just raw due cards)
- [ ] Handle timezone for users in Spain (next_review dates should respect user's local time or store UTC with clear documentation)

### 3.2 Missing Vocab API Endpoints
- [x] `POST /api/vocab/` — Manually create vocab item (with duplicate detection + encounter counting)
- [x] Text selection save — select Spanish text in exchange cards, floating "Save to Vocab" button
- [ ] `GET /api/vocab/{id}` — Get individual vocab item by ID
- [ ] `PATCH /api/vocab/{id}` — Edit existing vocab item (term, meaning, notes)

### 3.3 Database Safety
- [ ] Add `PRAGMA busy_timeout=30000` for lock contention handling
- [ ] Add `PRAGMA integrity_check` on startup (log warning if corruption detected)
- [ ] Ensure FTS triggers exist before first vocab insert

### 3.4 Dead Code Cleanup
- [x] Remove unused `import subprocess` from `orchestrator.py`
- [ ] Remove or document unused `decode_chunk()` method in `vad_buffer.py`
- [ ] Fix topic summary prompt condition — orchestrator initializes `topic_summary = ""` but prompt checks for `None`

### 3.5 Pipeline Error Reporting
- [ ] Send error callback to client when `_process_audio_segment()` fails silently
- [ ] Send error callback when partial transcription fails (currently swallowed in orchestrator)
- [ ] Log and notify when ffmpeg subprocess crashes

### 3.6 Translation Quality Metrics (from GeneralizedServiceChatbot RAGMetrics)

> **Reference:** `C:\Users\clint\GeneralizedServiceChatbot` — `RAGMetrics` quality tracking pattern

- [ ] Track LLM confidence scores over time (already returned in `TranslationResult.confidence`)
- [ ] Log low-confidence translations (< 0.3) with source text for review
- [ ] Track correction frequency per speaker (how often the LLM corrects ASR errors)
- [ ] Track idiom detection rate (pattern DB hits vs LLM-only detections)
- [ ] Add `GET /api/system/metrics` endpoint — translation stats, confidence distribution, correction rates
- [ ] Show quality summary in settings panel or system status

---

## Phase 4: DevOps & Deployment

Get the project production-ready for self-hosted use over Tailscale.

### 4.1 Git Hygiene
- [ ] Create `.gitignore` (`.pyc`, `__pycache__`, `*.db*`, `.env`, `venv/`, `.vscode/`, `*.log`, `data/audio/`, `.DS_Store`)
- [ ] Remove already-tracked files that should be ignored (`habla.db`, `__pycache__/`)

### 4.2 Dependency Pinning
- [ ] Pin exact versions in `requirements.txt` (run `pip freeze` and lock current working versions)
- [ ] Consider adding `requirements-dev.txt` for test/lint dependencies

### 4.3 Environment Configuration
- [ ] Create `.env.example` with all env vars (required and optional) and defaults
- [x] Validate `HF_TOKEN` at startup and log clear error if missing
- [x] Make `db_path` and `data_dir` configurable via env vars
- [x] Add `LOG_LEVEL` env var support

### 4.4 Docker Hardening
- [x] Add non-root user to Dockerfile
- [x] Add `HEALTHCHECK` directive (curl `/health`)
- [x] Add `restart: unless-stopped` to docker-compose services
- [x] Add resource limits (memory) to docker-compose
- [x] Pin Ollama image version (was `latest`, now `0.5`)
- [ ] Use Docker secrets or env_file for `HF_TOKEN` (not inline env var)
- [x] Add `depends_on` health check condition so Habla waits for Ollama readiness

### 4.5 Health & Monitoring
- [x] Add dedicated `GET /health` endpoint (checks Ollama, ffmpeg, DB, WhisperX, diarization)
- [x] Add structured logging with rotating file handlers (10MB x 5 + separate error log)
- [ ] Add request timing middleware (log slow requests > 5s)

### 4.6 HTTPS Setup
- [ ] Document Tailscale HTTPS setup for secure mic access on Android
- [ ] OR add nginx/Caddy reverse proxy config for TLS termination
- [ ] Verify WebSocket upgrade works through HTTPS proxy

### 4.7 Database Backup
- [ ] Create backup script (SQLite `.dump` → gzip, daily cron)
- [ ] Document restore procedure
- [ ] Add backup path to `.env.example`

---

## Phase 5: Testing

Build confidence that changes don't break the pipeline.

### 5.1 Unit Tests
- [ ] Idiom scanner — pattern matching, dedup, edge cases (empty input, overlapping matches)
- [ ] Speaker tracker — auto-labeling, rename, role hints
- [ ] Vocab service — CRUD, SM-2 calculations, Anki export format
- [ ] Translator — prompt building, JSON response parsing, timeout handling
- [ ] VAD buffer — speech detection, segment boundaries, energy fallback
- [ ] Config — env var overrides, defaults, validation

### 5.2 Integration Tests
- [ ] WebSocket connection lifecycle (connect, send audio, receive translation, disconnect)
- [ ] REST API endpoints (vocab CRUD, system status, direction toggle)
- [ ] Database operations (init, insert, query, FTS search)
- [ ] Full pipeline: text input → translation → exchange persisted → vocab saved

### 5.3 Test Infrastructure
- [ ] Add `pytest` and `pytest-asyncio` to dev requirements
- [ ] Create test fixtures (mock Ollama responses, sample audio, pre-seeded DB)
- [ ] Add `pytest.ini` or `pyproject.toml` test configuration

---

## Phase 6: Accessibility & Polish

Improve usability for all users.

### 6.1 Accessibility (WCAG 2.1 Level A)
- [ ] Add `aria-label` to all icon/emoji buttons (vocab, save, dismiss, direction toggle)
- [ ] Add `role="dialog"` and `aria-modal="true"` to speaker rename modal
- [ ] Add `aria-live="polite"` to transcript region for screen reader announcements
- [ ] Add text labels alongside status dot (Connected / Disconnected / Connecting)
- [ ] Add proper `<label>` elements for text input
- [ ] Add keyboard focus indicators (`:focus-visible` styles)

### 6.2 Client UX Polish
- [ ] Add landscape orientation handling (prevent overlap of controls and transcript)
- [ ] Add confirmation dialog before deleting vocab items
- [ ] Show connection quality (latency) not just connected/disconnected
- [ ] Add "scroll to bottom" button when user scrolls up in transcript
- [ ] Add long-press/hold on exchange card to copy translation text
- [ ] Handle session restore on page refresh (re-fetch current speakers, direction, mode via `/api/system/status`)

---

## Phase 7: Future Features (from CLAUDE.md)

Bigger features that extend the platform. Not required for "fully functional" but documented in project vision.

### 7.1 Piper TTS (Text-to-Speech)
- [ ] Wire `tts_enabled` config flag to pipeline
- [ ] Integrate Piper TTS (CPU-based, no VRAM impact)
- [ ] Add TTS toggle button in client
- [ ] Stream synthesized audio back to client via WebSocket binary frames
- [ ] Add voice selection (Spanish/English voices)

### 7.2 Audio Clip Storage
- [ ] Wire `save_audio_clips` config flag
- [ ] Save PCM segments to `data/audio/` with exchange ID reference
- [ ] Populate `audio_path` column in exchanges table
- [ ] Add playback button on exchange cards in client

### 7.3 Conversation History UI
- [ ] Build session browser (list past sessions with date, duration, exchange count)
- [ ] Session replay view (scroll through past exchanges with speaker attribution)
- [ ] Search across session history

### 7.4 Native Android Client (Kotlin/KMM)
- [ ] Background audio capture (works with screen off)
- [ ] Persistent WebSocket through Android lifecycle
- [ ] Notification integration for incoming translations
- [ ] Offline vocab review from synced local DB

### 7.5 Spanish Grammar Rule Engine (from ConjugationGameKMM)

> **Reference:** `C:\Users\clint\AndroidStudioProjects\ConjugationGameKMM` — conjugation rule engine with irregular verb handling

- [ ] Port Spanish conjugation rules to Python for server-side use
- [ ] Use rule engine in classroom mode to validate LLM grammar corrections against known patterns
- [ ] Generate targeted grammar drill suggestions based on errors detected in live conversation

### 7.6 Vocab Review Streaks & Motivation (from StoryTimeLanguageKMM)

> **Reference:** `C:\Users\clint\AndroidStudioProjects\StoryTimeLanguageKMM` — `DailyLoginManager`, streak tracking

- [ ] Track daily vocab review activity (date + cards reviewed count) in DB
- [ ] Calculate current streak (consecutive days with at least 1 review)
- [ ] Show streak counter on vocab review page
- [ ] Add `GET /api/vocab/streak` endpoint — current streak, longest streak, total review days
- [ ] Optional: daily review goal (configurable, e.g., "review 10 cards/day")

### 7.7 Conversation Export / Study Review (from GeneralizedServiceChatbot)

> **Reference:** `C:\Users\clint\GeneralizedServiceChatbot` — `ConversationLogger`, `ContentFormatter`

- [ ] Structured export of session exchanges (JSON, CSV) with speaker labels and corrections
- [ ] "Study summary" generation — LLM-produced session recap highlighting key vocab, corrections, and idioms encountered
- [ ] Export format compatible with Anki bulk import (extend existing TSV export to include exchange context)

---

## Phase 8: Multi-Client / Production Scaling

Not needed for personal/local use. Document of what would need to change to support multiple simultaneous users.

### 8.1 Per-Client Pipeline Isolation

Currently the server has a **single global `PipelineOrchestrator`** shared by all WebSocket clients. This causes state corruption when >1 client is connected.

**Shared state that must become per-client:**

| State | Current Location | Problem |
|-------|-----------------|---------|
| `direction` (es_to_en / en_to_es) | `orchestrator.direction` | Client A toggling direction flips Client B |
| `mode` (conversation / classroom) | `orchestrator.mode` | Same — one client's mode change affects all |
| `recent_exchanges` (context deque) | `orchestrator.recent_exchanges` | Conversation context from all clients bleeds together |
| `topic_summary` | `orchestrator.topic_summary` | Topic summary is pooled across unrelated conversations |
| `speaker_tracker` | `orchestrator.speaker_tracker` | Speaker labels (A, B, C) collide between clients |
| `session_id` | `orchestrator._current_session_id` | All clients log to one DB session |
| Pipeline callbacks | `orchestrator._on_translation`, etc. | **Last client to connect overwrites callbacks** — earlier clients stop receiving updates |

**Fix approach:** Move per-conversation state into a `ConversationContext` object, one per WebSocket client. The orchestrator keeps shared resources (ML models, translator, idiom scanner) but each client gets its own context, callbacks, and DB session.

- [ ] Create `ConversationContext` dataclass (direction, mode, speaker_tracker, recent_exchanges, topic_summary, session_id, callbacks)
- [ ] Refactor `PipelineOrchestrator` to accept context per-call instead of storing it as instance state
- [ ] Create one `ConversationContext` per `ClientSession` in websocket.py
- [ ] Fix `set_callbacks()` to use per-client callback registry instead of overwriting
- [ ] Create separate DB session per WebSocket connection
- [ ] Update API routes (`/api/system/direction`, `/api/system/mode`) to either be per-client or explicitly "server-wide default"

### 8.2 Processing Queue Improvements

Currently all clients share a **single asyncio.Queue(maxsize=5)** with **one sequential worker**. With N clients, each waits ~5-7s * N for their translation.

- [ ] Add timeout to `await self._queue.put()` (currently blocks forever if queue full)
- [ ] Consider per-client queues or priority queuing (active speaker gets priority)
- [ ] Add queue depth monitoring / backpressure signal to client ("server busy, please wait")
- [ ] Evaluate concurrent workers (limited by GPU — WhisperX and Ollama can only handle so many parallel requests on a 12GB RTX 3060)

### 8.3 Resource Contention

ML models are shared singletons and have their own concurrency limits:

| Resource | Concurrency | Bottleneck |
|----------|-------------|------------|
| WhisperX ASR | 1 (GPU, runs in thread) | ~2-3s per segment, serialized |
| Pyannote diarization | 1 (CPU, runs in thread) | ~1-2s per segment, serialized |
| Ollama LLM | ~2-3 concurrent (depends on model/VRAM) | Ollama handles its own queue internally |
| OpenAI API | High (cloud, rate-limited by tier) | Not a bottleneck unless hitting rate limits |
| `_partial_lock` | 1 (all clients' streaming partials serialize) | Latency degrades with more clients |

- [ ] Measure actual throughput: how many segments/minute can the pipeline sustain?
- [ ] Add asyncio.Semaphore for GPU-bound operations if moving to concurrent workers
- [ ] Consider separating ASR and translation into independent queues (ASR is the bottleneck, translation can run in parallel)
- [ ] Profile VRAM under concurrent WhisperX loads to find safe concurrency limit

### 8.4 Horizontal Scaling (Future)

If single-server capacity is exceeded:

- [ ] Extract ML models into separate microservices (WhisperX service, translation service)
- [ ] Add Redis or similar for inter-process job queuing (reference: StoryTime's Bull queue pattern at `C:\Users\clint\story-server\queues\storyQueue.js`)
- [ ] Sticky sessions or session affinity for WebSocket connections behind a load balancer
- [ ] Shared state store (Redis) for cross-instance session data
- [ ] GPU pool management for WhisperX across multiple GPUs/machines

---

## Status Key

- `[ ]` — Not started
- `[x]` — Complete
- `[-]` — Skipped / not applicable
