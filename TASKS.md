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
| [.dev/docs/PLAN-Failed-Segment-Tracking.md](.dev/docs/PLAN-Failed-Segment-Tracking.md) | Not started | Failed segment tracking & auto-tuning feedback loop (7.16) |

## Design References

Patterns drawn from the author's other projects (already ported into Habla where noted):

| Project | Reusable Patterns |
|---------|-------------------|
| **GeneralizedServiceChatbot** (Kotlin/C#/Node.js) | Multi-provider LLM switching (OpenAI/Claude/LM Studio), settings UI, health checking, LM Studio process management |
| **StoryTimeLanguageKMM** (Kotlin Multiplatform) | GPT-5 Nano/Mini Responses API, fallback chains, cost optimization, SSE streaming |
| **StockMaster** (Python async) | Server resilience patterns — already ported to Phase 2.5 (circuit breakers, retry, shutdown, logging) |
| **ConjugationGameKMM** (Kotlin Multiplatform) | Spanish verb conjugation game — potential vocab/drill patterns |
| **story-server** (Node.js) | Database migrations, structured logging, scheduled background tasks |

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

> **Design references:** GeneralizedServiceChatbot (provider switching, LM Studio management, model discovery, health checks) and StoryTimeLanguageKMM (GPT-5 Nano/Mini API, fallback chains, SSE streaming, cost tracking).

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

> **Design reference:** ConjugationGameKMM — `SrsAlgorithm`, `SrsModels`, `SrsScheduler`

- [x] Cap ease factor upper bound at 5.0 (currently unbounded — can grow infinitely)
- [x] Add lapse tracking — count how many times a card was forgotten after being learned (ConjugationGameKMM tracks this)
- [x] Add mature card detection — cards with interval > N days get different handling
- [x] Smart session planning from `SrsScheduler`: compose review sessions as 70% due, 20% new, 10% struggling
- [x] Add `GET /api/vocab/review-session` endpoint that returns a planned session mix (not just raw due cards)
- [x] Handle timezone for users in Spain (all next_review dates stored as UTC ISO strings)

### 3.2 Missing Vocab API Endpoints
- [x] `POST /api/vocab/` — Manually create vocab item (with duplicate detection + encounter counting)
- [x] Text selection save — select Spanish text in exchange cards, floating "Save to Vocab" button
- [x] `GET /api/vocab/{id}` — Get individual vocab item by ID
- [x] `PATCH /api/vocab/{id}` — Edit existing vocab item (term, meaning, notes)

### 3.3 Database Safety
- [x] Add `PRAGMA busy_timeout=30000` for lock contention handling
- [x] Add `PRAGMA integrity_check` on startup (log warning if corruption detected)
- [x] Ensure FTS triggers exist before first vocab insert

### 3.4 Dead Code Cleanup
- [x] Remove unused `import subprocess` from `orchestrator.py`
- [x] Remove or document unused `decode_chunk()` method in `vad_buffer.py`
- [x] Fix topic summary prompt condition — orchestrator initializes `topic_summary = ""` but prompt checks for `None`

### 3.5 Pipeline Error Reporting
- [x] Send error callback to client when `_process_audio_segment()` fails at ASR stage (translation-stage errors already reach client)
- [x] Send error callback when partial transcription fails (currently swallowed in orchestrator)
- [x] Log and notify when ffmpeg subprocess crashes (currently causes silent audio loss)

### 3.6 Translation Quality Metrics (from GeneralizedServiceChatbot RAGMetrics)

> **Design reference:** GeneralizedServiceChatbot — `RAGMetrics` quality tracking pattern

- [x] Track LLM confidence scores over time (already returned in `TranslationResult.confidence`)
- [x] Log low-confidence translations (< 0.3) with source text for review
- [x] Track correction frequency per speaker (how often the LLM corrects ASR errors)
- [x] Track idiom detection rate (pattern DB hits vs LLM-only detections)
- [x] Add `GET /api/system/metrics` endpoint — translation stats, confidence distribution, correction rates
- [x] Show quality summary in settings panel or system status

### 3.7 Dead / Partial Code Cleanup (Audit Feb 2026)

Items found during codebase audit — code that was started but never connected, or scaffolding left behind.

#### Dead Code (remove or decide to implement)

- [x] **`_align_model` in orchestrator** — removed dead field (was never set or used)
- [x] **`save_decoded_pcm` config flag** — removed unused flag + test assertion + doc reference
- [x] **`examples` column in `idiom_patterns` table** — removed from schema (existing DBs unaffected, column just sits unused)

#### Partial Implementations (finish wiring)

- [x] **No corrections page navigation** — added Corrections button to `index.html` nav bar + click handler in `app.js`
- [ ] **LM Studio auto-restart on health failure** — `lmstudio_manager.py` has full `ensure_running()` / restart capability, and `health.py:run_llm_health_monitor()` detects when LM Studio goes down, but the monitor only logs a warning and notifies the client — it never calls `lmstudio_manager.ensure_running()` to attempt recovery. Wire the auto-recovery loop.

#### Cleanup

- [x] **Remove `[RECORDING API]` debug logging** — reduced 7 verbose log lines to single `logger.info("Recording toggled: enabled=...")`

---

## Phase 4: DevOps & Deployment

Get the project production-ready for self-hosted use over Tailscale.

### 4.1 Git Hygiene
- [x] Create `.gitignore` (`.pyc`, `__pycache__`, `*.db*`, `.env`, `venv/`, `.vscode/`, `*.log`, `data/audio/`, `.DS_Store`)
- [x] Remove already-tracked files that should be ignored (`habla.db`, `__pycache__/`) — verified none tracked

### 4.2 Dependency Pinning
- [x] Pin exact versions in `requirements.txt` (run `pip freeze` and lock current working versions)
- [x] Add `requirements-dev.txt` for test dependencies

### 4.3 Environment Configuration
- [x] Create `.env.example` with all env vars (required and optional) and defaults
- [x] Validate `HF_TOKEN` at startup and log clear error if missing
- [x] Make `db_path` and `data_dir` configurable via env vars
- [x] Add `LOG_LEVEL` env var support

### 4.4 Docker Hardening
- [x] Add non-root user to Dockerfile
- [x] Add `HEALTHCHECK` directive (curl `/health`)
- [x] Add `restart: unless-stopped` to docker-compose services
- [x] Add resource limits (memory) to docker-compose
- [x] Pin Ollama image version (was `latest`, now `0.5`)
- [x] Use Docker `env_file` for secrets (HF_TOKEN, OPENAI_API_KEY via `.env`)
- [x] Add `depends_on` health check condition so Habla waits for Ollama readiness

### 4.5 Health & Monitoring
- [x] Add dedicated `GET /health` endpoint (checks Ollama, ffmpeg, DB, WhisperX, diarization)
- [x] Add structured logging with rotating file handlers (10MB x 5 + separate error log)
- [x] Add request timing middleware (log slow requests > 5s)

### 4.6 HTTPS Setup
- [x] Document Tailscale HTTPS setup for secure mic access on Android (`.dev/docs/HTTPS-Setup.md`)
- [x] Document Caddy reverse proxy config as alternative
- [x] Verify WebSocket upgrade works through HTTPS proxy (documented in setup guide)

### 4.7 Database Backup
- [x] Create backup script (`scripts/backup_db.sh` — SQLite `.dump` -> gzip, rotation)
- [x] Document restore procedure (in script comments)
- [x] Add backup path to `.gitignore`

---

## Phase 5: Testing

Build confidence that changes don't break the pipeline.

### 5.1 Unit Tests
- [x] Idiom scanner — pattern matching, dedup, edge cases (23 tests)
- [x] Speaker tracker — auto-labeling, rename, role hints (32 tests)
- [x] Vocab service — CRUD, SM-2 calculations, Anki export format (32 tests)
- [x] Translator — prompt building, JSON response parsing, timeout handling (49 tests)
- [x] VAD buffer — speech detection, segment boundaries, energy fallback (32 tests)
- [x] Config — env var overrides, defaults, validation (40 tests)

### 5.2 Integration Tests
- [x] WebSocket connection lifecycle (connect, send audio, receive translation, disconnect) (107 tests)
- [x] REST API endpoints (vocab CRUD, system status, direction toggle) (76 tests)
- [x] Database operations (init, insert, query, FTS search) (40 tests)
- [x] Full pipeline: text input → translation → exchange persisted → vocab saved (102 tests)

### 5.3 Test Infrastructure
- [x] Add `pytest` and `pytest-asyncio` to dev requirements
- [x] Create test fixtures (mock Ollama responses, sample audio, pre-seeded DB) — conftest.py (301 lines)
- [x] Add `pytest.ini` test configuration (markers: unit, integration, slow, benchmark)

---

## Phase 6: Accessibility & Polish

Improve usability for all users.

### 6.1 Accessibility (WCAG 2.1 Level A)
- [x] Add `aria-label` to all icon/emoji buttons (vocab, save, dismiss, direction toggle)
- [x] Add `role="dialog"` and `aria-modal="true"` to speaker rename modal
- [x] Add `aria-live="polite"` to transcript region for screen reader announcements
- [x] Add text labels alongside status dot (Connected / Disconnected / Connecting)
- [x] Add proper `<label>` elements for text input
- [x] Add keyboard focus indicators (`:focus-visible` styles)

### 6.2 Client UX Polish
- [x] Add landscape orientation handling (prevent overlap of controls and transcript)
- [x] Add confirmation dialog before deleting vocab items
- [x] Show connection quality (latency) not just connected/disconnected
- [x] Add "scroll to bottom" button when user scrolls up in transcript
- [x] Add long-press/hold on exchange card to copy translation text
- [x] Handle session restore on page refresh (re-fetch current speakers, direction, mode via `/api/system/status`)

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
- [x] Wire `save_audio_clips` config flag (saves normalized WAV after ASR preprocessing)
- [x] Save audio clips to `data/audio/clips/{session_id}/` with timestamp-based naming
- [x] Populate `audio_path` column in exchanges table + `has_audio` in API responses
- [x] Add playback button on exchange cards in client + `GET /api/sessions/{id}/exchanges/{id}/audio` endpoint
- [x] Clean up audio clips when session is deleted

### 7.3 Conversation History UI
- [x] Build session browser (list past sessions with date, duration, exchange count)
- [x] Session replay view (scroll through past exchanges with speaker attribution)
- [x] Search across session history (GET /api/sessions/search + search bar in history.html, 3 tests)

### 7.4 Cloud API Key Setup & Model Selection UI

Allow users without a local GPU to use cloud LLM providers (OpenAI, Claude) by entering their own API key. Provide a contextual model picker that shows available models per provider.

#### API Key Management
- [ ] Add settings UI section for entering/removing API keys (OpenAI, Claude/Anthropic)
- [ ] Store API keys in `.env` or browser `localStorage` (client-side only, never sent to third parties)
- [ ] Add `ANTHROPIC_API_KEY` env var and Claude provider support in `translator.py`
- [ ] Mask API keys in UI after entry (show last 4 chars only)
- [ ] Validate API key on entry (test call to provider's models endpoint)

#### Contextual Model Picker
- [ ] When OpenAI is selected, show dropdown with current models: `gpt-5.2`, `gpt-5.2-pro`, `gpt-4o-mini`, `gpt-4o`
- [ ] When Claude is selected, show dropdown with current models: `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-opus-4-6`
- [ ] When Ollama/LM Studio is selected, populate from live `/api/tags` or `/v1/models` query (existing behavior)
- [ ] Show estimated cost per provider/model (cheap indicator for nano/haiku, expensive for pro/opus)
- [ ] Persist model selection per provider in `data/llm_settings.json`

#### Documentation
- [ ] Add "Using Cloud Providers" section to README/SETUP.md explaining how users without a GPU can run Habla with a cloud API key
- [ ] Document cost expectations per provider (e.g., "~$0.01-0.05 per conversation hour with gpt-4o-mini")
- [ ] Note that cloud providers require internet but eliminate the GPU/VRAM requirement

### 7.4.1 Claude as Translation Provider (Backend)

Add Anthropic/Claude as a fourth LLM provider alongside Ollama, LM Studio, and OpenAI.

> **Reference:** Claude API uses `anthropic` Python SDK with `messages` API format.

| Provider | Endpoint | Request Format | Response Format |
|----------|----------|---------------|-----------------|
| **Claude** (cloud) | `POST https://api.anthropic.com/v1/messages` | `messages` array + `system` param | `content[0].text` |

- [ ] Add `anthropic` to `requirements.txt`
- [ ] Add `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` to `config.py` with env var support
- [ ] Add `"claude"` provider option in `translator.py` `_call_llm()` routing
- [ ] Handle Claude response parsing (`content[0].text`)
- [ ] Add Claude to provider probe in `GET /api/llm/providers` (check API key exists, list available models)
- [ ] Add Claude to `GET /api/llm/models?provider=claude`
- [ ] Add Claude to model validation in `POST /api/llm/select`
- [ ] Add Claude cost tracking (input/output token pricing per model)
- [ ] Add Claude to health check in `health.py`
- [ ] Update `.env.example` with `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` entries

### 7.5 Native Android Client (Kotlin/KMM)
- [ ] Background audio capture (works with screen off)
- [ ] Persistent WebSocket through Android lifecycle
- [ ] Notification integration for incoming translations
- [ ] Offline vocab review from synced local DB

### 7.6 Spanish Grammar Rule Engine (from ConjugationGameKMM)

> **Design reference:** ConjugationGameKMM — conjugation rule engine with irregular verb handling

- [ ] Port Spanish conjugation rules to Python for server-side use
- [ ] Use rule engine in classroom mode to validate LLM grammar corrections against known patterns
- [ ] Generate targeted grammar drill suggestions based on errors detected in live conversation

### 7.7 Vocab Review Streaks & Motivation (from StoryTimeLanguageKMM)

> **Design reference:** StoryTimeLanguageKMM — `DailyLoginManager`, streak tracking

- [ ] Track daily vocab review activity (date + cards reviewed count) in DB
- [ ] Calculate current streak (consecutive days with at least 1 review)
- [ ] Show streak counter on vocab review page
- [ ] Add `GET /api/vocab/streak` endpoint — current streak, longest streak, total review days
- [ ] Optional: daily review goal (configurable, e.g., "review 10 cards/day")

### 7.8 Conversation Export / Study Review (from GeneralizedServiceChatbot)

> **Design reference:** GeneralizedServiceChatbot — `ConversationLogger`, `ContentFormatter`

- [ ] Structured export of session exchanges (JSON, CSV) with speaker labels and corrections
- [ ] "Study summary" generation — LLM-produced session recap highlighting key vocab, corrections, and idioms encountered
- [ ] Export format compatible with Anki bulk import (extend existing TSV export to include exchange context)

### 7.9 Multi-Language Support

Expand beyond hardcoded Spanish ⇄ English to support arbitrary language pairs. WhisperX already supports 90+ languages; the LLM handles most translation pairs.

#### Backend — Language Configuration
- [ ] Replace hardcoded `es_to_en` / `en_to_es` direction with configurable `source_lang` + `target_lang` pair
- [ ] Add language registry with display names, ISO codes, and WhisperX language codes (e.g., `{"es": "Spanish", "en": "English", "fr": "French", ...}`)
- [ ] Update `config.py` with `DEFAULT_SOURCE_LANG` and `DEFAULT_TARGET_LANG` env vars
- [ ] Update translation prompts in `prompts.py` to be language-agnostic (parameterize language names)
- [ ] Update orchestrator direction toggle to cycle or select from configured language pairs
- [ ] Verify WhisperX model size supports target languages (some languages need `medium` or `large-v2`)

#### Backend — Idiom & Vocab Adaptation
- [ ] Make idiom pattern DB language-aware (currently `idioms/spain.json` is Spanish-only)
- [ ] Add idiom pattern file structure: `idioms/{lang_code}.json` (e.g., `fr.json`, `de.json`)
- [ ] Update vocab save to tag items with source/target language pair
- [ ] Update SRS review to filter by active language pair or show all

#### Frontend — Language Picker
- [ ] Replace direction toggle button with language pair selector (two dropdowns or a swap-able pair widget)
- [ ] Persist selected language pair in `localStorage` and `data/llm_settings.json`
- [ ] Show language flags or ISO codes on exchange cards instead of hardcoded "ES"/"EN" labels
- [ ] Add `GET /api/system/languages` endpoint returning supported languages

#### API Changes
- [ ] Update `POST /api/system/direction` to accept `source_lang` + `target_lang` instead of `es_to_en`/`en_to_es`
- [ ] Update WebSocket `set_direction` control message format
- [ ] Backward-compatible: map old `es_to_en`/`en_to_es` values to new format during transition

### 7.10 Always-Listening Mode (Hands-Free)

Continuous translation without push-to-talk — critical for classroom and conference use where the user can't hold a button.

- [ ] Add `listening_mode` toggle: `push_to_talk` (current) vs `continuous`
- [ ] In continuous mode, client streams audio constantly; server relies on VAD to segment speech
- [ ] Add silence gap threshold setting (how long a pause before finalizing a segment — default ~1.5s)
- [ ] Add client UI toggle button (mic icon changes to indicate always-listening state)
- [ ] Add visual indicator of VAD activity (pulsing border or waveform showing when speech is detected)
- [ ] Add energy gate / noise floor calibration — let user calibrate for room ambient noise on session start
- [ ] Handle long-running segments gracefully (cap at ~30s, force-finalize and start new segment)
- [ ] Add server-side `max_continuous_segment_duration` config
- [ ] Power management: reduce audio quality/sample rate in continuous mode to save bandwidth on mobile
- [ ] Add WebSocket control message `set_listening_mode` to switch at runtime

### 7.11 Auto Language Detection

Automatically detect which language is being spoken instead of requiring manual direction toggle.

- [ ] WhisperX returns detected language — extract and use it to auto-set translation direction
- [ ] Add `auto_detect` as a third direction option (alongside explicit source→target pairs)
- [ ] When auto-detect is active, determine target language from the detected source (if detected=Spanish, target=English and vice versa)
- [ ] Show detected language badge on each exchange card ("Detected: ES")
- [ ] Handle detection confidence — if WhisperX is uncertain, fall back to manually set direction
- [ ] Handle mixed-language speech gracefully (code-switching is common in bilingual classrooms)
- [ ] Add `ASR_AUTO_LANGUAGE` env var (already scaffolded in config) to enable by default
- [ ] Combine with multi-language support (7.9): auto-detect from the full set of configured languages

### 7.12 Content Import & Study

Paste an article, URL, or text block to get paragraph-by-paragraph translation with vocab/idiom extraction. Turns Habla into a reading comprehension tool.

#### Backend
- [ ] Add `POST /api/translate/text` endpoint — accepts plain text or URL, returns translated paragraphs
- [ ] For URLs: fetch page content, extract article text (strip nav/ads using `readability` or `trafilatura` library)
- [ ] Split text into paragraphs, translate each through the LLM pipeline (reuse `translator._call_llm()`)
- [ ] Run idiom scanner on each paragraph, return detected idioms alongside translation
- [ ] Extract vocab candidates (unfamiliar words, false friends) using LLM or frequency analysis
- [ ] Store imported content as a special session type (`session_type: "import"` vs `"live"`)

#### Frontend
- [ ] Add "Import" tab or button on main page (paste text or enter URL)
- [ ] Display side-by-side original and translated text, paragraph-aligned
- [ ] Highlight detected idioms and vocab candidates inline (clickable to save)
- [ ] Allow saving individual sentences or paragraphs to vocab for study
- [ ] Show in session history as imported content (distinct icon from live sessions)

### 7.13 Pronunciation Practice

Record yourself saying a phrase and get feedback on pronunciation quality. Pairs with vocab SRS and audio clip storage.

- [ ] Add "Practice" button on vocab cards and exchange cards that have saved audio
- [ ] Record user's attempt via mic (reuse existing audio capture pipeline)
- [ ] Compare user recording against original audio clip using audio similarity scoring
- [ ] Scoring approach: use Whisper to transcribe both clips, compare at text level (word error rate) as a baseline
- [ ] Advanced scoring: pitch contour comparison or audio embedding similarity (e.g., `resemblyzer` or `speechbrain`)
- [ ] Show visual feedback: waveform overlay of original vs user attempt
- [ ] Track pronunciation scores over time per vocab item (store in DB alongside SRS data)
- [ ] Add `GET /api/vocab/{id}/pronunciation` endpoint for score history
- [ ] Integrate with SRS: poor pronunciation = lower quality rating, triggering more frequent review

### 7.14 Conversation Bookmarks

Tap to bookmark interesting moments during live translation for quick review later.

- [ ] Add bookmark button on each exchange card (small flag/pin icon)
- [ ] `POST /api/sessions/{id}/exchanges/{eid}/bookmark` — toggle bookmark with optional note
- [ ] Add `is_bookmarked` field to exchanges table and API responses
- [ ] Add `GET /api/sessions/{id}/bookmarks` — return all bookmarked exchanges for a session
- [ ] Add `GET /api/bookmarks/recent` — return recent bookmarks across all sessions
- [ ] Show bookmark filter in session history view (view only bookmarked moments)
- [ ] Add keyboard shortcut or gesture for quick bookmarking during live translation
- [ ] Persist bookmarks to DB immediately (don't lose them if session disconnects)

### 7.15 Session Study Summary

After a session ends, auto-generate an LLM-produced summary of what was learned.

- [ ] On session close (WebSocket disconnect), queue a background summary generation task
- [ ] LLM prompt: given all exchanges from the session, produce a structured study report:
  - Key vocabulary encountered (with frequency)
  - Grammar patterns observed
  - Corrections made (by teacher or LLM)
  - Idioms detected
  - Suggested review topics
- [ ] Store summary in `sessions` table (`study_summary` column)
- [ ] Add `GET /api/sessions/{id}/summary` endpoint
- [ ] Show summary card at the top of session replay view
- [ ] Option to regenerate summary on demand (if user wants a different focus)
- [ ] Make summary generation configurable (auto vs manual, which LLM provider to use)

### 7.16 Failed Segment Tracking & Auto-Tuning Feedback Loop

When VAD detects speech but ASR produces empty or garbage output, the segment is silently dropped. This feature tracks those failures, surfaces them for review, and feeds them into the auto-tuning pipeline so the system learns from its mistakes.

> **Detailed plan:** [PLAN-Failed-Segment-Tracking.md](.dev/docs/PLAN-Failed-Segment-Tracking.md)

#### Partial Implementations to Complete

These items have scaffolding or partial code already in place but are not connected end-to-end.

- [ ] **Wire `quality_metrics` DB table** — schema exists in `database.py:132-148` with the right columns (`confidence`, `audio_rms`, `duration_seconds`, `speaker_id`, `clipped_onset`, `processing_time_ms`, `vad_threshold`, `model_name`) but nothing ever INSERTs into it. Add `status TEXT DEFAULT 'ok'` column + indexes. Write `_record_quality_metric()` method in orchestrator and call it on every segment outcome.
- [ ] **Connect `_is_bad_transcript()` rejection path** — detection logic exists in `orchestrator.py:1083-1096` and correctly identifies garbage ASR, but on rejection (line 903-905) it just logs a WARNING and returns empty. Need to: record to `quality_metrics`, increment metrics, enrich recorder metadata with `asr_status`/`asr_reject_reason`.
- [ ] **Enrich segment metadata on ASR failure** — `audio_recorder.py:add_segment_metadata()` already accepts arbitrary keys and `websocket.py:410-416` already enriches successful segments with `raw_transcript`, `confidence`, `speaker`. But rejected segments never reach this code path. Need a parallel path for failures.
- [ ] **Unify VAD threshold config** — `config.py` has `asr.vad_threshold = 0.35` and `vad_buffer.py:VADConfig` has `speech_threshold = 0.35` independently. `websocket.py:118` instantiates VADConfig with hardcoded values ignoring app config. Consolidate so there's a single source of truth, needed for recording `vad_threshold` in quality_metrics.
- [ ] **Complete `clipped_onset` detection** — `quality_metrics` table has the column, `auto_tune_parameters.py:76-90` has clipped Spanish word detection logic, but nothing in the live pipeline computes or records it. Need to: detect in orchestrator (check if first word matches clipped pattern), set flag in quality_metrics row.
- [ ] **Connect `auto_tune_parameters.py` to quality_metrics** — script currently only reads `metadata.json` files from disk. It should also query the `quality_metrics` DB table to correlate failures with audio conditions (RMS, VAD threshold, duration, speaker).

#### New Implementation Required

These items have no existing code.

- [ ] Compute audio RMS in `vad_buffer.py:_emit_segment()` — `np.sqrt(np.mean(samples**2))` on segment PCM bytes
- [ ] Track average VAD probability per segment — accumulate `_speech_prob_sum` during speech frames in `vad_buffer.py:_is_speech_frame()` (currently computes `prob` at line 128 but discards it), divide by `_speech_frames` at emission
- [ ] Extend VAD callback signature from `on_segment(pcm_bytes, duration)` to `on_segment(pcm_bytes, duration, audio_rms, vad_avg_prob)` — update all callers: `websocket.py:_on_speech_segment()`, test mocks
- [ ] Forward `audio_rms` and `vad_avg_prob` through websocket handler to recorder metadata and orchestrator
- [ ] Add `asr_rejected_count` and `asr_empty_count` to `orchestrator._metrics` dict (currently missing from lines 96-108)
- [ ] Expose `asr_rejected_count`, `asr_empty_count`, and `quality_metrics` status breakdown in `GET /api/system/metrics`
- [ ] Add `GET /api/corrections/failed-segments` endpoint — return recordings with ASR-rejected segments paired with WAV filenames for review
- [ ] Extend `auto_tune_parameters.py` to correlate ASR rejection rate with `audio_rms`, `vad_avg_prob`, `duration_seconds`, and speaker — generate recommendations (low RMS → AGC, borderline VAD → raise threshold, short duration → raise min_speech_ms)
- [ ] Add "Failed Segments" tab/filter to `corrections.html` — play audio, see RMS/VAD stats, manually transcribe or mark as `[noise]`
- [ ] Update `prepare_dataset.py` to filter `[noise]` markers and include corrected formerly-failed segments (highest-value training data)

### 7.17 Whisper Fine-Tuning Data Pipeline

Correction workflow and dataset assembly for LoRA fine-tuning. Built alongside the failed segment tracking (7.16) to create a complete data pipeline from raw classroom audio to training-ready dataset.

> **Reference:** [WHISPER_FINE_TUNING_PLAN.md](docs/WHISPER_FINE_TUNING_PLAN.md)

#### Backend — Correction API
- [x] `GET /api/recordings/{id}/ground-truth` — returns corrected (preferred) or original ground truth
- [x] `PUT /api/recordings/{id}/ground-truth` — saves corrected segments atomically to `corrected_ground_truth.json`
- [x] `GET /api/corrections/stats` — aggregate progress across all recordings

#### Frontend — Transcript Correction UI
- [x] Build `corrections.html` — recording list view with progress bars, segment correction view with audio playback
- [x] Keyboard navigation: Space=play, Enter=confirm+next, arrows=prev/next
- [x] Debounced auto-save on confirm (500ms)

#### Dataset Assembly Script
- [x] `scripts/prepare_dataset.py` — walks recordings, reads corrected/original ground truth, filters, builds 80/10/10 HuggingFace DatasetDict
- [x] CLI flags: `--min-duration`, `--max-duration`, `--min-confidence`, `--seed`
- [x] `requirements-finetune.txt` for offline fine-tuning dependencies (datasets, soundfile)

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
- [ ] Add Redis or similar for inter-process job queuing (reference: StoryTime's Bull queue pattern)
- [ ] Sticky sessions or session affinity for WebSocket connections behind a load balancer
- [ ] Shared state store (Redis) for cross-instance session data
- [ ] GPU pool management for WhisperX across multiple GPUs/machines

---

## Phase 9: Open Source Release & Distribution

Prepare the project for public GitHub visibility, first tagged release, and community engagement.

### 9.1 Public Repo Cleanup
- [x] Create MIT LICENSE file at project root
- [x] Create root README.md (project overview, architecture, quick start, usage)
- [x] Remove CLAUDE.md from git tracking (contains personal paths, kept local via .gitignore)
- [x] Fix service worker cache version mismatch (v31 → v33)
- [x] Update OpenAI model list to current real models
- [x] Sanitize `start-habla.bat` — use placeholder FQDN instead of hardcoded Tailscale hostname
- [x] Sanitize `AGENTS.md` — remove personal paths and Tailscale IP
- [x] Sanitize `TASKS.md` — remove or genericize personal machine paths in reference project table
- [x] Add `.dockerignore` (exclude `.env`, `*.db`, `*.log`, `data/audio/recordings/`, `.git/`, `tests/`)
- [x] Add `CONTRIBUTING.md` — how to report bugs, submit PRs, code style expectations, test requirements
- [x] Review git history for any accidentally committed secrets — found HF token in deleted `start_server_with_recording.sh` (rotated, no longer in tracked files; history cleanup via `git filter-repo` recommended before first public release)

### 9.2 GitHub Releases Workflow
- [ ] Define version numbering scheme (semver: `v0.1.0` for first public release)
- [ ] Create release checklist script that verifies before tagging:
  - No hardcoded personal paths in tracked files
  - Service worker versions in sync
  - All tests passing
  - `.env.example` up to date with all supported env vars
  - No `*.db`, `*.log`, `*.crt`, `*.key` files tracked
- [ ] Create first tagged release (`v0.1.0`) with release notes
- [ ] Create `CHANGELOG.md` to track changes between releases
- [ ] Add release notes template (features, fixes, breaking changes, upgrade instructions)

### 9.3 Community & Funding
- [ ] Set up GitHub Sponsors or Buy Me a Coffee account
- [ ] Add sponsor/tip badge to README.md (replace the TODO comment)
- [ ] Add GitHub issue templates (bug report, feature request)
- [ ] Add GitHub Discussions or link to Discord for community Q&A
- [ ] Create a short demo video or GIF showing the translator in action (for README)

---

## Status Key

- `[ ]` — Not started
- `[x]` — Complete
- `[-]` — Skipped / not applicable
