# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Habla is a self-hosted real-time bidirectional Spanish-English speech translator with speaker diarization, idiom detection, and vocabulary capture for language learning. Built to replace Google Translate for live conversation and classroom environments where Google's output is unreliable (passes Spanish through untranslated, butchers idioms, gives up on complex sentences, and badly mishandles quiet speech and noisy backgrounds).

## Hardware Constraints

Target hardware is an MSI laptop with an RTX 3060 (12GB VRAM) that has known thermal and power delivery problems. The system is designed conservatively at ~5GB VRAM steady state (~42% of capacity) to avoid crashes. Do not increase model sizes or add GPU-resident components without careful VRAM budgeting. The large headroom is intentional for stability.

### VRAM Budget
- WhisperX Small: ~1GB GPU (always loaded)
- Qwen3 4B Q3_K_M via Ollama: ~2.5GB GPU (always loaded)
- Pyannote 3.1 diarization: CPU only (zero GPU cost)
- Silero VAD: CPU only
- Idiom pattern DB: CPU only (regex)
- KV cache + overhead: ~1.5GB
- **Total: ~5GB**, leaving ~7GB headroom

## User & Deployment Context

- The owner accesses the server remotely from his Android phone over Tailscale VPN (Tailscale IP: 100.73.7.66), often from Spain
- The PWA client runs in Chrome on Android — must be mobile-first
- Primary use cases: live classroom translation (long sessions, always-on listening) and casual conversation translation
- Bandwidth is ~20MB/hour (Opus-encoded audio upstream, JSON downstream) — works on metered mobile data
- Vocab capture is a core feature, not an afterthought — idioms, false friends, corrections, and interesting constructions get saved for spaced repetition study and Anki export

## Commands

### Run the server (from `habla/` directory)
```bash
cd habla
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

### Run with Docker
```bash
export HF_TOKEN=hf_your_token_here
docker compose up
```

### Install dependencies
```bash
cd habla
pip install -r requirements.txt
```

### Required external services
- **Ollama** must be running with the translation model pulled: `ollama pull qwen3:4b`
- **ffmpeg** must be installed (used for audio decoding)
- **HF_TOKEN** env var required for Pyannote speaker diarization

## Architecture

All code lives under `habla/`. The project is a Python FastAPI server with a static HTML client.

### Pipeline Flow
```
Browser Audio (Opus/WebM) → WebSocket → AudioDecoder (ffmpeg) → StreamingVADBuffer (Silero VAD)
  → WhisperX ASR → Pyannote Diarization → Ollama LLM Translation → WebSocket → Browser
```

### Server Structure (`habla/server/`)

- **`main.py`** — FastAPI app with lifespan startup/shutdown. Loads all models, initializes DB, mounts client static files.
- **`config.py`** — All configuration via Pydantic models. Environment variable overrides: `HF_TOKEN`, `OLLAMA_URL`, `OLLAMA_MODEL`, `WHISPER_MODEL`, `WHISPER_DEVICE`.

#### `pipeline/`
- **`orchestrator.py`** — Central coordinator. Manages the processing queue, runs ASR+diarization in threads, coordinates translation, merges idiom matches (pattern DB + LLM), maintains conversation context (rolling deque of 10 exchanges) and topic summary. Also handles streaming partial transcription during ongoing speech.
- **`translator.py`** — Ollama LLM client (`httpx.AsyncClient`). Sends structured JSON prompts, parses JSON responses. Handles translation, correction detection, and topic summary updates.
- **`vad_buffer.py`** — `StreamingVADBuffer`: Silero VAD-based speech segmenter that detects utterance boundaries in continuous audio. `AudioDecoder`: ffmpeg subprocess wrapper for WebM/Opus to PCM conversion. Emits partial audio snapshots every ~1s for streaming transcription.

#### `routes/`
- **`websocket.py`** — WebSocket handler with `ClientSession` per connection. Supports continuous listening (VAD-based auto-segmentation) and push-to-talk. Handles binary audio frames and JSON control messages (`start_listening`, `stop_listening`, `text_input`, `toggle_direction`, `set_mode`, `rename_speaker`).
- **`api.py`** — REST endpoints under `/api/vocab/` (CRUD, spaced repetition review, Anki TSV export, search via FTS5) and `/api/system/` (status, direction/mode control, speaker rename).

#### `services/`
- **`idiom_scanner.py`** — Regex-based idiom detector. Loads patterns from JSON files in `data/idioms/`. Runs in <10ms. Matches are merged with LLM-detected idioms (pattern DB takes priority for dedup).
- **`speaker_tracker.py`** — In-memory speaker identity tracker. Auto-labels speakers (A, B, C...), supports custom naming and LLM-inferred role hints.
- **`vocab.py`** — Vocabulary CRUD with SM-2 spaced repetition algorithm. Tracks encounter counts, supports Anki export.

#### `models/`
- **`schemas.py`** — All Pydantic models: `SpeakerProfile`, `FlaggedPhrase`, `TranslationResult`, `Exchange`, `VocabItem`, and WebSocket message types (`WSTranslation`, `WSPartialTranscript`, `WSSpeakersUpdate`, `WSStatus`).
- **`prompts.py`** — LLM prompt templates. Translation prompt requests JSON output with corrected text, translation, flagged phrases, confidence, and speaker hints. Classroom mode adds grammar correction detection.

#### `db/`
- **`database.py`** — Async SQLite via `aiosqlite` with WAL mode. Tables: `sessions`, `speakers`, `exchanges`, `vocab`, `idiom_patterns`. FTS5 virtual table `vocab_fts` for full-text vocab search.

### Client (`habla/client/`)
- **`index.html`** — Single-file web client served as static files at `/`. Connects via WebSocket at `/ws/translate`.

### Data (`habla/data/`)
- **`idioms/`** — JSON files of regex idiom patterns (e.g., `spain.json`). Loaded at startup by the orchestrator.
- **`habla.db`** — SQLite database (created automatically).

## Key Design Decisions

- **All ML models stay loaded** — WhisperX on GPU, Pyannote on CPU, Ollama handles its own model. No model loading per-request. No sequential load/unload — all models coexist in VRAM simultaneously for instant response.
- **LLM-in-the-middle is the core differentiator** — Unlike Google Translate's mechanical ASR-to-MT pipeline, the LLM sees conversation context (last 10 exchanges + topic summary) and uses reasoning to fix ASR errors and handle idioms by meaning rather than literally. This is critical for real-world audio: quiet speech, noisy backgrounds (classrooms, restaurants, street), and natural mumbling produce misheard words that a dumb translator passes through as garbage. The LLM corrects homophones, dropped words, and garbled phrases using conversational context and topic awareness — inferring what was *meant* rather than transcribing what was *heard*. The LLM is explicitly instructed to never pass source language through untranslated.
- **Two-tier streaming display** — During speech, partial ASR + quick translation update every ~1s on the client (rough but instant). When speech ends, the full pipeline runs and the partial snaps to a polished final card with speaker attribution, idiom highlights, and correction detection. This mimics Google Translate's live feel while delivering much better final results.
- **Dual idiom detection** — Fast regex pattern DB (<10ms, CPU) catches known idioms before the LLM even runs. LLM catches novel/context-dependent phrases the DB missed. Pattern DB takes priority during dedup. When users save an LLM-detected idiom, it should eventually feed back into the pattern DB.
- **Classroom mode** — More aggressive vocab flagging. Detects when one speaker corrects another's grammar (e.g., teacher correcting a student) and creates structured correction cards with wrong form, right form, and explanation. This is a primary use case.
- **Single WebSocket per client** — Binary frames for Opus/WebM audio, text frames for JSON control/responses.
- **Blocking GPU ops run in threads** — ASR and diarization use `asyncio.to_thread()` to avoid blocking the event loop.
- **Text-only fallback** — If WhisperX fails to load, the server operates in text-input-only mode via the `process_text` method.
- **Two directions** — `es_to_en` and `en_to_es`, togglable at runtime.

## Task List

See **[TASKS.md](TASKS.md)** for the full implementation roadmap from current state to project completion, organized by phase and priority.

## Audit and Testing Standards

Before auditing, reviewing, or creating tests for any module, **read the relevant standards document first**:

- **Code audits/reviews**: Read **[.dev/docs/Audit-Checklist.md](.dev/docs/Audit-Checklist.md)** before beginning any audit. Use the checklist categories and report PASS/ISSUE/N/A for each applicable item.
- **Writing or reviewing tests**: Read **[.dev/docs/Testing-Standards.md](.dev/docs/Testing-Standards.md)** before writing new tests or auditing existing test coverage. Follow the naming conventions, assertion quality standards, and anti-trivial-pass guidelines.

These documents define the project's quality bar. Do not skip them.

## Reusable Code from Other Projects

Reference paths for patterns that can be ported into Habla. Do not copy blindly — adapt to Python/FastAPI conventions.

### StockMaster — `C:\Clint file drop\StockMaster`
Server resilience and operational patterns (Python, async).
- **Circuit breakers**: `ExecutionEngine/core/circuit_breaker_manager.py` — 7 breaker types, auto-reset, DB persistence
- **Retry with backoff**: `test/test_order_retry.py` — exponential backoff config, retryable vs permanent errors
- **Graceful shutdown + state save**: `core/main.py:397-695` — SIGINT/SIGTERM handlers, queue drain, final state JSON
- **Startup health checks**: `core/main.py:468-485` — component validation before accepting work
- **Structured logging**: `DataGatheringTools/monitoring/logger_config.py` — JSON structured logs, rotating files, metrics collection, separate error log
- **System monitoring**: `control_center/services/system_monitor.py` — CPU/memory/disk alerts with thresholds
- **Rate limiting**: `AIDecisionMakers/brokers/alpaca_client.py:1174-1183` — token-bucket rate limiter
- **Metrics tracking**: `AIDecisionMakers/brokers/alpaca_client.py:200-337` — orders, API calls, cache hits, uptime
- **Cache with TTL**: `AIDecisionMakers/brokers/alpaca_client.py:841-864` — 30s position cache
- **Docker health checks**: `control_center/Dockerfile` and `docker-compose.yml` — HEALTHCHECK, restart policies, service dependencies
- **Error categorization**: `AIDecisionMakers/brokers/alpaca_client.py:41-55` — TEMPORARY/PERMANENT/CRITICAL error types
- **WebSocket heartbeat**: `control_center/app.py:39-44` — ping_interval/timeout tuning

### GeneralizedServiceChatbot — `C:\Users\clint\GeneralizedServiceChatbot`
Multi-provider LLM integration and LM Studio management (Kotlin + C# + Node.js).
- **LLM provider interface**: `app/src/main/java/.../LlmClient.kt` — abstract interface with `generateResponse()`, `setModel()`
- **LM Studio client**: `app/src/main/java/.../LMStudioClient.kt` — OpenAI-compatible API calls, model discovery via `/v1/models`, health checks, auto-restart
- **LM Studio health checker**: `app/src/main/java/.../LMStudioHealthChecker.kt` — health + model verification, restart requests, model loading
- **LM Studio Monitor Service** (C# Windows service): `LMStudioMonitorService/` — manages LM Studio as child process on port 5000
  - `Controllers/LMStudioController.cs` — process start/stop/restart, model loading via CLI (`lms.exe`) with API fallback
  - `Controllers/HealthCheckController.cs` — `GET /api/health` returns running status + loaded model
  - `Controllers/LMStudioAPIController.cs` — `GET/POST /LMStudioApi/model`, `POST /restart`, `GET /models`
  - `appsettings.json` — executable path, model path, API endpoint, port config
- **Settings persistence**: `app/src/main/java/.../SettingsManager.kt` — provider selection, model, URL storage
- **Provider switching UI**: `app/src/main/res/layout/activity_settings.xml` — spinners for provider/model with loading indicators
- **API format differences**: Ollama uses `/api/generate` + `response` field; LM Studio uses `/v1/chat/completions` + `choices[0].message.content`

### story-server — `C:\Users\clint\story-server`
Production Node.js server patterns.
- Health monitoring with environment validation
- Database migrations (sequential, with partial failure handling)
- Winston structured logging with request correlation IDs
- Scheduled background tasks (backup, cleanup)

### storytimelanguage-web — `C:\Users\clint\storytimelanguage-web`
React/TypeScript PWA client.
- Multi-language support with translation infrastructure
- WebSocket client patterns for real-time server communication
- Vite build configuration, mobile-first design

## Future Directions (from design discussions)

- **Native Android/Kotlin app** — The owner has KMM experience. A native client would enable background audio capture, persistent WebSocket connections through screen-off, and notification integration.
- **Rust audio processing library** — Potential shared library for audio capture/encode/VAD that compiles for both Android NDK and Linux server.
- **Piper TTS** — CPU-based text-to-speech to optionally speak translations aloud. Scaffolded in config but not yet wired up.
- **Vocab review page** — `vocab.html` route exists but the page needs building.
