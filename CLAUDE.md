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

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete file tree, line counts, module dependency graph, and technology table. The summary below covers the essentials.

### Pipeline Flow
```
Browser Audio (Opus/WebM) → WebSocket → AudioDecoder (ffmpeg) → StreamingVADBuffer (Silero VAD)
  → WhisperX ASR → Pyannote Diarization → Ollama LLM Translation → WebSocket → Browser
```

### Server (`habla/server/`) — Key Modules
- **`main.py`** — FastAPI app, lifespan startup/shutdown, model loading, static mount
- **`config.py`** — Pydantic settings, env overrides (`HF_TOKEN`, `OLLAMA_*`, `WHISPER_*`)
- **`pipeline/`** — `orchestrator.py` (central coordinator, queue, context), `translator.py` (LLM client), `vad_buffer.py` (VAD + ffmpeg decoder)
- **`routes/`** — `websocket.py` (audio/control WS), `api.py` (base router), plus `api_vocab.py`, `api_idioms.py`, `api_llm.py`, `api_playback.py`, `api_sessions.py`, `api_system.py`
- **`services/`** — `health.py`, `idiom_scanner.py`, `speaker_tracker.py`, `vocab.py`, `audio_recorder.py`, `playback.py`, `lmstudio_manager.py`
- **`models/`** — `schemas.py` (Pydantic models), `prompts.py` (LLM templates)
- **`db/`** — `database.py` (async SQLite, WAL, FTS5)

### Client (`habla/client/`)
- **Pages**: `index.html` (translator), `vocab.html` (study), `history.html` (sessions)
- **JS modules**: `app.js`, `websocket.js`, `audio.js`, `core.js`, `settings.js`, `ui.js`
- **PWA**: `manifest.json`, `sw.js`, `styles.css`

### Data & Tooling
- **`data/`** — `idioms/spain.json`, `audio/recordings/`, `habla.db`, `last_session.json`
- **`scripts/`** — `auto_tune_parameters.py`, `compare_wer.py`
- **`tools/`** — `generate_ground_truth.py`, `test_recording.py`
- **`tests/`** — ~639 tests across `pipeline/`, `routes/`, `services/`, `db/`, `benchmark/`

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

## Context Efficiency and Session Continuity

To reduce token waste and preserve continuity across sessions:

- **Context budget first**: Start with the smallest relevant file set, then expand only through direct dependencies/callers/imports.
- **Session memory file**: Maintain `.dev/docs/SESSION_MEMORY.md` with:
  - Confirmed architecture facts
  - Open questions
  - Known pitfalls/regressions
  - `file + last_verified` markers for each item
- **Closeout protocol**: At the end of substantial work, append what changed, what was learned, and what should be rechecked next session.
- **Staleness rule**: If a memory item references files that changed, treat it as stale until revalidated.
- **Thrash guard**: If the same file is edited more than 3 times in one task, pause and restate the plan before continuing.
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

