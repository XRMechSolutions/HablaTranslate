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

- The owner accesses the server remotely from his Android phone over Tailscale VPN, often from Spain
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

## Design Influences

Patterns ported from other projects into Habla:
- **Server resilience** — Circuit breakers, retry with exponential backoff, graceful shutdown with state save, startup health checks, structured logging, rate limiting (from a Python async trading system)
- **Multi-provider LLM** — Provider abstraction, LM Studio management, model discovery, health checking, settings persistence (from a Kotlin/C#/Node.js chatbot)
- **Production ops** — Database migrations, request correlation IDs, scheduled background tasks (from a Node.js server)
- **PWA client** — Multi-language support, WebSocket patterns, mobile-first design (from a React/TypeScript app)

## Future Directions (from design discussions)

- **Native Android/Kotlin app** — The owner has KMM experience. A native client would enable background audio capture, persistent WebSocket connections through screen-off, and notification integration.
- **Rust audio processing library** — Potential shared library for audio capture/encode/VAD that compiles for both Android NDK and Linux server.
- **Piper TTS** — CPU-based text-to-speech to optionally speak translations aloud. Scaffolded in config but not yet wired up.
- **Vocab review page** — `vocab.html` route exists but the page needs building.
