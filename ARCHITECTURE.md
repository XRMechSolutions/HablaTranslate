# Habla — Architecture

Self-hosted real-time bidirectional speech translator with speaker diarization, idiom detection, and vocabulary capture for language learning. Built to replace Google Translate for live conversation and classroom environments.

## What Makes This Different

Google Translate's pipeline is mechanical: ASR transcribes what it hears, then a separate MT model translates the transcript literally. When the ASR mishears something — which happens constantly with quiet speech, noisy classrooms, mumbling, or accented speakers — the MT model faithfully translates the garbage. Idioms get translated word-for-word. Source language leaks through untranslated. The result is useless for real conversation.

Habla puts an LLM in the middle. The LLM sees the last 10 exchanges of conversation context plus a rolling topic summary. It uses that context to fix ASR errors (inferring what was *meant*, not what was *heard*), translate idioms by meaning, detect grammar corrections in classroom settings, and flag vocabulary worth studying. The LLM is explicitly instructed to never pass source language through untranslated.

## System Overview

```
Phone (Chrome PWA)                    MSI Laptop (RTX 3060, 12GB VRAM)
┌──────────────┐     Tailscale VPN    ┌─────────────────────────────────┐
│ Mic → Opus   │ ──── WebSocket ────→ │ ffmpeg → Silero VAD → WhisperX │
│              │                      │   → Pyannote → LLM → DB        │
│ UI ← JSON   │ ←─── WebSocket ────  │                                 │
└──────────────┘                      └─────────────────────────────────┘
     ~20MB/hr                              ~5GB VRAM steady state
```

The owner accesses the server from an Android phone over Tailscale. The PWA client captures Opus audio and streams it over a single WebSocket. The server decodes, segments, transcribes, diarizes, translates, and returns polished results as JSON. All ML models stay loaded — no per-request loading.

## Pipeline Flow

```
Browser Audio (Opus/WebM) → WebSocket → AudioDecoder (ffmpeg)
  → StreamingVADBuffer (Silero VAD) → WhisperX ASR → Pyannote Diarization
  → LLM Translation (Ollama / LM Studio / OpenAI / Claude) → WebSocket → Browser
```

**Two-tier streaming**: During speech, partial ASR + quick translation update every ~1s (rough but instant). When speech ends, the full pipeline runs and the partial snaps to a polished final card with speaker attribution, idiom highlights, and correction detection.

## Module Index

Each module has a detailed architecture doc covering current state, design rationale, and planned direction.

### Core Pipeline

| Module | Source | Description | Architecture |
|--------|--------|-------------|--------------|
| **Pipeline Overview** | `pipeline/orchestrator.py`, `pipeline/audio_processor.py`, `pipeline/context_manager.py` | Thin coordinator pattern — queue-based processing with concurrent ASR and translation | [pipeline-overview.md](docs/architecture/pipeline-overview.md) |
| **Audio Processing** | `pipeline/vad_buffer.py`, `pipeline/audio_processor.py` | Silero VAD segmentation, ffmpeg decode, WhisperX ASR, Pyannote diarization, audio normalization | [audio-processing.md](docs/architecture/audio-processing.md) |
| **Translation** | `pipeline/translator.py`, `models/prompts.py` | Multi-provider LLM client with fallback chains, cost tracking, retry, rate limiting, and prompt engineering | [translation.md](docs/architecture/translation.md) |
| **Annotation** | `services/idiom_scanner.py`, `services/speaker_tracker.py`, `pipeline/context_manager.py` | Dual-layer idiom detection, speaker diarization/tracking, correction detection, context management | [annotation.md](docs/architecture/annotation.md) |

### Communication

| Module | Source | Description | Architecture |
|--------|--------|-------------|--------------|
| **WebSocket** | `routes/websocket.py` | Real-time protocol — binary audio frames, JSON control messages, ClientSession lifecycle | [websocket.md](docs/architecture/websocket.md) |
| **REST API** | `routes/api_*.py` | Domain-split REST endpoints for vocab, sessions, LLM config, system control, corrections, bookmarks | [api.md](docs/architecture/api.md) |

### Services

| Module | Source | Description | Architecture |
|--------|--------|-------------|--------------|
| **Core Services** | `services/health.py`, `services/vocab.py` | Health monitoring, vocab CRUD with SM-2 spaced repetition and FTS5 search | [services.md](docs/architecture/services.md) |
| **LM Studio** | `services/lmstudio_manager.py` | Windows process management, model discovery, background health monitoring, settings persistence | [lmstudio.md](docs/architecture/lmstudio.md) |
| **Text-to-Speech** | `services/tts.py` | Piper TTS — CPU-based, lazy-loaded, opt-in. Zero VRAM cost | [tts.md](docs/architecture/tts.md) |
| **Audio Storage** | `services/audio_recorder.py`, `services/playback.py` | Debug recording, clip storage for exchange playback, recording replay through pipeline | [audio-storage.md](docs/architecture/audio-storage.md) |

### Data & Infrastructure

| Module | Source | Description | Architecture |
|--------|--------|-------------|--------------|
| **Database** | `db/database.py`, `models/schemas.py` | Async SQLite with WAL mode, FTS5, 6 tables, Pydantic models | [database.md](docs/architecture/database.md) |
| **Client** | `client/js/*.js`, `client/*.html` | Mobile-first PWA — vanilla JS, dark theme, audio capture, offline audio spooling | [client.md](docs/architecture/client.md) |
| **Deployment** | `Dockerfile`, `docker-compose.yml`, `config.py` | Docker, Tailscale VPN, VRAM budgeting, hierarchical config with env var overrides | [deployment.md](docs/architecture/deployment.md) |

## Key Technologies

| Component | Technology | Runs On | VRAM |
|-----------|-----------|---------|------|
| ASR | WhisperX Small | GPU | ~1GB |
| Translation | Qwen3 4B Q3_K_M via Ollama | GPU | ~2.5GB |
| Diarization | Pyannote 3.1 | CPU | 0 |
| VAD | Silero VAD | CPU | 0 |
| Idiom detection | Regex pattern DB + LLM | CPU | 0 |
| TTS | Piper (ONNX voices) | CPU | 0 |
| Server | FastAPI + uvicorn | CPU | — |
| Database | SQLite (aiosqlite, WAL, FTS5) | Disk | — |
| Audio decode | ffmpeg (subprocess) | CPU | — |
| Client | HTML5 PWA, vanilla JS | Browser | — |
| Audio codec | Opus/WebM (MediaRecorder) | Browser | — |
| **Total steady state** | | | **~5GB** |

## Module Dependency Graph

```
main.py
  ├── config.py
  ├── db/database.py
  ├── services/health.py
  ├── pipeline/orchestrator.py
  │     ├── pipeline/audio_processor.py ──→ WhisperX (GPU), Pyannote (CPU), ffmpeg
  │     ├── pipeline/context_manager.py
  │     │     ├── pipeline/translator.py ──→ Ollama / LM Studio / OpenAI / Claude
  │     │     ├── services/idiom_scanner.py ──→ data/idioms/*.json
  │     │     └── services/speaker_tracker.py
  │     └── pipeline/vad_buffer.py ──→ ffmpeg, Silero VAD
  ├── routes/websocket.py ──→ orchestrator, vad_buffer, services/tts.py
  ├── routes/api.py
  │     ├── api_vocab.py ──→ services/vocab.py ──→ db/database.py
  │     ├── api_idioms.py ──→ services/idiom_scanner.py
  │     ├── api_llm.py ──→ services/lmstudio_manager.py, translator
  │     ├── api_playback.py ──→ services/playback.py
  │     ├── api_sessions.py ──→ db/database.py
  │     ├── api_corrections.py ──→ services/audio_recorder.py
  │     ├── api_bookmarks.py ──→ db/database.py
  │     └── api_system.py
  └── client/ (static mount at /)
```

## Related Documents

| Document | Purpose |
|----------|---------|
| [TASKS.md](TASKS.md) | Implementation roadmap by phase |
| [.dev/docs/SESSION_MEMORY.md](.dev/docs/SESSION_MEMORY.md) | Cross-session architecture facts |
| [.dev/docs/BUG_REGISTRY.md](.dev/docs/BUG_REGISTRY.md) | Bug tracker |
| [.dev/docs/Flow-Maps.md](.dev/docs/Flow-Maps.md) | WebSocket protocol, processing flows, session lifecycle |
| [docs/AUDIO_TUNING_GUIDE.md](docs/AUDIO_TUNING_GUIDE.md) | Audio parameter tuning reference |
