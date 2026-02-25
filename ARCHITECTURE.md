# Architecture & File Structure

Last verified: 2026-02-25 | ~9,800 lines of application code

## Pipeline Flow

```
Browser Audio (Opus/WebM) --> WebSocket --> AudioDecoder (ffmpeg) --> StreamingVADBuffer (Silero VAD)
  --> WhisperX ASR --> Pyannote Diarization --> Ollama LLM Translation --> WebSocket --> Browser
```

## Directory Tree

```
hablatranslate/
|-- CLAUDE.md                  # AI assistant instructions & project context
|-- ARCHITECTURE.md            # This file
|-- TASKS.md                   # Implementation roadmap by phase
|-- AGENTS.md                  # Multi-agent configuration
|-- InitialConcept.txt         # Original design notes
|
|-- .dev/docs/                 # Developer standards & plans
|   |-- Audit-Checklist.md     # Code audit checklist (PASS/ISSUE/N/A)
|   |-- Testing-Standards.md   # Test quality bar & conventions
|   |-- TMP-FIX-PLAN-*.md     # Temporary fix plans (6 files, lifecycle through performance)
|   +-- Test-Audit-Review-*.md # Test audit results
|
|-- docs/                      # Technical reference docs
|   |-- AUDIO_TUNING_GUIDE.md
|   |-- CLIENT_AUDIO_COMPRESSION.md
|   |-- NOTES_MT_MODELS.md
|   +-- WHISPER_FINE_TUNING_PLAN.md
|
+-- habla/                     # Application root
    |-- README.md              # Quick start guide
    |-- requirements.txt       # Python dependencies
    |-- docker-compose.yml     # Docker deployment config
    |-- Dockerfile             # Container build (if present)
    |
    |-- server/                # Python FastAPI backend
    |   |-- main.py            (284L) App entry, lifespan startup/shutdown, model loading, static mount
    |   |-- config.py          (175L) Pydantic settings, env var overrides (HF_TOKEN, OLLAMA_*, WHISPER_*)
    |   |
    |   |-- pipeline/          # Core processing pipeline
    |   |   |-- orchestrator.py (1025L) Central coordinator: queue, ASR+diarization threads,
    |   |   |                           translation, idiom merge, context (10-exchange deque),
    |   |   |                           topic summary, streaming partial transcription
    |   |   |-- translator.py   (741L) Ollama/LLM client (httpx.AsyncClient), JSON prompt/parse,
    |   |   |                          translation, correction detection, topic summary
    |   |   +-- vad_buffer.py   (367L) StreamingVADBuffer (Silero VAD speech segmenter),
    |   |                              AudioDecoder (ffmpeg WebM/Opus->PCM), ~1s partial snapshots
    |   |
    |   |-- routes/            # API endpoints
    |   |   |-- _state.py       (32L) Shared server state (orchestrator, config refs)
    |   |   |-- api.py          (16L) Base router, mounts sub-routers
    |   |   |-- websocket.py   (615L) WebSocket handler: ClientSession, continuous/PTT modes,
    |   |   |                         binary audio + JSON control (start/stop_listening,
    |   |   |                         text_input, toggle_direction, set_mode, rename_speaker)
    |   |   |-- api_vocab.py   (142L) /api/vocab/ CRUD, SM-2 review, Anki TSV export, FTS5 search
    |   |   |-- api_idioms.py  (144L) /api/idioms/ pattern management
    |   |   |-- api_llm.py     (268L) /api/llm/ provider config (Ollama, LM Studio, OpenAI)
    |   |   |-- api_playback.py(129L) /api/playback/ recording playback
    |   |   |-- api_sessions.py(186L) /api/sessions/ session history
    |   |   +-- api_system.py  (153L) /api/system/ status, direction/mode control, speaker rename
    |   |
    |   |-- services/          # Business logic
    |   |   |-- health.py      (259L) Startup health checks (Ollama, WhisperX, ffmpeg, disk, etc.)
    |   |   |-- idiom_scanner.py(309L) Regex idiom detector, loads from data/idioms/*.json, <10ms
    |   |   |-- speaker_tracker.py(78L) In-memory speaker identity, auto-label A/B/C, custom names
    |   |   |-- vocab.py       (217L) Vocab CRUD + SM-2 spaced repetition, Anki export
    |   |   |-- audio_recorder.py(210L) Audio recording to disk with metadata
    |   |   |-- playback.py    (410L) Recording playback service + WebSocket integration
    |   |   +-- lmstudio_manager.py(430L) LM Studio process management, model discovery, health
    |   |
    |   |-- models/            # Data models & prompts
    |   |   |-- schemas.py     (139L) Pydantic: SpeakerProfile, FlaggedPhrase, TranslationResult,
    |   |   |                         Exchange, VocabItem, WS message types
    |   |   +-- prompts.py     (108L) LLM prompt templates (translation JSON, classroom corrections)
    |   |
    |   +-- db/                # Database
    |       +-- database.py    (199L) Async SQLite (aiosqlite), WAL mode
    |                                 Tables: sessions, speakers, exchanges, vocab, idiom_patterns
    |                                 FTS5: vocab_fts
    |
    |-- client/                # PWA frontend (mobile-first, dark theme)
    |   |-- index.html         (237L) Main translator interface
    |   |-- vocab.html         (495L) Vocabulary review/study page
    |   |-- history.html       (313L) Session history viewer
    |   |-- styles.css         (256L) Shared styles
    |   |-- manifest.json              PWA manifest
    |   |-- sw.js               (60L) Service worker
    |   |-- icon-192.png / icon-512.png
    |   +-- js/
    |       |-- app.js          (89L) App init, module loader
    |       |-- websocket.js   (136L) WebSocket client, reconnect logic
    |       |-- audio.js       (173L) Audio capture (MediaRecorder, Opus) & playback
    |       |-- core.js        (346L) Core state, translation processing, direction toggle
    |       |-- settings.js    (413L) Settings panel, LLM provider config UI
    |       +-- ui.js          (646L) DOM rendering, cards, partials, auto-scroll
    |
    |-- data/                  # Runtime data
    |   |-- idioms/spain.json          Regex idiom patterns (loaded at startup)
    |   |-- audio/recordings/*/        Recorded audio + metadata.json per session
    |   |-- habla.db                   SQLite database (auto-created)
    |   +-- last_session.json          Persisted session state for graceful restart
    |
    |-- scripts/               # Dev/tuning scripts
    |   |-- auto_tune_parameters.py    Parameter auto-tuning
    |   +-- compare_wer.py             Word error rate comparison
    |
    |-- tools/                 # Dev tools
    |   |-- generate_ground_truth.py   Ground truth generation for testing
    |   +-- test_recording.py          Recording test utility
    |
    +-- tests/                 # Test suite (~639 tests)
        |-- conftest.py                Shared fixtures
        |-- README.md                  Test documentation
        |-- TEST_STATUS.md             Current test status
        |-- EDGE_CASES.md             Edge case catalog
        |-- FIXES_SUMMARY.md          Bug fix log
        |
        |-- pipeline/                  orchestrator, translator, vad_buffer tests
        |-- routes/                    API, playback, websocket tests
        |-- services/                  All service tests (health, idiom, speaker, vocab, etc.)
        |-- db/                        Database tests
        |-- test_soak_stability.py     Long-running stability test
        +-- benchmark/                 Performance benchmarks
            |-- results/*.json         Benchmark results (idiom_scanner, speaker_tracker, translator)
            |-- test_audio_pipeline.py
            |-- test_pipeline_performance.py
            +-- audio_samples/         Test audio files
```

## Module Dependency Graph

```
main.py
  |-- config.py
  |-- db/database.py
  |-- services/health.py
  |-- pipeline/orchestrator.py
  |     |-- pipeline/translator.py --> Ollama / LM Studio / OpenAI
  |     |-- pipeline/vad_buffer.py --> ffmpeg, Silero VAD
  |     |-- services/idiom_scanner.py --> data/idioms/*.json
  |     +-- services/speaker_tracker.py
  |-- routes/websocket.py --> pipeline/orchestrator.py
  |-- routes/api.py
  |     |-- routes/api_vocab.py --> services/vocab.py --> db/database.py
  |     |-- routes/api_idioms.py --> services/idiom_scanner.py
  |     |-- routes/api_llm.py --> services/lmstudio_manager.py, pipeline/translator.py
  |     |-- routes/api_playback.py --> services/playback.py
  |     |-- routes/api_sessions.py --> db/database.py
  |     +-- routes/api_system.py
  +-- client/ (static mount at /)
```

## Key Technologies

| Component | Technology | Runs On |
|-----------|-----------|---------|
| ASR | WhisperX Small | GPU (~1GB) |
| Translation | Qwen3 4B Q3_K_M via Ollama | GPU (~2.5GB) |
| Diarization | Pyannote 3.1 | CPU |
| VAD | Silero VAD | CPU |
| Idiom detection | Regex pattern DB + LLM | CPU |
| Server | FastAPI + uvicorn | CPU |
| Database | SQLite (aiosqlite, WAL) | Disk |
| Audio decode | ffmpeg (subprocess) | CPU |
| Client | HTML5 PWA, vanilla JS | Browser |
| Audio codec | Opus/WebM (MediaRecorder) | Browser |
