# Pipeline Overview

The processing pipeline turns raw browser audio into translated exchange cards with speaker attribution, idiom highlights, and vocabulary flags. It was refactored from a single 1025-line orchestrator into three focused modules.

## Module Responsibilities

| Module | Owns | Delegates To |
|--------|------|-------------|
| **`orchestrator.py`** (1295L) | Processing queue, background worker, in-flight task tracking, async callbacks, metrics, DB operations, session lifecycle | AudioProcessor, ContextManager |
| **`audio_processor.py`** (300L) | WhisperX ASR (GPU), Pyannote diarization (CPU), ffmpeg decode/normalize, language detection voting, partial transcript state | — |
| **`context_manager.py`** (116L) | Translator lifecycle, SpeakerTracker, IdiomScanner, conversation context (direction, mode, topic, exchanges), idiom merge, audio clip saving | Translator, SpeakerTracker, IdiomScanner |

## Processing Flow

```
                              ┌─ Partial path (~1s): quick ASR + lightweight translation
                              │
WebSocket audio chunk         │
  → AudioDecoder (ffmpeg)     │
  → StreamingVADBuffer        │
  → on_segment callback ──────┤
                              │
                              └─ Final path (on silence):
                                   Queue (maxsize=5)
                                     → Background worker
                                       → ASR (GPU, serialized by _asr_lock)
                                       → Language detection voting
                                       → Diarization (CPU, parallel with next ASR)
                                       → [concurrent] Translation → Idiom merge → DB save → Callback
```

## Queue and Backpressure

The orchestrator uses a bounded `asyncio.Queue(maxsize=5)`. Each queued item carries:

- `kind`: "wav" or "raw"
- `payload`: audio bytes or text string
- `future`: the caller awaits this for the final Exchange result
- `meta`: dict with `audio_rms`, `vad_avg_prob`, `segment_id`

A single background worker drains the queue sequentially. ASR is GPU-bound and serialized. Once a transcript is ready, translation fires off as a concurrent background task so the worker immediately starts the next ASR — LLM latency never blocks audio processing.

## Two-Tier Streaming

**Partials** (every ~1s during speech): VAD emits a snapshot of accumulated audio. The orchestrator runs quick ASR (batch_size=4, no diarization) + a lightweight translation (temperature=0.2, no idiom scanning). The client shows this as a live-updating card at reduced opacity.

**Finals** (after silence gap): VAD closes the segment. The full pipeline runs — ASR with batch_size=8, Pyannote diarization, full LLM translation with context, idiom merge, correction detection. The partial card snaps to a polished final card.

## Concurrent Translation

ASR and translation are decoupled. Once ASR finishes for segment N, the worker queues the translation as a background task and immediately starts ASR for segment N+1. In-flight translation tasks are tracked in `_inflight_translations` and awaited during shutdown.

This means the pipeline can be transcribing the next utterance while the LLM is still translating the previous one — critical for natural conversation pace.

## Session Lifecycle

- `create_session()`: Insert row into `sessions` table, set session_id on context_manager
- `close_session()`: Write end timestamp, generate study summary (fire-and-forget), reset state
- `reset_session()`: Close current + create new, clear conversation context
- State persisted to `data/last_session.json` on shutdown (topic, exchanges, speakers, metrics)
- State restored on next startup for conversation continuity across restarts

## Metrics

Tracked in an in-memory dict (not persisted):

- `segments_processed`, `translations_completed`, `translation_errors`
- `asr_rejected_count`, `asr_empty_count`, `low_confidence_count`
- `corrections_detected`, `idiom_pattern_db_hits`, `idiom_llm_hits`
- `queue_depth`, `peak_queue_depth`
- Processing time distribution (avg, min, max)

Exposed via `GET /api/system/metrics`.

## Design Rationale

**Why a thin coordinator?** The orchestrator started as a monolith handling ASR, translation, idiom scanning, and DB writes. Extracting AudioProcessor and ContextManager made each concern independently testable and kept the orchestrator focused on coordination — queue management, concurrency control, and callback dispatch.

**Why a single worker?** WhisperX runs on GPU and isn't thread-safe for concurrent inference. A single sequential worker with GPU serialization is simpler and more reliable than semaphore-gated concurrent workers. The LLM call (the slower part) runs concurrently anyway.

**Why futures on the queue?** The WebSocket handler needs to know when a segment finishes processing (for recording metadata). Futures provide clean await-based signaling without callback spaghetti.

## Planned Direction

- **Per-client pipeline isolation** (Phase 8.1): Move per-conversation state (direction, mode, speakers, context) into a `ConversationContext` object per WebSocket client. The orchestrator keeps shared resources (ML models, translator) but each client gets its own context.
- **Processing queue improvements** (Phase 8.2): Per-client queues or priority queuing, backpressure signaling to the client.
- **Auto language detection** (7.11): Per-segment language detection already implemented via voting; needs UI toggle and `auto` direction mode.
