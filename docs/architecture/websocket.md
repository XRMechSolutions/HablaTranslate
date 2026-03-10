# WebSocket Protocol

A single WebSocket connection per client handles both binary audio streaming and JSON control messages. The server supports one active session at a time (single-user design).

## Connection

Endpoint: `WS /ws/translate`

On connect, the server creates a `ClientSession`, replaces any existing session, and sends a `status` message with pipeline readiness, session ID, current direction/mode, and speaker list.

## Message Protocol

### Client → Server

**Binary frames**: Opus/WebM audio chunks, streamed continuously during listening (~96-128 kbps).

**Text frames** (JSON with `type` field):

| Type | Purpose | Payload |
|------|---------|---------|
| `start_listening` | Begin continuous audio capture | — |
| `stop_listening` | End capture, flush remaining audio | — |
| `text_input` | Manual text entry (text-only mode) | `text`, `speaker_id` |
| `toggle_direction` | Cycle translation direction | — |
| `set_mode` | Switch operating mode | `mode` ("conversation" / "classroom") |
| `rename_speaker` | Update speaker label | `speaker_id`, `name` |
| `new_session` | Reset pipeline, start fresh session | — |
| `set_tts` | Toggle text-to-speech | `enabled`, `voice_es?`, `voice_en?` |
| `ping` | Keepalive / latency measurement | — |

### Server → Client

**Text frames** (JSON with `type` field):

| Type | When | Key Fields |
|------|------|------------|
| `status` | On connect | `pipeline_ready`, `session_id`, `direction`, `mode`, `speakers` |
| `listening_started` | After start_listening | `recording` (bool) |
| `listening_stopped` | After stop_listening | `segments`, `total_speech_duration` |
| `noise_calibration` | After first 2s of audio | `noise_floor_rms`, `level` (quiet/moderate/loud) |
| `partial_transcript` | Every ~1s during speech | `raw_transcript`, `translation_draft`, `speaker_id` |
| `translation` | After full pipeline completes | `raw_transcript`, `corrected_source`, `translation`, `speaker_label`, `confidence`, `idioms[]`, `exchange_id`, `processing_ms`, `has_audio`, `detected_language` |
| `speakers_updated` | When speaker roster changes | `speakers[]` with labels, colors, roles |
| `direction_changed` | After direction toggle | `direction` |
| `mode_changed` | After mode switch | `mode` |
| `session_reset` | After new_session | `session_id`, `direction`, `mode` |
| `tts_audio_start` | Before TTS binary frame | `exchange_id`, `size_bytes` |
| `tts_changed` | After TTS toggle | `enabled` |
| `error` | On any error | `message` |
| `pong` | Response to ping | — |

**Binary frames** (Server → Client): WAV audio for TTS playback, preceded by a `tts_audio_start` JSON message.

## ClientSession

One per WebSocket connection. Manages:

- **Listening state**: `listening` boolean, controls audio capture
- **Audio decoder**: Persistent ffmpeg subprocess for WebM→PCM conversion
- **VAD buffer**: Silero VAD instance for speech segmentation
- **Recorder**: Optional AudioRecorder (if recording enabled)
- **TTS**: Optional TTS dispatch (if enabled)
- **Playback mode**: For replaying saved recordings through the pipeline

### Segment Merge Logic

Short pauses between utterances (< 1.2s) are merged into a single segment rather than processed separately. This prevents excessive pipeline overhead on choppy speech.

- Merge gap: 1.2s (`MERGE_GAP_SECONDS`)
- Maximum pending: 60s (`MAX_PENDING_SECONDS`)
- Duration-weighted averaging of RMS and VAD probability across merged segments

### Heartbeat

Server tracks `_last_activity_time` on every received message. Connections with no activity for `ping_interval * missed_pings_threshold` seconds (default 90s) are auto-closed. Configurable via `WS_PING_INTERVAL` and `WS_MISSED_PINGS`.

### Cleanup

On disconnect:
- Cancel all background tasks (decode loop, heartbeat, pending flushes)
- Stop listening if active
- Stop playback if active
- Optionally trigger study summary generation (fire-and-forget)
- Post-cleanup invariant checks log warnings if state not fully cleaned

## Reconnection (Client-Side)

Exponential backoff: 3s → 6s → 12s → ... → max 60s, with random jitter (0.5x–1.5x).

- Max attempts: 20
- On give up: persistent "Connection lost" banner with manual retry button
- On reconnect: flush pending text queue, restore direction/mode from localStorage, re-sync session state

**Client is authoritative** for direction and mode. On reconnect, the client sends its localStorage values to the server rather than accepting the server's current state. This prevents direction/mode from resetting on page reload.

## Audio Spooling (Client-Side)

During brief disconnections while listening, audio chunks are buffered in IndexedDB (with memory fallback):

- Max age: 10 minutes (configurable via `habla_audio_spool_minutes`)
- Max size: 50MB (configurable via `habla_audio_spool_mb`)
- Flushed to server on reconnect
- Oldest chunks trimmed first when limits exceeded
- Toast warning if audio is being dropped

## Design Rationale

**Why a single WebSocket?** Separate audio and control channels would require synchronization. A single connection with binary/text frame multiplexing is simpler and provides natural ordering guarantees.

**Why single active session?** This is a personal tool accessed from one phone. Multi-session support is Phase 8 work — it requires per-client pipeline isolation.

**Why client-authoritative direction/mode?** The server is stateless between connections. If the user sets direction to en_to_es, refreshes the page, and reconnects, the server might have reset to es_to_en. LocalStorage preserves intent.

**Why segment merging?** Natural speech has micro-pauses that VAD interprets as segment boundaries. Processing each micro-segment independently wastes GPU time and fragments the conversation context. Merging produces more coherent segments.

## Planned Direction

- **Per-client sessions** (Phase 8.1): `ClientSession` already encapsulates per-connection state. The main work is moving pipeline state (direction, mode, speakers, context) out of the shared orchestrator into per-session context objects.
- **Multi-language WebSocket messages** (7.9): Update `set_direction` to accept `source_lang` + `target_lang` instead of cycling a hardcoded pair.
