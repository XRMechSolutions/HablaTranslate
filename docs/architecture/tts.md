# Text-to-Speech

`services/tts.py` (113L) — CPU-based text-to-speech using Piper, disabled by default. Speaks translations aloud so the user can hear the target language pronunciation.

## Architecture

### TTSService

- **Lazy loading**: Voice models loaded on first `synthesize()` call, not at startup
- **Caching**: Once loaded, voice objects stay in memory for the session
- **Thread serialization**: `asyncio.Lock` prevents double-loading when concurrent requests arrive
- **Async wrapper**: Blocking Piper synthesis runs in a thread via `asyncio.to_thread()`

### Voice Files

Piper uses ONNX voice models stored in `data/tts_models/`:

```
data/tts_models/
  es_ES-carlfm-x_low.onnx       # Spanish voice
  es_ES-carlfm-x_low.onnx.json  # Voice config
  en_US-lessac-medium.onnx       # English voice
  en_US-lessac-medium.onnx.json  # Voice config
```

Voice names are configurable via `TTS_VOICE_ES` and `TTS_VOICE_EN` env vars.

### Synthesis Flow

1. Client sends `{type: "set_tts", enabled: true}` via WebSocket
2. On each translation result, `_dispatch_tts()` fires as an async task
3. `TTSService.synthesize(text, lang)` loads the voice (if not cached), runs Piper, returns WAV bytes
4. Server sends `tts_audio_start` JSON message (with `exchange_id` and `size_bytes`)
5. Server sends WAV bytes as a binary WebSocket frame
6. Client decodes and plays the audio via Web Audio API

### WAV Container

`_build_wav()` wraps raw 16-bit mono PCM in a minimal WAV header (44 bytes). This avoids depending on external audio libraries for container framing.

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `TTS_ENABLED` | `false` | Master toggle |
| `TTS_VOICE_ES` | `es_ES-carlfm-x_low` | Spanish voice model name |
| `TTS_VOICE_EN` | `en_US-lessac-medium` | English voice model name |
| `TTS_MODELS_DIR` | `data/tts_models` | Voice model directory |

## VRAM Impact

Zero. Piper uses ONNX Runtime on CPU. No GPU memory consumed. This was a hard requirement — the VRAM budget is tight at 5GB on 12GB total.

## Graceful Degradation

- If `piper-tts` package isn't installed: `synthesize()` returns None, pipeline continues
- If voice model file is missing: returns None, logs warning
- If synthesis fails for any reason: returns None, no crash

The TTS path is entirely optional and never blocks translation.

## Design Rationale

**Why Piper?** CPU-only, fast (~1-2s for a sentence), MIT-licensed, small model files (~50MB per voice), good quality for the price. GPU-based alternatives (Bark, Tortoise) would eat VRAM that WhisperX and the LLM need.

**Why lazy loading?** Most sessions don't use TTS. Loading 100MB of voice models at startup would slow down server start for a rarely-used feature.

**Why disabled by default?** TTS requires downloading voice models manually. Enabling it without models would just produce errors.

**Why send WAV over WebSocket instead of a URL?** Streaming the audio inline avoids an extra HTTP round-trip and keeps the single-WebSocket architecture clean. The client decodes WAV in the browser using Web Audio API.

## Planned Direction

- **Voice selection UI**: Users can switch voices per-language from the settings panel. Voice config already sent via `set_tts` message.
- **Multi-language TTS** (7.9): Support voices for languages beyond Spanish/English. Piper has models for 20+ languages.
