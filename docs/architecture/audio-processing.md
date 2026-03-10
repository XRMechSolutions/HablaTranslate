# Audio Processing

Handles everything between raw browser audio and a transcript ready for translation: codec decode, voice activity detection, speech segmentation, ASR, diarization, and audio normalization.

## Components

### AudioDecoder (`vad_buffer.py`)

Wraps a persistent ffmpeg subprocess to decode Opus/WebM chunks from the browser into raw 16kHz mono PCM.

**Streaming mode** (continuous sessions): Single ffmpeg process stays alive for the entire listening session. Chunks are fed via stdin, PCM read from stdout in a background task. O(1) amortized per chunk — no re-decode overhead.

**Process crash detection**: If ffmpeg exits unexpectedly, the decoder sets a `_dead` flag and logs the error. The WebSocket handler detects this and notifies the client.

### StreamingVADBuffer (`vad_buffer.py`)

Server-side Voice Activity Detection using Silero VAD. Processes 30ms frames (512 samples at 16kHz), detects speech boundaries, and emits complete utterances.

**State machine**:
```
[CALIBRATING] → 2s ambient noise measurement → [READY]
                                                  ↓
                                    VAD prob > 0.35 → [SPEAKING]
                                    VAD prob ≤ 0.35 → [SILENT]

[SPEAKING]:
  - Accumulate frames into speech buffer
  - Every ~1s: emit partial audio snapshot
  - After 30s: force segment boundary (prevents runaway buffer)
  - After 600ms silence: emit complete segment → [SILENT]

[SILENT]:
  - Maintain ring buffer of last 300ms (pre-speech padding)
```

**Noise floor calibration**: First 2 seconds of audio measure ambient energy. The energy threshold is set to `max(300, noise_floor_rms * 3)`. This adapts to the environment — a quiet room gets a sensitive threshold, a noisy classroom gets a higher one.

**Pre-speech padding**: A 300ms ring buffer of recent frames is prepended to every segment. This prevents clipped consonants (critical for Spanish /g/, /b/, /d/).

**Segment emission**: Computes `audio_rms` (loudness) and `vad_avg_prob` (speech confidence) from accumulated frame data. Both are passed downstream for quality metrics.

**Fallback VAD**: If Silero fails to load (torch hub unavailable), falls back to energy-based detection: `energy > threshold`. Functional but less accurate.

**Tuning parameters** (all configurable via env vars):

| Parameter | Default | Env Var | Purpose |
|-----------|---------|---------|---------|
| `speech_threshold` | 0.35 | `VAD_THRESHOLD` | Silero probability threshold |
| `silence_duration_ms` | 600 | `VAD_SILENCE_MS` | Silence before segment boundary |
| `min_speech_ms` | 400 | `MIN_SPEECH_MS` | Minimum speech to emit (skip coughs) |
| `max_segment_seconds` | 30.0 | `MAX_SEGMENT_SECONDS` | Force-split long monologues |
| `pre_speech_padding_ms` | 300 | — | Include audio before speech onset |
| `calibration_duration_ms` | 2000 | `CALIBRATION_DURATION_MS` | Noise floor measurement period |

### AudioProcessor (`audio_processor.py`)

Owns the ML models for ASR and diarization. All GPU operations are serialized through `_asr_lock` (threading.Lock).

**ASR** (WhisperX, GPU, ~1GB VRAM):
- Full path: batch_size=8, word timestamps, language detection
- Partial path: batch_size=4, no diarization, speed priority
- Uses voted language hint when consensus exists; falls back to source language from direction setting

**Diarization** (Pyannote 3.1, CPU):
- Assigns speaker labels (SPEAKER_00, SPEAKER_01, ...) to transcript segments
- Requires `HF_TOKEN` env var for model access
- Gracefully skips if unavailable — ASR works without speaker attribution

**Audio normalization** (ffmpeg, per-segment):
- Highpass at 80Hz (remove rumble/wind)
- Lowpass at 7500Hz (remove hiss)
- Dynamic audio normalization: frame-by-frame gain adjustment (f=150, g=15, p=0.95)

**Garbage detection** (two distinct failure modes):
- `no_segments`: WhisperX VAD found no speech — too quiet or all noise
- `bad_transcript`: WhisperX produced output but <3 letters or <50% letters — garbage

Both are recorded in `quality_metrics` with distinct status values for downstream analysis.

**Language detection voting**:
- Last 5 detected languages in a deque
- Requires 70% supermajority to switch language
- Prevents a single misdetection from poisoning subsequent segments

## Threading Model

```
Event loop (async)
  ├── WebSocket handler
  │     └── feed audio chunks to decoder/VAD
  ├── VAD callbacks (on_segment, on_partial_audio)
  │     └── queue segments for processing
  └── Pipeline worker
        ├── asyncio.to_thread(WhisperX)  ← serialized by _asr_lock
        ├── asyncio.to_thread(Pyannote)  ← CPU, parallel with next ASR
        └── ffmpeg subprocess             ← per-segment normalization
```

GPU operations (WhisperX) are serialized. CPU operations (Pyannote, ffmpeg) run in threads without contention.

## Design Rationale

**Why server-side VAD?** Browser-side VAD would reduce bandwidth, but Silero VAD's quality is significantly better than anything available in the browser. The ~20MB/hour bandwidth is acceptable over Tailscale even on metered mobile data.

**Why persistent ffmpeg?** Launching a new ffmpeg process per chunk would add ~50ms startup overhead per chunk. The persistent subprocess amortizes this to zero after the first chunk.

**Why voting for language detection?** A single WhisperX misdetection (common with short or quiet segments) would flip the translation direction for all subsequent segments. Requiring 70% agreement over 5 segments prevents cascading errors.

**Why not use WhisperX's built-in VAD alone?** WhisperX's VAD operates on complete audio files, not streaming chunks. Silero VAD processes 30ms frames in real-time, enabling streaming partial transcripts during speech.

## Planned Direction

- **Auto language detection UI** (7.11): The voting mechanism exists; needs a client-side `auto` direction toggle and per-exchange detected language badges.
- **Multi-language support** (7.9): WhisperX supports 90+ languages. Language detection voting would adapt to arbitrary pairs.
- **WhisperX fine-tuning** (7.17): Correction workflow and dataset assembly scripts are complete. LoRA fine-tuning of the Small model for classroom Spanish is the next step.
