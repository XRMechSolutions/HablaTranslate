# Audio Storage

Two related systems for saving and replaying audio: the AudioRecorder (debug recording and training data) and the PlaybackService (replaying recordings through the pipeline).

## AudioRecorder (`services/audio_recorder.py`, 210L)

### Purpose

Save incoming audio for debugging, test sample collection, and Whisper fine-tuning training data.

### Recording Modes

Controlled by `RecordingConfig` flags:

| Flag | Default | What It Saves |
|------|---------|--------------|
| `enabled` | false | Master toggle |
| `save_raw_audio` | true | Incoming WebM/Opus stream (full session) |
| `save_vad_segments` | true | Individual VAD-detected segments as WAV files |
| `include_metadata` | true | JSON metadata per segment (duration, size, timestamps) |

### Storage Layout

```
data/audio/recordings/
  20260303_143022_abc123/          # Session directory (timestamp + session_id)
    raw_audio.webm                 # Full session audio (WebM/Opus)
    metadata.json                  # Session metadata
    segment_0001.wav               # VAD segment (16kHz mono PCM)
    segment_0002.wav
    ...
```

### Lifecycle

1. `start_recording()`: Create session directory, open raw audio file
2. `write_raw_chunk(data)`: Append WebM chunk to raw file
3. `save_pcm_segment(pcm, segment_num)`: Write WAV file for a VAD segment
4. `add_segment_metadata(segment_num, meta)`: Attach metadata after ASR/translation
5. `stop_recording()`: Flush metadata, trigger cleanup of old recordings

### Cleanup

`_cleanup_old_recordings()` enforces `max_recordings` limit (default 100). Deletes oldest directories first (by creation time). Uses regex pattern matching on directory names to avoid deleting unrelated files.

### RecorderManager

Multi-session wrapper that creates/retrieves/removes `AudioRecorder` instances by session key.

## Audio Clips (`context_manager._save_audio_clip()`)

Separate from debug recordings. When `save_audio_clips=true`, each translated segment's WAV is copied to:

```
data/audio/clips/{session_id}/clip_{timestamp}.wav
```

The `audio_path` column in the `exchanges` table links each exchange to its clip. The client shows a play button on exchanges that have saved audio. Clips are cleaned up when a session is deleted.

## PlaybackService (`services/playback.py`, 410L)

### Purpose

Replay saved recordings through the live pipeline for parameter tuning and testing. Two modes:

### Full Mode

Tests the entire pipeline: ffmpeg decode → VAD → ASR → translation.

1. Read the raw `raw_audio.webm` file
2. Split into 1-second chunks via ffmpeg
3. Feed chunks through the AudioDecoder at controlled speed
4. VAD processes the decoded PCM and triggers the normal pipeline

### Segments Mode

Skips decode and VAD for faster ASR/translation iteration.

1. Read individual `segment_NNNN.wav` files
2. Feed each directly to `pipeline.process_wav()` (bypasses decoder + VAD)
3. Useful for testing different WhisperX models or LLM prompts on the same audio

### Speed Control

- `speed=0`: Instant (no delay between chunks)
- `speed=1.0`: Real-time
- `speed=2.0`: 2x speed
- Up to 8x supported

### Progress Feedback

Sends WebSocket messages every ~10 chunks during full mode playback:
- `playback_started`: recording_id, mode, speed
- `playback_progress`: chunks_sent, total_chunks
- `playback_finished`: total_chunks, duration
- `playback_stopped`: if cancelled

### Ground Truth Comparison

During playback, the client loads ground truth data for the recording and displays word-level diffs between the pipeline output and the reference transcript. Uses Jaccard similarity (20% threshold) to match pipeline segments to ground truth segments.

### Safety

- Path traversal protection: `_safe_session_dir()` rejects directory traversal attacks
- Pipeline state management: Sets `playback_mode` flag, resets decoder/VAD
- Cancellation: Clean task cancellation with temp directory cleanup
- Mutual exclusion: Can't play back while listening (and vice versa)

## Design Rationale

**Why two separate audio storage systems?** Debug recordings save raw browser audio for forensic analysis and training data. Audio clips save the normalized, ASR-processed audio per exchange for user playback. Different purposes, different formats, different lifecycles.

**Why segment-based playback?** Testing ASR model changes (different WhisperX sizes, fine-tuned models) shouldn't require re-running VAD segmentation every time. Feeding pre-segmented WAVs directly to ASR gives fast iteration.

**Why not stream recordings as regular WebSocket audio?** Full mode does this — it feeds chunks through the decoder as if they came from the microphone. This tests the entire pipeline including ffmpeg decode and VAD, catching bugs that segment-only playback would miss.

**Why speed control?** Processing a 30-minute classroom recording at real-time takes 30 minutes. 4x or 8x speed makes parameter tuning practical. Speed=0 (instant) is useful for batch quality analysis.

## Planned Direction

- **Pronunciation practice** (7.13): Compare user recordings against saved exchange clips. Record user's attempt, transcribe both, compute word error rate.
- **Whisper fine-tuning** (7.17): Dataset assembly script (`prepare_dataset.py`) reads recordings and corrected ground truth to build HuggingFace DatasetDict for LoRA fine-tuning.
