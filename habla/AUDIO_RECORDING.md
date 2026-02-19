# Audio Recording Feature

The Habla server can optionally save audio from the web app for debugging, testing, and building test samples.

## Features

When enabled, the server automatically saves:

1. **Raw audio stream** - The original WebM/Opus data from the browser microphone
2. **VAD segments** - Individual speech segments detected by the VAD, saved as WAV files
3. **Metadata** - JSON file with transcript, translation, speaker, confidence, and idioms for each segment

## Enabling Audio Recording

### Method 1: Environment Variable (Recommended)

```bash
export SAVE_AUDIO_RECORDINGS=1
cd habla
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

### Method 2: Docker

```yaml
# docker-compose.yml
services:
  habla:
    environment:
      - SAVE_AUDIO_RECORDINGS=1
    volumes:
      - ./data:/app/data  # Recordings saved to data/audio/recordings/
```

### Method 3: Code Configuration

```python
# In your startup code
from server.config import load_config

config = load_config()
config.recording.enabled = True
```

## Recorded Data Structure

```
data/audio/recordings/
├── 12345_20260216_143022/
│   ├── raw_stream.webm          # Original browser audio
│   ├── segment_001.wav          # First VAD segment
│   ├── segment_002.wav          # Second VAD segment
│   ├── segment_003.wav          # Third VAD segment
│   └── metadata.json            # Full session metadata
└── 12346_20260216_143530/
    ├── raw_stream.webm
    ├── segment_001.wav
    └── metadata.json
```

### Metadata JSON Format

```json
{
  "session_id": "12345",
  "started_at": "2026-02-16T14:30:22.123456",
  "ended_at": "2026-02-16T14:32:15.789012",
  "raw_audio_format": "webm_opus",
  "sample_rate": 16000,
  "total_segments": 3,
  "segments": [
    {
      "segment_id": 1,
      "filename": "segment_001.wav",
      "size_bytes": 64000,
      "duration_seconds": 2.0,
      "recorded_at": "2026-02-16T14:30:24.456789",
      "raw_transcript": "Hola, ¿cómo estás?",
      "translation": "Hello, how are you?",
      "speaker": "SPEAKER_00",
      "confidence": 0.95,
      "idioms_detected": 0
    },
    {
      "segment_id": 2,
      "filename": "segment_002.wav",
      "size_bytes": 96000,
      "duration_seconds": 3.0,
      "recorded_at": "2026-02-16T14:30:28.123456",
      "raw_transcript": "No me importa un pepino lo que pienses.",
      "translation": "I don't care at all what you think.",
      "speaker": "SPEAKER_00",
      "confidence": 0.92,
      "idioms_detected": 1
    }
  ]
}
```

## Configuration Options

In `server/config.py`, the `RecordingConfig` class has these settings:

```python
class RecordingConfig(BaseModel):
    enabled: bool = False                    # Master toggle
    save_raw_audio: bool = True              # Save WebM/Opus stream
    save_decoded_pcm: bool = False           # Save decoded PCM (verbose)
    save_vad_segments: bool = True           # Save VAD segments (WAV)
    output_dir: Path = Path("data/audio/recordings")
    max_recordings: int = 100                # Auto-cleanup old recordings
    include_metadata: bool = True            # Save JSON metadata
```

## Use Cases

### 1. Building Test Samples

Recorded segments can be moved to `tests/benchmark/audio_samples/` for automated pipeline testing:

```bash
# Copy interesting samples
cp data/audio/recordings/*/segment_*.wav tests/benchmark/audio_samples/

# Rename for auto-discovery
mv segment_001.wav conversation_es_01.wav
mv segment_002.wav idioms_es_01.wav

# Run benchmarks
pytest tests/benchmark/test_audio_pipeline.py -v -s
```

### 2. Debugging ASR/Translation Issues

When users report "the translation was wrong", you can:

1. Enable recording
2. Reproduce the issue
3. Check `metadata.json` for the exact transcript and translation
4. Listen to the WAV segment to hear what the ASR heard
5. Compare raw transcript vs translation to identify LLM issues vs ASR issues

### 3. Model Comparison

Record a conversation once, then:

```bash
# Extract segments for benchmarking
cp data/audio/recordings/12345_*/segment_*.wav tests/benchmark/audio_samples/

# Run model comparison
pytest tests/benchmark/test_audio_pipeline.py::test_translator_lmstudio_model_comparison -v -s
```

### 4. Training Data Collection

For fine-tuning models or improving idiom detection:

1. Record classroom sessions with consent
2. Review `metadata.json` to find segments with low confidence
3. Manually verify transcripts
4. Use corrected data for fine-tuning

## Privacy & Storage

### Privacy Considerations

- **Voice samples contain personal data** - Do not commit recordings to version control
- **Get consent** - Inform users if recording is enabled in production
- `.gitignore` automatically excludes `data/audio/recordings/`
- Recordings are stored locally on the server, never sent to cloud

### Storage Management

- Each recording session creates a directory with timestamp
- Segments are WAV files (16kHz mono, ~32KB per second)
- A 5-minute conversation generates ~10-15 segments (~5MB total)
- Old recordings are auto-deleted when `max_recordings` limit is exceeded (default: 100 sessions)

### Manual Cleanup

```bash
# Remove all recordings
rm -rf data/audio/recordings/*

# Remove sessions older than 7 days
find data/audio/recordings -type d -mtime +7 -exec rm -rf {} +

# Keep only last 20 sessions
ls -t data/audio/recordings | tail -n +21 | xargs -I {} rm -rf data/audio/recordings/{}
```

## Disabling Recording

Set `SAVE_AUDIO_RECORDINGS=0` or simply don't set it (default is disabled).

```bash
export SAVE_AUDIO_RECORDINGS=0
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

Or stop the server and restart without the environment variable.

## Web UI Indicator

When recording is enabled on the server, the web app shows a status message in the LLM Settings modal:

```
Recording
[x] Save audio for testing
    Recording enabled on server
```

This is informational only - the toggle doesn't control server-side recording (it's configured via environment variable).

## Integration with Benchmarks

Recorded audio automatically works with the benchmark suite:

```bash
# 1. Enable recording and use the app normally
export SAVE_AUDIO_RECORDINGS=1
uvicorn server.main:app --host 0.0.0.0 --port 8002

# 2. After recording some audio, copy segments to test samples
cp data/audio/recordings/*/segment_*.wav tests/benchmark/audio_samples/

# 3. Rename following the naming convention (see tests/benchmark/audio_samples/README.md)
mv segment_001.wav conversation_es_01.wav
mv segment_002.wav fast_es_01.wav

# 4. Run benchmarks
pytest tests/benchmark/test_audio_pipeline.py -v -s
```

## Technical Details

### Implementation

- `server/services/audio_recorder.py` - Recording service
- `server/routes/websocket.py` - Integration into WebSocket handler
- `server/config.py` - Configuration model

### Audio Formats

- **Raw stream**: WebM container with Opus codec (browser native)
- **Segments**: WAV files (16kHz, 16-bit, mono PCM)
- **Compatibility**: WAV segments work with all audio tools (ffmpeg, Audacity, pytest benchmarks)

### Performance Impact

Minimal - audio saving is asynchronous and uses simple file writes. No transcoding or processing.

## Example Workflow

```bash
# 1. Start server with recording enabled
export SAVE_AUDIO_RECORDINGS=1
cd habla
uvicorn server.main:app --host 0.0.0.0 --port 8002

# 2. Use the web app, have a conversation in Spanish

# 3. Check recordings
ls -lh data/audio/recordings/

# 4. Review metadata
cat data/audio/recordings/*/metadata.json | jq

# 5. Listen to a segment
ffplay data/audio/recordings/*/segment_001.wav

# 6. Copy interesting samples for testing
cp data/audio/recordings/12345_*/segment_002.wav \
   tests/benchmark/audio_samples/idioms_es_01.wav

# 7. Run benchmarks
pytest tests/benchmark/test_audio_pipeline.py -v -s
```
