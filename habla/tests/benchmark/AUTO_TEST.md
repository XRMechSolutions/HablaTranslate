# Auto-Test Recordings

Automatically benchmark newly recorded audio to track translation quality over time and identify good test samples.

## Quick Start

```bash
# 1. Record audio with the web app (enable recording first)
export SAVE_AUDIO_RECORDINGS=1
uvicorn server.main:app --host 0.0.0.0 --port 8002

# 2. After recording, run auto-test on new recordings
python tests/benchmark/auto_test_recordings.py --all

# 3. Promote good segments to permanent test samples
python tests/benchmark/auto_test_recordings.py --promote --min-confidence 0.9
```

## Features

The auto-test script:

1. **Finds new recordings** - Scans `data/audio/recordings/` for untested sessions
2. **Re-translates segments** - Runs each segment through the translator again
3. **Measures performance** - Times idiom scanning and translation
4. **Checks consistency** - Verifies translation stability (same input → same output)
5. **Assesses quality** - Identifies segments good enough for test suite
6. **Generates reports** - Saves detailed JSON results
7. **Auto-promotes** - Optionally copies high-quality segments to test samples

## Usage Modes

### 1. Test All New Recordings

Process all recordings that haven't been tested yet:

```bash
python tests/benchmark/auto_test_recordings.py --all
```

Output:
```
Testing recording: 12345_20260216_143022
  Testing: segment_001
  Testing: segment_002
  Testing: segment_003
Results saved: tests/benchmark/results/auto_test/12345_20260216_143022_20260216_150000.json

============================================================
  Recording: 12345_20260216_143022
============================================================
  Segments: 3
  Avg Confidence: 0.923
  Avg Translation Time: 342.5ms
  Segments with Idioms: 1
  Good for Testing: 2
============================================================

Promotion Candidates:
  - segment_001.wav (confidence: 0.95, idioms: False)
  - segment_002.wav (confidence: 0.92, idioms: True)
```

### 2. Watch Mode (Continuous)

Monitor for new recordings and auto-test them:

```bash
# Check every 60 seconds (default)
python tests/benchmark/auto_test_recordings.py --watch

# Check every 5 minutes
python tests/benchmark/auto_test_recordings.py --watch --interval 300
```

Great for leaving running in the background during development or classroom sessions.

### 3. Test Specific Session

Test a single recording:

```bash
python tests/benchmark/auto_test_recordings.py --session 12345_20260216_143022
```

### 4. Promote Good Segments

Copy high-quality segments to test samples directory:

```bash
# Promote segments with confidence >= 0.9
python tests/benchmark/auto_test_recordings.py --promote --min-confidence 0.9

# Only promote segments with idioms
python tests/benchmark/auto_test_recordings.py --promote --require-idioms

# Promote high-confidence segments with idioms
python tests/benchmark/auto_test_recordings.py --promote --min-confidence 0.95 --require-idioms
```

Promoted files are named automatically:
- `idioms_es_01.wav` - Segments with idioms
- `conversation_es_01.wav` - High confidence (≥0.95)
- `natural_es_01.wav` - Good quality but not perfect

## What Gets Tested

For each segment, the script:

### Original Metadata
- Raw transcript
- Translation
- Speaker
- Confidence score
- Idioms detected
- Duration

### Retest Results
- Re-translate the same transcript
- Measure translation time
- Count idioms detected again
- Check if translation matches original

### Quality Assessment
- ✅ **confidence_ok**: Confidence ≥ 0.85
- ✅ **has_idioms**: Segment contains idioms
- ✅ **translation_stable**: Retest gives same translation
- ✅ **good_for_testing**: All checks pass

## Results Format

Results saved to `tests/benchmark/results/auto_test/`:

```json
{
  "session_id": "12345_20260216_143022",
  "tested_at": "2026-02-16T15:00:00.123456",
  "segments": [
    {
      "segment_id": 1,
      "filename": "segment_001.wav",
      "original": {
        "raw_transcript": "Hola, ¿cómo estás?",
        "translation": "Hello, how are you?",
        "speaker": "SPEAKER_00",
        "confidence": 0.95,
        "idioms_detected": 0
      },
      "retest": {
        "translation": "Hello, how are you?",
        "confidence": 0.94,
        "idioms_detected": 0,
        "translation_time_ms": 342.18,
        "matches_original": true
      },
      "idiom_scan_ms": 2.45,
      "pattern_matches": 0,
      "quality": {
        "confidence_ok": true,
        "has_idioms": false,
        "translation_stable": true,
        "good_for_testing": true,
        "issues": []
      }
    }
  ],
  "summary": {
    "total_segments": 3,
    "avg_confidence": 0.923,
    "min_confidence": 0.89,
    "max_confidence": 0.95,
    "avg_translation_time_ms": 342.5,
    "segments_with_idioms": 1,
    "good_for_testing": 2,
    "promotion_candidates": [
      {
        "segment_id": 1,
        "filename": "segment_001.wav",
        "confidence": 0.95,
        "has_idioms": false
      }
    ]
  }
}
```

## Workflow Examples

### During Development

```bash
# Terminal 1: Run server with recording
export SAVE_AUDIO_RECORDINGS=1
uvicorn server.main:app --host 0.0.0.0 --port 8002

# Terminal 2: Auto-test new recordings every minute
python tests/benchmark/auto_test_recordings.py --watch --interval 60
```

Now as you use the app, recordings are automatically tested and you get quality reports.

### Build Test Suite from Real Usage

```bash
# 1. Use app normally with recording enabled (collect real-world samples)
export SAVE_AUDIO_RECORDINGS=1
uvicorn server.main:app --host 0.0.0.0 --port 8002

# 2. After a day/week, test all recordings
python tests/benchmark/auto_test_recordings.py --all

# 3. Review results, then promote best segments
python tests/benchmark/auto_test_recordings.py --promote --min-confidence 0.92

# 4. Now your test suite has real-world samples
pytest tests/benchmark/test_audio_pipeline.py -v -s
```

### Validate Translation Consistency

Check if the LLM gives consistent translations:

```bash
# Test all recordings
python tests/benchmark/auto_test_recordings.py --all

# Review results for segments where matches_original is false
# These indicate non-deterministic translation behavior
```

### Track Quality Over Time

```bash
# Test recordings daily
python tests/benchmark/auto_test_recordings.py --all

# Compare results over time
ls tests/benchmark/results/auto_test/

# Look for trends:
# - Is avg_confidence improving?
# - Are translation_time_ms going down?
# - Are more segments passing quality checks?
```

## Integration with Full Pipeline Tests

Auto-promoted segments automatically work with the full pipeline benchmarks:

```bash
# 1. Record and promote
export SAVE_AUDIO_RECORDINGS=1
uvicorn server.main:app --host 0.0.0.0 --port 8002
# ... use app ...
python tests/benchmark/auto_test_recordings.py --promote

# 2. Run full pipeline tests (includes promoted segments)
pytest tests/benchmark/test_audio_pipeline.py -v -s

# 3. Run model comparison (tests all models on same samples)
pytest tests/benchmark/test_audio_pipeline.py::test_translator_lmstudio_model_comparison -v -s
```

## Processed Marker

After testing a recording, a `.tested` marker file is created in the recording directory:

```
data/audio/recordings/12345_20260216_143022/
├── raw_stream.webm
├── segment_001.wav
├── segment_002.wav
├── metadata.json
└── .tested               # Prevents re-testing
```

To force re-test:
```bash
# Remove marker
rm data/audio/recordings/12345_*/.tested

# Re-run tests
python tests/benchmark/auto_test_recordings.py --all
```

## Performance Notes

- **Translation speed**: Each segment is re-translated, so testing 10 segments = 10 LLM calls
- **Ollama required**: Script uses Ollama by default (change in script if using LM Studio/OpenAI)
- **Parallel testing**: Currently sequential (could be parallelized for speed)

Typical timing:
- 1 segment: ~0.5s (idiom scan + translation)
- 10 segments: ~5s
- 100 segments: ~50s

## Troubleshooting

### No recordings found
```
# Check if recording is enabled
export SAVE_AUDIO_RECORDINGS=1

# Check directory exists
ls data/audio/recordings/
```

### All recordings already tested
```
# Remove .tested markers to re-test
rm data/audio/recordings/*/.tested

# Or create new recordings
```

### Translation errors
```
# Make sure Ollama is running
curl http://localhost:11434/api/tags

# Check model is available
ollama list
```

## Advanced: Custom Promotion Logic

Edit `auto_test_recordings.py` to add custom promotion criteria:

```python
def _assess_quality(self, segment_result: dict) -> dict:
    quality = {
        "good_for_testing": False,
        "issues": []
    }

    # Add your custom criteria
    if has_certain_words(transcript):
        quality["good_for_testing"] = True

    # Add specific issue detection
    if translation_seems_wrong(original, retest):
        quality["issues"].append("Suspicious translation")

    return quality
```

## See Also

- `AUDIO_RECORDING.md` - How to enable audio recording
- `test_audio_pipeline.py` - Full pipeline benchmarks
- `test_pipeline_performance.py` - LLM model comparison
- `audio_samples/README.md` - Manual recording guidelines
