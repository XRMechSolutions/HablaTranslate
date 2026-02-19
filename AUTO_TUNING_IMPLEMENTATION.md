# Auto-Tuning Implementation Status

## Completed ✅

### 1. auto_tune_parameters.py
**Location**: `habla/scripts/auto_tune_parameters.py`

**Features**:
- Analyzes saved recordings from `data/audio/recordings/`
- Detects clipped onsets (missing first letters: "ola" → "hola")
- Calculates confidence statistics per speaker
- Recommends optimal VAD threshold, padding, silence duration
- Recommends AGC enablement based on volume variance
- Recommends model upgrade if needed
- Outputs human-readable report or JSON

**Usage**:
```bash
cd habla
python scripts/auto_tune_parameters.py
python scripts/auto_tune_parameters.py --json > recommendations.json
python scripts/auto_tune_parameters.py --apply  # Interactive application
```

---

### 2. compare_wer.py
**Location**: `habla/scripts/compare_wer.py`

**Features**:
- A/B tests different parameter configurations
- Calculates Word Error Rate (WER) for each config
- Tests on ground-truth audio samples
- Ranks configurations by accuracy
- Shows improvement over baseline

**Usage**:
```bash
# Create test samples first
mkdir -p tests/benchmark/audio_samples
# Copy recorded segments + create ground_truth.json

python scripts/compare_wer.py
python scripts/compare_wer.py --output wer_results.json
```

**Requires**: `ground_truth.json` with format:
```json
{
  "sample1": {
    "audio": "tests/benchmark/audio_samples/quiet_es_01.wav",
    "transcript": "Hola, ¿cómo estás?",
    "duration": 2.5
  }
}
```

---

### 3. quality_metrics Database Table
**Location**: `habla/server/db/database.py` (line 118)

**Schema**:
```sql
CREATE TABLE quality_metrics (
    id                  INTEGER PRIMARY KEY,
    session_id          INTEGER REFERENCES sessions(id),
    segment_id          INTEGER,
    timestamp           DATETIME,
    confidence          REAL,
    audio_rms           REAL,
    duration_seconds    REAL,
    speaker_id          TEXT,
    clipped_onset       BOOLEAN,
    processing_time_ms  INTEGER,
    vad_threshold       REAL,
    model_name          TEXT
)
```

**Purpose**: Tracks quality metrics for continuous learning

---

## Partially Complete ⚠️

### 4. Quality Metrics Tracking in Orchestrator
**Status**: Database table created, tracking code needs to be added

**What's Needed**:
Add to `orchestrator.py` after each transcription:

```python
async def _record_quality_metrics(self, exchange: Exchange, audio_pcm: bytes):
    """Track quality metrics for auto-tuning."""
    import numpy as np

    # Calculate audio RMS
    audio_array = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio_rms = float(np.sqrt(np.mean(audio_array ** 2)))

    # Detect clipped onsets
    transcript = exchange.raw_transcript.lower().strip()
    clipped_patterns = ["ola", "ueno", "racias", "asta", "ace"]
    clipped_onset = any(transcript.startswith(p) for p in clipped_patterns)

    # Save metrics
    db = await get_db()
    await db.execute("""
        INSERT INTO quality_metrics (
            session_id, segment_id, confidence, audio_rms,
            duration_seconds, speaker_id, clipped_onset,
            processing_time_ms, vad_threshold, model_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        self.session_id,
        exchange.id,
        exchange.confidence,
        audio_rms,
        len(audio_pcm) / 32000,  # 16kHz stereo
        exchange.speaker.speaker_id,
        clipped_onset,
        exchange.processing_ms,
        self.vad_buffer.config.speech_threshold,
        self.config.asr.model_size
    ))
    await db.commit()

# Call after process_audio completes
await self._record_quality_metrics(exchange, audio_bytes)
```

---

## Not Yet Implemented ❌

### 5. Continuous Auto-Tuning
**Location**: Needs to be added to `orchestrator.py`

**What's Needed**:
```python
async def _auto_tune_if_needed(self):
    """Periodically auto-tune parameters based on recent metrics."""

    # Run every 100 segments
    if self._segments_processed % 100 != 0:
        return

    db = await get_db()
    cursor = await db.execute("""
        SELECT AVG(confidence), STDDEV(confidence), SUM(clipped_onset)
        FROM quality_metrics
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT 100
    """, (self.session_id,))

    row = await cursor.fetchone()
    if not row:
        return

    avg_conf, std_conf, clipped_count = row

    # Auto-adjust VAD threshold
    if avg_conf < 0.75:
        new_threshold = 0.30
        if self.vad_buffer.config.speech_threshold != new_threshold:
            self.vad_buffer.config.speech_threshold = new_threshold
            logger.info(f"Auto-tuned VAD threshold: {new_threshold} (low confidence)")

    # Auto-adjust padding if many clipped onsets
    if clipped_count > 10:  # >10% clipped
        new_padding = 450
        if self.vad_buffer.config.pre_speech_padding_ms != new_padding:
            self.vad_buffer.config.pre_speech_padding_ms = new_padding
            logger.info(f"Auto-tuned pre-speech padding: {new_padding}ms (clipped onsets)")
```

---

### 6. API Endpoint for Auto-Tune
**Location**: Needs to be added to `habla/server/routes/api.py`

**What's Needed**:
```python
@system_router.post("/api/system/auto-tune")
async def run_auto_tune():
    """Trigger auto-tuning analysis on saved recordings."""

    # Run auto_tune_parameters.py as subprocess
    import subprocess
    import json

    result = subprocess.run(
        ["python", "scripts/auto_tune_parameters.py", "--json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )

    if result.returncode != 0:
        return {"error": "Auto-tune failed", "stderr": result.stderr}

    recommendations = json.loads(result.stdout)

    return {
        "success": True,
        "recommendations": recommendations["recommendations"],
        "stats": recommendations["results"]
    }


@system_router.post("/api/system/apply-tuning")
async def apply_tuning(recommendations: dict):
    """Apply auto-tuning recommendations."""

    # Update orchestrator parameters dynamically
    pipeline = get_pipeline()  # Global pipeline instance

    if "speech_threshold" in recommendations:
        pipeline.vad_buffer.config.speech_threshold = recommendations["speech_threshold"]

    if "pre_speech_padding_ms" in recommendations:
        pipeline.vad_buffer.config.pre_speech_padding_ms = recommendations["pre_speech_padding_ms"]

    if "silence_duration_ms" in recommendations:
        pipeline.vad_buffer.config.silence_duration_ms = recommendations["silence_duration_ms"]

    return {"success": True, "applied": recommendations}
```

---

### 7. Auto-Tune UI Button
**Location**: Needs to be added to `habla/client/index.html` settings modal

**What's Needed**:
```html
<div class="set-section">
  <div class="set-label">Auto-Tuning</div>
  <div class="set-row">
    <button class="set-btn" id="autoTuneBtn">Analyze & Optimize</button>
    <div class="set-status" id="autoTuneStatus"></div>
  </div>
  <div class="set-info" style="font-size: 11px; color: var(--fg3); margin-top: 4px;">
    Analyzes recorded audio to recommend optimal settings
  </div>
</div>
```

**JavaScript** (`habla/client/js/settings.js`):
```javascript
$('#autoTuneBtn').onclick = async () => {
  $('#autoTuneStatus').textContent = 'Analyzing...';
  $('#autoTuneBtn').disabled = true;

  try {
    const r = await fetch('/api/system/auto-tune', { method: 'POST' });
    const data = await r.json();

    if (data.success) {
      $('#autoTuneStatus').textContent = 'Analysis complete';

      // Show recommendations
      const recs = data.recommendations;
      const message = `
        Recommendations:
        - VAD Threshold: ${recs.speech_threshold}
        - Padding: ${recs.pre_speech_padding_ms}ms
        - AGC: ${recs.enable_agc ? 'Enable' : 'Disable'}

        Apply these settings?
      `;

      if (confirm(message)) {
        await fetch('/api/system/apply-tuning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(recs)
        });
        toast('Settings optimized', 'ok');
      }
    } else {
      $('#autoTuneStatus').textContent = 'Failed';
      toast('Auto-tune failed: ' + data.error, 'error');
    }
  } catch (e) {
    $('#autoTuneStatus').textContent = 'Error';
    toast('Auto-tune error: ' + e.message, 'error');
  } finally {
    $('#autoTuneBtn').disabled = false;
  }
};
```

---

### 8. Ground Truth Template
**Location**: Needs to be created at `tests/benchmark/audio_samples/ground_truth.json`

**Template**:
```json
{
  "quiet_es_01": {
    "audio": "tests/benchmark/audio_samples/quiet_es_01.wav",
    "transcript": "Hola, ¿cómo estás?",
    "duration": 2.5,
    "notes": "Quiet speaker, classroom environment"
  },
  "idiom_es_01": {
    "audio": "tests/benchmark/audio_samples/idiom_es_01.wav",
    "transcript": "No me importa un pepino lo que pienses",
    "duration": 3.2,
    "notes": "Idiom test: un pepino"
  },
  "fast_es_01": {
    "audio": "tests/benchmark/audio_samples/fast_es_01.wav",
    "transcript": "Necesito que me ayudes con esto porque no sé cómo hacerlo",
    "duration": 4.1,
    "notes": "Fast speech"
  }
}
```

---

## Testing Workflow

### Phase 1: Collect Data (Current)
```bash
# Enable recording
export SAVE_AUDIO_RECORDINGS=1
cd habla
uvicorn server.main:app --host 0.0.0.0 --port 8002

# Use Habla normally for 2-4 weeks
# Collect 20-50 recordings
```

### Phase 2: Analyze (After collection)
```bash
# Run auto-tune analysis
python scripts/auto_tune_parameters.py

# Output shows recommendations:
# - speech_threshold: 0.28
# - pre_speech_padding_ms: 450
# - enable_agc: True
```

### Phase 3: Test Configurations (Optional)
```bash
# Create ground truth samples
cp data/audio/recordings/*/segment_*.wav tests/benchmark/audio_samples/
# Manually verify transcripts, create ground_truth.json

# Compare WER across configs
python scripts/compare_wer.py

# Output ranks configs by accuracy
```

### Phase 4: Apply & Monitor (Future - with UI)
```bash
# Via web UI:
# 1. Open settings
# 2. Click "Analyze & Optimize"
# 3. Review recommendations
# 4. Click "Apply"

# Or via API:
curl -X POST http://localhost:8002/api/system/auto-tune
curl -X POST http://localhost:8002/api/system/apply-tuning \
  -H "Content-Type: application/json" \
  -d '{"speech_threshold": 0.28, "pre_speech_padding_ms": 450}'
```

---

## Next Steps to Complete

1. **Add metrics tracking to orchestrator** (1-2 hours)
   - Implement `_record_quality_metrics()` method
   - Call after each `process_audio()` completion

2. **Add continuous auto-tuning** (1-2 hours)
   - Implement `_auto_tune_if_needed()` method
   - Call every 100 segments

3. **Add API endpoints** (1-2 hours)
   - `/api/system/auto-tune` (trigger analysis)
   - `/api/system/apply-tuning` (apply recommendations)

4. **Add UI button** (1-2 hours)
   - Auto-Tune button in settings
   - Show recommendations dialog
   - Apply button

5. **Create ground truth template** (30 minutes)
   - Example `ground_truth.json`
   - Documentation

6. **Update documentation** (1 hour)
   - Add auto-tuning workflow to CLAUDE.md
   - Update AUDIO_TUNING_GUIDE.md with auto-tune instructions

**Total remaining**: ~8-10 hours of implementation work

---

## Current Status Summary

✅ **Working now**:
- `auto_tune_parameters.py` - Analyze recordings, get recommendations
- `compare_wer.py` - A/B test configurations
- Database schema for quality metrics

⚠️ **Partially complete**:
- Quality metrics tracking (needs orchestrator integration)

❌ **Not yet implemented**:
- Continuous auto-tuning in orchestrator
- API endpoints for auto-tune
- UI button for auto-tune
- Ground truth template file

**You can use `auto_tune_parameters.py` right now** once you have recordings. The rest is infrastructure to make it automatic/easier to use.
