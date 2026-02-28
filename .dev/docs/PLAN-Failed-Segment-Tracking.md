# Plan: Failed Segment Tracking & Auto-Tuning Feedback Loop

## Existing Scaffolding Audit (Feb 2026)

Several pieces of this feature were started but never connected. Current state:

| Component | What Exists | What's Missing |
|-----------|-------------|----------------|
| `quality_metrics` DB table | Schema in `database.py:132-148` with correct columns | **Never written to.** No INSERT anywhere in codebase. Missing `status` column + indexes. |
| `_is_bad_transcript()` | Detection logic in `orchestrator.py:1083-1096`, called at line 903 | On rejection: logs WARNING, returns `""`, **drops segment silently**. No DB record, no metrics, no metadata. |
| Segment metadata enrichment | `websocket.py:410-416` enriches successful segments with transcript/confidence/speaker | **Only runs on success.** Rejected segments never reach this code path. |
| `audio_recorder.add_segment_metadata()` | Accepts arbitrary dict, merges into segment info | Works, but **callers don't pass** `audio_rms`, `vad_avg_prob`, or failure info. |
| VAD probability | `vad_buffer.py:128` computes `prob = model(tensor).item()` every frame | **Discards the value** — only uses it for boolean is/isn't speech. Not accumulated. |
| VAD threshold config | `config.py` has `asr.vad_threshold=0.35`, `VADConfig` has `speech_threshold=0.35` | **Two independent values.** `websocket.py:118` hardcodes VADConfig, ignores app config. |
| `clipped_onset` column | Exists in `quality_metrics` table | **Never computed.** `auto_tune_parameters.py:76-90` has detection logic but only in offline script. |
| `auto_tune_parameters.py` | Reads `metadata.json`, detects low confidence + clipped onsets, recommends changes | **Reads files only**, never queries `quality_metrics` DB. Can't see rejected segments. |
| `orchestrator._metrics` | Tracks `low_confidence_count`, `segments_processed`, etc. | **Missing** `asr_rejected_count` and `asr_empty_count`. |

## Problem Statement

When VAD detects speech but ASR produces empty or garbage output, the segment is silently dropped — logged as a WARNING but never recorded in the database, metadata, or UI. This creates a blind spot:

- We can't measure how often speech goes untranscribed
- We can't review those audio clips to determine if they're genuine speech or VAD false positives
- We can't correlate failure patterns with audio conditions (quiet speakers, background noise, VAD threshold)
- The auto-tuning script (`auto_tune_parameters.py`) has no visibility into these failures
- The `quality_metrics` DB table was built for this purpose but is never populated

## Current Data Flow (with gap)

```
VAD detects speech (≥400ms above 0.35 threshold)
    ↓
Segment emitted, PCM saved as segment_NNN.wav (if recording enabled)
    ↓
ASR runs on WAV
    ├─ Success → Exchange created, stored in DB, sent to client
    ├─ Empty/garbage → WARNING log → GONE (the gap)
    └─ Low confidence (<0.3) → stored + metric incremented
```

**The gap**: Segments that VAD approved but ASR rejected are lost. The WAV file exists on disk (if recording was enabled), but nothing links it to the failure event.

## Proposed Data Flow (with tracking)

```
VAD detects speech (≥400ms above 0.35 threshold)
    ↓
Segment emitted, PCM saved as segment_NNN.wav + RMS/VAD prob recorded
    ↓
ASR runs on WAV
    ├─ Success → Exchange created, quality_metrics row written (status=ok)
    ├─ Empty/garbage → quality_metrics row written (status=asr_rejected)
    │                  segment metadata updated with failure reason
    │                  _metrics["asr_rejected_count"] incremented
    └─ Low confidence → Exchange created, quality_metrics row flagged (status=low_confidence)
```

## Implementation Plan

### Step 1: Add RMS and VAD probability to segment emission

**File: `habla/server/pipeline/vad_buffer.py`**

The `_emit_segment()` method currently passes `(segment_bytes, duration)` to its callback. Extend to also compute and pass:
- **`audio_rms`**: RMS energy of the segment (indicates volume level)
- **`vad_avg_prob`**: Average VAD probability across speech frames (indicates how confident VAD was)

Changes:
- In `_emit_segment()`, compute RMS from `segment_bytes` using numpy: `np.sqrt(np.mean(np.frombuffer(..., dtype=np.int16).astype(np.float32)**2))`
- Track cumulative VAD probability during speech frames (add `_speech_prob_sum` accumulator, divide by `_speech_frames` at emission)
- Change callback signature from `on_segment(pcm_bytes, duration)` to `on_segment(pcm_bytes, duration, audio_rms, vad_avg_prob)`
- Update `on_partial_audio` callback to also pass RMS (useful for partial normalization)

Impact on callers:
- `websocket.py` `_on_vad_segment()` — receives the new params, forwards to recorder and orchestrator
- All test mocks that use `on_segment` callbacks need signature update

### Step 2: Pass audio metadata through to the orchestrator

**File: `habla/server/routes/websocket.py`**

The `_on_vad_segment()` callback in `ClientSession` handles segments from VAD. Update it to:
- Accept `audio_rms` and `vad_avg_prob` from the VAD callback
- Pass them through to `recorder.save_pcm_segment()` as metadata fields
- Pass them to the orchestrator (new parameter on `process_wav()` or store as segment context)

**File: `habla/server/services/audio_recorder.py`**

- `save_pcm_segment()` already accepts a `metadata` dict — just ensure `audio_rms` and `vad_avg_prob` are included in the dict passed by the websocket handler
- No structural changes needed, just the caller passes richer metadata

### Step 3: Record quality metrics on every segment outcome

**File: `habla/server/pipeline/orchestrator.py`**

Add a new method `_record_quality_metric()` that INSERTs into the `quality_metrics` table:

```python
async def _record_quality_metric(
    self, segment_id: int, status: str, confidence: float | None,
    audio_rms: float | None, duration: float, speaker_id: str | None,
    vad_threshold: float, processing_ms: int
):
    """Record segment quality data for auto-tuning analysis."""
    await execute("""
        INSERT INTO quality_metrics
        (session_id, segment_id, confidence, audio_rms, duration_seconds,
         speaker_id, clipped_onset, processing_time_ms, vad_threshold, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [self._current_session_id, segment_id, confidence, audio_rms,
          duration, speaker_id, False, processing_ms, vad_threshold, self._model_name])
```

Call sites:
1. **After successful ASR + translation** (in `_process_audio_segment()` after diarization): status=ok, include confidence from ASR result
2. **After ASR rejection** (where `_is_bad_transcript()` returns True): status=asr_rejected, confidence=None
3. **After empty ASR** (where transcript is empty): status=asr_empty, confidence=None

Add to `self._metrics`:
- `"asr_rejected_count": 0` — segments where ASR returned garbage
- `"asr_empty_count": 0` — segments where ASR returned nothing

### Step 4: Add status column to quality_metrics table

**File: `habla/server/db/database.py`**

Add a `status` column to the `quality_metrics` table to categorize outcomes:

```sql
ALTER TABLE quality_metrics ADD COLUMN status TEXT DEFAULT 'ok';
```

Since we use `CREATE TABLE IF NOT EXISTS`, add the column via a migration-safe approach:
- After the CREATE TABLE, run `ALTER TABLE ... ADD COLUMN status TEXT DEFAULT 'ok'` wrapped in a try/except (column already exists → ignore)

Valid statuses: `ok`, `asr_rejected`, `asr_empty`, `low_confidence`

### Step 5: Update segment metadata with failure info

**File: `habla/server/services/audio_recorder.py`**

The existing `add_segment_metadata()` method already supports adding arbitrary keys to segment metadata. After ASR rejection, call:

```python
recorder.add_segment_metadata(segment_id, "asr_status", "rejected")
recorder.add_segment_metadata(segment_id, "asr_reject_reason", "bad_transcript")
```

This enriches `metadata.json` so the auto-tuning script can see which segments failed.

### Step 6: Expose failed segments via REST API

**File: `habla/server/routes/api_system.py`**

Add to the existing `/api/system/metrics` endpoint:
- `asr_rejected_count` and `asr_empty_count` from pipeline metrics
- Total quality_metrics breakdown by status (query DB)

**File: `habla/server/routes/api_corrections.py`** (or new endpoint)

Add endpoint:
- `GET /api/corrections/failed-segments` — returns segments from recordings where `asr_status == "rejected"` in metadata, paired with their WAV filenames for review in the corrections UI

### Step 7: Update auto-tuning script

**File: `habla/scripts/auto_tune_parameters.py`**

Currently reads only `metadata.json`. Extend to also:
1. Read `quality_metrics` from the database (or from enriched metadata.json)
2. Count ASR rejection rate: `rejected / (rejected + successful)` per session
3. Correlate rejections with:
   - `audio_rms` — are rejections clustered at low volume? → recommend lower VAD threshold or AGC
   - `vad_avg_prob` — are rejections from borderline VAD detections (0.35-0.50)? → raise VAD threshold
   - `duration_seconds` — are very short segments (0.4-0.8s) being rejected more? → raise `min_speech_ms`
   - Speaker-specific patterns → identify quiet speakers
4. Generate recommendations based on failure patterns:
   - High rejection rate + low RMS → "Enable AGC or lower mic gain"
   - High rejection rate + borderline VAD prob → "Raise speech_threshold from 0.35 to 0.45"
   - High rejection rate + short duration → "Raise min_speech_ms from 400 to 600"

### Step 8: Surface failed segments in corrections UI

**File: `habla/client/corrections.html`**

Add a "Failed Segments" tab/filter to the corrections page:
- Show recordings with ASR-rejected segments
- For each failed segment: play audio, see RMS/VAD stats, manually transcribe or mark as noise
- If manually transcribed → saves to `corrected_ground_truth.json` (same pipeline as normal corrections)
- If marked as noise → saves with `transcript: "[noise]"` so the dataset script can exclude it

This closes the loop: failed segments get human review, become training data, improve the model.

### Step 9: Update prepare_dataset.py to use failed+corrected segments

**File: `habla/scripts/prepare_dataset.py`**

Already handles `corrected_ground_truth.json` and filters `[inaudible]`. Extend:
- Also filter `[noise]` markers (from Step 8)
- Include manually transcribed formerly-failed segments (these are the most valuable training data — they represent exactly the cases the model struggles with)

## Database Schema Change

```sql
-- Add status column to existing quality_metrics table
ALTER TABLE quality_metrics ADD COLUMN status TEXT DEFAULT 'ok';

-- Index for efficient querying by status
CREATE INDEX IF NOT EXISTS idx_quality_metrics_status ON quality_metrics(status);

-- Index for session-level aggregation
CREATE INDEX IF NOT EXISTS idx_quality_metrics_session ON quality_metrics(session_id);
```

## Metrics Added

| Metric | Location | Description |
|--------|----------|-------------|
| `asr_rejected_count` | `orchestrator._metrics` | Segments where `_is_bad_transcript()` returned True |
| `asr_empty_count` | `orchestrator._metrics` | Segments where ASR returned no text at all |
| `audio_rms` | `quality_metrics` table | RMS energy per segment |
| `vad_avg_prob` | segment metadata | Average VAD probability during speech |
| `status` | `quality_metrics` table | Outcome category: ok/asr_rejected/asr_empty/low_confidence |

## Callback Signature Changes

### Before
```python
# VAD → callback
on_segment(pcm_bytes: bytes, duration: float)

# VAD → partial callback
on_partial_audio(pcm_bytes: bytes, duration: float)
```

### After
```python
# VAD → callback
on_segment(pcm_bytes: bytes, duration: float, audio_rms: float, vad_avg_prob: float)

# on_partial_audio stays the same (no need for extra data on partials)
```

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `server/pipeline/vad_buffer.py` | Compute RMS + avg VAD prob, pass to callback |
| `server/routes/websocket.py` | Accept new params, forward to recorder/orchestrator |
| `server/pipeline/orchestrator.py` | New `_record_quality_metric()`, new rejection metrics, call on every segment outcome |
| `server/db/database.py` | Add `status` column to `quality_metrics`, add indexes |
| `server/services/audio_recorder.py` | No structural changes — callers pass richer metadata dict |
| `server/routes/api_system.py` | Expose new metrics in `/api/system/metrics` |
| `server/routes/api_corrections.py` | Add `GET /api/corrections/failed-segments` endpoint |
| `scripts/auto_tune_parameters.py` | Read quality_metrics, correlate failures with audio conditions, generate recommendations |
| `client/corrections.html` | Add failed segment review tab |
| `scripts/prepare_dataset.py` | Filter `[noise]`, include corrected failed segments |

## Implementation Order

1. **DB schema** (Step 4) — add status column + indexes
2. **VAD buffer** (Step 1) — compute RMS + VAD prob
3. **WebSocket + recorder** (Steps 2, 5) — forward new metadata
4. **Orchestrator** (Step 3) — record quality metrics on every outcome
5. **API** (Step 6) — expose new metrics and failed segments endpoint
6. **Auto-tuning** (Step 7) — correlate failures with conditions
7. **Corrections UI** (Step 8) — failed segment review
8. **Dataset script** (Step 9) — handle noise markers

Steps 1-4 form the core tracking pipeline. Steps 5-9 are the consumption/feedback layer.

## Testing Considerations

- Mock the DB for orchestrator tests that verify `_record_quality_metric()` calls
- Test VAD buffer RMS calculation with known audio (silence → RMS ≈ 0, tone → known RMS)
- Test that ASR rejection path writes quality_metrics row with status=asr_rejected
- Test auto-tuning recommendations with synthetic quality_metrics data
- Integration test: feed a very quiet audio segment, verify it appears in failed segments API
