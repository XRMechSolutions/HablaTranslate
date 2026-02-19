# Audio Quality Tuning Guide

This document details all tunable parameters in the Habla audio pipeline for improving transcription quality, especially for quiet or noisy speech. Use this guide while gathering test samples to systematically experiment with different configurations.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Client-Side Audio (Browser)](#client-side-audio-browser)
3. [Server-Side Audio Decoding](#server-side-audio-decoding)
4. [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
5. [ASR (WhisperX)](#asr-whisperx)
6. [Speaker Diarization (Pyannote)](#speaker-diarization-pyannote)
7. [LLM Translation](#llm-translation)
8. [Testing Workflow](#testing-workflow)
9. [Common Issues & Solutions](#common-issues--solutions)

---

## Quick Reference

| Component | Key Parameter | Default | Quiet Speech Fix | Noisy Environment Fix |
|-----------|---------------|---------|------------------|-----------------------|
| **Browser** | Opus bitrate | 48kbps | ✅ Auto 128kbps (w/ AGC) | ✅ Auto 128kbps (w/ AGC) |
| **Browser** | Dynamics compression | Off | ✅ Enable in settings | Enable in settings |
| **VAD** | `speech_threshold` | 0.35 | ↓ 0.25 | ↑ 0.45 |
| **VAD** | `silence_duration_ms` | 600ms | ↑ 800ms | ↓ 400ms |
| **VAD** | `pre_speech_padding_ms` | 300ms | ↑ 500ms | = 300ms |
| **WhisperX** | `model_size` | small | → medium | → medium |
| **WhisperX** | `beam_size` | 3 | ↑ 5 | ↑ 5 |
| **WhisperX** | `vad_onset` | 0.01 | ↓ 0.001 | ↑ 0.1 |
| **Pyannote** | `min_speakers` | 1 | = 1 | = 1 |
| **LLM** | Context exchanges | 5 | ↑ 10 | = 5 |

---

## Client-Side Audio (Browser)

Location: `habla/client/index.html` (search for `MediaRecorder`)

### 1. Opus Encoding Bitrate

**Current**: ✅ **IMPLEMENTED** - Adaptive (48kbps normal, 128kbps with AGC)

**Purpose**: Higher bitrate preserves more audio detail from quiet speakers.

**How it Works**:
- **Without AGC**: 48kbps (standard quality, low bandwidth)
- **With AGC enabled**: 128kbps automatically (high quality to preserve boosted audio)

**Implementation**: `habla/client/js/audio.js` line 42

**Impact**: +80kbps bandwidth when AGC enabled (~600KB/min vs ~360KB/min)
**When enabled**: Automatically with "Boost quiet speech" toggle

---

### 2. Audio Dynamics Compression (AGC)

**Current**: ✅ **IMPLEMENTED** - Available via settings toggle

**Purpose**: Automatically boost quiet audio and compress loud peaks. Like the "loudness equalization" feature in audio players.

**How to Enable**:
1. Open the web app
2. Tap the gear icon (⚙️) in top-right
3. Enable "Boost quiet speech (AGC)"
4. Start listening

**Implementation**: `habla/client/js/audio.js` lines 18-39

**Settings Explained**:
- `threshold: -50`: Start compressing when audio exceeds -50dB (very sensitive)
- `knee: 40`: Smooth transition into compression (gentle)
- `ratio: 12`: 12:1 compression ratio (strong AGC effect)
- `attack: 0`: Instant compression (no delay)
- `release: 0.25`: Return to normal after 250ms

**Tuning for Different Scenarios**:

| Scenario | Threshold | Ratio | Knee | Notes |
|----------|-----------|-------|------|-------|
| **Quiet speaker** | -50 | 12 | 40 | Aggressive boost |
| **Noisy classroom** | -30 | 6 | 20 | Moderate compression |
| **Whispered speech** | -60 | 20 | 50 | Maximum boost |
| **Normal conversation** | -24 | 4 | 10 | Light leveling |

**Side Effects**:
- May amplify background noise (hiss, HVAC, keyboard)
- May create "pumping" artifacts if attack/release too fast
- Test in target environment (quiet room vs classroom)

---

### 3. Noise Gate (for Noisy Environments)

**Purpose**: Cut audio below a threshold to eliminate background noise during silence.

**Implementation**:
```javascript
// Add a gate before the compressor
const gate = audioContext.createDynamicsCompressor();
gate.threshold.setValueAtTime(-60, audioContext.currentTime);  // Cut below -60dB
gate.knee.setValueAtTime(0, audioContext.currentTime);         // Hard knee
gate.ratio.setValueAtTime(20, audioContext.currentTime);       // Strong cut
gate.attack.setValueAtTime(0.003, audioContext.currentTime);   // 3ms attack
gate.release.setValueAtTime(0.1, audioContext.currentTime);    // 100ms release

source.connect(gate);
gate.connect(compressor);
compressor.connect(dest);
```

**When to use**: Noisy classrooms, restaurants, street conversations

---

## Server-Side Audio Decoding

Location: `habla/server/pipeline/vad_buffer.py` (class `AudioDecoder`)

### 4. ffmpeg Sample Rate

**Current**: 16kHz (line 289)

**Purpose**: WhisperX and Pyannote expect 16kHz. Do not change unless you also change ASR model.

**Options**:
- `8000`: Phone quality (very low, not recommended)
- `16000`: Standard for speech (optimal for WhisperX Small)
- `22050`: CD quality (marginal improvement, slower processing)
- `48000`: Studio quality (4x data, minimal benefit for speech)

**Recommendation**: Leave at 16kHz unless testing with larger ASR models.

---

### 5. Server-Side Normalization (Experimental)

**Current**: None

**Purpose**: Normalize quiet audio after decoding but before VAD/ASR.

**Implementation** (add to `vad_buffer.py`):
```python
import numpy as np

def normalize_pcm_loudness(pcm_bytes: bytes, target_lufs: float = -20.0) -> bytes:
    """Apply loudness normalization to PCM int16 audio."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("pyloudnorm not installed, skipping normalization")
        return pcm_bytes

    # Convert to float32 [-1, 1]
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Measure loudness
    meter = pyln.Meter(16000)
    loudness = meter.integrated_loudness(audio)

    if loudness < -60:  # Too quiet to normalize
        return pcm_bytes

    # Normalize to target LUFS
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)

    # Convert back to int16
    return (normalized * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
```

**Usage**: Call before `vad.feed_pcm()` in `websocket.py`

**When to use**: When client-side AGC is not feasible (e.g., testing with pre-recorded files)

**Note**: Less effective than client-side AGC because Opus compression already discarded quiet audio detail.

---

## Voice Activity Detection (VAD)

Location: `habla/server/pipeline/vad_buffer.py` (class `VADConfig`)

### 6. Speech Threshold

**Current**: `speech_threshold: float = 0.35` (line 35)

**Purpose**: Silero VAD probability threshold. Below this, audio is considered silence.

**Options**:
- `0.2`: Very sensitive (catches whispers, but more false positives)
- `0.3`: Sensitive (good for quiet speakers)
- `0.35`: Default (balanced)
- `0.5`: Conservative (may miss quiet speech onsets)
- `0.7`: Very conservative (only loud clear speech)

**How to Change**:
```python
# In config.py (not yet exposed, add to AudioConfig):
class AudioConfig(BaseModel):
    vad_speech_threshold: float = 0.35  # Add this line

# In vad_buffer.py VADConfig.__init__:
speech_threshold: float = config.audio.vad_speech_threshold if config else 0.35
```

**Testing Strategy**:
1. Record sample with quiet speaker at default (0.35)
2. Replay through pipeline with threshold at 0.25, 0.30, 0.35, 0.40
3. Check if quiet portions are detected vs truncated
4. Check false positive rate (background noise triggering)

---

### 7. Silence Duration

**Current**: `silence_duration_ms: int = 600` (line 29)

**Purpose**: How long to wait after speech stops before finalizing the segment.

**Options**:
- `300ms`: Fast response (quick speakers, may cut off sentence endings)
- `600ms`: Default (balanced)
- `800ms`: Patient (good for hesitant speakers, classroom Q&A)
- `1200ms`: Very patient (wait for "um..." pauses)

**When to increase**: Quiet speakers who trail off, classroom with pauses
**When to decrease**: Fast conversations, minimize latency

---

### 8. Pre-Speech Padding

**Current**: `pre_speech_padding_ms: int = 300` (line 39)

**Purpose**: Include this much audio *before* speech onset. Captures the first syllable that triggered VAD.

**Options**:
- `200ms`: Minimal (may clip first phoneme)
- `300ms`: Default (usually captures full first word)
- `500ms`: Safe (definitely captures onset, includes leading breath)
- `1000ms`: Very safe (captures context, but may include noise)

**When to increase**: ASR missing first words, quiet speech with slow onset

---

### 9. Minimum Speech Duration

**Current**: `min_speech_ms: int = 400` (line 31)

**Purpose**: Discard segments shorter than this (filters coughs, clicks, background noise).

**Options**:
- `200ms`: Catches very short utterances ("yes", "no", "okay")
- `400ms`: Default (filters most noise)
- `600ms`: Conservative (only process full sentences)

**When to decrease**: Single-word responses, quick acknowledgments
**When to increase**: Very noisy environment with many false triggers

---

### 10. Maximum Segment Length

**Current**: `max_segment_seconds: float = 30.0` (line 33)

**Purpose**: Force-split long monologues to prevent memory buildup and keep latency reasonable.

**Options**:
- `15.0`: Short segments (more frequent updates, better for streaming)
- `30.0`: Default (good for classroom lectures)
- `60.0`: Long segments (rare, for speeches/presentations)

**Note**: Segments are split with 300ms crossfade to avoid losing words at boundaries.

---

### 11. Fallback Energy-Based VAD

**Current**: Silero VAD with energy fallback at threshold 500 (line 133)

**Purpose**: If Silero fails to load, use simple RMS energy detection.

**Tuning the Fallback**:
```python
# In vad_buffer.py, line 133:
return energy > 500  # Default for 16-bit audio

# For quiet speech:
return energy > 300  # More sensitive

# For noisy environment:
return energy > 800  # Less sensitive
```

**When to tune**: Only if Silero VAD fails to load (rare).

---

## ASR (WhisperX)

Location: `habla/server/config.py` (class `ASRConfig`)

### 12. Model Size

**Current**: `model_size: str = "small"` (line 10)

**Purpose**: Larger models are more accurate, especially for noisy/accented speech.

**Options**:

| Model | Params | VRAM | Speed | Accuracy | Use Case |
|-------|--------|------|-------|----------|----------|
| `tiny` | 39M | ~500MB | 8x | Low | Testing only |
| `base` | 74M | ~700MB | 6x | Medium | Fast draft mode |
| `small` | 244M | ~1GB | 4x | Good | **Default** |
| `medium` | 769M | ~2.5GB | 2x | Very good | Quiet/noisy speech |
| `large-v2` | 1550M | ~5GB | 1x | Excellent | Maximum quality |

**VRAM Budget** (RTX 3060 12GB):
- Current usage: ~5GB (Small + Qwen3 4B + overhead)
- With Medium: ~6.5GB (still safe, 5.5GB headroom)
- With Large: ~8.5GB (tight, 3.5GB headroom, risky thermal)

**How to Change**:
```bash
# Via environment variable:
export WHISPER_MODEL=medium
uvicorn server.main:app --host 0.0.0.0 --port 8002

# Via config.py:
model_size: str = "medium"
```

**Testing Strategy**:
1. Record problematic samples with current model (small)
2. Re-run saved WAV through pipeline with medium model
3. Compare WER (Word Error Rate) and transcript quality
4. Measure latency increase (expect ~2x slower)

---

### 13. Compute Type

**Current**: `compute_type: str = "int8"` (line 12)

**Purpose**: Quantization level. Lower precision = faster + less VRAM, but slightly less accurate.

**Options**:
- `int8`: 8-bit quantization (fast, 1GB VRAM for small model)
- `float16`: Half precision (slower, 2GB VRAM, slightly better accuracy)
- `float32`: Full precision (slowest, 4GB VRAM, best accuracy)

**Recommendation**: Stick with `int8` unless you have VRAM to spare and need maximum accuracy.

---

### 14. Beam Size

**Current**: `beam_size: int = 3` (line 13)

**Purpose**: Beam search width. Larger = explores more alternatives, more accurate but slower.

**Options**:
- `1`: Greedy search (fastest, least accurate)
- `3`: Default (balanced)
- `5`: High accuracy (1.5x slower)
- `10`: Maximum accuracy (3x slower, rarely needed)

**How to Change**:
```python
# In config.py:
beam_size: int = 5  # was 3
```

**When to increase**: High-value transcripts (lectures, legal), quiet/mumbled speech
**When to decrease**: Latency-critical applications, fast conversations

---

### 15. WhisperX Internal VAD

**Current**: `vad_filter: bool = True`, `vad_onset: 0.01`, `vad_offset: 0.01` (lines 14, 105)

**Purpose**: WhisperX has internal VAD to segment audio *within* each ASR chunk. Very aggressive (0.01 = 1% confidence).

**Interaction with Silero VAD**:
- Silero VAD (outer): Segments continuous stream into utterances
- WhisperX VAD (inner): Further segments each utterance for ASR

**Tuning**:
```python
# In orchestrator.py, line 105:
vad_options={"vad_onset": 0.01, "vad_offset": 0.01}  # Default (aggressive)

# For quiet speech (more sensitive):
vad_options={"vad_onset": 0.001, "vad_offset": 0.001}

# For noisy speech (less sensitive):
vad_options={"vad_onset": 0.1, "vad_offset": 0.1}
```

**Note**: Very low values (0.001) may cause ASR to hallucinate words in silence.

---

### 16. Language Detection

**Current**: `auto_language: bool = True` (line 18)

**Purpose**: Let WhisperX auto-detect Spanish vs English, or force a language.

**Options**:
```python
auto_language: bool = True   # Automatic (default)
# OR
auto_language: bool = False
language: str = "es"  # Force Spanish (or "en" for English)
```

**When to force language**:
- You know the source language (classroom, single-language environment)
- Auto-detection is picking wrong language
- Slight speed improvement (~5%)

**When to auto-detect**:
- Bilingual conversations (code-switching)
- User may switch direction mid-session

---

### 17. Word-Level Timestamps

**Current**: `word_timestamps: bool = True` (line 16)

**Purpose**: Get timestamps for each word (used by diarization alignment).

**Recommendation**: Leave enabled. Disabling saves minimal compute and breaks diarization alignment.

---

## Speaker Diarization (Pyannote)

Location: `habla/server/config.py` (class `DiarizationConfig`)

### 18. Device

**Current**: `device: str = "cpu"` (line 60)

**Purpose**: Pyannote runs on CPU to save GPU VRAM for WhisperX + LLM.

**Options**:
- `cpu`: Default (zero GPU cost, ~2s per 30s audio)
- `cuda`: GPU acceleration (~10x faster, but uses ~1.5GB VRAM)

**When to use GPU**: If you have VRAM headroom and need lower latency (rare).

---

### 19. Speaker Count

**Current**: `min_speakers: int = 1`, `max_speakers: int = 8` (lines 61-62)

**Purpose**: Constrain diarization to expected speaker count.

**Tuning**:
```python
# One-on-one conversation:
min_speakers: int = 1
max_speakers: int = 2

# Classroom (teacher + students):
min_speakers: int = 2
max_speakers: int = 5

# Large group discussion:
min_speakers: int = 3
max_speakers: int = 8
```

**Impact**: Tighter bounds improve accuracy. If you know "exactly 2 speakers", set both to 2.

---

## LLM Translation

Location: `habla/server/config.py` (class `TranslatorConfig`)

### 20. Context Window

**Current**: `max_context_exchanges: int = 5` (line 40)

**Purpose**: How many previous exchanges to include in LLM prompt for context-aware correction.

**Options**:
- `3`: Short context (fast, less correction power)
- `5`: Default (balanced)
- `10`: Long context (better correction, slower, may exceed small LLM context)
- `20`: Very long (only for large context models)

**When to increase**: Quiet speech with frequent mishearings, classroom corrections
**When to decrease**: Fast conversations, small LLM models

**How to Change**:
```python
# In config.py:
max_context_exchanges: int = 10  # was 5
```

---

### 21. Temperature / Reasoning Effort

**Current**: `temperature: float = 0.3` (line 39)

**Purpose**: Controls LLM creativity vs consistency.

**Options**:
- `0.0`: Deterministic (same output every time)
- `0.3`: Default (slightly varied, good for translation)
- `0.7`: Creative (more varied translations)
- `1.0`: Very creative (may deviate from source meaning)

**For OpenAI GPT-5**: Use `reasoning.effort` instead (see MEMORY.md)

**Recommendation**: Keep at 0.3 for translation (you want consistency, not creativity).

---

### 22. Quick Model (Partial Transcripts)

**Current**: `quick_model: str = ""` (line 43, uses main model if empty)

**Purpose**: Use a faster/smaller model for streaming partial transcripts, save main model for final.

**Example**:
```bash
# Use Qwen3 1.5B for partials, 4B for finals:
export QUICK_MODEL=qwen3:1.5b
export OLLAMA_MODEL=qwen3:4b
```

**When to use**: If partial translation latency is too high (>500ms).

---

## Testing Workflow

### Phase 1: Gather Baseline Samples

1. **Enable audio recording**:
   ```bash
   export SAVE_AUDIO_RECORDINGS=1
   cd habla
   uvicorn server.main:app --host 0.0.0.0 --port 8002
   ```

2. **Record diverse samples**:
   - Quiet speaker (normal volume)
   - Quiet speaker (whispered)
   - Noisy classroom background
   - Fast speech
   - Mumbled/unclear speech
   - Ideal conditions (control)

3. **Review recordings**:
   ```bash
   ls -lh data/audio/recordings/
   cat data/audio/recordings/*/metadata.json | jq
   ```

4. **Identify problems**:
   - Missing first/last words → adjust `pre_speech_padding_ms` or `speech_threshold`
   - Entire quiet utterances missing → lower `speech_threshold`
   - Wrong words transcribed → try larger model or beam size
   - Background noise triggering false speech → raise `speech_threshold` or add noise gate

---

### Phase 2: Systematic Parameter Tuning

5. **Copy problematic samples to test suite**:
   ```bash
   cp data/audio/recordings/12345_*/segment_002.wav \
      tests/benchmark/audio_samples/quiet_es_01.wav
   ```

6. **Create test matrix** (example):

| Test ID | Threshold | Padding | Model | Beam | Expected Improvement |
|---------|-----------|---------|-------|------|---------------------|
| baseline | 0.35 | 300ms | small | 3 | (control) |
| test_01 | 0.25 | 300ms | small | 3 | Catch onset |
| test_02 | 0.25 | 500ms | small | 3 | Catch onset + full word |
| test_03 | 0.35 | 300ms | medium | 3 | Better accuracy |
| test_04 | 0.25 | 500ms | medium | 5 | All improvements |

7. **Run each test**:
   ```bash
   # Modify config.py with test parameters
   # Restart server
   # Re-process saved audio
   # Compare transcript quality
   ```

8. **Measure metrics**:
   - **Transcript accuracy**: Manual review or WER calculation
   - **Detection rate**: Did it catch the utterance?
   - **Latency**: Time from speech end to translation
   - **False positive rate**: Background noise triggering

---

### Phase 3: Client-Side Testing

9. **Test browser AGC** (requires code change):
   - Add dynamics compressor to `index.html`
   - Record same speaker (quiet) with/without AGC
   - Compare server-side transcript quality

10. **Test Opus bitrate**:
    - Record at 48kbps (default) vs 128kbps
    - Compare audio quality in saved WAV segments
    - Measure bandwidth impact

---

### Phase 4: Model Comparison

11. **WhisperX Small vs Medium**:
    ```bash
    # Baseline with small
    export WHISPER_MODEL=small
    uvicorn server.main:app --host 0.0.0.0 --port 8002
    # Process test samples, record WER

    # Test with medium
    export WHISPER_MODEL=medium
    uvicorn server.main:app --host 0.0.0.0 --port 8002
    # Process same samples, compare WER
    ```

12. **LLM Translation Quality**:
    - Compare Ollama vs LM Studio vs GPT-5
    - Focus on mishearing correction, not just translation
    - Example: ASR said "No me importa un *pepito*" (mishearing)
      - Bad LLM: translates literally "I don't care about a *pepito*"
      - Good LLM: corrects to "un pepino" → "I don't care at all"

---

## Common Issues & Solutions

### Issue 1: Quiet Speaker Not Detected

**Symptoms**: VAD doesn't trigger, no transcript appears

**Diagnosis**:
```bash
# Check VAD segments in recording
ls data/audio/recordings/*/segment_*.wav
# If no segments, VAD threshold too high
```

**Solutions** (try in order):
1. Lower `speech_threshold` from 0.35 → 0.25
2. Add client-side AGC (dynamics compressor)
3. Increase `pre_speech_padding_ms` to 500ms
4. Lower WhisperX `vad_onset` to 0.001

---

### Issue 2: First Word Clipped

**Symptoms**: "ola, ¿cómo estás?" instead of "Hola, ¿cómo estás?"

**Diagnosis**: VAD onset too late, padding insufficient

**Solutions**:
1. Increase `pre_speech_padding_ms` from 300 → 500ms
2. Lower `speech_threshold` from 0.35 → 0.30

---

### Issue 3: Background Noise Triggering False Speech

**Symptoms**: Many empty/noise segments in recordings

**Diagnosis**: VAD too sensitive

**Solutions**:
1. Raise `speech_threshold` from 0.35 → 0.45
2. Increase `min_speech_ms` from 400 → 600ms
3. Add client-side noise gate (see Client-Side Audio section)
4. Raise WhisperX `vad_onset` to 0.1

---

### Issue 4: Wrong Words Transcribed (ASR Hallucination)

**Symptoms**: ASR produces plausible but incorrect words

**Diagnosis**: Audio unclear, model guessing from limited data

**Solutions**:
1. Upgrade to `medium` model
2. Increase `beam_size` from 3 → 5
3. Add client-side AGC to boost quiet audio
4. Increase Opus bitrate to 128kbps
5. Rely on LLM context correction (already enabled)

---

### Issue 5: Sentence Cut Mid-Speech

**Symptoms**: Single sentence split into two segments

**Diagnosis**: Speaker paused longer than `silence_duration_ms`

**Solutions**:
1. Increase `silence_duration_ms` from 600 → 800ms
2. Check if speaker naturally pauses (may be correct behavior)

---

### Issue 6: High Latency (>3s from speech end to translation)

**Diagnosis**: Model size, beam size, or LLM timeout

**Solutions**:
1. Use `small` model instead of `medium` (4x faster)
2. Lower `beam_size` from 5 → 3
3. Use `quick_model` for streaming partials
4. Check LLM response time (should be <500ms for local models)
5. Ensure WhisperX on GPU (`device: cuda`), Pyannote on CPU

---

### Issue 7: Translation Doesn't Fix ASR Errors

**Symptoms**: LLM passes through garbled transcript

**Diagnosis**: Insufficient context or weak LLM

**Solutions**:
1. Increase `max_context_exchanges` from 5 → 10
2. Switch to stronger LLM (LM Studio → GPT-5)
3. Add explicit correction hint to prompt (see prompts.py)
4. Check if ASR error is too severe (upgrade ASR model first)

---

## Parameter Change Checklist

Before changing parameters, document:

- [ ] Current parameter value
- [ ] New parameter value
- [ ] Hypothesis (what you expect to improve)
- [ ] Test sample (filename of WAV to re-test)
- [ ] Baseline metric (current transcript quality)
- [ ] Side effects to monitor (latency, false positives, etc.)

After testing:

- [ ] Measured improvement (better/worse/same)
- [ ] Unexpected side effects
- [ ] Decision (keep/revert/iterate)

---

## Recommended Tuning Order

1. **Start with client-side** (biggest impact, no server restart):
   - Add Opus bitrate 128kbps
   - Test with/without dynamics compressor

2. **Tune VAD next** (fast iteration):
   - Lower `speech_threshold` to 0.25
   - Increase `pre_speech_padding_ms` to 500ms

3. **Try larger model** (if steps 1-2 insufficient):
   - Upgrade `small` → `medium`
   - Increase `beam_size` to 5

4. **Fine-tune edge cases**:
   - Adjust silence duration for speaker style
   - Tune min_speech_ms for environment

---

## Benchmarking Tools

### Measure Word Error Rate (WER)

```bash
# Install jiwer
pip install jiwer

# Python script to calculate WER
from jiwer import wer

reference = "Hola, ¿cómo estás?"
hypothesis = "Ola, ¿cómo estás?"  # ASR output
error_rate = wer(reference, hypothesis)
print(f"WER: {error_rate:.2%}")  # 20.00%
```

### Automated Pipeline Testing

```bash
# Run full benchmark suite
cd habla
pytest tests/benchmark/test_audio_pipeline.py -v -s

# Test specific audio samples
pytest tests/benchmark/test_audio_pipeline.py::test_quiet_speech -v -s
```

### Manual Listening Test

```bash
# Listen to recorded segment
ffplay data/audio/recordings/12345_*/segment_001.wav

# Compare with transcript
cat data/audio/recordings/12345_*/metadata.json | jq '.segments[0].raw_transcript'
```

---

## Environment Variable Quick Reference

All config can be overridden without code changes:

```bash
# ASR
export WHISPER_MODEL=medium              # small|base|medium|large-v2
export WHISPER_DEVICE=cuda               # cuda|cpu
export ASR_AUTO_LANGUAGE=true            # true|false

# LLM
export LLM_PROVIDER=lmstudio             # ollama|lmstudio|openai
export OLLAMA_MODEL=qwen3:4b
export OLLAMA_URL=http://localhost:11434
export LMSTUDIO_MODEL=auto               # auto-detect
export LMSTUDIO_URL=http://localhost:1234
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-5-nano
export QUICK_MODEL=qwen3:1.5b            # For partials

# Diarization
export HF_TOKEN=hf_...                   # Required for Pyannote

# Recording
export SAVE_AUDIO_RECORDINGS=1           # Enable audio capture

# Database
export DB_PATH=data/habla.db
export DATA_DIR=data

# Logging
export LOG_LEVEL=DEBUG                   # DEBUG|INFO|WARNING|ERROR
export LOG_DIR=data
```

---

## Next Steps

1. Enable audio recording and gather 10-20 diverse samples
2. Identify the most common failure mode (missing onset, wrong words, etc.)
3. Pick 2-3 parameters to tune based on this guide
4. Test systematically with saved samples
5. Document results and iterate

For questions or unexpected behavior, check `data/habla.log` and `data/habla_errors.log`.
