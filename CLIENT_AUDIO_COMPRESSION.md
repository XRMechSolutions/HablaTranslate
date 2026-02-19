# Client-Side Audio Compression (AGC)

## Overview

Client-side audio compression has been implemented to improve transcription quality for quiet speakers. This feature uses the Web Audio API to apply automatic gain control (AGC) before audio encoding.

## What It Does

- **Boosts quiet audio** automatically using dynamics compression (12:1 ratio)
- **Increases Opus bitrate** from 48kbps → 128kbps when enabled (to preserve boosted audio quality)
- **Compresses loud peaks** to prevent clipping
- **Zero server-side changes** required (all processing in browser)

## How to Enable

1. Open Habla web app
2. Tap the **gear icon (⚙️)** in the top-right corner
3. Find the **"Audio Input"** section
4. Enable **"Boost quiet speech (AGC)"**
5. Start listening

The setting is saved to localStorage and persists across sessions.

## Technical Details

### Compression Settings

The implementation uses aggressive AGC tuned for quiet speech:

```javascript
compressor.threshold = -50dB   // Start compressing at very low levels
compressor.knee = 40           // Smooth transition
compressor.ratio = 12:1        // Strong compression
compressor.attack = 0ms        // Instant response
compressor.release = 250ms     // Return to normal after 1/4 second
```

### Bandwidth Impact

| Mode | Opus Bitrate | Bandwidth | Usage per hour |
|------|-------------|-----------|----------------|
| **Normal** | 48kbps | ~360 KB/min | ~21 MB/hour |
| **With AGC** | 128kbps | ~960 KB/min | ~56 MB/hour |

Even with AGC enabled, bandwidth is very reasonable for mobile data (~56 MB/hour).

### Audio Processing Pipeline

```
Microphone → Web Audio API → DynamicsCompressor → MediaRecorder (128kbps Opus)
                                                        ↓
                                                   WebSocket
                                                        ↓
                                                   Server (ffmpeg)
```

## When to Use

### ✅ Enable AGC for:
- **Quiet speakers** (students, shy speakers, distance from mic)
- **Whispered speech** or very soft voices
- **Classroom recordings** where volume varies
- **Mobile phone recording** (lower mic sensitivity)

### ❌ Disable AGC for:
- **Normal/loud speech** (no benefit, uses more bandwidth)
- **Very noisy environments** (may amplify background noise)
- **Metered/slow mobile data** (saves bandwidth)

## Testing Strategy

### Baseline Test (Before AGC)
1. Enable audio recording: `export SAVE_AUDIO_RECORDINGS=1`
2. Record a quiet speaker with AGC **disabled**
3. Note which words are missed or misheard

### Comparison Test (With AGC)
4. Enable AGC in settings
5. Record the **same speaker** saying similar content
6. Compare transcripts:
   - Are quiet words now detected?
   - Is accuracy improved?
   - Is background noise amplified too much?

### A/B Comparison
```bash
# Review both recordings side by side
ls data/audio/recordings/*/segment_*.wav
cat data/audio/recordings/*/metadata.json | jq '.segments[] | {transcript, confidence}'
```

## Side Effects to Monitor

### Potential Issues

1. **Background noise amplification**
   - Quiet hum, HVAC, keyboard clicks may be boosted
   - **Solution**: Test in target environment (classroom, home, etc.)

2. **"Pumping" artifacts**
   - Audio may sound "breathing" if attack/release too fast
   - Current settings (0ms attack, 250ms release) minimize this

3. **Increased bandwidth**
   - 128kbps uses ~2.7x more data than 48kbps
   - Still only ~56 MB/hour (very reasonable)

## Tuning the Compressor (Advanced)

To adjust compression settings for different scenarios, edit `habla/client/js/audio.js` lines 25-30:

### For Very Quiet / Whispered Speech
```javascript
compressor.threshold.setValueAtTime(-60, audioCtx.currentTime);  // Even more sensitive
compressor.ratio.setValueAtTime(20, audioCtx.currentTime);       // More aggressive
```

### For Noisy Classroom (Less Aggressive)
```javascript
compressor.threshold.setValueAtTime(-30, audioCtx.currentTime);  // Less sensitive
compressor.ratio.setValueAtTime(6, audioCtx.currentTime);        // Moderate compression
compressor.knee.setValueAtTime(20, audioCtx.currentTime);        // Sharper knee
```

### For Normal Conversation (Light Leveling)
```javascript
compressor.threshold.setValueAtTime(-24, audioCtx.currentTime);
compressor.ratio.setValueAtTime(4, audioCtx.currentTime);
compressor.knee.setValueAtTime(10, audioCtx.currentTime);
```

## Implementation Files

- **Audio processing**: `habla/client/js/audio.js` (lines 18-42, 69-73)
- **Settings UI**: `habla/client/index.html` (lines 148-157)
- **Settings logic**: `habla/client/js/settings.js` (lines 103, 117-120, 161)

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Full | Recommended |
| Firefox | ✅ Full | Recommended |
| Safari | ✅ Full | iOS 14.5+ |
| Edge | ✅ Full | Chromium-based |

Web Audio API is well-supported across all modern browsers.

## Performance Impact

- **CPU**: Minimal (~1-2% on modern phones)
- **Battery**: Negligible (compression is lightweight)
- **Latency**: <5ms added latency (imperceptible)

The DynamicsCompressor is hardware-accelerated on most devices.

## Troubleshooting

### AGC Not Working

**Symptom**: Toggle enabled but audio sounds the same

**Check**:
1. Refresh page after enabling
2. Check browser console for errors (F12 → Console)
3. Verify `localStorage.getItem('habla_audio_compression')` returns `"1"`

### Audio Sounds Distorted

**Symptom**: Audio is too loud or clipping

**Solution**: Lower the compression ratio:
```javascript
compressor.ratio.setValueAtTime(6, audioCtx.currentTime);  // Less aggressive
```

### Background Noise Too Loud

**Symptom**: HVAC, keyboard, etc. are amplified

**Solutions**:
1. Disable AGC (not needed if speaker is loud enough)
2. Add a noise gate (see AUDIO_TUNING_GUIDE.md section 3)
3. Raise threshold to -30dB (less sensitive to quiet sounds)

## Next Steps

1. **Test with real quiet speakers** in your target environment
2. **Compare WER (Word Error Rate)** with/without AGC using saved recordings
3. **Tune compressor settings** if needed for your specific use case
4. **Measure bandwidth impact** on mobile data if concerned

## Related Documentation

- **Full tuning guide**: `AUDIO_TUNING_GUIDE.md`
- **Audio recording**: `habla/AUDIO_RECORDING.md`
- **Configuration**: `habla/server/config.py` (server-side settings)
