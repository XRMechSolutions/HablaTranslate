# UI Improvements Summary

## Issues Fixed

### 1. Mobile Layout - Vocab Button Pushed Off Screen ✅

**Problem:** On narrow mobile screens, the control bar buttons were overflowing, pushing the Vocab button off-screen.

**Solution:**
- Added `flex-wrap: wrap` to `.ctrl` - buttons wrap to second row on narrow screens
- Added `min-width: 0` to text input area - prevents overflow
- Added `flex-shrink: 0` to buttons - maintains button sizes
- Added `white-space: nowrap` to nav buttons - prevents text wrapping

**Result:** All buttons remain visible and accessible on mobile. The mic button, text input, and send button stay on the first row, while History and Vocab buttons wrap to a second row if needed.

### 2. Save Conversation UX ✅

**Problem:** Used browser `prompt()` which was jarring and didn't match app design.

**Solution:**
- Created proper modal dialog (`#saveModal`) matching app style
- Styled input field for session name/notes
- Added Cancel/Save buttons
- Keyboard shortcuts: Enter to save, Escape to cancel

**Result:** Native-feeling save experience that matches the app's dark theme and mobile design.

### 3. History Page Date/Time Display ✅

**Problem:** Date parsing was inconsistent - SQLite returns `YYYY-MM-DD HH:MM:SS` but JavaScript expected ISO format with 'T'.

**Solution:**
- Improved `fmtDate()` and `fmtTime()` to handle both formats
- Added validation to catch invalid dates
- Proper timezone handling (SQLite CURRENT_TIMESTAMP is UTC)

**Result:** Dates display correctly as "Today 14:30", "Yesterday 09:15", etc.

### 4. Audio Recording Toggle ✅

**Problem:** Recording toggle in UI was just informational - didn't actually control recording.

**Solution:**
- Added API endpoint `POST /api/system/recording` to toggle recording
- Wired up checkbox to call API and update server config
- Shows current status from server on page load
- Displays info text when enabled
- Creates recordings directory automatically when enabled

**Result:** Users can now enable/disable audio recording directly from the web UI without restarting the server.

## How to Use Audio Recording

### From Web UI (New!)

1. Open LLM Settings (gear icon)
2. Scroll to "Recording (for benchmarks)"
3. Toggle "Save audio to server"
4. Start a conversation
5. Audio will be saved to `data/audio/recordings/`

### From Command Line (Original Method)

```bash
export SAVE_AUDIO_RECORDINGS=1
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

## Files Modified

### Mobile Layout
- `client/styles.css` - Added flex-wrap, min-width, flex-shrink

### Save Modal
- `client/index.html` - Added #saveModal dialog
- `client/js/app.js` - Replaced prompt() with modal, added keyboard shortcuts

### History Dates
- `client/history.html` - Improved date parsing in fmtDate() and fmtTime()

### Recording Toggle
- `server/routes/api.py` - Added POST /api/system/recording endpoint
- `server/routes/api.py` - Added recording_enabled to GET /api/system/status
- `client/index.html` - Updated recording section label and info text
- `client/js/settings.js` - Added setRecording(), updateRecordingInfo(), wired up toggle

## Technical Details

### API Endpoint

```javascript
POST /api/system/recording
{
  "enabled": true
}

Response:
{
  "recording_enabled": true,
  "output_dir": "data/audio/recordings"
}
```

### Recording Flow

1. User toggles checkbox in UI
2. JavaScript calls `/api/system/recording`
3. Server updates `config.recording.enabled`
4. Server creates recordings directory if enabling
5. WebSocket sessions use updated config for new connections
6. Audio is saved per `RecordingConfig` settings

### Recordings Structure

```
data/audio/recordings/
└── <session_id>_<timestamp>/
    ├── raw_stream.webm      # Original browser audio
    ├── segment_001.wav      # VAD segments
    ├── segment_002.wav
    └── metadata.json        # Transcripts, translations, etc.
```

## Testing

All features have been tested and work correctly:
- ✅ Mobile layout - buttons wrap properly on narrow screens
- ✅ Save modal - shows/hides correctly, Enter/Escape work
- ✅ History dates - parse correctly from SQLite format
- ✅ Recording toggle - enables/disables recording, shows status

## Next Steps

Users can now:
1. Use the app on mobile without layout issues
2. Save conversations with a native modal dialog
3. View session history with correct dates
4. Enable audio recording from the UI for building test samples

The auto-test workflow works seamlessly:
```bash
# 1. Enable recording from UI (or command line)
# 2. Have conversations
# 3. Run auto-test on recordings
python tests/benchmark/auto_test_recordings.py --all

# 4. Promote good samples to test suite
python tests/benchmark/auto_test_recordings.py --promote --min-confidence 0.9

# 5. Run full pipeline benchmarks
pytest tests/benchmark/test_audio_pipeline.py -v -s
```
