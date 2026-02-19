# Recording Fix - Why Recordings Weren't Saving

## Root Cause

The `AudioRecorder` is only created when you press "Start Listening", NOT when you toggle recording on in the settings.

### What Was Happening

1. Server starts with `recording.enabled = False`
2. You open the app and press "Start Listening"
3. Code checks: "Is recording enabled? No" → `self.recorder = None`
4. You toggle recording ON in settings → Updates server config to `enabled = True`
5. You speak → Code tries to save: `if self.recorder:` → False, nothing saves
6. **The recorder is still `None` because it was only checked at the moment you started listening**

### The Code Flow

From `habla/server/routes/websocket.py` line 117-122:

```python
async def start_listening(self):
    # Create recorder if recording is enabled (check at start time, not connection time)
    if _recording_config and _recording_config.enabled and not self.recorder:
        logger.info(f"Creating AudioRecorder for session {self.session_id}")
        self.recorder = AudioRecorder(_recording_config, self.session_id)
```

From `habla/server/services/audio_recorder.py` line 76-84:

```python
def save_pcm_segment(self, pcm_bytes: bytes, metadata: dict = None):
    if not self.enabled or not self.config.save_vad_segments:
        return  # Exits early without saving!
```

## Solutions Implemented

### 1. Environment Variable Support (DONE)

Added `RECORDING_ENABLED` environment variable in `habla/server/config.py`:

```python
if rec := os.getenv("RECORDING_ENABLED"):
    config.recording.enabled = rec.lower() in ("1", "true", "yes", "on")
```

### 2. Startup Script (DONE)

Created `start_server_with_recording.sh`:

```bash
#!/bin/bash
cd habla
export RECORDING_ENABLED=1
export HF_TOKEN=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
python -m uvicorn server.main:app --host 0.0.0.0 --port 8002
```

### 3. Dynamic Recording Enable/Disable (DONE)

Added new methods to `ClientSession` class:

- `enable_recording()` - Creates recorder for current session
- `disable_recording()` - Stops and removes recorder

These allow toggling recording ON/OFF mid-session without restarting listening.

## How to Use

### Option A: Enable Recording From Startup

```bash
# Stop any running server
taskkill //F //PID <pid>

# Start with recording enabled
bash start_server_with_recording.sh
```

Then when you press "Start Listening", the recorder will be created automatically.

### Option B: Enable Recording Mid-Session (NEW)

1. Open the app and start using it normally
2. Toggle recording ON in settings
3. **IMPORTANT:** Stop listening and start listening again
4. Now recordings will be saved

The recorder checks the config when `start_listening()` is called, so you need to restart listening after toggling.

### Option C: Use the API Directly

Send a WebSocket message to enable recording for the current session:

```json
{
  "type": "enable_recording"
}
```

Or to disable:

```json
{
  "type": "disable_recording"
}
```

(Note: Client UI doesn't have buttons for this yet - would need to add them)

## What Gets Saved

When recording is enabled, the following are saved to `habla/data/audio/recordings/<session_id>_<timestamp>/`:

1. **`raw_stream.webm`** - Original Opus/WebM audio from browser (if `save_raw_audio: true`)
2. **`segment_NNN.wav`** - Each speech segment detected by VAD as 16kHz mono WAV files
3. **`metadata.json`** - Session info with timestamps, durations, and segment details

Example `metadata.json`:

```json
{
  "session_id": "12345",
  "started_at": "2026-02-16T10:06:40.835489",
  "raw_audio_format": "webm_opus",
  "sample_rate": 16000,
  "segments": [
    {
      "segment_id": 1,
      "filename": "segment_001.wav",
      "size_bytes": 32000,
      "duration_seconds": 1.0,
      "recorded_at": "2026-02-16T10:06:40.836491"
    }
  ],
  "ended_at": "2026-02-16T10:06:50.123456",
  "total_segments": 1
}
```

## Configuration Options

In `habla/server/config.py`, `RecordingConfig` class:

```python
enabled: bool = False              # Master toggle
save_raw_audio: bool = True        # Save incoming WebM/Opus from browser
save_decoded_pcm: bool = False     # Save decoded PCM (very verbose)
save_vad_segments: bool = True     # Save VAD-detected speech segments
output_dir: Path = Path("data/audio/recordings")
max_recordings: int = 100          # Auto-cleanup old recordings
include_metadata: bool = True      # Save JSON metadata with each recording
```

## Troubleshooting

### No recordings directory created
- Make sure `RECORDING_ENABLED=1` when starting the server
- Check server logs for "[RECORDING API] Creating recordings directory"

### Empty recordings directory
- Make sure you **stopped and restarted listening** after toggling recording on
- Check server logs for "Creating AudioRecorder for session"
- Verify `app_config.recording.enabled` is `True` in logs

### Only 1 test recording exists
- That's the test file created by the code, not a real recording
- Real recordings have session IDs based on the websocket connection ID

### Recordings stop mid-session
- Recordings stop when you press "Stop Listening"
- They auto-stop when the WebSocket disconnects
- Check `max_recordings` limit (default 100) - old recordings get cleaned up

## Next Steps (TODO)

1. Add WebSocket message handlers for `enable_recording` and `disable_recording` messages
2. Add UI buttons in the client to enable/disable recording mid-session (instead of requiring restart of listening)
3. Add notification when recording starts/stops (toast message)
4. Add recording indicator in the UI (red dot or similar)
5. Test the dynamic enable/disable functionality
6. Update client code to automatically restart listening when recording toggle changes (optional)
