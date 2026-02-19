"""Quick test to verify recording setup works."""

import sys
from pathlib import Path

# Test 1: Import recorder
print("Test 1: Importing AudioRecorder...")
try:
    from server.services.audio_recorder import AudioRecorder
    from server.config import RecordingConfig
    print("[OK] Imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Create config
print("\nTest 2: Creating RecordingConfig...")
config = RecordingConfig(
    enabled=True,
    save_raw_audio=True,
    save_vad_segments=True,
    output_dir=Path("data/audio/recordings"),
)
print(f"[OK] Config created: enabled={config.enabled}")

# Test 3: Create recorder
print("\nTest 3: Creating AudioRecorder...")
try:
    recorder = AudioRecorder(config, "test_session_12345")
    print(f"[OK] Recorder created for session: {recorder.session_id}")
    print(f"  Session dir: {recorder.session_dir}")
except Exception as e:
    print(f"[FAIL] Recorder creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Start recording
print("\nTest 4: Starting recording...")
try:
    recorder.start_recording()
    print(f"[OK] Recording started")
    print(f"  Raw file: {recorder.raw_file}")
except Exception as e:
    print(f"[FAIL] Start recording failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Write some data
print("\nTest 5: Writing test chunk...")
try:
    test_chunk = b"fake audio data"
    recorder.write_raw_chunk(test_chunk)
    print(f"[OK] Wrote {len(test_chunk)} bytes")
except Exception as e:
    print(f"[FAIL] Write failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Save a segment
print("\nTest 6: Saving test segment...")
try:
    import struct
    # Create fake PCM data (1 second of silence at 16kHz, 16-bit)
    pcm_bytes = struct.pack(f"{16000}h", *([0] * 16000))
    recorder.save_pcm_segment(pcm_bytes, metadata={"test": True})
    print(f"[OK] Saved segment")
except Exception as e:
    print(f"[FAIL] Save segment failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Stop recording
print("\nTest 7: Stopping recording...")
try:
    recorder.stop_recording()
    print(f"[OK] Recording stopped")
except Exception as e:
    print(f"[FAIL] Stop failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Check files created
print("\nTest 8: Verifying files...")
if recorder.session_dir.exists():
    files = list(recorder.session_dir.glob("*"))
    print(f"[OK] Session directory exists with {len(files)} files:")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size} bytes)")
else:
    print(f"[FAIL] Session directory not found: {recorder.session_dir}")

print("\n" + "="*60)
print("Recording test complete!")
print("If all tests passed, recording should work in the app.")
print("="*60)
