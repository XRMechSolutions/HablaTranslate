"""Tests for server.services.audio_recorder — AudioRecorder and RecorderManager."""

import json
import struct
import wave
from pathlib import Path

import pytest

from server.config import RecordingConfig
from server.services.audio_recorder import AudioRecorder, RecorderManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def recording_config(tmp_path):
    from server.config import RecordingConfig
    return RecordingConfig(
        enabled=True,
        save_raw_audio=True,
        save_vad_segments=True,
        output_dir=tmp_path / "recordings",
        max_recordings=3,
        include_metadata=True,
    )


@pytest.fixture
def disabled_config(tmp_path):
    return RecordingConfig(
        enabled=False,
        output_dir=tmp_path / "recordings",
    )


@pytest.fixture
def recorder(recording_config):
    """An enabled AudioRecorder ready for use."""
    return AudioRecorder(recording_config, "sess-001")


@pytest.fixture
def pcm_one_second():
    """One second of silence as 16-bit 16kHz mono PCM."""
    return b"\x00\x00" * 16000


# ---------------------------------------------------------------------------
# TestAudioRecorderInit
# ---------------------------------------------------------------------------

class TestAudioRecorderInit:

    @pytest.mark.unit
    def test_init_disabled_skips_directory_creation(self, disabled_config):
        recorder = AudioRecorder(disabled_config, "sess-off")
        assert recorder.enabled is False
        # output_dir should not have been created
        assert not disabled_config.output_dir.exists()

    @pytest.mark.unit
    def test_init_enabled_creates_session_directory(self, recording_config):
        recorder = AudioRecorder(recording_config, "sess-001")
        assert recorder.enabled is True
        assert recorder.session_dir.exists()
        assert recorder.session_dir.is_dir()
        # session_dir lives inside output_dir
        assert recorder.session_dir.parent == recording_config.output_dir

    @pytest.mark.unit
    def test_init_sets_metadata_fields(self, recording_config):
        recorder = AudioRecorder(recording_config, "sess-meta")
        meta = recorder.metadata
        assert meta["session_id"] == "sess-meta"
        assert "started_at" in meta
        assert meta["raw_audio_format"] == "webm_opus"
        assert meta["sample_rate"] == 16000
        assert meta["segments"] == []


# ---------------------------------------------------------------------------
# TestStartRecording
# ---------------------------------------------------------------------------

class TestStartRecording:

    @pytest.mark.unit
    def test_start_recording_creates_raw_file(self, recorder):
        recorder.start_recording()
        assert recorder.raw_file is not None
        assert recorder.raw_file.name == "raw_stream.webm"
        assert recorder.raw_handle is not None
        # Clean up handle
        recorder.raw_handle.close()

    @pytest.mark.unit
    def test_start_recording_disabled_does_nothing(self, disabled_config):
        rec = AudioRecorder(disabled_config, "sess-off")
        rec.start_recording()
        assert not hasattr(rec, "raw_file") or rec.raw_file is None

    @pytest.mark.unit
    def test_start_recording_raw_audio_disabled_does_nothing(self, recording_config):
        recording_config.save_raw_audio = False
        rec = AudioRecorder(recording_config, "sess-noraw")
        rec.start_recording()
        assert rec.raw_file is None
        assert rec.raw_handle is None


# ---------------------------------------------------------------------------
# TestWriteRawChunk
# ---------------------------------------------------------------------------

class TestWriteRawChunk:

    @pytest.mark.unit
    def test_write_raw_chunk_writes_bytes(self, recorder):
        recorder.start_recording()
        payload = b"\x1a\x45\xdf\xa3" + b"\xab" * 100
        recorder.write_raw_chunk(payload)
        recorder.raw_handle.flush()

        content = recorder.raw_file.read_bytes()
        assert content == payload
        recorder.raw_handle.close()

    @pytest.mark.unit
    def test_write_raw_chunk_disabled_does_nothing(self, disabled_config):
        rec = AudioRecorder(disabled_config, "sess-off")
        # Should not raise
        rec.write_raw_chunk(b"\x00" * 50)

    @pytest.mark.unit
    def test_write_raw_chunk_no_handle_does_nothing(self, recorder):
        # start_recording was never called, so raw_handle is None
        recorder.write_raw_chunk(b"\x00" * 50)
        # No exception, nothing written


# ---------------------------------------------------------------------------
# TestSavePcmSegment
# ---------------------------------------------------------------------------

class TestSavePcmSegment:

    @pytest.mark.unit
    def test_save_pcm_segment_creates_wav_file(self, recorder, pcm_one_second):
        recorder.save_pcm_segment(pcm_one_second)

        wav_path = recorder.session_dir / "segment_001.wav"
        assert wav_path.exists()

        # Verify the WAV is valid and has the right properties
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000  # 1 second

    @pytest.mark.unit
    def test_save_pcm_segment_increments_counter(self, recorder, pcm_one_second):
        recorder.save_pcm_segment(pcm_one_second)
        recorder.save_pcm_segment(pcm_one_second)
        recorder.save_pcm_segment(pcm_one_second)

        assert recorder.segment_counter == 3
        assert (recorder.session_dir / "segment_001.wav").exists()
        assert (recorder.session_dir / "segment_002.wav").exists()
        assert (recorder.session_dir / "segment_003.wav").exists()

    @pytest.mark.unit
    def test_save_pcm_segment_adds_metadata(self, recorder, pcm_one_second):
        recorder.save_pcm_segment(pcm_one_second)

        segments = recorder.metadata["segments"]
        assert len(segments) == 1
        seg = segments[0]
        assert seg["segment_id"] == 1
        assert seg["filename"] == "segment_001.wav"
        assert seg["size_bytes"] == len(pcm_one_second)
        assert seg["duration_seconds"] == pytest.approx(1.0)
        assert "recorded_at" in seg

    @pytest.mark.unit
    def test_save_pcm_segment_with_extra_metadata(self, recorder, pcm_one_second):
        extra = {"speaker": "Speaker A", "language": "es"}
        recorder.save_pcm_segment(pcm_one_second, metadata=extra)

        seg = recorder.metadata["segments"][0]
        assert seg["speaker"] == "Speaker A"
        assert seg["language"] == "es"

    @pytest.mark.unit
    def test_save_pcm_segment_disabled_does_nothing(self, disabled_config, pcm_one_second):
        rec = AudioRecorder(disabled_config, "sess-off")
        rec.save_pcm_segment(pcm_one_second)
        # No exception, no attributes to check beyond enabled=False
        assert rec.enabled is False


# ---------------------------------------------------------------------------
# TestStopRecording
# ---------------------------------------------------------------------------

class TestStopRecording:

    @pytest.mark.unit
    def test_stop_recording_closes_raw_file(self, recorder):
        recorder.start_recording()
        recorder.write_raw_chunk(b"\x00" * 50)
        assert not recorder.raw_handle.closed

        recorder.stop_recording()
        assert recorder.raw_handle.closed

    @pytest.mark.unit
    def test_stop_recording_saves_metadata_json(self, recorder, pcm_one_second):
        recorder.save_pcm_segment(pcm_one_second)
        recorder.stop_recording()

        metadata_file = recorder.session_dir / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert saved["session_id"] == "sess-001"
        assert saved["total_segments"] == 1
        assert len(saved["segments"]) == 1
        assert saved["segments"][0]["filename"] == "segment_001.wav"

    @pytest.mark.unit
    def test_stop_recording_disabled_does_nothing(self, disabled_config):
        rec = AudioRecorder(disabled_config, "sess-off")
        # Should not raise
        rec.stop_recording()

    @pytest.mark.unit
    def test_stop_recording_metadata_has_ended_at(self, recorder):
        recorder.stop_recording()

        metadata_file = recorder.session_dir / "metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert "ended_at" in saved
        assert "started_at" in saved
        # ended_at should be a parseable ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(saved["ended_at"])


# ---------------------------------------------------------------------------
# TestCleanupOldRecordings
# ---------------------------------------------------------------------------

class TestCleanupOldRecordings:

    @pytest.mark.unit
    def test_cleanup_removes_oldest_over_limit(self, recording_config):
        import time

        # max_recordings is 3, so create 4 session dirs to trigger cleanup
        output_dir = recording_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        session_dirs = []
        for i in range(4):
            d = output_dir / f"session_{i}_20260101_00000{i}"
            d.mkdir()
            # Write a dummy file so rmdir works after file deletion
            (d / "dummy.txt").write_text("x")
            session_dirs.append(d)
            # Small sleep so st_ctime differs
            time.sleep(0.05)

        # The 5th recorder triggers cleanup on stop
        rec = AudioRecorder(recording_config, "session_4")
        rec.stop_recording()

        # After cleanup, total dirs should be <= max_recordings (3)
        remaining = list(output_dir.iterdir())
        assert len(remaining) <= recording_config.max_recordings

    @pytest.mark.unit
    def test_cleanup_no_limit_does_nothing(self, tmp_path):
        config = RecordingConfig(
            enabled=True,
            output_dir=tmp_path / "recordings",
            max_recordings=0,  # 0 means no limit (falsy)
            include_metadata=False,
        )
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create several session dirs
        for i in range(5):
            d = output_dir / f"sess_{i}_20260101_00000{i}"
            d.mkdir()

        rec = AudioRecorder(config, "sess_extra")
        rec.stop_recording()

        # All dirs should still exist (5 pre-existing + 1 from the recorder)
        remaining = list(output_dir.iterdir())
        assert len(remaining) == 6


# ---------------------------------------------------------------------------
# TestAddSegmentMetadata
# ---------------------------------------------------------------------------

class TestAddSegmentMetadata:

    @pytest.mark.unit
    def test_add_segment_metadata_valid_id(self, recorder, pcm_one_second):
        recorder.save_pcm_segment(pcm_one_second)
        recorder.add_segment_metadata(1, "transcript", "Hola mundo")

        seg = recorder.metadata["segments"][0]
        assert seg["transcript"] == "Hola mundo"

    @pytest.mark.unit
    def test_add_segment_metadata_invalid_id_does_nothing(self, recorder, pcm_one_second):
        recorder.save_pcm_segment(pcm_one_second)
        # segment_id 99 is out of range
        recorder.add_segment_metadata(99, "transcript", "Should not appear")

        seg = recorder.metadata["segments"][0]
        assert "transcript" not in seg


# ---------------------------------------------------------------------------
# TestRecorderManager
# ---------------------------------------------------------------------------

class TestRecorderManager:

    @pytest.mark.unit
    def test_create_recorder_returns_recorder(self, recording_config):
        mgr = RecorderManager(recording_config)
        rec = mgr.create_recorder("sess-a")

        assert isinstance(rec, AudioRecorder)
        assert rec.enabled is True
        assert "sess-a" in mgr.recorders

    @pytest.mark.unit
    def test_create_recorder_disabled_not_stored(self, disabled_config):
        mgr = RecorderManager(disabled_config)
        rec = mgr.create_recorder("sess-off")

        assert isinstance(rec, AudioRecorder)
        assert rec.enabled is False
        assert "sess-off" not in mgr.recorders

    @pytest.mark.unit
    def test_get_recorder_returns_existing(self, recording_config):
        mgr = RecorderManager(recording_config)
        mgr.create_recorder("sess-b")

        found = mgr.get_recorder("sess-b")
        assert found is not None
        assert found.session_id == "sess-b"

    @pytest.mark.unit
    def test_get_recorder_returns_none_for_unknown(self, recording_config):
        mgr = RecorderManager(recording_config)
        assert mgr.get_recorder("nonexistent") is None

    @pytest.mark.unit
    def test_remove_recorder_stops_and_deletes(self, recording_config):
        mgr = RecorderManager(recording_config)
        rec = mgr.create_recorder("sess-c")
        rec.start_recording()
        rec.write_raw_chunk(b"\x00" * 20)

        mgr.remove_recorder("sess-c")

        assert "sess-c" not in mgr.recorders
        # Raw file handle should be closed by stop_recording
        assert rec.raw_handle.closed
        # Metadata JSON should have been written
        assert (rec.session_dir / "metadata.json").exists()

    @pytest.mark.unit
    def test_stop_all_stops_all_recorders(self, recording_config):
        mgr = RecorderManager(recording_config)
        rec_a = mgr.create_recorder("sess-x")
        rec_b = mgr.create_recorder("sess-y")

        rec_a.start_recording()
        rec_b.start_recording()

        mgr.stop_all()

        assert len(mgr.recorders) == 0
        assert rec_a.raw_handle.closed
        assert rec_b.raw_handle.closed
        assert (rec_a.session_dir / "metadata.json").exists()
        assert (rec_b.session_dir / "metadata.json").exists()
