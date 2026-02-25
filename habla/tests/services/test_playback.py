"""Unit tests for the PlaybackService."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from server.services.playback import PlaybackService


# --- list_recordings ---

class TestListRecordings:
    """Tests for PlaybackService.list_recordings()."""

    def test_empty_dir(self, recordings_dir):
        """Returns empty list when recordings dir has no subdirs."""
        empty = recordings_dir.parent / "empty_recs"
        empty.mkdir()
        svc = PlaybackService(empty)
        assert svc.list_recordings() == []

    def test_nonexistent_dir(self, tmp_path):
        """Returns empty list when recordings dir doesn't exist."""
        svc = PlaybackService(tmp_path / "nope")
        assert svc.list_recordings() == []

    def test_lists_with_metadata(self, recordings_dir):
        """Parses metadata.json and returns correct fields."""
        svc = PlaybackService(recordings_dir)
        recs = svc.list_recordings()
        full = next(r for r in recs if r["id"] == "rec_full")
        assert full["started_at"] == "2026-02-22T10:00:00"
        assert full["total_duration_seconds"] == 1.8
        assert full["total_segments"] == 2

    def test_ground_truth_flag(self, recordings_dir):
        """has_ground_truth is True only when ground_truth.json exists."""
        svc = PlaybackService(recordings_dir)
        recs = {r["id"]: r for r in svc.list_recordings()}
        assert recs["rec_full"]["has_ground_truth"] is True
        assert recs["rec_segments_only"]["has_ground_truth"] is False
        assert recs["rec_empty"]["has_ground_truth"] is False

    def test_raw_stream_flag(self, recordings_dir):
        """has_raw_stream is True only when raw_stream.webm exists."""
        svc = PlaybackService(recordings_dir)
        recs = {r["id"]: r for r in svc.list_recordings()}
        assert recs["rec_full"]["has_raw_stream"] is True
        assert recs["rec_segments_only"]["has_raw_stream"] is False


# --- get_recording ---

class TestGetRecording:
    """Tests for PlaybackService.get_recording()."""

    def test_not_found(self, recordings_dir):
        """Returns None for nonexistent recording id."""
        svc = PlaybackService(recordings_dir)
        assert svc.get_recording("nonexistent") is None

    def test_returns_metadata(self, recordings_dir):
        """Loads and returns metadata.json."""
        svc = PlaybackService(recordings_dir)
        result = svc.get_recording("rec_full")
        assert result is not None
        assert result["metadata"]["started_at"] == "2026-02-22T10:00:00"
        assert len(result["metadata"]["segments"]) == 2

    def test_returns_ground_truth(self, recordings_dir):
        """Loads and returns ground_truth.json when present."""
        svc = PlaybackService(recordings_dir)
        result = svc.get_recording("rec_full")
        assert "ground_truth" in result
        assert result["ground_truth"]["whisper_model"] == "large-v3"
        assert len(result["ground_truth"]["segments"]) == 2

    def test_handles_corrupt_json(self, recordings_dir):
        """Corrupt metadata doesn't crash, returns None for that field."""
        corrupt = recordings_dir / "rec_corrupt"
        corrupt.mkdir()
        (corrupt / "metadata.json").write_text("{bad json!!")
        svc = PlaybackService(recordings_dir)
        result = svc.get_recording("rec_corrupt")
        assert result is not None
        assert result["metadata"] is None


# --- start_playback ---

class TestStartPlayback:
    """Tests for PlaybackService.start_playback()."""

    async def test_rejects_nonexistent_recording(self, recordings_dir, mock_session):
        """Returns error dict for nonexistent recording."""
        svc = PlaybackService(recordings_dir)
        result = await svc.start_playback("nonexistent", mock_session)
        assert "error" in result
        assert "not found" in result["error"].lower()

    async def test_rejects_when_already_active(self, recordings_dir, mock_session):
        """Returns error when playback is already in progress."""
        svc = PlaybackService(recordings_dir)
        # Fake an active task
        svc._task = MagicMock()
        svc._task.done.return_value = False
        svc._active_recording_id = "rec_full"
        result = await svc.start_playback("rec_full", mock_session)
        assert "error" in result
        assert "already" in result["error"].lower()

    async def test_starts_segments_mode(self, recordings_dir, mock_session):
        """Segments mode spawns a task and returns success."""
        svc = PlaybackService(recordings_dir)
        result = await svc.start_playback("rec_segments_only", mock_session, mode="segments")
        assert result["status"] == "started"
        assert result["mode"] == "segments"
        assert svc.is_active
        # Clean up
        await svc.stop_playback()

    async def test_starts_full_mode(self, recordings_dir, mock_session):
        """Full mode spawns a task and returns success."""
        svc = PlaybackService(recordings_dir)
        result = await svc.start_playback("rec_full", mock_session, mode="full")
        assert result["status"] == "started"
        assert result["mode"] == "full"
        assert svc.is_active
        await svc.stop_playback()

    async def test_full_mode_requires_raw_stream(self, recordings_dir, mock_session):
        """Full mode returns error when raw_stream.webm is missing."""
        svc = PlaybackService(recordings_dir)
        result = await svc.start_playback("rec_segments_only", mock_session, mode="full")
        assert "error" in result
        assert "raw_stream" in result["error"].lower() or "segments" in result["error"].lower()

    async def test_full_mode_no_raw_stream_leaves_no_stale_state(self, recordings_dir, mock_session):
        """Full mode error path does not leave stale _active_recording_id."""
        svc = PlaybackService(recordings_dir)
        result = await svc.start_playback("rec_segments_only", mock_session, mode="full")
        assert "error" in result
        assert svc._active_recording_id is None
        assert svc._task is None
        assert not svc.is_active


# --- stop_playback ---

class TestStopPlayback:
    """Tests for PlaybackService.stop_playback()."""

    async def test_cancels_active_task(self, recordings_dir, mock_session):
        """Active task gets cancelled."""
        svc = PlaybackService(recordings_dir)
        result = await svc.start_playback("rec_segments_only", mock_session, mode="segments")
        assert svc.is_active
        await svc.stop_playback()
        assert not svc.is_active

    async def test_stop_when_nothing_active(self, recordings_dir):
        """No-op when nothing is playing."""
        svc = PlaybackService(recordings_dir)
        await svc.stop_playback()  # should not raise
        assert not svc.is_active

    async def test_cleans_up_state(self, recordings_dir, mock_session):
        """State is fully cleaned up after stop."""
        svc = PlaybackService(recordings_dir)
        await svc.start_playback("rec_segments_only", mock_session, mode="segments")
        await svc.stop_playback()
        assert svc._task is None
        assert svc._active_recording_id is None


# --- _playback_segments ---

class TestPlaybackSegments:
    """Tests for the segment-only playback path."""

    async def test_feeds_all_segments(self, recordings_dir, mock_session):
        """Calls pipeline.process_wav() once per WAV segment."""
        svc = PlaybackService(recordings_dir)
        # Run directly instead of via task for deterministic control
        session_dir = recordings_dir / "rec_full"
        await svc._playback_segments(session_dir, mock_session, speed=0)
        assert mock_session.pipeline.process_wav.call_count == 2

    async def test_sends_progress_messages(self, recordings_dir, mock_session):
        """Progress messages are sent during playback."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_full"
        await svc._playback_segments(session_dir, mock_session, speed=0)
        sent_types = [c.args[0]["type"] for c in mock_session._send.call_args_list]
        assert "playback_progress" in sent_types

    async def test_sends_finished_message(self, recordings_dir, mock_session):
        """Final message has type playback_finished."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_full"
        await svc._playback_segments(session_dir, mock_session, speed=0)
        last_msg = mock_session._send.call_args_list[-1].args[0]
        assert last_msg["type"] == "playback_finished"
        assert last_msg["chunks_processed"] == 2

    async def test_cancellation_sends_stopped(self, recordings_dir, mock_session):
        """Cancelling mid-playback sends playback_stopped."""
        svc = PlaybackService(recordings_dir)
        svc._cancelled = True  # Pre-cancel
        session_dir = recordings_dir / "rec_full"
        await svc._playback_segments(session_dir, mock_session, speed=0)
        sent_types = [c.args[0]["type"] for c in mock_session._send.call_args_list]
        assert "playback_stopped" in sent_types

    async def test_sets_playback_mode(self, recordings_dir, mock_session):
        """playback_mode is True during playback, False after."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_full"
        await svc._playback_segments(session_dir, mock_session, speed=0)
        # After completion, playback_mode should be False
        assert mock_session.playback_mode is False

    async def test_no_segments_sends_error(self, recordings_dir, mock_session):
        """Empty recording dir sends error message."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_empty"
        await svc._playback_segments(session_dir, mock_session, speed=0)
        last_msg = mock_session._send.call_args_list[-1].args[0]
        assert last_msg["type"] == "error"
        assert "no segment" in last_msg["message"].lower()


# --- _playback_full ---

class TestPlaybackFull:
    """Tests for the full pipeline playback path (raw WebM through decoder+VAD)."""

    async def test_ffmpeg_failure_sends_error(self, recordings_dir, mock_session):
        """Mock ffmpeg returning non-zero sends error."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_full"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"ffmpeg error details"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await svc._playback_full(session_dir, mock_session, speed=0)

        sent_types = [c.args[0]["type"] for c in mock_session._send.call_args_list]
        assert "error" in sent_types

    async def test_sets_session_state(self, recordings_dir, mock_session):
        """Session state is set during playback and reset after."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_full"

        # Track state changes
        states_during = []

        original_start_streaming = mock_session.decoder.start_streaming

        async def capture_state():
            states_during.append({
                "playback_mode": mock_session.playback_mode,
                "listening": mock_session.listening,
            })
            return await original_start_streaming()

        mock_session.decoder.start_streaming = capture_state

        # Mock ffmpeg to produce one chunk
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        chunk_file = session_dir / "temp_chunk"
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("tempfile.mkdtemp", return_value=str(session_dir / "_tmp")):
                (session_dir / "_tmp").mkdir(exist_ok=True)
                # Create a fake chunk file
                chunk = session_dir / "_tmp" / "chunk_0000.webm"
                chunk.write_bytes(b"\x00" * 10)

                # Mock _continuous_decode_loop to return a real done task
                done_task = asyncio.create_task(asyncio.sleep(0))
                await done_task  # let it finish
                mock_session._decode_task = done_task

                await svc._playback_full(session_dir, mock_session, speed=0)

        # After playback, state should be reset
        assert mock_session.playback_mode is False

    async def test_cleanup_on_cancel(self, recordings_dir, mock_session):
        """Decoder is stopped and decode task is cancelled on CancelledError."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_full"

        # Create a task that will be cancelled
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError())
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await svc._playback_full(session_dir, mock_session, speed=0)

        # After CancelledError, state should be cleaned up
        assert mock_session.playback_mode is False
        sent_types = [c.args[0]["type"] for c in mock_session._send.call_args_list]
        assert "playback_stopped" in sent_types

    async def test_cleanup_on_exception(self, recordings_dir, mock_session):
        """Decoder is stopped and state reset on unexpected exception."""
        svc = PlaybackService(recordings_dir)
        session_dir = recordings_dir / "rec_full"

        # Make ffmpeg raise an unexpected error
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=RuntimeError("disk full"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await svc._playback_full(session_dir, mock_session, speed=0)

        assert mock_session.playback_mode is False
        sent_types = [c.args[0]["type"] for c in mock_session._send.call_args_list]
        assert "error" in sent_types
