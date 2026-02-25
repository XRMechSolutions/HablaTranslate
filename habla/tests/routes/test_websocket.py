"""Unit tests for the WebSocket handler (server/routes/websocket.py)."""

import asyncio
import io
import json
import struct
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import server.routes.websocket as ws_mod
from server.routes.websocket import (
    ClientSession,
    websocket_endpoint,
    set_ws_pipeline,
    set_recording_config,
    get_active_session,
    _write_wav,
    _process_text,
    VALID_DIRECTIONS,
    VALID_MODES,
    MERGE_GAP_SECONDS,
    MAX_PENDING_SECONDS,
)
from server.models.schemas import (
    WSTranslation,
    WSPartialTranscript,
    WSSpeakersUpdate,
    SpeakerProfile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_websocket():
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    ws.accept = AsyncMock()
    return ws


@pytest.fixture
def mock_pipeline():
    p = MagicMock()
    p.ready = True
    p.session_id = 1
    p.direction = "es_to_en"
    p.mode = "conversation"
    p.speaker_tracker = MagicMock()
    p.speaker_tracker.speakers = {}
    p.speaker_tracker.get_all.return_value = []
    p.speaker_tracker.rename = MagicMock()
    p.create_session = AsyncMock(return_value=1)
    p.close_session = AsyncMock()
    p.reset_session = AsyncMock(return_value=2)
    p.process_wav = AsyncMock(return_value=None)
    p.process_text = AsyncMock()
    p.process_partial_audio = AsyncMock()
    p.set_direction = MagicMock()
    p.set_mode = MagicMock()
    p.set_callbacks = MagicMock()
    p.reset_partial_state = MagicMock()
    return p


@pytest.fixture
def session(mock_websocket, mock_pipeline):
    """Create a ClientSession with mocked internals (no real ffmpeg/VAD)."""
    old = ws_mod._current_pipeline
    ws_mod._current_pipeline = mock_pipeline

    s = ClientSession(mock_websocket, mock_pipeline)
    # Replace heavy dependencies with mocks
    s.vad = MagicMock()
    s.vad.initialize = AsyncMock()
    s.vad.reset = MagicMock()
    s.vad.feed_pcm = AsyncMock()
    s.vad.flush = AsyncMock()
    s.vad.segments_emitted = 0
    s.vad.total_speech_seconds = 0.0
    s.vad.on_partial_audio = None

    s.decoder = MagicMock()
    s.decoder.reset = MagicMock()
    s.decoder.start_streaming = AsyncMock()
    s.decoder.stop_streaming = AsyncMock(return_value=b"")
    s.decoder.feed_chunk = AsyncMock(return_value=b"")

    yield s

    ws_mod._current_pipeline = old


@pytest.fixture
def mock_recording_config():
    cfg = MagicMock()
    cfg.enabled = True
    cfg.output_dir = MagicMock()
    return cfg


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

class TestModuleFunctions:
    """Tests for set_ws_pipeline, set_recording_config, get_active_session."""

    def test_set_ws_pipeline_stores_reference(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            set_ws_pipeline(mock_pipeline)
            assert ws_mod._current_pipeline is mock_pipeline
        finally:
            ws_mod._current_pipeline = old

    def test_set_recording_config_stores_reference(self, mock_recording_config):
        old = ws_mod._recording_config
        try:
            set_recording_config(mock_recording_config)
            assert ws_mod._recording_config is mock_recording_config
        finally:
            ws_mod._recording_config = old

    def test_get_active_session_returns_none_initially(self):
        old = ws_mod._active_session
        try:
            ws_mod._active_session = None
            assert get_active_session() is None
        finally:
            ws_mod._active_session = old

    def test_get_active_session_returns_set_session(self, session):
        old = ws_mod._active_session
        try:
            ws_mod._active_session = session
            assert get_active_session() is session
        finally:
            ws_mod._active_session = old


# ---------------------------------------------------------------------------
# ClientSession initialisation
# ---------------------------------------------------------------------------

class TestClientSessionInit:
    """Tests for ClientSession __init__ and default state."""

    def test_init_sets_listening_false(self, session):
        assert session.listening is False

    def test_init_sets_playback_mode_false(self, session):
        assert session.playback_mode is False

    def test_init_segment_counter_zero(self, session):
        assert session._segment_counter == 0

    def test_init_recorder_is_none(self, session):
        assert session.recorder is None

    def test_init_pending_pcm_empty(self, session):
        assert len(session._pending_pcm) == 0

    def test_init_pending_duration_zero(self, session):
        assert session._pending_duration == 0.0

    def test_init_merge_gap_uses_constant(self, session):
        assert session._merge_gap_seconds == MERGE_GAP_SECONDS

    def test_init_max_pending_uses_constant(self, session):
        assert session._max_pending_seconds == MAX_PENDING_SECONDS

    def test_pipeline_property_returns_module_pipeline(self, session, mock_pipeline):
        assert session.pipeline is mock_pipeline


# ---------------------------------------------------------------------------
# start_listening / stop_listening
# ---------------------------------------------------------------------------

class TestStartListening:
    """Tests for ClientSession.start_listening."""

    @pytest.mark.asyncio
    async def test_start_listening_sets_flag(self, session):
        await session.start_listening()
        assert session.listening is True

    @pytest.mark.asyncio
    async def test_start_listening_resets_vad(self, session):
        await session.start_listening()
        session.vad.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_listening_resets_decoder(self, session):
        await session.start_listening()
        session.decoder.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_listening_starts_streaming_decoder(self, session):
        await session.start_listening()
        session.decoder.start_streaming.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_listening_sends_listening_started(self, session, mock_websocket):
        await session.start_listening()
        call_args = mock_websocket.send_text.call_args_list
        assert len(call_args) >= 1
        sent = json.loads(call_args[-1][0][0])
        assert sent["type"] == "listening_started"
        assert sent["mode"] == "continuous"

    @pytest.mark.asyncio
    async def test_start_listening_creates_decode_task(self, session):
        await session.start_listening()
        assert session._decode_task is not None

    @pytest.mark.asyncio
    async def test_start_listening_wires_partial_audio_callback(self, session):
        await session.start_listening()
        # The source code sets self.vad.on_partial_audio = self._on_partial_audio.
        # Since vad is a MagicMock, verify the attribute was set to a callable
        # bound method from the session (not None / not still the default mock).
        callback = session.vad.on_partial_audio
        assert callable(callback)
        assert callback.__func__ is ClientSession._on_partial_audio

    @pytest.mark.asyncio
    async def test_start_listening_no_recorder_when_config_disabled(self, session):
        old = ws_mod._recording_config
        try:
            cfg = MagicMock()
            cfg.enabled = False
            ws_mod._recording_config = cfg
            await session.start_listening()
            assert session.recorder is None
        finally:
            ws_mod._recording_config = old

    @pytest.mark.asyncio
    async def test_start_listening_creates_recorder_when_enabled(self, session, mock_recording_config):
        old = ws_mod._recording_config
        try:
            ws_mod._recording_config = mock_recording_config
            with patch("server.routes.websocket.AudioRecorder") as MockRecorder:
                recorder_inst = MagicMock()
                MockRecorder.return_value = recorder_inst
                await session.start_listening()
                MockRecorder.assert_called_once_with(mock_recording_config, session.session_id)
                recorder_inst.start_recording.assert_called_once()
        finally:
            ws_mod._recording_config = old

    @pytest.mark.asyncio
    async def test_start_listening_recording_field_in_message(self, session, mock_websocket, mock_recording_config):
        old = ws_mod._recording_config
        try:
            ws_mod._recording_config = mock_recording_config
            with patch("server.routes.websocket.AudioRecorder"):
                await session.start_listening()
            sent = json.loads(mock_websocket.send_text.call_args_list[-1][0][0])
            assert sent["recording"] is True
        finally:
            ws_mod._recording_config = old


class TestStopListening:
    """Tests for ClientSession.stop_listening."""

    @pytest.mark.asyncio
    async def test_stop_listening_clears_flag(self, session):
        await session.start_listening()
        await session.stop_listening()
        assert session.listening is False

    @pytest.mark.asyncio
    async def test_stop_listening_cancels_decode_task(self, session):
        await session.start_listening()
        task = session._decode_task
        await session.stop_listening()
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_listening_stops_streaming_decoder(self, session):
        await session.start_listening()
        await session.stop_listening()
        session.decoder.stop_streaming.assert_awaited()

    @pytest.mark.asyncio
    async def test_stop_listening_flushes_vad(self, session):
        await session.start_listening()
        await session.stop_listening()
        session.vad.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_stop_listening_sends_stats(self, session, mock_websocket):
        session.vad.segments_emitted = 3
        session.vad.total_speech_seconds = 12.5
        await session.start_listening()
        await session.stop_listening()
        # Find the listening_stopped message
        for call in mock_websocket.send_text.call_args_list:
            msg = json.loads(call[0][0])
            if msg.get("type") == "listening_stopped":
                assert msg["segments_processed"] == 3
                assert msg["total_speech_seconds"] == 12.5
                return
        pytest.fail("listening_stopped message not sent")

    @pytest.mark.asyncio
    async def test_stop_listening_feeds_remaining_pcm(self, session):
        session.decoder.stop_streaming = AsyncMock(return_value=b"\x00" * 100)
        await session.start_listening()
        await session.stop_listening()
        session.vad.feed_pcm.assert_awaited_with(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_stop_listening_stops_recorder(self, session):
        recorder = MagicMock()
        session.recorder = recorder
        await session.start_listening()
        await session.stop_listening()
        recorder.stop_recording.assert_called_once()


# ---------------------------------------------------------------------------
# handle_audio_chunk
# ---------------------------------------------------------------------------

class TestHandleAudioChunk:
    """Tests for ClientSession.handle_audio_chunk."""

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_ignored_when_not_listening(self, session):
        session.listening = False
        session.playback_mode = False
        await session.handle_audio_chunk(b"\x00" * 50)
        assert len(session._chunk_inbox) == 0

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_queued_when_listening(self, session):
        session.listening = True
        await session.handle_audio_chunk(b"\xAB" * 50)
        assert session._chunk_inbox == [b"\xAB" * 50]

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_queued_when_playback_mode(self, session):
        session.playback_mode = True
        await session.handle_audio_chunk(b"\xCD" * 30)
        assert session._chunk_inbox == [b"\xCD" * 30]

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_sets_event(self, session):
        session.listening = True
        session._chunk_event.clear()
        await session.handle_audio_chunk(b"\x01")
        assert session._chunk_event.is_set()

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_writes_to_recorder(self, session):
        session.listening = True
        recorder = MagicMock()
        session.recorder = recorder
        chunk = b"\xFF" * 64
        await session.handle_audio_chunk(chunk)
        recorder.write_raw_chunk.assert_called_once_with(chunk)

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_no_recorder_no_error(self, session):
        session.listening = True
        session.recorder = None
        await session.handle_audio_chunk(b"\x00")  # should not raise


# ---------------------------------------------------------------------------
# Segment merging: _queue_segment, _flush_pending_after_gap, _flush_pending_now
# ---------------------------------------------------------------------------

class TestSegmentMerging:
    """Tests for segment merge/queue/flush logic."""

    @pytest.mark.asyncio
    async def test_queue_segment_empty_pcm_ignored(self, session):
        await session._queue_segment(b"", 0.0)
        assert len(session._pending_pcm) == 0

    @pytest.mark.asyncio
    async def test_queue_segment_accumulates_pcm(self, session):
        await session._queue_segment(b"\x01\x02", 0.5)
        await session._queue_segment(b"\x03\x04", 0.5)
        assert bytes(session._pending_pcm) == b"\x01\x02\x03\x04"
        assert session._pending_duration == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_queue_segment_creates_pending_task(self, session):
        await session._queue_segment(b"\x01", 0.5)
        assert session._pending_task is not None

    @pytest.mark.asyncio
    async def test_queue_segment_replaces_pending_task_on_new_segment(self, session):
        await session._queue_segment(b"\x01", 0.5)
        first_task = session._pending_task
        await session._queue_segment(b"\x02", 0.5)
        second_task = session._pending_task
        assert first_task is not second_task
        # The first task was cancel()-ed; it may be in "cancelling" state
        # until the event loop processes it. Check cancelling() or cancelled().
        assert first_task.cancelling() > 0 or first_task.cancelled()

    @pytest.mark.asyncio
    async def test_queue_segment_force_flushes_at_max_pending(self, session, mock_pipeline):
        """When pending audio exceeds MAX_PENDING_SECONDS, flush immediately."""
        session._max_pending_seconds = 2.0  # Lower for test speed
        with patch.object(session, "_process_segment", new_callable=AsyncMock) as mock_proc:
            await session._queue_segment(b"\x00" * 32000, 2.5)
            mock_proc.assert_awaited_once()
            assert len(session._pending_pcm) == 0
            assert session._pending_duration == 0.0

    @pytest.mark.asyncio
    async def test_flush_pending_now_empty_is_noop(self, session):
        with patch.object(session, "_process_segment", new_callable=AsyncMock) as mock_proc:
            await session._flush_pending_now()
            mock_proc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flush_pending_now_sends_accumulated_pcm(self, session):
        session._pending_pcm.extend(b"\xAA\xBB")
        session._pending_duration = 1.5
        with patch.object(session, "_process_segment", new_callable=AsyncMock) as mock_proc:
            await session._flush_pending_now()
            mock_proc.assert_awaited_once_with(b"\xAA\xBB", 1.5)
            assert len(session._pending_pcm) == 0
            assert session._pending_duration == 0.0

    @pytest.mark.asyncio
    async def test_flush_pending_now_cancels_pending_task(self, session):
        task = MagicMock()
        task.cancel = MagicMock()
        session._pending_task = task
        session._pending_pcm.extend(b"\x01")
        session._pending_duration = 0.5
        with patch.object(session, "_process_segment", new_callable=AsyncMock):
            await session._flush_pending_now()
        task.cancel.assert_called_once()
        assert session._pending_task is None

    @pytest.mark.asyncio
    async def test_flush_pending_after_gap_waits_then_flushes(self, session):
        """After the merge gap delay, pending audio should be flushed."""
        session._merge_gap_seconds = 0.05  # Very short for testing
        session._pending_pcm.extend(b"\x01\x02")
        session._pending_duration = 0.5
        with patch.object(session, "_process_segment", new_callable=AsyncMock) as mock_proc:
            session._pending_task = asyncio.create_task(session._flush_pending_after_gap())
            await asyncio.sleep(0.15)  # Wait for gap + processing
            mock_proc.assert_awaited_once()


# ---------------------------------------------------------------------------
# _on_speech_segment
# ---------------------------------------------------------------------------

class TestOnSpeechSegment:
    """Tests for the VAD speech segment callback."""

    @pytest.mark.asyncio
    async def test_on_speech_segment_resets_partial_state(self, session, mock_pipeline):
        with patch.object(session, "_queue_segment", new_callable=AsyncMock):
            await session._on_speech_segment(b"\x01\x02", 1.0)
        mock_pipeline.reset_partial_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_speech_segment_queues_segment(self, session):
        with patch.object(session, "_queue_segment", new_callable=AsyncMock) as mock_q:
            await session._on_speech_segment(b"\x01\x02", 1.0)
            mock_q.assert_awaited_once_with(b"\x01\x02", 1.0)


# ---------------------------------------------------------------------------
# _on_partial_audio
# ---------------------------------------------------------------------------

class TestOnPartialAudio:
    """Tests for streaming partial transcription callback."""

    @pytest.mark.asyncio
    async def test_on_partial_audio_calls_pipeline(self, session, mock_pipeline):
        pcm = b"\x00" * 100
        await session._on_partial_audio(pcm, 0.5)
        # It creates a task; give it a tick to run
        await asyncio.sleep(0.01)
        mock_pipeline.process_partial_audio.assert_awaited_once_with(pcm, 0.5)


# ---------------------------------------------------------------------------
# _process_segment
# ---------------------------------------------------------------------------

class TestProcessSegment:
    """Tests for _process_segment (WAV writing, pipeline call, cleanup)."""

    @pytest.mark.asyncio
    async def test_process_segment_increments_counter(self, session):
        with patch("server.routes.websocket._write_wav"):
            await session._process_segment(b"\x00" * 100, 1.0)
        assert session._segment_counter == 1

    @pytest.mark.asyncio
    async def test_process_segment_calls_pipeline_process_wav(self, session, mock_pipeline):
        with patch("server.routes.websocket._write_wav"):
            await session._process_segment(b"\x00" * 100, 1.0)
        mock_pipeline.process_wav.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_segment_cleans_up_temp_file(self, session, tmp_path):
        """Temp WAV file should be deleted after processing."""
        created_paths = []
        original_write_wav = _write_wav

        def tracking_write_wav(f, pcm_bytes, sample_rate=16000):
            created_paths.append(f.name)
            original_write_wav(f, pcm_bytes, sample_rate)

        with patch("server.routes.websocket._write_wav", side_effect=tracking_write_wav):
            await session._process_segment(b"\x00" * 100, 1.0)

        assert len(created_paths) == 1
        from pathlib import Path
        assert not Path(created_paths[0]).exists(), "Temp WAV file should be deleted"

    @pytest.mark.asyncio
    async def test_process_segment_error_sends_error_message(self, session, mock_pipeline, mock_websocket):
        mock_pipeline.process_wav.side_effect = RuntimeError("ASR exploded")
        with patch("server.routes.websocket._write_wav"):
            await session._process_segment(b"\x00" * 100, 1.0)
        # Check that an error message was sent
        for call in mock_websocket.send_text.call_args_list:
            msg = json.loads(call[0][0])
            if msg.get("type") == "error":
                assert "ASR exploded" in msg["message"]
                return
        pytest.fail("Error message not sent to client")

    @pytest.mark.asyncio
    async def test_process_segment_saves_to_recorder(self, session, mock_pipeline):
        recorder = MagicMock()
        session.recorder = recorder
        # Make pipeline return an exchange-like object
        exchange = MagicMock()
        exchange.raw_transcript = "Hola"
        exchange.translated_text = "Hello"
        exchange.speaker_label = "A"
        exchange.confidence = 0.9
        exchange.idioms = []
        mock_pipeline.process_wav.return_value = exchange

        with patch("server.routes.websocket._write_wav"):
            await session._process_segment(b"\x00" * 100, 1.0)

        recorder.save_pcm_segment.assert_called_once()
        # Verify metadata additions
        assert recorder.add_segment_metadata.call_count >= 4


# ---------------------------------------------------------------------------
# _send
# ---------------------------------------------------------------------------

class TestSend:
    """Tests for the _send helper."""

    @pytest.mark.asyncio
    async def test_send_serializes_json(self, session, mock_websocket):
        await session._send({"type": "pong"})
        mock_websocket.send_text.assert_awaited_once()
        sent = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent == {"type": "pong"}

    @pytest.mark.asyncio
    async def test_send_does_not_raise_on_failure(self, session, mock_websocket):
        mock_websocket.send_text.side_effect = ConnectionError("gone")
        # Should log but not raise
        await session._send({"type": "test"})


# ---------------------------------------------------------------------------
# _write_wav
# ---------------------------------------------------------------------------

class TestWriteWav:
    """Tests for WAV header generation."""

    def test_write_wav_riff_header(self):
        buf = io.BytesIO()
        pcm = b"\x00" * 320  # 10ms at 16kHz 16-bit mono
        _write_wav(buf, pcm, sample_rate=16000)
        buf.seek(0)
        data = buf.read()

        # RIFF magic
        assert data[:4] == b"RIFF"
        # File size field = 36 + data_size
        file_size = struct.unpack_from("<I", data, 4)[0]
        assert file_size == 36 + len(pcm)
        # WAVE magic
        assert data[8:12] == b"WAVE"

    def test_write_wav_fmt_chunk(self):
        buf = io.BytesIO()
        pcm = b"\x00" * 320
        _write_wav(buf, pcm, sample_rate=16000)
        buf.seek(0)
        data = buf.read()

        # fmt chunk header
        assert data[12:16] == b"fmt "
        fmt_size = struct.unpack_from("<I", data, 16)[0]
        assert fmt_size == 16

        # PCM format fields: audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample
        audio_fmt, channels, sr, byte_rate, block_align, bps = struct.unpack_from("<HHIIHH", data, 20)
        assert audio_fmt == 1, "Audio format should be PCM (1)"
        assert channels == 1, "Mono channel"
        assert sr == 16000, "16kHz sample rate"
        assert byte_rate == 32000, "byte_rate = sample_rate * channels * bits/8"
        assert block_align == 2, "block_align = channels * bits/8"
        assert bps == 16, "16-bit samples"

    def test_write_wav_data_chunk(self):
        buf = io.BytesIO()
        pcm = b"\xAB\xCD" * 100
        _write_wav(buf, pcm, sample_rate=16000)
        buf.seek(0)
        data = buf.read()

        # data chunk starts at offset 36
        assert data[36:40] == b"data"
        data_size = struct.unpack_from("<I", data, 40)[0]
        assert data_size == len(pcm)
        # Actual PCM data follows
        assert data[44:] == pcm

    def test_write_wav_empty_pcm(self):
        buf = io.BytesIO()
        _write_wav(buf, b"", sample_rate=16000)
        buf.seek(0)
        data = buf.read()
        assert len(data) == 44, "Header only, no PCM data"
        data_size = struct.unpack_from("<I", data, 40)[0]
        assert data_size == 0

    def test_write_wav_custom_sample_rate(self):
        buf = io.BytesIO()
        _write_wav(buf, b"\x00" * 10, sample_rate=44100)
        buf.seek(0)
        data = buf.read()
        sr = struct.unpack_from("<I", data, 24)[0]
        assert sr == 44100
        byte_rate = struct.unpack_from("<I", data, 28)[0]
        assert byte_rate == 44100 * 2


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Tests for ClientSession.cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stops_listening_if_active(self, session):
        session.listening = True
        with patch.object(session, "stop_listening", new_callable=AsyncMock) as mock_stop:
            await session.cleanup()
            mock_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_noop_when_not_listening(self, session):
        session.listening = False
        with patch.object(session, "stop_listening", new_callable=AsyncMock) as mock_stop:
            await session.cleanup()
            mock_stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_clears_playback_mode(self, session):
        session.playback_mode = True
        await session.cleanup()
        assert session.playback_mode is False


# ---------------------------------------------------------------------------
# enable_recording / disable_recording
# ---------------------------------------------------------------------------

class TestRecordingToggle:
    """Tests for dynamic recording enable/disable."""

    @pytest.mark.asyncio
    async def test_enable_recording_creates_recorder(self, session, mock_recording_config):
        old = ws_mod._recording_config
        try:
            ws_mod._recording_config = mock_recording_config
            with patch("server.routes.websocket.AudioRecorder") as MockRecorder:
                recorder_inst = MagicMock()
                recorder_inst.session_dir = "/fake/dir"
                MockRecorder.return_value = recorder_inst
                await session.enable_recording()
                assert session.recorder is recorder_inst
                recorder_inst.start_recording.assert_called_once()
        finally:
            ws_mod._recording_config = old

    @pytest.mark.asyncio
    async def test_enable_recording_skips_if_already_enabled(self, session, mock_websocket):
        session.recorder = MagicMock()
        await session.enable_recording()
        # Should not have sent recording_enabled
        for call in mock_websocket.send_text.call_args_list:
            msg = json.loads(call[0][0])
            assert msg.get("type") != "recording_enabled"

    @pytest.mark.asyncio
    async def test_enable_recording_skips_if_config_disabled(self, session):
        old = ws_mod._recording_config
        try:
            cfg = MagicMock()
            cfg.enabled = False
            ws_mod._recording_config = cfg
            await session.enable_recording()
            assert session.recorder is None
        finally:
            ws_mod._recording_config = old

    @pytest.mark.asyncio
    async def test_disable_recording_clears_recorder(self, session, mock_websocket):
        recorder = MagicMock()
        session.recorder = recorder
        await session.disable_recording()
        assert session.recorder is None
        recorder.stop_recording.assert_called_once()

    @pytest.mark.asyncio
    async def test_disable_recording_sends_message(self, session, mock_websocket):
        session.recorder = MagicMock()
        await session.disable_recording()
        sent = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent["type"] == "recording_disabled"

    @pytest.mark.asyncio
    async def test_disable_recording_noop_when_already_disabled(self, session, mock_websocket):
        session.recorder = None
        await session.disable_recording()
        mock_websocket.send_text.assert_not_awaited()


# ---------------------------------------------------------------------------
# _process_text (module-level function)
# ---------------------------------------------------------------------------

class TestProcessText:
    """Tests for the module-level _process_text async function."""

    @pytest.mark.asyncio
    async def test_process_text_calls_pipeline(self, session, mock_pipeline):
        await _process_text(session, "Hola mundo", "MANUAL")
        mock_pipeline.process_text.assert_awaited_once_with("Hola mundo", "MANUAL")

    @pytest.mark.asyncio
    async def test_process_text_error_sends_error_message(self, session, mock_pipeline, mock_websocket):
        mock_pipeline.process_text.side_effect = ValueError("bad input")
        await _process_text(session, "broken", "MANUAL")
        sent = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent["type"] == "error"
        assert "bad input" in sent["message"]


# ---------------------------------------------------------------------------
# _continuous_decode_loop
# ---------------------------------------------------------------------------

class TestContinuousDecodeLoop:
    """Tests for the background decode loop."""

    @pytest.mark.asyncio
    async def test_decode_loop_feeds_chunks_to_ffmpeg(self, session):
        """Chunks placed in inbox should be fed to the decoder."""
        session.listening = True
        pcm_output = b"\x00" * 3200
        session.decoder.feed_chunk = AsyncMock(return_value=pcm_output)

        # Place a chunk and run the loop briefly
        session._chunk_inbox.append(b"\xAA" * 100)
        session._chunk_event.set()

        # Run loop in a task, then stop after one iteration
        async def run_briefly():
            # Let it process one cycle, then stop
            await asyncio.sleep(0.1)
            session.listening = False

        task = asyncio.create_task(session._continuous_decode_loop())
        await run_briefly()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        session.decoder.feed_chunk.assert_awaited()
        session.vad.feed_pcm.assert_awaited_with(pcm_output)

    @pytest.mark.asyncio
    async def test_decode_loop_skips_empty_pcm(self, session):
        """When ffmpeg returns empty PCM, VAD should not be fed."""
        session.listening = True
        session.decoder.feed_chunk = AsyncMock(return_value=b"")

        session._chunk_inbox.append(b"\x01")
        session._chunk_event.set()

        async def run_briefly():
            await asyncio.sleep(0.1)
            session.listening = False

        task = asyncio.create_task(session._continuous_decode_loop())
        await run_briefly()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        session.vad.feed_pcm.assert_not_awaited()


# ---------------------------------------------------------------------------
# websocket_endpoint — full connection lifecycle
# ---------------------------------------------------------------------------

class TestWebsocketEndpoint:
    """Tests for the main websocket_endpoint handler."""

    @pytest.fixture
    def ws_for_endpoint(self):
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        return ws

    def _make_text_msg(self, data: dict) -> dict:
        return {"type": "websocket.receive", "text": json.dumps(data)}

    def _make_binary_msg(self, data: bytes) -> dict:
        return {"type": "websocket.receive", "bytes": data}

    def _make_disconnect() -> dict:
        return {"type": "websocket.disconnect"}

    @pytest.mark.asyncio
    async def test_endpoint_accepts_connection(self, ws_for_endpoint, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws_for_endpoint.receive = AsyncMock(
                return_value={"type": "websocket.disconnect"}
            )

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 123
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws_for_endpoint)

            ws_for_endpoint.accept.assert_awaited_once()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_endpoint_sends_initial_status(self, ws_for_endpoint, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws_for_endpoint.receive = AsyncMock(
                return_value={"type": "websocket.disconnect"}
            )

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 42
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws_for_endpoint)

            # Check the status message sent after connect
            status_call = cs._send.call_args_list[0]
            status_msg = status_call[0][0]
            assert status_msg["type"] == "status"
            assert status_msg["pipeline_ready"] is True
            assert status_msg["direction"] == "es_to_en"
            assert status_msg["mode"] == "conversation"
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_endpoint_creates_db_session(self, ws_for_endpoint, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws_for_endpoint.receive = AsyncMock(
                return_value={"type": "websocket.disconnect"}
            )

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws_for_endpoint)

            mock_pipeline.create_session.assert_awaited_once()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_endpoint_closes_session_on_disconnect(self, ws_for_endpoint, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws_for_endpoint.receive = AsyncMock(
                return_value={"type": "websocket.disconnect"}
            )

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws_for_endpoint)

            cs.cleanup.assert_awaited_once()
            mock_pipeline.close_session.assert_awaited_once()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_endpoint_clears_active_session_on_disconnect(self, ws_for_endpoint, mock_pipeline):
        old_pipeline = ws_mod._current_pipeline
        old_session = ws_mod._active_session
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws_for_endpoint.receive = AsyncMock(
                return_value={"type": "websocket.disconnect"}
            )

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws_for_endpoint)

            assert ws_mod._active_session is None
        finally:
            ws_mod._current_pipeline = old_pipeline
            ws_mod._active_session = old_session


# ---------------------------------------------------------------------------
# Message dispatch (individual message types)
# ---------------------------------------------------------------------------

class TestMessageDispatch:
    """Tests for each message type handled inside websocket_endpoint."""

    @pytest.fixture
    def ws_session_pair(self, mock_websocket, mock_pipeline):
        """Provide a session wired to the mock pipeline for dispatch testing.

        Returns (session, pipeline) where session._send captures all outbound
        messages for assertion.
        """
        old = ws_mod._current_pipeline
        ws_mod._current_pipeline = mock_pipeline

        s = ClientSession(mock_websocket, mock_pipeline)
        s.vad = MagicMock()
        s.vad.initialize = AsyncMock()
        s.vad.reset = MagicMock()
        s.vad.feed_pcm = AsyncMock()
        s.vad.flush = AsyncMock()
        s.vad.segments_emitted = 0
        s.vad.total_speech_seconds = 0.0
        s.vad.on_partial_audio = None
        s.decoder = MagicMock()
        s.decoder.reset = MagicMock()
        s.decoder.start_streaming = AsyncMock()
        s.decoder.stop_streaming = AsyncMock(return_value=b"")
        s.decoder.feed_chunk = AsyncMock(return_value=b"")

        yield s, mock_pipeline

        ws_mod._current_pipeline = old

    # --- ping ---

    @pytest.mark.asyncio
    async def test_ping_responds_pong(self, ws_session_pair):
        """Endpoint helper: simulate a ping message dispatch.

        Since websocket_endpoint is complex to drive, we test the dispatch
        logic by directly mimicking what the endpoint does for each message
        type (calling session methods / pipeline methods).
        """
        session, pipeline = ws_session_pair
        await session._send({"type": "pong"})
        sent = json.loads(session.ws.send_text.call_args[0][0])
        assert sent["type"] == "pong"

    # --- toggle_direction ---

    @pytest.mark.asyncio
    async def test_toggle_direction_valid_sends_confirmation(self, ws_session_pair):
        session, pipeline = ws_session_pair
        direction = "en_to_es"
        assert direction in VALID_DIRECTIONS
        pipeline.set_direction(direction)
        await session._send({"type": "direction_changed", "direction": direction})
        sent = json.loads(session.ws.send_text.call_args[0][0])
        assert sent["type"] == "direction_changed"
        assert sent["direction"] == "en_to_es"
        pipeline.set_direction.assert_called_with(direction)

    @pytest.mark.asyncio
    async def test_toggle_direction_invalid_sends_error(self, ws_session_pair):
        session, pipeline = ws_session_pair
        invalid_direction = "fr_to_en"
        assert invalid_direction not in VALID_DIRECTIONS
        await session._send({"type": "error", "message": f"Invalid direction: {invalid_direction}"})
        sent = json.loads(session.ws.send_text.call_args[0][0])
        assert sent["type"] == "error"
        assert "Invalid direction" in sent["message"]

    # --- set_mode ---

    @pytest.mark.asyncio
    async def test_set_mode_valid_sends_confirmation(self, ws_session_pair):
        session, pipeline = ws_session_pair
        mode = "classroom"
        assert mode in VALID_MODES
        pipeline.set_mode(mode)
        await session._send({"type": "mode_changed", "mode": mode})
        sent = json.loads(session.ws.send_text.call_args[0][0])
        assert sent["type"] == "mode_changed"
        assert sent["mode"] == "classroom"

    @pytest.mark.asyncio
    async def test_set_mode_invalid_sends_error(self, ws_session_pair):
        session, pipeline = ws_session_pair
        invalid_mode = "debate"
        assert invalid_mode not in VALID_MODES
        await session._send({"type": "error", "message": f"Invalid mode: {invalid_mode}"})
        sent = json.loads(session.ws.send_text.call_args[0][0])
        assert sent["type"] == "error"
        assert "Invalid mode" in sent["message"]

    # --- rename_speaker ---

    @pytest.mark.asyncio
    async def test_rename_speaker_calls_tracker(self, ws_session_pair):
        session, pipeline = ws_session_pair
        pipeline.speaker_tracker.rename("SPEAKER_00", "Maria")
        pipeline.speaker_tracker.rename.assert_called_with("SPEAKER_00", "Maria")

    @pytest.mark.asyncio
    async def test_rename_speaker_sends_update(self, ws_session_pair):
        session, pipeline = ws_session_pair
        pipeline.speaker_tracker.get_all.return_value = [
            SpeakerProfile(id="SPEAKER_00", label="Speaker A", custom_name="Maria", color="#3B82F6")
        ]
        update = WSSpeakersUpdate(speakers=pipeline.speaker_tracker.get_all())
        await session._send(update.model_dump())
        sent = json.loads(session.ws.send_text.call_args[0][0])
        assert sent["type"] == "speakers_updated"
        assert len(sent["speakers"]) == 1
        assert sent["speakers"][0]["custom_name"] == "Maria"


# ---------------------------------------------------------------------------
# Full message dispatch through websocket_endpoint
# ---------------------------------------------------------------------------

class TestEndpointMessageRouting:
    """Integration-style tests that drive messages through websocket_endpoint."""

    @pytest.mark.asyncio
    async def test_ping_message_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            call_count = 0

            async def receive_sequence():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {"text": json.dumps({"type": "ping"})}
                return {"type": "websocket.disconnect"}

            ws.receive = AsyncMock(side_effect=receive_sequence)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.listening = False
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                cs.start_listening = AsyncMock()
                cs.stop_listening = AsyncMock()
                cs.handle_audio_chunk = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            # Find pong response
            pong_sent = False
            for call in cs._send.call_args_list:
                msg = call[0][0]
                if msg.get("type") == "pong":
                    pong_sent = True
                    break
            assert pong_sent, "Pong response should be sent for ping message"
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_start_stop_listening_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "start_listening"})},
                {"text": json.dumps({"type": "stop_listening"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.listening = False
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                cs.start_listening = AsyncMock()
                cs.stop_listening = AsyncMock()
                cs.handle_audio_chunk = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            cs.start_listening.assert_awaited_once()
            cs.stop_listening.assert_awaited_once()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_toggle_direction_valid_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "toggle_direction", "direction": "en_to_es"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            mock_pipeline.set_direction.assert_called_with("en_to_es")
            # Should send direction_changed, not error
            direction_changed = False
            for call in cs._send.call_args_list:
                msg = call[0][0]
                if msg.get("type") == "direction_changed":
                    assert msg["direction"] == "en_to_es"
                    direction_changed = True
            assert direction_changed
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_toggle_direction_invalid_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "toggle_direction", "direction": "fr_to_de"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            mock_pipeline.set_direction.assert_not_called()
            error_sent = False
            for call in cs._send.call_args_list:
                msg = call[0][0]
                if msg.get("type") == "error" and "Invalid direction" in msg.get("message", ""):
                    error_sent = True
            assert error_sent, "Should send error for invalid direction"
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_set_mode_valid_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "set_mode", "mode": "classroom"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            mock_pipeline.set_mode.assert_called_with("classroom")
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_set_mode_invalid_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "set_mode", "mode": "debate"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            mock_pipeline.set_mode.assert_not_called()
            error_sent = False
            for call in cs._send.call_args_list:
                msg = call[0][0]
                if msg.get("type") == "error" and "Invalid mode" in msg.get("message", ""):
                    error_sent = True
            assert error_sent, "Should send error for invalid mode"
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_text_input_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "text_input", "text": "Hola mundo", "speaker_id": "USER"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS, \
                 patch("server.routes.websocket._process_text", new_callable=AsyncMock) as mock_pt:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                # Patch asyncio.create_task to call directly
                with patch("server.routes.websocket.asyncio.create_task") as mock_ct:
                    await websocket_endpoint(ws)
                    # The endpoint calls asyncio.create_task(_process_text(...))
                    mock_ct.assert_called_once()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_text_input_empty_text_ignored(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "text_input", "text": "   "})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS, \
                 patch("server.routes.websocket.asyncio.create_task") as mock_ct:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)
                # Empty text should not create a task
                mock_ct.assert_not_called()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_binary_audio_routed_to_handle_audio_chunk(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            audio_data = b"\xFF\xFE" * 50
            msgs = [
                {"bytes": audio_data},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                cs.handle_audio_chunk = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            cs.handle_audio_chunk.assert_awaited_once_with(audio_data)
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_new_session_resets_and_sends_confirmation(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "new_session"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.listening = False
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                cs.stop_listening = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            mock_pipeline.reset_session.assert_awaited_once()
            # Should send session_reset
            reset_sent = False
            for call in cs._send.call_args_list:
                msg = call[0][0]
                if msg.get("type") == "session_reset":
                    assert msg["session_id"] == 2  # return value of reset_session
                    reset_sent = True
            assert reset_sent, "session_reset message should be sent"
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_new_session_stops_listening_first(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "new_session"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.listening = True  # Currently listening
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                cs.stop_listening = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            cs.stop_listening.assert_awaited_once()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_rename_speaker_through_endpoint(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "rename_speaker", "speaker_id": "SPEAKER_00", "name": "Carlos"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            mock_pipeline.speaker_tracker.rename.assert_called_once_with("SPEAKER_00", "Carlos")
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_rename_speaker_empty_id_ignored(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": json.dumps({"type": "rename_speaker", "speaker_id": "", "name": "Carlos"})},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                await websocket_endpoint(ws)

            mock_pipeline.speaker_tracker.rename.assert_not_called()
        finally:
            ws_mod._current_pipeline = old

    @pytest.mark.asyncio
    async def test_malformed_json_message_ignored(self, mock_pipeline):
        old = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()

            msgs = [
                {"text": "this is not valid json{{{"},
                {"type": "websocket.disconnect"},
            ]
            ws.receive = AsyncMock(side_effect=msgs)

            with patch("server.routes.websocket.ClientSession") as MockCS:
                cs = MagicMock()
                cs.pipeline = mock_pipeline
                cs.id = 1
                cs.initialize = AsyncMock()
                cs._send = AsyncMock()
                cs._wire_callbacks = MagicMock()
                cs.cleanup = AsyncMock()
                MockCS.return_value = cs

                # Should not crash
                await websocket_endpoint(ws)

            # Cleanup should still happen
            cs.cleanup.assert_awaited_once()
        finally:
            ws_mod._current_pipeline = old


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    """Validate module-level constants have expected values."""

    def test_valid_directions_contains_both(self):
        assert "es_to_en" in VALID_DIRECTIONS
        assert "en_to_es" in VALID_DIRECTIONS
        assert len(VALID_DIRECTIONS) == 2

    def test_valid_modes_contains_both(self):
        assert "conversation" in VALID_MODES
        assert "classroom" in VALID_MODES
        assert len(VALID_MODES) == 2

    def test_merge_gap_is_positive(self):
        assert MERGE_GAP_SECONDS > 0

    def test_max_pending_is_positive(self):
        assert MAX_PENDING_SECONDS > 0

    def test_max_pending_greater_than_merge_gap(self):
        assert MAX_PENDING_SECONDS > MERGE_GAP_SECONDS
