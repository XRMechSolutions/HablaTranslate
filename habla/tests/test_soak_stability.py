"""Soak and repetition stability tests for lifecycle operations.

These tests run N cycles of connect/listen/stop/disconnect and
playback start/stop to verify no leaked tasks, stale state, or
resource drift across repeated operations.

Marked as 'soak' for optional CI gating — run with:
    pytest tests/test_soak_stability.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import server.routes.websocket as ws_mod
from server.routes.websocket import ClientSession
from server.services.playback import PlaybackService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pipeline():
    p = MagicMock()
    p.ready = True
    p.direction = "es_to_en"
    p.mode = "conversation"
    p.session_id = 1
    p.topic_summary = ""
    p.speaker_tracker = MagicMock()
    p.speaker_tracker.speakers = {}
    p.speaker_tracker.get_all.return_value = []
    p.create_session = AsyncMock(return_value=1)
    p.close_session = AsyncMock()
    p.reset_session = AsyncMock(return_value=2)
    p.process_wav = AsyncMock(return_value=None)
    p.process_partial_audio = AsyncMock()
    p.set_direction = MagicMock()
    p.set_mode = MagicMock()
    p.set_callbacks = MagicMock()
    p.reset_partial_state = MagicMock()
    return p


@pytest.fixture
def mock_websocket():
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    ws.accept = AsyncMock()
    return ws


def _make_session(ws, pipeline):
    """Create a ClientSession with mocked VAD/decoder (no real ffmpeg)."""
    s = ClientSession(ws, pipeline)
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
    return s


# ---------------------------------------------------------------------------
# Repeated listen/stop lifecycle
# ---------------------------------------------------------------------------

class TestRepeatedListenStopCycles:
    """Verify start_listening / stop_listening N times leaves clean state."""

    CYCLES = 20

    async def test_repeated_listen_stop(self, mock_websocket, mock_pipeline):
        """N cycles of start/stop listening must leave no leaked state."""
        session = _make_session(mock_websocket, mock_pipeline)
        await session.initialize()

        for i in range(self.CYCLES):
            await session.start_listening()
            assert session.listening is True
            await session.stop_listening()
            assert session.listening is False
            # No lingering decode task
            assert session._decode_task is None or session._decode_task.done()

    async def test_repeated_cleanup_idempotent(self, mock_websocket, mock_pipeline):
        """Calling cleanup multiple times must not raise."""
        session = _make_session(mock_websocket, mock_pipeline)
        await session.initialize()

        await session.start_listening()
        for _ in range(5):
            await session.cleanup()
            assert session._cleaned_up is True
            assert session.listening is False


# ---------------------------------------------------------------------------
# Repeated session create/close
# ---------------------------------------------------------------------------

class TestRepeatedSessionLifecycle:
    """Verify creating and closing WebSocket sessions N times is stable."""

    CYCLES = 15

    async def test_repeated_connect_disconnect(self, mock_websocket, mock_pipeline):
        """N cycles of session create + cleanup + close must leave clean state."""
        old_pipeline = ws_mod._current_pipeline
        ws_mod._current_pipeline = mock_pipeline

        try:
            for i in range(self.CYCLES):
                session = _make_session(mock_websocket, mock_pipeline)
                ws_mod._active_session = session
                await session.initialize()
                await session.pipeline.create_session()

                # Simulate some listening
                await session.start_listening()
                await session.stop_listening()

                # Cleanup
                await session.cleanup()
                await session.pipeline.close_session()
                ws_mod._active_session = None

                # Verify clean state
                assert session._cleaned_up is True
                assert session.listening is False
                assert session.playback_mode is False
        finally:
            ws_mod._current_pipeline = old_pipeline
            ws_mod._active_session = None


# ---------------------------------------------------------------------------
# Repeated playback start/stop
# ---------------------------------------------------------------------------

class TestRepeatedPlaybackCycles:
    """Verify playback start/stop N times leaves clean state."""

    CYCLES = 20

    @pytest.fixture
    def playback_service(self, tmp_path):
        rec_dir = tmp_path / "recordings"
        rec_dir.mkdir()
        session_dir = rec_dir / "test_rec"
        session_dir.mkdir()
        (session_dir / "raw_stream.webm").write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 50)
        return PlaybackService(rec_dir)

    async def test_repeated_stop_idempotent(self, playback_service):
        """Calling stop_playback when nothing is active must not raise."""
        for _ in range(self.CYCLES):
            await playback_service.stop_playback()
            assert playback_service.is_active is False
            assert playback_service._active_recording_id is None

    async def test_no_stale_state_after_missing_recording(self, playback_service):
        """Starting playback for a missing recording must not leave stale state."""
        mock_session = MagicMock()
        for _ in range(self.CYCLES):
            result = await playback_service.start_playback(
                recording_id="nonexistent",
                session=mock_session,
                speed=0,
            )
            assert "error" in result
            assert playback_service._active_recording_id is None
            assert playback_service.is_active is False


# ---------------------------------------------------------------------------
# Health cache behavior
# ---------------------------------------------------------------------------

class TestHealthCacheBehavior:
    """Verify health check caching works correctly."""

    async def test_cache_returns_same_result_within_ttl(self):
        from server.services.health import (
            run_runtime_checks, _runtime_cache, _RUNTIME_CACHE_TTL,
            SystemHealth, ComponentStatus, HealthCheck,
        )

        # Seed the cache with a known result
        cached = SystemHealth()
        cached.add(HealthCheck("test", ComponentStatus.OK, "cached"))
        _runtime_cache["result"] = cached
        _runtime_cache["timestamp"] = __import__("time").monotonic()

        # Create a mock pipeline
        mock_pipeline = MagicMock()

        # Should return cached result without calling any checks
        result = await run_runtime_checks(mock_pipeline)
        assert result is cached

        # Clean up
        _runtime_cache["result"] = None
        _runtime_cache["timestamp"] = 0.0

    async def test_cache_expires_after_ttl(self):
        import time as time_mod
        from server.services.health import (
            run_runtime_checks, _runtime_cache, _RUNTIME_CACHE_TTL,
            SystemHealth, ComponentStatus, HealthCheck,
        )

        # Seed cache with an expired timestamp
        old_result = SystemHealth()
        old_result.add(HealthCheck("test", ComponentStatus.OK, "old"))
        _runtime_cache["result"] = old_result
        _runtime_cache["timestamp"] = time_mod.monotonic() - _RUNTIME_CACHE_TTL - 1

        # Mock pipeline with enough structure for run_runtime_checks
        mock_pipeline = MagicMock()
        mock_pipeline.config.translator.provider = "ollama"
        mock_pipeline.config.translator.ollama_url = "http://localhost:11434"
        mock_pipeline.config.translator.ollama_model = "test"
        mock_pipeline.config.db_path = "/tmp/test.db"
        mock_pipeline._whisperx_model = None
        mock_pipeline._diarize_pipeline = None

        # Patch expensive checks to avoid real network/subprocess calls
        from unittest.mock import patch
        with patch("server.services.health._check_active_llm", new_callable=AsyncMock) as mock_llm, \
             patch("server.services.health.check_ffmpeg", new_callable=AsyncMock) as mock_ff, \
             patch("server.services.health.check_database", new_callable=AsyncMock) as mock_db:
            mock_llm.return_value = HealthCheck("llm", ComponentStatus.OK, "ok")
            mock_ff.return_value = HealthCheck("ffmpeg", ComponentStatus.OK, "ok")
            mock_db.return_value = HealthCheck("database", ComponentStatus.OK, "ok")

            result = await run_runtime_checks(mock_pipeline)
            assert result is not old_result  # Got fresh result

        # Clean up
        _runtime_cache["result"] = None
        _runtime_cache["timestamp"] = 0.0


# ---------------------------------------------------------------------------
# Orchestrator metrics tracking
# ---------------------------------------------------------------------------

class TestOrchestratorMetrics:
    """Verify metrics dict is populated correctly."""

    def test_metrics_initial_state(self):
        """Fresh orchestrator has zero metrics."""
        from server.pipeline.orchestrator import PipelineOrchestrator
        config = MagicMock()
        config.session.direction = "es_to_en"
        config.session.mode = "conversation"
        config.translator.provider = "ollama"
        config.data_dir = Path("/tmp")

        orch = PipelineOrchestrator(config)
        m = orch.metrics
        assert m["segments_processed"] == 0
        assert m["translations_completed"] == 0
        assert m["translation_errors"] == 0
        assert m["peak_queue_depth"] == 0
        assert m["sessions_created"] == 0
        assert m["sessions_closed"] == 0
        assert m["queue_depth"] == 0
        assert m["inflight_translations"] == 0
        assert m["worker_alive"] is False
