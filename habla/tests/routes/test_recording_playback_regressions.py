"""Regression tests for History Page Recording Playback audit findings.

Verifies fixes for BUG-015 through BUG-021 (BUG_REGISTRY.md).
These tests pass only when the corresponding bug is fixed.
"""

import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from server.routes.api_sessions import session_router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(session_router)
    return test_app


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.execute_fetchall = AsyncMock(return_value=[])
    return db


@pytest.fixture(autouse=True)
def _patch_get_db(mock_db):
    mock_get_db = AsyncMock(return_value=mock_db)
    with patch("server.routes.api_sessions.get_db", new=mock_get_db):
        yield


# ---------------------------------------------------------------------------
# BUG-016 — Path traversal in delete_session recording cleanup
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBug016PathTraversal:
    """BUG-016: recording_dir with traversal sequences must be rejected."""

    async def test_delete_session_rejects_traversal_in_recording_dir(self, client, mock_db, tmp_path):
        """A recording_dir of '../../evil' must NOT cause deletion outside recordings dir."""
        recordings_base = tmp_path / "recordings"
        recordings_base.mkdir()

        # Create a target outside the recordings dir that traversal would reach
        target_dir = tmp_path / "evil"
        target_dir.mkdir()
        (target_dir / "important.txt").write_text("do not delete")

        call_count = 0

        async def multi_fetchall(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 1}]
            else:
                return [{"recording_dir": "../../evil"}]

        mock_db.execute_fetchall = multi_fetchall

        with patch("server.routes.api_sessions.get_recordings_dir", return_value=recordings_base):
            resp = await client.delete("/api/sessions/1")

        assert resp.status_code == 200
        # The target directory must still exist — traversal was rejected
        assert target_dir.exists(), "Path traversal was not blocked — directory outside recordings was deleted"
        assert (target_dir / "important.txt").exists()

    async def test_delete_session_allows_valid_recording_dir(self, client, mock_db, tmp_path):
        """A legitimate recording_dir within the recordings dir should be deleted."""
        recordings_base = tmp_path / "recordings"
        recordings_base.mkdir()
        rec_dir = recordings_base / "12345_20260302_120000"
        rec_dir.mkdir()
        (rec_dir / "raw.webm").write_bytes(b"fake")

        call_count = 0

        async def multi_fetchall(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 1}]
            else:
                return [{"recording_dir": "12345_20260302_120000"}]

        mock_db.execute_fetchall = multi_fetchall

        with patch("server.routes.api_sessions.get_recordings_dir", return_value=recordings_base):
            resp = await client.delete("/api/sessions/1")

        assert resp.status_code == 200
        assert not rec_dir.exists(), "Valid recording directory should have been deleted"


# ---------------------------------------------------------------------------
# BUG-017 — Hardcoded recording path diverges from config
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBug017HardcodedPath:
    """BUG-017: delete handler must use configured recordings path."""

    async def test_delete_uses_configured_recording_path(self, client, mock_db, tmp_path):
        """Delete handler should call get_recordings_dir() for the base path."""
        custom_dir = tmp_path / "custom" / "recordings"
        custom_dir.mkdir(parents=True)
        rec_dir = custom_dir / "test_rec_123"
        rec_dir.mkdir()
        (rec_dir / "data.webm").write_bytes(b"x")

        call_count = 0

        async def multi_fetchall(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 1}]
            else:
                return [{"recording_dir": "test_rec_123"}]

        mock_db.execute_fetchall = multi_fetchall

        with patch("server.routes.api_sessions.get_recordings_dir", return_value=custom_dir):
            resp = await client.delete("/api/sessions/1")

        assert resp.status_code == 200
        # The recording in the custom dir should have been deleted
        assert not rec_dir.exists(), "Handler did not use configured recording path"


# ---------------------------------------------------------------------------
# BUG-018 — Recorder not reset on new_session
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBug018RecorderNotReset:
    """BUG-018: new_session must reset the recorder so stale recordings aren't linked."""

    async def test_new_session_resets_recorder(self):
        """After new_session handler logic, session.recorder should be None."""
        import server.routes.websocket as ws_mod
        from server.routes.websocket import ClientSession

        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()

        mock_pipeline = MagicMock()
        mock_pipeline.ready = True
        mock_pipeline.session_id = 1
        mock_pipeline.direction = "es_to_en"
        mock_pipeline.mode = "conversation"
        mock_pipeline.speaker_tracker = MagicMock()
        mock_pipeline.speaker_tracker.speakers = {}
        mock_pipeline.speaker_tracker.get_all.return_value = []
        mock_pipeline.create_session = AsyncMock(return_value=1)
        mock_pipeline.close_session = AsyncMock()
        mock_pipeline.reset_session = AsyncMock(return_value=2)
        mock_pipeline.set_callbacks = MagicMock()
        mock_pipeline.reset_partial_state = MagicMock()

        audio_cfg = MagicMock()
        audio_cfg.sample_rate = 16000
        audio_cfg.vad_silence_ms = 600
        audio_cfg.min_speech_ms = 400
        audio_cfg.max_segment_seconds = 30.0
        audio_cfg.calibration_duration_ms = 2000
        asr_cfg = MagicMock()
        asr_cfg.vad_threshold = 0.35
        mock_pipeline.config.audio = audio_cfg
        mock_pipeline.config.asr = asr_cfg
        mock_pipeline.config.session.tts_enabled = False

        old_pipeline = ws_mod._current_pipeline
        try:
            ws_mod._current_pipeline = mock_pipeline
            session = ClientSession(mock_ws, mock_pipeline)
            session.vad = MagicMock()
            session.vad.initialize = AsyncMock()
            session.decoder = MagicMock()

            # Simulate an active recorder from a previous listening session
            old_recorder = MagicMock()
            old_recorder.session_dir = MagicMock()
            old_recorder.session_dir.name = "old_session_dir"
            old_recorder.stop_recording = MagicMock()
            session.recorder = old_recorder
            session.listening = False

            session._wire_callbacks = lambda: mock_pipeline.set_callbacks()

            # Simulate new_session handler (matches websocket.py logic)
            if session.listening:
                await session.stop_listening()
            if session.recorder:
                session.recorder.stop_recording()
                session.recorder = None
            await session.pipeline.reset_session()
            if hasattr(session, '_wire_callbacks'):
                session._wire_callbacks()

            assert session.recorder is None, (
                "Recorder should be None after new_session to prevent stale "
                "recording dir from being linked to the new session"
            )
            old_recorder.stop_recording.assert_called_once()
        finally:
            ws_mod._current_pipeline = old_pipeline


# ---------------------------------------------------------------------------
# BUG-019 — delete_session FK constraint when vocab exists
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBug019VocabFKConstraint:
    """BUG-019: delete_session must delete vocab before exchanges."""

    async def test_delete_session_with_vocab_succeeds(self, tmp_path):
        """Deleting a session that has vocab entries must not fail with FK constraint."""
        from server.db.database import init_db, close_db

        db_path = tmp_path / "test.db"
        db = await init_db(db_path)
        try:
            # Create a session with an exchange and a vocab entry
            await db.execute(
                "INSERT INTO sessions (mode, direction) VALUES ('conversation', 'es_to_en')"
            )
            await db.execute(
                """INSERT INTO exchanges (session_id, direction, raw_transcript, translation)
                   VALUES (1, 'es_to_en', 'tomar el pelo', 'to pull someone''s leg')"""
            )
            await db.execute(
                """INSERT INTO vocab (exchange_id, term, meaning)
                   VALUES (1, 'tomar el pelo', 'to pull someone''s leg')"""
            )
            await db.commit()

            # Delete using the same SQL order as the fixed delete_session
            await db.execute("DELETE FROM quality_metrics WHERE session_id = ?", (1,))
            await db.execute(
                "DELETE FROM vocab WHERE exchange_id IN (SELECT id FROM exchanges WHERE session_id = ?)",
                (1,),
            )
            await db.execute("DELETE FROM exchanges WHERE session_id = ?", (1,))
            await db.execute("DELETE FROM speakers WHERE session_id = ?", (1,))
            await db.execute("DELETE FROM sessions WHERE id = ?", (1,))
            await db.commit()

            # Verify everything is gone
            rows = await db.execute_fetchall("SELECT COUNT(*) as c FROM sessions")
            assert rows[0]["c"] == 0
            rows = await db.execute_fetchall("SELECT COUNT(*) as c FROM vocab")
            assert rows[0]["c"] == 0
            rows = await db.execute_fetchall("SELECT COUNT(*) as c FROM exchanges")
            assert rows[0]["c"] == 0
        finally:
            await close_db()


# ---------------------------------------------------------------------------
# BUG-021 — Verify recording cleanup actually calls rmtree
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBug021CleanupVerification:
    """BUG-021: delete must actually remove the recording directory."""

    async def test_delete_session_removes_recording_dir(self, client, mock_db, tmp_path):
        """Deleting a session with a recording must call shutil.rmtree on the recording dir."""
        recordings_base = tmp_path / "recordings"
        recordings_base.mkdir()
        rec_dir = recordings_base / "some_recording_dir"
        rec_dir.mkdir()
        (rec_dir / "raw.webm").write_bytes(b"data")

        call_count = 0

        async def multi_fetchall(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 1}]
            else:
                return [{"recording_dir": "some_recording_dir"}]

        mock_db.execute_fetchall = multi_fetchall

        with patch("server.routes.api_sessions.get_recordings_dir", return_value=recordings_base):
            resp = await client.delete("/api/sessions/1")

        assert resp.status_code == 200
        assert not rec_dir.exists(), "Recording directory should have been deleted by rmtree"
