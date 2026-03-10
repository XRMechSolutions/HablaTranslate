"""Tests for corrections API routes including the failed-segments endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from server.routes.api_corrections import corrections_router
import server.routes._state as _state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(corrections_router)
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
    with patch("server.routes.api_corrections.get_db", new=mock_get_db):
        yield


@pytest.fixture(autouse=True)
def _clear_playback_service():
    original = _state._playback_service
    _state._playback_service = None
    yield
    _state._playback_service = original


# ---------------------------------------------------------------------------
# TestFailedSegments
# ---------------------------------------------------------------------------

class TestFailedSegments:
    """Tests for GET /api/corrections/failed-segments."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_rejections(self, client, mock_db):
        """Empty quality_metrics → empty failed_segments list."""
        mock_db.execute_fetchall = AsyncMock(return_value=[])

        r = await client.get("/api/corrections/failed-segments")

        assert r.status_code == 200
        data = r.json()
        assert data["failed_segments"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_asr_rejected_rows(self, client, mock_db):
        """Rows with status=asr_rejected should appear in response."""
        row = {
            "id": 1,
            "session_id": 42,
            "segment_id": 7,
            "timestamp": "2026-03-02T10:00:00",
            "audio_rms": 320.5,
            "duration_seconds": 1.8,
            "processing_time_ms": 450,
            "vad_threshold": 0.35,
            "model_name": "small",
            "clipped_onset": False,
            "status": "asr_rejected",
            "recording_dir": None,
        }
        mock_db.execute_fetchall = AsyncMock(return_value=[row])

        r = await client.get("/api/corrections/failed-segments")

        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        seg = data["failed_segments"][0]
        assert seg["session_id"] == 42
        assert seg["segment_id"] == 7
        assert seg["audio_rms"] == pytest.approx(320.5)
        assert seg["duration_seconds"] == pytest.approx(1.8)
        assert seg["status"] == "asr_rejected"
        assert seg["audio_url"] is None  # no recording_dir → no clip

    @pytest.mark.asyncio
    async def test_returns_asr_empty_rows(self, client, mock_db):
        """Rows with status=asr_empty should also appear — these were invisible before the fix."""
        row = {
            "id": 3,
            "session_id": 10,
            "segment_id": 2,
            "timestamp": "2026-03-02T12:00:00",
            "audio_rms": 180.0,
            "duration_seconds": 0.8,
            "processing_time_ms": 310,
            "vad_threshold": 0.35,
            "model_name": "small",
            "clipped_onset": False,
            "status": "asr_empty",
            "recording_dir": None,
        }
        mock_db.execute_fetchall = AsyncMock(return_value=[row])

        r = await client.get("/api/corrections/failed-segments")

        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        seg = data["failed_segments"][0]
        assert seg["status"] == "asr_empty"
        assert seg["audio_url"] is None

    @pytest.mark.asyncio
    async def test_limit_param_default(self, client, mock_db):
        """Default limit is 50 — endpoint should accept without error."""
        mock_db.execute_fetchall = AsyncMock(return_value=[])
        r = await client.get("/api/corrections/failed-segments")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_limit_param_custom(self, client, mock_db):
        """Custom limit param should be accepted."""
        mock_db.execute_fetchall = AsyncMock(return_value=[])
        r = await client.get("/api/corrections/failed-segments?limit=10")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_session_id_filter(self, client, mock_db):
        """session_id query param should be passed to query."""
        mock_db.execute_fetchall = AsyncMock(return_value=[])
        r = await client.get("/api/corrections/failed-segments?session_id=5")
        assert r.status_code == 200
        # Verify the session_id filter was applied — execute_fetchall called once
        mock_db.execute_fetchall.assert_called_once()
        call_args = mock_db.execute_fetchall.call_args
        # session_id=5 should appear in the positional args tuple
        assert 5 in call_args[0][1]

    @pytest.mark.asyncio
    async def test_clipped_onset_included_in_response(self, client, mock_db):
        """clipped_onset flag should pass through to response."""
        row = {
            "id": 2, "session_id": 1, "segment_id": 3,
            "timestamp": "2026-03-02T11:00:00", "audio_rms": 100.0,
            "duration_seconds": 0.9, "processing_time_ms": 200,
            "vad_threshold": 0.35, "model_name": "small",
            "clipped_onset": True, "status": "asr_rejected", "recording_dir": None,
        }
        mock_db.execute_fetchall = AsyncMock(return_value=[row])

        r = await client.get("/api/corrections/failed-segments")

        assert r.status_code == 200
        seg = r.json()["failed_segments"][0]
        assert seg["clipped_onset"] is True


# ---------------------------------------------------------------------------
# TestCorrectionStats
# ---------------------------------------------------------------------------

class TestCorrectionStats:
    """Smoke tests for GET /api/corrections/stats."""

    @pytest.mark.asyncio
    async def test_returns_503_when_no_playback_service(self, client):
        """Without playback service, stats endpoint returns 503."""
        # _playback_service is None (set by autouse fixture)
        r = await client.get("/api/corrections/stats")
        assert r.status_code == 503
