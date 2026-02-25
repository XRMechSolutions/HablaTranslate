"""Integration tests for playback REST API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from server.routes.api_playback import playback_router
from server.routes._state import set_playback_service


def _create_test_app():
    """Create a minimal FastAPI app with only the playback router."""
    app = FastAPI()
    app.include_router(playback_router)
    return app


def _mock_playback_service(recordings=None, recording_detail=None, is_active=False):
    """Create a mock PlaybackService."""
    svc = MagicMock()
    svc.list_recordings.return_value = recordings or []
    svc.get_recording.return_value = recording_detail
    svc.is_active = is_active
    svc.start_playback = AsyncMock(return_value={
        "status": "started", "recording_id": "rec_full", "speed": 1.0, "mode": "full"
    })
    svc.stop_playback = AsyncMock()
    return svc


def _mock_session(listening=False):
    """Create a lightweight mock session for API tests."""
    s = MagicMock()
    s.listening = listening
    return s


# --- Recordings endpoints ---

class TestRecordingsEndpoints:
    """Tests for GET /api/recordings and GET /api/recordings/{id}."""

    async def test_list_recordings(self):
        """GET /api/recordings returns list from service."""
        app = _create_test_app()
        svc = _mock_playback_service(recordings=[
            {"id": "rec_1", "has_raw_stream": True, "segment_count": 5},
            {"id": "rec_2", "has_raw_stream": False, "segment_count": 2},
        ])
        set_playback_service(svc)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/recordings")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            assert data[0]["id"] == "rec_1"
        finally:
            set_playback_service(None)

    async def test_get_recording_found(self):
        """GET /api/recordings/{id} returns recording data."""
        app = _create_test_app()
        svc = _mock_playback_service(recording_detail={
            "id": "rec_full",
            "metadata": {"started_at": "2026-02-22T10:00:00"},
            "ground_truth": {"whisper_model": "large-v3"},
        })
        set_playback_service(svc)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/recordings/rec_full")
            assert resp.status_code == 200
            assert resp.json()["id"] == "rec_full"
        finally:
            set_playback_service(None)

    async def test_get_recording_not_found(self):
        """GET /api/recordings/{id} returns 404 for unknown id."""
        app = _create_test_app()
        svc = _mock_playback_service(recording_detail=None)
        set_playback_service(svc)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/recordings/nonexistent")
            assert resp.status_code == 404
        finally:
            set_playback_service(None)

    async def test_service_not_initialized(self):
        """Returns 503 when playback service is None."""
        app = _create_test_app()
        set_playback_service(None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/recordings")
        assert resp.status_code == 503


# --- Playback start endpoint ---

class TestPlaybackStartEndpoint:
    """Tests for POST /api/playback/start."""

    async def test_no_ws_client(self):
        """Returns 409 when no WebSocket client is connected."""
        app = _create_test_app()
        svc = _mock_playback_service()
        set_playback_service(svc)
        try:
            with patch("server.routes.websocket.get_active_session", return_value=None):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post("/api/playback/start", json={
                        "recording_id": "rec_full", "speed": 1.0, "mode": "full"
                    })
            assert resp.status_code == 409
            assert "no active" in resp.json()["detail"].lower()
        finally:
            set_playback_service(None)

    async def test_client_listening(self):
        """Returns 409 when client is actively listening via microphone."""
        app = _create_test_app()
        svc = _mock_playback_service()
        set_playback_service(svc)
        try:
            with patch("server.routes.websocket.get_active_session", return_value=_mock_session(listening=True)):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post("/api/playback/start", json={
                        "recording_id": "rec_full", "speed": 1.0, "mode": "full"
                    })
            assert resp.status_code == 409
            assert "listening" in resp.json()["detail"].lower()
        finally:
            set_playback_service(None)

    async def test_invalid_speed(self):
        """Returns 400 for negative speed."""
        app = _create_test_app()
        svc = _mock_playback_service()
        set_playback_service(svc)
        try:
            with patch("server.routes.websocket.get_active_session", return_value=_mock_session()):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post("/api/playback/start", json={
                        "recording_id": "rec_full", "speed": -1.0, "mode": "full"
                    })
            assert resp.status_code == 400
        finally:
            set_playback_service(None)

    async def test_invalid_mode(self):
        """Returns 400 for unknown mode."""
        app = _create_test_app()
        svc = _mock_playback_service()
        set_playback_service(svc)
        try:
            with patch("server.routes.websocket.get_active_session", return_value=_mock_session()):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post("/api/playback/start", json={
                        "recording_id": "rec_full", "speed": 1.0, "mode": "turbo"
                    })
            assert resp.status_code == 400
        finally:
            set_playback_service(None)

    async def test_success(self):
        """Returns 200 with correct response body on success."""
        app = _create_test_app()
        svc = _mock_playback_service()
        set_playback_service(svc)
        try:
            with patch("server.routes.websocket.get_active_session", return_value=_mock_session()):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post("/api/playback/start", json={
                        "recording_id": "rec_full", "speed": 2.0, "mode": "full"
                    })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "started"
            assert data["recording_id"] == "rec_full"
        finally:
            set_playback_service(None)


# --- Playback stop endpoint ---

class TestPlaybackStopEndpoint:
    """Tests for POST /api/playback/stop."""

    async def test_stop_active(self):
        """Returns stopped status when playback is active."""
        app = _create_test_app()
        svc = _mock_playback_service(is_active=True)
        set_playback_service(svc)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/api/playback/stop")
            assert resp.status_code == 200
            assert resp.json()["status"] == "stopped"
            svc.stop_playback.assert_awaited_once()
        finally:
            set_playback_service(None)

    async def test_stop_nothing_active(self):
        """Returns no_playback_active when nothing is playing."""
        app = _create_test_app()
        svc = _mock_playback_service(is_active=False)
        set_playback_service(svc)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/api/playback/stop")
            assert resp.status_code == 200
            assert resp.json()["status"] == "no_playback_active"
        finally:
            set_playback_service(None)

    async def test_service_not_initialized(self):
        """Returns 503 when playback service is None."""
        app = _create_test_app()
        set_playback_service(None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/playback/stop")
        assert resp.status_code == 503
