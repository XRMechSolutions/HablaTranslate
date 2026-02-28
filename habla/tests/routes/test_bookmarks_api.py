"""Tests for bookmark API routes."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from server.routes.api_bookmarks import bookmarks_router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fastapi_app():
    test_app = FastAPI()
    test_app.include_router(bookmarks_router)
    return test_app


@pytest.fixture
async def client(fastapi_app):
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test") as c:
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
    with patch("server.routes.api_bookmarks.get_db", new=mock_get_db):
        yield


# ---------------------------------------------------------------------------
# Toggle bookmark
# ---------------------------------------------------------------------------

class TestToggleBookmark:
    @pytest.mark.asyncio
    async def test_toggle_on_unbookmarked_exchange(self, client, mock_db):
        """Toggling an unbookmarked exchange sets bookmarked=True."""
        mock_db.execute_fetchall.return_value = [{"id": 1, "bookmarked": False}]

        resp = await client.post("/api/exchanges/1/bookmark")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bookmarked"] is True
        assert data["id"] == 1
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_toggle_off_bookmarked_exchange(self, client, mock_db):
        """Toggling a bookmarked exchange sets bookmarked=False and clears note."""
        mock_db.execute_fetchall.return_value = [{"id": 1, "bookmarked": True}]

        resp = await client.post("/api/exchanges/1/bookmark")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bookmarked"] is False

    @pytest.mark.asyncio
    async def test_toggle_nonexistent_exchange_returns_404(self, client, mock_db):
        """Toggling a non-existent exchange returns 404."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.post("/api/exchanges/999/bookmark")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_toggle_with_note(self, client, mock_db):
        """Toggling on with a note stores the note."""
        mock_db.execute_fetchall.return_value = [{"id": 1, "bookmarked": False}]

        resp = await client.post("/api/exchanges/1/bookmark", json={"note": "interesting idiom"})
        assert resp.status_code == 200
        assert resp.json()["bookmarked"] is True
        # Verify the SQL included the note
        call_args = mock_db.execute.call_args
        assert "interesting idiom" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_toggle_off_clears_note(self, client, mock_db):
        """Toggling off clears the bookmark_note to NULL."""
        mock_db.execute_fetchall.return_value = [{"id": 1, "bookmarked": True}]

        resp = await client.post("/api/exchanges/1/bookmark")
        assert resp.status_code == 200
        call_args = mock_db.execute.call_args
        assert "bookmarked = 0" in call_args[0][0]
        assert "bookmark_note = NULL" in call_args[0][0]


# ---------------------------------------------------------------------------
# Get recent bookmarks
# ---------------------------------------------------------------------------

class TestRecentBookmarks:
    @pytest.mark.asyncio
    async def test_returns_empty_list(self, client, mock_db):
        """Returns empty bookmarks list and zero total when none exist."""
        mock_db.execute_fetchall.side_effect = [
            [],  # bookmarks query
            [{"cnt": 0}],  # count query
        ]

        resp = await client.get("/api/bookmarks/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bookmarks"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_bookmarked_exchanges(self, client, mock_db):
        """Returns bookmarked exchanges with session info and has_audio computed."""
        mock_db.execute_fetchall.side_effect = [
            [
                {
                    "id": 5, "session_id": 1, "speaker_id": "SPEAKER_00",
                    "timestamp": "2026-02-27T10:00:00", "direction": "es_to_en",
                    "raw_transcript": "Hola mundo", "corrected_source": None,
                    "translation": "Hello world", "confidence": 0.9,
                    "is_correction": False, "correction_json": None,
                    "processing_ms": 500, "audio_path": "data/audio/clips/1/5.wav",
                    "bookmarked": True, "bookmark_note": "test note",
                    "session_started": "2026-02-27T09:00:00", "session_name": "Test Session",
                },
            ],
            [{"cnt": 1}],
        ]

        resp = await client.get("/api/bookmarks/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["bookmarks"]) == 1
        assert data["total"] == 1
        bk = data["bookmarks"][0]
        assert bk["has_audio"] is True
        assert bk["session_name"] == "Test Session"

    @pytest.mark.asyncio
    async def test_has_audio_false_when_no_path(self, client, mock_db):
        """has_audio is False when audio_path is empty."""
        mock_db.execute_fetchall.side_effect = [
            [
                {
                    "id": 3, "session_id": 1, "speaker_id": None,
                    "timestamp": "2026-02-27T10:00:00", "direction": "en_to_es",
                    "raw_transcript": "Hello", "corrected_source": None,
                    "translation": "Hola", "confidence": 0.8,
                    "is_correction": False, "correction_json": None,
                    "processing_ms": 300, "audio_path": None,
                    "bookmarked": True, "bookmark_note": None,
                    "session_started": None, "session_name": None,
                },
            ],
            [{"cnt": 1}],
        ]

        resp = await client.get("/api/bookmarks/recent")
        data = resp.json()
        assert data["bookmarks"][0]["has_audio"] is False

    @pytest.mark.asyncio
    async def test_respects_limit_and_offset(self, client, mock_db):
        """Passes limit and offset to the query."""
        mock_db.execute_fetchall.side_effect = [[], [{"cnt": 0}]]

        resp = await client.get("/api/bookmarks/recent?limit=10&offset=5")
        assert resp.status_code == 200
        call_args = mock_db.execute_fetchall.call_args_list[0]
        assert call_args[0][1] == (10, 5)


# ---------------------------------------------------------------------------
# Get session bookmarks
# ---------------------------------------------------------------------------

class TestSessionBookmarks:
    @pytest.mark.asyncio
    async def test_returns_session_bookmarks(self, client, mock_db):
        """Returns bookmarked exchanges for a specific session."""
        mock_db.execute_fetchall.side_effect = [
            [{"id": 1}],  # session exists
            [
                {
                    "id": 10, "session_id": 1, "speaker_id": "SPEAKER_01",
                    "timestamp": "2026-02-27T11:00:00", "direction": "es_to_en",
                    "raw_transcript": "Buenos dias", "corrected_source": None,
                    "translation": "Good morning", "confidence": 0.95,
                    "is_correction": False, "correction_json": None,
                    "processing_ms": 400, "audio_path": None,
                    "bookmarked": True, "bookmark_note": None,
                },
            ],
        ]

        resp = await client.get("/api/sessions/1/bookmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == 10

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(self, client, mock_db):
        """Returns 404 if session does not exist."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.get("/api/sessions/999/bookmarks")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_bookmarks_for_session(self, client, mock_db):
        """Returns empty list when session has no bookmarks."""
        mock_db.execute_fetchall.side_effect = [
            [{"id": 1}],  # session exists
            [],  # no bookmarks
        ]

        resp = await client.get("/api/sessions/1/bookmarks")
        assert resp.status_code == 200
        assert resp.json() == []
