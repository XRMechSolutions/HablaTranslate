"""Comprehensive tests for REST API routes (vocab, system, sessions, idioms, LLM, LM Studio)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from server.routes._state import set_pipeline, set_lmstudio_manager, set_playback_service
from server.routes.api_vocab import vocab_router
from server.routes.api_system import system_router
from server.routes.api_sessions import session_router
from server.routes.api_idioms import idiom_router, _generate_pattern
from server.routes.api_llm import llm_router, lmstudio_router
from server.routes.api_playback import playback_router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create test FastAPI app with all routers."""
    test_app = FastAPI()
    test_app.include_router(vocab_router)
    test_app.include_router(system_router)
    test_app.include_router(session_router)
    test_app.include_router(idiom_router)
    test_app.include_router(llm_router)
    test_app.include_router(lmstudio_router)
    test_app.include_router(playback_router)
    return test_app


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def mock_db():
    """Mock database connection returned by get_db()."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.execute_fetchall = AsyncMock(return_value=[])
    return db


@pytest.fixture(autouse=True)
def _patch_get_db(mock_db):
    """Patch get_db in all route modules so no real DB is hit."""
    mock_get_db = AsyncMock(return_value=mock_db)
    with (
        patch("server.routes.api_vocab.get_db", new=mock_get_db),
        patch("server.routes.api_sessions.get_db", new=mock_get_db),
        patch("server.routes.api_idioms.get_db", new=mock_get_db),
    ):
        yield


@pytest.fixture
def mock_vocab_service():
    """Mock VocabService methods."""
    svc = MagicMock()
    svc.get_all = AsyncMock(return_value=[])
    svc.search = AsyncMock(return_value=[])
    svc.get_due_for_review = AsyncMock(return_value=[])
    svc.get_stats = AsyncMock(return_value={"total": 0, "due_for_review": 0, "by_category": {}})
    svc.record_review = AsyncMock(return_value={
        "id": 1, "ease_factor": 2.5, "interval_days": 1, "next_review": "2026-03-01", "repetitions": 1,
    })
    svc.delete = AsyncMock(return_value=True)
    svc.export_anki_csv = AsyncMock(return_value="front\tback\ttags\n")
    svc.export_json = AsyncMock(return_value=[])
    return svc


@pytest.fixture(autouse=True)
def _patch_vocab_service(mock_vocab_service):
    """Replace the module-level _vocab_service with our mock."""
    with patch("server.routes.api_vocab._vocab_service", mock_vocab_service):
        yield


@pytest.fixture
def mock_pipeline():
    """Mock PipelineOrchestrator with common attributes."""
    p = MagicMock()
    p.ready = True
    p.direction = "es_to_en"
    p.mode = "conversation"
    p.topic_summary = "General conversation"
    p.config.asr.auto_language = True
    p._queue.qsize.return_value = 0

    # Speaker tracker
    p.speaker_tracker.get_all.return_value = []
    p.speaker_tracker.rename.return_value = MagicMock(
        model_dump=MagicMock(return_value={"id": "SPEAKER_00", "label": "Speaker A", "custom_name": "Carlos"})
    )

    # Idiom scanner
    p.idiom_scanner.count = 42
    p.idiom_scanner.patterns = []

    # Translator
    p.translator.config.provider = "ollama"
    p.translator.config.model = "qwen3:4b"
    p.translator.config.quick_model = ""
    p.translator.config.ollama_url = "http://localhost:11434"
    p.translator.config.lmstudio_url = "http://localhost:1234"
    p.translator.config.openai_api_key = ""
    p.translator.config.openai_model = "gpt-5-nano"
    p.translator.metrics = {"translations": 0}
    p.translator.costs = {"total": 0.0}
    p.translator.switch_provider = MagicMock()

    p.set_direction = MagicMock()
    p.set_mode = MagicMock()
    p.reload_idiom_patterns = AsyncMock()

    return p


@pytest.fixture
def _set_pipeline(mock_pipeline):
    """Set the module-level _pipeline and clean up after."""
    set_pipeline(mock_pipeline)
    yield mock_pipeline
    set_pipeline(None)


@pytest.fixture
def mock_lmstudio_manager():
    """Mock LM Studio manager."""
    mgr = MagicMock()
    mgr.is_running = AsyncMock(return_value=True)
    mgr.get_loaded_models.return_value = ["test-model"]
    mgr.restart = AsyncMock()
    return mgr


@pytest.fixture
def _set_lmstudio_manager(mock_lmstudio_manager):
    """Set the module-level _lmstudio_manager and clean up after."""
    set_lmstudio_manager(mock_lmstudio_manager)
    yield mock_lmstudio_manager
    set_lmstudio_manager(None)


# ---------------------------------------------------------------------------
# TestGeneratePattern
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGeneratePattern:
    """Unit tests for the _generate_pattern regex generator function."""

    def test_generate_pattern_simple_phrase_joins_words(self):
        """Simple multi-word phrase produces words joined by \\s+."""
        result = _generate_pattern("buena onda")
        assert "buena" in result
        assert "onda" in result
        assert "\\s+" in result

    def test_generate_pattern_with_article_makes_optional(self):
        """Spanish articles (el, la, etc.) become optional groups in pattern."""
        result = _generate_pattern("tomar el pelo")
        # 'el' should be wrapped in an optional group
        assert "el" in result
        assert "?" in result  # optional marker

    def test_generate_pattern_verb_stem_flexibility(self):
        """First word ending in -ar/-er/-ir gets stem + \\w* for conjugation flexibility."""
        result = _generate_pattern("tomar el pelo")
        # 'tomar' -> stem 'tom' + \w*
        assert "tom" in result
        assert "\\w*" in result
        # Should NOT contain the full word 'tomar' as a literal
        assert "tomar" not in result

    def test_generate_pattern_verb_arse_ending(self):
        """Reflexive verb ending in -arse gets stem extracted."""
        result = _generate_pattern("quedarse dormido")
        assert "qued" in result
        assert "\\w*" in result

    def test_generate_pattern_short_first_word_no_stem(self):
        """First word with 3 or fewer chars is not stem-expanded."""
        result = _generate_pattern("dar en el clavo")
        # 'dar' is only 3 chars, so len(word) > 3 is False; treated as literal
        assert "dar" in result

    def test_generate_pattern_empty_string_returns_escaped(self):
        """Empty or whitespace-only phrase returns re.escape of the input."""
        result = _generate_pattern("")
        # With empty input, words list is empty, so returns re.escape("")
        assert result == ""

    def test_generate_pattern_single_word_verb(self):
        """Single verb word gets stem flexibility."""
        result = _generate_pattern("hablar")
        assert "habl" in result
        assert "\\w*" in result

    def test_generate_pattern_multiple_optional_words(self):
        """Multiple articles/prepositions each become optional."""
        result = _generate_pattern("estar en las nubes")
        # 'en' and 'las' are optional_words
        assert result.count("?") >= 2


# ---------------------------------------------------------------------------
# TestVocabRoutes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVocabRoutes:
    """Tests for /api/vocab endpoints."""

    async def test_create_vocab_success_returns_new_id(self, client, mock_db):
        """POST /api/vocab with valid term creates entry and returns id."""
        cursor_mock = MagicMock()
        cursor_mock.lastrowid = 7
        mock_db.execute.return_value = cursor_mock
        mock_db.execute_fetchall.return_value = []  # no duplicate

        resp = await client.post("/api/vocab", json={
            "term": "hola", "meaning": "hello",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 7
        assert body["duplicate"] is False

    async def test_create_vocab_empty_term_returns_400(self, client):
        """POST /api/vocab with blank term returns 400."""
        resp = await client.post("/api/vocab", json={
            "term": "   ", "meaning": "nothing",
        })
        assert resp.status_code == 400

    async def test_create_vocab_duplicate_increments_encounter(self, client, mock_db):
        """POST /api/vocab with existing term bumps times_encountered."""
        mock_db.execute_fetchall.return_value = [
            {"id": 3, "times_encountered": 2},
        ]

        resp = await client.post("/api/vocab", json={
            "term": "hola", "meaning": "hello",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["duplicate"] is True
        assert body["times_encountered"] == 3
        assert body["id"] == 3

    async def test_list_vocab_returns_service_result(self, client, mock_vocab_service):
        """GET /api/vocab delegates to VocabService.get_all."""
        mock_vocab_service.get_all.return_value = [
            {"id": 1, "term": "hola", "meaning": "hello"},
        ]

        resp = await client.get("/api/vocab")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["term"] == "hola"
        mock_vocab_service.get_all.assert_awaited_once_with(limit=50, offset=0, category=None)

    async def test_list_vocab_with_category_filter(self, client, mock_vocab_service):
        """GET /api/vocab?category=idiom passes category to service."""
        mock_vocab_service.get_all.return_value = []

        resp = await client.get("/api/vocab?category=idiom&limit=10&offset=5")

        assert resp.status_code == 200
        mock_vocab_service.get_all.assert_awaited_once_with(limit=10, offset=5, category="idiom")

    async def test_search_vocab_delegates_to_service(self, client, mock_vocab_service):
        """GET /api/vocab/search?q=test calls VocabService.search."""
        mock_vocab_service.search.return_value = [
            {"id": 1, "term": "test", "meaning": "prueba"},
        ]

        resp = await client.get("/api/vocab/search?q=test")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        mock_vocab_service.search.assert_awaited_once_with("test", limit=20)

    async def test_review_vocab_valid_quality_returns_updated(self, client, mock_vocab_service):
        """POST /api/vocab/1/review with quality 0-5 succeeds."""
        resp = await client.post("/api/vocab/1/review", json={"quality": 4})

        assert resp.status_code == 200
        body = resp.json()
        assert "ease_factor" in body
        mock_vocab_service.record_review.assert_awaited_once_with(1, 4)

    async def test_review_vocab_quality_zero_succeeds(self, client, mock_vocab_service):
        """POST /api/vocab/1/review with quality 0 (forgot) is valid."""
        resp = await client.post("/api/vocab/1/review", json={"quality": 0})
        assert resp.status_code == 200

    async def test_review_vocab_invalid_quality_negative_returns_400(self, client):
        """POST /api/vocab/1/review with quality < 0 returns 400."""
        resp = await client.post("/api/vocab/1/review", json={"quality": -1})
        assert resp.status_code == 400

    async def test_review_vocab_invalid_quality_above_5_returns_400(self, client):
        """POST /api/vocab/1/review with quality > 5 returns 400."""
        resp = await client.post("/api/vocab/1/review", json={"quality": 6})
        assert resp.status_code == 400

    async def test_delete_vocab_success_returns_true(self, client, mock_vocab_service):
        """DELETE /api/vocab/1 returns deleted: true when item exists."""
        mock_vocab_service.delete.return_value = True

        resp = await client.delete("/api/vocab/1")

        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    async def test_delete_vocab_not_found_returns_404(self, client, mock_vocab_service):
        """DELETE /api/vocab/999 returns 404 when item does not exist."""
        mock_vocab_service.delete.return_value = False

        resp = await client.delete("/api/vocab/999")

        assert resp.status_code == 404

    async def test_export_anki_returns_tsv(self, client, mock_vocab_service):
        """GET /api/vocab/export/anki returns TSV content."""
        mock_vocab_service.export_anki_csv.return_value = "front\tback\ttags\n"

        resp = await client.get("/api/vocab/export/anki")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/tab-separated-values")
        assert "habla-vocab.tsv" in resp.headers.get("content-disposition", "")
        assert resp.text == "front\tback\ttags\n"

    async def test_export_json_returns_list(self, client, mock_vocab_service):
        """GET /api/vocab/export/json returns JSON list."""
        mock_vocab_service.export_json.return_value = [{"id": 1, "term": "hola"}]

        resp = await client.get("/api/vocab/export/json")

        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_vocab_stats_returns_counts(self, client, mock_vocab_service):
        """GET /api/vocab/stats returns stat summary."""
        mock_vocab_service.get_stats.return_value = {
            "total": 15, "due_for_review": 3, "by_category": {"idiom": 10, "phrase": 5},
        }

        resp = await client.get("/api/vocab/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 15
        assert body["due_for_review"] == 3

    async def test_vocab_due_returns_list(self, client, mock_vocab_service):
        """GET /api/vocab/due returns items due for review."""
        mock_vocab_service.get_due_for_review.return_value = [{"id": 1, "term": "hola"}]

        resp = await client.get("/api/vocab/due")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1


# ---------------------------------------------------------------------------
# TestSystemRoutes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSystemRoutes:
    """Tests for /api/system endpoints."""

    async def test_status_not_initialized_returns_not_initialized(self, client):
        """GET /api/system/status without pipeline returns not initialized."""
        set_pipeline(None)
        try:
            resp = await client.get("/api/system/status")
            assert resp.status_code == 200
            assert resp.json()["status"] == "not initialized"
        finally:
            set_pipeline(None)

    async def test_status_initialized_returns_full_status(self, client, _set_pipeline):
        """GET /api/system/status with pipeline returns detailed status."""
        with patch("server.main.app_config") as mock_config:
            mock_config.recording.enabled = False

            resp = await client.get("/api/system/status")

            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ready"
            assert body["direction"] == "es_to_en"
            assert body["mode"] == "conversation"
            assert body["idiom_patterns_loaded"] == 42
            assert body["llm_provider"] == "ollama"
            assert body["llm_model"] == "qwen3:4b"

    async def test_set_direction_valid_es_to_en(self, client, _set_pipeline):
        """POST /api/system/direction with valid direction succeeds."""
        resp = await client.post("/api/system/direction", json={"direction": "es_to_en"})

        assert resp.status_code == 200
        assert resp.json()["direction"] == "es_to_en"
        _set_pipeline.set_direction.assert_called_once_with("es_to_en")

    async def test_set_direction_valid_en_to_es(self, client, _set_pipeline):
        """POST /api/system/direction with en_to_es succeeds."""
        resp = await client.post("/api/system/direction", json={"direction": "en_to_es"})

        assert resp.status_code == 200
        assert resp.json()["direction"] == "en_to_es"

    async def test_set_direction_invalid_returns_400(self, client, _set_pipeline):
        """POST /api/system/direction with bogus value returns 400."""
        resp = await client.post("/api/system/direction", json={"direction": "fr_to_de"})

        assert resp.status_code == 400
        assert "es_to_en" in resp.json()["detail"]

    async def test_set_mode_valid_conversation(self, client, _set_pipeline):
        """POST /api/system/mode with 'conversation' succeeds."""
        resp = await client.post("/api/system/mode", json={"mode": "conversation"})

        assert resp.status_code == 200
        assert resp.json()["mode"] == "conversation"

    async def test_set_mode_valid_classroom(self, client, _set_pipeline):
        """POST /api/system/mode with 'classroom' succeeds."""
        resp = await client.post("/api/system/mode", json={"mode": "classroom"})

        assert resp.status_code == 200
        assert resp.json()["mode"] == "classroom"
        _set_pipeline.set_mode.assert_called_once_with("classroom")

    async def test_set_mode_invalid_returns_400(self, client, _set_pipeline):
        """POST /api/system/mode with unknown mode returns 400."""
        resp = await client.post("/api/system/mode", json={"mode": "debate"})

        assert resp.status_code == 400
        assert "conversation" in resp.json()["detail"]

    async def test_set_asr_language_no_pipeline_returns_503(self, client):
        """POST /api/system/asr/language without pipeline returns 503."""
        set_pipeline(None)
        try:
            resp = await client.post("/api/system/asr/language", json={"auto_language": True})
            assert resp.status_code == 503
        finally:
            set_pipeline(None)

    async def test_set_asr_language_success(self, client, _set_pipeline):
        """POST /api/system/asr/language toggles auto_language."""
        resp = await client.post("/api/system/asr/language", json={"auto_language": False})

        assert resp.status_code == 200
        body = resp.json()
        assert "auto_language" in body

    async def test_rename_speaker_success(self, client, _set_pipeline):
        """PUT /api/system/speakers/{id} renames speaker."""
        resp = await client.put(
            "/api/system/speakers/SPEAKER_00",
            json={"name": "Carlos", "role": "student"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["custom_name"] == "Carlos"
        _set_pipeline.speaker_tracker.rename.assert_called_once_with("SPEAKER_00", "Carlos")
        _set_pipeline.speaker_tracker.set_role_hint.assert_called_once_with("SPEAKER_00", "student")

    async def test_rename_speaker_not_found_returns_404(self, client, _set_pipeline):
        """PUT /api/system/speakers/{id} returns 404 when speaker unknown."""
        _set_pipeline.speaker_tracker.rename.return_value = None

        resp = await client.put(
            "/api/system/speakers/SPEAKER_99",
            json={"name": "Nobody"},
        )

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestSessionRoutes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSessionRoutes:
    """Tests for /api/sessions endpoints."""

    async def test_list_sessions_returns_rows(self, client, mock_db):
        """GET /api/sessions returns list of sessions."""
        mock_db.execute_fetchall.return_value = [
            {"id": 1, "started_at": "2026-02-20", "exchange_count": 5},
            {"id": 2, "started_at": "2026-02-21", "exchange_count": 10},
        ]

        resp = await client.get("/api/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == 1

    async def test_list_sessions_with_pagination(self, client, mock_db):
        """GET /api/sessions?limit=1&offset=1 passes params to query."""
        mock_db.execute_fetchall.return_value = [
            {"id": 2, "started_at": "2026-02-21", "exchange_count": 10},
        ]

        resp = await client.get("/api/sessions?limit=1&offset=1")

        assert resp.status_code == 200
        # Verify the SQL received limit=1, offset=1
        call_args = mock_db.execute_fetchall.call_args
        assert call_args[0][1] == (1, 1)

    async def test_get_session_not_found_returns_404(self, client, mock_db):
        """GET /api/sessions/999 returns 404 when session does not exist."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.get("/api/sessions/999")

        assert resp.status_code == 404

    async def test_get_session_success_includes_speakers(self, client, mock_db):
        """GET /api/sessions/1 returns session with speakers and exchange_count."""
        call_count = 0

        async def multi_fetchall(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Session query
                return [{"id": 1, "started_at": "2026-02-20", "notes": ""}]
            elif call_count == 2:
                # Speakers query
                return [{"id": "SPEAKER_00", "session_id": 1, "auto_label": "A"}]
            else:
                # Exchange count
                return [{"c": 5}]

        mock_db.execute_fetchall = multi_fetchall

        resp = await client.get("/api/sessions/1")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 1
        assert len(body["speakers"]) == 1
        assert body["exchange_count"] == 5

    async def test_get_session_exchanges_not_found_returns_404(self, client, mock_db):
        """GET /api/sessions/999/exchanges returns 404 for nonexistent session."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.get("/api/sessions/999/exchanges")

        assert resp.status_code == 404

    async def test_get_session_exchanges_success(self, client, mock_db):
        """GET /api/sessions/1/exchanges returns exchange list."""
        call_count = 0

        async def multi_fetchall(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Session exists check
                return [{"id": 1}]
            else:
                # Exchanges
                return [
                    {"id": 1, "raw_transcript": "Hola", "translation": "Hello", "correction_json": None},
                ]

        mock_db.execute_fetchall = multi_fetchall

        resp = await client.get("/api/sessions/1/exchanges")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["raw_transcript"] == "Hola"

    async def test_save_session_not_found_returns_404(self, client, mock_db):
        """POST /api/sessions/999/save returns 404 for nonexistent session."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.post("/api/sessions/999/save", json={})

        assert resp.status_code == 404

    async def test_save_session_success(self, client, mock_db):
        """POST /api/sessions/1/save marks session as saved."""
        mock_db.execute_fetchall.return_value = [{"id": 1}]

        resp = await client.post("/api/sessions/1/save", json={"notes": "Good lesson"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "saved"
        assert body["session_id"] == 1

    async def test_delete_session_success_returns_deleted(self, client, mock_db):
        """DELETE /api/sessions/1 deletes session and related data."""
        mock_db.execute_fetchall.return_value = [{"id": 1}]

        resp = await client.delete("/api/sessions/1")

        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] is True
        assert body["session_id"] == 1
        # Should delete from quality_metrics, exchanges, speakers, sessions
        assert mock_db.execute.await_count == 4
        mock_db.commit.assert_awaited_once()

    async def test_delete_session_not_found_returns_404(self, client, mock_db):
        """DELETE /api/sessions/999 returns 404 when session does not exist."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.delete("/api/sessions/999")

        assert resp.status_code == 404

    async def test_export_session_not_found_returns_404(self, client, mock_db):
        """GET /api/sessions/999/export returns 404 for nonexistent session."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.get("/api/sessions/999/export")

        assert resp.status_code == 404

    async def test_export_session_success_returns_plaintext(self, client, mock_db):
        """GET /api/sessions/1/export returns plain text transcript."""
        call_count = 0

        async def multi_fetchall(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Session query
                return [{"id": 1, "started_at": "2026-02-20", "notes": "Test", "topic_summary": "", "direction": "es_to_en"}]
            else:
                # Exchanges with speaker info
                return [
                    {
                        "id": 1, "raw_transcript": "Hola", "corrected_source": "Hola",
                        "translation": "Hello", "timestamp": "10:00:00",
                        "custom_name": "Carlos", "auto_label": "A",
                    },
                ]

        mock_db.execute_fetchall = multi_fetchall

        resp = await client.get("/api/sessions/1/export")

        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "Hola" in resp.text
        assert "Hello" in resp.text
        assert "Carlos" in resp.text


# ---------------------------------------------------------------------------
# TestIdiomRoutes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIdiomRoutes:
    """Tests for /api/idioms endpoints."""

    async def test_list_idioms_returns_rows(self, client, mock_db):
        """GET /api/idioms returns list of idiom patterns."""
        mock_db.execute_fetchall.return_value = [
            {"id": 1, "canonical": "tomar el pelo", "meaning": "to pull someone's leg"},
        ]

        resp = await client.get("/api/idioms")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["canonical"] == "tomar el pelo"

    async def test_create_idiom_pattern_success(self, client, mock_db, _set_pipeline):
        """POST /api/idioms creates pattern and returns id."""
        cursor_mock = MagicMock()
        cursor_mock.lastrowid = 5
        mock_db.execute.return_value = cursor_mock
        mock_db.execute_fetchall.return_value = []  # no duplicate

        resp = await client.post("/api/idioms", json={
            "phrase": "tomar el pelo",
            "meaning": "to pull someone's leg",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 5
        assert body["canonical"] == "tomar el pelo"
        assert body["pattern"]  # auto-generated regex
        assert body["total_patterns"] == 42
        _set_pipeline.reload_idiom_patterns.assert_awaited_once()

    async def test_create_idiom_duplicate_returns_409(self, client, mock_db, _set_pipeline):
        """POST /api/idioms with existing canonical returns 409."""
        mock_db.execute_fetchall.return_value = [{"id": 3}]

        resp = await client.post("/api/idioms", json={
            "phrase": "tomar el pelo",
            "meaning": "to pull someone's leg",
        })

        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    async def test_create_idiom_duplicate_in_scanner_returns_409(self, client, mock_db, _set_pipeline):
        """POST /api/idioms with canonical matching scanner patterns returns 409."""
        mock_db.execute_fetchall.return_value = []  # not in DB

        pattern_mock = MagicMock()
        pattern_mock.canonical = "tomar el pelo"
        _set_pipeline.idiom_scanner.patterns = [pattern_mock]

        resp = await client.post("/api/idioms", json={
            "phrase": "tomar el pelo",
            "meaning": "to pull someone's leg",
        })

        assert resp.status_code == 409
        assert "pattern files" in resp.json()["detail"]

    async def test_create_idiom_invalid_regex_returns_400(self, client, mock_db, _set_pipeline):
        """POST /api/idioms with invalid manual regex returns 400."""
        mock_db.execute_fetchall.return_value = []

        resp = await client.post("/api/idioms", json={
            "phrase": "test",
            "meaning": "test",
            "pattern": "[invalid(",  # bad regex
        })

        assert resp.status_code == 400
        assert "Invalid regex" in resp.json()["detail"]

    async def test_create_idiom_with_manual_pattern(self, client, mock_db, _set_pipeline):
        """POST /api/idioms with manual pattern uses it instead of auto-generating."""
        cursor_mock = MagicMock()
        cursor_mock.lastrowid = 6
        mock_db.execute.return_value = cursor_mock
        mock_db.execute_fetchall.return_value = []

        custom_pattern = r"tom\w+\s+el\s+pelo"
        resp = await client.post("/api/idioms", json={
            "phrase": "tomar el pelo",
            "meaning": "to pull someone's leg",
            "pattern": custom_pattern,
        })

        assert resp.status_code == 200
        assert resp.json()["pattern"] == custom_pattern

    async def test_delete_idiom_success(self, client, mock_db, _set_pipeline):
        """DELETE /api/idioms/1 deletes and reloads scanner."""
        cursor_mock = MagicMock()
        cursor_mock.rowcount = 1
        mock_db.execute.return_value = cursor_mock

        resp = await client.delete("/api/idioms/1")

        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        _set_pipeline.reload_idiom_patterns.assert_awaited_once()

    async def test_delete_idiom_not_found_returns_404(self, client, mock_db):
        """DELETE /api/idioms/999 returns 404 when pattern does not exist."""
        cursor_mock = MagicMock()
        cursor_mock.rowcount = 0
        mock_db.execute.return_value = cursor_mock

        resp = await client.delete("/api/idioms/999")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestLLMRoutes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMRoutes:
    """Tests for /api/llm endpoints."""

    async def test_llm_current_not_initialized_returns_status(self, client):
        """GET /api/llm/current without pipeline returns not initialized."""
        set_pipeline(None)
        try:
            resp = await client.get("/api/llm/current")
            assert resp.status_code == 200
            assert resp.json()["status"] == "not initialized"
        finally:
            set_pipeline(None)

    async def test_llm_current_initialized_returns_provider_info(self, client, _set_pipeline):
        """GET /api/llm/current with pipeline returns provider and model."""
        resp = await client.get("/api/llm/current")

        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "ollama"
        assert body["model"] == "qwen3:4b"
        assert "metrics" in body

    async def test_llm_providers_not_initialized_returns_empty(self, client):
        """GET /api/llm/providers without pipeline returns empty list."""
        set_pipeline(None)
        try:
            resp = await client.get("/api/llm/providers")
            assert resp.status_code == 200
            assert resp.json()["providers"] == []
        finally:
            set_pipeline(None)

    async def test_llm_select_invalid_provider_returns_400(self, client, _set_pipeline):
        """POST /api/llm/select with unknown provider returns 400."""
        resp = await client.post("/api/llm/select", json={
            "provider": "gemini", "model": "gemini-pro",
        })

        assert resp.status_code == 400
        assert "ollama" in resp.json()["detail"]

    async def test_llm_select_no_pipeline_returns_503(self, client):
        """POST /api/llm/select without pipeline returns 503."""
        set_pipeline(None)
        try:
            resp = await client.post("/api/llm/select", json={
                "provider": "ollama", "model": "qwen3:4b",
            })
            assert resp.status_code == 503
        finally:
            set_pipeline(None)

    async def test_llm_select_openai_no_key_returns_400(self, client, _set_pipeline):
        """POST /api/llm/select for openai without API key returns 400."""
        _set_pipeline.translator.config.openai_api_key = ""

        resp = await client.post("/api/llm/select", json={
            "provider": "openai", "model": "gpt-5-nano",
        })

        assert resp.status_code == 400
        assert "OPENAI_API_KEY" in resp.json()["detail"]

    async def test_llm_select_valid_provider_switches(self, client, _set_pipeline):
        """POST /api/llm/select with valid provider switches translator."""
        resp = await client.post("/api/llm/select", json={
            "provider": "ollama",
        })

        assert resp.status_code == 200
        _set_pipeline.translator.switch_provider.assert_called_once()

    async def test_llm_models_no_pipeline_returns_503(self, client):
        """GET /api/llm/models?provider=ollama without pipeline returns 503."""
        set_pipeline(None)
        try:
            resp = await client.get("/api/llm/models?provider=ollama")
            assert resp.status_code == 503
        finally:
            set_pipeline(None)

    async def test_llm_models_unknown_provider_returns_400(self, client, _set_pipeline):
        """GET /api/llm/models?provider=gemini returns 400."""
        resp = await client.get("/api/llm/models?provider=gemini")
        assert resp.status_code == 400

    async def test_llm_models_openai_no_key_returns_400(self, client, _set_pipeline):
        """GET /api/llm/models?provider=openai without key returns 400."""
        _set_pipeline.translator.config.openai_api_key = ""

        resp = await client.get("/api/llm/models?provider=openai")
        assert resp.status_code == 400

    async def test_llm_models_openai_with_key_returns_list(self, client, _set_pipeline):
        """GET /api/llm/models?provider=openai with key returns model list."""
        _set_pipeline.translator.config.openai_api_key = "sk-test"

        resp = await client.get("/api/llm/models?provider=openai")

        assert resp.status_code == 200
        models = resp.json()["models"]
        assert "gpt-5-nano" in models

    async def test_llm_costs_not_initialized_returns_null(self, client):
        """GET /api/llm/costs without pipeline returns costs: null."""
        set_pipeline(None)
        try:
            resp = await client.get("/api/llm/costs")
            assert resp.status_code == 200
            assert resp.json()["costs"] is None
        finally:
            set_pipeline(None)

    async def test_llm_costs_local_provider_returns_free(self, client, _set_pipeline):
        """GET /api/llm/costs with ollama returns 'Free (local)'."""
        resp = await client.get("/api/llm/costs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["cost_tracking"] == "Free (local)"


# ---------------------------------------------------------------------------
# TestLMStudioRoutes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLMStudioRoutes:
    """Tests for /api/lmstudio endpoints."""

    async def test_lmstudio_status_no_manager_returns_not_active(self, client):
        """GET /api/lmstudio/status without manager returns running=False."""
        set_lmstudio_manager(None)

        resp = await client.get("/api/lmstudio/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] is False
        assert "not active" in body["note"]

    async def test_lmstudio_status_with_manager_returns_running(self, client, _set_lmstudio_manager):
        """GET /api/lmstudio/status with running manager returns status."""
        resp = await client.get("/api/lmstudio/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] is True
        assert "test-model" in body["models"]

    async def test_lmstudio_restart_no_manager_returns_503(self, client):
        """POST /api/lmstudio/restart without manager returns 503."""
        set_lmstudio_manager(None)

        resp = await client.post("/api/lmstudio/restart")

        assert resp.status_code == 503

    async def test_lmstudio_restart_with_manager_returns_restarting(self, client, _set_lmstudio_manager):
        """POST /api/lmstudio/restart with manager kicks off restart."""
        resp = await client.post("/api/lmstudio/restart")

        assert resp.status_code == 200
        assert resp.json()["status"] == "restarting"

    async def test_lmstudio_models_no_manager_returns_503(self, client):
        """GET /api/lmstudio/models without manager returns 503."""
        set_lmstudio_manager(None)

        resp = await client.get("/api/lmstudio/models")

        assert resp.status_code == 503
