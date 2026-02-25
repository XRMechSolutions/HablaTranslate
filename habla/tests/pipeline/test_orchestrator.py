"""Unit tests for the PipelineOrchestrator."""

import asyncio
import json
import pytest
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from server.config import AppConfig, TranslatorConfig, RecordingConfig, SessionConfig
from server.models.schemas import (
    Exchange, FlaggedPhrase, SpeakerProfile, TranslationResult,
    WSTranslation, WSPartialTranscript, CorrectionDetail,
)
from server.pipeline.orchestrator import PipelineOrchestrator, _is_bad_transcript
from server.services.idiom_scanner import IdiomMatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_config(tmp_path):
    """AppConfig with tmp_path for data/db, recording disabled."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "idioms").mkdir()
    return AppConfig(
        data_dir=data_dir,
        db_path=tmp_path / "test.db",
        translator=TranslatorConfig(provider="ollama"),
        recording=RecordingConfig(enabled=False),
        session=SessionConfig(direction="es_to_en", mode="conversation"),
    )


def _make_translation_result(**overrides):
    """Build a TranslationResult with sensible defaults."""
    defaults = dict(
        corrected="Corrected text",
        translated="Translated text",
        flagged_phrases=[],
        confidence=0.9,
        speaker_hint=None,
        is_correction=False,
        correction_detail=None,
    )
    defaults.update(overrides)
    return TranslationResult(**defaults)


@pytest.fixture
def orchestrator(app_config):
    """A PipelineOrchestrator with the translator mocked out and marked ready."""
    orch = PipelineOrchestrator(app_config)
    orch.translator = MagicMock()
    orch.translator.translate = AsyncMock(return_value=_make_translation_result())
    orch.translator.update_topic_summary = AsyncMock(return_value="updated summary")
    orch.translator.close = AsyncMock()
    orch.translator.metrics = {"requests": 0, "successes": 0, "failures": 0}
    orch._ready = True
    return orch


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Verify constructor defaults and initial state."""

    def test_init_sets_direction_from_config(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert orch.direction == "es_to_en"

    def test_init_sets_mode_from_config(self, app_config):
        app_config.session.mode = "classroom"
        orch = PipelineOrchestrator(app_config)
        assert orch.mode == "classroom"

    def test_init_not_ready(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert orch.ready is False

    def test_init_empty_exchanges(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert len(orch.recent_exchanges) == 0

    def test_init_topic_summary_empty(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert orch.topic_summary == ""

    def test_init_session_id_none(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert orch.session_id is None

    def test_init_queue_maxsize(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert orch._queue.maxsize == 5

    def test_init_language_votes_maxlen(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert orch._language_votes.maxlen == 5

    def test_init_callbacks_none(self, app_config):
        orch = PipelineOrchestrator(app_config)
        assert orch._on_translation is None
        assert orch._on_partial is None
        assert orch._on_speakers is None
        assert orch._on_final_transcript is None
        assert orch._on_error is None


# ---------------------------------------------------------------------------
# TestDirectionMode
# ---------------------------------------------------------------------------

class TestDirectionMode:
    """Test direction/mode switching and the _source_language property."""

    def test_source_language_es_to_en(self, orchestrator):
        orchestrator.direction = "es_to_en"
        assert orchestrator._source_language == "es"

    def test_source_language_en_to_es(self, orchestrator):
        orchestrator.direction = "en_to_es"
        assert orchestrator._source_language == "en"

    def test_set_direction_updates_direction(self, orchestrator):
        orchestrator.set_direction("en_to_es")
        assert orchestrator.direction == "en_to_es"

    def test_set_direction_clears_detected_language(self, orchestrator):
        orchestrator._last_detected_language = "es"
        orchestrator.set_direction("en_to_es")
        assert orchestrator._last_detected_language is None

    def test_set_mode_updates_mode(self, orchestrator):
        orchestrator.set_mode("classroom")
        assert orchestrator.mode == "classroom"


# ---------------------------------------------------------------------------
# TestCallbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    """Test callback registration and invocation."""

    def test_set_callbacks_stores_all(self, orchestrator):
        on_t = AsyncMock()
        on_p = AsyncMock()
        on_s = AsyncMock()
        on_ft = AsyncMock()
        on_e = AsyncMock()
        orchestrator.set_callbacks(
            on_translation=on_t,
            on_partial=on_p,
            on_speakers=on_s,
            on_final_transcript=on_ft,
            on_error=on_e,
        )
        assert orchestrator._on_translation is on_t
        assert orchestrator._on_partial is on_p
        assert orchestrator._on_speakers is on_s
        assert orchestrator._on_final_transcript is on_ft
        assert orchestrator._on_error is on_e

    @pytest.mark.asyncio
    async def test_process_text_fires_translation_callback(self, orchestrator):
        on_t = AsyncMock()
        orchestrator.set_callbacks(on_translation=on_t)

        await orchestrator.process_text("Hola mundo")

        on_t.assert_awaited_once()
        msg = on_t.call_args[0][0]
        assert isinstance(msg, WSTranslation)
        assert msg.source == "Hola mundo"
        assert msg.translated == "Translated text"

    @pytest.mark.asyncio
    async def test_process_text_no_callback_no_error(self, orchestrator):
        """process_text should work fine even when no callbacks are registered."""
        exchange = await orchestrator.process_text("Hola")
        assert exchange.translation == "Translated text"

    @pytest.mark.asyncio
    async def test_translate_and_notify_fires_error_callback_on_failure(self, orchestrator):
        orchestrator.translator.translate = AsyncMock(side_effect=RuntimeError("LLM down"))
        on_error = AsyncMock()
        orchestrator.set_callbacks(on_error=on_error)

        await orchestrator._translate_and_notify("Hola", "SPEAKER_00", 0.0)

        on_error.assert_awaited_once()
        error_msg = on_error.call_args[0][0]
        assert "Translation failed" in error_msg


# ---------------------------------------------------------------------------
# TestSessionLifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    """Test create/close/reset session database interactions."""

    @pytest.mark.asyncio
    async def test_create_session_returns_id(self, orchestrator):
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 42
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            session_id = await orchestrator.create_session()

        assert session_id == 42
        assert orchestrator.session_id == 42

    @pytest.mark.asyncio
    async def test_create_session_db_failure_returns_zero(self, orchestrator):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("DB locked"))

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            session_id = await orchestrator.create_session()

        assert session_id == 0

    @pytest.mark.asyncio
    async def test_close_session_noop_without_session_id(self, orchestrator):
        orchestrator.session_id = None
        # Should not raise
        await orchestrator.close_session()

    @pytest.mark.asyncio
    async def test_close_session_persists_speakers(self, orchestrator):
        orchestrator.session_id = 1
        orchestrator.speaker_tracker.record_utterance("SPEAKER_00")
        orchestrator.speaker_tracker.record_utterance("SPEAKER_01")

        mock_db = AsyncMock()
        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            await orchestrator.close_session()

        # 1 session UPDATE + 2 speaker INSERTs + 1 commit = at least 3 execute calls
        assert mock_db.execute.await_count >= 3

    @pytest.mark.asyncio
    async def test_reset_session_clears_state(self, orchestrator):
        # Populate some state
        orchestrator.recent_exchanges.append({"source": "test"})
        orchestrator.topic_summary = "some topic"
        orchestrator._last_partial_text = "partial"

        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 99
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            orchestrator.session_id = 1
            new_id = await orchestrator.reset_session()

        assert new_id == 99
        assert len(orchestrator.recent_exchanges) == 0
        assert orchestrator.topic_summary == ""
        assert orchestrator._last_partial_text == ""


# ---------------------------------------------------------------------------
# TestProcessText
# ---------------------------------------------------------------------------

class TestProcessText:
    """Test the text-input pipeline (process_text)."""

    @pytest.mark.asyncio
    async def test_process_text_returns_exchange(self, orchestrator):
        exchange = await orchestrator.process_text("Hola mundo")

        assert isinstance(exchange, Exchange)
        assert exchange.raw_transcript == "Hola mundo"
        assert exchange.translation == "Translated text"
        assert exchange.corrected_source == "Corrected text"
        assert exchange.direction == "es_to_en"

    @pytest.mark.asyncio
    async def test_process_text_uses_correct_direction(self, orchestrator):
        orchestrator.set_direction("en_to_es")

        await orchestrator.process_text("Hello world")

        call_kwargs = orchestrator.translator.translate.call_args[1]
        assert call_kwargs["direction"] == "en_to_es"

    @pytest.mark.asyncio
    async def test_process_text_uses_correct_mode(self, orchestrator):
        orchestrator.set_mode("classroom")

        await orchestrator.process_text("Hola")

        call_kwargs = orchestrator.translator.translate.call_args[1]
        assert call_kwargs["mode"] == "classroom"

    @pytest.mark.asyncio
    async def test_process_text_default_speaker_is_manual(self, orchestrator):
        exchange = await orchestrator.process_text("Hola")
        assert exchange.speaker.id == "MANUAL"

    @pytest.mark.asyncio
    async def test_process_text_custom_speaker_id(self, orchestrator):
        exchange = await orchestrator.process_text("Hola", speaker_id="SPEAKER_01")
        assert exchange.speaker.id == "SPEAKER_01"

    @pytest.mark.asyncio
    async def test_process_text_increments_utterance_count(self, orchestrator):
        await orchestrator.process_text("Hola", speaker_id="SPK")
        await orchestrator.process_text("Mundo", speaker_id="SPK")

        profile = orchestrator.speaker_tracker.speakers["SPK"]
        assert profile.utterance_count == 2

    @pytest.mark.asyncio
    async def test_process_text_passes_context_exchanges(self, orchestrator):
        # Pre-populate context
        orchestrator.recent_exchanges.append({"speaker_label": "A", "source": "prev"})

        await orchestrator.process_text("Hola")

        call_kwargs = orchestrator.translator.translate.call_args[1]
        assert len(call_kwargs["context_exchanges"]) == 1
        assert call_kwargs["context_exchanges"][0]["source"] == "prev"

    @pytest.mark.asyncio
    async def test_process_text_passes_topic_summary(self, orchestrator):
        orchestrator.topic_summary = "Discussing weather"

        await orchestrator.process_text("Hola")

        call_kwargs = orchestrator.translator.translate.call_args[1]
        assert call_kwargs["topic_summary"] == "Discussing weather"

    @pytest.mark.asyncio
    async def test_process_text_updates_recent_exchanges(self, orchestrator):
        await orchestrator.process_text("Hola mundo")

        assert len(orchestrator.recent_exchanges) == 1
        entry = orchestrator.recent_exchanges[0]
        assert entry["source"] == "Hola mundo"
        assert entry["corrected"] == "Corrected text"
        assert entry["translated"] == "Translated text"

    @pytest.mark.asyncio
    async def test_process_text_sets_speaker_hint(self, orchestrator):
        orchestrator.translator.translate = AsyncMock(
            return_value=_make_translation_result(speaker_hint="teacher")
        )

        await orchestrator.process_text("Hola", speaker_id="SPK_HINT")

        profile = orchestrator.speaker_tracker.speakers["SPK_HINT"]
        assert profile.role_hint == "teacher"

    @pytest.mark.asyncio
    async def test_process_text_skips_hint_if_custom_name(self, orchestrator):
        orchestrator.translator.translate = AsyncMock(
            return_value=_make_translation_result(speaker_hint="student")
        )
        # Pre-create speaker with custom name
        orchestrator.speaker_tracker.record_utterance("NAMED")
        orchestrator.speaker_tracker.rename("NAMED", "Maria")

        await orchestrator.process_text("Hola", speaker_id="NAMED")

        profile = orchestrator.speaker_tracker.speakers["NAMED"]
        assert profile.role_hint is None, "Should not overwrite when custom_name exists"

    @pytest.mark.asyncio
    async def test_process_text_saves_exchange_to_db(self, orchestrator):
        orchestrator.session_id = 10
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 77
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            exchange = await orchestrator.process_text("Hola")

        assert exchange.id == 77

    @pytest.mark.asyncio
    async def test_process_text_no_save_without_session(self, orchestrator):
        orchestrator.session_id = None
        exchange = await orchestrator.process_text("Hola")
        assert exchange.id is None


# ---------------------------------------------------------------------------
# TestProcessWav
# ---------------------------------------------------------------------------

class TestProcessWav:
    """Test process_wav routing through the queue."""

    @pytest.mark.asyncio
    async def test_process_wav_not_ready_returns_none(self, orchestrator):
        orchestrator._ready = False
        result = await orchestrator.process_wav("/fake/path.wav")
        assert result is None

    @pytest.mark.asyncio
    async def test_process_wav_queues_item(self, orchestrator):
        """process_wav should put an item into the queue and wait for the future."""
        # We won't start the worker; instead, manually consume from the queue
        async def fake_worker():
            kind, payload, future = await orchestrator._queue.get()
            assert kind == "wav"
            assert payload == "/fake/path.wav"
            future.set_result(None)

        worker = asyncio.create_task(fake_worker())
        result = await orchestrator.process_wav("/fake/path.wav")
        await worker
        assert result is None


# ---------------------------------------------------------------------------
# TestContextManagement
# ---------------------------------------------------------------------------

class TestContextManagement:
    """Test recent_exchanges deque behavior and topic summary."""

    @pytest.mark.asyncio
    async def test_recent_exchanges_maxlen_is_10(self, orchestrator):
        assert orchestrator.recent_exchanges.maxlen == 10

    @pytest.mark.asyncio
    async def test_context_window_holds_exactly_10(self, orchestrator):
        for i in range(12):
            await orchestrator.process_text(f"Utterance {i}")

        assert len(orchestrator.recent_exchanges) == 10, "Deque should cap at 10"

    @pytest.mark.asyncio
    async def test_context_window_drops_oldest(self, orchestrator):
        for i in range(12):
            await orchestrator.process_text(f"Utterance {i}")

        oldest = orchestrator.recent_exchanges[0]
        assert oldest["source"] == "Utterance 2", "Oldest two should have been evicted"

    @pytest.mark.asyncio
    async def test_topic_summary_updated_via_fire_and_forget(self, orchestrator):
        """_update_topic is launched as a fire-and-forget task from process_text."""
        orchestrator.translator.update_topic_summary = AsyncMock(return_value="new topic")

        await orchestrator.process_text("Hola mundo")
        # Give the fire-and-forget task time to complete
        await asyncio.sleep(0.05)

        assert orchestrator.topic_summary == "new topic"

    @pytest.mark.asyncio
    async def test_topic_summary_unchanged_on_update_failure(self, orchestrator):
        orchestrator.topic_summary = "old topic"
        orchestrator.translator.update_topic_summary = AsyncMock(side_effect=RuntimeError("fail"))

        await orchestrator.process_text("Hola")
        await asyncio.sleep(0.05)

        assert orchestrator.topic_summary == "old topic"

    @pytest.mark.asyncio
    async def test_topic_summary_unchanged_when_update_returns_none(self, orchestrator):
        orchestrator.topic_summary = "old topic"
        orchestrator.translator.update_topic_summary = AsyncMock(return_value=None)

        await orchestrator.process_text("Hola")
        await asyncio.sleep(0.05)

        assert orchestrator.topic_summary == "old topic"

    def test_reset_partial_state_clears_last_partial(self, orchestrator):
        orchestrator._last_partial_text = "some partial"
        orchestrator.reset_partial_state()
        assert orchestrator._last_partial_text == ""


# ---------------------------------------------------------------------------
# TestIdiomMerge
# ---------------------------------------------------------------------------

class TestIdiomMerge:
    """Test _merge_idioms dedup logic with pattern DB priority."""

    def test_merge_empty_lists(self, orchestrator):
        result = orchestrator._merge_idioms([], [])
        assert result == []

    def test_merge_pattern_only(self, orchestrator):
        patterns = [
            IdiomMatch(
                canonical="tomar el pelo",
                literal="to take the hair",
                meaning="to pull someone's leg",
                region="spain",
                match_start=0,
                match_end=14,
            )
        ]
        result = orchestrator._merge_idioms(patterns, [])
        assert len(result) == 1
        assert result[0].phrase == "tomar el pelo"
        assert result[0].source == "pattern_db"
        assert result[0].save_worthy is True

    def test_merge_llm_only(self, orchestrator):
        llm_phrases = [
            FlaggedPhrase(phrase="dar en el clavo", meaning="to hit the nail", source="llm")
        ]
        result = orchestrator._merge_idioms([], llm_phrases)
        assert len(result) == 1
        assert result[0].source == "llm"

    def test_merge_dedup_pattern_takes_priority(self, orchestrator):
        """When both pattern DB and LLM detect the same idiom, pattern DB wins."""
        patterns = [
            IdiomMatch(
                canonical="Tomar el pelo",
                literal="to take the hair",
                meaning="to pull someone's leg (DB)",
                match_start=0,
                match_end=14,
            )
        ]
        llm_phrases = [
            FlaggedPhrase(phrase="tomar el pelo", meaning="to pull someone's leg (LLM)", source="llm")
        ]
        result = orchestrator._merge_idioms(patterns, llm_phrases)
        assert len(result) == 1, "Should dedup to one entry"
        assert result[0].source == "pattern_db", "Pattern DB should take priority"
        assert "DB" in result[0].meaning

    def test_merge_case_insensitive_dedup(self, orchestrator):
        patterns = [
            IdiomMatch(canonical="Tomar El Pelo", literal="lit", meaning="m1", match_start=0, match_end=14)
        ]
        llm_phrases = [
            FlaggedPhrase(phrase="tomar el pelo", meaning="m2", source="llm")
        ]
        result = orchestrator._merge_idioms(patterns, llm_phrases)
        assert len(result) == 1

    def test_merge_different_idioms_kept(self, orchestrator):
        patterns = [
            IdiomMatch(canonical="idiom A", literal="lit A", meaning="m A", match_start=0, match_end=7)
        ]
        llm_phrases = [
            FlaggedPhrase(phrase="idiom B", meaning="m B", source="llm")
        ]
        result = orchestrator._merge_idioms(patterns, llm_phrases)
        assert len(result) == 2

    def test_merge_pattern_preserves_span(self, orchestrator):
        patterns = [
            IdiomMatch(canonical="test", literal="lit", meaning="m", match_start=5, match_end=9)
        ]
        result = orchestrator._merge_idioms(patterns, [])
        assert result[0].span_start == 5
        assert result[0].span_end == 9

    def test_merge_pattern_preserves_region(self, orchestrator):
        patterns = [
            IdiomMatch(canonical="test", literal="lit", meaning="m", region="spain", match_start=0, match_end=4)
        ]
        result = orchestrator._merge_idioms(patterns, [])
        assert result[0].region == "spain"


# ---------------------------------------------------------------------------
# TestLanguageDetection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    """Test _update_detected_language voting and supermajority logic."""

    def test_first_detection_accepted(self, orchestrator):
        orchestrator._last_detected_language = None
        orchestrator._update_detected_language("es")
        assert orchestrator._last_detected_language == "es"

    def test_same_language_stays(self, orchestrator):
        orchestrator._last_detected_language = "es"
        orchestrator._language_votes.clear()
        for _ in range(3):
            orchestrator._update_detected_language("es")
        assert orchestrator._last_detected_language == "es"

    def test_switch_requires_supermajority(self, orchestrator):
        """A single disagreeing detection should not switch the language."""
        orchestrator._last_detected_language = "es"
        orchestrator._language_votes.clear()
        orchestrator._language_votes.extend(["es", "es", "es"])

        orchestrator._update_detected_language("en")

        # 1 en out of 4 = 25%, below 70% threshold
        assert orchestrator._last_detected_language == "es"

    def test_switch_happens_at_supermajority(self, orchestrator):
        """Language should switch when the new language reaches 70%+ of votes."""
        orchestrator._last_detected_language = "es"
        orchestrator._language_votes.clear()

        # Fill deque (maxlen=5) with enough "en" votes
        for _ in range(4):
            orchestrator._update_detected_language("en")

        # 4 out of 4 = 100% -> should switch
        assert orchestrator._last_detected_language == "en"

    def test_votes_deque_capped_at_maxlen(self, orchestrator):
        orchestrator._last_detected_language = "es"
        orchestrator._language_votes.clear()

        for _ in range(10):
            orchestrator._update_detected_language("es")

        assert len(orchestrator._language_votes) == 5

    def test_gradual_switch(self, orchestrator):
        """Simulate a realistic gradual language switch."""
        orchestrator._last_detected_language = "es"
        orchestrator._language_votes.clear()

        # Start with Spanish majority
        orchestrator._update_detected_language("es")
        orchestrator._update_detected_language("es")
        orchestrator._update_detected_language("en")  # stray detection
        assert orchestrator._last_detected_language == "es"

        # Now push English to supermajority
        orchestrator._update_detected_language("en")
        orchestrator._update_detected_language("en")
        # Votes now: [es, es, en, en, en] -> 60% en, not enough
        assert orchestrator._last_detected_language == "es"

        orchestrator._update_detected_language("en")
        # Votes now: [es, en, en, en, en] -> 80% en, above threshold
        assert orchestrator._last_detected_language == "en"


# ---------------------------------------------------------------------------
# TestBadTranscript
# ---------------------------------------------------------------------------

class TestBadTranscript:
    """Test the module-level _is_bad_transcript function."""

    def test_empty_string(self):
        assert _is_bad_transcript("") is True

    def test_none(self):
        assert _is_bad_transcript(None) is True

    def test_whitespace_only(self):
        assert _is_bad_transcript("   ") is True

    def test_too_few_letters(self):
        assert _is_bad_transcript("ab") is True

    def test_exactly_min_letters(self):
        assert _is_bad_transcript("abc") is False

    def test_punctuation_only(self):
        assert _is_bad_transcript("...!!!???") is True

    def test_numbers_only(self):
        assert _is_bad_transcript("12345") is True

    def test_low_letter_ratio(self):
        """Mostly digits with few letters should be rejected."""
        assert _is_bad_transcript("a12345678") is True

    def test_normal_text(self):
        assert _is_bad_transcript("Hola mundo") is False

    def test_short_valid_text(self):
        assert _is_bad_transcript("Yes") is False

    def test_mixed_alphanumeric_ok(self):
        """Text with a healthy letter ratio should pass."""
        assert _is_bad_transcript("Hello 123") is False

    @pytest.mark.parametrize("text,expected", [
        ("...", True),
        ("abc", False),
        ("  ab  ", True),
        ("Hola, como estas?", False),
        ("1234abc5678", True),  # 3 letters / 10 alnum = 0.3 < 0.5
        ("abc123", False),  # 3 letters / 6 alnum = 0.5, exactly at threshold
    ])
    def test_parametrized_cases(self, text, expected):
        assert _is_bad_transcript(text) is expected


# ---------------------------------------------------------------------------
# TestStatePersistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """Test _save_shutdown_state and _restore_shutdown_state."""

    @pytest.mark.asyncio
    async def test_save_state_creates_file(self, orchestrator, app_config):
        orchestrator.topic_summary = "Test topic"
        orchestrator.recent_exchanges.append({"source": "Hola", "translated": "Hello"})

        await orchestrator._save_shutdown_state()

        state_path = app_config.data_dir / "last_session.json"
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data["topic_summary"] == "Test topic"
        assert len(data["recent_exchanges"]) == 1
        assert data["direction"] == "es_to_en"
        assert data["mode"] == "conversation"

    @pytest.mark.asyncio
    async def test_save_state_includes_speakers(self, orchestrator, app_config):
        orchestrator.speaker_tracker.record_utterance("SPEAKER_00")
        orchestrator.speaker_tracker.rename("SPEAKER_00", "Maria")

        await orchestrator._save_shutdown_state()

        data = json.loads((app_config.data_dir / "last_session.json").read_text())
        speakers = data["speakers"]
        assert "SPEAKER_00" in speakers
        assert speakers["SPEAKER_00"]["custom_name"] == "Maria"
        assert speakers["SPEAKER_00"]["utterance_count"] == 1

    @pytest.mark.asyncio
    async def test_save_state_includes_translator_metrics(self, orchestrator, app_config):
        orchestrator.translator.metrics = {"requests": 5, "successes": 4, "failures": 1}

        await orchestrator._save_shutdown_state()

        data = json.loads((app_config.data_dir / "last_session.json").read_text())
        assert data["translator_metrics"]["requests"] == 5

    @pytest.mark.asyncio
    async def test_restore_state_loads_topic(self, orchestrator, app_config):
        state = {
            "shutdown_time": "2026-02-24T10:00:00",
            "topic_summary": "Restored topic",
            "recent_exchanges": [],
            "speakers": {},
        }
        (app_config.data_dir / "last_session.json").write_text(json.dumps(state))

        await orchestrator._restore_shutdown_state()

        assert orchestrator.topic_summary == "Restored topic"

    @pytest.mark.asyncio
    async def test_restore_state_loads_exchanges(self, orchestrator, app_config):
        state = {
            "shutdown_time": "2026-02-24T10:00:00",
            "topic_summary": "",
            "recent_exchanges": [
                {"source": "Hola", "translated": "Hello"},
                {"source": "Mundo", "translated": "World"},
            ],
            "speakers": {},
        }
        (app_config.data_dir / "last_session.json").write_text(json.dumps(state))

        await orchestrator._restore_shutdown_state()

        assert len(orchestrator.recent_exchanges) == 2
        assert orchestrator.recent_exchanges[0]["source"] == "Hola"

    @pytest.mark.asyncio
    async def test_restore_state_loads_speakers(self, orchestrator, app_config):
        state = {
            "shutdown_time": "2026-02-24T10:00:00",
            "topic_summary": "",
            "recent_exchanges": [],
            "speakers": {
                "SPK_0": {
                    "auto_label": "Speaker A",
                    "custom_name": "Carlos",
                    "role_hint": "teacher",
                    "utterance_count": 7,
                },
            },
        }
        (app_config.data_dir / "last_session.json").write_text(json.dumps(state))

        await orchestrator._restore_shutdown_state()

        assert "SPK_0" in orchestrator.speaker_tracker.speakers
        sp = orchestrator.speaker_tracker.speakers["SPK_0"]
        assert sp.custom_name == "Carlos"
        assert sp.role_hint == "teacher"
        assert sp.utterance_count == 7
        assert sp.label == "Speaker A"

    @pytest.mark.asyncio
    async def test_restore_state_no_file_noop(self, orchestrator):
        """Should do nothing if no state file exists."""
        await orchestrator._restore_shutdown_state()
        assert orchestrator.topic_summary == ""

    @pytest.mark.asyncio
    async def test_restore_state_corrupt_file_no_crash(self, orchestrator, app_config):
        (app_config.data_dir / "last_session.json").write_text("NOT VALID JSON {{{")
        # Should log warning but not raise
        await orchestrator._restore_shutdown_state()
        assert orchestrator.topic_summary == ""

    @pytest.mark.asyncio
    async def test_save_and_restore_roundtrip(self, orchestrator, app_config):
        """Save state then restore it and verify identical content."""
        orchestrator.topic_summary = "Roundtrip topic"
        orchestrator.recent_exchanges.append({"source": "test", "translated": "prueba"})
        orchestrator.speaker_tracker.record_utterance("SPK")

        await orchestrator._save_shutdown_state()

        # Create fresh orchestrator and restore
        orch2 = PipelineOrchestrator(app_config)
        orch2.translator = MagicMock()
        orch2.translator.metrics = {}
        await orch2._restore_shutdown_state()

        assert orch2.topic_summary == "Roundtrip topic"
        assert len(orch2.recent_exchanges) == 1
        assert "SPK" in orch2.speaker_tracker.speakers


# ---------------------------------------------------------------------------
# TestShutdown
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestShutdown:
    """Test graceful shutdown behavior (30s queue drain timeout)."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_not_ready(self, orchestrator, app_config):
        # Start a no-op worker
        orchestrator._worker_task = asyncio.create_task(asyncio.sleep(10))

        await orchestrator.shutdown()

        assert orchestrator.ready is False

    @pytest.mark.asyncio
    async def test_shutdown_cancels_worker(self, orchestrator, app_config):
        task = asyncio.create_task(asyncio.sleep(100))
        orchestrator._worker_task = task

        await orchestrator.shutdown()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_saves_state(self, orchestrator, app_config):
        orchestrator._worker_task = asyncio.create_task(asyncio.sleep(10))
        orchestrator.topic_summary = "Shutdown topic"

        await orchestrator.shutdown()

        state_path = app_config.data_dir / "last_session.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["topic_summary"] == "Shutdown topic"

    @pytest.mark.asyncio
    async def test_shutdown_closes_translator(self, orchestrator, app_config):
        orchestrator._worker_task = asyncio.create_task(asyncio.sleep(10))

        await orchestrator.shutdown()

        orchestrator.translator.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_queued_futures(self, orchestrator, app_config):
        orchestrator._worker_task = asyncio.create_task(asyncio.sleep(10))

        # Put an item in the queue with an unresolved future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await orchestrator._queue.put(("wav", "/fake.wav", future))

        await orchestrator.shutdown()

        assert future.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_inflight_translations(self, orchestrator, app_config):
        orchestrator._worker_task = asyncio.create_task(asyncio.sleep(10))

        # Create a "fast" inflight task
        completed = asyncio.Event()

        async def inflight():
            completed.set()

        task = asyncio.create_task(inflight())
        orchestrator._inflight_translations.add(task)
        task.add_done_callback(orchestrator._inflight_translations.discard)

        await orchestrator.shutdown()

        assert completed.is_set()


# ---------------------------------------------------------------------------
# TestPartialAudio
# ---------------------------------------------------------------------------

class TestPartialAudio:
    """Test process_partial_audio (streaming partials)."""

    @pytest.mark.asyncio
    async def test_partial_audio_noop_without_whisperx(self, orchestrator):
        """Should return immediately if no WhisperX model loaded."""
        orchestrator._whisperx_model = None
        # Should not raise
        await orchestrator.process_partial_audio(b"\x00" * 3200, 0.1)

    @pytest.mark.asyncio
    async def test_partial_audio_fires_partial_callback(self, orchestrator):
        on_partial = AsyncMock()
        orchestrator.set_callbacks(on_partial=on_partial)

        # Mock the WhisperX model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "Hola parcial"}],
            "language": "es",
        }
        orchestrator._whisperx_model = mock_model

        # Mock quick translate
        orchestrator._quick_translate = AsyncMock(return_value="Hello partial")

        # Generate some valid PCM audio (16-bit, 16kHz, 0.1s)
        import numpy as np
        pcm = (np.random.randn(1600) * 1000).astype(np.int16).tobytes()

        await orchestrator.process_partial_audio(pcm, 0.1)

        # Should have sent at least one partial callback (source text)
        assert on_partial.await_count >= 1

    @pytest.mark.asyncio
    async def test_partial_audio_skips_duplicate(self, orchestrator):
        """Should not send callback if partial text matches previous."""
        on_partial = AsyncMock()
        orchestrator.set_callbacks(on_partial=on_partial)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "Same text"}],
            "language": "es",
        }
        orchestrator._whisperx_model = mock_model
        orchestrator._last_partial_text = "Same text"

        import numpy as np
        pcm = (np.random.randn(1600) * 1000).astype(np.int16).tobytes()

        await orchestrator.process_partial_audio(pcm, 0.1)

        on_partial.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestProcessQueue
# ---------------------------------------------------------------------------

class TestProcessQueue:
    """Test the background queue worker."""

    @pytest.mark.asyncio
    async def test_queue_worker_processes_wav_item(self, orchestrator):
        orchestrator._process_audio_segment_from_wav = AsyncMock(return_value=None)

        worker = asyncio.create_task(orchestrator._process_queue())

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await orchestrator._queue.put(("wav", "/path.wav", future))

        result = await asyncio.wait_for(future, timeout=2.0)
        assert result is None
        orchestrator._process_audio_segment_from_wav.assert_awaited_once_with("/path.wav")

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        assert worker.done()

    @pytest.mark.asyncio
    async def test_queue_worker_sets_exception_on_failure(self, orchestrator):
        orchestrator._process_audio_segment_from_wav = AsyncMock(
            side_effect=RuntimeError("ASR crashed")
        )

        worker = asyncio.create_task(orchestrator._process_queue())

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await orchestrator._queue.put(("wav", "/path.wav", future))

        with pytest.raises(RuntimeError, match="ASR crashed"):
            await asyncio.wait_for(future, timeout=2.0)

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        assert worker.done()

    @pytest.mark.asyncio
    async def test_queue_worker_handles_raw_kind(self, orchestrator):
        orchestrator._process_audio_segment = AsyncMock(return_value=None)

        worker = asyncio.create_task(orchestrator._process_queue())

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await orchestrator._queue.put(("raw", b"\x00\x00", future))

        result = await asyncio.wait_for(future, timeout=2.0)
        assert result is None
        orchestrator._process_audio_segment.assert_awaited_once()

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        assert worker.done()


# ---------------------------------------------------------------------------
# TestSaveExchange
# ---------------------------------------------------------------------------

class TestSaveExchange:
    """Test _save_exchange database persistence."""

    @pytest.mark.asyncio
    async def test_save_exchange_returns_id(self, orchestrator):
        orchestrator.session_id = 5
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 101
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        speaker = orchestrator.speaker_tracker.record_utterance("SPK")
        exchange = Exchange(
            session_id=5,
            speaker=speaker,
            direction="es_to_en",
            raw_transcript="Hola",
            translation="Hello",
            confidence=0.9,
        )

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            result_id = await orchestrator._save_exchange(exchange)

        assert result_id == 101
        assert exchange.id == 101

    @pytest.mark.asyncio
    async def test_save_exchange_no_session_returns_none(self, orchestrator):
        orchestrator.session_id = None
        speaker = orchestrator.speaker_tracker.record_utterance("SPK")
        exchange = Exchange(
            session_id=0,
            speaker=speaker,
            direction="es_to_en",
            raw_transcript="Hola",
            translation="Hello",
        )

        result_id = await orchestrator._save_exchange(exchange)
        assert result_id is None

    @pytest.mark.asyncio
    async def test_save_exchange_with_correction_detail(self, orchestrator):
        orchestrator.session_id = 1
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 50
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        speaker = orchestrator.speaker_tracker.record_utterance("SPK")
        exchange = Exchange(
            session_id=1,
            speaker=speaker,
            direction="es_to_en",
            raw_transcript="yo es bueno",
            translation="I am good",
            is_correction=True,
            correction_detail=CorrectionDetail(
                wrong="yo es",
                right="yo soy",
                explanation="Verb conjugation",
            ),
        )

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            await orchestrator._save_exchange(exchange)

        # Verify correction_json was passed in the execute call
        call_args = mock_db.execute.call_args[0]
        params = call_args[1]
        correction_json = params[8]  # correction_json is the 9th parameter (index 8)
        assert correction_json is not None
        parsed = json.loads(correction_json)
        assert parsed["wrong"] == "yo es"

    @pytest.mark.asyncio
    async def test_save_exchange_db_failure_returns_none(self, orchestrator):
        orchestrator.session_id = 1
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("disk full"))

        speaker = orchestrator.speaker_tracker.record_utterance("SPK")
        exchange = Exchange(
            session_id=1,
            speaker=speaker,
            direction="es_to_en",
            raw_transcript="Hola",
            translation="Hello",
        )

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            result_id = await orchestrator._save_exchange(exchange)

        assert result_id is None


# ---------------------------------------------------------------------------
# TestQuickTranslate
# ---------------------------------------------------------------------------

class TestQuickTranslate:
    """Test _quick_translate for partial streaming translations."""

    @pytest.mark.asyncio
    async def test_quick_translate_empty_returns_empty(self, orchestrator):
        result = await orchestrator._quick_translate("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_quick_translate_es_to_en(self, orchestrator):
        orchestrator.translator._call_llm = AsyncMock(return_value="Hello world")
        orchestrator.direction = "es_to_en"

        result = await orchestrator._quick_translate("Hola mundo")

        assert result == "Hello world"
        call_args = orchestrator.translator._call_llm.call_args
        prompt = call_args[0][1]
        assert "Spanish" in prompt
        assert "English" in prompt

    @pytest.mark.asyncio
    async def test_quick_translate_en_to_es(self, orchestrator):
        orchestrator.translator._call_llm = AsyncMock(return_value="Hola mundo")
        orchestrator.direction = "en_to_es"

        result = await orchestrator._quick_translate("Hello world")

        call_args = orchestrator.translator._call_llm.call_args
        prompt = call_args[0][1]
        assert "English" in prompt
        assert "Spanish" in prompt

    @pytest.mark.asyncio
    async def test_quick_translate_error_returns_empty(self, orchestrator):
        orchestrator.translator._call_llm = AsyncMock(side_effect=RuntimeError("timeout"))

        result = await orchestrator._quick_translate("Hola")

        assert result == ""

    @pytest.mark.asyncio
    async def test_quick_translate_lmstudio_override(self, orchestrator):
        orchestrator.config.translator.provider = "lmstudio"
        orchestrator.config.translator.quick_model = "fast-model"
        orchestrator.translator._call_provider = AsyncMock(return_value="Quick result")

        result = await orchestrator._quick_translate("Hola")

        assert result == "Quick result"
        call_kwargs = orchestrator.translator._call_provider.call_args[1]
        assert call_kwargs["lmstudio_model"] == "fast-model"


# ---------------------------------------------------------------------------
# TestReloadIdiomPatterns
# ---------------------------------------------------------------------------

class TestReloadIdiomPatterns:
    """Test idiom pattern reloading."""

    @pytest.mark.asyncio
    async def test_reload_clears_and_reloads(self, orchestrator, app_config):
        # Add a pattern to the scanner
        orchestrator.idiom_scanner.patterns.append(MagicMock())
        assert orchestrator.idiom_scanner.count >= 1

        # Mock DB loading
        mock_db = AsyncMock()
        mock_db.execute_fetchall = AsyncMock(return_value=[])

        with patch("server.pipeline.orchestrator.get_db", AsyncMock(return_value=mock_db)):
            await orchestrator.reload_idiom_patterns()

        # Patterns should be cleared (no JSON files in empty idioms dir, no DB patterns)
        assert orchestrator.idiom_scanner.count == 0
