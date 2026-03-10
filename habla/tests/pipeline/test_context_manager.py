"""Unit tests for ContextManager."""

import pytest
from collections import deque
from unittest.mock import AsyncMock, MagicMock

from server.config import AppConfig, RecordingConfig, SessionConfig, TranslatorConfig
from server.models.schemas import FlaggedPhrase, TranslationResult
from server.pipeline.context_manager import ContextManager
from server.services.idiom_scanner import IdiomMatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_config(tmp_path):
    """AppConfig with test defaults."""
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


@pytest.fixture
def ctx(app_config):
    """ContextManager with translator mocked out."""
    cm = ContextManager(app_config)
    cm.translator = MagicMock()
    cm.translator.translate = AsyncMock()
    cm.translator.update_topic_summary = AsyncMock(return_value="new topic")
    return cm


def _make_translation_result(**overrides) -> TranslationResult:
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


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Verify ContextManager default state."""

    def test_direction_from_config(self, app_config):
        cm = ContextManager(app_config)
        assert cm.direction == "es_to_en"

    def test_mode_from_config(self, app_config):
        app_config.session.mode = "classroom"
        cm = ContextManager(app_config)
        assert cm.mode == "classroom"

    def test_topic_summary_empty(self, app_config):
        cm = ContextManager(app_config)
        assert cm.topic_summary == ""

    def test_recent_exchanges_empty(self, app_config):
        cm = ContextManager(app_config)
        assert len(cm.recent_exchanges) == 0

    def test_recent_exchanges_maxlen_10(self, app_config):
        cm = ContextManager(app_config)
        assert cm.recent_exchanges.maxlen == 10

    def test_session_id_none(self, app_config):
        cm = ContextManager(app_config)
        assert cm.session_id is None

    def test_creates_translator(self, app_config):
        cm = ContextManager(app_config)
        # Translator is created (not None) even without calling startup
        assert cm.translator is not None

    def test_creates_speaker_tracker(self, app_config):
        cm = ContextManager(app_config)
        assert cm.speaker_tracker is not None

    def test_creates_idiom_scanner(self, app_config):
        cm = ContextManager(app_config)
        assert cm.idiom_scanner is not None


# ---------------------------------------------------------------------------
# TestMergeIdioms
# ---------------------------------------------------------------------------

class TestMergeIdioms:
    """Test _merge_idioms dedup logic with pattern DB priority."""

    def test_empty_inputs_return_empty(self, ctx):
        result = ctx._merge_idioms([], [])
        assert result == []

    def test_pattern_only_returns_flagged_phrase(self, ctx):
        patterns = [
            IdiomMatch(
                canonical="echar de menos",
                literal="to throw from less",
                meaning="to miss someone",
                region="spain",
                match_start=0,
                match_end=14,
            )
        ]
        result = ctx._merge_idioms(patterns, [])
        assert len(result) == 1
        assert result[0].phrase == "echar de menos"
        assert result[0].source == "pattern_db"
        assert result[0].save_worthy is True

    def test_pattern_preserves_span(self, ctx):
        patterns = [
            IdiomMatch(canonical="test", literal="lit", meaning="m", match_start=5, match_end=9)
        ]
        result = ctx._merge_idioms(patterns, [])
        assert result[0].span_start == 5
        assert result[0].span_end == 9

    def test_pattern_preserves_region(self, ctx):
        patterns = [
            IdiomMatch(canonical="test", literal="lit", meaning="m", region="spain", match_start=0, match_end=4)
        ]
        result = ctx._merge_idioms(patterns, [])
        assert result[0].region == "spain"

    def test_llm_only_returns_flagged_phrase(self, ctx):
        llm_phrases = [
            FlaggedPhrase(phrase="dar en el clavo", meaning="to hit the nail", source="llm")
        ]
        result = ctx._merge_idioms([], llm_phrases)
        assert len(result) == 1
        assert result[0].source == "llm"

    def test_dedup_pattern_takes_priority_over_llm(self, ctx):
        """When both detect the same phrase, pattern DB wins."""
        patterns = [
            IdiomMatch(
                canonical="Tomar el pelo",
                literal="to take the hair",
                meaning="to pull someone's leg (DB)",
                match_start=0, match_end=14,
            )
        ]
        llm_phrases = [
            FlaggedPhrase(phrase="tomar el pelo", meaning="to pull someone's leg (LLM)", source="llm")
        ]
        result = ctx._merge_idioms(patterns, llm_phrases)
        assert len(result) == 1, "Same phrase from both sources should be deduped"
        assert result[0].source == "pattern_db", "Pattern DB must take priority"
        assert "DB" in result[0].meaning

    def test_dedup_is_case_insensitive(self, ctx):
        patterns = [
            IdiomMatch(canonical="ECHAR DE MENOS", literal="lit", meaning="m", match_start=0, match_end=13)
        ]
        llm_phrases = [
            FlaggedPhrase(phrase="echar de menos", meaning="m2", source="llm")
        ]
        result = ctx._merge_idioms(patterns, llm_phrases)
        assert len(result) == 1, "Case-insensitive dedup must remove the LLM duplicate"

    def test_different_idioms_kept(self, ctx):
        patterns = [
            IdiomMatch(canonical="idiom A", literal="lit A", meaning="m A", match_start=0, match_end=7)
        ]
        llm_phrases = [
            FlaggedPhrase(phrase="idiom B", meaning="m B", source="llm")
        ]
        result = ctx._merge_idioms(patterns, llm_phrases)
        assert len(result) == 2, "Different idioms from both sources should all be included"

    def test_result_order_pattern_first(self, ctx):
        """Pattern DB entries always come before LLM entries in the merged list."""
        patterns = [
            IdiomMatch(canonical="pattern idiom", literal="lit", meaning="m", match_start=0, match_end=13)
        ]
        llm_phrases = [
            FlaggedPhrase(phrase="llm idiom", meaning="m2", source="llm")
        ]
        result = ctx._merge_idioms(patterns, llm_phrases)
        assert result[0].source == "pattern_db"
        assert result[1].source == "llm"


# ---------------------------------------------------------------------------
# TestUpdateTopic
# ---------------------------------------------------------------------------

class TestUpdateTopic:
    """Test _update_topic async topic summary update."""

    @pytest.mark.asyncio
    async def test_updates_topic_summary_on_success(self, ctx):
        # Arrange
        ctx.translator.update_topic_summary = AsyncMock(return_value="new summary")
        ctx.topic_summary = "old summary"
        result = _make_translation_result()

        # Act
        await ctx._update_topic(result, "Teacher")

        # Assert
        assert ctx.topic_summary == "new summary"

    @pytest.mark.asyncio
    async def test_does_not_update_when_update_returns_none(self, ctx):
        # Arrange
        ctx.translator.update_topic_summary = AsyncMock(return_value=None)
        ctx.topic_summary = "old summary"
        result = _make_translation_result()

        # Act
        await ctx._update_topic(result, "Teacher")

        # Assert
        assert ctx.topic_summary == "old summary", "None return from LLM must not overwrite existing summary"

    @pytest.mark.asyncio
    async def test_does_not_raise_on_llm_failure(self, ctx):
        # Arrange
        ctx.translator.update_topic_summary = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        ctx.topic_summary = "old summary"
        result = _make_translation_result()

        # Act — should not raise
        await ctx._update_topic(result, "Teacher")

        # Assert
        assert ctx.topic_summary == "old summary", "LLM failure must not overwrite summary"

    @pytest.mark.asyncio
    async def test_passes_correct_arguments_to_translator(self, ctx):
        # Arrange
        ctx.topic_summary = "current topic"
        result = _make_translation_result(corrected="Hola", translated="Hello")

        # Act
        await ctx._update_topic(result, "Speaker A")

        # Assert
        ctx.translator.update_topic_summary.assert_awaited_once_with(
            previous_summary="current topic",
            latest_source="Hola",
            latest_translation="Hello",
            speaker_label="Speaker A",
        )


# ---------------------------------------------------------------------------
# TestSaveAudioClip
# ---------------------------------------------------------------------------

class TestSaveAudioClip:
    """Test _save_audio_clip file persistence."""

    def test_returns_none_when_clips_disabled(self, ctx):
        # Arrange
        ctx.config.session.save_audio_clips = False
        ctx.session_id = 1

        # Act
        result = ctx._save_audio_clip("/fake/audio.wav")

        # Assert
        assert result is None

    def test_returns_none_without_session_id(self, ctx):
        # Arrange
        ctx.config.session.save_audio_clips = True
        ctx.session_id = None

        # Act
        result = ctx._save_audio_clip("/fake/audio.wav")

        # Assert
        assert result is None

    def test_returns_path_on_success(self, ctx, tmp_path):
        # Arrange
        ctx.config.session.save_audio_clips = True
        ctx.session_id = 42
        wav_path = str(tmp_path / "audio.wav")
        (tmp_path / "audio.wav").write_bytes(b"\x00" * 100)

        import unittest.mock
        with unittest.mock.patch("server.pipeline.context_manager.Path") as mock_path_class:
            clips_dir = MagicMock()
            clip_file = MagicMock()
            mock_path_class.return_value.__truediv__.return_value = clips_dir
            clips_dir.__truediv__.return_value = clip_file
            clip_file.__str__.return_value = "data/audio/clips/42/clip_test.wav"

            with unittest.mock.patch("shutil.copy2"):
                result = ctx._save_audio_clip(wav_path)

        # Just verify it returns a string (path) when enabled+session exists
        # The exact path is system-dependent
        assert result is not None

    def test_returns_none_on_io_failure(self, ctx):
        # Arrange
        ctx.config.session.save_audio_clips = True
        ctx.session_id = 1

        import unittest.mock
        with unittest.mock.patch("server.pipeline.context_manager.Path") as mock_path:
            mock_path.return_value.__truediv__.return_value.mkdir.side_effect = OSError("disk full")
            result = ctx._save_audio_clip("/fake/audio.wav")

        # Assert
        assert result is None, "IO failure in clip saving must return None, not raise"


# ---------------------------------------------------------------------------
# TestContextManagerDirectionMode
# ---------------------------------------------------------------------------

class TestContextManagerDirectionMode:
    """Test direction/mode state management."""

    def test_direction_initialized_from_config(self, app_config):
        cm = ContextManager(app_config)
        assert cm.direction == "es_to_en"

    def test_direction_settable(self, ctx):
        ctx.direction = "en_to_es"
        assert ctx.direction == "en_to_es"

    def test_mode_settable(self, ctx):
        ctx.mode = "classroom"
        assert ctx.mode == "classroom"

    def test_session_id_settable(self, ctx):
        ctx.session_id = 99
        assert ctx.session_id == 99

    def test_topic_summary_settable(self, ctx):
        ctx.topic_summary = "discussing verbs"
        assert ctx.topic_summary == "discussing verbs"
