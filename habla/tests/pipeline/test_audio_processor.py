"""Unit tests for AudioProcessor."""

import asyncio
import threading
import pytest
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

from server.config import AppConfig, RecordingConfig, SessionConfig, TranslatorConfig
from server.pipeline.audio_processor import AudioProcessor, _is_bad_transcript


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_config(tmp_path):
    """AppConfig with recording disabled."""
    return AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "test.db",
        translator=TranslatorConfig(provider="ollama"),
        recording=RecordingConfig(enabled=False),
        session=SessionConfig(direction="es_to_en", mode="conversation"),
    )


@pytest.fixture
def audio_processor(app_config):
    """AudioProcessor with es_to_en direction."""
    return AudioProcessor(app_config, source_language_getter=lambda: "es")


@pytest.fixture
def mock_whisperx_model():
    """Mock WhisperX model that returns a clean Spanish transcript."""
    model = MagicMock()
    model.transcribe.return_value = {
        "segments": [{"text": "Hola mundo"}],
        "language": "es",
    }
    return model


# ---------------------------------------------------------------------------
# TestRunAsrAndDiarize
# ---------------------------------------------------------------------------

class TestRunAsrAndDiarize:
    """Test _run_asr_and_diarize: transcription + diarization in thread."""

    @pytest.fixture(autouse=True)
    def _fake_audio_and_detection(self, monkeypatch):
        """No real WAV on disk: load_audio returns silence, and per-segment
        detection reports the language the mock transcript is in."""
        import numpy as np
        import whisperx
        monkeypatch.setattr(whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
        monkeypatch.setattr(
            AudioProcessor, "_detect_segment_language",
            lambda self, audio: self._whisperx_model.transcribe.return_value.get("language", "es"),
        )

    def test_success_returns_transcript_and_segments(self, audio_processor, mock_whisperx_model):
        # Arrange
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None  # no diarization

        # Act
        transcript, segments, detected_lang, reason = audio_processor._run_asr_and_diarize("/fake/audio.wav")

        # Assert
        assert transcript == "Hola mundo"
        assert segments == []
        assert detected_lang == "es"
        assert reason == "ok"

    def test_empty_result_returns_no_segments_reason(self, audio_processor, mock_whisperx_model):
        # Arrange — WhisperX VAD found no speech
        mock_whisperx_model.transcribe.return_value = {"segments": [], "language": "es"}
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None

        # Act
        transcript, segments, detected_lang, reason = audio_processor._run_asr_and_diarize("/fake/audio.wav")

        # Assert — distinct reason from bad_transcript: audio was too quiet / non-speech for WhisperX VAD
        assert transcript == ""
        assert segments == []
        assert detected_lang == "es"
        assert reason == "no_segments", "Zero segments from WhisperX should report 'no_segments'"

    def test_bad_transcript_returns_bad_transcript_reason(self, audio_processor, mock_whisperx_model):
        # Arrange — segments present but content is garbage
        mock_whisperx_model.transcribe.return_value = {
            "segments": [{"text": "12345"}],  # noise-like garbage
            "language": "es",
        }
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None

        # Act
        transcript, segments, detected_lang, reason = audio_processor._run_asr_and_diarize("/fake/audio.wav")

        # Assert — distinct reason from no_segments: ASR ran but output was garbage
        assert transcript == ""
        assert reason == "bad_transcript", "Numbers-only output should report 'bad_transcript'"

    def test_asr_lock_is_used_as_context_manager(self, audio_processor, mock_whisperx_model):
        # Arrange — replace threading.Lock with a MagicMock that supports `with`
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None
        mock_lock = MagicMock()
        audio_processor._asr_lock = mock_lock

        # Act
        audio_processor._run_asr_and_diarize("/fake/audio.wav")

        # Assert — `with self._asr_lock:` calls __enter__ and __exit__
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    def test_updates_detected_language_from_asr_result(self, audio_processor, mock_whisperx_model):
        # Arrange
        mock_whisperx_model.transcribe.return_value = {
            "segments": [{"text": "Hello world"}],
            "language": "en",
        }
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None

        # Act
        audio_processor._run_asr_and_diarize("/fake/audio.wav")

        # Assert
        assert audio_processor._last_detected_language == "en", \
            "ASR-detected language should be recorded via voting"

    def test_language_switch_reaches_voter(self, audio_processor, mock_whisperx_model):
        """Regression: WhisperX reuses its first detected language, so detection
        must run per segment or auto-direction never switches."""
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None
        detections = iter(["es", "en", "en", "en", "en"])
        with patch.object(AudioProcessor, "_detect_segment_language",
                          lambda self, audio: next(detections)):
            for _ in range(5):
                audio_processor._run_asr_and_diarize("/fake/audio.wav")
        assert audio_processor._last_detected_language == "en"

    def test_final_transcribed_in_its_own_detected_language(self, audio_processor, mock_whisperx_model):
        """An English turn in a Spanish conversation is transcribed as English
        even before the vote switches."""
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None
        detections = iter(["es", "es", "en"])
        with patch.object(AudioProcessor, "_detect_segment_language",
                          lambda self, audio: next(detections)):
            for _ in range(3):
                _, _, det, _ = audio_processor._run_asr_and_diarize("/fake/audio.wav")
        assert det == "en"
        assert mock_whisperx_model.transcribe.call_args.kwargs["language"] == "en"
        assert audio_processor._last_detected_language == "es"  # vote unchanged

    def test_fixed_language_skips_detection(self, audio_processor, mock_whisperx_model):
        audio_processor.config.asr.auto_language = False
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._diarize_pipeline = None
        with patch.object(AudioProcessor, "_detect_segment_language",
                          side_effect=AssertionError("should not detect")):
            audio_processor._run_asr_and_diarize("/fake/audio.wav")
        assert mock_whisperx_model.transcribe.call_args.kwargs["language"] == "es"

    def test_diarize_failure_returns_transcript_with_empty_segments(
        self, audio_processor, mock_whisperx_model
    ):
        # Arrange
        audio_processor._whisperx_model = mock_whisperx_model
        mock_diarize = MagicMock()
        mock_diarize.side_effect = RuntimeError("diarize failed")
        audio_processor._diarize_pipeline = mock_diarize

        # Act
        transcript, segments, detected_lang, reason = audio_processor._run_asr_and_diarize("/fake/audio.wav")

        # Assert — diarization error should degrade gracefully, not crash
        assert transcript == "Hola mundo", "Transcript should succeed despite diarization failure"
        assert segments == [], "Failed diarization returns empty segments"
        assert reason == "ok"


# ---------------------------------------------------------------------------
# TestRunQuickAsr
# ---------------------------------------------------------------------------

class TestRunQuickAsr:
    """Test _run_quick_asr: fast partial transcription."""

    def test_success_returns_transcript(self, audio_processor, mock_whisperx_model):
        # Arrange
        audio_processor._whisperx_model = mock_whisperx_model

        # Act
        text, detected = audio_processor._run_quick_asr("/fake/partial.wav")

        # Assert
        assert text == "Hola mundo"
        assert detected == "es"
        mock_whisperx_model.transcribe.assert_called_once()

    def test_uses_smaller_batch_size_than_full_asr(self, audio_processor, mock_whisperx_model):
        # Arrange — full ASR uses batch_size=8; partial should use batch_size=4
        audio_processor._whisperx_model = mock_whisperx_model

        # Act
        audio_processor._run_quick_asr("/fake/partial.wav")

        # Assert
        call_kwargs = mock_whisperx_model.transcribe.call_args[1]
        assert call_kwargs["batch_size"] == 4, "Quick ASR must use smaller batch for speed"

    def test_empty_segments_returns_empty_string(self, audio_processor, mock_whisperx_model):
        # Arrange
        mock_whisperx_model.transcribe.return_value = {"segments": [], "language": "es"}
        audio_processor._whisperx_model = mock_whisperx_model

        # Act
        text, detected = audio_processor._run_quick_asr("/fake/partial.wav")

        # Assert
        assert text == "", "Empty segments should return empty string"

    def test_exception_returns_empty_string(self, audio_processor, mock_whisperx_model):
        # Arrange
        mock_whisperx_model.transcribe.side_effect = RuntimeError("GPU out of memory")
        audio_processor._whisperx_model = mock_whisperx_model

        # Act
        text, detected = audio_processor._run_quick_asr("/fake/partial.wav")

        # Assert
        assert text == "", "Exception in quick ASR should return empty string, not raise"
        assert detected is None

    def test_uses_detected_language_when_auto_language_enabled(self, audio_processor, mock_whisperx_model):
        # Arrange
        audio_processor._last_detected_language = "es"
        audio_processor.config.asr.auto_language = True
        audio_processor._whisperx_model = mock_whisperx_model

        # Act
        audio_processor._run_quick_asr("/fake/partial.wav")

        # Assert
        call_kwargs = mock_whisperx_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "es", "Should use detected language when auto_language=True"

    def test_uses_source_language_when_auto_language_disabled(self, audio_processor, mock_whisperx_model):
        # Arrange
        audio_processor.config.asr.auto_language = False
        audio_processor._whisperx_model = mock_whisperx_model

        # Act
        audio_processor._run_quick_asr("/fake/partial.wav")

        # Assert
        call_kwargs = mock_whisperx_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "es", "Should use fixed source language when auto_language=False"


# ---------------------------------------------------------------------------
# TestUpdateDetectedLanguage
# ---------------------------------------------------------------------------

class TestUpdateDetectedLanguage:
    """Test _update_detected_language voting and supermajority logic."""

    def test_first_detection_accepted_immediately(self, audio_processor):
        # Arrange
        audio_processor._last_detected_language = None
        audio_processor._language_votes.clear()

        # Act
        audio_processor._update_detected_language("es")

        # Assert
        assert audio_processor._last_detected_language == "es"

    def test_same_language_stays(self, audio_processor):
        # Arrange
        audio_processor._last_detected_language = "es"
        audio_processor._language_votes.clear()

        # Act
        for _ in range(5):
            audio_processor._update_detected_language("es")

        # Assert
        assert audio_processor._last_detected_language == "es"

    def test_single_disagreement_does_not_switch(self, audio_processor):
        # Arrange
        audio_processor._last_detected_language = "es"
        audio_processor._language_votes.clear()
        audio_processor._language_votes.extend(["es", "es", "es"])

        # Act — one dissenting vote (1/4 = 25%, below 70% threshold)
        audio_processor._update_detected_language("en")

        # Assert
        assert audio_processor._last_detected_language == "es", \
            "Single dissenting detection should not override established language"

    def test_supermajority_triggers_switch(self, audio_processor):
        # Arrange
        audio_processor._last_detected_language = "es"
        audio_processor._language_votes.clear()

        # Act — push 4 consecutive "en" detections (4/4 = 100%)
        for _ in range(4):
            audio_processor._update_detected_language("en")

        # Assert
        assert audio_processor._last_detected_language == "en", \
            "100% majority of recent detections should switch language"

    def test_votes_deque_capped_at_maxlen_5(self, audio_processor):
        # Arrange
        audio_processor._last_detected_language = "es"
        audio_processor._language_votes.clear()

        # Act
        for _ in range(10):
            audio_processor._update_detected_language("es")

        # Assert
        assert len(audio_processor._language_votes) == 5, "Deque must cap at maxlen=5"

    def test_gradual_shift_switches_at_threshold(self, audio_processor):
        """Simulate realistic gradual language shift."""
        # Arrange
        audio_processor._last_detected_language = "es"
        audio_processor._language_votes.clear()

        # Act — build up English detections past 70%
        audio_processor._update_detected_language("es")   # [es]
        audio_processor._update_detected_language("es")   # [es, es]
        audio_processor._update_detected_language("en")   # [es, es, en] — 33%, no switch
        assert audio_processor._last_detected_language == "es"

        audio_processor._update_detected_language("en")   # [es, es, en, en] — 50%, no switch
        assert audio_processor._last_detected_language == "es"

        audio_processor._update_detected_language("en")   # [es, es, en, en, en] — 60%, no switch
        assert audio_processor._last_detected_language == "es"

        audio_processor._update_detected_language("en")   # [es, en, en, en, en] — 80%, switch!
        assert audio_processor._last_detected_language == "en"


# ---------------------------------------------------------------------------
# TestResetPartialState
# ---------------------------------------------------------------------------

class TestResetPartialState:
    """Test reset_partial_state clears the streaming buffer."""

    def test_clears_last_partial_text(self, audio_processor):
        # Arrange
        audio_processor._last_partial_text = "some ongoing speech"

        # Act
        audio_processor.reset_partial_state()

        # Assert
        assert audio_processor._last_partial_text == "", "Partial buffer must be cleared"

    def test_idempotent_when_already_empty(self, audio_processor):
        # Arrange
        audio_processor._last_partial_text = ""

        # Act
        audio_processor.reset_partial_state()

        # Assert
        assert audio_processor._last_partial_text == ""


# ---------------------------------------------------------------------------
# TestDecodeToWav
# ---------------------------------------------------------------------------

class TestDecodeToWav:
    """Test _decode_to_wav: ffmpeg Opus/WebM -> WAV conversion."""

    @pytest.mark.asyncio
    async def test_success_returns_wav_path(self, audio_processor, tmp_path):
        # Arrange
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("pathlib.Path.unlink"):  # don't try to delete non-existent temp file
                result = await audio_processor._decode_to_wav(b"\x00" * 100)

        # Assert
        assert result is not None
        assert result.endswith(".wav"), "Output must be a WAV file"

    @pytest.mark.asyncio
    async def test_ffmpeg_failure_returns_none(self, audio_processor):
        # Arrange
        mock_proc = AsyncMock()
        mock_proc.returncode = 1  # ffmpeg error
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("pathlib.Path.unlink"):
                result = await audio_processor._decode_to_wav(b"\x00" * 100)

        # Assert
        assert result is None, "Failed ffmpeg decode must return None, not raise"

    @pytest.mark.asyncio
    async def test_subprocess_exception_returns_none(self, audio_processor):
        # Arrange
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("ffmpeg not found")):
            result = await audio_processor._decode_to_wav(b"\x00" * 100)

        # Assert
        assert result is None, "Subprocess exception must return None, not propagate"


# ---------------------------------------------------------------------------
# TestNormalizeWav
# ---------------------------------------------------------------------------

class TestNormalizeWav:
    """Test _normalize_wav: bandpass + dynaudnorm via ffmpeg."""

    @pytest.mark.asyncio
    async def test_success_returns_wav_path(self, audio_processor, tmp_path):
        # Arrange
        wav_path = str(tmp_path / "input.wav")
        Path(wav_path).write_bytes(b"\x00" * 100)  # create the input file

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()

        # Simulate: normalized file is "created" by ffmpeg
        norm_path = wav_path + ".norm.wav"

        def create_norm_file():
            Path(norm_path).write_bytes(b"\x00" * 100)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=lambda *a, **kw: create_norm_file()):
            result = await audio_processor._normalize_wav(wav_path)

        # Assert
        assert result == wav_path, "Normalized WAV should replace the original path"

    @pytest.mark.asyncio
    async def test_ffmpeg_failure_falls_back_to_original(self, audio_processor, tmp_path):
        # Arrange
        wav_path = str(tmp_path / "input.wav")
        Path(wav_path).write_bytes(b"\x00" * 100)

        mock_proc = AsyncMock()
        mock_proc.returncode = 1  # normalization failed
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await audio_processor._normalize_wav(wav_path)

        # Assert
        assert result == wav_path, "On ffmpeg failure, original WAV path must be returned"
        assert Path(wav_path).exists(), "Original WAV must NOT be deleted when normalization fails"


# ---------------------------------------------------------------------------
# TestAudioProcessorSourceLanguage
# ---------------------------------------------------------------------------

class TestAudioProcessorSourceLanguage:
    """Test _source_language property via the getter callable."""

    def test_getter_called_at_access_time(self, app_config):
        # Arrange — getter returns different values over time
        calls = []

        def dynamic_getter():
            calls.append(len(calls))
            return "es" if len(calls) <= 2 else "en"

        processor = AudioProcessor(app_config, source_language_getter=dynamic_getter)

        # Act
        lang1 = processor._source_language
        lang2 = processor._source_language
        lang3 = processor._source_language

        # Assert — getter is called fresh each time
        assert lang1 == "es"
        assert lang2 == "es"
        assert lang3 == "en"
        assert len(calls) == 3, "Getter should be called on every property access"


class TestDetectSegmentLanguage:
    """_detect_segment_language picks the best supported language."""

    def _model(self, probs):
        model = MagicMock()
        model.model.feat_kwargs = {"feature_size": 80}
        model.model.encode.return_value = "enc"
        model.model.model.detect_language.return_value = [probs]
        return model

    def test_ignores_unsupported_languages(self, audio_processor):
        import numpy as np
        audio_processor._whisperx_model = self._model(
            [("<|fr|>", 0.6), ("<|en|>", 0.3), ("<|es|>", 0.1)]
        )
        with patch("whisperx.audio.log_mel_spectrogram", return_value="mel"):
            assert audio_processor._detect_segment_language(np.zeros(16000, dtype=np.float32)) == "en"

    def test_no_supported_language_keeps_current(self, audio_processor):
        import numpy as np
        audio_processor._whisperx_model = self._model([("<|fr|>", 0.9)])
        audio_processor._last_detected_language = "en"
        with patch("whisperx.audio.log_mel_spectrogram", return_value="mel"):
            assert audio_processor._detect_segment_language(np.zeros(16000, dtype=np.float32)) == "en"


class TestQuickAsrDoesNotVote:
    def test_partials_do_not_vote(self, audio_processor, mock_whisperx_model):
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._last_detected_language = "en"
        mock_whisperx_model.transcribe.return_value = {"segments": [{"text": "Hola mundo"}], "language": "es"}
        for _ in range(5):
            audio_processor._run_quick_asr("/fake/partial.wav")
        assert len(audio_processor._language_votes) == 0
        assert audio_processor._last_detected_language == "en"


class TestQuickAsrLanguageBeforeFirstFinal:
    def test_partial_before_any_vote_uses_source_language(self, audio_processor, mock_whisperx_model):
        audio_processor._whisperx_model = mock_whisperx_model
        assert audio_processor._last_detected_language is None
        audio_processor._run_quick_asr("/fake/partial.wav")
        assert mock_whisperx_model.transcribe.call_args.kwargs["language"] == "es"

    def test_partial_uses_voted_language_once_known(self, audio_processor, mock_whisperx_model):
        audio_processor._whisperx_model = mock_whisperx_model
        audio_processor._last_detected_language = "en"
        audio_processor._run_quick_asr("/fake/partial.wav")
        assert mock_whisperx_model.transcribe.call_args.kwargs["language"] == "en"
