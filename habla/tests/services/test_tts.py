"""Tests for TTSService."""

import io
import struct
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from server.services.tts import TTSService, _build_wav


# ---------------------------------------------------------------------------
# _build_wav
# ---------------------------------------------------------------------------

class TestBuildWav:
    def test_wav_header_structure(self):
        pcm = b"\x00\x01" * 100  # 200 bytes of fake 16-bit PCM
        wav = _build_wav(pcm, 22050)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert wav[36:40] == b"data"

    def test_wav_data_size(self):
        pcm = b"\x00" * 400
        wav = _build_wav(pcm, 16000)
        # data chunk size field at byte 40
        data_size = struct.unpack_from("<I", wav, 40)[0]
        assert data_size == 400
        # RIFF chunk size = 36 + data_size
        riff_size = struct.unpack_from("<I", wav, 4)[0]
        assert riff_size == 436

    def test_wav_sample_rate_encoded(self):
        pcm = b"\x00" * 100
        wav = _build_wav(pcm, 22050)
        # sample_rate at byte 24 in fmt chunk
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 22050

    def test_wav_mono_pcm16(self):
        pcm = b"\x00" * 100
        wav = _build_wav(pcm, 16000)
        # audio format = 1 (PCM)
        fmt = struct.unpack_from("<H", wav, 20)[0]
        assert fmt == 1
        # channels = 1
        channels = struct.unpack_from("<H", wav, 22)[0]
        assert channels == 1
        # bits per sample = 16
        bps = struct.unpack_from("<H", wav, 34)[0]
        assert bps == 16


# ---------------------------------------------------------------------------
# TTSService
# ---------------------------------------------------------------------------

@pytest.fixture
def models_dir(tmp_path):
    return tmp_path / "tts_models"


@pytest.fixture
def svc(models_dir):
    return TTSService(models_dir=models_dir, voice_es="es_ES-carlfm-x_low", voice_en="en_US-lessac-medium")


class TestTTSServiceInit:
    def test_defaults(self, svc, models_dir):
        assert svc._voice_es == "es_ES-carlfm-x_low"
        assert svc._voice_en == "en_US-lessac-medium"
        assert svc._models_dir == models_dir
        assert svc._voices == {}

    def test_set_voices_clears_cache(self, svc):
        fake_voice = MagicMock()
        svc._voices["es"] = fake_voice
        svc._voices["en"] = fake_voice
        svc.set_voices("es_ES-davefx-medium", "en_US-lessac-medium")
        # es changed → evicted; en unchanged → kept
        assert "es" not in svc._voices
        assert "en" in svc._voices

    def test_set_voices_both_changed(self, svc):
        svc._voices["es"] = MagicMock()
        svc._voices["en"] = MagicMock()
        svc.set_voices("es_ES-davefx-medium", "en_US-ryan-medium")
        assert svc._voices == {}


class TestTTSServiceLoadVoice:
    def test_missing_model_returns_none(self, svc):
        result = svc._load_voice_sync("en")
        assert result is None

    def test_missing_model_warns(self, svc, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="habla.tts"):
            svc._load_voice_sync("es")
        assert "not found" in caplog.text

    def test_piper_import_error_returns_none(self, svc, models_dir):
        # Create a fake .onnx file so the existence check passes
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "en_US-lessac-medium.onnx").write_bytes(b"fake")
        with patch.dict("sys.modules", {"piper": None}):
            result = svc._load_voice_sync("en")
        assert result is None

    def test_voice_cached_after_load(self, svc, models_dir):
        models_dir.mkdir(parents=True, exist_ok=True)
        onnx = models_dir / "en_US-lessac-medium.onnx"
        onnx.write_bytes(b"fake")
        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050
        mock_piper = MagicMock()
        mock_piper.PiperVoice.load.return_value = mock_voice
        with patch.dict("sys.modules", {"piper": mock_piper}):
            v1 = svc._load_voice_sync("en")
            v2 = svc._load_voice_sync("en")
        assert v1 is v2
        # PiperVoice.load called exactly once (cached on second call)
        assert mock_piper.PiperVoice.load.call_count == 1


class TestTTSServiceSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_returns_none_when_no_model(self, svc):
        result = await svc.synthesize("Hola", "es")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_returns_wav_bytes(self, svc, models_dir):
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "es_ES-carlfm-x_low.onnx").write_bytes(b"fake")
        pcm = b"\x00\x01" * 50
        mock_voice = MagicMock()
        mock_voice.synthesize_stream_raw.return_value = [pcm]
        mock_voice.config.sample_rate = 22050
        mock_piper = MagicMock()
        mock_piper.PiperVoice.load.return_value = mock_voice
        with patch.dict("sys.modules", {"piper": mock_piper}):
            result = await svc.synthesize("Hola mundo", "es")
        assert result is not None
        assert result[:4] == b"RIFF"
        mock_voice.synthesize_stream_raw.assert_called_once_with("Hola mundo")

    @pytest.mark.asyncio
    async def test_synthesize_handles_synthesis_exception(self, svc, models_dir):
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "en_US-lessac-medium.onnx").write_bytes(b"fake")
        mock_voice = MagicMock()
        mock_voice.synthesize_stream_raw.side_effect = RuntimeError("synth boom")
        mock_voice.config.sample_rate = 22050
        mock_piper = MagicMock()
        mock_piper.PiperVoice.load.return_value = mock_voice
        with patch.dict("sys.modules", {"piper": mock_piper}):
            result = await svc.synthesize("Hello", "en")
        assert result is None
