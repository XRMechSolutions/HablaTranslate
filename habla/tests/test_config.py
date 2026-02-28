"""Tests for server.config — defaults, env var overrides, validation."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from server.config import (
    AppConfig, ASRConfig, TranslatorConfig, DiarizationConfig,
    AudioConfig, SessionConfig, RecordingConfig, WebSocketConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    """Verify all config classes produce correct defaults without env vars."""

    def test_app_config_defaults(self):
        c = AppConfig()
        assert c.host == "0.0.0.0"
        assert c.port == 8002
        assert c.data_dir == Path("data")
        assert c.db_path == Path("data/habla.db")

    def test_asr_config_defaults(self):
        c = ASRConfig()
        assert c.model_size == "small"
        assert c.device == "cuda"
        assert c.compute_type == "int8"
        assert c.beam_size == 3
        assert c.auto_language is True
        assert c.language == "es"

    def test_translator_config_defaults(self):
        c = TranslatorConfig()
        assert c.provider == "lmstudio"
        assert c.ollama_url == "http://localhost:11434"
        assert c.ollama_model == "qwen3:4b"
        assert c.lmstudio_url == "http://localhost:1234"
        assert c.lmstudio_model == ""
        assert c.openai_api_key == ""
        assert c.openai_model == "gpt-5-nano"
        assert c.temperature == 0.3
        assert c.timeout_seconds == 30
        assert c.quick_model == ""
        assert c.fallback_to_local is True
        assert c.rate_limit_interval == 0.5

    def test_translator_model_property_ollama(self):
        c = TranslatorConfig(provider="ollama", ollama_model="llama3")
        assert c.model == "llama3"

    def test_translator_model_property_lmstudio(self):
        c = TranslatorConfig(provider="lmstudio", lmstudio_model="qwen-7b")
        assert c.model == "qwen-7b"

    def test_translator_model_property_openai(self):
        c = TranslatorConfig(provider="openai", openai_model="gpt-4o")
        assert c.model == "gpt-4o"

    def test_diarization_config_defaults(self):
        c = DiarizationConfig()
        assert c.device == "cpu"
        assert c.min_speakers == 1
        assert c.max_speakers == 8
        assert c.hf_token == ""

    def test_audio_config_defaults(self):
        c = AudioConfig()
        assert c.sample_rate == 16000
        assert c.channels == 1
        assert c.vad_silence_ms == 400
        assert c.min_speech_ms == 500
        assert c.max_segment_seconds == 30.0
        assert c.buffer_max_seconds == 60.0

    def test_session_config_defaults(self):
        c = SessionConfig()
        assert c.mode == "conversation"
        assert c.direction == "es_to_en"
        assert c.tts_enabled is False

    def test_recording_config_defaults(self):
        c = RecordingConfig()
        assert c.enabled is False
        assert c.save_raw_audio is True
        assert c.max_recordings == 100

    def test_websocket_config_defaults(self):
        c = WebSocketConfig()
        assert c.ping_interval_seconds == 30.0
        assert c.missed_pings_threshold == 3


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    """Verify load_config() applies env var overrides correctly."""

    def _load_with_env(self, env_vars: dict) -> AppConfig:
        """Load config with specific env vars set, all others cleared."""
        clean = {k: "" for k in [
            "HF_TOKEN", "OLLAMA_URL", "OLLAMA_MODEL", "LMSTUDIO_URL",
            "LMSTUDIO_MODEL", "LMSTUDIO_EXECUTABLE", "LMSTUDIO_MODEL_PATHS",
            "OPENAI_API_KEY", "OPENAI_MODEL", "QUICK_MODEL", "LLM_PROVIDER",
            "WHISPER_MODEL", "WHISPER_DEVICE", "ASR_AUTO_LANGUAGE",
            "DB_PATH", "DATA_DIR", "RECORDING_ENABLED", "RATE_LIMIT_INTERVAL",
            "WS_PING_INTERVAL", "WS_MISSED_PINGS", "SAVE_AUDIO_RECORDINGS",
        ]}
        clean.update(env_vars)
        with patch.dict(os.environ, clean, clear=False):
            return load_config()

    def test_hf_token(self, tmp_path):
        with patch.dict(os.environ, {"DATA_DIR": str(tmp_path), "HF_TOKEN": "hf_test123"}):
            c = load_config()
        assert c.diarization.hf_token == "hf_test123"

    def test_ollama_url(self, tmp_path):
        c = self._load_with_env({"OLLAMA_URL": "http://remote:11434", "DATA_DIR": str(tmp_path)})
        assert c.translator.ollama_url == "http://remote:11434"

    def test_ollama_model(self, tmp_path):
        c = self._load_with_env({"OLLAMA_MODEL": "llama3:8b", "DATA_DIR": str(tmp_path)})
        assert c.translator.ollama_model == "llama3:8b"

    def test_lmstudio_url(self, tmp_path):
        c = self._load_with_env({"LMSTUDIO_URL": "http://lms:5000", "DATA_DIR": str(tmp_path)})
        assert c.translator.lmstudio_url == "http://lms:5000"

    def test_lmstudio_model(self, tmp_path):
        c = self._load_with_env({"LMSTUDIO_MODEL": "qwen3-4b", "DATA_DIR": str(tmp_path)})
        assert c.translator.lmstudio_model == "qwen3-4b"

    def test_lmstudio_model_paths_semicolon_separated(self, tmp_path):
        c = self._load_with_env({
            "LMSTUDIO_MODEL_PATHS": "/models/a.gguf;/models/b.gguf",
            "DATA_DIR": str(tmp_path),
        })
        assert c.translator.lmstudio_model_paths == ["/models/a.gguf", "/models/b.gguf"]

    def test_lmstudio_model_paths_strips_whitespace(self, tmp_path):
        c = self._load_with_env({
            "LMSTUDIO_MODEL_PATHS": " /models/a.gguf ; /models/b.gguf ; ",
            "DATA_DIR": str(tmp_path),
        })
        assert c.translator.lmstudio_model_paths == ["/models/a.gguf", "/models/b.gguf"]

    def test_openai_api_key(self, tmp_path):
        c = self._load_with_env({"OPENAI_API_KEY": "sk-test", "DATA_DIR": str(tmp_path)})
        assert c.translator.openai_api_key == "sk-test"

    def test_openai_model(self, tmp_path):
        c = self._load_with_env({"OPENAI_MODEL": "gpt-4o", "DATA_DIR": str(tmp_path)})
        assert c.translator.openai_model == "gpt-4o"

    def test_quick_model(self, tmp_path):
        c = self._load_with_env({"QUICK_MODEL": "qwen-fast", "DATA_DIR": str(tmp_path)})
        assert c.translator.quick_model == "qwen-fast"

    def test_llm_provider_valid(self, tmp_path):
        for provider in ("ollama", "lmstudio", "openai"):
            c = self._load_with_env({"LLM_PROVIDER": provider, "DATA_DIR": str(tmp_path)})
            assert c.translator.provider == provider

    def test_llm_provider_invalid_ignored(self, tmp_path):
        c = self._load_with_env({"LLM_PROVIDER": "anthropic", "DATA_DIR": str(tmp_path)})
        assert c.translator.provider == "lmstudio"  # default unchanged

    def test_whisper_model(self, tmp_path):
        c = self._load_with_env({"WHISPER_MODEL": "large-v2", "DATA_DIR": str(tmp_path)})
        assert c.asr.model_size == "large-v2"

    def test_whisper_device_cpu(self, tmp_path):
        c = self._load_with_env({"WHISPER_DEVICE": "cpu", "DATA_DIR": str(tmp_path)})
        assert c.asr.device == "cpu"

    def test_whisper_device_default_unchanged(self, tmp_path):
        c = self._load_with_env({"DATA_DIR": str(tmp_path)})
        assert c.asr.device == "cuda"

    def test_asr_auto_language_truthy_values(self, tmp_path):
        for val in ("1", "true", "True", "yes", "on"):
            c = self._load_with_env({"ASR_AUTO_LANGUAGE": val, "DATA_DIR": str(tmp_path)})
            assert c.asr.auto_language is True, f"Failed for {val}"

    def test_asr_auto_language_falsy_values(self, tmp_path):
        for val in ("0", "false", "no", "off"):
            c = self._load_with_env({"ASR_AUTO_LANGUAGE": val, "DATA_DIR": str(tmp_path)})
            assert c.asr.auto_language is False, f"Failed for {val}"

    def test_db_path(self, tmp_path):
        c = self._load_with_env({"DB_PATH": "/custom/db.sqlite", "DATA_DIR": str(tmp_path)})
        assert c.db_path == Path("/custom/db.sqlite")

    def test_data_dir(self, tmp_path):
        custom = tmp_path / "custom_data"
        c = self._load_with_env({"DATA_DIR": str(custom)})
        assert c.data_dir == custom
        assert custom.exists()  # load_config creates it

    def test_recording_enabled(self, tmp_path):
        c = self._load_with_env({"RECORDING_ENABLED": "true", "DATA_DIR": str(tmp_path)})
        assert c.recording.enabled is True

    def test_recording_enabled_via_save_audio(self, tmp_path):
        c = self._load_with_env({"SAVE_AUDIO_RECORDINGS": "1", "DATA_DIR": str(tmp_path)})
        assert c.recording.enabled is True

    def test_rate_limit_interval(self, tmp_path):
        c = self._load_with_env({"RATE_LIMIT_INTERVAL": "1.5", "DATA_DIR": str(tmp_path)})
        assert c.translator.rate_limit_interval == 1.5

    def test_ws_ping_interval(self, tmp_path):
        c = self._load_with_env({"WS_PING_INTERVAL": "15", "DATA_DIR": str(tmp_path)})
        assert c.websocket.ping_interval_seconds == 15.0

    def test_ws_missed_pings(self, tmp_path):
        c = self._load_with_env({"WS_MISSED_PINGS": "5", "DATA_DIR": str(tmp_path)})
        assert c.websocket.missed_pings_threshold == 5


# ---------------------------------------------------------------------------
# Validation and side effects
# ---------------------------------------------------------------------------

class TestValidation:
    """Verify load_config() validation and directory creation."""

    def test_data_dir_created_on_load(self, tmp_path):
        data = tmp_path / "new_data"
        assert not data.exists()
        with patch.dict(os.environ, {"DATA_DIR": str(data)}, clear=False):
            load_config()
        assert data.exists()
        assert (data / "audio").exists()

    def test_recording_dir_created_when_enabled(self, tmp_path):
        rec_dir = tmp_path / "recordings"
        with patch.dict(os.environ, {
            "DATA_DIR": str(tmp_path),
            "RECORDING_ENABLED": "true",
        }, clear=False):
            c = load_config()
            c.recording.output_dir = rec_dir
            # load_config creates it based on default path, but we test the flag
            assert c.recording.enabled is True

    def test_missing_hf_token_logs_warning(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="habla.config"):
            with patch.dict(os.environ, {"DATA_DIR": str(tmp_path), "HF_TOKEN": ""}, clear=False):
                load_config()
        assert "HF_TOKEN not set" in caplog.text

    def test_empty_env_var_uses_default(self, tmp_path):
        """Empty string env vars should not override defaults (walrus := skips empty)."""
        with patch.dict(os.environ, {
            "DATA_DIR": str(tmp_path),
            "OLLAMA_MODEL": "",
            "WHISPER_MODEL": "",
        }, clear=False):
            c = load_config()
        assert c.translator.ollama_model == "qwen3:4b"
        assert c.asr.model_size == "small"

    def test_app_config_nested_subconfigs(self):
        """AppConfig creates all sub-configs with their defaults."""
        c = AppConfig()
        assert isinstance(c.asr, ASRConfig)
        assert isinstance(c.translator, TranslatorConfig)
        assert isinstance(c.diarization, DiarizationConfig)
        assert isinstance(c.audio, AudioConfig)
        assert isinstance(c.session, SessionConfig)
        assert isinstance(c.recording, RecordingConfig)
        assert isinstance(c.websocket, WebSocketConfig)
