"""Habla configuration — all settings in one place."""

from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from the habla/ directory (one level up from server/)
load_dotenv(Path(__file__).parent.parent / ".env")


# Language registry — single source of truth for all supported languages.
# To add a new language: add one entry here and it works everywhere.
# Keys are ISO 639-1 codes (same as WhisperX detection output).
LANGUAGES: dict[str, dict] = {
    "es": {"name": "Spanish", "flag": "\U0001F1EA\U0001F1F8"},  # 🇪🇸
    "en": {"name": "English", "flag": "\U0001F1EC\U0001F1E7"},  # 🇬🇧
    # Ready to enable:
    # "fr": {"name": "French", "flag": "\U0001F1EB\U0001F1F7"},
    # "pt": {"name": "Portuguese", "flag": "\U0001F1E7\U0001F1F7"},
    # "de": {"name": "German", "flag": "\U0001F1E9\U0001F1EA"},
    # "it": {"name": "Italian", "flag": "\U0001F1EE\U0001F1F9"},
    # "ja": {"name": "Japanese", "flag": "\U0001F1EF\U0001F1F5"},
    # "zh": {"name": "Chinese", "flag": "\U0001F1E8\U0001F1F3"},
}

# Default target language when auto-detecting. If someone speaks a language
# and we need to pick what to translate TO, use this.
DEFAULT_TARGET_LANG = "en"


def parse_direction(direction: str) -> tuple[str, str]:
    """Parse 'xx_to_yy' direction string into (source_code, target_code).

    Returns language codes. Falls back to ("es", "en") for unknown formats.
    """
    parts = direction.split("_to_")
    if len(parts) == 2 and parts[0] in LANGUAGES and parts[1] in LANGUAGES:
        return parts[0], parts[1]
    return "es", "en"


def lang_name(code: str) -> str:
    """Look up display name for a language code. Returns code itself if unknown."""
    return LANGUAGES.get(code, {}).get("name", code)


class ASRConfig(BaseModel):
    """WhisperX ASR settings."""
    model_size: str = "small"
    device: str = "cuda"
    compute_type: str = "int8"
    beam_size: int = 3
    vad_filter: bool = True
    vad_threshold: float = 0.35
    word_timestamps: bool = True
    language: str = "es"  # toggled by direction
    auto_language: bool = True  # if True, let WhisperX detect language


class TranslatorConfig(BaseModel):
    """Multi-provider LLM translator settings."""
    # Active provider: "ollama", "lmstudio", "openai", or "claude"
    provider: str = "lmstudio"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    # LM Studio (OpenAI-compatible API)
    lmstudio_url: str = "http://localhost:1234"
    lmstudio_model: str = ""  # auto-detected from /v1/models
    lmstudio_executable: str = "lms"  # override via LMSTUDIO_EXECUTABLE; Windows-only feature
    lmstudio_model_paths: list[str] = []  # full GGUF paths to load on startup

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"

    # Claude (Anthropic)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Shared
    temperature: float = 0.3
    max_context_exchanges: int = 5
    timeout_seconds: int = 30
    # Quick translation (partials) model override
    quick_model: str = ""

    # Fallback: if cloud (OpenAI) fails, try local providers. Never local→cloud.
    fallback_to_local: bool = True

    # Rate limiting: minimum seconds between LLM requests
    rate_limit_interval: float = 0.5

    @property
    def model(self) -> str:
        """Return the active model name for the current provider."""
        if self.provider == "lmstudio":
            return self.lmstudio_model
        elif self.provider == "openai":
            return self.openai_model
        elif self.provider == "claude":
            return self.anthropic_model
        return self.ollama_model


class DiarizationConfig(BaseModel):
    """Pyannote speaker diarization settings."""
    device: str = "cpu"  # always CPU to save GPU
    min_speakers: int = 1
    max_speakers: int = 8
    hf_token: str = ""  # set via env var


class AudioConfig(BaseModel):
    """Audio ingress settings."""
    sample_rate: int = 16000
    channels: int = 1
    vad_silence_ms: int = 600  # ms of silence to finalize segment
    min_speech_ms: int = 200  # minimum speech length to process
    max_segment_seconds: float = 30.0  # max segment before forced split
    buffer_max_seconds: float = 60.0  # ring buffer size
    calibration_duration_ms: int = 2000  # noise floor calibration period on session start


class SessionConfig(BaseModel):
    """Session defaults."""
    mode: str = "conversation"  # or "classroom"
    direction: str = "es_to_en"  # or "en_to_es"
    tts_enabled: bool = False
    save_audio_clips: bool = False
    vocab_sensitivity: str = "normal"  # or "high" for classroom


class TTSConfig(BaseModel):
    """Piper TTS settings (CPU-based, no VRAM cost)."""
    enabled: bool = False
    voice_es: str = "es_ES-carlfm-x_low"
    voice_en: str = "en_US-lessac-medium"
    models_dir: Path = Path("data/tts_models")


class RecordingConfig(BaseModel):
    """Audio recording settings for debugging/testing."""
    enabled: bool = False  # Master toggle
    save_raw_audio: bool = True  # Save incoming WebM/Opus from browser
    save_vad_segments: bool = True  # Save VAD-detected speech segments
    output_dir: Path = Path("data/audio/recordings")
    max_recordings: int = 100  # Auto-cleanup old recordings
    include_metadata: bool = True  # Save JSON metadata with each recording


class WebSocketConfig(BaseModel):
    """WebSocket connection settings."""
    ping_interval_seconds: float = 30.0  # Expected client ping interval
    missed_pings_threshold: int = 3  # Close after this many missed intervals


class AppConfig(BaseModel):
    """Root configuration."""
    host: str = "0.0.0.0"
    port: int = 8002
    data_dir: Path = Path("data")
    db_path: Path = Path("data/habla.db")

    asr: ASRConfig = ASRConfig()
    translator: TranslatorConfig = TranslatorConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    audio: AudioConfig = AudioConfig()
    session: SessionConfig = SessionConfig()
    tts: TTSConfig = TTSConfig()
    recording: RecordingConfig = RecordingConfig()
    websocket: WebSocketConfig = WebSocketConfig()


def load_config() -> AppConfig:
    """Load config with environment variable overrides."""
    config = AppConfig()

    # Environment overrides
    if token := os.getenv("HF_TOKEN"):
        config.diarization.hf_token = token
    if url := os.getenv("OLLAMA_URL"):
        config.translator.ollama_url = url
    if model := os.getenv("OLLAMA_MODEL"):
        config.translator.ollama_model = model
    if url := os.getenv("LMSTUDIO_URL"):
        config.translator.lmstudio_url = url
    if model := os.getenv("LMSTUDIO_MODEL"):
        config.translator.lmstudio_model = model
    if exe := os.getenv("LMSTUDIO_EXECUTABLE"):
        config.translator.lmstudio_executable = exe
    if paths := os.getenv("LMSTUDIO_MODEL_PATHS"):  # semicolon-separated full GGUF paths
        config.translator.lmstudio_model_paths = [p.strip() for p in paths.split(";") if p.strip()]
    if key := os.getenv("OPENAI_API_KEY"):
        config.translator.openai_api_key = key
    if model := os.getenv("OPENAI_MODEL"):
        config.translator.openai_model = model
    if key := os.getenv("ANTHROPIC_API_KEY"):
        config.translator.anthropic_api_key = key
    if model := os.getenv("ANTHROPIC_MODEL"):
        config.translator.anthropic_model = model
    if model := os.getenv("QUICK_MODEL"):
        config.translator.quick_model = model
    if provider := os.getenv("LLM_PROVIDER"):
        if provider in ("ollama", "lmstudio", "openai", "claude"):
            config.translator.provider = provider
    if size := os.getenv("WHISPER_MODEL"):
        config.asr.model_size = size
    if os.getenv("WHISPER_DEVICE", "").lower() == "cpu":
        config.asr.device = "cpu"
    if auto_lang := os.getenv("ASR_AUTO_LANGUAGE"):
        config.asr.auto_language = auto_lang.lower() in ("1", "true", "yes", "on")
    if db := os.getenv("DB_PATH"):
        config.db_path = Path(db)
    if data := os.getenv("DATA_DIR"):
        config.data_dir = Path(data)
    if rec := os.getenv("RECORDING_ENABLED"):
        config.recording.enabled = rec.lower() in ("1", "true", "yes", "on")
    if interval := os.getenv("RATE_LIMIT_INTERVAL"):
        config.translator.rate_limit_interval = float(interval)
    if ping := os.getenv("WS_PING_INTERVAL"):
        config.websocket.ping_interval_seconds = float(ping)
    if missed := os.getenv("WS_MISSED_PINGS"):
        config.websocket.missed_pings_threshold = int(missed)
    if os.getenv("TTS_ENABLED", "").lower() in ("1", "true", "yes", "on"):
        config.tts.enabled = True
    if voice := os.getenv("TTS_VOICE_ES"):
        config.tts.voice_es = voice
    if voice := os.getenv("TTS_VOICE_EN"):
        config.tts.voice_en = voice
    if tts_dir := os.getenv("TTS_MODELS_DIR"):
        config.tts.models_dir = Path(tts_dir)

    # VAD / Audio segmentation overrides
    if val := os.getenv("VAD_SILENCE_MS"):
        config.audio.vad_silence_ms = int(val)
    if val := os.getenv("MIN_SPEECH_MS"):
        config.audio.min_speech_ms = int(val)
    if val := os.getenv("MAX_SEGMENT_SECONDS"):
        config.audio.max_segment_seconds = float(val)
    if val := os.getenv("VAD_THRESHOLD"):
        config.asr.vad_threshold = float(val)
    if val := os.getenv("CALIBRATION_DURATION_MS"):
        config.audio.calibration_duration_ms = int(val)

    # Validate critical config
    if not config.diarization.hf_token:
        import logging
        logging.getLogger("habla.config").warning(
            "HF_TOKEN not set — speaker diarization will be disabled"
        )

    # Environment overrides for recording
    if os.getenv("SAVE_AUDIO_RECORDINGS", "").lower() in ("1", "true", "yes", "on"):
        config.recording.enabled = True

    # Ensure data directories exist
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "audio").mkdir(exist_ok=True)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    if config.recording.enabled:
        config.recording.output_dir.mkdir(parents=True, exist_ok=True)

    import logging
    logging.getLogger("habla.config").info(
        "Database path: %s", config.db_path.resolve()
    )

    return config
