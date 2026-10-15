"""Habla configuration — all settings in one place."""

from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv

from server import shared_assets

# Load .env from the habla/ directory (one level up from server/)
load_dotenv(Path(__file__).parent.parent / ".env")


# Language registry — single source of truth for all supported languages,
# shared with the Android app (shared/languages.json). Keys are ISO 639-1
# codes (same as WhisperX detection output).
LANGUAGES: dict[str, dict] = shared_assets.languages()

# Default target language when auto-detecting. If someone speaks a language
# and we need to pick what to translate TO, use this.
DEFAULT_TARGET_LANG: str = shared_assets.load_json("languages.json")["default_target"]

# Tuning shared with the Android app (shared/asr_profile.json).
_PROFILE = shared_assets.asr_profile()


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
    model_size: str = _PROFILE["whisper"]["model"]
    device: str = "cuda"
    compute_type: str = "int8"
    beam_size: int = _PROFILE["whisper"]["beam_size"]
    vad_filter: bool = True
    vad_threshold: float = _PROFILE["vad"]["speech_threshold"]
    word_timestamps: bool = True
    language: str = _PROFILE["whisper"]["default_language"]  # toggled by direction
    auto_language: bool = _PROFILE["whisper"]["auto_language"]  # if True, let WhisperX detect language


class TranslatorConfig(BaseModel):
    """Multi-provider LLM translator settings."""
    # Active provider: "ollama", "lmstudio", "openai", or "claude".
    # Cloud by default (D021); without OPENAI_API_KEY calls fail fast and fall back to local.
    provider: str = "openai"

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
    openai_model: str = shared_assets.models()["openai"]["final_default"]

    # Claude (Anthropic)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Shared
    temperature: float = 0.3
    max_context_exchanges: int = _PROFILE["context"]["max_context_exchanges"]
    timeout_seconds: int = 30
    # Quick translation (partials) model override. Empty on OpenAI means
    # shared/models.json quick_default; empty elsewhere means the main model.
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
    sample_rate: int = _PROFILE["audio"]["sample_rate"]
    channels: int = _PROFILE["audio"]["channels"]
    vad_silence_ms: int = _PROFILE["segmenter"]["silence_duration_ms"]  # ms of silence to finalize segment
    min_speech_ms: int = _PROFILE["segmenter"]["min_speech_ms"]  # minimum speech length to process
    max_segment_seconds: float = _PROFILE["segmenter"]["max_segment_seconds"]  # max segment before forced split
    buffer_max_seconds: float = 60.0  # ring buffer size
    calibration_duration_ms: int = _PROFILE["calibration"]["duration_ms"]  # noise floor calibration period on session start
    vad_frame_ms: int = _PROFILE["vad"]["frame_ms"]  # one 512-sample Silero frame at 16 kHz
    pre_speech_padding_ms: int = _PROFILE["segmenter"]["pre_speech_padding_ms"]


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
