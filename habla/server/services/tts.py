"""CPU-based text-to-speech using Piper (no VRAM cost)."""

import asyncio
import io
import logging
import struct
from pathlib import Path

logger = logging.getLogger("habla.tts")


class TTSService:
    """Synthesize translated text to audio via Piper.

    Voices are loaded lazily on first use and cached. Missing model files
    are tolerated — synthesize() returns None and logs a warning so the
    rest of the pipeline continues unaffected.

    Thread safety: voice loading is serialized via an asyncio lock so
    two concurrent requests for the same language don't double-load.
    """

    def __init__(self, models_dir: Path, voice_es: str, voice_en: str) -> None:
        self._models_dir = models_dir
        self._voice_es = voice_es
        self._voice_en = voice_en
        self._voices: dict[str, object] = {}  # "es" | "en" → PiperVoice
        self._lock = asyncio.Lock()

    def set_voices(self, voice_es: str, voice_en: str) -> None:
        """Replace voice names and evict cached voice objects for changed languages."""
        if voice_es != self._voice_es:
            self._voice_es = voice_es
            self._voices.pop("es", None)
        if voice_en != self._voice_en:
            self._voice_en = voice_en
            self._voices.pop("en", None)

    def _load_voice_sync(self, lang: str) -> object | None:
        """Load and cache a PiperVoice (blocking). Returns None if unavailable."""
        if lang in self._voices:
            return self._voices[lang]

        voice_name = self._voice_es if lang == "es" else self._voice_en
        model_path = self._models_dir / f"{voice_name}.onnx"
        config_path = self._models_dir / f"{voice_name}.onnx.json"

        if not model_path.exists():
            logger.warning("TTS model not found: %s", model_path)
            return None

        try:
            from piper import PiperVoice  # type: ignore[import]
            kw: dict = {}
            if config_path.exists():
                kw["config_path"] = str(config_path)
            voice = PiperVoice.load(str(model_path), **kw)
            self._voices[lang] = voice
            logger.info("TTS voice loaded: %s (%s)", voice_name, lang)
            return voice
        except ImportError:
            logger.warning("piper-tts not installed — TTS unavailable. Install with: pip install piper-tts")
            return None
        except Exception as e:
            logger.error("Failed to load TTS voice %s: %s", voice_name, e, exc_info=True)
            return None

    async def synthesize(self, text: str, lang: str) -> bytes | None:
        """Synthesize text to WAV bytes. Returns None if TTS is unavailable.

        Runs Piper in a thread to avoid blocking the event loop.
        lang: "es" or "en"
        """
        async with self._lock:
            voice = await asyncio.to_thread(self._load_voice_sync, lang)

        if voice is None:
            return None

        def _synth() -> bytes:
            pcm = b"".join(voice.synthesize_stream_raw(text))
            return _build_wav(pcm, voice.config.sample_rate)

        try:
            return await asyncio.to_thread(_synth)
        except Exception as e:
            logger.error("TTS synthesis failed for lang=%s: %s", lang, e, exc_info=True)
            return None


def _build_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap 16-bit mono PCM in a minimal WAV container."""
    data_size = len(pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # fmt chunk size
    buf.write(struct.pack(
        "<HHIIHH",
        1,               # PCM format
        1,               # mono
        sample_rate,
        sample_rate * 2, # byte rate (sample_rate × channels × bits/8)
        2,               # block align
        16,              # bits per sample
    ))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()
