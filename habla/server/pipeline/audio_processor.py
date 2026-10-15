"""Audio processing — ASR, diarization, language detection, and audio decode/normalize."""

import asyncio
import logging
import tempfile
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from server.config import AppConfig

logger = logging.getLogger("habla.pipeline")

# Target RMS for partial audio normalization (~-16.5 dBFS for 16-bit)
_PARTIAL_TARGET_RMS = 3000.0
_PARTIAL_MAX_GAIN = 10.0

# Thresholds for rejecting garbage ASR output (mirrors orchestrator._is_bad_transcript)
_MIN_LETTERS = 3
_MIN_LETTER_RATIO = 0.5


def _is_bad_transcript(text: str) -> bool:
    """Return True if the ASR output looks like noise/garbage rather than real speech."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < _MIN_LETTERS:
        return True
    alnum = sum(ch.isalnum() for ch in stripped)
    if alnum and letters / alnum < _MIN_LETTER_RATIO:
        return True
    return False


# Spanish word fragments that appear when VAD clips the onset of a word.
# Each pair is (clipped_form, correct_form). Lowercase, no accents on the
# clipped side so simple casefold comparison works.
_CLIPPED_ONSET_PAIRS: tuple[tuple[str, str], ...] = (
    ("ola", "hola"),
    ("ueno", "bueno"),
    ("racias", "gracias"),
    ("omo", "como"),
    ("uiero", "quiero"),
    ("engo", "tengo"),
    ("stoy", "estoy"),
    ("oy", "voy"),
    ("ué", "qué"),
    ("ónde", "dónde"),
    ("uándo", "cuándo"),
    ("orque", "porque"),
)


def _has_clipped_onset(text: str) -> bool:
    """Return True if the first word of the transcript looks like a clipped Spanish word.

    Checks for short fragments that match the tail of a common word — a sign
    that VAD fired too late and clipped the word's onset.
    """
    words = text.strip().split()
    if not words:
        return False
    first = words[0].casefold()
    return any(first == clipped for clipped, _ in _CLIPPED_ONSET_PAIRS)


class AudioProcessor:
    """Owns all audio/ASR/language detection state and operations.

    Responsibilities:
    - WhisperX ASR transcription (GPU, serialized via _asr_lock)
    - Pyannote speaker diarization (CPU)
    - Audio decode (ffmpeg WebM/Opus -> WAV) and normalization
    - Language detection voting (prevents single bad detection from poisoning subsequent segments)
    - Streaming partial transcript state

    Lifecycle:
    - Created by PipelineOrchestrator.__init__() before model loading.
    - Model instances (_whisperx_model, _diarize_pipeline) are set during
      PipelineOrchestrator.startup() after loading.
    - shutdown() is a no-op; models are released when the process exits.

    Thread safety:
    - _asr_lock serializes all WhisperX calls (partial and full ASR share the same GPU model).
    - _partial_lock serializes concurrent partial-audio processing to prevent overlap.
    """

    _norm_counter = 0  # class-level counter for unique debug filenames

    def __init__(self, config: AppConfig, source_language_getter: Callable[[], str]):
        self.config = config
        self._source_language_getter = source_language_getter

        # ASR/diarization models (set during startup)
        self._whisperx_model = None
        self._diarize_pipeline = None

        # Thread-safety: WhisperX model is not thread-safe.
        # Serializes all transcribe() calls between partial and final ASR.
        self._asr_lock = threading.Lock()

        # Language detection voting: track recent detections to prevent snowball
        # from a single bad detection poisoning all subsequent segments.
        self._last_detected_language: str | None = None
        self._language_votes: deque[str] = deque(maxlen=5)
        self._language_confidence_threshold = 0.7  # require 70%+ to switch

        # Streaming partials state
        self._last_partial_text = ""
        self._partial_lock = asyncio.Lock()

    @property
    def _source_language(self) -> str:
        """Current source language code ('es' or 'en') based on translation direction."""
        return self._source_language_getter()

    def reset_partial_state(self) -> None:
        """Reset streaming partial state between speech segments."""
        self._last_partial_text = ""

    def _update_detected_language(self, detected: str) -> None:
        """Update language detection using voting to prevent snowball from one bad detection.

        Only switches _last_detected_language when a supermajority of recent
        detections agree. This prevents a single misdetection (e.g. Spanish
        audio detected as English at 40% confidence) from poisoning all
        subsequent segments.
        """
        self._language_votes.append(detected)

        if not self._last_detected_language:
            # First detection — accept it immediately
            self._last_detected_language = detected
            return

        # Count votes for each language
        vote_counts: dict[str, int] = {}
        for vote in self._language_votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # Find the majority language
        majority_lang = max(vote_counts, key=vote_counts.get)
        majority_ratio = vote_counts[majority_lang] / len(self._language_votes)

        # Only switch if the new language has a strong majority AND it differs
        if majority_lang != self._last_detected_language:
            if majority_ratio >= self._language_confidence_threshold:
                logger.info(
                    f"Language switch: {self._last_detected_language} -> {majority_lang} "
                    f"({majority_ratio:.0%} of last {len(self._language_votes)} detections)"
                )
                self._last_detected_language = majority_lang
            else:
                logger.debug(
                    f"Language detection '{detected}' disagrees with current "
                    f"'{self._last_detected_language}' but only {majority_ratio:.0%} "
                    f"majority — keeping current language"
                )

    def _run_asr_and_diarize(self, wav_path: str) -> tuple[str, list[dict], str | None, str]:
        """Run WhisperX ASR and Pyannote diarization (blocking, runs in thread).

        Returns (transcript, diarized_segments, detected_language, reject_reason).

        reject_reason is one of:
          "ok"            — transcript is valid, use it
          "no_segments"   — WhisperX returned zero segments; its internal VAD found no speech
          "bad_transcript" — segments were present but output was garbage (noise, digits-only, etc.)

        The reason is intentionally surfaced so callers can distinguish between audio
        that was too quiet for WhisperX's VAD versus audio that triggered ASR but
        produced garbage — these point to different tuning actions.
        """
        import whisperx

        # Transcribe (lock shared with partial ASR for thread safety)
        lang_arg = self._last_detected_language if self.config.asr.auto_language else self._source_language
        with self._asr_lock:
            result = self._whisperx_model.transcribe(
                wav_path,
                batch_size=8,
                language=lang_arg,
                task="transcribe",
            )
            detected = result.get("language")
            if detected:
                self._update_detected_language(detected)

        # Capture per-segment detected language (raw, not voted)
        segment_detected_lang = detected

        # Extract text — check for empty segments before bad-transcript to get distinct reason
        segments = result.get("segments", [])
        if not segments:
            logger.warning("WhisperX returned no segments — internal VAD found no speech in audio")
            return "", [], segment_detected_lang, "no_segments"

        transcript = " ".join(seg.get("text", "").strip() for seg in segments).strip()
        if _is_bad_transcript(transcript):
            logger.warning(
                f"ASR produced garbage transcript ({len(segments)} segments, text='{transcript[:80]}')"
            )
            return "", [], segment_detected_lang, "bad_transcript"
        logger.info(f"Final transcript: {transcript}")

        # Diarize if available
        diarized_segments = []
        if self._diarize_pipeline and segments:
            try:
                # Pyannote 4.x requires torchcodec for file-path input, which
                # doesn't work on Windows.  Pre-load audio as a waveform dict
                # so Pyannote skips its internal decoder entirely.
                import torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                diarize_input = {"waveform": waveform, "sample_rate": sample_rate}
                diarize_result = self._diarize_pipeline(diarize_input)

                # Pyannote 4.x returns DiarizeOutput; extract the Annotation
                annotation = getattr(diarize_result, "speaker_diarization", diarize_result)

                # Align diarization with transcription
                import pandas as pd
                diarize_segments_raw = []
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    diarize_segments_raw.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker,
                    })

                # whisperx 3.8+ expects a DataFrame with iterrows()
                diarize_df = pd.DataFrame(diarize_segments_raw)
                result_aligned = whisperx.assign_word_speakers(diarize_df, result)
                diarized_segments = result_aligned.get("segments", [])

            except Exception as e:
                logger.warning(f"Diarization failed: {e}")

        return transcript, diarized_segments, segment_detected_lang, "ok"

    def _run_quick_asr(self, wav_path: str) -> tuple[str, str | None]:
        """Fast ASR on a partial segment — transcription only, no alignment or diarization.

        Returns (transcript, detected_language).
        """
        try:
            lang_arg = self._last_detected_language if self.config.asr.auto_language else self._source_language
            with self._asr_lock:
                result = self._whisperx_model.transcribe(
                    wav_path,
                    batch_size=4,  # smaller batch for speed
                    language=lang_arg,
                    task="transcribe",
                )
                detected = result.get("language")
                if detected:
                    self._update_detected_language(detected)
            segments = result.get("segments", [])
            text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
            return ("", detected) if _is_bad_transcript(text) else (text, detected)
        except Exception as e:
            logger.debug(f"Quick ASR error: {e}")
            return "", None

    async def _normalize_wav(self, wav_path: str) -> str:
        """Apply bandpass filter + dynamic loudness normalization to a WAV file for ASR.

        Uses dynaudnorm for frame-by-frame gain adjustment — makes quiet speech
        loud and keeps loud speech consistent, even on short segments.
        Returns path to the normalized WAV (replaces original in-place).
        Also saves a debug copy to recordings/processed/ if recording is enabled.
        """
        normalized_path = wav_path + ".norm.wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", wav_path,
            "-af", "highpass=f=80,lowpass=f=7500,dynaudnorm=f=150:g=15:p=0.95:m=10",
            "-ar", "16000", "-ac", "1", "-f", "wav",
            normalized_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

        if proc.returncode == 0:
            # Save processed audio for debugging if recording is enabled
            if self.config.recording.enabled:
                try:
                    save_dir = Path(self.config.recording.output_dir) / "processed"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    AudioProcessor._norm_counter += 1
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = save_dir / f"asr_input_{ts}_{AudioProcessor._norm_counter:04d}.wav"
                    import shutil
                    shutil.copy2(normalized_path, save_path)
                    logger.info(f"Saved processed audio: {save_path}")
                except Exception as e:
                    logger.debug(f"Could not save processed audio: {e}")

            # Replace original with normalized version
            Path(wav_path).unlink(missing_ok=True)
            Path(normalized_path).rename(wav_path)
        else:
            # Normalization failed, fall back to original
            logger.warning("Audio normalization failed, using original WAV")
            Path(normalized_path).unlink(missing_ok=True)

        return wav_path

    async def _decode_to_wav(self, audio_bytes: bytes) -> str | None:
        """Decode Opus/WebM audio to 16kHz mono WAV for WhisperX."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                input_path = f.name

            output_path = input_path.replace(".webm", ".wav")

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-f", "wav",
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            Path(input_path).unlink(missing_ok=True)

            if proc.returncode == 0:
                return output_path
            else:
                logger.error("ffmpeg decode failed")
                return None

        except Exception as e:
            logger.error(f"Audio decode error: {e}")
            return None

