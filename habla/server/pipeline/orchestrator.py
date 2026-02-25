"""Pipeline orchestrator — coordinates all processing stages."""

import asyncio
import json
import logging
import time
import tempfile
import threading
from pathlib import Path
from collections import deque
from datetime import datetime, UTC

from server.config import AppConfig
from server.db.database import get_db
from server.models.schemas import (
    Exchange, FlaggedPhrase, SpeakerProfile, TranslationResult,
    WSTranslation, WSPartialTranscript, WSSpeakersUpdate,
)
from server.services.speaker_tracker import SpeakerTracker, SPEAKER_COLORS
from server.services.idiom_scanner import IdiomScanner, IdiomMatch
from server.pipeline.translator import Translator

import numpy as np

logger = logging.getLogger("habla.pipeline")


class PipelineOrchestrator:
    """Manages the full translation pipeline:
    Audio -> ASR (WhisperX) -> Diarization (Pyannote) -> Translation (LLM) -> Output

    Lifecycle:
    - Created once during app lifespan (main.py). Stored in routes._state._pipeline.
    - startup() loads WhisperX, Pyannote, idiom patterns; starts worker task.
    - shutdown() drains queue, saves state to JSON, cancels worker.
    - On console restart: old instance shutdown(), new instance created and startup().

    Ownership:
    - Owns speaker_tracker, idiom_scanner, translator instances.
    - Owns processing queue and worker task.
    - Session lifecycle (create_session/close_session) is DB-backed; session_id is
      set per WebSocket connection and cleared on disconnect.
    - Callbacks are set by ClientSession (websocket.py) and cleared on disconnect.

    Error contract:
    - create_session returns 0 on DB error (logged, non-fatal).
    - process_audio/process_text queue work; errors are delivered via _on_error callback.
    - ASR/diarization failures are logged and skipped (degraded output, not crash).

    All models stay loaded. ASR + diarization on GPU+CPU, translator uses configured provider.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.speaker_tracker = SpeakerTracker()
        self.idiom_scanner = IdiomScanner()
        self.translator = Translator(config.translator)

        # State
        self.direction = config.session.direction
        self.mode = config.session.mode
        self.topic_summary = ""
        self.recent_exchanges: deque[dict] = deque(maxlen=10)
        self.session_id: int | None = None

        # ASR model (loaded on startup)
        self._whisperx_model = None
        self._align_model = None
        self._diarize_pipeline = None
        self._ready = False

        # Processing queue: (kind, payload, future) where kind is "raw" or "wav"
        self._queue: asyncio.Queue[tuple[str, bytes | str, asyncio.Future]] = asyncio.Queue(maxsize=5)
        self._worker_task: asyncio.Task | None = None

        # Callbacks
        self._on_translation = None  # async callback(WSTranslation)
        self._on_partial = None  # async callback(WSPartialTranscript)
        self._on_speakers = None  # async callback(WSSpeakersUpdate)
        self._on_final_transcript = None  # async callback(dict) — source text locked, translation pending
        self._on_error = None  # async callback(str) — translation or pipeline error

        # Streaming partials state
        self._last_partial_text = ""
        self._partial_lock = asyncio.Lock()
        self._last_detected_language: str | None = None

        # Language detection voting: track recent detections to prevent snowball
        # from a single bad detection poisoning all subsequent segments
        self._language_votes: deque[str] = deque(maxlen=5)
        self._language_confidence_threshold = 0.7  # require 70%+ confidence to switch

        # Track in-flight translation tasks for clean shutdown
        self._inflight_translations: set[asyncio.Task] = set()

        # Runtime metrics (lightweight, log-based)
        self._metrics = {
            "segments_processed": 0,
            "translations_completed": 0,
            "translation_errors": 0,
            "peak_queue_depth": 0,
            "sessions_created": 0,
            "sessions_closed": 0,
        }

        # Thread safety: WhisperX model is not thread-safe.
        # Serialize all transcribe() calls between partial and final ASR.
        self._asr_lock = threading.Lock()

    @property
    def metrics(self) -> dict:
        """Return a snapshot of runtime metrics including current queue state."""
        return {
            **self._metrics,
            "queue_depth": self._queue.qsize(),
            "inflight_translations": len(self._inflight_translations),
            "worker_alive": self._worker_task is not None and not self._worker_task.done(),
        }

    async def startup(self):
        """Load all models and prepare the pipeline."""
        logger.info("Loading pipeline models...")

        # Load idiom patterns from JSON files
        idioms_dir = self.config.data_dir / "idioms"
        for json_file in idioms_dir.glob("*.json"):
            self.idiom_scanner.load_from_json(json_file)
        json_count = self.idiom_scanner.count

        # Load user-contributed patterns from database
        await self._load_db_idiom_patterns()
        logger.info(
            f"Loaded {self.idiom_scanner.count} idiom patterns "
            f"({json_count} from files, {self.idiom_scanner.count - json_count} from DB)"
        )

        # Load WhisperX (ASR + alignment)
        try:
            import torch
            import whisperx

            # Workaround: WhisperX's bundled Pyannote VAD crashes on load
            # due to cudnnGetLibConfig symbol mismatch (cuDNN 9.x + torch 2.5).
            # Disabling cuDNN here is safe — ctranslate2 (faster-whisper) uses
            # its own CUDA path, and the VAD model is tiny.
            torch.backends.cudnn.enabled = False

            lang_arg = None if self.config.asr.auto_language else self._source_language
            self._whisperx_model = whisperx.load_model(
                self.config.asr.model_size,
                device=self.config.asr.device,
                compute_type=self.config.asr.compute_type,
                language=lang_arg,
                vad_options={"vad_onset": 0.01, "vad_offset": 0.01},
            )
            torch.backends.cudnn.enabled = True
            logger.info(f"WhisperX {self.config.asr.model_size} loaded on {self.config.asr.device}")
        except Exception as e:
            logger.error(f"Failed to load WhisperX: {e}")
            logger.info("Pipeline will operate in text-only mode (no ASR)")

        # Load Pyannote diarization (CPU)
        try:
            if self.config.diarization.hf_token:
                from pyannote.audio import Pipeline as PyannotePipeline

                self._diarize_pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.config.diarization.hf_token,
                )
                import torch
                if self._diarize_pipeline is None:
                    raise RuntimeError(
                        "from_pretrained returned None — check HF token is valid and "
                        "terms accepted at hf.co/pyannote/speaker-diarization-3.1 "
                        "and hf.co/pyannote/segmentation-3.0"
                    )
                self._diarize_pipeline.to(torch.device("cpu"))
                logger.info("Pyannote 3.1 diarization loaded on CPU")
            else:
                logger.warning("No HF_TOKEN — diarization disabled. Set HF_TOKEN env var.")
        except Exception as e:
            logger.warning(f"Pyannote load failed: {e}. Diarization disabled.")

        # Auto-detect LLM model if needed (e.g. LM Studio with no model configured)
        await self.translator.auto_detect_model()

        self._ready = True

        # Restore previous session context (topic, exchanges, speakers) if available
        await self._restore_shutdown_state()

        # Start background worker
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("Pipeline ready")

    async def shutdown(self):
        """Graceful shutdown: drain queue, save state, close resources."""
        logger.info(f"Pipeline shutting down... metrics={self.metrics}")

        # 1. Stop accepting new work
        self._ready = False

        # 2. Drain the processing queue (give in-flight items a chance to finish)
        if not self._queue.empty():
            remaining = self._queue.qsize()
            logger.info(f"Draining {remaining} queued item(s)...")
            try:
                # Wait up to 30s for queue to drain
                deadline = time.monotonic() + 30
                while not self._queue.empty() and time.monotonic() < deadline:
                    await asyncio.sleep(0.5)
                if not self._queue.empty():
                    logger.warning(f"Shutdown timeout: {self._queue.qsize()} items still in queue")
            except Exception as e:
                logger.warning(f"Error draining queue: {e}")

        # 3. Cancel the background worker
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # 4. Cancel any remaining queued futures
        while not self._queue.empty():
            try:
                _, _, future = self._queue.get_nowait()
                if not future.done():
                    future.cancel()
            except asyncio.QueueEmpty:
                break

        # 5. Wait for in-flight translations to complete (max 15s)
        if self._inflight_translations:
            pending = len(self._inflight_translations)
            logger.info(f"Waiting for {pending} in-flight translation(s)...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._inflight_translations, return_exceptions=True),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for in-flight translations")

        # 6. Save session state for potential resume
        await self._save_shutdown_state()

        # 7. Close resources
        await self.translator.close()
        logger.info("Pipeline shut down")

    async def _save_shutdown_state(self):
        """Persist conversation context and metrics so state survives restarts."""
        try:
            state = {
                "shutdown_time": datetime.now(UTC).isoformat(),
                "direction": self.direction,
                "mode": self.mode,
                "topic_summary": self.topic_summary,
                "recent_exchanges": list(self.recent_exchanges),
                "speakers": {
                    sid: {
                        "auto_label": sp.label,
                        "custom_name": sp.custom_name,
                        "role_hint": sp.role_hint,
                        "utterance_count": sp.utterance_count,
                    }
                    for sid, sp in self.speaker_tracker.speakers.items()
                },
                "translator_metrics": self.translator.metrics,
            }
            state_path = self.config.data_dir / "last_session.json"
            state_path.write_text(json.dumps(state, indent=2, default=str))
            logger.info(f"Session state saved to {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save session state: {e}")

    async def _restore_shutdown_state(self):
        """Restore conversation context from last shutdown if the file exists."""
        state_path = self.config.data_dir / "last_session.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text())
            shutdown_time = state.get("shutdown_time", "unknown")

            if state.get("topic_summary"):
                self.topic_summary = state["topic_summary"]

            for ex in state.get("recent_exchanges", []):
                self.recent_exchanges.append(ex)

            for sid, sp_data in state.get("speakers", {}).items():
                idx = len(self.speaker_tracker.speakers)
                self.speaker_tracker.speakers[sid] = SpeakerProfile(
                    id=sid,
                    label=sp_data.get("auto_label", f"Speaker {chr(65 + idx)}"),
                    custom_name=sp_data.get("custom_name"),
                    role_hint=sp_data.get("role_hint"),
                    color=SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
                    utterance_count=sp_data.get("utterance_count", 0),
                )

            logger.info(
                f"Restored session state from {shutdown_time}: "
                f"{len(self.recent_exchanges)} exchanges, "
                f"{len(self.speaker_tracker.speakers)} speakers, "
                f"topic='{self.topic_summary[:60] if self.topic_summary else '(none)'}'"
            )
        except Exception as e:
            logger.warning(f"Failed to restore session state: {e}")

    @property
    def _source_language(self) -> str:
        return "es" if self.direction == "es_to_en" else "en"

    @property
    def ready(self) -> bool:
        return self._ready

    def set_callbacks(self, on_translation=None, on_partial=None, on_speakers=None, on_final_transcript=None, on_error=None):
        """Set async callbacks for pipeline events."""
        self._on_translation = on_translation
        self._on_partial = on_partial
        self._on_speakers = on_speakers
        self._on_final_transcript = on_final_transcript
        self._on_error = on_error

    def reset_partial_state(self):
        """Reset streaming partial state between speech segments."""
        self._last_partial_text = ""

    def set_direction(self, direction: str):
        """Toggle translation direction."""
        self.direction = direction
        self._last_detected_language = None
        logger.info(f"Direction set to {direction}")

    def set_mode(self, mode: str):
        """Switch between conversation and classroom mode."""
        self.mode = mode
        logger.info(f"Mode set to {mode}")

    async def create_session(self) -> int:
        """Create a new session record in the database. Returns session ID."""
        try:
            db = await get_db()
            cursor = await db.execute(
                """INSERT INTO sessions (mode, direction)
                   VALUES (?, ?)""",
                (self.mode, self.direction),
            )
            await db.commit()
            self.session_id = cursor.lastrowid
            self._metrics["sessions_created"] += 1
            logger.info(f"Session {self.session_id} created")
            return self.session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}", exc_info=True)
            return 0

    async def close_session(self):
        """Close the current session: set end time, persist topic summary and speakers."""
        if not self.session_id:
            return
        try:
            db = await get_db()
            # Get cost data from translator for persistence
            costs = self.translator.get_session_costs()

            await db.execute(
                """UPDATE sessions SET ended_at = CURRENT_TIMESTAMP,
                   topic_summary = ?, speaker_count = ?,
                   llm_provider = ?, llm_input_tokens = ?,
                   llm_output_tokens = ?, llm_cost_usd = ?
                   WHERE id = ?""",
                (
                    self.topic_summary or None,
                    len(self.speaker_tracker.speakers),
                    self.translator.config.provider,
                    costs["session_input_tokens"],
                    costs["session_output_tokens"],
                    costs["session_cost_usd"],
                    self.session_id,
                ),
            )
            # Persist all speaker profiles for this session
            for sid, sp in self.speaker_tracker.speakers.items():
                await db.execute(
                    """INSERT OR REPLACE INTO speakers
                       (id, session_id, auto_label, custom_name, role_hint, color, utterance_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (sid, self.session_id, sp.label, sp.custom_name,
                     sp.role_hint, sp.color, sp.utterance_count),
                )
            await db.commit()
            self._metrics["sessions_closed"] += 1
            logger.info(f"Session {self.session_id} closed ({len(self.speaker_tracker.speakers)} speakers)")
        except Exception as e:
            logger.error(f"Failed to close session: {e}", exc_info=True)

    async def reset_session(self) -> int:
        """Close current session and start a fresh one. Returns new session ID."""
        await self.close_session()
        self.speaker_tracker = SpeakerTracker()
        self.recent_exchanges.clear()
        self.topic_summary = ""
        self.reset_partial_state()
        new_id = await self.create_session()
        logger.info(f"Session reset -> new session {new_id}")
        return new_id

    async def _save_exchange(self, exchange: Exchange) -> int | None:
        """Persist a completed exchange to the database. Returns the exchange ID."""
        if not self.session_id:
            return None
        try:
            db = await get_db()
            correction_json = None
            if exchange.correction_detail:
                correction_json = json.dumps(exchange.correction_detail.model_dump())
            cursor = await db.execute(
                """INSERT INTO exchanges
                   (session_id, speaker_id, direction, raw_transcript,
                    corrected_source, translation, confidence,
                    is_correction, correction_json, processing_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id,
                    exchange.speaker.id,
                    exchange.direction,
                    exchange.raw_transcript,
                    exchange.corrected_source,
                    exchange.translation,
                    exchange.confidence,
                    exchange.is_correction,
                    correction_json,
                    exchange.processing_ms,
                ),
            )
            await db.commit()
            exchange.id = cursor.lastrowid
            return exchange.id
        except Exception as e:
            logger.error(f"Failed to save exchange: {e}", exc_info=True)
            return None

    async def process_wav(self, wav_path: str) -> Exchange | None:
        """
        Process a WAV file through the full pipeline (used by continuous mode).
        Routes through the queue for proper serialization and backpressure.
        """
        if not self._ready:
            logger.warning("Pipeline not ready, dropping audio")
            return None

        future = asyncio.get_event_loop().create_future()
        await self._queue.put(("wav", wav_path, future))
        return await future

    async def process_text(self, text: str, speaker_id: str = "MANUAL") -> Exchange:
        """Process text directly (skip ASR). Useful for testing or typed input."""
        start_time = time.monotonic()

        speaker = self.speaker_tracker.record_utterance(speaker_id)
        speaker_name = self.speaker_tracker.get_display_name(speaker_id)

        # Idiom scan
        idiom_matches = self.idiom_scanner.scan(text)

        # Translate
        result = await self.translator.translate(
            transcript=text,
            speaker_label=speaker_name,
            direction=self.direction,
            mode=self.mode,
            context_exchanges=list(self.recent_exchanges),
            topic_summary=self.topic_summary,
        )

        # Merge idiom matches
        merged_phrases = self._merge_idioms(idiom_matches, result.flagged_phrases)

        # Update speaker hint
        if result.speaker_hint and not speaker.custom_name:
            self.speaker_tracker.set_role_hint(speaker_id, result.speaker_hint)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        exchange = Exchange(
            session_id=self.session_id or 0,
            speaker=speaker,
            direction=self.direction,
            raw_transcript=text,
            corrected_source=result.corrected,
            translation=result.translated,
            flagged_phrases=merged_phrases,
            confidence=result.confidence,
            is_correction=result.is_correction,
            correction_detail=result.correction_detail,
            processing_ms=elapsed_ms,
        )

        # Update context
        self.recent_exchanges.append({
            "speaker_label": speaker_name,
            "source": text,
            "corrected": result.corrected,
            "translated": result.translated,
        })

        # Persist exchange to database
        await self._save_exchange(exchange)

        # Async topic summary update (fire and forget)
        asyncio.create_task(self._update_topic(result, speaker_name))

        # Notify via callback
        if self._on_translation:
            ws_msg = WSTranslation(
                exchange_id=exchange.id,
                speaker=speaker,
                source=text,
                corrected=result.corrected,
                translated=result.translated,
                idioms=merged_phrases,
                is_correction=result.is_correction,
                correction_detail=result.correction_detail,
                confidence=result.confidence,
                timestamp=datetime.now(UTC).isoformat(),
            )
            await self._on_translation(ws_msg)

        return exchange

    async def process_partial_audio(self, pcm_bytes: bytes, duration: float):
        """
        Run a fast ASR pass on a partial audio snapshot (while speech is ongoing).
        Sends partial transcript + quick translation to the client for live display.
        """
        if not self._whisperx_model:
            return

        async with self._partial_lock:
            wav_path = None
            try:
                # Bandpass filter + normalize PCM volume for partial ASR
                import struct as _struct
                from scipy.signal import butter, sosfilt
                audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

                # Bandpass 80Hz-7500Hz to remove rumble and hiss
                sos = butter(4, [80, 7500], btype='band', fs=16000, output='sos')
                audio = sosfilt(sos, audio)

                # RMS normalization
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 10:  # skip near-silence
                    gain = min(_PARTIAL_TARGET_RMS / rms, _PARTIAL_MAX_GAIN)
                    audio = np.clip(audio * gain, -32768, 32767)
                normalized_bytes = audio.astype(np.int16).tobytes()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="habla_partial_") as f:
                    wav_path = f.name
                    data_size = len(normalized_bytes)
                    f.write(b'RIFF')
                    f.write(_struct.pack('<I', 36 + data_size))
                    f.write(b'WAVE')
                    f.write(b'fmt ')
                    f.write(_struct.pack('<I', 16))
                    f.write(_struct.pack('<HHIIHH', 1, 1, 16000, 32000, 2, 16))
                    f.write(b'data')
                    f.write(_struct.pack('<I', data_size))
                    f.write(normalized_bytes)

                # Fast ASR — run in thread
                transcript = await asyncio.to_thread(self._run_quick_asr, wav_path)

                if not transcript or transcript.strip() == self._last_partial_text:
                    return  # no change, don't spam the client

                self._last_partial_text = transcript.strip()
                logger.info(f"Partial transcript: {self._last_partial_text}")

                # Send partial source text immediately
                if self._on_partial:
                    await self._on_partial(WSPartialTranscript(
                        type="partial",
                        text=transcript.strip(),
                        speaker_id=None,
                    ))

                # Quick translate — lightweight LLM call for partial translation
                quick_translation = await self._quick_translate(transcript.strip())

                if quick_translation and self._on_partial:
                    await self._on_partial(WSPartialTranscript(
                        type="partial_translation",
                        text=quick_translation,
                        speaker_id=None,
                    ))

            except Exception as e:
                logger.warning(f"Partial processing error: {e}")
            finally:
                if wav_path:
                    Path(wav_path).unlink(missing_ok=True)

    def _run_quick_asr(self, wav_path: str) -> str:
        """Fast ASR on a partial segment — transcription only, no alignment or diarization."""
        try:
            if self.config.asr.auto_language:
                lang_arg = self._last_detected_language
            else:
                lang_arg = self._source_language
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
            return "" if _is_bad_transcript(text) else text
        except Exception as e:
            logger.debug(f"Quick ASR error: {e}")
            return ""

    async def _quick_translate(self, text: str) -> str:
        """
        Lightweight translation for partials — shorter prompt, no idiom detection.
        Returns just the translated text, nothing else.
        """
        if not text:
            return ""

        if self.direction == "es_to_en":
            src, tgt = "Spanish", "English"
        else:
            src, tgt = "English", "Spanish"

        try:
            model_override = self.config.translator.quick_model
            provider = self.config.translator.provider
            if provider == "lmstudio" and model_override:
                result = await self.translator._call_provider(
                    provider, "", f"Translate this {src} to natural {tgt}. Output ONLY the translation, nothing else.\n\n{text}",
                    retries=0, max_tokens=256, temperature=0.2, json_mode=False,
                    lmstudio_model=model_override,
                )
            else:
                result = await self.translator._call_llm(
                    "",
                    f"Translate this {src} to natural {tgt}. Output ONLY the translation, nothing else.\n\n{text}",
                    retries=0, max_tokens=256, temperature=0.2, json_mode=False,
                )
            return result.strip()
        except Exception as e:
            logger.debug(f"Quick translate error: {e}")
            return ""

    async def _process_queue(self):
        """Background worker that processes audio from the queue."""
        while True:
            try:
                item = await self._queue.get()
                kind, payload, future = item
                # Track peak queue depth
                depth = self._queue.qsize() + 1  # +1 for the item we just took
                if depth > self._metrics["peak_queue_depth"]:
                    self._metrics["peak_queue_depth"] = depth
                try:
                    if kind == "wav":
                        result = await self._process_audio_segment_from_wav(payload)
                    else:
                        result = await self._process_audio_segment(payload)
                    self._metrics["segments_processed"] += 1
                    if result:
                        self._metrics["translations_completed"] += 1
                    if not future.done():
                        future.set_result(result)
                except asyncio.CancelledError:
                    if not future.done():
                        future.cancel()
                    raise
                except Exception as e:
                    self._metrics["translation_errors"] += 1
                    if not future.done():
                        future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}", exc_info=True)

    async def _process_audio_segment(self, audio_bytes: bytes) -> Exchange | None:
        """Run full pipeline on one audio segment (from raw Opus/WebM bytes)."""
        wav_path = await self._decode_to_wav(audio_bytes)
        if not wav_path:
            return None
        try:
            return await self._process_audio_segment_from_wav(wav_path)
        finally:
            Path(wav_path).unlink(missing_ok=True)

    _norm_counter = 0  # class-level counter for unique filenames

    async def _normalize_wav(self, wav_path: str) -> str:
        """Apply bandpass filter + dynamic loudness normalization to a WAV file for ASR.

        Uses dynaudnorm for frame-by-frame gain adjustment — makes quiet speech
        loud and keeps loud speech consistent, even on short segments.
        Returns path to the normalized WAV (replaces original in-place).
        Also saves a copy to the recordings directory if recording is enabled.
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
                    PipelineOrchestrator._norm_counter += 1
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = save_dir / f"asr_input_{ts}_{PipelineOrchestrator._norm_counter:04d}.wav"
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

    async def _process_audio_segment_from_wav(self, wav_path: str) -> Exchange | None:
        """Run full pipeline on a WAV file (used by both PTT and continuous modes).

        ASR runs in the queue (GPU-serial). Once transcript is ready, translation
        is fired off concurrently so the queue can process the next segment's ASR
        without waiting for a slow LLM call.
        """
        start_time = time.monotonic()

        # Normalize audio (bandpass + loudness) before ASR
        wav_path = await self._normalize_wav(wav_path)

        # Run ASR + diarization in thread (blocking GPU ops)
        transcript, diarized_segments = await asyncio.to_thread(
            self._run_asr_and_diarize, wav_path
        )

        if not transcript or not transcript.strip():
            logger.warning("Full ASR returned empty transcript — WhisperX VAD may have rejected the audio")
            return None

        # Determine primary speaker from diarization
        speaker_id = "SPEAKER_00"
        if diarized_segments:
            speaker_times: dict[str, float] = {}
            for seg in diarized_segments:
                spk = seg.get("speaker", "SPEAKER_00")
                duration = seg.get("end", 0) - seg.get("start", 0)
                speaker_times[spk] = speaker_times.get(spk, 0) + duration
            speaker_id = max(speaker_times, key=speaker_times.get)

        # Send finalized source text immediately (before translation starts)
        if self._on_final_transcript:
            speaker = self.speaker_tracker.get_or_create(speaker_id)
            await self._on_final_transcript({
                "type": "transcript_final",
                "text": transcript.strip(),
                "speaker": speaker.model_dump() if speaker else None,
                "direction": self.direction,
            })

        # Fire translation concurrently — don't block the ASR queue waiting for LLM
        task = asyncio.create_task(self._translate_and_notify(transcript, speaker_id, start_time))
        self._inflight_translations.add(task)
        task.add_done_callback(self._inflight_translations.discard)

        return None  # Translation result sent via callback, not via future

    async def _translate_and_notify(self, transcript: str, speaker_id: str, start_time: float):
        """Run translation outside the queue so ASR can continue processing."""
        try:
            exchange = await self.process_text(transcript, speaker_id)
            if exchange:
                exchange.processing_ms = int((time.monotonic() - start_time) * 1000)
                logger.info(f"Translation done in {exchange.processing_ms}ms: '{transcript[:60]}'")
        except Exception as e:
            logger.error(f"Translation error: {e}")
            if self._on_error:
                try:
                    await self._on_error(f"Translation failed: {e}")
                except Exception:
                    pass

    def _run_asr_and_diarize(self, wav_path: str) -> tuple[str, list[dict]]:
        """Run WhisperX ASR and Pyannote diarization (blocking, runs in thread)."""
        import whisperx

        # Transcribe (lock shared with partial ASR for thread safety)
        if self.config.asr.auto_language:
            lang_arg = self._last_detected_language
        else:
            lang_arg = self._source_language
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

        # Extract text
        segments = result.get("segments", [])
        transcript = " ".join(seg.get("text", "").strip() for seg in segments).strip()
        if _is_bad_transcript(transcript):
            logger.warning(f"ASR produced bad/empty transcript ({len(segments)} segments, text='{transcript[:80]}')")
            return "", []
        logger.info(f"Final transcript: {transcript}")

        # Diarize if available
        diarized_segments = []
        if self._diarize_pipeline and segments:
            try:
                diarize_result = self._diarize_pipeline(wav_path)

                # Align diarization with transcription
                diarize_segments_raw = []
                for turn, _, speaker in diarize_result.itertracks(yield_label=True):
                    diarize_segments_raw.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker,
                    })

                # Assign speakers to whisper segments
                result_aligned = whisperx.assign_word_speakers(
                    diarize_segments_raw, result
                )
                diarized_segments = result_aligned.get("segments", [])

            except Exception as e:
                logger.warning(f"Diarization failed: {e}")

        return transcript, diarized_segments

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

    def _merge_idioms(
        self, pattern_matches: list[IdiomMatch], llm_phrases: list[FlaggedPhrase]
    ) -> list[FlaggedPhrase]:
        """Merge pattern DB matches with LLM-detected phrases, deduplicating."""
        result = []
        seen = set()

        # Pattern DB matches first (more curated)
        for m in pattern_matches:
            key = m.canonical.lower()
            if key not in seen:
                seen.add(key)
                result.append(FlaggedPhrase(
                    phrase=m.canonical,
                    literal=m.literal,
                    meaning=m.meaning,
                    type="idiom",
                    region=m.region,
                    source="pattern_db",
                    save_worthy=True,
                    span_start=m.match_start,
                    span_end=m.match_end,
                ))

        # LLM-detected (novel ones only)
        for fp in llm_phrases:
            key = fp.phrase.lower()
            if key not in seen:
                seen.add(key)
                fp.source = "llm"
                result.append(fp)

        return result

    async def _load_db_idiom_patterns(self):
        """Load user-contributed idiom patterns from the database."""
        try:
            db = await get_db()
            rows = await db.execute_fetchall("SELECT * FROM idiom_patterns")
            if rows:
                self.idiom_scanner.load_from_db([dict(r) for r in rows])
        except Exception as e:
            logger.warning(f"Failed to load DB idiom patterns: {e}")

    async def reload_idiom_patterns(self):
        """Reload all idiom patterns (JSON files + DB). Called after new patterns are added."""
        self.idiom_scanner.patterns.clear()
        idioms_dir = self.config.data_dir / "idioms"
        for json_file in idioms_dir.glob("*.json"):
            self.idiom_scanner.load_from_json(json_file)
        await self._load_db_idiom_patterns()
        logger.info(f"Reloaded {self.idiom_scanner.count} idiom patterns")

    def _update_detected_language(self, detected: str):
        """Update language detection using voting to prevent snowball from one bad detection.

        Only switches _last_detected_language when a supermajority of recent
        detections agree. This prevents a single misdetection (e.g. Spanish
        audio detected as English at 40% confidence) from poisoning all
        subsequent segments.
        """
        self._language_votes.append(detected)

        if not self._last_detected_language:
            # First detection — accept it but only if we have no prior language
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

    async def _update_topic(self, result: TranslationResult, speaker_name: str):
        """Update topic summary asynchronously."""
        try:
            new_summary = await self.translator.update_topic_summary(
                previous_summary=self.topic_summary,
                latest_source=result.corrected,
                latest_translation=result.translated,
                speaker_label=speaker_name,
            )
            if new_summary:
                self.topic_summary = new_summary
        except Exception as e:
            logger.debug(f"Topic update failed (non-critical): {e}")



# Thresholds for rejecting garbage ASR output
_MIN_LETTERS = 3           # At least this many alphabetic chars required
_MIN_LETTER_RATIO = 0.5    # At least this fraction of alnum chars must be letters

# Target RMS for partial audio normalization (~-16.5 dBFS for 16-bit)
_PARTIAL_TARGET_RMS = 3000.0
_PARTIAL_MAX_GAIN = 10.0   # Cap at 20dB boost


def _is_bad_transcript(text: str) -> bool:
    if not text or not text.strip():
        return True
    stripped = text.strip()
    # Count actual alphabetic characters
    letters = sum(ch.isalpha() for ch in stripped)
    # Reject if too few letters (noise, punctuation-only, numbers)
    if letters < _MIN_LETTERS:
        return True
    # Reject if mostly non-alphabetic (timestamps, garbage)
    alnum = sum(ch.isalnum() for ch in stripped)
    if alnum and letters / alnum < _MIN_LETTER_RATIO:
        return True
    return False
