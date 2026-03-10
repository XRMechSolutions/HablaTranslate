"""Pipeline orchestrator — thin coordinator that owns the queue and delegates to submodules."""

import asyncio
import logging
import time
import tempfile
import wave
from datetime import datetime, UTC
from pathlib import Path

from server.config import AppConfig, LANGUAGES, DEFAULT_TARGET_LANG, parse_direction, lang_name
from server.models.schemas import (
    Exchange, FlaggedPhrase, TranslationResult,
    WSTranslation, WSPartialTranscript, WSSpeakersUpdate,
)
from server.services.idiom_scanner import IdiomMatch
from server.pipeline.audio_processor import (
    AudioProcessor, _is_bad_transcript, _has_clipped_onset,
    _CLIPPED_ONSET_PAIRS, _MIN_LETTERS, _MIN_LETTER_RATIO,
    _PARTIAL_TARGET_RMS, _PARTIAL_MAX_GAIN,
)
from server.pipeline.context_manager import ContextManager
from server.pipeline.session_manager import SessionManager

import numpy as np

logger = logging.getLogger("habla.pipeline")


class PipelineOrchestrator:
    """Coordinates the full translation pipeline: Audio → ASR → Diarization → Translation → Output.

    Owns directly:
    - Processing queue and background worker task
    - In-flight translation task set
    - Async callbacks (on_translation, on_partial, on_speakers, on_final_transcript, on_error)
    - Ready flag and runtime metrics dict

    Delegates to:
    - AudioProcessor: ASR, diarization, audio decode/normalize, language detection, partial state
    - ContextManager: translator, speaker_tracker, idiom_scanner, conversation context
    - SessionManager: DB operations, session lifecycle, state persistence

    Lifecycle:
    - Created once during app lifespan (main.py). Stored in routes._state._pipeline.
    - startup() loads WhisperX, Pyannote, idiom patterns; starts worker task.
    - shutdown() drains queue, saves state to JSON, cancels worker.
    - On console restart: old instance shutdown(), new instance created and startup().

    Error contract:
    - create_session returns 0 on DB error (logged, non-fatal).
    - process_audio/process_text queue work; errors are delivered via _on_error callback.
    - ASR/diarization failures are logged and skipped (degraded output, not crash).

    All models stay loaded. ASR + diarization on GPU+CPU, translator uses configured provider.
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Submodule: translation/context state
        self.context_manager = ContextManager(config)

        # Submodule: audio/ASR state
        # source_language_getter lets AudioProcessor resolve direction at call time
        self.audio_processor = AudioProcessor(
            config,
            source_language_getter=lambda: self._source_language,
        )

        # Submodule: DB operations and session lifecycle
        # Orchestrator-owned state
        self._ready = False

        # Processing queue: (kind, payload, future, meta) where kind is "raw" or "wav"
        self._queue: asyncio.Queue[tuple[str, bytes | str, asyncio.Future, dict]] = asyncio.Queue(maxsize=5)
        self._worker_task: asyncio.Task | None = None

        # Callbacks
        self._on_translation = None  # async callback(WSTranslation)
        self._on_partial = None      # async callback(WSPartialTranscript)
        self._on_speakers = None     # async callback(WSSpeakersUpdate)
        self._on_final_transcript = None  # async callback(dict)
        self._on_error = None        # async callback(str)

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
            "low_confidence_count": 0,
            "corrections_detected": 0,
            "idiom_pattern_db_hits": 0,
            "idiom_llm_hits": 0,
            "asr_rejected_count": 0,
            "asr_empty_count": 0,
        }

        # Session manager (created after metrics dict exists)
        self.session_manager = SessionManager(config, self.context_manager, self._metrics)

    # ---------------------------------------------------------------------------
    # Computed properties
    # ---------------------------------------------------------------------------

    @property
    def _source_language(self) -> str:
        if self.context_manager.direction == "auto":
            return self.audio_processor._last_detected_language or "es"
        return "es" if self.context_manager.direction == "es_to_en" else "en"

    @property
    def metrics(self) -> dict:
        """Return a snapshot of runtime metrics including current queue state."""
        return {
            **self._metrics,
            "queue_depth": self._queue.qsize(),
            "inflight_translations": len(self._inflight_translations),
            "worker_alive": self._worker_task is not None and not self._worker_task.done(),
        }

    @property
    def ready(self) -> bool:
        return self._ready

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    async def startup(self):
        """Load all models and prepare the pipeline."""
        logger.info("Loading pipeline models...")

        # Load idiom patterns from JSON files
        idioms_dir = self.config.data_dir / "idioms"
        for json_file in idioms_dir.glob("*.json"):
            self.context_manager.idiom_scanner.load_from_json(json_file)
        json_count = self.context_manager.idiom_scanner.count

        # Load user-contributed patterns from database
        await self.session_manager.load_db_idiom_patterns()
        logger.info(
            f"Loaded {self.context_manager.idiom_scanner.count} idiom patterns "
            f"({json_count} from files, {self.context_manager.idiom_scanner.count - json_count} from DB)"
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
            self.audio_processor._whisperx_model = whisperx.load_model(
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

                self.audio_processor._diarize_pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.config.diarization.hf_token,
                )
                import torch
                if self.audio_processor._diarize_pipeline is None:
                    raise RuntimeError(
                        "from_pretrained returned None — check HF token is valid and "
                        "terms accepted at hf.co/pyannote/speaker-diarization-3.1 "
                        "and hf.co/pyannote/segmentation-3.0"
                    )
                self.audio_processor._diarize_pipeline.to(torch.device("cpu"))
                logger.info("Pyannote 3.1 diarization loaded on CPU")
            else:
                logger.warning("No HF_TOKEN — diarization disabled. Set HF_TOKEN env var.")
        except Exception as e:
            logger.warning(f"Pyannote load failed: {e}. Diarization disabled.")

        # Auto-detect LLM model if needed (e.g. LM Studio with no model configured)
        await self.context_manager.translator.auto_detect_model()

        self._ready = True

        # Restore previous session context (topic, exchanges, speakers) if available
        await self.session_manager.restore_shutdown_state()

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
                _, _, future, *_ = self._queue.get_nowait()
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
        await self.session_manager.save_shutdown_state()

        # 7. Close resources
        await self.context_manager.translator.close()
        logger.info("Pipeline shut down")

    # ---------------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------------

    def set_callbacks(self, on_translation=None, on_partial=None, on_speakers=None, on_final_transcript=None, on_error=None):
        """Set async callbacks for pipeline events."""
        self._on_translation = on_translation
        self._on_partial = on_partial
        self._on_speakers = on_speakers
        self._on_final_transcript = on_final_transcript
        self._on_error = on_error

    # ---------------------------------------------------------------------------
    # Direction / mode / partial state
    # ---------------------------------------------------------------------------

    def reset_partial_state(self):
        """Reset streaming partial state between speech segments."""
        self.audio_processor.reset_partial_state()

    def set_direction(self, direction: str):
        """Toggle translation direction. 'auto' enables per-segment detection."""
        self.context_manager.direction = direction
        if direction != "auto":
            self.audio_processor._last_detected_language = None
            self.audio_processor._language_votes.clear()
        logger.info(f"Direction set to {direction}")

    def set_mode(self, mode: str):
        """Switch between conversation and classroom mode."""
        self.context_manager.mode = mode
        logger.info(f"Mode set to {mode}")

    def _resolve_effective_direction(self, segment_detected_lang: str | None = None) -> tuple[str, str | None]:
        """Resolve effective direction for a segment.

        Returns (effective_direction, detected_language).
        In auto mode, uses per-segment detection (not voted consensus).
        Falls back to "es_to_en" when detection is unavailable.
        """
        if self.context_manager.direction != "auto":
            return self.context_manager.direction, None

        detected = segment_detected_lang
        if detected and detected in LANGUAGES:
            if detected != DEFAULT_TARGET_LANG:
                target = DEFAULT_TARGET_LANG
            else:
                # Speaker is speaking the target language — translate to
                # the first other supported language.
                others = [c for c in LANGUAGES if c != detected]
                target = others[0] if others else "es"
            return f"{detected}_to_{target}", detected
        else:
            return "es_to_en", detected  # fallback for unknown/None

    # ---------------------------------------------------------------------------
    # Session lifecycle convenience wrappers
    # ---------------------------------------------------------------------------

    async def create_session(self) -> int:
        """Create a new session record. Delegates to session_manager."""
        return await self.session_manager.create_session()

    async def close_session(self):
        """Close the current session. Delegates to session_manager."""
        await self.session_manager.close_session()

    async def generate_study_summary(self, session_id: int | None = None) -> str | None:
        """Generate study summary. Delegates to session_manager."""
        return await self.session_manager.generate_study_summary(session_id)

    async def reset_session(self) -> int:
        """Close current session, reset all state, start fresh. Coordinates submodules."""
        new_id = await self.session_manager.reset_session()
        self.audio_processor.reset_partial_state()
        return new_id

    async def reload_idiom_patterns(self):
        """Reload all idiom patterns (JSON files + DB). Called after new patterns are added."""
        self.context_manager.idiom_scanner.patterns.clear()
        idioms_dir = self.config.data_dir / "idioms"
        for json_file in idioms_dir.glob("*.json"):
            self.context_manager.idiom_scanner.load_from_json(json_file)
        await self.session_manager.load_db_idiom_patterns()
        logger.info(f"Reloaded {self.context_manager.idiom_scanner.count} idiom patterns")

    # ---------------------------------------------------------------------------
    # Idiom merge
    # ---------------------------------------------------------------------------

    def _merge_idioms(
        self, pattern_matches: list[IdiomMatch], llm_phrases: list[FlaggedPhrase]
    ) -> list[FlaggedPhrase]:
        """Merge pattern DB matches with LLM-detected phrases, deduplicating."""
        result = []
        seen: set[str] = set()

        # Pattern DB matches first (more curated)
        for m in pattern_matches:
            key = m.canonical.lower()
            if key not in seen:
                seen.add(key)
                self._metrics["idiom_pattern_db_hits"] += 1
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
                self._metrics["idiom_llm_hits"] += 1
                fp.source = "llm"
                result.append(fp)

        return result

    # ---------------------------------------------------------------------------
    # Text processing
    # ---------------------------------------------------------------------------

    async def process_text(
        self, text: str, speaker_id: str = "MANUAL", audio_path: str | None = None,
        effective_direction: str | None = None, detected_language: str | None = None,
    ) -> Exchange:
        """Process text directly (skip ASR). Useful for testing or typed input."""
        start_time = time.monotonic()

        # Resolve direction: use effective_direction if provided, else resolve from auto
        direction = effective_direction or self.context_manager.direction
        if direction == "auto":
            direction, detected_language = self._resolve_effective_direction()

        speaker = self.context_manager.speaker_tracker.record_utterance(speaker_id)
        speaker_name = self.context_manager.speaker_tracker.get_display_name(speaker_id)

        # Idiom scan
        idiom_matches = self.context_manager.idiom_scanner.scan(text)

        # Translate
        result = await self.context_manager.translator.translate(
            transcript=text,
            speaker_label=speaker_name,
            direction=direction,
            mode=self.context_manager.mode,
            context_exchanges=list(self.context_manager.recent_exchanges),
            topic_summary=self.context_manager.topic_summary,
        )

        # Merge idiom matches
        merged_phrases = self._merge_idioms(idiom_matches, result.flagged_phrases)

        # Track quality metrics
        if result.confidence < 0.3:
            self._metrics["low_confidence_count"] += 1
            logger.warning(
                f"Low confidence ({result.confidence:.2f}) translation: "
                f"src='{text[:100]}' -> '{result.translated[:100]}'"
            )
        if result.is_correction:
            self._metrics["corrections_detected"] += 1

        # Update speaker hint
        if result.speaker_hint and not speaker.custom_name:
            self.context_manager.speaker_tracker.set_role_hint(speaker_id, result.speaker_hint)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        exchange = Exchange(
            session_id=self.context_manager.session_id or 0,
            speaker=speaker,
            direction=direction,
            raw_transcript=text,
            corrected_source=result.corrected,
            translation=result.translated,
            flagged_phrases=merged_phrases,
            confidence=result.confidence,
            is_correction=result.is_correction,
            correction_detail=result.correction_detail,
            processing_ms=elapsed_ms,
            audio_path=audio_path,
            detected_language=detected_language,
        )

        # Update context
        self.context_manager.recent_exchanges.append({
            "speaker_label": speaker_name,
            "source": text,
            "corrected": result.corrected,
            "translated": result.translated,
        })

        # Persist exchange to database
        await self.session_manager.save_exchange(exchange)

        # Async topic summary update (fire and forget)
        asyncio.create_task(self.context_manager._update_topic(result, speaker_name))

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
                has_audio=bool(exchange.audio_path),
                detected_language=detected_language,
                effective_direction=direction,
                timestamp=datetime.now(UTC).isoformat(),
            )
            await self._on_translation(ws_msg)

        return exchange

    # ---------------------------------------------------------------------------
    # Partial audio processing
    # ---------------------------------------------------------------------------

    async def process_partial_audio(self, pcm_bytes: bytes, duration: float):
        """
        Run a fast ASR pass on a partial audio snapshot (while speech is ongoing).
        Sends partial transcript + quick translation to the client for live display.
        """
        if not self.audio_processor._whisperx_model:
            return

        async with self.audio_processor._partial_lock:
            wav_path = None
            try:
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
                transcript, partial_detected = await asyncio.to_thread(
                    self.audio_processor._run_quick_asr, wav_path
                )

                if not transcript or transcript.strip() == self.audio_processor._last_partial_text:
                    return  # no change, don't spam the client

                self.audio_processor._last_partial_text = transcript.strip()
                logger.info(f"Partial transcript: {self.audio_processor._last_partial_text}")

                # Resolve direction for partials
                effective_direction, _ = self._resolve_effective_direction(partial_detected)

                # Send partial source text immediately
                if self._on_partial:
                    await self._on_partial(WSPartialTranscript(
                        type="partial",
                        text=transcript.strip(),
                        speaker_id=None,
                        detected_language=partial_detected,
                    ))

                # Quick translate — lightweight LLM call for partial translation
                quick_translation = await self._quick_translate(
                    transcript.strip(), effective_direction=effective_direction
                )

                if quick_translation and self._on_partial:
                    await self._on_partial(WSPartialTranscript(
                        type="partial_translation",
                        text=quick_translation,
                        speaker_id=None,
                        detected_language=partial_detected,
                    ))

            except Exception as e:
                logger.warning(f"Partial processing error: {e}")
                if self._on_error:
                    try:
                        await self._on_error(f"Partial transcription error: {e}")
                    except Exception:
                        pass
            finally:
                if wav_path:
                    Path(wav_path).unlink(missing_ok=True)

    async def _quick_translate(self, text: str, effective_direction: str | None = None) -> str:
        """
        Lightweight translation for partials — shorter prompt, no idiom detection.
        Returns just the translated text, nothing else.
        """
        if not text:
            return ""

        direction = effective_direction or self.context_manager.direction
        if direction == "auto":
            direction = "es_to_en"  # fallback
        src_code, tgt_code = parse_direction(direction)
        src, tgt = lang_name(src_code), lang_name(tgt_code)

        try:
            model_override = self.config.translator.quick_model
            provider = self.config.translator.provider
            if provider == "lmstudio" and model_override:
                result = await self.context_manager.translator._call_provider(
                    provider, "", f"Translate this {src} to natural {tgt}. Output ONLY the translation, nothing else.\n\n{text}",
                    retries=0, max_tokens=256, temperature=0.2, json_mode=False,
                    lmstudio_model=model_override,
                )
            else:
                result = await self.context_manager.translator._call_llm(
                    "",
                    f"Translate this {src} to natural {tgt}. Output ONLY the translation, nothing else.\n\n{text}",
                    retries=0, max_tokens=256, temperature=0.2, json_mode=False,
                )
            return result.strip()
        except Exception as e:
            logger.debug(f"Quick translate error: {e}")
            return ""

    # ---------------------------------------------------------------------------
    # Queue and audio segment processing
    # ---------------------------------------------------------------------------

    async def process_wav(
        self, wav_path: str, audio_rms: float = 0.0, vad_avg_prob: float = 0.0,
        segment_id: int | None = None,
    ) -> Exchange | None:
        """
        Process a WAV file through the full pipeline (used by continuous mode).
        Routes through the queue for proper serialization and backpressure.
        """
        if not self._ready:
            logger.warning("Pipeline not ready, dropping audio")
            return None

        meta = {"audio_rms": audio_rms, "vad_avg_prob": vad_avg_prob, "segment_id": segment_id}
        future = asyncio.get_event_loop().create_future()
        await self._queue.put(("wav", wav_path, future, meta))
        return await future

    async def _process_queue(self):
        """Background worker that processes audio from the queue."""
        while True:
            try:
                item = await self._queue.get()
                kind, payload, future, *rest = item
                meta = rest[0] if rest else {}
                depth = self._queue.qsize() + 1
                if depth > self._metrics["peak_queue_depth"]:
                    self._metrics["peak_queue_depth"] = depth
                try:
                    if kind == "wav":
                        result = await self._process_audio_segment_from_wav(payload, meta=meta)
                    else:
                        result = await self._process_audio_segment(payload, meta=meta)
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
                    logger.error(f"Pipeline processing error: {e}", exc_info=True)
                    if self._on_error:
                        try:
                            await self._on_error(f"Processing failed: {e}")
                        except Exception:
                            pass
                    if not future.done():
                        future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}", exc_info=True)

    async def _process_audio_segment(self, audio_bytes: bytes, meta: dict | None = None) -> Exchange | None:
        """Run full pipeline on one audio segment (from raw Opus/WebM bytes)."""
        wav_path = await self.audio_processor._decode_to_wav(audio_bytes)
        if not wav_path:
            return None
        try:
            return await self._process_audio_segment_from_wav(wav_path, meta=meta)
        finally:
            Path(wav_path).unlink(missing_ok=True)

    async def _process_audio_segment_from_wav(self, wav_path: str, meta: dict | None = None) -> Exchange | None:
        """Run full pipeline on a WAV file (used by both PTT and continuous modes).

        ASR runs in the queue (GPU-serial). Once transcript is ready, translation
        is fired off concurrently so the queue can process the next segment's ASR
        without waiting for a slow LLM call.
        """
        start_time = time.monotonic()
        meta = meta or {}

        # Compute segment duration from WAV header (reads 44 bytes — zero cost)
        try:
            with wave.open(wav_path) as wf:
                duration_seconds = wf.getnframes() / wf.getframerate()
        except Exception:
            duration_seconds = 0.0

        # Normalize audio (bandpass + loudness) before ASR
        wav_path = await self.audio_processor._normalize_wav(wav_path)

        # Save audio clip before ASR (WAV gets deleted after processing)
        clip_path = self.context_manager._save_audio_clip(wav_path)

        # Run ASR + diarization in thread (blocking GPU ops)
        transcript, diarized_segments, segment_detected_lang, reject_reason = await asyncio.to_thread(
            self.audio_processor._run_asr_and_diarize, wav_path
        )

        if not transcript or not transcript.strip():
            if reject_reason == "no_segments":
                # WhisperX VAD found no speech — audio too quiet or non-speech
                logger.warning("WhisperX returned no speech segments — audio may be too quiet or non-speech")
                self._metrics["asr_empty_count"] += 1
                qm_status = "asr_empty"
            else:
                # ASR ran but produced garbage (noise markers, digits-only, etc.)
                logger.warning("ASR produced garbage output — background noise or non-speech audio")
                self._metrics["asr_rejected_count"] += 1
                qm_status = "asr_rejected"
            await self.session_manager.record_quality_metric(
                status=qm_status, confidence=None, meta=meta,
                processing_ms=int((time.monotonic() - start_time) * 1000),
                duration_seconds=duration_seconds,
            )
            if clip_path:
                Path(clip_path).unlink(missing_ok=True)
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

        # Resolve effective direction from per-segment detection
        effective_direction, detected_language = self._resolve_effective_direction(segment_detected_lang)

        # Send finalized source text immediately (before translation starts)
        if self._on_final_transcript:
            speaker = self.context_manager.speaker_tracker.get_or_create(speaker_id)
            await self._on_final_transcript({
                "type": "transcript_final",
                "text": transcript.strip(),
                "speaker": speaker.model_dump() if speaker else None,
                "direction": effective_direction,
                "detected_language": detected_language,
            })

        clipped_onset = _has_clipped_onset(transcript)
        if clipped_onset:
            logger.info(f"Segment #{meta.get('segment_id', '?')}: clipped onset detected in '{transcript[:40]}'")

        # Fire translation concurrently — don't block the ASR queue waiting for LLM
        task = asyncio.create_task(
            self._translate_and_notify(
                transcript, speaker_id, start_time,
                audio_path=clip_path, meta=meta,
                duration_seconds=duration_seconds,
                clipped_onset=clipped_onset,
                effective_direction=effective_direction,
                detected_language=detected_language,
            )
        )
        self._inflight_translations.add(task)
        task.add_done_callback(self._inflight_translations.discard)

        return None  # Translation result sent via callback, not via future

    async def _translate_and_notify(
        self, transcript: str, speaker_id: str, start_time: float,
        audio_path: str | None = None, meta: dict | None = None,
        duration_seconds: float = 0.0, clipped_onset: bool = False,
        effective_direction: str | None = None, detected_language: str | None = None,
    ):
        """Run translation outside the queue so ASR can continue processing."""
        try:
            exchange = await self.process_text(
                transcript, speaker_id, audio_path=audio_path,
                effective_direction=effective_direction,
                detected_language=detected_language,
            )
            if exchange:
                exchange.processing_ms = int((time.monotonic() - start_time) * 1000)
                logger.info(f"Translation done in {exchange.processing_ms}ms: '{transcript[:60]}'")
                status = "low_confidence" if exchange.confidence < 0.3 else "ok"
                await self.session_manager.record_quality_metric(
                    status=status,
                    confidence=exchange.confidence,
                    meta=meta,
                    processing_ms=exchange.processing_ms,
                    speaker_id=speaker_id,
                    duration_seconds=duration_seconds,
                    clipped_onset=clipped_onset,
                )
        except Exception as e:
            logger.error(f"Translation error: {e}")
            if self._on_error:
                try:
                    await self._on_error(f"Translation failed: {e}")
                except Exception:
                    pass
