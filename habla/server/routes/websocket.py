"""
WebSocket handler for real-time audio translation.

Supports two modes:
  1. CONTINUOUS (default, like Google Translate conversation mode):
     Client starts streaming, server VAD detects utterance boundaries,
     translates each automatically. No user interaction after pressing Start.

  2. PUSH-TO-TALK (fallback):
     Client signals segment start/end manually.

Protocol:
  Client → Server:
    Binary frames:  Opus/WebM audio chunks (continuous stream)
    Text frames:    JSON control messages

  Server → Client:
    Text frames:    JSON (partials, translations, speaker updates, status)
"""

import asyncio
import json
import logging
import tempfile
import time
import struct
from pathlib import Path
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from server import shared_assets
from server.db.database import get_db
from server.pipeline.orchestrator import PipelineOrchestrator
from server.pipeline.vad_buffer import StreamingVADBuffer, AudioDecoder, VADConfig
from server.models.schemas import WSTranslation, WSPartialTranscript, WSSpeakersUpdate
from server.services.audio_recorder import AudioRecorder
from server.services.lmstudio_manager import LMStudioManager
from server.config import RecordingConfig
from server.services.tts import TTSService

logger = logging.getLogger("habla.ws")

# Segment merge timing (shared/asr_profile.json "merge"): wait this long after
# the last VAD segment before flushing to the pipeline, so consecutive short
# segments merge into one.
MERGE_GAP_SECONDS = shared_assets.asr_profile()["merge"]["gap_seconds"]
# Maximum pending audio before forced flush (prevents unbounded accumulation).
MAX_PENDING_SECONDS = shared_assets.asr_profile()["merge"]["max_pending_seconds"]

# Valid values for direction and mode control messages
VALID_DIRECTIONS = ("es_to_en", "en_to_es", "auto")
VALID_MODES = ("conversation", "classroom")

# Only one client at a time — second connection gets rejected.

# Module-level pipeline reference — updated on restart so active sessions
# always use the current pipeline instead of a stale (closed) one.
_current_pipeline: PipelineOrchestrator | None = None
_recording_config: RecordingConfig | None = None
_ws_config = None  # WebSocketConfig, set from main.py

# Active session reference — allows REST API (playback) to access the session.
_active_session: Optional['ClientSession'] = None

# Module-level playback service reference — set from main.py, used for cleanup on disconnect.
_playback_service = None

# Module-level TTS service reference — set from main.py.
_tts_service: TTSService | None = None


def set_ws_playback_service(service):
    """Called from main.py so cleanup can stop active playback on disconnect."""
    global _playback_service
    _playback_service = service


def set_ws_tts_service(service: TTSService | None) -> None:
    """Called from main.py to supply the TTS service (or None if TTS is disabled)."""
    global _tts_service
    _tts_service = service


def set_ws_pipeline(pipeline: PipelineOrchestrator):
    """Called from main.py on startup and after restart."""
    global _current_pipeline
    _current_pipeline = pipeline


def set_ws_config(config):
    """Set WebSocket configuration (heartbeat intervals)."""
    global _ws_config
    _ws_config = config


def set_recording_config(config: RecordingConfig):
    """Set recording configuration from main app config."""
    global _recording_config
    logger.info(f"[RECORDING CONFIG] Setting recording config: enabled={config.enabled}, output_dir={config.output_dir}")
    _recording_config = config
    logger.info(f"[RECORDING CONFIG] Global _recording_config updated: enabled={_recording_config.enabled}")


def get_active_session() -> Optional['ClientSession']:
    """Return the active client session (if any). Used by playback REST API."""
    return _active_session


class ClientSession:
    """Manages state for one connected WebSocket client.

    Lifecycle: created per WebSocket connection in websocket_endpoint().
    Owns decoder (ffmpeg subprocess), VAD buffer (Silero), and optional audio recorder.
    Registers callbacks on the shared pipeline for receiving translations/partials.
    Stored as module-global _active_session (single-client design).

    Cleanup: cleanup() stops listening/recording; pipeline.close_session() persists to DB.
    The pipeline reference comes from module-global _current_pipeline (set by main.py).
    """

    def __init__(self, websocket: WebSocket, pipeline: PipelineOrchestrator):
        self.ws = websocket
        self.pipeline = pipeline
        self.id = id(websocket)
        self.session_id = str(self.id)

        # Audio processing — derive VAD config from pipeline's app config
        audio_cfg = pipeline.config.audio
        asr_cfg = pipeline.config.asr
        self.decoder = AudioDecoder(sample_rate=audio_cfg.sample_rate)
        self.vad = StreamingVADBuffer(
            config=VADConfig(
                sample_rate=audio_cfg.sample_rate,
                silence_duration_ms=audio_cfg.vad_silence_ms,
                min_speech_ms=audio_cfg.min_speech_ms,
                max_segment_seconds=audio_cfg.max_segment_seconds,
                speech_threshold=asr_cfg.vad_threshold,
                frame_ms=audio_cfg.vad_frame_ms,
                pre_speech_padding_ms=audio_cfg.pre_speech_padding_ms,
                calibration_duration_ms=audio_cfg.calibration_duration_ms,
            ),
            on_segment=self._on_speech_segment,
        )

        # Audio recording (created when listening starts if enabled)
        self.recorder: Optional[AudioRecorder] = None

        # State
        self.listening = False
        self.playback_mode = False
        self._segment_counter = 0
        self.tts_enabled: bool = pipeline.config.session.tts_enabled

        # Cleanup guard
        self._cleaned_up = False

        # Heartbeat monitoring
        self._last_activity_time = time.monotonic()
        self._heartbeat_task: asyncio.Task | None = None

        # Background decode loop
        self._decode_task: asyncio.Task | None = None
        self._chunk_event = asyncio.Event()
        self._chunk_inbox: list[bytes] = []
        self._pending_pcm = bytearray()
        self._pending_duration = 0.0
        self._pending_task: asyncio.Task | None = None
        self._pending_rms_sum = 0.0  # weighted RMS accumulator for merged segments
        self._pending_prob_sum = 0.0  # weighted VAD prob accumulator for merged segments
        self._pending_weight = 0.0  # total duration weight for averaging
        self._merge_gap_seconds = MERGE_GAP_SECONDS
        self._max_pending_seconds = MAX_PENDING_SECONDS

    async def initialize(self):
        """Load VAD model."""
        await self.vad.initialize()

    async def _link_recording_to_session(self):
        """Write recording directory name to the DB session row."""
        if not self.recorder or not self.pipeline.context_manager.session_id:
            return
        try:
            db = await get_db()
            await db.execute(
                "UPDATE sessions SET recording_dir = ? WHERE id = ?",
                (self.recorder.session_dir.name, self.pipeline.context_manager.session_id),
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to link recording to session: {e}")

    async def start_listening(self):
        """Begin continuous listening mode."""
        self.listening = True
        self.vad.reset()
        self.decoder.reset()

        # Create recorder if recording is enabled (check at start time, not connection time)
        logger.info(f"Recording check: _recording_config={_recording_config}, enabled={_recording_config.enabled if _recording_config else None}, self.recorder={self.recorder}")
        if _recording_config and _recording_config.enabled and not self.recorder:
            logger.info(f"Creating AudioRecorder for session {self.session_id}")
            self.recorder = AudioRecorder(_recording_config, self.session_id)
            logger.info(f"AudioRecorder created: {self.recorder}, session_dir={self.recorder.session_dir}")
        else:
            if not _recording_config:
                logger.warning("No _recording_config set!")
            elif not _recording_config.enabled:
                logger.info("Recording disabled in config")
            elif self.recorder:
                logger.info("Recorder already exists")

        # Start audio recording if enabled
        if self.recorder:
            logger.info(f"Starting recording for session {self.session_id}")
            self.recorder.start_recording()
            logger.info(f"Recording started, raw_file={self.recorder.raw_file}")
            await self._link_recording_to_session()

        # Re-wire callbacks to current pipeline (handles restart mid-session)
        if hasattr(self, '_wire_callbacks'):
            self._wire_callbacks()

        # Wire up partial audio callback for streaming transcription
        self.vad.on_partial_audio = self._on_partial_audio
        # Wire noise floor calibration callback
        self.vad.on_calibration_complete = self._on_calibration_complete

        # Start streaming ffmpeg and background decode loop
        await self.decoder.start_streaming()
        self._decode_task = asyncio.create_task(self._continuous_decode_loop())

        await self._send({
            "type": "listening_started",
            "mode": "continuous",
            "recording": self.recorder is not None,
        })
        logger.info(f"Client {self.id}: continuous listening started (recording: {self.recorder is not None})")

    async def _on_partial_audio(self, pcm_bytes: bytes, duration: float):
        """Called by VAD every ~1s during speech — triggers streaming partial transcription."""
        asyncio.create_task(
            self.pipeline.process_partial_audio(pcm_bytes, duration)
        )

    async def _on_calibration_complete(self, noise_floor_rms: float):
        """Called by VAD after noise floor calibration finishes."""
        if noise_floor_rms < 200:
            level = "quiet"
        elif noise_floor_rms < 800:
            level = "moderate"
        else:
            level = "loud"
        logger.info(f"Noise floor calibration: RMS={noise_floor_rms:.0f}, level={level}")
        await self._send({
            "type": "noise_calibration",
            "noise_floor_rms": round(noise_floor_rms, 1),
            "noise_level": level,
        })
        if level == "loud":
            await self._send({
                "type": "error",
                "message": f"Loud environment detected (RMS {noise_floor_rms:.0f}) — speech detection threshold raised automatically",
            })

    async def stop_listening(self):
        """Stop continuous listening, flush remaining audio."""
        self.listening = False

        if self._decode_task:
            self._decode_task.cancel()
            try:
                await self._decode_task
            except asyncio.CancelledError:
                pass

        # Flush remaining PCM from the streaming ffmpeg process
        try:
            remaining_pcm = await self.decoder.stop_streaming()
            if remaining_pcm:
                logger.info(f"Final stream flush produced {len(remaining_pcm)} bytes PCM")
                await self.vad.feed_pcm(remaining_pcm)
        except Exception as e:
            logger.warning(f"Final stream flush error: {e}")

        await self.vad.flush()
        await self._flush_pending_now()

        # Stop audio recording
        if self.recorder:
            self.recorder.stop_recording()

        await self._send({
            "type": "listening_stopped",
            "segments_processed": self.vad.segments_emitted,
            "total_speech_seconds": round(self.vad.total_speech_seconds, 1),
        })
        logger.info(f"Client {self.id}: stopped, {self.vad.segments_emitted} segments")

    async def enable_recording(self):
        """Enable recording for current session (can be called mid-session)."""
        if self.recorder:
            logger.info("Recording already enabled for this session")
            return

        if not _recording_config or not _recording_config.enabled:
            logger.warning("Cannot enable recording: global recording config is disabled")
            return

        logger.info(f"Dynamically enabling recording for session {self.session_id}")
        self.recorder = AudioRecorder(_recording_config, self.session_id)
        self.recorder.start_recording()
        logger.info(f"Recording enabled: session_dir={self.recorder.session_dir}")
        await self._link_recording_to_session()

        await self._send({
            "type": "recording_enabled",
            "session_dir": str(self.recorder.session_dir),
        })

    async def disable_recording(self):
        """Disable recording for current session."""
        if not self.recorder:
            logger.info("Recording already disabled")
            return

        logger.info(f"Dynamically disabling recording for session {self.session_id}")
        self.recorder.stop_recording()
        self.recorder = None

        await self._send({"type": "recording_disabled"})

    async def handle_audio_chunk(self, chunk: bytes):
        """Handle incoming audio chunk from client or playback stream."""
        if not self.listening and not self.playback_mode:
            return

        # Save raw audio chunk if recording
        if self.recorder:
            self.recorder.write_raw_chunk(chunk)
            logger.debug(f"Wrote {len(chunk)} bytes to recording")
        else:
            logger.debug(f"No recorder active, not saving chunk")

        self._chunk_inbox.append(chunk)
        self._chunk_event.set()
        logger.debug(f"Audio chunk received: {len(chunk)} bytes")

    async def _continuous_decode_loop(self):
        """
        Feeds WebM chunks to the streaming ffmpeg process and routes
        decoded PCM to the VAD. O(1) per chunk — no re-decode.
        """
        FEED_INTERVAL = 0.5  # Process chunks every 0.5s
        empty_cycles = 0

        try:
            while self.listening or self.playback_mode:
                # Wait for new chunks or timeout
                try:
                    await asyncio.wait_for(self._chunk_event.wait(), timeout=FEED_INTERVAL)
                except asyncio.TimeoutError:
                    pass
                self._chunk_event.clear()

                # Grab all pending chunks
                chunks = self._chunk_inbox[:]
                self._chunk_inbox.clear()

                if not chunks:
                    continue

                # Feed each chunk to streaming ffmpeg
                for chunk in chunks:
                    pcm = await self.decoder.feed_chunk(chunk)
                    if self.decoder.is_dead:
                        await self._send({"type": "error", "message": "Audio decoder crashed — restart listening to recover"})
                        return
                    if pcm:
                        empty_cycles = 0
                        logger.debug(f"Decode produced {len(pcm)} bytes PCM")
                        await self.vad.feed_pcm(pcm)
                    else:
                        empty_cycles += 1
                        if empty_cycles >= 6:
                            logger.warning("ffmpeg producing no PCM for multiple cycles")
                            empty_cycles = 0

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Decode loop error: {e}")

    async def _on_speech_segment(self, pcm_bytes: bytes, duration: float,
                                  audio_rms: float = 0.0, vad_avg_prob: float = 0.0):
        """Callback from VAD - queues segment for merge before translation."""
        self.pipeline.reset_partial_state()
        await self._queue_segment(pcm_bytes, duration, audio_rms, vad_avg_prob)

    async def _queue_segment(self, pcm_bytes: bytes, duration: float,
                              audio_rms: float = 0.0, vad_avg_prob: float = 0.0):
        if not pcm_bytes:
            return
        self._pending_pcm.extend(pcm_bytes)
        self._pending_duration += duration
        # Accumulate weighted audio metrics for merged segments
        self._pending_rms_sum += audio_rms * duration
        self._pending_prob_sum += vad_avg_prob * duration
        self._pending_weight += duration
        if self._pending_duration >= self._max_pending_seconds:
            await self._flush_pending_now()
            return
        if self._pending_task:
            self._pending_task.cancel()
        self._pending_task = asyncio.create_task(self._flush_pending_after_gap())

    async def _flush_pending_after_gap(self):
        try:
            await asyncio.sleep(self._merge_gap_seconds)
            # Clear self-reference BEFORE calling flush, otherwise
            # _flush_pending_now cancels us (the current running task).
            self._pending_task = None
            await self._flush_pending_now()
        except asyncio.CancelledError:
            pass

    async def _flush_pending_now(self):
        if not self._pending_pcm:
            return
        if self._pending_task:
            self._pending_task.cancel()
            self._pending_task = None
        pcm_bytes = bytes(self._pending_pcm)
        duration = self._pending_duration
        # Compute duration-weighted averages for merged segments
        audio_rms = self._pending_rms_sum / self._pending_weight if self._pending_weight > 0 else 0.0
        vad_avg_prob = self._pending_prob_sum / self._pending_weight if self._pending_weight > 0 else 0.0
        self._pending_pcm.clear()
        self._pending_duration = 0.0
        self._pending_rms_sum = 0.0
        self._pending_prob_sum = 0.0
        self._pending_weight = 0.0
        await self._process_segment(pcm_bytes, duration, audio_rms, vad_avg_prob)

    async def _process_segment(self, pcm_bytes: bytes, duration: float,
                                audio_rms: float = 0.0, vad_avg_prob: float = 0.0):
        """Runs full translation pipeline on the merged segment via the orchestrator queue."""
        self._segment_counter += 1
        seg = self._segment_counter

        logger.info(f"Segment #{seg} ({duration:.1f}s, RMS={audio_rms:.0f}, VAD={vad_avg_prob:.2f})")

        # Save segment if recording enabled
        if self.recorder:
            self.recorder.save_pcm_segment(pcm_bytes, metadata={
                "segment_id": seg,
                "duration_seconds": duration,
                "audio_rms": round(audio_rms, 1),
                "vad_avg_prob": round(vad_avg_prob, 3),
            })
        else:
            logger.info(f"No recorder active, segment #{seg} not saved")

        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix=f"habla_{seg}_"
            ) as f:
                wav_path = f.name
                _write_wav(f, pcm_bytes, sample_rate=16000)

            start = time.monotonic()
            exchange = await self.pipeline.process_wav(
                wav_path, audio_rms=audio_rms, vad_avg_prob=vad_avg_prob,
                segment_id=seg,
            )
            elapsed = time.monotonic() - start

            if exchange:
                logger.info(
                    f"#{seg} done in {elapsed:.1f}s: "
                    f"'{exchange.raw_transcript[:60]}'"
                )

                # Add translation metadata to recording
                if self.recorder:
                    self.recorder.add_segment_metadata(seg, "raw_transcript", exchange.raw_transcript)
                    self.recorder.add_segment_metadata(seg, "translation", exchange.translated_text)
                    self.recorder.add_segment_metadata(seg, "speaker", exchange.speaker_label)
                    self.recorder.add_segment_metadata(seg, "confidence", exchange.confidence)
                    self.recorder.add_segment_metadata(seg, "idioms_detected", len(exchange.idioms))
            else:
                # ASR produced nothing — enrich recording with failure info
                if self.recorder:
                    self.recorder.add_segment_metadata(seg, "asr_status", "rejected")

        except Exception as e:
            logger.error(f"Segment #{seg} error: {e}")
            await self._send({"type": "error", "message": str(e)})
        finally:
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

    async def _send(self, data: dict):
        try:
            await self.ws.send_text(json.dumps(data))
        except Exception as e:
            logger.debug(f"WebSocket send failed (client may have disconnected): {e}")

    async def _send_tts_audio(self, wav_bytes: bytes, exchange_id: int | None = None) -> None:
        """Send TTS audio to client: JSON announcement then binary WAV frame."""
        try:
            await self.ws.send_text(json.dumps({
                "type": "tts_audio_start",
                "exchange_id": exchange_id,
                "size_bytes": len(wav_bytes),
            }))
            await self.ws.send_bytes(wav_bytes)
        except Exception as e:
            logger.debug("TTS audio send failed: %s", e)

    async def _heartbeat_monitor(self, interval: float, max_missed: int):
        """Monitor client liveness. Close connection if no activity for too long."""
        timeout = interval * max_missed
        try:
            while True:
                await asyncio.sleep(interval)
                elapsed = time.monotonic() - self._last_activity_time
                if elapsed > timeout:
                    logger.warning(
                        f"Client {self.id}: no activity for {elapsed:.0f}s "
                        f"(>{timeout:.0f}s), closing stale connection"
                    )
                    try:
                        await self.ws.close(code=1000, reason="Heartbeat timeout")
                    except Exception:
                        pass
                    break
        except asyncio.CancelledError:
            pass

    async def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        self.playback_mode = False
        if self.listening:
            await self.stop_listening()
        # Stop active playback if running (prevents orphaned playback task after disconnect)
        if _playback_service and _playback_service.is_active:
            try:
                await _playback_service.stop_playback()
            except Exception as e:
                logger.warning(f"Playback stop during cleanup failed: {e}")

        # Post-cleanup invariant checks (warn, don't raise)
        if self.listening:
            logger.warning(f"Client {self.id}: cleanup finished but listening still True")
        if self.playback_mode:
            logger.warning(f"Client {self.id}: cleanup finished but playback_mode still True")
        if self._decode_task and not self._decode_task.done():
            logger.warning(f"Client {self.id}: cleanup finished but decode_task still running")
        if self._pending_task and not self._pending_task.done():
            logger.warning(f"Client {self.id}: cleanup finished but pending_task still running")


async def _dispatch_tts(session: 'ClientSession', msg: WSTranslation) -> None:
    """Synthesize translation to audio and send to client as a binary frame."""
    direction = msg.effective_direction or session.pipeline.context_manager.direction
    if direction == "auto":
        direction = "es_to_en"  # fallback
    target_lang = "en" if direction.endswith("_to_en") else "es"
    wav = await _tts_service.synthesize(msg.translated, target_lang)
    if wav:
        await session._send_tts_audio(wav, msg.exchange_id)


def _write_wav(f, pcm_bytes: bytes, sample_rate: int = 16000):
    """Write minimal WAV header + PCM data."""
    data_size = len(pcm_bytes)
    f.write(b'RIFF')
    f.write(struct.pack('<I', 36 + data_size))
    f.write(b'WAVE')
    f.write(b'fmt ')
    f.write(struct.pack('<I', 16))
    f.write(struct.pack('<HHIIHH', 1, 1, sample_rate,
                        sample_rate * 2, 2, 16))
    f.write(b'data')
    f.write(struct.pack('<I', data_size))
    f.write(pcm_bytes)


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket handler. Only one client at a time.

    New connections immediately replace old ones (single-user app).
    """
    global _active_session
    await websocket.accept()

    # Close any stale session so the new one takes over immediately
    old = _active_session
    if old is not None:
        logger.info("Replacing previous session for new connection")
        _active_session = None
        try:
            await old.cleanup()
        except Exception:
            pass
        try:
            await old.websocket.close()
        except Exception:
            pass

    session = ClientSession(websocket, _current_pipeline)
    _active_session = session
    db_session_created = False

    try:
        await session.initialize()
        logger.info(f"Client connected: {session.id}")

        # Create a DB session for this connection
        await session.pipeline.create_session()
        db_session_created = True

        # Define callbacks (wired to current pipeline on connect and start_listening)
        async def on_translation(msg: WSTranslation):
            await session._send(msg.model_dump())
            if session.tts_enabled and _tts_service:
                asyncio.create_task(_dispatch_tts(session, msg))

        async def on_partial(msg: WSPartialTranscript):
            await session._send(msg.model_dump())

        async def on_speakers(msg: WSSpeakersUpdate):
            await session._send(msg.model_dump())

        async def on_final_transcript(msg: dict):
            await session._send(msg)

        async def on_error(msg: str):
            await session._send({"type": "error", "message": msg})

        session._wire_callbacks = lambda: session.pipeline.set_callbacks(
            on_translation=on_translation,
            on_partial=on_partial,
            on_speakers=on_speakers,
            on_final_transcript=on_final_transcript,
            on_error=on_error,
        )
        session._wire_callbacks()

        # Start heartbeat monitor
        if _ws_config:
            session._heartbeat_task = asyncio.create_task(
                session._heartbeat_monitor(
                    _ws_config.ping_interval_seconds,
                    _ws_config.missed_pings_threshold,
                )
            )

        p = session.pipeline
        ctx = p.context_manager
        await session._send({
            "type": "status",
            "pipeline_ready": p.ready,
            "session_id": ctx.session_id,
            "direction": ctx.direction,
            "mode": ctx.mode,
            "speaker_count": len(ctx.speaker_tracker.speakers),
        })

        while True:
            message = await websocket.receive()
            session._last_activity_time = time.monotonic()
            if message.get("type") == "websocket.disconnect":
                logger.info(f"Client {session.id}: disconnect received")
                break

            if "bytes" in message and message["bytes"]:
                await session.handle_audio_chunk(message["bytes"])

            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning(f"Client {session.id}: malformed JSON in text frame")
                    await session._send({"type": "error", "message": "Invalid JSON"})
                    continue

                t = msg.get("type", "")

                if t == "start_listening":
                    logger.info(f"Client {session.id}: start_listening received")
                    await session.start_listening()

                elif t == "stop_listening":
                    logger.info(f"Client {session.id}: stop_listening received")
                    await session.stop_listening()

                elif t == "segment_end":
                    # Legacy push-to-talk: process accumulated audio
                    pass  # continuous mode handles this via VAD

                elif t == "text_input":
                    text = msg.get("text", "").strip()
                    spk = msg.get("speaker_id", "MANUAL")
                    logger.info(f"Text input received (len={len(text)} spk={spk})")
                    if text:
                        asyncio.create_task(
                            _process_text(session, text, spk)
                        )

                elif t == "toggle_direction":
                    d = msg.get("direction", "es_to_en")
                    if d not in VALID_DIRECTIONS:
                        logger.warning(f"Invalid direction '{d}', ignoring")
                        await session._send({"type": "error", "message": f"Invalid direction: {d}"})
                    else:
                        session.pipeline.set_direction(d)
                        await session._send({"type": "direction_changed", "direction": d})
                        # Persist so it survives server restarts
                        _persist_ui_setting(session, "direction", d)

                elif t == "set_mode":
                    m = msg.get("mode", "conversation")
                    if m not in VALID_MODES:
                        logger.warning(f"Invalid mode '{m}', ignoring")
                        await session._send({"type": "error", "message": f"Invalid mode: {m}"})
                    else:
                        session.pipeline.set_mode(m)
                        await session._send({"type": "mode_changed", "mode": m})
                        # Persist so it survives server restarts
                        _persist_ui_setting(session, "mode", m)

                elif t == "rename_speaker":
                    sid = msg.get("speaker_id", "")
                    name = msg.get("name", "")
                    if sid and name:
                        session.pipeline.context_manager.speaker_tracker.rename(sid, name)
                        await session._send(
                            WSSpeakersUpdate(
                                speakers=session.pipeline.context_manager.speaker_tracker.get_all()
                            ).model_dump()
                        )

                elif t == "new_session":
                    logger.info(f"Client {session.id}: new_session requested")
                    if session.listening:
                        await session.stop_listening()
                    # Detach recorder so next start_listening creates a fresh one
                    if session.recorder:
                        session.recorder.stop_recording()
                        session.recorder = None
                    new_id = await session.pipeline.reset_session()
                    if hasattr(session, '_wire_callbacks'):
                        session._wire_callbacks()
                    await session._send({
                        "type": "session_reset",
                        "session_id": new_id,
                        "direction": session.pipeline.context_manager.direction,
                        "mode": session.pipeline.context_manager.mode,
                    })

                elif t == "set_tts":
                    enabled = bool(msg.get("enabled", False))
                    session.tts_enabled = enabled
                    if enabled and _tts_service:
                        voice_es = msg.get("voice_es", "")
                        voice_en = msg.get("voice_en", "")
                        if voice_es or voice_en:
                            _tts_service.set_voices(
                                voice_es or _tts_service._voice_es,
                                voice_en or _tts_service._voice_en,
                            )
                    await session._send({"type": "tts_changed", "enabled": enabled})

                elif t == "ping":
                    await session._send({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session.id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        _active_session = None
        await session.cleanup()
        if db_session_created:
            await session.pipeline.close_session()
            # Fire-and-forget: generate study summary in background after session closes.
            # Skips automatically if the session has fewer than 3 exchanges.
            asyncio.create_task(session.pipeline.generate_study_summary())


def _persist_ui_setting(session, key: str, value: str):
    """Save a UI setting (direction, mode) to llm_settings.json."""
    try:
        data_dir = session.pipeline.config.data_dir
        settings = LMStudioManager.load_settings(data_dir)
        settings[key] = value
        LMStudioManager.save_settings(data_dir, settings)
    except Exception as exc:
        logger.warning("Could not persist %s=%s: %s", key, value, exc)


async def _process_text(session, text, speaker_id):
    try:
        await session.pipeline.process_text(text, speaker_id)
    except Exception as e:
        logger.error(f"Text error: {e}")
        await session._send({"type": "error", "message": str(e)})
