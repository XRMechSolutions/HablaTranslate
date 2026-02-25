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

from server.pipeline.orchestrator import PipelineOrchestrator
from server.pipeline.vad_buffer import StreamingVADBuffer, AudioDecoder, VADConfig
from server.models.schemas import WSTranslation, WSPartialTranscript, WSSpeakersUpdate
from server.services.audio_recorder import AudioRecorder
from server.config import RecordingConfig

logger = logging.getLogger("habla.ws")

# Segment merge timing: wait this long after last VAD segment before flushing
# to the pipeline. Allows consecutive short segments to merge into one.
MERGE_GAP_SECONDS = 1.2
# Maximum pending audio before forced flush (prevents unbounded accumulation).
MAX_PENDING_SECONDS = 60.0

# Valid values for direction and mode control messages
VALID_DIRECTIONS = ("es_to_en", "en_to_es")
VALID_MODES = ("conversation", "classroom")

# Only one client at a time — second connection gets rejected.
_active_ws_lock = asyncio.Lock()

# Module-level pipeline reference — updated on restart so active sessions
# always use the current pipeline instead of a stale (closed) one.
_current_pipeline: PipelineOrchestrator | None = None
_recording_config: RecordingConfig | None = None

# Active session reference — allows REST API (playback) to access the session.
_active_session: Optional['ClientSession'] = None

# Module-level playback service reference — set from main.py, used for cleanup on disconnect.
_playback_service = None


def set_ws_playback_service(service):
    """Called from main.py so cleanup can stop active playback on disconnect."""
    global _playback_service
    _playback_service = service


def set_ws_pipeline(pipeline: PipelineOrchestrator):
    """Called from main.py on startup and after restart."""
    global _current_pipeline
    _current_pipeline = pipeline


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

        # Audio processing
        self.decoder = AudioDecoder(sample_rate=16000)
        self.vad = StreamingVADBuffer(
            config=VADConfig(
                silence_duration_ms=600,
                min_speech_ms=200,
                max_segment_seconds=30.0,
                speech_threshold=0.35,
                frame_ms=32,
                pre_speech_padding_ms=300,
            ),
            on_segment=self._on_speech_segment,
        )

        # Audio recording (created when listening starts if enabled)
        self.recorder: Optional[AudioRecorder] = None

        # State
        self.listening = False
        self.playback_mode = False
        self._segment_counter = 0

        # Cleanup guard
        self._cleaned_up = False

        # Background decode loop
        self._decode_task: asyncio.Task | None = None
        self._chunk_event = asyncio.Event()
        self._chunk_inbox: list[bytes] = []
        self._pending_pcm = bytearray()
        self._pending_duration = 0.0
        self._pending_task: asyncio.Task | None = None
        self._merge_gap_seconds = MERGE_GAP_SECONDS
        self._max_pending_seconds = MAX_PENDING_SECONDS

    async def initialize(self):
        """Load VAD model."""
        await self.vad.initialize()

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

        # Re-wire callbacks to current pipeline (handles restart mid-session)
        if hasattr(self, '_wire_callbacks'):
            self._wire_callbacks()

        # Wire up partial audio callback for streaming transcription
        self.vad.on_partial_audio = self._on_partial_audio

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

    async def _on_speech_segment(self, pcm_bytes: bytes, duration: float):
        """Callback from VAD - queues segment for merge before translation."""
        self.pipeline.reset_partial_state()
        await self._queue_segment(pcm_bytes, duration)

    async def _queue_segment(self, pcm_bytes: bytes, duration: float):
        if not pcm_bytes:
            return
        self._pending_pcm.extend(pcm_bytes)
        self._pending_duration += duration
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
        self._pending_pcm.clear()
        self._pending_duration = 0.0
        await self._process_segment(pcm_bytes, duration)

    async def _process_segment(self, pcm_bytes: bytes, duration: float):
        """Runs full translation pipeline on the merged segment via the orchestrator queue."""
        self._segment_counter += 1
        seg = self._segment_counter

        logger.info(f"Segment #{seg} ({duration:.1f}s)")

        # Save segment if recording enabled
        if self.recorder:
            logger.info(f"Saving segment #{seg} to recording ({len(pcm_bytes)} bytes PCM)")
            self.recorder.save_pcm_segment(pcm_bytes, metadata={
                "segment_id": seg,
                "duration_seconds": duration,
            })
            logger.info(f"Segment #{seg} saved to recording")
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
            exchange = await self.pipeline.process_wav(wav_path)
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

    async def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
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

    Uses a lock so reconnecting clients wait for the previous connection's
    cleanup to finish rather than being rejected during the brief teardown window.
    A second tab will wait until the first disconnects (intended single-client behavior).
    """
    await websocket.accept()

    async with _active_ws_lock:
        global _active_session
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

            p = session.pipeline
            await session._send({
                "type": "status",
                "pipeline_ready": p.ready,
                "session_id": p.session_id,
                "direction": p.direction,
                "mode": p.mode,
                "speaker_count": len(p.speaker_tracker.speakers),
            })

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    logger.info(f"Client {session.id}: disconnect received")
                    break

                if "bytes" in message and message["bytes"]:
                    await session.handle_audio_chunk(message["bytes"])

                elif "text" in message and message["text"]:
                    try:
                        msg = json.loads(message["text"])
                    except json.JSONDecodeError:
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

                    elif t == "set_mode":
                        m = msg.get("mode", "conversation")
                        if m not in VALID_MODES:
                            logger.warning(f"Invalid mode '{m}', ignoring")
                            await session._send({"type": "error", "message": f"Invalid mode: {m}"})
                        else:
                            session.pipeline.set_mode(m)
                            await session._send({"type": "mode_changed", "mode": m})

                    elif t == "rename_speaker":
                        sid = msg.get("speaker_id", "")
                        name = msg.get("name", "")
                        if sid and name:
                            session.pipeline.speaker_tracker.rename(sid, name)
                            await session._send(
                                WSSpeakersUpdate(
                                    speakers=session.pipeline.speaker_tracker.get_all()
                                ).model_dump()
                            )

                    elif t == "new_session":
                        logger.info(f"Client {session.id}: new_session requested")
                        if session.listening:
                            await session.stop_listening()
                        new_id = await session.pipeline.reset_session()
                        if hasattr(session, '_wire_callbacks'):
                            session._wire_callbacks()
                        await session._send({
                            "type": "session_reset",
                            "session_id": new_id,
                            "direction": session.pipeline.direction,
                            "mode": session.pipeline.mode,
                        })

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


async def _process_text(session, text, speaker_id):
    try:
        await session.pipeline.process_text(text, speaker_id)
    except Exception as e:
        logger.error(f"Text error: {e}")
        await session._send({"type": "error", "message": str(e)})
