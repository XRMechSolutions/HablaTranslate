"""
Server-side Voice Activity Detection and audio segmentation.

Receives a continuous stream of Opus/WebM audio chunks from the client,
decodes them, runs Silero VAD to detect speech boundaries, and emits
complete utterance segments for the ASR pipeline.

This is the key piece that makes always-on listening work like Google Translate.
"""

import asyncio
import logging
import tempfile
import struct
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from server import shared_assets

logger = logging.getLogger("habla.vad")


_PROFILE = shared_assets.asr_profile()
_CAL = _PROFILE["calibration"]
_SEG = _PROFILE["segmenter"]
_VAD = _PROFILE["vad"]


@dataclass
class VADConfig:
    """VAD tuning parameters."""
    sample_rate: int = _PROFILE["audio"]["sample_rate"]
    # How many ms of silence before we consider speech ended
    silence_duration_ms: int = _SEG["silence_duration_ms"]
    # Minimum speech duration to process (skip coughs, clicks)
    min_speech_ms: int = _SEG["min_speech_ms"]
    # Maximum segment length before forced split (for long monologues)
    max_segment_seconds: float = _SEG["max_segment_seconds"]
    # VAD probability threshold — lower = more sensitive
    speech_threshold: float = _VAD["speech_threshold"]
    # Duration of one VAD frame (512 samples at 16 kHz = 32 ms)
    frame_ms: int = _VAD["frame_ms"]
    # Padding: include this much audio before speech onset
    pre_speech_padding_ms: int = _SEG["pre_speech_padding_ms"]
    # Noise floor calibration period at session start (ms)
    calibration_duration_ms: int = _CAL["duration_ms"]
    # Interval between partial snapshots while speaking (ms)
    partial_interval_ms: int = _SEG["partial_interval_ms"]


class StreamingVADBuffer:
    """
    Accumulates decoded PCM audio, runs VAD frame-by-frame, and emits
    complete speech segments when silence is detected.

    Usage:
        vad = StreamingVADBuffer(config, on_segment=my_callback)
        # Feed decoded PCM frames continuously:
        await vad.feed_pcm(pcm_int16_bytes)
        await vad.feed_pcm(pcm_int16_bytes)
        ...
        # Callback fires automatically when speech segments are detected
    """

    def __init__(
        self,
        config: VADConfig | None = None,
        on_segment: Callable[[bytes, float, float, float], Awaitable[None]] | None = None,
    ):
        self.config = config or VADConfig()
        # Force 16kHz/512-sample frames for Silero VAD compatibility
        self.config.sample_rate = 16000
        self.on_segment = on_segment  # async callback(pcm_bytes, duration, audio_rms, vad_avg_prob)

        # Silero VAD model
        self._vad_model = None
        self._vad_ready = False

        # Audio state
        # Silero VAD requires fixed frame sizes: 512 samples @16kHz
        self._frame_size = _VAD["frame_samples"]
        self._pcm_buffer = bytearray()  # incoming PCM accumulator
        self._speech_buffer = bytearray()  # current speech segment
        self._pre_speech_ring: deque[bytes] = deque(
            maxlen=max(1, self.config.pre_speech_padding_ms // self.config.frame_ms)
        )

        # State machine
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._speech_prob_sum = 0.0  # accumulated VAD probability during speech
        self._silence_threshold_frames = max(
            1, self.config.silence_duration_ms // self.config.frame_ms
        )
        self._max_segment_frames = int(
            self.config.max_segment_seconds * 1000 / self.config.frame_ms
        )
        self._min_speech_frames = max(
            1, self.config.min_speech_ms // self.config.frame_ms
        )

        # Streaming partials — snapshot speech audio every N frames for interim ASR
        self._partial_interval_frames = max(
            1, self.config.partial_interval_ms // self.config.frame_ms  # ~every 1 second
        )
        self._frames_since_partial = 0
        self.on_partial_audio: Callable[[bytes, float], Awaitable[None]] | None = None

        # Noise floor calibration state
        self._calibrating = self.config.calibration_duration_ms > 0
        self._calibration_frames_needed = max(
            1, self.config.calibration_duration_ms // self.config.frame_ms
        ) if self._calibrating else 0
        self._calibration_frame_count = 0
        self._calibration_energy_sum = 0.0
        self._noise_floor_rms = _CAL["uncalibrated_threshold"]  # fallback until calibrated
        self._energy_threshold = _CAL["uncalibrated_threshold"]  # dynamic, updated after calibration
        self.on_calibration_complete: Callable[[float], Awaitable[None]] | None = None

        # Stats
        self.segments_emitted = 0
        self.total_speech_seconds = 0.0

    async def initialize(self):
        """Load the Silero VAD model."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
            )
            self._vad_model = model
            self._vad_ready = True
            logger.info("Silero VAD loaded (~2MB)")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            logger.info("Falling back to energy-based VAD")

    def _is_speech_frame(self, pcm_frame: np.ndarray) -> tuple[bool, float]:
        """Run VAD on a single frame. Returns (is_speech, probability)."""
        if pcm_frame.shape[0] != self._frame_size:
            logger.debug(f"VAD frame size mismatch: {pcm_frame.shape[0]} != {self._frame_size}")
            return False, 0.0
        if self._vad_ready and self._vad_model is not None:
            import torch
            # Silero expects float samples in [-1, 1]; raw int16 values make it
            # miss up to half the speech frames (worst on loud or noisy input).
            tensor = torch.from_numpy(pcm_frame.astype(np.float32) / 32768.0)
            prob = self._vad_model(tensor, self.config.sample_rate).item()
            return prob > self.config.speech_threshold, prob
        else:
            # Fallback: simple energy-based detection
            energy = np.sqrt(np.mean(pcm_frame.astype(np.float32) ** 2))
            prob = min(1.0, energy / (self._energy_threshold * _VAD["energy_fallback_prob_scale"]))
            return energy > self._energy_threshold, prob

    async def feed_pcm(self, pcm_bytes: bytes):
        """
        Feed raw PCM int16 mono audio into the VAD.
        Can be called with any chunk size; internally buffers to frame boundaries.
        """
        # Ensure 16-bit alignment to avoid partial samples
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        self._pcm_buffer.extend(pcm_bytes)

        bytes_per_frame = self._frame_size * 2  # 16-bit = 2 bytes per sample

        while len(self._pcm_buffer) >= bytes_per_frame:
            # Extract one frame
            frame_bytes = bytes(self._pcm_buffer[:bytes_per_frame])
            del self._pcm_buffer[:bytes_per_frame]

            # Convert to numpy (ensure correct frame size for Silero: 512 samples @16kHz)
            frame = np.frombuffer(frame_bytes, dtype=np.int16).copy()
            if frame.shape[0] != self._frame_size:
                if frame.shape[0] > self._frame_size:
                    frame = frame[:self._frame_size]
                else:
                    frame = np.pad(frame, (0, self._frame_size - frame.shape[0]), mode="constant")
                frame_bytes = frame.astype(np.int16).tobytes()

            # Calibration gate — accumulate ambient energy before running VAD
            if self._calibrating:
                energy = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
                self._calibration_energy_sum += energy
                self._calibration_frame_count += 1
                # Buffer frames for pre-speech ring during calibration
                self._pre_speech_ring.append(frame_bytes)
                if self._calibration_frame_count >= self._calibration_frames_needed:
                    self._calibrating = False
                    self._noise_floor_rms = self._calibration_energy_sum / self._calibration_frame_count
                    self._energy_threshold = max(
                        _CAL["min_energy_threshold"], self._noise_floor_rms * _CAL["noise_floor_multiplier"]
                    )
                    logger.info(
                        f"Noise floor calibrated: RMS={self._noise_floor_rms:.0f}, "
                        f"threshold={self._energy_threshold:.0f} "
                        f"({self._calibration_frame_count} frames)"
                    )
                    if self.on_calibration_complete:
                        await self.on_calibration_complete(self._noise_floor_rms)
                continue

            # Run VAD
            is_speech, frame_prob = self._is_speech_frame(frame)

            if is_speech:
                if not self._is_speaking:
                    # Speech onset — include pre-speech padding
                    self._is_speaking = True
                    self._silence_frames = 0
                    self._speech_frames = 0
                    self._speech_prob_sum = 0.0
                    self._frames_since_partial = 0
                    self._speech_buffer = bytearray()

                    # Prepend buffered pre-speech audio
                    for pre_frame in self._pre_speech_ring:
                        self._speech_buffer.extend(pre_frame)

                    logger.debug("Speech started")

                self._speech_buffer.extend(frame_bytes)
                self._speech_frames += 1
                self._speech_prob_sum += frame_prob
                self._silence_frames = 0
                self._frames_since_partial += 1

                # Emit partial audio snapshot for streaming transcription
                if (self._frames_since_partial >= self._partial_interval_frames
                        and self._speech_frames >= self._min_speech_frames
                        and self.on_partial_audio):
                    self._frames_since_partial = 0
                    partial_pcm = bytes(self._speech_buffer)
                    partial_duration = len(partial_pcm) / (self.config.sample_rate * 2)
                    await self.on_partial_audio(partial_pcm, partial_duration)

                # Check max segment length — force split on long monologues
                if self._speech_frames >= self._max_segment_frames:
                    logger.debug(f"Max segment reached ({self.config.max_segment_seconds}s), forcing split")
                    # Save tail audio for crossfade into next segment
                    tail_bytes = self.config.pre_speech_padding_ms * self.config.sample_rate * 2 // 1000
                    tail = bytes(self._speech_buffer[-tail_bytes:]) if len(self._speech_buffer) > tail_bytes else b''
                    await self._emit_segment()
                    # Continue speaking — seed next segment with tail
                    self._is_speaking = True
                    self._speech_buffer = bytearray(tail)
                    self._speech_frames = len(tail) // (self._frame_size * 2)

            else:
                if self._is_speaking:
                    # We're in speech, counting silence
                    self._speech_buffer.extend(frame_bytes)
                    self._silence_frames += 1

                    if self._silence_frames >= self._silence_threshold_frames:
                        # Enough silence — end of utterance
                        await self._emit_segment()
                else:
                    # Not speaking — keep ring buffer for pre-speech padding
                    self._pre_speech_ring.append(frame_bytes)

    async def _emit_segment(self):
        """Emit a completed speech segment via callback."""
        self._is_speaking = False

        if self._speech_frames < self._min_speech_frames:
            logger.debug(f"Segment too short ({self._speech_frames} frames), discarding")
            self._speech_buffer = bytearray()
            self._speech_frames = 0
            self._speech_prob_sum = 0.0
            return

        segment_bytes = bytes(self._speech_buffer)
        duration = len(segment_bytes) / (self.config.sample_rate * 2)  # 16-bit mono

        # Compute audio RMS (root mean square energy)
        samples = np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32)
        audio_rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

        # Average VAD probability across speech frames
        vad_avg_prob = self._speech_prob_sum / self._speech_frames if self._speech_frames > 0 else 0.0

        self.segments_emitted += 1
        self.total_speech_seconds += duration

        logger.info(f"Speech segment #{self.segments_emitted}: {duration:.1f}s (RMS={audio_rms:.0f}, VAD={vad_avg_prob:.2f})")

        # Reset
        self._speech_buffer = bytearray()
        self._speech_frames = 0
        self._silence_frames = 0
        self._speech_prob_sum = 0.0

        # Fire callback
        if self.on_segment:
            await self.on_segment(segment_bytes, duration, audio_rms, vad_avg_prob)

    async def flush(self):
        """Flush any remaining audio (call when stopping listening)."""
        if self._is_speaking and self._speech_frames >= self._min_speech_frames:
            await self._emit_segment()
        self._pcm_buffer.clear()
        self._speech_buffer = bytearray()
        self._is_speaking = False

    def reset(self):
        """Reset all state for a new session."""
        self._pcm_buffer.clear()
        self._speech_buffer = bytearray()
        self._pre_speech_ring.clear()
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        # Re-enable calibration for new session
        self._calibrating = self.config.calibration_duration_ms > 0
        self._calibration_frame_count = 0
        self._calibration_energy_sum = 0.0
        # Silero carries recurrent state and 64 samples of context between frames
        if self._vad_model is not None and hasattr(self._vad_model, "reset_states"):
            self._vad_model.reset_states()


_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_WEBM_CLUSTER_ID = b"\x1f\x43\xb6\x75"


def _webm_init_segment(first_chunk: bytes) -> bytes | None:
    """Return the WebM header (everything before the first Cluster) from a stream's first chunk.

    Replaying only the header, not the first cluster, means a restarted decoder
    doesn't re-emit audio that was already processed. Returns None for non-WebM
    input (e.g. Ogg), which can't be restarted this way.
    """
    if not first_chunk.startswith(_EBML_MAGIC):
        return None
    cluster = first_chunk.find(_WEBM_CLUSTER_ID)
    return first_chunk if cluster == -1 else first_chunk[:cluster]


class AudioDecoder:
    """
    Decodes Opus/WebM chunks from the browser into raw PCM.
    Uses ffmpeg subprocess for decode.

    Two modes:
      - Streaming: persistent ffmpeg process, feed chunks via stdin, read PCM from stdout.
        Efficient for long sessions (O(1) per chunk, no re-decode).
      - Blob: one-shot decode of a complete buffer (used for final flush).
    """

    # ffmpeg restarts allowed per listening stream before giving up and reporting dead
    MAX_RESTARTS = 3

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._process: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task | None = None
        self._pcm_chunks: list[bytes] = []
        self._pcm_lock = asyncio.Lock()
        self._dead = False  # set True when ffmpeg crashes and can't be restarted
        # WebM header (EBML + Segment info + Tracks) from the stream's first chunk.
        # Only the first MediaRecorder chunk carries it, so a restarted ffmpeg
        # needs it replayed before it can decode later clusters.
        self._init_segment: bytes | None = None
        self._first_chunk_seen = False
        self._restarts = 0

    @property
    def is_dead(self) -> bool:
        """True if ffmpeg process has crashed and audio decoding is lost."""
        return self._dead

    async def start_streaming(self, _restart: bool = False):
        """Start a persistent ffmpeg process for streaming decode.

        A normal start begins a new recording stream and forgets the previous
        stream's header and restart count; an internal restart keeps both.
        """
        await self.stop_streaming()
        self._pcm_chunks = []
        self._dead = False
        if not _restart:
            self._init_segment = None
            self._first_chunk_seen = False
            self._restarts = 0
        self._process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._read_task = asyncio.create_task(self._read_stdout())
        logger.debug("Streaming ffmpeg process started")

    async def _read_stdout(self):
        """Background task: continuously read PCM from ffmpeg stdout."""
        try:
            while self._process and self._process.stdout:
                chunk = await self._process.stdout.read(8192)
                if not chunk:
                    break
                async with self._pcm_lock:
                    self._pcm_chunks.append(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"ffmpeg stdout reader ended: {e}")

    async def feed_chunk(self, webm_bytes: bytes) -> bytes:
        """Feed a WebM chunk to the streaming ffmpeg process. Returns any available PCM."""
        if not self._process or not self._process.stdin:
            return b""
        if not self._first_chunk_seen:
            self._first_chunk_seen = True
            self._init_segment = _webm_init_segment(webm_bytes)
        # Detect crashed ffmpeg process and restart it with the stream header replayed
        if self._process.returncode is not None:
            if self._dead or not await self._restart_after_crash(self._process.returncode):
                return b""
        try:
            self._process.stdin.write(webm_bytes)
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            if not self._dead:
                self._dead = True
                logger.error(f"ffmpeg process died (write error: {e}) — audio decoding lost for this session")
            return b""

        # Yield briefly to let the reader task collect output
        await asyncio.sleep(0.05)

        async with self._pcm_lock:
            if not self._pcm_chunks:
                return b""
            pcm = b"".join(self._pcm_chunks)
            self._pcm_chunks.clear()
            return pcm

    async def _restart_after_crash(self, exit_code: int) -> bool:
        """Restart ffmpeg and replay the WebM header. Returns False (and marks dead) if it can't."""
        if self._init_segment is None:
            reason = "no WebM header to replay"
        elif self._restarts >= self.MAX_RESTARTS:
            reason = f"{self.MAX_RESTARTS} restarts already used"
        else:
            reason = ""
        if reason:
            self._dead = True
            logger.error(f"ffmpeg process crashed (exit code {exit_code}), {reason} — audio decoding lost for this session")
            return False

        self._restarts += 1
        logger.warning(
            f"ffmpeg process crashed (exit code {exit_code}) — restarting "
            f"({self._restarts}/{self.MAX_RESTARTS})"
        )
        header = self._init_segment
        await self.start_streaming(_restart=True)
        try:
            self._process.stdin.write(header)
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._dead = True
            logger.error(f"ffmpeg restart failed (write error: {e}) — audio decoding lost for this session")
            return False
        return True

    async def stop_streaming(self) -> bytes:
        """Close the streaming ffmpeg process and return any remaining PCM."""
        remaining_pcm = b""
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
                await self._process.stdin.wait_closed()
            except Exception:
                pass
            # Wait briefly for ffmpeg to flush remaining output
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._process.kill()

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        async with self._pcm_lock:
            if self._pcm_chunks:
                remaining_pcm = b"".join(self._pcm_chunks)
                self._pcm_chunks.clear()

        self._process = None
        return remaining_pcm

    def reset(self):
        """Reset decoder state."""
        self._pcm_chunks.clear()
