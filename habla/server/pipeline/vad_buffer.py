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

logger = logging.getLogger("habla.vad")


@dataclass
class VADConfig:
    """VAD tuning parameters."""
    sample_rate: int = 16000
    # How many ms of silence before we consider speech ended
    silence_duration_ms: int = 600
    # Minimum speech duration to process (skip coughs, clicks)
    min_speech_ms: int = 400
    # Maximum segment length before forced split (for long monologues)
    max_segment_seconds: float = 30.0
    # VAD probability threshold — lower = more sensitive
    speech_threshold: float = 0.35
    # Chunk size for VAD processing (30ms frames)
    frame_ms: int = 30
    # Padding: include this much audio before speech onset
    pre_speech_padding_ms: int = 300


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
        on_segment: Callable[[bytes, float], Awaitable[None]] | None = None,
    ):
        self.config = config or VADConfig()
        # Force 16kHz/512-sample frames for Silero VAD compatibility
        self.config.sample_rate = 16000
        self.on_segment = on_segment  # async callback(pcm_bytes, duration_seconds)

        # Silero VAD model
        self._vad_model = None
        self._vad_ready = False

        # Audio state
        # Silero VAD requires fixed frame sizes: 512 samples @16kHz
        self._frame_size = 512
        self._pcm_buffer = bytearray()  # incoming PCM accumulator
        self._speech_buffer = bytearray()  # current speech segment
        self._pre_speech_ring: deque[bytes] = deque(
            maxlen=max(1, self.config.pre_speech_padding_ms // self.config.frame_ms)
        )

        # State machine
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
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
            1, 1000 // self.config.frame_ms  # ~every 1 second
        )
        self._frames_since_partial = 0
        self.on_partial_audio: Callable[[bytes, float], Awaitable[None]] | None = None

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

    def _is_speech_frame(self, pcm_frame: np.ndarray) -> bool:
        """Run VAD on a single frame. Returns True if speech detected."""
        if pcm_frame.shape[0] != self._frame_size:
            logger.debug(f"VAD frame size mismatch: {pcm_frame.shape[0]} != {self._frame_size}")
            return False
        if self._vad_ready and self._vad_model is not None:
            import torch
            tensor = torch.from_numpy(pcm_frame).float()
            prob = self._vad_model(tensor, self.config.sample_rate).item()
            return prob > self.config.speech_threshold
        else:
            # Fallback: simple energy-based detection
            energy = np.sqrt(np.mean(pcm_frame.astype(np.float32) ** 2))
            return energy > 500  # rough threshold for 16-bit audio

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

            # Run VAD
            is_speech = self._is_speech_frame(frame)

            if is_speech:
                if not self._is_speaking:
                    # Speech onset — include pre-speech padding
                    self._is_speaking = True
                    self._silence_frames = 0
                    self._speech_frames = 0
                    self._frames_since_partial = 0
                    self._speech_buffer = bytearray()

                    # Prepend buffered pre-speech audio
                    for pre_frame in self._pre_speech_ring:
                        self._speech_buffer.extend(pre_frame)

                    logger.debug("Speech started")

                self._speech_buffer.extend(frame_bytes)
                self._speech_frames += 1
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
            return

        segment_bytes = bytes(self._speech_buffer)
        duration = len(segment_bytes) / (self.config.sample_rate * 2)  # 16-bit mono

        self.segments_emitted += 1
        self.total_speech_seconds += duration

        logger.info(f"Speech segment #{self.segments_emitted}: {duration:.1f}s")

        # Reset
        self._speech_buffer = bytearray()
        self._speech_frames = 0
        self._silence_frames = 0

        # Fire callback
        if self.on_segment:
            await self.on_segment(segment_bytes, duration)

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


class AudioDecoder:
    """
    Decodes Opus/WebM chunks from the browser into raw PCM.
    Uses ffmpeg subprocess for decode.

    Two modes:
      - Streaming: persistent ffmpeg process, feed chunks via stdin, read PCM from stdout.
        Efficient for long sessions (O(1) per chunk, no re-decode).
      - Blob: one-shot decode of a complete buffer (used for final flush).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._process: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task | None = None
        self._pcm_chunks: list[bytes] = []
        self._pcm_lock = asyncio.Lock()

    async def start_streaming(self):
        """Start a persistent ffmpeg process for streaming decode."""
        await self.stop_streaming()
        self._pcm_chunks = []
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
        try:
            self._process.stdin.write(webm_bytes)
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.debug(f"ffmpeg stdin write error: {e}")
            return b""

        # Yield briefly to let the reader task collect output
        await asyncio.sleep(0.05)

        async with self._pcm_lock:
            if not self._pcm_chunks:
                return b""
            pcm = b"".join(self._pcm_chunks)
            self._pcm_chunks.clear()
            return pcm

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
