"""Recording playback service for reprocessing audio through the live pipeline.

Reads saved recordings and feeds them back through the full audio pipeline
(decode -> VAD -> ASR -> translate) as if they were coming from a live
microphone. Supports speed control for real-time display testing or fast
batch iteration.

Two playback modes:
  1. FULL PIPELINE: Reads raw_stream.webm, splits into ~1s WebM chunks via
     ffmpeg, feeds through decode->VAD->ASR->translate. Tests everything.
  2. SEGMENTS ONLY: Feeds pre-segmented WAVs directly to the orchestrator's
     process_wav(), skipping decode/VAD. Faster for ASR/translation tuning.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("habla.playback")


class PlaybackService:
    """Manages playback of recorded audio through the live pipeline."""

    def __init__(self, recordings_dir: Path):
        self.recordings_dir = recordings_dir
        self._task: Optional[asyncio.Task] = None
        self._active_recording_id: Optional[str] = None
        self._cancelled = False

    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    def list_recordings(self) -> list[dict]:
        """List all available recording sessions."""
        recordings = []
        if not self.recordings_dir.exists():
            return recordings

        for session_dir in sorted(self.recordings_dir.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue

            metadata_path = session_dir / "metadata.json"
            has_raw = (session_dir / "raw_stream.webm").exists()
            has_ground_truth = (session_dir / "ground_truth.json").exists()

            segment_count = len(list(session_dir.glob("segment_*.wav")))

            info = {
                "id": session_dir.name,
                "has_raw_stream": has_raw,
                "has_ground_truth": has_ground_truth,
                "segment_count": segment_count,
            }

            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                    info["started_at"] = meta.get("started_at", "")
                    info["ended_at"] = meta.get("ended_at", "")
                    info["total_segments"] = meta.get("total_segments", segment_count)

                    # Calculate total duration from segments
                    total_duration = sum(
                        s.get("duration_seconds", 0) for s in meta.get("segments", [])
                    )
                    info["total_duration_seconds"] = round(total_duration, 1)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Bad metadata in {session_dir.name}: {e}")

            recordings.append(info)

        return recordings

    def _safe_session_dir(self, recording_id: str) -> Optional[Path]:
        """Resolve a recording ID to a directory, rejecting path traversal."""
        session_dir = (self.recordings_dir / recording_id).resolve()
        if not session_dir.is_relative_to(self.recordings_dir.resolve()):
            logger.warning(f"Path traversal rejected: {recording_id}")
            return None
        if not session_dir.is_dir():
            return None
        return session_dir

    def get_recording(self, recording_id: str) -> Optional[dict]:
        """Get full metadata + ground truth for a recording."""
        session_dir = self._safe_session_dir(recording_id)
        if not session_dir:
            return None

        result = {"id": recording_id}

        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    result["metadata"] = json.load(f)
            except (json.JSONDecodeError, OSError):
                result["metadata"] = None

        gt_path = session_dir / "ground_truth.json"
        if gt_path.exists():
            try:
                with open(gt_path, "r") as f:
                    result["ground_truth"] = json.load(f)
            except (json.JSONDecodeError, OSError):
                result["ground_truth"] = None

        return result

    async def start_playback(
        self,
        recording_id: str,
        session,  # ClientSession — avoid circular import
        speed: float = 1.0,
        mode: str = "full",  # "full" or "segments"
    ) -> dict:
        """Start playback of a recording through the live pipeline.

        Args:
            recording_id: Directory name under recordings/
            session: Active ClientSession with WebSocket
            speed: Playback speed multiplier (1=real-time, 2=2x, 0=instant)
            mode: "full" = raw WebM through decode+VAD, "segments" = WAVs direct to ASR
        """
        if self.is_active:
            return {"error": "Playback already in progress", "recording_id": self._active_recording_id}

        session_dir = self._safe_session_dir(recording_id)
        if not session_dir:
            return {"error": f"Recording not found: {recording_id}"}

        self._cancelled = False

        if mode == "segments":
            self._task = asyncio.create_task(
                self._playback_segments(session_dir, session, speed)
            )
            self._active_recording_id = recording_id
        else:
            raw_stream = session_dir / "raw_stream.webm"
            if not raw_stream.exists():
                return {"error": "No raw_stream.webm in recording — use 'segments' mode"}
            self._task = asyncio.create_task(
                self._playback_full(session_dir, session, speed)
            )
            self._active_recording_id = recording_id

        return {"status": "started", "recording_id": recording_id, "speed": speed, "mode": mode}

    async def stop_playback(self):
        """Cancel active playback."""
        self._cancelled = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._active_recording_id = None

        # Post-stop invariant check
        if self.is_active:
            logger.warning("stop_playback completed but is_active still True")

    async def _playback_full(self, session_dir: Path, session, speed: float):
        """Full pipeline playback: split raw WebM into chunks and feed through decoder+VAD."""
        raw_stream = session_dir / "raw_stream.webm"
        tmp_dir = None

        try:
            # Split WebM into ~1s chunks using ffmpeg
            tmp_dir = Path(tempfile.mkdtemp(prefix="habla_playback_"))
            chunk_pattern = str(tmp_dir / "chunk_%04d.webm")

            logger.info(f"Splitting {raw_stream} into 1s chunks...")
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", str(raw_stream),
                "-f", "segment",
                "-segment_time", "1.0",
                "-segment_format", "webm",
                "-c", "copy",
                chunk_pattern,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode(errors="replace")[-500:]
                logger.error(f"ffmpeg split failed: {err}")
                await session._send({"type": "error", "message": f"Failed to split recording: {err[:200]}"})
                return

            # Gather and sort chunks
            chunks = sorted(tmp_dir.glob("chunk_*.webm"))
            total_chunks = len(chunks)

            if total_chunks == 0:
                await session._send({"type": "error", "message": "No chunks produced from recording"})
                return

            logger.info(f"Playback starting: {total_chunks} chunks at {speed}x speed")

            # Notify client
            await session._send({
                "type": "playback_started",
                "recording_id": session_dir.name,
                "total_chunks": total_chunks,
                "speed": speed,
                "mode": "full",
            })

            # Start the decoder and decode loop (like start_listening but without recorder)
            # Only set playback_mode, not listening — listening is controlled by the
            # client's start/stop buttons and must not be modified by playback.
            session.playback_mode = True
            session.vad.reset()
            session.decoder.reset()
            session.vad.on_partial_audio = session._on_partial_audio
            await session.decoder.start_streaming()
            session._decode_task = asyncio.create_task(session._continuous_decode_loop())

            # Feed chunks at controlled speed
            for i, chunk_path in enumerate(chunks):
                if self._cancelled:
                    break

                chunk_data = chunk_path.read_bytes()
                if chunk_data:
                    # Feed directly into the chunk inbox (same as handle_audio_chunk but skip recording)
                    session._chunk_inbox.append(chunk_data)
                    session._chunk_event.set()

                # Progress update every 10 chunks
                if (i + 1) % 10 == 0:
                    await session._send({
                        "type": "playback_progress",
                        "chunk_index": i + 1,
                        "total_chunks": total_chunks,
                    })

                # Pacing
                if speed > 0:
                    await asyncio.sleep(1.0 / speed)
                else:
                    # Instant mode: yield to event loop periodically
                    if i % 5 == 0:
                        await asyncio.sleep(0)

            # Flush the pipeline: stop decoder, flush VAD
            if session._decode_task:
                session._decode_task.cancel()
                try:
                    await session._decode_task
                except asyncio.CancelledError:
                    pass

            try:
                remaining_pcm = await session.decoder.stop_streaming()
                if remaining_pcm:
                    await session.vad.feed_pcm(remaining_pcm)
            except Exception as e:
                logger.warning(f"Playback final flush error: {e}")

            await session.vad.flush()
            await session._flush_pending_now()

            # Wait for pipeline queue to drain
            if hasattr(session.pipeline, '_queue'):
                deadline = asyncio.get_event_loop().time() + 60
                while not session.pipeline._queue.empty():
                    if asyncio.get_event_loop().time() > deadline:
                        logger.warning("Pipeline queue did not drain within 60s")
                        break
                    await asyncio.sleep(0.5)

            session.playback_mode = False

            if self._cancelled:
                await session._send({"type": "playback_stopped"})
                logger.info("Playback cancelled by user")
            else:
                await session._send({
                    "type": "playback_finished",
                    "recording_id": session_dir.name,
                    "chunks_processed": total_chunks,
                })
                logger.info(f"Playback complete: {total_chunks} chunks processed")

        except asyncio.CancelledError:
            if session._decode_task and not session._decode_task.done():
                session._decode_task.cancel()
                try:
                    await session._decode_task
                except asyncio.CancelledError:
                    pass
            try:
                await session.decoder.stop_streaming()
            except Exception:
                pass
            session.playback_mode = False
            await session._send({"type": "playback_stopped"})
            logger.info("Playback cancelled")
        except Exception as e:
            if session._decode_task and not session._decode_task.done():
                session._decode_task.cancel()
                try:
                    await session._decode_task
                except asyncio.CancelledError:
                    pass
            try:
                await session.decoder.stop_streaming()
            except Exception:
                pass
            session.playback_mode = False
            logger.error(f"Playback error: {e}")
            await session._send({"type": "error", "message": f"Playback error: {e}"})
        finally:
            self._task = None
            self._active_recording_id = None
            # Clean up temp chunks
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _playback_segments(self, session_dir: Path, session, speed: float):
        """Segment-only playback: feed WAVs directly to pipeline.process_wav()."""
        try:
            segments = sorted(session_dir.glob("segment_*.wav"))
            total = len(segments)

            if total == 0:
                await session._send({"type": "error", "message": "No segment WAVs in recording"})
                return

            logger.info(f"Segment playback starting: {total} segments at {speed}x speed")

            await session._send({
                "type": "playback_started",
                "recording_id": session_dir.name,
                "total_chunks": total,
                "speed": speed,
                "mode": "segments",
            })

            session.playback_mode = True

            for i, wav_path in enumerate(segments):
                if self._cancelled:
                    break

                # Get segment duration for pacing
                file_size = wav_path.stat().st_size
                # WAV header is 44 bytes, 16-bit mono 16kHz = 32000 bytes/sec
                duration = max(0, (file_size - 44)) / 32000.0

                logger.info(f"Segment {i+1}/{total}: {wav_path.name} ({duration:.1f}s)")

                # Feed directly to orchestrator (bypasses decode+VAD)
                try:
                    await session.pipeline.process_wav(str(wav_path))
                except Exception as e:
                    logger.error(f"Segment {wav_path.name} error: {e}")
                    await session._send({"type": "error", "message": f"Segment error: {e}"})

                # Progress
                if (i + 1) % 5 == 0 or i == total - 1:
                    await session._send({
                        "type": "playback_progress",
                        "chunk_index": i + 1,
                        "total_chunks": total,
                    })

                # Pacing based on segment duration
                if speed > 0 and duration > 0:
                    await asyncio.sleep(duration / speed)
                else:
                    await asyncio.sleep(0)

            session.playback_mode = False

            if self._cancelled:
                await session._send({"type": "playback_stopped"})
            else:
                await session._send({
                    "type": "playback_finished",
                    "recording_id": session_dir.name,
                    "chunks_processed": total,
                })
                logger.info(f"Segment playback complete: {total} segments")

        except asyncio.CancelledError:
            session.playback_mode = False
            await session._send({"type": "playback_stopped"})
        except Exception as e:
            session.playback_mode = False
            logger.error(f"Segment playback error: {e}")
            await session._send({"type": "error", "message": f"Segment playback error: {e}"})
        finally:
            self._task = None
            self._active_recording_id = None
