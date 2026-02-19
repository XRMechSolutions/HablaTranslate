"""Audio recording service for debugging and test sample collection.

Saves incoming audio from the web app for:
1. Building test samples from real usage
2. Debugging ASR/translation issues with problematic audio
3. Collecting training data

Respects RecordingConfig settings for what to save and where.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import wave

from server.config import RecordingConfig

logger = logging.getLogger("habla.recorder")


class AudioRecorder:
    """Records audio streams with optional metadata."""

    def __init__(self, config: RecordingConfig, session_id: str):
        self.config = config
        self.session_id = session_id
        self.enabled = config.enabled

        if not self.enabled:
            return

        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = config.output_dir / f"{session_id}_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # File handles
        self.raw_file: Optional[Path] = None
        self.raw_handle = None
        self.segment_counter = 0

        # Metadata
        self.metadata = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "raw_audio_format": "webm_opus",
            "sample_rate": 16000,
            "segments": []
        }

        logger.info(f"Audio recorder initialized: {self.session_dir}")

    def start_recording(self):
        """Start recording raw audio stream."""
        if not self.enabled or not self.config.save_raw_audio:
            return

        # Create raw audio file (WebM/Opus from browser)
        self.raw_file = self.session_dir / "raw_stream.webm"
        self.raw_handle = open(self.raw_file, "wb")
        logger.info(f"Started recording to: {self.raw_file}")

    def write_raw_chunk(self, chunk: bytes):
        """Write raw WebM/Opus chunk from browser."""
        if not self.enabled or not self.config.save_raw_audio or not self.raw_handle:
            return

        try:
            self.raw_handle.write(chunk)
        except Exception as e:
            logger.error(f"Failed to write raw audio chunk: {e}")

    def save_pcm_segment(self, pcm_bytes: bytes, metadata: dict = None):
        """Save a decoded PCM segment (from VAD).

        Args:
            pcm_bytes: Raw PCM audio data (16-bit, 16kHz, mono)
            metadata: Optional metadata about this segment (speaker, duration, etc.)
        """
        if not self.enabled or not self.config.save_vad_segments:
            return

        self.segment_counter += 1
        segment_file = self.session_dir / f"segment_{self.segment_counter:03d}.wav"

        try:
            # Write as WAV file
            with wave.open(str(segment_file), "wb") as wav:
                wav.setnchannels(1)  # Mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(16000)  # 16kHz
                wav.writeframes(pcm_bytes)

            # Add to metadata
            segment_info = {
                "segment_id": self.segment_counter,
                "filename": segment_file.name,
                "size_bytes": len(pcm_bytes),
                "duration_seconds": len(pcm_bytes) / (16000 * 2),  # 16kHz, 16-bit
                "recorded_at": datetime.now().isoformat(),
            }
            if metadata:
                segment_info.update(metadata)

            self.metadata["segments"].append(segment_info)

            logger.debug(f"Saved segment: {segment_file.name}")

        except Exception as e:
            logger.error(f"Failed to save PCM segment: {e}")

    def stop_recording(self):
        """Stop recording and save metadata."""
        if not self.enabled:
            return

        # Close raw file
        if self.raw_handle:
            try:
                self.raw_handle.close()
                logger.info(f"Closed raw audio file: {self.raw_file}")
            except Exception as e:
                logger.error(f"Error closing raw file: {e}")

        # Save metadata
        if self.config.include_metadata:
            self.metadata["ended_at"] = datetime.now().isoformat()
            self.metadata["total_segments"] = self.segment_counter

            metadata_file = self.session_dir / "metadata.json"
            try:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2)
                logger.info(f"Saved metadata: {metadata_file}")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

        # Cleanup old recordings if over limit
        self._cleanup_old_recordings()

        logger.info(f"Recording stopped: {self.segment_counter} segments saved")

    def _cleanup_old_recordings(self):
        """Remove old recordings if exceeding max_recordings limit."""
        if not self.config.max_recordings:
            return

        try:
            # Get all session directories sorted by creation time
            sessions = sorted(
                self.config.output_dir.glob("*_*"),
                key=lambda p: p.stat().st_ctime
            )

            # Remove oldest sessions if over limit
            while len(sessions) > self.config.max_recordings:
                oldest = sessions.pop(0)
                logger.info(f"Removing old recording: {oldest.name}")
                # Remove all files in directory
                for file in oldest.glob("*"):
                    file.unlink()
                oldest.rmdir()

        except Exception as e:
            logger.error(f"Failed to cleanup old recordings: {e}")

    def add_segment_metadata(self, segment_id: int, key: str, value):
        """Add additional metadata to a segment after it's been saved."""
        if not self.enabled or segment_id > len(self.metadata["segments"]):
            return

        try:
            self.metadata["segments"][segment_id - 1][key] = value
        except Exception as e:
            logger.error(f"Failed to add segment metadata: {e}")


class RecorderManager:
    """Manages audio recorders for multiple sessions."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self.recorders: dict[str, AudioRecorder] = {}

    def create_recorder(self, session_id: str) -> AudioRecorder:
        """Create a new recorder for a session."""
        recorder = AudioRecorder(self.config, session_id)
        if recorder.enabled:
            self.recorders[session_id] = recorder
        return recorder

    def get_recorder(self, session_id: str) -> Optional[AudioRecorder]:
        """Get recorder for a session."""
        return self.recorders.get(session_id)

    def remove_recorder(self, session_id: str):
        """Remove and cleanup recorder for a session."""
        if session_id in self.recorders:
            recorder = self.recorders[session_id]
            recorder.stop_recording()
            del self.recorders[session_id]

    def stop_all(self):
        """Stop all active recorders."""
        for recorder in self.recorders.values():
            recorder.stop_recording()
        self.recorders.clear()
