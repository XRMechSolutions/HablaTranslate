"""Shared mutable state for API route modules.

Globals are set once during app lifespan startup via the setter functions.
Route modules import from here to avoid circular dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.pipeline.orchestrator import PipelineOrchestrator
    from server.services.playback import PlaybackService
    from server.config import RecordingConfig

_pipeline: PipelineOrchestrator | None = None
_lmstudio_manager = None
_playback_service: PlaybackService | None = None
_recording_config: RecordingConfig | None = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


def get_pipeline():
    return _pipeline


def set_lmstudio_manager(manager):
    global _lmstudio_manager
    _lmstudio_manager = manager


def set_playback_service(service):
    global _playback_service
    _playback_service = service


def set_recording_config(config):
    global _recording_config
    _recording_config = config


def get_recordings_dir() -> Path:
    """Return the configured recordings directory, falling back to default."""
    if _recording_config:
        return _recording_config.output_dir
    return Path("data/audio/recordings")
