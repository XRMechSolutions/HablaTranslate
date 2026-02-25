"""Shared mutable state for API route modules.

Globals are set once during app lifespan startup via the setter functions.
Route modules import from here to avoid circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.pipeline.orchestrator import PipelineOrchestrator
    from server.services.playback import PlaybackService

_pipeline: PipelineOrchestrator | None = None
_lmstudio_manager = None
_playback_service: PlaybackService | None = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


def set_lmstudio_manager(manager):
    global _lmstudio_manager
    _lmstudio_manager = manager


def set_playback_service(service):
    global _playback_service
    _playback_service = service
