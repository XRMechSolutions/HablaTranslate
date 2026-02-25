"""Playback and recording reprocessing routes."""

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import server.routes._state as _state


playback_router = APIRouter(prefix="/api", tags=["playback"])


@playback_router.get("/recordings")
async def list_recordings():
    """List all saved recording sessions. Returns 200 or 503 if service not ready."""
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")
    return _state._playback_service.list_recordings()


@playback_router.get("/recordings/{recording_id}")
async def get_recording(recording_id: str):
    """Get metadata and ground truth for a recording. Returns 404 or 503."""
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")
    result = _state._playback_service.get_recording(recording_id)
    if result is None:
        raise HTTPException(404, "Recording not found")
    return result


@playback_router.get("/recordings/{recording_id}/audio")
async def get_recording_audio(recording_id: str):
    """Stream the raw WebM audio file for a recording. Returns 404 or 503."""
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")
    session_dir = _state._playback_service._safe_session_dir(recording_id)
    if not session_dir:
        raise HTTPException(404, "Recording not found")
    audio_path = session_dir / "raw_stream.webm"
    if not audio_path.exists():
        raise HTTPException(404, "No raw audio file in this recording")
    return FileResponse(
        str(audio_path),
        media_type="audio/webm",
        filename=f"{recording_id}.webm",
    )


@playback_router.get("/recordings/{recording_id}/audio/{filename}")
async def get_segment_audio(recording_id: str, filename: str):
    """Stream an individual segment WAV from a recording.

    Returns 400 for invalid filename pattern. Returns 404 or 503.
    """
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")
    session_dir = _state._playback_service._safe_session_dir(recording_id)
    if not session_dir:
        raise HTTPException(404, "Recording not found")
    # Only allow segment_NNN.wav files
    if not re.match(r'^segment_\d+\.wav$', filename):
        raise HTTPException(400, "Invalid segment filename")
    audio_path = session_dir / filename
    if not audio_path.exists():
        raise HTTPException(404, f"Segment not found: {filename}")
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=filename,
    )


class PlaybackStartRequest(BaseModel):
    recording_id: str
    speed: float = 1.0  # 0=instant, 0.5, 1, 2, 4, 8
    mode: str = "full"  # "full" or "segments"


@playback_router.post("/playback/start")
async def start_playback(req: PlaybackStartRequest):
    """Start playback of a recording through the live pipeline.

    Returns 200 with playback status. Returns 400 for invalid speed/mode.
    Returns 409 if no WebSocket client or client is currently listening.
    Returns 503 if service not ready.
    Side effects: creates async playback task, feeds audio into pipeline.
    """
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")

    from server.routes.websocket import get_active_session
    session = get_active_session()
    if session is None:
        raise HTTPException(409, "No active WebSocket client connected")

    if session.listening:
        raise HTTPException(409, "Client is currently listening via microphone -- stop listening first")

    if req.speed < 0:
        raise HTTPException(400, "Speed must be >= 0")
    if req.mode not in ("full", "segments"):
        raise HTTPException(400, "Mode must be 'full' or 'segments'")

    result = await _state._playback_service.start_playback(
        recording_id=req.recording_id,
        session=session,
        speed=req.speed,
        mode=req.mode,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@playback_router.post("/playback/stop")
async def stop_playback():
    """Stop active playback. Returns {status: "stopped"} or {status: "no_playback_active"}.

    Returns 503 if service not ready. Side effects: cancels playback task.
    """
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")
    if not _state._playback_service.is_active:
        return {"status": "no_playback_active"}
    await _state._playback_service.stop_playback()
    return {"status": "stopped"}
