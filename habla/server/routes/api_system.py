"""System management routes: status, direction, mode, ASR language, speakers, recording."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.routes._state import _pipeline, _lmstudio_manager
import server.routes._state as _state


system_router = APIRouter(prefix="/api/system", tags=["system"])


@system_router.get("/status")
async def system_status():
    """Return full system status including pipeline state, speakers, and LLM info.

    Returns 200 {status: "not initialized"} when pipeline is None.
    Returns 200 with detailed status dict when pipeline is ready or loading.
    """
    if not _state._pipeline:
        return {"status": "not initialized"}

    # Get recording status from global config
    from server.main import app_config

    return {
        "status": "ready" if _state._pipeline.ready else "loading",
        "direction": _state._pipeline.direction,
        "mode": _state._pipeline.mode,
        "asr_auto_language": _state._pipeline.config.asr.auto_language,
        "speakers": [s.model_dump() for s in _state._pipeline.speaker_tracker.get_all()],
        "topic_summary": _state._pipeline.topic_summary,
        "idiom_patterns_loaded": _state._pipeline.idiom_scanner.count,
        "queue_depth": _state._pipeline._queue.qsize(),
        "llm_provider": _state._pipeline.translator.config.provider,
        "llm_model": _state._pipeline.translator.config.model,
        "recording_enabled": app_config.recording.enabled if app_config else False,
        "lmstudio_running": (await _state._lmstudio_manager.is_running()) if _state._lmstudio_manager else False,
        "lmstudio_models": _state._lmstudio_manager.get_loaded_models() if _state._lmstudio_manager else [],
    }


class DirectionRequest(BaseModel):
    direction: str


@system_router.post("/direction")
async def set_direction(req: DirectionRequest):
    """Set translation direction. Returns 200 or 400 for invalid value, 503 if pipeline not ready.

    Side effects: mutates pipeline direction for all subsequent translations.
    """
    if not _state._pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    if req.direction not in ("es_to_en", "en_to_es"):
        raise HTTPException(400, "Direction must be 'es_to_en' or 'en_to_es'")
    _state._pipeline.set_direction(req.direction)
    return {"direction": req.direction}


class ModeRequest(BaseModel):
    mode: str


@system_router.post("/mode")
async def set_mode(req: ModeRequest):
    """Set operating mode ('conversation' or 'classroom'). Returns 400/503 on error.

    Side effects: mutates pipeline mode; classroom enables grammar correction detection.
    """
    if not _state._pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    if req.mode not in ("conversation", "classroom"):
        raise HTTPException(400, "Mode must be 'conversation' or 'classroom'")
    _state._pipeline.set_mode(req.mode)
    return {"mode": req.mode}


class ASRLanguageRequest(BaseModel):
    auto_language: bool


@system_router.post("/asr/language")
async def set_asr_language(req: ASRLanguageRequest):
    """Toggle ASR auto-language detection. Returns 503 if pipeline not ready.

    Side effects: mutates pipeline ASR config for subsequent transcriptions.
    """
    if not _state._pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    _state._pipeline.config.asr.auto_language = req.auto_language
    return {"auto_language": _state._pipeline.config.asr.auto_language}


class RenameRequest(BaseModel):
    name: str
    role: str | None = None


@system_router.put("/speakers/{speaker_id}")
async def rename_speaker(speaker_id: str, req: RenameRequest):
    """Rename a speaker and optionally set role hint.

    Returns 200 with speaker dict, 404 if speaker_id unknown, 503 if pipeline not ready.
    Side effects: mutates speaker_tracker in-memory state.
    """
    if not _state._pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    result = _state._pipeline.speaker_tracker.rename(speaker_id, req.name)
    if req.role:
        _state._pipeline.speaker_tracker.set_role_hint(speaker_id, req.role)
    if not result:
        raise HTTPException(404, "Speaker not found")
    return result.model_dump()


class RecordingRequest(BaseModel):
    enabled: bool


@system_router.post("/recording")
async def set_recording(req: RecordingRequest):
    """Toggle audio recording on/off."""
    import logging
    from server.main import app_config
    from server.routes.websocket import set_recording_config

    logger = logging.getLogger("habla.api")
    logger.info(f"[RECORDING API] Request to set recording: enabled={req.enabled}")

    if not app_config:
        raise HTTPException(503, "Server not initialized")

    # Update the global config
    logger.info(f"[RECORDING API] Before update: app_config.recording.enabled={app_config.recording.enabled}")
    app_config.recording.enabled = req.enabled
    logger.info(f"[RECORDING API] After update: app_config.recording.enabled={app_config.recording.enabled}")

    # Update websocket handler's config reference
    logger.info(f"[RECORDING API] Calling set_recording_config with enabled={req.enabled}")
    set_recording_config(app_config.recording)

    # Create recordings directory if enabling
    if req.enabled:
        logger.info(f"[RECORDING API] Creating recordings directory: {app_config.recording.output_dir}")
        app_config.recording.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[RECORDING API] Directory created/verified")

    logger.info(f"[RECORDING API] Final state: recording_enabled={app_config.recording.enabled}")
    return {
        "recording_enabled": app_config.recording.enabled,
        "output_dir": str(app_config.recording.output_dir)
    }
