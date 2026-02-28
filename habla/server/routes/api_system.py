"""System management routes: status, direction, mode, ASR language, speakers, recording, metrics."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.db.database import get_db
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

    if not app_config:
        raise HTTPException(503, "Server not initialized")

    app_config.recording.enabled = req.enabled
    set_recording_config(app_config.recording)

    if req.enabled:
        app_config.recording.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Recording toggled: enabled={req.enabled}")
    return {
        "recording_enabled": app_config.recording.enabled,
        "output_dir": str(app_config.recording.output_dir)
    }


@system_router.get("/metrics")
async def system_metrics():
    """Return translation quality metrics: confidence distribution, correction rates, idiom stats.

    Combines in-memory pipeline/translator metrics with DB aggregates.
    Returns 503 if pipeline not initialized.
    """
    if not _state._pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    db = await get_db()
    pipeline = _state._pipeline

    # Confidence distribution from DB
    confidence_rows = await db.execute_fetchall("""
        SELECT
            COUNT(*) as total,
            ROUND(AVG(confidence), 3) as avg_confidence,
            COUNT(CASE WHEN confidence < 0.3 THEN 1 END) as low_count,
            COUNT(CASE WHEN confidence >= 0.3 AND confidence < 0.7 THEN 1 END) as medium_count,
            COUNT(CASE WHEN confidence >= 0.7 THEN 1 END) as high_count
        FROM exchanges
    """)
    conf = dict(confidence_rows[0]) if confidence_rows else {}

    # Correction frequency per speaker
    correction_rows = await db.execute_fetchall("""
        SELECT e.speaker_id,
               COALESCE(s.custom_name, s.auto_label, e.speaker_id) as speaker_name,
               COUNT(*) as total_exchanges,
               SUM(CASE WHEN e.is_correction THEN 1 ELSE 0 END) as corrections
        FROM exchanges e
        LEFT JOIN speakers s ON e.speaker_id = s.id AND e.session_id = s.session_id
        GROUP BY e.speaker_id
    """)
    corrections_by_speaker = [
        {
            "speaker_id": r["speaker_id"],
            "speaker_name": r["speaker_name"],
            "total_exchanges": r["total_exchanges"],
            "corrections": r["corrections"],
        }
        for r in correction_rows
    ]

    # Average processing time
    proc_rows = await db.execute_fetchall("""
        SELECT ROUND(AVG(processing_ms), 0) as avg_ms,
               MIN(processing_ms) as min_ms,
               MAX(processing_ms) as max_ms
        FROM exchanges WHERE processing_ms > 0
    """)
    processing = dict(proc_rows[0]) if proc_rows else {}

    # Quality metrics breakdown from DB
    qm_rows = await db.execute_fetchall("""
        SELECT status, COUNT(*) as cnt
        FROM quality_metrics
        GROUP BY status
    """)
    quality_breakdown = {r["status"]: r["cnt"] for r in qm_rows} if qm_rows else {}

    # Pipeline in-memory metrics
    pipeline_metrics = pipeline.metrics
    translator_metrics = pipeline.translator.metrics

    # Compute translator avg latency
    avg_latency = 0.0
    if translator_metrics["requests"] > 0:
        avg_latency = round(translator_metrics["total_latency_ms"] / translator_metrics["requests"], 1)

    return {
        "confidence": {
            "total_exchanges": conf.get("total", 0),
            "average": conf.get("avg_confidence"),
            "low_count": conf.get("low_count", 0),
            "medium_count": conf.get("medium_count", 0),
            "high_count": conf.get("high_count", 0),
        },
        "corrections": {
            "total_detected": pipeline_metrics.get("corrections_detected", 0),
            "by_speaker": corrections_by_speaker,
        },
        "idioms": {
            "pattern_db_hits": pipeline_metrics.get("idiom_pattern_db_hits", 0),
            "llm_hits": pipeline_metrics.get("idiom_llm_hits", 0),
            "patterns_loaded": pipeline.idiom_scanner.count,
        },
        "pipeline": {
            "segments_processed": pipeline_metrics.get("segments_processed", 0),
            "translations_completed": pipeline_metrics.get("translations_completed", 0),
            "translation_errors": pipeline_metrics.get("translation_errors", 0),
            "low_confidence_count": pipeline_metrics.get("low_confidence_count", 0),
            "asr_rejected_count": pipeline_metrics.get("asr_rejected_count", 0),
            "asr_empty_count": pipeline_metrics.get("asr_empty_count", 0),
            "queue_depth": pipeline_metrics.get("queue_depth", 0),
            "peak_queue_depth": pipeline_metrics.get("peak_queue_depth", 0),
        },
        "quality_metrics": quality_breakdown,
        "translator": {
            "provider": translator_metrics.get("provider"),
            "model": translator_metrics.get("model"),
            "avg_latency_ms": avg_latency,
            "total_requests": translator_metrics.get("requests", 0),
            "successes": translator_metrics.get("successes", 0),
            "failures": translator_metrics.get("failures", 0),
            "timeouts": translator_metrics.get("timeouts", 0),
            "degraded": translator_metrics.get("degraded", False),
        },
        "processing": processing,
    }
