"""REST API routes for vocab, sessions, system management, and LLM provider control."""

import asyncio
import json
import re
from datetime import datetime
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from server.db.database import get_db
from server.services.vocab import VocabService
from server.pipeline.orchestrator import PipelineOrchestrator


# --- Vocab Routes ---

vocab_router = APIRouter(prefix="/api/vocab", tags=["vocab"])
_vocab_service = VocabService()


class CreateVocabRequest(BaseModel):
    term: str
    meaning: str = ""
    literal: str | None = None
    category: str = "phrase"
    source_sentence: str | None = None
    region: str = "universal"


@vocab_router.post("")
async def create_vocab(req: CreateVocabRequest):
    """Manually create a vocab item (from client save button or text selection)."""
    if not req.term.strip():
        raise HTTPException(400, "Term is required")
    db = await get_db()

    # Check for duplicate
    existing = await db.execute_fetchall(
        "SELECT id, times_encountered FROM vocab WHERE LOWER(term) = LOWER(?)",
        (req.term.strip(),),
    )
    if existing:
        row = existing[0]
        await db.execute(
            "UPDATE vocab SET times_encountered = ? WHERE id = ?",
            (row["times_encountered"] + 1, row["id"]),
        )
        await db.commit()
        return {"id": row["id"], "duplicate": True, "times_encountered": row["times_encountered"] + 1}

    from datetime import datetime
    cursor = await db.execute(
        """INSERT INTO vocab (term, literal, meaning, category, source_sentence, region, next_review)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (req.term.strip(), req.literal or "", req.meaning, req.category,
         req.source_sentence, req.region, datetime.utcnow().isoformat()),
    )
    await db.commit()
    return {"id": cursor.lastrowid, "duplicate": False}


@vocab_router.get("")
async def list_vocab(limit: int = 50, offset: int = 0, category: str | None = None):
    return await _vocab_service.get_all(limit=limit, offset=offset, category=category)


@vocab_router.get("/due")
async def vocab_due(limit: int = 20):
    return await _vocab_service.get_due_for_review(limit=limit)


@vocab_router.get("/stats")
async def vocab_stats():
    return await _vocab_service.get_stats()


@vocab_router.get("/search")
async def search_vocab(q: str, limit: int = 20):
    return await _vocab_service.search(q, limit=limit)


class ReviewRequest(BaseModel):
    quality: int  # 0-5


@vocab_router.post("/{vocab_id}/review")
async def review_vocab(vocab_id: int, req: ReviewRequest):
    if not 0 <= req.quality <= 5:
        raise HTTPException(400, "Quality must be 0-5")
    return await _vocab_service.record_review(vocab_id, req.quality)


@vocab_router.delete("/{vocab_id}")
async def delete_vocab(vocab_id: int):
    deleted = await _vocab_service.delete(vocab_id)
    if not deleted:
        raise HTTPException(404, "Vocab item not found")
    return {"deleted": True}


@vocab_router.get("/export/anki")
async def export_anki():
    csv_data = await _vocab_service.export_anki_csv()
    return PlainTextResponse(csv_data, media_type="text/tab-separated-values",
                             headers={"Content-Disposition": "attachment; filename=habla-vocab.tsv"})


@vocab_router.get("/export/json")
async def export_json():
    return await _vocab_service.export_json()


# --- System Routes ---

system_router = APIRouter(prefix="/api/system", tags=["system"])
_pipeline: PipelineOrchestrator | None = None
_lmstudio_manager = None


def set_pipeline(pipeline: PipelineOrchestrator):
    global _pipeline
    _pipeline = pipeline


def set_lmstudio_manager(manager):
    global _lmstudio_manager
    _lmstudio_manager = manager


@system_router.get("/status")
async def system_status():
    if not _pipeline:
        return {"status": "not initialized"}

    # Get recording status from global config
    from server.main import app_config

    return {
        "status": "ready" if _pipeline.ready else "loading",
        "direction": _pipeline.direction,
        "mode": _pipeline.mode,
        "asr_auto_language": _pipeline.config.asr.auto_language,
        "speakers": [s.model_dump() for s in _pipeline.speaker_tracker.get_all()],
        "topic_summary": _pipeline.topic_summary,
        "idiom_patterns_loaded": _pipeline.idiom_scanner.count,
        "queue_depth": _pipeline._queue.qsize(),
        "llm_provider": _pipeline.translator.config.provider,
        "llm_model": _pipeline.translator.config.model,
        "recording_enabled": app_config.recording.enabled if app_config else False,
        "lmstudio_running": (await _lmstudio_manager.is_running()) if _lmstudio_manager else False,
        "lmstudio_models": _lmstudio_manager.get_loaded_models() if _lmstudio_manager else [],
    }


class DirectionRequest(BaseModel):
    direction: str


@system_router.post("/direction")
async def set_direction(req: DirectionRequest):
    if req.direction not in ("es_to_en", "en_to_es"):
        raise HTTPException(400, "Direction must be 'es_to_en' or 'en_to_es'")
    _pipeline.set_direction(req.direction)
    return {"direction": req.direction}


class ModeRequest(BaseModel):
    mode: str


@system_router.post("/mode")
async def set_mode(req: ModeRequest):
    if req.mode not in ("conversation", "classroom"):
        raise HTTPException(400, "Mode must be 'conversation' or 'classroom'")
    _pipeline.set_mode(req.mode)
    return {"mode": req.mode}


class ASRLanguageRequest(BaseModel):
    auto_language: bool


@system_router.post("/asr/language")
async def set_asr_language(req: ASRLanguageRequest):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    _pipeline.config.asr.auto_language = req.auto_language
    return {"auto_language": _pipeline.config.asr.auto_language}


class RenameRequest(BaseModel):
    name: str
    role: str | None = None


@system_router.put("/speakers/{speaker_id}")
async def rename_speaker(speaker_id: str, req: RenameRequest):
    result = _pipeline.speaker_tracker.rename(speaker_id, req.name)
    if req.role:
        _pipeline.speaker_tracker.set_role_hint(speaker_id, req.role)
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


# --- Session History Routes ---

session_router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@session_router.get("")
async def list_sessions(limit: int = 20, offset: int = 0):
    """List past sessions, most recent first."""
    db = await get_db()
    rows = await db.execute_fetchall(
        """SELECT s.*,
                  (SELECT COUNT(*) FROM exchanges e WHERE e.session_id = s.id) as exchange_count
           FROM sessions s
           ORDER BY s.started_at DESC
           LIMIT ? OFFSET ?""",
        (limit, offset),
    )
    return [dict(r) for r in rows]


@session_router.get("/{session_id}")
async def get_session(session_id: int):
    """Get a single session with its speakers."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    )
    if not rows:
        raise HTTPException(404, "Session not found")
    session = dict(rows[0])

    speakers = await db.execute_fetchall(
        "SELECT * FROM speakers WHERE session_id = ?", (session_id,)
    )
    session["speakers"] = [dict(s) for s in speakers]

    exchange_count = await db.execute_fetchall(
        "SELECT COUNT(*) as c FROM exchanges WHERE session_id = ?", (session_id,)
    )
    session["exchange_count"] = exchange_count[0]["c"]

    return session


@session_router.get("/{session_id}/exchanges")
async def get_session_exchanges(session_id: int, limit: int = 100, offset: int = 0):
    """Get exchanges for a session, in chronological order."""
    db = await get_db()

    # Verify session exists
    session_rows = await db.execute_fetchall(
        "SELECT id FROM sessions WHERE id = ?", (session_id,)
    )
    if not session_rows:
        raise HTTPException(404, "Session not found")

    rows = await db.execute_fetchall(
        """SELECT * FROM exchanges
           WHERE session_id = ?
           ORDER BY timestamp ASC
           LIMIT ? OFFSET ?""",
        (session_id, limit, offset),
    )
    exchanges = []
    for r in rows:
        ex = dict(r)
        # Parse correction_json back to dict if present
        if ex.get("correction_json"):
            try:
                ex["correction_detail"] = json.loads(ex["correction_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        exchanges.append(ex)
    return exchanges


@session_router.post("/{session_id}/save")
async def save_session(session_id: int, body: dict = {}):
    """Mark a session as explicitly saved with an optional name/note."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id FROM sessions WHERE id = ?", (session_id,)
    )
    if not rows:
        raise HTTPException(404, "Session not found")

    notes = (body.get("notes") or "").strip()
    await db.execute(
        "UPDATE sessions SET notes = ? WHERE id = ?",
        (notes or f"Saved {datetime.now().strftime('%Y-%m-%d %H:%M')}", session_id),
    )
    await db.commit()
    return {"status": "saved", "session_id": session_id}


@session_router.get("/{session_id}/export")
async def export_session(session_id: int):
    """Export a session transcript as plain text."""
    db = await get_db()
    session_rows = await db.execute_fetchall(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    )
    if not session_rows:
        raise HTTPException(404, "Session not found")
    session = dict(session_rows[0])

    rows = await db.execute_fetchall(
        """SELECT e.*, s.auto_label, s.custom_name
           FROM exchanges e
           LEFT JOIN speakers s ON e.speaker_id = s.id AND e.session_id = s.session_id
           WHERE e.session_id = ?
           ORDER BY e.timestamp ASC""",
        (session_id,),
    )

    lines = []
    lines.append(f"Habla Transcript - Session {session_id}")
    lines.append(f"Date: {session.get('started_at', 'Unknown')}")
    if session.get("notes"):
        lines.append(f"Notes: {session['notes']}")
    if session.get("topic_summary"):
        lines.append(f"Topic: {session['topic_summary']}")
    lines.append(f"Direction: {session.get('direction', 'es_to_en')}")
    lines.append("=" * 50)
    lines.append("")

    for r in rows:
        ex = dict(r)
        speaker = ex.get("custom_name") or ex.get("auto_label") or "Speaker"
        ts = ex.get("timestamp", "")
        src = ex.get("corrected_source") or ex.get("raw_transcript", "")
        tgt = ex.get("translation", "")
        lines.append(f"[{ts}] {speaker}:")
        lines.append(f"  {src}")
        lines.append(f"  -> {tgt}")
        lines.append("")

    return PlainTextResponse(
        "\n".join(lines),
        headers={"Content-Disposition": f'attachment; filename="habla_session_{session_id}.txt"'},
    )


# --- Idiom Pattern Routes ---

idiom_router = APIRouter(prefix="/api/idioms", tags=["idioms"])


def _generate_pattern(phrase: str) -> str:
    """Generate a regex pattern from an idiom phrase.

    Handles common Spanish verb conjugation flexibility and optional words.
    E.g., "tomar el pelo" -> r"tomar\\s+(el\\s+)?pelo"
    """
    words = phrase.strip().lower().split()
    if not words:
        return re.escape(phrase)

    parts = []
    # Common Spanish articles/prepositions that might be optional
    optional_words = {"el", "la", "los", "las", "un", "una", "de", "del", "en", "a", "al"}

    for i, word in enumerate(words):
        if i > 0 and word in optional_words:
            # Make articles/prepositions optional
            parts.append(f"({re.escape(word)}\\s+)?")
        elif i == 0 and len(word) > 3:
            # First word is often a verb - allow conjugation variants
            # Strip common infinitive endings and allow flexibility
            stem = word
            for ending in ("arse", "erse", "irse", "ar", "er", "ir"):
                if word.endswith(ending) and len(word) - len(ending) >= 2:
                    stem = word[:-len(ending)]
                    break
            if stem != word:
                parts.append(f"{re.escape(stem)}\\w*")
            else:
                parts.append(re.escape(word))
        else:
            parts.append(re.escape(word))

    return "\\s+".join(p for p in parts if p)


class IdiomPatternRequest(BaseModel):
    phrase: str
    canonical: str | None = None
    literal: str | None = None
    meaning: str
    region: str = "universal"
    frequency: str = "common"
    pattern: str | None = None  # optional manual regex override


@idiom_router.get("")
async def list_idiom_patterns(limit: int = 100, offset: int = 0):
    """List user-contributed idiom patterns from the database."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM idiom_patterns ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    return [dict(r) for r in rows]


@idiom_router.post("")
async def create_idiom_pattern(req: IdiomPatternRequest):
    """Create a new idiom pattern from a saved phrase. Auto-generates regex if not provided."""
    db = await get_db()
    canonical = req.canonical or req.phrase
    pattern = req.pattern or _generate_pattern(req.phrase)

    # Validate the regex compiles
    try:
        re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise HTTPException(400, f"Invalid regex pattern: {e}")

    # Check for duplicate canonical (case-insensitive)
    existing = await db.execute_fetchall(
        "SELECT id FROM idiom_patterns WHERE LOWER(canonical) = LOWER(?)",
        (canonical,),
    )
    if existing:
        raise HTTPException(409, f"Pattern for '{canonical}' already exists (id={existing[0]['id']})")

    # Also check against JSON-loaded patterns in the scanner
    if _pipeline:
        for p in _pipeline.idiom_scanner.patterns:
            if p.canonical.lower() == canonical.lower():
                raise HTTPException(409, f"Pattern for '{canonical}' already exists in pattern files")

    cursor = await db.execute(
        """INSERT INTO idiom_patterns (pattern, canonical, literal, meaning, region, frequency)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (pattern, canonical, req.literal or "", req.meaning, req.region, req.frequency),
    )
    await db.commit()

    # Reload the scanner so the new pattern is active immediately
    if _pipeline:
        await _pipeline.reload_idiom_patterns()

    return {
        "id": cursor.lastrowid,
        "pattern": pattern,
        "canonical": canonical,
        "total_patterns": _pipeline.idiom_scanner.count if _pipeline else None,
    }


@idiom_router.delete("/{pattern_id}")
async def delete_idiom_pattern(pattern_id: int):
    """Delete a user-contributed idiom pattern."""
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM idiom_patterns WHERE id = ?", (pattern_id,)
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(404, "Pattern not found")

    # Reload scanner
    if _pipeline:
        await _pipeline.reload_idiom_patterns()

    return {"deleted": True}


# --- LLM Provider Routes ---

llm_router = APIRouter(prefix="/api/llm", tags=["llm"])


@llm_router.get("/current")
async def llm_current():
    """Return current LLM provider, model, and metrics."""
    if not _pipeline:
        return {"status": "not initialized"}
    t = _pipeline.translator
    result = {
        "provider": t.config.provider,
        "model": t.config.model,
        "quick_model": t.config.quick_model,
        "metrics": t.metrics,
    }
    if t.config.provider == "openai":
        result["costs"] = t.costs
    return result


@llm_router.get("/providers")
async def llm_providers():
    """Probe all configured providers and return connection status + models."""
    if not _pipeline:
        return {"providers": []}

    cfg = _pipeline.translator.config
    providers = []

    # Ollama
    ollama_info = {"name": "ollama", "url": cfg.ollama_url, "status": "unknown", "models": []}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{cfg.ollama_url}/api/tags")
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            ollama_info["status"] = "ok"
            ollama_info["models"] = models
    except httpx.ConnectError:
        ollama_info["status"] = "down"
    except Exception:
        ollama_info["status"] = "error"
    providers.append(ollama_info)

    # LM Studio
    lms_info = {"name": "lmstudio", "url": cfg.lmstudio_url, "status": "unknown", "models": []}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{cfg.lmstudio_url}/v1/models")
            resp.raise_for_status()
            models = [m.get("id", "") for m in resp.json().get("data", [])]
            lms_info["status"] = "ok"
            lms_info["models"] = models
    except httpx.ConnectError:
        lms_info["status"] = "down"
    except Exception:
        lms_info["status"] = "error"
    providers.append(lms_info)

    # OpenAI
    openai_info = {"name": "openai", "status": "unknown", "models": []}
    if cfg.openai_api_key:
        openai_info["status"] = "ok"
        openai_info["models"] = ["gpt-5", "gpt-5-mini", "gpt-4o-mini", "gpt-5-nano"]
    else:
        openai_info["status"] = "no_api_key"
    providers.append(openai_info)

    return {"providers": providers, "active": cfg.provider}


@llm_router.get("/models")
async def llm_models(provider: str):
    """List models for a specific provider."""
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    cfg = _pipeline.translator.config

    if provider == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{cfg.ollama_url}/api/tags")
                resp.raise_for_status()
                return {"models": [m.get("name", "") for m in resp.json().get("models", [])]}
        except Exception as e:
            raise HTTPException(502, f"Cannot reach Ollama: {e}")

    elif provider == "lmstudio":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{cfg.lmstudio_url}/v1/models")
                resp.raise_for_status()
                return {"models": [m.get("id", "") for m in resp.json().get("data", [])]}
        except Exception as e:
            raise HTTPException(502, f"Cannot reach LM Studio: {e}")

    elif provider == "openai":
        if not cfg.openai_api_key:
            raise HTTPException(400, "OPENAI_API_KEY not configured")
        return {"models": ["gpt-5", "gpt-5-mini", "gpt-4o-mini", "gpt-5-nano"]}

    else:
        raise HTTPException(400, f"Unknown provider: {provider}")


class LLMSelectRequest(BaseModel):
    provider: str
    model: str = ""
    url: str = ""
    quick_model: str = ""


@llm_router.post("/select")
async def llm_select(req: LLMSelectRequest):
    """Switch LLM provider and model at runtime."""
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    if req.provider not in ("ollama", "lmstudio", "openai"):
        raise HTTPException(400, "Provider must be 'ollama', 'lmstudio', or 'openai'")
    if req.provider == "openai" and not _pipeline.translator.config.openai_api_key:
        raise HTTPException(400, "OPENAI_API_KEY not configured")
    if req.model:
        cfg = _pipeline.translator.config
        if req.provider == "ollama":
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{cfg.ollama_url}/api/tags")
                    resp.raise_for_status()
                    models = [m.get("name", "") for m in resp.json().get("models", [])]
                if req.model not in models:
                    raise HTTPException(400, f"Ollama model not found: {req.model}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"Cannot verify Ollama model: {e}")
        if req.provider == "lmstudio":
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{cfg.lmstudio_url}/v1/models")
                    resp.raise_for_status()
                    models = [m.get("id", "") for m in resp.json().get("data", [])]
                if req.model not in models:
                    raise HTTPException(400, f"LM Studio model not found: {req.model}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"Cannot verify LM Studio model: {e}")

    if req.quick_model:
        _pipeline.translator.config.quick_model = req.quick_model
    _pipeline.translator.switch_provider(req.provider, req.model, req.url)
    return {
        "provider": _pipeline.translator.config.provider,
        "model": _pipeline.translator.config.model,
        "quick_model": _pipeline.translator.config.quick_model,
    }


@llm_router.get("/costs")
async def llm_costs():
    """Get OpenAI API cost tracking data."""
    if not _pipeline:
        return {"costs": None}
    t = _pipeline.translator
    if t.config.provider != "openai":
        return {"provider": t.config.provider, "cost_tracking": "Free (local)"}
    return {
        "provider": "openai",
        "model": t.config.openai_model,
        "costs": t.costs,
    }


# --- LM Studio management routes ---

lmstudio_router = APIRouter(prefix="/api/lmstudio", tags=["lmstudio"])


@lmstudio_router.get("/status")
async def lmstudio_status():
    """Return LM Studio running state and loaded model names."""
    if _lmstudio_manager is None:
        return {"running": False, "models": [], "note": "lmstudio provider not active"}
    running = await _lmstudio_manager.is_running()
    return {
        "running": running,
        "models": _lmstudio_manager.get_loaded_models(),
    }


@lmstudio_router.post("/restart")
async def lmstudio_restart():
    """Restart LM Studio and reload configured models."""
    if _lmstudio_manager is None:
        raise HTTPException(503, "LM Studio manager not active")
    asyncio.create_task(_lmstudio_manager.restart())
    return {"status": "restarting"}


@lmstudio_router.get("/models")
async def lmstudio_models():
    """Proxy /v1/models from LM Studio (shows all currently loaded models)."""
    if _lmstudio_manager is None:
        raise HTTPException(503, "LM Studio manager not active")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{_pipeline.translator.config.lmstudio_url}/v1/models"
                if _pipeline else "http://localhost:1234/v1/models"
            )
        if resp.is_success:
            return resp.json()
        raise HTTPException(502, f"LM Studio returned HTTP {resp.status_code}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Cannot reach LM Studio: {exc}")
