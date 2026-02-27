"""Session history routes: list, get, exchanges, save, export, delete."""

import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from server.db.database import get_db


session_router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@session_router.get("")
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List past sessions, most recent first. Includes exchange_count per session."""
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


@session_router.get("/search")
async def search_sessions(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=30, ge=1, le=100),
):
    """Search across session history by matching exchange text (source, translation).

    Returns sessions containing matching exchanges, most recent first,
    with matched exchange snippets.
    """
    db = await get_db()
    pattern = f"%{q}%"
    rows = await db.execute_fetchall(
        """SELECT DISTINCT s.*,
                  (SELECT COUNT(*) FROM exchanges e2 WHERE e2.session_id = s.id) as exchange_count
           FROM sessions s
           JOIN exchanges e ON e.session_id = s.id
           WHERE e.raw_transcript LIKE ? COLLATE NOCASE
              OR e.corrected_source LIKE ? COLLATE NOCASE
              OR e.translation LIKE ? COLLATE NOCASE
           ORDER BY s.started_at DESC
           LIMIT ?""",
        (pattern, pattern, pattern, limit),
    )
    results = []
    for r in rows:
        session = dict(r)
        # Fetch matching exchange snippets for this session
        matches = await db.execute_fetchall(
            """SELECT raw_transcript, corrected_source, translation, speaker_id, timestamp
               FROM exchanges
               WHERE session_id = ?
                 AND (raw_transcript LIKE ? COLLATE NOCASE
                   OR corrected_source LIKE ? COLLATE NOCASE
                   OR translation LIKE ? COLLATE NOCASE)
               ORDER BY timestamp ASC
               LIMIT 5""",
            (session["id"], pattern, pattern, pattern),
        )
        session["matched_exchanges"] = [dict(m) for m in matches]
        results.append(session)
    return results


@session_router.get("/{session_id}")
async def get_session(session_id: int):
    """Get a single session with speakers list and exchange_count. Returns 404 if not found."""
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
async def get_session_exchanges(
    session_id: int,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """Get exchanges for a session in chronological order. Returns 404 if session not found.

    Parses correction_json into correction_detail dict when present.
    """
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
        ex["has_audio"] = bool(ex.get("audio_path"))
        exchanges.append(ex)
    return exchanges


@session_router.post("/{session_id}/save")
async def save_session(session_id: int, body: dict | None = None):
    """Mark a session as explicitly saved with an optional name/note. Returns 404 if not found.

    Side effects: updates session notes in DB.
    """
    body = body or {}
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
    """Export a session transcript as plain text with speaker attribution. Returns 404 if not found."""
    from fastapi.responses import PlainTextResponse

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


@session_router.get("/{session_id}/exchanges/{exchange_id}/audio")
async def get_exchange_audio(session_id: int, exchange_id: int):
    """Serve the audio clip for a specific exchange. Returns 404 if not found."""
    from fastapi.responses import FileResponse

    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT audio_path FROM exchanges WHERE id = ? AND session_id = ?",
        (exchange_id, session_id),
    )
    if not rows:
        raise HTTPException(404, "Exchange not found")
    audio_path = rows[0]["audio_path"]
    if not audio_path:
        raise HTTPException(404, "No audio clip for this exchange")
    path = Path(audio_path)
    if not path.exists():
        raise HTTPException(404, "Audio file missing")
    return FileResponse(path, media_type="audio/wav", filename=f"exchange_{exchange_id}.wav")


@session_router.delete("/{session_id}")
async def delete_session(session_id: int):
    """Delete a session and all associated data (metrics, exchanges, speakers). Returns 404 if not found.

    Side effects: cascading delete across quality_metrics, exchanges, speakers, sessions tables.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id FROM sessions WHERE id = ?", (session_id,)
    )
    if not rows:
        raise HTTPException(404, "Session not found")

    # Clean up audio clips directory for this session
    clips_dir = Path("data/audio/clips") / str(session_id)
    if clips_dir.exists():
        import shutil
        shutil.rmtree(clips_dir, ignore_errors=True)

    await db.execute("DELETE FROM quality_metrics WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM exchanges WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM speakers WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    await db.commit()
    return {"deleted": True, "session_id": session_id}
