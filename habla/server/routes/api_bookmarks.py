"""Conversation bookmark routes.

Toggle bookmarks on exchange cards for quick review later. Bookmarks persist
to the exchanges table and can be browsed across all sessions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.db.database import get_db

logger = logging.getLogger("habla.bookmarks")

bookmarks_router = APIRouter(prefix="/api", tags=["bookmarks"])


class BookmarkBody(BaseModel):
    note: Optional[str] = None


@bookmarks_router.post("/exchanges/{exchange_id}/bookmark")
async def toggle_bookmark(exchange_id: int, body: BookmarkBody | None = None):
    """Toggle bookmark on an exchange. Optionally attach a note."""
    db = await get_db()

    rows = await db.execute_fetchall(
        "SELECT id, bookmarked FROM exchanges WHERE id = ?", (exchange_id,)
    )
    if not rows:
        raise HTTPException(404, "Exchange not found")

    current = bool(rows[0]["bookmarked"])
    new_state = not current
    note = body.note if body else None

    if new_state:
        await db.execute(
            "UPDATE exchanges SET bookmarked = 1, bookmark_note = ? WHERE id = ?",
            (note, exchange_id),
        )
    else:
        await db.execute(
            "UPDATE exchanges SET bookmarked = 0, bookmark_note = NULL WHERE id = ?",
            (exchange_id,),
        )
    await db.commit()

    logger.info(f"Exchange {exchange_id} bookmark={'on' if new_state else 'off'}")
    return {"id": exchange_id, "bookmarked": new_state}


@bookmarks_router.get("/bookmarks/recent")
async def get_recent_bookmarks(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Return recent bookmarked exchanges across all sessions."""
    db = await get_db()

    rows = await db.execute_fetchall(
        """SELECT e.*, s.started_at AS session_started, s.name AS session_name
           FROM exchanges e
           JOIN sessions s ON s.id = e.session_id
           WHERE e.bookmarked = 1
           ORDER BY e.timestamp DESC
           LIMIT ? OFFSET ?""",
        (limit, offset),
    )
    results = []
    for r in rows:
        ex = dict(r)
        ex["has_audio"] = bool(ex.get("audio_path"))
        results.append(ex)

    count_row = await db.execute_fetchall(
        "SELECT COUNT(*) AS cnt FROM exchanges WHERE bookmarked = 1"
    )
    total = count_row[0]["cnt"] if count_row else 0

    return {"bookmarks": results, "total": total}


@bookmarks_router.get("/sessions/{session_id}/bookmarks")
async def get_session_bookmarks(session_id: int):
    """Return bookmarked exchanges for a specific session."""
    db = await get_db()

    session_rows = await db.execute_fetchall(
        "SELECT id FROM sessions WHERE id = ?", (session_id,)
    )
    if not session_rows:
        raise HTTPException(404, "Session not found")

    rows = await db.execute_fetchall(
        """SELECT * FROM exchanges
           WHERE session_id = ? AND bookmarked = 1
           ORDER BY timestamp ASC""",
        (session_id,),
    )
    results = []
    for r in rows:
        ex = dict(r)
        ex["has_audio"] = bool(ex.get("audio_path"))
        results.append(ex)

    return results
