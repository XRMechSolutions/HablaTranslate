"""Idiom pattern CRUD routes."""

import re

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.db.database import get_db
import server.routes._state as _state


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
async def list_idiom_patterns(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List user-contributed idiom patterns from the database (DB-stored only, not JSON files)."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM idiom_patterns ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    return [dict(r) for r in rows]


@idiom_router.post("")
async def create_idiom_pattern(req: IdiomPatternRequest):
    """Create a new idiom pattern from a saved phrase. Auto-generates regex if not provided.

    Returns 200 with id, pattern, canonical, total_patterns.
    Returns 400 if manual regex is invalid. Returns 409 if canonical already exists (DB or JSON files).
    Side effects: inserts DB row, reloads idiom scanner so pattern is immediately active.
    """
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
    if _state._pipeline:
        for p in _state._pipeline.idiom_scanner.patterns:
            if p.canonical.lower() == canonical.lower():
                raise HTTPException(409, f"Pattern for '{canonical}' already exists in pattern files")

    cursor = await db.execute(
        """INSERT INTO idiom_patterns (pattern, canonical, literal, meaning, region, frequency)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (pattern, canonical, req.literal or "", req.meaning, req.region, req.frequency),
    )
    await db.commit()

    # Reload the scanner so the new pattern is active immediately
    if _state._pipeline:
        await _state._pipeline.reload_idiom_patterns()

    return {
        "id": cursor.lastrowid,
        "pattern": pattern,
        "canonical": canonical,
        "total_patterns": _state._pipeline.idiom_scanner.count if _state._pipeline else None,
    }


@idiom_router.delete("/{pattern_id}")
async def delete_idiom_pattern(pattern_id: int):
    """Delete a user-contributed idiom pattern. Returns 404 if not found.

    Side effects: deletes DB row, reloads idiom scanner.
    """
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM idiom_patterns WHERE id = ?", (pattern_id,)
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(404, "Pattern not found")

    # Reload scanner
    if _state._pipeline:
        await _state._pipeline.reload_idiom_patterns()

    return {"deleted": True}
