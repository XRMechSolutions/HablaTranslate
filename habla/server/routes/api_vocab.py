"""Vocab CRUD, review, search, and export routes."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from server.db.database import get_db
from server.services.vocab import VocabService


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
    """Create a vocab item or increment encounter count if duplicate.

    Returns 200 {id, duplicate: false} on new insert.
    Returns 200 {id, duplicate: true, times_encountered} if term exists (case-insensitive).
    Returns 400 if term is blank.
    Side effects: inserts or updates vocab row, commits DB.
    """
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

    from datetime import datetime, UTC
    cursor = await db.execute(
        """INSERT INTO vocab (term, literal, meaning, category, source_sentence, region, next_review)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (req.term.strip(), req.literal or "", req.meaning, req.category,
         req.source_sentence, req.region, datetime.now(UTC).isoformat()),
    )
    await db.commit()
    return {"id": cursor.lastrowid, "duplicate": False}


@vocab_router.get("")
async def list_vocab(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    category: str | None = None,
):
    """List vocab items with optional category filter.

    Returns 200 with list of vocab dicts. 422 if limit/offset out of bounds.
    """
    return await _vocab_service.get_all(limit=limit, offset=offset, category=category)


@vocab_router.get("/due")
async def vocab_due(limit: int = Query(default=20, ge=1, le=200)):
    """Return vocab items due for spaced repetition review.

    Returns 200 with list of vocab dicts where next_review <= now.
    """
    return await _vocab_service.get_due_for_review(limit=limit)


@vocab_router.get("/stats")
async def vocab_stats():
    """Return aggregate vocab statistics (total, due_for_review, by_category counts)."""
    return await _vocab_service.get_stats()


@vocab_router.get("/search")
async def search_vocab(q: str, limit: int = Query(default=20, ge=1, le=200)):
    """Full-text search across vocab terms and meanings via FTS5.

    Returns 200 with matching vocab list. Returns 400 if query syntax is invalid.
    """
    try:
        return await _vocab_service.search(q, limit=limit)
    except ValueError as e:
        raise HTTPException(400, str(e))


class ReviewRequest(BaseModel):
    quality: int  # 0-5


@vocab_router.post("/{vocab_id}/review")
async def review_vocab(vocab_id: int, req: ReviewRequest):
    """Record a spaced repetition review using SM-2 algorithm.

    Returns 200 with updated ease_factor, interval_days, next_review, repetitions.
    Returns 400 if quality not in 0-5. Returns 404 if vocab_id not found.
    Side effects: updates vocab SM-2 fields and commits DB.
    """
    if not 0 <= req.quality <= 5:
        raise HTTPException(400, "Quality must be 0-5")
    result = await _vocab_service.record_review(vocab_id, req.quality)
    if "error" in result:
        raise HTTPException(404, "Vocab item not found")
    return result


@vocab_router.delete("/{vocab_id}")
async def delete_vocab(vocab_id: int):
    """Delete a vocab item. Returns 200 {deleted: true} or 404 if not found."""
    deleted = await _vocab_service.delete(vocab_id)
    if not deleted:
        raise HTTPException(404, "Vocab item not found")
    return {"deleted": True}


@vocab_router.get("/export/anki")
async def export_anki():
    """Export all vocab as Anki-compatible TSV (front/back/tags). Returns text/tab-separated-values."""
    csv_data = await _vocab_service.export_anki_csv()
    return PlainTextResponse(csv_data, media_type="text/tab-separated-values",
                             headers={"Content-Disposition": "attachment; filename=habla-vocab.tsv"})


@vocab_router.get("/export/json")
async def export_json():
    """Export all vocab as JSON array."""
    return await _vocab_service.export_json()
