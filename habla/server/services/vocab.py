"""Vocab service — CRUD, spaced repetition, and export."""

import csv
import io
import json
from datetime import datetime, timedelta
from server.db.database import get_db
from server.models.schemas import VocabItem, FlaggedPhrase


class VocabService:
    """Manages vocabulary items with spaced repetition scheduling."""

    async def save_from_phrase(
        self,
        phrase: FlaggedPhrase,
        exchange_id: int | None = None,
        speaker_id: str | None = None,
        source_sentence: str = "",
    ) -> int:
        """Save a flagged phrase as a vocab item. Returns the vocab ID."""
        db = await get_db()

        # Check if term already exists — bump encounter count if so
        existing = await db.execute_fetchall(
            "SELECT id, times_encountered FROM vocab WHERE LOWER(term) = LOWER(?)",
            (phrase.phrase,),
        )
        if existing:
            row = existing[0]
            await db.execute(
                "UPDATE vocab SET times_encountered = ?, source_sentence = COALESCE(?, source_sentence) WHERE id = ?",
                (row["times_encountered"] + 1, source_sentence or None, row["id"]),
            )
            await db.commit()
            return row["id"]

        # Insert new
        cursor = await db.execute(
            """INSERT INTO vocab
               (exchange_id, speaker_id, term, literal, meaning, category,
                source_sentence, region, save_worthy, next_review)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exchange_id,
                speaker_id,
                phrase.phrase,
                phrase.literal,
                phrase.meaning,
                phrase.type,
                source_sentence,
                phrase.region or "universal",
                phrase.save_worthy,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()
        return cursor.lastrowid

    async def get_all(
        self, limit: int = 50, offset: int = 0, category: str | None = None
    ) -> list[dict]:
        """Get all vocab items, optionally filtered."""
        db = await get_db()
        query = "SELECT * FROM vocab WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = await db.execute_fetchall(query, params)
        return [dict(r) for r in rows]

    async def get_due_for_review(self, limit: int = 20) -> list[dict]:
        """Get vocab items due for spaced repetition review."""
        db = await get_db()
        now = datetime.utcnow().isoformat()
        rows = await db.execute_fetchall(
            """SELECT * FROM vocab
               WHERE next_review <= ? OR next_review IS NULL
               ORDER BY next_review ASC LIMIT ?""",
            (now, limit),
        )
        return [dict(r) for r in rows]

    async def record_review(self, vocab_id: int, quality: int) -> dict:
        """
        Record a spaced repetition review using SM-2 algorithm.
        quality: 0-5 (0=forgot, 3=hard, 5=easy)
        """
        db = await get_db()
        rows = await db.execute_fetchall(
            "SELECT * FROM vocab WHERE id = ?", (vocab_id,)
        )
        if not rows:
            return {"error": "not found"}

        item = dict(rows[0])
        ease = item["ease_factor"]
        interval = item["interval_days"]
        reps = item["repetitions"]

        # SM-2 algorithm
        if quality < 3:
            # Failed — reset
            reps = 0
            interval = 1
        else:
            if reps == 0:
                interval = 1
            elif reps == 1:
                interval = 6
            else:
                interval = int(interval * ease)
            reps += 1

        # Update ease factor
        ease = max(1.3, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))

        next_review = (datetime.utcnow() + timedelta(days=interval)).isoformat()

        await db.execute(
            """UPDATE vocab SET ease_factor = ?, interval_days = ?,
               repetitions = ?, next_review = ? WHERE id = ?""",
            (ease, interval, reps, next_review, vocab_id),
        )
        await db.commit()

        return {
            "id": vocab_id,
            "ease_factor": ease,
            "interval_days": interval,
            "next_review": next_review,
            "repetitions": reps,
        }

    async def delete(self, vocab_id: int) -> bool:
        db = await get_db()
        cursor = await db.execute("DELETE FROM vocab WHERE id = ?", (vocab_id,))
        await db.commit()
        return cursor.rowcount > 0

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across vocab."""
        db = await get_db()
        rows = await db.execute_fetchall(
            """SELECT v.* FROM vocab v
               JOIN vocab_fts fts ON v.id = fts.rowid
               WHERE vocab_fts MATCH ?
               LIMIT ?""",
            (query, limit),
        )
        return [dict(r) for r in rows]

    async def export_anki_csv(self) -> str:
        """Export all vocab as Anki-compatible CSV."""
        db = await get_db()
        rows = await db.execute_fetchall(
            "SELECT * FROM vocab ORDER BY created_at"
        )

        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t")

        for row in rows:
            row = dict(row)
            front = row["term"]
            if row["literal"]:
                front += f" (lit: {row['literal']})"

            back = row["meaning"]
            if row["source_sentence"]:
                back += f"\n\nExample: {row['source_sentence']}"

            tags = f"habla {row['category']} {row.get('region', 'universal')}"

            writer.writerow([front, back, tags])

        return output.getvalue()

    async def export_json(self) -> list[dict]:
        """Export all vocab as JSON."""
        return await self.get_all(limit=10000)

    async def get_stats(self) -> dict:
        """Get vocab statistics."""
        db = await get_db()
        total = await db.execute_fetchall("SELECT COUNT(*) as c FROM vocab")
        due = await db.execute_fetchall(
            "SELECT COUNT(*) as c FROM vocab WHERE next_review <= ?",
            (datetime.utcnow().isoformat(),),
        )
        by_cat = await db.execute_fetchall(
            "SELECT category, COUNT(*) as c FROM vocab GROUP BY category"
        )

        return {
            "total": total[0]["c"],
            "due_for_review": due[0]["c"],
            "by_category": {r["category"]: r["c"] for r in by_cat},
        }
