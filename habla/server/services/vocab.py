"""Vocab service — CRUD, spaced repetition, and export."""

import csv
import io
import json
from datetime import datetime, timedelta, UTC
from server.db.database import get_db
from server.models.schemas import VocabItem, FlaggedPhrase


class VocabService:
    """Manages vocabulary items with spaced repetition scheduling.

    Error contract:
    - record_review returns {"error": "not found"} for missing IDs (route maps to 404).
    - search raises ValueError on malformed FTS queries (route maps to 400).
    - delete returns False for missing IDs (route maps to 404).
    - All other methods assume valid inputs; DB errors propagate as exceptions.

    Side effects: all mutating methods call get_db() and commit within the call.
    """

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
                datetime.now(UTC).isoformat(),
            ),
        )
        await db.commit()
        return cursor.lastrowid

    async def get_by_id(self, vocab_id: int) -> dict | None:
        """Get a single vocab item by ID. Returns None if not found."""
        db = await get_db()
        rows = await db.execute_fetchall(
            "SELECT * FROM vocab WHERE id = ?", (vocab_id,)
        )
        return dict(rows[0]) if rows else None

    async def update(self, vocab_id: int, **fields) -> dict | None:
        """Update specific fields on a vocab item. Returns updated item or None if not found."""
        db = await get_db()
        rows = await db.execute_fetchall(
            "SELECT * FROM vocab WHERE id = ?", (vocab_id,)
        )
        if not rows:
            return None

        allowed = {"term", "meaning", "literal", "category", "source_sentence", "region", "notes"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return dict(rows[0])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [vocab_id]
        await db.execute(f"UPDATE vocab SET {set_clause} WHERE id = ?", values)
        await db.commit()

        updated = await db.execute_fetchall(
            "SELECT * FROM vocab WHERE id = ?", (vocab_id,)
        )
        return dict(updated[0])

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
        now = datetime.now(UTC).isoformat()
        rows = await db.execute_fetchall(
            """SELECT * FROM vocab
               WHERE next_review <= ? OR next_review IS NULL
               ORDER BY next_review ASC LIMIT ?""",
            (now, limit),
        )
        return [dict(r) for r in rows]

    # Mature card threshold — cards with interval above this are "mature"
    MATURE_INTERVAL_DAYS = 21

    async def record_review(self, vocab_id: int, quality: int) -> dict:
        """
        Record a spaced repetition review using SM-2 algorithm.
        quality: 0-5 (0=forgot, 3=hard, 5=easy)

        All next_review timestamps are stored as UTC ISO strings.
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
        lapse_count = item.get("lapse_count", 0) or 0

        # SM-2 algorithm
        if quality < 3:
            # Failed — track lapse if card was previously learned (reps > 0)
            if reps > 0:
                lapse_count += 1
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

        # Update ease factor — capped to [1.3, 5.0]
        ease = min(5.0, max(1.3, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))))

        next_review = (datetime.now(UTC) + timedelta(days=interval)).isoformat()

        await db.execute(
            """UPDATE vocab SET ease_factor = ?, interval_days = ?,
               repetitions = ?, next_review = ?, lapse_count = ? WHERE id = ?""",
            (ease, interval, reps, next_review, lapse_count, vocab_id),
        )
        await db.commit()

        return {
            "id": vocab_id,
            "ease_factor": ease,
            "interval_days": interval,
            "next_review": next_review,
            "repetitions": reps,
            "lapse_count": lapse_count,
            "is_mature": interval >= self.MATURE_INTERVAL_DAYS,
        }

    async def delete(self, vocab_id: int) -> bool:
        """Delete a vocab item by ID. Returns True if deleted, False if not found."""
        db = await get_db()
        cursor = await db.execute("DELETE FROM vocab WHERE id = ?", (vocab_id,))
        await db.commit()
        return cursor.rowcount > 0

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across vocab. Raises ValueError on malformed FTS syntax."""
        import sqlite3
        db = await get_db()
        try:
            rows = await db.execute_fetchall(
                """SELECT v.* FROM vocab v
                   JOIN vocab_fts fts ON v.id = fts.rowid
                   WHERE vocab_fts MATCH ?
                   LIMIT ?""",
                (query, limit),
            )
        except sqlite3.OperationalError as e:
            raise ValueError(f"Invalid search query: {e}")
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

    async def get_review_session(self, session_size: int = 20) -> dict:
        """Plan a review session: 70% due, 20% new (never reviewed), 10% struggling.

        Struggling = lapse_count > 0 or (repetitions == 0 and times_encountered > 1).
        New = never reviewed (repetitions == 0 and next_review is past/null, no lapses).
        Due = next_review <= now and repetitions > 0.

        Returns {"due": [...], "new": [...], "struggling": [...], "total": int}.
        All next_review timestamps are UTC.
        """
        db = await get_db()
        now = datetime.now(UTC).isoformat()

        due_count = int(session_size * 0.70)
        new_count = int(session_size * 0.20)
        struggling_count = session_size - due_count - new_count  # remainder ~10%

        # Due items: reviewed before (reps > 0), next_review has passed
        due_rows = await db.execute_fetchall(
            """SELECT * FROM vocab
               WHERE repetitions > 0 AND next_review <= ?
               ORDER BY next_review ASC LIMIT ?""",
            (now, due_count),
        )
        due = [dict(r) for r in due_rows]

        # Struggling items: have lapsed, or stuck at reps=0 after multiple encounters
        struggling_rows = await db.execute_fetchall(
            """SELECT * FROM vocab
               WHERE (lapse_count > 0 AND next_review <= ?)
                  OR (repetitions = 0 AND times_encountered > 1 AND (next_review <= ? OR next_review IS NULL))
               ORDER BY lapse_count DESC, next_review ASC LIMIT ?""",
            (now, now, struggling_count),
        )
        struggling = [dict(r) for r in struggling_rows]
        struggling_ids = {r["id"] for r in struggling}

        # New items: never reviewed, not struggling
        new_rows = await db.execute_fetchall(
            """SELECT * FROM vocab
               WHERE repetitions = 0 AND lapse_count = 0
                 AND (next_review <= ? OR next_review IS NULL)
               ORDER BY created_at ASC LIMIT ?""",
            (now, new_count),
        )
        new = [dict(r) for r in new_rows if r["id"] not in struggling_ids]

        # Fill remaining slots if any category came up short
        used_ids = {r["id"] for r in due} | struggling_ids | {r["id"] for r in new}
        total_planned = len(due) + len(new) + len(struggling)
        if total_planned < session_size:
            fill_limit = session_size - total_planned
            fill_rows = await db.execute_fetchall(
                """SELECT * FROM vocab
                   WHERE next_review <= ? OR next_review IS NULL
                   ORDER BY next_review ASC LIMIT ?""",
                (now, fill_limit + len(used_ids)),
            )
            fill = [dict(r) for r in fill_rows if r["id"] not in used_ids][:fill_limit]
            due.extend(fill)

        return {
            "due": due,
            "new": new,
            "struggling": struggling,
            "total": len(due) + len(new) + len(struggling),
        }

    async def get_stats(self) -> dict:
        """Get vocab statistics."""
        db = await get_db()
        total = await db.execute_fetchall("SELECT COUNT(*) as c FROM vocab")
        due = await db.execute_fetchall(
            "SELECT COUNT(*) as c FROM vocab WHERE next_review <= ?",
            (datetime.now(UTC).isoformat(),),
        )
        by_cat = await db.execute_fetchall(
            "SELECT category, COUNT(*) as c FROM vocab GROUP BY category"
        )

        return {
            "total": total[0]["c"],
            "due_for_review": due[0]["c"],
            "by_category": {r["category"]: r["c"] for r in by_cat},
        }
