"""Async SQLite database for Habla."""

import logging
import aiosqlite
from pathlib import Path

logger = logging.getLogger("habla.db")

_db: aiosqlite.Connection | None = None


async def init_db(db_path: Path) -> aiosqlite.Connection:
    """Initialize database and create tables."""
    global _db
    _db = await aiosqlite.connect(str(db_path))
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")
    await _db.execute("PRAGMA busy_timeout=30000")
    await _create_tables(_db)
    await _db.commit()

    # Integrity check on startup — warns but does not abort
    rows = await _db.execute_fetchall("PRAGMA integrity_check")
    result = rows[0][0] if rows else "unknown"
    if result != "ok":
        logger.warning("Database integrity check FAILED: %s", result)
    else:
        logger.debug("Database integrity check passed")

    return _db


async def get_db() -> aiosqlite.Connection:
    """Get the database connection."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    return _db


async def close_db():
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None


async def _create_tables(db: aiosqlite.Connection):
    """Create all tables if they don't exist."""

    await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            ended_at        DATETIME,
            mode            TEXT DEFAULT 'conversation',
            direction       TEXT DEFAULT 'es_to_en',
            speaker_count   INTEGER DEFAULT 0,
            topic_summary   TEXT,
            notes           TEXT
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            id              TEXT NOT NULL,
            session_id      INTEGER NOT NULL REFERENCES sessions(id),
            auto_label      TEXT NOT NULL,
            custom_name     TEXT,
            role_hint       TEXT,
            color           TEXT NOT NULL,
            utterance_count INTEGER DEFAULT 0,
            first_seen_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id, session_id)
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS exchanges (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER NOT NULL REFERENCES sessions(id),
            speaker_id      TEXT,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
            direction       TEXT NOT NULL,
            raw_transcript  TEXT NOT NULL,
            corrected_source TEXT,
            translation     TEXT NOT NULL,
            confidence      REAL,
            is_correction   BOOLEAN DEFAULT FALSE,
            correction_json TEXT,
            processing_ms   INTEGER,
            audio_path      TEXT
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange_id         INTEGER REFERENCES exchanges(id),
            speaker_id          TEXT,
            term                TEXT NOT NULL,
            literal             TEXT,
            meaning             TEXT NOT NULL,
            category            TEXT DEFAULT 'idiom',
            source_sentence     TEXT,
            region              TEXT DEFAULT 'universal',
            save_worthy         BOOLEAN DEFAULT TRUE,
            created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
            ease_factor         REAL DEFAULT 2.5,
            interval_days       INTEGER DEFAULT 1,
            next_review         DATETIME,
            repetitions         INTEGER DEFAULT 0,
            times_encountered   INTEGER DEFAULT 1
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS idiom_patterns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern     TEXT NOT NULL,
            canonical   TEXT NOT NULL,
            literal     TEXT,
            meaning     TEXT NOT NULL,
            region      TEXT DEFAULT 'universal',
            frequency   TEXT DEFAULT 'common',
            examples    TEXT
        )
    """)

    # Quality metrics for auto-tuning
    await db.execute("""
        CREATE TABLE IF NOT EXISTS quality_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER NOT NULL REFERENCES sessions(id),
            segment_id      INTEGER,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
            confidence      REAL,
            audio_rms       REAL,
            duration_seconds REAL,
            speaker_id      TEXT,
            clipped_onset   BOOLEAN DEFAULT FALSE,
            processing_time_ms INTEGER,
            vad_threshold   REAL,
            model_name      TEXT
        )
    """)

    # Full-text search on vocab
    await db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vocab_fts
        USING fts5(term, meaning, literal, category,
                   content='vocab', content_rowid='id')
    """)

    # FTS5 sync triggers — keep vocab_fts in sync with vocab table
    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS vocab_ai AFTER INSERT ON vocab BEGIN
            INSERT INTO vocab_fts(rowid, term, meaning, literal, category)
            VALUES (new.id, new.term, new.meaning, new.literal, new.category);
        END
    """)
    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS vocab_ad AFTER DELETE ON vocab BEGIN
            INSERT INTO vocab_fts(vocab_fts, rowid, term, meaning, literal, category)
            VALUES ('delete', old.id, old.term, old.meaning, old.literal, old.category);
        END
    """)
    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS vocab_au AFTER UPDATE ON vocab BEGIN
            INSERT INTO vocab_fts(vocab_fts, rowid, term, meaning, literal, category)
            VALUES ('delete', old.id, old.term, old.meaning, old.literal, old.category);
            INSERT INTO vocab_fts(rowid, term, meaning, literal, category)
            VALUES (new.id, new.term, new.meaning, new.literal, new.category);
        END
    """)

    # Rebuild FTS index only if needed (first run or pre-trigger data)
    fts_row = await db.execute_fetchall("SELECT COUNT(*) FROM vocab_fts")
    fts_count = fts_row[0][0] if fts_row else 0
    if fts_count == 0:
        vocab_row = await db.execute_fetchall("SELECT COUNT(*) FROM vocab")
        vocab_count = vocab_row[0][0] if vocab_row else 0
        if vocab_count > 0:
            await db.execute("INSERT INTO vocab_fts(vocab_fts) VALUES ('rebuild')")

    # Add cost columns to sessions if they don't exist (migration for existing DBs)
    for col, typ in [
        ("llm_provider", "TEXT"),
        ("llm_input_tokens", "INTEGER DEFAULT 0"),
        ("llm_output_tokens", "INTEGER DEFAULT 0"),
        ("llm_cost_usd", "REAL DEFAULT 0.0"),
    ]:
        try:
            await db.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typ}")
        except Exception:
            pass  # Column already exists

    # Indexes
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_exchanges_session
        ON exchanges(session_id)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_vocab_next_review
        ON vocab(next_review)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_vocab_term
        ON vocab(term)
    """)
