"""Integration tests for server.db.database module.

Tests real SQLite behavior using tmp_path-based databases.
Each test gets a fresh DB via the `db` fixture.
"""

import asyncio
import sqlite3

import pytest
import aiosqlite

import server.db.database as db_module
from server.db.database import init_db, get_db, close_db


@pytest.fixture
async def db(tmp_path):
    """Initialize a fresh database for each test, tear down after."""
    db_path = tmp_path / "test.db"
    conn = await init_db(db_path)
    yield conn
    await close_db()


@pytest.fixture
async def db_with_session(db):
    """DB with a pre-inserted session row (id=1) for FK references."""
    await db.execute(
        "INSERT INTO sessions (mode, direction) VALUES ('conversation', 'es_to_en')"
    )
    await db.commit()
    return db


# ---------------------------------------------------------------------------
# TestInitDb
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInitDb:
    """Tests for init_db behavior."""

    async def test_init_db_creates_all_tables(self, db):
        """init_db creates all 6 expected tables."""
        rows = await db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'vocab_fts%' "
            "ORDER BY name"
        )
        table_names = sorted(row[0] for row in rows)
        expected = sorted([
            "sessions", "speakers", "exchanges",
            "vocab", "idiom_patterns", "quality_metrics",
        ])
        assert table_names == expected

    async def test_init_db_enables_wal_journal_mode(self, db):
        """init_db sets journal_mode to WAL."""
        rows = await db.execute_fetchall("PRAGMA journal_mode")
        assert rows[0][0] == "wal"

    async def test_init_db_enables_foreign_keys(self, db):
        """init_db turns on foreign key enforcement."""
        rows = await db.execute_fetchall("PRAGMA foreign_keys")
        assert rows[0][0] == 1

    async def test_init_db_creates_fts5_virtual_table(self, db):
        """init_db creates the vocab_fts FTS5 virtual table."""
        rows = await db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vocab_fts'"
        )
        assert len(rows) == 1
        assert rows[0][0] == "vocab_fts"

    async def test_init_db_creates_indexes(self, db):
        """init_db creates all 3 expected indexes."""
        rows = await db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_%' ORDER BY name"
        )
        index_names = sorted(row[0] for row in rows)
        expected = sorted([
            "idx_exchanges_bookmarked",
            "idx_exchanges_session",
            "idx_quality_metrics_session",
            "idx_quality_metrics_status",
            "idx_vocab_next_review",
            "idx_vocab_term",
        ])
        assert index_names == expected

    async def test_init_db_integrity_check_passes_on_clean_db(self, db):
        """Integrity check succeeds on a freshly created database."""
        rows = await db.execute_fetchall("PRAGMA integrity_check")
        assert rows[0][0] == "ok"

    async def test_init_db_is_idempotent(self, tmp_path):
        """Calling init_db twice on the same path does not error."""
        db_path = tmp_path / "idem.db"
        conn1 = await init_db(db_path)
        await close_db()
        conn2 = await init_db(db_path)
        try:
            rows = await conn2.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            assert len(rows) == 1
        finally:
            await close_db()

    async def test_init_db_sets_row_factory(self, db):
        """init_db sets row_factory to aiosqlite.Row."""
        assert db.row_factory is aiosqlite.Row


# ---------------------------------------------------------------------------
# TestGetDb
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGetDb:
    """Tests for get_db behavior."""

    async def test_get_db_returns_connection_after_init(self, db):
        """get_db returns the active connection after init_db."""
        conn = await get_db()
        assert conn is db

    async def test_get_db_raises_runtime_error_before_init(self, tmp_path):
        """get_db raises RuntimeError when no DB has been initialized."""
        old_db = db_module._db
        db_module._db = None
        try:
            with pytest.raises(RuntimeError, match="Database not initialized"):
                await get_db()
        finally:
            db_module._db = old_db

    async def test_get_db_returns_same_connection_on_repeated_calls(self, db):
        """get_db returns the same connection object each time."""
        conn1 = await get_db()
        conn2 = await get_db()
        assert conn1 is conn2


# ---------------------------------------------------------------------------
# TestCloseDb
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCloseDb:
    """Tests for close_db behavior."""

    async def test_close_db_closes_connection(self, tmp_path):
        """close_db closes the underlying connection."""
        db_path = tmp_path / "close_test.db"
        await init_db(db_path)
        await close_db()
        assert db_module._db is None

    async def test_close_db_sets_db_to_none(self, tmp_path):
        """close_db sets the module-level _db to None."""
        db_path = tmp_path / "close_none.db"
        await init_db(db_path)
        await close_db()
        assert db_module._db is None

    async def test_close_db_safe_when_already_closed(self, tmp_path):
        """close_db does not error when called on an already-closed DB."""
        db_path = tmp_path / "double_close.db"
        await init_db(db_path)
        await close_db()
        await close_db()
        assert db_module._db is None


# ---------------------------------------------------------------------------
# TestTableSchema
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTableSchema:
    """Tests for table column structure and constraints."""

    async def test_sessions_table_has_expected_columns(self, db):
        """sessions table has all expected columns."""
        rows = await db.execute_fetchall("PRAGMA table_info(sessions)")
        col_names = {row[1] for row in rows}
        expected = {
            "id", "started_at", "ended_at", "mode",
            "direction", "speaker_count", "topic_summary", "notes",
            "llm_provider", "llm_input_tokens", "llm_output_tokens", "llm_cost_usd",
        }
        assert col_names == expected

    async def test_speakers_table_has_composite_primary_key(self, db):
        """speakers table has composite PK on (id, session_id)."""
        rows = await db.execute_fetchall("PRAGMA table_info(speakers)")
        pk_cols = {row[1] for row in rows if row[5] > 0}
        assert pk_cols == {"id", "session_id"}

    async def test_exchanges_table_has_foreign_key_to_sessions(self, db):
        """exchanges.session_id references sessions(id)."""
        rows = await db.execute_fetchall("PRAGMA foreign_key_list(exchanges)")
        fk_tables = {row[2] for row in rows}
        assert "sessions" in fk_tables

    async def test_vocab_table_has_all_expected_columns(self, db):
        """vocab table has all expected columns."""
        rows = await db.execute_fetchall("PRAGMA table_info(vocab)")
        col_names = {row[1] for row in rows}
        expected = {
            "id", "exchange_id", "speaker_id", "term", "literal",
            "meaning", "category", "source_sentence", "region",
            "save_worthy", "created_at", "ease_factor", "interval_days",
            "next_review", "repetitions", "lapse_count", "times_encountered",
        }
        assert col_names == expected

    async def test_idiom_patterns_table_has_expected_columns(self, db):
        """idiom_patterns table has all expected columns."""
        rows = await db.execute_fetchall("PRAGMA table_info(idiom_patterns)")
        col_names = {row[1] for row in rows}
        expected = {
            "id", "pattern", "canonical", "literal",
            "meaning", "region", "frequency",
        }
        assert col_names == expected

    async def test_quality_metrics_table_has_foreign_key_to_sessions(self, db):
        """quality_metrics.session_id references sessions(id)."""
        rows = await db.execute_fetchall("PRAGMA foreign_key_list(quality_metrics)")
        fk_tables = {row[2] for row in rows}
        assert "sessions" in fk_tables


# ---------------------------------------------------------------------------
# TestFTS5
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFTS5:
    """Tests for FTS5 virtual table and sync triggers."""

    async def test_insert_into_vocab_populates_fts(self, db):
        """Inserting a vocab row auto-populates vocab_fts via trigger."""
        await db.execute(
            "INSERT INTO vocab (term, meaning, literal, category) "
            "VALUES ('echar de menos', 'to miss someone', 'to throw of less', 'idiom')"
        )
        await db.commit()
        rows = await db.execute_fetchall(
            "SELECT term FROM vocab_fts WHERE vocab_fts MATCH 'echar'"
        )
        assert len(rows) == 1
        assert rows[0][0] == "echar de menos"

    async def test_delete_from_vocab_removes_from_fts(self, db):
        """Deleting a vocab row removes it from vocab_fts via trigger."""
        await db.execute(
            "INSERT INTO vocab (term, meaning, literal, category) "
            "VALUES ('tener ganas', 'to feel like', 'to have desires', 'idiom')"
        )
        await db.commit()
        await db.execute("DELETE FROM vocab WHERE term = 'tener ganas'")
        await db.commit()
        rows = await db.execute_fetchall(
            "SELECT term FROM vocab_fts WHERE vocab_fts MATCH 'tener'"
        )
        assert len(rows) == 0

    async def test_update_vocab_updates_fts(self, db):
        """Updating a vocab row updates vocab_fts via trigger."""
        await db.execute(
            "INSERT INTO vocab (term, meaning, literal, category) "
            "VALUES ('dar igual', 'to not matter', 'to give equal', 'idiom')"
        )
        await db.commit()
        await db.execute(
            "UPDATE vocab SET meaning = 'to not care' WHERE term = 'dar igual'"
        )
        await db.commit()
        old = await db.execute_fetchall(
            "SELECT * FROM vocab_fts WHERE vocab_fts MATCH 'matter'"
        )
        assert len(old) == 0
        new = await db.execute_fetchall(
            "SELECT meaning FROM vocab_fts WHERE vocab_fts MATCH 'care'"
        )
        assert len(new) == 1
        assert new[0][0] == "to not care"

    async def test_fts5_match_on_term_field(self, db):
        """FTS5 MATCH finds rows by term content."""
        await db.execute(
            "INSERT INTO vocab (term, meaning) "
            "VALUES ('por si acaso', 'just in case')"
        )
        await db.execute(
            "INSERT INTO vocab (term, meaning) "
            "VALUES ('sin embargo', 'however')"
        )
        await db.commit()
        rows = await db.execute_fetchall(
            "SELECT term FROM vocab_fts WHERE vocab_fts MATCH 'embargo'"
        )
        assert len(rows) == 1
        assert rows[0][0] == "sin embargo"

    async def test_fts5_match_on_meaning_field(self, db):
        """FTS5 MATCH finds rows by meaning content."""
        await db.execute(
            "INSERT INTO vocab (term, meaning) "
            "VALUES ('a lo mejor', 'maybe or perhaps')"
        )
        await db.commit()
        rows = await db.execute_fetchall(
            "SELECT term FROM vocab_fts WHERE vocab_fts MATCH 'perhaps'"
        )
        assert len(rows) == 1
        assert rows[0][0] == "a lo mejor"


# ---------------------------------------------------------------------------
# TestCRUD
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCRUD:
    """Tests for basic insert/retrieve operations on all tables."""

    async def test_insert_and_retrieve_session(self, db):
        """Insert a session and read it back."""
        await db.execute(
            "INSERT INTO sessions (mode, direction) VALUES ('classroom', 'en_to_es')"
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM sessions WHERE id = 1")
        assert len(rows) == 1
        assert rows[0]["mode"] == "classroom"
        assert rows[0]["direction"] == "en_to_es"

    async def test_insert_and_retrieve_speaker(self, db_with_session):
        """Insert a speaker linked to a session and read it back."""
        db = db_with_session
        await db.execute(
            "INSERT INTO speakers (id, session_id, auto_label, color) "
            "VALUES ('SPEAKER_00', 1, 'Speaker A', '#3B82F6')"
        )
        await db.commit()
        rows = await db.execute_fetchall(
            "SELECT * FROM speakers WHERE id = 'SPEAKER_00' AND session_id = 1"
        )
        assert len(rows) == 1
        assert rows[0]["auto_label"] == "Speaker A"
        assert rows[0]["color"] == "#3B82F6"

    async def test_insert_and_retrieve_exchange(self, db_with_session):
        """Insert an exchange linked to a session and read it back."""
        db = db_with_session
        await db.execute(
            "INSERT INTO exchanges (session_id, direction, raw_transcript, translation) "
            "VALUES (1, 'es_to_en', 'Hola mundo', 'Hello world')"
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM exchanges WHERE id = 1")
        assert len(rows) == 1
        assert rows[0]["raw_transcript"] == "Hola mundo"
        assert rows[0]["translation"] == "Hello world"

    async def test_insert_and_retrieve_vocab(self, db):
        """Insert a vocab item and read it back."""
        await db.execute(
            "INSERT INTO vocab (term, meaning, category, region) "
            "VALUES ('meter la pata', 'to put your foot in it', 'idiom', 'spain')"
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM vocab WHERE id = 1")
        assert len(rows) == 1
        assert rows[0]["term"] == "meter la pata"
        assert rows[0]["region"] == "spain"

    async def test_insert_and_retrieve_idiom_pattern(self, db):
        """Insert an idiom pattern and read it back."""
        await db.execute(
            "INSERT INTO idiom_patterns (pattern, canonical, meaning, region) "
            "VALUES ('tirar\\s+la\\s+toalla', 'tirar la toalla', 'to give up', 'universal')"
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM idiom_patterns WHERE id = 1")
        assert len(rows) == 1
        assert rows[0]["canonical"] == "tirar la toalla"

    async def test_foreign_key_enforcement_rejects_orphan_exchange(self, db):
        """Inserting an exchange with nonexistent session_id fails with FK on."""
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            await db.execute(
                "INSERT INTO exchanges (session_id, direction, raw_transcript, translation) "
                "VALUES (9999, 'es_to_en', 'orphan', 'orphan')"
            )
            await db.commit()

    async def test_vocab_defaults_ease_factor_interval_encounters(self, db):
        """Vocab row gets expected defaults for ease_factor, interval_days, times_encountered."""
        await db.execute(
            "INSERT INTO vocab (term, meaning) VALUES ('prueba', 'test')"
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM vocab WHERE id = 1")
        row = rows[0]
        assert row["ease_factor"] == 2.5
        assert row["interval_days"] == 1
        assert row["times_encountered"] == 1
        assert row["category"] == "idiom"
        assert row["region"] == "universal"
        assert row["repetitions"] == 0

    async def test_insert_and_retrieve_quality_metrics(self, db_with_session):
        """Insert a quality_metrics row linked to a session and read it back."""
        db = db_with_session
        await db.execute(
            "INSERT INTO quality_metrics "
            "(session_id, confidence, audio_rms, duration_seconds, processing_time_ms, model_name) "
            "VALUES (1, 0.92, 0.05, 3.2, 450, 'whisper-small')"
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM quality_metrics WHERE id = 1")
        assert len(rows) == 1
        assert rows[0]["confidence"] == pytest.approx(0.92)
        assert rows[0]["model_name"] == "whisper-small"


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConcurrency:
    """Tests proving WAL-mode concurrent read/write behavior."""

    async def _open_conn(self, db_path):
        """Open a new connection with WAL and busy_timeout matching production."""
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=30000")
        return conn

    async def test_reader_not_blocked_by_active_writer(self, tmp_path):
        """A reader can query while a separate connection holds an open write transaction."""
        db_path = tmp_path / "concurrent.db"
        await init_db(db_path)

        writer = await self._open_conn(db_path)
        reader = await self._open_conn(db_path)
        try:
            # Seed a row so reader has something to find
            await writer.execute(
                "INSERT INTO sessions (mode, direction) VALUES ('conversation', 'es_to_en')"
            )
            await writer.commit()

            # Writer starts a transaction and inserts without committing
            await writer.execute("BEGIN IMMEDIATE")
            await writer.execute(
                "INSERT INTO sessions (mode, direction) VALUES ('classroom', 'en_to_es')"
            )

            # Reader should still see the committed row (snapshot isolation)
            rows = await reader.execute_fetchall("SELECT * FROM sessions")
            assert len(rows) == 1
            assert rows[0]["mode"] == "conversation"

            # Writer commits
            await writer.execute("COMMIT")

            # Now reader sees both rows
            rows = await reader.execute_fetchall("SELECT * FROM sessions")
            assert len(rows) == 2
        finally:
            await writer.close()
            await reader.close()
            await close_db()

    async def test_parallel_writes_succeed_with_busy_timeout(self, tmp_path):
        """Two writers inserting concurrently both succeed thanks to busy_timeout."""
        db_path = tmp_path / "parallel_write.db"
        await init_db(db_path)

        num_rows_per_writer = 20
        errors = []

        async def writer_task(writer_id):
            conn = await self._open_conn(db_path)
            try:
                for i in range(num_rows_per_writer):
                    await conn.execute(
                        "INSERT INTO vocab (term, meaning) VALUES (?, ?)",
                        (f"term_{writer_id}_{i}", f"meaning_{writer_id}_{i}"),
                    )
                    await conn.commit()
            except Exception as exc:
                errors.append((writer_id, exc))
            finally:
                await conn.close()

        await asyncio.gather(writer_task(1), writer_task(2))

        assert errors == [], f"Writer errors: {errors}"

        # Verify all rows landed
        primary = await get_db()
        rows = await primary.execute_fetchall("SELECT COUNT(*) FROM vocab")
        assert rows[0][0] == num_rows_per_writer * 2
        await close_db()

    async def test_concurrent_read_write_vocab_with_fts(self, tmp_path):
        """Concurrent writes and FTS reads on vocab table do not deadlock or error."""
        db_path = tmp_path / "fts_concurrent.db"
        await init_db(db_path)

        # Seed some searchable data
        primary = await get_db()
        for i in range(5):
            await primary.execute(
                "INSERT INTO vocab (term, meaning) VALUES (?, ?)",
                (f"modismo_{i}", f"idiom meaning {i}"),
            )
        await primary.commit()

        read_results = []
        write_errors = []

        async def fts_reader():
            conn = await self._open_conn(db_path)
            try:
                for _ in range(10):
                    rows = await conn.execute_fetchall(
                        "SELECT term FROM vocab_fts WHERE vocab_fts MATCH 'modismo*'"
                    )
                    read_results.append(len(rows))
                    await asyncio.sleep(0)
            finally:
                await conn.close()

        async def vocab_writer():
            conn = await self._open_conn(db_path)
            try:
                for i in range(5, 15):
                    await conn.execute(
                        "INSERT INTO vocab (term, meaning) VALUES (?, ?)",
                        (f"modismo_{i}", f"idiom meaning {i}"),
                    )
                    await conn.commit()
                    await asyncio.sleep(0)
            except Exception as exc:
                write_errors.append(exc)
            finally:
                await conn.close()

        await asyncio.gather(fts_reader(), vocab_writer())

        assert write_errors == [], f"Write errors: {write_errors}"
        # Reader should have gotten results on every iteration
        assert all(count >= 5 for count in read_results)
        # After all writes, total vocab should be 15
        rows = await primary.execute_fetchall("SELECT COUNT(*) FROM vocab")
        assert rows[0][0] == 15
        await close_db()


# ---------------------------------------------------------------------------
# TestSessionCostPersistence
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSessionCostPersistence:
    """Tests for LLM cost columns on the sessions table."""

    async def test_session_cost_columns_exist(self, db):
        """Sessions table has llm cost columns after init."""
        rows = await db.execute_fetchall(
            "SELECT llm_provider, llm_input_tokens, llm_output_tokens, llm_cost_usd "
            "FROM sessions LIMIT 0"
        )
        # Query succeeds without error — columns exist
        assert rows == []  # no rows, but columns are valid

    async def test_session_cost_defaults_to_zero(self, db):
        """New session rows default to zero cost."""
        await db.execute("INSERT INTO sessions (mode, direction) VALUES ('conversation', 'es_to_en')")
        await db.commit()
        rows = await db.execute_fetchall("SELECT llm_input_tokens, llm_output_tokens, llm_cost_usd FROM sessions WHERE id = 1")
        row = rows[0]
        assert row["llm_input_tokens"] == 0
        assert row["llm_output_tokens"] == 0
        assert row["llm_cost_usd"] == 0.0

    async def test_session_cost_persistence_round_trip(self, db):
        """Costs written to a session can be read back."""
        await db.execute("INSERT INTO sessions (mode, direction) VALUES ('conversation', 'es_to_en')")
        await db.commit()
        await db.execute(
            """UPDATE sessions SET llm_provider = ?, llm_input_tokens = ?,
               llm_output_tokens = ?, llm_cost_usd = ? WHERE id = 1""",
            ("openai", 1500, 800, 0.0023),
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT llm_provider, llm_input_tokens, llm_output_tokens, llm_cost_usd FROM sessions WHERE id = 1")
        row = rows[0]
        assert row["llm_provider"] == "openai"
        assert row["llm_input_tokens"] == 1500
        assert row["llm_output_tokens"] == 800
        assert abs(row["llm_cost_usd"] - 0.0023) < 1e-6

    async def test_all_time_cost_aggregation(self, db):
        """SUM query across multiple sessions returns correct totals."""
        for i, (tokens_in, tokens_out, cost) in enumerate([
            (1000, 500, 0.001),
            (2000, 1000, 0.003),
            (500, 200, 0.0005),
        ]):
            await db.execute("INSERT INTO sessions (mode, direction) VALUES ('conversation', 'es_to_en')")
            await db.commit()
            await db.execute(
                """UPDATE sessions SET llm_provider = 'openai',
                   llm_input_tokens = ?, llm_output_tokens = ?, llm_cost_usd = ?
                   WHERE id = ?""",
                (tokens_in, tokens_out, cost, i + 1),
            )
        # Also add a non-openai session that should be excluded
        await db.execute("INSERT INTO sessions (mode, direction) VALUES ('conversation', 'es_to_en')")
        await db.execute(
            "UPDATE sessions SET llm_provider = 'ollama', llm_input_tokens = 9999 WHERE id = 4"
        )
        await db.commit()

        rows = await db.execute_fetchall(
            """SELECT COALESCE(SUM(llm_input_tokens), 0) as total_input,
                      COALESCE(SUM(llm_output_tokens), 0) as total_output,
                      COALESCE(SUM(llm_cost_usd), 0.0) as total_cost
               FROM sessions WHERE llm_provider = 'openai'"""
        )
        row = rows[0]
        assert row["total_input"] == 3500
        assert row["total_output"] == 1700
        assert abs(row["total_cost"] - 0.0045) < 1e-6
