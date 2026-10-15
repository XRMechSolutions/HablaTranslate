package com.xrmech.habla.data

import com.xrmech.habla.data.db.HablaDatabase
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/**
 * The v1 schema guard. With no `.sqm` files `verifySqlDelightMigration` passes vacuously (KB quirk 1),
 * so this test pins every table, column, type, nullability, default, key and index. Any `.sq` edit
 * fails here, which forces whoever makes it to add `1.sqm` (and so a real migration check).
 */
class SchemaV1Test {
    private val t = TestDb()

    @AfterEach
    fun close() = t.close()

    private data class Col(val name: String, val type: String, val notNull: Int, val dflt: String?, val pk: Int)

    private data class Fk(val table: String, val from: String, val to: String, val onDelete: String)

    private fun columns(table: String) = t.rows("PRAGMA table_info($table)") {
        Col(it.getString(1)!!, it.getString(2)!!, it.getLong(3)!!.toInt(), it.getString(4), it.getLong(5)!!.toInt())
    }

    private fun foreignKeys(table: String) = t.rows("PRAGMA foreign_key_list($table)") {
        Fk(it.getString(2)!!, it.getString(3)!!, it.getString(4)!!, it.getString(6)!!)
    }

    @Test
    fun `schema version is 1`() {
        assertEquals(1L, HablaDatabase.Schema.version)
    }

    @Test
    fun `tables and indexes`() {
        val objects = t.rows("SELECT type, name FROM sqlite_master WHERE type IN ('table','index') ORDER BY type, name") {
            "${it.getString(0)}:${it.getString(1)}"
        }
        assertEquals(
            listOf(
                "index:exchanges_session_idx",
                "index:sqlite_autoindex_training_segments_1",
                "index:sqlite_autoindex_vocab_1",
                "index:usage_log_session_idx",
                "table:exchanges",
                "table:sessions",
                "table:sqlite_sequence",
                "table:training_segments",
                "table:usage_log",
                "table:vocab",
            ),
            objects,
        )
    }

    @Test
    fun `index columns`() {
        val idx = t.rows("PRAGMA index_info(exchanges_session_idx)") { it.getString(2)!! } +
            t.rows("PRAGMA index_info(usage_log_session_idx)") { it.getString(2)!! } +
            t.rows("PRAGMA index_info(sqlite_autoindex_vocab_1)") { it.getString(2)!! } +
            t.rows("PRAGMA index_info(sqlite_autoindex_training_segments_1)") { it.getString(2)!! }
        assertEquals(listOf("session_id", "segment_id", "session_id", "term", "session_id", "segment_id"), idx)
    }

    @Test
    fun `sessions columns`() {
        assertEquals(
            listOf(
                Col("id", "INTEGER", 0, null, 1),
                Col("started_at", "TEXT", 1, null, 0),
                Col("ended_at", "TEXT", 0, null, 0),
                Col("mode", "TEXT", 1, "'conversation'", 0),
                Col("direction", "TEXT", 1, "'es_to_en'", 0),
                Col("topic_summary", "TEXT", 0, null, 0),
                Col("notes", "TEXT", 0, null, 0),
                Col("asr_model", "TEXT", 1, null, 0),
                Col("translate_model", "TEXT", 0, null, 0),
            ),
            columns("sessions"),
        )
        assertEquals(emptyList<Fk>(), foreignKeys("sessions"))
    }

    @Test
    fun `exchanges columns and keys`() {
        assertEquals(
            listOf(
                Col("id", "INTEGER", 0, null, 1),
                Col("session_id", "INTEGER", 1, null, 0),
                Col("segment_id", "INTEGER", 1, null, 0),
                Col("timestamp", "TEXT", 1, null, 0),
                Col("direction", "TEXT", 1, null, 0),
                Col("raw_transcript", "TEXT", 1, null, 0),
                Col("corrected_source", "TEXT", 0, null, 0),
                Col("translation", "TEXT", 1, null, 0),
                Col("confidence", "REAL", 0, null, 0),
                Col("is_correction", "INTEGER", 1, "0", 0),
                Col("correction_json", "TEXT", 0, null, 0),
                Col("flagged_json", "TEXT", 1, "'[]'", 0),
                Col("processing_ms", "INTEGER", 0, null, 0),
                Col("detected_language", "TEXT", 0, null, 0),
                Col("translation_error", "INTEGER", 1, "0", 0),
                Col("audio_path", "TEXT", 0, null, 0),
                Col("corrected_transcript", "TEXT", 0, null, 0),
            ),
            columns("exchanges"),
        )
        assertEquals(listOf(Fk("sessions", "session_id", "id", "CASCADE")), foreignKeys("exchanges"))
    }

    @Test
    fun `vocab columns and keys`() {
        assertEquals(
            listOf(
                Col("id", "INTEGER", 0, null, 1),
                Col("exchange_id", "INTEGER", 0, null, 0),
                Col("term", "TEXT", 1, null, 0),
                Col("literal", "TEXT", 0, null, 0),
                Col("meaning", "TEXT", 1, null, 0),
                Col("category", "TEXT", 1, "'idiom'", 0),
                Col("source_sentence", "TEXT", 0, null, 0),
                Col("region", "TEXT", 0, "'universal'", 0),
                Col("created_at", "TEXT", 1, null, 0),
                Col("times_encountered", "INTEGER", 1, "1", 0),
            ),
            columns("vocab"),
        )
        assertEquals(listOf(Fk("exchanges", "exchange_id", "id", "SET NULL")), foreignKeys("vocab"))
    }

    @Test
    fun `usage_log columns and keys`() {
        assertEquals(
            listOf(
                Col("id", "INTEGER", 0, null, 1),
                Col("session_id", "INTEGER", 0, null, 0),
                Col("at", "TEXT", 1, null, 0),
                Col("model", "TEXT", 1, null, 0),
                Col("kind", "TEXT", 1, null, 0),
                Col("input_tokens", "INTEGER", 1, null, 0),
                Col("cached_input_tokens", "INTEGER", 1, "0", 0),
                Col("output_tokens", "INTEGER", 1, null, 0),
                Col("cost_usd", "REAL", 1, null, 0),
            ),
            columns("usage_log"),
        )
        assertEquals(listOf(Fk("sessions", "session_id", "id", "SET NULL")), foreignKeys("usage_log"))
    }

    @Test
    fun `training_segments columns and keys`() {
        assertEquals(
            listOf(
                Col("id", "INTEGER", 0, null, 1),
                Col("session_id", "INTEGER", 1, null, 0),
                Col("segment_id", "INTEGER", 1, null, 0),
                Col("filename", "TEXT", 1, null, 0),
                Col("size_bytes", "INTEGER", 1, null, 0),
                Col("duration_seconds", "REAL", 1, null, 0),
                Col("recorded_at", "TEXT", 1, null, 0),
                Col("audio_rms", "REAL", 1, null, 0),
                Col("vad_avg_prob", "REAL", 1, null, 0),
                Col("detected_language", "TEXT", 0, null, 0),
                Col("asr_status", "TEXT", 1, null, 0),
                Col("asr_time_seconds", "REAL", 1, "0", 0),
                Col("asr_avg_logprob", "REAL", 0, null, 0),
            ),
            columns("training_segments"),
        )
        assertEquals(listOf(Fk("sessions", "session_id", "id", "CASCADE")), foreignKeys("training_segments"))
    }

    @Test
    fun `vocab term is unique without ASCII case`() {
        t.exec("INSERT INTO vocab (term, meaning, created_at) VALUES ('echar aguas', 'm', 'x')")
        val dup = runCatching { t.exec("INSERT INTO vocab (term, meaning, created_at) VALUES ('Echar Aguas', 'm', 'x')") }
        assertEquals(true, dup.isFailure)
    }
}
