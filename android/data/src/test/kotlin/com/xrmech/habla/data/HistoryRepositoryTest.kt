package com.xrmech.habla.data

import app.cash.turbine.test
import com.xrmech.habla.translate.model.CorrectionDetail
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import java.io.File
import java.time.Instant
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

@OptIn(ExperimentalCoroutinesApi::class)
class HistoryRepositoryTest {
    @TempDir
    lateinit var filesDir: File

    private val t = TestDb()
    private val clock = MutableClock()

    @AfterEach
    fun close() = t.close()

    private fun TestScope.repo() = SqlHistoryRepository(t.db, filesDir, UnconfinedTestDispatcher(testScheduler), clock)

    private fun exchange(sessionId: Long, segmentId: Int = 1, raw: String = "hola") = NewExchange(
        sessionId = sessionId,
        segmentId = segmentId,
        direction = Direction.ES_TO_EN,
        rawTranscript = raw,
        correctedSource = null,
        translation = "hello",
        confidence = 0.9,
        isCorrection = false,
        correction = null,
        flagged = emptyList(),
        processingMs = 100,
        detectedLanguage = "es",
        translationError = false,
        audioPath = null,
    )

    @Test
    fun `start and end a session`() = runTest {
        val repo = repo()
        val id = repo.startSession(Mode.CLASSROOM, Direction.AUTO, "ggml-small-q8_0")
        clock.advanceSeconds(60)
        repo.endSession(id, "  Ordering at a café  ")
        val s = repo.session(id)!!
        assertEquals(Instant.parse("2026-10-15T14:15:02.120Z"), s.startedAt)
        assertEquals(Instant.parse("2026-10-15T14:16:02.120Z"), s.endedAt)
        assertEquals(Mode.CLASSROOM, s.mode)
        assertEquals(Direction.AUTO, s.direction)
        assertEquals("Ordering at a café", s.topicSummary)
        assertEquals("ggml-small-q8_0", s.asrModel)
        assertNull(s.translateModel)
        assertEquals(0, s.exchangeCount)
        assertEquals(0.0, s.costUsd)
    }

    @Test
    fun `endSession twice is a no-op and blank summary is null`() = runTest {
        val repo = repo()
        val id = repo.startSession(Mode.CONVERSATION, Direction.ES_TO_EN, "m")
        clock.advanceSeconds(5)
        repo.endSession(id, "   ")
        val first = repo.session(id)!!
        assertNull(first.topicSummary)
        clock.advanceSeconds(5)
        repo.endSession(id, "later summary")
        assertEquals(first, repo.session(id))
        repo.endSession(999, "unknown")
    }

    @Test
    fun `dangling sessions are closed by the next start`() = runTest {
        val repo = repo()
        val withExchanges = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        clock.advanceSeconds(10)
        repo.addExchange(exchange(withExchanges, 1))
        clock.advanceSeconds(10)
        repo.addExchange(exchange(withExchanges, 2))
        val lastExchangeAt = clock.instant()
        clock.advanceSeconds(100)
        val empty = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        val emptyStart = clock.instant()
        assertEquals(lastExchangeAt, repo.session(withExchanges)!!.endedAt)
        clock.advanceSeconds(100)
        val next = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        assertEquals(emptyStart, repo.session(empty)!!.endedAt)
        assertNull(repo.session(next)!!.endedAt)
    }

    @Test
    fun `addExchange round trips every field`() = runTest {
        val repo = repo()
        val sid = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        clock.advanceSeconds(3)
        val flagged = listOf(
            FlaggedPhrase(
                phrase = "echar aguas", literal = "throw waters", meaning = "watch out", type = "idiom",
                region = "mexico", source = "pattern_db", saveWorthy = true, spanStart = 0, spanEnd = 11,
            ),
            FlaggedPhrase(phrase = "güey", meaning = "dude", region = null),
        )
        val e = NewExchange(
            sessionId = sid, segmentId = 7, direction = Direction.EN_TO_ES,
            rawTranscript = "I has a dog", correctedSource = "I have a dog", translation = "Tengo un perro",
            confidence = 0.75, isCorrection = true,
            correction = CorrectionDetail("has", "have", "subject-verb agreement"),
            flagged = flagged, processingMs = 1234, detectedLanguage = "en", translationError = true,
            audioPath = "sessions/$sid/segment_007.wav",
        )
        val id = repo.addExchange(e)
        val row = repo.exchanges(sid).first().single()
        assertEquals(id, row.id)
        assertEquals(sid, row.sessionId)
        assertEquals(7, row.segmentId)
        assertEquals(clock.instant(), row.timestamp)
        assertEquals(Direction.EN_TO_ES, row.direction)
        assertEquals("I has a dog", row.rawTranscript)
        assertEquals("I have a dog", row.correctedSource)
        assertEquals("Tengo un perro", row.translation)
        assertEquals(0.75, row.confidence)
        assertTrue(row.isCorrection)
        assertEquals(CorrectionDetail("has", "have", "subject-verb agreement"), row.correction)
        assertEquals(flagged, row.flagged)
        assertEquals(1234L, row.processingMs)
        assertEquals("en", row.detectedLanguage)
        assertTrue(row.translationError)
        assertEquals("sessions/$sid/segment_007.wav", row.audioPath)
        assertNull(row.correctedTranscript)
        val stored = t.rows("SELECT flagged_json, correction_json, is_correction FROM exchanges") {
            Triple(it.getString(0), it.getString(1), it.getLong(2))
        }.single()
        assertTrue(stored.first!!.contains("\"save_worthy\":true"))
        assertTrue(stored.first!!.contains("\"span_start\":0"))
        assertEquals("{\"wrong\":\"has\",\"right\":\"have\",\"explanation\":\"subject-verb agreement\"}", stored.second)
        assertEquals(1L, stored.third)
        assertEquals(1, repo.session(sid)!!.exchangeCount)
    }

    @Test
    fun `AUTO direction and segment 0 are rejected`() = runTest {
        val repo = repo()
        val sid = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        assertThrows<IllegalArgumentException> { repo.addExchange(exchange(sid).copy(direction = Direction.AUTO)) }
        assertThrows<IllegalArgumentException> { repo.addExchange(exchange(sid, segmentId = 0)) }
    }

    @Test
    fun `unknown session is a DataException`() = runTest {
        assertThrows<DataException> { repo().addExchange(exchange(4242)) }
    }

    @Test
    fun `setCorrectedTranscript trims, clears on blank and rejects unknown ids`() = runTest {
        val repo = repo()
        val sid = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        val id = repo.addExchange(exchange(sid))
        repo.setCorrectedTranscript(id, "  ¿Qué onda, güey?  ")
        assertEquals("¿Qué onda, güey?", repo.exchanges(sid).first().single().correctedTranscript)
        repo.setCorrectedTranscript(id, "   ")
        assertNull(repo.exchanges(sid).first().single().correctedTranscript)
        assertThrows<NoSuchElementException> { repo.setCorrectedTranscript(id + 100, "x") }
    }

    @Test
    fun `exchanges flow re-emits on insert in segment order`() = runTest {
        val repo = repo()
        val sid = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        repo.exchanges(sid).test {
            assertEquals(emptyList<ExchangeRow>(), awaitItem())
            repo.addExchange(exchange(sid, 2, "dos"))
            assertEquals(listOf("dos"), awaitItem().map { it.rawTranscript })
            repo.addExchange(exchange(sid, 1, "uno"))
            assertEquals(listOf("uno", "dos"), awaitItem().map { it.rawTranscript })
        }
    }

    @Test
    fun `sessions are newest first with counts and cost`() = runTest {
        val repo = repo()
        val a = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        repo.addExchange(exchange(a))
        t.db.usageQueries.insertUsage(a, "x", "gpt", "final", 1, 0, 1, 0.25)
        t.db.usageQueries.insertUsage(a, "x", "gpt", "quick", 1, 0, 1, 0.5)
        clock.advanceSeconds(1)
        val b = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        val rows = repo.sessions().first()
        assertEquals(listOf(b, a), rows.map { it.id })
        assertEquals(1, rows[1].exchangeCount)
        assertEquals(0.75, rows[1].costUsd)
    }

    @Test
    fun `deleteSession cascades, nulls links and deletes the audio dir`() = runTest {
        val repo = repo()
        val sid = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        val other = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        val exId = repo.addExchange(exchange(sid))
        t.db.trainingSegmentQueries.insertTrainingSegment(sid, 1, "segment_001.wav", 2, 0.1, "x", 1.0, 0.5, null, "accepted", 0.1, null)
        t.db.vocabQueries.insertVocab(exId, "echar aguas", null, "watch out", "idiom", null, "universal", "x")
        t.db.usageQueries.insertUsage(sid, "x", "gpt", "final", 10, 0, 5, 0.01)
        val dir = File(filesDir, "sessions/$sid").apply { mkdirs() }
        File(dir, "segment_001.wav").writeBytes(ByteArray(46))
        val otherDir = File(filesDir, "sessions/$other").apply { mkdirs() }

        repo.deleteSession(sid)

        assertNull(repo.session(sid))
        assertEquals(0L, t.rows("SELECT COUNT(*) FROM exchanges") { it.getLong(0)!! }.single())
        assertEquals(0L, t.rows("SELECT COUNT(*) FROM training_segments") { it.getLong(0)!! }.single())
        assertEquals(listOf<Long?>(null), t.rows("SELECT exchange_id FROM vocab") { it.getLong(0) })
        assertEquals(listOf<Long?>(null), t.rows("SELECT session_id FROM usage_log") { it.getLong(0) })
        assertFalse(dir.exists())
        assertTrue(otherDir.exists())
        repo.deleteSession(sid)
    }

    @Test
    fun `translate_model is the most used final model`() = runTest {
        val repo = repo()
        val sid = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        val u = t.db.usageQueries
        u.insertUsage(sid, "x", "gpt-cheap", "final", 1, 0, 1, 0.0)
        u.insertUsage(sid, "x", "gpt-main", "final", 1, 0, 1, 0.0)
        u.insertUsage(sid, "x", "gpt-main", "final", 1, 0, 1, 0.0)
        u.insertUsage(sid, "x", "gpt-quick", "quick", 1, 0, 1, 0.0)
        u.insertUsage(sid, "x", "gpt-quick", "quick", 1, 0, 1, 0.0)
        u.insertUsage(sid, "x", "gpt-quick", "quick", 1, 0, 1, 0.0)
        repo.endSession(sid, null)
        assertEquals("gpt-main", repo.session(sid)!!.translateModel)
    }

    @Test
    fun `corrupt json and unknown enums degrade per row`() = runTest {
        val repo = repo()
        val sid = repo.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        repo.addExchange(exchange(sid))
        t.exec("UPDATE exchanges SET flagged_json = '{not json', correction_json = '[1,2]', direction = 'sideways'")
        t.exec("UPDATE sessions SET mode = 'lecture', direction = 'up'")
        val row = repo.exchanges(sid).first().single()
        assertEquals(emptyList<FlaggedPhrase>(), row.flagged)
        assertNull(row.correction)
        assertEquals(Direction.ES_TO_EN, row.direction)
        val s = repo.session(sid)!!
        assertEquals(Mode.CONVERSATION, s.mode)
        assertEquals(Direction.ES_TO_EN, s.direction)
    }

    @Test
    fun `session of an unknown id is null`() = runTest {
        assertNull(repo().session(77))
    }
}
