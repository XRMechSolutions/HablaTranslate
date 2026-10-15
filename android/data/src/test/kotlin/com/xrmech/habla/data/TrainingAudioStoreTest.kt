package com.xrmech.habla.data

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.sound.sampled.AudioSystem
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

@OptIn(ExperimentalCoroutinesApi::class)
class TrainingAudioStoreTest {
    @TempDir
    lateinit var filesDir: File

    private val t = TestDb()
    private val clock = MutableClock()

    @AfterEach
    fun close() = t.close()

    private fun TestScope.store(usableSpace: (File) -> Long = { Long.MAX_VALUE }) =
        TrainingAudioStore(filesDir, t.db, UnconfinedTestDispatcher(testScheduler), clock, usableSpace)

    private fun TestScope.history() = SqlHistoryRepository(t.db, filesDir, UnconfinedTestDispatcher(testScheduler), clock)

    private fun pcm(n: Int) = ShortArray(n) { ((it * 97) % 65536 - 32768).toShort() }

    private fun capture(sid: Long, seg: Int, samples: ShortArray = pcm(1600), status: AsrStatus = AsrStatus.ACCEPTED) =
        SegmentCapture(sid, seg, samples, 812.44, 0.7123, "es", status, 1.844, -0.3124)

    private suspend fun TestScope.newSession() = history().startSession(Mode.CONVERSATION, Direction.AUTO, "ggml-small-q8_0")

    private fun wavFiles() = filesDir.walkTopDown().filter { it.isFile }.map { it.relativeTo(filesDir).invariantSeparatorsPath }.toList()

    @Test
    fun `writes a 44 byte header plus 2n bytes and returns the relative path`() = runTest {
        val sid = newSession()
        val samples = pcm(16_000)
        val path = store().saveSegment(capture(sid, 1, samples))
        assertEquals("sessions/$sid/segment_001.wav", path)
        val file = File(filesDir, path!!)
        assertEquals(44L + 2 * 16_000, file.length())
        val fmt = AudioSystem.getAudioFileFormat(file).format
        assertEquals(16_000f, fmt.sampleRate)
        assertEquals(16, fmt.sampleSizeInBits)
        assertEquals(1, fmt.channels)
        assertFalse(fmt.isBigEndian)
        val body = file.readBytes().copyOfRange(44, file.length().toInt())
        val back = ShortArray(samples.size).also { ByteBuffer.wrap(body).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(it) }
        assertArrayEquals(samples, back)
        assertEquals(listOf("sessions/$sid/segment_001.wav"), wavFiles())
    }

    @Test
    fun `samples longer than the 64 KiB buffer round trip`() = runTest {
        val sid = newSession()
        val samples = pcm(100_003)
        val file = File(filesDir, store().saveSegment(capture(sid, 2, samples))!!)
        val body = file.readBytes().copyOfRange(44, file.length().toInt())
        val back = ShortArray(samples.size).also { ByteBuffer.wrap(body).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(it) }
        assertArrayEquals(samples, back)
    }

    @Test
    fun `row stores the metadata`() = runTest {
        val sid = newSession()
        store().saveSegment(capture(sid, 3, pcm(48_000), AsrStatus.REJECTED))
        val row = store().segments(sid).single()
        assertEquals(3, row.segmentId)
        assertEquals("segment_003.wav", row.filename)
        assertEquals(96_000L, row.sizeBytes)
        assertEquals(3.0, row.durationSeconds)
        assertEquals(clock.instant(), row.recordedAt)
        assertEquals(812.44, row.audioRms)
        assertEquals(0.7123, row.vadAvgProb)
        assertEquals("es", row.detectedLanguage)
        assertEquals(AsrStatus.REJECTED, row.status)
        assertEquals(1.844, row.asrTimeSeconds)
        assertEquals(-0.3124, row.asrAvgLogprob)
        assertNull(row.exchangeId)
        assertEquals(0, row.flaggedCount)
        assertEquals(listOf("rejected"), t.rows("SELECT asr_status FROM training_segments") { it.getString(0) })
    }

    @Test
    fun `file names pad to three digits and widen past 999`() = runTest {
        val sid = newSession()
        assertEquals("sessions/$sid/segment_007.wav", store().saveSegment(capture(sid, 7)))
        assertEquals("sessions/$sid/segment_1000.wav", store().saveSegment(capture(sid, 1000)))
    }

    @Test
    fun `saving the same segment twice returns the existing path and writes nothing`() = runTest {
        val sid = newSession()
        val store = store()
        val first = store.saveSegment(capture(sid, 4, pcm(100)))
        val file = File(filesDir, first!!)
        val before = file.readBytes()
        val second = store.saveSegment(capture(sid, 4, pcm(5000)))
        assertEquals(first, second)
        assertArrayEquals(before, file.readBytes())
        assertEquals(1, store.segments(sid).size)
    }

    @Test
    fun `no tmp file is left behind`() = runTest {
        val sid = newSession()
        store().saveSegment(capture(sid, 1))
        assertTrue(wavFiles().none { it.endsWith(".tmp") })
    }

    @Test
    fun `database failure after the write removes the wav and returns null`() = runTest {
        val sid = newSession()
        // The space check runs between the session check and the insert: delete the session there.
        val store = store(usableSpace = {
            t.exec("DELETE FROM sessions WHERE id = $sid")
            Long.MAX_VALUE
        })
        assertNull(store.saveSegment(capture(sid, 1)))
        assertEquals(emptyList<String>(), wavFiles())
    }

    @Test
    fun `low space returns null and writes nothing`() = runTest {
        val sid = newSession()
        val samples = pcm(16_000)
        val needed = TrainingAudioStore.MIN_FREE_BYTES + samples.size * 2 + 44
        assertNull(store(usableSpace = { needed - 1 }).saveSegment(capture(sid, 1, samples)))
        assertEquals(emptyList<String>(), wavFiles())
        assertEquals(0, store().segments(sid).size)
        assertEquals("sessions/$sid/segment_001.wav", store(usableSpace = { needed }).saveSegment(capture(sid, 1, samples)))
    }

    @Test
    fun `bad arguments throw`() = runTest {
        val sid = newSession()
        val store = store()
        assertThrows<IllegalArgumentException> { store.saveSegment(capture(sid, 0)) }
        assertThrows<IllegalArgumentException> { store.saveSegment(capture(sid, 1, ShortArray(0))) }
        assertThrows<IllegalArgumentException> { store.saveSegment(capture(sid + 50, 1)) }
        assertEquals(emptyList<String>(), wavFiles())
    }

    @Test
    fun `segments join the lowest-id exchange and count flagged phrases`() = runTest {
        val history = history()
        val sid = newSession()
        val store = store()
        store.saveSegment(capture(sid, 2))
        store.saveSegment(capture(sid, 1))
        val base = NewExchange(
            sessionId = sid, segmentId = 1, direction = Direction.ES_TO_EN, rawTranscript = "first",
            correctedSource = null, translation = "uno", confidence = 0.8, isCorrection = false,
            correction = null, flagged = listOf(FlaggedPhrase(phrase = "a", meaning = "b"), FlaggedPhrase(phrase = "c", meaning = "d")),
            processingMs = null, detectedLanguage = "es", translationError = false, audioPath = null,
        )
        val firstId = history.addExchange(base)
        history.addExchange(base.copy(rawTranscript = "second", flagged = emptyList()))
        history.setCorrectedTranscript(firstId, "fixed")
        val rows = store.segments(sid)
        assertEquals(listOf(1, 2), rows.map { it.segmentId })
        assertEquals(firstId, rows[0].exchangeId)
        assertEquals("first", rows[0].rawTranscript)
        assertEquals("uno", rows[0].translation)
        assertEquals(0.8, rows[0].confidence)
        assertEquals(2, rows[0].flaggedCount)
        assertEquals("fixed", rows[0].correctedTranscript)
        assertNull(rows[1].exchangeId)
        assertEquals(0, rows[1].flaggedCount)
    }

    @Test
    fun `sweepOrphans removes unknown dirs and stray tmp files and keeps valid files`() = runTest {
        val sid = newSession()
        val store = store()
        val kept = File(filesDir, store.saveSegment(capture(sid, 1))!!)
        val sessions = File(filesDir, "sessions")
        File(sessions, "abc").apply { mkdirs() }.resolve("x.wav").writeText("x")
        File(sessions, "9999").apply { mkdirs() }.resolve("segment_001.wav").writeText("x")
        File(sessions, "00$sid").mkdirs()
        val strayTmp = File(sessions, "$sid/segment_002.wav.tmp").apply { writeText("partial") }
        assertEquals(4, store.sweepOrphans())
        assertTrue(kept.isFile)
        assertFalse(strayTmp.exists())
        assertEquals(listOf("$sid"), sessions.list()!!.toList())
        assertEquals(0, store.sweepOrphans())
    }

    @Test
    fun `sweepOrphans on a clean install returns zero`() = runTest {
        assertEquals(0, store().sweepOrphans())
    }
}
