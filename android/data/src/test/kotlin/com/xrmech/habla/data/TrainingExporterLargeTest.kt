package com.xrmech.habla.data

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import java.io.File
import java.io.OutputStream
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir

/**
 * Bounded memory: 60 segments x 30 s (~57 MB of WAV) export under `:data:largeTest`, which runs with a
 * 64 MB heap. Buffering whole WAVs or the zip would fail with OutOfMemoryError.
 */
@Tag("large")
@OptIn(ExperimentalCoroutinesApi::class)
class TrainingExporterLargeTest {
    @TempDir
    lateinit var filesDir: File

    private val t = TestDb()

    @AfterEach
    fun close() = t.close()

    /** Counts bytes and discards them, so the test itself holds no zip in memory. */
    private class CountingSink : OutputStream() {
        var count = 0L

        override fun write(b: Int) {
            count++
        }

        override fun write(b: ByteArray, off: Int, len: Int) {
            count += len
        }
    }

    @Test
    fun `sixty 30-second segments export within a 64 MB heap`() = runTest(timeout = kotlin.time.Duration.parse("5m")) {
        val d = UnconfinedTestDispatcher(testScheduler)
        val clock = MutableClock()
        val history = SqlHistoryRepository(t.db, filesDir, d, clock)
        val training = TrainingAudioStore(filesDir, t.db, d, clock) { Long.MAX_VALUE }
        val sid = history.startSession(Mode.CONVERSATION, Direction.ES_TO_EN, "ggml-small-q8_0")
        var seed = 12345
        val samples = ShortArray(30 * 16_000)
        for (seg in 1..60) {
            for (i in samples.indices) {
                seed = seed * 1103515245 + 12345
                samples[i] = (seed ushr 16).toShort()
            }
            training.saveSegment(SegmentCapture(sid, seg, samples, 500.0, 0.6, "es", AsrStatus.ACCEPTED, 2.0, -0.4))
        }
        val sink = CountingSink()
        val summary = TrainingExporter(filesDir, history, training, d, clock).export(sid, sink)
        assertEquals(ExportSummary(60, 0, false), summary)
        assertTrue(sink.count > 50L * 1024 * 1024, "zip size ${sink.count}")
    }
}
