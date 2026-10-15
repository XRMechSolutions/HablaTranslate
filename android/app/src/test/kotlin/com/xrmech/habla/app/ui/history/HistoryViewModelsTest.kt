package com.xrmech.habla.app.ui.history

import android.net.Uri
import com.xrmech.habla.app.fakes.FakeHistoryRepository
import com.xrmech.habla.app.fakes.FakeLiveController
import com.xrmech.habla.app.fakes.FakeVocabRepository
import com.xrmech.habla.app.fakes.MainDispatcherExtension
import com.xrmech.habla.app.pipeline.CardTranslation
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.data.ExchangeRow
import com.xrmech.habla.data.ExportException
import com.xrmech.habla.data.ExportSummary
import com.xrmech.habla.data.SessionRow
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import io.mockk.mockk
import java.io.File
import java.time.Instant
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.RegisterExtension
import org.junit.jupiter.api.io.TempDir

private val STARTED = Instant.parse("2026-10-15T20:15:00Z")

private fun sessionRow(id: Long) = SessionRow(
    id, STARTED, STARTED.plusSeconds(600), Mode.CONVERSATION, Direction.ES_TO_EN, "Ordering lunch", null,
    "ggml-small-q8_0", "gpt", exchangeCount = 2, costUsd = 0.012,
)

private fun exchange(id: Long, segment: Int, audio: String?, error: Boolean = false) = ExchangeRow(
    id = id, sessionId = 5, segmentId = segment, timestamp = STARTED.plusSeconds(segment * 10L), direction = Direction.ES_TO_EN,
    rawTranscript = "hola raw $segment", correctedSource = "Hola $segment.", translation = "Hello $segment.", confidence = 0.9,
    isCorrection = false, correction = null, flagged = emptyList(), processingMs = 100, detectedLanguage = "es",
    translationError = error, audioPath = audio, correctedTranscript = null,
)

class HistoryViewModelTest {
    @JvmField
    @RegisterExtension
    val main = MainDispatcherExtension()

    private val history = FakeHistoryRepository().apply { sessionRows.value = listOf(sessionRow(6), sessionRow(5)) }
    private val live = FakeLiveController()

    private fun TestScope.newVm() = HistoryViewModel(history, live).also { vm ->
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { vm.ui.collect {} }
    }

    @Test
    fun `should refuse to delete the active session`() = runTest {
        live.state.value = LiveState(phase = SessionPhase.Listening, sessionId = 6)
        val vm = newVm()
        assertEquals(6L, vm.ui.value.activeSessionId)
        vm.delete(6)
        assertTrue(history.deleted.isEmpty())
        assertEquals("Stop listening before deleting this session", vm.ui.value.error)
    }

    @Test
    fun `should delete an inactive session`() = runTest {
        live.state.value = LiveState(phase = SessionPhase.Listening, sessionId = 6)
        val vm = newVm()
        vm.delete(5)
        assertEquals(listOf(5L), history.deleted)
        assertEquals(listOf(6L), vm.ui.value.sessions.map { it.id })
    }

    @Test
    fun `should allow deleting the last session once listening stopped`() = runTest {
        live.state.value = LiveState(phase = SessionPhase.Idle, sessionId = 6)
        val vm = newVm()
        assertNull(vm.ui.value.activeSessionId)
        vm.delete(6)
        assertEquals(listOf(6L), history.deleted)
    }
}

class SessionDetailViewModelTest {
    @JvmField
    @RegisterExtension
    val main = MainDispatcherExtension()

    private val history = FakeHistoryRepository().apply { sessionRows.value = listOf(sessionRow(5)) }
    private val vocab = FakeVocabRepository()

    private class FakeExports : ExportPort {
        var error: Exception? = null
        var calls = 0
        var summary = ExportSummary(2, 1, correctedIncluded = false)

        override suspend fun exportTo(sessionId: Long, uri: Uri): ExportSummary {
            calls++
            error?.let { throw it }
            return summary
        }

        override suspend fun exportForShare(sessionId: Long, name: String): SharedExport {
            calls++
            error?.let { throw it }
            return SharedExport(mockk(relaxed = true), summary)
        }

        override fun suggestedName(startedAtUtc: Instant) = ExportController.nameFor(startedAtUtc)
    }

    private val exports = FakeExports()

    private fun TestScope.newVm() = SessionDetailViewModel(5, history, vocab, exports).also { vm ->
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { vm.ui.collect {} }
    }

    @Test
    fun `should save the trimmed transcript and send blank to remove it`() = runTest {
        history.exchangeRows.value = mapOf(5L to listOf(exchange(1, 1, "sessions/5/segment_001.wav")))
        val vm = newVm()
        vm.editTranscript(1, "  hola, que tal  ")
        assertEquals("hola, que tal", history.corrected[1])
        vm.editTranscript(1, "   ")
        assertEquals("", history.corrected[1])
    }

    @Test
    fun `should disable export with the reason when no exchange has audio`() = runTest {
        history.exchangeRows.value = mapOf(5L to listOf(exchange(1, 1, null), exchange(2, 2, null)))
        val vm = newVm()
        val s = vm.ui.value
        assertFalse(s.canExport)
        assertEquals("No training audio in this session (turn on Training capture before listening)", s.exportDisabledReason)
        vm.exportTo(mockk())
        vm.share()
        assertEquals(0, exports.calls)
    }

    @Test
    fun `should show the export counts after a successful export`() = runTest {
        history.exchangeRows.value = mapOf(5L to listOf(exchange(1, 1, "a.wav"), exchange(2, 2, null)))
        val vm = newVm()
        assertTrue(vm.ui.value.canExport)
        assertNull(vm.ui.value.exportDisabledReason)
        vm.exportTo(mockk())
        assertEquals("2 segments, 1 missing", vm.ui.value.local.exportResult)
        assertFalse(vm.ui.value.local.exporting)
    }

    @Test
    fun `should show an export failure message verbatim`() = runTest {
        history.exchangeRows.value = mapOf(5L to listOf(exchange(1, 1, "a.wav")))
        exports.error = ExportException("session 5 has no training segments", ExportException.Kind.NO_SEGMENTS)
        val vm = newVm()
        vm.exportTo(mockk())
        assertEquals("session 5 has no training segments", vm.ui.value.local.exportError)
        assertNull(vm.ui.value.local.exportResult)
    }

    @Test
    fun `should share with the suggested habla file name`() = runTest {
        history.exchangeRows.value = mapOf(5L to listOf(exchange(1, 1, "a.wav")))
        val vm = newVm()
        assertEquals("habla-20261015T201500Z.zip", vm.suggestedName())
        val effects = mutableListOf<DetailEffect>()
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { vm.effects.collect { effects += it } }
        vm.share()
        assertEquals(1, effects.size)
        assertEquals("2 segments, 1 missing", vm.ui.value.local.exportResult)
    }

    @Test
    fun `should map stored rows to cards and failed rows to failed cards`() = runTest {
        history.exchangeRows.value = mapOf(5L to listOf(exchange(1, 1, null), exchange(2, 2, null, error = true)))
        val vm = newVm()
        val items = vm.ui.value.items
        assertEquals("Hola 1.", (items[0].card.translation as CardTranslation.Done).result.corrected)
        assertTrue(items[1].card.translation is CardTranslation.Failed)
        assertEquals("Ordering lunch", vm.ui.value.session?.topicSummary)
    }
}

class ExportCacheTest {
    @TempDir
    lateinit var cache: File

    @Test
    fun `should delete export files older than 24 hours only`() {
        val dir = File(cache, "exports").apply { mkdirs() }
        val now = 1_800_000_000_000L
        val old = File(dir, "old.zip").apply { writeText("x"); setLastModified(now - 25 * 3_600_000L) }
        val fresh = File(dir, "fresh.zip").apply { writeText("x"); setLastModified(now - 3_600_000L) }
        assertEquals(1, ExportController.cleanCache(cache, now))
        assertFalse(old.exists())
        assertTrue(fresh.exists())
    }
}
