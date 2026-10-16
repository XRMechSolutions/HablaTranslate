package com.xrmech.habla.app.pipeline

import com.xrmech.habla.app.capture.AudioSourceException
import com.xrmech.habla.app.fakes.FakeApiKeyStore
import com.xrmech.habla.app.fakes.FakeHistoryRepository
import com.xrmech.habla.app.fakes.FakeRecognizerHolder
import com.xrmech.habla.app.fakes.FakeSegmenter
import com.xrmech.habla.app.fakes.FakeSettingsRepository
import com.xrmech.habla.app.fakes.FakeSpeechRecognizer.Step
import com.xrmech.habla.app.fakes.FakeTrainingCapture
import com.xrmech.habla.app.fakes.FakeTranslationProvider
import com.xrmech.habla.app.fakes.FakeUsageRepository
import com.xrmech.habla.app.fakes.IdleAudioSource
import com.xrmech.habla.app.fakes.TestAssets
import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.app.state.AppSettingsState
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.state.ThermalMonitor
import com.xrmech.habla.asr.AsrException
import com.xrmech.habla.asr.AsrPass
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.audio.VadModel
import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.AsrStatus
import com.xrmech.habla.translate.ConversationContext
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.IdiomScanner
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.TopicTracker
import com.xrmech.habla.translate.Translator
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.UsageKind
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout
import org.junit.jupiter.api.assertThrows

/** Everything real except the seams (design section 11); one StandardTestDispatcher for all four dispatchers. */
class Harness(
    scope: TestScope,
    settings: AppSettings = AppSettingsDefaults(TestAssets.shared).value,
    key: String? = "sk-proj-testkey1234567890",
) {
    val assets = TestAssets.shared
    private val dispatcher = StandardTestDispatcher(scope.testScheduler)

    /** Foreground app scope on the test scheduler (backgroundScope work is not driven by advanceUntilIdle). */
    val appScope = CoroutineScope(SupervisorJob() + dispatcher)
    val keyStore = FakeApiKeyStore(key)
    val provider = FakeTranslationProvider { keyStore.get() }
    val costMeter = CostMeter(assets.models)
    val settingsRepo = FakeSettingsRepository(settings)
    val appSettings = AppSettingsState(settingsRepo, appScope)
    private val ts = { appSettings.translateSettings(assets.models) }
    val translator = Translator(provider, assets, ts, costMeter)
    val quick = QuickTranslator(provider, assets, ts, costMeter)
    val context = ConversationContext(assets.asrProfile)
    val topic = TopicTracker(translator, context)
    val voter = LanguageVoter(assets.asrProfile)
    val keyHealth = KeyHealthTracker(keyStore.hasKey, appScope)
    val history = FakeHistoryRepository()
    val usage = FakeUsageRepository()
    val training = FakeTrainingCapture()
    val thermalStatus = MutableStateFlow(0)
    val thermal = ThermalMonitor(thermalStatus, appScope)
    val segmenter = FakeSegmenter()
    val recognizers = FakeRecognizerHolder()
    val rec get() = recognizers.recognizer
    val source = IdleAudioSource()
    var vadClosed = false

    val coordinator = PipelineCoordinator(
        assets = assets,
        segmenterFactory = { _, _, listener -> segmenter.also { it.listener = listener } },
        vadFactory = {
            object : VadModel {
                override fun speechProbability(frame: FloatArray) = 0f
                override fun reset() = Unit
                override fun close() {
                    vadClosed = true
                }
            }
        },
        recognizers = recognizers,
        voter = voter,
        translator = translator,
        quickTranslator = quick,
        topicTracker = topic,
        context = context,
        idiomScanner = IdiomScanner(assets.idioms),
        costMeter = costMeter,
        settings = appSettings,
        keyStore = keyStore,
        keyHealth = keyHealth,
        history = history,
        usage = usage,
        training = training,
        thermal = thermal,
        dispatchers = PipelineDispatchers(dispatcher, dispatcher, dispatcher, dispatcher),
        clock = { scope.testScheduler.currentTime },
        wallClock = { 1_000L },
        cleanupScope = appScope,
        metricsLogIntervalMs = 0,
        publishIntervalMs = 0,
    )

    val state get() = coordinator.state.value

    /** Calibrate, then emit one raw segment and let the 1.2 s gap pass so it flushes alone. */
    fun utterance(id: Int, seconds: Double = 1.0) {
        segmenter.emitSegment(id, seconds)
        segmenter.advanceSeconds(1.3)
    }
}

@Timeout(30)
class PipelineCoordinatorTest {
    private suspend fun TestScope.started(h: Harness): Long {
        val id = h.coordinator.start(h.source)
        h.segmenter.emitCalibrated()
        h.segmenter.advanceSeconds(2.0)
        advanceUntilIdle()
        return id
    }

    // (1)
    @Test
    fun `should produce two cards es then en with auto direction`() = runTest {
        val h = Harness(this)
        h.rec.detectLanguages += listOf("es", "en")
        h.rec.finals += Step("hola amigo como estas hoy")
        h.rec.finals += Step("hello my friend how are you")
        val id = started(h)
        assertEquals(42L, id)
        assertEquals(SessionPhase.Listening, h.state.phase)
        h.utterance(1)
        advanceUntilIdle()
        h.utterance(2)
        advanceUntilIdle()
        h.coordinator.stop(StopReason.UserStop)

        val cards = h.state.exchanges
        assertEquals(listOf(Direction.ES_TO_EN, Direction.EN_TO_ES), cards.map { it.direction })
        assertTrue(cards.all { it.translation is CardTranslation.Done })
        assertEquals(listOf(1, 2), h.history.added.map { it.segmentId })
        assertEquals(listOf("es", "en"), h.history.added.map { it.detectedLanguage })
        assertEquals(2, h.context.recent().size)
        assertEquals(2, h.provider.requestsOf(UsageKind.TOPIC).size)
        assertTrue(h.usage.rows.count { it.kind == UsageKind.FINAL && it.sessionId == 42L } == 2)
        assertEquals(listOf(FakeHistoryRepository.Ended(42, "topic 2")), h.history.ended)
        assertEquals(SessionPhase.Idle, h.state.phase)
        assertTrue(h.source.closed)
        assertTrue(h.vadClosed)
        assertEquals(1, h.recognizers.releases)
    }

    @Test
    fun `should load the speech model with the live model and engine from the settings snapshot`() = runTest {
        val base = AppSettingsDefaults(TestAssets.shared).value
        val h = Harness(this, settings = base.copy(liveModelId = "ggml-base-q8_0", asrGpu = true))
        started(h)
        h.coordinator.stop(StopReason.UserStop)
        assertEquals(listOf(Triple(base.asrModelId, "ggml-base-q8_0", true)), h.recognizers.acquired)
    }

    @Test
    fun `should pass no live model when the setting is empty`() = runTest {
        val h = Harness(this)
        started(h)
        h.coordinator.stop(StopReason.UserStop)
        assertEquals(null, h.recognizers.acquired.single().second)
        assertEquals(false, h.recognizers.acquired.single().third)
    }

    // (3)
    @Test
    fun `should persist failed no key cards and keep context empty without a key`() = runTest {
        val h = Harness(this, key = null)
        started(h)
        h.segmenter.emitPartial(1)
        advanceUntilIdle()
        h.utterance(1)
        advanceUntilIdle()
        h.coordinator.stop(StopReason.UserStop)

        val card = h.state.exchanges.single()
        assertEquals(LlmException.Kind.NO_KEY, (card.translation as CardTranslation.Failed).kind)
        assertTrue(h.history.added.single().translationError)
        assertEquals(card.rawTranscript, h.history.added.single().translation)
        assertTrue(h.provider.requests.isEmpty(), "no request of any kind without a key")
        assertEquals(0, h.context.recent().size)
        assertEquals(KeyState.ABSENT, h.keyHealth.state.value)
    }

    // (4)
    @Test
    fun `should show the verbatim failure then clear it on the next ok`() = runTest {
        val h = Harness(this)
        h.rec.finals += Step("uno dos tres cuatro cinco")
        h.rec.finals += Step("seis siete ocho nueve diez")
        h.provider.onFinal = { req ->
            if ("uno dos" in req.input) throw LlmException("upstream 500", LlmException.Kind.SERVER, 500)
            FakeTranslationProvider.okFinal("six seven")
        }
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        assertEquals(LlmException.Kind.SERVER, h.state.lastTranslationError?.kind)
        h.utterance(2)
        advanceUntilIdle()
        val first = h.state.exchanges[0].translation as CardTranslation.Failed
        assertEquals("upstream 500", first.message)
        assertTrue(h.state.exchanges[1].translation is CardTranslation.Done)
        assertNull(h.state.lastTranslationError)
        assertEquals(1, h.context.recent().size)
        h.coordinator.stop(StopReason.UserStop)
    }

    // (5)
    @Test
    fun `should merge at 0_8 s split at 1_3 s and flush at 60 s through the coordinator`() = runTest {
        val h = Harness(this)
        started(h)
        h.segmenter.emitSegment(1)
        h.segmenter.advanceSeconds(0.8)
        h.segmenter.emitSegment(2)
        h.segmenter.advanceSeconds(1.2)
        advanceUntilIdle()
        assertEquals(1, h.rec.requests.count { it.pass == AsrPass.FINAL })
        h.segmenter.emitSegment(3)
        h.segmenter.advanceSeconds(1.3)
        advanceUntilIdle()
        assertEquals(2, h.rec.requests.count { it.pass == AsrPass.FINAL })
        h.segmenter.emitSegment(4, seconds = 30.0)
        h.segmenter.emitSegment(5, seconds = 31.0)
        advanceUntilIdle()
        assertEquals(3, h.rec.requests.count { it.pass == AsrPass.FINAL })
        assertEquals(listOf(1, 2, 3), h.state.exchanges.map { it.seq })
        assertEquals(listOf(listOf(1, 2), listOf(3), listOf(4, 5)), h.state.exchanges.map { it.rawSegmentIds })
        h.coordinator.stop(StopReason.UserStop)
    }

    // (6)
    @Test
    fun `should drop the oldest waiting final when seven arrive behind a gated one`() = runTest {
        val h = Harness(this)
        val gate = CompletableDeferred<Unit>()
        h.rec.finals += Step("hola amigo como estas hoy", gate = gate)
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        (2..7).forEach { h.utterance(it) }
        advanceUntilIdle()
        assertEquals(6, h.state.queueDepth)
        assertTrue(SessionNotice.FALLING_BEHIND in h.state.notices)
        gate.complete(Unit)
        advanceUntilIdle()
        h.coordinator.stop(StopReason.UserStop)
        assertEquals(1, h.state.metrics.finalsDropped)
        assertEquals(listOf(1, 3, 4, 5, 6, 7), h.state.exchanges.map { it.seq })
        // cards carry their source segment ids, so a dropped final does not shift the pairing
        assertEquals(listOf(1, 3, 4, 5, 6, 7).map { listOf(it) }, h.state.exchanges.map { it.rawSegmentIds })
        assertEquals(6, h.state.metrics.peakQueueDepth)
    }

    // (7)
    @Test
    fun `should make no card for empty and bad transcripts and save them as rejected`() = runTest {
        val h = Harness(this, settings = AppSettingsDefaults(TestAssets.shared).value.copy(trainingCapture = true))
        h.rec.finals += Step("", segments = false)
        h.rec.finals += Step("...")
        started(h)
        h.utterance(1)
        h.utterance(2)
        advanceUntilIdle()
        h.coordinator.stop(StopReason.UserStop)
        assertTrue(h.state.exchanges.isEmpty())
        assertEquals(1, h.state.metrics.asrEmptyCount)
        assertEquals(1, h.state.metrics.asrRejectedCount)
        assertEquals(listOf(AsrStatus.REJECTED, AsrStatus.REJECTED), h.training.saved.map { it.status })
        assertTrue(h.history.added.isEmpty())
    }

    @Test
    fun `should save accepted training audio as raw pcm with the exchange path`() = runTest {
        val h = Harness(this, settings = AppSettingsDefaults(TestAssets.shared).value.copy(trainingCapture = true))
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        h.coordinator.stop(StopReason.UserStop)
        val saved = h.training.saved.single()
        assertEquals(AsrStatus.ACCEPTED, saved.status)
        assertEquals(16_000, saved.pcm.size)
        assertEquals("sessions/42/segment_001.wav", h.history.added.single().audioPath)
    }

    // (8)
    @Test
    fun `should still detect with a fixed direction and reset the voter when switching to it`() = runTest {
        val h = Harness(this)
        h.rec.detectLanguages += listOf("es", "es")
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        assertEquals("es", h.voter.current)
        h.settingsRepo.update { it.copy(direction = Direction.EN_TO_ES) }
        advanceUntilIdle()
        assertNull(h.voter.current)
        h.utterance(2)
        advanceUntilIdle()
        assertEquals(2, h.rec.detectCalls)
        assertEquals(Direction.EN_TO_ES, h.state.exchanges[1].direction)
        assertEquals("es", h.rec.requests.last { it.pass == AsrPass.FINAL }.language)
        h.coordinator.stop(StopReason.UserStop)
    }

    // (11)
    @Test
    fun `should drain a gated final and its translation on stop and end with the topic`() = runTest {
        val h = Harness(this)
        val gate = CompletableDeferred<Unit>()
        h.rec.finals += Step("hola amigo como estas hoy", gate = gate)
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        val stopping = launch { h.coordinator.stop(StopReason.UserStop) }
        runCurrent()
        assertTrue(h.state.phase is SessionPhase.Finishing)
        gate.complete(Unit)
        advanceUntilIdle()
        assertTrue(stopping.isCompleted)
        assertTrue(h.state.exchanges.single().translation is CardTranslation.Done)
        assertEquals(listOf(FakeHistoryRepository.Ended(42, "topic 1")), h.history.ended)
        assertEquals(SessionPhase.Idle, h.state.phase)
    }

    @Test
    fun `should persist a timeout failure when a translation outlives the 20 s drain`() = runTest {
        val h = Harness(this)
        h.provider.onFinal = { awaitCancellation() }
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        h.coordinator.stop(StopReason.UserStop)
        val failed = h.state.exchanges.single().translation as CardTranslation.Failed
        assertEquals(LlmException.Kind.TIMEOUT, failed.kind)
        assertEquals("Stopped before translation finished", failed.message)
        assertTrue(h.history.added.single().translationError)
        assertEquals(1, h.history.ended.size)
    }

    @Test
    fun `should persist a shown final as stopped when the finals drain times out before its translation starts`() = runTest {
        val h = Harness(this, settings = AppSettingsDefaults(TestAssets.shared).value.copy(trainingCapture = true))
        h.training.gate = CompletableDeferred() // training save never returns: the item stays past its card
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        assertEquals(CardTranslation.Pending, h.state.exchanges.single().translation)
        h.coordinator.stop(StopReason.UserStop) // 30 s finals drain times out on the test clock
        val failed = h.state.exchanges.single().translation as CardTranslation.Failed
        assertEquals(LlmException.Kind.TIMEOUT, failed.kind)
        assertEquals("Stopped before translation finished", failed.message)
        val row = h.history.added.single()
        assertTrue(row.translationError)
        assertEquals(1, row.segmentId)
        assertNull(row.audioPath)
        assertEquals(listOf(42L), h.history.ended.map { it.id })
        assertEquals(SessionPhase.Idle, h.state.phase)
    }

    // (12)
    @Test
    fun `should fail the session and end it when the recognizer cannot load`() = runTest {
        val h = Harness(this)
        h.recognizers.loadError = AsrException("corrupt file", AsrException.Kind.MODEL_LOAD_FAILED)
        started(h)
        advanceUntilIdle()
        val phase = h.state.phase as SessionPhase.Failed
        assertTrue(phase.error is SessionError.ModelLoad)
        assertEquals("Speech model failed to load (corrupt file) - delete and re-download it in Settings", phase.error.message)
        assertEquals(listOf(42L), h.history.ended.map { it.id })
        assertTrue(h.source.closed)
    }

    @Test
    fun `should refuse a second start while a session is active`() = runTest {
        val h = Harness(this)
        started(h)
        val e = assertThrows<IllegalStateException> { h.coordinator.start(IdleAudioSource()) }
        assertEquals("session already active", e.message)
        h.coordinator.stop(StopReason.UserStop)
    }

    @Test
    fun `should fail start with the verbatim audio message and end the session row`() = runTest {
        val h = Harness(this)
        h.source.openError = AudioSourceException("Microphone unavailable: AudioRecord not initialized", AudioSourceException.Kind.INIT_FAILED)
        val e = assertThrows<SessionStartException> { h.coordinator.start(h.source) }
        assertEquals(SessionError.Audio(AudioSourceException.Kind.INIT_FAILED, "Microphone unavailable: AudioRecord not initialized"), e.error)
        assertEquals(SessionPhase.Failed(e.error), h.state.phase)
        assertEquals(listOf(42L), h.history.ended.map { it.id })
        // A new session may start after a failure.
        h.source.openError = null
        assertEquals(43L, h.coordinator.start(IdleAudioSource()))
        h.coordinator.stop(StopReason.UserStop)
    }

    @Test
    fun `should stop with AsrFailed after three consecutive native errors`() = runTest {
        val h = Harness(this)
        repeat(3) { h.rec.finals += Step("x", error = AsrException("whisper rc -3", AsrException.Kind.NATIVE_ERROR)) }
        started(h)
        (1..3).forEach { h.utterance(it) }
        advanceUntilIdle()
        val phase = h.state.phase as SessionPhase.Failed
        assertTrue(phase.error is SessionError.Asr)
        assertEquals(3, h.state.metrics.asrErrors)
    }

    @Test
    fun `should keep processing finals after an unexpected recognizer exception`() = runTest {
        val h = Harness(this)
        h.rec.finals += Step("x", error = IllegalStateException("native state corrupted"))
        h.rec.finals += Step("hola amigo como estas hoy")
        started(h)
        h.utterance(1)
        h.utterance(2)
        advanceUntilIdle()
        assertEquals(1, h.state.metrics.asrErrors)
        assertEquals(listOf(2), h.state.exchanges.map { it.seq })
        h.coordinator.stop(StopReason.UserStop)
        assertEquals(SessionPhase.Idle, h.state.phase)
    }

    @Test
    fun `should show a history write failure and keep the card`() = runTest {
        val h = Harness(this)
        h.history.addError = com.xrmech.habla.data.DataException("disk full")
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        assertTrue(SessionNotice.HISTORY_WRITE_FAILED in h.state.notices)
        assertEquals("disk full", h.state.lastHistoryError)
        assertTrue(h.state.exchanges.single().translation is CardTranslation.Done)
        h.coordinator.stop(StopReason.UserStop)
    }

    @Test
    fun `should do nothing on stop and abort without a session`() = runTest {
        val h = Harness(this)
        h.coordinator.stop(StopReason.UserStop)
        h.coordinator.abort()
        assertEquals(SessionPhase.Idle, h.state.phase)
    }

    @Test
    fun `should end the session row on abort`() = runTest {
        val h = Harness(this)
        started(h)
        h.coordinator.abort()
        advanceUntilIdle()
        assertEquals(SessionPhase.Idle, h.state.phase)
        assertEquals(listOf(42L), h.history.ended.map { it.id })
        assertTrue(h.source.closed)
    }

    @Test
    fun `should leave a newer session alone when the stop it interrupted finishes after an abort`() = runTest {
        val h = Harness(this)
        h.rec.finals += Step("hola amigo como estas hoy", gate = CompletableDeferred())
        started(h)
        h.utterance(1)
        advanceUntilIdle()
        launch { h.coordinator.stop(StopReason.UserStop) }
        runCurrent()
        h.coordinator.abort() // service destroyed while the stop is still draining
        val releasesAfterAbort = h.recognizers.releases
        val next = h.coordinator.start(IdleAudioSource())
        advanceUntilIdle()
        assertEquals(43L, next)
        assertEquals(43L, h.state.sessionId)
        assertEquals(SessionPhase.Calibrating, h.state.phase, "the old stop must not publish Idle over the new session")
        assertEquals(releasesAfterAbort, h.recognizers.releases, "the old stop must not free the new session's model")
        h.coordinator.stop(StopReason.UserStop)
        assertEquals(SessionPhase.Idle, h.state.phase)
    }

    // ---- partials (scenarios 2, 9, 10)

    // (2)
    @Test
    fun `should transcribe only the first and latest partial while the first is gated`() = runTest {
        val h = Harness(this)
        val gate = CompletableDeferred<Unit>()
        h.rec.partials += Step("uno dos tres", gate = gate)
        h.rec.partials += Step("uno dos tres cuatro cinco")
        started(h)
        h.segmenter.emitPartial(1, 1.0)
        runCurrent()
        h.segmenter.emitPartial(1, 2.0)
        h.segmenter.emitPartial(1, 3.0)
        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(2, h.rec.requests.count { it.pass == AsrPass.PARTIAL })
        assertEquals("uno dos tres cuatro cinco", h.state.partial?.text)
        h.coordinator.stop(StopReason.UserStop)
        assertEquals(2, h.state.metrics.partialsShown)
    }

    @Test
    fun `should drop a partial result that returns after its segment finalized`() = runTest {
        val h = Harness(this)
        val gate = CompletableDeferred<Unit>()
        h.rec.partials += Step("hola que tal", gate = gate)
        started(h)
        h.segmenter.emitPartial(1)
        runCurrent()
        h.segmenter.emitSegment(1)
        runCurrent()
        gate.complete(Unit)
        advanceUntilIdle()
        assertNull(h.state.partial?.takeIf { it.text == "hola que tal" })
        h.coordinator.stop(StopReason.UserStop)
    }

    // (9)
    @Test
    fun `should run partials in the session source language without voting or detecting`() = runTest {
        val h = Harness(this)
        started(h)
        h.segmenter.emitPartial(1)
        advanceUntilIdle()
        assertEquals("es", h.rec.requests.single { it.pass == AsrPass.PARTIAL }.language)
        assertEquals(0, h.rec.detectCalls)
        assertNull(h.voter.current)
        h.settingsRepo.update { it.copy(direction = Direction.EN_TO_ES) }
        advanceUntilIdle()
        h.rec.partials += Step("hello my friend")
        h.segmenter.emitPartial(1, 2.0)
        advanceUntilIdle()
        assertEquals("en", h.rec.requests.last { it.pass == AsrPass.PARTIAL }.language)
        assertEquals(Direction.EN_TO_ES, h.state.partial?.direction)
        h.coordinator.stop(StopReason.UserStop)
    }

    // (10)
    @Test
    fun `should cancel the in flight quick translation when the final arrives`() = runTest {
        val h = Harness(this)
        val quickGate = CompletableDeferred<Unit>()
        var quickCancelled = false
        h.provider.onQuick = {
            try {
                quickGate.await()
                com.xrmech.habla.translate.LlmResponse("late quick", FakeTranslationProvider.usage(), false)
            } catch (e: CancellationException) {
                quickCancelled = true
                throw e
            }
        }
        started(h)
        h.segmenter.emitPartial(1)
        advanceUntilIdle()
        assertEquals(1, h.provider.requestsOf(UsageKind.QUICK).size)
        h.segmenter.emitSegment(1)
        runCurrent()
        assertTrue(quickCancelled)
        assertTrue(h.state.partial?.finalizing ?: true)
        quickGate.complete(Unit)
        advanceUntilIdle()
        assertNull(h.state.partial?.translation)
        h.coordinator.stop(StopReason.UserStop)
    }

    @Test
    fun `should show a quick translation on the partial line`() = runTest {
        val h = Harness(this)
        started(h)
        h.segmenter.emitPartial(1)
        advanceUntilIdle()
        assertEquals("quick", h.state.partial?.translation)
        h.coordinator.stop(StopReason.UserStop)
    }

    @Test
    fun `should not transcribe partials while thermally paused`() = runTest {
        val h = Harness(this)
        started(h)
        h.thermalStatus.value = 4
        advanceUntilIdle()
        assertTrue(SessionNotice.THERMAL_PAUSE in h.state.notices)
        h.segmenter.emitPartial(1)
        advanceUntilIdle()
        assertEquals(0, h.rec.requests.count { it.pass == AsrPass.PARTIAL })
        h.coordinator.stop(StopReason.UserStop)
    }

    @Test
    fun `should clear the partial line when its final card appears`() = runTest {
        val h = Harness(this)
        started(h)
        h.segmenter.emitPartial(1)
        advanceUntilIdle()
        assertNotNull(h.state.partial)
        h.utterance(1)
        advanceUntilIdle()
        assertNull(h.state.partial)
        assertFalse(h.state.exchanges.isEmpty())
        h.coordinator.stop(StopReason.UserStop)
    }
}
