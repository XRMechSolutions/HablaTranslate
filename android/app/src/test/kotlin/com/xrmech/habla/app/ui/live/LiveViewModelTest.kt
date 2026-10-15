package com.xrmech.habla.app.ui.live

import com.xrmech.habla.app.fakes.FakeApiKeyStore
import com.xrmech.habla.app.fakes.FakeLiveController
import com.xrmech.habla.app.fakes.FakeSessionControl
import com.xrmech.habla.app.fakes.FakeSettingsRepository
import com.xrmech.habla.app.fakes.FakeVocabRepository
import com.xrmech.habla.app.fakes.MainDispatcherExtension
import com.xrmech.habla.app.fakes.NoNetworkDownloader
import com.xrmech.habla.app.fakes.TestAssets
import com.xrmech.habla.app.fakes.installVerifiedModel
import com.xrmech.habla.app.pipeline.CardTranslation
import com.xrmech.habla.app.pipeline.ExchangeCard
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.SessionError
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.state.ModelDownloadController
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.TranslationResult
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.RegisterExtension
import org.junit.jupiter.api.io.TempDir

class LiveViewModelTest {
    @JvmField
    @RegisterExtension
    val main = MainDispatcherExtension()

    @TempDir
    lateinit var dir: File

    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
    private val assets = TestAssets.shared
    private val settings = FakeSettingsRepository()
    private val keyStore = FakeApiKeyStore("sk-proj-stored0000000000000000")
    private val keyHealth = KeyHealthTracker(keyStore.hasKey, appScope)
    private val live = FakeLiveController()
    private val service = FakeSessionControl()
    private val online = MutableStateFlow(true)
    private val vocab = FakeVocabRepository()

    @AfterEach
    fun tearDown() {
        appScope.cancel()
    }

    private fun TestScope.newVm(modelReady: Boolean = true): Pair<LiveViewModel, MutableList<LiveEffect>> {
        if (modelReady) installVerifiedModel(dir, assets.asrModels.byId(settings.current.asrModelId)!!)
        val models = ModelManager(dir, assets.asrModels, NoNetworkDownloader)
        val vm = LiveViewModel(
            live, service, settings, models, ModelDownloadController(models, appScope), keyHealth, online, vocab, assets,
            startTimeoutMs = 10_000,
        )
        val effects = mutableListOf<LiveEffect>()
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { vm.ui.collect {} }
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { vm.effects.collect { effects += it } }
        return vm to effects
    }

    @Test
    fun `should request the microphone permission before starting`() = runTest {
        val (vm, effects) = newVm()
        vm.onListen()
        assertEquals(listOf<LiveEffect>(LiveEffect.RequestMicPermission), effects)
        assertEquals(0, service.starts)
    }

    @Test
    fun `should open app settings when the permission is permanently denied`() = runTest {
        val (vm, effects) = newVm()
        vm.setMicPermission(MicPermission.PERMANENTLY_DENIED)
        vm.onListen()
        assertEquals(listOf<LiveEffect>(LiveEffect.OpenAppSettings), effects)
        assertEquals(0, service.starts)
    }

    @Test
    fun `should scroll to the model banner and start nothing when the model is missing`() = runTest {
        val (vm, effects) = newVm(modelReady = false)
        vm.setMicPermission(MicPermission.GRANTED)
        vm.onListen()
        assertEquals(listOf<LiveEffect>(LiveEffect.ScrollToBanner(BannerKind.MODEL_MISSING)), effects)
        assertEquals(0, service.starts)
        assertTrue(vm.ui.value.allBanners.any { it.kind == BannerKind.MODEL_MISSING })
    }

    @Test
    fun `should start the service when permission and model are ready`() = runTest {
        val (vm, effects) = newVm()
        vm.setMicPermission(MicPermission.GRANTED)
        assertTrue(vm.ui.value.status.modelState is ModelState.Ready)
        vm.onListen()
        assertEquals(1, service.starts)
        assertTrue(effects.isEmpty())
        assertTrue(vm.ui.value.local.startRequested)
        assertFalse(vm.ui.value.canListen)
    }

    @Test
    fun `should ignore a second Listen tap before the session reports back`() = runTest {
        val (vm, _) = newVm()
        vm.setMicPermission(MicPermission.GRANTED)
        vm.onListen()
        vm.onListen()
        assertEquals(1, service.starts)
        live.state.value = LiveState(phase = SessionPhase.Calibrating, sessionId = 1)
        vm.onListen()
        assertEquals(1, service.starts)
        assertFalse(vm.ui.value.local.startRequested)
    }

    @Test
    fun `should allow Listen again after the guard times out or the session fails`() = runTest {
        val (vm, _) = newVm()
        vm.setMicPermission(MicPermission.GRANTED)
        vm.onListen()
        advanceTimeBy(10_001)
        vm.onListen()
        assertEquals(2, service.starts)
        live.state.value = LiveState(phase = SessionPhase.Failed(SessionError.ModelLoad("Speech model missing")))
        vm.onListen()
        assertEquals(3, service.starts)
    }

    @Test
    fun `should stop through the service only while a session is active`() = runTest {
        val (vm, _) = newVm()
        vm.onStop()
        assertEquals(0, service.stops)
        live.state.value = LiveState(phase = SessionPhase.Listening, sessionId = 1)
        vm.onStop()
        assertEquals(1, service.stops)
    }

    @Test
    fun `should assemble pipeline status from live state, model, key and connectivity`() = runTest {
        val (vm, _) = newVm()
        live.state.value = LiveState(phase = SessionPhase.Listening, sessionId = 1, queueDepth = 2, lastAsrError = "boom")
        online.value = false
        val s = vm.ui.value.status
        assertTrue(s.listening)
        assertFalse(s.calibrating)
        assertEquals(2, s.queueDepth)
        assertEquals(KeyState.PRESENT, s.keyState)
        assertEquals("boom", s.lastError)
        assertFalse(s.online)
        assertTrue(s.modelState is ModelState.Ready)
        assertTrue(vm.ui.value.allBanners.any { it.kind == BannerKind.OFFLINE })

        online.value = true
        assertTrue(vm.ui.value.status.online)
        assertTrue(vm.ui.value.allBanners.none { it.kind == BannerKind.OFFLINE })
    }

    @Test
    fun `should label auto direction with the voted language`() = runTest {
        settings.update { it.copy(direction = Direction.AUTO) }
        val (vm, _) = newVm()
        assertEquals("Auto", vm.ui.value.directionLabel)
        live.state.value = LiveState(votedLanguage = "es")
        assertEquals("Auto (ES)", vm.ui.value.directionLabel)
        vm.setDirection(Direction.EN_TO_ES)
        assertEquals(Direction.EN_TO_ES, settings.current.direction)
        assertEquals("EN -> ES", vm.ui.value.directionLabel)
    }

    @Test
    fun `should save a chip phrase to vocab once with the corrected sentence`() = runTest {
        val (vm, _) = newVm()
        val phrase = FlaggedPhrase(phrase = "echar una mano", meaning = "to help")
        val card = ExchangeCard(
            seq = 3, exchangeId = 11, createdAtMs = 0, rawTranscript = "me echas una mano", detectedLanguage = "es",
            direction = Direction.ES_TO_EN,
            translation = CardTranslation.Done(TranslationResult("¿Me echas una mano?", "Can you help me?"), listOf(phrase), "m", 10),
        )
        vm.saveVocab(card, phrase)
        vm.saveVocab(card, phrase)
        assertEquals(1, vocab.saved.size)
        assertEquals(11L, vocab.saved[0].exchangeId)
        assertEquals("¿Me echas una mano?", vocab.saved[0].sentence)
        assertTrue(vm.ui.value.isSaved(card, phrase))
    }

    @Test
    fun `should show the setup card until done and finish it when every row completes`() = runTest {
        settings.update { it.copy(firstRunDone = false) }
        val (vm, _) = newVm()
        val rows = vm.ui.value.setup!!
        assertFalse(rows.micDone)
        assertTrue(rows.modelDone)
        assertTrue(rows.keyDone)
        vm.setMicPermission(MicPermission.GRANTED)
        assertTrue(settings.current.firstRunDone)
        assertNull(vm.ui.value.setup)
    }

    @Test
    fun `should mark the key row done when the user skips it`() = runTest {
        keyStore.clear()
        settings.update { it.copy(firstRunDone = false) }
        val (vm, _) = newVm(modelReady = false)
        assertFalse(vm.ui.value.setup!!.keyDone)
        vm.skipKey()
        assertTrue(vm.ui.value.setup!!.keyDone)
        assertFalse(settings.current.firstRunDone)
        vm.dismissSetup()
        assertTrue(settings.current.firstRunDone)
    }
}
