package com.xrmech.habla.app.ui.settings

import com.xrmech.habla.app.fakes.FakeApiKeyStore
import com.xrmech.habla.app.fakes.FakeLiveController
import com.xrmech.habla.app.fakes.FakeSettingsRepository
import com.xrmech.habla.app.fakes.FakeUsageRepository
import com.xrmech.habla.app.fakes.MainDispatcherExtension
import com.xrmech.habla.app.fakes.NoNetworkDownloader
import com.xrmech.habla.app.fakes.TestAssets
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.state.ModelDownloadController
import com.xrmech.habla.asr.AsrBackend
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.KeyTestResult
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.RegisterExtension
import org.junit.jupiter.api.io.TempDir

class SettingsViewModelTest {
    @JvmField
    @RegisterExtension
    val main = MainDispatcherExtension()

    @TempDir
    lateinit var dir: File

    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
    private val assets = TestAssets.shared
    private val settings = FakeSettingsRepository()
    private val keyStore = FakeApiKeyStore("sk-proj-stored0000000000000000")
    private val usage = FakeUsageRepository()
    private val costMeter = CostMeter(assets.models)
    private val keyHealth = KeyHealthTracker(keyStore.hasKey, appScope)
    private val live = FakeLiveController()
    private val testedWith = mutableListOf<String?>()
    private var nextResult: KeyTestResult = KeyTestResult.Valid(TokenUsage(20, 0, 5, 0, assets.models.quickDefault))
    private val tester = KeyTester { candidate ->
        testedWith += candidate
        nextResult
    }

    @AfterEach
    fun tearDown() {
        appScope.cancel()
    }

    private fun TestScope.newVm(): SettingsViewModel {
        val models = ModelManager(dir, assets.asrModels, NoNetworkDownloader)
        val vm = SettingsViewModel(
            settings, keyStore, tester, costMeter, usage, keyHealth, models,
            ModelDownloadController(models, appScope), assets, live,
        )
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { vm.ui.collect {} }
        return vm
    }

    @Test
    fun `should say Enter a key and keep the stored key when saving blank text`() = runTest {
        val vm = newVm()
        vm.saveKey("   ")
        assertEquals("Enter a key", vm.ui.value.local.keyError)
        assertEquals("sk-proj-stored0000000000000000", keyStore.peek())
    }

    @Test
    fun `should store the key and reset key health on save`() = runTest {
        val vm = newVm()
        keyHealth.report(LlmException.Kind.INVALID_KEY)
        assertEquals(KeyState.INVALID, keyHealth.state.value)
        assertEquals("Key rejected", vm.ui.value.keyStatus)

        vm.saveKey("  sk-proj-new111111111111111111  ")

        assertEquals("sk-proj-new111111111111111111", keyStore.peek())
        assertEquals(KeyState.PRESENT, keyHealth.state.value)
        assertEquals("Key saved", vm.ui.value.keyStatus)
        assertNull(vm.ui.value.local.keyError)
    }

    @Test
    fun `should show No key after clearing`() = runTest {
        val vm = newVm()
        vm.clearKey()
        assertNull(keyStore.peek())
        assertEquals("No key", vm.ui.value.keyStatus)
    }

    @Test
    fun `should test the typed candidate and fall back to the stored key when blank`() = runTest {
        val vm = newVm()
        vm.testKey(" sk-proj-typed22222222222222222 ")
        vm.testKey("")
        assertEquals(listOf("sk-proj-typed22222222222222222", null), testedWith)
    }

    @Test
    fun `should show the key test result texts`() = runTest {
        val vm = newVm()
        vm.testKey(null)
        assertEquals("Key works (${assets.models.quickDefault} replied)", vm.ui.value.local.keyTestMessage)

        nextResult = KeyTestResult.Failed(LlmException("bad", LlmException.Kind.INVALID_KEY, httpStatus = 401))
        vm.testKey(null)
        assertEquals("Key rejected by OpenAI (401)", vm.ui.value.local.keyTestMessage)
        assertEquals(KeyState.INVALID, keyHealth.state.value)

        nextResult = KeyTestResult.Failed(LlmException("no credit", LlmException.Kind.QUOTA, httpStatus = 429))
        vm.testKey(null)
        assertEquals("No credit on this OpenAI account (429) - add billing at platform.openai.com", vm.ui.value.local.keyTestMessage)

        nextResult = KeyTestResult.Failed(LlmException("No OpenAI API key stored", LlmException.Kind.NO_KEY))
        vm.testKey(null)
        assertEquals("Enter a key first", vm.ui.value.local.keyTestMessage)
    }

    @Test
    fun `should not change stored key health when an unsaved candidate fails`() = runTest {
        val vm = newVm()
        nextResult = KeyTestResult.Failed(LlmException("bad", LlmException.Kind.INVALID_KEY, httpStatus = 401))
        vm.testKey("sk-proj-typed22222222222222222")
        assertEquals(KeyState.PRESENT, keyHealth.state.value)
    }

    @Test
    fun `should record a valid test as TEST usage with no session even while listening`() = runTest {
        live.state.value = LiveState(phase = SessionPhase.Listening, sessionId = 7)
        val vm = newVm()
        vm.testKey(null)
        assertEquals(1, usage.rows.size)
        assertNull(usage.rows[0].sessionId)
        assertEquals(UsageKind.TEST, usage.rows[0].kind)
        assertTrue(usage.rows[0].costUsd > 0.0)
        assertEquals(usage.rows[0].costUsd, costMeter.totals.value.totalCostUsd, 1e-12)
        assertEquals(0.0, costMeter.totals.value.sessionCostUsd)
    }

    @Test
    fun `should refuse to delete the speech model while a session runs`() = runTest {
        live.state.value = LiveState(phase = SessionPhase.Listening, sessionId = 7)
        val modelFile = File(dir, "models/${assets.asrModels.default.file}").apply {
            parentFile.mkdirs()
            writeText("weights")
        }
        val vm = newVm()
        vm.deleteModel()
        assertEquals(SettingsViewModel.DELETE_WHILE_LISTENING, vm.ui.value.local.modelMessage)
        assertTrue(modelFile.exists())
        assertTrue(vm.ui.value.sessionActive)
    }

    @Test
    fun `should list every model with deprecated ones marked and write picker changes`() = runTest {
        val vm = newVm()
        val ui = vm.ui.value
        assertEquals(assets.models.models.keys.toList(), ui.finalModels.map { it.id })
        assets.models.models.filterValues { it.deprecated != null }.forEach { (id, info) ->
            assertEquals("$id (deprecated ${info.deprecated})", ui.finalModels.single { it.id == id }.label)
        }
        // Partial-only catalog entries (base, D044) are not offered as the speech model.
        assertEquals(listOf("ggml-small-q8_0", "ggml-small-q5_1", "ggml-small"), ui.speechModels.map { it.id })
        assertEquals(assets.asrModels.models.filter { it.usableForFinals }.map { it.id }, ui.speechModels.map { it.id })

        vm.setMode(Mode.CLASSROOM)
        vm.setTrainingCapture(true)
        assertEquals(Mode.CLASSROOM, settings.current.mode)
        assertTrue(settings.current.trainingCapture)
        assertFalse(ui.sessionActive)
    }

    @Test
    fun `should offer only partial models as the live model and write live model and GPU changes`() = runTest {
        val vm = newVm()
        assertEquals(listOf("", "ggml-base-q8_0", "ggml-base"), vm.ui.value.liveModels.map { it.id })
        assertNull(vm.ui.value.liveModelState, "no live row while partials share the speech model")

        vm.setLiveModel("ggml-base-q8_0")
        assertEquals("ggml-base-q8_0", settings.current.liveModelId)
        assertEquals("ggml-base-q8_0", vm.ui.value.selectedLiveModel?.id)
        assertEquals(ModelState.Missing, vm.ui.value.liveModelState)

        vm.setLiveModel("not-a-model")
        assertEquals("ggml-base-q8_0", settings.current.liveModelId)
        vm.setLiveModel("")
        assertEquals("", settings.current.liveModelId)

        vm.setAsrGpu(true)
        assertTrue(settings.current.asrGpu)
        assertEquals(AsrBackend.gpuBuilt, vm.ui.value.gpuBuilt)
    }
}
