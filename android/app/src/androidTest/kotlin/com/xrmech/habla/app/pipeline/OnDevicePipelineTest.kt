package com.xrmech.habla.app.pipeline

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.app.capture.WavFileAudioSource
import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.app.state.AppSettingsState
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.state.ThermalMonitor
import com.xrmech.habla.app.testing.InMemoryHistory
import com.xrmech.habla.app.testing.InMemoryKeyStore
import com.xrmech.habla.app.testing.InMemorySettings
import com.xrmech.habla.app.testing.InMemoryUsage
import com.xrmech.habla.app.testing.TestFixtures
import com.xrmech.habla.app.translate.MlKitTranslator
import com.xrmech.habla.app.translate.PackState
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.audio.Segmenter
import com.xrmech.habla.audio.SileroVad
import com.xrmech.habla.data.EngineChoice
import com.xrmech.habla.translate.ConversationContext
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.IdiomScanner
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.LlmRequest
import com.xrmech.habla.translate.LlmResponse
import com.xrmech.habla.translate.OnDeviceTranslator
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.TopicTracker
import com.xrmech.habla.translate.TranslationProvider
import com.xrmech.habla.translate.Translator
import com.xrmech.habla.translate.model.Direction
import java.io.BufferedInputStream
import java.io.IOException
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Free tier end to end on a real phone (D050): no OpenAI key, EngineChoice.AUTO resolves to the
 * phone, and every final and preview of the 42 s mixed clip is translated by ML Kit. The cloud
 * provider here fails every call and counts them, so any LLM use fails the test.
 *
 * Run by class with `am instrument` after pushing ggml-small-q8_0 and ggml-base-q8_0 to
 * TestFixtures.DEVICE_MODEL_DIR (see LiveOpenAiPipelineTest). Needs network once for the pack.
 */
@RunWith(AndroidJUnit4::class)
class OnDevicePipelineTest {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val assets = TestFixtures.shared

    @After
    fun tearDown() {
        scope.cancel()
    }

    @Test
    fun mixedClipTranslatesOnThePhoneWithoutAKey(): Unit = runBlocking {
        val speech = checkNotNull(assets.asrModels.byId("ggml-small-q8_0"))
        val live = checkNotNull(assets.asrModels.byId("ggml-base-q8_0"))
        assumeTrue(
            "models not pushed to ${TestFixtures.DEVICE_MODEL_DIR}",
            TestFixtures.installPushedModel(speech) && TestFixtures.installPushedModel(live),
        )
        val mlKit = MlKitTranslator()
        mlKit.refresh()
        if (mlKit.state.value != PackState.Ready) withTimeout(180_000) { mlKit.download() }
        assertEquals(PackState.Ready, mlKit.state.value)

        val offline = object : ModelDownloader {
            override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
                throw IOException("offline")
        }
        val models = ModelManager(TestFixtures.context.noBackupFilesDir, assets.asrModels, offline)
        val settings = InMemorySettings(
            AppSettingsDefaults(assets).value.copy(
                asrModelId = speech.id, liveModelId = live.id, asrGpu = false, direction = Direction.AUTO,
                partialsEnabled = true, partialTranslation = true, trainingCapture = false, firstRunDone = true,
                finalEngine = EngineChoice.AUTO, onDevicePreviews = true,
            ),
        )
        val keyStore = InMemoryKeyStore(null)
        val keyHealth = KeyHealthTracker(keyStore.hasKey, scope)
        keyHealth.state.first { it != KeyState.UNKNOWN }
        val settingsState = AppSettingsState(
            settings, scope,
            hasKey = { keyHealth.state.value != KeyState.ABSENT },
            packReady = { mlKit.state.value == PackState.Ready },
        )
        val cloudCalls = AtomicInteger()
        val noCloud = object : TranslationProvider {
            override suspend fun complete(request: LlmRequest): LlmResponse {
                cloudCalls.incrementAndGet()
                throw LlmException("no cloud in this test", LlmException.Kind.NO_KEY)
            }
        }
        val costMeter = CostMeter(assets.models)
        val ts = { settingsState.translateSettings(assets.models) }
        val translator = Translator(noCloud, assets, ts, costMeter, mlKit)
        val context = ConversationContext(assets.asrProfile)
        val coordinator = PipelineCoordinator(
            assets = assets,
            segmenterFactory = { profile, vad, listener -> AudioSegmenterPort(Segmenter(profile, vad, listener)) },
            vadFactory = { SileroVad(SileroVad.bundledModelBytes()) },
            recognizers = RecognizerHolder(models, assets, Dispatchers.IO),
            voter = LanguageVoter(assets.asrProfile),
            translator = translator,
            quickTranslator = QuickTranslator(noCloud, assets, ts, costMeter, mlKit),
            topicTracker = TopicTracker(translator, context),
            context = context,
            idiomScanner = IdiomScanner(assets.idioms),
            costMeter = costMeter,
            settings = settingsState,
            keyStore = keyStore,
            keyHealth = keyHealth,
            history = InMemoryHistory(),
            usage = InMemoryUsage(),
            training = TrainingCapture { null },
            thermal = ThermalMonitor(kotlinx.coroutines.flow.flowOf(0), scope),
            dispatchers = PipelineDispatchers.production(),
            clock = SystemClock::elapsedRealtime,
        )
        settingsState.awaitLoaded()

        val previews = mutableSetOf<String>()
        val watcher = scope.launch {
            coordinator.state.collect { st ->
                st.partial?.takeIf { it.translation != null && it.translatedOnDevice }?.let { synchronized(previews) { previews += it.translation!! } }
            }
        }
        val started = SystemClock.elapsedRealtime()
        withTimeout(180_000) {
            coordinator.start(WavFileAudioSource(BufferedInputStream(TestFixtures.clip("mixed_es_en_es.wav")), paced = true))
            coordinator.state.first { it.phase is SessionPhase.Finishing || it.phase == SessionPhase.Idle && it.sessionId == null }
            coordinator.state.first { it.phase == SessionPhase.Idle || it.phase is SessionPhase.Failed }
        }
        watcher.cancel()
        val state = coordinator.state.value
        state.exchanges.forEach { card ->
            val t = card.translation
            val text = if (t is CardTranslation.Done) "${t.model}: ${t.result.translated}" else t.toString()
            Log.i(TAG, "#${card.seq} ${card.detectedLanguage} ${card.direction} '${card.rawTranscript}' -> $text")
        }
        Log.i(TAG, "elapsedMs=${SystemClock.elapsedRealtime() - started} cards=${state.exchanges.size} previews=${previews.size} metrics=${coordinator.lastMetricsLine.value}")

        assertTrue("session failed: ${state.phase}", state.phase == SessionPhase.Idle)
        assertTrue("at least 4 of the 6 utterances became cards, got ${state.exchanges.size}", state.exchanges.size >= 4)
        val notOnDevice = state.exchanges.filter { (it.translation as? CardTranslation.Done)?.model != OnDeviceTranslator.MODEL_ID }
        assertTrue("cards not translated on the phone: ${notOnDevice.map { it.translation }}", notOnDevice.isEmpty())
        assertTrue("both directions in the mixed clip", state.exchanges.map { it.direction }.toSet().size == 2)
        assertTrue("at least one on-device preview", previews.isNotEmpty())
        assertEquals("cloud calls", 0, cloudCalls.get())
        assertEquals(0.0, costMeter.totals.value.sessionCostUsd, 0.0)
        assertEquals("no key reported invalid", KeyState.ABSENT, keyHealth.state.value)
    }

    private companion object {
        const val TAG = "HablaOnDeviceE2E"
    }
}
