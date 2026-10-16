package com.xrmech.habla.app.pipeline

import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.xrmech.habla.app.capture.WavFileAudioSource
import com.xrmech.habla.app.seed.BuildSeed
import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.app.state.AppSettingsState
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.ThermalMonitor
import com.xrmech.habla.app.testing.InMemoryHistory
import com.xrmech.habla.app.testing.InMemoryKeyStore
import com.xrmech.habla.app.testing.InMemorySettings
import com.xrmech.habla.app.testing.InMemoryUsage
import com.xrmech.habla.app.testing.TestFixtures
import com.xrmech.habla.asr.AsrBackend
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.audio.Segmenter
import com.xrmech.habla.audio.SileroVad
import com.xrmech.habla.shared.AsrModelEntry
import com.xrmech.habla.translate.ConversationContext
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.IdiomScanner
import com.xrmech.habla.translate.OpenAiHttp
import com.xrmech.habla.translate.OpenAiProvider
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.ResilientProvider
import com.xrmech.habla.translate.TopicTracker
import com.xrmech.habla.translate.Translator
import com.xrmech.habla.translate.model.Direction
import java.io.BufferedInputStream
import java.io.File
import java.io.IOException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Personal flavor, real device, real money (about $0.01): the production pipeline end to end on
 * the owner's phone. Mic replaced by the 42 s public-domain mixed clip (paced), speech on the
 * A56 profile (small q8_0 finals, base q8_0 live partials, GPU unless `habla.gpu=false`),
 * translation through the real OpenAI stack with the key seeded from android/.env (D022).
 *
 * Not part of any suite: run it by class with `am instrument` after
 * `gradlew :asr:prepareDeviceBench -Phabla.benchModels=ggml-small-q8_0,ggml-base-q8_0`. It also
 * copies both models into the app's model directory, so the installed app needs no download.
 */
@RunWith(AndroidJUnit4::class)
class LiveOpenAiPipelineTest {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val assets = TestFixtures.shared
    private val args get() = InstrumentationRegistry.getArguments()

    @After
    fun tearDown() {
        scope.cancel()
    }

    /** Copies a catalog model pushed under /data/local/tmp (shell-readable only) into the app's models dir. */
    private fun installModel(entry: AsrModelEntry): Boolean {
        val dir = File(TestFixtures.context.noBackupFilesDir, "models").apply { mkdirs() }
        val dst = File(dir, entry.file)
        if (dst.length() != entry.sizeBytes || TestFixtures.sha256(dst) != entry.sha256) {
            val shell = InstrumentationRegistry.getInstrumentation().uiAutomation
                .executeShellCommand("cat ${TestFixtures.DEVICE_MODEL_DIR}/${entry.file}")
            ParcelFileDescriptor.AutoCloseInputStream(shell).use { i -> dst.outputStream().use { i.copyTo(it, 1 shl 20) } }
            if (dst.length() != entry.sizeBytes || TestFixtures.sha256(dst) != entry.sha256) {
                dst.delete()
                return false
            }
        }
        // ModelManager's marker, so fileFor() reports the file Ready without hashing.
        File(dir, entry.file + ".verified").writeText("${entry.sha256} ${entry.sizeBytes} ${dst.lastModified()}\n")
        return true
    }

    @Test
    fun mixedClipTranslatesThroughOpenAiOnTheA56Profile() = runBlocking {
        val key = BuildSeed.key
        assumeTrue("no seeded key (android/.env)", key != null)
        val speech = checkNotNull(assets.asrModels.byId("ggml-small-q8_0"))
        val live = checkNotNull(assets.asrModels.byId("ggml-base-q8_0"))
        assumeTrue("models not pushed to ${TestFixtures.DEVICE_MODEL_DIR}", installModel(speech) && installModel(live))
        val gpu = args.getString("habla.gpu") != "false"

        val offline = object : ModelDownloader {
            override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
                throw IOException("offline")
        }
        val models = ModelManager(TestFixtures.context.noBackupFilesDir, assets.asrModels, offline)
        val settings = InMemorySettings(
            AppSettingsDefaults(assets).value.copy(
                asrModelId = speech.id, liveModelId = live.id, asrGpu = gpu, direction = Direction.AUTO,
                partialsEnabled = true, partialTranslation = true, trainingCapture = false, firstRunDone = true,
            ),
        )
        val settingsState = AppSettingsState(settings, scope)
        val keyStore = InMemoryKeyStore(key)
        val costMeter = CostMeter(assets.models)
        val http = OpenAiHttp.create(OpenAiHttp.okHttpEngine(), debugLogging = false)
        try {
            val provider = ResilientProvider(OpenAiProvider(http, { keyStore.get() }, assets.models))
            val ts = { settingsState.translateSettings(assets.models) }
            val translator = Translator(provider, assets, ts, costMeter)
            val context = ConversationContext(assets.asrProfile)
            val coordinator = PipelineCoordinator(
                assets = assets,
                segmenterFactory = { profile, vad, listener -> AudioSegmenterPort(Segmenter(profile, vad, listener)) },
                vadFactory = { SileroVad(SileroVad.bundledModelBytes()) },
                recognizers = RecognizerHolder(models, assets, Dispatchers.IO),
                voter = LanguageVoter(assets.asrProfile),
                translator = translator,
                quickTranslator = QuickTranslator(provider, assets, ts, costMeter),
                topicTracker = TopicTracker(translator, context),
                context = context,
                idiomScanner = IdiomScanner(assets.idioms),
                costMeter = costMeter,
                settings = settingsState,
                keyStore = keyStore,
                keyHealth = KeyHealthTracker(keyStore.hasKey, scope),
                history = InMemoryHistory(),
                usage = InMemoryUsage(),
                training = TrainingCapture { null },
                thermal = ThermalMonitor(kotlinx.coroutines.flow.flowOf(0), scope),
                dispatchers = PipelineDispatchers.production(),
                clock = SystemClock::elapsedRealtime,
            )
            settingsState.awaitLoaded()

            val started = SystemClock.elapsedRealtime()
            withTimeout(180_000) {
                coordinator.start(WavFileAudioSource(BufferedInputStream(TestFixtures.clip("mixed_es_en_es.wav")), paced = true))
                // End of stream stops the session by itself and drains the final queue.
                coordinator.state.first { it.phase is SessionPhase.Finishing || it.phase == SessionPhase.Idle && it.sessionId == null }
                coordinator.state.first { it.phase == SessionPhase.Idle || it.phase is SessionPhase.Failed }
            }
            val state = coordinator.state.value
            val elapsed = SystemClock.elapsedRealtime() - started
            state.exchanges.forEach { card ->
                val t = card.translation
                val text = if (t is CardTranslation.Done) "${t.model}: ${t.result.translated}" else t.toString()
                Log.i(TAG, "#${card.seq} ${card.detectedLanguage} ${card.direction} '${card.rawTranscript}' -> $text")
            }
            val cost = costMeter.totals.value.sessionCostUsd
            Log.i(TAG, "engine=${AsrBackend.loadedVariant} elapsedMs=$elapsed cards=${state.exchanges.size} costUsd=$cost metrics=${coordinator.lastMetricsLine.value}")

            assertTrue("session failed: ${state.phase}", state.phase == SessionPhase.Idle)
            if (gpu && AsrBackend.gpuBuilt) assertEquals("GPU engine loaded", "vk", AsrBackend.loadedVariant)
            assertTrue("at least 4 of the 6 utterances became cards, got ${state.exchanges.size}", state.exchanges.size >= 4)
            val failed = state.exchanges.filter { it.translation !is CardTranslation.Done }
            assertTrue("untranslated cards: ${failed.map { it.translation }}", failed.isEmpty())
            assertTrue("both directions in the mixed clip", state.exchanges.map { it.direction }.toSet().size == 2)
            assertTrue("cost $cost stays under \$0.05", cost < 0.05)
        } finally {
            http.close()
        }
    }

    private companion object {
        const val TAG = "HablaLiveE2E"
    }
}
