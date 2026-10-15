            // End of stream stops the session by itself (StopReason.EndOfStream) and drains.
package com.xrmech.habla.app.pipeline

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.app.capture.WavFileAudioSource
import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.app.state.AppSettingsState
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.ThermalMonitor
import com.xrmech.habla.app.testing.InMemoryHistory
import com.xrmech.habla.app.testing.InMemoryKeyStore
import com.xrmech.habla.app.testing.InMemorySettings
import com.xrmech.habla.app.testing.InMemoryUsage
import com.xrmech.habla.app.testing.TestFixtures
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.audio.PartialSnapshot
import com.xrmech.habla.audio.SegmenterListener
import com.xrmech.habla.audio.Segmenter
import com.xrmech.habla.audio.SileroVad
import com.xrmech.habla.audio.SpeechSegment
import com.xrmech.habla.translate.ConversationContext
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.IdiomScanner
import com.xrmech.habla.translate.LlmRequest
import com.xrmech.habla.translate.LlmResponse
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.TopicTracker
import com.xrmech.habla.translate.TranslationProvider
import com.xrmech.habla.translate.Translator
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import java.io.BufferedInputStream
import java.io.File
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End to end on the emulator (design section 11): real Segmenter + SileroVad + FinalPrep/PartialPrep
 * + WhisperCppRecognizer (ggml-tiny-q8_0, 2 threads) + real translate classes over a fake provider
 * + in-memory data, fed by WavFileAudioSource over the public-domain clips (D033). Transcript text
 * is not compared (tiny differs from the server's small; WER parity is asr D6).
 */
@RunWith(AndroidJUnit4::class)
class PipelineSmokeTest {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val assets = TestFixtures.shared

    /** Answers every request instantly with a valid translation; no network. */
    private class FakeProvider : TranslationProvider {
        override suspend fun complete(request: LlmRequest): LlmResponse {
            val usage = TokenUsage(50, 0, 20, 0, request.model)
            val text = when (request.kind) {
                UsageKind.FINAL ->
                    """{"corrected": "c", "translated": "t", "flagged_phrases": [], "confidence": 0.9, "is_correction": false}"""
                else -> "ok"
            }
            return LlmResponse(text, usage, false)
        }
    }

    private class Run(val coordinator: PipelineCoordinator, val segments: List<SpeechSegment>, val partialIds: Set<Int>, val merged: Int)

    private class Built(
        val coordinator: PipelineCoordinator,
        val segments: CopyOnWriteArrayList<SpeechSegment>,
        val partialIds: MutableSet<Int>,
        val merged: AtomicInteger,
    )

    private fun build(): Built {
        val merged = AtomicInteger()
        val cache = TestFixtures.context.cacheDir
        TestFixtures.installTinyModel(cache)
        val offline = object : ModelDownloader {
            override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
                throw IOException("offline")
        }
        val models = ModelManager(cache, TestFixtures.tinyCatalog, offline)
        val settings = InMemorySettings(
            AppSettingsDefaults(assets).value.copy(
                asrModelId = TestFixtures.tinyEntry.id, direction = Direction.AUTO, partialsEnabled = true,
                partialTranslation = true, trainingCapture = false, firstRunDone = true,
            ),
        )
        val settingsState = AppSettingsState(settings, scope)
        val keyStore = InMemoryKeyStore("sk-test-not-a-real-key-000000")
        val costMeter = CostMeter(assets.models)
        val provider = FakeProvider()
        val ts = { settingsState.translateSettings(assets.models) }
        val translator = Translator(provider, assets, ts, costMeter)
        val context = ConversationContext(assets.asrProfile)
        val segments = CopyOnWriteArrayList<SpeechSegment>()
        val partialIds: MutableSet<Int> = ConcurrentHashMap.newKeySet()
        val coordinator = PipelineCoordinator(
            assets = assets,
            segmenterFactory = { profile, vad, listener ->
                // Records raw segments and replays the 1.2 s merge rule on the same stream positions.
                val replay = SegmentMerger(profile.merge, { merged.incrementAndGet() })
                val real = AudioSegmenterPort(
                    Segmenter(
                        profile, vad,
                        object : SegmenterListener by listener {
                            override fun onSegment(segment: SpeechSegment) {
                                segments += segment
                                replay.onSegment(segment)
                                listener.onSegment(segment)
                            }

                            override fun onPartial(snapshot: PartialSnapshot) = listener.onPartial(snapshot)
                        },
                    ),
                )
                object : SegmenterPort by real {
                    private var fed = 0L

                    override fun feed(pcm: ShortArray) {
                        real.feed(pcm)
                        fed += pcm.size
                        replay.onAudio(fed)
                    }

                    override fun flush() {
                        real.flush()
                        replay.flushNow()
                    }
                }
            },
            vadFactory = { SileroVad(SileroVad.bundledModelBytes()) },
            recognizers = RecognizerHolder(models, assets, Dispatchers.IO, threads = 2),
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
        scope.launch { coordinator.state.collect { s -> s.partial?.let { partialIds += it.segmentId } } }
        runBlocking { settingsState.awaitLoaded() }
        return Built(coordinator, segments, partialIds, merged)
    }

    private fun runClip(name: String, paced: Boolean, timeoutMs: Long): Run = runBlocking {
        val b = build()
        val coordinator = b.coordinator
        val started = SystemClock.elapsedRealtime()
        withTimeout(timeoutMs) {
            coordinator.start(WavFileAudioSource(BufferedInputStream(TestFixtures.clip(name)), paced))
            // End of stream stops the session by itself (StopReason.EndOfStream) and drains.
            coordinator.state.first { it.phase is SessionPhase.Finishing || it.phase == SessionPhase.Idle && it.sessionId == null }
            coordinator.state.first { it.phase == SessionPhase.Idle || it.phase is SessionPhase.Failed }
        }
        val elapsed = SystemClock.elapsedRealtime() - started
        val line = coordinator.lastMetricsLine.value
        Log.i("HablaSmoke", "$name paced=$paced elapsedMs=$elapsed metrics=$line")
        File(TestFixtures.context.getExternalFilesDir(null), "smoke-$name.metrics.json").writeText((line ?: "null") + "\n")
        Run(coordinator, b.segments.toList(), b.partialIds.toSet(), b.merged.get())
    }

    @After
    fun tearDown() {
        scope.cancel()
    }

    @Test
    fun mixedClipPacedMatchesTheGoldenSegmentation() {
        val golden = Json.parseToJsonElement(TestFixtures.golden("mixed_es_en_es.json")).jsonObject
        val gSegs = golden["segments"]!!.jsonArray.map { it.jsonObject }
        val gUtts = golden["utterances"]!!.jsonArray.map { it.jsonObject }

        val run = runClip("mixed_es_en_es.wav", paced = true, timeoutMs = 120_000)
        val state = run.coordinator.state.value
        assertTrue("session failed: ${state.phase}", state.phase == SessionPhase.Idle)

        assertEquals("raw segment count", gSegs.size, run.segments.size)
        run.segments.zip(gSegs).forEachIndexed { i, (s, g) ->
            val gs = g["start_sample"]!!.jsonPrimitive.long
            val ge = g["end_sample"]!!.jsonPrimitive.long
            assertTrue("segment $i start ${s.startSample} vs golden $gs", kotlin.math.abs(s.startSample - gs) <= 512)
            assertTrue("segment $i end ${s.endSample} vs golden $ge", kotlin.math.abs(s.endSample - ge) <= 512)
        }

        val m = state.metrics
        val rejected = m.asrRejectedCount + m.asrEmptyCount
        assertEquals("1.2 s merge over the stream gives the golden utterance count", gUtts.size, run.merged)
        assertEquals("exchanges == merged utterances - rejections", gUtts.size - rejected, state.exchanges.size)

        // Language: compare exchanges to the golden utterance holding the same first raw segment.
        var matches = 0
        state.exchanges.forEachIndexed { i, card ->
            val g = gUtts.getOrNull(i) ?: return@forEachIndexed
            if (card.detectedLanguage == g["detected_language"]?.jsonPrimitive?.content) matches++
        }
        assertTrue("detected language matched only $matches of ${gUtts.size}", matches >= gUtts.size - 1)

        run.segments.filter { it.durationSec > 2.0 }.forEach { s ->
            assertTrue("no partial shown for segment ${s.id} (${s.durationSec} s); saw ${run.partialIds}", s.id in run.partialIds)
        }
    }

    @Test
    fun englishClipUnpacedAccountsForEverySegment() {
        val run = runClip("en_aesop_01.wav", paced = false, timeoutMs = 300_000)
        val state = run.coordinator.state.value
        assertTrue("session failed: ${state.phase}", state.phase == SessionPhase.Idle)
        val merged = run.merged
        assertTrue("no segments found", merged > 0)
        val m = state.metrics
        assertEquals(
            "exchanges + rejected + finalsDropped == merged",
            merged,
            state.exchanges.size + m.asrRejectedCount + m.asrEmptyCount + m.finalsDropped,
        )
    }
}
