package com.xrmech.habla.asr

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import com.xrmech.habla.asr.AsrTestFixtures.SAMPLE_RATE
import com.xrmech.habla.asr.internal.CpuVariant
import com.xrmech.habla.asr.internal.CpuVariantSelector
import com.xrmech.habla.shared.AsrProfile
import java.io.File
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Emulator suite on `ggml-tiny-q8_0` (x86_64 loads `habla_asr_base`). Checks mechanics, not accuracy:
 * tiny on an emulated CPU is far from the A56 + small configuration that D5/D6 measure.
 * Prerequisite: `android\gradlew.bat -p android :asr:prepareTestModels`.
 */
@RunWith(AndroidJUnit4::class)
class WhisperCppRecognizerInstrumentedTest {
    private lateinit var model: File
    private val profile: AsrProfile get() = AsrTestFixtures.profile
    private val candidates: List<String> get() = AsrTestFixtures.languages.codes
    private val threads = minOf(4, Runtime.getRuntime().availableProcessors())
    private val open = mutableListOf<WhisperCppRecognizer>()

    @Before
    fun setUp() {
        model = AsrTestFixtures.installTestModel()
    }

    @After
    fun tearDown() {
        open.forEach { it.close() }
    }

    private fun recognizer(p: AsrProfile = profile) = WhisperCppRecognizer(model, p, threads).also { open += it }

    private fun parityProfile(): AsrProfile =
        profile.copy(whisperCpp = profile.whisperCpp.copy(partial = profile.whisperCpp.partial.copy(level = "parity")))

    private val smoke = listOf(
        "es_marti_02.wav" to 2,
        "es_marti_02.wav" to 4,
        "en_aesop_01.wav" to 1,
        "en_aesop_01.wav" to 0,
    )

    @Test
    fun libraryLoadsTheSelectedVariantAndPreloads() = runBlocking<Unit> {
        val r = recognizer()
        r.preload()
        val info = r.nativeInfo
        assertNotNull(info)
        info!!
        // base on the x86_64 emulator, v86 on the A56 (asimddp + i8mm + asimdhp).
        val expected = CpuVariantSelector.select(System.getProperty("os.arch").orEmpty(), File("/proc/cpuinfo").readText())
        assertTrue(info, info.startsWith("${expected.label} | "))
        assertTrue(info, info.length > "${expected.label} | ".length)
        if (expected == CpuVariant.V86) {
            assertTrue(info, info.contains("DOTPROD = 1") && info.contains("MATMUL_INT8 = 1"))
        }
        r.preload()
    }

    @Test
    fun smokeUtterancesTranscribeAndDetectTheirLanguage() = runBlocking<Unit> {
        val r = recognizer()
        for ((clip, index) in smoke) {
            val u = AsrTestFixtures.utterance(clip, index)
            val guess = r.detectLanguage(u.samples, candidates)
            assertEquals("${u.clip}#${u.index} detected", u.language, guess.language)
            assertTrue(guess.probability in 0f..1f)
            val result = r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL))
            val wer = AsrTestFixtures.wer(u.transcript, result.text)
            android.util.Log.i("HablaAsrTest", "${u.clip}#${u.index} wer=$wer elapsedMs=${result.elapsedMs}")
            assertTrue("${u.clip}#${u.index} WER $wer > 0.6 (ref '${u.transcript}', got '${result.text}')", wer <= 0.6)
            assertFalse(result.text.contains(Char(0xFFFD)))
            assertTrue(result.segments.isNotEmpty())
        }
    }

    @Test
    fun finalAfterDetectionReusesItsEncoderPassWithIdenticalOutput() = runBlocking<Unit> {
        val r = recognizer()
        val u = AsrTestFixtures.utterance(smoke[0].first, smoke[0].second)
        val other = AsrTestFixtures.utterance(smoke[1].first, smoke[1].second)

        // Fresh mel and encoder: transcribe with no detection in front of it.
        val plain = r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL))
        assertEquals(0, r.reusedDetections.get())

        r.detectLanguage(u.samples, candidates)
        val reused = r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL))
        assertEquals("the final after detection reuses its encoder pass", 1, r.reusedDetections.get())
        assertEquals(plain.text, reused.text)
        assertEquals(plain.segments.map { it.startSec to it.endSec }, reused.segments.map { it.startSec to it.endSec })

        // Detection on other audio must not leak into this final.
        r.detectLanguage(other.samples, candidates)
        val afterOther = r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL))
        assertEquals(1, r.reusedDetections.get())
        assertEquals(plain.text, afterOther.text)

        // A reuse is single-use: a second final on the same samples recomputes.
        r.detectLanguage(u.samples, candidates)
        r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL))
        val again = r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL))
        assertEquals(2, r.reusedDetections.get())
        assertEquals(plain.text, again.text)
        android.util.Log.i(
            "HablaAsrTest",
            "reuse plain=${plain.elapsedMs}ms reused=${reused.elapsedMs}ms afterOther=${afterOther.elapsedMs}ms",
        )
    }

    @Test
    fun partialDetectionRunsOnThePartialSlotAndItsPartialReusesTheEncoderPass() = runBlocking<Unit> {
        // D049: partials before the first vote in auto direction detect on the partial slot. With a
        // full-context partial level (audio_ctx 0) the partial then skips its encoder pass.
        val r = WhisperCppRecognizer(model, profile, threads, partialModelFile = model).also { open += it }
        val u = AsrTestFixtures.utterance(smoke[0].first, smoke[0].second)
        val plain = r.transcribe(u.samples, AsrRequest(u.language, AsrPass.PARTIAL))

        assertEquals(u.language, r.detectLanguage(u.samples, candidates, AsrPass.PARTIAL).language)
        assertEquals(u.language, r.detectLanguage(u.samples, candidates).language)
        val partial = r.transcribe(u.samples, AsrRequest(u.language, AsrPass.PARTIAL))
        assertEquals(plain.text, partial.text)
        if (profile.whisperCpp.partial.level.contains("full_ctx")) {
            assertEquals("the partial after a partial detection reuses its encoder pass", 1, r.reusedDetections.get())
        }
        // The final slot's own detection is still there for the final.
        r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL))
        assertTrue(r.reusedDetections.get() >= 1)
    }

    @Test
    fun reuseCoversOnlyTheFirstWindowOfAudioLongerThan30Seconds() = runBlocking<Unit> {
        val r = recognizer()
        // > 30 s: whisper encodes a second window at a later offset, which must not hit the cache.
        val parts = smoke.map { (clip, index) -> AsrTestFixtures.utterance(clip, index).samples }
        var long = FloatArray(0)
        var i = 0
        while (long.size < 34 * 16_000) long += parts[i++ % parts.size]
        val language = AsrTestFixtures.utterance(smoke[0].first, smoke[0].second).language

        val plain = r.transcribe(long, AsrRequest(language, AsrPass.FINAL))
        r.detectLanguage(long, candidates)
        val reused = r.transcribe(long, AsrRequest(language, AsrPass.FINAL))
        assertEquals(1, r.reusedDetections.get())
        assertEquals(plain.text, reused.text)
        assertEquals(plain.segments.map { it.startSec to it.endSec }, reused.segments.map { it.startSec to it.endSec })
        assertTrue("expected a second window, got ${plain.segments.lastOrNull()?.endSec}", (plain.segments.lastOrNull()?.endSec ?: 0.0) > 30.0)
    }

    @Test
    fun aSeparatePartialModelServesPartialsWhileFinalsKeepTheMainOne() = runBlocking<Unit> {
        // Same tiny file twice: proves the second context and its state, not a quality difference.
        val r = WhisperCppRecognizer(model, profile, threads, partialModelFile = model).also { open += it }
        val u = AsrTestFixtures.utterance(smoke[0].first, smoke[0].second)
        val single = recognizer()
        val expectedPartial = single.transcribe(u.samples, AsrRequest(u.language, AsrPass.PARTIAL)).text
        val expectedFinal = single.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL)).text

        assertEquals(u.language, r.detectLanguage(u.samples, candidates).language)
        assertEquals(expectedFinal, r.transcribe(u.samples, AsrRequest(u.language, AsrPass.FINAL)).text)
        assertEquals(expectedPartial, r.transcribe(u.samples, AsrRequest(u.language, AsrPass.PARTIAL)).text)
    }

    @Test
    fun spanishAccentsSurviveWithoutReplacementCharacters() = runBlocking<Unit> {
        val r = recognizer()
        val texts = listOf(2, 3, 8, 9).map { i ->
            val u = AsrTestFixtures.utterance("es_marti_02.wav", i)
            r.transcribe(u.samples, AsrRequest("es", AsrPass.FINAL))
        }
        for (res in texts) {
            assertFalse(res.text, res.text.contains(Char(0xFFFD)))
            res.segments.forEach { assertFalse(it.text.contains(Char(0xFFFD))) }
        }
        val all = texts.joinToString(" ") { it.text }
        assertTrue("no accented character in '$all'", all.any { it in "áéíóúñÁÉÍÓÚÑ" })
    }

    @Test
    fun veryShortAndEmptyInputGiveEmptyResults() = runBlocking<Unit> {
        val r = recognizer()
        val empty = r.transcribe(FloatArray(0), AsrRequest("es", AsrPass.FINAL))
        assertEquals("", empty.text)
        assertTrue(empty.segments.isEmpty())
        val short = r.transcribe(FloatArray(SAMPLE_RATE / 20), AsrRequest("es", AsrPass.FINAL))
        assertEquals("", short.text)
        assertTrue(short.segments.isEmpty())
        val partial = r.transcribe(FloatArray(SAMPLE_RATE / 20), AsrRequest("es", AsrPass.PARTIAL))
        assertTrue(partial.segments.isEmpty())
        // Detection pads to 100 ms and must not fail on empty input.
        assertTrue(r.detectLanguage(FloatArray(0), candidates).language in candidates)
    }

    @Test
    fun finalAbortsARunningParityPartial() = runBlocking<Unit> {
        val r = recognizer(parityProfile())
        r.preload()
        val long = AsrTestFixtures.clip("es_marti_02.wav").copyOfRange(0, 28 * SAMPLE_RATE)
        val short = AsrTestFixtures.utterance("es_marti_02.wav", 2)
        var aborted = false
        for (attempt in 1..3) {
            val partial = async(Dispatchers.Default, start = CoroutineStart.DEFAULT) {
                runCatching { r.transcribe(long, AsrRequest("es", AsrPass.PARTIAL)) }
            }
            delay(50)
            val final = r.transcribe(short.samples, AsrRequest("es", AsrPass.FINAL))
            assertTrue(final.segments.isNotEmpty())
            val outcome = partial.await()
            val e = outcome.exceptionOrNull()
            if (e is AsrException && e.kind == AsrException.Kind.ABORTED) {
                aborted = true
                break
            }
            assertTrue("partial failed unexpectedly: $e", outcome.isSuccess)
        }
        assertTrue("PARTIAL was never ABORTED by a FINAL in 3 attempts", aborted)
    }

    @Test
    fun cancellingAFinalLeavesTheRecognizerUsable() = runBlocking<Unit> {
        val r = recognizer(parityProfile())
        r.preload()
        val long = AsrTestFixtures.clip("es_marti_02.wav").copyOfRange(0, 28 * SAMPLE_RATE)
        val job = launch(Dispatchers.Default) { r.transcribe(long, AsrRequest("es", AsrPass.FINAL)) }
        delay(300)
        withTimeout(30_000) {
            job.cancel()
            job.join()
        }
        assertTrue(job.isCancelled)
        val u = AsrTestFixtures.utterance("en_aesop_01.wav", 1)
        val result = r.transcribe(u.samples, AsrRequest("en", AsrPass.FINAL))
        assertTrue(AsrTestFixtures.wer(u.transcript, result.text) <= 0.6)
    }

    @Test
    fun closeDuringACallAbortsIt() = runBlocking<Unit> {
        val r = recognizer(parityProfile())
        r.preload()
        val long = AsrTestFixtures.clip("es_marti_02.wav").copyOfRange(0, 28 * SAMPLE_RATE)
        val call = async(Dispatchers.Default) { runCatching { r.transcribe(long, AsrRequest("es", AsrPass.FINAL)) } }
        delay(300)
        r.close()
        val e = withTimeout(30_000) { call.await() }.exceptionOrNull()
        assertTrue("expected ABORTED, got $e", e is AsrException && e.kind == AsrException.Kind.ABORTED)
        try {
            r.transcribe(long, AsrRequest("es", AsrPass.FINAL))
            fail("call after close must throw")
        } catch (expected: IllegalStateException) {
            assertEquals("recognizer closed", expected.message)
        }
    }

    @Test
    fun pipelineFinalOnTinyDetectsVotesAndTranscribes() = runBlocking<Unit> {
        val r = recognizer()
        val voter = LanguageVoter(profile)
        val pipeline = AsrPipeline(r, voter, profile, AsrTestFixtures.languages)
        val u = AsrTestFixtures.utterance("en_aesop_01.wav", 1)
        val out = pipeline.final(u.samples, sourceLanguage = "es")
        assertEquals("en", out.language)
        assertEquals("en", voter.current)
        assertEquals(RejectReason.OK, out.reason)
        val partial = pipeline.partial(u.samples, "es")
        assertEquals("en", partial.language)
    }

    @LargeTest
    @Test
    fun realDownloadOfTinyResumesAfterCancellation() = runBlocking<Unit> {
        val filesDir = File(AsrTestFixtures.context.noBackupFilesDir, "model-download-test").apply {
            deleteRecursively()
            mkdirs()
        }
        val entry = AsrTestFixtures.tinyEntry
        KtorModelDownloader().use { http ->
            val manager = ModelManager(filesDir, AsrTestFixtures.testCatalog, http)
            val job = launch(Dispatchers.IO) { manager.ensure(entry.id) }
            withTimeout(300_000) {
                manager.state.first { it is ModelState.Downloading && it.bytes >= 10L * 1024 * 1024 || it is ModelState.Failed }
            }
            assertFalse("download failed: ${manager.state.value} (${entry.url}, sha256 ${entry.sha256})", manager.state.value is ModelState.Failed)
            job.cancel()
            job.join()
            assertEquals(ModelState.Missing, manager.state.value)
            val part = File(filesDir, "models/${entry.file}.part")
            val kept = part.length()
            assertTrue("part kept for resume ($kept bytes)", kept >= 10L * 1024 * 1024 && kept < entry.sizeBytes)
            assertTrue(File(filesDir, "models/${entry.file}.part.json").isFile)

            withTimeout(600_000) { manager.ensure(entry.id) }
            val state = manager.state.value
            assertTrue("expected Ready, got $state (${entry.url}, sha256 ${entry.sha256})", state is ModelState.Ready)
            val file = manager.fileFor(entry.id)!!
            assertEquals(entry.sizeBytes, file.length())
            assertEquals(entry.sha256, AsrTestFixtures.sha256(file))
        }
        filesDir.deleteRecursively()
    }
}
