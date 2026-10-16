package com.xrmech.habla.asr

import com.xrmech.habla.asr.fakes.FakeEngineBinding
import com.xrmech.habla.asr.fakes.FakeEngineBinding.Behaviour
import com.xrmech.habla.asr.fakes.FakeEngineBinding.Companion.failure
import com.xrmech.habla.asr.fakes.FakeEngineBinding.Companion.ok
import com.xrmech.habla.asr.fakes.FakeEngineBinding.Companion.segments
import com.xrmech.habla.asr.internal.AsrLog
import com.xrmech.habla.asr.internal.AsrLogSink
import com.xrmech.habla.shared.AsrProfile
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.yield
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

class WhisperCppRecognizerTest {
    @TempDir
    lateinit var dir: File

    private val fake = FakeEngineBinding()
    private lateinit var model: File
    private val audio = FloatArray(32_000) { 0.01f }
    private val logs = CopyOnWriteArrayList<String>()

    private val es = AsrRequest("es", AsrPass.FINAL)
    private val esPartial = AsrRequest("es", AsrPass.PARTIAL)

    @BeforeEach
    fun setUp() {
        model = File(dir, "ggml-test.bin").apply { writeBytes(ByteArray(16)) }
        AsrLog.sink = AsrLogSink { level, message, _ -> logs += "$level $message" }
    }

    @AfterEach
    fun tearDown() {
        AsrLog.sink = AsrLogSink { _, _, _ -> }
    }

    private fun recognizer(profile: AsrProfile = TestAssets.profile, threads: Int = 2) =
        WhisperCppRecognizer(model, profile, threads, fake)

    private fun blocking(slot: Int, text: String = "blocked") =
        Behaviour { f, _ -> f.blockUntilAbortedOrReleased(slot, ok(text)) }

    private fun slowToAbort(slot: Int, text: String = "slow") =
        Behaviour { f, _ -> f.blockUntilReleased(slot, ok(text)) }

    private suspend fun asrFailure(block: suspend () -> Unit): AsrException {
        try {
            block()
        } catch (e: AsrException) {
            return e
        }
        throw AssertionError("expected AsrException")
    }

    private fun <T> CoroutineScope.start(block: suspend () -> T) =
        async(start = CoroutineStart.UNDISPATCHED) { runCatching { block() } }

    private fun waitFor(timeoutMs: Long = 5_000, condition: () -> Boolean) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (!condition()) {
            check(System.currentTimeMillis() < deadline) { "condition not met in $timeoutMs ms" }
            Thread.sleep(5)
        }
    }

    // ---- preemption -------------------------------------------------------------------------

    @Test
    fun `final aborts a running partial and runs next`() = runBlocking {
        val r = recognizer()
        fake.script(blocking(slot = 1))
        val partial = start { r.transcribe(audio, esPartial) }
        fake.awaitTranscribeCalls(1)
        val final = start { r.transcribe(audio, es) }

        val pe = partial.await().exceptionOrNull() as AsrException
        assertEquals(AsrException.Kind.ABORTED, pe.kind)
        assertEquals("hello", final.await().getOrThrow().text)
        assertEquals(listOf(1, 0), fake.transcribeCalls.map { it.slot })
        r.close()
    }

    @Test
    fun `partial that gets the lock while a final is pending returns ABORTED without running`() = runBlocking {
        val r = recognizer()
        fake.script(slowToAbort(slot = 0, text = "first"))
        val f1 = start { r.transcribe(audio, es) }
        fake.awaitTranscribeCalls(1)
        val partial = start { r.transcribe(audio, esPartial) } // queued first
        val f2 = start { r.transcribe(audio, es) } // queued second, pendingFinals > 0
        fake.release()

        assertEquals("first", f1.await().getOrThrow().text)
        val pe = partial.await().exceptionOrNull() as AsrException
        assertEquals(AsrException.Kind.ABORTED, pe.kind)
        assertEquals("hello", f2.await().getOrThrow().text)
        assertEquals(listOf(0, 0), fake.transcribeCalls.map { it.slot })
        r.close()
    }

    @Test
    fun `abort landing after the partial finished keeps its result and the next partial clears the stale flag`() =
        runBlocking {
            val r = recognizer()
            fake.script(
                Behaviour { f, c ->
                    f.requestAbort(c.handle, c.slot)
                    ok("late")
                },
                Behaviour { f, c -> if (f.isAborted(c.slot)) failure(-6) else ok("next") },
            )
            assertEquals("late", r.transcribe(audio, esPartial).text)
            assertEquals("next", r.transcribe(audio, esPartial).text)
            r.close()
        }

    @Test
    fun `two finals run FIFO and are never aborted`() = runBlocking {
        val r = recognizer()
        fake.script(slowToAbort(slot = 0, text = "one"), Behaviour { _, _ -> ok("two") })
        val f1 = start { r.transcribe(audio, es) }
        fake.awaitTranscribeCalls(1)
        val f2 = start { r.transcribe(audio, es) }
        assertEquals(1, fake.transcribeCalls.size)
        fake.release()
        assertEquals("one", f1.await().getOrThrow().text)
        assertEquals("two", f2.await().getOrThrow().text)
        assertTrue(fake.abortRequests.none { it.second == 0 }, "a final was aborted: ${fake.abortRequests}")
        r.close()
    }

    // ---- cancellation -----------------------------------------------------------------------

    @Test
    fun `cancelling a running call aborts it and keeps the lock until native returns`() = runBlocking {
        val r = recognizer()
        fake.script(slowToAbort(slot = 0))
        val f1 = launch(start = CoroutineStart.UNDISPATCHED) { r.transcribe(audio, es) }
        fake.awaitTranscribeCalls(1)
        f1.cancel()
        yield()
        assertTrue(fake.abortRequests.any { it.second == 0 }, "no abort requested for the FINAL slot")

        val f2 = start { r.transcribe(audio, es) }
        yield()
        assertEquals(1, fake.transcribeCalls.size, "second call ran while the first was still in native code")
        assertFalse(f1.isCompleted, "lock released before native returned")

        fake.release()
        f1.join()
        assertTrue(f1.isCancelled)
        assertEquals("hello", f2.await().getOrThrow().text)
        r.close()
    }

    @Test
    fun `cancel while waiting on the mutex decrements pendingFinals`() = runBlocking {
        val r = recognizer()
        fake.script(slowToAbort(slot = 0))
        val f1 = start { r.transcribe(audio, es) }
        fake.awaitTranscribeCalls(1)
        val f2 = launch(start = CoroutineStart.UNDISPATCHED) { r.transcribe(audio, es) }
        f2.cancel()
        f2.join()
        fake.release()
        f1.await().getOrThrow()

        assertEquals("hello", r.transcribe(audio, esPartial).text)
        assertEquals(listOf(0, 1), fake.transcribeCalls.map { it.slot })
        r.close()
    }

    // ---- close --------------------------------------------------------------------------------

    @Test
    fun `close during a call aborts it, fails waiters and frees once after the call`() = runBlocking {
        val r = recognizer()
        fake.script(blocking(slot = 0))
        val f1 = start { r.transcribe(audio, es) }
        fake.awaitTranscribeCalls(1)
        val waiter = start { r.transcribe(audio, esPartial) }

        r.close()
        r.close()

        val e1 = f1.await().exceptionOrNull() as AsrException
        assertEquals(AsrException.Kind.ABORTED, e1.kind)
        assertEquals("recognizer closed", e1.message)
        val e2 = waiter.await().exceptionOrNull() as AsrException
        assertEquals(AsrException.Kind.ABORTED, e2.kind)
        assertEquals("recognizer closed", e2.message)

        waitFor { fake.frees.isNotEmpty() }
        Thread.sleep(50)
        assertEquals(listOf(1L), fake.frees.toList())
        assertEquals(listOf("transcribe-end:1", "free:1"), fake.events.toList())

        val ise = assertThrows<IllegalStateException> { runBlocking { r.transcribe(audio, es) } }
        assertEquals("recognizer closed", ise.message)
        assertThrows<IllegalStateException> { runBlocking { r.detectLanguage(audio, listOf("es")) } }
        assertThrows<IllegalStateException> { runBlocking { r.preload() } }
    }

    @Test
    fun `an abort request racing close never reaches a freed engine`() {
        val r = recognizer()
        runBlocking { r.preload() }
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        fake.gateNextAbort(entered, release)

        // A FINAL's asFinal() aborts the PARTIAL slot from the caller thread; the fake stalls it there.
        val finalCaller = thread { runBlocking { runCatching { r.transcribe(audio, es) } } }
        assertTrue(entered.await(5, TimeUnit.SECONDS))
        val closer = thread { r.close() }
        // Without the handle lock, close() retires the handle now and the idle executor frees it.
        Thread.sleep(100)
        release.countDown()
        closer.join(5_000)
        finalCaller.join(5_000)

        waitFor { fake.frees.isNotEmpty() }
        assertEquals(listOf("gated-abort-end:1", "free:1"), fake.events.filter { !it.startsWith("transcribe") })
    }

    @Test
    fun `close before any load frees nothing`() {
        val r = recognizer()
        r.close()
        Thread.sleep(20)
        assertTrue(fake.frees.isEmpty())
        assertTrue(fake.loads.isEmpty())
    }

    // ---- native return codes and loading ------------------------------------------------------

    @Test
    fun `rc -7 is NATIVE_ERROR and the next call works`() = runBlocking {
        val r = recognizer()
        fake.script(Behaviour { _, _ -> failure(-7) })
        val e = asrFailure { r.transcribe(audio, es) }
        assertEquals(AsrException.Kind.NATIVE_ERROR, e.kind)
        assertEquals("kv cache allocation failed", e.message)
        assertEquals("hello", r.transcribe(audio, es).text)
        assertEquals(1, fake.loads.size)
        r.close()
    }

    @Test
    fun `other rc is NATIVE_ERROR naming it`() = runBlocking {
        val r = recognizer()
        fake.script(Behaviour { _, _ -> failure(-6) })
        val e = asrFailure { r.transcribe(audio, es) }
        assertEquals(AsrException.Kind.NATIVE_ERROR, e.kind)
        assertEquals("whisper_full rc=-6", e.message)
        r.close()
    }

    @Test
    fun `rc -100 frees the engine and reloads on the same call`() = runBlocking {
        val r = recognizer()
        fake.script(Behaviour { _, _ -> failure(-100) }, Behaviour { _, _ -> ok("again") })
        assertEquals("again", r.transcribe(audio, es).text)
        assertEquals(2, fake.loads.size)
        assertEquals(listOf(1L), fake.frees.toList())
        assertEquals(listOf(1L, 2L), fake.transcribeCalls.map { it.handle })
        r.close()
    }

    @Test
    fun `load returning 0 is MODEL_LOAD_FAILED and the next call retries`() = runBlocking {
        val r = recognizer()
        fake.loadResults += 0L
        val e = asrFailure { r.transcribe(audio, es) }
        assertEquals(AsrException.Kind.MODEL_LOAD_FAILED, e.kind)
        assertEquals("whisper init failed for ggml-test.bin", e.message)
        assertEquals("hello", r.transcribe(audio, es).text)
        assertEquals(2, fake.loads.size)
        r.close()
    }

    @Test
    fun `missing model file is MODEL_MISSING without calling load`() = runBlocking {
        val r = recognizer()
        model.delete()
        val e = asrFailure { r.transcribe(audio, es) }
        assertEquals(AsrException.Kind.MODEL_MISSING, e.kind)
        assertTrue(fake.loads.isEmpty())
        r.close()
    }

    @Test
    fun `a partial model is loaded alongside the main one, none by default`() = runBlocking {
        recognizer().use { it.preload() }
        val partial = File(dir, "ggml-partial.bin").apply { writeBytes(ByteArray(16)) }
        WhisperCppRecognizer(model, TestAssets.profile, 2, fake, partial).use { it.preload() }
        assertEquals(listOf(null, partial.absolutePath), fake.partialLoads.toList())
        assertEquals(listOf(model.absolutePath, model.absolutePath), fake.loads.toList())
    }

    @Test
    fun `missing partial model file is MODEL_MISSING without calling load`() = runBlocking {
        val r = WhisperCppRecognizer(model, TestAssets.profile, 2, fake, File(dir, "absent.bin"))
        val e = asrFailure { r.transcribe(audio, es) }
        assertEquals(AsrException.Kind.MODEL_MISSING, e.kind)
        assertTrue(e.message!!.contains("absent.bin"))
        assertTrue(fake.loads.isEmpty())
        r.close()
    }

    @Test
    fun `library load failure is NATIVE_ERROR with the link error as cause`() = runBlocking {
        val r = recognizer()
        val link = UnsatisfiedLinkError("dlopen failed")
        fake.linkError = link
        val e = asrFailure { r.preload() }
        assertEquals(AsrException.Kind.NATIVE_ERROR, e.kind)
        assertSame(link, e.cause)
        assertNull(r.nativeInfo)
        r.close()
    }

    @Test
    fun `preload loads once on the habla-asr thread and sets nativeInfo`() = runBlocking {
        val r = recognizer()
        assertNull(r.nativeInfo)
        r.preload()
        r.preload()
        assertEquals(1, fake.loads.size)
        assertEquals(model.absolutePath, fake.loads.single())
        assertEquals("habla-asr", fake.threadNames.single())
        assertEquals("base | AVX = 0 | NEON = 0 | SSE3 = 1", r.nativeInfo)
        r.transcribe(audio, es)
        assertEquals("habla-asr", fake.transcribeCalls.single().thread)
        r.close()
    }

    // ---- result mapping ------------------------------------------------------------------

    @Test
    fun `split UTF-8 bytes lose only the broken character`() = runBlocking {
        val r = recognizer()
        val bytes = " está bien".toByteArray(Charsets.UTF_8)
        val cut = bytes.indexOfFirst { it == 0xC3.toByte() } + 1 // inside "á"
        fake.script(Behaviour { _, _ -> segments(bytes.copyOfRange(0, cut), bytes.copyOfRange(cut, bytes.size)) })
        val result = r.transcribe(audio, es)
        assertEquals(" est", result.segments[0].text)
        assertEquals(" bien", result.segments[1].text)
        assertTrue(result.segments.none { it.text.contains(Char(0xFFFD)) })
        assertEquals("est bien", result.text)
        r.close()
    }

    @Test
    fun `segments are joined with Python strip semantics and mapped`() = runBlocking {
        val r = recognizer()
        val nel = Char(0x85)
        val nbsp = Char(0xA0)
        fake.script(Behaviour { _, _ -> ok(" Hola$nel", "${nbsp}amigo ") })
        val result = r.transcribe(audio, es)
        assertEquals("Hola amigo", result.text)
        assertEquals(" Hola$nel", result.segments[0].text) // segment text is not trimmed
        assertEquals(0.0, result.segments[0].startSec)
        assertEquals(1.5, result.segments[0].endSec)
        assertEquals(1.0, result.segments[1].startSec)
        assertEquals(-0.25f, result.segments[0].avgLogprob)
        assertNull(result.segments[1].avgLogprob) // NaN -> null
        assertEquals(0.01f, result.segments[0].noSpeechProb)
        assertEquals(42L, result.elapsedMs)
        assertNull(result.detectedLanguage)
        assertNull(result.languageProb)
        r.close()
    }

    @Test
    fun `empty samples return an empty result without native calls`() = runBlocking {
        val r = recognizer()
        assertEquals(AsrResult("", null, null, emptyList(), 0), r.transcribe(FloatArray(0), es))
        assertTrue(fake.loads.isEmpty())
        assertTrue(fake.transcribeCalls.isEmpty())
        r.close()
    }

    @Test
    fun `zero segments is an empty result, not an error`() = runBlocking {
        val r = recognizer()
        fake.script(Behaviour { _, _ -> segments() })
        val result = r.transcribe(FloatArray(800), es)
        assertEquals("", result.text)
        assertTrue(result.segments.isEmpty())
        r.close()
    }

    @Test
    fun `transcript text is never logged`() = runBlocking {
        val r = recognizer()
        fake.script(Behaviour { _, _ -> ok(" secret phrase") })
        r.transcribe(audio, es)
        assertTrue(logs.isNotEmpty())
        assertTrue(logs.none { it.contains("secret") }, logs.toString())
        r.close()
    }

    // ---- language detection ----------------------------------------------------------------

    @Test
    fun `detection pads short input, prefers the first candidate on ties and returns the raw probability`() =
        runBlocking {
            val r = recognizer()
            fake.detectProbs = FloatArray(FakeEngineBinding.LANG_MAX_ID + 1).also {
                it[3] = 0.25f // es
                it[0] = 0.25f // en
                it[7] = 0.5f // a language outside the candidates
            }
            assertEquals(LanguageGuess("es", 0.25f), r.detectLanguage(FloatArray(10), listOf("es", "en")))
            assertEquals(1600, fake.detectCalls.single().size)
            assertEquals(LanguageGuess("en", 0.25f), r.detectLanguage(audio, listOf("en", "es")))
            assertEquals(audio.size, fake.detectCalls[1].size)

            fake.detectProbs[0] = 0.3f
            assertEquals(LanguageGuess("en", 0.3f), r.detectLanguage(audio, listOf("es", "en")))
            r.close()
        }

    @Test
    fun `detection rc below zero is NATIVE_ERROR`() = runBlocking {
        val r = recognizer()
        fake.detectRcs += -6
        val e = asrFailure { r.detectLanguage(audio, listOf("es", "en")) }
        assertEquals(AsrException.Kind.NATIVE_ERROR, e.kind)
        assertEquals("language detection failed rc=-6", e.message)
        r.close()
    }

    @Test
    fun `detection counts as a final and aborts a running partial`() = runBlocking {
        val r = recognizer()
        fake.script(blocking(slot = 1))
        val partial = start { r.transcribe(audio, esPartial) }
        fake.awaitTranscribeCalls(1)
        fake.detectProbs[3] = 0.9f
        val guess = withTimeout(5_000) { r.detectLanguage(audio, listOf("es", "en")) }
        assertEquals("es", guess.language)
        assertEquals(AsrException.Kind.ABORTED, (partial.await().exceptionOrNull() as AsrException).kind)
        r.close()
    }

    @Test
    fun `detection runs on the final slot by default and on the partial slot for a partial`() = runBlocking {
        val r = recognizer()
        fake.detectProbs[0] = 0.9f // en
        assertEquals("en", r.detectLanguage(audio, listOf("es", "en")).language)
        assertEquals("en", r.detectLanguage(audio, listOf("es", "en"), AsrPass.PARTIAL).language)
        assertEquals(listOf(0, 1), fake.detectSlots)
        r.close()
    }

    @Test
    fun `partial detection that gets the lock while a final is pending returns ABORTED without running`() = runBlocking {
        val r = recognizer()
        fake.script(slowToAbort(slot = 0, text = "first"))
        val f1 = start { r.transcribe(audio, es) }
        fake.awaitTranscribeCalls(1)
        val detect = start { r.detectLanguage(audio, listOf("es", "en"), AsrPass.PARTIAL) } // queued first
        val f2 = start { r.transcribe(audio, es) } // queued second, pendingFinals > 0
        fake.release()

        assertEquals("first", f1.await().getOrThrow().text)
        assertEquals(AsrException.Kind.ABORTED, (detect.await().exceptionOrNull() as AsrException).kind)
        assertEquals("hello", f2.await().getOrThrow().text)
        assertTrue(fake.detectSlots.isEmpty())
        r.close()
    }

    // ---- argument validation ------------------------------------------------------------------

    @Test
    fun `invalid languages and candidates are IllegalArgumentException`() = runBlocking {
        val r = recognizer()
        assertThrows<IllegalArgumentException> { runBlocking { r.transcribe(audio, AsrRequest("auto", AsrPass.FINAL)) } }
        assertThrows<IllegalArgumentException> { runBlocking { r.transcribe(audio, AsrRequest(" ", AsrPass.FINAL)) } }
        assertThrows<IllegalArgumentException> { runBlocking { r.transcribe(audio, AsrRequest("xx", AsrPass.FINAL)) } }
        assertThrows<IllegalArgumentException> { runBlocking { r.detectLanguage(audio, emptyList()) } }
        assertThrows<IllegalArgumentException> { runBlocking { r.detectLanguage(audio, listOf("es", "xx")) } }
        assertTrue(fake.transcribeCalls.isEmpty())
        assertTrue(fake.detectCalls.isEmpty())
        r.close()
    }

    @Test
    fun `constructor validates the profile with the JSON key`() {
        val bad = TestAssets.profileWith(finalBeam = 9)
        val e = assertThrows<IllegalArgumentException> { recognizer(profile = bad) }
        assertEquals("whisper_cpp.final.beam_size must be 1..8, was 9", e.message)
        assertTrue(fake.loads.isEmpty())
    }

    @Test
    fun `threads are clamped to the available processors`() = runBlocking {
        val cores = Runtime.getRuntime().availableProcessors()
        recognizer(threads = 0).also {
            it.transcribe(audio, es)
            it.close()
        }
        recognizer(threads = 10_000).also {
            it.transcribe(audio, es)
            it.close()
        }
        assertEquals(listOf(1, cores), fake.transcribeCalls.map { it.params.threads })
        assertFalse(fake.transcribeCalls.any { it.params.threads > cores })
    }

    @Test
    fun `cancellation exception is rethrown, not mapped`() = runBlocking {
        val r = recognizer()
        fake.script(blocking(slot = 0))
        val job = async(start = CoroutineStart.UNDISPATCHED) { r.transcribe(audio, es) }
        fake.awaitTranscribeCalls(1)
        job.cancel()
        val thrown = runCatching { job.await() }.exceptionOrNull()
        assertTrue(thrown is CancellationException, "got $thrown")
        // Recognizer still usable.
        assertEquals("hello", r.transcribe(audio, es).text)
        r.close()
    }
}
