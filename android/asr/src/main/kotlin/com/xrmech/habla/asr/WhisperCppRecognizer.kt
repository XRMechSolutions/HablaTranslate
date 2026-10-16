package com.xrmech.habla.asr

import com.xrmech.habla.asr.internal.AsrLog
import com.xrmech.habla.asr.internal.DecodeParams
import com.xrmech.habla.asr.internal.EngineBinding
import com.xrmech.habla.asr.internal.NativeEngineBinding
import com.xrmech.habla.asr.internal.NativeResult
import com.xrmech.habla.asr.internal.PyText
import com.xrmech.habla.asr.internal.WhisperNative
import com.xrmech.habla.asr.internal.WhisperParamsBuilder
import com.xrmech.habla.shared.AsrProfile
import java.io.File
import java.lang.ref.PhantomReference
import java.lang.ref.ReferenceQueue
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * whisper.cpp recognizer: one model, two `whisper_state`s (FINAL slot 0, PARTIAL slot 1), one FIFO
 * [Mutex], one single-thread executor `habla-asr` (design.md sections 4 and 7).
 *
 * - Construction validates [profile] (IllegalArgumentException naming the JSON key), clamps [threads]
 *   to `1..availableProcessors()` and does no I/O. The library and model load lazily inside the mutex
 *   on the executor (or in [preload]); a failed load is retried on the next call.
 * - FINAL preempts PARTIAL: a final bumps `pendingFinals` before waiting for the lock and aborts the
 *   partial slot; a partial that gets the lock while a final is pending returns ABORTED unrun.
 * - Cancelling a caller aborts its native call and keeps the lock until native code has returned.
 * - [close] aborts both slots, frees native memory on the executor after any running call and never
 *   blocks the caller. There is no finalizer; debug builds WARN if the recognizer leaks while loaded.
 *
 * Transcript text is never logged (counts, rc, timings and audio_ctx only).
 */
class WhisperCppRecognizer internal constructor(
    private val modelFile: File,
    private val profile: AsrProfile,
    threads: Int,
    private val binding: EngineBinding,
) : SpeechRecognizer {

    constructor(modelFile: File, profile: AsrProfile, threads: Int) :
        this(modelFile, profile, threads, NativeEngineBinding())

    private val threads: Int = threads.coerceIn(1, maxOf(1, Runtime.getRuntime().availableProcessors()))

    // Single thread: every native call comes from one stable thread, blocking JNI never occupies a
    // coroutine dispatcher thread, and `free` queued by close() runs only after the in-flight call.
    // Tasks are submitted only while holding [mutex] (plus close()'s free), so the queue holds at most two.
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "habla-asr").apply { isDaemon = true }
    }

    private val mutex = Mutex()
    private val pendingFinals = AtomicInteger()
    private val handle = AtomicLong(0)

    /**
     * Guards "read [handle] + requestAbort" on caller threads against the handle being retired for
     * `free`. Every swap of a live handle to 0 happens under this lock, so once it is released no other
     * thread can still be about to write into that engine's abort flags; `free` itself runs outside
     * it (on the executor). Held only for an atomic store or a swap, never for native work.
     */
    private val handleLock = Any()
    private val closed = AtomicBoolean(false)

    /** Kotlin mirror of the native per-slot abort flags ("rc != 0 and abort flag set" means ABORTED). */
    private val abortFlags = arrayOf(AtomicBoolean(false), AtomicBoolean(false))
    private val langIds = ConcurrentHashMap<String, Int>()

    @Volatile
    private var libraryReady = false

    private val leakWatch: LeakWatch.Watch? = LeakWatch.register(this, handle)

    /** FINAL runs that reused the preceding detection's mel and encoder output (D042); for tests. */
    internal val reusedDetections = AtomicInteger()

    /** `"<variant> | <whisper_print_system_info()>"`; null before the native library is loaded. */
    @Volatile
    var nativeInfo: String? = null
        private set

    init {
        WhisperParamsBuilder.validate(profile)
    }

    /** Loads library, model and both states now. Idempotent. */
    suspend fun preload() {
        checkOpen()
        mutex.withLock {
            if (closed.get()) throw closedAborted()
            runJob(WhisperNative.SLOT_FINAL) {
                ensureLibrary()
                ensureEngine()
            }
        }
    }

    override suspend fun detectLanguage(samples: FloatArray, candidates: List<String>): LanguageGuess {
        checkOpen()
        require(candidates.isNotEmpty()) { "candidates must not be empty" }
        candidates.forEach { requireUsableCode(it) }
        val padded = if (samples.size < MIN_DETECT_SAMPLES) samples.copyOf(MIN_DETECT_SAMPLES) else samples
        return asFinal {
            runJob(WhisperNative.SLOT_FINAL) { detectJob(padded, candidates) }
        }
    }

    override suspend fun transcribe(samples: FloatArray, request: AsrRequest): AsrResult {
        checkOpen()
        requireUsableCode(request.language)
        if (samples.isEmpty()) return AsrResult("", null, null, emptyList(), 0)
        return when (request.pass) {
            AsrPass.FINAL -> asFinal {
                runJob(WhisperNative.SLOT_FINAL) { transcribeJob(WhisperNative.SLOT_FINAL, request, samples, it) }
            }
            AsrPass.PARTIAL -> mutex.withLock {
                if (closed.get()) throw closedAborted()
                runJob(WhisperNative.SLOT_PARTIAL) { transcribeJob(WhisperNative.SLOT_PARTIAL, request, samples, it) }
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        requestAbortSlot(WhisperNative.SLOT_FINAL)
        requestAbortSlot(WhisperNative.SLOT_PARTIAL)
        val old = synchronized(handleLock) { handle.getAndSet(0) }
        if (old != 0L) {
            try {
                executor.execute { binding.free(old) }
            } catch (e: RejectedExecutionException) {
                AsrLog.w("free after close could not be queued", e)
            }
        }
        executor.shutdown()
        leakWatch?.close()
    }

    // ---- serialization and the job pattern -------------------------------------------------

    /** FINAL work (transcribe or detection): counts in pendingFinals and aborts a running partial. */
    private suspend fun <T> asFinal(block: suspend () -> T): T {
        pendingFinals.incrementAndGet()
        try {
            requestAbortSlot(WhisperNative.SLOT_PARTIAL)
            return mutex.withLock {
                if (closed.get()) throw closedAborted()
                block()
            }
        } finally {
            pendingFinals.decrementAndGet()
        }
    }

    /**
     * Runs [work] on the executor and awaits it. On cancellation: abort [slot], wait non-cancellably
     * until native code has returned (the caller still holds the mutex, so no other call can reuse a
     * running state), then rethrow.
     */
    private suspend fun <T> runJob(slot: Int, work: (jobAbort: AtomicBoolean) -> T): T {
        val jobAbort = AtomicBoolean(false)
        val deferred = CompletableDeferred<T>()
        try {
            executor.execute {
                try {
                    deferred.complete(work(jobAbort))
                } catch (t: Throwable) {
                    deferred.completeExceptionally(t)
                }
            }
        } catch (e: RejectedExecutionException) {
            throw closedAborted()
        }
        try {
            return deferred.await()
        } catch (e: CancellationException) {
            if (!deferred.isCompleted) {
                jobAbort.set(true)
                requestAbortSlot(slot)
                withContext(NonCancellable) { deferred.join() }
            }
            throw e
        }
    }

    // ---- executor-thread work --------------------------------------------------------------

    private fun transcribeJob(slot: Int, request: AsrRequest, samples: FloatArray, jobAbort: AtomicBoolean): AsrResult {
        ensureLibrary()
        requireKnownLanguage(request.language)
        var h = ensureEngine()
        // Clear first, then check: a final that set the flag just before the clear has already
        // incremented pendingFinals, so the check below still sees it.
        clearSlot(h, slot)
        if (closed.get()) throw closedAborted()
        if (jobAbort.get()) throw AsrException("cancelled", AsrException.Kind.ABORTED)
        if (slot == WhisperNative.SLOT_PARTIAL && pendingFinals.get() > 0) {
            throw AsrException("partial preempted by a pending final", AsrException.Kind.ABORTED)
        }

        val params = WhisperParamsBuilder.build(request.pass, request.language, samples.size, profile, threads)
        var result = binding.transcribe(h, slot, samples, params)
        if (result.rc == WhisperNative.RC_NO_STATE && !abortFlags[slot].get() && !closed.get()) {
            AsrLog.w("pass=${request.pass} slot $slot has no state; reloading the model")
            h = reloadEngine(h)
            if (abortFlags[slot].get()) binding.requestAbort(h, slot)
            result = binding.transcribe(h, slot, samples, params)
        }
        return mapResult(result, slot, request.pass, params)
    }

    private fun detectJob(samples: FloatArray, candidates: List<String>): LanguageGuess {
        ensureLibrary()
        val ids = candidates.map { requireKnownLanguage(it) }
        var h = ensureEngine()
        if (closed.get()) throw closedAborted()
        val probs = FloatArray(binding.langMaxId() + 1)
        var rc = binding.detectLanguage(h, samples, threads, probs)
        if (rc == WhisperNative.RC_NO_STATE && !closed.get()) {
            AsrLog.w("detection slot has no state; reloading the model")
            h = reloadEngine(h)
            rc = binding.detectLanguage(h, samples, threads, probs)
        }
        if (closed.get()) throw closedAborted()
        if (rc < 0) throw AsrException("language detection failed rc=$rc", AsrException.Kind.NATIVE_ERROR)

        // Strict '>' in candidate order: the first candidate wins ties, like Python max().
        var best = 0
        for (i in 1 until ids.size) {
            if (probs[ids[i]] > probs[ids[best]]) best = i
        }
        return LanguageGuess(candidates[best], probs[ids[best]])
    }

    private fun mapResult(r: NativeResult, slot: Int, pass: AsrPass, params: DecodeParams): AsrResult {
        if (r.rc != 0) {
            AsrLog.i("pass=$pass rc=${r.rc} wallMs=${r.wallMs} audioCtx=${params.audioCtx}")
            if (closed.get()) throw closedAborted()
            if (abortFlags[slot].get()) throw AsrException("aborted", AsrException.Kind.ABORTED)
            if (r.rc == WhisperNative.RC_KV_CACHE_FAILED) {
                throw AsrException("kv cache allocation failed", AsrException.Kind.NATIVE_ERROR)
            }
            throw AsrException("whisper_full rc=${r.rc}", AsrException.Kind.NATIVE_ERROR)
        }
        val segments = r.texts.indices.map { i ->
            AsrSegment(
                startSec = r.t0[i] / 100.0,
                endSec = r.t1[i] / 100.0,
                text = decodeUtf8(r.texts[i]),
                avgLogprob = r.avgLogprob[i].takeUnless { it.isNaN() },
                noSpeechProb = r.noSpeechProb[i],
            )
        }
        val text = PyText.strip(segments.joinToString(" ") { PyText.strip(it.text) })
        if (r.reusedDetection) reusedDetections.incrementAndGet()
        AsrLog.i(
            "pass=$pass rc=0 segments=${segments.size} chars=${PyText.length(text)} " +
                "wallMs=${r.wallMs} audioCtx=${params.audioCtx} threads=${params.threads} reused=${r.reusedDetection}",
        )
        return AsrResult(text, null, null, segments, r.wallMs)
    }

    private fun ensureLibrary() {
        if (libraryReady) return
        try {
            nativeInfo = "${binding.variant} | ${binding.systemInfo()}"
        } catch (e: LinkageError) {
            throw AsrException("native library failed to load", AsrException.Kind.NATIVE_ERROR, e)
        }
        libraryReady = true
    }

    private fun ensureEngine(): Long {
        handle.get().let { if (it != 0L) return it }
        if (closed.get()) throw closedAborted()
        ensureLibrary()
        if (!modelFile.isFile || !modelFile.canRead()) {
            throw AsrException("model file missing or unreadable: ${modelFile.name}", AsrException.Kind.MODEL_MISSING)
        }
        val started = System.nanoTime()
        val h = try {
            binding.load(modelFile.absolutePath, FLASH_ATTN)
        } catch (e: LinkageError) {
            throw AsrException("native library failed to load", AsrException.Kind.NATIVE_ERROR, e)
        }
        if (h == 0L) {
            throw AsrException("whisper init failed for ${modelFile.name}", AsrException.Kind.MODEL_LOAD_FAILED)
        }
        if (!handle.compareAndSet(0L, h)) {
            binding.free(h)
            throw closedAborted()
        }
        if (closed.get()) {
            // close() ran while we were loading and saw no handle; free it here unless it already took it.
            if (synchronized(handleLock) { handle.compareAndSet(h, 0L) }) binding.free(h)
            throw closedAborted()
        }
        AsrLog.i("model ${modelFile.name} loaded in ${(System.nanoTime() - started) / 1_000_000} ms")
        return h
    }

    /** Frees a broken engine and loads a fresh one, unless close() has already taken the handle. */
    private fun reloadEngine(old: Long): Long {
        if (!synchronized(handleLock) { handle.compareAndSet(old, 0L) }) throw closedAborted()
        binding.free(old)
        return ensureEngine()
    }

    private fun requestAbortSlot(slot: Int) {
        abortFlags[slot].set(true)
        synchronized(handleLock) {
            val h = handle.get()
            if (h != 0L) binding.requestAbort(h, slot)
        }
    }

    private fun clearSlot(h: Long, slot: Int) {
        abortFlags[slot].set(false)
        binding.clearAbort(h, slot)
    }

    // ---- validation ------------------------------------------------------------------------

    private fun checkOpen() {
        check(!closed.get()) { "recognizer closed" }
    }

    private fun requireUsableCode(code: String) {
        require(code.isNotBlank()) { "language must not be blank" }
        require(code != "auto") { "language must not be \"auto\"; detect it with detectLanguage first" }
    }

    /** Executor thread, library loaded. */
    private fun requireKnownLanguage(code: String): Int {
        val id = langIds.getOrPut(code) { binding.langId(code) }
        require(id >= 0) { "language '$code' is unknown to Whisper" }
        return id
    }

    private fun closedAborted() = AsrException("recognizer closed", AsrException.Kind.ABORTED)

    private companion object {
        /** 100 ms at 16 kHz; shorter input makes detection fail with "offset past end" (-2). */
        const val MIN_DETECT_SAMPLES = 1_600

        const val FLASH_ATTN = true

        /** UTF-8 with replacement, then drop U+FFFD, which only comes from a character split at a segment edge. */
        fun decodeUtf8(bytes: ByteArray): String = String(bytes, Charsets.UTF_8).replace(REPLACEMENT_CHAR, "")

        private val REPLACEMENT_CHAR = Char(0xFFFD).toString()
    }
}

/**
 * Debug builds only: WARN when a recognizer becomes unreachable while its model is still loaded
 * (close() was never called). No finalizer: it would run on the GC thread concurrently with a native
 * call. Built on PhantomReference because `java.lang.ref.Cleaner` needs API 33 and minSdk is 29.
 */
private object LeakWatch {
    private val queue = ReferenceQueue<Any>()

    /** Keeps the phantom references themselves reachable until they are enqueued or closed. */
    private val live: MutableSet<Watch> = ConcurrentHashMap.newKeySet()
    private val started = AtomicBoolean(false)

    class Watch(owner: Any, val handle: AtomicLong) : PhantomReference<Any>(owner, queue) {
        fun close() {
            live.remove(this)
            clear()
        }
    }

    fun register(owner: Any, handle: AtomicLong): Watch? {
        if (!BuildConfig.DEBUG) return null
        startOnce()
        return Watch(owner, handle).also { live += it }
    }

    // One daemon thread per process, debug builds only. Driven by the GC enqueuing references; it
    // blocks in queue.remove() and is never cancelled (dies with the process); an unexpected error
    // ends the watch, which only loses the debug WARN.
    private fun startOnce() {
        if (!started.compareAndSet(false, true)) return
        Thread({
            try {
                while (true) {
                    val watch = queue.remove() as Watch
                    live.remove(watch)
                    if (watch.handle.get() != 0L) {
                        AsrLog.w(
                            "WhisperCppRecognizer became unreachable without close(); " +
                                "native model memory leaks until process death",
                        )
                    }
                }
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }, "habla-asr-leakwatch").apply { isDaemon = true }.start()
    }
}
