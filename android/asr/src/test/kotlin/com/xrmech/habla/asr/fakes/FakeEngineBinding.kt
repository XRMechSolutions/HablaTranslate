package com.xrmech.habla.asr.fakes

import com.xrmech.habla.asr.internal.DecodeParams
import com.xrmech.habla.asr.internal.EngineBinding
import com.xrmech.habla.asr.internal.NativeResult
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * In-memory stand-in for the native engine. Abort flags behave like the C++ atomics; scripted
 * transcribe behaviours can block until their slot is aborted or the test releases them.
 */
internal class FakeEngineBinding : EngineBinding {
    private val lock = Object()

    val loads = CopyOnWriteArrayList<String>()
    val frees = CopyOnWriteArrayList<Long>()
    val transcribeCalls = CopyOnWriteArrayList<Call>()
    val detectCalls = CopyOnWriteArrayList<FloatArray>()
    val abortRequests = CopyOnWriteArrayList<Pair<Long, Int>>()
    val threadNames = CopyOnWriteArrayList<String>()

    /** "transcribe-end:<handle>" and "free:<handle>" in the order they happened. */
    val events = CopyOnWriteArrayList<String>()

    /** Handles returned by successive load() calls; when empty, 1, 2, 3... */
    val loadResults = ArrayDeque<Long>()
    private var nextHandle = 1L

    var linkError: UnsatisfiedLinkError? = null
    var langIds: Map<String, Int> = mapOf("en" to 0, "es" to 3, "fr" to 5)
    var detectProbs: FloatArray = FloatArray(LANG_MAX_ID + 1)
    val detectRcs = ArrayDeque<Int>()

    @Volatile
    private var abortGate: Pair<CountDownLatch, CountDownLatch>? = null

    private val behaviours = ArrayDeque<Behaviour>()
    private val aborted = BooleanArray(2)
    private var released = 0

    data class Call(val handle: Long, val slot: Int, val samples: Int, val params: DecodeParams, val thread: String)

    fun interface Behaviour {
        fun run(fake: FakeEngineBinding, call: Call): NativeResult
    }

    fun script(vararg b: Behaviour) = synchronized(lock) { behaviours.addAll(b) }

    /** Lets one blocked call finish normally (rc 0) if it was not aborted. */
    fun release() = synchronized(lock) {
        released++
        lock.notifyAll()
    }

    /**
     * The next requestAbort counts down [entered], then blocks until [release] opens, then records
     * "gated-abort-end:<handle>" in [events] (a caller thread stalled between reading the handle and
     * writing the native flag).
     */
    fun gateNextAbort(entered: CountDownLatch, release: CountDownLatch) {
        abortGate = entered to release
    }

    fun isAborted(slot: Int): Boolean = synchronized(lock) { aborted[slot] }

    /** Waits until [n] transcribe calls have started. */
    fun awaitTranscribeCalls(n: Int, timeoutMs: Long = 5_000) {
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs)
        synchronized(lock) {
            while (transcribeCalls.size < n) {
                val left = TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime())
                check(left > 0) { "only ${transcribeCalls.size} of $n transcribe calls started" }
                lock.wait(left)
            }
        }
    }

    /** Blocks the calling (ASR) thread until [slot] is aborted (-> rc -6) or [release]d (-> ok). */
    fun blockUntilAbortedOrReleased(slot: Int, ok: NativeResult): NativeResult = synchronized(lock) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (!aborted[slot] && released == 0) {
            val left = TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime())
            check(left > 0) { "blocked call never aborted or released" }
            lock.wait(left)
        }
        if (aborted[slot]) return failure(-6)
        released--
        ok
    }

    /** Blocks until released, ignoring aborts until then (a native call slow to notice the flag). */
    fun blockUntilReleased(slot: Int, ok: NativeResult, onAbort: NativeResult = failure(-6)): NativeResult = synchronized(lock) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (released == 0) {
            val left = TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime())
            check(left > 0) { "blocked call never released" }
            lock.wait(left)
        }
        released--
        if (aborted[slot]) onAbort else ok
    }

    override val variant: String
        get() {
            linkError?.let { throw it }
            return "base"
        }

    override fun systemInfo(): String {
        linkError?.let { throw it }
        return "AVX = 0 | NEON = 0 | SSE3 = 1"
    }

    override fun load(modelPath: String, flashAttn: Boolean): Long {
        linkError?.let { throw it }
        loads += modelPath
        threadNames += Thread.currentThread().name
        return loadResults.removeFirstOrNull() ?: nextHandle++
    }

    override fun free(handle: Long) {
        frees += handle
        events += "free:$handle"
    }

    override fun langId(code: String): Int = langIds[code] ?: -1

    override fun langMaxId(): Int = LANG_MAX_ID

    override fun detectLanguage(handle: Long, samples: FloatArray, threads: Int, probsOut: FloatArray): Int {
        detectCalls += samples
        val rc = detectRcs.removeFirstOrNull() ?: 0
        if (rc < 0) return rc
        detectProbs.copyInto(probsOut)
        return rc
    }

    override fun transcribe(handle: Long, slot: Int, samples: FloatArray, p: DecodeParams): NativeResult {
        val call = Call(handle, slot, samples.size, p, Thread.currentThread().name)
        val behaviour = synchronized(lock) {
            transcribeCalls += call
            lock.notifyAll()
            behaviours.removeFirstOrNull()
        } ?: Behaviour { _, _ -> ok("hello") }
        return behaviour.run(this, call).also { events += "transcribe-end:$handle" }
    }

    override fun requestAbort(handle: Long, slot: Int) {
        abortGate?.let { (entered, release) ->
            abortGate = null
            entered.countDown()
            check(release.await(10, TimeUnit.SECONDS)) { "abort gate never released" }
            events += "gated-abort-end:$handle"
        }
        abortRequests += handle to slot
        synchronized(lock) {
            aborted[slot] = true
            lock.notifyAll()
        }
    }

    override fun clearAbort(handle: Long, slot: Int) {
        synchronized(lock) { aborted[slot] = false }
    }

    companion object {
        const val LANG_MAX_ID = 99

        fun ok(vararg texts: String): NativeResult = segments(*texts.map { it.toByteArray() }.toTypedArray())

        fun segments(vararg bytes: ByteArray): NativeResult = NativeResult(
            rc = 0,
            wallMs = 42,
            texts = arrayOf(*bytes),
            t0 = LongArray(bytes.size) { it * 100L },
            t1 = LongArray(bytes.size) { it * 100L + 150 },
            noSpeechProb = FloatArray(bytes.size) { 0.01f },
            avgLogprob = FloatArray(bytes.size) { if (it == 0) -0.25f else Float.NaN },
            textTokens = IntArray(bytes.size) { if (it == 0) 3 else 0 },
        )

        fun failure(rc: Int): NativeResult =
            NativeResult(rc, 7, emptyArray(), LongArray(0), LongArray(0), FloatArray(0), FloatArray(0), IntArray(0))
    }
}
