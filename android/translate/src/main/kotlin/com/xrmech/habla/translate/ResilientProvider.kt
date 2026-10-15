package com.xrmech.habla.translate

import com.xrmech.habla.translate.LlmException.Kind
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.getAndUpdate
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import kotlin.math.min
import kotlin.math.pow
import kotlin.time.ComparableTimeMark
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

/**
 * The single shared retry loop (port of the server's `_call_provider`): rate gate, retry with
 * exponential backoff and `Retry-After`, and degraded mode after repeated timeouts. One instance
 * serves every caller, so the gate and the degraded state are shared as on the server.
 */
class ResilientProvider(
    private val inner: TranslationProvider,
    private val timeSource: TimeSource.WithComparableMarks = TimeSource.Monotonic,
) : TranslationProvider {

    private val log = Log()

    // Kotlin's Mutex is fair: waiters acquire in FIFO order.
    private val gate = Mutex()
    private var lastStart: ComparableTimeMark? = null

    private val stateLock = Any()
    private var consecutiveTimeouts = 0
    private val _degraded = MutableStateFlow(false)

    /** True after [DEGRADED_THRESHOLD] consecutive timeouts; cleared by any success. */
    val degraded: StateFlow<Boolean> = _degraded.asStateFlow()

    override suspend fun complete(request: LlmRequest): LlmResponse {
        awaitGate()
        val maxRetries = if (_degraded.value) 0 else request.maxRetries
        var attempt = 0
        while (true) {
            try {
                val response = attemptOnce(request)
                onSuccess()
                return response
            } catch (e: LlmException) {
                if (e.kind == Kind.TIMEOUT) onTimeout()
                if (!e.retryable || attempt >= maxRetries) throw e
                var waitMs = min(INITIAL_BACKOFF_MS * BACKOFF_FACTOR.pow(attempt), MAX_BACKOFF_MS.toDouble()).toLong()
                val retryAfter = e.retryAfterMs
                if (retryAfter != null) {
                    // A translation that waits this long is stale; the caller's model fallback takes over.
                    if (retryAfter > MAX_BACKOFF_MS) throw e
                    waitMs = maxOf(waitMs, retryAfter)
                }
                log.warn("OpenAI ${request.kind} attempt ${attempt + 1}/${maxRetries + 1} failed (${e.kind}); retrying in $waitMs ms")
                delay(waitMs)
                attempt++
            }
        }
    }

    /** Spaces call starts by [MIN_REQUEST_INTERVAL_MS]; a cancelled waiter leaves [lastStart] alone. */
    private suspend fun awaitGate() {
        gate.withLock {
            val last = lastStart
            if (last != null) {
                val remaining = MIN_REQUEST_INTERVAL_MS.milliseconds - last.elapsedNow()
                if (remaining.isPositive()) delay(remaining)
            }
            lastStart = timeSource.markNow()
        }
    }

    private suspend fun attemptOnce(request: LlmRequest): LlmResponse {
        if (!_degraded.value) return inner.complete(request)
        return try {
            withTimeout(DEGRADED_TIMEOUT_MS) { inner.complete(request) }
        } catch (e: TimeoutCancellationException) {
            // A caller's own (outer) timeout or cancellation must propagate untouched; only our
            // deadline is converted.
            currentCoroutineContext().ensureActive()
            throw LlmException("OpenAI request timed out (degraded mode, ${DEGRADED_TIMEOUT_MS} ms)", Kind.TIMEOUT, cause = e)
        }
    }

    private fun onSuccess() {
        val wasDegraded = synchronized(stateLock) {
            consecutiveTimeouts = 0
            _degraded.getAndUpdate { false }
        }
        if (wasDegraded) log.info("OpenAI recovered; leaving degraded mode")
    }

    private fun onTimeout() {
        val entered = synchronized(stateLock) {
            consecutiveTimeouts++
            consecutiveTimeouts >= DEGRADED_THRESHOLD && !_degraded.getAndUpdate { true }
        }
        if (entered) {
            log.warn("OpenAI entered degraded mode after $DEGRADED_THRESHOLD consecutive timeouts: no retries, ${DEGRADED_TIMEOUT_MS} ms cap")
        }
    }

    companion object {
        const val MAX_RETRIES_CLOUD = 3
        const val INITIAL_BACKOFF_MS = 1_000L
        const val BACKOFF_FACTOR = 2.0
        const val MAX_BACKOFF_MS = 15_000L
        const val DEGRADED_THRESHOLD = 3
        const val DEGRADED_TIMEOUT_MS = 8_000L
        const val MIN_REQUEST_INTERVAL_MS = 500L
    }
}
