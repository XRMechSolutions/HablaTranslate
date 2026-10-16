package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.TranslationResult
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withTimeoutOrNull
import java.time.Instant
import java.time.format.DateTimeParseException
import java.util.UUID

/** Result of one Pro translation: the gateway's answer, or why the phone should take over. */
sealed interface GatewayResult {
    data class Ok(val result: TranslationResult, val model: String) : GatewayResult
    data class Unavailable(val error: GatewayException) : GatewayResult
}

/**
 * What the Pro path last saw, for the UI ("Pro allowance used - translating on this phone").
 * [blockedUntilMs] > now means the gateway is skipped without a network call.
 */
data class GatewayStatus(val lastError: GatewayException? = null, val blockedUntilMs: Long = 0)

/**
 * Pro translation through llm-gateway (D051). The gateway owns prompts, parsing, re-asks and model
 * failover. This class only holds a 15 s budget, resends the same requestId when the gateway says
 * `retry` (a dropped socket or an in-flight duplicate then returns the stored result), and stops
 * calling for a while after refusals that will not change per request (no entitlement, allowance
 * used, app switched off). Callers fall back to [OnDeviceTranslator] on [GatewayResult.Unavailable].
 */
class GatewayTranslator(
    private val client: GatewayClient,
    /** `context.max_context_exchanges` (5); the contract allows up to 8. */
    private val maxContext: Int,
    private val clock: () -> Long = System::currentTimeMillis,
    private val newRequestId: () -> String = { UUID.randomUUID().toString() },
    private val sleep: suspend (Long) -> Unit = { delay(it) },
) {
    private val log = Log()
    private val _status = MutableStateFlow(GatewayStatus())
    val status: StateFlow<GatewayStatus> = _status.asStateFlow()

    /** Clears a refusal block, e.g. after sign-in or a verified purchase. */
    fun reset() {
        _status.value = GatewayStatus()
    }

    suspend fun translate(
        transcript: String,
        direction: Direction,
        mode: Mode,
        context: List<ContextExchange>,
        topicSummary: String,
    ): GatewayResult {
        blocked()?.let { return GatewayResult.Unavailable(it) }
        if (transcript.length > MAX_TRANSCRIPT) {
            return GatewayResult.Unavailable(GatewayException("VALIDATION_ERROR", "Transcript over $MAX_TRANSCRIPT chars", fallback = GatewayFallback.ON_DEVICE))
        }
        val request = GatewayTranslateRequest(
            requestId = newRequestId(),
            direction = (if (direction == Direction.AUTO) Direction.ES_TO_EN else direction).serialName,
            mode = if (mode == Mode.CLASSROOM) "classroom" else "conversation",
            transcript = transcript,
            speakerLabel = Translator.SPEAKER_LABEL,
            context = context.takeLast(maxContext.coerceAtMost(CONTRACT_MAX_CONTEXT)).map {
                GatewayContextLine(Translator.SPEAKER_LABEL, it.source.take(MAX_CONTEXT_TEXT), it.translated.take(MAX_CONTEXT_TEXT))
            },
            topicSummary = topicSummary.take(MAX_SUMMARY),
        )
        return when (val r = withRetry(BUDGET_MS) { client.translate(request) }) {
            is Attempt.Ok -> {
                if (_status.value.lastError != null) _status.value = GatewayStatus()
                GatewayResult.Ok(r.value.toResult(), r.value.meta.model.ifEmpty { "gateway" })
            }
            is Attempt.Failed -> GatewayResult.Unavailable(r.error.also(::record))
        }
    }

    /** The new summary, or null (the caller keeps the previous one). */
    suspend fun topic(previous: String, latestSource: String, latestTranslation: String): String? {
        if (blocked() != null) return null
        val request = GatewayTopicRequest(
            newRequestId(),
            previous.take(MAX_SUMMARY),
            GatewayTopicLatest(Translator.SPEAKER_LABEL, latestSource.take(MAX_TRANSCRIPT), latestTranslation.take(MAX_TRANSCRIPT)),
        )
        return when (val r = withRetry(TOPIC_BUDGET_MS) { client.topic(request) }) {
            is Attempt.Ok -> PyText.strip(r.value.summary).ifEmpty { null }
            is Attempt.Failed -> {
                record(r.error)
                null
            }
        }
    }

    private sealed interface Attempt<out T> {
        data class Ok<T>(val value: T) : Attempt<T>
        data class Failed(val error: GatewayException) : Attempt<Nothing>
    }

    /** One call, resent with the same body while the gateway answers `retry` and the budget lasts. */
    private suspend fun <T> withRetry(budgetMs: Long, call: suspend () -> T): Attempt<T> {
        var last: GatewayException? = null
        val result = withTimeoutOrNull(budgetMs) {
            repeat(MAX_ATTEMPTS) { attempt ->
                try {
                    return@withTimeoutOrNull Attempt.Ok(call())
                } catch (e: GatewayException) {
                    last = e
                    if (e.fallback != GatewayFallback.RETRY || attempt == MAX_ATTEMPTS - 1) return@withTimeoutOrNull Attempt.Failed(e)
                    log.debug("Gateway asked to retry (${e.code}); attempt ${attempt + 2}")
                    sleep(((e.retryAfterSeconds ?: 1).coerceIn(1, MAX_RETRY_WAIT_S)) * 1000)
                }
            }
            Attempt.Failed(last ?: GatewayException(GatewayException.TIMEOUT, "Gateway retries exhausted"))
        }
        return result ?: Attempt.Failed(GatewayException(GatewayException.TIMEOUT, "Gateway budget of $budgetMs ms used", fallback = GatewayFallback.ON_DEVICE))
    }

    private fun blocked(): GatewayException? {
        val s = _status.value
        return if (s.blockedUntilMs > clock()) s.lastError else null
    }

    private fun record(e: GatewayException) {
        log.warn("Gateway unavailable: ${e.code} (${e.httpStatus ?: "-"})")
        val until = if (e.code in STICKY) clock() + blockFor(e) else 0
        _status.value = GatewayStatus(e, until)
    }

    /** Until the allowance resets, else Retry-After, else 10 minutes; at most a day. */
    private fun blockFor(e: GatewayException): Long {
        val now = clock()
        val reset = e.resetsAt?.let {
            try {
                Instant.parse(it).toEpochMilli() - now
            } catch (x: DateTimeParseException) {
                null
            }
        }
        val ms = reset ?: e.retryAfterSeconds?.times(1000) ?: DEFAULT_BLOCK_MS
        return ms.coerceIn(MIN_BLOCK_MS, MAX_BLOCK_MS)
    }

    companion object {
        /** Client budgets above the gateway's deadlines (translate 12 s, topic 8 s). */
        const val BUDGET_MS = 15_000L
        const val TOPIC_BUDGET_MS = 10_000L
        const val MAX_ATTEMPTS = 3
        private const val MAX_RETRY_WAIT_S = 3L
        const val MAX_TRANSCRIPT = 2000
        private const val MAX_SUMMARY = 1000
        private const val MAX_CONTEXT_TEXT = 2000
        private const val CONTRACT_MAX_CONTEXT = 8
        private const val DEFAULT_BLOCK_MS = 10 * 60_000L
        private const val MIN_BLOCK_MS = 60_000L
        private const val MAX_BLOCK_MS = 24 * 3600_000L

        /** Refusals that hold for every request until something changes (account, purchase, month, ops). */
        val STICKY = setOf(
            "NOT_ENTITLED", "SPEND_CEILING_REACHED", "ACCOUNT_REQUIRED", "EMAIL_NOT_VERIFIED", "ACCOUNT_DISABLED",
            "TASK_DISABLED", "APP_DISABLED", "MAINTENANCE", "UPGRADE_REQUIRED", "ACCOUNT_DELETED",
        )
    }
}
