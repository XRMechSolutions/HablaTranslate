package com.xrmech.habla.translate

import com.xrmech.habla.shared.ModelTable
import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import io.ktor.client.HttpClient
import io.ktor.client.network.sockets.ConnectTimeoutException
import io.ktor.client.plugins.HttpRequestTimeoutException
import io.ktor.client.plugins.timeout
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import io.ktor.http.content.TextContent
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import java.io.IOException
import java.net.SocketTimeoutException
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import kotlin.coroutines.cancellation.CancellationException

/** Result of [OpenAiProvider.testKey]. Callers record [Valid.usage] with [UsageKind.TEST]. */
sealed interface KeyTestResult {
    data class Valid(val usage: TokenUsage) : KeyTestResult
    data class Failed(val error: LlmException) : KeyTestResult
}

/**
 * One HTTP attempt against the OpenAI Responses API: request body (exact port of the server's
 * `_openai_request_body`), response reader and error mapper. Retries, rate gating and degraded
 * mode live in [ResilientProvider].
 *
 * The API key comes from [keyProvider] once per attempt, goes only into that request's
 * Authorization header, and is never stored, logged or put in a message.
 */
class OpenAiProvider(
    private val http: HttpClient,
    private val keyProvider: suspend () -> String?,
    private val models: ModelTable,
) : TranslationProvider {

    private val log = Log()

    override suspend fun complete(request: LlmRequest): LlmResponse = attempt(request, keyProvider())

    /**
     * Checks a key with one tiny request (no retry, fallback or rate gate). Uses [candidate] when
     * given (testing before saving), else the stored key. Never throws [LlmException].
     */
    suspend fun testKey(candidate: String? = null): KeyTestResult {
        val key = candidate ?: keyProvider()
        val request = LlmRequest(
            model = models.quickDefault, instructions = "", input = TEST_INPUT,
            maxOutputTokens = TEST_MAX_OUTPUT_TOKENS, json = false, kind = UsageKind.TEST,
        )
        return try {
            KeyTestResult.Valid(attempt(request, key).usage)
        } catch (e: LlmException) {
            KeyTestResult.Failed(e)
        }
    }

    private suspend fun attempt(request: LlmRequest, key: String?): LlmResponse {
        if (key.isNullOrBlank()) throw LlmException("No OpenAI API key stored", Kind.NO_KEY)
        if (key.any { it.code !in 0x21..0x7E }) throw LlmException("API key contains invalid characters", Kind.INVALID_KEY)

        val body = requestBody(request, models).toString()
        log.debug("OpenAI request model=${request.model} kind=${request.kind} max_output_tokens=${request.maxOutputTokens}")
        val response: HttpResponse
        val text: String
        try {
            response = http.post("${OpenAiHttp.BASE_URL}/responses") {
                header(HttpHeaders.Authorization, "Bearer $key")
                setBody(TextContent(body, ContentType.Application.Json))
                timeout { requestTimeoutMillis = timeoutFor(request.kind) }
            }
            text = response.bodyAsText()
        } catch (e: CancellationException) {
            throw e
        } catch (e: HttpRequestTimeoutException) {
            throw LlmException("OpenAI request timed out", Kind.TIMEOUT, cause = e)
        } catch (e: ConnectTimeoutException) {
            throw LlmException("OpenAI connect timed out", Kind.TIMEOUT, cause = e)
        } catch (e: SocketTimeoutException) {
            // Ktor's socket timeout extends java.net.SocketTimeoutException on the JVM.
            throw LlmException("OpenAI socket timed out", Kind.TIMEOUT, cause = e)
        } catch (e: IOException) {
            throw LlmException(Redact.secrets("Network error: ${e.javaClass.simpleName}: ${e.message ?: ""}"), Kind.NETWORK, cause = e)
        }

        val status = response.status.value
        if (status !in 200..299) throw mapError(status, response.status.description, text, response.headers)
        return readSuccess(request, status, text)
    }

    private fun readSuccess(request: LlmRequest, status: Int, text: String): LlmResponse {
        val root = parseObject(text)
            ?: throw LlmException("OpenAI $status: response body is not a JSON object", Kind.MALFORMED, status).also {
                log.debug("OpenAI non-JSON 2xx body: ${text.take(ERROR_BODY_LOG_CHARS)}")
            }
        val incomplete = root.str("status") == "incomplete"
        val incompleteReason = if (incomplete) (root["incomplete_details"] as? JsonObject)?.str("reason") ?: "unknown" else null
        when (root.str("status")) {
            "failed" -> {
                val err = root["error"] as? JsonObject
                val code = err?.str("code") ?: err?.str("type") ?: "failed"
                throw LlmException(message(status, code, err?.str("message") ?: "response status failed"), Kind.SERVER, status)
            }
            "incomplete" -> log.warn("OpenAI response incomplete (reason=$incompleteReason) model=${request.model}")
        }
        val usage = root["usage"] as? JsonObject
        val inputDetails = usage?.get("input_tokens_details") as? JsonObject
        val outputDetails = usage?.get("output_tokens_details") as? JsonObject
        return LlmResponse(
            text = outputText(root),
            usage = TokenUsage(
                inputTokens = usage.int("input_tokens"),
                cachedInputTokens = inputDetails.int("cached_tokens"),
                outputTokens = usage.int("output_tokens"),
                reasoningTokens = outputDetails.int("reasoning_tokens"),
                model = request.model,
            ),
            incomplete = incomplete,
            incompleteReason = incompleteReason,
        )
    }

    /** First `output_text` of the first `message` item; `""` when there is none. */
    private fun outputText(root: JsonObject): String {
        val output = root["output"] as? JsonArray ?: return ""
        val message = output.firstOrNull { (it as? JsonObject)?.str("type") == "message" } as? JsonObject ?: return ""
        val content = message["content"] as? JsonArray ?: return ""
        val item = content.firstOrNull { (it as? JsonObject)?.str("type") == "output_text" } as? JsonObject ?: return ""
        return item.str("text") ?: ""
    }

    private fun mapError(status: Int, description: String, text: String, headers: Headers): LlmException {
        log.debug("OpenAI error body ($status): ${text.take(ERROR_BODY_LOG_CHARS)}")
        val err = parseObject(text)?.get("error") as? JsonObject
        val code = err?.str("code")?.takeIf { it.isNotEmpty() } ?: err?.str("type") ?: ""
        val msg = message(status, code, err?.str("message") ?: description)
        return when {
            status == 401 || status == 403 -> LlmException(msg, Kind.INVALID_KEY, status)
            status == 429 && code in QUOTA_CODES -> LlmException(msg, Kind.QUOTA, status)
            status == 429 -> LlmException(msg, Kind.RATE_LIMITED, status, retryAfterMs = RetryAfter.parse(headers))
            status == 404 -> LlmException(msg, Kind.MODEL_UNAVAILABLE, status)
            status in 400..499 -> {
                log.error("OpenAI rejected the request: $msg (param=${err?.str("param") ?: "-"})")
                LlmException(msg, Kind.BAD_REQUEST, status)
            }
            // The server honours Retry-After on every retryable HTTP error (`_retry_after_seconds`),
            // so any 5xx carries it, not only 503.
            status >= 500 -> LlmException(msg, Kind.SERVER, status, retryAfterMs = RetryAfter.parse(headers))
            // Anything else unexpected (1xx/3xx) is treated as a transient server fault.
            else -> LlmException(msg, Kind.SERVER, status)
        }
    }

    private fun message(status: Int, code: String, detail: String): String =
        Redact.secrets(if (code.isEmpty()) "OpenAI $status: $detail" else "OpenAI $status $code: $detail")

    private fun parseObject(text: String): JsonObject? = try {
        Json.parseToJsonElement(text) as? JsonObject
    } catch (_: SerializationException) {
        null
    } catch (_: IllegalArgumentException) {
        null
    }

    private fun JsonObject.str(key: String): String? = (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.contentOrNull

    private fun JsonObject?.int(key: String): Int = (this?.get(key) as? JsonPrimitive)?.intOrNull ?: 0

    companion object {
        /** 429 error codes/types that mean "out of credit", not "slow down" (server `_QUOTA_ERROR_CODES`). */
        val QUOTA_CODES: Set<String> = setOf(
            "insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted",
            "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
            "organization_usage_limit_exceeded",
        )

        private const val TEST_INPUT = "Reply with OK."
        private const val TEST_MAX_OUTPUT_TOKENS = 16
        private const val ERROR_BODY_LOG_CHARS = 200

        internal fun timeoutFor(kind: UsageKind): Long = when (kind) {
            UsageKind.FINAL, UsageKind.TOPIC -> 30_000L
            UsageKind.QUICK -> 10_000L
            UsageKind.TEST -> 15_000L
        }

        /**
         * Exact port of the server's `_openai_request_body`: optional fields only for models in the
         * table, and only parameters the model accepts. Public for the golden tests.
         */
        fun requestBody(request: LlmRequest, models: ModelTable): JsonObject {
            val info = models.models[request.model]
            return buildJsonObject {
                put("model", request.model)
                put("input", request.input)
                put("max_output_tokens", request.maxOutputTokens)
                put("store", false)
                if (request.instructions.isNotEmpty()) put("instructions", request.instructions)
                val effort = info?.effort
                if (!effort.isNullOrEmpty() && effort in info.efforts) {
                    putJsonObject("reasoning") { put("effort", effort) }
                }
                val verbosity = info?.verbosity == true
                if (verbosity || request.json) {
                    putJsonObject("text") {
                        if (verbosity) put("verbosity", "low")
                        if (request.json) putJsonObject("format") { put("type", "json_object") }
                    }
                }
            }
        }
    }
}

/** `Retry-After` handling: `retry-after-ms` wins, then decimal seconds, then an HTTP-date. */
internal object RetryAfter {
    fun parse(headers: Headers, nowMs: Long = System.currentTimeMillis()): Long? {
        headers["retry-after-ms"]?.let { raw ->
            return raw.trim().toDoubleOrNull()?.takeIf { it >= 0 && !it.isNaN() }?.toLong()
        }
        val raw = headers["retry-after"]?.trim() ?: return null
        raw.toDoubleOrNull()?.let { secs ->
            return if (secs >= 0 && !secs.isNaN()) (secs * 1000).toLong() else null
        }
        return try {
            val at = ZonedDateTime.parse(raw, DateTimeFormatter.RFC_1123_DATE_TIME).toInstant().toEpochMilli()
            (at - nowMs).takeIf { it >= 0 }
        } catch (_: Exception) {
            null
        }
    }
}

