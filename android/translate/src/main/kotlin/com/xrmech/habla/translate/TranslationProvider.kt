package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind

/** One LLM completion. Throws [LlmException]; `CancellationException` propagates untouched. */
interface TranslationProvider {
    suspend fun complete(request: LlmRequest): LlmResponse
}

/**
 * A single completion request.
 *
 * @property instructions system prompt; `""` omits the `instructions` field.
 * @property maxRetries extra attempts [ResilientProvider] may make (0 = one attempt). Ignored by
 *   [OpenAiProvider], which always makes exactly one HTTP attempt.
 */
data class LlmRequest(
    val model: String,
    val instructions: String,
    val input: String,
    val maxOutputTokens: Int,
    val json: Boolean,
    val kind: UsageKind,
    val maxRetries: Int = 0,
)

/**
 * @property text first `output_text` of the first `message` output item, `""` when there is none.
 * @property incomplete true when OpenAI reported `status: "incomplete"`.
 * @property incompleteReason `incomplete_details.reason` (e.g. `max_output_tokens`,
 *   `content_filter`) when [incomplete], else null.
 */
data class LlmResponse(
    val text: String,
    val usage: TokenUsage,
    val incomplete: Boolean,
    val incompleteReason: String? = null,
)

/**
 * A failed completion. [message] is always scrubbed of API keys before construction and is safe
 * to show in the UI.
 */
class LlmException(
    message: String,
    val kind: Kind,
    val httpStatus: Int? = null,
    cause: Throwable? = null,
    val retryAfterMs: Long? = null,
) : Exception(message, cause) {

    enum class Kind { NO_KEY, INVALID_KEY, QUOTA, RATE_LIMITED, MODEL_UNAVAILABLE, BAD_REQUEST, SERVER, TIMEOUT, NETWORK, MALFORMED }

    /** Port of the server's `_is_retryable`: transient failures only. */
    val retryable: Boolean
        get() = kind == Kind.RATE_LIMITED || kind == Kind.SERVER || kind == Kind.TIMEOUT || kind == Kind.NETWORK

    /**
     * False for account-wide or connectivity-wide failures, where every other model would fail
     * identically (D036), so the fallback walk is skipped.
     */
    val allowsFallback: Boolean
        get() = kind != Kind.NO_KEY && kind != Kind.INVALID_KEY && kind != Kind.QUOTA && kind != Kind.NETWORK
}
