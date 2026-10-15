package com.xrmech.habla.translate

import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.TranslationResult
import com.xrmech.habla.translate.model.UsageKind

/**
 * Final translations (port of the server's `Translator.translate`, `_retry_passthrough` and
 * `update_topic_summary`). Holds no mutable state, so concurrent finals are safe.
 */
class Translator(
    private val provider: TranslationProvider,
    assets: SharedAssets,
    private val settings: () -> TranslateSettings,
    private val costs: CostMeter,
) {
    private val log = Log()
    private val models = assets.models
    private val prompts = PromptBuilder(assets)
    private val fallback = ModelFallback(models)

    /**
     * Never throws except `CancellationException`: every [LlmException] becomes
     * [TranslationOutcome.Failed] with the transcript as `raw`.
     */
    suspend fun translate(
        transcript: String,
        direction: Direction,
        mode: Mode,
        context: List<ContextExchange>,
        topicSummary: String,
    ): TranslationOutcome {
        val model = fallback.resolveModel(settings().finalModel, models.finalDefault)
        val system: String
        val user: String
        try {
            system = prompts.system(direction, mode)
            user = prompts.user(transcript, context, topicSummary)
        } catch (e: IllegalArgumentException) {
            log.error("Prompt template failed: ${e.message}")
            return TranslationOutcome.Failed(transcript, LlmException("Prompt template failed: ${e.message}", Kind.BAD_REQUEST, cause = e))
        }
        log.debug("Translating '${Log.snippet(transcript)}' ${direction.serialName} with $model")

        return try {
            val primary = LlmRequest(model, system, user, FINAL_MAX_TOKENS, json = true, kind = UsageKind.FINAL, maxRetries = ResilientProvider.MAX_RETRIES_CLOUD)
            var (response, used) = fallback.call(provider, primary, ::recordFinal)
            if (response.incomplete) {
                // Usually reasoning tokens used up the budget; one larger re-ask on the same model.
                log.warn("Final from $used was incomplete; re-issuing at $INCOMPLETE_MAX_TOKENS tokens")
                response = provider.complete(primary.copy(model = used, maxOutputTokens = INCOMPLETE_MAX_TOKENS, maxRetries = 0))
                recordFinal(response)
            }
            val result = when (val parsed = ResponseParser.parse(response.text.ifEmpty { "{}" }, transcript)) {
                is ParseResult.Invalid -> {
                    log.warn("Model output failed validation: ${parsed.reason}")
                    return TranslationOutcome.Failed(transcript, LlmException("Model output failed validation: ${parsed.reason}", Kind.MALFORMED))
                }
                is ParseResult.Ok -> parsed.result
            }
            if (!Passthrough.isPassthrough(transcript, result.translated, direction.serialName)) {
                return TranslationOutcome.Ok(result, used)
            }
            log.warn("Model passed the source through untranslated ($used); retrying")
            log.debug("Passthrough: '${Log.snippet(transcript)}' -> '${Log.snippet(result.translated)}'")
            val retry = retryPassthrough(transcript, direction, system, model)
            if (retry != null) {
                TranslationOutcome.Ok(retry.first.copy(confidence = minOf(retry.first.confidence, 0.5)), retry.second)
            } else {
                TranslationOutcome.Ok(result.copy(confidence = minOf(result.confidence, 0.2)), used)
            }
        } catch (e: LlmException) {
            log.warn("Translation failed: ${e.message}")
            TranslationOutcome.Failed(transcript, e)
        }
    }

    /** The retry result and its model, or null when the retry failed or was itself a passthrough. */
    private suspend fun retryPassthrough(
        transcript: String,
        direction: Direction,
        system: String,
        model: String,
    ): Pair<TranslationResult, String>? {
        return try {
            val prompt = prompts.passthroughRetry(transcript, direction)
            val request = LlmRequest(model, system, prompt, RETRY_MAX_TOKENS, json = true, kind = UsageKind.FINAL, maxRetries = 0)
            val (response, used) = fallback.call(provider, request, ::recordFinal)
            val parsed = ResponseParser.parse(response.text.ifEmpty { "{}" }, transcript)
            if (parsed !is ParseResult.Ok) return null
            if (Passthrough.isPassthrough(transcript, parsed.result.translated, direction.serialName)) {
                log.warn("Passthrough retry also returned the source")
                return null
            }
            parsed.result to used
        } catch (e: LlmException) {
            log.warn("Passthrough retry failed: ${e.message}")
            null
        } catch (e: IllegalArgumentException) {
            log.error("Passthrough retry prompt failed: ${e.message}")
            null
        }
    }

    /**
     * Updates the rolling topic summary with one cheap call (with model fallback). Returns the
     * trimmed text, or [previous] when it is blank or the call fails. Throws only
     * `CancellationException`.
     */
    suspend fun updateTopicSummary(previous: String, latestSource: String, latestTranslation: String): String {
        val model = fallback.resolveModel(settings().finalModel, models.finalDefault)
        return try {
            val prompt = prompts.topic(previous, latestSource, latestTranslation)
            val request = LlmRequest(model, "", prompt, TOPIC_MAX_TOKENS, json = false, kind = UsageKind.TOPIC, maxRetries = 0)
            val (response, _) = fallback.call(provider, request) { costs.record(it.usage, UsageKind.TOPIC) }
            PyText.strip(response.text).ifEmpty { previous }
        } catch (e: LlmException) {
            log.debug("Topic summary update skipped: ${e.message}")
            previous
        } catch (e: IllegalArgumentException) {
            log.error("Topic prompt failed: ${e.message}")
            previous
        }
    }

    private fun recordFinal(response: LlmResponse) = costs.record(response.usage, UsageKind.FINAL)

    companion object {
        /** Speaker label for every prompt (the server's single-speaker label). */
        const val SPEAKER_LABEL = "Speaker A"

        private const val FINAL_MAX_TOKENS = 1024
        private const val INCOMPLETE_MAX_TOKENS = 2048
        private const val RETRY_MAX_TOKENS = 512
        private const val TOPIC_MAX_TOKENS = 200
    }
}
