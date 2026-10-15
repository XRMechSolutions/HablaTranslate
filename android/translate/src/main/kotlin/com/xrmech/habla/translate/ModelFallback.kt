package com.xrmech.habla.translate

import com.xrmech.habla.shared.ModelTable
import java.util.concurrent.ConcurrentHashMap

/**
 * Model selection and the OpenAI model-fallback walk (port of the cloud part of the server's
 * `_call_llm`; local-provider fallback is dropped, D019).
 */
internal class ModelFallback(private val models: ModelTable) {
    private val log = Log()
    private val warnedUnknown = ConcurrentHashMap.newKeySet<String>()
    private val warnedDeprecated = ConcurrentHashMap.newKeySet<String>()

    /** [requested] when it is in the table, else [default] (WARN once per unknown id). */
    fun resolveModel(requested: String, default: String): String {
        val id = if (requested in models.models) {
            requested
        } else {
            if (warnedUnknown.add(requested)) log.warn("Model '$requested' is not in models.json; using $default")
            default
        }
        val deprecated = models.models[id]?.deprecated
        if (deprecated != null && warnedDeprecated.add(id)) log.warn("Model $id is deprecated: $deprecated")
        return id
    }

    /**
     * Calls [request]; when it fails with a kind that [LlmException.allowsFallback], tries each
     * `fallbackOrder` id after the primary's position (the whole list when the primary is not in
     * it), skipping ids missing from the table, with `maxRetries = 0`. First success wins; when
     * all fail the primary error is thrown. [onResponse] sees every successful response (usage).
     */
    suspend fun call(
        provider: TranslationProvider,
        request: LlmRequest,
        onResponse: (LlmResponse) -> Unit,
    ): Pair<LlmResponse, String> {
        val primaryError = try {
            return provider.complete(request).also(onResponse) to request.model
        } catch (e: LlmException) {
            e
        }
        if (!primaryError.allowsFallback) throw primaryError

        val order = models.fallbackOrder
        val start = order.indexOf(request.model) + 1
        for (model in order.drop(start)) {
            if (model !in models.models) continue
            log.warn("OpenAI ${request.model} failed (${primaryError.kind}); trying $model")
            try {
                val response = provider.complete(request.copy(model = model, maxRetries = 0))
                onResponse(response)
                log.warn("OpenAI fallback to $model succeeded")
                return response to model
            } catch (e: LlmException) {
                log.warn("OpenAI $model also failed (${e.kind})")
            }
        }
        throw primaryError
    }
}
