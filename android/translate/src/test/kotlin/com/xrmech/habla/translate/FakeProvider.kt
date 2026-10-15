package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.TokenUsage

/**
 * Scripted [TranslationProvider]: each call takes the next step, which returns a response,
 * throws an [LlmException], or runs arbitrary suspend code. Every request is recorded.
 */
class FakeProvider(vararg steps: Step) : TranslationProvider {
    sealed interface Step
    data class Respond(val text: String, val incomplete: Boolean = false, val input: Int = 10, val output: Int = 5) : Step
    data class Fail(val kind: LlmException.Kind, val status: Int? = null) : Step
    class Run(val block: suspend (LlmRequest) -> LlmResponse) : Step

    private val queue = ArrayDeque(steps.toList())
    val requests = mutableListOf<LlmRequest>()

    fun enqueue(vararg steps: Step) { queue.addAll(steps) }

    override suspend fun complete(request: LlmRequest): LlmResponse {
        synchronized(requests) { requests += request }
        val step = synchronized(queue) { queue.removeFirstOrNull() } ?: error("no scripted step for request ${requests.size}: $request")
        return when (step) {
            is Respond -> LlmResponse(step.text, TokenUsage(step.input, 0, step.output, 0, request.model), step.incomplete)
            is Fail -> throw LlmException("OpenAI ${step.status ?: 0} scripted: ${step.kind}", step.kind, step.status)
            is Run -> step.block(request)
        }
    }

    companion object {
        fun json(corrected: String, translated: String, confidence: Double = 0.9): String =
            """{"corrected": "$corrected", "translated": "$translated", "confidence": $confidence}"""
    }
}
