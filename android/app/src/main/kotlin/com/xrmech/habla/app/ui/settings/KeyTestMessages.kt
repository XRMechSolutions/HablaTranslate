package com.xrmech.habla.app.ui.settings

import com.xrmech.habla.translate.KeyTestResult
import com.xrmech.habla.translate.LlmException

/** Text under the Test key button (interfaces.md @ Test key result text). */
object KeyTestMessages {
    /** [testModel] is the model the test request used (`models.json` quick default). */
    fun text(result: KeyTestResult, testModel: String): String = when (result) {
        is KeyTestResult.Valid -> "Key works (${result.usage.model} replied)"
        is KeyTestResult.Failed -> failed(result.error, testModel)
    }

    private fun failed(e: LlmException, testModel: String): String = when (e.kind) {
        LlmException.Kind.NO_KEY -> "Enter a key first"
        LlmException.Kind.INVALID_KEY -> "Key rejected by OpenAI (${e.httpStatus ?: "invalid"})"
        LlmException.Kind.QUOTA -> "No credit on this OpenAI account (${e.httpStatus ?: 429}) - add billing at platform.openai.com"
        LlmException.Kind.MODEL_UNAVAILABLE -> "Model $testModel not enabled for this key (404)"
        LlmException.Kind.RATE_LIMITED -> "Rate-limited - the key works, try again shortly"
        LlmException.Kind.NETWORK, LlmException.Kind.TIMEOUT -> "Network error: ${e.message}"
        else -> "${e.kind}: ${e.message}"
    }
}
