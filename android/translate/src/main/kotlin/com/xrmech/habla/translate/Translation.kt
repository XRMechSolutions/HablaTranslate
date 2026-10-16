package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.TranslationResult

/** Result of [Translator.translate]. */
sealed interface TranslationOutcome {
    /** [model] is the id that produced [result] (primary, fallback or passthrough-retry model). */
    data class Ok(val result: TranslationResult, val model: String) : TranslationOutcome

    /** [raw] is the original transcript, shown flagged in the UI. */
    data class Failed(val raw: String, val error: LlmException) : TranslationOutcome
}

/** [finalEngine] / [previewEngine]: who translates finals and partials (D050); CLOUD keeps the LLM path. */
data class TranslateSettings(
    val finalModel: String,
    val quickModel: String,
    val partialTranslation: Boolean,
    val finalEngine: TranslateEngine = TranslateEngine.CLOUD,
    val previewEngine: TranslateEngine = TranslateEngine.CLOUD,
)

/** One prior exchange for the prompt context. [source] is `TranslationResult.corrected`. */
data class ContextExchange(val source: String, val translated: String)
