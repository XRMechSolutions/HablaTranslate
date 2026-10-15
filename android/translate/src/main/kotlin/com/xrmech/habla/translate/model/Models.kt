package com.xrmech.habla.translate.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Translation direction. [serialName] is the server's direction string; [sourceCode] and
 * [targetCode] are the language codes. AUTO carries `es`/`en`, which is what the server's
 * `parse_direction("auto")` falls back to.
 */
@Serializable
enum class Direction(val serialName: String, val sourceCode: String, val targetCode: String) {
    @SerialName("es_to_en") ES_TO_EN("es_to_en", "es", "en"),
    @SerialName("en_to_es") EN_TO_ES("en_to_es", "en", "es"),
    @SerialName("auto") AUTO("auto", "es", "en");

    companion object {
        /** The direction whose [serialName] is [s], or null for an unknown name. */
        fun fromSerialName(s: String): Direction? = entries.firstOrNull { it.serialName == s }
    }
}

@Serializable
enum class Mode {
    @SerialName("conversation") CONVERSATION,
    @SerialName("classroom") CLASSROOM,
}

@Serializable
enum class UsageKind {
    @SerialName("final") FINAL,
    @SerialName("quick") QUICK,
    @SerialName("topic") TOPIC,
    @SerialName("test") TEST,
}

/**
 * Port of the server's pydantic `TranslationResult`. Encoded with
 * `Json { encodeDefaults = true; explicitNulls = true }` it produces the same key set and names
 * as the server's `model_dump()` (stored by `:data` as `flagged_json`).
 */
@Serializable
data class TranslationResult(
    val corrected: String,
    val translated: String,
    @SerialName("flagged_phrases") val flaggedPhrases: List<FlaggedPhrase> = emptyList(),
    val confidence: Double = 0.0,
    @SerialName("speaker_hint") val speakerHint: String? = null,
    @SerialName("is_correction") val isCorrection: Boolean = false,
    @SerialName("correction_detail") val correctionDetail: CorrectionDetail? = null,
)

@Serializable
data class FlaggedPhrase(
    val phrase: String,
    val literal: String? = null,
    val meaning: String,
    /** idiom | slang | false_friend | correction | grammar_note */
    val type: String = "idiom",
    val region: String? = "universal",
    /** "pattern_db" | "llm" */
    val source: String = "llm",
    @SerialName("save_worthy") val saveWorthy: Boolean = true,
    /** Code-point offset into the transcript (server parity), not a UTF-16 index. */
    @SerialName("span_start") val spanStart: Int? = null,
    @SerialName("span_end") val spanEnd: Int? = null,
)

@Serializable
data class CorrectionDetail(val wrong: String, val right: String, val explanation: String)

@Serializable
data class TokenUsage(
    @SerialName("input_tokens") val inputTokens: Int,
    @SerialName("cached_input_tokens") val cachedInputTokens: Int,
    /** Includes reasoning tokens. */
    @SerialName("output_tokens") val outputTokens: Int,
    @SerialName("reasoning_tokens") val reasoningTokens: Int,
    /** The request model id (the price-table key), not the snapshot id OpenAI echoes back. */
    val model: String,
)

@Serializable
data class CostTotals(
    @SerialName("session_input_tokens") val sessionInputTokens: Long = 0,
    @SerialName("session_output_tokens") val sessionOutputTokens: Long = 0,
    @SerialName("session_cost_usd") val sessionCostUsd: Double = 0.0,
    @SerialName("total_cost_usd") val totalCostUsd: Double = 0.0,
)
