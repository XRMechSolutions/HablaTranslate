package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.Direction

/**
 * Plain machine translation that runs on the phone (D050; ML Kit in `:app`). It is not an LLM:
 * no JSON, no corrections, no idioms and no confidence, so [Translator] wraps its text in a
 * [com.xrmech.habla.translate.model.TranslationResult] itself.
 */
interface OnDeviceTranslator {
    /**
     * Translates [text] in [direction] (AUTO is read as es -> en). Never throws except
     * `CancellationException`.
     */
    suspend fun translate(text: String, direction: Direction): OnDeviceResult

    companion object {
        /** `TranslationOutcome.Ok.model` and the stored session model for on-device results. */
        const val MODEL_ID = "on-device"
    }
}

sealed interface OnDeviceResult {
    data class Ok(val text: String) : OnDeviceResult

    /** The language pack is not on the phone yet. */
    data object NotReady : OnDeviceResult

    data class Failed(val message: String) : OnDeviceResult
}

/**
 * Who translates a final or a preview. `:app` resolves the user's choice to one of these per call.
 * CLOUD is the user's own OpenAI key; GATEWAY is Pro through llm-gateway (D051, finals and topic only).
 */
enum class TranslateEngine { CLOUD, ON_DEVICE, GATEWAY }
