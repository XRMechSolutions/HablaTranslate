package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.CorrectionDetail
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.TranslationResult
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

/**
 * llm-gateway v1 wire types for Habla (llm-gateway `.dev/planning/interfaces.md`, confirmed
 * 2026-10-16). camelCase on the wire; [toResult] maps to the snake_case [TranslationResult] the
 * app persists, so nothing downstream changes.
 */
internal object GatewayWire {
    /** Requests: no nulls sent; responses: unknown fields ignored, so the server can add fields. */
    val json = Json {
        encodeDefaults = true
        explicitNulls = false
        ignoreUnknownKeys = true
    }
}

@Serializable
data class GatewayContextLine(val speaker: String, val source: String, val translated: String)

@Serializable
data class GatewayTranslateRequest(
    val requestId: String,
    /** `es_to_en` | `en_to_es`; AUTO is resolved before the call. */
    val direction: String,
    val mode: String,
    val transcript: String,
    val speakerLabel: String,
    val context: List<GatewayContextLine>,
    val topicSummary: String,
)

@Serializable
data class GatewayFlaggedPhrase(
    val phrase: String,
    val literal: String? = null,
    val meaning: String,
    val type: String = "idiom",
    val saveWorthy: Boolean = true,
)

@Serializable
data class GatewayCorrection(val wrong: String, val right: String, val explanation: String)

@Serializable
data class GatewayMeta(val model: String = "", val attempts: Int = 1, val salvaged: Boolean = false, val latencyMs: Long? = null)

@Serializable
data class GatewayTranslateResponse(
    val corrected: String,
    val translated: String,
    val flaggedPhrases: List<GatewayFlaggedPhrase> = emptyList(),
    val confidence: Double = 0.0,
    val speakerHint: String? = null,
    val isCorrection: Boolean = false,
    val correctionDetail: GatewayCorrection? = null,
    val meta: GatewayMeta = GatewayMeta(),
) {
    /** Flagged phrases come from the model, so `source = "llm"` and no spans (spans are pattern-DB only). */
    fun toResult() = TranslationResult(
        corrected = corrected,
        translated = translated,
        flaggedPhrases = flaggedPhrases.map {
            FlaggedPhrase(phrase = it.phrase, literal = it.literal, meaning = it.meaning, type = it.type, source = "llm", saveWorthy = it.saveWorthy)
        },
        confidence = confidence,
        speakerHint = speakerHint,
        isCorrection = isCorrection,
        correctionDetail = correctionDetail?.let { CorrectionDetail(it.wrong, it.right, it.explanation) },
    )
}

@Serializable
data class GatewayTopicLatest(val speaker: String, val source: String, val translated: String)

@Serializable
data class GatewayTopicRequest(val requestId: String, val previousSummary: String, val latest: GatewayTopicLatest)

@Serializable
data class GatewayTopicResponse(val summary: String)

@Serializable
data class GatewayEntitlement(
    val tier: String = "free",
    val active: Boolean = false,
    val source: String? = null,
    val productId: String? = null,
    val accessUntil: String? = null,
    val state: String = "none",
)

@Serializable
data class GatewaySpend(val usedFraction: Double = 0.0, val ceilingReached: Boolean = false, val periodEnd: String? = null)

/** `GET /v1/me`: who the user is to the gateway and how much of the monthly allowance is used. */
@Serializable
data class GatewayMe(val userId: String = "", val entitlement: GatewayEntitlement = GatewayEntitlement(), val spend: GatewaySpend = GatewaySpend())

@Serializable
data class GatewayTasks(val translate: Boolean = false, val topic: Boolean = false, val quick: Boolean = false)

/** `GET /v1/config`: no auth, never version-gated. */
@Serializable
data class GatewayConfig(
    val minVersion: String = "0.0.0",
    val storeUrl: String? = null,
    val maintenance: Boolean = false,
    val aiEnabled: Boolean = false,
    val tasks: GatewayTasks = GatewayTasks(),
)

@Serializable
internal data class GatewayErrorBody(
    val code: String,
    val message: String = "",
    val requestId: String? = null,
    val retryAfterSeconds: Long? = null,
    val fallback: String = "none",
    val details: JsonObject? = null,
)

@Serializable
internal data class GatewayErrorEnvelope(val error: GatewayErrorBody)
