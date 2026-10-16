package com.xrmech.habla.app.ui.live

import com.xrmech.habla.translate.LlmException

/** The one-line text and action of a failed card (interfaces.md @ BannerKind, failed card line). */
object CardTexts {
    fun failedLine(kind: LlmException.Kind, message: String): String = when (kind) {
        LlmException.Kind.NO_KEY -> "Not translated - no OpenAI key"
        LlmException.Kind.INVALID_KEY -> "Not translated - key rejected"
        LlmException.Kind.QUOTA -> "Not translated - no credit"
        LlmException.Kind.NETWORK, LlmException.Kind.TIMEOUT -> "Not translated - network: $message"
        LlmException.Kind.MODEL_UNAVAILABLE -> "Not translated - $message"
        else -> "Not translated - $kind: $message"
    }

    fun failedAction(kind: LlmException.Kind): BannerAction? = when (kind) {
        LlmException.Kind.NO_KEY -> BannerAction.ADD_KEY
        LlmException.Kind.INVALID_KEY, LlmException.Kind.QUOTA, LlmException.Kind.MODEL_UNAVAILABLE -> BannerAction.OPEN_SETTINGS
        else -> null
    }
}
