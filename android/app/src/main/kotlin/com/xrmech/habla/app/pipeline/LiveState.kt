package com.xrmech.habla.app.pipeline

import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.model.CostTotals
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.TranslationResult

/** Everything the Live screen and the notification render for one session (interfaces.md @ LiveState). */
data class LiveState(
    val phase: SessionPhase = SessionPhase.Idle,
    val sessionId: Long? = null,
    /** clock() at start, for the timer. */
    val startedAtMs: Long? = null,
    val partial: PartialLine? = null,
    /** Oldest first, at most [MAX_CARDS] (web client MAX_EXCHANGES); older ones stay in History. */
    val exchanges: List<ExchangeCard> = emptyList(),
    /** Waiting + running finals. */
    val queueDepth: Int = 0,
    val noise: NoiseLevel? = null,
    val noiseFloorRms: Double? = null,
    val votedLanguage: String? = null,
    val audio: AudioCondition = AudioCondition.Normal,
    /** Session-scoped warnings (see BannerKind). */
    val notices: Set<SessionNotice> = emptySet(),
    /** Last "Transcription failed" message; cleared by the next accepted final. */
    val lastAsrError: String? = null,
    /** Cleared by the next Ok translation. */
    val lastTranslationError: LlmException? = null,
    /** Last HISTORY_WRITE_FAILED message. */
    val lastHistoryError: String? = null,
    val costs: CostTotals = CostTotals(),
    val metrics: MetricsSnapshot = MetricsSnapshot(),
) {
    companion object {
        const val MAX_CARDS = 200
    }
}

sealed interface SessionPhase {
    data object Idle : SessionPhase
    data object Starting : SessionPhase
    data object Calibrating : SessionPhase
    data object Listening : SessionPhase
    data class Finishing(val pending: Int) : SessionPhase
    data class Failed(val error: SessionError) : SessionPhase

    val active: Boolean get() = this !is Idle && this !is Failed
}

/** Calibration noise floor: < 200 quiet, < 800 moderate, else loud (server `_on_calibration_complete`). */
enum class NoiseLevel {
    QUIET, MODERATE, LOUD;

    companion object {
        fun of(rms: Double): NoiseLevel = when {
            rms < 200 -> QUIET
            rms < 800 -> MODERATE
            else -> LOUD
        }
    }
}

/** The in-progress line; [direction] is the effective direction (never AUTO). */
data class PartialLine(
    val segmentId: Int,
    val text: String,
    val translation: String?,
    val language: String,
    val direction: Direction,
    val finalizing: Boolean,
    /** [translation] came from the phone's engine, so the Google attribution shows beside it (D050). */
    val translatedOnDevice: Boolean = false,
)

/**
 * One final; [direction] is the effective direction (never AUTO). [rawSegmentIds] are the
 * `SpeechSegment.id`s merged into it (empty for cards rebuilt from history).
 */
data class ExchangeCard(
    val seq: Int,
    val exchangeId: Long?,
    val createdAtMs: Long,
    val rawTranscript: String,
    val detectedLanguage: String?,
    val direction: Direction,
    val translation: CardTranslation,
    val rawSegmentIds: List<Int> = emptyList(),
)

sealed interface CardTranslation {
    data object Pending : CardTranslation

    data class Done(
        val result: TranslationResult,
        val flagged: List<FlaggedPhrase>,
        val model: String,
        val processingMs: Long,
    ) : CardTranslation

    data class Failed(val kind: LlmException.Kind, val message: String) : CardTranslation
}

enum class SessionNotice {
    VAD_FALLBACK, AUDIO_DROPPED, FALLING_BEHIND, THERMAL_PAUSE, HISTORY_WRITE_FAILED, LOUD_ENVIRONMENT,
}
