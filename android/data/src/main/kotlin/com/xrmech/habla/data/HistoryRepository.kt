package com.xrmech.habla.data

import com.xrmech.habla.translate.model.CorrectionDetail
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import java.time.Instant
import kotlinx.coroutines.flow.Flow

/** Sessions and their exchanges (port of the server's history tables). */
interface HistoryRepository {
    /** Closes sessions left open by a killed process, inserts a new one, returns its id. */
    suspend fun startSession(mode: Mode, direction: Direction, asrModel: String): Long

    /** Sets `ended_at`, the trimmed topic summary (blank -> null) and `translate_model`; no-op if already ended or unknown. */
    suspend fun endSession(id: Long, topicSummary: String?)

    /** @throws IllegalArgumentException for AUTO direction or segmentId < 1; [DataException] for an unknown session. */
    suspend fun addExchange(e: NewExchange): Long

    /** Stores the trimmed text, or NULL when blank. @throws NoSuchElementException for an unknown exchange. */
    suspend fun setCorrectedTranscript(exchangeId: Long, text: String)

    /** Newest first by `started_at`, then id. */
    fun sessions(): Flow<List<SessionRow>>

    /** Ordered by `segment_id`, id; an unknown session gives an empty list. */
    fun exchanges(sessionId: Long): Flow<List<ExchangeRow>>

    /** Deletes the rows (cascades), then `files/sessions/<id>/`. Unknown id is a no-op. */
    suspend fun deleteSession(id: Long)

    /** One session, or null. */
    suspend fun session(id: Long): SessionRow?
}

data class NewExchange(
    val sessionId: Long,
    /** >= 1, the pipeline segment number. */
    val segmentId: Int,
    /** ES_TO_EN or EN_TO_ES; AUTO is rejected. */
    val direction: Direction,
    val rawTranscript: String,
    val correctedSource: String?,
    /** On a failed translation: the raw transcript. */
    val translation: String,
    val confidence: Double?,
    val isCorrection: Boolean,
    val correction: CorrectionDetail?,
    val flagged: List<FlaggedPhrase>,
    val processingMs: Long?,
    /** "es" | "en" | null */
    val detectedLanguage: String?,
    val translationError: Boolean,
    /** The value returned by `TrainingAudioStore.saveSegment`, else null. */
    val audioPath: String?,
)

data class SessionRow(
    val id: Long,
    val startedAt: Instant,
    val endedAt: Instant?,
    val mode: Mode,
    val direction: Direction,
    val topicSummary: String?,
    val notes: String?,
    val asrModel: String,
    val translateModel: String?,
    val exchangeCount: Int,
    val costUsd: Double,
)

data class ExchangeRow(
    val id: Long,
    val sessionId: Long,
    val segmentId: Int,
    val timestamp: Instant,
    val direction: Direction,
    val rawTranscript: String,
    val correctedSource: String?,
    val translation: String,
    val confidence: Double?,
    val isCorrection: Boolean,
    val correction: CorrectionDetail?,
    val flagged: List<FlaggedPhrase>,
    val processingMs: Long?,
    val detectedLanguage: String?,
    val translationError: Boolean,
    val audioPath: String?,
    val correctedTranscript: String?,
)
