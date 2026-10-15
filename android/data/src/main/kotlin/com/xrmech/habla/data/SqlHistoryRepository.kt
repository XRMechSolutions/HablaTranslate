package com.xrmech.habla.data

import app.cash.sqldelight.coroutines.asFlow
import app.cash.sqldelight.coroutines.mapToList
import com.xrmech.habla.data.db.HablaDatabase
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import java.io.File
import java.time.Clock
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext

/**
 * [HistoryRepository] over `habla.db`. Every suspend function runs on [dispatcher]; transactions
 * hold only synchronous query calls, and session files are deleted after the commit.
 */
class SqlHistoryRepository(
    db: HablaDatabase,
    private val filesDir: File,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val clock: Clock = Clock.systemUTC(),
) : HistoryRepository {

    private val sessionQueries = db.sessionQueries
    private val exchangeQueries = db.exchangeQueries

    override suspend fun startSession(mode: Mode, direction: Direction, asrModel: String): Long =
        dbCall(dispatcher, "start session") {
            sessionQueries.transactionWithResult {
                // A process killed mid-session never called endSession; close it with its real end time.
                sessionQueries.closeDanglingSessions()
                sessionQueries.insertSession(Timestamps.now(clock), EnumCodec.encode(mode), EnumCodec.encode(direction), asrModel)
                sessionQueries.lastInsertRowId().executeAsOne()
            }
        }

    override suspend fun endSession(id: Long, topicSummary: String?) {
        dbCall(dispatcher, "end session") {
            sessionQueries.endSession(Timestamps.now(clock), topicSummary?.trim()?.ifEmpty { null }, id)
        }
    }

    override suspend fun addExchange(e: NewExchange): Long = dbCall(dispatcher, "add exchange") {
        require(e.direction != Direction.AUTO) { "exchange direction must be resolved, not AUTO" }
        require(e.segmentId >= 1) { "segmentId must be >= 1" }
        val flagged = RowCodec.encodeFlagged(e.flagged)
        val correction = RowCodec.encodeCorrection(e.correction)
        exchangeQueries.transactionWithResult {
            exchangeQueries.insertExchange(
                session_id = e.sessionId,
                segment_id = e.segmentId.toLong(),
                timestamp = Timestamps.now(clock),
                direction = EnumCodec.encode(e.direction),
                raw_transcript = e.rawTranscript,
                corrected_source = e.correctedSource,
                translation = e.translation,
                confidence = e.confidence,
                is_correction = RowCodec.bit(e.isCorrection),
                correction_json = correction,
                flagged_json = flagged,
                processing_ms = e.processingMs,
                detected_language = e.detectedLanguage,
                translation_error = RowCodec.bit(e.translationError),
                audio_path = e.audioPath,
            )
            sessionQueries.lastInsertRowId().executeAsOne()
        }
    }

    override suspend fun setCorrectedTranscript(exchangeId: Long, text: String) {
        val value = text.trim().ifEmpty { null }
        dbCall(dispatcher, "set corrected transcript") {
            exchangeQueries.transaction {
                if (exchangeQueries.exchangeExists(exchangeId).executeAsOne() == 0L) {
                    throw NoSuchElementException("exchange $exchangeId not found")
                }
                exchangeQueries.setCorrectedTranscript(value, exchangeId)
            }
        }
    }

    override fun sessions(): Flow<List<SessionRow>> =
        sessionQueries.selectSessions(::sessionRow).asFlow().mapToList(dispatcher).asDataFlow("read sessions")

    override fun exchanges(sessionId: Long): Flow<List<ExchangeRow>> =
        exchangeQueries.selectExchangesForSession(sessionId, ::exchangeRow).asFlow().mapToList(dispatcher)
            .asDataFlow("read exchanges")

    override suspend fun session(id: Long): SessionRow? = dbCall(dispatcher, "read session") {
        sessionQueries.selectSessionById(id, ::sessionRow).executeAsOneOrNull()
    }

    override suspend fun deleteSession(id: Long) {
        dbCall(dispatcher, "delete session") {
            sessionQueries.transaction { sessionQueries.deleteSession(id) }
        }
        // Rows first: a crash in between leaves orphan files (swept at startup), never rows
        // pointing at deleted audio.
        withContext(dispatcher) {
            val dir = File(filesDir, "${DataFiles.SESSIONS_DIR}/$id")
            val deleted = try {
                !dir.exists() || dir.deleteRecursively()
            } catch (e: SecurityException) {
                false
            }
            if (!deleted) DataLog.w("could not delete audio for session $id; left for sweepOrphans")
        }
    }

    @Suppress("LongParameterList")
    private fun sessionRow(
        id: Long,
        startedAt: String,
        endedAt: String?,
        mode: String,
        direction: String,
        topicSummary: String?,
        notes: String?,
        asrModel: String,
        translateModel: String?,
        exchangeCount: Long,
        costUsd: Double,
    ) = SessionRow(
        id = id,
        startedAt = Timestamps.parse(startedAt),
        endedAt = endedAt?.let(Timestamps::parse),
        mode = RowCodec.mode(mode),
        direction = RowCodec.direction(direction),
        topicSummary = topicSummary,
        notes = notes,
        asrModel = asrModel,
        translateModel = translateModel,
        exchangeCount = exchangeCount.toInt(),
        costUsd = costUsd,
    )

    @Suppress("LongParameterList")
    private fun exchangeRow(
        id: Long,
        sessionId: Long,
        segmentId: Long,
        timestamp: String,
        direction: String,
        rawTranscript: String,
        correctedSource: String?,
        translation: String,
        confidence: Double?,
        isCorrection: Long,
        correctionJson: String?,
        flaggedJson: String,
        processingMs: Long?,
        detectedLanguage: String?,
        translationError: Long,
        audioPath: String?,
        correctedTranscript: String?,
    ) = ExchangeRow(
        id = id,
        sessionId = sessionId,
        segmentId = segmentId.toInt(),
        timestamp = Timestamps.parse(timestamp),
        direction = RowCodec.direction(direction),
        rawTranscript = rawTranscript,
        correctedSource = correctedSource,
        translation = translation,
        confidence = confidence,
        isCorrection = RowCodec.bool(isCorrection),
        correction = RowCodec.correction(correctionJson),
        flagged = RowCodec.flagged(flaggedJson),
        processingMs = processingMs,
        detectedLanguage = detectedLanguage,
        translationError = RowCodec.bool(translationError),
        audioPath = audioPath,
        correctedTranscript = correctedTranscript,
    )
}
