package com.xrmech.habla.data

import com.xrmech.habla.data.db.HablaDatabase
import java.io.File
import java.io.IOException
import java.time.Clock
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Training capture (DM1, DM2): every final segment, accepted or rejected, as
 * `files/sessions/<sessionId>/segment_NNN.wav` plus a `training_segments` row. `:app` decides
 * whether capture is on; this class writes whatever it is handed. Capture must never break the
 * live pipeline, so storage problems return null instead of throwing.
 */
class TrainingAudioStore(
    private val filesDir: File,
    db: HablaDatabase,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val clock: Clock = Clock.systemUTC(),
    internal val usableSpace: (File) -> Long = File::getUsableSpace,
) {
    private val sessionQueries = db.sessionQueries
    private val trainingQueries = db.trainingSegmentQueries
    private val writer = WavWriter()

    /**
     * Writes the WAV and its row and returns `"sessions/<sid>/segment_NNN.wav"` (relative to
     * filesDir). An existing `(session, segment)` returns its path without writing.
     *
     * @throws IllegalArgumentException for segmentId < 1, empty pcm or an unknown session.
     * @return null (logged) on low space, an I/O error or a database failure.
     */
    suspend fun saveSegment(capture: SegmentCapture): String? {
        require(capture.segmentId >= 1) { "segmentId must be >= 1" }
        require(capture.pcm.isNotEmpty()) { "pcm is empty" }
        val sid = capture.sessionId
        val seg = capture.segmentId
        return withContext(dispatcher) {
            val existing = try {
                require(sessionQueries.sessionExists(sid).executeAsOne() > 0L) { "unknown session $sid" }
                trainingQueries.selectTrainingFilename(sid, seg.toLong()).executeAsOneOrNull()
            } catch (e: IllegalArgumentException) {
                throw e
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                DataLog.w("training lookup failed for session $sid segment $seg: ${e.javaClass.simpleName}")
                return@withContext null
            }
            if (existing != null) return@withContext relativePath(sid, existing)

            val name = fileName(seg)
            val dataBytes = capture.pcm.size.toLong() * WavWriter.BYTES_PER_SAMPLE
            // The floor leaves room for SQLite journals and DataStore writes, which fail far worse
            // than a skipped training clip.
            if (usableSpace(filesDir) < MIN_FREE_BYTES + dataBytes + WavWriter.HEADER_BYTES) {
                DataLog.w("low storage; skipped training audio for session $sid segment $seg")
                return@withContext null
            }
            val dir = File(filesDir, "${DataFiles.SESSIONS_DIR}/$sid")
            val wav = File(dir, name)
            val tmp = File(dir, "$name$TMP_SUFFIX")
            try {
                if (!dir.isDirectory && !dir.mkdirs()) throw IOException("cannot create session directory")
                writer.write(tmp, capture.pcm)
                // A complete WAV left by a crash between rename and insert has no row; replace it.
                if (wav.exists()) wav.delete()
                if (!tmp.renameTo(wav)) throw IOException("rename failed")
            } catch (e: IOException) {
                DataLog.w("training audio write failed for session $sid segment $seg: ${e.javaClass.simpleName}")
                tmp.delete()
                return@withContext null
            }
            try {
                trainingQueries.insertTrainingSegment(
                    session_id = sid,
                    segment_id = seg.toLong(),
                    filename = name,
                    size_bytes = dataBytes,
                    duration_seconds = dataBytes / BYTES_PER_SECOND,
                    recorded_at = Timestamps.now(clock),
                    audio_rms = capture.audioRms,
                    vad_avg_prob = capture.vadAvgProb,
                    detected_language = capture.detectedLanguage,
                    asr_status = capture.status.dbValue,
                    asr_time_seconds = capture.asrTimeSeconds,
                    asr_avg_logprob = capture.asrAvgLogprob,
                )
            } catch (e: CancellationException) {
                wav.delete()
                throw e
            } catch (e: Exception) {
                DataLog.w("training row insert failed for session $sid segment $seg: ${e.javaClass.simpleName}")
                wav.delete()
                return@withContext null
            }
            relativePath(sid, name)
        }
    }

    /** Rows ordered by `segment_id`, each joined to its lowest-id exchange. @throws DataException */
    suspend fun segments(sessionId: Long): List<TrainingSegmentRow> = dbCall(dispatcher, "read training segments") {
        trainingQueries.selectTrainingSegmentsForSession(sessionId, ::segmentRow).executeAsList()
    }

    /**
     * Deletes `sessions/<name>` directories whose name is not a session id in the database, and
     * stray `*.wav.tmp` files in the rest. Called once per process start, after the DB opens.
     *
     * @return the number of entries deleted. @throws DataException if the session ids cannot be read.
     */
    suspend fun sweepOrphans(): Int = withContext(dispatcher) {
        val root = File(filesDir, DataFiles.SESSIONS_DIR)
        // List before reading ids: a session created in between is then in the id set, never swept.
        val children = root.listFiles() ?: return@withContext 0
        val ids = dbCall(dispatcher, "read session ids") { sessionQueries.selectAllSessionIds().executeAsList().toSet() }
        var deleted = 0
        for (child in children) {
            val id = child.name.takeIf { SESSION_DIR_NAME.matches(it) }?.toLongOrNull()
            if (id == null || id.toString() != child.name || id !in ids) {
                if (deleteQuietly(child)) deleted++
                continue
            }
            child.listFiles()?.filter { it.isFile && it.name.endsWith(TMP_SUFFIX) }?.forEach {
                if (deleteQuietly(it)) deleted++
            }
        }
        if (deleted > 0) DataLog.i("swept $deleted orphan training entries")
        deleted
    }

    private fun deleteQuietly(f: File): Boolean = try {
        f.deleteRecursively().also { if (!it) DataLog.w("could not delete orphan training entry") }
    } catch (e: SecurityException) {
        DataLog.w("could not delete orphan training entry: ${e.javaClass.simpleName}")
        false
    }

    private fun relativePath(sessionId: Long, filename: String) = "${DataFiles.SESSIONS_DIR}/$sessionId/$filename"

    @Suppress("LongParameterList")
    private fun segmentRow(
        sessionId: Long,
        segmentId: Long,
        filename: String,
        sizeBytes: Long,
        durationSeconds: Double,
        recordedAt: String,
        audioRms: Double,
        vadAvgProb: Double,
        detectedLanguage: String?,
        asrStatus: String,
        asrTimeSeconds: Double,
        asrAvgLogprob: Double?,
        exchangeId: Long?,
        rawTranscript: String?,
        translation: String?,
        confidence: Double?,
        flaggedJson: String?,
        correctedTranscript: String?,
    ) = TrainingSegmentRow(
        sessionId = sessionId,
        segmentId = segmentId.toInt(),
        filename = filename,
        sizeBytes = sizeBytes,
        durationSeconds = durationSeconds,
        recordedAt = Timestamps.parse(recordedAt),
        audioRms = audioRms,
        vadAvgProb = vadAvgProb,
        detectedLanguage = detectedLanguage,
        status = AsrStatus.entries.firstOrNull { it.dbValue == asrStatus }
            ?: AsrStatus.REJECTED.also { DataLog.w("unknown asr_status in row; using rejected") },
        asrTimeSeconds = asrTimeSeconds,
        asrAvgLogprob = asrAvgLogprob,
        exchangeId = exchangeId,
        rawTranscript = rawTranscript,
        translation = translation,
        confidence = confidence,
        flaggedCount = RowCodec.flagged(flaggedJson).size,
        correctedTranscript = correctedTranscript,
    )

    companion object {
        /** Free-space floor kept for the database and DataStore. */
        const val MIN_FREE_BYTES: Long = 50L * 1024 * 1024

        private const val TMP_SUFFIX = ".tmp"
        private const val BYTES_PER_SECOND = (WavWriter.SAMPLE_RATE * WavWriter.BYTES_PER_SAMPLE).toDouble()
        private val SESSION_DIR_NAME = Regex("[0-9]+")

        /** Server parity: `f"segment_{n:03d}.wav"`; ids above 999 widen. */
        internal fun fileName(segmentId: Int): String = String.format(Locale.ROOT, "segment_%03d.wav", segmentId)
    }
}
