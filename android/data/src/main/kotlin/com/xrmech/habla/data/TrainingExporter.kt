package com.xrmech.habla.data

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.time.Clock
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext

/** Result of one export, for the "N segments, M missing" message. */
data class ExportSummary(val segmentsWritten: Int, val segmentsMissing: Int, val correctedIncluded: Boolean)

class ExportException(message: String, val kind: Kind, cause: Throwable? = null) : Exception(message, cause) {
    enum class Kind { SESSION_NOT_FOUND, NO_SEGMENTS, IO }
}

/**
 * Streams one session's training data as a zip in the server's recording layout
 * (`android-<id>_<yyyyMMdd_HHmmss>/segment_NNN.wav`, `metadata.json`, `ground_truth.json`, and
 * `corrected_ground_truth.json` when the user edited a transcript). Memory stays at one 64 KiB copy
 * buffer however long the session is. The caller owns [OutputStream] and it is never closed here.
 */
class TrainingExporter(
    private val filesDir: File,
    private val history: HistoryRepository,
    private val training: TrainingAudioStore,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val clock: Clock = Clock.systemUTC(),
) {
    /**
     * @throws ExportException `SESSION_NOT_FOUND`, `NO_SEGMENTS` (no rows, or every WAV missing,
     *   before any JSON is written) or `IO` (writing [out] failed).
     * @throws kotlinx.coroutines.CancellationException leaving [out] partial and open.
     */
    suspend fun export(sessionId: Long, out: OutputStream): ExportSummary {
        val session = history.session(sessionId)
            ?: throw ExportException("session $sessionId not found", ExportException.Kind.SESSION_NOT_FOUND)
        val rows = training.segments(sessionId)
        if (rows.isEmpty()) throw ExportException("session $sessionId has no training segments", ExportException.Kind.NO_SEGMENTS)
        val directions = history.exchanges(sessionId).first().map { it.direction }

        return withContext(dispatcher) {
            val root = rootName(session)
            val zip = ZipOutputStream(BufferedOutputStream(out, BUFFER_BYTES))
            zip.setLevel(Deflater.BEST_SPEED)
            val buffer = ByteArray(BUFFER_BYTES)
            val written = mutableListOf<TrainingSegmentRow>()
            var missing = 0
            try {
                // WAVs first, so the JSON lists only files that actually made it into the zip.
                for (row in rows) {
                    val input = openWav(sessionId, row)
                    if (input == null) {
                        missing++
                        continue
                    }
                    input.use {
                        zip.putNextEntry(ZipEntry("$root/${row.filename}"))
                        val complete = copy(it, zip, buffer)
                        zip.closeEntry()
                        if (complete) written += row else missing++
                    }
                }
                if (written.isEmpty()) {
                    throw ExportException("no training audio found for session $sessionId", ExportException.Kind.NO_SEGMENTS)
                }
                val now = Timestamps.now(clock)
                val direction = ExportJson.resolveDirection(session.direction, directions)
                putJson(zip, "$root/metadata.json", ExportJson.encode(ExportJson.metadata(session, written)))
                putJson(zip, "$root/ground_truth.json", ExportJson.encode(ExportJson.groundTruth(session, written, direction, now)))
                val corrected = written.any(ExportJson::isHumanCorrected)
                if (corrected) {
                    putJson(
                        zip,
                        "$root/corrected_ground_truth.json",
                        ExportJson.encode(ExportJson.correctedGroundTruth(session, written, direction, now)),
                    )
                }
                // finish() + flush, never close(): closing would close the caller's stream.
                zip.finish()
                zip.flush()
                ExportSummary(written.size, missing, corrected)
            } catch (e: IOException) {
                throw ExportException("export of session $sessionId failed: ${e.javaClass.simpleName}", ExportException.Kind.IO, e)
            }
        }
    }

    private fun openWav(sessionId: Long, row: TrainingSegmentRow): InputStream? {
        val file = File(filesDir, "${DataFiles.SESSIONS_DIR}/$sessionId/${row.filename}")
        return try {
            FileInputStream(file)
        } catch (e: IOException) {
            DataLog.w("training audio missing for session $sessionId segment ${row.segmentId}")
            null
        } catch (e: SecurityException) {
            DataLog.w("training audio unreadable for session $sessionId segment ${row.segmentId}")
            null
        }
    }

    /**
     * Copies [input] into the open entry, checking for cancellation between chunks. Returns false
     * when reading the WAV failed part way (the partial entry stays in the zip but is left out of
     * the JSON); write failures propagate as [IOException].
     */
    private fun kotlinx.coroutines.CoroutineScope.copy(input: InputStream, zip: ZipOutputStream, buffer: ByteArray): Boolean {
        while (true) {
            ensureActive()
            val n = try {
                input.read(buffer)
            } catch (e: IOException) {
                DataLog.w("training audio read failed: ${e.javaClass.simpleName}")
                return false
            }
            if (n < 0) return true
            zip.write(buffer, 0, n)
        }
    }

    private fun putJson(zip: ZipOutputStream, name: String, bytes: ByteArray) {
        zip.putNextEntry(ZipEntry(name))
        zip.write(bytes)
        zip.closeEntry()
    }

    companion object {
        // 64 KiB: one read/write chunk; the whole export holds this plus the segment list.
        private const val BUFFER_BYTES = 64 * 1024

        private val ROOT_TIME: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").withZone(ZoneOffset.UTC)

        /**
         * `android-<id>_<yyyyMMdd_HHmmss>` from the UTC start, matching the server's recording-dir
         * pattern `^.+_\d{8}_\d{6}$` (DM10).
         */
        internal fun rootName(session: SessionRow): String = "android-${session.id}_${ROOT_TIME.format(session.startedAt)}"

        /** File name for the SAF "create document" dialog, e.g. `android-12_20261015_141502.zip`. */
        fun suggestedFileName(session: SessionRow): String = "${rootName(session)}.zip"
    }
}
