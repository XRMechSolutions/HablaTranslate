package com.xrmech.habla.data

import com.xrmech.habla.translate.model.Direction
import java.math.BigDecimal
import java.math.RoundingMode
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject

/**
 * Builds the three files of a training export in the server's recording layout, so the PC dataset
 * builder (`prepare_dataset.py`) reads phone and server sessions alike:
 * - `metadata.json`: `AudioRecorder` (`__init__`, `save_pcm_segment`, `add_segment_metadata`, `stop_recording`)
 * - `ground_truth.json`: `tools/generate_ground_truth.py`
 * - `corrected_ground_truth.json`: `api_corrections.py` `save_corrected_ground_truth`
 */
internal object ExportJson {
    /** The generator's empty-transcript marker; the dataset builder skips it. */
    const val INAUDIBLE = "[inaudible]"

    @OptIn(ExperimentalSerializationApi::class)
    private val pretty = Json {
        prettyPrint = true
        prettyPrintIndent = "  "
    }

    /** UTF-8 bytes, 2-space indent, non-ASCII unescaped (Python `ensure_ascii=False`). */
    fun encode(element: JsonElement): ByteArray = pretty.encodeToString(JsonElement.serializer(), element).toByteArray(Charsets.UTF_8)

    /** Python `round(x, n)`: half-even on the exact binary value. */
    fun round(value: Double, decimals: Int): Double =
        if (value.isFinite()) BigDecimal(value).setScale(decimals, RoundingMode.HALF_EVEN).toDouble() else value

    fun metadata(session: SessionRow, rows: List<TrainingSegmentRow>): JsonObject = buildJsonObject {
        put("session_id", "android-${session.id}")
        put("started_at", Timestamps.format(session.startedAt))
        // The phone keeps no raw stream, only the per-segment WAVs.
        put("raw_audio_format", "none")
        put("sample_rate", WavWriter.SAMPLE_RATE)
        putJsonArray("segments") {
            for (r in rows) addJsonObject {
                put("segment_id", r.segmentId)
                put("filename", r.filename)
                put("size_bytes", r.sizeBytes)
                put("duration_seconds", r.durationSeconds)
                put("recorded_at", Timestamps.format(r.recordedAt))
                put("audio_rms", round(r.audioRms, 1))
                put("vad_avg_prob", round(r.vadAvgProb, 3))
                put("detected_language", r.detectedLanguage)
                when {
                    r.status == AsrStatus.REJECTED -> put("asr_status", "rejected")
                    r.exchangeId != null -> {
                        put("raw_transcript", r.rawTranscript)
                        put("translation", r.translation)
                        // No diarization on the phone.
                        put("speaker", JsonNull)
                        put("confidence", r.confidence)
                        put("idioms_detected", r.flaggedCount)
                    }
                }
            }
        }
        val endedAt = session.endedAt ?: rows.maxOfOrNull { it.recordedAt } ?: session.startedAt
        put("ended_at", Timestamps.format(endedAt))
        put("total_segments", rows.size)
    }

    fun groundTruth(session: SessionRow, rows: List<TrainingSegmentRow>, direction: Direction, generatedAt: String): JsonObject =
        buildJsonObject {
            header(session, rows, direction, generatedAt)
            putJsonArray("segments") {
                for (r in rows) addJsonObject { baseSegment(r, transcript(r)) }
            }
            put("full_transcript", rows.joinToString("\n") { "[${it.segmentId}] ${transcript(it)}" })
            put("full_translation", rows.joinToString("\n") { translation(it) })
        }

    fun correctedGroundTruth(session: SessionRow, rows: List<TrainingSegmentRow>, direction: Direction, generatedAt: String): JsonObject =
        buildJsonObject {
            header(session, rows, direction, generatedAt)
            var corrected = 0
            putJsonArray("segments") {
                for (r in rows) addJsonObject {
                    val human = isHumanCorrected(r)
                    if (human) corrected++
                    baseSegment(r, r.correctedTranscript ?: transcript(r))
                    // Editing is reviewing: the phone has no separate "reviewed" action.
                    put("reviewed", human)
                    put("human_corrected", human)
                }
            }
            put("corrected_at", generatedAt)
            putJsonObject("correction_stats") {
                put("reviewed", corrected)
                put("human_corrected", corrected)
                put("total", rows.size)
            }
        }

    fun isHumanCorrected(r: TrainingSegmentRow): Boolean =
        r.correctedTranscript != null && r.correctedTranscript != r.rawTranscript

    /** Session direction unless `auto`; then the most frequent exchange direction, ties -> `es_to_en`. */
    fun resolveDirection(sessionDirection: Direction, exchangeDirections: List<Direction>): Direction {
        if (sessionDirection != Direction.AUTO) return sessionDirection
        val es = exchangeDirections.count { it == Direction.ES_TO_EN }
        val en = exchangeDirections.count { it == Direction.EN_TO_ES }
        return if (en > es) Direction.EN_TO_ES else Direction.ES_TO_EN
    }

    private fun kotlinx.serialization.json.JsonObjectBuilder.header(
        session: SessionRow,
        rows: List<TrainingSegmentRow>,
        direction: Direction,
        generatedAt: String,
    ) {
        put("generated_at", generatedAt)
        put("whisper_model", session.asrModel)
        put("direction", EnumCodec.encode(direction))
        put("total_segments", rows.size)
    }

    private fun kotlinx.serialization.json.JsonObjectBuilder.baseSegment(r: TrainingSegmentRow, transcript: String) {
        put("segment_id", r.segmentId)
        put("filename", r.filename)
        put("transcript", transcript)
        // Generator meaning: mean Whisper avg_logprob, or integer 0 when there is none.
        val logprob = r.asrAvgLogprob
        if (logprob != null) put("confidence", round(logprob, 3)) else put("confidence", 0)
        put("asr_time_seconds", round(r.asrTimeSeconds, 2))
        put("translation", translation(r))
        // No LLM review pass on the phone.
        put("asr_corrections", "")
    }

    private fun transcript(r: TrainingSegmentRow): String {
        val raw = r.rawTranscript
        return if (r.status == AsrStatus.ACCEPTED && r.exchangeId != null && !raw.isNullOrBlank()) raw else INAUDIBLE
    }

    private fun translation(r: TrainingSegmentRow): String =
        if (r.status == AsrStatus.ACCEPTED) r.translation.orEmpty() else ""
}
