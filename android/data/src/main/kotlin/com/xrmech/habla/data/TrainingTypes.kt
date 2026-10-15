package com.xrmech.habla.data

import java.time.Instant

/** One final segment handed to [TrainingAudioStore.saveSegment] after ASR and the filters ran. */
data class SegmentCapture(
    val sessionId: Long,
    /** >= 1; also the NNN in `segment_NNN.wav`. */
    val segmentId: Int,
    /** 16 kHz mono, the raw segment as emitted by the Segmenter (not FinalPrep output). */
    val pcm: ShortArray,
    /** SpeechSegment.rms */
    val audioRms: Double,
    /** SpeechSegment.vadAvgProb */
    val vadAvgProb: Double,
    /** AsrResult.detectedLanguage */
    val detectedLanguage: String?,
    val status: AsrStatus,
    /** AsrResult.elapsedMs / 1000.0 for the FINAL pass. */
    val asrTimeSeconds: Double,
    /** Mean of the non-null AsrResult.segments[].avgLogprob; null if none. */
    val asrAvgLogprob: Double?,
) {
    override fun equals(other: Any?): Boolean = other is SegmentCapture &&
        sessionId == other.sessionId && segmentId == other.segmentId && pcm.contentEquals(other.pcm) &&
        audioRms == other.audioRms && vadAvgProb == other.vadAvgProb &&
        detectedLanguage == other.detectedLanguage && status == other.status &&
        asrTimeSeconds == other.asrTimeSeconds && asrAvgLogprob == other.asrAvgLogprob

    override fun hashCode(): Int {
        var h = sessionId.hashCode()
        h = 31 * h + segmentId
        h = 31 * h + pcm.contentHashCode()
        h = 31 * h + status.hashCode()
        return h
    }
}

/** ASR outcome of a captured segment. Stored as `"accepted"` / `"rejected"`. */
enum class AsrStatus(val dbValue: String) {
    ACCEPTED("accepted"),
    REJECTED("rejected"),
}

/** A `training_segments` row joined to the lowest-id exchange with the same session and segment. */
data class TrainingSegmentRow(
    val sessionId: Long,
    val segmentId: Int,
    val filename: String,
    val sizeBytes: Long,
    val durationSeconds: Double,
    val recordedAt: Instant,
    val audioRms: Double,
    val vadAvgProb: Double,
    val detectedLanguage: String?,
    val status: AsrStatus,
    val asrTimeSeconds: Double,
    val asrAvgLogprob: Double?,
    val exchangeId: Long?,
    val rawTranscript: String?,
    val translation: String?,
    val confidence: Double?,
    /** Size of the exchange's decoded `flagged_json`; 0 when there is no exchange or it is corrupt. */
    val flaggedCount: Int,
    val correctedTranscript: String?,
)
