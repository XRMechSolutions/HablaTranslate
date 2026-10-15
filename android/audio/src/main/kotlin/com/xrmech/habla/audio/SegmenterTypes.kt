package com.xrmech.habla.audio

/**
 * Receives [Segmenter] results. Every callback runs synchronously on the thread that called
 * `feed`/`flush` and must not block; arrays inside the arguments are fresh copies the listener owns.
 * Per frame at most one of: [onCalibrated]; or [onPartial] and/or [onSegment] (partial first).
 * [onVadFallback] fires before that frame's other callbacks.
 */
interface SegmenterListener {
    /** Calibration finished (after construction and after each reset); never when `calibration.duration_ms == 0`. */
    fun onCalibrated(noiseFloorRms: Double)

    /** Snapshot of the in-progress segment, about every `partial_interval_ms` of speech. */
    fun onPartial(snapshot: PartialSnapshot)

    /** A completed speech segment. */
    fun onSegment(segment: SpeechSegment)

    /** Silero failed mid-session; the Segmenter now runs the energy fallback until it is rebuilt. */
    fun onVadFallback(cause: VadException) {}
}

/**
 * A final speech segment (server `_emit_segment`). Not a data class: array equality would be by
 * reference.
 *
 * @property id 1-based, = segments emitted since the last reset
 * @property pcm pre-speech ring + speech + silence frames (+ the 4800-sample tail after a max split)
 * @property durationSec `pcm.size / 16000.0`
 * @property rms int16-unit RMS of [pcm]
 * @property vadAvgProb probability sum / speech frames (server quirk after a split, see design)
 * @property startSample stream position (since reset) of `pcm[0]`
 * @property endSample stream position one past the last frame processed (exclusive);
 *   `endSample - startSample > pcm.size` when stale pre-speech ring frames were prepended
 */
class SpeechSegment(
    val id: Int,
    val pcm: ShortArray,
    val durationSec: Double,
    val rms: Double,
    val vadAvgProb: Double,
    val startSample: Long,
    val endSample: Long,
) {
    override fun toString(): String =
        "SpeechSegment(id=$id, samples=${pcm.size}, durationSec=$durationSec, rms=$rms, " +
            "vadAvgProb=$vadAvgProb, startSample=$startSample, endSample=$endSample)"
}

/**
 * A copy of the in-progress segment's audio for partial ASR.
 *
 * @property segmentId id the in-progress segment will get when emitted
 * @property pcm the whole speech buffer so far
 * @property durationSec `pcm.size / 16000.0`
 */
class PartialSnapshot(
    val segmentId: Int,
    val pcm: ShortArray,
    val durationSec: Double,
) {
    override fun toString(): String = "PartialSnapshot(segmentId=$segmentId, samples=${pcm.size}, durationSec=$durationSec)"
}
