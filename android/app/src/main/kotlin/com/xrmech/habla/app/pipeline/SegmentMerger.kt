package com.xrmech.habla.app.pipeline

import com.xrmech.habla.audio.SpeechSegment
import com.xrmech.habla.shared.MergeProfile

/** One unit of final work: one or more raw segments joined by the merge rule. */
class MergedSegment(
    /** Server `_segment_counter`: 1-based per session; exchanges.segment_id and training file number. */
    val seq: Int,
    val pcm: ShortArray,
    val durationSec: Double,
    /** Duration-weighted mean of the raw segments' RMS. */
    val rms: Double,
    /** Duration-weighted mean of the raw segments' VAD probability. */
    val vadAvgProb: Double,
    val rawIds: List<Int>,
    /** Stream position (samples since session start) at which the last raw segment was emitted. */
    val endSample: Long,
) {
    override fun toString(): String =
        "MergedSegment(seq=$seq, samples=${pcm.size}, durationSec=$durationSec, rawIds=$rawIds, endSample=$endSample)"
}

/**
 * Port of the server's `_queue_segment` / `_flush_pending_after_gap` / `_flush_pending_now`
 * (design section 3.4). The 1.2 s gap is counted in stream samples, not wall time: identical to
 * the server live, deterministic when a WAV is fed faster than real time. Not thread-safe: the
 * coordinator calls it only on its control dispatcher.
 */
class SegmentMerger(
    profile: MergeProfile,
    private val emit: (MergedSegment) -> Unit,
    sampleRate: Int = 16_000,
) {
    private val gapSamples: Long = Math.round(profile.gapSeconds * sampleRate)
    private val maxPendingSeconds: Double = profile.maxPendingSeconds

    private val pending = ArrayList<ShortArray>()
    private var pendingSamples = 0
    private var pendingDuration = 0.0
    private var rmsSum = 0.0
    private var probSum = 0.0
    private var weight = 0.0
    private val rawIds = ArrayList<Int>()
    private var lastEmitSample = 0L
    private var seq = 0

    val hasPending: Boolean get() = pending.isNotEmpty()

    /** Server `_queue_segment`: empty pcm ignored; 60 s of pending audio flushes at once. */
    fun onSegment(segment: SpeechSegment) {
        if (segment.pcm.isEmpty()) return
        pending += segment.pcm
        pendingSamples += segment.pcm.size
        pendingDuration += segment.durationSec
        rmsSum += segment.rms * segment.durationSec
        probSum += segment.vadAvgProb * segment.durationSec
        weight += segment.durationSec
        rawIds += segment.id
        lastEmitSample = segment.endSample
        if (pendingDuration >= maxPendingSeconds) flushNow()
    }

    /** Stream progress after each fed frame; flushes once the gap since the last segment has passed. */
    fun onAudio(samplesFed: Long) {
        if (pending.isNotEmpty() && samplesFed - lastEmitSample >= gapSamples) flushNow()
    }

    /** Server `_flush_pending_now`; no-op when nothing is pending. */
    fun flushNow() {
        if (pending.isEmpty()) return
        val pcm = ShortArray(pendingSamples)
        var at = 0
        for (part in pending) {
            part.copyInto(pcm, at)
            at += part.size
        }
        val merged = MergedSegment(
            seq = ++seq,
            pcm = pcm,
            durationSec = pendingDuration,
            rms = if (weight > 0) rmsSum / weight else 0.0,
            vadAvgProb = if (weight > 0) probSum / weight else 0.0,
            rawIds = rawIds.toList(),
            endSample = lastEmitSample,
        )
        pending.clear()
        pendingSamples = 0
        pendingDuration = 0.0
        rmsSum = 0.0
        probSum = 0.0
        weight = 0.0
        rawIds.clear()
        emit(merged)
    }
}
