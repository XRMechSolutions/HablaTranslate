package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Port of the server's `StreamingVADBuffer` (`habla/server/pipeline/vad_buffer.py`: `feed_pcm`,
 * `_emit_segment`, `flush`, `reset`), including its quirks (D007): calibration frames never reach
 * the VAD, the pre-speech ring is not cleared at onset, a max split re-seeds the next segment with
 * a 300 ms tail and `speechFrames = 9` while the probability sum restarts at 0.
 *
 * Every frame count is the server's integer division by `vad.frame_ms`. Single-threaded and not
 * thread-safe; listener callbacks run synchronously inside [feed] and [flush]. The Segmenter
 * never closes the [VadModel] it is given.
 *
 * @param vad the model; `null` (Silero failed to load) or an [EnergyVad] means the server's energy
 *   fallback, decided by `energy > threshold`
 * @throws IllegalArgumentException naming the profile key when the profile cannot drive the port
 */
class Segmenter(private val profile: AsrProfile, vad: VadModel?, private val listener: SegmenterListener) {
    private val sampleRate = profile.audio.sampleRate
    private val frameSamples = profile.vad.frameSamples
    private val speechThreshold = profile.vad.speechThreshold
    private val minEnergyThreshold = profile.calibration.minEnergyThreshold
    private val noiseFloorMultiplier = profile.calibration.noiseFloorMultiplier

    internal val preRingFrames: Int
    internal val silenceFrames: Int
    internal val minSpeechFrames: Int
    internal val partialIntervalFrames: Int
    internal val maxSegmentFrames: Int
    internal val calibrationFrames: Int
    internal val tailSamples: Int
    internal val initialCapacity: Int
    internal val maxBufferSamples: Int

    init {
        val seg = profile.segmenter
        val cal = profile.calibration
        require(sampleRate == 16000) { "audio.sample_rate must be 16000, was $sampleRate" }
        require(frameSamples == VAD_FRAME_SAMPLES) { "vad.frame_samples must be $VAD_FRAME_SAMPLES, was $frameSamples" }
        val frameMs = profile.vad.frameMs
        require(frameMs > 0 && frameSamples.toLong() * 1000 == sampleRate.toLong() * frameMs) {
            "vad.frame_ms must equal frame_samples * 1000 / sample_rate (32), was $frameMs"
        }
        require(seg.silenceDurationMs >= 0) { "segmenter.silence_duration_ms must be >= 0, was ${seg.silenceDurationMs}" }
        require(seg.minSpeechMs >= 0) { "segmenter.min_speech_ms must be >= 0, was ${seg.minSpeechMs}" }
        require(seg.preSpeechPaddingMs >= 0) { "segmenter.pre_speech_padding_ms must be >= 0, was ${seg.preSpeechPaddingMs}" }
        require(seg.partialIntervalMs >= 0) { "segmenter.partial_interval_ms must be >= 0, was ${seg.partialIntervalMs}" }
        require(cal.durationMs >= 0) { "calibration.duration_ms must be >= 0, was ${cal.durationMs}" }
        require(seg.maxSegmentSeconds > 0.0) { "segmenter.max_segment_seconds must be > 0, was ${seg.maxSegmentSeconds}" }

        preRingFrames = max(1, seg.preSpeechPaddingMs / frameMs)
        silenceFrames = max(1, seg.silenceDurationMs / frameMs)
        minSpeechFrames = max(1, seg.minSpeechMs / frameMs)
        partialIntervalFrames = max(1, seg.partialIntervalMs / frameMs)
        maxSegmentFrames = (seg.maxSegmentSeconds * 1000.0 / frameMs).toInt()
        require(maxSegmentFrames >= 1) {
            "segmenter.max_segment_seconds gives $maxSegmentFrames frames of ${frameMs} ms, need >= 1"
        }
        calibrationFrames = if (cal.durationMs > 0) max(1, cal.durationMs / frameMs) else 0
        // The server computes the tail in bytes: padding_ms * rate * 2 // 1000, then halves it.
        tailSamples = (seg.preSpeechPaddingMs.toLong() * sampleRate * 2 / 1000 / 2).toInt()
        require((preRingFrames + maxSegmentFrames.toLong()) * frameSamples <= Int.MAX_VALUE / 4) {
            "segmenter.max_segment_seconds is too large for an in-memory segment buffer, was ${seg.maxSegmentSeconds}"
        }
        initialCapacity = (preRingFrames + maxSegmentFrames) * frameSamples
        // Worst case: max_segment_seconds counts speech frames only, so one speech frame per
        // (silenceFrames - 1) silent frames keeps a segment open until maxSegmentFrames.
        val bound = max(tailSamples, preRingFrames * frameSamples).toLong() +
            (maxSegmentFrames.toLong() + (maxSegmentFrames - 1).toLong() * (silenceFrames - 1) + silenceFrames) * frameSamples
        maxBufferSamples = min(bound, Int.MAX_VALUE.toLong() - 8).toInt()
    }

    private var model: VadModel = vad ?: EnergyVad(profile)
    private var energyModel: EnergyVad? = model as? EnergyVad

    private val pending = ShortArray(frameSamples)
    private var pendingLen = 0
    private val frameF = FloatArray(frameSamples)

    private val ring = ShortArray(preRingFrames * frameSamples)
    private val ringFrameIndex = LongArray(preRingFrames)
    private var ringStart = 0
    private var ringCount = 0

    private var speech = ShortArray(initialCapacity)
    private var speechLen = 0

    private var speaking = false
    private var silenceCount = 0
    private var speechFrames = 0
    private var probSum = 0.0
    private var framesSincePartial = 0
    private var segmentStartSample = 0L
    private var framesConsumed = 0L

    private var calCount = 0
    private var calSum = 0.0
    private var energyThreshold = profile.calibration.uncalibratedThreshold

    private val events = ArrayList<Event>(3)

    /** True until `calibration.duration_ms` of audio has been fed since construction or [reset]. */
    var calibrating: Boolean = calibrationFrames > 0
        private set

    /** Segments emitted since the last [reset] (server `segments_emitted`). */
    var segmentsEmitted: Int = 0
        private set

    /** Seconds of emitted segment audio since the last [reset] (server `total_speech_seconds`). */
    var totalSpeechSeconds: Double = 0.0
        private set

    /** Average int16 RMS of the calibration frames; `uncalibrated_threshold` until the first calibration. */
    var noiseFloorRms: Double = profile.calibration.uncalibratedThreshold
        private set

    /** True when built with `vad == null` or an [EnergyVad], or after Silero failed mid-session. */
    val usingEnergyFallback: Boolean get() = energyModel != null

    /** Current speech buffer capacity in samples (tests check it shrinks back after a long segment). */
    internal val speechCapacity: Int get() = speech.size

    /**
     * Consumes `pcm[offset until offset + length]`, processing every completed 512-sample frame
     * and keeping the remainder for the next call; output is identical for any chunking.
     *
     * @throws IndexOutOfBoundsException for a bad range (nothing consumed)
     * @throws IllegalStateException if a segment would exceed the proven buffer bound (a bug)
     */
    fun feed(pcm: ShortArray, offset: Int = 0, length: Int = pcm.size) {
        if (offset < 0 || length < 0 || offset > pcm.size - length) {
            throw IndexOutOfBoundsException("offset=$offset length=$length size=${pcm.size}")
        }
        var pos = offset
        val end = offset + length
        while (pos < end) {
            val n = min(frameSamples - pendingLen, end - pos)
            pcm.copyInto(pending, pendingLen, pos, pos + n)
            pendingLen += n
            pos += n
            if (pendingLen < frameSamples) return
            pendingLen = 0
            processFrame()
        }
    }

    /**
     * Server `flush`: emits the in-progress segment if it has at least `min_speech_ms` of speech,
     * then drops the pending partial frame and the speech buffer. Calibration, ring and VAD state
     * are kept.
     */
    fun flush() {
        events.clear()
        if (speaking && speechFrames >= minSpeechFrames) emit()
        pendingLen = 0
        speechLen = 0
        speaking = false
        shrinkSpeech()
        dispatch()
    }

    /**
     * Server `reset` for a new session: clears buffers and counters (including [segmentsEmitted]
     * and the sample position), restarts calibration and resets the VAD model. An energy fallback
     * taken mid-session stays in force. Never invokes callbacks.
     *
     * @throws IllegalStateException if the VadModel was closed
     */
    fun reset() {
        model.reset()
        pendingLen = 0
        speechLen = 0
        shrinkSpeech()
        ringStart = 0
        ringCount = 0
        speaking = false
        silenceCount = 0
        speechFrames = 0
        probSum = 0.0
        framesSincePartial = 0
        calibrating = calibrationFrames > 0
        calCount = 0
        calSum = 0.0
        framesConsumed = 0
        segmentsEmitted = 0
        totalSpeechSeconds = 0.0
        events.clear()
    }

    private fun processFrame() {
        val frameIndex = framesConsumed
        framesConsumed++
        events.clear()

        if (calibrating) {
            var sum = 0.0
            for (s in pending) sum += s.toDouble() * s
            calSum += sqrt(sum / frameSamples)
            calCount++
            pushRing(frameIndex)
            if (calCount >= calibrationFrames) {
                calibrating = false
                noiseFloorRms = calSum / calCount
                energyThreshold = max(minEnergyThreshold, noiseFloorRms * noiseFloorMultiplier)
                energyModel?.energyThreshold = energyThreshold
                events += Event.Calibrated(noiseFloorRms)
            }
            dispatch()
            return
        }

        for (i in 0 until frameSamples) frameF[i] = pending[i] / 32768f
        val prob = try {
            model.speechProbability(frameF)
        } catch (e: VadException) {
            val fallback = EnergyVad(profile).also { it.energyThreshold = energyThreshold }
            model = fallback
            energyModel = fallback
            events += Event.Fallback(e)
            fallback.speechProbability(frameF)
        }
        val em = energyModel
        val isSpeech = if (em != null) em.lastEnergy > em.energyThreshold else prob.toDouble() > speechThreshold

        if (isSpeech) {
            if (!speaking) startSegment(frameIndex)
            appendSpeech(pending, 0, frameSamples)
            speechFrames++
            probSum += prob.toDouble()
            silenceCount = 0
            framesSincePartial++
            if (framesSincePartial >= partialIntervalFrames && speechFrames >= minSpeechFrames) {
                framesSincePartial = 0
                events += Event.Partial(PartialSnapshot(segmentsEmitted + 1, speech.copyOf(speechLen), speechLen / sampleRate.toDouble()))
            }
            if (speechFrames >= maxSegmentFrames) {
                // An empty tail when padding is 0 (Python's buf[-0:] would re-seed the whole segment).
                val tail = if (speechLen > tailSamples) speech.copyOfRange(speechLen - tailSamples, speechLen) else ShortArray(0)
                val endSample = emit()
                speaking = true
                tail.copyInto(speech)
                speechLen = tail.size
                speechFrames = tail.size / frameSamples
                segmentStartSample = endSample - tail.size
            }
        } else if (speaking) {
            appendSpeech(pending, 0, frameSamples)
            silenceCount++
            if (silenceCount >= silenceFrames) emit()
        } else {
            pushRing(frameIndex)
        }
        dispatch()
    }

    private fun startSegment(frameIndex: Long) {
        speaking = true
        silenceCount = 0
        speechFrames = 0
        probSum = 0.0
        framesSincePartial = 0
        speechLen = 0
        // The ring is not cleared here: frames from before an earlier segment can be prepended (server quirk).
        segmentStartSample = (if (ringCount > 0) ringFrameIndex[ringStart] else frameIndex) * frameSamples
        for (k in 0 until ringCount) {
            val slot = (ringStart + k) % preRingFrames
            appendSpeech(ring, slot * frameSamples, frameSamples)
        }
    }

    /** `_emit_segment`; returns the segment's end sample (also when the segment was too short). */
    private fun emit(): Long {
        speaking = false
        val endSample = framesConsumed * frameSamples
        if (speechFrames < minSpeechFrames) {
            speechLen = 0
            speechFrames = 0
            probSum = 0.0
            return endSample
        }
        val pcm = speech.copyOf(speechLen)
        val durationSec = speechLen / sampleRate.toDouble()
        var sum = 0.0
        for (s in pcm) sum += s.toDouble() * s
        val rms = if (pcm.isEmpty()) 0.0 else sqrt(sum / pcm.size)
        val vadAvgProb = probSum / speechFrames
        segmentsEmitted++
        totalSpeechSeconds += durationSec
        speechLen = 0
        speechFrames = 0
        silenceCount = 0
        probSum = 0.0
        shrinkSpeech()
        events += Event.Segment(SpeechSegment(segmentsEmitted, pcm, durationSec, rms, vadAvgProb, segmentStartSample, endSample))
        return endSample
    }

    private fun pushRing(frameIndex: Long) {
        val slot: Int
        if (ringCount < preRingFrames) {
            slot = (ringStart + ringCount) % preRingFrames
            ringCount++
        } else {
            slot = ringStart
            ringStart = (ringStart + 1) % preRingFrames
        }
        pending.copyInto(ring, slot * frameSamples, 0, frameSamples)
        ringFrameIndex[slot] = frameIndex
    }

    private fun appendSpeech(src: ShortArray, from: Int, n: Int) {
        val needed = speechLen + n
        if (needed > speech.size) {
            check(needed <= maxBufferSamples) {
                "segment buffer would exceed its bound of $maxBufferSamples samples (Segmenter invariant broken)"
            }
            var cap = speech.size.toLong()
            while (cap < needed) cap *= 2
            speech = speech.copyOf(min(cap, maxBufferSamples.toLong()).toInt())
        }
        src.copyInto(speech, speechLen, from, from + n)
        speechLen = needed
    }

    /** One pathological segment must not pin a grown buffer for the rest of the session. */
    private fun shrinkSpeech() {
        if (speech.size > initialCapacity && speechLen <= initialCapacity) {
            speech = speech.copyOf(initialCapacity)
        }
    }

    /** Callbacks run after the frame's state is fully updated; a throwing listener skips the rest. */
    private fun dispatch() {
        if (events.isEmpty()) return
        val batch = events.toTypedArray()
        events.clear()
        for (e in batch) {
            when (e) {
                is Event.Fallback -> listener.onVadFallback(e.cause)
                is Event.Calibrated -> listener.onCalibrated(e.noiseFloorRms)
                is Event.Partial -> listener.onPartial(e.snapshot)
                is Event.Segment -> listener.onSegment(e.segment)
            }
        }
    }

    private sealed interface Event {
        class Fallback(val cause: VadException) : Event
        class Calibrated(val noiseFloorRms: Double) : Event
        class Partial(val snapshot: PartialSnapshot) : Event
        class Segment(val segment: SpeechSegment) : Event
    }
}
