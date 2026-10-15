package com.xrmech.habla.app.pipeline

import com.xrmech.habla.audio.Segmenter
import com.xrmech.habla.audio.SegmenterListener
import com.xrmech.habla.audio.VadModel
import com.xrmech.habla.shared.AsrProfile

/** Builds the per-session segmenter; tests return a FakeSegmenter they drive by hand. */
fun interface SegmenterFactory {
    fun create(profile: AsrProfile, vad: VadModel?, listener: SegmenterListener): SegmenterPort
}

/** The slice of `com.xrmech.habla.audio.Segmenter` the coordinator uses. */
interface SegmenterPort {
    fun feed(pcm: ShortArray)
    fun flush()
    fun reset()
    val calibrating: Boolean
}

/**
 * Stream position seam: the coordinator's segmenter listener also implements this. The feeder
 * calls it after every fed frame (the merge gap is counted in samples); FakeSegmenter calls it from
 * `advanceSamples` so tests drive the gap without audio.
 */
interface StreamProgressListener {
    fun onAudio(samplesFed: Long)
}

/** Production port over the real segmenter. */
class AudioSegmenterPort(private val segmenter: Segmenter) : SegmenterPort {
    override fun feed(pcm: ShortArray) = segmenter.feed(pcm)
    override fun flush() = segmenter.flush()
    override fun reset() = segmenter.reset()
    override val calibrating: Boolean get() = segmenter.calibrating

    companion object {
        val FACTORY = SegmenterFactory { profile, vad, listener -> AudioSegmenterPort(Segmenter(profile, vad, listener)) }
    }
}
