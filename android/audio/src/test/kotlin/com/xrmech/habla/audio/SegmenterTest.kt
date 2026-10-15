package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test

class SegmenterTest {
    private val profile: AsrProfile = TestProfiles.shipped

    /** Records every callback in order. */
    private class Recorder : SegmenterListener {
        val events = ArrayList<String>()
        val calibrated = ArrayList<Double>()
        val partials = ArrayList<PartialSnapshot>()
        val segments = ArrayList<SpeechSegment>()
        val fallbacks = ArrayList<VadException>()
        var throwOnSegment = false

        override fun onCalibrated(noiseFloorRms: Double) {
            events += "calibrated"
            calibrated += noiseFloorRms
        }

        override fun onPartial(snapshot: PartialSnapshot) {
            events += "partial"
            partials += snapshot
        }

        override fun onSegment(segment: SpeechSegment) {
            events += "segment"
            segments += segment
            if (throwOnSegment) throw IllegalStateException("listener failure")
        }

        override fun onVadFallback(cause: VadException) {
            events += "fallback"
            fallbacks += cause
        }
    }

    /** A frame of alternating +-amplitude samples: RMS == amplitude exactly. */
    private fun frame(amplitude: Int) = ShortArray(512) { if (it % 2 == 0) amplitude.toShort() else (-amplitude).toShort() }

    private fun frames(n: Int, amplitude: Int = 120): ShortArray {
        val f = frame(amplitude)
        return ShortArray(n * 512) { f[it % 512] }
    }

    private fun probs(vararg runs: Pair<Int, Float>): FloatArray {
        val out = ArrayList<Float>()
        for ((n, p) in runs) repeat(n) { out += p }
        return out.toFloatArray()
    }

    private fun AsrProfile.withSegmenter(f: (com.xrmech.habla.shared.SegmenterProfile) -> com.xrmech.habla.shared.SegmenterProfile) =
        copy(segmenter = f(segmenter))

    @Test
    fun `frame arithmetic matches the server's integer division`() {
        val s = Segmenter(profile, ScriptedVad(FloatArray(0)), Recorder())
        assertEquals(9, s.preRingFrames)
        assertEquals(18, s.silenceFrames)
        assertEquals(6, s.minSpeechFrames)
        assertEquals(31, s.partialIntervalFrames)
        assertEquals(937, s.maxSegmentFrames)
        assertEquals(62, s.calibrationFrames)
        assertEquals(4800, s.tailSamples)
        assertEquals(484_352, s.initialCapacity)
        assertEquals(8_640_704, s.maxBufferSamples)
        assertEquals(484_352, s.speechCapacity)
    }

    @Test
    fun `calibration measures the noise floor without calling the VAD`() {
        val vad = ScriptedVad(FloatArray(0))
        val energy = EnergyVad(profile)
        val rec = Recorder()
        val s = Segmenter(profile, vad, rec)
        assertTrue(s.calibrating)
        assertEquals(500.0, s.noiseFloorRms)
        s.feed(frames(61))
        assertTrue(rec.calibrated.isEmpty())
        s.feed(frames(1))
        assertEquals(1, rec.calibrated.size)
        assertEquals(120.0, rec.calibrated[0], 120.0 * 1e-4)
        assertFalse(s.calibrating)
        assertEquals(0, vad.calls)
        s.feed(frames(1))
        assertEquals(1, vad.calls)

        val es = Segmenter(profile, energy, Recorder())
        es.feed(frames(62))
        assertEquals(360.0, energy.energyThreshold, 1e-9)
        assertTrue(es.usingEnergyFallback)
    }

    @Test
    fun `zero calibration duration runs the VAD from the first frame`() {
        val p = profile.copy(calibration = profile.calibration.copy(durationMs = 0))
        val vad = ScriptedVad(FloatArray(0))
        val rec = Recorder()
        val energy = EnergyVad(p)
        val s = Segmenter(p, vad, rec)
        assertFalse(s.calibrating)
        s.feed(frames(3))
        assertEquals(3, vad.calls)
        assertTrue(rec.calibrated.isEmpty())
        Segmenter(p, energy, Recorder()).feed(frames(70))
        assertEquals(500.0, energy.energyThreshold)
    }

    @Test
    fun `utterance segment carries ring, speech and trailing silence`() {
        val speech = 47
        val vad = ScriptedVad(probs(20 to 0.01f, speech to 0.9f, 18 to 0.01f))
        val rec = Recorder()
        val s = Segmenter(profile, vad, rec)
        s.feed(frames(62 + 20 + speech + 18))
        assertEquals(1, rec.segments.size)
        val seg = rec.segments[0]
        assertEquals((9 + speech + 18) * 512, seg.pcm.size)
        assertEquals(seg.pcm.size.toLong(), seg.endSample - seg.startSample)
        assertEquals((62L + 20 + speech + 18) * 512, seg.endSample)
        assertEquals(1, seg.id)
        assertEquals(0.9, seg.vadAvgProb, 1e-7)
        assertEquals(120.0, seg.rms, 1e-9)
        assertEquals(seg.pcm.size / 16000.0, seg.durationSec)
        assertEquals(1, s.segmentsEmitted)
        assertEquals(seg.durationSec, s.totalSpeechSeconds)
        assertFalse('[' in seg.toString(), "toString must not print the pcm array")
    }

    @Test
    fun `stale pre-speech ring is prepended to a quickly resumed segment`() {
        val vad = ScriptedVad(probs(20 to 0.01f, 10 to 0.9f, 18 to 0.01f, 10 to 0.9f, 18 to 0.01f))
        val rec = Recorder()
        Segmenter(profile, vad, rec).feed(frames(62 + 20 + 10 + 18 + 10 + 18))
        assertEquals(2, rec.segments.size)
        val second = rec.segments[1]
        assertEquals((9 + 10 + 18) * 512, second.pcm.size)
        assertTrue(second.endSample - second.startSample > second.pcm.size)
        assertEquals(rec.segments[0].startSample, second.startSample)
    }

    @Test
    fun `utterance shorter than min speech is discarded`() {
        val rec = Recorder()
        Segmenter(profile, ScriptedVad(probs(20 to 0f, 5 to 0.9f, 30 to 0f)), rec).feed(frames(62 + 55))
        assertTrue(rec.segments.isEmpty())
    }

    @Test
    fun `probability exactly at the threshold is not speech`() {
        val at = Recorder()
        Segmenter(profile, ScriptedVad(probs(10 to 0f, 20 to 0.35f, 20 to 0f)), at).feed(frames(62 + 50))
        assertTrue(at.segments.isEmpty())
        val above = Recorder()
        Segmenter(profile, ScriptedVad(probs(10 to 0f, 20 to Math.nextUp(0.35f), 20 to 0f)), above).feed(frames(62 + 50))
        assertEquals(1, above.segments.size)
    }

    @Test
    fun `max split emits the segment and seeds the next with a 4800-sample tail`() {
        val vad = ScriptedVad(probs(10 to 0f, 937 to 0.9f, 100 to 0.8f, 18 to 0f))
        val rec = Recorder()
        Segmenter(profile, vad, rec).feed(frames(62 + 10 + 937 + 100 + 18))
        assertEquals(2, rec.segments.size)
        val (first, second) = rec.segments
        assertEquals((9 + 937) * 512, first.pcm.size)
        assertEquals(first.endSample - 4800, second.startSample)
        assertEquals(4800 + (100 + 18) * 512, second.pcm.size)
        // speechFrames restarts at 9 while the probability sum restarts at 0 (server quirk)
        var sum = 0.0
        repeat(100) { sum += 0.8f.toDouble() }
        assertEquals(sum / (9 + 100), second.vadAvgProb, 1e-12)
        // Partials every 31 speech frames: 30 in the first segment (frames 31..930), then
        // framesSincePartial is NOT reset by the split (server quirk), so 937 - 930 = 7 carry over
        // and the next partial comes 24 speech frames after the split, not 31.
        val firstPartials = rec.partials.filter { it.segmentId == 1 }.map { it.pcm.size }
        assertEquals((1..30).map { (9 + 31 * it) * 512 }, firstPartials)
        val secondPartials = rec.partials.filter { it.segmentId == 2 }.map { it.pcm.size }
        assertEquals(listOf(24, 55, 86).map { 4800 + it * 512 }, secondPartials)
    }

    @Test
    fun `zero pre-speech padding gives an empty split tail`() {
        val p = profile.withSegmenter { it.copy(preSpeechPaddingMs = 0) }
        val rec = Recorder()
        val s = Segmenter(p, ScriptedVad(probs(10 to 0f, 937 to 0.9f, 100 to 0.8f, 18 to 0f)), rec)
        assertEquals(1, s.preRingFrames)
        assertEquals(0, s.tailSamples)
        s.feed(frames(62 + 10 + 937 + 100 + 18))
        assertEquals(2, rec.segments.size)
        assertEquals((1 + 937) * 512, rec.segments[0].pcm.size)
        assertEquals((100 + 18) * 512, rec.segments[1].pcm.size)
        assertEquals(0.8f.toDouble(), rec.segments[1].vadAvgProb, 1e-12)
    }

    @Test
    fun `flush emits only with enough speech and drops the pending partial frame`() {
        val enough = Recorder()
        val vad = ScriptedVad(probs(10 to 0f, 6 to 0.9f))
        val s = Segmenter(profile, vad, enough)
        s.feed(frames(62 + 16))
        s.feed(ShortArray(100))
        s.flush()
        assertEquals(1, enough.segments.size)
        assertEquals((9 + 6) * 512, enough.segments[0].pcm.size)
        val callsBefore = vad.calls
        s.feed(ShortArray(412))
        assertEquals(callsBefore, vad.calls, "pending samples must be dropped by flush")

        val few = Recorder()
        val s2 = Segmenter(profile, ScriptedVad(probs(10 to 0f, 5 to 0.9f)), few)
        s2.feed(frames(62 + 15))
        s2.flush()
        assertTrue(few.segments.isEmpty())
    }

    @Test
    fun `reset restarts calibration, ids and the VAD state`() {
        val once = probs(10 to 0f, 10 to 0.9f, 18 to 0f)
        val vad = ScriptedVad(once + once)
        val rec = Recorder()
        val s = Segmenter(profile, vad, rec)
        s.feed(frames(62 + 38))
        s.feed(ShortArray(100))
        assertEquals(1, rec.segments.size)
        s.reset()
        assertEquals(1, vad.resets)
        assertTrue(s.calibrating)
        assertEquals(0, s.segmentsEmitted)
        assertEquals(0.0, s.totalSpeechSeconds)
        val callsAtReset = vad.calls
        s.feed(frames(62))
        assertEquals(2, rec.calibrated.size)
        assertEquals(callsAtReset, vad.calls, "recalibration frames never reach the VAD")
        s.feed(frames(38))
        assertEquals(2, rec.segments.size)
        val again = rec.segments[1]
        assertEquals(1, again.id)
        // positions restart at 0: the ring holds frames 63..71 of the new session
        assertEquals(63L * 512, again.startSample)
        assertEquals(rec.segments[0].startSample, again.startSample)
        assertEquals(rec.segments[0].endSample, again.endSample)
    }

    @Test
    fun `mid-session VAD failure falls back to energy on the same frame`() {
        val vad = ScriptedVad(probs(200 to 0f), throwAtCall = 100)
        val rec = Recorder()
        val s = Segmenter(profile, vad, rec)
        s.feed(frames(62 + 100, amplitude = 50))
        assertFalse(s.usingEnergyFallback)
        s.feed(frames(1, amplitude = 5000))
        assertTrue(s.usingEnergyFallback)
        assertEquals(1, rec.fallbacks.size)
        assertEquals(101, vad.calls)
        s.feed(frames(9, amplitude = 5000))
        s.feed(frames(18, amplitude = 50))
        assertEquals(1, rec.segments.size, "frame 100 and the following loud frames count as speech")
        assertEquals((9 + 10 + 18) * 512, rec.segments[0].pcm.size)
        assertEquals("fallback", rec.events[1])
        s.reset()
        assertTrue(s.usingEnergyFallback)
        assertEquals(0, vad.resets)
    }

    @Test
    fun `listener exception propagates and the segmenter stays usable`() {
        val vad = ScriptedVad(probs(10 to 0f, 10 to 0.9f, 18 to 0f, 30 to 0f, 10 to 0.9f, 18 to 0f))
        val rec = Recorder().apply { throwOnSegment = true }
        val s = Segmenter(profile, vad, rec)
        assertFailsWith<IllegalStateException> { s.feed(frames(62 + 38 + 20)) }
        val callsAtThrow = vad.calls
        assertEquals(10 + 10 + 18, callsAtThrow, "the rest of the chunk is dropped")
        rec.throwOnSegment = false
        s.feed(frames(30 + 10 + 18 - 0))
        assertEquals(2, rec.segments.size)
        assertEquals(2, rec.segments[1].id)
    }

    @Test
    fun `sparse speech grows the buffer and it shrinks back after emit`() {
        val pattern = ArrayList<Float>()
        repeat(937) {
            pattern += 0.9f
            repeat(17) { pattern += 0f }
        }
        val vad = ScriptedVad(pattern.toFloatArray())
        val rec = Recorder()
        val s = Segmenter(profile, vad, rec)
        s.feed(frames(62))
        val chunk = frames(18)
        var maxCap = 0
        repeat(937) {
            s.feed(chunk)
            maxCap = maxOf(maxCap, s.speechCapacity)
        }
        assertEquals(1, rec.segments.size)
        assertTrue(maxCap > 484_352, "capacity grew to $maxCap")
        assertTrue(rec.segments[0].pcm.size > 484_352)
        assertEquals(484_352, s.speechCapacity)
    }

    @Test
    fun `invalid profile values name the key`() {
        val cases = listOf(
            "audio.sample_rate" to profile.copy(audio = profile.audio.copy(sampleRate = 8000)),
            "vad.frame_samples" to profile.copy(vad = profile.vad.copy(frameSamples = 256)),
            "vad.frame_ms" to profile.copy(vad = profile.vad.copy(frameMs = 30)),
            "segmenter.silence_duration_ms" to profile.withSegmenter { it.copy(silenceDurationMs = -1) },
            "segmenter.min_speech_ms" to profile.withSegmenter { it.copy(minSpeechMs = -1) },
            "segmenter.pre_speech_padding_ms" to profile.withSegmenter { it.copy(preSpeechPaddingMs = -1) },
            "segmenter.partial_interval_ms" to profile.withSegmenter { it.copy(partialIntervalMs = -1) },
            "calibration.duration_ms" to profile.copy(calibration = profile.calibration.copy(durationMs = -1)),
            "segmenter.max_segment_seconds" to profile.withSegmenter { it.copy(maxSegmentSeconds = 0.0) },
            "segmenter.max_segment_seconds" to profile.withSegmenter { it.copy(maxSegmentSeconds = 0.01) },
        )
        for ((key, p) in cases) {
            val e = assertFailsWith<IllegalArgumentException>(key) { Segmenter(p, null, Recorder()) }
            assertTrue(key in e.message.orEmpty(), "message '${e.message}' should name $key")
        }
    }

    @Test
    fun `bad offset or length consumes nothing`() {
        val vad = ScriptedVad(FloatArray(0))
        val rec = Recorder()
        val s = Segmenter(profile, vad, rec)
        val pcm = frames(70)
        assertFailsWith<IndexOutOfBoundsException> { s.feed(pcm, -1, 10) }
        assertFailsWith<IndexOutOfBoundsException> { s.feed(pcm, 10, pcm.size) }
        assertFailsWith<IndexOutOfBoundsException> { s.feed(pcm, 0, -1) }
        assertTrue(rec.events.isEmpty())
        s.feed(pcm, 0, 0)
        s.feed(pcm, 512, 62 * 512)
        assertEquals(1, rec.calibrated.size)
        assertEquals(0, vad.calls)
    }

    @Test
    fun `null vad uses the energy fallback with the calibrated threshold`() {
        val rec = Recorder()
        val s = Segmenter(profile, null, rec)
        assertTrue(s.usingEnergyFallback)
        s.feed(frames(62, amplitude = 50))
        s.feed(frames(10, amplitude = 50))
        s.feed(frames(10, amplitude = 400))
        s.feed(frames(18, amplitude = 50))
        assertEquals(1, rec.segments.size)
        assertEquals((9 + 10 + 18) * 512, rec.segments[0].pcm.size)
    }
}
