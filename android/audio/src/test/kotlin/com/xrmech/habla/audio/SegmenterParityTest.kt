package com.xrmech.habla.audio

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource

/**
 * The Segmenter against the server's own `StreamingVADBuffer` (goldens from
 * `habla/scripts/make_audio_goldens.py`): the JIT probability vector replayed through
 * [ScriptedVad], the energy fallback run, and chunk-size invariance.
 */
class SegmenterParityTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("sessions")
    fun `replayed JIT probabilities give the server's segments`(name: String) {
        val s = GoldenFixtures.session(name)
        val actual = SegmenterRun.run(s.pcm, ScriptedVad(s.probs), chunk = 512)
        assertEquals(s.probs.size, actual.vadCalls, "$name: VAD calls (calibration frames must not reach the VAD)")
        SegmenterRun.assertMatches("$name/silero", s.silero, actual, vadAvgProbTolerance = 1e-9)
        if (name == "session_noise") assertTrue(actual.segments.isEmpty())
        if (name == "session_es") assertTrue(s.silero.splits >= 1 && actual.segments.any { it.pcm.size >= 937 * 512 })
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("sessions")
    fun `energy fallback gives the server's energy-mode segments`(name: String) {
        val s = GoldenFixtures.session(name)
        val actual = SegmenterRun.run(s.pcm, null, chunk = 512)
        // The server's fallback probability is float32 numpy arithmetic; the port's is Double
        // rounded to Float, so per-frame probabilities can differ by one float ulp (~6e-8).
        SegmenterRun.assertMatches("$name/energy", s.energy, actual, vadAvgProbTolerance = 1e-6)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("sessions")
    fun `callbacks do not depend on chunk size`(name: String) {
        val s = GoldenFixtures.session(name)
        val reference = SegmenterRun.run(s.pcm, ScriptedVad(s.probs), chunk = 512).signature()
        for (chunk in listOf(1, 511, 513, 4096, -1)) {
            val run = SegmenterRun.run(s.pcm, ScriptedVad(s.probs), chunk = chunk)
            assertEquals(reference, run.signature(), "$name: chunk size $chunk")
        }
    }

    @Test
    fun `a truncated probability vector is reported with the first diverging frame`() {
        val s = GoldenFixtures.session("session_es")
        val truncated = s.probs.copyOf(s.probs.size / 2)
        val actual = SegmenterRun.run(s.pcm, ScriptedVad(truncated), chunk = 512)
        val e = assertFailsWith<AssertionError> {
            SegmenterRun.assertMatches("session_es/truncated", s.silero, actual, vadAvgProbTolerance = 1e-9)
        }
        assertTrue("first diverging frame" in e.message.orEmpty(), e.message)
    }

    companion object {
        @JvmStatic
        fun sessions(): List<String> = GoldenFixtures.sessionNames
    }
}

/** Runs a Segmenter over a whole session (then flush, as the generator does) and records everything. */
internal class SegmenterRun private constructor() : SegmenterListener {
    val segments = ArrayList<SpeechSegment>()
    val partials = ArrayList<PartialSnapshot>()
    val calibrated = ArrayList<Double>()
    val events = ArrayList<String>()
    var vadCalls = 0

    override fun onCalibrated(noiseFloorRms: Double) {
        calibrated += noiseFloorRms
        events += "cal:$noiseFloorRms"
    }

    override fun onPartial(snapshot: PartialSnapshot) {
        partials += snapshot
        events += "partial:${snapshot.segmentId}:${snapshot.pcm.size}:${GoldenFixtures.sha256(snapshot.pcm)}"
    }

    override fun onSegment(segment: SpeechSegment) {
        segments += segment
        events += "segment:${segment.id}:${segment.startSample}:${segment.endSample}:" +
            "${GoldenFixtures.sha256(segment.pcm)}:${segment.rms}:${segment.vadAvgProb}"
    }

    fun signature(): List<String> = events

    companion object {
        /** `chunk = -1` feeds seeded random chunk sizes in 1..2000. */
        fun run(pcm: ShortArray, vad: VadModel?, chunk: Int): SegmenterRun {
            val rec = SegmenterRun()
            val seg = Segmenter(TestProfiles.shipped, vad, rec)
            val rnd = Random(20261015)
            var pos = 0
            while (pos < pcm.size) {
                val n = minOf(if (chunk > 0) chunk else rnd.nextInt(1, 2001), pcm.size - pos)
                seg.feed(pcm, pos, n)
                pos += n
            }
            seg.flush()
            rec.vadCalls = (vad as? ScriptedVad)?.calls ?: (vad as? CountingVad)?.calls ?: 0
            return rec
        }

        fun assertMatches(
            name: String,
            expected: GoldenFixtures.ExpectedRun,
            actual: SegmenterRun,
            vadAvgProbTolerance: Double,
        ) {
            val exp = expected.segments
            val act = actual.segments
            var maxProb = 0.0
            var maxRms = 0.0
            for (i in 0 until maxOf(exp.size, act.size)) {
                val e = exp.getOrNull(i)
                val a = act.getOrNull(i)
                val same = e != null && a != null && e.startSample == a.startSample && e.endSample == a.endSample &&
                    e.samples == a.pcm.size && e.pcmSha256 == GoldenFixtures.sha256(a.pcm) && e.id == a.id
                if (!same) {
                    val frame = listOfNotNull(e?.startSample, a?.startSample, e?.endSample, a?.endSample)
                        .let { if (e != null && a != null && e.startSample == a.startSample) listOf(minOf(e.endSample, a.endSample)) else it }
                        .min() / 512
                    throw AssertionError(
                        "$name: segment #${i + 1} differs, first diverging frame $frame: expected " +
                            (e?.let { "[${it.startSample}, ${it.endSample}) ${it.samples} samples" } ?: "none") +
                            ", actual " + (a?.let { "[${it.startSample}, ${it.endSample}) ${it.pcm.size} samples" } ?: "none") +
                            " (${exp.size} expected, ${act.size} actual segments)",
                    )
                }
                e!!
                a!!
                maxProb = maxOf(maxProb, abs(e.vadAvgProb - a.vadAvgProb))
                maxRms = maxOf(maxRms, abs(e.rms - a.rms) / e.rms)
                assertTrue(abs(e.vadAvgProb - a.vadAvgProb) <= vadAvgProbTolerance,
                    "$name: segment ${e.id} vadAvgProb ${a.vadAvgProb} vs server ${e.vadAvgProb}")
                assertTrue(abs(e.rms - a.rms) <= 1e-4 * e.rms, "$name: segment ${e.id} rms ${a.rms} vs server ${e.rms}")
            }
            assertEquals(expected.partials.map { it.segmentId to it.samples },
                actual.partials.map { it.segmentId to it.pcm.size }, "$name: partial snapshots (segment id, samples)")
            val floor = expected.noiseFloorRms
            println(
                "parity $name: ${act.size} segments, ${actual.partials.size} partials, max|dVadAvgProb|=$maxProb, " +
                    "max rel dRms=$maxRms, noise floor ${actual.calibrated.firstOrNull()} vs server $floor",
            )
            if (floor == null) {
                assertTrue(actual.calibrated.isEmpty(), "$name: unexpected calibration")
            } else {
                assertEquals(1, actual.calibrated.size, "$name: calibrations")
                assertTrue(abs(actual.calibrated[0] - floor) <= 1e-4 * floor,
                    "$name: noise floor ${actual.calibrated[0]} vs server $floor")
            }
        }
    }
}

/** Delegating VadModel that counts calls (end-to-end runs with the real SileroVad). */
internal class CountingVad(private val inner: VadModel) : VadModel by inner {
    var calls = 0
        private set

    override fun speechProbability(frame: FloatArray): Float {
        calls++
        return inner.speechProbability(frame)
    }
}
