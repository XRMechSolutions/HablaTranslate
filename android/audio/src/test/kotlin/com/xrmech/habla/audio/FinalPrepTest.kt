package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.DynaudnormProfile
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test

class FinalPrepTest {
    private val profile: AsrProfile = TestProfiles.shipped

    private fun peak(x: ShortArray) = x.maxOf { abs(it.toInt()) }

    /** Speech-like: a 300 Hz tone gated at 4 Hz, scaled to the given peak. */
    private fun speechLike(n: Int, peak: Int) = ShortArray(n) {
        val env = 0.5 + 0.5 * sin(2 * PI * 4.0 * it / 16000.0)
        Math.rint(peak * env * sin(2 * PI * 300.0 * it / 16000.0)).toInt().toShort()
    }

    @Test
    fun `biquad tables equal the design table exactly`() {
        assertContentEquals(
            doubleArrayOf(0.9780272696186294, -1.9560545392372588, 0.9780272696186294, -1.9555718227233034, 0.9565372557512138),
            Biquads.HP80,
        )
        assertContentEquals(
            doubleArrayOf(0.8703148444087448, 1.7406296888174897, 0.8703148444087448, 1.7237446121141573, 0.7575147655208223),
            Biquads.LP7500,
        )
    }

    @Test
    fun `erf matches scipy within one ulp`() {
        // scipy.special.erf (cephes) values, printed with repr by the venv's scipy 1.18.1
        val ref = mapOf(
            0.0 to 0.0,
            0.5 to 0.5204998778130465,
            1.0 to 0.8427007929497148,
            2.0 to 0.9953222650189527,
            6.0 to 1.0,
            -1.0 to -0.8427007929497148,
            0.1 to 0.1124629160182849,
            0.9 to 0.7969082124228322,
            1.5 to 0.9661051464753108,
            3.0 to 0.9999779095030014,
            0.08862269254527579 to 0.09973881631072791,
        )
        for ((x, expected) in ref) {
            val actual = Erf.erf(x)
            assertTrue(abs(actual - expected) <= Math.ulp(expected), "erf($x) = $actual, scipy $expected")
        }
        assertEquals(1.0, Erf.erf(Double.POSITIVE_INFINITY))
        assertEquals(-1.0, Erf.erf(Double.NEGATIVE_INFINITY))
        assertTrue(Erf.erf(Double.NaN).isNaN())
    }

    @Test
    fun `empty, all-zero and odd lengths`() {
        assertEquals(0, FinalPrep.process(ShortArray(0), profile).size)
        assertContentEquals(ShortArray(16000), FinalPrep.process(ShortArray(16000), profile))
        val odd = speechLike(2400 * 3 + 700, 2000)
        assertEquals(odd.size, FinalPrep.process(odd, profile).size)
    }

    @Test
    fun `short clip keeps its level and a long clip is boosted (D029)`() {
        val outShort = FinalPrep.process(speechLike(12800, 578), profile)
        assertTrue(abs(peak(outShort) - 578) <= 578 * 0.01, "0.8 s peak ${peak(outShort)}")
        val outLong = FinalPrep.process(speechLike(128000, 578), profile)
        assertTrue(peak(outLong) > 5000, "8 s peak ${peak(outLong)}")
    }

    @Test
    fun `input array is not modified`() {
        val x = speechLike(16000, 3000)
        val copy = x.copyOf()
        FinalPrep.process(x, profile)
        assertContentEquals(copy, x)
    }

    @Test
    fun `invalid profile fields name the key`() {
        fun dn(f: (DynaudnormProfile) -> DynaudnormProfile) =
            profile.copy(finalPrep = profile.finalPrep.copy(dynaudnorm = f(profile.finalPrep.dynaudnorm)))
        val cases = listOf(
            "final_prep.highpass_hz" to profile.copy(finalPrep = profile.finalPrep.copy(highpassHz = 100)),
            "final_prep.lowpass_hz" to profile.copy(finalPrep = profile.finalPrep.copy(lowpassHz = 7000)),
            "audio.sample_rate" to profile.copy(audio = profile.audio.copy(sampleRate = 48000)),
            "dynaudnorm.frame_ms" to dn { it.copy(frameMs = 0) },
            "dynaudnorm.gauss_size" to dn { it.copy(gaussSize = 14) },
            "dynaudnorm.gauss_size" to dn { it.copy(gaussSize = 303) },
            "dynaudnorm.gauss_size" to dn { it.copy(gaussSize = 1) },
            "dynaudnorm.peak" to dn { it.copy(peak = 0.0) },
            "dynaudnorm.peak" to dn { it.copy(peak = 1.5) },
            "dynaudnorm.max_gain" to dn { it.copy(maxGain = 0.5) },
            "dynaudnorm.max_gain" to dn { it.copy(maxGain = 101.0) },
        )
        for ((key, p) in cases) {
            val e = assertFailsWith<IllegalArgumentException>(key) { FinalPrep.process(ShortArray(10), p) }
            assertTrue(key in e.message.orEmpty(), "message '${e.message}' should name $key")
        }
    }
}
