package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test

class PartialPrepTest {
    private val profile: AsrProfile = TestProfiles.shipped

    private fun rms(x: ShortArray): Double {
        var s = 0.0
        for (v in x) s += v.toDouble() * v
        return sqrt(s / x.size)
    }

    private fun tone(rms: Double, hz: Double) =
        ShortArray(16000) { Math.rint(rms * sqrt(2.0) * sin(2 * PI * hz * it / 16000.0)).toInt().toShort() }

    @Test
    fun `SOS table equals the design table exactly`() {
        val expected = arrayOf(
            doubleArrayOf(0.7420434859024201, 1.4840869718048402, 0.7420434859024201, 1.6612958431214457, 0.6940169382407124),
            doubleArrayOf(1.0, 2.0, 1.0, 1.8255448495100886, 0.8613419770425558),
            doubleArrayOf(1.0, -2.0, 1.0, -1.9425131629847214, 0.9434772385780453),
            doubleArrayOf(1.0, -2.0, 1.0, -1.9753201768232749, 0.9762962394835808),
        )
        assertEquals(expected.size, PartialPrep.SOS.size)
        for (i in expected.indices) assertContentEquals(expected[i], PartialPrep.SOS[i])
    }

    @Test
    fun `empty and silent inputs`() {
        assertEquals(0, PartialPrep.process(ShortArray(0), profile).size)
        assertContentEquals(ShortArray(16000), PartialPrep.process(ShortArray(16000), profile))
    }

    @Test
    fun `quiet tone is boosted by the capped gain`() {
        val out = PartialPrep.process(tone(300.0, 1000.0), profile)
        assertEquals(16000, out.size)
        val r = rms(out)
        assertTrue(r in 2900.0 * 0.95..2900.0 * 1.05, "rms $r")
    }

    @Test
    fun `full-scale square wave is attenuated and stays in range`() {
        val sq = ShortArray(16000) { if ((it / 40) % 2 == 0) 32767 else -32768 }
        val out = PartialPrep.process(sq, profile)
        assertTrue(rms(out) < rms(sq), "gain < 1 expected")
        assertTrue(rms(out) in 2900.0..3001.0, "rms ${rms(out)} is the 3000 target minus clipping")
    }

    @Test
    fun `near-silent input gets no gain`() {
        val out = PartialPrep.process(tone(5.0, 500.0), profile)
        assertTrue(rms(out) < 10.0, "rms ${rms(out)}")
    }

    @Test
    fun `input array is not modified`() {
        val x = ShortArray(4000) { ((it * 37) % 2000 - 1000).toShort() }
        val copy = x.copyOf()
        PartialPrep.process(x, profile)
        assertContentEquals(copy, x)
    }

    @Test
    fun `changed band names the regeneration command`() {
        val p = profile.copy(partialPrep = profile.partialPrep.copy(bandpassHz = listOf(100, 7500)))
        val e = assertFailsWith<IllegalArgumentException> { PartialPrep.process(ShortArray(10), p) }
        assertTrue("--print-sos" in e.message.orEmpty())
        assertTrue("partial_prep.bandpass_hz" in e.message.orEmpty())
        val o = profile.copy(partialPrep = profile.partialPrep.copy(butterworthOrder = 5))
        assertFailsWith<IllegalArgumentException> { PartialPrep.process(ShortArray(10), o) }
    }
}
