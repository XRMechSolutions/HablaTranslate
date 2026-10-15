package com.xrmech.habla.audio

import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource

/**
 * PartialPrep against the server's `prepare_partial_pcm`, FinalPrep against the server's ffmpeg
 * command (8.0.1), and the FinalPrep stages against ffmpeg's f64 highpass/lowpass output and the
 * R3 Python dynaudnorm reference. Gates: design.md @ Test plan.
 */
class PrepParityTest {
    private val profile = TestProfiles.shipped

    private class Diff(val max: Double, val firstIndex: Int, val nonZero: Int)

    private fun diff(expected: ShortArray, actual: ShortArray): Diff {
        var max = 0.0
        var first = -1
        var count = 0
        for (i in expected.indices) {
            val d = abs(expected[i] - actual[i]).toDouble()
            if (d > 0) {
                count++
                if (first < 0) first = i
            }
            if (d > max) max = d
        }
        return Diff(max, first, count)
    }

    private fun diff(expected: DoubleArray, actual: DoubleArray): Diff {
        var max = 0.0
        var first = -1
        var count = 0
        for (i in expected.indices) {
            val d = abs(expected[i] - actual[i])
            if (d > 0) {
                count++
                if (first < 0) first = i
            }
            if (d > max) max = d
        }
        return Diff(max, first, count)
    }

    /** dB RMS of each 2400-sample (150 ms) frame. */
    private fun envelopeDb(x: ShortArray): DoubleArray {
        val k = x.size / 2400
        return DoubleArray(k) { f ->
            var s = 0.0
            for (i in f * 2400 until (f + 1) * 2400) s += x[i].toDouble() * x[i]
            20 * log10(sqrt(s / 2400) / 32768.0 + 1e-12)
        }
    }

    private fun rms(x: ShortArray): Double {
        var s = 0.0
        for (v in x) s += v.toDouble() * v
        return sqrt(s / x.size)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("partialClips")
    fun `PartialPrep matches the server snippet`(key: String) {
        val clip = GoldenFixtures.partialPrepClips.first { it.key == key }
        val expected = GoldenFixtures.shorts(clip.file("output"))
        val actual = PartialPrep.process(clip.input, profile)
        assertEquals(expected.size, actual.size, key)
        val d = diff(expected, actual)
        println("parity partial $key: max ${d.max} LSB, ${d.nonZero} samples differ")
        assertTrue(d.max <= 1.0, "$key: max diff ${d.max} LSB, first diverging sample ${d.firstIndex}")
        // The 1 LSB gate alone cannot tell rounding from the server's truncation toward zero
        // (rounding moves about half the samples by 1 LSB); measured diffs are 0, so also require
        // >= 99.9% identical samples, as for FinalPrep.
        assertTrue(d.nonZero <= expected.size / 1000, "$key: ${d.nonZero} of ${expected.size} samples differ, first ${d.firstIndex}")
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("stageClips")
    fun `highpass and lowpass match ffmpeg in f64`(key: String) {
        val clip = GoldenFixtures.finalPrepClips.first { it.key == key }
        val expected = GoldenFixtures.doubles(clip.file("hplp"))
        val x = DoubleArray(clip.input.size) { clip.input[it] / 32768.0 }
        Biquads.highpass80(x)
        Biquads.lowpass7500(x)
        val d = diff(expected, x)
        println("parity hplp $key: max abs ${d.max}")
        assertTrue(d.max <= 1e-12, "$key: max diff ${d.max}, first diverging sample ${d.firstIndex}")
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("stageClips")
    fun `dynaudnorm matches the Python reference`(key: String) {
        val clip = GoldenFixtures.finalPrepClips.first { it.key == key }
        val input = GoldenFixtures.doubles(clip.file("hplp"))
        val expected = GoldenFixtures.doubles(clip.file("dyn"))
        val dn = profile.finalPrep.dynaudnorm
        val actual = Dynaudnorm.apply(input, 2400, dn.gaussSize, dn.peak, dn.maxGain)
        val d = diff(expected, actual)
        println("parity dynaudnorm $key: max abs ${d.max}")
        assertTrue(d.max <= 1e-12, "$key: max diff ${d.max}, first diverging sample ${d.firstIndex}")
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("finalClips")
    fun `FinalPrep matches ffmpeg`(key: String) {
        val clip = GoldenFixtures.finalPrepClips.first { it.key == key }
        val expected = GoldenFixtures.shorts(clip.file("final"))
        val actual = FinalPrep.process(clip.input, profile)
        assertEquals(expected.size, actual.size, "$key: length")
        val d = diff(expected, actual)
        val identical = 1.0 - d.nonZero.toDouble() / expected.size
        val ea = envelopeDb(expected)
        val eb = envelopeDb(actual)
        var env = 0.0
        for (i in ea.indices) if (ea[i] > -60.0) env = maxOf(env, abs(ea[i] - eb[i]))
        println("parity final $key: max ${d.max} LSB, identical ${"%.5f".format(identical)}, envelope max ${"%.4f".format(env)} dB")
        if (key == "hot_synth") {
            assertTrue(d.max / 32768.0 <= 1.5e-3, "$key: max diff ${d.max / 32768.0} FS, first diverging sample ${d.firstIndex}")
        } else {
            assertTrue(d.max <= 2.0, "$key: max diff ${d.max} LSB, first diverging sample ${d.firstIndex}")
            assertTrue(identical >= 0.999, "$key: only $identical identical, first diverging sample ${d.firstIndex}")
        }
        assertTrue(env <= 0.1, "$key: per-150 ms envelope differs by $env dB")
    }

    @Test
    fun `short clips keep their level like ffmpeg (D029)`() {
        for (key in listOf("cut0.8s_x1", "cut1.5s_x1", "cut1.95s_x1", "cut0.8s_x0.1", "cut1.5s_x0.1", "cut1.95s_x0.1")) {
            val clip = GoldenFixtures.finalPrepClips.first { it.key == key }
            val ratio = rms(FinalPrep.process(clip.input, profile)) / rms(clip.input)
            assertTrue(ratio in 0.9..1.1, "$key: output/input RMS ratio $ratio, expected about 1 (no dynaudnorm boost)")
        }
        val long = GoldenFixtures.finalPrepClips.first { it.key == "cut8.0s_x0.1" }
        assertTrue(rms(FinalPrep.process(long.input, profile)) / rms(long.input) > 3.0, "8 s quiet clip is boosted")
    }

    @Test
    fun `owner recordings match ffmpeg (skipped unless HABLA_OWNER_CLIPS is set and ffmpeg is on PATH)`() {
        val dir = System.getenv("HABLA_OWNER_CLIPS")?.let(::File)
        assumeTrue(dir != null && dir.isDirectory, "HABLA_OWNER_CLIPS not set")
        val clips = dir!!.listFiles { f -> f.name.endsWith(".wav", ignoreCase = true) }.orEmpty().sortedBy { it.name }
        assumeTrue(clips.isNotEmpty(), "no .wav files in $dir")
        val filter = "highpass=f=${profile.finalPrep.highpassHz},lowpass=f=${profile.finalPrep.lowpassHz}," +
            "dynaudnorm=f=150:g=15:p=0.95:m=10"
        for (clip in clips) {
            val input = GoldenFixtures.readWav(clip)
            val out = File.createTempFile("habla_owner_", ".wav")
            try {
                val proc = try {
                    ProcessBuilder("ffmpeg", "-v", "error", "-y", "-i", clip.absolutePath, "-af", filter,
                        "-ar", "16000", "-ac", "1", "-f", "wav", out.absolutePath).redirectErrorStream(true).start()
                } catch (e: java.io.IOException) {
                    null
                }
                assumeTrue(proc != null, "ffmpeg not on PATH")
                assertTrue(proc!!.waitFor(120, TimeUnit.SECONDS) && proc.exitValue() == 0, "ffmpeg failed on ${clip.name}")
                val d = diff(GoldenFixtures.readWav(out), FinalPrep.process(input, profile))
                println("owner final ${clip.name}: max ${d.max} LSB, ${d.nonZero} samples differ")
                assertTrue(d.max <= 2.0 && d.nonZero <= input.size / 1000, "${clip.name}: max ${d.max} LSB, first ${d.firstIndex}")
            } finally {
                out.delete()
            }
        }
    }

    companion object {
        @JvmStatic
        fun partialClips(): List<String> = GoldenFixtures.partialPrepClips.map { it.key }

        @JvmStatic
        fun finalClips(): List<String> = GoldenFixtures.finalPrepClips.map { it.key }

        @JvmStatic
        fun stageClips(): List<String> = GoldenFixtures.finalPrepClips.filter { it.has("hplp") }.map { it.key }
    }
}
