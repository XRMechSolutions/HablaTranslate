package com.xrmech.habla.audio

import java.io.File
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource

/**
 * SileroVad (ONNX Runtime desktop) against the server's Silero JIT model and the server's
 * segment boundaries (R2 Q3: max |dp| about 1e-6 when the 64-sample context is reproduced).
 */
class SileroVadParityTest {
    private val calibrationFrames = 62

    /** Fresh SileroVad over the post-calibration frames, as the server's model sees them. */
    private fun onnxProbs(pcm: ShortArray, frames: Int, contextEnabled: Boolean = true): FloatArray =
        SileroVad(SileroVad.bundledModelBytes()).use { vad ->
            vad.contextEnabledForTest = contextEnabled
            FloatArray(frames) { k ->
                val start = (calibrationFrames + k) * 512
                vad.speechProbability(pcm.copyOfRange(start, start + 512).toFloatSamples())
            }
        }

    private fun assertClose(name: String, jit: FloatArray, onnx: FloatArray) {
        var max = 0.0
        var at = -1
        for (i in jit.indices) {
            assertTrue(onnx[i] in 0f..1f, "$name: probability ${onnx[i]} at VAD call $i outside [0, 1]")
            val d = abs(jit[i].toDouble() - onnx[i].toDouble())
            if (d > max) {
                max = d
                at = i
            }
        }
        println("parity $name: ${jit.size} frames, max|ONNX - JIT| = $max at VAD call $at")
        assertTrue(max < 1e-4, "$name: max |ONNX - JIT| = $max at VAD call $at")
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("sessions")
    fun `ONNX probabilities match the JIT model`(name: String) {
        val s = GoldenFixtures.session(name)
        assertClose(name, s.probs, onnxProbs(s.pcm, s.probs.size))
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("sessions")
    fun `Segmenter with SileroVad reproduces the server's boundaries`(name: String) {
        val s = GoldenFixtures.session(name)
        SileroVad(SileroVad.bundledModelBytes()).use { silero ->
            val vad = CountingVad(silero)
            val run = SegmenterRun.run(s.pcm, vad, chunk = 1600)
            assertEquals(s.probs.size, vad.calls, "$name: VAD calls")
            SegmenterRun.assertMatches("$name/onnx-end-to-end", s.silero, run, vadAvgProbTolerance = 1e-4)
        }
    }

    @Test
    fun `without the 64-sample context the model misses speech`() {
        val s = GoldenFixtures.session("session_es")
        val probs = onnxProbs(s.pcm, s.probs.size, contextEnabled = false)
        val speech = s.probs.indices.filter { s.probs[it] > 0.35f }
        assertTrue(speech.size > 500, "fixture has ${speech.size} speech frames")
        val mean = speech.sumOf { probs[it].toDouble() } / speech.size
        println("parity no-context: mean probability over ${speech.size} JIT speech frames = $mean")
        assertTrue(mean < 0.05, "mean probability without context $mean")
    }

    @Test
    fun `local Piper fixtures match the JIT model (skipped when absent)`() {
        assumeTrue(GoldenFixtures.exists("local/manifest.json"), "no goldens/local (run make_audio_goldens.py --piper)")
        val sessions = Json.parseToJsonElement(GoldenFixtures.text("local/manifest.json")).jsonObject
            .getValue("sessions").jsonObject
        for ((name, v) in sessions) {
            val o = v as JsonObject
            val pcm = GoldenFixtures.shorts("local/" + o.getValue("pcm").jsonPrimitive.content)
            val jit = GoldenFixtures.floats("local/" + o.getValue("probs").jsonPrimitive.content)
            assertClose("piper/$name", jit, onnxProbs(pcm, jit.size))
            SileroVad(SileroVad.bundledModelBytes()).use { silero ->
                val run = SegmenterRun.run(pcm, silero, chunk = 512)
                val expected = o.getValue("segments").jsonArray.map {
                    val g = it.jsonObject
                    g.getValue("startSample").jsonPrimitive.int to g.getValue("endSample").jsonPrimitive.int
                }
                assertEquals(expected, run.segments.map { it.startSample.toInt() to it.endSample.toInt() }, "piper/$name boundaries")
            }
        }
    }

    @Test
    fun `owner recordings run cleanly (skipped unless HABLA_OWNER_CLIPS is set)`() {
        val dir = System.getenv("HABLA_OWNER_CLIPS")?.let(::File)
        assumeTrue(dir != null && dir.isDirectory, "HABLA_OWNER_CLIPS not set")
        val clips = dir!!.listFiles { f -> f.name.endsWith(".wav", ignoreCase = true) }.orEmpty().sortedBy { it.name }
        assumeTrue(clips.isNotEmpty(), "no .wav files in $dir")
        for (clip in clips) {
            val pcm = GoldenFixtures.readWav(clip)
            val frames = maxOf(0, pcm.size / 512 - calibrationFrames)
            val probs = onnxProbs(pcm, frames)
            assertTrue(probs.all { it in 0f..1f }, "${clip.name}: probabilities in [0, 1]")
            SileroVad(SileroVad.bundledModelBytes()).use { silero ->
                val run = SegmenterRun.run(pcm, silero, chunk = 1600)
                println("owner ${clip.name}: ${run.segments.size} segments")
            }
        }
    }

    companion object {
        @JvmStatic
        fun sessions(): List<String> = GoldenFixtures.sessionNames
    }
}
