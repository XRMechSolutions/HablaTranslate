package com.xrmech.habla.asr.internal

import com.xrmech.habla.asr.AsrPass
import com.xrmech.habla.asr.TestAssets
import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.WhisperCppProfile
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import org.junit.jupiter.api.assertThrows

class WhisperParamsBuilderTest {
    private val profile = TestAssets.profile

    private fun seconds(s: Double) = (s * 16_000).toInt()

    @Test
    fun `FINAL is beam search with whisper beam size and best_of equal to beam`() {
        val p = WhisperParamsBuilder.build(AsrPass.FINAL, "es", seconds(5.0), profile, 4)
        assertEquals(WhisperNative.STRATEGY_BEAM_SEARCH, p.strategy)
        assertEquals(5, p.beamSize)
        assertEquals(5, p.bestOf)
        assertEquals(0, p.audioCtx)
        assertFalse(p.singleSegment)
        assertEquals(0f, p.temperature)
        assertEquals(0f, p.temperatureInc)
        assertEquals(0f, p.entropyThold)
        assertEquals(-1f, p.logprobThold)
        assertEquals(1f, p.noSpeechThold)
        assertEquals(4, p.threads)
        assertEquals("es", p.language)
    }

    @Test
    fun `final beam override wins and best_of follows it`() {
        val p = WhisperParamsBuilder.build(AsrPass.FINAL, "en", 16_000, TestAssets.profileWith(finalBeam = 3), 2)
        assertEquals(3, p.beamSize)
        assertEquals(3, p.bestOf)
        assertEquals(2, p.threads)
    }

    @Test
    fun `PARTIAL parity level equals FINAL`() {
        val prof = TestAssets.profileWith(level = "parity")
        val partial = WhisperParamsBuilder.build(AsrPass.PARTIAL, "es", seconds(3.0), prof, 4)
        val final = WhisperParamsBuilder.build(AsrPass.FINAL, "es", seconds(3.0), prof, 4)
        assertEquals(final, partial)
    }

    @Test
    fun `PARTIAL greedy_full_ctx is greedy single segment with full context`() {
        val prof = TestAssets.profileWith(level = "greedy_full_ctx")
        val p = WhisperParamsBuilder.build(AsrPass.PARTIAL, "es", seconds(3.0), prof, 4)
        assertEquals(WhisperNative.STRATEGY_GREEDY, p.strategy)
        assertEquals(-1, p.beamSize)
        assertEquals(1, p.bestOf)
        assertTrue(p.singleSegment)
        assertEquals(0, p.audioCtx)
        assertEquals(0f, p.temperatureInc)
        assertEquals(1f, p.noSpeechThold)
    }

    @Test
    fun `PARTIAL greedy_scaled_ctx scales audio_ctx with the clip`() {
        val prof = TestAssets.profileWith(level = "greedy_scaled_ctx")
        val p = WhisperParamsBuilder.build(AsrPass.PARTIAL, "en", seconds(2.0), prof, 4)
        assertEquals(WhisperNative.STRATEGY_GREEDY, p.strategy)
        assertEquals(-1, p.beamSize)
        assertEquals(1, p.bestOf)
        assertTrue(p.singleSegment)
        assertEquals(164, p.audioCtx)
        assertEquals("en", p.language)
    }

    @Test
    fun `scaled audio_ctx formula`() {
        assertEquals(128, WhisperParamsBuilder.scaledAudioCtx(0, 64, 128))
        assertEquals(128, WhisperParamsBuilder.scaledAudioCtx(seconds(1.0), 64, 128))
        assertEquals(164, WhisperParamsBuilder.scaledAudioCtx(seconds(2.0), 64, 128))
        assertEquals(1500, WhisperParamsBuilder.scaledAudioCtx(seconds(29.0), 64, 128))
        assertEquals(1500, WhisperParamsBuilder.scaledAudioCtx(seconds(40.0), 64, 128))
        // ceil: one extra sample opens a new encoder frame.
        assertEquals(51 + 64, WhisperParamsBuilder.scaledAudioCtx(seconds(1.0) + 1, 64, 0))
    }

    @Test
    fun `shipped profile validates`() {
        assertDoesNotThrow { WhisperParamsBuilder.validate(profile) }
    }

    private fun invalid(expected: String, mutate: (AsrProfile) -> AsrProfile) {
        val e = assertThrows<IllegalArgumentException> { WhisperParamsBuilder.validate(mutate(profile)) }
        assertEquals(expected, e.message)
    }

    private fun AsrProfile.cpp(f: (WhisperCppProfile) -> WhisperCppProfile): AsrProfile = copy(whisperCpp = f(whisperCpp))

    @Test
    fun `validation limits name the JSON key`() {
        invalid("whisper_cpp.threads must be 1..16, was 0") { it.cpp { c -> c.copy(threads = 0) } }
        invalid("whisper_cpp.threads must be 1..16, was 17") { it.cpp { c -> c.copy(threads = 17) } }
        invalid("whisper_cpp.final.beam_size must be 1..8, was 9") {
            it.cpp { c -> c.copy(final = c.final.copy(beamSize = 9)) }
        }
        invalid("whisper_cpp.final.beam_size must be 1..8, was 0") {
            it.cpp { c -> c.copy(final = c.final.copy(beamSize = 0)) }
        }
        invalid("whisper.beam_size must be 1..8, was 10") { it.copy(whisper = it.whisper.copy(beamSize = 10)) }
        invalid("whisper_cpp.final.temperature must be 0.0..1.0, was 1.5") {
            it.cpp { c -> c.copy(final = c.final.copy(temperature = 1.5)) }
        }
        invalid("whisper_cpp.final.temperature_inc must be 0.0..1.0, was -0.2") {
            it.cpp { c -> c.copy(final = c.final.copy(temperatureInc = -0.2)) }
        }
        invalid("whisper_cpp.partial.level must be one of [parity, greedy_full_ctx, greedy_scaled_ctx], was 'fast'") {
            it.cpp { c -> c.copy(partial = c.partial.copy(level = "fast")) }
        }
        invalid("whisper_cpp.partial.audio_ctx_pad_frames must be 0..1500, was -1") {
            it.cpp { c -> c.copy(partial = c.partial.copy(audioCtxPadFrames = -1)) }
        }
        invalid("whisper_cpp.partial.audio_ctx_pad_frames must be 0..1500, was 1501") {
            it.cpp { c -> c.copy(partial = c.partial.copy(audioCtxPadFrames = 1501)) }
        }
        invalid("whisper_cpp.partial.audio_ctx_min_frames must be 1..1500, was 0") {
            it.cpp { c -> c.copy(partial = c.partial.copy(audioCtxMinFrames = 0)) }
        }
        invalid("whisper_cpp.partial.audio_ctx_min_frames must be 1..1500, was 1501") {
            it.cpp { c -> c.copy(partial = c.partial.copy(audioCtxMinFrames = 1501)) }
        }
    }

    @Test
    fun `validation accepts the boundaries`() {
        assertDoesNotThrow {
            WhisperParamsBuilder.validate(
                profile.cpp { c ->
                    c.copy(
                        threads = 16,
                        final = c.final.copy(beamSize = 8, temperature = 1.0, temperatureInc = 0.0),
                        partial = c.partial.copy(audioCtxPadFrames = 0, audioCtxMinFrames = 1500),
                    )
                },
            )
        }
        assertDoesNotThrow { WhisperParamsBuilder.validate(TestAssets.profileWith(finalBeam = 1)) }
    }
}
