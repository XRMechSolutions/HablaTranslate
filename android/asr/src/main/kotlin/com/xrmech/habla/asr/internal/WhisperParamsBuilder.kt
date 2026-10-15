package com.xrmech.habla.asr.internal

import com.xrmech.habla.asr.AsrPass
import com.xrmech.habla.shared.AsrProfile
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/** Arguments of one `WhisperNative.transcribe` call (fixed fields live in C++). */
internal data class DecodeParams(
    val strategy: Int,
    val threads: Int,
    val beamSize: Int,
    val bestOf: Int,
    val audioCtx: Int,
    val singleSegment: Boolean,
    val temperature: Float,
    val temperatureInc: Float,
    val entropyThold: Float,
    val logprobThold: Float,
    val noSpeechThold: Float,
    val language: String,
)

/**
 * Maps the profile to whisper.cpp decoder settings (design.md section 5).
 *
 * FINAL is WhisperX's batched decoder: one beam pass, `best_of == beam` (so whisper.cpp never allocates
 * more decoders than beams), no temperature fallback, no-speech/entropy guards unreachable. PARTIAL
 * follows the `whisper_cpp.partial.level` ladder and never falls back either.
 */
internal object WhisperParamsBuilder {
    const val LEVEL_PARITY = "parity"
    const val LEVEL_GREEDY_FULL_CTX = "greedy_full_ctx"
    const val LEVEL_GREEDY_SCALED_CTX = "greedy_scaled_ctx"
    val LEVELS: List<String> = listOf(LEVEL_PARITY, LEVEL_GREEDY_FULL_CTX, LEVEL_GREEDY_SCALED_CTX)

    /** Encoder frames of a full 30 s window; larger audio_ctx makes whisper_full return -5. */
    const val MAX_AUDIO_CTX = 1500

    /** whisper.cpp `WHISPER_MAX_DECODERS`; more makes whisper_full return -4. */
    const val MAX_BEAM = 8

    const val MAX_THREADS = 16

    private const val SAMPLE_RATE = 16_000.0

    /** 100 mel frames/s, conv stride 2. */
    private const val ENCODER_FRAMES_PER_SECOND = 50

    fun finalBeamSize(profile: AsrProfile): Int = profile.whisperCpp.final.beamSize ?: profile.whisper.beamSize

    fun build(pass: AsrPass, language: String, sampleCount: Int, profile: AsrProfile, threads: Int): DecodeParams {
        val final = profile.whisperCpp.final
        val beam = finalBeamSize(profile)
        val beamSearch = DecodeParams(
            strategy = WhisperNative.STRATEGY_BEAM_SEARCH,
            threads = threads,
            beamSize = beam,
            bestOf = beam,
            audioCtx = 0,
            singleSegment = false,
            temperature = final.temperature.toFloat(),
            temperatureInc = final.temperatureInc.toFloat(),
            entropyThold = final.entropyThold.toFloat(),
            logprobThold = final.logprobThold.toFloat(),
            noSpeechThold = final.noSpeechThold.toFloat(),
            language = language,
        )
        if (pass == AsrPass.FINAL) return beamSearch

        val partial = profile.whisperCpp.partial
        // No fallback on partials at any level: it can multiply latency by up to 6x and the server has none.
        val partialBeam = beamSearch.copy(temperatureInc = 0f)
        return when (partial.level) {
            LEVEL_PARITY -> partialBeam
            LEVEL_GREEDY_FULL_CTX -> partialBeam.greedy(audioCtx = 0)
            LEVEL_GREEDY_SCALED_CTX -> partialBeam.greedy(
                audioCtx = scaledAudioCtx(sampleCount, partial.audioCtxPadFrames, partial.audioCtxMinFrames),
            )
            else -> throw IllegalArgumentException(
                "whisper_cpp.partial.level must be one of $LEVELS, was '${partial.level}'",
            )
        }
    }

    /** `min(1500, max(minFrames, ceil(n / 16000.0 * 50) + padFrames))`. */
    fun scaledAudioCtx(sampleCount: Int, padFrames: Int, minFrames: Int): Int {
        val speechFrames = ceil(sampleCount / SAMPLE_RATE * ENCODER_FRAMES_PER_SECOND).toInt()
        return min(MAX_AUDIO_CTX, max(minFrames, speechFrames + padFrames))
    }

    /**
     * Throws [IllegalArgumentException] naming the JSON key of the first value outside the limits of
     * design.md section 6.
     */
    fun validate(profile: AsrProfile) {
        val cpp = profile.whisperCpp
        requireIn("whisper_cpp.threads", cpp.threads, 1..MAX_THREADS)
        val beamKey = if (cpp.final.beamSize != null) "whisper_cpp.final.beam_size" else "whisper.beam_size"
        requireIn(beamKey, finalBeamSize(profile), 1..MAX_BEAM)
        requireUnit("whisper_cpp.final.temperature", cpp.final.temperature)
        requireUnit("whisper_cpp.final.temperature_inc", cpp.final.temperatureInc)
        require(cpp.partial.level in LEVELS) {
            "whisper_cpp.partial.level must be one of $LEVELS, was '${cpp.partial.level}'"
        }
        requireIn("whisper_cpp.partial.audio_ctx_pad_frames", cpp.partial.audioCtxPadFrames, 0..MAX_AUDIO_CTX)
        requireIn("whisper_cpp.partial.audio_ctx_min_frames", cpp.partial.audioCtxMinFrames, 1..MAX_AUDIO_CTX)
    }

    private fun DecodeParams.greedy(audioCtx: Int): DecodeParams = copy(
        strategy = WhisperNative.STRATEGY_GREEDY,
        beamSize = -1,
        bestOf = 1,
        singleSegment = true,
        audioCtx = audioCtx,
    )

    private fun requireIn(key: String, value: Int, range: IntRange) {
        require(value in range) { "$key must be ${range.first}..${range.last}, was $value" }
    }

    private fun requireUnit(key: String, value: Double) {
        require(value in 0.0..1.0) { "$key must be 0.0..1.0, was $value" }
    }
}
