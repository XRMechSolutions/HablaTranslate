package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile
import kotlin.math.min
import kotlin.math.sqrt

/**
 * The server's energy fallback (`StreamingVADBuffer._is_speech_frame` without Silero):
 * `energy = sqrt(mean((x * 32768)^2))` in int16 units, probability
 * `min(1, energy / (energyThreshold * energy_fallback_prob_scale))`. The [Segmenter] decides
 * speech with `lastEnergy > energyThreshold`, not with the probability (server behaviour).
 */
class EnergyVad(profile: AsrProfile) : VadModel {
    private val probScale = profile.vad.energyFallbackProbScale

    /** int16-RMS threshold; starts at `calibration.uncalibrated_threshold`, set by the Segmenter. */
    var energyThreshold: Double = profile.calibration.uncalibratedThreshold
        set(value) {
            require(value.isFinite() && value > 0.0) { "energyThreshold must be finite and > 0, was $value" }
            field = value
        }

    /** int16-RMS energy of the most recent frame; 0.0 before the first frame and after [reset]. */
    var lastEnergy: Double = 0.0
        private set

    override fun speechProbability(frame: FloatArray): Float {
        require(frame.size == VAD_FRAME_SAMPLES) { "frame must have $VAD_FRAME_SAMPLES samples, had ${frame.size}" }
        var sum = 0.0
        for (x in frame) {
            val s = x * 32768.0
            sum += s * s
        }
        val energy = sqrt(sum / frame.size)
        lastEnergy = energy
        return min(1.0, energy / (energyThreshold * probScale)).toFloat()
    }

    override fun reset() {
        lastEnergy = 0.0
    }

    override fun close() = Unit
}
