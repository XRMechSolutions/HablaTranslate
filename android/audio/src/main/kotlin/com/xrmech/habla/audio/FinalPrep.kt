package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile

/**
 * Clean-room port of the server's final-segment ffmpeg chain
 * `highpass=f=80,lowpass=f=7500,dynaudnorm=f=150:g=15:p=0.95:m=10` (`audio_processor
 * _final_prep_filter/_normalize_wav`; D008, D029, R3 Q1-Q2), all in Double:
 * `s / 32768.0` -> RBJ highpass -> RBJ lowpass -> [Dynaudnorm] -> `rint(y * 32768)` clamped.
 *
 * Provenance: ffmpeg filter documentation, the Dynamic Audio Normalizer README prose, the RBJ
 * Audio EQ Cookbook (W3C Note 2021) and black-box measurement of ffmpeg 8.0.1. No ffmpeg or
 * Dynamic Audio Normalizer source was read or copied. Stateless; safe from any thread.
 */
object FinalPrep {
    /**
     * Returns the prepared copy of [pcm] (same length; [pcm] is not modified). Memory: one
     * `DoubleArray(pcm.size)` filtered in place plus the output.
     *
     * @throws IllegalArgumentException naming the key if `final_prep.highpass_hz != 80`,
     *   `final_prep.lowpass_hz != 7500`, `audio.sample_rate != 16000`, `dynaudnorm.frame_ms`
     *   does not give a whole number of samples, `dynaudnorm.gauss_size` is not odd in 3..301,
     *   `dynaudnorm.peak` is not in (0, 1] or `dynaudnorm.max_gain` is not in [1, 100]
     */
    fun process(pcm: ShortArray, profile: AsrProfile): ShortArray {
        val fp = profile.finalPrep
        val dn = fp.dynaudnorm
        require(fp.highpassHz == 80) { "final_prep.highpass_hz must be 80 (fixed RBJ table), was ${fp.highpassHz}" }
        require(fp.lowpassHz == 7500) { "final_prep.lowpass_hz must be 7500 (fixed RBJ table), was ${fp.lowpassHz}" }
        require(profile.audio.sampleRate == 16000) { "audio.sample_rate must be 16000, was ${profile.audio.sampleRate}" }
        require(dn.frameMs > 0 && dn.frameMs.toLong() * 16000 % 1000 == 0L) {
            "final_prep.dynaudnorm.frame_ms must give a whole number of samples at 16 kHz, was ${dn.frameMs}"
        }
        require(dn.gaussSize in 3..301 && dn.gaussSize % 2 == 1) {
            "final_prep.dynaudnorm.gauss_size must be odd in 3..301, was ${dn.gaussSize}"
        }
        require(dn.peak > 0.0 && dn.peak <= 1.0) { "final_prep.dynaudnorm.peak must be in (0, 1], was ${dn.peak}" }
        require(dn.maxGain >= 1.0 && dn.maxGain <= 100.0) {
            "final_prep.dynaudnorm.max_gain must be in [1, 100], was ${dn.maxGain}"
        }
        val n = pcm.size
        if (n == 0) return ShortArray(0)

        val x = DoubleArray(n) { pcm[it] / 32768.0 }
        Biquads.highpass80(x)
        Biquads.lowpass7500(x)
        Dynaudnorm.applyInPlace(x, (dn.frameMs.toLong() * 16000 / 1000).toInt(), dn.gaussSize, dn.peak, dn.maxGain)
        return ShortArray(n) { Math.rint(x[it] * 32768.0).coerceIn(-32768.0, 32767.0).toInt().toShort() }
    }
}

/**
 * RBJ cookbook biquads with Q = 0.707 (ffmpeg's default width, measured exact) at 16 kHz, applied
 * in place with zero initial state in scipy `lfilter` direct form II transposed order (R3 Q1).
 */
internal object Biquads {
    /** `[b0, b1, b2, a1, a2]`, a0 normalized to 1. */
    internal val HP80 = doubleArrayOf(0.9780272696186294, -1.9560545392372588, 0.9780272696186294, -1.9555718227233034, 0.9565372557512138)
    internal val LP7500 = doubleArrayOf(0.8703148444087448, 1.7406296888174897, 0.8703148444087448, 1.7237446121141573, 0.7575147655208223)

    fun highpass80(x: DoubleArray) = filter(x, HP80)

    fun lowpass7500(x: DoubleArray) = filter(x, LP7500)

    private fun filter(x: DoubleArray, c: DoubleArray) {
        val b0 = c[0]
        val b1 = c[1]
        val b2 = c[2]
        val a1 = c[3]
        val a2 = c[4]
        var z0 = 0.0
        var z1 = 0.0
        for (i in x.indices) {
            val v = x[i]
            val y = b0 * v + z0
            z0 = (b1 * v + z1) - a1 * y
            z1 = b2 * v - a2 * y
            x[i] = y
        }
    }
}
