package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Port of the server's partial-ASR prep (`audio_processor.prepare_partial_pcm`, R3 Q3): scipy
 * `butter(4, [80, 7500], 'band', fs=16000, output='sos')` applied with `sosfilt` in Double
 * (cascaded direct form II transposed, zero initial state), then RMS gain toward `target_rms`
 * capped at `max_gain` unless the filtered RMS is <= `min_rms_for_gain`, then int16 truncation
 * toward zero. Stateless; safe from any thread.
 */
object PartialPrep {
    /** `butter(4, [80, 7500], 'band', fs=16000, output='sos')` rows as `[b0, b1, b2, a1, a2]` (a0 = 1). */
    internal val SOS: Array<DoubleArray> = arrayOf(
        doubleArrayOf(0.7420434859024201, 1.4840869718048402, 0.7420434859024201, 1.6612958431214457, 0.6940169382407124),
        doubleArrayOf(1.0, 2.0, 1.0, 1.8255448495100886, 0.8613419770425558),
        doubleArrayOf(1.0, -2.0, 1.0, -1.9425131629847214, 0.9434772385780453),
        doubleArrayOf(1.0, -2.0, 1.0, -1.9753201768232749, 0.9762962394835808),
    )

    private const val REGENERATE =
        "partial_prep band changed: regenerate SOS with habla/scripts/make_audio_goldens.py --print-sos"

    /**
     * Returns the prepared copy of [pcm] (same length; [pcm] is not modified).
     *
     * @throws IllegalArgumentException if `partial_prep.bandpass_hz != [80, 7500]`,
     *   `partial_prep.butterworth_order != 4` or `audio.sample_rate != 16000` (the SOS table is fixed)
     */
    fun process(pcm: ShortArray, profile: AsrProfile): ShortArray {
        val pp = profile.partialPrep
        require(pp.bandpassHz == listOf(80, 7500)) { "partial_prep.bandpass_hz is ${pp.bandpassHz}; $REGENERATE" }
        require(pp.butterworthOrder == 4) { "partial_prep.butterworth_order is ${pp.butterworthOrder}; $REGENERATE" }
        require(profile.audio.sampleRate == 16000) { "audio.sample_rate is ${profile.audio.sampleRate}; $REGENERATE" }
        val n = pcm.size
        if (n == 0) return ShortArray(0)

        val y = DoubleArray(n) { pcm[it].toDouble() }
        for (sec in SOS) {
            val b0 = sec[0]
            val b1 = sec[1]
            val b2 = sec[2]
            val a1 = sec[3]
            val a2 = sec[4]
            var z0 = 0.0
            var z1 = 0.0
            for (i in 0 until n) {
                val v = y[i]
                val out = b0 * v + z0
                z0 = b1 * v - a1 * out + z1
                z1 = b2 * v - a2 * out
                y[i] = out
            }
        }
        var sum = 0.0
        for (v in y) sum += v * v
        val rms = sqrt(sum / n)
        if (rms > pp.minRmsForGain) {
            val gain = min(pp.targetRms / rms, pp.maxGain)
            for (i in 0 until n) y[i] = (y[i] * gain).coerceIn(-32768.0, 32767.0)
        }
        return ShortArray(n) { y[it].toInt().coerceIn(-32768, 32767).toShort() }
    }
}
