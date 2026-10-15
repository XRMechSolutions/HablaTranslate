package com.xrmech.habla.audio

import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Clean-room offline model of ffmpeg's `dynaudnorm` for the server's parameters (D008, D029; R3
 * Q2). Provenance: the ffmpeg filter documentation, the Dynamic Audio Normalizer README prose
 * ("How It Works") and black-box measurement of the ffmpeg 8.0.1 binary. No ffmpeg or Dynamic
 * Audio Normalizer source was read or copied.
 *
 * Per frame of `frameLen` samples the local gain is `bnd(peak / framePeak)` with the sigmoid bound
 * `bnd(v) = maxGain * erf((sqrt(pi)/2) * v / maxGain)`; frames before the clip count as 1.0 and
 * frames after it as `bnd(1.0)`. A minimum filter of width `gaussSize` then a normalized Gaussian
 * (sigma = gaussSize / 6) smooth the gains, applied with a one-frame lag and clamped to the local
 * gain, and each sample gets a linear ramp between neighbouring frame gains. Clips shorter than
 * `2 * (gaussSize / 2)` frames follow the short-clip rules fitted to ffmpeg 8.0.1, which pin their
 * gain near 1.0 (D029, owner question 1: reproduced on purpose).
 */
internal object Dynaudnorm {
    /** Returns `x * gain` as a new array; `x` is not modified. */
    fun apply(x: DoubleArray, frameLen: Int, gaussSize: Int, peak: Double, maxGain: Double): DoubleArray =
        x.copyOf().also { applyInPlace(it, frameLen, gaussSize, peak, maxGain) }

    /** Multiplies `x` in place by the per-sample gain (computed on the fly, no gain array). */
    fun applyInPlace(x: DoubleArray, frameLen: Int, gaussSize: Int, peak: Double, maxGain: Double) {
        require(frameLen > 0) { "frameLen must be > 0, was $frameLen" }
        require(gaussSize >= 3 && gaussSize % 2 == 1) { "gaussSize must be odd and >= 3, was $gaussSize" }
        val n = x.size
        if (n == 0) return
        val r = gaussSize / 2
        val pad = 2 * r
        val k = (n + frameLen - 1) / frameLen
        val sqrtPiHalf = sqrt(PI) / 2
        fun bnd(v: Double) = maxGain * Erf.erf(sqrtPiHalf * v / maxGain)

        // Local gains, then the short-clip padding (fitted to ffmpeg 8.0.1).
        val kp = when {
            k < pad -> gaussSize
            k == pad -> k + 1
            else -> k
        }
        val local = DoubleArray(kp) { 1.0 }
        for (f in 0 until k) {
            var pk = 0.0
            val end = min(n, (f + 1) * frameLen)
            for (i in f * frameLen until end) {
                val a = Math.abs(x[i])
                if (a > pk) pk = a
            }
            local[f] = if (pk > 0.0) bnd(peak / pk) else maxGain
        }
        if (k == pad) local[k] = Double.POSITIVE_INFINITY

        // Extended gains: 2r frames of 1.0 before the clip, 2r frames of bnd(1.0) after it.
        val e = bnd(1.0)
        val len = pad + kp + pad
        val lp = DoubleArray(len) { i ->
            when {
                i < pad -> 1.0
                i < pad + kp -> local[i - pad]
                else -> e
            }
        }
        val m = DoubleArray(len) { i ->
            var v = Double.POSITIVE_INFINITY
            for (j in maxOf(0, i - r)..minOf(len - 1, i + r)) if (lp[j] < v) v = lp[j]
            v
        }
        val sigma = gaussSize / 6.0
        val w = DoubleArray(2 * r + 1) { exp(-((it - r) * (it - r)).toDouble() / (2 * sigma * sigma)) }
        var wSum = 0.0
        for (v in w) wSum += v
        for (i in w.indices) w[i] /= wSum
        fun smoothed(i: Int): Double {
            var s = 0.0
            for (d in 0..2 * r) s += w[d] * m[i - r + d]
            return s
        }
        // Frame gains with a one-frame lag, clamped to the local gain.
        val t = DoubleArray(kp + 1)
        t[0] = min(1.0, lp[pad])
        for (j in 1..kp) t[j] = min(smoothed(pad + j - 1), lp[pad + j])

        for (f in 0 until k) {
            val start = f * frameLen
            val nk = min(frameLen, n - start)
            val t0 = t[f]
            val dt = t[f + 1] - t[f]
            for (i in 0 until nk) {
                x[start + i] *= t0 + dt * (i + 1) / nk
            }
        }
    }
}
