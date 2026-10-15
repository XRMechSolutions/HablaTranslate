package com.xrmech.habla.asr.internal

import androidx.annotation.Keep

/**
 * Result of one `whisper_full_with_state` call, constructed only from JNI (constructor descriptor
 * `(IJ[[B[J[J[F[F[I)V`, kept by consumer-rules.pro and [Keep]). Arrays are parallel, one entry per
 * segment, and all empty when [rc] != 0.
 */
@Keep
internal class NativeResult(
    @JvmField val rc: Int,
    @JvmField val wallMs: Long,
    /** Raw UTF-8 per segment; may end inside a multibyte character. */
    @JvmField val texts: Array<ByteArray>,
    /** Segment start, centiseconds. */
    @JvmField val t0: LongArray,
    /** Segment end, centiseconds. */
    @JvmField val t1: LongArray,
    @JvmField val noSpeechProb: FloatArray,
    /** Mean token log-probability over text tokens; NaN when the segment has none. */
    @JvmField val avgLogprob: FloatArray,
    @JvmField val textTokens: IntArray,
)
