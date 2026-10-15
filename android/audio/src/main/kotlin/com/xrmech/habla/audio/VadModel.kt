package com.xrmech.habla.audio

/**
 * Per-frame speech probability model used by [Segmenter]: [SileroVad] or the energy fallback
 * [EnergyVad]. One instance per session thread; not thread-safe.
 */
interface VadModel : AutoCloseable {
    /**
     * Speech probability in [0, 1] for one 512-sample frame of samples in [-1, 1]; advances the
     * model's recurrent state.
     *
     * @throws IllegalArgumentException if `frame.size != 512`
     * @throws VadException on inference failure ([SileroVad] only)
     * @throws IllegalStateException after [close]
     */
    fun speechProbability(frame: FloatArray): Float

    /** Forgets all recurrent state and context. @throws IllegalStateException after [close]. */
    fun reset()
}

/** ONNX Runtime load or inference failure, or a model whose I/O does not match Silero v6. */
class VadException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

internal const val VAD_FRAME_SAMPLES = 512
