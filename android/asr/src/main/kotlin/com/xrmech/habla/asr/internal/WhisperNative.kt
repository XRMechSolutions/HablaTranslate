package com.xrmech.habla.asr.internal

/**
 * JNI entry points of `libhabla_asr_v86.so` / `libhabla_asr_base.so` (design.md section 3).
 *
 * Only [NativeEngineBinding] calls these, after [NativeLibraryLoader.load]. Every call except
 * [requestAbort] must come from the single `habla-asr` thread. A handle of 0 yields rc -100 (or a
 * no-op for [free], [requestAbort], [clearAbort]); no native call ever throws.
 */
internal object WhisperNative {
    /** Slot of the FINAL `whisper_state` (also used by language detection). */
    const val SLOT_FINAL: Int = 0

    /** Slot of the PARTIAL `whisper_state`. */
    const val SLOT_PARTIAL: Int = 1

    /** `WHISPER_SAMPLING_GREEDY`. */
    const val STRATEGY_GREEDY: Int = 0

    /** `WHISPER_SAMPLING_BEAM_SEARCH`. */
    const val STRATEGY_BEAM_SEARCH: Int = 1

    /** Run stopped by an abort callback (also reported when whisper returned 0 after stopping). */
    const val RC_ABORTED: Int = -3

    /** whisper_full rc: KV cache allocation failed; native code has re-created the slot state. */
    const val RC_KV_CACHE_FAILED: Int = -7

    /** Ours: handle 0, invalid slot, or the slot has no state (re-creation after rc -7 failed). */
    const val RC_NO_STATE: Int = -100

    /** Ours: invalid argument passed from Kotlin (programming error). */
    const val RC_BAD_ARGUMENT: Int = -101

    /** Ours: a C++ exception (e.g. allocation failure) was caught at the JNI boundary. */
    const val RC_NATIVE_FAILURE: Int = -102

    /**
     * Loads the model and both slot states; returns an engine handle, 0 on any failure. A non-null
     * [partialModelPath] loads a second model for the PARTIAL slot only (D045). [partialOnCpu]
     * (experimental, vk library only) keeps the PARTIAL slot's context on the CPU.
     */
    @JvmStatic external fun init(modelPath: String, partialModelPath: String?, flashAttn: Boolean, partialOnCpu: Boolean): Long

    @JvmStatic external fun free(handle: Long)

    /** Whisper language id for [code], -1 if unknown. */
    @JvmStatic external fun langId(code: String): Int

    @JvmStatic external fun langMaxId(): Int

    /** `whisper_print_system_info()`. */
    @JvmStatic external fun systemInfo(): String

    /** On [slot]: fills [probsOut] (size `langMaxId() + 1`); returns the argmax language id or a negative rc. */
    @JvmStatic external fun detectLanguage(handle: Long, slot: Int, samples: FloatArray, nThreads: Int, probsOut: FloatArray): Int

    @JvmStatic external fun transcribe(
        handle: Long,
        slot: Int,
        samples: FloatArray,
        strategy: Int,
        nThreads: Int,
        beamSize: Int,
        bestOf: Int,
        audioCtx: Int,
        singleSegment: Boolean,
        temperature: Float,
        temperatureInc: Float,
        entropyThold: Float,
        logprobThold: Float,
        noSpeechThold: Float,
        language: String,
    ): NativeResult

    /** Sets the slot's abort flag; safe from any thread. */
    @JvmStatic external fun requestAbort(handle: Long, slot: Int)

    /** Clears the slot's abort flag; ASR thread only. */
    @JvmStatic external fun clearAbort(handle: Long, slot: Int)
}
