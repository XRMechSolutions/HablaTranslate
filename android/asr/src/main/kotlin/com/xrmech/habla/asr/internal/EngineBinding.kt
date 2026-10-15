package com.xrmech.habla.asr.internal

import android.util.Log

/**
 * Kotlin-side seam over the native engine so [com.xrmech.habla.asr.WhisperCppRecognizer]'s threading
 * and mapping run on the JVM with a fake. Only [NativeEngineBinding] reaches [WhisperNative].
 */
internal interface EngineBinding {
    /** Loads model + both states; 0 on failure. */
    fun load(modelPath: String, flashAttn: Boolean): Long
    fun free(handle: Long)
    /** -1 if unknown. */
    fun langId(code: String): Int
    fun langMaxId(): Int
    /** >= 0 argmax id, < 0 rc. */
    fun detectLanguage(handle: Long, samples: FloatArray, threads: Int, probsOut: FloatArray): Int
    fun transcribe(handle: Long, slot: Int, samples: FloatArray, p: DecodeParams): NativeResult
    /** Any thread. */
    fun requestAbort(handle: Long, slot: Int)
    /** ASR thread only. */
    fun clearAbort(handle: Long, slot: Int)
    fun systemInfo(): String
    /** "v86" | "base". */
    val variant: String
}

/**
 * Production binding. Every call first makes sure the native library is loaded (cheap after the first
 * time), so a link failure surfaces as [UnsatisfiedLinkError] from whichever call comes first.
 */
internal class NativeEngineBinding : EngineBinding {
    init {
        AsrLog.sink = LogcatSink
    }

    override val variant: String get() = NativeLibraryLoader.load().label

    override fun load(modelPath: String, flashAttn: Boolean): Long {
        NativeLibraryLoader.load()
        return WhisperNative.init(modelPath, flashAttn)
    }

    override fun free(handle: Long) {
        NativeLibraryLoader.load()
        WhisperNative.free(handle)
    }

    override fun langId(code: String): Int {
        NativeLibraryLoader.load()
        return WhisperNative.langId(code)
    }

    override fun langMaxId(): Int {
        NativeLibraryLoader.load()
        return WhisperNative.langMaxId()
    }

    override fun detectLanguage(handle: Long, samples: FloatArray, threads: Int, probsOut: FloatArray): Int {
        NativeLibraryLoader.load()
        return WhisperNative.detectLanguage(handle, samples, threads, probsOut)
    }

    override fun transcribe(handle: Long, slot: Int, samples: FloatArray, p: DecodeParams): NativeResult {
        NativeLibraryLoader.load()
        return WhisperNative.transcribe(
            handle, slot, samples, p.strategy, p.threads, p.beamSize, p.bestOf, p.audioCtx, p.singleSegment,
            p.temperature, p.temperatureInc, p.entropyThold, p.logprobThold, p.noSpeechThold, p.language,
        )
    }

    override fun requestAbort(handle: Long, slot: Int) {
        WhisperNative.requestAbort(handle, slot)
    }

    override fun clearAbort(handle: Long, slot: Int) {
        WhisperNative.clearAbort(handle, slot)
    }

    override fun systemInfo(): String {
        NativeLibraryLoader.load()
        return WhisperNative.systemInfo()
    }

    private object LogcatSink : AsrLogSink {
        private const val TAG = "HablaAsr"

        override fun log(level: AsrLog.Level, message: String, error: Throwable?) {
            when (level) {
                AsrLog.Level.DEBUG -> Log.d(TAG, message, error)
                AsrLog.Level.INFO -> Log.i(TAG, message, error)
                AsrLog.Level.WARN -> Log.w(TAG, message, error)
                AsrLog.Level.ERROR -> Log.e(TAG, message, error)
            }
        }
    }
}
