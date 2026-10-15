package com.xrmech.habla.asr.internal

/** Where :asr's Kotlin log lines go. Never pass transcript text: logcat is readable over adb. */
internal fun interface AsrLogSink {
    fun log(level: AsrLog.Level, message: String, error: Throwable?)
}

/**
 * Log seam that keeps the recognizer free of `android.*` imports so it runs on the JVM. The
 * [NativeEngineBinding] installs the logcat sink; JVM tests leave it silent or install a capturing one.
 */
internal object AsrLog {
    enum class Level { DEBUG, INFO, WARN, ERROR }

    @Volatile
    var sink: AsrLogSink = AsrLogSink { _, _, _ -> }

    fun i(message: String) = sink.log(Level.INFO, message, null)
    fun w(message: String, error: Throwable? = null) = sink.log(Level.WARN, message, error)
    fun e(message: String, error: Throwable? = null) = sink.log(Level.ERROR, message, error)
}
