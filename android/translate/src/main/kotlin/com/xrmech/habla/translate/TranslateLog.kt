package com.xrmech.habla.translate

/**
 * Log sink for the translate module. `:app` installs one in [TranslateLogging.sink]; every message
 * this module emits has already been scrubbed by `Redact.secrets`.
 */
fun interface TranslateLog {
    fun log(level: Level, message: String)

    enum class Level { DEBUG, INFO, WARN, ERROR }

    companion object {
        /** Discards everything. */
        val NONE: TranslateLog = TranslateLog { _, _ -> }
    }
}

/** Process-wide sink, set once by `:app` at startup. */
object TranslateLogging {
    @Volatile
    var sink: TranslateLog = TranslateLog.NONE
}

/**
 * The only path from this module's code to a [TranslateLog]: scrubs API keys from every message.
 * Classes take a [TranslateLog] (default: the process sink) so tests can capture output.
 */
internal class Log(private val sink: () -> TranslateLog = { TranslateLogging.sink }) {
    constructor(fixed: TranslateLog) : this({ fixed })

    fun debug(message: String?) = emit(TranslateLog.Level.DEBUG, message)
    fun info(message: String?) = emit(TranslateLog.Level.INFO, message)
    fun warn(message: String?) = emit(TranslateLog.Level.WARN, message)
    fun error(message: String?) = emit(TranslateLog.Level.ERROR, message)

    fun emit(level: TranslateLog.Level, message: String?) {
        if (message.isNullOrEmpty()) return
        sink().log(level, Redact.secrets(message))
    }

    companion object {
        const val TRANSCRIPT_LOG_CHARS = 60

        /** Transcript text for DEBUG lines only: at most 60 chars, so a log never holds a whole utterance. */
        fun snippet(text: String?): String {
            if (text == null) return ""
            return text.take(TRANSCRIPT_LOG_CHARS)
        }
    }
}
