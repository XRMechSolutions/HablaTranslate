package com.xrmech.habla.app.log

import com.xrmech.habla.app.BuildConfig

/** Destination for [HLog] lines; [priority] uses android.util.Log's numbering (2..7). */
fun interface LogSink {
    fun log(priority: Int, tag: String, msg: String, t: Throwable?)
}

/**
 * App-wide logger (design section 9). Every message and throwable message passes [redactSecrets]
 * before reaching the sink, as a last net: callers never pass a key on purpose. Transcript,
 * translation and vocab text go only through [text], which is compiled in but never evaluated
 * outside DEBUG builds.
 */
object HLog {
    const val DEBUG = 3
    const val INFO = 4
    const val WARN = 5
    const val ERROR = 6

    const val SERVICE = "Habla.Service"
    const val CAPTURE = "Habla.Capture"
    const val PIPELINE = "Habla.Pipeline"
    const val PARTIAL = "Habla.Partial"
    const val TRANSLATE = "Habla.Translate"
    const val UI = "Habla.UI"
    const val KEY = "Habla.Key"
    const val METRICS = "Habla.Metrics"

    @Volatile
    var sink: LogSink = PrintSink

    /** Seam for tests; production value is BuildConfig.DEBUG. */
    @Volatile
    var textEnabled: Boolean = BuildConfig.DEBUG

    private val SECRET = Regex("""sk-(proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{8,}""")

    fun redactSecrets(s: String): String = SECRET.replace(s, "sk-***")

    fun d(tag: String, msg: String) = emit(DEBUG, tag, msg, null)
    fun i(tag: String, msg: String) = emit(INFO, tag, msg, null)
    fun w(tag: String, msg: String, t: Throwable? = null) = emit(WARN, tag, msg, t)
    fun e(tag: String, msg: String, t: Throwable? = null) = emit(ERROR, tag, msg, t)

    /** User text (transcripts, translations): DEBUG builds and DEBUG level only. */
    inline fun text(tag: String, block: () -> String) {
        if (textEnabled) d(tag, block())
    }

    private fun emit(priority: Int, tag: String, msg: String, t: Throwable?) {
        sink.log(priority, tag, redactSecrets(msg), t?.let(::RedactedThrowable))
    }
}

/** Copy of a throwable whose message (and causes' messages) are redacted; stack trace kept. */
class RedactedThrowable(original: Throwable) : Throwable(
    HLog.redactSecrets(original.javaClass.name + (original.message?.let { ": $it" } ?: "")),
    original.cause?.let(::RedactedThrowable),
) {
    init {
        stackTrace = original.stackTrace
    }

    override fun toString(): String = message ?: javaClass.name
}

/** JVM tests and the pre-Koin window: plain stdout. */
object PrintSink : LogSink {
    private val names = mapOf(HLog.DEBUG to "D", HLog.INFO to "I", HLog.WARN to "W", HLog.ERROR to "E")

    override fun log(priority: Int, tag: String, msg: String, t: Throwable?) {
        println("${names[priority] ?: priority}/$tag: $msg" + (t?.let { " | $it" } ?: ""))
    }
}
