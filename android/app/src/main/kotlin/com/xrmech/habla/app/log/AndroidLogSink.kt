package com.xrmech.habla.app.log

import android.util.Log

/** Production sink: logcat. Installed by HablaApp. */
object AndroidLogSink : LogSink {
    override fun log(priority: Int, tag: String, msg: String, t: Throwable?) {
        val line = if (t != null) msg + '\n' + Log.getStackTraceString(t) else msg
        Log.println(priority, tag, line)
    }
}
