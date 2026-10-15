package com.xrmech.habla.app.service

import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import com.xrmech.habla.app.log.HLog

/** How the UI starts and stops listening (seam so ViewModel tests need no Context). */
interface SessionControl {
    /** Call only from a resumed activity's click handler with RECORD_AUDIO granted (while-in-use rule). */
    fun start()

    /** No effect when no session runs. */
    fun stop()
}

/** Starts and stops [ListeningService] with intents; the service is never bound. */
open class ServiceController(private val context: Context) : SessionControl {
    override fun start() {
        ContextCompat.startForegroundService(context, Intent(context, ListeningService::class.java).setAction(ListeningService.ACTION_START))
    }

    override fun stop() {
        try {
            context.startService(Intent(context, ListeningService::class.java).setAction(ListeningService.ACTION_STOP))
        } catch (e: IllegalStateException) {
            HLog.w(HLog.SERVICE, "stop request refused: ${e.message}")
        }
    }
}
