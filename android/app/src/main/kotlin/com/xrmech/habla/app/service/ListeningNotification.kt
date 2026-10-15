package com.xrmech.habla.app.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.xrmech.habla.app.MainActivity
import com.xrmech.habla.app.R
import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.SessionPhase

/** Notification text per session state (design section 4); pure so it is JVM-tested. */
object NotificationText {
    fun of(state: LiveState, nowMs: Long): String? = when (val p = state.phase) {
        SessionPhase.Starting, SessionPhase.Calibrating -> "Calibrating..."
        SessionPhase.Listening -> if (state.audio == AudioCondition.Silenced) {
            "Mic silenced by another app"
        } else {
            val n = state.exchanges.size
            "Listening - $n ${if (n == 1) "exchange" else "exchanges"} - ${elapsed(nowMs - (state.startedAtMs ?: nowMs))}"
        }
        is SessionPhase.Finishing -> "Finishing ${p.pending} ${if (p.pending == 1) "segment" else "segments"}..."
        SessionPhase.Idle, is SessionPhase.Failed -> null
    }

    /** m:ss (minutes keep growing past 59). */
    fun elapsed(ms: Long): String {
        val total = maxOf(0L, ms) / 1000
        return "%d:%02d".format(total / 60, total % 60)
    }
}

/** At most one notification update per [minIntervalMs]; identical text is never re-posted. */
class NotificationThrottle(private val minIntervalMs: Long = 2_000) {
    private var lastText: String? = null
    private var lastAtMs: Long = Long.MIN_VALUE

    /** True when [text] should be posted now (and records it as posted). */
    fun offer(text: String, nowMs: Long): Boolean {
        if (text == lastText) return false
        if (lastAtMs != Long.MIN_VALUE && nowMs - lastAtMs < minIntervalMs) return false
        lastText = text
        lastAtMs = nowMs
        return true
    }
}

/** Builds and posts the ongoing "listening" notification (id 1001, channel `listening`). */
class ListeningNotification(private val context: Context) {
    fun ensureChannel() {
        val channel = NotificationChannel(CHANNEL_ID, "Listening", NotificationManager.IMPORTANCE_LOW).apply {
            description = "Shown while Habla is listening"
            setSound(null, null)
        }
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    fun build(text: String): Notification {
        val open = PendingIntent.getActivity(
            context,
            0,
            Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val stop = PendingIntent.getService(
            context,
            1,
            Intent(context, ListeningService::class.java)
                .setAction(ListeningService.ACTION_STOP)
                .putExtra(ListeningService.EXTRA_FROM_NOTIFICATION, true),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_mic)
            .setContentTitle(context.getString(R.string.app_name))
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setContentIntent(open)
            .addAction(0, "Stop", stop)
            .build()
    }

    /** No-op when POST_NOTIFICATIONS is denied (the session still runs; Stop is in the app). */
    fun post(text: String) {
        val manager = NotificationManagerCompat.from(context)
        if (!manager.areNotificationsEnabled()) return
        try {
            manager.notify(NOTIFICATION_ID, build(text))
        } catch (e: SecurityException) {
            // Permission revoked between the check and the post.
        }
    }

    companion object {
        const val CHANNEL_ID = "listening"
        const val NOTIFICATION_ID = 1001
    }
}
