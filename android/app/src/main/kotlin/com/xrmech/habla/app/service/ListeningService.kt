package com.xrmech.habla.app.service

import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.ServiceCompat
import com.xrmech.habla.app.capture.AudioSource
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.pipeline.PipelineCoordinator
import com.xrmech.habla.app.pipeline.SessionError
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.pipeline.SessionStartException
import com.xrmech.habla.app.pipeline.StopReason
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.koin.android.ext.android.inject

/**
 * The microphone foreground service (design section 4). Started only through [ServiceController]
 * from a visible activity; START_NOT_STICKY because a system restart could not reopen the mic
 * from the background and the session state is gone anyway.
 */
class ListeningService : Service() {
    private val coordinator: PipelineCoordinator by inject()
    private val wakeLock: WakeLockHolder by inject()
    private val sources: ServiceAudioSourceFactory by inject()

    private val notification by lazy { ListeningNotification(this) }
    private val throttle = NotificationThrottle()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var startJob: Job? = null
    private var running = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> handleStart()
            ACTION_STOP -> handleStop(
                if (intent.getBooleanExtra(EXTRA_FROM_NOTIFICATION, false)) StopReason.NotificationStop else StopReason.UserStop,
            )
            else -> if (!running) stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun handleStart() {
        if (running) {
            HLog.i(HLog.SERVICE, "start ignored: already listening")
            return
        }
        // Synchronously, before any suspension (FGS start rule).
        try {
            ServiceCompat.startForeground(
                this,
                ListeningNotification.NOTIFICATION_ID,
                notification.build("Calibrating..."),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } catch (e: SecurityException) {
            refuseStart(e)
            return
        } catch (e: IllegalStateException) {
            // ForegroundServiceStartNotAllowedException (API 31+) is an IllegalStateException.
            refuseStart(e)
            return
        }
        running = true
        wakeLock.acquire()
        val source: AudioSource = sources.create(this)
        // Driven by coordinator.start; a start failure already shows Failed in LiveState, so the
        // service only tears itself down. Cancelled with the service scope (onDestroy).
        startJob = scope.launch {
            try {
                coordinator.start(source)
                scope.launch { watchSession() }
            } catch (e: CancellationException) {
                throw e
            } catch (e: SessionStartException) {
                finish()
            } catch (e: IllegalStateException) {
                HLog.w(HLog.SERVICE, "start refused: ${e.message}")
                finish()
            }
        }
    }

    private fun refuseStart(e: Exception) {
        val message = e.message ?: e.javaClass.simpleName
        HLog.w(HLog.SERVICE, "startForeground refused: $message")
        coordinator.reportStartFailure(SessionError.StartNotAllowed(message))
        stopSelf()
    }

    /**
     * Keeps the notification current (throttled to one post per 2 s; a 1 s tick advances the
     * timer) and stops the service when the session ends for any reason. Runs until the session
     * is Idle/Failed or the service scope is cancelled.
     */
    private suspend fun watchSession() {
        while (scope.isActive) {
            val state = coordinator.state.value
            val text = NotificationText.of(state, SystemClock.elapsedRealtime())
            if (text == null) {
                if (state.phase is SessionPhase.Idle || state.phase is SessionPhase.Failed) {
                    finish()
                    return
                }
            } else if (throttle.offer(text, SystemClock.elapsedRealtime())) {
                notification.post(text)
            }
            delay(TICK_MS)
        }
    }

    private fun handleStop(reason: StopReason) {
        if (!running) {
            stopSelf()
            return
        }
        // Stop during Starting waits for start to return (or fail), then drains normally.
        scope.launch {
            startJob?.join()
            coordinator.stop(reason)
            finish()
        }
    }

    private fun finish() {
        if (!running) return
        running = false
        wakeLock.release()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        // Swiping the app away keeps listening (screen-off use case); the notification has Stop.
    }

    override fun onDestroy() {
        if (coordinator.state.value.phase !is SessionPhase.Idle && coordinator.state.value.phase !is SessionPhase.Failed) {
            coordinator.abort()
        }
        wakeLock.release()
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        const val ACTION_START = "com.xrmech.habla.app.action.START"
        const val ACTION_STOP = "com.xrmech.habla.app.action.STOP"
        const val EXTRA_FROM_NOTIFICATION = "from_notification"
        private const val TICK_MS = 1_000L
    }
}

/** Which AudioSource a service session reads: the mic in production, a WAV in instrumented tests. */
fun interface ServiceAudioSourceFactory {
    fun create(service: Service): AudioSource
}
