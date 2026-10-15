package com.xrmech.habla.app.state

import android.os.PowerManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.runningFold
import kotlinx.coroutines.flow.stateIn

/** Hysteresis for partial ASR (design section 8): pause at SEVERE or worse, resume at MODERATE or better. */
object ThermalPolicy {
    // PowerManager.THERMAL_STATUS_* values (API 29); kept here so the policy is JVM-testable.
    const val MODERATE = 2
    const val SEVERE = 3

    fun paused(currentlyPaused: Boolean, status: Int): Boolean = when {
        status >= SEVERE -> true
        status <= MODERATE -> false
        else -> currentlyPaused
    }
}

/**
 * Pauses live partials when the phone is hot; finals never pause. The internal constructor takes
 * the raw status flow so JVM tests need no PowerManager.
 */
class ThermalMonitor internal constructor(statuses: Flow<Int>, appScope: CoroutineScope) {
    constructor(pm: PowerManager, appScope: CoroutineScope) : this(pm.statusFlow(), appScope)

    // Collected for the app's lifetime; the listener is removed only when the app scope ends.
    val status: StateFlow<Int> = statuses.stateIn(appScope, SharingStarted.Eagerly, 0)

    val partialsPaused: StateFlow<Boolean> = status
        .runningFold(false) { paused, s -> ThermalPolicy.paused(paused, s) }
        .stateIn(appScope, SharingStarted.Eagerly, false)

    private companion object {
        fun PowerManager.statusFlow(): Flow<Int> = callbackFlow {
            trySend(currentThermalStatus)
            val listener = PowerManager.OnThermalStatusChangedListener { trySend(it) }
            addThermalStatusListener(listener)
            awaitClose { removeThermalStatusListener(listener) }
        }
    }
}
