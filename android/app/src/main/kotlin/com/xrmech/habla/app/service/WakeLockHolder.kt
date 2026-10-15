package com.xrmech.habla.app.service

import android.os.PowerManager

/**
 * Partial wake lock for the whole session (design section 4: screen-off sessions need the
 * segmenter, Whisper and network running). Not reference-counted; 4 h acquire timeout as a cap.
 */
class WakeLockHolder(private val powerManager: PowerManager) {
    private var lock: PowerManager.WakeLock? = null

    @Synchronized
    fun acquire() {
        if (lock?.isHeld == true) return
        lock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, TAG).apply {
            setReferenceCounted(false)
            acquire(TIMEOUT_MS)
        }
    }

    /** Idempotent. */
    @Synchronized
    fun release() {
        lock?.let { if (it.isHeld) it.release() }
        lock = null
    }

    private companion object {
        const val TAG = "habla:listening"
        const val TIMEOUT_MS = 4L * 60 * 60 * 1000
    }
}
