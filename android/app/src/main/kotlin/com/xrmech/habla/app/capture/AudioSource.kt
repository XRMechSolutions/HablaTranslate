package com.xrmech.habla.app.capture

import kotlinx.coroutines.flow.StateFlow

/**
 * Where PCM comes from: the microphone on device, a WAV file in tests (interfaces.md @ capture).
 * 16 kHz mono PCM16 always; [read] blocks until samples exist.
 */
interface AudioSource : AutoCloseable {
    /** Always 16 000; CapturePump refuses anything else. */
    val sampleRate: Int
    val condition: StateFlow<AudioCondition>

    /** Starts capture or parses the file header. Throws [AudioSourceException]. */
    fun open()

    /** Returns 1..buffer.size samples, 0 (nothing yet), or -1 (end of stream / closed). */
    fun read(buffer: ShortArray): Int

    /** Idempotent; a pending or later [read] returns -1. */
    override fun close()
}

sealed interface AudioCondition {
    data object Normal : AudioCondition

    /** isClientSilenced, or 2 s of exact zeros (privacy toggle, input sharing during a call). */
    data object Silenced : AudioCondition

    /** Reopen attempt 1..5 after a dead AudioRecord. */
    data class Reopening(val attempt: Int) : AudioCondition
}

class AudioSourceException(
    message: String,
    val kind: Kind,
    val code: Int? = null,
    cause: Throwable? = null,
) : Exception(message, cause) {
    enum class Kind { PERMISSION, INIT_FAILED, START_FAILED, DEAD, IO }
}
