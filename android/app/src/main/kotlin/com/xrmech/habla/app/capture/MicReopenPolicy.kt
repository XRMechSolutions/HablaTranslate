package com.xrmech.habla.app.capture

/**
 * Pure rules for [MicAudioSource] (JVM-testable): bounded reopen backoff (KB microphone
 * lifecycle) and the mapping from open stages and AudioRecord read codes to errors.
 */
internal object MicReopenPolicy {
    /** Delay before reopen attempt 1..5; after the 5th failure the session fails. */
    val BACKOFF_MS: List<Long> = listOf(250, 500, 1_000, 2_000, 4_000)
    val MAX_ATTEMPTS: Int = BACKOFF_MS.size

    /** Delay before [attempt] (1-based), or null when the source must give up. */
    fun delayBefore(attempt: Int): Long? = BACKOFF_MS.getOrNull(attempt - 1)

    /** Where `open()` failed, mapped to the public error kind. */
    enum class OpenStage { PERMISSION_DENIED, MIN_BUFFER, NOT_INITIALIZED, START }

    fun kindFor(stage: OpenStage): AudioSourceException.Kind = when (stage) {
        OpenStage.PERMISSION_DENIED -> AudioSourceException.Kind.PERMISSION
        OpenStage.MIN_BUFFER, OpenStage.NOT_INITIALIZED -> AudioSourceException.Kind.INIT_FAILED
        OpenStage.START -> AudioSourceException.Kind.START_FAILED
    }

    /** Any negative read result (ERROR, BAD_VALUE, INVALID_OPERATION, DEAD_OBJECT) means reopen. */
    fun needsReopen(readResult: Int): Boolean = readResult < 0

    /** AudioRecord read codes by name (values from android.media.AudioRecord). */
    fun describe(code: Int): String = when (code) {
        -1 -> "ERROR"
        -2 -> "ERROR_BAD_VALUE"
        -3 -> "ERROR_INVALID_OPERATION"
        -6 -> "ERROR_DEAD_OBJECT"
        else -> "error $code"
    }

    /** Recording buffer: at least 1 s of audio and twice the device minimum (design section 5). */
    fun bufferBytes(minBufferBytes: Int): Int = maxOf(2 * minBufferBytes, 32_000)
}
