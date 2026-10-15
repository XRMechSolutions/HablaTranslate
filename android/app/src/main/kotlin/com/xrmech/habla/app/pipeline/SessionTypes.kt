package com.xrmech.habla.app.pipeline

import com.xrmech.habla.app.capture.AudioSourceException
import com.xrmech.habla.asr.AsrException
import kotlinx.coroutines.flow.StateFlow

/** What the UI depends on, so UI tests can swap a fake. Start/stop go through ServiceController. */
interface LiveController {
    val state: StateFlow<LiveState>

    /** Last metrics JSON line (every 60 s and at session end), for Settings > Diagnostics. */
    val lastMetricsLine: StateFlow<String?>
}

sealed interface StopReason {
    data object UserStop : StopReason
    data object NotificationStop : StopReason
    data object EndOfStream : StopReason
    data class AudioFailed(val error: AudioSourceException) : StopReason

    /** Three consecutive NATIVE_ERROR finals. */
    data class AsrFailed(val error: AsrException) : StopReason
}

/** Thrown by `start` when the session could not begin; the state shows `Failed(error)` too. */
class SessionStartException(val error: SessionError) : Exception(error.message)

sealed interface SessionError {
    /** The verbatim fault shown to the user. */
    val message: String

    data class StartNotAllowed(override val message: String) : SessionError
    data class Audio(val kind: AudioSourceException.Kind, override val message: String) : SessionError
    data class ModelLoad(override val message: String) : SessionError
    data class Storage(override val message: String) : SessionError
    data class Asr(override val message: String) : SessionError
}
