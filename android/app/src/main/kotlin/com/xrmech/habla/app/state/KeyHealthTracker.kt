package com.xrmech.habla.app.state

import com.xrmech.habla.translate.LlmException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.flow.stateIn

/** PipelineStatus.keyState (contracts.md @ :app accepted change requests). */
enum class KeyState { UNKNOWN, ABSENT, PRESENT, INVALID, NO_QUOTA }

/** What the last translation or key test said about the key, independent of whether one exists. */
enum class KeyVerdict { NONE, INVALID, NO_QUOTA }

/** Pure state machine so the coordinator and settings tests can drive it. */
object KeyHealth {
    /** Outcome of a call: null = success, else the failure kind. Other kinds say nothing about the key. */
    fun afterOutcome(current: KeyVerdict, kind: LlmException.Kind?): KeyVerdict = when (kind) {
        null -> KeyVerdict.NONE
        LlmException.Kind.INVALID_KEY -> KeyVerdict.INVALID
        LlmException.Kind.QUOTA -> KeyVerdict.NO_QUOTA
        else -> current
    }

    fun state(hasKey: Boolean?, verdict: KeyVerdict): KeyState = when {
        hasKey == null -> KeyState.UNKNOWN
        !hasKey -> KeyState.ABSENT
        verdict == KeyVerdict.INVALID -> KeyState.INVALID
        verdict == KeyVerdict.NO_QUOTA -> KeyState.NO_QUOTA
        else -> KeyState.PRESENT
    }
}

/**
 * ABSENT when no key is stored; INVALID / NO_QUOTA after such a translation or key-test outcome;
 * PRESENT otherwise. [reset] on Save/Clear key.
 */
class KeyHealthTracker(hasKey: Flow<Boolean>, appScope: CoroutineScope) {
    private val verdict = MutableStateFlow(KeyVerdict.NONE)

    // Collected for the app's lifetime; hasKey is the key store's flow (never errors per data docs).
    val state: StateFlow<KeyState> = combine(
        hasKey.map<Boolean, Boolean?> { it }.onStart { emit(null) },
        verdict,
    ) { has, v -> KeyHealth.state(has, v) }
        .stateIn(appScope, SharingStarted.Eagerly, KeyState.UNKNOWN)

    fun report(kind: LlmException.Kind?) {
        verdict.value = KeyHealth.afterOutcome(verdict.value, kind)
    }

    fun reset() {
        verdict.value = KeyVerdict.NONE
    }
}
