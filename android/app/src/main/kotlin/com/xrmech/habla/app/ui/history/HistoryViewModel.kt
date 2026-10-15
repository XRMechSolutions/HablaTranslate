package com.xrmech.habla.app.ui.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.pipeline.LiveController
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.SessionRow
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class HistoryUiState(
    val sessions: List<SessionRow> = emptyList(),
    /** The live session's id while one runs; its Delete is disabled. */
    val activeSessionId: Long? = null,
    val loading: Boolean = true,
    val error: String? = null,
)

/** Past sessions newest first (design section 6, History). */
class HistoryViewModel(private val history: HistoryRepository, private val live: LiveController) : ViewModel() {
    private val error = MutableStateFlow<String?>(null)

    val ui: StateFlow<HistoryUiState> = combine(history.sessions(), live.state, error) { rows, l, e ->
        HistoryUiState(rows, l.sessionId.takeIf { l.phase.active }, loading = false, error = e)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), HistoryUiState())

    /** Refused for the active session (edge case "Session deleted from History while it is the live session"). */
    fun delete(sessionId: Long) {
        val l = live.state.value
        if (l.phase.active && l.sessionId == sessionId) {
            error.value = "Stop listening before deleting this session"
            return
        }
        viewModelScope.launch {
            try {
                history.deleteSession(sessionId)
                error.value = null
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                HLog.w(HLog.UI, "session delete failed: ${e.message}")
                error.value = "Couldn't delete the session: ${e.message}"
            }
        }
    }
}
