package com.xrmech.habla.app.ui.vocab

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.xrmech.habla.data.VocabRepository
import com.xrmech.habla.data.VocabRow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn

data class VocabUiState(
    val filter: String = "",
    /** Rows matching [filter], newest first (repository order). */
    val rows: List<VocabRow> = emptyList(),
    /** Saved terms before filtering; distinguishes "nothing saved" from "nothing matches". */
    val totalCount: Int = 0,
    val loading: Boolean = true,
)

/** Saved phrases with an in-memory contains-filter on term and meaning (design section 6, Vocab). */
class VocabViewModel(vocab: VocabRepository) : ViewModel() {
    private val filter = MutableStateFlow("")

    val ui: StateFlow<VocabUiState> = combine(vocab.all(), filter) { rows, f ->
        VocabUiState(filter = f, rows = filterRows(rows, f), totalCount = rows.size, loading = false)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), VocabUiState())

    fun setFilter(text: String) {
        filter.value = text
    }

    companion object {
        /** Case-insensitive contains-match on term or meaning; a blank filter keeps every row. */
        fun filterRows(rows: List<VocabRow>, filter: String): List<VocabRow> {
            val f = filter.trim()
            if (f.isEmpty()) return rows
            return rows.filter { it.term.contains(f, ignoreCase = true) || it.meaning.contains(f, ignoreCase = true) }
        }
    }
}
