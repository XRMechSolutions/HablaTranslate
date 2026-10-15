package com.xrmech.habla.app.ui.vocab

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.xrmech.habla.app.ui.nav.Navigator
import com.xrmech.habla.app.ui.theme.HablaColors
import com.xrmech.habla.data.VocabRow
import org.koin.compose.viewmodel.koinViewModel

@Composable
fun VocabScreen(@Suppress("UNUSED_PARAMETER") navigator: Navigator, vm: VocabViewModel = koinViewModel()) {
    val state by vm.ui.collectAsStateWithLifecycle()
    VocabContent(state, vm::setFilter)
}

@Composable
fun VocabContent(state: VocabUiState, onFilter: (String) -> Unit) {
    Column(Modifier.fillMaxSize()) {
        Text("Vocab", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(start = 16.dp, top = 16.dp, end = 16.dp))
        OutlinedTextField(
            value = state.filter,
            onValueChange = onFilter,
            label = { Text("Filter by term or meaning") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
            modifier = Modifier.fillMaxWidth().padding(16.dp).testTag("vocab_filter"),
        )
        when {
            state.loading -> Unit
            state.totalCount == 0 -> EmptyText("No saved phrases yet. Tap a flagged phrase on a Live or History card and choose Save to vocab.")
            state.rows.isEmpty() -> EmptyText("No saved phrase matches \"${state.filter.trim()}\".")
            else -> LazyColumn(
                contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.testTag("vocab_list"),
            ) {
                items(state.rows, key = { it.id }) { VocabItem(it) }
            }
        }
    }
}

@Composable
private fun EmptyText(text: String) {
    Text(text, color = HablaColors.TextMuted, modifier = Modifier.padding(16.dp).testTag("vocab_empty"))
}

@Composable
private fun VocabItem(row: VocabRow) {
    Card(colors = CardDefaults.cardColors(containerColor = HablaColors.Raised), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(row.term, style = MaterialTheme.typography.titleMedium)
            Text(row.meaning)
            row.literal?.takeIf { it.isNotBlank() }?.let { Text("Literally: $it", color = HablaColors.TextMuted) }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(row.category.replace('_', ' '), color = HablaColors.AccentText, style = MaterialTheme.typography.labelMedium)
                row.region?.takeIf { it.isNotBlank() }?.let {
                    Text(it, color = HablaColors.TextMuted, style = MaterialTheme.typography.labelMedium)
                }
                Text(
                    if (row.timesEncountered == 1) "seen once" else "seen ${row.timesEncountered} times",
                    color = HablaColors.TextMuted,
                    style = MaterialTheme.typography.labelMedium,
                )
            }
        }
    }
}
