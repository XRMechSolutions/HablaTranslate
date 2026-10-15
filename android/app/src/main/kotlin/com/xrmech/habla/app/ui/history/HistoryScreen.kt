package com.xrmech.habla.app.ui.history

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.xrmech.habla.app.ui.nav.Navigator
import com.xrmech.habla.app.ui.nav.SessionDetail
import com.xrmech.habla.app.ui.theme.HablaColors
import com.xrmech.habla.data.SessionRow
import java.time.Duration
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import org.koin.compose.viewmodel.koinViewModel

@Composable
fun HistoryScreen(navigator: Navigator, vm: HistoryViewModel = koinViewModel()) {
    val state by vm.ui.collectAsStateWithLifecycle()
    HistoryContent(state, onOpen = { navigator.navigate(SessionDetail(it)) }, onDelete = vm::delete)
}

private val DATE = DateTimeFormatter.ofPattern("EEE d MMM yyyy, HH:mm", Locale.getDefault())

@Composable
fun HistoryContent(state: HistoryUiState, onOpen: (Long) -> Unit, onDelete: (Long) -> Unit) {
    var confirm by remember { mutableStateOf<SessionRow?>(null) }
    Column(Modifier.fillMaxSize()) {
        Text("History", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(16.dp))
        state.error?.let { Text(it, color = HablaColors.ErrorText, modifier = Modifier.padding(horizontal = 16.dp).testTag("history_error")) }
        when {
            state.loading -> Unit
            state.sessions.isEmpty() -> Text(
                "No sessions yet. Sessions appear here after you listen on Live.",
                color = HablaColors.TextMuted,
                modifier = Modifier.padding(16.dp).testTag("history_empty"),
            )
            else -> LazyColumn(
                contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.testTag("history_list"),
            ) {
                items(state.sessions, key = { it.id }) { s ->
                    SessionItem(s, active = s.id == state.activeSessionId, onOpen = { onOpen(s.id) }, onDelete = { confirm = s })
                }
            }
        }
    }
    confirm?.let { s ->
        AlertDialog(
            onDismissRequest = { confirm = null },
            title = { Text("Delete this session?") },
            text = { Text("Its ${s.exchangeCount} exchanges and any training audio are removed. Saved vocab stays.") },
            confirmButton = {
                TextButton(onClick = { confirm = null; onDelete(s.id) }, modifier = Modifier.testTag("history_delete_confirm")) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirm = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun SessionItem(s: SessionRow, active: Boolean, onOpen: () -> Unit, onDelete: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    Card(
        colors = CardDefaults.cardColors(containerColor = HablaColors.Raised),
        modifier = Modifier.fillMaxWidth().clickable(onClick = onOpen).testTag("session_${s.id}"),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 12.dp, top = 8.dp, bottom = 8.dp)) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(DATE.format(s.startedAt.atZone(ZoneId.systemDefault())), style = MaterialTheme.typography.titleSmall)
                Text(
                    listOf(
                        if (active) "Listening now" else durationText(s),
                        s.mode.name.lowercase().replaceFirstChar { it.titlecase(Locale.ROOT) },
                        if (s.exchangeCount == 1) "1 exchange" else "${s.exchangeCount} exchanges",
                        "$" + String.format(Locale.US, "%.3f", s.costUsd),
                    ).joinToString("  -  "),
                    color = HablaColors.TextMuted,
                    style = MaterialTheme.typography.bodySmall,
                )
                s.topicSummary?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            }
            Box {
                TextButton(onClick = { menu = true }, modifier = Modifier.testTag("session_${s.id}_menu")) { Text("...") }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(
                        text = { Text(if (active) "Delete (stop listening first)" else "Delete") },
                        enabled = !active,
                        onClick = { menu = false; onDelete() },
                        modifier = Modifier.testTag("session_${s.id}_delete"),
                    )
                }
            }
        }
    }
}

internal fun durationText(s: SessionRow): String {
    val end = s.endedAt ?: return "Not finished"
    val d = Duration.between(s.startedAt, end).coerceAtLeast(Duration.ZERO)
    val m = d.toMinutes()
    return if (m >= 60) "${m / 60} h ${m % 60} min" else if (m >= 1) "$m min" else "${d.seconds} s"
}
