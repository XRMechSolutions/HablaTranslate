package com.xrmech.habla.app.ui.history

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.xrmech.habla.app.ui.live.ExchangeCardView
import com.xrmech.habla.app.ui.nav.Navigator
import com.xrmech.habla.app.ui.theme.HablaColors
import org.koin.compose.viewmodel.koinViewModel
import org.koin.core.parameter.parametersOf

@Composable
fun SessionDetailScreen(
    sessionId: Long,
    @Suppress("UNUSED_PARAMETER") navigator: Navigator,
    vm: SessionDetailViewModel = koinViewModel(key = "session-$sessionId") { parametersOf(sessionId) },
) {
    val state by vm.ui.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val saf = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        if (uri != null) vm.exportTo(uri) // null = picker cancelled: nothing happens
    }
    LaunchedEffect(vm) {
        vm.effects.collect { e ->
            when (e) {
                is DetailEffect.Share -> context.startActivity(ExportController.shareChooser(e.uri))
            }
        }
    }
    SessionDetailContent(
        state = state,
        onSaveZip = { vm.suggestedName()?.let(saf::launch) },
        onShare = vm::share,
        onEdit = vm::editTranscript,
        onSaveVocab = vm::saveVocab,
    )
}

@Composable
fun SessionDetailContent(
    state: SessionDetailUiState,
    onSaveZip: () -> Unit,
    onShare: () -> Unit,
    onEdit: (Long, String) -> Unit,
    onSaveVocab: (DetailItem, com.xrmech.habla.translate.model.FlaggedPhrase) -> Unit,
) {
    var editing by remember { mutableStateOf<DetailItem?>(null) }
    Column(Modifier.fillMaxSize()) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 8.dp, top = 8.dp)) {
            Text("Session", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
            ExportMenu(state, onSaveZip, onShare)
        }
        state.exportDisabledReason?.let {
            if (!state.loading) Text(it, color = HablaColors.TextMuted, modifier = Modifier.padding(horizontal = 16.dp).testTag("export_disabled_reason"))
        }
        if (state.local.exporting) {
            Column(Modifier.padding(horizontal = 16.dp, vertical = 4.dp).testTag("export_progress")) {
                Text("Exporting...", color = HablaColors.TextMuted)
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
        }
        state.local.exportResult?.let { Text(it, color = HablaColors.Success, modifier = Modifier.padding(horizontal = 16.dp).testTag("export_result")) }
        state.local.exportError?.let { Text(it, color = HablaColors.ErrorText, modifier = Modifier.padding(horizontal = 16.dp).testTag("export_error")) }
        state.local.editError?.let { Text(it, color = HablaColors.ErrorText, modifier = Modifier.padding(horizontal = 16.dp)) }
        state.session?.topicSummary?.let {
            Text("Topic: $it", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp).testTag("topic_summary"))
        }
        when {
            state.loading -> Unit
            state.session == null -> Text("This session no longer exists.", color = HablaColors.TextMuted, modifier = Modifier.padding(16.dp))
            state.items.isEmpty() -> Text("No exchanges in this session.", color = HablaColors.TextMuted, modifier = Modifier.padding(16.dp))
            else -> LazyColumn(
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.testTag("detail_list"),
            ) {
                items(state.items, key = { it.row.id }) { item ->
                    Column {
                        ExchangeCardView(
                            card = item.card,
                            isSaved = { p -> state.isSaved(item, p) },
                            onSave = { p -> onSaveVocab(item, p) },
                            onAction = {},
                            onEdit = if (item.hasAudio) ({ editing = item }) else null,
                            failedText = "Not translated",
                        )
                        item.row.correctedTranscript?.let {
                            Text(
                                "Corrected transcript: $it",
                                style = MaterialTheme.typography.bodySmall,
                                color = HablaColors.TextMuted,
                                modifier = Modifier.padding(start = 12.dp, top = 4.dp).testTag("corrected_${item.row.id}"),
                            )
                        }
                    }
                }
            }
        }
    }
    editing?.let { item ->
        var text by remember(item.row.id) { mutableStateOf(item.editableTranscript) }
        AlertDialog(
            onDismissRequest = { editing = null },
            title = { Text("Edit transcript") },
            text = {
                Column {
                    Text("What was actually said. Leave blank to remove your correction.", color = HablaColors.TextMuted)
                    OutlinedTextField(value = text, onValueChange = { text = it }, modifier = Modifier.fillMaxWidth().testTag("edit_field"))
                }
            },
            confirmButton = {
                TextButton(onClick = { editing = null; onEdit(item.row.id, text) }, modifier = Modifier.testTag("edit_save")) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { editing = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun ExportMenu(state: SessionDetailUiState, onSaveZip: () -> Unit, onShare: () -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        TextButton(onClick = { open = true }, enabled = state.canExport, modifier = Modifier.testTag("export_menu")) { Text("Export") }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            DropdownMenuItem(text = { Text("Save .zip...") }, onClick = { open = false; onSaveZip() }, modifier = Modifier.testTag("export_save"))
            DropdownMenuItem(text = { Text("Share...") }, onClick = { open = false; onShare() }, modifier = Modifier.testTag("export_share"))
        }
    }
}
