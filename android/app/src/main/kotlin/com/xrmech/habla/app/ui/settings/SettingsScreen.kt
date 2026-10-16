package com.xrmech.habla.app.ui.settings

import android.content.ActivityNotFoundException
import android.content.Intent
import android.provider.Settings as AndroidSettings
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
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
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.xrmech.habla.app.BuildConfig
import com.xrmech.habla.app.ui.live.ModelFailureText
import com.xrmech.habla.app.ui.nav.Navigator
import com.xrmech.habla.app.ui.nav.Settings
import com.xrmech.habla.app.ui.theme.HablaColors
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import java.util.Locale
import org.koin.compose.viewmodel.koinViewModel

/** Callbacks of [SettingsContent]; the screen wires them to [SettingsViewModel]. */
data class SettingsActions(
    val saveKey: (String) -> Unit = {},
    val testKey: (String?) -> Unit = {},
    val clearKey: () -> Unit = {},
    val setFinalModel: (String) -> Unit = {},
    val setQuickModel: (String) -> Unit = {},
    val setPartialTranslation: (Boolean) -> Unit = {},
    val setPartialsEnabled: (Boolean) -> Unit = {},
    val setDirection: (Direction) -> Unit = {},
    val setMode: (Mode) -> Unit = {},
    val setAsrModel: (String) -> Unit = {},
    val downloadModel: (Boolean) -> Unit = {},
    val dismissMeteredDownload: () -> Unit = {},
    val cancelDownload: () -> Unit = {},
    val deleteModel: () -> Unit = {},
    val setTrainingCapture: (Boolean) -> Unit = {},
    val setLiveModel: (String) -> Unit = {},
    val downloadLiveModel: (Boolean) -> Unit = {},
    val setAsrGpu: (Boolean) -> Unit = {},
)

@Composable
fun SettingsScreen(section: String?, @Suppress("UNUSED_PARAMETER") navigator: Navigator, vm: SettingsViewModel = koinViewModel()) {
    val state by vm.ui.collectAsStateWithLifecycle()
    var keySavedTick by remember { mutableStateOf(0) }
    LaunchedEffect(vm) {
        vm.effects.collect { if (it == SettingsEffect.KeySaved) keySavedTick++ }
    }
    SettingsContent(
        state = state,
        section = section,
        keySavedTick = keySavedTick,
        actions = SettingsActions(
            saveKey = vm::saveKey, testKey = vm::testKey, clearKey = vm::clearKey,
            setFinalModel = vm::setFinalModel, setQuickModel = vm::setQuickModel,
            setPartialTranslation = vm::setPartialTranslation, setPartialsEnabled = vm::setPartialsEnabled,
            setDirection = vm::setDirection, setMode = vm::setMode, setAsrModel = vm::setAsrModel,
            downloadModel = vm::downloadModel, dismissMeteredDownload = vm::dismissMeteredDownload,
            cancelDownload = vm::cancelDownload, deleteModel = vm::deleteModel,
            setTrainingCapture = vm::setTrainingCapture,
            setLiveModel = vm::setLiveModel, downloadLiveModel = vm::downloadLiveModel, setAsrGpu = vm::setAsrGpu,
        ),
    )
}

private val SECTION_ORDER = listOf(
    Settings.KEY, Settings.TRANSLATION, Settings.LISTENING, Settings.MODEL, Settings.TRAINING, Settings.COST,
    "battery", "diagnostics", "about",
)

/** One scrolling screen; [section] scrolls to that section once (design section 6, Settings). */
@Composable
fun SettingsContent(state: SettingsUiState, section: String?, keySavedTick: Int, actions: SettingsActions) {
    val list = rememberLazyListState()
    LaunchedEffect(section) {
        val index = SECTION_ORDER.indexOf(section)
        if (index >= 0) list.scrollToItem(index + 1)
    }
    LazyColumn(
        state = list,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxSize().testTag("settings_list"),
    ) {
        item("title") { Text("Settings", style = MaterialTheme.typography.headlineSmall) }
        item(Settings.KEY) { KeySection(state, keySavedTick, actions) }
        item(Settings.TRANSLATION) { TranslationSection(state, actions) }
        item(Settings.LISTENING) { ListeningSection(state, actions) }
        item(Settings.MODEL) { SpeechModelSection(state, actions) }
        item(Settings.TRAINING) { TrainingSection(state, actions) }
        item(Settings.COST) { CostSection(state) }
        item("battery") { BatterySection() }
        item("diagnostics") { DiagnosticsSection(state) }
        item("about") { AboutSection() }
    }
}

@Composable
private fun SectionCard(title: String, tag: String, content: @Composable () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = HablaColors.Surface),
        modifier = Modifier.fillMaxWidth().testTag("section_$tag"),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
private fun KeySection(state: SettingsUiState, keySavedTick: Int, actions: SettingsActions) {
    // The key lives only in this transient field; it is never pre-filled and never shown after Save.
    var text by remember { mutableStateOf("") }
    var visible by remember { mutableStateOf(false) }
    var confirmClear by remember { mutableStateOf(false) }
    LaunchedEffect(keySavedTick) {
        if (keySavedTick > 0) {
            text = ""
            visible = false
        }
    }
    SectionCard("OpenAI key", Settings.KEY) {
        Text(
            state.keyStatus,
            color = when (state.keyStatus) {
                "Key saved" -> HablaColors.Success
                "Key rejected", "No credit" -> HablaColors.ErrorText
                else -> HablaColors.TextMuted
            },
            modifier = Modifier.testTag("key_status"),
        )
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text(if (state.keyStatus == "Key saved") "Replace key" else "API key") },
            singleLine = true,
            visualTransformation = if (visible) VisualTransformation.None else PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, autoCorrectEnabled = false),
            trailingIcon = {
                TextButton(onClick = { visible = !visible }, modifier = Modifier.testTag("key_visibility")) {
                    Text(if (visible) "Hide" else "Show")
                }
            },
            isError = state.local.keyError != null,
            supportingText = state.local.keyError?.let { msg -> { Text(msg, modifier = Modifier.testTag("key_error")) } },
            modifier = Modifier.fillMaxWidth().testTag("key_field"),
        )
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { actions.saveKey(text) }, modifier = Modifier.testTag("key_save")) { Text("Save") }
            OutlinedButton(
                onClick = { actions.testKey(text.ifBlank { null }) },
                enabled = !state.local.keyTestRunning,
                modifier = Modifier.testTag("key_test"),
            ) { Text(if (state.local.keyTestRunning) "Testing..." else "Test key") }
            OutlinedButton(onClick = { confirmClear = true }, modifier = Modifier.testTag("key_clear")) { Text("Clear key") }
        }
        state.local.keyTestMessage?.let { Text(it, modifier = Modifier.testTag("key_test_result")) }
    }
    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear the OpenAI key?") },
            text = { Text("Translation stops until you add a key again. Transcription keeps working.") },
            confirmButton = {
                TextButton(onClick = { confirmClear = false; actions.clearKey() }, modifier = Modifier.testTag("key_clear_confirm")) {
                    Text("Clear key")
                }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun TranslationSection(state: SettingsUiState, actions: SettingsActions) {
    val s = state.settings
    SectionCard("Translation", Settings.TRANSLATION) {
        Picker("Final model", s?.finalModel, state.finalModels.map { it.id to it.label }, actions.setFinalModel, "final_model")
        Picker("Quick model", s?.quickModel, state.quickModels.map { it.id to it.label }, actions.setQuickModel, "quick_model")
        ToggleRow("Translate live partials", s?.partialTranslation == true, s != null, actions.setPartialTranslation, "partial_translation")
    }
}

@Composable
private fun ListeningSection(state: SettingsUiState, actions: SettingsActions) {
    val s = state.settings
    SectionCard("Listening", Settings.LISTENING) {
        Picker(
            "Direction", s?.direction?.name,
            Direction.entries.map { it.name to directionLabel(it) },
            { name -> actions.setDirection(Direction.valueOf(name)) }, "direction",
        )
        Picker(
            "Mode", s?.mode?.name,
            Mode.entries.map { it.name to it.name.lowercase().replaceFirstChar { c -> c.titlecase(Locale.ROOT) } },
            { name -> actions.setMode(Mode.valueOf(name)) }, "mode",
        )
        ToggleRow("Show live partials", s?.partialsEnabled == true, s != null, actions.setPartialsEnabled, "partials_enabled")
    }
}

@Composable
private fun SpeechModelSection(state: SettingsUiState, actions: SettingsActions) {
    val selected = state.selectedSpeechModel
    SectionCard("Speech model", Settings.MODEL) {
        Picker("Model", state.settings?.asrModelId, state.speechModels.map { it.id to it.label }, actions.setAsrModel, "asr_model")
        val m = state.modelState
        val status = modelStatusText(m)
        Text(status, color = if (m is ModelState.Failed) HablaColors.ErrorText else HablaColors.Text, modifier = Modifier.testTag("model_status"))
        if (m is ModelState.Downloading && m.total > 0) {
            LinearProgressIndicator(progress = { m.bytes.toFloat() / m.total }, modifier = Modifier.fillMaxWidth().testTag("model_progress"))
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.downloadRunning || m is ModelState.Downloading) {
                OutlinedButton(onClick = actions.cancelDownload, modifier = Modifier.testTag("model_cancel")) { Text("Cancel") }
            } else if (m !is ModelState.Ready) {
                Button(onClick = { actions.downloadModel(false) }, modifier = Modifier.testTag("model_download")) {
                    Text(selected?.let { "Download (${mb(it.sizeBytes)} MB)" } ?: "Download")
                }
            }
            if (m is ModelState.Ready || m is ModelState.Failed) {
                OutlinedButton(onClick = actions.deleteModel, enabled = !state.sessionActive, modifier = Modifier.testTag("model_delete")) {
                    Text("Delete")
                }
            }
        }
        if (state.sessionActive) Text("Delete is unavailable while listening", color = HablaColors.TextMuted)
        state.local.modelMessage?.let { Text(it, color = HablaColors.ErrorText) }

        // Live preview (D045): a smaller model for partials; finals keep the speech model.
        Picker("Live preview model", state.settings?.liveModelId, state.liveModels.map { it.id to it.label }, actions.setLiveModel, "live_model")
        state.liveModelState?.let { lm ->
            Text(modelStatusText(lm), color = if (lm is ModelState.Failed) HablaColors.ErrorText else HablaColors.Text, modifier = Modifier.testTag("live_model_status"))
            if (lm is ModelState.Downloading && lm.total > 0) {
                LinearProgressIndicator(progress = { lm.bytes.toFloat() / lm.total }, modifier = Modifier.fillMaxWidth().testTag("live_model_progress"))
            }
            if (lm !is ModelState.Ready && lm !is ModelState.Downloading && !state.downloadRunning) {
                Button(onClick = { actions.downloadLiveModel(false) }, modifier = Modifier.testTag("live_model_download")) {
                    Text(state.selectedLiveModel?.let { "Download (${mb(it.sizeBytes)} MB)" } ?: "Download")
                }
            }
        }
        if (state.gpuBuilt) {
            ToggleRow("GPU acceleration (experimental)", state.settings?.asrGpu == true, state.settings != null, actions.setAsrGpu, "asr_gpu")
            Text(
                "Applies after restarting the app. Engine now: ${state.engineNow ?: "not loaded yet"}",
                color = HablaColors.TextMuted, modifier = Modifier.testTag("asr_engine_now"),
            )
        }
    }
    if (state.local.confirmMeteredDownload) {
        val live = state.local.meteredForLive
        val size = (if (live) state.selectedLiveModel else selected)?.let { mb(it.sizeBytes) } ?: "?"
        AlertDialog(
            onDismissRequest = actions.dismissMeteredDownload,
            title = { Text("Download on mobile data?") },
            text = { Text("The ${if (live) "live preview" else "speech"} model is $size MB and this network is metered.") },
            confirmButton = {
                TextButton(onClick = { if (live) actions.downloadLiveModel(true) else actions.downloadModel(true) }) { Text("Download") }
            },
            dismissButton = { TextButton(onClick = actions.dismissMeteredDownload) { Text("Cancel") } },
        )
    }
}

private fun modelStatusText(m: ModelState): String = when (m) {
    ModelState.Missing -> "Not downloaded"
    is ModelState.Downloading -> "Downloading ${mb(m.bytes)} / ${mb(m.total)} MB"
    ModelState.Verifying -> "Verifying..."
    is ModelState.Ready -> "Ready"
    is ModelState.Failed -> "Download failed: ${ModelFailureText.of(m.reason)}"
}

@Composable
private fun TrainingSection(state: SettingsUiState, actions: SettingsActions) {
    val s = state.settings
    SectionCard("Training capture", Settings.TRAINING) {
        ToggleRow("Save segment audio for training", s?.trainingCapture == true, s != null, actions.setTrainingCapture, "training_capture")
        Text(
            "Keeps each final segment's audio on this phone so you can export a training set from History.",
            color = HablaColors.TextMuted,
        )
        if (state.sessionActive) Text("Changes apply from the next session", color = HablaColors.Warning, modifier = Modifier.testTag("training_next_session"))
    }
}

@Composable
private fun CostSection(state: SettingsUiState) {
    SectionCard("Cost", Settings.COST) {
        Text("This session: ${usd(state.costs.sessionCostUsd)}")
        Text("All time: ${usd(state.costs.totalCostUsd)}", modifier = Modifier.testTag("cost_total"))
    }
}

@Composable
private fun BatterySection() {
    val context = LocalContext.current
    SectionCard("Battery", "battery") {
        Text(
            "Samsung phones may put Habla to sleep during long sessions. Set Habla to \"Unrestricted\" battery use.",
            color = HablaColors.TextMuted,
        )
        OutlinedButton(onClick = {
            try {
                context.startActivity(Intent(AndroidSettings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
            } catch (_: ActivityNotFoundException) {
                context.startActivity(Intent(AndroidSettings.ACTION_SETTINGS))
            }
        }) { Text("Open battery settings") }
    }
}

@Composable
private fun DiagnosticsSection(state: SettingsUiState) {
    SectionCard("Diagnostics", "diagnostics") {
        Text(
            state.metricsLine ?: "No session metrics yet",
            fontFamily = FontFamily.Monospace,
            style = MaterialTheme.typography.bodySmall,
            color = HablaColors.TextMuted,
            modifier = Modifier.testTag("metrics_line"),
        )
    }
}

@Composable
private fun AboutSection() {
    SectionCard("About", "about") {
        Text("Habla ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
        Text("whisper.cpp - MIT License", color = HablaColors.TextMuted)
        Text("Silero VAD - MIT License", color = HablaColors.TextMuted)
        Text("ONNX Runtime - MIT License", color = HablaColors.TextMuted)
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, enabled: Boolean, onChange: (Boolean) -> Unit, tag: String) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange, enabled = enabled, modifier = Modifier.testTag(tag))
    }
}

/** Label plus a button showing the current choice; tapping opens every option. */
@Composable
private fun Picker(label: String, selectedId: String?, options: List<Pair<String, String>>, onSelect: (String) -> Unit, tag: String) {
    var open by remember { mutableStateOf(false) }
    Column {
        Text(label, style = MaterialTheme.typography.labelMedium, color = HablaColors.TextMuted)
        Box {
            OutlinedButton(onClick = { open = true }, enabled = selectedId != null, modifier = Modifier.fillMaxWidth().testTag(tag)) {
                Text(options.firstOrNull { it.first == selectedId }?.second ?: selectedId ?: "...", modifier = Modifier.weight(1f))
                Text("Change")
            }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                options.forEach { (id, text) ->
                    DropdownMenuItem(
                        text = { Text(text) },
                        onClick = { open = false; onSelect(id) },
                        modifier = Modifier.testTag("${tag}_option_$id"),
                    )
                }
            }
        }
    }
}

internal fun directionLabel(d: Direction): String = when (d) {
    Direction.ES_TO_EN -> "Spanish -> English"
    Direction.EN_TO_ES -> "English -> Spanish"
    Direction.AUTO -> "Auto-detect"
}

private fun mb(bytes: Long): Long = (bytes + 500_000) / 1_000_000

internal fun usd(v: Double): String = "$" + String.format(Locale.US, "%.3f", v)
