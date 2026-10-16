package com.xrmech.habla.app.ui.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.pipeline.LiveController
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.state.ModelDownloadController
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.data.ApiKeyStore
import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.KeyTestResult
import com.xrmech.habla.translate.OpenAiProvider
import com.xrmech.habla.translate.model.CostTotals
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Seam over [OpenAiProvider.testKey] (a final class) so the ViewModel is testable without HTTP. */
fun interface KeyTester {
    suspend fun test(candidate: String?): KeyTestResult

    companion object {
        fun of(provider: OpenAiProvider): KeyTester = KeyTester { provider.testKey(it) }
    }
}

data class ModelOption(val id: String, val label: String)

data class SpeechModelOption(val id: String, val label: String, val sizeBytes: Long)

/** Transient Settings state owned by the ViewModel; the key text itself is never stored here. */
data class SettingsLocal(
    val keyError: String? = null,
    val keyTestRunning: Boolean = false,
    val keyTestMessage: String? = null,
    val modelMessage: String? = null,
    val confirmMeteredDownload: Boolean = false,
)

data class SettingsUiState(
    val keyStatus: String = "",
    val keyState: KeyState = KeyState.UNKNOWN,
    val settings: AppSettings? = null,
    val finalModels: List<ModelOption> = emptyList(),
    val quickModels: List<ModelOption> = emptyList(),
    val speechModels: List<SpeechModelOption> = emptyList(),
    val modelState: ModelState = ModelState.Missing,
    val downloadRunning: Boolean = false,
    val sessionActive: Boolean = false,
    val costs: CostTotals = CostTotals(),
    val metricsLine: String? = null,
    val local: SettingsLocal = SettingsLocal(),
) {
    val selectedSpeechModel: SpeechModelOption? get() = speechModels.firstOrNull { it.id == settings?.asrModelId }
}

sealed interface SettingsEffect {
    /** Save succeeded: the screen clears its key field. */
    data object KeySaved : SettingsEffect
}

/** Settings screen (design section 6, Settings). */
class SettingsViewModel(
    private val settings: SettingsRepository,
    private val keyStore: ApiKeyStore,
    private val tester: KeyTester,
    private val costMeter: CostMeter,
    private val usage: UsageRepository,
    private val keyHealth: KeyHealthTracker,
    models: ModelManager,
    private val downloads: ModelDownloadController,
    private val assets: SharedAssets,
    private val live: LiveController,
) : ViewModel() {
    private val local = MutableStateFlow(SettingsLocal())
    private val _effects = Channel<SettingsEffect>(Channel.BUFFERED)
    val effects: Flow<SettingsEffect> = _effects.receiveAsFlow()

    private val finalOptions = assets.models.models.map { (id, info) ->
        ModelOption(id, if (info.deprecated != null) "$id (deprecated ${info.deprecated})" else id)
    }
    // Partial-only catalog entries (base, D044) are never offered as the speech model.
    private val speechOptions = assets.asrModels.models.filter { it.usableForFinals }.map {
        SpeechModelOption(it.id, "${it.base} ${it.quantization} (${(it.sizeBytes + 500_000) / 1_000_000} MB)", it.sizeBytes)
    }

    private val base = combine(settings.settings, keyHealth.state, models.state, downloads.running) { s, k, m, d ->
        SettingsUiState(
            keyStatus = keyStatusText(k), keyState = k, settings = s,
            finalModels = finalOptions, quickModels = finalOptions, speechModels = speechOptions,
            modelState = m, downloadRunning = d,
        )
    }

    val ui: StateFlow<SettingsUiState> = combine(base, live.state, costMeter.totals, live.lastMetricsLine, local) { b, l, c, m, loc ->
        b.copy(sessionActive = l.phase.active, costs = c, metricsLine = m, local = loc)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SettingsUiState())

    init {
        // Show Ready/Missing for the selected model from disk; never touch a running download.
        viewModelScope.launch {
            val id = settings.settings.first().asrModelId
            if (!downloads.running.value) downloads.refresh(id)
        }
    }

    // ---- OpenAI key -------------------------------------------------------------------------

    fun saveKey(text: String) {
        if (text.isBlank()) {
            local.update { it.copy(keyError = ENTER_A_KEY, keyTestMessage = null) }
            return
        }
        viewModelScope.launch {
            try {
                keyStore.set(text)
                keyHealth.reset()
                local.update { it.copy(keyError = null, keyTestMessage = null) }
                _effects.send(SettingsEffect.KeySaved)
                HLog.i(HLog.KEY, "key saved")
            } catch (e: CancellationException) {
                throw e
            } catch (e: IllegalArgumentException) {
                // The store's messages describe the format, never the key itself.
                local.update { it.copy(keyError = if (text.isBlank()) ENTER_A_KEY else e.message ?: "Invalid key") }
            } catch (e: Exception) {
                HLog.w(HLog.KEY, "key save failed", e)
                local.update { it.copy(keyError = "Couldn't save the key: ${e.message ?: e::class.java.simpleName}") }
            }
        }
    }

    /** [candidate] = the field text when non-blank (test before saving), else the stored key. */
    fun testKey(candidate: String?) {
        if (local.value.keyTestRunning) return
        val typed = candidate?.takeIf { it.isNotBlank() }?.trim()
        local.update { it.copy(keyTestRunning = true, keyTestMessage = null, keyError = null) }
        viewModelScope.launch {
            val result = try {
                tester.test(typed)
            } finally {
                local.update { it.copy(keyTestRunning = false) }
            }
            local.update { it.copy(keyTestMessage = KeyTestMessages.text(result, assets.models.quickDefault)) }
            when (result) {
                is KeyTestResult.Valid -> {
                    if (typed == null) keyHealth.report(null)
                    recordTestUsage(result)
                }
                // Only the stored key's health is tracked; a typed, unsaved candidate says nothing about it.
                is KeyTestResult.Failed -> if (typed == null) keyHealth.report(result.error.kind)
            }
        }
    }

    private suspend fun recordTestUsage(result: KeyTestResult.Valid) {
        // Not through CostMeter.record: during a session the coordinator would persist that record
        // under the session id. A key test has no session (edge case "Key test while a session runs").
        val cost = costMeter.costOf(result.usage)
        costMeter.seedTotal(costMeter.totals.value.totalCostUsd + cost)
        try {
            usage.record(null, result.usage, UsageKind.TEST, cost)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            HLog.w(HLog.KEY, "test usage not recorded: ${e.message}")
        }
    }

    fun clearKey() {
        viewModelScope.launch {
            keyStore.clear()
            keyHealth.reset()
            local.update { it.copy(keyError = null, keyTestMessage = null) }
            HLog.i(HLog.KEY, "key cleared")
        }
    }

    // ---- Translation and listening ----------------------------------------------------------

    fun setFinalModel(id: String) = edit { it.copy(finalModel = id) }
    fun setQuickModel(id: String) = edit { it.copy(quickModel = id) }
    fun setPartialTranslation(on: Boolean) = edit { it.copy(partialTranslation = on) }
    fun setPartialsEnabled(on: Boolean) = edit { it.copy(partialsEnabled = on) }
    fun setDirection(d: Direction) = edit { it.copy(direction = d) }
    fun setMode(m: Mode) = edit { it.copy(mode = m) }
    fun setTrainingCapture(on: Boolean) = edit { it.copy(trainingCapture = on) }

    fun setAsrModel(id: String) {
        if (assets.asrModels.byId(id) == null) return
        viewModelScope.launch {
            settings.update { it.copy(asrModelId = id) }
            if (!downloads.running.value) downloads.refresh(id)
        }
    }

    private fun edit(transform: (AppSettings) -> AppSettings) {
        viewModelScope.launch { settings.update(transform) }
    }

    // ---- Speech model -----------------------------------------------------------------------

    /** On a metered network the first call only asks; [confirmed] = the user accepted. */
    fun downloadModel(confirmed: Boolean = false) {
        if (!confirmed && downloads.isMetered()) {
            local.update { it.copy(confirmMeteredDownload = true) }
            return
        }
        local.update { it.copy(confirmMeteredDownload = false, modelMessage = null) }
        viewModelScope.launch { downloads.download(settings.settings.first().asrModelId) }
    }

    fun dismissMeteredDownload() = local.update { it.copy(confirmMeteredDownload = false) }

    fun cancelDownload() = downloads.cancel()

    fun deleteModel() {
        if (live.state.value.phase.active) {
            local.update { it.copy(modelMessage = DELETE_WHILE_LISTENING) }
            return
        }
        viewModelScope.launch {
            downloads.delete(settings.settings.first().asrModelId)
            local.update { it.copy(modelMessage = null) }
        }
    }

    companion object {
        const val ENTER_A_KEY = "Enter a key"
        const val DELETE_WHILE_LISTENING = "Stop listening before deleting the speech model"

        fun keyStatusText(k: KeyState): String = when (k) {
            KeyState.UNKNOWN -> "Checking..."
            KeyState.ABSENT -> "No key"
            KeyState.PRESENT -> "Key saved"
            KeyState.INVALID -> "Key rejected"
            KeyState.NO_QUOTA -> "No credit"
        }
    }
}
