package com.xrmech.habla.app.di

import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.ui.history.ExportController
import com.xrmech.habla.data.TrainingAudioStore
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.CostMeter
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

sealed interface StartupState {
    data object Loading : StartupState
    data object Ready : StartupState

    /** Shared assets failed to parse: the fatal screen shows [message] (edge case "Shared assets damaged"). */
    data class Fatal(val message: String) : StartupState
}

/**
 * Startup work after Koin is up (design section 10): shared-assets warm-up gates the UI, then the
 * training-file sweep, the all-time cost seed and the export cache cleanup run in the background.
 * Dependencies are resolved lazily so a damaged asset set never touches the classes that need it.
 */
class AppStartup(
    private val assets: () -> SharedAssets,
    private val training: () -> TrainingAudioStore,
    private val costMeter: () -> CostMeter,
    private val usage: () -> UsageRepository,
    private val cacheDir: File,
    private val scope: CoroutineScope,
    /** Reads the on-device language pack state from disk (D050), so the first preview can use it. */
    private val languagePack: suspend () -> Unit = {},
) {
    private val _state = MutableStateFlow<StartupState>(StartupState.Loading)
    val state: StateFlow<StartupState> = _state.asStateFlow()

    fun run() {
        scope.launch(Dispatchers.IO) {
            try {
                assets()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // SharedAssetException (validation) or an ExceptionInInitializerError cause from ICU-only regex.
                HLog.e(HLog.UI, "shared assets failed to load", e)
                _state.value = StartupState.Fatal(e.message ?: e.javaClass.simpleName)
                return@launch
            } catch (e: ExceptionInInitializerError) {
                HLog.e(HLog.UI, "shared assets failed to load", e)
                _state.value = StartupState.Fatal(e.cause?.message ?: e.javaClass.simpleName)
                return@launch
            }
            _state.value = StartupState.Ready
            step("training sweep") { HLog.i(HLog.UI, "swept ${training().sweepOrphans()} orphan training files") }
            step("cost seed") { costMeter().seedTotal(usage().totalCostUsd().first()) }
            step("export cleanup") { ExportController.cleanCache(cacheDir) }
            step("language pack check") { languagePack() }
        }
    }

    private suspend fun step(name: String, block: suspend () -> Unit) {
        try {
            block()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            HLog.w(HLog.UI, "startup $name failed: ${e.message}")
        }
    }
}
