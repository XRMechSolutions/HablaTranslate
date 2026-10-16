package com.xrmech.habla.app.state

import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.EngineChoice
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.shared.ModelTable
import com.xrmech.habla.translate.TranslateEngine
import com.xrmech.habla.translate.TranslateSettings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn

/**
 * App-wide view of the DataStore settings. [settings] is null until DataStore's first real
 * emission, so nothing acts on defaults that are not yet the user's values.
 */
class AppSettingsState(
    repo: SettingsRepository,
    appScope: CoroutineScope,
    /** Whether an OpenAI key is stored (resolves EngineChoice.AUTO, D050). */
    private val hasKey: () -> Boolean = { true },
    /** Whether the on-device language pack is ready. */
    private val packReady: () -> Boolean = { false },
    /** Whether the signed-in user has an active Pro entitlement (D051). */
    private val proActive: () -> Boolean = { false },
) {
    // Collected for the life of the app scope; DataStore errors already emit defaults (data docs).
    val settings: StateFlow<AppSettings?> = repo.settings.stateIn(appScope, SharingStarted.Eagerly, null)

    suspend fun awaitLoaded(): AppSettings = settings.filterNotNull().first()

    /** Read per translation call by Translator/QuickTranslator; models.json defaults before load. */
    fun translateSettings(models: ModelTable): TranslateSettings {
        val s = settings.value ?: return TranslateSettings(models.finalDefault, models.quickDefault, partialTranslation = true)
        val (final, preview) = Engines.resolve(s.finalEngine, s.onDevicePreviews, hasKey(), packReady(), proActive())
        return TranslateSettings(s.finalModel, s.quickModel, s.partialTranslation, final, preview)
    }
}

/** D050/D051 engine resolution, pure so it can be tested on its own. */
object Engines {
    /**
     * Finals: an explicit choice is taken as chosen (PRO without an entitlement still goes to the
     * gateway, which refuses, and the phone takes over). AUTO is Pro when active, else the user's
     * key, else the phone. Previews never use the gateway (D007): the cloud only when finals use the
     * user's key and phone previews are off or the pack is not ready; otherwise the phone.
     */
    fun resolve(
        choice: EngineChoice,
        onDevicePreviews: Boolean,
        hasKey: Boolean,
        packReady: Boolean,
        proActive: Boolean = false,
    ): Pair<TranslateEngine, TranslateEngine> {
        val final = when (choice) {
            EngineChoice.CLOUD -> TranslateEngine.CLOUD
            EngineChoice.ON_DEVICE -> TranslateEngine.ON_DEVICE
            EngineChoice.PRO -> TranslateEngine.GATEWAY
            EngineChoice.AUTO -> when {
                proActive -> TranslateEngine.GATEWAY
                hasKey -> TranslateEngine.CLOUD
                else -> TranslateEngine.ON_DEVICE
            }
        }
        val preview = if (final != TranslateEngine.CLOUD || (onDevicePreviews && packReady)) TranslateEngine.ON_DEVICE else TranslateEngine.CLOUD
        return final to preview
    }
}
