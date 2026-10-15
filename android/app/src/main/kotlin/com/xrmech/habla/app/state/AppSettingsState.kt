package com.xrmech.habla.app.state

import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.shared.ModelTable
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
class AppSettingsState(repo: SettingsRepository, appScope: CoroutineScope) {
    // Collected for the life of the app scope; DataStore errors already emit defaults (data docs).
    val settings: StateFlow<AppSettings?> = repo.settings.stateIn(appScope, SharingStarted.Eagerly, null)

    suspend fun awaitLoaded(): AppSettings = settings.filterNotNull().first()

    /** Read per translation call by Translator/QuickTranslator; models.json defaults before load. */
    fun translateSettings(models: ModelTable): TranslateSettings {
        val s = settings.value ?: return TranslateSettings(models.finalDefault, models.quickDefault, partialTranslation = true)
        return TranslateSettings(s.finalModel, s.quickModel, s.partialTranslation)
    }
}
