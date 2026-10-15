package com.xrmech.habla.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.preferencesDataStore
import java.io.IOException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map

/**
 * `files/datastore/settings.preferences_pb`, one instance per process. A corrupt file becomes empty
 * preferences, so every setting reads as its default (first-run screens show again; accepted).
 */
internal val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(
    name = DataFiles.SETTINGS_STORE,
    corruptionHandler = ReplaceFileCorruptionHandler { emptyPreferences() },
)

/**
 * Settings in a Preferences DataStore. [defaults] comes from `:app` (built from SharedAssets), which
 * keeps `:data` free of `:shared-assets`.
 */
class DataStoreSettingsRepository internal constructor(
    private val store: DataStore<Preferences>,
    private val defaults: AppSettings,
) : SettingsRepository {

    constructor(context: Context, defaults: AppSettings) : this(context.applicationContext.settingsDataStore, defaults)

    override val settings: Flow<AppSettings> =
        store.data
            .catch { if (it is IOException) emit(emptyPreferences()) else throw it }
            .map { SettingsCodec.decode(it, defaults) }
            .distinctUntilChanged()

    /** DataStore runs `edit` blocks one at a time on the latest data, so concurrent updates both apply. */
    override suspend fun update(transform: (AppSettings) -> AppSettings) {
        store.edit { p -> SettingsCodec.encode(transform(SettingsCodec.decode(p, defaults)), p) }
    }
}
