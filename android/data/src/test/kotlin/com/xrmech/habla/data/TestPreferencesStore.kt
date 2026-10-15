package com.xrmech.habla.data

import androidx.datastore.core.DataStore
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler
import androidx.datastore.core.okio.OkioStorage
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.PreferencesSerializer
import androidx.datastore.preferences.core.emptyPreferences
import java.io.File
import kotlinx.coroutines.CoroutineScope
import okio.FileSystem
import okio.Path.Companion.toOkioPath

/**
 * A temp-file Preferences DataStore for JVM tests, with the same serializer and corruption handler
 * production uses. Okio storage replaces the file with an atomic move; DataStore's `File`-based
 * storage cannot overwrite an existing file on a Windows host (`File.renameTo` onto an existing
 * target fails there), a host quirk that does not exist on Android.
 */
internal fun testPreferencesStore(scope: CoroutineScope, file: File): DataStore<Preferences> =
    PreferenceDataStoreFactory.create(
        storage = OkioStorage(FileSystem.SYSTEM, PreferencesSerializer) { file.toOkioPath() },
        corruptionHandler = ReplaceFileCorruptionHandler { emptyPreferences() },
        scope = scope,
    )
