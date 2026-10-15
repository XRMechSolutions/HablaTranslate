package com.xrmech.habla.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.preferencesDataStore

/**
 * `files/datastore/secrets.preferences_pb`. A top-level delegate gives exactly one instance per
 * process, as DataStore requires. A corrupt file is replaced with empty preferences: the blob is
 * then gone (the key must be re-entered or re-seeded) and the orphan alias is reused by the next set.
 */
internal val Context.secretsDataStore: DataStore<Preferences> by preferencesDataStore(
    name = DataFiles.SECRETS_STORE,
    corruptionHandler = ReplaceFileCorruptionHandler { emptyPreferences() },
)
