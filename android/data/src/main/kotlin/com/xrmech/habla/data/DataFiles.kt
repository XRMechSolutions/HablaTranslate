package com.xrmech.habla.data

/**
 * Every on-disk name `:data` owns. The backup rule XML files repeat these literals; `BackupRulesTest`
 * keeps the two in step.
 */
object DataFiles {
    /** SQLDelight database, `databases/habla.db`. */
    const val DATABASE_NAME = "habla.db"

    /** Settings DataStore, `files/datastore/settings.preferences_pb`. */
    const val SETTINGS_STORE = "settings"

    /** Secrets DataStore, `files/datastore/secrets.preferences_pb`. */
    const val SECRETS_STORE = "secrets"

    /** Training audio root, `files/sessions/<sessionId>/segment_NNN.wav`. */
    const val SESSIONS_DIR = "sessions"

    /** Android Keystore alias of the AES key that wraps the API key; also the envelope AAD. */
    const val KEY_ALIAS = "habla_api_key_v1"
}
