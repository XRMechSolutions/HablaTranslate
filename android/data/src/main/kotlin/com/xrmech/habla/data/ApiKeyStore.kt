package com.xrmech.habla.data

import kotlinx.coroutines.flow.Flow

/** The user's OpenAI key, encrypted at rest (D010, D031). */
interface ApiKeyStore {
    /** True while an encrypted blob is stored. Does not decrypt; runs the one-time seed check. */
    val hasKey: Flow<Boolean>

    /** The decrypted, trimmed key, or null when absent or undecryptable (the store is then cleared). */
    suspend fun get(): String?

    /** Trims, validates, encrypts and stores [key]. @throws IllegalArgumentException on a bad key. */
    suspend fun set(key: String)

    /** Deletes the ciphertext and the Keystore alias. Idempotent. */
    suspend fun clear()
}
