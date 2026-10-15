package com.xrmech.habla.data

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.byteArrayPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import java.io.IOException
import java.security.GeneralSecurityException
import java.security.KeyStoreException
import java.security.ProviderException
import java.security.UnrecoverableKeyException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

internal val OPENAI_KEY_BLOB = byteArrayPreferencesKey("openai_key_blob")
internal val OPENAI_KEY_SEEDED_FROM_BUILD = booleanPreferencesKey("openai_key_seeded_from_build")

/**
 * Pure-JVM key store logic over three seams (DataStore, [KeyMaterial], [SecretEnvelope]).
 *
 * One [Mutex] serializes get/set/clear and seeding, so the in-memory cache always matches the
 * stored blob. Any decrypt failure deletes the blob and then the key (self-heal, D031); transient
 * keystore-daemon errors get one retry first. [seedKey] is the personal-flavor build seed (D022),
 * stored at most once per install.
 */
internal class ApiKeyStoreCore(
    private val store: DataStore<Preferences>,
    private val keys: KeyMaterial,
    private val seedKey: String?,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val retryDelayMs: Long = 100,
) : ApiKeyStore {

    private val mutex = Mutex()

    // Guarded by mutex.
    private var cached: String? = null
    private var seedChecked = false

    override val hasKey: Flow<Boolean> =
        store.data
            .catch { if (it is IOException) emit(emptyPreferences()) else throw it }
            .map { it[OPENAI_KEY_BLOB] != null }
            .distinctUntilChanged()
            .onStart { locked { } }

    override suspend fun get(): String? = locked {
        cached?.let { return@locked it }
        val blob = try {
            store.data.first()[OPENAI_KEY_BLOB]
        } catch (e: IOException) {
            // An I/O error is not proof of a bad blob: keep everything and let the next call retry.
            DataLog.w("secrets read failed: ${e.javaClass.simpleName}")
            null
        } ?: return@locked null
        decrypt(blob)
    }

    override suspend fun set(key: String) {
        val normalized = ApiKeyFormat.normalize(key)
        locked {
            val blob = seal(normalized)
            store.edit { it[OPENAI_KEY_BLOB] = blob }
            cached = normalized
        }
    }

    override suspend fun clear() {
        locked {
            // Cache first: if the blob removal or the alias delete throws, get() must not keep
            // returning the old plaintext from memory.
            cached = null
            store.edit { it.remove(OPENAI_KEY_BLOB) }
            keys.delete()
        }
    }

    override fun toString(): String = "ApiKeyStoreCore"

    /** Runs [block] under the mutex on the dispatcher, after the one-time seed check. */
    private suspend fun <T> locked(block: suspend () -> T): T = mutex.withLock {
        withContext(dispatcher) {
            seedOnce()
            block()
        }
    }

    private fun seal(plaintext: String): ByteArray =
        SecretEnvelope.seal(keys.getOrCreate(), plaintext.toByteArray(Charsets.UTF_8), SecretEnvelope.keyAliasAad)

    private suspend fun decrypt(blob: ByteArray): String? {
        var retried = false
        while (true) {
            try {
                val key = keys.existing()
                if (key == null) {
                    selfHeal("key material missing")
                    return null
                }
                val plaintext = String(SecretEnvelope.open(key, blob, SecretEnvelope.keyAliasAad), Charsets.UTF_8)
                cached = plaintext
                return plaintext
            } catch (e: Exception) {
                when {
                    e is CancellationException -> throw e
                    isTransient(e) && !retried -> {
                        retried = true
                        delay(retryDelayMs)
                    }
                    isTransient(e) || e is GeneralSecurityException || e is IllegalArgumentException -> {
                        selfHeal(e.javaClass.simpleName)
                        return null
                    }
                    else -> throw e
                }
            }
        }
    }

    private fun isTransient(e: Exception): Boolean =
        e is KeyStoreException || e is UnrecoverableKeyException || e is ProviderException

    /**
     * Ciphertext goes first, so a crash in between leaves only a harmless orphan alias. The seeded
     * flag is never touched: a self-heal must not re-arm the build seed.
     */
    private suspend fun selfHeal(reason: String) {
        cached = null
        try {
            store.edit { it.remove(OPENAI_KEY_BLOB) }
            keys.delete()
            DataLog.w("stored API key unusable ($reason); cleared stored key")
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            DataLog.w("self-heal after $reason failed: ${e.javaClass.simpleName}")
        }
    }

    /**
     * Stores the build seed once per install (D022/D031). Runs inside the mutex at the start of the
     * first operation, so the UI never sees "no key" before a seed lands. `seedChecked` is set only
     * when a decision is reached; a Keystore or DataStore failure leaves it false so the next
     * operation retries, and the persisted flag is not written.
     */
    private suspend fun seedOnce() {
        if (seedChecked) return
        val seed = seedKey
        if (seed.isNullOrBlank()) {
            seedChecked = true
            return
        }
        try {
            val prefs = store.data.first()
            if (prefs[OPENAI_KEY_SEEDED_FROM_BUILD] == true) {
                seedChecked = true
                return
            }
            val normalized = try {
                ApiKeyFormat.normalize(seed)
            } catch (e: IllegalArgumentException) {
                // Flag stays unset so a rebuild with a fixed android/.env seeds on the next launch.
                DataLog.w("seed key rejected (format)")
                seedChecked = true
                return
            }
            if (prefs[OPENAI_KEY_BLOB] == null) {
                val blob = seal(normalized)
                store.edit {
                    it[OPENAI_KEY_BLOB] = blob
                    it[OPENAI_KEY_SEEDED_FROM_BUILD] = true
                }
                cached = normalized
                DataLog.i("build seed stored")
            } else {
                // The user's key wins; the flag still records that the seed was considered.
                store.edit { it[OPENAI_KEY_SEEDED_FROM_BUILD] = true }
            }
            seedChecked = true
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            DataLog.w("seeding failed: ${e.javaClass.simpleName}")
        }
    }
}
