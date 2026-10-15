package com.xrmech.habla.data

import android.content.Context
import android.security.keystore.KeyInfo
import android.util.Log
import androidx.datastore.preferences.core.edit
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.security.KeyStore
import javax.crypto.SecretKey
import javax.crypto.SecretKeyFactory
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** The real AndroidKeyStore path behind [KeystoreApiKeyStore] (D031). */
@RunWith(AndroidJUnit4::class)
class KeystoreApiKeyStoreTest {
    private val context: Context = ApplicationProvider.getApplicationContext()

    // Obviously fake; does not match the OpenAI key pattern the APK scan looks for.
    private val fakeKey = "fake-device-test-key-0001"

    private fun androidKeyStore(): KeyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }

    private fun reset() = runBlocking {
        context.secretsDataStore.edit { it.clear() }
        val ks = androidKeyStore()
        if (ks.containsAlias(DataFiles.KEY_ALIAS)) ks.deleteEntry(DataFiles.KEY_ALIAS)
    }

    @Before
    fun setUp() = reset()

    @After
    fun tearDown() = reset()

    @Test
    fun roundTripThroughTheRealKeystore() = runBlocking {
        KeystoreApiKeyStore(context).set("  $fakeKey  ")
        // A new instance has an empty cache, so this decrypts with the Keystore key.
        assertEquals(fakeKey, KeystoreApiKeyStore(context).get())
        assertTrue(KeystoreApiKeyStore(context).hasKey.first())
    }

    @Test
    fun aliasExistsAfterSetAndIsGoneAfterClear() = runBlocking {
        val store = KeystoreApiKeyStore(context)
        store.set(fakeKey)
        assertTrue(androidKeyStore().containsAlias("habla_api_key_v1"))
        logSecurityLevel()
        store.clear()
        assertFalse(androidKeyStore().containsAlias("habla_api_key_v1"))
        assertNull(store.get())
        assertFalse(store.hasKey.first())
        store.clear()
    }

    @Test
    fun externallyDeletedAliasGivesNullAndRemovesTheBlob() = runBlocking {
        KeystoreApiKeyStore(context).set(fakeKey)
        androidKeyStore().deleteEntry(DataFiles.KEY_ALIAS)
        val fresh = KeystoreApiKeyStore(context)
        assertNull(fresh.get())
        assertNull(context.secretsDataStore.data.first()[OPENAI_KEY_BLOB])
        assertFalse(fresh.hasKey.first())
    }

    @Test
    fun flippedBlobBytesGiveNull() = runBlocking {
        KeystoreApiKeyStore(context).set(fakeKey)
        val blob = context.secretsDataStore.data.first()[OPENAI_KEY_BLOB]
        assertNotNull(blob)
        val flipped = blob!!.copyOf().also { it[it.size - 1] = (it[it.size - 1].toInt() xor 0x01).toByte() }
        context.secretsDataStore.edit { it[OPENAI_KEY_BLOB] = flipped }
        val fresh = KeystoreApiKeyStore(context)
        assertNull(fresh.get())
        assertNull(context.secretsDataStore.data.first()[OPENAI_KEY_BLOB])
        assertFalse(androidKeyStore().containsAlias(DataFiles.KEY_ALIAS))
    }

    /** Emulators report SOFTWARE; logged for the record, never asserted. */
    private fun logSecurityLevel() {
        val key = androidKeyStore().getKey(DataFiles.KEY_ALIAS, null) as SecretKey
        val info = SecretKeyFactory.getInstance(key.algorithm, "AndroidKeyStore")
            .getKeySpec(key, KeyInfo::class.java) as KeyInfo
        val level = if (android.os.Build.VERSION.SDK_INT >= 31) info.securityLevel.toString() else "pre-31 isInsideSecureHardware=${legacyInsideSecureHardware(info)}"
        Log.i("HablaDataTest", "habla_api_key_v1 security level: $level")
    }

    @Suppress("DEPRECATION")
    private fun legacyInsideSecureHardware(info: KeyInfo): Boolean = info.isInsideSecureHardware
}
