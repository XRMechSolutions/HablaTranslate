package com.xrmech.habla.data

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import app.cash.turbine.test
import java.io.File
import java.io.IOException
import java.security.KeyStoreException
import java.security.ProviderException
import javax.crypto.SecretKey
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

@OptIn(ExperimentalCoroutinesApi::class)
class ApiKeyStoreCoreTest {

    @TempDir
    lateinit var tempDir: File

    // Obviously fake values; none matches the OpenAI key pattern the APK scan looks for.
    private val userKey = "fake-user-key-0001"
    private val seed = "fake-seed-key-0002"

    private fun TestScope.newStore(): DataStore<Preferences> =
        testPreferencesStore(backgroundScope, tempDir.resolve("secrets.preferences_pb"))

    private fun TestScope.core(
        store: DataStore<Preferences>,
        keys: KeyMaterial = SoftwareKeyMaterial(),
        seedKey: String? = null,
    ) = ApiKeyStoreCore(store, keys, seedKey, UnconfinedTestDispatcher(testScheduler), retryDelayMs = 100)

    private suspend fun DataStore<Preferences>.blob(): ByteArray? = data.first()[OPENAI_KEY_BLOB]

    private suspend fun DataStore<Preferences>.seeded(): Boolean? = data.first()[OPENAI_KEY_SEEDED_FROM_BUILD]

    private suspend fun DataStore<Preferences>.putBlob(bytes: ByteArray) = edit { it[OPENAI_KEY_BLOB] = bytes }

    @Test
    fun `set then get round trips and trims`() = runTest {
        val store = newStore()
        val core = core(store)
        core.set("  $userKey \n")
        assertEquals(userKey, core.get())
        assertNotNull(store.blob())
    }

    @Test
    fun `a new instance decrypts the stored blob`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        core(store, keys).set(userKey)
        assertEquals(userKey, core(store, keys).get())
    }

    @Test
    fun `stored blob is a v1 envelope, not plaintext`() = runTest {
        val store = newStore()
        core(store).set(userKey)
        val blob = store.blob()!!
        assertEquals(1, blob[0].toInt())
        assertEquals(2 + blob[1].toInt() + userKey.length + 16, blob.size)
        assertFalse(String(blob, Charsets.ISO_8859_1).contains(userKey))
    }

    @Test
    fun `invalid keys are rejected and nothing is stored`() = runTest {
        val store = newStore()
        val core = core(store)
        for (bad in listOf("", "   ", "fake key", "fake\nkey", "k".repeat(1025))) {
            assertThrows<IllegalArgumentException> { core.set(bad) }
        }
        assertNull(store.blob())
        assertNull(core.get())
    }

    @Test
    fun `get with no blob returns null and deletes nothing`() = runTest {
        val keys = SoftwareKeyMaterial()
        assertNull(core(newStore(), keys).get())
        assertEquals(0, keys.deleteCalls)
    }

    @Test
    fun `corrupt blobs self-heal to null`() = runTest {
        val corruptions: List<(ByteArray) -> ByteArray> = listOf(
            { it.copyOf(20) },
            { b -> b.copyOf().also { it[it.size - 1] = (it[it.size - 1].toInt() xor 0x01).toByte() } },
            { b -> b.copyOf().also { it[0] = 2 } },
        )
        for (corrupt in corruptions) {
            val store = newStoreNamed("c${corruptions.indexOf(corrupt)}")
            val keys = SoftwareKeyMaterial()
            core(store, keys).set(userKey)
            store.putBlob(corrupt(store.blob()!!))
            val fresh = core(store, keys)
            assertNull(fresh.get())
            assertNull(store.blob())
            assertEquals(1, keys.deleteCalls)
            assertNull(keys.key)
        }
    }

    private fun TestScope.newStoreNamed(name: String): DataStore<Preferences> =
        testPreferencesStore(backgroundScope, tempDir.resolve("$name.preferences_pb"))

    @Test
    fun `missing key material with a blob present clears the blob`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        core(store, keys).set(userKey)
        keys.key = null
        assertNull(core(store, keys).get())
        assertNull(store.blob())
        assertEquals(1, keys.deleteCalls)
    }

    @Test
    fun `transient keystore error is retried once after the delay`() = runTest {
        val store = newStore()
        val keys = FlakyKeyMaterial(failures = 1)
        core(store, keys).set(userKey)
        val start = currentTime
        assertEquals(userKey, core(store, keys).get())
        assertEquals(100, currentTime - start)
        assertNotNull(store.blob())
        assertEquals(0, keys.deleteCalls)
    }

    @Test
    fun `transient keystore error twice self-heals`() = runTest {
        val store = newStore()
        val keys = FlakyKeyMaterial(failures = 2)
        core(store, keys).set(userKey)
        assertNull(core(store, keys).get())
        assertNull(store.blob())
        assertEquals(1, keys.deleteCalls)
    }

    @Test
    fun `datastore read IOException gives null and deletes nothing`() = runTest {
        val keys = SoftwareKeyMaterial()
        keys.getOrCreate()
        val core = core(FailingReadStore(), keys)
        assertNull(core.get())
        assertEquals(0, keys.deleteCalls)
        assertNotNull(keys.key)
        core.hasKey.test {
            assertFalse(awaitItem())
            awaitComplete()
        }
    }

    @Test
    fun `clear twice is fine and get returns null`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        val core = core(store, keys)
        core.set(userKey)
        core.clear()
        core.clear()
        assertNull(core.get())
        assertNull(store.blob())
        assertNull(keys.key)
    }

    @Test
    fun `clear whose alias delete throws still drops the cached key`() = runTest {
        val store = newStore()
        val keys = FailingDeleteKeyMaterial()
        val core = core(store, keys)
        core.set(userKey)
        assertEquals(userKey, core.get())
        assertThrows<KeyStoreException> { core.clear() }
        assertNull(store.blob())
        assertNull(core.get())
    }

    @Test
    fun `set replaces the previous key`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        val core = core(store, keys)
        core.set(userKey)
        core.set("fake-user-key-0003")
        assertEquals("fake-user-key-0003", core.get())
        assertEquals("fake-user-key-0003", core(store, keys).get())
    }

    @Test
    fun `hasKey emits false, true, false across set and clear`() = runTest {
        val core = core(newStore())
        core.hasKey.test {
            assertFalse(awaitItem())
            core.set(userKey)
            assertTrue(awaitItem())
            core.clear()
            assertFalse(awaitItem())
            expectNoEvents()
        }
    }

    @Test
    fun `self-heal flips hasKey to false`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        core(store, keys).set(userKey)
        keys.key = null
        val core = core(store, keys)
        core.hasKey.test {
            assertTrue(awaitItem())
            assertNull(core.get())
            assertFalse(awaitItem())
        }
    }

    @Test
    fun `plaintext never appears in toString`() = runTest {
        val core = core(newStore())
        core.set(userKey)
        core.get()
        assertFalse(core.toString().contains(userKey))
        assertFalse(core.toString().contains("fake-user"))
    }

    @Test
    fun `envelope stored by set opens with the same key material`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        core(store, keys).set(userKey)
        val opened = SecretEnvelope.open(keys.key!!, store.blob()!!, SecretEnvelope.keyAliasAad)
        assertArrayEquals(userKey.toByteArray(), opened)
    }

    // --- personal-flavor seeding ---

    @Test
    fun `seeds once and sets the flag`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        assertEquals(seed, core(store, keys, seed).get())
        assertEquals(true, store.seeded())
        assertEquals(seed, core(store, keys, seed).get())
    }

    @Test
    fun `does not re-seed after clear`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        val first = core(store, keys, seed)
        assertEquals(seed, first.get())
        first.clear()
        val nextLaunch = core(store, keys, seed)
        assertNull(nextLaunch.get())
        assertNull(store.blob())
        assertEquals(true, store.seeded())
    }

    @Test
    fun `keeps a user key present before seeding and still sets the flag`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        core(store, keys).set(userKey)
        assertEquals(userKey, core(store, keys, seed).get())
        assertEquals(true, store.seeded())
    }

    @Test
    fun `invalid seed leaves the flag unset and a fixed seed works next launch`() = runTest {
        val store = newStore()
        val keys = SoftwareKeyMaterial()
        assertNull(core(store, keys, "fake seed with spaces").get())
        assertNull(store.seeded())
        assertNull(store.blob())
        assertEquals(seed, core(store, keys, seed).get())
        assertEquals(true, store.seeded())
    }

    @Test
    fun `null and blank seeds do nothing`() = runTest {
        val store = newStore()
        assertNull(core(store, seedKey = null).get())
        assertNull(core(store, seedKey = "  ").get())
        assertNull(store.seeded())
        assertNull(store.blob())
    }

    @Test
    fun `hasKey first collection with a valid seed emits true`() = runTest {
        val core = core(newStore(), seedKey = seed)
        core.hasKey.test {
            assertTrue(awaitItem())
            expectNoEvents()
        }
    }

    @Test
    fun `seeding failure leaves the flag unset and retries on the next operation`() = runTest {
        val store = newStore()
        val keys = FailingCreateKeyMaterial(failures = 1)
        val core = core(store, keys, seed)
        assertNull(core.get())
        assertNull(store.seeded())
        assertNull(store.blob())
        assertEquals(seed, core.get())
        assertEquals(true, store.seeded())
    }

    /** A DataStore whose reads fail with a disk error; writes are never expected. */
    private class FailingReadStore : DataStore<Preferences> {
        override val data: Flow<Preferences> = flow { throw IOException("disk read failed") }

        override suspend fun updateData(transform: suspend (t: Preferences) -> Preferences): Preferences =
            throw IOException("disk write failed")
    }

    /** Throws [ProviderException] from [existing] the first [failures] times after a key exists. */
    private class FlakyKeyMaterial(private var failures: Int) : SoftwareKeyMaterial() {
        override fun existing(): SecretKey? {
            if (key != null && failures > 0) {
                failures--
                throw ProviderException("keystore daemon hiccup")
            }
            return super.existing()
        }
    }

    /** Alias deletion always fails, as a Keystore daemon error would. */
    private class FailingDeleteKeyMaterial : SoftwareKeyMaterial() {
        override fun delete() = throw KeyStoreException("keystore daemon hiccup")
    }

    /** Throws [ProviderException] from [getOrCreate] the first [failures] times. */
    private class FailingCreateKeyMaterial(private var failures: Int) : SoftwareKeyMaterial() {
        override fun getOrCreate(): SecretKey {
            if (failures > 0) {
                failures--
                throw ProviderException("keystore daemon hiccup")
            }
            return super.getOrCreate()
        }
    }
}
