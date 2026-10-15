package com.xrmech.habla.data

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey

/**
 * Non-exportable AES-256-GCM key in the Android Keystore (D031).
 *
 * No StrongBox (TEE is enough, one generation path to test), no user authentication, no
 * unlocked-device requirement and no validity dates: the foreground service decrypts with the
 * screen locked and off. The key is generated lazily by the first [getOrCreate].
 */
internal class AndroidKeystoreKeyMaterial(private val alias: String = DataFiles.KEY_ALIAS) : KeyMaterial {

    private fun keyStore(): KeyStore = KeyStore.getInstance(PROVIDER).apply { load(null) }

    override fun existing(): SecretKey? = keyStore().getKey(alias, null) as? SecretKey

    override fun getOrCreate(): SecretKey = existing() ?: generate()

    override fun delete() {
        val store = keyStore()
        if (store.containsAlias(alias)) store.deleteEntry(alias)
    }

    private fun generate(): SecretKey {
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, PROVIDER)
        generator.init(
            KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(KEY_BITS)
                .build(),
        )
        return generator.generateKey()
    }

    private companion object {
        const val PROVIDER = "AndroidKeyStore"
        const val KEY_BITS = 256
    }
}
