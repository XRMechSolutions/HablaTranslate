package com.xrmech.habla.data

import javax.crypto.KeyGenerator
import javax.crypto.SecretKey

/** In-memory AES-256 key standing in for the Android Keystore in JVM tests. */
internal open class SoftwareKeyMaterial : KeyMaterial {
    var key: SecretKey? = null
    var deleteCalls = 0
        private set

    override fun existing(): SecretKey? = key

    override fun getOrCreate(): SecretKey =
        key ?: KeyGenerator.getInstance("AES").apply { init(256) }.generateKey().also { key = it }

    override fun delete() {
        deleteCalls++
        key = null
    }
}
