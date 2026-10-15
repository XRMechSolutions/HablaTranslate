package com.xrmech.habla.data

import javax.crypto.SecretKey

/** Lifecycle of the AES key that wraps the API key. Android: [AndroidKeystoreKeyMaterial]. */
internal interface KeyMaterial {
    /** The key if it exists, else null (alias wiped by restore, transfer or a keystore reset). */
    fun existing(): SecretKey?

    /** The existing key, or a newly generated one. */
    fun getOrCreate(): SecretKey

    /** Deletes the key. A missing key is not an error. */
    fun delete()
}
