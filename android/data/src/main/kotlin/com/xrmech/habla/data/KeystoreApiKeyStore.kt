package com.xrmech.habla.data

import android.content.Context

/**
 * The production [ApiKeyStore]: the OpenAI key sealed with an Android Keystore AES-256-GCM key and
 * stored as ciphertext in the `secrets` DataStore (D010, D031).
 *
 * [seedKey] is the personal-flavor build seed (`:app` passes `BuildSeed.key`; null in `public`).
 * It is stored at most once per install, inside the first store operation (D022). `:data` itself
 * generates no build-config class and never contains a key literal.
 */
class KeystoreApiKeyStore private constructor(core: ApiKeyStoreCore) : ApiKeyStore by core {
    constructor(context: Context, seedKey: String? = null) : this(
        ApiKeyStoreCore(
            store = context.applicationContext.secretsDataStore,
            keys = AndroidKeystoreKeyMaterial(),
            seedKey = seedKey,
        ),
    )

    override fun toString(): String = "KeystoreApiKeyStore"
}
