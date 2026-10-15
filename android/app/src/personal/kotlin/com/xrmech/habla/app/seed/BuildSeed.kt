package com.xrmech.habla.app.seed

import com.xrmech.habla.app.BuildConfig

/** Personal flavor only (D022): the owner's key from android/.env, seeded once into the key store. */
object BuildSeed {
    val key: String? = BuildConfig.SEED_OPENAI_KEY.ifBlank { null }
}
