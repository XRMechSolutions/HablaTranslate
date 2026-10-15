package com.xrmech.habla.app.fakes

import com.xrmech.habla.data.ApiKeyStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map

/** In-memory key store with the real store's blank rejection; [gets] counts reads. */
class FakeApiKeyStore(initial: String? = null) : ApiKeyStore {
    private val key = MutableStateFlow(initial)
    var gets = 0
        private set

    override val hasKey: Flow<Boolean> = key.map { it != null }

    override suspend fun get(): String? {
        gets++
        return key.value
    }

    override suspend fun set(key: String) {
        val trimmed = key.trim()
        require(trimmed.isNotEmpty()) { "key is blank" }
        this.key.value = trimmed
    }

    override suspend fun clear() {
        key.value = null
    }

    fun peek(): String? = key.value
}
