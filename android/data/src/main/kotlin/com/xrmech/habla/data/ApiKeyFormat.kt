package com.xrmech.habla.data

/**
 * The one validation rule for API keys, used by `set` and by personal-flavor seeding. Printable
 * ASCII only keeps CR/LF out of the `Authorization` header. No `sk-` prefix check: key formats
 * change.
 */
internal object ApiKeyFormat {
    /** About 6x the longest current OpenAI key; only stops absurd pastes. */
    const val MAX_LENGTH = 1024

    /** @throws IllegalArgumentException when the trimmed key is blank, too long or has a bad char. */
    fun normalize(raw: String): String {
        val key = raw.trim()
        require(key.isNotEmpty()) { "API key is blank" }
        require(key.length <= MAX_LENGTH) { "API key is longer than $MAX_LENGTH characters" }
        require(key.all { it in '!'..'~' }) { "API key contains whitespace or control characters" }
        return key
    }
}
