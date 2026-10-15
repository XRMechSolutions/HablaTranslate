package com.xrmech.habla.translate

/** Removes OpenAI API keys (and the masked fragments OpenAI echoes in 401 messages) from text. */
internal object Redact {
    private val KEY = Regex("""sk-[A-Za-z0-9_\-]{8,}""")

    fun secrets(text: String?): String {
        if (text.isNullOrEmpty()) return ""
        return KEY.replace(text, "sk-***")
    }
}
