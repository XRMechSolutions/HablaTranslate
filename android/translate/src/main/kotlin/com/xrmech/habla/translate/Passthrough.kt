package com.xrmech.habla.translate

import java.util.Locale

/** Port of the server's `_is_passthrough`: did the model echo the source instead of translating? */
internal object Passthrough {
    // Python `[^\w\s]` on str: \w is letters, numbers and underscore; \s is str.isspace.
    private val NON_WORD = Regex("[^\\p{L}\\p{N}_${PyText.WS_CLASS}]")
    private val SPANISH_CHARS = Regex("[áéíóúüñ¿¡]")
    private val LETTERS = Regex("[a-záéíóúüñ]")

    fun isPassthrough(source: String, translated: String, direction: String): Boolean {
        if (source.isEmpty() || translated.isEmpty()) return false
        val src = words(source)
        val tgt = words(translated)
        if (src.isEmpty()) return false
        val overlap = src.count { it in tgt }.toDouble() / src.size
        if (overlap > 0.7) return true

        if (direction.endsWith("_to_en")) {
            val lower = translated.lowercase(Locale.ROOT)
            val spanish = SPANISH_CHARS.findAll(lower).count()
            val totalAlpha = maxOf(LETTERS.findAll(lower).count(), 1)
            if (spanish.toDouble() / totalAlpha > 0.02 && spanish > 3) return true
        }
        return false
    }

    private fun words(text: String): Set<String> =
        PyText.split(NON_WORD.replace(text.lowercase(Locale.ROOT), "")).toSet()
}
