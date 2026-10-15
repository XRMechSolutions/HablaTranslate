package com.xrmech.habla.asr

import com.xrmech.habla.asr.internal.PyText
import com.xrmech.habla.shared.AsrProfile

/**
 * Ports of the server's `_is_bad_transcript` and `_has_clipped_onset`
 * (`habla/server/pipeline/audio_processor.py`), with Python string semantics via [PyText].
 * Pure, thread-safe, never throws.
 */
object TranscriptFilters {

    /**
     * True when the text looks like noise rather than speech: empty or whitespace-only, fewer than
     * `filters.min_letters` letters, or letters/alphanumerics below `filters.min_letter_ratio`.
     */
    fun isBadTranscript(text: String, profile: AsrProfile): Boolean {
        val stripped = PyText.strip(text)
        if (stripped.isEmpty()) return true
        var letters = 0
        var alnum = 0
        var i = 0
        while (i < stripped.length) {
            val cp = stripped.codePointAt(i)
            if (PyText.isAlpha(cp)) letters++
            if (PyText.isAlnum(cp)) alnum++
            i += Character.charCount(cp)
        }
        val filters = profile.filters
        if (letters < filters.minLetters) return true
        if (alnum > 0 && letters.toDouble() / alnum < filters.minLetterRatio) return true
        return false
    }

    /** True when the first word, casefolded, is a known clipped Spanish onset (e.g. `ola` for `hola`). */
    fun hasClippedOnset(text: String, profile: AsrProfile): Boolean {
        val words = PyText.split(PyText.strip(text))
        if (words.isEmpty()) return false
        val first = PyText.casefold(words[0])
        return profile.filters.clippedOnsetPairs.any { it.isNotEmpty() && first == it[0] }
    }
}
