package com.xrmech.habla.asr.internal

import java.util.Locale

/**
 * Python 3 `str` semantics the server's ASR filters rely on, over code points (edge-cases.md
 * "Python string semantics"). Kotlin's `trim()` misses U+0085, Java regex `\s` is ASCII-only and
 * `isLetterOrDigit` misses Nl/No numerics, so none of those are used here.
 */
internal object PyText {

    /** Python `str.isspace` for one code point. */
    fun isSpace(cp: Int): Boolean = when (cp) {
        in 0x09..0x0D, in 0x1C..0x1F, 0x20, 0x85, 0xA0, 0x1680, in 0x2000..0x200A,
        0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
        -> true
        else -> false
    }

    /** Python `str.isalpha`: general category Lu, Ll, Lt, Lm, Lo. */
    fun isAlpha(cp: Int): Boolean = when (Character.getType(cp).toByte()) {
        Character.UPPERCASE_LETTER, Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER,
        Character.MODIFIER_LETTER, Character.OTHER_LETTER,
        -> true
        else -> false
    }

    /** Python `str.isalnum`: isAlpha or a numeric category (Nd, Nl, No). */
    fun isAlnum(cp: Int): Boolean = isAlpha(cp) || when (Character.getType(cp).toByte()) {
        Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
        else -> false
    }

    /** Python `str.strip()` with no arguments. */
    fun strip(s: String): String {
        var start = 0
        var end = s.length
        while (start < end) {
            val cp = s.codePointAt(start)
            if (!isSpace(cp)) break
            start += Character.charCount(cp)
        }
        while (end > start) {
            val cp = s.codePointBefore(end)
            if (!isSpace(cp)) break
            end -= Character.charCount(cp)
        }
        return s.substring(start, end)
    }

    /** Python `str.split()` with no arguments: split on whitespace runs, no empty parts. */
    fun split(s: String): List<String> {
        val parts = ArrayList<String>()
        var i = 0
        var wordStart = -1
        while (i < s.length) {
            val cp = s.codePointAt(i)
            if (isSpace(cp)) {
                if (wordStart >= 0) {
                    parts += s.substring(wordStart, i)
                    wordStart = -1
                }
            } else if (wordStart < 0) {
                wordStart = i
            }
            i += Character.charCount(cp)
        }
        if (wordStart >= 0) parts += s.substring(wordStart)
        return parts
    }

    /** Python `str.casefold`: root-locale lowercase plus the full foldings that differ from it. */
    fun casefold(s: String): String {
        val lower = s.lowercase(Locale.ROOT)
        if (lower.none { it in FOLD_EXPANSIONS }) return lower
        val out = StringBuilder(lower.length + 4)
        for (ch in lower) out.append(FOLD_EXPANSIONS[ch] ?: ch)
        return out.toString()
    }

    /** Code-point count (Python `len`). */
    fun length(s: String): Int = s.codePointCount(0, s.length)

    private val FOLD_EXPANSIONS: Map<Char, String> = mapOf(
        'ß' to "ss",       // ß (ẞ lowercases to ß first)
        'ς' to "σ",   // ς -> σ
        'ﬀ' to "ff",
        'ﬁ' to "fi",
        'ﬂ' to "fl",
        'ﬃ' to "ffi",
        'ﬄ' to "ffl",
        'ﬅ' to "st",
        'ﬆ' to "st",
        'ŉ' to "ʼn",  // ŉ -> ʼn
    )
}
