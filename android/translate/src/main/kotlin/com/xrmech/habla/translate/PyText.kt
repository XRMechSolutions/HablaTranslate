package com.xrmech.habla.translate

/**
 * Python `str` whitespace semantics (`str.isspace`, `strip()`, `split()`, regex `\s`), which the
 * server ports depend on. Kotlin's `trim()` differs on U+0085, and Java's `\s` differs on
 * U+001C..U+001F, so the ports use these instead.
 */
internal object PyText {
    /** Regex class body matching exactly the characters Python's `str.isspace()` accepts. */
    const val WS_CLASS = "\\t\\n\\x0B\\f\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"

    /** Python regex `\s` on a `str` pattern. */
    const val WS = "[$WS_CLASS]"

    fun isSpace(c: Char): Boolean = when (c) {
        '\t', '\n', '\u000B', '\u000C', '\r', ' ', '\u0085', ' ', ' ',
        ' ', ' ', ' ', ' ', '　' -> true
        in '\u001C'..'\u001F', in ' '..' ' -> true
        else -> false
    }

    /** Python `str.strip()` with no argument. */
    fun strip(s: String): String = s.trim(::isSpace)

    /** Python `str.split()` with no argument: runs of whitespace, empties dropped. */
    fun split(s: String): List<String> {
        val out = ArrayList<String>()
        var start = -1
        for (i in s.indices) {
            if (isSpace(s[i])) {
                if (start >= 0) { out += s.substring(start, i); start = -1 }
            } else if (start < 0) {
                start = i
            }
        }
        if (start >= 0) out += s.substring(start)
        return out
    }
}
