package com.xrmech.habla.shared

import java.util.regex.Pattern

/**
 * The single place idiom patterns written for Python `re` (with `IGNORECASE`) become
 * `java.util.regex` patterns that match the same text (design.md section 8).
 *
 * Only a whitelist of constructs whose behaviour is identical in Python, the desktop JVM and
 * Android's ICU engine is accepted; everything else throws, so a future pattern cannot silently
 * diverge between server and phone. `\s` is rewritten to Python's explicit whitespace set because
 * the three engines disagree on it.
 */
object IdiomRegex {
    /** Unicode-aware case folding, as Python str patterns do under `IGNORECASE`. */
    const val FLAGS: Int = Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE

    /** Members of Python's `\s` (every `str.isspace()` char), without brackets. */
    private const val WHITESPACE_MEMBERS =
        "\\x{9}-\\x{D}\\x{1C}-\\x{20}\\x{85}\\x{A0}\\x{1680}\\x{2000}-\\x{200A}\\x{2028}\\x{2029}\\x{202F}\\x{205F}\\x{3000}"
    private const val WHITESPACE_CLASS = "[$WHITESPACE_MEMBERS]"

    // The closing brace is escaped: Android's ICU engine rejects a bare '}' (the JVM accepts it).
    private val QUANTIFIER = Regex("""\{\d+(,\d*)?\}""")

    /**
     * Checks [pythonPattern] against the whitelist and returns the equivalent Java pattern text.
     * Syntax errors the whitelist does not cover (e.g. unbalanced parentheses) surface from
     * [compile] as `PatternSyntaxException`.
     *
     * @throws IllegalArgumentException `unsupported regex construct '<token>' at index i: <reason>`
     */
    fun translate(pythonPattern: String): String {
        val p = pythonPattern
        val out = StringBuilder(p.length + 64)
        var i = 0
        while (i < p.length) {
            val c = p[i]
            when (c) {
                '\\' -> {
                    out.append(translateEscape(p, i, inClass = false))
                    i += 2
                }
                '.' -> reject(".", i, "'.' line-terminator handling differs between Python and Java")
                '^', '$' -> reject(c.toString(), i, "anchor line-terminator handling differs between Python and Java")
                '(' -> {
                    if (i + 1 < p.length && p[i + 1] == '?') {
                        if (i + 2 < p.length && p[i + 2] == ':') {
                            out.append("(?:")
                            i += 3
                        } else {
                            val token = p.substring(i, minOf(i + 3, p.length))
                            reject(token, i, "only (?: groups are supported (no named groups, lookarounds or inline flags)")
                        }
                    } else {
                        out.append('(')
                        i++
                    }
                }
                ')', '|' -> { out.append(c); i++ }
                '?', '*', '+' -> {
                    out.append(c)
                    i = appendQuantifierSuffix(p, i + 1, out, c.toString())
                }
                '{' -> {
                    val m = QUANTIFIER.matchAt(p, i)
                        ?: reject("{", i, "'{' not forming a {m}, {m,} or {m,n} quantifier (Python literal, Java error)")
                    out.append(m.value)
                    i = appendQuantifierSuffix(p, i + m.value.length, out, m.value)
                }
                // Literal in Python; escaped so no engine can read it as syntax.
                '}', ']' -> { out.append('\\').append(c); i++ }
                '[' -> i = translateClass(p, i, out)
                else -> { out.append(c); i++ }
            }
        }
        return out.toString()
    }

    /** `Pattern.compile(translate(pythonPattern), FLAGS)`. */
    fun compile(pythonPattern: String): Pattern = Pattern.compile(translate(pythonPattern), FLAGS)

    private fun appendQuantifierSuffix(p: String, at: Int, out: StringBuilder, quantifier: String): Int {
        if (at < p.length) {
            when (p[at]) {
                '?' -> { out.append('?'); return at + 1 }
                '+' -> reject("$quantifier+", at - quantifier.length, "possessive quantifiers are not supported")
            }
        }
        return at
    }

    private fun translateClass(p: String, start: Int, out: StringBuilder): Int {
        out.append('[')
        var i = start + 1
        if (i < p.length && p[i] == '^') {
            out.append('^')
            i++
        }
        if (i < p.length && p[i] == ']') {
            reject("]", i, "a leading ']' in a class is a literal in Python but an error in Java")
        }
        while (i < p.length) {
            val c = p[i]
            when {
                c == ']' -> { out.append(']'); return i + 1 }
                c == '\\' -> { out.append(translateEscape(p, i, inClass = true)); i += 2 }
                c == '[' -> reject("[", i, "'[' inside a class is a Java union")
                c == '&' && i + 1 < p.length && p[i + 1] == '&' -> reject("&&", i, "'&&' inside a class is a Java intersection")
                else -> { out.append(c); i++ }
            }
        }
        // Unterminated class: left for Pattern.compile to report as a syntax error.
        return i
    }

    private fun translateEscape(p: String, i: Int, inClass: Boolean): String {
        if (i + 1 >= p.length) reject("\\", i, "trailing backslash")
        val e = p[i + 1]
        return when {
            e == 's' -> if (inClass) WHITESPACE_MEMBERS else WHITESPACE_CLASS
            e.code in 33..126 && !e.isLetterOrDigit() -> "\\$e"
            e in "wdbSWDB" -> reject("\\$e", i, "Unicode vs ASCII semantics differ between Python and Java")
            else -> reject("\\$e", i, "escape not in the supported set (\\s and escaped ASCII punctuation)")
        }
    }

    private fun reject(token: String, index: Int, reason: String): Nothing =
        throw IllegalArgumentException("unsupported regex construct '$token' at index $index: $reason")
}
