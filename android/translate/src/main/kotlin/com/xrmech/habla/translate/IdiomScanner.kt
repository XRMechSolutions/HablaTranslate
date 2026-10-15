package com.xrmech.habla.translate

import com.xrmech.habla.shared.IdiomPatternDef
import com.xrmech.habla.shared.IdiomRegex
import com.xrmech.habla.translate.model.FlaggedPhrase
import java.util.Locale
import java.util.regex.Pattern

/**
 * Port of the server's `IdiomScanner` (`load_from_json` + `scan`). Patterns compile only through
 * [IdiomRegex], the single Python-`re`-to-Java translator; one that fails is skipped (the server's
 * `re.error` skip), logged WARN and listed in [skipped].
 */
class IdiomScanner(patterns: List<IdiomPatternDef>) {
    private class Compiled(val pattern: Pattern, val def: IdiomPatternDef)

    private val compiled: List<Compiled>

    /** Canonicals whose pattern failed to compile. */
    val skipped: List<String>

    init {
        val log = Log()
        val ok = ArrayList<Compiled>(patterns.size)
        val bad = ArrayList<String>()
        for (def in patterns) {
            try {
                ok += Compiled(IdiomRegex.compile(def.pattern), def)
            } catch (e: IllegalArgumentException) {
                // PatternSyntaxException is an IllegalArgumentException too.
                bad += def.canonical
                log.warn("Skipping idiom '${def.canonical}': ${e.message}")
            }
        }
        compiled = ok
        skipped = bad
    }

    /** Number of compiled patterns. */
    val count: Int get() = compiled.size

    /**
     * One phrase per distinct canonical (case-insensitive) whose pattern matches, in pattern order.
     * Spans are code-point offsets, matching the server's Python indices.
     */
    fun scan(text: String): List<FlaggedPhrase> {
        val seen = HashSet<String>()
        val out = ArrayList<FlaggedPhrase>()
        for (c in compiled) {
            val m = c.pattern.matcher(text)
            if (!m.find()) continue
            if (!seen.add(c.def.canonical.lowercase(Locale.ROOT))) continue
            out += FlaggedPhrase(
                phrase = c.def.canonical,
                literal = c.def.literal,
                meaning = c.def.meaning,
                type = "idiom",
                region = c.def.region,
                source = "pattern_db",
                saveWorthy = true,
                spanStart = text.codePointCount(0, m.start()),
                spanEnd = text.codePointCount(0, m.end()),
            )
        }
        return out
    }
}
