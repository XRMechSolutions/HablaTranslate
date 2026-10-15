package com.xrmech.habla.shared

import java.util.regex.PatternSyntaxException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.serialization.builtins.ListSerializer

class IdiomRegexTest {

    private val ws = "[\\x{9}-\\x{D}\\x{1C}-\\x{20}\\x{85}\\x{A0}\\x{1680}\\x{2000}-\\x{200A}\\x{2028}\\x{2029}\\x{202F}\\x{205F}\\x{3000}]"

    @Test
    fun `should expand backslash-s to the Python whitespace class when translating the interfaces example`() {
        assertEquals("tomar$ws+(el$ws+)?pelo", IdiomRegex.translate("tomar\\s+(el\\s+)?pelo"))
    }

    @Test
    fun `should match non-ASCII letters case-insensitively when compiled`() {
        val p = IdiomRegex.compile("est[aá]")
        assertTrue(p.matcher("ESTÁ").find())
        assertTrue(p.matcher("Está").find())
        assertTrue(IdiomRegex.compile("señor").matcher("SEÑOR").find())
    }

    @Test
    fun `should match Python whitespace and not zero-width space when using backslash-s`() {
        val p = IdiomRegex.compile("a\\sb")
        for (c in listOf(' ', ' ', '\t', '　', '\u0085', '\u001C')) {
            assertTrue(p.matcher("a${c}b").matches(), "U+%04X".format(c.code))
        }
        assertFalse(p.matcher("a​b").matches())
        assertTrue(IdiomRegex.compile("a[\\s,]b").matcher("a b").matches())
    }

    @Test
    fun `should accept every whitelisted construct when translating`() {
        IdiomRegex.compile("(?:a|b)*?c+?d??e{2}f{1,}g{1,3}?\\.\\-[^a-z\\s\\]]")
    }

    @Test
    fun `should reject each construct outside the whitelist naming token and index`() {
        val cases = listOf(
            "a.b" to ("." to 1),
            "^a" to ("^" to 0),
            "a$" to ("$" to 1),
            "\\w" to ("\\w" to 0),
            "x\\d" to ("\\d" to 1),
            "\\b" to ("\\b" to 0),
            "\\S" to ("\\S" to 0),
            "\\W" to ("\\W" to 0),
            "\\D" to ("\\D" to 0),
            "\\B" to ("\\B" to 0),
            "\\n" to ("\\n" to 0),
            "(?P<x>a)" to ("(?P" to 0),
            "(?=a)" to ("(?=" to 0),
            "(?i)a" to ("(?i" to 0),
            "a*+" to ("*+" to 1),
            "a{2}+" to ("{2}+" to 1),
            "x{,3}" to ("{" to 1),
            "a{b" to ("{" to 1),
            "[a[b]]" to ("[" to 2),
            "[a&&b]" to ("&&" to 2),
            "[]a]" to ("]" to 1),
            "[^]a]" to ("]" to 2),
            "a\\" to ("\\" to 1),
        )
        for ((pattern, expected) in cases) {
            val (token, index) = expected
            val e = assertFailsWith<IllegalArgumentException>(pattern) { IdiomRegex.translate(pattern) }
            assertTrue(
                e.message!!.startsWith("unsupported regex construct '$token' at index $index: "),
                "$pattern -> ${e.message}",
            )
        }
    }

    @Test
    fun `should pass a syntax error through as PatternSyntaxException when parentheses are unbalanced`() {
        assertFailsWith<PatternSyntaxException> { IdiomRegex.compile("(unclosed") }
    }

    @Test
    fun `should translate and compile every pattern when reading the real spain json`() {
        val text = FileAssetSource.fromSystemProperty().readText("idioms/spain.json")
        val entries = SharedJson.decodeFromString(ListSerializer(IdiomPatternDef.serializer()), text)
        assertEquals(24, entries.size)
        for (e in entries) IdiomRegex.compile(e.pattern)
    }
}
