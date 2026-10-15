package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.CorrectionDetail
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.TranslationResult
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import kotlin.test.assertEquals
import kotlin.test.assertIs

class ResponseParserTest {

    private val original = "hola mundo"

    private fun ok(raw: String): TranslationResult {
        val r = ResponseParser.parse(raw, original)
        return assertIs<ParseResult.Ok>(r, "expected Ok for: $raw").result
    }

    private fun assertInvalid(raw: String) {
        assertIs<ParseResult.Invalid>(ResponseParser.parse(raw, original), "expected Invalid for: $raw")
    }

    @Test
    fun pathologicalNestingIsInvalidNotAStackOverflow() {
        // Python raises RecursionError (an error result on the server), not a salvage case.
        val n = 100_000
        assertInvalid("{\"translated\": " + "[".repeat(n) + "]".repeat(n) + "}")
        assertInvalid("[".repeat(n) + "]".repeat(n))
        // Nesting up to the cap still decodes normally.
        val d = StrictJson.MAX_DEPTH - 1
        assertEquals("ok", ok("{\"translated\": \"ok\", \"x\": " + "[".repeat(d) + "]".repeat(d) + "}").translated)
    }

    @Test
    fun validJsonAllFields() {
        val r = ok(
            """{"corrected":"Hola, mundo.","translated":"Hello, world.","confidence":0.9,"speaker_hint":"teacher",
               "is_correction":true,"correction_detail":{"wrong":"a","right":"b","explanation":"c"},
               "flagged_phrases":[{"phrase":"p","literal":"l","meaning":"m","type":"slang","save_worthy":false,"region":"mx"}]}""",
        )
        assertEquals(
            TranslationResult(
                corrected = "Hola, mundo.", translated = "Hello, world.",
                flaggedPhrases = listOf(FlaggedPhrase("p", "l", "m", "slang", "universal", "llm", false)),
                confidence = 0.9, speakerHint = "teacher", isCorrection = true,
                correctionDetail = CorrectionDetail("a", "b", "c"),
            ),
            r,
        )
    }

    @Test
    fun defaultsWhenFieldsMissing() {
        val r = ok("""{"translated":"Hello"}""")
        assertEquals(TranslationResult(corrected = original, translated = "Hello", confidence = 0.5), r)
    }

    @Test
    fun falsyCorrectedAndTranslatedFallBackToOriginal() {
        for (v in listOf("null", "\"\"", "0", "false", "[]", "{}", "0.0")) {
            val r = ok("""{"corrected":$v,"translated":$v}""")
            assertEquals(original, r.corrected, v)
            assertEquals(original, r.translated, v)
        }
    }

    @Test
    fun truthyNonStringCorrectedIsInvalid() {
        assertInvalid("""{"corrected":5,"translated":"x"}""")
        assertInvalid("""{"corrected":"x","translated":["a"]}""")
        assertInvalid("""{"corrected":"x","translated":true}""")
    }

    @Test
    fun thinkTagsStripped() {
        assertEquals("Hi", ok("""<think>{"translated":"no"}</think>  {"translated":"Hi"}""").translated)
    }

    @Test
    fun thinkingPrefixCutToFirstBrace() {
        assertEquals("Hi", ok("Thinking Process:\n1. read\n{\"translated\":\"Hi\"}").translated)
    }

    @Test
    fun codeFencesRemoved() {
        assertEquals("Hi", ok("```json\n{\"translated\":\"Hi\"}\n```").translated)
    }

    @Test
    fun nestedBracesInsideStringsAndTrailingText() {
        val r = ok("""Sure! {"translated":"a } b { c","corrected":"x \" }"} and more""")
        assertEquals("a } b { c", r.translated)
        assertEquals("x \" }", r.corrected)
    }

    @Test
    fun unclosedJsonSalvagesFields() {
        val r = ok("""{"corrected": "Hola", "translated": "Hello there""")
        assertEquals(TranslationResult(corrected = "Hola", translated = "{\"corrected\": \"Hola\", \"translated\": \"Hello there", confidence = 0.3), r)
    }

    @Test
    fun salvageReplacesLiteralBackslashN() {
        val r = ok("""{"corrected": "a\nb", "translated": "c\nd", oops}""")
        assertEquals("a b", r.corrected)
        assertEquals("c d", r.translated)
        assertEquals(0.3, r.confidence)
    }

    @Test
    fun rawControlCharInsideStringIsRejectedThenSalvaged() {
        val r = ok("{\"corrected\":\"x\",\"translated\":\"line1\nline2\"}")
        assertEquals("line1\nline2", r.translated)
        assertEquals(0.3, r.confidence)
    }

    @Test
    fun translationLinePatterns() {
        assertEquals(TranslationResult(original, "Good morning", confidence = 0.3), ok("Translation: \"Good morning\""))
        assertEquals("Good morning", ok("Some preamble\nTRANSLATED = Good morning   \nbye").translated)
        assertEquals("Good morning", ok("translated: Good morning\r\nnext line").translated)
    }

    @Test
    fun bareQuotedString() {
        // A valid JSON string decodes (non-object -> raw text); only undecodable quoted text is unwrapped.
        assertEquals(TranslationResult(original, "\"Hello there\"", confidence = 0.1), ok("  \"Hello there\"  "))
        assertEquals(TranslationResult(original, "He said \"hi\" to me", confidence = 0.3), ok("\"He said \"hi\" to me\""))
    }

    @Test
    fun plainTextFallsBackWithLowConfidence() {
        assertEquals(TranslationResult(original, "just words", confidence = 0.1), ok("  just words "))
    }

    @Test
    fun nonObjectJsonUsesRawText() {
        assertEquals(TranslationResult(original, "42", confidence = 0.1), ok("42"))
        assertEquals(TranslationResult(original, "[1, 2]", confidence = 0.1), ok("[1, 2]"))
    }

    @Test
    fun crlfInput() {
        assertEquals("Hi", ok("\r\n{\"translated\":\r\n\"Hi\"}\r\n").translated)
    }

    @Test
    fun nanLiteralTakesSalvagePath() {
        val r = ok("""{"corrected":"a","translated":"b","confidence":NaN}""")
        assertEquals(0.3, r.confidence)
    }

    @Test
    fun confidenceRules() {
        assertEquals(0.9, ok("""{"confidence":"0.9"}""").confidence)
        assertEquals(0.9, ok("""{"confidence":" 0.9 "}""").confidence)
        assertEquals(1.0, ok("""{"confidence":true}""").confidence)
        assertEquals(1.0, ok("""{"confidence":1}""").confidence)
        assertInvalid("""{"confidence":"high"}""")
        assertInvalid("""{"confidence":null}""")
        assertInvalid("""{"confidence":[0.5]}""")
        assertInvalid("""{"confidence":{}}""")
        assertInvalid("""{"confidence":"0x10"}""")
    }

    @Test
    fun boolRules() {
        assertEquals(true, ok("""{"is_correction":"Yes"}""").isCorrection)
        assertEquals(false, ok("""{"is_correction":"off"}""").isCorrection)
        assertEquals(true, ok("""{"is_correction":1}""").isCorrection)
        assertEquals(false, ok("""{"is_correction":0.0}""").isCorrection)
        assertInvalid("""{"is_correction":2}""")
        assertInvalid("""{"is_correction":"maybe"}""")
        assertInvalid("""{"is_correction":null}""")
        assertInvalid("""{"flagged_phrases":[{"phrase":"p","meaning":"m","save_worthy":null}]}""")
    }

    @Test
    fun speakerHintRules() {
        assertEquals(null, ok("""{"speaker_hint":null}""").speakerHint)
        assertInvalid("""{"speaker_hint":3}""")
    }

    @Test
    fun flaggedPhrasesShapes() {
        assertEquals(emptyList(), ok("""{"flagged_phrases":""}""").flaggedPhrases)
        assertEquals(emptyList(), ok("""{"flagged_phrases":{}}""").flaggedPhrases)
        assertInvalid("""{"flagged_phrases":null}""")
        assertInvalid("""{"flagged_phrases":"abc"}""")
        assertInvalid("""{"flagged_phrases":{"a":1}}""")
        assertInvalid("""{"flagged_phrases":5}""")
        assertInvalid("""{"flagged_phrases":["x"]}""")
        assertInvalid("""{"flagged_phrases":[{"phrase":null,"meaning":"m"}]}""")
        assertInvalid("""{"flagged_phrases":[{"phrase":"p","meaning":3}]}""")
        assertInvalid("""{"flagged_phrases":[{"phrase":"p","meaning":"m","literal":7}]}""")
        val fp = ok("""{"flagged_phrases":[{}]}""").flaggedPhrases.single()
        assertEquals(FlaggedPhrase(phrase = "", literal = null, meaning = "", type = "idiom", source = "llm", saveWorthy = true), fp)
    }

    @Test
    fun correctionDetailRules() {
        assertEquals(CorrectionDetail("", "", "use ser"), ok("""{"correction_detail":"use ser"}""").correctionDetail)
        assertEquals(null, ok("""{"correction_detail":""}""").correctionDetail)
        assertEquals(null, ok("""{"correction_detail":{}}""").correctionDetail)
        assertEquals(CorrectionDetail("w", "", ""), ok("""{"correction_detail":{"wrong":"w"}}""").correctionDetail)
        assertInvalid("""{"correction_detail":{"wrong":null}}""")
        assertInvalid("""{"correction_detail":["a"]}""")
        assertInvalid("""{"correction_detail":5}""")
    }

    @Test
    fun duplicateKeysLastWins() {
        assertEquals("second", ok("""{"translated":"first","translated":"second"}""").translated)
    }

    @ParameterizedTest
    @ValueSource(strings = ["""{"a":01}""", """{"a":1.}""", """{a:1}""", """{"a":1,}""", """{"a":abc}""", """{"a":"\x"}"""])
    fun strictDecoderRejectsWhatPythonRejects(json: String) {
        // Each falls to salvage: no corrected/translated, no translation line -> 0.1 raw text.
        assertEquals(0.1, ok(json).confidence)
    }

    @Test
    fun extractFirstJsonCases() {
        assertEquals("no braces", ResponseParser.extractFirstJson("no braces"))
        assertEquals("{\"a\":{\"b\":1}}", ResponseParser.extractFirstJson("x {\"a\":{\"b\":1}} y"))
        assertEquals("{\"a\":\"\\\"}\"", ResponseParser.extractFirstJson("{\"a\":\"\\\"}\""))
        assertEquals("{\"a\":", ResponseParser.extractFirstJson("pre {\"a\":"))
    }

    @Test
    fun extractRawTranslationFirstLineOnly() {
        assertEquals("one", ResponseParser.extractRawTranslation("Translation: one\nTranslation: two"))
        assertEquals("", ResponseParser.extractRawTranslation("nothing here"))
        assertEquals("multi\nline", ResponseParser.extractRawTranslation("\"multi\nline\"\n"))
    }
}
