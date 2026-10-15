package com.xrmech.habla.asr

import com.xrmech.habla.asr.internal.PyText
import java.io.File
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assertions.fail
import org.junit.jupiter.api.Test

/**
 * Parity with the server's own `_update_detected_language`, `_is_bad_transcript` and
 * `_has_clipped_onset`, from goldens written by
 * `habla\.venv\Scripts\python -I habla/scripts/gen_shared_goldens.py --asr`.
 */
class ServerGoldenParityTest {
    private val profile = TestAssets.profile

    private fun golden(name: String): JsonObject {
        val text = javaClass.getResource("/golden/$name")?.readText(Charsets.UTF_8)
            ?: fail("golden missing: run $REGEN_COMMAND")
        return Json.parseToJsonElement(text).jsonObject
    }

    private fun assertFresh(g: JsonObject) {
        val recorded = g.getValue("sources").jsonObject.getValue("asr_profile.json").jsonPrimitive.content
        val shared = File(checkNotNull(System.getProperty("habla.sharedDir")) { "habla.sharedDir not set" })
        val bytes = File(shared, "asr_profile.json").readBytes()
        val normalized = String(bytes, Charsets.ISO_8859_1).replace("\r\n", "\n").toByteArray(Charsets.ISO_8859_1)
        val current = MessageDigest.getInstance("SHA-256").digest(normalized).joinToString("") { "%02x".format(it) }
        assertEquals(recorded, current, "golden stale: run $REGEN_COMMAND")
    }

    @Test
    fun `language vote goldens are fresh and match the profile`() {
        val g = golden("language_vote.json")
        assertFresh(g)
        assertEquals(profile.languageVote.window, g.getValue("window").jsonPrimitive.int)
        assertEquals(profile.languageVote.switchRatio, g.getValue("switch_ratio").jsonPrimitive.double)
    }

    @Test
    fun `LanguageVoter matches the server step by step`() {
        val g = golden("language_vote.json")
        val reset = g.getValue("reset_step").jsonPrimitive.content
        val cases = g.getValue("cases").jsonArray
        assertTrue(cases.size >= 10)
        val ids = cases.map { it.jsonObject.getValue("id").jsonPrimitive.content }.toSet()
        for (required in listOf("first_detection_es", "three_of_five_does_not_switch", "four_of_five_switches", "window_rollover")) {
            assertTrue(required in ids, "golden lacks case $required")
        }
        assertTrue(ids.any { it.startsWith("reset") }, "golden lacks a reset case")

        for (case in cases) {
            val o = case.jsonObject
            val id = o.getValue("id").jsonPrimitive.content
            val steps = o.getValue("steps").jsonArray.map { it.jsonPrimitive.content }
            val expected = o.getValue("current").jsonArray.map { if (it is JsonNull) null else it.jsonPrimitive.contentOrNull }
            val voter = LanguageVoter(profile)
            val actual = steps.map { step ->
                if (step == reset) voter.reset() else voter.vote(step)
                voter.current
            }
            assertEquals(expected, actual, "case $id")
        }
    }

    @Test
    fun `TranscriptFilters match the server on every golden text`() {
        val g = golden("transcript_filters.json")
        assertFresh(g)
        assertEquals(profile.filters.minLetters, g.getValue("min_letters").jsonPrimitive.int)
        assertEquals(profile.filters.minLetterRatio, g.getValue("min_letter_ratio").jsonPrimitive.double)
        for (case in g.getValue("cases").jsonArray) {
            val o = case.jsonObject
            val text = o.getValue("text").jsonPrimitive.content
            assertEquals(o.getValue("is_bad").jsonPrimitive.boolean, TranscriptFilters.isBadTranscript(text, profile), "is_bad '$text'")
            assertEquals(o.getValue("clipped_onset").jsonPrimitive.boolean, TranscriptFilters.hasClippedOnset(text, profile), "clipped_onset '$text'")
        }
    }

    @Test
    fun `filter golden contains every edge-cases case`() {
        val texts = golden("transcript_filters.json").getValue("cases").jsonArray
            .map { it.jsonObject.getValue("text").jsonPrimitive.content }.toSet()
        val nel = Char(0x85)
        val required = listOf(
            "", "   ", "${nel}hola$nel", " ok ", "ok", "abc", "123", "12 ab", "ab12c", "½½ abc", "²³ abc", "Ⅻ abc",
            "١٢٣ abc", "１２ ａｂｃ", "😀😀 abc", "...!!!", "¿Qué?", "ola amigo", "OLA amigo", "Ola, amigo",
            "  ueno pues", "ué pasa", "Ué pasa", "qué pasa", "ﬆoy aquí", "stoy${Char(9)}aquí", "hola", "Hola",
            "Gracias.", "[Música]",
        )
        val missing = required.filterNot { it in texts }
        assertTrue(missing.isEmpty(), "golden lacks $missing; run $REGEN_COMMAND")
    }

    @Test
    fun `PyText matches Python str semantics on the golden inputs`() {
        for (case in golden("pytext.json").getValue("cases").jsonArray) {
            val o = case.jsonObject
            val text = o.getValue("text").jsonPrimitive.content
            assertEquals(o.getValue("strip").jsonPrimitive.content, PyText.strip(text), "strip '$text'")
            assertEquals(
                (o.getValue("split") as JsonArray).map { it.jsonPrimitive.content },
                PyText.split(text),
                "split '$text'",
            )
            assertEquals(o.getValue("casefold").jsonPrimitive.content, PyText.casefold(text), "casefold '$text'")
            assertEquals(o.getValue("len").jsonPrimitive.int, PyText.length(text), "len '$text'")
            val cps = text.codePoints().toArray()
            assertEquals(o.getValue("alpha").jsonPrimitive.int, cps.count { PyText.isAlpha(it) }, "isalpha '$text'")
            assertEquals(o.getValue("alnum").jsonPrimitive.int, cps.count { PyText.isAlnum(it) }, "isalnum '$text'")
            assertEquals(o.getValue("space").jsonPrimitive.int, cps.count { PyText.isSpace(it) }, "isspace '$text'")
        }
    }

    private companion object {
        const val REGEN_COMMAND = "habla\\.venv\\Scripts\\python -I habla/scripts/gen_shared_goldens.py --asr"
    }
}
