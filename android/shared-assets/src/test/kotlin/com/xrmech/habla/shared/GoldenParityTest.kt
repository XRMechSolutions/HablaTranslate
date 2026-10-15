package com.xrmech.habla.shared

import java.io.File
import java.security.MessageDigest
import java.util.regex.Pattern
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.fail
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.io.TempDir

/**
 * Replays the Python goldens from habla/scripts/gen_shared_goldens.py against PromptTemplate and
 * IdiomRegex. Each suite first checks the goldens were generated from the current shared/ files.
 */
class GoldenParityTest {

    private val shared = FileAssetSource.fromSystemProperty()
    private val promptGolden = golden("prompt_format.json")
    private val regexGolden = golden("idiom_regex.json")

    // ---- staleness ---------------------------------------------------------------------------

    @Test
    fun `should match current shared files when checking prompt golden sources`() {
        assertNull(staleness(promptGolden, shared))
    }

    @Test
    fun `should match current shared files when checking regex golden sources`() {
        assertNull(staleness(regexGolden, shared))
    }

    @Test
    fun `should report golden stale when a source file changed`(@TempDir tmp: File) {
        val original = shared.readText("prompts.json")
        val changed = original.replaceFirst("Translate", "Translatf")
        check(changed != original)
        File(tmp, "prompts.json").writeText(changed, Charsets.UTF_8)
        val message = staleness(promptGolden, FileAssetSource(tmp))
        assertNotNull(message)
        assertTrue(message.startsWith("golden stale: run python -I habla/scripts/gen_shared_goldens.py"), message)
    }

    @Test
    fun `should hash a CRLF copy the same as the LF file when checking staleness`(@TempDir tmp: File) {
        File(tmp, "prompts.json").writeText(shared.readText("prompts.json").replace("\n", "\r\n"), Charsets.UTF_8)
        assertNull(staleness(promptGolden, FileAssetSource(tmp)))
    }

    // ---- prompt_format.json ------------------------------------------------------------------

    @Test
    fun `should reproduce every Python str format case when formatting goldens`() {
        val cases = promptGolden.getValue("cases").jsonArray.map { it.jsonObject }
        assertTrue(cases.size >= 50, "only ${cases.size} cases")
        val failures = cases.mapNotNull { checkPromptCase(it) }
        assertEquals(emptyList(), failures)
    }

    @Test
    fun `should name the case when a golden expectation is wrong`() {
        val case = promptGolden.getValue("cases").jsonArray.first().jsonObject
        val wrong = JsonObject(case + ("expected" to JsonPrimitive("deliberately wrong")))
        val failure = checkPromptCase(wrong)
        assertNotNull(failure)
        assertTrue(failure.startsWith("[${case.str("id")}]"), failure)
    }

    private fun checkPromptCase(case: JsonObject): String? {
        val id = case.str("id")
        val template = case.str("template")
        val values = case.getValue("values").jsonObject.mapValues { it.value.jsonPrimitive.content }
        val kind = case.str("kind")
        val result = runCatching { PromptTemplate.format(template, values) }
        return when (kind) {
            "ok" -> {
                val expected = case.str("expected")
                val actual = result.getOrElse { return "[$id] threw ${it.message}" }
                if (actual == expected) null else "[$id] expected <$expected> but was <$actual>"
            }
            "error", "unsupported" -> {
                val e = result.exceptionOrNull()
                when {
                    e == null -> "[$id] expected IllegalArgumentException but got <${result.getOrNull()}>"
                    e !is IllegalArgumentException -> "[$id] expected IllegalArgumentException but got $e"
                    else -> null
                }
            }
            else -> "[$id] unknown kind '$kind'"
        }
    }

    // ---- idiom_regex.json --------------------------------------------------------------------

    @Test
    fun `should match exactly Python whitespace over the BMP when using the rewritten backslash-s`() {
        val expected = regexGolden.getValue("whitespace_bmp").jsonArray.map { it.jsonPrimitive.int }
        val ws = Pattern.compile(IdiomRegex.translate("\\s"), IdiomRegex.FLAGS)
        assertEquals(expected, bmpMatches(ws))
    }

    @Test
    fun `should fold the same code points as Python for every letter used in patterns`() {
        val casefold = regexGolden.getValue("casefold").jsonObject
        assertTrue(casefold.isNotEmpty())
        val failures = casefold.mapNotNull { (ch, golden) ->
            val expected = golden.jsonArray.map { it.jsonPrimitive.int }
            val actual = bmpMatches(IdiomRegex.compile(ch))
            if (actual == expected) null else "[casefold '$ch'] expected $expected but was $actual"
        }
        assertEquals(emptyList(), failures)
    }

    @Test
    fun `should find the same hit set as Python re search when matching every pattern against every probe`() {
        val refs = regexGolden.getValue("patterns").jsonArray.map { it.jsonPrimitive.content }
        val probes = regexGolden.getValue("probes").jsonArray.map { it.jsonPrimitive.content }
        val pythonUncompilable = regexGolden.getValue("python_compile_errors").jsonObject.keys.map { it.toInt() }.toSet()
        val patterns = resolvePatterns(refs)

        val expected = regexGolden.getValue("hits").jsonArray.map { h ->
            val o = h.jsonObject
            Hit(o.getValue("pattern").jsonPrimitive.int, o.getValue("probe").jsonPrimitive.int,
                o.getValue("start").jsonPrimitive.int, o.getValue("end").jsonPrimitive.int)
        }.toSet()
        val actual = HashSet<Hit>()
        patterns.forEachIndexed { pi, p ->
            if (pi in pythonUncompilable) return@forEachIndexed
            val rx = IdiomRegex.compile(p)
            probes.forEachIndexed { qi, probe ->
                val m = rx.matcher(probe)
                if (m.find()) actual += Hit(pi, qi, m.start(), m.end())
            }
        }
        val missing = (expected - actual).map { describe(it, refs, probes) }
        val extra = (actual - expected).map { describe(it, refs, probes) }
        assertTrue(missing.isEmpty() && extra.isEmpty(), "missing (Python only): $missing\nextra (Kotlin only): $extra")
        assertTrue(expected.isNotEmpty())
    }

    private data class Hit(val pattern: Int, val probe: Int, val start: Int, val end: Int)

    private fun describe(h: Hit, refs: List<String>, probes: List<String>) =
        "[${refs[h.pattern]} x probe ${h.probe} '${probes[h.probe]}' ${h.start}..${h.end}]"

    private fun resolvePatterns(refs: List<String>): List<String> {
        val byFile = HashMap<String, List<IdiomPatternDef>>()
        return refs.map { ref ->
            val file = ref.substringBefore('#')
            val index = ref.substringAfter('#').toInt()
            byFile.getOrPut(file) {
                SharedJson.decodeFromString(ListSerializer(IdiomPatternDef.serializer()), shared.readText(file))
            }[index].pattern
        }
    }

    private fun bmpMatches(p: Pattern): List<Int> =
        (0..0xFFFF).filter { cp -> p.matcher(cp.toChar().toString()).matches() }

    // ---- helpers -----------------------------------------------------------------------------

    private fun golden(name: String): JsonObject {
        val url = javaClass.getResource("/golden/$name") ?: fail("golden/$name missing from test resources")
        return Json.parseToJsonElement(url.readText(Charsets.UTF_8)).jsonObject
    }

    /** Null when every recorded source sha256 equals the current file's (CRLF->LF normalized). */
    private fun staleness(golden: JsonObject, source: AssetSource): String? {
        val sources = golden.getValue("sources").jsonObject
        for ((path, sha) in sources) {
            val text = source.readText(path).replace("\r\n", "\n")
            val actual = MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
            if (actual != sha.jsonPrimitive.content) {
                return "golden stale: run python -I habla/scripts/gen_shared_goldens.py ($path sha256 $actual != recorded ${sha.jsonPrimitive.content})"
            }
        }
        return null
    }

    private fun JsonObject.str(key: String): String {
        val e: JsonElement = getValue(key)
        check(e != JsonNull) { "golden field $key is null" }
        return e.jsonPrimitive.content
    }
}
