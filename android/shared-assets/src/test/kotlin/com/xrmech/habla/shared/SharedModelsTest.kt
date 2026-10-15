package com.xrmech.habla.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.SerializationException

class SharedModelsTest {

    private val languagesJson = """
        {"default_target": "en", "languages": {"es": {"name": "Spanish", "flag": "x"}, "en": {"name": "English", "flag": "y"}}}
    """.trimIndent()

    @Test
    fun `should fail naming the key when an unknown key is present`() {
        val e = assertFailsWith<SerializationException> {
            SharedJson.decodeFromString(LanguageInfo.serializer(), """{"name": "Spanish", "flag": "x", "foo": 1}""")
        }
        assertTrue("foo" in e.message!!, e.message)
    }

    @Test
    fun `should fail when a required key is missing`() {
        val e = assertFailsWith<SerializationException> {
            SharedJson.decodeFromString(LanguageInfo.serializer(), """{"name": "Spanish"}""")
        }
        assertTrue("flag" in e.message!!, e.message)
    }

    @Test
    fun `should accept a file when the comment key is absent`() {
        val r = SharedJson.decodeFromString(LanguageRegistry.serializer(), languagesJson)
        assertNull(r.comment)
    }

    @Test
    fun `should apply server defaults when optional idiom keys are absent`() {
        val d = SharedJson.decodeFromString(
            IdiomPatternDef.serializer(),
            """{"pattern": "a", "canonical": "b", "meaning": "c"}""",
        )
        assertEquals("universal", d.region)
        assertEquals("", d.literal)
        assertEquals("common", d.frequency)
    }

    @Test
    fun `should decode null when whisper_cpp final beam_size is null`() {
        val f = SharedJson.decodeFromString(
            WhisperCppFinal.serializer(),
            """{"beam_size": null, "temperature": 0.0, "temperature_inc": 0.0, "entropy_thold": 0.0, "logprob_thold": -1.0, "no_speech_thold": 1.0}""",
        )
        assertNull(f.beamSize)
    }

    @Test
    fun `should keep file order and reject unknown codes when using the language registry`() {
        val r = SharedJson.decodeFromString(LanguageRegistry.serializer(), languagesJson)
        assertEquals(listOf("es", "en"), r.codes)
        assertEquals("Spanish", r.name("es"))
        assertTrue(r.isSupported("en"))
        val e = assertFailsWith<IllegalArgumentException> { r.name("xx") }
        assertEquals("unknown language code 'xx'", e.message)
    }

    @Test
    fun `should expose pair helpers when profile lists are well formed`() {
        val pp = PartialPrepProfile(listOf(80, 7500), 4, 3000.0, 10.0, 10.0)
        assertEquals(80, pp.lowHz)
        assertEquals(7500, pp.highHz)
        val f = FilterProfile(3, 0.5, listOf(listOf("ola", "hola")))
        assertEquals(listOf("ola" to "hola"), f.onsetPairs)
    }

    @Test
    fun `should look up catalog and model entries when ids exist`() {
        val entry = AsrModelEntry("m", "m.bin", "small", "q8_0", "a".repeat(40), "u", 1, "b".repeat(64))
        val catalog = ModelCatalog(defaultId = "m", models = listOf(entry))
        assertEquals(entry, catalog.default)
        assertNull(catalog.byId("other"))
        val info = ModelInfo(0.1, 0.01, 0.5, emptyList(), null, false, true)
        assertEquals(info, ModelTable("x", "x", listOf("x"), mapOf("x" to info)).info("x"))
    }
}
