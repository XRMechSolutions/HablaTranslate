package com.xrmech.habla.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlinx.serialization.KSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Decode -> encode -> tree comparison per shared file: proves no key is dropped, renamed or
 * retyped by the Kotlin models. Numbers compare by BigDecimal value; a JSON null and an absent
 * member are equal (explicitNulls = false drops nulls on encode).
 */
class SharedAssetsRoundTripTest {

    private val source = FileAssetSource.fromSystemProperty()

    // encodeDefaults so idiom/model defaults present in the file are re-emitted.
    private val encoder = Json(SharedJson) { encodeDefaults = true }

    private val files: List<Pair<String, KSerializer<*>>> = listOf(
        "asr_profile.json" to AsrProfile.serializer(),
        "languages.json" to LanguageRegistry.serializer(),
        "prompts.json" to Prompts.serializer(),
        "models.json" to ModelsFile.serializer(),
        "asr_models.json" to ModelCatalog.serializer(),
        "idioms/index.json" to IdiomIndex.serializer(),
        "idioms/spain.json" to ListSerializer(IdiomPatternDef.serializer()),
    )

    @Suppress("UNCHECKED_CAST")
    private fun roundTrip(text: String, serializer: KSerializer<*>): JsonElement {
        val s = serializer as KSerializer<Any?>
        return encoder.encodeToJsonElement(s, SharedJson.decodeFromString(s, text))
    }

    @Test
    fun `should reproduce every shared file tree when decoded and re-encoded`() {
        for ((path, serializer) in files) {
            val text = source.readText(path)
            val diff = treeDiff(Json.parseToJsonElement(text), roundTrip(text, serializer), "$")
            assertNull(diff, "shared/$path: $diff")
        }
    }

    @Test
    fun `should detect a quoted number when it decodes silently into an Int`() {
        val text = source.readText("asr_profile.json")
        val quoted = text.replace("\"silence_duration_ms\": 600", "\"silence_duration_ms\": \"600\"")
        check(quoted != text)
        val diff = treeDiff(Json.parseToJsonElement(quoted), roundTrip(quoted, AsrProfile.serializer()), "$")
        assertNotNull(diff)
        assertEquals("$.segmenter.silence_duration_ms", diff.substringBefore(":"))
    }

    @Test
    fun `should treat null and absent as equal and compare numbers by value`() {
        assertNull(treeDiff(Json.parseToJsonElement("""{"a": null, "b": 0.10}"""), Json.parseToJsonElement("""{"b": 0.1}"""), "$"))
        assertNotNull(treeDiff(Json.parseToJsonElement("""{"a": 1}"""), Json.parseToJsonElement("""{}"""), "$"))
    }

    private fun treeDiff(a: JsonElement, b: JsonElement, path: String): String? {
        when {
            a is JsonObject && b is JsonObject -> {
                for (key in (a.keys + b.keys)) {
                    val av = a[key] ?: JsonNull
                    val bv = b[key] ?: JsonNull
                    treeDiff(av, bv, "$path.$key")?.let { return it }
                }
                return null
            }
            a is JsonArray && b is JsonArray -> {
                if (a.size != b.size) return "$path: array size ${a.size} != ${b.size}"
                a.indices.forEach { i -> treeDiff(a[i], b[i], "$path[$i]")?.let { return it } }
                return null
            }
            a is JsonNull && b is JsonNull -> return null
            a is JsonPrimitive && b is JsonPrimitive && a !is JsonNull && b !is JsonNull -> {
                if (a.isString != b.isString) return "$path: $a != $b (string vs non-string)"
                if (a.isString) return if (a.content == b.content) null else "$path: $a != $b"
                val an = a.content.toBigDecimalOrNull()
                val bn = b.content.toBigDecimalOrNull()
                val equal = if (an != null && bn != null) an.compareTo(bn) == 0 else a.content == b.content
                return if (equal) null else "$path: $a != $b"
            }
            else -> return "$path: $a != $b"
        }
    }
}
