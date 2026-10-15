package com.xrmech.habla.translate.model

import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.LlmRequest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ModelTypesTest {

    private val json = Json { encodeDefaults = true; explicitNulls = true }

    @Test
    fun translationResultEncodesServerModelDumpKeys() {
        val result = TranslationResult(
            corrected = "Me importa un pepino.",
            translated = "I couldn't care less.",
            flaggedPhrases = listOf(FlaggedPhrase(phrase = "importa un pepino", meaning = "not care")),
        )
        val obj = json.parseToJsonElement(json.encodeToString(TranslationResult.serializer(), result)).jsonObject

        assertEquals(
            listOf("corrected", "translated", "flagged_phrases", "confidence", "speaker_hint", "is_correction", "correction_detail"),
            obj.keys.toList(),
        )
        assertEquals(JsonNull, obj["speaker_hint"])
        assertEquals(JsonNull, obj["correction_detail"])
        val phrase = obj["flagged_phrases"]!!.jsonArray[0] as JsonObject
        assertEquals(
            listOf("phrase", "literal", "meaning", "type", "region", "source", "save_worthy", "span_start", "span_end"),
            phrase.keys.toList(),
        )
        assertEquals(JsonNull, phrase["literal"])
        assertEquals(JsonNull, phrase["span_start"])
    }

    @Test
    fun translationResultRoundTrips() {
        val result = TranslationResult(
            corrected = "a", translated = "b",
            flaggedPhrases = listOf(FlaggedPhrase("p", "lit", "m", "slang", null, "pattern_db", false, 3, 20)),
            confidence = 0.75, speakerHint = "teacher", isCorrection = true,
            correctionDetail = CorrectionDetail("x", "y", "z"),
        )
        val encoded = json.encodeToString(TranslationResult.serializer(), result)
        assertEquals(result, json.decodeFromString(TranslationResult.serializer(), encoded))
    }

    @Test
    fun tokenUsageAndUsageKindUseSnakeCaseNames() {
        val usage = TokenUsage(1, 2, 3, 4, "gpt-x")
        assertEquals(
            """{"input_tokens":1,"cached_input_tokens":2,"output_tokens":3,"reasoning_tokens":4,"model":"gpt-x"}""",
            json.encodeToString(TokenUsage.serializer(), usage),
        )
        assertEquals("\"final\"", json.encodeToString(UsageKind.serializer(), UsageKind.FINAL))
        assertEquals("\"es_to_en\"", json.encodeToString(Direction.serializer(), Direction.ES_TO_EN))
        assertEquals("\"classroom\"", json.encodeToString(Mode.serializer(), Mode.CLASSROOM))
    }

    @Test
    fun directionCodes() {
        assertEquals(Triple("es_to_en", "es", "en"), Direction.ES_TO_EN.let { Triple(it.serialName, it.sourceCode, it.targetCode) })
        assertEquals(Triple("en_to_es", "en", "es"), Direction.EN_TO_ES.let { Triple(it.serialName, it.sourceCode, it.targetCode) })
        assertEquals(Triple("auto", "es", "en"), Direction.AUTO.let { Triple(it.serialName, it.sourceCode, it.targetCode) })
    }

    @Test
    fun fromSerialNameKnownAndUnknown() {
        Direction.entries.forEach { assertEquals(it, Direction.fromSerialName(it.serialName)) }
        assertNull(Direction.fromSerialName("fr_to_en"))
        assertNull(Direction.fromSerialName(""))
    }

    @Test
    fun retryableAndAllowsFallbackTruthTable() {
        val retryable = setOf(
            LlmException.Kind.RATE_LIMITED, LlmException.Kind.SERVER, LlmException.Kind.TIMEOUT, LlmException.Kind.NETWORK,
        )
        val noFallback = setOf(
            LlmException.Kind.NO_KEY, LlmException.Kind.INVALID_KEY, LlmException.Kind.QUOTA, LlmException.Kind.NETWORK,
        )
        assertEquals(10, LlmException.Kind.entries.size)
        for (kind in LlmException.Kind.entries) {
            val e = LlmException("m", kind)
            assertEquals(kind in retryable, e.retryable, "retryable for $kind")
            assertEquals(kind !in noFallback, e.allowsFallback, "allowsFallback for $kind")
        }
    }

    @Test
    fun llmRequestMaxRetriesDefaultsToZero() {
        val request = LlmRequest("m", "", "in", 16, false, UsageKind.TEST)
        assertEquals(0, request.maxRetries)
        assertFalse(request.json)
        assertTrue(LlmException("m", LlmException.Kind.SERVER).retryAfterMs == null)
    }
}
