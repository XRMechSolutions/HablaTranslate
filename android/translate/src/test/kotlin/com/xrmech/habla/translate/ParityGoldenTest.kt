package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.TranslationResult
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Compares every Kotlin port with goldens generated from the real server functions by
 * `habla/scripts/gen_translate_goldens.py`. The goldens are the arbiter: a disagreement is fixed
 * in the Kotlin port, never in the golden. A missing or empty golden fails rather than passing
 * vacuously.
 */
class ParityGoldenTest {

    private val json = Json { ignoreUnknownKeys = false }
    private val assets = TestAssets.assets

    private fun cases(name: String): List<JsonObject> {
        val stream = javaClass.classLoader.getResourceAsStream("golden/$name")
            ?: error("golden/$name is missing; run habla/scripts/gen_translate_goldens.py")
        val root = json.parseToJsonElement(stream.use { it.readBytes().toString(Charsets.UTF_8) }).jsonObject
        val list = root["cases"]!!.jsonArray.map { it.jsonObject }
        assertTrue(list.isNotEmpty(), "golden/$name has no cases")
        return list
    }

    private fun JsonObject.str(key: String): String = this[key]!!.jsonPrimitive.content
    private fun JsonObject.strOrNull(key: String): String? = this[key]?.takeIf { it != JsonNull }?.jsonPrimitive?.content

    private fun direction(name: String): Direction = Direction.fromSerialName(name) ?: error("unknown direction $name")

    private fun label(i: Int, s: String) = "#$i ${s.take(50).replace('\n', ' ').replace('\r', ' ')}"

    @TestFactory
    fun parse(): List<DynamicTest> = cases("parse.json").mapIndexed { i, c ->
        DynamicTest.dynamicTest(label(i, c.str("id"))) {
            val actual = ResponseParser.parse(c.str("raw"), c.str("original"))
            val expected = c["expected"]!!.jsonObject
            if (expected["invalid"] != null) {
                assertIs<ParseResult.Invalid>(actual, "server raised ${expected["error"]}")
            } else {
                val ok = assertIs<ParseResult.Ok>(actual)
                assertEquals(json.decodeFromJsonElement(TranslationResult.serializer(), expected), ok.result)
            }
        }
    }

    @TestFactory
    fun passthrough(): List<DynamicTest> = cases("passthrough.json").mapIndexed { i, c ->
        DynamicTest.dynamicTest(label(i, c.str("source") + " | " + c.str("translated"))) {
            assertEquals(
                c["expected"]!!.jsonPrimitive.boolean,
                Passthrough.isPassthrough(c.str("source"), c.str("translated"), c.str("direction")),
            )
        }
    }

    @TestFactory
    fun extract(): List<DynamicTest> = cases("extract.json").mapIndexed { i, c ->
        DynamicTest.dynamicTest(label(i, c.str("kind") + " " + c.str("text"))) {
            val actual = when (c.str("kind")) {
                "field" -> ResponseParser.extractField(c.str("text"), c.str("field"))
                else -> ResponseParser.extractRawTranslation(c.str("text"))
            }
            assertEquals(c.str("expected"), actual)
        }
    }

    @TestFactory
    fun requestBodies(): List<DynamicTest> = cases("request_bodies.json").mapIndexed { i, c ->
        DynamicTest.dynamicTest(label(i, "${c.str("model")} ${c.str("max_tokens")} json=${c.str("json")}")) {
            val request = LlmRequest(
                model = c.str("model"), instructions = c.str("system"), input = c.str("user"),
                maxOutputTokens = c["max_tokens"]!!.jsonPrimitive.int, json = c["json"]!!.jsonPrimitive.boolean,
                kind = UsageKind.FINAL,
            )
            assertEquals(c["expected"]!!.jsonObject, OpenAiProvider.requestBody(request, assets.models))
        }
    }

    @TestFactory
    fun prompts(): List<DynamicTest> {
        val builder = PromptBuilder(assets)
        return cases("prompts.json").mapIndexed { i, c ->
            DynamicTest.dynamicTest(label(i, c.str("kind"))) {
                val actual = when (c.str("kind")) {
                    "system" -> builder.system(direction(c.str("direction")), Mode.valueOf(c.str("mode").uppercase()))
                    "user" -> builder.user(
                        c.str("transcript"),
                        c["context"]!!.jsonArray.map { pair ->
                            val p = pair.jsonArray
                            ContextExchange(p[0].jsonPrimitive.content, p[1].jsonPrimitive.content)
                        },
                        c.str("topic"),
                    )
                    "quick" -> builder.quick(c.str("text"), direction(c.str("direction")))
                    "topic" -> builder.topic(c.str("previous"), c.str("source"), c.str("translation"))
                    "passthrough_retry" -> builder.passthroughRetry(c.str("transcript"), direction(c.str("direction")))
                    else -> error("unknown prompt kind ${c.str("kind")}")
                }
                assertEquals(c.str("expected"), actual)
            }
        }
    }

    @TestFactory
    fun idioms(): List<DynamicTest> {
        val scanner = IdiomScanner(assets.idioms)
        val phrases = ListSerializer(FlaggedPhrase.serializer())
        return cases("idioms.json").mapIndexed { i, c ->
            DynamicTest.dynamicTest(label(i, c.str("text"))) {
                assertEquals(c["pattern_count"]!!.jsonPrimitive.int, scanner.count)
                val hits = scanner.scan(c.str("text"))
                val scanExpected = c["scan"]!!.jsonArray.map { it.jsonObject }
                assertEquals(scanExpected.map { it.str("canonical") }, hits.map { it.phrase })
                assertEquals(scanExpected.map { it["start"]!!.jsonPrimitive.int to it["end"]!!.jsonPrimitive.int }, hits.map { it.spanStart to it.spanEnd })
                val llm = json.decodeFromJsonElement(phrases, c["llm"]!!)
                assertEquals(json.decodeFromJsonElement(phrases, c["expected"]!!), IdiomMerge.merge(hits, llm))
            }
        }
    }

    @TestFactory
    fun direction(): List<DynamicTest> = cases("direction.json").mapIndexed { i, c ->
        DynamicTest.dynamicTest(label(i, "${c.str("setting")} detected=${c.strOrNull("detected")}")) {
            val actual = DirectionResolver.resolve(direction(c.str("setting")), c.strOrNull("detected"), null, assets.languages)
            assertEquals(c.str("expected"), actual.serialName)
        }
    }

    @TestFactory
    fun costs(): List<DynamicTest> = cases("costs.json").mapIndexed { i, c ->
        DynamicTest.dynamicTest(label(i, "${c.str("model")} ${c.str("input")}/${c.str("cached")}/${c.str("output")}")) {
            val meter = CostMeter(assets.models)
            meter.record(
                TokenUsage(
                    inputTokens = c["input"]!!.jsonPrimitive.int, cachedInputTokens = c["cached"]!!.jsonPrimitive.int,
                    outputTokens = c["output"]!!.jsonPrimitive.int, reasoningTokens = 0, model = c.str("model"),
                ),
                UsageKind.FINAL,
            )
            val expected = c["expected"]!!.jsonObject
            val totals = meter.totals.value
            assertEquals(expected["session_cost_usd"]!!.jsonPrimitive.double, totals.sessionCostUsd, 1e-12)
            assertEquals(expected["session_input_tokens"]!!.jsonPrimitive.double.toLong(), totals.sessionInputTokens)
            assertEquals(expected["session_output_tokens"]!!.jsonPrimitive.double.toLong(), totals.sessionOutputTokens)
        }
    }
}
