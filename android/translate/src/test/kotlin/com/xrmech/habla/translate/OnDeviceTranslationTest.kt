package com.xrmech.habla.translate

import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** D050: finals and previews through [OnDeviceTranslator] never reach the LLM provider. */
class OnDeviceTranslationTest {

    private val assets = TestAssets.assets
    private val models = assets.models

    private class FakeOnDevice(private val result: (String, Direction) -> OnDeviceResult) : OnDeviceTranslator {
        val calls = mutableListOf<Pair<String, Direction>>()
        override suspend fun translate(text: String, direction: Direction): OnDeviceResult {
            calls += text to direction
            return result(text, direction)
        }
    }

    private fun settings(final: TranslateEngine = TranslateEngine.ON_DEVICE, preview: TranslateEngine = TranslateEngine.ON_DEVICE) =
        TranslateSettings("gpt-5.4-mini", "gpt-6-luna", partialTranslation = true, finalEngine = final, previewEngine = preview)

    @Test
    fun onDeviceFinalKeepsTheTranscriptAndMakesNoLlmCall() = runTest {
        val provider = FakeProvider()
        val engine = FakeOnDevice { _, _ -> OnDeviceResult.Ok("  I couldn't care less. ") }
        val costs = CostMeter(models)
        val t = Translator(provider, assets, { settings() }, costs, engine)

        val ok = assertIs<TranslationOutcome.Ok>(t.translate("me importa un pepino", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), ""))

        assertEquals(OnDeviceTranslator.MODEL_ID, ok.model)
        assertEquals("me importa un pepino", ok.result.corrected)
        assertEquals("I couldn't care less.", ok.result.translated)
        assertTrue(ok.result.flaggedPhrases.isEmpty())
        assertEquals(false, ok.result.isCorrection)
        assertEquals(listOf("me importa un pepino" to Direction.ES_TO_EN), engine.calls)
        assertTrue(provider.requests.isEmpty())
        assertEquals(0.0, costs.totals.value.sessionCostUsd)
    }

    @Test
    fun missingPackIsModelUnavailableWithTheTranscriptAsRaw() = runTest {
        val t = Translator(FakeProvider(), assets, { settings() }, CostMeter(models), FakeOnDevice { _, _ -> OnDeviceResult.NotReady })
        val failed = assertIs<TranslationOutcome.Failed>(t.translate("hola", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), ""))
        assertEquals(Kind.MODEL_UNAVAILABLE, failed.error.kind)
        assertEquals(Translator.PACK_MISSING, failed.error.message)
        assertEquals("hola", failed.raw)
    }

    @Test
    fun engineFailureAndNoEngineAreFailures() = runTest {
        val broken = Translator(FakeProvider(), assets, { settings() }, CostMeter(models), FakeOnDevice { _, _ -> OnDeviceResult.Failed("boom") })
        assertEquals(Kind.SERVER, assertIs<TranslationOutcome.Failed>(broken.translate("hola", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), "")).error.kind)

        val none = Translator(FakeProvider(), assets, { settings() }, CostMeter(models))
        assertEquals(Kind.MODEL_UNAVAILABLE, assertIs<TranslationOutcome.Failed>(none.translate("hola", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), "")).error.kind)
    }

    @Test
    fun cloudFinalEngineIgnoresTheOnDeviceTranslator() = runTest {
        val provider = FakeProvider(FakeProvider.Respond(FakeProvider.json("Hola.", "Hello.", 0.9)))
        val engine = FakeOnDevice { _, _ -> OnDeviceResult.Ok("wrong") }
        val t = Translator(provider, assets, { settings(final = TranslateEngine.CLOUD) }, CostMeter(models), engine)
        val ok = assertIs<TranslationOutcome.Ok>(t.translate("hola", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), ""))
        assertEquals("Hello.", ok.result.translated)
        assertTrue(engine.calls.isEmpty())
    }

    @Test
    fun topicSummaryIsSkippedForOnDeviceFinals() = runTest {
        val provider = FakeProvider()
        val t = Translator(provider, assets, { settings() }, CostMeter(models), FakeOnDevice { _, _ -> OnDeviceResult.Ok("x") })
        assertEquals("previous", t.updateTopicSummary("previous", "hola", "hello"))
        assertTrue(provider.requests.isEmpty())
    }

    @Test
    fun onDevicePreviewResolvesAutoAndMakesNoLlmCall() = runTest {
        val provider = FakeProvider()
        val engine = FakeOnDevice { _, _ -> OnDeviceResult.Ok(" hello ") }
        val q = QuickTranslator(provider, assets, { settings() }, CostMeter(models), engine)

        assertEquals("hello", q.translate("hola", Direction.AUTO))
        assertEquals(listOf("hola" to Direction.ES_TO_EN), engine.calls)
        assertTrue(provider.requests.isEmpty())
    }

    @Test
    fun onDevicePreviewIsNullWhenNotReadyFailedOrOff() = runTest {
        val notReady = QuickTranslator(FakeProvider(), assets, { settings() }, CostMeter(models), FakeOnDevice { _, _ -> OnDeviceResult.NotReady })
        assertNull(notReady.translate("hola", Direction.ES_TO_EN))
        val failed = QuickTranslator(FakeProvider(), assets, { settings() }, CostMeter(models), FakeOnDevice { _, _ -> OnDeviceResult.Failed("x") })
        assertNull(failed.translate("hola", Direction.ES_TO_EN))
        val engine = FakeOnDevice { _, _ -> OnDeviceResult.Ok("hello") }
        val off = QuickTranslator(FakeProvider(), assets, { settings().copy(partialTranslation = false) }, CostMeter(models), engine)
        assertNull(off.translate("hola", Direction.ES_TO_EN))
        assertTrue(engine.calls.isEmpty())
    }
}
