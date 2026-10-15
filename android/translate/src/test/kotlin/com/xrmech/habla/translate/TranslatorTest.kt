package com.xrmech.habla.translate

import com.xrmech.habla.translate.FakeProvider.Fail
import com.xrmech.habla.translate.FakeProvider.Respond
import com.xrmech.habla.translate.FakeProvider.Run
import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class TranslatorTest {

    private val assets = TestAssets.assets
    private val models = assets.models
    private val transcript = "me importa un pepino"
    private val good = FakeProvider.json("Me importa un pepino.", "I couldn't care less.", 0.95)

    private fun translator(provider: FakeProvider, finalModel: String = "gpt-5.4-mini", costs: CostMeter = CostMeter(models)) =
        Translator(provider, assets, { TranslateSettings(finalModel, "gpt-6-luna", true) }, costs)

    private suspend fun Translator.run(text: String = transcript, direction: Direction = Direction.ES_TO_EN) =
        translate(text, direction, Mode.CONVERSATION, emptyList(), "")

    @Test
    fun primaryRequestFields() = runTest {
        val provider = FakeProvider(Respond(good))
        val ok = assertIs<TranslationOutcome.Ok>(translator(provider).run())
        assertEquals("I couldn't care less.", ok.result.translated)
        assertEquals("gpt-5.4-mini", ok.model)
        val r = provider.requests.single()
        assertEquals(LlmRequest("gpt-5.4-mini", PromptBuilder(assets).system(Direction.ES_TO_EN, Mode.CONVERSATION),
            PromptBuilder(assets).user(transcript, emptyList(), ""), 1024, true, UsageKind.FINAL, 3), r)
    }

    @Test
    fun kindsThatFallBack() = runTest {
        for (kind in listOf(Kind.MODEL_UNAVAILABLE, Kind.BAD_REQUEST, Kind.RATE_LIMITED, Kind.SERVER, Kind.TIMEOUT, Kind.MALFORMED)) {
            val provider = FakeProvider(Fail(kind), Respond(good))
            val ok = assertIs<TranslationOutcome.Ok>(translator(provider).run(), "$kind")
            assertEquals("gpt-6-luna", ok.model, "Ok.model is the producing model ($kind)")
            assertEquals(listOf("gpt-5.4-mini", "gpt-6-luna"), provider.requests.map { it.model })
            assertEquals(0, provider.requests[1].maxRetries)
            assertEquals(provider.requests[0].input, provider.requests[1].input)
        }
    }

    @Test
    fun kindsThatDoNotFallBack() = runTest {
        for (kind in listOf(Kind.NO_KEY, Kind.INVALID_KEY, Kind.QUOTA, Kind.NETWORK)) {
            val provider = FakeProvider(Fail(kind))
            val failed = assertIs<TranslationOutcome.Failed>(translator(provider).run(), "$kind")
            assertEquals(kind, failed.error.kind)
            assertEquals(transcript, failed.raw)
            assertEquals(1, provider.requests.size, "$kind must not walk the fallback list")
        }
    }

    @Test
    fun primaryErrorSurfacedWhenAllFail() = runTest {
        val provider = FakeProvider(Fail(Kind.SERVER, 500), Fail(Kind.MODEL_UNAVAILABLE, 404))
        val failed = assertIs<TranslationOutcome.Failed>(translator(provider).run())
        assertEquals(Kind.SERVER, failed.error.kind)
        assertEquals(500, failed.error.httpStatus)
    }

    @Test
    fun primaryAbsentFromFallbackOrderWalksWholeList() = runTest {
        val provider = FakeProvider(Fail(Kind.MODEL_UNAVAILABLE), Fail(Kind.SERVER), Respond(good))
        val ok = assertIs<TranslationOutcome.Ok>(translator(provider, finalModel = "gpt-5.4-nano").run())
        assertEquals(listOf("gpt-5.4-nano", "gpt-5.4-mini", "gpt-6-luna"), provider.requests.map { it.model })
        assertEquals("gpt-6-luna", ok.model)
    }

    @Test
    fun fallbackSkipsIdsMissingFromTheTable() = runTest {
        val table = models.copy(fallbackOrder = listOf("gpt-5.4-mini", "ghost-model", "gpt-6-luna"))
        val provider = FakeProvider(Fail(Kind.SERVER), Respond("x"))
        val (_, model) = ModelFallback(table).call(provider, LlmRequest("gpt-5.4-mini", "s", "u", 10, true, UsageKind.FINAL, 3)) {}
        assertEquals("gpt-6-luna", model)
        assertEquals(listOf("gpt-5.4-mini", "gpt-6-luna"), provider.requests.map { it.model })
    }

    @Test
    fun unknownFinalModelUsesFinalDefault() = runTest {
        val provider = FakeProvider(Respond(good))
        translator(provider, finalModel = "gpt-removed").run()
        assertEquals(models.finalDefault, provider.requests.single().model)
    }

    @Test
    fun incompleteReissuedOnceAt2048AndBothUsagesRecorded() = runTest {
        val costs = CostMeter(models)
        val provider = FakeProvider(Fail(Kind.SERVER), Respond("{\"translated\": \"I co", incomplete = true, input = 100), Respond(good, incomplete = true, input = 200))
        val ok = assertIs<TranslationOutcome.Ok>(translator(provider, costs = costs).run())
        assertEquals("I couldn't care less.", ok.result.translated)
        assertEquals("gpt-6-luna", ok.model)
        val reissue = provider.requests[2]
        assertEquals("gpt-6-luna", reissue.model, "re-issue on the model that produced the incomplete output")
        assertEquals(2048, reissue.maxOutputTokens)
        assertEquals(0, reissue.maxRetries)
        assertEquals(300L, costs.totals.value.sessionInputTokens)
    }

    @Test
    fun emptyTextBecomesBracesThenPassthroughRetrySucceeds() = runTest {
        val costs = CostMeter(models)
        val records = mutableListOf<CostRecord>()
        val job = launch { costs.records.collect { records += it } }
        testScheduler.runCurrent()
        val provider = FakeProvider(Respond(""), Respond(FakeProvider.json("Me importa un pepino", "I don't care", 0.9)))
        val ok = assertIs<TranslationOutcome.Ok>(translator(provider, costs = costs).run())
        assertEquals("I don't care", ok.result.translated)
        assertEquals(0.5, ok.result.confidence)
        val retry = provider.requests[1]
        assertEquals(512, retry.maxOutputTokens)
        assertEquals(0, retry.maxRetries)
        assertTrue(retry.json)
        assertEquals(provider.requests[0].instructions, retry.instructions)
        assertEquals(PromptBuilder(assets).passthroughRetry(transcript, Direction.ES_TO_EN), retry.input)
        testScheduler.runCurrent()
        job.cancel()
        assertEquals(listOf(UsageKind.FINAL, UsageKind.FINAL), records.map { it.kind })
    }

    @Test
    fun failedPassthroughRetryCapsConfidenceAtPointTwo() = runTest {
        val echo = FakeProvider.json(transcript, transcript, 0.9)
        val provider = FakeProvider(Respond(echo), Fail(Kind.SERVER), Fail(Kind.SERVER))
        val ok = assertIs<TranslationOutcome.Ok>(translator(provider).run())
        assertEquals(transcript, ok.result.translated)
        assertEquals(0.2, ok.result.confidence)
        assertEquals("gpt-5.4-mini", ok.model)
    }

    @Test
    fun retryThatIsStillPassthroughKeepsFirstResult() = runTest {
        val echo = FakeProvider.json(transcript, transcript, 0.1)
        val provider = FakeProvider(Respond(echo), Respond(echo))
        val ok = assertIs<TranslationOutcome.Ok>(translator(provider).run())
        assertEquals(0.1, ok.result.confidence)
        assertEquals(2, provider.requests.size)
    }

    @Test
    fun invalidModelOutputIsMalformed() = runTest {
        val provider = FakeProvider(Respond("{\"corrected\": 5, \"translated\": \"x\"}"))
        val failed = assertIs<TranslationOutcome.Failed>(translator(provider).run())
        assertEquals(Kind.MALFORMED, failed.error.kind)
        assertEquals(transcript, failed.raw)
    }

    @Test
    fun cancellationPropagates() = runTest {
        val provider = FakeProvider(Run { throw CancellationException("session stopped") })
        assertThrows<CancellationException> { translator(provider).run() }
    }

    @Test
    fun updateTopicSummary() = runTest {
        val costs = CostMeter(models)
        val provider = FakeProvider(
            Respond("  Shopping; asking a price.\n"),
            Respond("   "),
            Fail(Kind.NETWORK),
            Fail(Kind.SERVER), Fail(Kind.SERVER),
        )
        val t = translator(provider, costs = costs)
        assertEquals("Shopping; asking a price.", t.updateTopicSummary("", "¿Cuánto cuesta?", "How much is it?"))
        val r = provider.requests[0]
        assertEquals(LlmRequest("gpt-5.4-mini", "", PromptBuilder(assets).topic("", "¿Cuánto cuesta?", "How much is it?"), 200, false, UsageKind.TOPIC, 0), r)
        assertEquals("prev", t.updateTopicSummary("prev", "a", "b"))
        assertEquals("prev", t.updateTopicSummary("prev", "a", "b"))
        assertEquals("prev", t.updateTopicSummary("prev", "a", "b"))
        assertEquals(listOf("gpt-5.4-mini", "gpt-5.4-mini", "gpt-5.4-mini", "gpt-5.4-mini", "gpt-6-luna"), provider.requests.map { it.model })
        assertEquals(20L, costs.totals.value.sessionInputTokens)
    }

    @Test
    fun recordsKindsForTopicCalls() = runTest {
        val costs = CostMeter(models)
        val got = mutableListOf<CostRecord>()
        val job = launch { costs.records.collect { got += it } }
        testScheduler.runCurrent()
        translator(FakeProvider(Respond("summary")), costs = costs).updateTopicSummary("", "a", "b")
        testScheduler.runCurrent()
        job.cancel()
        assertEquals(listOf(UsageKind.TOPIC), got.map { it.kind })
    }
}
