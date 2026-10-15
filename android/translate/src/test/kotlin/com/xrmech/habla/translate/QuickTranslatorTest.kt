package com.xrmech.habla.translate

import com.xrmech.habla.translate.FakeProvider.Fail
import com.xrmech.habla.translate.FakeProvider.Respond
import com.xrmech.habla.translate.FakeProvider.Run
import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class QuickTranslatorTest {

    private val assets = TestAssets.assets
    private val models = assets.models

    private fun quick(provider: FakeProvider, enabled: Boolean = true, quickModel: String = "gpt-6-luna", costs: CostMeter = CostMeter(models)) =
        QuickTranslator(provider, assets, { TranslateSettings("gpt-5.4-mini", quickModel, enabled) }, costs)

    private fun slow(text: String, ms: Long = 1_000) = Run { r ->
        delay(ms)
        LlmResponse(text, TokenUsage(10, 0, 3, 0, r.model), incomplete = false)
    }

    @Test
    fun disabledOrBlankMakesNoCall() = runTest {
        val provider = FakeProvider()
        assertNull(quick(provider, enabled = false).translate("hola", Direction.ES_TO_EN))
        assertNull(quick(provider).translate("   ", Direction.ES_TO_EN))
        assertEquals(0, provider.requests.size)
    }

    @Test
    fun requestFieldsAndTrimmedResult() = runTest {
        val costs = CostMeter(models)
        val provider = FakeProvider(Respond("  where is  \n"))
        assertEquals("where is", quick(provider, costs = costs).translate("¿dónde está", Direction.AUTO))
        assertEquals(
            LlmRequest("gpt-6-luna", "", PromptBuilder(assets).quick("¿dónde está", Direction.ES_TO_EN), 256, false, UsageKind.QUICK, 0),
            provider.requests.single(),
        )
        assertEquals(10L, costs.totals.value.sessionInputTokens)
    }

    @Test
    fun unknownQuickModelUsesQuickDefaultAndNoFallback() = runTest {
        val provider = FakeProvider(Fail(Kind.SERVER))
        assertNull(quick(provider, quickModel = "gone-model").translate("hola", Direction.ES_TO_EN))
        assertEquals(listOf(models.quickDefault), provider.requests.map { it.model })
    }

    @Test
    fun blankOutputIsNullAndIncompleteTextIsUsed() = runTest {
        val provider = FakeProvider(Respond("  "), Respond("partial text", incomplete = true))
        val q = quick(provider)
        assertNull(q.translate("hola", Direction.ES_TO_EN))
        assertEquals("partial text", q.translate("hola", Direction.ES_TO_EN))
    }

    @Test
    fun llmExceptionIsNull() = runTest {
        assertNull(quick(FakeProvider(Fail(Kind.NO_KEY))).translate("hola", Direction.ES_TO_EN))
    }

    @Test
    fun overlappingCallsLatestWins() = runTest {
        val provider = FakeProvider(slow("first"), slow("second"))
        val q = quick(provider)
        val a = async { q.translate("uno", Direction.ES_TO_EN) }
        advanceTimeBy(300)
        val b = async { q.translate("uno dos", Direction.ES_TO_EN) }
        assertNull(a.await())
        assertEquals("second", b.await())
    }

    @Test
    fun cancelInFlightYieldsNull() = runTest {
        val q = quick(FakeProvider(slow("late")))
        val a = async { q.translate("uno", Direction.ES_TO_EN) }
        advanceTimeBy(300)
        q.cancelInFlight()
        assertNull(a.await())
    }

    @Test
    fun callerCancellationRethrows() = runTest {
        val q = quick(FakeProvider(slow("late")))
        var thrown: Throwable? = null
        val job = launch {
            try {
                q.translate("uno", Direction.ES_TO_EN)
            } catch (e: Throwable) {
                thrown = e
                throw e
            }
        }
        advanceTimeBy(300)
        job.cancel()
        job.join()
        assertTrue(thrown is CancellationException, "got $thrown")
    }

    @Test
    fun olderCallerNeverClearsNewerJob() = runTest {
        val provider = FakeProvider(slow("first"), slow("second", 5_000))
        val q = quick(provider)
        val a = async { q.translate("uno", Direction.ES_TO_EN) }
        advanceTimeBy(300)
        val b = async { q.translate("uno dos", Direction.ES_TO_EN) }
        runCurrent()
        assertNull(a.await()) // A's finally has run; B must still be registered.
        q.cancelInFlight()
        assertNull(b.await(), "cancelInFlight must still reach the newer call")
    }
}
