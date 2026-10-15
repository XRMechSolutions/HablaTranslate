package com.xrmech.habla.translate

import com.xrmech.habla.translate.FakeProvider.Fail
import com.xrmech.habla.translate.FakeProvider.Respond
import com.xrmech.habla.translate.FakeProvider.Run
import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.TokenUsage
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TopicTrackerTest {

    private val assets = TestAssets.assets

    private fun setup(provider: FakeProvider): Pair<ConversationContext, TopicTracker> {
        val ctx = ConversationContext(assets.asrProfile)
        val translator = Translator(provider, assets, { TranslateSettings("gpt-5.4-mini", "gpt-6-luna", true) }, CostMeter(assets.models))
        return ctx to TopicTracker(translator, ctx)
    }

    private fun slow(text: String, ms: Long) = Run { r ->
        delay(ms)
        LlmResponse(text, TokenUsage(1, 0, 1, 0, r.model), incomplete = false)
    }

    @Test
    fun concurrentUpdatesRunInOrderAndSecondSeesFirst() = runTest {
        val provider = FakeProvider(slow("Summary one.", 1_000), slow("Summary two.", 1_000))
        val (ctx, tracker) = setup(provider)
        val a = launch { tracker.update("uno", "one") }
        val b = launch { tracker.update("dos", "two") }
        a.join()
        b.join()
        assertEquals("Summary two.", ctx.topicSummary)
        assertEquals(2, provider.requests.size)
        val prompt = PromptBuilder(assets)
        assertEquals(prompt.topic("", "uno", "one"), provider.requests[0].input)
        assertEquals(prompt.topic("Summary one.", "dos", "two"), provider.requests[1].input)
    }

    @Test
    fun blankResultOrFailureKeepsPrevious() = runTest {
        val (ctx, tracker) = setup(FakeProvider(Respond("Start."), Respond("   "), Fail(Kind.NO_KEY)))
        tracker.update("a", "b")
        assertEquals("Start.", ctx.topicSummary)
        tracker.update("c", "d")
        assertEquals("Start.", ctx.topicSummary)
        tracker.update("e", "f")
        assertEquals("Start.", ctx.topicSummary)
    }

    @Test
    fun cancelledUpdateWritesNothing() = runTest {
        val (ctx, tracker) = setup(FakeProvider(slow("Never written.", 5_000)))
        ctx.topicSummary = "Kept."
        val job = launch { tracker.update("a", "b") }
        advanceTimeBy(1_000)
        job.cancel()
        job.join()
        assertTrue(job.isCancelled)
        assertEquals("Kept.", ctx.topicSummary)
    }
}
