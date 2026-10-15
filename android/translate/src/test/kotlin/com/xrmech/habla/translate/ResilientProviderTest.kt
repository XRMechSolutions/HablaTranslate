package com.xrmech.habla.translate

import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import java.net.UnknownHostException
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ResilientProviderTest {

    private val models = TestAssets.assets.models
    private val final = LlmRequest("gpt-5.4-mini", "SYS", "USER", 1024, true, UsageKind.FINAL, maxRetries = 3)
    private val quick = LlmRequest("gpt-6-luna", "", "x", 256, false, UsageKind.QUICK)

    private fun TestScope.setup(vararg replies: Reply): Pair<FakeOpenAi, ResilientProvider> {
        val fake = FakeOpenAi(this, replies.toList())
        val provider = ResilientProvider(OpenAiProvider(fake.client, { "sk-test-0000000000" }, models), testScheduler.timeSource)
        return fake to provider
    }

    private suspend fun ResilientProvider.fail(request: LlmRequest): LlmException =
        try {
            complete(request)
            error("expected LlmException")
        } catch (e: LlmException) {
            e
        }

    private val ok = Reply.Fixture("ok_text.json")
    private val err500 = Reply.Fixture("err_500.json", 500)

    @Test
    fun serverServerOkWaitsOneThenTwoSeconds() = runTest {
        val (fake, provider) = setup(err500, err500, ok)
        provider.complete(final)
        assertEquals(listOf(0L, 1_000L, 3_000L), fake.requestTimes)
    }

    @Test
    fun networkFourTimesBacksOffOneTwoFourThenRethrows() = runTest {
        val down = Reply.Throw(UnknownHostException("api.openai.com"))
        val (fake, provider) = setup(down, down, down, down)
        assertEquals(Kind.NETWORK, provider.fail(final).kind)
        assertEquals(listOf(0L, 1_000L, 3_000L, 7_000L), fake.requestTimes)
        assertEquals(7_000L, currentTime)
    }

    @Test
    fun retryAfterMsBeatsRetryAfterAndBackoff() = runTest {
        val (fake, provider) = setup(
            Reply.Fixture("err_429_rate.json", 429, mapOf("retry-after-ms" to "1800", "retry-after" to "5")),
            ok,
        )
        provider.complete(final)
        assertEquals(listOf(0L, 1_800L), fake.requestTimes)
    }

    @Test
    fun retryAfterSecondsLongerThanBackoff() = runTest {
        val (fake, provider) = setup(Reply.Fixture("err_429_rate.json", 429, mapOf("retry-after" to "2")), ok)
        provider.complete(final)
        assertEquals(listOf(0L, 2_000L), fake.requestTimes)
    }

    @Test
    fun retryAfterShorterThanBackoffUsesBackoff() = runTest {
        val (fake, provider) = setup(
            err500,
            Reply.Fixture("err_503_overloaded.json", 503, mapOf("retry-after" to "0.5")),
            ok,
        )
        provider.complete(final)
        assertEquals(listOf(0L, 1_000L, 3_000L), fake.requestTimes)
    }

    @Test
    fun retryAfterAboveMaxBackoffIsNotRetried() = runTest {
        val (fake, provider) = setup(Reply.Fixture("err_429_rate_long.json", 429, mapOf("retry-after" to "60")), ok)
        val e = provider.fail(final)
        assertEquals(Kind.RATE_LIMITED, e.kind)
        assertEquals(60_000L, e.retryAfterMs)
        assertEquals(1, fake.requests.size)
        assertEquals(0L, currentTime)
    }

    @Test
    fun quotaAndInvalidKeyAreNotRetried() = runTest {
        val (fake, provider) = setup(Reply.Fixture("err_429_insufficient_quota.json", 429), Reply.Fixture("err_401.json", 401))
        assertEquals(Kind.QUOTA, provider.fail(final).kind)
        assertEquals(Kind.INVALID_KEY, provider.fail(final).kind)
        assertEquals(2, fake.requests.size)
    }

    @Test
    fun maxRetriesZeroMakesOneAttempt() = runTest {
        val (fake, provider) = setup(err500, ok)
        assertEquals(Kind.SERVER, provider.fail(final.copy(maxRetries = 0)).kind)
        assertEquals(1, fake.requests.size)
    }

    @Test
    fun rateGateSpacesConcurrentStartsBy500ms() = runTest {
        val (fake, provider) = setup(ok, ok, ok)
        (1..3).map { async { provider.complete(quick) } }.awaitAll()
        assertEquals(listOf(0L, 500L, 1_000L), fake.requestTimes)
    }

    @Test
    fun rateGateAppliesOncePerCallNotPerRetry() = runTest {
        val (fake, provider) = setup(err500, ok, ok)
        provider.complete(final)
        // Retry at 1 000 ms is not gated; the next call starts 500 ms after the first call's start (already past).
        provider.complete(quick)
        assertEquals(listOf(0L, 1_000L, 1_000L), fake.requestTimes)
    }

    @Test
    fun degradedModeEntryCapAndExit() = runTest {
        val slow = Reply.Fixture("ok_text.json", delayMs = 20_000)
        val (fake, provider) = setup(slow, slow, slow, Reply.Fixture("ok_text.json", delayMs = 9_000), ok)
        val seen = mutableListOf<Boolean>()
        val watcher = launch { provider.degraded.collect { seen += it } }
        runCurrent()

        repeat(3) { assertEquals(Kind.TIMEOUT, provider.fail(quick).kind) }
        assertTrue(provider.degraded.value)

        // Degraded: a final gets 0 retries and an 8 s cap (the response would take 9 s).
        val before = fake.requests.size
        assertEquals(Kind.TIMEOUT, provider.fail(final).kind)
        assertEquals(before + 1, fake.requests.size)
        assertEquals(8_000L, currentTime - fake.requestTimes.last())

        provider.complete(quick)
        assertFalse(provider.degraded.value)
        runCurrent()
        watcher.cancel()
        assertEquals(listOf(false, true, false), seen)
    }

    @Test
    fun nonTimeoutFailuresDoNotResetTheTimeoutCounter() = runTest {
        val slow = Reply.Fixture("ok_text.json", delayMs = 20_000)
        val (_, provider) = setup(slow, slow, err500, slow)
        repeat(2) { provider.fail(quick) }
        assertEquals(Kind.SERVER, provider.fail(quick).kind)
        assertFalse(provider.degraded.value)
        provider.fail(quick)
        assertTrue(provider.degraded.value)
    }

    @Test
    fun cancellationDuringBackoffPropagatesAndIsNotCounted() = runTest {
        val slow = Reply.Fixture("ok_text.json", delayMs = 20_000)
        val (fake, provider) = setup(slow, slow, err500, slow)
        repeat(2) { provider.fail(quick) }
        var thrown: Throwable? = null
        val job = launch {
            try {
                provider.complete(final)
            } catch (e: Throwable) {
                thrown = e
                throw e
            }
        }
        advanceTimeBy(500) // first attempt failed SERVER; now inside the 1 s backoff
        job.cancel()
        job.join()
        assertTrue(thrown is CancellationException, "got $thrown")
        assertFalse(provider.degraded.value)
        val requestsBefore = fake.requests.size
        provider.fail(quick) // third real timeout
        assertTrue(provider.degraded.value)
        assertEquals(requestsBefore + 1, fake.requests.size)
    }

    @Test
    fun cancelledGateWaiterDoesNotMoveLastStart() = runTest {
        val (fake, provider) = setup(ok, ok)
        provider.complete(quick) // starts at 0
        advanceTimeBy(100)
        val waiter = launch { provider.complete(quick) } // must wait until 500
        advanceTimeBy(100)
        waiter.cancel()
        waiter.join()
        advanceTimeBy(100)
        provider.complete(quick) // still gated relative to the first start
        assertEquals(listOf(0L, 500L), fake.requestTimes)
    }
}
