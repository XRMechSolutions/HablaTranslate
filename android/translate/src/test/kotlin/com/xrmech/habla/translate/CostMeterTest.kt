package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.yield
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class CostMeterTest {

    private val models = TestAssets.assets.models

    private fun usage(model: String, input: Int, cached: Int, output: Int) = TokenUsage(input, cached, output, 0, model)

    @Test
    fun interfacesExamples() {
        val meter = CostMeter(models)
        assertEquals(0.00012, meter.costOf(usage("gpt-6-luna", 600, 0, 120)), 1e-12)
        assertEquals(0.0012948, meter.costOf(usage("gpt-5.4-mini", 2048, 1024, 100)), 1e-12)
    }

    @Test
    fun cachedGreaterThanInputIsCapped() {
        val meter = CostMeter(models)
        // All 100 input tokens billed at the cached rate: 100 * 0.01e-6 + 10 * 0.50e-6.
        assertEquals(100 * 0.01e-6 + 10 * 0.50e-6, meter.costOf(usage("gpt-6-luna", 100, 500, 10)), 1e-15)
    }

    @Test
    fun reasoningTokensAreNotAddedAgain() {
        val meter = CostMeter(models)
        val withReasoning = TokenUsage(600, 0, 120, 100, "gpt-6-luna")
        assertEquals(0.00012, meter.costOf(withReasoning), 1e-12)
    }

    @Test
    fun unknownModelCostsNothingButCountsTokensAndWarnsOnce() {
        val lines = mutableListOf<String>()
        val previous = TranslateLogging.sink
        TranslateLogging.sink = TranslateLog { _, m -> lines += m }
        try {
            val meter = CostMeter(models)
            meter.record(usage("gpt-unknown", 600, 0, 120), UsageKind.FINAL)
            meter.record(usage("gpt-unknown", 400, 0, 80), UsageKind.FINAL)
            val t = meter.totals.value
            assertEquals(0.0, t.sessionCostUsd)
            assertEquals(1000L, t.sessionInputTokens)
            assertEquals(200L, t.sessionOutputTokens)
        } finally {
            TranslateLogging.sink = previous
        }
        assertEquals(1, lines.count { "gpt-unknown" in it })
    }

    @Test
    fun concurrentRecordsSumExactly() = runBlocking {
        val meter = CostMeter(models)
        (1..1000).map {
            async(Dispatchers.Default) { meter.record(usage("gpt-6-luna", 10, 0, 2), UsageKind.QUICK) }
        }.awaitAll()
        val t = meter.totals.value
        assertEquals(10_000L, t.sessionInputTokens)
        assertEquals(2_000L, t.sessionOutputTokens)
        assertEquals(1000 * meter.costOf(usage("gpt-6-luna", 10, 0, 2)), t.sessionCostUsd, 1e-12)
    }

    @Test
    fun recordsEmitsOnePerCall() = runTest {
        val meter = CostMeter(models)
        val collected = async { meter.records.take(2).toList() }
        yield()
        launch {
            meter.record(usage("gpt-6-luna", 600, 0, 120), UsageKind.FINAL)
            meter.record(usage("gpt-5.4-mini", 2048, 1024, 100), UsageKind.TOPIC)
        }
        val records = collected.await()
        assertEquals(listOf(UsageKind.FINAL, UsageKind.TOPIC), records.map { it.kind })
        assertEquals(0.00012, records[0].costUsd, 1e-12)
        assertEquals(0.0012948, records[1].costUsd, 1e-12)
        assertEquals("gpt-5.4-mini", records[1].usage.model)
    }

    @Test
    fun resetSessionKeepsTotalAndSeedTotalSetsIt() {
        val meter = CostMeter(models)
        meter.seedTotal(1.5)
        meter.record(usage("gpt-6-luna", 600, 0, 120), UsageKind.FINAL)
        assertEquals(1.5 + 0.00012, meter.totals.value.totalCostUsd, 1e-12)
        meter.resetSession()
        val t = meter.totals.value
        assertEquals(0L, t.sessionInputTokens)
        assertEquals(0L, t.sessionOutputTokens)
        assertEquals(0.0, t.sessionCostUsd)
        assertEquals(1.5 + 0.00012, t.totalCostUsd, 1e-12)
    }
}
