package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Opt-in check against the real OpenAI API through the production stack (OkHttp engine,
 * OpenAiProvider, ResilientProvider, real shared assets). Skipped unless `OPENAI_API_KEY` holds a
 * key; excluded from CI by the Gradle test filter. The key is read from the environment only and
 * never printed or put in an assertion message. Total cost is well under $0.001.
 */
class OpenAiLiveTest {

    @Test
    fun liveRoundTrip() = runBlocking {
        val key = System.getenv("OPENAI_API_KEY")
        assumeTrue(key?.startsWith("sk-") == true, "OPENAI_API_KEY not set; live test skipped")

        val assets = TestAssets.assets
        val http = OpenAiHttp.create(OpenAiHttp.okHttpEngine(), debugLogging = false)
        try {
            val openAi = OpenAiProvider(http, { key }, assets.models)
            val provider = ResilientProvider(openAi)
            val costs = CostMeter(assets.models)
            val settings = { TranslateSettings(assets.models.finalDefault, assets.models.quickDefault, partialTranslation = true) }

            val valid = openAi.testKey()
            assertIs<KeyTestResult.Valid>(valid, "testKey with the configured key")
            costs.record(valid.usage, UsageKind.TEST)

            val outcome = Translator(provider, assets, settings, costs)
                .translate("¿Dónde está la biblioteca?", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), "")
            val ok = assertIs<TranslationOutcome.Ok>(outcome, "final translation")
            assertTrue("library" in ok.result.translated.lowercase(), "translation mentions the library")
            assertTrue(costs.totals.value.sessionInputTokens > 0)

            assertNotNull(QuickTranslator(provider, assets, settings, costs).translate("¿dónde está", Direction.ES_TO_EN))

            val invalid = openAi.testKey("sk-invalid-000000000000")
            assertEquals(LlmException.Kind.INVALID_KEY, assertIs<KeyTestResult.Failed>(invalid).error.kind)

            assertTrue(costs.totals.value.sessionCostUsd < 0.001, "live test stays under \$0.001")
        } finally {
            http.close()
        }
    }
}
