package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PromptBuilderTest {

    private val assets = TestAssets.assets
    private val builder = PromptBuilder(assets)

    @Test
    fun systemPromptBothDirectionsAndModes() {
        val esEn = builder.system(Direction.ES_TO_EN, Mode.CONVERSATION)
        assertTrue(esEn.startsWith("You are a Spanish-to-English interpreter"))
        val enEs = builder.system(Direction.EN_TO_ES, Mode.CONVERSATION)
        assertTrue(enEs.startsWith("You are a English-to-Spanish interpreter"))
        val extra = assets.prompts.translatorClassroomExtra
        assertTrue(extra.isNotEmpty())
        assertTrue(extra in builder.system(Direction.ES_TO_EN, Mode.CLASSROOM))
        assertTrue(extra in builder.system(Direction.EN_TO_ES, Mode.CLASSROOM))
        assertFalse(extra in esEn)
        // AUTO carries parse_direction("auto")'s fallback codes.
        assertEquals(esEn, builder.system(Direction.AUTO, Mode.CONVERSATION))
    }

    @Test
    fun userPromptWithoutContextOrTopic() {
        assertEquals(
            "Recent conversation:\n(none yet)\n\nTopic context: (no summary yet)\n\nCurrent utterance:\n" +
                "Speaker: Speaker A\nTranscript: \"hola\"\n\nTranslate now. Respond with a single JSON object only. No extra text.",
            builder.user("hola", emptyList(), ""),
        )
    }

    @Test
    fun userPromptWithThreeContextItemsAndTopic() {
        val ctx = listOf(ContextExchange("a", "b"), ContextExchange("c", "d"), ContextExchange("e", "f"))
        val prompt = builder.user("t", ctx, "Topic.")
        assertTrue("  [Speaker A] a → b\n  [Speaker A] c → d\n  [Speaker A] e → f\n" in prompt, prompt)
        assertTrue("Topic context: Topic.\n" in prompt)
    }

    @Test
    fun sevenContextItemsAreSlicedToTheLastFive() {
        val ctx = (1..7).map { ContextExchange("s$it", "t$it") }
        val prompt = builder.user("t", ctx, "")
        assertFalse("s1 →" in prompt)
        assertFalse("s2 →" in prompt)
        assertTrue("  [Speaker A] s3 → t3\n" in prompt)
        assertTrue("  [Speaker A] s7 → t7\n" in prompt)
    }

    @Test
    fun bracesAndQuotesInTranscriptAreInsertedLiterally() {
        val transcript = "dijo \"hola\" y {x} con {{llaves}} y {0}"
        assertTrue("Transcript: \"$transcript\"" in builder.user(transcript, emptyList(), ""))
    }

    @Test
    fun quickPrompt() {
        assertEquals(
            "Translate this Spanish to natural English. Output ONLY the translation, nothing else.\n\n¿dónde está",
            builder.quick("¿dónde está", Direction.ES_TO_EN),
        )
    }

    @Test
    fun topicPromptPreviousEmptyAndSet() {
        assertTrue("Previous summary: (start of conversation)\n" in builder.topic("", "a", "b"))
        val p = builder.topic("Shopping.", "¿Cuánto cuesta?", "How much is it?")
        assertTrue("Previous summary: Shopping.\n" in p)
        assertTrue("  [Speaker A] ¿Cuánto cuesta? → How much is it?\n" in p)
    }

    @Test
    fun passthroughRetryPromptDirections() {
        val esEn = builder.passthroughRetry("hola", Direction.ES_TO_EN)
        assertTrue(esEn.startsWith("The following Spanish text MUST be translated to English."))
        assertTrue(esEn.endsWith("{\"corrected\": \"<corrected Spanish>\", \"translated\": \"<English translation>\", \"confidence\": 0.5}"))
        assertTrue(builder.passthroughRetry("hi", Direction.EN_TO_ES).startsWith("The following English text MUST be translated to Spanish."))
        // The server's _retry_passthrough uses parse_direction("auto") = es -> en.
        assertEquals(esEn, builder.passthroughRetry("hola", Direction.AUTO))
    }
}
