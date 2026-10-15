package com.xrmech.habla.translate

import com.xrmech.habla.shared.LanguageInfo
import com.xrmech.habla.shared.LanguageRegistry
import com.xrmech.habla.translate.model.Direction
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class DirectionResolverTest {

    private val languages = TestAssets.assets.languages

    @Test
    fun interfacesExamples() {
        assertEquals(Direction.EN_TO_ES, DirectionResolver.resolve(Direction.AUTO, "en", null, languages))
        assertEquals(Direction.ES_TO_EN, DirectionResolver.resolve(Direction.AUTO, "fr", "es", languages))
        assertEquals(Direction.ES_TO_EN, DirectionResolver.resolve(Direction.AUTO, null, null, languages))
        assertEquals(Direction.EN_TO_ES, DirectionResolver.resolve(Direction.EN_TO_ES, "es", "es", languages))
    }

    @Test
    fun detectedWinsOverVoted() {
        assertEquals(Direction.ES_TO_EN, DirectionResolver.resolve(Direction.AUTO, "es", "en", languages))
        assertEquals(Direction.EN_TO_ES, DirectionResolver.resolve(Direction.AUTO, null, "en", languages))
    }

    @Test
    fun fixedSettingIgnoresDetection() {
        assertEquals(Direction.ES_TO_EN, DirectionResolver.resolve(Direction.ES_TO_EN, "en", "en", languages))
    }

    @Test
    fun unmappedPairFallsBackToEsToEn() {
        val withFrench = LanguageRegistry(
            defaultTarget = "en",
            languages = linkedMapOf(
                "es" to LanguageInfo("Spanish", "x"), "en" to LanguageInfo("English", "x"), "fr" to LanguageInfo("French", "x"),
            ),
        )
        assertEquals(Direction.ES_TO_EN, DirectionResolver.resolve(Direction.AUTO, "fr", null, withFrench))
    }

    @Test
    fun targetLanguageUsesFirstOtherKeyInRegistryOrder() {
        val reordered = LanguageRegistry(
            defaultTarget = "es",
            languages = linkedMapOf("es" to LanguageInfo("Spanish", "x"), "en" to LanguageInfo("English", "x")),
        )
        assertEquals(Direction.ES_TO_EN, DirectionResolver.resolve(Direction.AUTO, "es", null, reordered))
        assertEquals(Direction.EN_TO_ES, DirectionResolver.resolve(Direction.AUTO, "en", null, reordered))
    }
}
