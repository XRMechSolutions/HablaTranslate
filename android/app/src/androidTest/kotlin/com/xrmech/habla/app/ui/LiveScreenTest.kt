package com.xrmech.habla.app.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.app.pipeline.CardTranslation
import com.xrmech.habla.app.pipeline.ExchangeCard
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.PartialLine
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.testing.KoinFakesRule
import com.xrmech.habla.app.testing.TestFixtures
import com.xrmech.habla.app.ui.live.LiveActions
import com.xrmech.habla.app.ui.live.LiveContent
import com.xrmech.habla.app.ui.live.LiveUiState
import com.xrmech.habla.app.ui.nav.HablaNavHost
import com.xrmech.habla.app.ui.theme.HablaTheme
import com.xrmech.habla.data.EngineChoice
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.OnDeviceTranslator
import com.xrmech.habla.translate.model.CorrectionDetail
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.TranslationResult
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LiveScreenTest {
    private val koin = KoinFakesRule()
    private val compose = createComposeRule()

    @get:Rule
    val rules: RuleChain = RuleChain.outerRule(koin).around(compose)

    private val phrase = FlaggedPhrase(phrase = "echar una mano", literal = "throw a hand", meaning = "to help", spanStart = 3, spanEnd = 17)

    private fun card(seq: Int, t: CardTranslation) =
        ExchangeCard(seq, seq.toLong(), 1_760_000_000_000, "me echas una mano", "es", Direction.ES_TO_EN, t)

    private val state = LiveUiState(
        live = LiveState(
            phase = SessionPhase.Listening,
            sessionId = 1,
            startedAtMs = 0,
            exchanges = listOf(
                card(1, CardTranslation.Pending),
                card(
                    2,
                    CardTranslation.Done(
                        TranslationResult(
                            "¿Me echas una mano?", "Can you give me a hand?", listOf(phrase), 0.87,
                            isCorrection = true, correctionDetail = CorrectionDetail("echas", "echas", "fine"),
                        ),
                        listOf(phrase), "m", 100,
                    ),
                ),
                card(3, CardTranslation.Failed(LlmException.Kind.NO_KEY, "No OpenAI API key stored")),
            ),
            partial = PartialLine(4, "y luego vamos", null, "es", Direction.ES_TO_EN, finalizing = false),
        ),
        settings = com.xrmech.habla.app.state.AppSettingsDefaults(TestFixtures.shared).value.copy(firstRunDone = true),
    )

    @Test
    fun rendersPartialPendingDoneAndFailedCards() {
        compose.setContent { HablaTheme { LiveContent(state, LiveActions(), nowMs = { 65_000 }) } }
        compose.onNodeWithTag("partial_text").assertTextContains("y luego vamos", substring = true)
        compose.onNodeWithTag("card_1_pending").assertIsDisplayed()
        compose.onNodeWithTag("card_2_translation").assertTextContains("Can you give me a hand?")
        compose.onNodeWithTag("card_2_confidence").assertTextContains("87%")
        compose.onNodeWithTag("card_2_correction").assertIsDisplayed()
        compose.onNodeWithTag("card_3_failed").assertTextContains("Not translated - no OpenAI key")
        compose.onNodeWithTag("card_3_action").assertTextContains("Add key")
        compose.onNodeWithTag("status_label").assertTextContains("Listening 1:05")
    }

    @Test
    fun chipSaveCallsVocab() {
        val saved = mutableListOf<Pair<Int, String>>()
        compose.setContent {
            HablaTheme { LiveContent(state, LiveActions(saveVocab = { c, p -> saved += c.seq to p.phrase })) }
        }
        compose.onNodeWithTag("chip_2_echar una mano").performClick()
        compose.onNodeWithText("Literally: throw a hand").assertIsDisplayed()
        compose.onNodeWithTag("chip_2_echar una mano_save").performClick()
        assertEquals(listOf(2 to "echar una mano"), saved)
    }

    @Test
    fun chipShowsSavedAfterSavingThroughTheViewModel() {
        koin.live.state.value = state.live
        compose.setContent { HablaTheme { HablaNavHost() } }
        compose.onNodeWithTag("chip_2_echar una mano").performClick()
        compose.onNodeWithTag("chip_2_echar una mano_save").performClick()
        compose.waitUntil(5_000) { koin.vocab.saved.isNotEmpty() }
        compose.onNodeWithTag("chip_2_echar una mano_saved").assertIsDisplayed()
        assertEquals("echar una mano", koin.vocab.saved.single().phrase)
    }

    @Test
    fun noKeyBannerAddKeyOpensSettings() {
        // Only the OpenAI engine needs a key; Automatic with no key translates on the phone (D050).
        runBlocking { koin.settings.update { it.copy(finalEngine = EngineChoice.CLOUD) } }
        compose.setContent { HablaTheme { HablaNavHost() } }
        compose.waitUntil(5_000) { compose.onAllNodes(androidx.compose.ui.test.hasTestTag("banner_NO_KEY")).fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithText("Translation off - add your OpenAI key").assertIsDisplayed()
        compose.onNodeWithTag("banner_NO_KEY_action").performClick()
        compose.waitForIdle()
        compose.onNodeWithTag("section_key").assertIsDisplayed()
        compose.onNodeWithTag("nav_settings").assertIsDisplayedSelected()
    }

    @Test
    fun onDeviceCardShowsGoogleAttributionAndNoConfidence() {
        val onDevice = state.copy(
            live = state.live.copy(
                exchanges = listOf(
                    card(5, CardTranslation.Done(TranslationResult("hola", "hello", confidence = 1.0), emptyList(), OnDeviceTranslator.MODEL_ID, 40)),
                ),
                partial = null,
            ),
        )
        compose.setContent { HablaTheme { LiveContent(onDevice, LiveActions(), nowMs = { 65_000 }) } }
        compose.onNodeWithTag("card_5_translation").assertTextContains("hello")
        compose.onNodeWithTag("google_attribution").assertIsDisplayed()
        compose.onNodeWithTag("card_5_confidence").assertDoesNotExist()
    }
}

private fun androidx.compose.ui.test.SemanticsNodeInteraction.assertIsDisplayedSelected() {
    assertIsDisplayed()
    assertIsSelected()
}
