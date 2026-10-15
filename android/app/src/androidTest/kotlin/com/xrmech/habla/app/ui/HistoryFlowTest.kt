package com.xrmech.habla.app.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextReplacement
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.app.testing.KoinFakesRule
import com.xrmech.habla.app.ui.nav.HablaNavHost
import com.xrmech.habla.app.ui.theme.HablaTheme
import com.xrmech.habla.data.ExchangeRow
import com.xrmech.habla.data.SessionRow
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HistoryFlowTest {
    private val koin = KoinFakesRule()
    private val compose = createComposeRule()

    @get:Rule
    val rules: RuleChain = RuleChain.outerRule(koin).around(compose)

    private val t = Instant.parse("2026-10-15T20:15:00Z")

    private fun exchange(id: Long, session: Long, audio: String?) = ExchangeRow(
        id, session, 1, t, Direction.ES_TO_EN, "hola que tal", "Hola, ¿qué tal?", "Hi, how are you?", 0.9,
        false, null, emptyList(), 100, "es", false, audio, null,
    )

    @Before
    fun seed() {
        koin.history.sessionRows.value = listOf(
            SessionRow(2, t.plusSeconds(3600), t.plusSeconds(3700), Mode.CONVERSATION, Direction.ES_TO_EN, "Lunch", null, "m", null, 1, 0.01),
            SessionRow(1, t, t.plusSeconds(60), Mode.CLASSROOM, Direction.ES_TO_EN, null, null, "m", null, 1, 0.0),
        )
        koin.history.exchangeRows.value = mapOf(
            2L to listOf(exchange(20, 2, "sessions/2/segment_001.wav")),
            1L to listOf(exchange(10, 1, null)),
        )
    }

    @Test
    fun editTranscriptFromSessionDetailSaves() {
        compose.setContent { HablaTheme { HablaNavHost() } }
        compose.onNodeWithTag("nav_history").performClick()
        compose.onNodeWithTag("session_2").performClick()
        compose.waitForIdle()
        compose.onNodeWithTag("topic_summary").assertTextContains("Lunch", substring = true)
        compose.onNodeWithTag("export_menu").assertIsEnabled()
        compose.onNodeWithTag("card_1_edit").performClick()
        compose.onNodeWithTag("edit_field").performTextReplacement("  hola, ¿qué tal?  ")
        compose.onNodeWithTag("edit_save").performClick()
        compose.waitUntil(5_000) { koin.history.corrected.value.isNotEmpty() }
        assertEquals("hola, ¿qué tal?", koin.history.corrected.value[20])
        compose.onNodeWithTag("corrected_20").assertTextContains("hola, ¿qué tal?", substring = true)
    }

    @Test
    fun exportIsDisabledWithoutAudio() {
        compose.setContent { HablaTheme { HablaNavHost() } }
        compose.onNodeWithTag("nav_history").performClick()
        compose.onNodeWithTag("session_1").performClick()
        compose.waitForIdle()
        compose.onNodeWithTag("export_menu").assertIsNotEnabled()
        compose.onNodeWithTag("export_disabled_reason")
            .assertIsDisplayed()
            .assertTextContains("No training audio in this session (turn on Training capture before listening)")
        compose.onNodeWithTag("card_1_edit").assertDoesNotExist()
    }
}
