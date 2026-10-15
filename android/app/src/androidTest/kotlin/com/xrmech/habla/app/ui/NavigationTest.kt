package com.xrmech.habla.app.ui

import androidx.compose.ui.test.DeviceConfigurationOverride
import androidx.compose.ui.test.ForcedSize
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.app.testing.KoinFakesRule
import com.xrmech.habla.app.ui.nav.HablaNavHost
import com.xrmech.habla.app.ui.nav.TopLevelDestination
import com.xrmech.habla.app.ui.theme.HablaTheme
import com.xrmech.habla.data.SessionRow
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import java.time.Instant
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NavigationTest {
    private val koin = KoinFakesRule()
    private val compose = createComposeRule()

    @get:Rule
    val rules: RuleChain = RuleChain.outerRule(koin).around(compose)

    private fun tag(d: TopLevelDestination) = "nav_${d.name.lowercase()}"

    @Test
    fun everyTopLevelDestinationIsReachableFromEveryOtherWithItsItemSelected() {
        compose.setContent { HablaTheme { HablaNavHost() } }
        compose.onNodeWithTag("nav_bar").assertIsDisplayed()
        compose.onNodeWithTag(tag(TopLevelDestination.LIVE)).assertIsSelected()
        for (from in TopLevelDestination.entries) {
            for (to in TopLevelDestination.entries) {
                compose.onNodeWithTag(tag(from)).performClick()
                compose.onNodeWithTag(tag(from)).assertIsSelected()
                compose.onNodeWithTag(tag(to)).performClick()
                compose.waitForIdle()
                compose.onNodeWithTag(tag(to)).assertIsSelected()
            }
        }
    }

    @Test
    fun sessionDetailMarksHistory() {
        val t = Instant.parse("2026-10-15T20:15:00Z")
        koin.history.sessionRows.value = listOf(
            SessionRow(7, t, t.plusSeconds(60), Mode.CONVERSATION, Direction.ES_TO_EN, null, null, "m", null, 0, 0.0),
        )
        compose.setContent { HablaTheme { HablaNavHost() } }
        compose.onNodeWithTag(tag(TopLevelDestination.HISTORY)).performClick()
        compose.onNodeWithTag("session_7").performClick()
        compose.waitForIdle()
        compose.onNodeWithTag("export_menu").assertIsDisplayed()
        compose.onNodeWithTag(tag(TopLevelDestination.HISTORY)).assertIsSelected()
    }

    @Test
    fun wideShortWindowUsesTheRailAndPutsListenInTheTopBar() {
        compose.setContent {
            DeviceConfigurationOverride(DeviceConfigurationOverride.ForcedSize(DpSize(800.dp, 400.dp))) {
                HablaTheme { HablaNavHost() }
            }
        }
        compose.onNodeWithTag("nav_rail").assertIsDisplayed()
        compose.onNodeWithTag("listen_top").assertIsDisplayed()
        compose.onNodeWithTag("listen_bottom").assertDoesNotExist()
        compose.onNodeWithTag(tag(TopLevelDestination.VOCAB)).performClick()
        compose.onNodeWithTag(tag(TopLevelDestination.VOCAB)).assertIsSelected()
    }

    @Test
    fun phoneWindowKeepsListenInTheBottomBar() {
        compose.setContent { HablaTheme { HablaNavHost() } }
        compose.onNodeWithTag("listen_bottom").assertIsDisplayed()
        compose.onNodeWithTag("listen_top").assertDoesNotExist()
    }
}
