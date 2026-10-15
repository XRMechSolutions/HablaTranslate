package com.xrmech.habla.app.ui

import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.hasTestTag
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.testing.TestFixtures
import com.xrmech.habla.app.ui.settings.KeyTestMessages
import com.xrmech.habla.app.ui.settings.SettingsActions
import com.xrmech.habla.app.ui.settings.SettingsContent
import com.xrmech.habla.app.ui.settings.SettingsLocal
import com.xrmech.habla.app.ui.settings.SettingsUiState
import com.xrmech.habla.app.ui.settings.SpeechModelOption
import com.xrmech.habla.app.ui.theme.HablaTheme
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.translate.KeyTestResult
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.model.TokenUsage
import androidx.compose.runtime.mutableStateOf
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SettingsScreenTest {
    @get:Rule
    val compose = createComposeRule()

    private val defaults = com.xrmech.habla.app.state.AppSettingsDefaults(TestFixtures.shared).value
    private val base = SettingsUiState(
        keyStatus = "No key",
        keyState = KeyState.ABSENT,
        settings = defaults,
        speechModels = TestFixtures.shared.asrModels.models.map { SpeechModelOption(it.id, it.id, it.sizeBytes) },
    )

    @Test
    fun keyFieldHasPasswordSemantics() {
        compose.setContent { HablaTheme { SettingsContent(base, null, 0, SettingsActions()) } }
        compose.onNodeWithTag("key_field").assert(SemanticsMatcher.keyIsDefined(SemanticsProperties.Password))
    }

    @Test
    fun rendersEveryKeyTestResultText() {
        val quick = TestFixtures.shared.models.quickDefault
        val results = listOf(
            KeyTestResult.Valid(TokenUsage(5, 0, 1, 0, quick)),
            KeyTestResult.Failed(LlmException("x", LlmException.Kind.NO_KEY)),
            KeyTestResult.Failed(LlmException("x", LlmException.Kind.INVALID_KEY, httpStatus = 401)),
            KeyTestResult.Failed(LlmException("x", LlmException.Kind.QUOTA, httpStatus = 429)),
            KeyTestResult.Failed(LlmException("x", LlmException.Kind.MODEL_UNAVAILABLE, httpStatus = 404)),
            KeyTestResult.Failed(LlmException("x", LlmException.Kind.RATE_LIMITED, httpStatus = 429)),
            KeyTestResult.Failed(LlmException("connection reset", LlmException.Kind.NETWORK)),
        )
        val texts = results.map { KeyTestMessages.text(it, quick) }
        val state = mutableStateOf(base)
        compose.setContent { HablaTheme { SettingsContent(state.value, null, 0, SettingsActions()) } }
        texts.forEach { t ->
            state.value = base.copy(local = SettingsLocal(keyTestMessage = t))
            compose.onNodeWithTag("key_test_result").assertTextContains(t)
        }
        assertEquals("Network error: connection reset", texts.last())
    }

    @Test
    fun clearKeyAsksForConfirmation() {
        var cleared = 0
        compose.setContent { HablaTheme { SettingsContent(base, null, 0, SettingsActions(clearKey = { cleared++ })) } }
        compose.onNodeWithTag("key_clear").performClick()
        assertEquals(0, cleared)
        compose.onNodeWithTag("key_clear_confirm").assertIsDisplayed().performClick()
        assertEquals(1, cleared)
    }

    @Test
    fun rendersDownloadProgressAndFailure() {
        val state = mutableStateOf(base.copy(modelState = ModelState.Downloading(50_000_000, 100_000_000), downloadRunning = true))
        compose.setContent { HablaTheme { SettingsContent(state.value, null, 0, SettingsActions()) } }
        compose.onNodeWithTag("settings_list").performScrollToNode(hasTestTag("section_model"))
        compose.onNodeWithTag("model_status").assertTextContains("Downloading 50 / 100 MB")
        compose.onNodeWithTag("model_progress").assertIsDisplayed()
        compose.onNodeWithTag("model_cancel").assertIsDisplayed()

        state.value = base.copy(modelState = ModelState.Failed("checksum_mismatch", retryable = false))
        compose.onNodeWithTag("model_status").assertTextContains("Download failed: checksum mismatch")
        compose.onNodeWithTag("model_download").assertIsDisplayed()
    }
}
