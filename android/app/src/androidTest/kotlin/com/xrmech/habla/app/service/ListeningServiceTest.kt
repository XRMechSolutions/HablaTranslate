package com.xrmech.habla.app.service

import android.Manifest
import android.os.Build
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.xrmech.habla.app.MainActivity
import com.xrmech.habla.app.capture.WavFileAudioSource
import com.xrmech.habla.app.di.appModules
import com.xrmech.habla.app.pipeline.PipelineCoordinator
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.app.testing.InMemoryHistory
import com.xrmech.habla.app.testing.InMemoryKeyStore
import com.xrmech.habla.app.testing.InMemorySettings
import com.xrmech.habla.app.testing.InMemoryUsage
import com.xrmech.habla.app.testing.InMemoryVocab
import com.xrmech.habla.app.testing.TestFixtures
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.data.ApiKeyStore
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.data.VocabRepository
import java.io.BufferedInputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.rules.TestWatcher
import org.junit.runner.Description
import org.junit.runner.RunWith
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.GlobalContext
import org.koin.core.context.startKoin
import org.koin.core.context.stopKoin
import org.koin.dsl.module

/**
 * The real service and pipeline (tiny model, paced WAV in place of the mic): Listen in the
 * activity posts the notification, and its Stop action ends the session.
 */
@RunWith(AndroidJUnit4::class)
class ListeningServiceTest {
    private val history = InMemoryHistory()

    private val koin = object : TestWatcher() {
        override fun starting(description: Description) {
            TestFixtures.installTinyModel()
            stopKoin()
            val settings = InMemorySettings(
                AppSettingsDefaults(TestFixtures.shared).value.copy(
                    asrModelId = TestFixtures.tinyEntry.id, firstRunDone = true, trainingCapture = false,
                ),
            )
            startKoin {
                androidContext(TestFixtures.context)
                allowOverride(true)
                modules(
                    appModules + module {
                        single<SettingsRepository> { settings }
                        single<HistoryRepository> { history }
                        single<VocabRepository> { InMemoryVocab() }
                        single<UsageRepository> { InMemoryUsage() }
                        single<ApiKeyStore> { InMemoryKeyStore() }
                        single { ModelManager(TestFixtures.context.noBackupFilesDir, TestFixtures.tinyCatalog, get<ModelDownloader>()) }
                        single {
                            ServiceAudioSourceFactory {
                                WavFileAudioSource(BufferedInputStream(TestFixtures.clip("mixed_es_en_es.wav")), paced = true)
                            }
                        }
                    },
                )
            }
        }

        override fun finished(description: Description) {
            GlobalContext.getOrNull()?.get<PipelineCoordinator>()?.abort()
            stopKoin()
        }
    }

    private val permissions: GrantPermissionRule = if (Build.VERSION.SDK_INT >= 33) {
        GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO, Manifest.permission.POST_NOTIFICATIONS)
    } else {
        GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO)
    }
    private val compose = createAndroidComposeRule<MainActivity>()

    @get:Rule
    val rules: RuleChain = RuleChain.outerRule(permissions).around(koin).around(compose)

    @Test
    fun listenPostsTheNotificationAndItsStopActionEndsTheSession() {
        val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        val coordinator = GlobalContext.get().get<PipelineCoordinator>()

        compose.waitUntil(15_000) { compose.onAllNodes(androidx.compose.ui.test.hasTestTag("listen_bottom")).fetchSemanticsNodes().isNotEmpty() }
        // The ViewModel refreshes the model state on start; wait until Listen would actually start.
        compose.waitUntil(15_000) { GlobalContext.get().get<ModelManager>().state.value is com.xrmech.habla.asr.ModelState.Ready }
        compose.onNodeWithTag("listen_bottom").performClick()

        compose.waitUntil(30_000) { coordinator.state.value.phase.let { it == SessionPhase.Listening || it == SessionPhase.Calibrating } }
        val sessionId = coordinator.state.value.sessionId
        assertNotNull(sessionId)

        // The notification is posted by the foreground service with a "Stop" action.
        val nm = TestFixtures.context.getSystemService(android.app.NotificationManager::class.java)
        compose.waitUntil(15_000) { nm.activeNotifications.any { it.id == ListeningNotification.NOTIFICATION_ID } }
        val posted = nm.activeNotifications.first { it.id == ListeningNotification.NOTIFICATION_ID }.notification
        val stopAction = posted.actions?.firstOrNull { it.title == "Stop" }
        assertNotNull("listening notification has no Stop action", stopAction)
        // It is visible in the shade (UiAutomator), then its Stop action is fired as the shade would.
        device.openNotification()
        assertNotNull("notification not shown in the shade", device.wait(Until.findObject(By.textContains("Habla")), 10_000))
        val stopButton = device.findObject(By.text("Stop")) ?: device.findObject(By.text("STOP"))
        if (stopButton != null) stopButton.click() else stopAction!!.actionIntent.send()

        compose.waitUntil(60_000) { coordinator.state.value.phase == SessionPhase.Idle }
        device.pressBack()
        assertEquals(SessionPhase.Idle, coordinator.state.value.phase)
        assertTrue("session row ended", history.sessionRows.value.first { it.id == sessionId }.endedAt != null)
    }
}
