package com.xrmech.habla.app.testing

import com.xrmech.habla.app.di.appModules
import com.xrmech.habla.app.pipeline.LiveController
import com.xrmech.habla.app.service.SessionControl
import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.app.ui.settings.KeyTester
import com.xrmech.habla.data.ApiKeyStore
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.data.VocabRepository
import org.junit.rules.TestWatcher
import org.junit.runner.Description
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.startKoin
import org.koin.core.context.stopKoin
import org.koin.core.module.Module
import org.koin.dsl.module

/**
 * Restarts Koin with the production graph plus overrides of the UI's seams (LiveController,
 * SessionControl, repositories, key store, key tester), so Compose tests never touch the mic, the
 * network or the device database. [extra] adds per-test overrides.
 */
class KoinFakesRule(
    val live: FakeLiveController = FakeLiveController(),
    val session: FakeSessionControl = FakeSessionControl(),
    val history: InMemoryHistory = InMemoryHistory(),
    val vocab: InMemoryVocab = InMemoryVocab(),
    val usage: InMemoryUsage = InMemoryUsage(),
    val keyStore: InMemoryKeyStore = InMemoryKeyStore(),
    val keyTester: FakeKeyTester = FakeKeyTester(),
    firstRunDone: Boolean = true,
    private val extra: Module = module {},
) : TestWatcher() {
    val settings = InMemorySettings(AppSettingsDefaults(TestFixtures.shared).value.copy(firstRunDone = firstRunDone))

    override fun starting(description: Description) {
        stopKoin()
        val fakes = module {
            single<LiveController> { live }
            single<SessionControl> { session }
            single<HistoryRepository> { history }
            single<VocabRepository> { vocab }
            single<UsageRepository> { usage }
            single<ApiKeyStore> { keyStore }
            single<KeyTester> { keyTester }
            single<SettingsRepository> { settings }
        }
        startKoin {
            androidContext(TestFixtures.context)
            allowOverride(true)
            modules(appModules + fakes + extra)
        }
    }

    override fun finished(description: Description) {
        stopKoin()
    }
}
