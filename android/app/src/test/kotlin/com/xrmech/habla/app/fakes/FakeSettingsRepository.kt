package com.xrmech.habla.app.fakes

import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.SettingsRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.filterNotNull

/** In-memory settings; [loaded] false holds back the first emission like a slow DataStore. */
class FakeSettingsRepository(
    initial: AppSettings = AppSettingsDefaults(TestAssets.shared).value,
    loaded: Boolean = true,
) : SettingsRepository {
    private val state = MutableStateFlow(if (loaded) initial else null)
    private var pending = initial

    override val settings: Flow<AppSettings> = state.filterNotNull()

    val current: AppSettings get() = state.value ?: pending

    fun load() {
        state.value = pending
    }

    override suspend fun update(transform: (AppSettings) -> AppSettings) {
        val next = transform(current)
        pending = next
        state.value = next
    }
}
