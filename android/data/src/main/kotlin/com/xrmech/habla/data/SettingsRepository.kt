package com.xrmech.habla.data

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import kotlinx.coroutines.flow.Flow

/** User preferences (data-models section 4). */
interface SettingsRepository {
    /** Current settings with defaults filled in; re-emits on every distinct change. */
    val settings: Flow<AppSettings>

    /** Atomically reads, applies [transform] (pure, no I/O) and writes all nine keys. */
    suspend fun update(transform: (AppSettings) -> AppSettings)
}

data class AppSettings(
    val direction: Direction,
    val mode: Mode,
    val partialsEnabled: Boolean,
    val partialTranslation: Boolean,
    val finalModel: String,
    val quickModel: String,
    val asrModelId: String,
    val trainingCapture: Boolean,
    val firstRunDone: Boolean,
)
