package com.xrmech.habla.data

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import kotlinx.coroutines.flow.Flow

/** User preferences (data-models section 4). */
interface SettingsRepository {
    /** Current settings with defaults filled in; re-emits on every distinct change. */
    val settings: Flow<AppSettings>

    /** Atomically reads, applies [transform] (pure, no I/O) and writes all thirteen keys. */
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
    /** Separate model for live partials (D045, e.g. base); "" = the speech model itself. */
    val liveModelId: String = "",
    /** Experimental GPU speech engine (D043/D047); applies from the next app start. */
    val asrGpu: Boolean = false,
    /** Who translates finals (D050/D051). AUTO = Pro when active, else the OpenAI key, else on-device. */
    val finalEngine: EngineChoice = EngineChoice.AUTO,
    /** Previews on the phone whenever the language pack is ready, even with a key (free and faster). */
    val onDevicePreviews: Boolean = true,
)

/** The user's translation engine choice; `:app` resolves AUTO per call. PRO is llm-gateway (D051). */
enum class EngineChoice { AUTO, CLOUD, ON_DEVICE, PRO }
