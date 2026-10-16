package com.xrmech.habla.data

import androidx.datastore.preferences.core.MutablePreferences
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode

/**
 * Enum <-> stored string. An explicit exhaustive `when` (not Json round-tripping), so renaming a
 * Kotlin constant fails to compile instead of silently changing what is stored.
 */
internal object EnumCodec {
    fun encode(direction: Direction): String = when (direction) {
        Direction.ES_TO_EN -> "es_to_en"
        Direction.EN_TO_ES -> "en_to_es"
        Direction.AUTO -> "auto"
    }

    fun decodeDirection(value: String): Direction? = when (value) {
        "es_to_en" -> Direction.ES_TO_EN
        "en_to_es" -> Direction.EN_TO_ES
        "auto" -> Direction.AUTO
        else -> null
    }

    fun encode(mode: Mode): String = when (mode) {
        Mode.CONVERSATION -> "conversation"
        Mode.CLASSROOM -> "classroom"
    }

    fun decodeMode(value: String): Mode? = when (value) {
        "conversation" -> Mode.CONVERSATION
        "classroom" -> Mode.CLASSROOM
        else -> null
    }
}

/** Preferences <-> [AppSettings] over the eleven data-models keys. */
internal object SettingsCodec {
    val DIRECTION = stringPreferencesKey("direction")
    val MODE = stringPreferencesKey("mode")
    val PARTIALS_ENABLED = booleanPreferencesKey("partials_enabled")
    val PARTIAL_TRANSLATION_ENABLED = booleanPreferencesKey("partial_translation_enabled")
    val FINAL_MODEL = stringPreferencesKey("final_model")
    val QUICK_MODEL = stringPreferencesKey("quick_model")
    val ASR_MODEL_ID = stringPreferencesKey("asr_model_id")
    val TRAINING_CAPTURE = booleanPreferencesKey("training_capture")
    val FIRST_RUN_DONE = booleanPreferencesKey("first_run_done")
    val LIVE_MODEL_ID = stringPreferencesKey("live_model_id")
    val ASR_GPU = booleanPreferencesKey("asr_gpu")

    /** Missing keys take [defaults]; an unknown enum string takes that field's default with a warning. */
    fun decode(p: Preferences, defaults: AppSettings): AppSettings = AppSettings(
        direction = p[DIRECTION]?.let { raw ->
            EnumCodec.decodeDirection(raw) ?: defaults.direction.also { DataLog.w("unknown stored direction; using default") }
        } ?: defaults.direction,
        mode = p[MODE]?.let { raw ->
            EnumCodec.decodeMode(raw) ?: defaults.mode.also { DataLog.w("unknown stored mode; using default") }
        } ?: defaults.mode,
        partialsEnabled = p[PARTIALS_ENABLED] ?: defaults.partialsEnabled,
        partialTranslation = p[PARTIAL_TRANSLATION_ENABLED] ?: defaults.partialTranslation,
        finalModel = p[FINAL_MODEL] ?: defaults.finalModel,
        quickModel = p[QUICK_MODEL] ?: defaults.quickModel,
        asrModelId = p[ASR_MODEL_ID] ?: defaults.asrModelId,
        trainingCapture = p[TRAINING_CAPTURE] ?: defaults.trainingCapture,
        firstRunDone = p[FIRST_RUN_DONE] ?: defaults.firstRunDone,
        liveModelId = p[LIVE_MODEL_ID] ?: defaults.liveModelId,
        asrGpu = p[ASR_GPU] ?: defaults.asrGpu,
    )

    /** Writes all eleven keys, so the file is always complete. */
    fun encode(s: AppSettings, p: MutablePreferences) {
        p[DIRECTION] = EnumCodec.encode(s.direction)
        p[MODE] = EnumCodec.encode(s.mode)
        p[PARTIALS_ENABLED] = s.partialsEnabled
        p[PARTIAL_TRANSLATION_ENABLED] = s.partialTranslation
        p[FINAL_MODEL] = s.finalModel
        p[QUICK_MODEL] = s.quickModel
        p[ASR_MODEL_ID] = s.asrModelId
        p[TRAINING_CAPTURE] = s.trainingCapture
        p[FIRST_RUN_DONE] = s.firstRunDone
        p[LIVE_MODEL_ID] = s.liveModelId
        p[ASR_GPU] = s.asrGpu
    }
}
