package com.xrmech.habla.app.ui.nav

import kotlinx.serialization.Serializable

/** Type-safe navigation routes (interfaces.md @ ui); every screen is directly addressable. */
@Serializable
data object Live

@Serializable
data object History

@Serializable
data class SessionDetail(val sessionId: Long)

@Serializable
data object Vocab

/** [section] scrolls Settings to `key`, `translation`, `listening`, `model`, `training` or `cost`. */
@Serializable
data class Settings(val section: String? = null) {
    companion object {
        const val KEY = "key"
        const val TRANSLATION = "translation"
        const val LISTENING = "listening"
        const val MODEL = "model"
        const val TRAINING = "training"
        const val COST = "cost"
    }
}
