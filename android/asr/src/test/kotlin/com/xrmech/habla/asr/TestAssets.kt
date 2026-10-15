package com.xrmech.habla.asr

import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.FileAssetSource
import com.xrmech.habla.shared.LanguageRegistry
import com.xrmech.habla.shared.SharedAssets

/** The real `shared/` files (path from the `habla.sharedDir` system property set by Gradle). */
internal object TestAssets {
    val shared: SharedAssets by lazy { SharedAssets(FileAssetSource.fromSystemProperty()) }
    val profile: AsrProfile get() = shared.asrProfile
    val languages: LanguageRegistry get() = shared.languages

    fun profileWith(
        autoLanguage: Boolean = profile.whisper.autoLanguage,
        level: String = profile.whisperCpp.partial.level,
        finalBeam: Int? = profile.whisperCpp.final.beamSize,
    ): AsrProfile = profile.copy(
        whisper = profile.whisper.copy(autoLanguage = autoLanguage),
        whisperCpp = profile.whisperCpp.copy(
            final = profile.whisperCpp.final.copy(beamSize = finalBeam),
            partial = profile.whisperCpp.partial.copy(level = level),
        ),
    )
}
