package com.xrmech.habla.app.fakes

import com.xrmech.habla.shared.FileAssetSource
import com.xrmech.habla.shared.SharedAssets

/** The repo's real shared/ assets (system property habla.sharedDir, set by the Gradle test task). */
object TestAssets {
    val shared: SharedAssets by lazy { SharedAssets(FileAssetSource.fromSystemProperty()) }
}
