package com.xrmech.habla.translate

import com.xrmech.habla.shared.FileAssetSource
import com.xrmech.habla.shared.SharedAssets

/** The repo's real `shared/` assets, loaded once per test JVM. */
object TestAssets {
    val assets: SharedAssets by lazy { SharedAssets(FileAssetSource.fromSystemProperty()) }
}
