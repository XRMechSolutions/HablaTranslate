package com.xrmech.habla.app.di

import android.content.res.AssetManager
import com.xrmech.habla.shared.AssetSource

/** `shared/<path>` from the APK assets (synced by :shared-assets), read as UTF-8. */
class AndroidAssetSource(private val assets: AssetManager) : AssetSource {
    override fun readText(path: String): String =
        assets.open("shared/$path").use { it.readBytes().toString(Charsets.UTF_8) }
}
