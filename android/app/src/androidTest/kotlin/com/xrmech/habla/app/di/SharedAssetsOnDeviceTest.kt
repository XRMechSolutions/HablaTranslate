package com.xrmech.habla.app.di

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.app.testing.TestFixtures
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.IdiomScanner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The APK's shared assets through the production [AndroidAssetSource] on ART, whose regex engine
 * is ICU: a pattern that only the JVM accepts fails here (it once threw ExceptionInInitializerError).
 */
@RunWith(AndroidJUnit4::class)
class SharedAssetsOnDeviceTest {
    @Test
    fun loadsEverySharedFileFromTheApk() {
        val assets = SharedAssets(AndroidAssetSource(TestFixtures.context.assets))
        assertTrue(assets.asrModels.models.isNotEmpty())
        assertTrue(assets.models.models.containsKey(assets.models.finalDefault))
        assertTrue(assets.prompts.toString().isNotEmpty())
    }

    @Test
    fun compilesEveryShippedIdiomPatternOnArt() {
        val assets = SharedAssets(AndroidAssetSource(TestFixtures.context.assets))
        assertEquals("idiom entries skipped on ART: ${assets.skippedIdioms}", emptyList<String>(), assets.skippedIdioms)
        assertTrue(assets.idioms.isNotEmpty())
        val scanner = IdiomScanner(assets.idioms)
        assertEquals("patterns the scanner could not compile: ${scanner.skipped}", emptyList<String>(), scanner.skipped)
        // Every pattern compiles through the production Python->Java translation on ART.
        assets.idioms.forEach { def -> com.xrmech.habla.shared.IdiomRegex.compile(def.pattern) }
    }
}
