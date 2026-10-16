package com.xrmech.habla.app.translate

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.xrmech.habla.translate.OnDeviceResult
import com.xrmech.habla.translate.model.Direction
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Real ML Kit on the phone (D050). Needs network the first time (the ~30 MB pack); later runs read
 * it from disk. Leaves the pack installed, as a user's download would.
 */
@RunWith(AndroidJUnit4::class)
class MlKitTranslatorTest {
    @Test
    fun downloadsThePackAndTranslatesBothWays(): Unit = runBlocking {
        val t = MlKitTranslator()
        t.refresh()
        if (t.state.value != PackState.Ready) withTimeout(DOWNLOAD_TIMEOUT_MS) { t.download() }
        assertEquals(PackState.Ready, t.state.value)

        val started = System.nanoTime()
        val en = t.translate("¿Dónde está la estación de tren?", Direction.ES_TO_EN)
        val firstMs = (System.nanoTime() - started) / 1_000_000
        val es = t.translate("Where is the train station?", Direction.EN_TO_ES)
        val auto = t.translate("Buenos días", Direction.AUTO)

        val enText = (en as OnDeviceResult.Ok).text.lowercase()
        val esText = (es as OnDeviceResult.Ok).text.lowercase()
        assertTrue(enText, "train station" in enText)
        assertTrue(esText, "estación" in esText)
        assertTrue((auto as OnDeviceResult.Ok).text, auto.text.lowercase().contains("good"))
        android.util.Log.i("MlKitTranslatorTest", "first es->en ${firstMs} ms: '$enText' / '$esText'")
    }

    @Test
    fun aWarmTranslationIsFastEnoughForLivePreviews(): Unit = runBlocking {
        val t = MlKitTranslator()
        t.refresh()
        if (t.state.value != PackState.Ready) withTimeout(DOWNLOAD_TIMEOUT_MS) { t.download() }
        t.translate("hola", Direction.ES_TO_EN)
        val times = (1..10).map {
            val s = System.nanoTime()
            t.translate("Me gustaría reservar una mesa para dos personas esta noche", Direction.ES_TO_EN)
            (System.nanoTime() - s) / 1_000_000
        }.sorted()
        android.util.Log.i("MlKitTranslatorTest", "warm es->en ms: $times")
        assertTrue("median ${times[5]} ms", times[5] < 500)
    }

    private companion object {
        const val DOWNLOAD_TIMEOUT_MS = 180_000L
    }
}
