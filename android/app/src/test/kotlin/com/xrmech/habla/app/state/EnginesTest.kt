package com.xrmech.habla.app.state

import com.xrmech.habla.data.EngineChoice
import com.xrmech.habla.translate.TranslateEngine.CLOUD
import com.xrmech.habla.translate.TranslateEngine.GATEWAY
import com.xrmech.habla.translate.TranslateEngine.ON_DEVICE
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/** D050 engine resolution. */
class EnginesTest {
    @Test
    fun `auto uses the key when one is saved and the phone otherwise`() {
        assertEquals(CLOUD to CLOUD, Engines.resolve(EngineChoice.AUTO, onDevicePreviews = true, hasKey = true, packReady = false))
        assertEquals(CLOUD to ON_DEVICE, Engines.resolve(EngineChoice.AUTO, onDevicePreviews = true, hasKey = true, packReady = true))
        assertEquals(ON_DEVICE to ON_DEVICE, Engines.resolve(EngineChoice.AUTO, onDevicePreviews = false, hasKey = false, packReady = false))
    }

    @Test
    fun `explicit choices are kept whatever the key and pack`() {
        assertEquals(CLOUD to CLOUD, Engines.resolve(EngineChoice.CLOUD, onDevicePreviews = true, hasKey = false, packReady = false))
        assertEquals(ON_DEVICE to ON_DEVICE, Engines.resolve(EngineChoice.ON_DEVICE, onDevicePreviews = false, hasKey = true, packReady = true))
    }

    @Test
    fun `previews stay in the cloud when the user turns phone previews off`() {
        assertEquals(CLOUD to CLOUD, Engines.resolve(EngineChoice.CLOUD, onDevicePreviews = false, hasKey = true, packReady = true))
    }

    @Test
    fun `pro wins in auto when active and previews never use the gateway`() {
        assertEquals(GATEWAY to ON_DEVICE, Engines.resolve(EngineChoice.AUTO, onDevicePreviews = false, hasKey = true, packReady = false, proActive = true))
        assertEquals(CLOUD to CLOUD, Engines.resolve(EngineChoice.AUTO, onDevicePreviews = true, hasKey = true, packReady = false, proActive = false))
        assertEquals(GATEWAY to ON_DEVICE, Engines.resolve(EngineChoice.PRO, onDevicePreviews = true, hasKey = false, packReady = true, proActive = false))
    }
}
