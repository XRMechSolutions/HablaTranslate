package com.xrmech.habla.app.capture

import com.xrmech.habla.app.capture.MicReopenPolicy.OpenStage
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class MicReopenPolicyTest {
    @Test
    fun `should back off 250 to 4000 ms then give up`() {
        assertEquals(listOf(250L, 500L, 1_000L, 2_000L, 4_000L), (1..5).map { MicReopenPolicy.delayBefore(it) })
        assertNull(MicReopenPolicy.delayBefore(6))
        assertEquals(5, MicReopenPolicy.MAX_ATTEMPTS)
    }

    @Test
    fun `should map open stages to error kinds`() {
        assertEquals(AudioSourceException.Kind.PERMISSION, MicReopenPolicy.kindFor(OpenStage.PERMISSION_DENIED))
        assertEquals(AudioSourceException.Kind.INIT_FAILED, MicReopenPolicy.kindFor(OpenStage.MIN_BUFFER))
        assertEquals(AudioSourceException.Kind.INIT_FAILED, MicReopenPolicy.kindFor(OpenStage.NOT_INITIALIZED))
        assertEquals(AudioSourceException.Kind.START_FAILED, MicReopenPolicy.kindFor(OpenStage.START))
    }

    @Test
    fun `should reopen on every negative read code and name dead object`() {
        listOf(-1, -2, -3, -6).forEach { assertTrue(MicReopenPolicy.needsReopen(it)) }
        assertFalse(MicReopenPolicy.needsReopen(0))
        assertFalse(MicReopenPolicy.needsReopen(512))
        assertEquals("ERROR_DEAD_OBJECT", MicReopenPolicy.describe(-6))
        assertEquals("error -42", MicReopenPolicy.describe(-42))
    }

    @Test
    fun `should size the buffer to at least one second and twice the minimum`() {
        assertEquals(32_000, MicReopenPolicy.bufferBytes(1_280))
        assertEquals(40_000, MicReopenPolicy.bufferBytes(20_000))
    }
}
