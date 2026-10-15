package com.xrmech.habla.data

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class ApiKeyFormatTest {

    @Test
    fun `trims surrounding whitespace`() {
        assertEquals("fake-key-abc", ApiKeyFormat.normalize("  \tfake-key-abc \r\n"))
    }

    @Test
    fun `rejects blank`() {
        assertThrows<IllegalArgumentException> { ApiKeyFormat.normalize("") }
        assertThrows<IllegalArgumentException> { ApiKeyFormat.normalize("   \n") }
    }

    @Test
    fun `rejects inner whitespace and control characters`() {
        for (bad in listOf("fake key", "fake\tkey", "fake\rkey", "fake\nkey", "fake\u0000key")) {
            val e = assertThrows<IllegalArgumentException> { ApiKeyFormat.normalize(bad) }
            assertEquals("API key contains whitespace or control characters", e.message)
        }
    }

    @Test
    fun `rejects non-ASCII`() {
        assertThrows<IllegalArgumentException> { ApiKeyFormat.normalize("fake-kéy") }
        assertThrows<IllegalArgumentException> { ApiKeyFormat.normalize("fake key") }
    }

    @Test
    fun `length limit is 1024`() {
        assertEquals(1024, ApiKeyFormat.normalize("k".repeat(1024)).length)
        assertThrows<IllegalArgumentException> { ApiKeyFormat.normalize("k".repeat(1025)) }
    }

    @Test
    fun `accepts quotes`() {
        assertEquals("\"fake-key\"", ApiKeyFormat.normalize("\"fake-key\""))
        assertEquals("'fake-key'", ApiKeyFormat.normalize("'fake-key'"))
    }
}
