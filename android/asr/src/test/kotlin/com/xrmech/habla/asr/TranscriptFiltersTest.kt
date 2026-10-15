package com.xrmech.habla.asr

import com.xrmech.habla.asr.internal.PyText
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class TranscriptFiltersTest {
    private val profile = TestAssets.profile

    private fun bad(s: String) = TranscriptFilters.isBadTranscript(s, profile)
    private fun clipped(s: String) = TranscriptFilters.hasClippedOnset(s, profile)

    @Test
    fun `interfaces examples`() {
        assertTrue(bad("123"))
        assertTrue(bad("ok"))
        // interfaces.md lists this as true, but the server splits on whitespace: first word "ola," != "ola".
        assertFalse(clipped("ola, ¿cómo estás?"))
        assertTrue(clipped("ola ¿cómo estás?"))
        assertFalse(clipped("Hola"))
        assertFalse(bad("Hola"))
    }

    @Test
    fun `empty and whitespace-only are bad`() {
        assertTrue(bad(""))
        assertTrue(bad("   "))
        assertTrue(bad("${Char(0x85)}${Char(0xA0)}${Char(0x3000)}"))
    }

    @Test
    fun `letter ratio uses Python numerics`() {
        assertTrue(bad("12 ab")) // 2 letters < 3
        assertFalse(bad("ab12c")) // 3 letters / 5 alnum = 0.6
        assertTrue(bad("½½½½ abc")) // ½ is No: 3 / 7 < 0.5
        assertTrue(bad("ⅫⅫⅫⅫ abc")) // Ⅻ is Nl
        assertFalse(bad("😀😀 abc")) // emoji are neither letters nor alnum
        assertTrue(bad("...!!!"))
    }

    @Test
    fun `clipped onset uses Python strip split and casefold`() {
        assertTrue(clipped("OLA amigo"))
        assertFalse(clipped("Ola, amigo")) // first word is "ola," as on the server
        assertTrue(clipped("  ueno pues"))
        assertTrue(clipped("Ué pasa"))
        assertFalse(clipped("qué pasa"))
        assertTrue(clipped("ﬆoy aquí")) // ligature folds to "stoy"
        assertTrue(clipped("stoy\taquí"))
        assertTrue(clipped("\u0085ola\u0085"))
        assertFalse(clipped(""))
    }

    @Test
    fun `PyText basics`() {
        assertEquals("hola", PyText.strip("${Char(0x85)} hola${Char(0xA0)}"))
        assertEquals(listOf("a", "b"), PyText.split("${Char(0x2028)}a${Char(0x3000)}${Char(0x3000)}b "))
        assertEquals("strasse", PyText.casefold("STRAẞE"))
        assertEquals("σοφοσ", PyText.casefold("σοφος"))
        assertEquals(1, PyText.length("😀"))
    }
}
