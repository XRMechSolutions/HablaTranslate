package com.xrmech.habla.asr

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class LanguageVoterTest {
    private fun voter() = LanguageVoter(TestAssets.profile)

    private fun LanguageVoter.trace(vararg votes: String): List<String?> = votes.map { vote(it); current }

    @Test
    fun `interfaces example es en en en`() {
        // vote 2: [es,en] tie -> first-seen es; vote 3: en 2/3 < 0.7; vote 4: en 3/4 >= 0.7.
        assertEquals(listOf("es", "es", "es", "en"), voter().trace("es", "en", "en", "en"))
    }

    @Test
    fun `first detection is accepted immediately`() {
        val v = voter()
        assertNull(v.current)
        v.vote("en")
        assertEquals("en", v.current)
    }

    @Test
    fun `window of five - 3 of 5 does not switch, 4 of 5 does`() {
        val v = voter()
        v.trace("es", "es", "en", "en")
        v.vote("en") // window [es,es,en,en,en]: 3/5 = 0.6
        assertEquals("es", v.current)
        v.vote("en") // window [es,en,en,en,en]: 4/5 = 0.8
        assertEquals("en", v.current)
    }

    @Test
    fun `old votes roll out of the window`() {
        val v = voter()
        v.trace("es", "es", "es", "es", "es")
        assertEquals(listOf("es", "es", "es", "en"), v.trace("en", "en", "en", "en"))
    }

    @Test
    fun `reset clears votes and current`() {
        val v = voter()
        v.trace("es", "es", "es")
        v.reset()
        assertNull(v.current)
        v.vote("en")
        assertEquals("en", v.current)
        // Only one vote in the window after reset, so es needs a majority again.
        assertEquals(listOf("en", "en"), v.trace("es", "es"))
    }

    @Test
    fun `blank language is rejected`() {
        assertThrows<IllegalArgumentException> { voter().vote("") }
        assertThrows<IllegalArgumentException> { voter().vote("  ") }
    }
}
