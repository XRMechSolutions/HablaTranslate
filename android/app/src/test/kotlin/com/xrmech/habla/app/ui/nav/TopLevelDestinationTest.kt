package com.xrmech.habla.app.ui.nav

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test

class TopLevelDestinationTest {
    @Test
    fun `should list four destinations with unique routes and labels`() {
        val all = TopLevelDestination.entries
        assertEquals(listOf("Live", "History", "Vocab", "Settings"), all.map { it.label })
        assertEquals(4, all.map { it.route::class }.toSet().size)
        assertEquals(4, all.map { it.icon }.toSet().size)
    }

    @Test
    fun `should mark history for session detail`() {
        assertEquals(TopLevelDestination.HISTORY, TopLevelDestination.forRoute(SessionDetail::class))
        assertEquals(TopLevelDestination.SETTINGS, TopLevelDestination.forRoute(Settings::class))
        assertNull(TopLevelDestination.forRoute(String::class))
    }
}
