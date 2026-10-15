package com.xrmech.habla.data

import java.time.Clock
import java.time.Instant
import java.time.ZoneId
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class TimestampsTest {

    @Test
    fun `fixed clock formats with millis and Z`() {
        val clock = Clock.fixed(Instant.parse("2026-10-15T14:15:02.120Z"), ZoneId.of("America/Chicago"))
        assertEquals("2026-10-15T14:15:02.120Z", Timestamps.now(clock))
    }

    @Test
    fun `whole seconds still carry three millis digits`() {
        val clock = Clock.fixed(Instant.parse("2026-01-02T03:04:05Z"), ZoneId.of("UTC"))
        assertEquals("2026-01-02T03:04:05.000Z", Timestamps.now(clock))
    }

    @Test
    fun `round trips through Instant parse`() {
        val instant = Instant.parse("2026-10-15T14:15:02.987Z")
        assertEquals(instant, Timestamps.parse(Timestamps.format(instant)))
    }

    @Test
    fun `lexicographic order equals chronological order`() {
        val earlier = Instant.parse("2026-09-30T23:59:59.999Z")
        val later = Instant.parse("2026-10-01T00:00:00.000Z")
        assertTrue(Timestamps.format(earlier) < Timestamps.format(later))
        val a = Instant.parse("2026-10-15T09:00:00.005Z")
        val b = Instant.parse("2026-10-15T10:00:00.004Z")
        assertTrue(Timestamps.format(a) < Timestamps.format(b))
    }
}
