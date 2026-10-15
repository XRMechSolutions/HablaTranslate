package com.xrmech.habla.data

import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/**
 * Timestamp columns are fixed-width UTC TEXT (`uuuu-MM-dd'T'HH:mm:ss.SSS'Z'`), so lexicographic
 * order equals chronological order and Python's `datetime.fromisoformat` reads exports.
 */
internal object Timestamps {
    private val FORMAT: DateTimeFormatter =
        DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH:mm:ss.SSS'Z'").withZone(ZoneOffset.UTC)

    fun now(clock: Clock): String = format(clock.instant())

    fun format(instant: Instant): String = FORMAT.format(instant)

    fun parse(text: String): Instant = Instant.parse(text)
}
