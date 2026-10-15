package com.xrmech.habla.translate

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RedactTest {

    @Test
    fun scrubsKeysOfSeveralShapes() {
        assertEquals("key sk-*** here", Redact.secrets("key sk-abcdefgh1234 here"))
        assertEquals("ends with sk-***", Redact.secrets("ends with sk-proj-AbC_dEf-123456789"))
        assertEquals("sk-***", Redact.secrets("sk-svcacct-0123456789abcdef"))
        assertEquals("Incorrect API key provided: sk-***...wxyz.", Redact.secrets("Incorrect API key provided: sk-proj-abc...wxyz."))
        assertEquals("a sk-*** b sk-***", Redact.secrets("a sk-11111111 b sk-22222222"))
    }

    @Test
    fun leavesShortAndKeylessTextAlone() {
        assertEquals("sk-ab", Redact.secrets("sk-ab"))
        assertEquals("sk-1234567", Redact.secrets("sk-1234567"))
        assertEquals("¿Dónde está la biblioteca?", Redact.secrets("¿Dónde está la biblioteca?"))
    }

    @Test
    fun nullAndEmpty() {
        assertEquals("", Redact.secrets(null))
        assertEquals("", Redact.secrets(""))
    }

    @Test
    fun helperRoutesToSinkWithScrubbing() {
        val lines = mutableListOf<Pair<TranslateLog.Level, String>>()
        val log = Log(TranslateLog { level, message -> lines += level to message })
        log.warn("OpenAI 401 invalid_api_key: Incorrect API key provided: sk-live1234567890.")
        log.debug("debug line")
        log.info(null)
        log.error("")
        assertEquals(2, lines.size)
        assertEquals(TranslateLog.Level.WARN, lines[0].first)
        assertTrue("sk-live" !in lines[0].second)
        assertTrue("sk-***" in lines[0].second)
        assertEquals(TranslateLog.Level.DEBUG to "debug line", lines[1])
    }

    @Test
    fun helperFollowsTheProcessSink() {
        val lines = mutableListOf<String>()
        val previous = TranslateLogging.sink
        try {
            TranslateLogging.sink = TranslateLog { _, m -> lines += m }
            Log().info("key=sk-abcdefghijk")
            TranslateLogging.sink = TranslateLog.NONE
            Log().info("dropped")
        } finally {
            TranslateLogging.sink = previous
        }
        assertEquals(listOf("key=sk-***"), lines)
    }

    @Test
    fun snippetTruncatesAt60Chars() {
        val long = "a".repeat(100)
        assertEquals(60, Log.snippet(long).length)
        assertEquals("short", Log.snippet("short"))
        assertEquals("b".repeat(60), Log.snippet("b".repeat(60)))
        assertEquals("", Log.snippet(null))
    }
}
