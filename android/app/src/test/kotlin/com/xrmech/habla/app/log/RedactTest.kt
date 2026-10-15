package com.xrmech.habla.app.log

import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

class RedactTest {
    private data class Line(val priority: Int, val tag: String, val msg: String, val t: Throwable?)

    private val lines = mutableListOf<Line>()
    private val previousSink = HLog.sink
    private val previousText = HLog.textEnabled

    @BeforeEach
    fun setUp() {
        HLog.sink = LogSink { p, tag, msg, t -> lines += Line(p, tag, msg, t) }
    }

    @AfterEach
    fun tearDown() {
        HLog.sink = previousSink
        HLog.textEnabled = previousText
    }

    @Test
    fun `should redact a project key`() {
        assertEquals("key sk-***", HLog.redactSecrets("key sk-proj-abcdEFGH1234"))
    }

    @Test
    fun `should redact service account admin and plain keys`() {
        assertEquals("sk-***", HLog.redactSecrets("sk-svcacct-abcd_EFGH-1234"))
        assertEquals("sk-***", HLog.redactSecrets("sk-admin-abcdEFGH1234"))
        assertEquals("sk-***", HLog.redactSecrets("sk-abcdEFGH1234"))
    }

    @Test
    fun `should keep a short sk token`() {
        assertEquals("sk-short", HLog.redactSecrets("sk-short"))
    }

    @Test
    fun `should redact both keys when a message holds two`() {
        assertEquals("a sk-*** b sk-***", HLog.redactSecrets("a sk-proj-11112222333 b sk-44445555666"))
    }

    @Test
    fun `should redact the message passed to the sink`() {
        HLog.i(HLog.KEY, "saved sk-proj-abcdEFGH1234")
        assertEquals("saved sk-***", lines.single().msg)
    }

    @Test
    fun `should redact an exception message when warning`() {
        HLog.w(HLog.TRANSLATE, "failed", IllegalStateException("bad key sk-proj-abcdEFGH1234", RuntimeException("again sk-abcdefghijk")))
        val t = lines.single().t!!
        assertFalse(t.toString().contains("abcdEFGH1234"))
        assertTrue(t.toString().contains("sk-***"))
        assertFalse(t.cause.toString().contains("abcdefghijk"))
    }

    @Test
    fun `should not evaluate the text block when text logging is off`() {
        HLog.textEnabled = false
        var evaluated = false
        HLog.text(HLog.PIPELINE) { evaluated = true; "transcript" }
        assertFalse(evaluated)
        assertTrue(lines.isEmpty())
    }

    @Test
    fun `should emit text at debug level when text logging is on`() {
        HLog.textEnabled = true
        HLog.text(HLog.PIPELINE) { "hola" }
        assertEquals(HLog.DEBUG, lines.single().priority)
    }
}
