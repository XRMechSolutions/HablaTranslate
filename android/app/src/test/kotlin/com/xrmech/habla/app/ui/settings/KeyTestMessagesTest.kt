package com.xrmech.habla.app.ui.settings

import com.xrmech.habla.translate.KeyTestResult
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.TokenUsage
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class KeyTestMessagesTest {
    private fun failed(kind: Kind, status: Int? = null, msg: String = "boom") =
        KeyTestMessages.text(KeyTestResult.Failed(LlmException(msg, kind, status)), "gpt-quick")

    @Test
    fun `should report a working key with the replying model`() {
        val r = KeyTestResult.Valid(TokenUsage(5, 0, 1, 0, "gpt-quick-2026"))
        assertEquals("Key works (gpt-quick-2026 replied)", KeyTestMessages.text(r, "gpt-quick"))
    }

    @Test
    fun `should give the text for every failure kind`() {
        assertEquals("Enter a key first", failed(Kind.NO_KEY))
        assertEquals("Key rejected by OpenAI (401)", failed(Kind.INVALID_KEY, 401))
        assertEquals("No credit on this OpenAI account (429) - add billing at platform.openai.com", failed(Kind.QUOTA, 429))
        assertEquals("Model gpt-quick not enabled for this key (404)", failed(Kind.MODEL_UNAVAILABLE, 404))
        assertEquals("Rate-limited - the key works, try again shortly", failed(Kind.RATE_LIMITED, 429))
        assertEquals("Network error: no route", failed(Kind.NETWORK, msg = "no route"))
        assertEquals("Network error: timed out", failed(Kind.TIMEOUT, msg = "timed out"))
        assertEquals("SERVER: boom", failed(Kind.SERVER, 500))
        assertEquals("BAD_REQUEST: boom", failed(Kind.BAD_REQUEST, 400))
        assertEquals("MALFORMED: boom", failed(Kind.MALFORMED))
    }
}
