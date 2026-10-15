package com.xrmech.habla.app.service

import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.app.pipeline.CardTranslation
import com.xrmech.habla.app.pipeline.ExchangeCard
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.SessionError
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.translate.model.Direction
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NotificationTextTest {
    private fun cards(n: Int) = List(n) { ExchangeCard(it + 1, null, 0, "t", "es", Direction.ES_TO_EN, CardTranslation.Pending) }

    @Test
    fun `should show calibrating while starting and calibrating`() {
        assertEquals("Calibrating...", NotificationText.of(LiveState(phase = SessionPhase.Starting), 0))
        assertEquals("Calibrating...", NotificationText.of(LiveState(phase = SessionPhase.Calibrating), 0))
    }

    @Test
    fun `should show exchanges and elapsed time while listening`() {
        val s = LiveState(phase = SessionPhase.Listening, startedAtMs = 1_000, exchanges = cards(12))
        assertEquals("Listening - 12 exchanges - 4:32", NotificationText.of(s, 1_000 + 272_000))
        assertEquals("Listening - 1 exchange - 0:05", NotificationText.of(s.copy(exchanges = cards(1)), 6_000))
        assertEquals("75:00", NotificationText.elapsed(4_500_000))
    }

    @Test
    fun `should show silenced and finishing`() {
        val silenced = LiveState(phase = SessionPhase.Listening, startedAtMs = 0, audio = AudioCondition.Silenced)
        assertEquals("Mic silenced by another app", NotificationText.of(silenced, 0))
        assertEquals("Finishing 2 segments...", NotificationText.of(LiveState(phase = SessionPhase.Finishing(2)), 0))
    }

    @Test
    fun `should have no text when idle or failed`() {
        assertNull(NotificationText.of(LiveState(), 0))
        assertNull(NotificationText.of(LiveState(phase = SessionPhase.Failed(SessionError.Asr("x"))), 0))
    }

    @Test
    fun `should drop updates inside 2 s and repeats of the same text`() {
        val t = NotificationThrottle()
        assertTrue(t.offer("a", 0))
        assertFalse(t.offer("b", 1_999))
        assertTrue(t.offer("b", 2_000))
        assertFalse(t.offer("b", 9_000))
        assertTrue(t.offer("c", 9_000))
    }
}
