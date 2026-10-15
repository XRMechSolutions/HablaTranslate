package com.xrmech.habla.app.ui.live

import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.MetricsSnapshot
import com.xrmech.habla.app.pipeline.SessionError
import com.xrmech.habla.app.pipeline.SessionNotice
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.translate.LlmException
import java.io.File
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class BannerReducerTest {
    private val quiet = LiveUiInputs(
        live = LiveState(),
        modelState = ModelState.Ready(File("m.bin")),
        modelSizeBytes = 264_464_607,
        keyState = KeyState.PRESENT,
        online = true,
        micPermission = MicPermission.GRANTED,
        finalModel = "gpt-5.4-mini",
    )

    private fun single(inputs: LiveUiInputs): Banner = BannerReducer.reduce(inputs).single()

    @Test
    fun `should show nothing when all is well`() {
        assertTrue(BannerReducer.reduce(quiet).isEmpty())
    }

    @Test
    fun `should show the session failure verbatim`() {
        val b = single(quiet.copy(live = LiveState(phase = SessionPhase.Failed(SessionError.ModelLoad("Speech model missing - download it in Settings")))))
        assertEquals(BannerKind.SESSION_FAILED, b.kind)
        assertEquals("Speech model missing - download it in Settings - tap Listen to retry", b.text)
    }

    @Test
    fun `should offer grant then open settings for the microphone`() {
        assertEquals(BannerAction.GRANT_MIC, single(quiet.copy(micPermission = MicPermission.DENIED)).action)
        val b = single(quiet.copy(micPermission = MicPermission.PERMANENTLY_DENIED))
        assertEquals("Microphone permission needed", b.text)
        assertEquals(BannerAction.OPEN_APP_SETTINGS, b.action)
        assertTrue(BannerReducer.reduce(quiet.copy(micPermission = MicPermission.NOT_ASKED)).isEmpty())
    }

    @Test
    fun `should show model missing failed and downloading`() {
        val missing = single(quiet.copy(modelState = ModelState.Missing))
        assertEquals("Speech model not downloaded (264 MB) - needed to transcribe", missing.text)
        assertEquals(BannerAction.DOWNLOAD_MODEL, missing.action)
        val failed = single(quiet.copy(modelState = ModelState.Failed("checksum_mismatch", false)))
        assertEquals("Model download failed: checksum mismatch", failed.text)
        assertEquals(BannerAction.DELETE_AND_RETRY, failed.action)
        assertEquals(BannerAction.RETRY_DOWNLOAD, single(quiet.copy(modelState = ModelState.Failed("network", true))).action)
        val dl = single(quiet.copy(modelState = ModelState.Downloading(120_000_000, 264_464_607)))
        assertEquals(BannerKind.MODEL_DOWNLOADING, dl.kind)
        assertEquals("Downloading speech model 120/264 MB", dl.text)
        assertEquals(BannerAction.CANCEL_DOWNLOAD, dl.action)
    }

    @Test
    fun `should show key states`() {
        val invalid = single(
            quiet.copy(
                keyState = KeyState.INVALID,
                live = LiveState(lastTranslationError = LlmException("bad key", LlmException.Kind.INVALID_KEY, 401)),
            ),
        )
        assertEquals("OpenAI rejected the key (401) - check or replace it", invalid.text)
        assertEquals("OpenAI account has no credit - add billing at platform.openai.com", single(quiet.copy(keyState = KeyState.NO_QUOTA)).text)
        val noKey = single(quiet.copy(keyState = KeyState.ABSENT))
        assertEquals("Translation off - add your OpenAI key", noKey.text)
        assertEquals(BannerAction.ADD_KEY, noKey.action)
    }

    @Test
    fun `should name the model when it is unavailable`() {
        val b = single(quiet.copy(live = LiveState(lastTranslationError = LlmException("nf", LlmException.Kind.MODEL_UNAVAILABLE, 404))))
        assertEquals("Model gpt-5.4-mini not available for this key (404)", b.text)
    }

    @Test
    fun `should show translate and asr errors verbatim`() {
        val t = single(quiet.copy(live = LiveState(lastTranslationError = LlmException("upstream 500", LlmException.Kind.SERVER, 500))))
        assertEquals("Translation failed: SERVER - upstream 500", t.text)
        val a = single(quiet.copy(live = LiveState(lastAsrError = "whisper rc -3")))
        assertEquals(BannerKind.ASR_ERROR, a.kind)
        assertEquals("Transcription failed: whisper rc -3", a.text)
    }

    @Test
    fun `should clear the asr banner once the next accepted final cleared the error`() {
        assertTrue(BannerReducer.reduce(quiet.copy(live = LiveState(lastAsrError = null))).isEmpty())
    }

    @Test
    fun `should show offline and mic silenced while listening`() {
        assertEquals(BannerKind.OFFLINE, single(quiet.copy(online = false)).kind)
        val silenced = quiet.copy(live = LiveState(phase = SessionPhase.Listening, audio = AudioCondition.Silenced))
        assertEquals("Microphone in use by another app or muted - paused", single(silenced).text)
        assertTrue(BannerReducer.reduce(quiet.copy(live = LiveState(audio = AudioCondition.Silenced))).isEmpty())
    }

    @Test
    fun `should show every session notice with its numbers`() {
        val live = LiveState(
            notices = SessionNotice.entries.toSet(),
            metrics = MetricsSnapshot(finalsDropped = 2, framesDropped = 31),
            lastHistoryError = "disk full",
            noiseFloorRms = 912.4,
        )
        val texts = BannerReducer.reduce(quiet.copy(live = live)).associate { it.kind to it.text }
        assertEquals("Phone falling behind - skipped 2 segments", texts[BannerKind.FALLING_BEHIND])
        assertEquals("Audio dropped (31 frames) - phone overloaded", texts[BannerKind.AUDIO_DROPPED])
        assertEquals("Couldn't save to history: disk full", texts[BannerKind.HISTORY_WRITE_FAILED])
        assertEquals("Voice detector failed - using energy detection", texts[BannerKind.VAD_FALLBACK])
        assertEquals("Phone is hot - live partials paused", texts[BannerKind.THERMAL_PAUSE])
        assertEquals(
            "Loud environment detected (RMS 912) - speech detection threshold raised automatically",
            texts[BannerKind.LOUD_ENVIRONMENT],
        )
    }

    @Test
    fun `should order by severity and cap at three with a more count`() {
        val inputs = quiet.copy(
            live = LiveState(phase = SessionPhase.Failed(SessionError.Asr("x")), notices = setOf(SessionNotice.THERMAL_PAUSE)),
            keyState = KeyState.ABSENT,
            online = false,
            modelState = ModelState.Missing,
        )
        val all = BannerReducer.reduce(inputs)
        assertEquals(
            listOf(BannerKind.SESSION_FAILED, BannerKind.MODEL_MISSING, BannerKind.OFFLINE, BannerKind.THERMAL_PAUSE, BannerKind.NO_KEY),
            all.map { it.kind },
        )
        val set = BannerReducer.cap(all)
        assertEquals(3, set.shown.size)
        assertEquals(2, set.more)
    }

    @Test
    fun `should not repeat a kind`() {
        val inputs = quiet.copy(
            keyState = KeyState.INVALID,
            live = LiveState(lastTranslationError = LlmException("bad", LlmException.Kind.INVALID_KEY, 403)),
        )
        val all = BannerReducer.reduce(inputs)
        assertEquals(1, all.size)
        assertNull(all.firstOrNull { it.kind == BannerKind.TRANSLATE_ERROR })
    }
}

class CardTextsTest {
    @Test
    fun `should give the failed line for every kind`() {
        val expected = mapOf(
            LlmException.Kind.NO_KEY to "Not translated - no OpenAI key",
            LlmException.Kind.INVALID_KEY to "Not translated - key rejected",
            LlmException.Kind.QUOTA to "Not translated - no credit",
            LlmException.Kind.NETWORK to "Not translated - network: m",
            LlmException.Kind.TIMEOUT to "Not translated - network: m",
            LlmException.Kind.RATE_LIMITED to "Not translated - RATE_LIMITED: m",
            LlmException.Kind.MODEL_UNAVAILABLE to "Not translated - MODEL_UNAVAILABLE: m",
            LlmException.Kind.BAD_REQUEST to "Not translated - BAD_REQUEST: m",
            LlmException.Kind.SERVER to "Not translated - SERVER: m",
            LlmException.Kind.MALFORMED to "Not translated - MALFORMED: m",
        )
        assertEquals(LlmException.Kind.entries.toSet(), expected.keys)
        expected.forEach { (kind, text) -> assertEquals(text, CardTexts.failedLine(kind, "m")) }
        assertEquals(BannerAction.ADD_KEY, CardTexts.failedAction(LlmException.Kind.NO_KEY))
        assertEquals(BannerAction.OPEN_SETTINGS, CardTexts.failedAction(LlmException.Kind.QUOTA))
        assertNull(CardTexts.failedAction(LlmException.Kind.SERVER))
    }
}
