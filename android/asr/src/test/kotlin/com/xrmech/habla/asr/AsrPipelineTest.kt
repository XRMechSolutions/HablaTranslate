package com.xrmech.habla.asr

import com.xrmech.habla.asr.fakes.FakeSpeechRecognizer
import com.xrmech.habla.shared.AsrProfile
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.yield
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class AsrPipelineTest {
    private val audio = FloatArray(16_000)
    private val rec = FakeSpeechRecognizer()

    private fun pipeline(profile: AsrProfile = TestAssets.profile, voter: LanguageVoter = LanguageVoter(profile)) =
        AsrPipeline(rec, voter, profile, TestAssets.languages) to voter

    @Test
    fun `final detects, votes and transcribes in the detected language`() = runTest {
        val (p, voter) = pipeline()
        rec.detects("en", 0.81f).transcribes("Where is the station?")
        val out = p.final(audio, sourceLanguage = "es")
        assertEquals(listOf(listOf("es", "en")), rec.detectCalls)
        assertEquals(AsrRequest("en", AsrPass.FINAL), rec.transcribeCalls.single())
        assertEquals("en", voter.current)
        assertEquals(RejectReason.OK, out.reason)
        assertEquals("Where is the station?", out.transcript)
        assertEquals("en", out.language)
        assertEquals("en", out.detectedLanguage)
        assertEquals(0.81f, out.languageProb)
        assertEquals("en", out.result.detectedLanguage)
        assertEquals(0.81f, out.result.languageProb)
        assertFalse(out.clippedOnset)
    }

    @Test
    fun `final is transcribed in its own detection even when the vote does not switch`() = runTest {
        val (p, voter) = pipeline()
        rec.detects("es").transcribes("Hola amigos").detects("en").transcribes("Hello there")
        p.final(audio, "es")
        val out = p.final(audio, "es")
        assertEquals("es", voter.current) // 1 of 2 votes: no switch
        assertEquals("en", out.language)
        assertEquals("en", rec.transcribeCalls[1].language)
    }

    @Test
    fun `detection NATIVE_ERROR falls back to voted then source language without voting`() = runTest {
        val (p, voter) = pipeline()
        rec.detectFails(AsrException("language detection failed rc=-6", AsrException.Kind.NATIVE_ERROR))
            .transcribes("Hola amigos")
        val first = p.final(audio, "es")
        assertEquals("es", first.language)
        assertNull(first.detectedLanguage)
        assertNull(first.languageProb)
        assertNull(voter.current)

        voter.vote("en")
        rec.detectFails(AsrException("x", AsrException.Kind.NATIVE_ERROR)).transcribes("Hello there")
        val second = p.final(audio, "es")
        assertEquals("en", second.language)
        assertEquals("en", voter.current)
    }

    @Test
    fun `other detection failures propagate`() = runTest {
        val (p, _) = pipeline()
        rec.detectFails(AsrException("closed", AsrException.Kind.ABORTED))
        val e = assertThrows<AsrException> { p.final(audio, "es") }
        assertEquals(AsrException.Kind.ABORTED, e.kind)
        assertTrue(rec.transcribeCalls.isEmpty())
    }

    @Test
    fun `transcribe failures in final propagate`() = runTest {
        val (p, _) = pipeline()
        rec.detects("es").transcribeFails(AsrException("whisper_full rc=-6", AsrException.Kind.NATIVE_ERROR))
        val e = assertThrows<AsrException> { p.final(audio, "es") }
        assertEquals(AsrException.Kind.NATIVE_ERROR, e.kind)
    }

    @Test
    fun `auto_language off uses source language with no detection and no vote`() = runTest {
        val (p, voter) = pipeline(TestAssets.profileWith(autoLanguage = false))
        rec.transcribes("Hola amigos")
        val out = p.final(audio, "es")
        assertTrue(rec.detectCalls.isEmpty())
        assertEquals("es", rec.transcribeCalls.single().language)
        assertNull(voter.current)
        assertNull(out.detectedLanguage)
        assertEquals(RejectReason.OK, out.reason)
    }

    @Test
    fun `reject reasons in server order`() = runTest {
        val (p, _) = pipeline(TestAssets.profileWith(autoLanguage = false))
        rec.transcribes("", segments = 0).transcribes("123", segments = 0).transcribes("123").transcribes("ola amigo, ¿cómo estás?")
        val none = p.final(audio, "es")
        assertEquals(RejectReason.NO_SEGMENTS, none.reason)
        assertEquals("", none.transcript)
        // NO_SEGMENTS wins over a bad text.
        assertEquals(RejectReason.NO_SEGMENTS, p.final(audio, "es").reason)
        val bad = p.final(audio, "es")
        assertEquals(RejectReason.BAD_TRANSCRIPT, bad.reason)
        assertEquals("", bad.transcript)
        assertFalse(bad.clippedOnset)
        val clipped = p.final(audio, "es")
        assertEquals(RejectReason.OK, clipped.reason)
        assertTrue(clipped.clippedOnset)
    }

    @Test
    fun `partial before the first vote uses the source language without detection`() = runTest {
        val (p, _) = pipeline()
        rec.transcribes("Hola amigos", elapsedMs = 321)
        val out = p.partial(audio, "es")
        assertTrue(rec.detectCalls.isEmpty())
        assertEquals(AsrRequest("es", AsrPass.PARTIAL), rec.transcribeCalls.single())
        assertEquals(PartialAsrOutcome("Hola amigos", "es", false, 321), out)
    }

    @Test
    fun `partial after a vote uses the voted language and never votes`() = runTest {
        val (p, voter) = pipeline()
        voter.vote("en")
        rec.transcribes("Hello there").transcribes("Hello again")
        p.partial(audio, "es")
        p.partial(audio, "es")
        assertEquals(listOf("en", "en"), rec.transcribeCalls.map { it.language })
        assertTrue(rec.detectCalls.isEmpty())
        assertEquals("en", voter.current)
    }

    @Test
    fun `partial errors become empty outcomes with the aborted flag`() = runTest {
        val (p, _) = pipeline()
        rec.transcribeFails(AsrException("preempted", AsrException.Kind.ABORTED))
            .transcribeFails(AsrException("rc=-6", AsrException.Kind.NATIVE_ERROR))
        assertEquals(PartialAsrOutcome("", null, true, 0), p.partial(audio, "es"))
        assertEquals(PartialAsrOutcome("", null, false, 0), p.partial(audio, "es"))
    }

    @Test
    fun `partial bad transcript becomes empty text but keeps the language`() = runTest {
        val (p, _) = pipeline()
        rec.transcribes("ok", elapsedMs = 50)
        assertEquals(PartialAsrOutcome("", "es", false, 50), p.partial(audio, "es"))
    }

    @Test
    fun `cancellation propagates from partial and final`() = runTest {
        val (p, _) = pipeline()
        rec.transcribeRuns { awaitCancellation() }
        val job = launch { p.partial(audio, "es") }
        yield()
        job.cancel()
        job.join()
        assertTrue(job.isCancelled)

        rec.transcribeFails(CancellationException("caller gone"))
        assertThrows<CancellationException> { p.partial(audio, "es") }
        rec.detects("es").transcribeFails(CancellationException("caller gone"))
        assertThrows<CancellationException> { p.final(audio, "es") }
    }
}
