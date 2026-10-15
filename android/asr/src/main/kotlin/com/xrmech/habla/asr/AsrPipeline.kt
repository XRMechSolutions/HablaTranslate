package com.xrmech.habla.asr

import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.LanguageRegistry
import kotlin.coroutines.cancellation.CancellationException

/** Server `reject_reason` values: `"ok"`, `"no_segments"`, `"bad_transcript"`. */
enum class RejectReason { OK, NO_SEGMENTS, BAD_TRANSCRIPT }

data class FinalAsrOutcome(
    /** `""` unless [reason] is [RejectReason.OK]. */
    val transcript: String,
    val reason: RejectReason,
    /** Language the final was transcribed in. */
    val language: String,
    /** Raw per-segment detection; null when auto_language is off or detection failed. */
    val detectedLanguage: String?,
    val languageProb: Float?,
    /** [TranscriptFilters.hasClippedOnset] of the transcript; false unless OK. */
    val clippedOnset: Boolean,
    /** Full result including segments and elapsedMs. */
    val result: AsrResult,
)

data class PartialAsrOutcome(
    /** `""` for a bad transcript or any [AsrException]. */
    val text: String,
    /** Language forced for this partial; null on [AsrException]. */
    val language: String?,
    /** True when the [AsrException] kind was ABORTED (preempted by a final). */
    val aborted: Boolean,
    /** 0 when no transcription ran. */
    val elapsedMs: Long,
)

/**
 * Port of the server's ASR control flow (design.md section 8): `_run_asr_and_diarize` minus
 * diarization for finals, `_run_quick_asr` for partials. Candidates for detection are
 * `languages.codes` in `languages.json` order.
 *
 * Not thread-safe because [LanguageVoter] is not: call from the coordinator's serial dispatcher.
 */
class AsrPipeline(
    private val recognizer: SpeechRecognizer,
    private val voter: LanguageVoter,
    private val profile: AsrProfile,
    languages: LanguageRegistry,
) {
    private val candidates: List<String> = languages.codes

    /**
     * Finals are transcribed in their own detected language (D026); the detection also votes, which
     * only steers partials and the direction fallback. Errors from transcribe propagate; a failed
     * detection falls back to `voter.current ?: sourceLanguage` without voting, like the server's
     * "no supported language detected" branch.
     */
    suspend fun final(samples: FloatArray, sourceLanguage: String): FinalAsrOutcome {
        var guess: LanguageGuess? = null
        val language = if (profile.whisper.autoLanguage) {
            try {
                recognizer.detectLanguage(samples, candidates).also {
                    guess = it
                    voter.vote(it.language)
                }.language
            } catch (e: AsrException) {
                if (e.kind != AsrException.Kind.NATIVE_ERROR) throw e
                voter.current ?: sourceLanguage
            }
        } else {
            sourceLanguage
        }

        val raw = recognizer.transcribe(samples, AsrRequest(language, AsrPass.FINAL))
        val result = raw.copy(detectedLanguage = guess?.language, languageProb = guess?.probability)

        val reason = when {
            result.segments.isEmpty() -> RejectReason.NO_SEGMENTS
            TranscriptFilters.isBadTranscript(result.text, profile) -> RejectReason.BAD_TRANSCRIPT
            else -> RejectReason.OK
        }
        val ok = reason == RejectReason.OK
        return FinalAsrOutcome(
            transcript = if (ok) result.text else "",
            reason = reason,
            language = language,
            detectedLanguage = result.detectedLanguage,
            languageProb = result.languageProb,
            clippedOnset = ok && TranscriptFilters.hasClippedOnset(result.text, profile),
            result = result,
        )
    }

    /**
     * Partials use the voted language, or [sourceLanguage] before the first vote (D035). They never
     * detect, never vote and never throw [AsrException]; cancellation is rethrown.
     */
    suspend fun partial(samples: FloatArray, sourceLanguage: String): PartialAsrOutcome {
        val language = voter.current ?: sourceLanguage
        val result = try {
            recognizer.transcribe(samples, AsrRequest(language, AsrPass.PARTIAL))
        } catch (e: CancellationException) {
            throw e
        } catch (e: AsrException) {
            return PartialAsrOutcome("", null, e.kind == AsrException.Kind.ABORTED, 0)
        }
        val text = if (TranscriptFilters.isBadTranscript(result.text, profile)) "" else result.text
        return PartialAsrOutcome(text, language, aborted = false, elapsedMs = result.elapsedMs)
    }
}
