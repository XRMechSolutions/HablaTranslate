package com.xrmech.habla.asr

/**
 * Speech-to-text over 16 kHz mono float audio (contracts.md @ :asr).
 *
 * `samples` are 16 kHz mono PCM in [-1, 1] (`:audio` `ShortArray.toFloatSamples()`), any length.
 * Values are neither scanned nor clamped; NaN samples are a caller contract violation.
 *
 * Implementations serialize calls internally; FINAL work preempts PARTIAL work.
 */
interface SpeechRecognizer : AutoCloseable {
    /**
     * Picks the most probable language among [candidates] (non-empty, in preference order; the first
     * wins ties) and returns its raw probability over all Whisper languages. [pass] PARTIAL runs on the
     * partial model at partial priority (a pending final preempts it with ABORTED), for partials before
     * the first language vote (D049); FINAL is the finals' detection.
     */
    suspend fun detectLanguage(samples: FloatArray, candidates: List<String>, pass: AsrPass = AsrPass.FINAL): LanguageGuess

    /** Transcribes [samples] in [AsrRequest.language] with the decoder settings of [AsrRequest.pass]. */
    suspend fun transcribe(samples: FloatArray, request: AsrRequest): AsrResult
}

/** One transcription request. [language] is a Whisper language code and never `"auto"`. */
data class AsrRequest(val language: String, val pass: AsrPass)

/** FINAL: beam search with the profile's `whisper_cpp.final` settings. PARTIAL: the profile's ladder level. */
enum class AsrPass { FINAL, PARTIAL }

/** Result of a restricted language detection; [probability] is the raw softmax over all Whisper languages. */
data class LanguageGuess(val language: String, val probability: Float)

/** Recognizer failure with a machine-readable [kind]. */
class AsrException(message: String, val kind: Kind, cause: Throwable? = null) : Exception(message, cause) {
    enum class Kind { MODEL_MISSING, MODEL_LOAD_FAILED, ABORTED, NATIVE_ERROR }
}

/** Transcription result (data-models.md section 1). */
data class AsrResult(
    /** Segment texts Python-stripped, joined with one space, stripped; `""` when there are none. */
    val text: String,
    /** `null` from [SpeechRecognizer.transcribe]; `AsrPipeline` fills it from the detection that chose the language. */
    val detectedLanguage: String?,
    /** Raw softmax over all Whisper languages; `null` as for [detectedLanguage]. */
    val languageProb: Float?,
    val segments: List<AsrSegment>,
    /** Native wall time of `whisper_full` (excludes waiting for the recognizer). */
    val elapsedMs: Long,
)

/** One decoded segment. */
data class AsrSegment(
    /** Whisper t0 centiseconds / 100.0. */
    val startSec: Double,
    val endSec: Double,
    /** UTF-8 decoded with split characters removed; NOT trimmed. */
    val text: String,
    /** Mean token log-probability over text tokens; `null` when the segment has none. */
    val avgLogprob: Float?,
    val noSpeechProb: Float?,
)
