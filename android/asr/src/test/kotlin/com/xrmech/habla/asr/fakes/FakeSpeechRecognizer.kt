package com.xrmech.habla.asr.fakes

import com.xrmech.habla.asr.AsrPass
import com.xrmech.habla.asr.AsrRequest
import com.xrmech.habla.asr.AsrResult
import com.xrmech.habla.asr.AsrSegment
import com.xrmech.habla.asr.LanguageGuess
import com.xrmech.habla.asr.SpeechRecognizer

/** Scripted recognizer: each call pops the next scripted answer (or throwable) and records the request. */
class FakeSpeechRecognizer : SpeechRecognizer {
    val detectCalls = mutableListOf<List<String>>()
    val detectPasses = mutableListOf<AsrPass>()
    val transcribeCalls = mutableListOf<AsrRequest>()

    private val detections = ArrayDeque<() -> LanguageGuess>()
    private val transcriptions = ArrayDeque<suspend () -> AsrResult>()

    fun detects(language: String, probability: Float = 0.9f) = apply { detections += { LanguageGuess(language, probability) } }
    fun detectFails(t: Throwable) = apply { detections += { throw t } }
    fun transcribes(text: String, segments: Int = 1, elapsedMs: Long = 100) = apply {
        transcriptions += { result(text, segments, elapsedMs) }
    }
    fun transcribeFails(t: Throwable) = apply { transcriptions += { throw t } }
    fun transcribeRuns(block: suspend () -> AsrResult) = apply { transcriptions += block }

    override suspend fun detectLanguage(samples: FloatArray, candidates: List<String>, pass: AsrPass): LanguageGuess {
        detectCalls += candidates
        detectPasses += pass
        return checkNotNull(detections.removeFirstOrNull()) { "unexpected detectLanguage call" }()
    }

    override suspend fun transcribe(samples: FloatArray, request: AsrRequest): AsrResult {
        transcribeCalls += request
        return checkNotNull(transcriptions.removeFirstOrNull()) { "unexpected transcribe call" }()
    }

    override fun close() = Unit

    companion object {
        fun result(text: String, segments: Int = 1, elapsedMs: Long = 100) = AsrResult(
            text = text,
            detectedLanguage = null,
            languageProb = null,
            segments = List(segments) { AsrSegment(it.toDouble(), it + 1.0, " $text", -0.2f, 0.01f) },
            elapsedMs = elapsedMs,
        )
    }
}
