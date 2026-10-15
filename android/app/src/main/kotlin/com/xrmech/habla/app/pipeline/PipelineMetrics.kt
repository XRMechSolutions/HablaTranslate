package com.xrmech.habla.app.pipeline

import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ceil
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Per-session counters (design section 9). Names mirror the server orchestrator's `_metrics` keys
 * plus phone-only fields. Counters are atomic (the capture thread bumps [framesDropped]); timing
 * lists are synchronized because the final worker and translation jobs append from different
 * coroutines.
 */
class PipelineMetrics {
    @Volatile var sessionId: Long? = null

    val segmentsProcessed = AtomicInteger()
    val translationsCompleted = AtomicInteger()
    val translationErrors = AtomicInteger()
    val peakQueueDepth = AtomicInteger()
    val lowConfidenceCount = AtomicInteger()
    val correctionsDetected = AtomicInteger()
    val idiomPatternDbHits = AtomicInteger()
    val idiomLlmHits = AtomicInteger()
    val asrRejectedCount = AtomicInteger()
    val asrEmptyCount = AtomicInteger()

    val framesDropped = AtomicInteger()
    val finalsDropped = AtomicInteger()
    val partialsShown = AtomicInteger()
    val partialsAborted = AtomicInteger()
    val clippedOnsetCount = AtomicInteger()
    val asrErrors = AtomicInteger()
    val thermalChanges = AtomicInteger()

    @Volatile var batteryStartPct: Int? = null
    @Volatile var batteryEndPct: Int? = null

    private val finalAsrMs = mutableListOf<Long>()
    private val translateMs = mutableListOf<Long>()
    private val speechEndToTranslationMs = mutableListOf<Long>()
    private val partialCadenceMs = mutableListOf<Long>()

    fun recordFinalAsr(ms: Long) = synchronized(finalAsrMs) { finalAsrMs += ms }
    fun recordTranslate(ms: Long) = synchronized(translateMs) { translateMs += ms }
    fun recordSpeechEndToTranslation(ms: Long) = synchronized(speechEndToTranslationMs) { speechEndToTranslationMs += ms }
    fun recordPartialCadence(ms: Long) = synchronized(partialCadenceMs) { partialCadenceMs += ms }

    fun notePeakQueueDepth(depth: Int) {
        peakQueueDepth.accumulateAndGet(depth) { a, b -> maxOf(a, b) }
    }

    fun snapshot(): MetricsSnapshot = MetricsSnapshot(
        sessionId = sessionId,
        segmentsProcessed = segmentsProcessed.get(),
        translationsCompleted = translationsCompleted.get(),
        translationErrors = translationErrors.get(),
        peakQueueDepth = peakQueueDepth.get(),
        lowConfidenceCount = lowConfidenceCount.get(),
        correctionsDetected = correctionsDetected.get(),
        idiomPatternDbHits = idiomPatternDbHits.get(),
        idiomLlmHits = idiomLlmHits.get(),
        asrRejectedCount = asrRejectedCount.get(),
        asrEmptyCount = asrEmptyCount.get(),
        framesDropped = framesDropped.get(),
        finalsDropped = finalsDropped.get(),
        partialsShown = partialsShown.get(),
        partialsAborted = partialsAborted.get(),
        clippedOnsetCount = clippedOnsetCount.get(),
        asrErrors = asrErrors.get(),
        thermalChanges = thermalChanges.get(),
        batteryStartPct = batteryStartPct,
        batteryEndPct = batteryEndPct,
        finalAsrMs = synchronized(finalAsrMs) { finalAsrMs.toList() },
        translateMs = synchronized(translateMs) { translateMs.toList() },
        speechEndToTranslationMs = synchronized(speechEndToTranslationMs) { speechEndToTranslationMs.toList() },
        partialCadenceMs = synchronized(partialCadenceMs) { partialCadenceMs.toList() },
    )

    fun toJsonLine(): String = snapshot().toJsonLine()
}

/** Immutable copy for LiveState and the Diagnostics row. */
data class MetricsSnapshot(
    val sessionId: Long? = null,
    val segmentsProcessed: Int = 0,
    val translationsCompleted: Int = 0,
    val translationErrors: Int = 0,
    val peakQueueDepth: Int = 0,
    val lowConfidenceCount: Int = 0,
    val correctionsDetected: Int = 0,
    val idiomPatternDbHits: Int = 0,
    val idiomLlmHits: Int = 0,
    val asrRejectedCount: Int = 0,
    val asrEmptyCount: Int = 0,
    val framesDropped: Int = 0,
    val finalsDropped: Int = 0,
    val partialsShown: Int = 0,
    val partialsAborted: Int = 0,
    val clippedOnsetCount: Int = 0,
    val asrErrors: Int = 0,
    val thermalChanges: Int = 0,
    val batteryStartPct: Int? = null,
    val batteryEndPct: Int? = null,
    val finalAsrMs: List<Long> = emptyList(),
    val translateMs: List<Long> = emptyList(),
    val speechEndToTranslationMs: List<Long> = emptyList(),
    val partialCadenceMs: List<Long> = emptyList(),
) {
    /**
     * One JSON line, server key names in snake_case. Timing lists appear as `<name>_p50` /
     * `<name>_p95` (nearest rank) and are omitted while empty; null battery values are omitted.
     */
    fun toJsonLine(): String = toJson().toString()

    fun toJson(): JsonObject = buildJsonObject {
        sessionId?.let { put("session", it) }
        put("segments_processed", segmentsProcessed)
        put("translations_completed", translationsCompleted)
        put("translation_errors", translationErrors)
        put("peak_queue_depth", peakQueueDepth)
        put("low_confidence_count", lowConfidenceCount)
        put("corrections_detected", correctionsDetected)
        put("idiom_pattern_db_hits", idiomPatternDbHits)
        put("idiom_llm_hits", idiomLlmHits)
        put("asr_rejected_count", asrRejectedCount)
        put("asr_empty_count", asrEmptyCount)
        put("frames_dropped", framesDropped)
        put("finals_dropped", finalsDropped)
        put("partials_shown", partialsShown)
        put("partials_aborted", partialsAborted)
        put("clipped_onset_count", clippedOnsetCount)
        put("asr_errors", asrErrors)
        put("thermal_changes", thermalChanges)
        batteryStartPct?.let { put("battery_start_pct", it) }
        batteryEndPct?.let { put("battery_end_pct", it) }
        for ((name, values) in listOf(
            "final_asr_ms" to finalAsrMs,
            "translate_ms" to translateMs,
            "speech_end_to_translation_ms" to speechEndToTranslationMs,
            "partial_cadence_ms" to partialCadenceMs,
        )) {
            if (values.isEmpty()) continue
            put("${name}_p50", percentile(values, 0.50))
            put("${name}_p95", percentile(values, 0.95))
        }
    }

    companion object {
        /** Nearest-rank percentile of a non-empty list. */
        fun percentile(values: List<Long>, q: Double): Long {
            val sorted = values.sorted()
            val rank = ceil(q * sorted.size).toInt().coerceIn(1, sorted.size)
            return sorted[rank - 1]
        }
    }
}
