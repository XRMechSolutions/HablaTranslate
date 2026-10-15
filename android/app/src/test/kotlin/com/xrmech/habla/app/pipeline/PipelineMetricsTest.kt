package com.xrmech.habla.app.pipeline

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class PipelineMetricsTest {
    @Test
    fun `should write server key names and session id`() {
        val m = PipelineMetrics()
        m.sessionId = 42
        m.segmentsProcessed.addAndGet(12)
        m.framesDropped.incrementAndGet()
        m.notePeakQueueDepth(3)
        m.notePeakQueueDepth(2)
        val json = Json.parseToJsonElement(m.toJsonLine()).jsonObject
        assertEquals(42L, json["session"]!!.jsonPrimitive.long)
        assertEquals(12L, json["segments_processed"]!!.jsonPrimitive.long)
        assertEquals(1L, json["frames_dropped"]!!.jsonPrimitive.long)
        assertEquals(3L, json["peak_queue_depth"]!!.jsonPrimitive.long)
        listOf(
            "translations_completed", "translation_errors", "low_confidence_count", "corrections_detected",
            "idiom_pattern_db_hits", "idiom_llm_hits", "asr_rejected_count", "asr_empty_count",
            "finals_dropped", "partials_shown", "partials_aborted", "clipped_onset_count", "asr_errors",
        ).forEach { assertTrue(it in json, "missing $it") }
    }

    @Test
    fun `should report nearest rank p50 and p95 for a timing list`() {
        val m = PipelineMetrics()
        (1L..20L).map { it * 100 }.shuffled(java.util.Random(7)).forEach(m::recordSpeechEndToTranslation)
        val json = Json.parseToJsonElement(m.toJsonLine()).jsonObject
        assertEquals(1_000L, json["speech_end_to_translation_ms_p50"]!!.jsonPrimitive.long)
        assertEquals(1_900L, json["speech_end_to_translation_ms_p95"]!!.jsonPrimitive.long)
    }

    @Test
    fun `should omit empty timing lists and null battery values`() {
        val json = Json.parseToJsonElement(PipelineMetrics().toJsonLine()).jsonObject
        assertFalse("final_asr_ms_p50" in json)
        assertFalse("partial_cadence_ms_p95" in json)
        assertFalse("battery_start_pct" in json)
        assertFalse("session" in json)
    }

    @Test
    fun `should give the single value for a one element list`() {
        assertEquals(7L, MetricsSnapshot.percentile(listOf(7L), 0.95))
        assertEquals(7L, MetricsSnapshot.percentile(listOf(7L), 0.5))
    }
}
