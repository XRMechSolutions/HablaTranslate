package com.xrmech.habla.data

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.OutputStream
import java.util.zip.ZipInputStream
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.int
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assertions.fail
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

/**
 * Export tests. Structure parity is checked against `src/test/resources/server-export-fixture/`,
 * hand-built from the server's own dict literals:
 * - `metadata.json`: `habla/server/services/audio_recorder.py` `AudioRecorder.__init__` (session keys),
 *   `save_pcm_segment` (segment base keys), `stop_recording` (`ended_at`, `total_segments`) and the
 *   `add_segment_metadata` calls in `habla/server/routes/websocket.py` (accepted: `raw_transcript`,
 *   `translation`, `speaker`, `confidence`, `idioms_detected`; rejected: `asr_status`) plus the
 *   `audio_rms`/`vad_avg_prob` metadata passed to `save_pcm_segment`. `detected_language` is the one
 *   documented phone addition (data design.md section 8).
 * - `ground_truth.json`: `habla/tools/generate_ground_truth.py` `gt_segments.append` and the
 *   `ground_truth = {...}` literal.
 * - `corrected_ground_truth.json`: `habla/server/routes/api_corrections.py` `CorrectionSegment` and
 *   the `corrected_doc = {...}` literal in `save_corrected_ground_truth`.
 * Key sets and JSON kinds are compared, not values.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class TrainingExporterTest {
    @TempDir
    lateinit var filesDir: File

    private val t = TestDb()
    private val clock = MutableClock()

    @AfterEach
    fun close() = t.close()

    private class Env(val history: SqlHistoryRepository, val training: TrainingAudioStore, val exporter: TrainingExporter)

    private fun TestScope.env(): Env {
        val d = UnconfinedTestDispatcher(testScheduler)
        val history = SqlHistoryRepository(t.db, filesDir, d, clock)
        val training = TrainingAudioStore(filesDir, t.db, d, clock) { Long.MAX_VALUE }
        return Env(history, training, TrainingExporter(filesDir, history, training, d, clock))
    }

    private fun pcm(n: Int) = ShortArray(n) { (it % 200 - 100).toShort() }

    private fun capture(sid: Long, seg: Int, status: AsrStatus, samples: Int = 8000, logprob: Double? = -0.3124) =
        SegmentCapture(sid, seg, pcm(samples), 812.44, 0.71234, if (status == AsrStatus.ACCEPTED) "es" else null, status, 1.8449, logprob)

    private fun exchange(sid: Long, seg: Int, raw: String, translation: String, direction: Direction = Direction.ES_TO_EN) =
        NewExchange(
            sessionId = sid, segmentId = seg, direction = direction, rawTranscript = raw, correctedSource = null,
            translation = translation, confidence = 0.92, isCorrection = false, correction = null,
            flagged = listOf(FlaggedPhrase(phrase = "qué onda", meaning = "what's up")), processingMs = 900,
            detectedLanguage = "es", translationError = false, audioPath = "sessions/$sid/segment_00$seg.wav",
        )

    /** A session with an accepted segment (exchange) and a rejected one. Returns (sessionId, exchangeId). */
    private suspend fun Env.twoSegmentSession(direction: Direction = Direction.AUTO): Pair<Long, Long> {
        val sid = history.startSession(Mode.CONVERSATION, direction, "ggml-small-q8_0")
        clock.advanceSeconds(7)
        training.saveSegment(capture(sid, 1, AsrStatus.ACCEPTED, samples = 48_000))
        val exId = history.addExchange(exchange(sid, 1, "¿Qué onda?", "What's up?"))
        clock.advanceSeconds(3)
        training.saveSegment(capture(sid, 2, AsrStatus.REJECTED, samples = 8000, logprob = null))
        return sid to exId
    }

    private fun unzip(bytes: ByteArray): Map<String, ByteArray> {
        val out = linkedMapOf<String, ByteArray>()
        ZipInputStream(ByteArrayInputStream(bytes)).use { zip ->
            while (true) {
                val e = zip.nextEntry ?: break
                out[e.name] = zip.readBytes()
            }
        }
        return out
    }

    private fun json(bytes: ByteArray): JsonObject = Json.parseToJsonElement(bytes.toString(Charsets.UTF_8)).jsonObject

    private fun fixture(name: String): JsonObject =
        json(javaClass.classLoader!!.getResourceAsStream("server-export-fixture/$name")!!.readBytes())

    private fun kind(e: JsonElement): String = when (e) {
        is JsonNull -> "null"
        is JsonObject -> "object"
        is JsonArray -> "array"
        is JsonPrimitive -> when {
            e.isString -> "string"
            e.content == "true" || e.content == "false" -> "boolean"
            else -> "number"
        }
    }

    // Fields the phone may write as null where the server fixture holds a value.
    private val nullable = setOf("speaker", "detected_language", "confidence")

    /** Same key set and same JSON kind at every level; `segments` is compared per variant by the caller. */
    private fun assertSameShape(expected: JsonElement, actual: JsonElement, path: String) {
        if (expected is JsonObject) {
            assertTrue(actual is JsonObject, "$path: expected object, got ${kind(actual)}")
            actual as JsonObject
            assertEquals(expected.keys, actual.keys, "$path: key set")
            for (k in expected.keys) {
                if (k == "segments") continue
                val a = actual.getValue(k)
                if (a is JsonNull && k in nullable) continue
                assertSameShape(expected.getValue(k), a, "$path.$k")
            }
        } else {
            assertEquals(kind(expected), kind(actual), "$path: JSON kind")
        }
    }

    private fun assertParity(fixtureName: String, actual: JsonObject) {
        val expected = fixture(fixtureName)
        assertSameShape(expected, actual, fixtureName)
        val expectedSegments = expected.getValue("segments").jsonArray
        val actualSegments = actual.getValue("segments").jsonArray
        assertEquals(2, actualSegments.size)
        // Variant order in both: [accepted with exchange, rejected].
        for (i in 0..1) assertSameShape(expectedSegments[i], actualSegments[i], "$fixtureName.segments[$i]")
    }

    @Test
    fun `zip has one root, the wavs byte for byte, and server-shaped json`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession()
        val out = ByteArrayOutputStream()
        val summary = env.exporter.export(sid, out)
        assertEquals(ExportSummary(segmentsWritten = 2, segmentsMissing = 0, correctedIncluded = false), summary)

        val entries = unzip(out.toByteArray())
        val roots = entries.keys.map { it.substringBefore('/') }.toSet()
        assertEquals(1, roots.size)
        assertTrue(Regex("^android-\\d+_\\d{8}_\\d{6}$").matches(roots.single()), roots.single())
        val root = roots.single()
        assertEquals("android-${sid}_20261015_141502", root)
        assertEquals(
            setOf("$root/segment_001.wav", "$root/segment_002.wav", "$root/metadata.json", "$root/ground_truth.json"),
            entries.keys,
        )
        for (name in listOf("segment_001.wav", "segment_002.wav")) {
            assertArrayEquals(File(filesDir, "sessions/$sid/$name").readBytes(), entries.getValue("$root/$name"))
        }
        val meta = json(entries.getValue("$root/metadata.json"))
        val gt = json(entries.getValue("$root/ground_truth.json"))
        assertParity("metadata.json", meta)
        assertParity("ground_truth.json", gt)
    }

    @Test
    fun `field derivations`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession()
        clock.advanceSeconds(3600)
        val out = ByteArrayOutputStream()
        env.exporter.export(sid, out)
        val entries = unzip(out.toByteArray())
        val root = entries.keys.first().substringBefore('/')
        val meta = json(entries.getValue("$root/metadata.json"))
        assertEquals("android-$sid", meta.getValue("session_id").jsonPrimitive.content)
        assertEquals("none", meta.getValue("raw_audio_format").jsonPrimitive.content)
        assertEquals(16000, meta.getValue("sample_rate").jsonPrimitive.int)
        assertEquals("2026-10-15T14:15:02.120Z", meta.getValue("started_at").jsonPrimitive.content)
        // Session still open: ended_at falls back to the last recorded_at.
        assertEquals("2026-10-15T14:15:12.120Z", meta.getValue("ended_at").jsonPrimitive.content)
        assertEquals(2, meta.getValue("total_segments").jsonPrimitive.int)
        val accepted = meta.getValue("segments").jsonArray[0].jsonObject
        assertEquals("812.4", accepted.getValue("audio_rms").jsonPrimitive.content)
        assertEquals("0.712", accepted.getValue("vad_avg_prob").jsonPrimitive.content)
        assertEquals(96000, accepted.getValue("size_bytes").jsonPrimitive.int)
        assertEquals("3.0", accepted.getValue("duration_seconds").jsonPrimitive.content)
        assertEquals(JsonNull, accepted.getValue("speaker"))
        assertEquals(1, accepted.getValue("idioms_detected").jsonPrimitive.int)
        assertEquals("rejected", meta.getValue("segments").jsonArray[1].jsonObject.getValue("asr_status").jsonPrimitive.content)

        val gt = json(entries.getValue("$root/ground_truth.json"))
        assertEquals("ggml-small-q8_0", gt.getValue("whisper_model").jsonPrimitive.content)
        assertEquals("es_to_en", gt.getValue("direction").jsonPrimitive.content)
        assertEquals("2026-10-15T15:15:12.120Z", gt.getValue("generated_at").jsonPrimitive.content)
        val segs = gt.getValue("segments").jsonArray.map { it.jsonObject }
        assertEquals("¿Qué onda?", segs[0].getValue("transcript").jsonPrimitive.content)
        assertEquals("-0.312", segs[0].getValue("confidence").jsonPrimitive.content)
        assertEquals("1.84", segs[0].getValue("asr_time_seconds").jsonPrimitive.content)
        assertEquals("What's up?", segs[0].getValue("translation").jsonPrimitive.content)
        assertEquals("", segs[0].getValue("asr_corrections").jsonPrimitive.content)
        assertEquals("[inaudible]", segs[1].getValue("transcript").jsonPrimitive.content)
        assertEquals("0", segs[1].getValue("confidence").jsonPrimitive.content)
        assertEquals("", segs[1].getValue("translation").jsonPrimitive.content)
        assertEquals("[1] ¿Qué onda?\n[2] [inaudible]", gt.getValue("full_transcript").jsonPrimitive.content)
        assertEquals("What's up?\n", gt.getValue("full_translation").jsonPrimitive.content)
    }

    @Test
    fun `json is 2-space pretty printed utf-8 with non-ASCII unescaped`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession()
        val out = ByteArrayOutputStream()
        env.exporter.export(sid, out)
        val entries = unzip(out.toByteArray())
        val text = entries.entries.first { it.key.endsWith("/ground_truth.json") }.value.toString(Charsets.UTF_8)
        assertTrue(text.contains("¿Qué onda?"))
        assertFalse(text.contains("\\u00"))
        assertTrue(text.startsWith("{\n  \"generated_at\""), text.take(40))
    }

    @Test
    fun `corrected file appears only with a real edit, with stats`() = runTest {
        val env = env()
        val (sid, exId) = env.twoSegmentSession()
        env.history.setCorrectedTranscript(exId, "¿Qué onda?")
        val same = ByteArrayOutputStream()
        assertFalse(env.exporter.export(sid, same).correctedIncluded)
        assertTrue(unzip(same.toByteArray()).keys.none { it.endsWith("corrected_ground_truth.json") })

        env.history.setCorrectedTranscript(exId, "¿Qué onda, güey?")
        val out = ByteArrayOutputStream()
        assertTrue(env.exporter.export(sid, out).correctedIncluded)
        val entries = unzip(out.toByteArray())
        val corrected = json(entries.entries.first { it.key.endsWith("/corrected_ground_truth.json") }.value)
        assertParity("corrected_ground_truth.json", corrected)
        val segs = corrected.getValue("segments").jsonArray.map { it.jsonObject }
        assertEquals("¿Qué onda, güey?", segs[0].getValue("transcript").jsonPrimitive.content)
        assertEquals("true", segs[0].getValue("human_corrected").jsonPrimitive.content)
        assertEquals("true", segs[0].getValue("reviewed").jsonPrimitive.content)
        assertEquals("false", segs[1].getValue("human_corrected").jsonPrimitive.content)
        assertEquals("[inaudible]", segs[1].getValue("transcript").jsonPrimitive.content)
        val stats = corrected.getValue("correction_stats").jsonObject
        assertEquals(1, stats.getValue("reviewed").jsonPrimitive.int)
        assertEquals(1, stats.getValue("human_corrected").jsonPrimitive.int)
        assertEquals(2, stats.getValue("total").jsonPrimitive.int)
        // The original ground truth keeps the raw transcript.
        val gt = json(entries.entries.first { it.key.endsWith("/ground_truth.json") }.value)
        assertEquals("¿Qué onda?", gt.getValue("segments").jsonArray[0].jsonObject.getValue("transcript").jsonPrimitive.content)
    }

    @Test
    fun `a missing wav is skipped, counted and absent from the json`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession()
        File(filesDir, "sessions/$sid/segment_002.wav").delete()
        val out = ByteArrayOutputStream()
        assertEquals(ExportSummary(1, 1, false), env.exporter.export(sid, out))
        val entries = unzip(out.toByteArray())
        assertTrue(entries.keys.none { it.endsWith("segment_002.wav") })
        val meta = json(entries.entries.first { it.key.endsWith("/metadata.json") }.value)
        assertEquals(listOf("segment_001.wav"), meta.getValue("segments").jsonArray.map { it.jsonObject.getValue("filename").jsonPrimitive.content })
        assertEquals(1, meta.getValue("total_segments").jsonPrimitive.int)
        val gt = json(entries.entries.first { it.key.endsWith("/ground_truth.json") }.value)
        assertEquals(1, gt.getValue("segments").jsonArray.size)
    }

    @Test
    fun `unknown session and empty session throw the right kind`() = runTest {
        val env = env()
        val unknown = assertThrows<ExportException> { env.exporter.export(404, ByteArrayOutputStream()) }
        assertEquals(ExportException.Kind.SESSION_NOT_FOUND, unknown.kind)
        val sid = env.history.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        val out = ByteArrayOutputStream()
        val empty = assertThrows<ExportException> { env.exporter.export(sid, out) }
        assertEquals(ExportException.Kind.NO_SEGMENTS, empty.kind)
        assertEquals(0, out.size())
    }

    @Test
    fun `every wav missing is NO_SEGMENTS before any json`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession()
        File(filesDir, "sessions/$sid").deleteRecursively()
        val out = ByteArrayOutputStream()
        val e = assertThrows<ExportException> { env.exporter.export(sid, out) }
        assertEquals(ExportException.Kind.NO_SEGMENTS, e.kind)
        assertFalse(out.toByteArray().toString(Charsets.ISO_8859_1).contains("metadata.json"))
    }

    @Test
    fun `the caller's stream is never closed`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession()
        val out = object : ByteArrayOutputStream() {
            override fun close() = fail<Unit>("export must not close the caller's stream")
        }
        env.exporter.export(sid, out)
        assertTrue(out.size() > 0)
    }

    @Test
    fun `write failure on out is ExportException IO`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession()
        val out = object : OutputStream() {
            override fun write(b: Int) = throw IOException("disk full")
            override fun write(b: ByteArray, off: Int, len: Int) = throw IOException("disk full")
        }
        val e = assertThrows<ExportException> { env.exporter.export(sid, out) }
        assertEquals(ExportException.Kind.IO, e.kind)
    }

    @Test
    fun `auto direction resolves to the most frequent exchange direction`() {
        assertEquals(Direction.EN_TO_ES, ExportJson.resolveDirection(Direction.AUTO, listOf(Direction.EN_TO_ES, Direction.EN_TO_ES, Direction.ES_TO_EN)))
        assertEquals(Direction.ES_TO_EN, ExportJson.resolveDirection(Direction.AUTO, listOf(Direction.EN_TO_ES, Direction.ES_TO_EN)))
        assertEquals(Direction.ES_TO_EN, ExportJson.resolveDirection(Direction.AUTO, emptyList()))
        assertEquals(Direction.EN_TO_ES, ExportJson.resolveDirection(Direction.EN_TO_ES, listOf(Direction.ES_TO_EN)))
    }

    @Test
    fun `session direction en_to_es is written as is`() = runTest {
        val env = env()
        val (sid, _) = env.twoSegmentSession(Direction.EN_TO_ES)
        val out = ByteArrayOutputStream()
        env.exporter.export(sid, out)
        val gt = json(unzip(out.toByteArray()).entries.first { it.key.endsWith("/ground_truth.json") }.value)
        assertEquals("en_to_es", gt.getValue("direction").jsonPrimitive.content)
    }

    @Test
    fun `python rounding is half-even on the exact binary value`() {
        assertEquals(2.67, ExportJson.round(2.675, 2))
        assertEquals(0.712, ExportJson.round(0.71234, 3))
        assertEquals(812.4, ExportJson.round(812.44, 1))
        assertEquals(0.5, ExportJson.round(0.5, 1))
    }

    @Test
    fun `suggestedFileName uses the UTC start`() {
        val session = SessionRow(
            id = 12, startedAt = java.time.Instant.parse("2026-10-15T14:15:02.120Z"), endedAt = null,
            mode = Mode.CONVERSATION, direction = Direction.AUTO, topicSummary = null, notes = null,
            asrModel = "m", translateModel = null, exchangeCount = 0, costUsd = 0.0,
        )
        assertEquals("android-12_20261015_141502.zip", TrainingExporter.suggestedFileName(session))
    }
}
