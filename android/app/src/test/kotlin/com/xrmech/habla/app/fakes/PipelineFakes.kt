package com.xrmech.habla.app.fakes

import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.app.capture.AudioSource
import com.xrmech.habla.app.pipeline.RecognizerHolder
import com.xrmech.habla.app.pipeline.SegmenterPort
import com.xrmech.habla.app.pipeline.StreamProgressListener
import com.xrmech.habla.app.pipeline.TrainingCapture
import com.xrmech.habla.asr.AsrException
import com.xrmech.habla.asr.AsrPass
import com.xrmech.habla.asr.AsrRequest
import com.xrmech.habla.asr.AsrResult
import com.xrmech.habla.asr.AsrSegment
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.LanguageGuess
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.SpeechRecognizer
import com.xrmech.habla.audio.PartialSnapshot
import com.xrmech.habla.audio.SegmenterListener
import com.xrmech.habla.audio.SpeechSegment
import com.xrmech.habla.data.DataException
import com.xrmech.habla.data.ExchangeRow
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.NewExchange
import com.xrmech.habla.data.SegmentCapture
import com.xrmech.habla.data.SessionRow
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.data.VocabRepository
import com.xrmech.habla.data.VocabRow
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.LlmRequest
import com.xrmech.habla.translate.LlmResponse
import com.xrmech.habla.translate.TranslationProvider
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import java.io.IOException
import java.nio.file.Files
import java.time.Instant
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map

/** Blocks in read() until closed, then returns -1: a live mic that the test never feeds. */
class IdleAudioSource : AudioSource {
    override val sampleRate = 16_000
    override val condition = MutableStateFlow<AudioCondition>(AudioCondition.Normal)
    private val closedLatch = CountDownLatch(1)
    @Volatile var opened = false
    @Volatile var closed = false
    var openError: Exception? = null

    override fun open() {
        openError?.let { throw it }
        opened = true
    }

    override fun read(buffer: ShortArray): Int {
        closedLatch.await(30, TimeUnit.SECONDS)
        return -1
    }

    override fun close() {
        closed = true
        closedLatch.countDown()
    }
}

/** Segmenter the test drives by hand; ids are 1-based like the real one. */
class FakeSegmenter : SegmenterPort {
    lateinit var listener: SegmenterListener
    var resets = 0
    var flushes = 0
    private var samples = 0L

    override fun feed(pcm: ShortArray) = Unit
    override fun flush() {
        flushes++
    }

    override fun reset() {
        resets++
        samples = 0
    }

    override val calibrating: Boolean = false

    fun emitCalibrated(rms: Double = 120.0) = listener.onCalibrated(rms)

    fun emitPartial(segmentId: Int, seconds: Double = 1.0) =
        listener.onPartial(PartialSnapshot(segmentId, ShortArray((seconds * 16_000).toInt()) { 500 }, seconds))

    /** A raw segment of [seconds] ending at the current stream position. */
    fun emitSegment(id: Int, seconds: Double = 1.0, rms: Double = 300.0, prob: Double = 0.8) {
        val n = (seconds * 16_000).toInt()
        val pcm = ShortArray(n) { ((it % 64) * 100 - 3_200).toShort() }
        listener.onSegment(SpeechSegment(id, pcm, n / 16_000.0, rms, prob, samples - n, samples))
    }

    /** Moves the stream position forward (drives the 1.2 s merge gap). */
    fun advanceSamples(n: Long) {
        samples += n
        (listener as StreamProgressListener).onAudio(samples)
    }

    fun advanceSeconds(s: Double) = advanceSamples((s * 16_000).toLong())
}

/** Scripted recognizer. Finals and partials pull steps from their queues; missing steps default. */
class FakeSpeechRecognizer : SpeechRecognizer {
    class Step(
        val text: String,
        val gate: CompletableDeferred<Unit>? = null,
        val error: Exception? = null,
        val segments: Boolean = true,
    )

    val detectLanguages = ArrayDeque<String>()
    val finals = ArrayDeque<Step>()
    val partials = ArrayDeque<Step>()
    val requests = mutableListOf<AsrRequest>()
    var detectCalls = 0
    var closed = false

    override suspend fun detectLanguage(samples: FloatArray, candidates: List<String>): LanguageGuess {
        detectCalls++
        return LanguageGuess(detectLanguages.removeFirstOrNull() ?: "es", 0.9f)
    }

    override suspend fun transcribe(samples: FloatArray, request: AsrRequest): AsrResult {
        requests += request
        val step = (if (request.pass == AsrPass.FINAL) finals else partials).removeFirstOrNull()
            ?: Step(if (request.pass == AsrPass.FINAL) "hola amigo como estas hoy" else "hola amigo")
        step.gate?.await()
        step.error?.let { throw it }
        val segs = if (step.segments && step.text.isNotEmpty()) listOf(AsrSegment(0.0, 1.0, step.text, -0.3f, 0.01f)) else emptyList()
        return AsrResult(step.text, null, null, segs, 120)
    }

    override fun close() {
        closed = true
    }
}

/** RecognizerHolder whose acquire returns [recognizer] or throws [loadError]. */
class FakeRecognizerHolder(
    val recognizer: FakeSpeechRecognizer = FakeSpeechRecognizer(),
    var loadError: Exception? = null,
) : RecognizerHolder(
    ModelManager(Files.createTempDirectory("habla-models").toFile(), TestAssets.shared.asrModels, NoDownloads),
    TestAssets.shared,
    kotlinx.coroutines.Dispatchers.Unconfined,
    1,
) {
    var releases = 0
    var gate: CompletableDeferred<Unit>? = null

    /** (modelId, liveModelId, gpu) of every acquire, for the coordinator's settings snapshot. */
    val acquired = mutableListOf<Triple<String, String?, Boolean>>()

    override suspend fun acquire(modelId: String, liveModelId: String?, gpu: Boolean): SpeechRecognizer {
        acquired += Triple(modelId, liveModelId, gpu)
        gate?.await()
        loadError?.let { throw it }
        return recognizer
    }

    override fun release() {
        releases++
    }

    private object NoDownloads : ModelDownloader {
        override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
            throw IOException("no downloads in tests")
    }
}

/**
 * Scripted OpenAI stand-in. Honours [keyProvider] like OpenAiProvider (null -> NO_KEY, no
 * request). [onFinal] answers FINAL requests; QUICK and TOPIC have simple defaults.
 */
class FakeTranslationProvider(private val keyProvider: suspend () -> String?) : TranslationProvider {
    val requests = mutableListOf<LlmRequest>()
    var finalCount = 0
    var onFinal: suspend (LlmRequest) -> LlmResponse = { okFinal("translated ${++finalCount}") }
    var onQuick: suspend (LlmRequest) -> LlmResponse = { LlmResponse("quick", usage(), false) }
    var topicCount = 0

    override suspend fun complete(request: LlmRequest): LlmResponse {
        keyProvider() ?: throw LlmException("No OpenAI API key", LlmException.Kind.NO_KEY)
        requests += request
        return when (request.kind) {
            UsageKind.FINAL -> onFinal(request)
            UsageKind.QUICK -> onQuick(request)
            UsageKind.TOPIC -> LlmResponse("topic ${++topicCount}", usage(), false)
            UsageKind.TEST -> LlmResponse("ok", usage(), false)
        }
    }

    fun requestsOf(kind: UsageKind) = requests.filter { it.kind == kind }

    companion object {
        fun usage() = TokenUsage(100, 0, 50, 0, TestAssets.shared.models.finalDefault)

        fun okFinal(translated: String, corrected: String = "corrected", confidence: Double = 0.9) = LlmResponse(
            """{"corrected": "$corrected", "translated": "$translated", "flagged_phrases": [], "confidence": $confidence, "is_correction": false}""",
            usage(),
            false,
        )
    }
}

class FakeHistoryRepository : HistoryRepository {
    data class Ended(val id: Long, val topic: String?)

    var nextSession = 42L
    var startError: DataException? = null
    var addError: DataException? = null
    val started = mutableListOf<Long>()
    val ended = mutableListOf<Ended>()
    val added = mutableListOf<NewExchange>()
    val corrected = mutableMapOf<Long, String>()
    private var nextExchange = 1L
    val sessionRows = MutableStateFlow<List<SessionRow>>(emptyList())
    val exchangeRows = MutableStateFlow<Map<Long, List<ExchangeRow>>>(emptyMap())
    val deleted = mutableListOf<Long>()

    override suspend fun startSession(mode: Mode, direction: Direction, asrModel: String): Long {
        startError?.let { throw it }
        return nextSession++.also { started += it }
    }

    override suspend fun endSession(id: Long, topicSummary: String?) {
        ended += Ended(id, topicSummary)
    }

    override suspend fun addExchange(e: NewExchange): Long {
        addError?.let { throw it }
        added += e
        return nextExchange++
    }

    override suspend fun setCorrectedTranscript(exchangeId: Long, text: String) {
        corrected[exchangeId] = text
    }

    override fun sessions(): Flow<List<SessionRow>> = sessionRows
    override fun exchanges(sessionId: Long): Flow<List<ExchangeRow>> = exchangeRows.map { it[sessionId].orEmpty() }
    override suspend fun deleteSession(id: Long) {
        deleted += id
        sessionRows.value = sessionRows.value.filterNot { it.id == id }
    }

    override suspend fun session(id: Long): SessionRow? = sessionRows.value.firstOrNull { it.id == id }
}

class FakeUsageRepository : UsageRepository {
    data class Row(val sessionId: Long?, val kind: UsageKind, val costUsd: Double)

    val rows = mutableListOf<Row>()
    val total = MutableStateFlow(0.0)

    override suspend fun record(sessionId: Long?, usage: TokenUsage, kind: UsageKind, costUsd: Double) {
        rows += Row(sessionId, kind, costUsd)
    }

    override fun totalCostUsd(): Flow<Double> = total
}

class FakeVocabRepository : VocabRepository {
    data class Saved(val phrase: FlaggedPhrase, val exchangeId: Long?, val sentence: String?)

    val saved = mutableListOf<Saved>()
    val rows = MutableStateFlow<List<VocabRow>>(emptyList())

    override suspend fun save(phrase: FlaggedPhrase, exchangeId: Long?, sourceSentence: String?) {
        saved += Saved(phrase, exchangeId, sourceSentence)
    }

    override fun all(): Flow<List<VocabRow>> = rows

    companion object {
        fun row(id: Long, term: String, meaning: String, createdAt: Instant) =
            VocabRow(id, null, term, null, meaning, "idiom", null, null, createdAt, 1)
    }
}

class FakeTrainingCapture : TrainingCapture {
    val saved = mutableListOf<SegmentCapture>()

    /** When set, every save waits on it (a slow or stuck disk). */
    var gate: CompletableDeferred<Unit>? = null

    override suspend fun save(capture: SegmentCapture): String {
        gate?.await()
        saved += capture
        return "sessions/${capture.sessionId}/segment_%03d.wav".format(capture.segmentId)
    }
}
