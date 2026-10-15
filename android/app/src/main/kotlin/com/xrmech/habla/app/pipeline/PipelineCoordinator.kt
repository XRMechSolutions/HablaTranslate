package com.xrmech.habla.app.pipeline

import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.app.capture.AudioSource
import com.xrmech.habla.app.capture.AudioSourceException
import com.xrmech.habla.app.capture.CapturePump
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.state.AppSettingsState
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.ThermalMonitor
import com.xrmech.habla.asr.AsrException
import com.xrmech.habla.asr.AsrPipeline
import com.xrmech.habla.asr.FinalAsrOutcome
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.asr.RejectReason
import com.xrmech.habla.asr.SpeechRecognizer
import com.xrmech.habla.audio.FinalPrep
import com.xrmech.habla.audio.PartialSnapshot
import com.xrmech.habla.audio.SegmenterListener
import com.xrmech.habla.audio.SpeechSegment
import com.xrmech.habla.audio.VadException
import com.xrmech.habla.audio.VadModel
import com.xrmech.habla.audio.toFloatSamples
import com.xrmech.habla.data.ApiKeyStore
import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.AsrStatus
import com.xrmech.habla.data.DataException
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.NewExchange
import com.xrmech.habla.data.SegmentCapture
import com.xrmech.habla.data.TrainingAudioStore
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.ConversationContext
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.DirectionResolver
import com.xrmech.habla.translate.IdiomMerge
import com.xrmech.habla.translate.IdiomScanner
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.TopicTracker
import com.xrmech.habla.translate.TranslationOutcome
import com.xrmech.habla.translate.Translator
import com.xrmech.habla.translate.ContextExchange
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

/** One-method seam over `TrainingAudioStore.saveSegment` (D037 j): the store is final and needs a database. */
fun interface TrainingCapture {
    suspend fun save(capture: SegmentCapture): String?

    companion object {
        fun of(store: TrainingAudioStore): TrainingCapture = TrainingCapture { store.saveSegment(it) }
    }
}

/**
 * Drives one listening session through :audio, :asr, :translate and :data as the server's
 * `routes/websocket.py` + `pipeline/orchestrator.py` do (design section 3; every divergence is in
 * section 3.9). All mutable session state lives on the serial `control` dispatcher; the segmenter
 * listener only `trySend`s into control-side channels.
 */
class PipelineCoordinator(
    private val assets: SharedAssets,
    private val segmenterFactory: SegmenterFactory,
    private val vadFactory: () -> VadModel?,
    private val recognizers: RecognizerHolder,
    private val voter: LanguageVoter,
    private val translator: Translator,
    private val quickTranslator: QuickTranslator,
    private val topicTracker: TopicTracker,
    private val context: ConversationContext,
    private val idiomScanner: IdiomScanner,
    private val costMeter: CostMeter,
    private val settings: AppSettingsState,
    private val keyStore: ApiKeyStore,
    private val keyHealth: KeyHealthTracker,
    private val history: HistoryRepository,
    private val usage: UsageRepository,
    private val training: TrainingCapture,
    private val thermal: ThermalMonitor,
    private val dispatchers: PipelineDispatchers,
    /** Elapsed ms (SystemClock.elapsedRealtime on device). */
    private val clock: () -> Long,
    /** Wall-clock ms for card timestamps. */
    private val wallClock: () -> Long = System::currentTimeMillis,
    /** Runs abort()'s endSession after the session scope is gone. */
    private val cleanupScope: CoroutineScope = CoroutineScope(SupervisorJob() + dispatchers.io),
    /** Metrics line period; <= 0 disables the ticker (JVM tests that use advanceUntilIdle). */
    private val metricsLogIntervalMs: Long = 60_000,
    /** LiveState.metrics / AUDIO_DROPPED refresh period; <= 0 disables (updated at stop only). */
    private val publishIntervalMs: Long = 1_000,
) : LiveController {
    private val profile get() = assets.asrProfile

    private val _state = MutableStateFlow(LiveState())
    override val state: StateFlow<LiveState> = _state.asStateFlow()

    private val _lastMetricsLine = MutableStateFlow<String?>(null)

    /** Last session metrics line, for Settings > Diagnostics. */
    override val lastMetricsLine: StateFlow<String?> = _lastMetricsLine.asStateFlow()

    private val busy = AtomicBoolean(false)
    @Volatile private var session: Session? = null

    /** Lives as long as the coordinator; internal stops are launched here so they never cancel themselves. */
    private val coordinatorScope = CoroutineScope(SupervisorJob() + dispatchers.control)

    private sealed interface Event {
        class Calibrated(val rms: Double) : Event
        class Segment(val segment: SpeechSegment) : Event
        class Audio(val samplesFed: Long) : Event
        data object VadFallback : Event
        class FeederDone(val cause: Throwable?) : Event
    }

    private inner class Session(
        val id: Long,
        val source: AudioSource,
        /** trainingCapture and asrModelId apply from the next session (snapshotted here). */
        val snapshot: AppSettings,
        val scope: CoroutineScope,
        val startedAtMs: Long,
    ) {
        val metrics = PipelineMetrics().also { it.sessionId = id }

        /** Unbounded: segmenter events are a few per second plus one progress tick per 32 ms frame. */
        val events = Channel<Event>(Channel.UNLIMITED)

        /** 256 frames = 8.2 s of audio; a full channel drops frames in CapturePump (counted), never blocks capture. */
        val frames = Channel<ShortArray>(FRAMES_CAPACITY)

        val queue = FinalQueue()

        /** CONFLATED doorbell: the worker drains the whole queue per ring. */
        val doorbell = Channel<Unit>(Channel.CONFLATED)
        val depth = MutableStateFlow(0)
        val merger = SegmentMerger(profile.merge, { enqueue(this, it) })
        val translations = SupervisorJob(scope.coroutineContext[Job])
        val translationScope = CoroutineScope(scope.coroutineContext + translations)
        val feederDrained = CompletableDeferred<Unit>()
        val listener = Listener(this)

        lateinit var recognizer: Deferred<SpeechRecognizer>
        lateinit var segmenter: SegmenterPort
        lateinit var partials: PartialWorker
        var pump: CapturePump? = null
        var asr: AsrPipeline? = null
        var vad: VadModel? = null
        var finalWorker: Job? = null
        var stopping: Deferred<Unit>? = null
        var settingsNow: AppSettings = snapshot
        @Volatile var hasKey = false
        var consecutiveNativeErrors = 0
        var trainingWarned = false
    }

    /** Runs on the segmenter thread: only hands work to the control side. */
    private inner class Listener(private val s: Session) : SegmenterListener, StreamProgressListener {
        override fun onCalibrated(noiseFloorRms: Double) {
            s.events.trySend(Event.Calibrated(noiseFloorRms))
        }

        override fun onPartial(snapshot: PartialSnapshot) {
            s.partials.offer(snapshot)
        }

        override fun onSegment(segment: SpeechSegment) {
            s.events.trySend(Event.Segment(segment))
        }

        override fun onVadFallback(cause: VadException) {
            HLog.w(HLog.PIPELINE, "VAD failed mid-session (${cause.message}); energy fallback")
            s.events.trySend(Event.VadFallback)
        }

        override fun onAudio(samplesFed: Long) {
            s.events.trySend(Event.Audio(samplesFed))
        }
    }

    // ---------------------------------------------------------------------------------------
    // start (design 3.2)

    /**
     * Starts a session over an unopened [source] and returns its id.
     * @throws IllegalStateException when a session is already active
     * @throws SessionStartException history, VAD-less segmenter or `source.open()` failure
     */
    suspend fun start(source: AudioSource): Long {
        if (!busy.compareAndSet(false, true)) throw IllegalStateException("session already active")
        _state.value = LiveState(phase = SessionPhase.Starting, costs = costMeter.totals.value)
        var sessionId: Long? = null
        var started: Session? = null
        try {
            return withContext(dispatchers.control) {
                val appSettings = settings.awaitLoaded()
                val id = try {
                    withContext(dispatchers.io) {
                        history.startSession(appSettings.mode, appSettings.direction, appSettings.asrModelId)
                    }
                } catch (e: DataException) {
                    throw SessionStartException(SessionError.Storage("Couldn't start a session: ${e.message}"))
                }
                sessionId = id
                costMeter.resetSession()
                context.reset()
                voter.reset()
                quickTranslator.cancelInFlight()

                val scope = CoroutineScope(
                    SupervisorJob() + dispatchers.control +
                        CoroutineExceptionHandler { _, e -> HLog.e(HLog.PIPELINE, "session coroutine failed", e) },
                )
                val s = Session(id, source, appSettings, scope, clock())
                started = s
                session = s
                _state.update { it.copy(sessionId = id, startedAtMs = s.startedAtMs) }

                // Model load (1-3 s) runs beside calibration (2 s), so it is hidden. Driven to completion
                // by acquire; cancelled with the session scope; a failure fails the session (watcher below).
                s.recognizer = scope.async(dispatchers.io) { recognizers.acquire(appSettings.asrModelId) }

                val vad = vadFactory()
                if (vad == null) {
                    HLog.w(HLog.PIPELINE, "Silero VAD unavailable; energy fallback")
                    addNotice(SessionNotice.VAD_FALLBACK)
                }
                s.vad = vad
                s.partials = newPartialWorker(s)
                s.segmenter = segmenterFactory.create(profile, vad, s.listener)
                s.segmenter.reset()

                try {
                    source.open()
                } catch (e: AudioSourceException) {
                    throw SessionStartException(SessionError.Audio(e.kind, e.message ?: "Microphone unavailable"))
                }
                s.pump = CapturePump(source, s.frames, s.metrics, dispatchers.captureThreadSetup).also { it.start() }
                _state.update { it.copy(phase = SessionPhase.Calibrating) }
                launchSessionCoroutines(s)
                HLog.i(HLog.PIPELINE, "session $id started (model ${appSettings.asrModelId}, capture ${appSettings.trainingCapture})")
                id
            }
        } catch (e: CancellationException) {
            cleanupFailedStart(started, sessionId, null)
            throw e
        } catch (e: SessionStartException) {
            cleanupFailedStart(started, sessionId, e.error)
            throw e
        } catch (e: Exception) {
            val error = SessionError.Storage(e.message ?: e.javaClass.simpleName)
            cleanupFailedStart(started, sessionId, error)
            throw SessionStartException(error)
        }
    }

    private suspend fun cleanupFailedStart(s: Session?, sessionId: Long?, error: SessionError?) {
        withContext(NonCancellable) {
            runCatching { (s?.source)?.close() }
            s?.pump?.let { p -> withContext(dispatchers.io) { p.stopAndJoin() } }
            s?.scope?.cancel()
            recognizers.release()
            runCatching { s?.vad?.close() }
            if (sessionId != null) {
                try {
                    withContext(dispatchers.io) { history.endSession(sessionId, null) }
                } catch (e: Exception) {
                    HLog.e(HLog.PIPELINE, "endSession after failed start: ${e.message}")
                }
            }
            session = null
            HLog.w(HLog.PIPELINE, "session start failed: ${error?.message ?: "cancelled"}")
            _state.update { it.copy(phase = error?.let { SessionPhase.Failed(it) } ?: SessionPhase.Idle) }
            busy.set(false)
        }
    }

    private fun newPartialWorker(s: Session) = PartialWorker(
        profile = profile,
        languages = assets.languages,
        voter = voter,
        quickTranslator = quickTranslator,
        partialsPaused = thermal.partialsPaused,
        compute = dispatchers.compute,
        clock = clock,
        metrics = s.metrics,
        scope = s.scope,
        settings = { s.settingsNow },
        asr = { s.asr },
        hasKey = { s.hasKey },
        sourceLanguage = ::sourceLanguage,
        state = _state,
    )

    // Every coroutine below is a child of the session scope: cancelled at the end of stop/abort.
    private fun launchSessionCoroutines(s: Session) {
        // Feeder: drains `frames` until CapturePump closes it (end of stream / source error /
        // stop), then flushes the segmenter and tells the control loop.
        s.scope.launch(dispatchers.segmenter) {
            var fed = 0L
            var cause: Throwable? = null
            try {
                for (frame in s.frames) {
                    s.segmenter.feed(frame)
                    fed += frame.size
                    s.listener.onAudio(fed)
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                cause = e
            }
            s.segmenter.flush()
            s.events.send(Event.FeederDone(cause))
        }

        // Control loop: segmenter events in arrival order; ends when the scope is cancelled.
        s.scope.launch { for (ev in s.events) handle(s, ev) }

        // Final worker: strictly in order, one at a time (server `_process_queue`).
        s.finalWorker = s.scope.launch {
            for (ring in s.doorbell) {
                while (true) {
                    val item = s.queue.take() ?: break
                    publishDepth(s)
                    try {
                        processFinal(s, item)
                    } finally {
                        s.queue.done()
                        publishDepth(s)
                    }
                }
            }
        }

        // Partial worker: until the session scope ends.
        s.scope.launch { s.partials.run() }

        // Model load watcher: builds the per-session AsrPipeline or fails the session.
        s.scope.launch {
            try {
                val rec = s.recognizer.await()
                s.asr = AsrPipeline(rec, voter, profile, assets.languages)
            } catch (e: CancellationException) {
                throw e
            } catch (e: AsrException) {
                fail(s, SessionError.ModelLoad(modelLoadMessage(e)))
            } catch (e: Exception) {
                fail(s, SessionError.ModelLoad("Speech model failed to load (${e.message}) - delete and re-download it in Settings"))
            }
        }

        // Usage recorder: every cost record of this session becomes a usage_log row.
        s.scope.launch {
            costMeter.records.collect { r ->
                try {
                    withContext(dispatchers.io) { usage.record(s.id, r.usage, r.kind, r.costUsd) }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    HLog.w(HLog.PIPELINE, "usage record failed: ${e.javaClass.simpleName}: ${e.message}")
                }
            }
        }
        s.scope.launch { costMeter.totals.collect { t -> _state.update { it.copy(costs = t) } } }
        s.scope.launch { keyStore.hasKey.collect { s.hasKey = it } }

        // Settings watcher: direction changes to a fixed direction reset the voter (server
        // `set_direction`); partials switched off clear the line.
        s.scope.launch {
            var previous = s.snapshot.direction
            settings.settings.filterNotNull().collect { now ->
                s.settingsNow = now
                if (now.direction != previous && now.direction != Direction.AUTO) voter.reset()
                previous = now.direction
                if (!now.partialsEnabled) s.partials.clearLine()
            }
        }

        val pump = checkNotNull(s.pump)
        s.scope.launch {
            combine(s.source.condition, pump.zeroSilenced) { c, zeros ->
                if (zeros && c == AudioCondition.Normal) AudioCondition.Silenced else c
            }.collect { c -> _state.update { it.copy(audio = c) } }
        }
        s.scope.launch {
            thermal.partialsPaused.collect { paused ->
                if (paused) addNotice(SessionNotice.THERMAL_PAUSE) else removeNotice(SessionNotice.THERMAL_PAUSE)
            }
        }
        s.scope.launch { thermal.status.drop(1).collect { s.metrics.thermalChanges.incrementAndGet() } }

        if (metricsLogIntervalMs > 0) {
            s.scope.launch {
                while (true) {
                    delay(metricsLogIntervalMs)
                    HLog.i(HLog.METRICS, s.metrics.toJsonLine())
                }
            }
        }
        if (publishIntervalMs > 0) {
            s.scope.launch {
                while (true) {
                    delay(publishIntervalMs)
                    publishMetrics(s)
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // control-side events (design 3.3, 3.4)

    private fun handle(s: Session, ev: Event) {
        when (ev) {
            is Event.Calibrated -> {
                val level = NoiseLevel.of(ev.rms)
                _state.update {
                    it.copy(
                        noise = level,
                        noiseFloorRms = ev.rms,
                        phase = if (it.phase == SessionPhase.Calibrating) SessionPhase.Listening else it.phase,
                    )
                }
                if (level == NoiseLevel.LOUD) {
                    HLog.i(HLog.PIPELINE, "Loud environment detected (RMS ${ev.rms.toInt()}) - speech detection threshold raised automatically")
                    addNotice(SessionNotice.LOUD_ENVIRONMENT)
                }
            }
            is Event.Segment -> {
                s.partials.onRawSegment(ev.segment.id)
                s.merger.onSegment(ev.segment)
            }
            is Event.Audio -> s.merger.onAudio(ev.samplesFed)
            Event.VadFallback -> addNotice(SessionNotice.VAD_FALLBACK)
            is Event.FeederDone -> {
                s.merger.flushNow()
                s.feederDrained.complete(Unit)
                if (s.stopping == null) {
                    val reason = when (val c = ev.cause) {
                        null -> StopReason.EndOfStream
                        is AudioSourceException -> StopReason.AudioFailed(c)
                        else -> StopReason.AudioFailed(
                            AudioSourceException(c.message ?: c.javaClass.simpleName, AudioSourceException.Kind.IO, cause = c),
                        )
                    }
                    coordinatorScope.launch { stop(reason) }
                }
            }
        }
    }

    private fun enqueue(s: Session, merged: MergedSegment) {
        val dropped = s.queue.offer(merged)
        s.metrics.notePeakQueueDepth(s.queue.depth)
        if (dropped != null) {
            s.metrics.finalsDropped.incrementAndGet()
            HLog.w(HLog.PIPELINE, "final queue full: dropped segment #${dropped.seq}")
            addNotice(SessionNotice.FALLING_BEHIND)
        }
        publishDepth(s)
        s.doorbell.trySend(Unit)
    }

    private fun publishDepth(s: Session) {
        val depth = s.queue.depth
        s.depth.value = depth
        _state.update {
            it.copy(
                queueDepth = depth,
                phase = if (it.phase is SessionPhase.Finishing) SessionPhase.Finishing(depth) else it.phase,
                metrics = s.metrics.snapshot(),
            )
        }
    }

    private fun publishMetrics(s: Session) {
        if (s.metrics.framesDropped.get() > 0) addNotice(SessionNotice.AUDIO_DROPPED)
        _state.update { it.copy(metrics = s.metrics.snapshot()) }
    }

    /** Server `_source_language`. */
    private fun sourceLanguage(direction: Direction): String = when (direction) {
        Direction.ES_TO_EN -> "es"
        Direction.EN_TO_ES -> "en"
        Direction.AUTO -> voter.current ?: profile.whisper.defaultLanguage
    }

    // ---------------------------------------------------------------------------------------
    // final worker (design 3.5)

    private suspend fun processFinal(s: Session, item: MergedSegment) {
        val t0 = clock()
        val rec = try {
            s.recognizer.await()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            return // the session is already failing (model load watcher)
        }
        val asr = s.asr ?: AsrPipeline(rec, voter, profile, assets.languages).also { s.asr = it }
        val samples = withContext(dispatchers.compute) { FinalPrep.process(item.pcm, profile).toFloatSamples() }
        val direction = s.settingsNow.direction
        val outcome = try {
            asr.final(samples, sourceLanguage(direction))
        } catch (e: AsrException) {
            s.metrics.asrErrors.incrementAndGet()
            HLog.w(HLog.PIPELINE, "segment #${item.seq} transcription failed (${e.kind}): ${e.message}")
            _state.update { it.copy(lastAsrError = e.message ?: e.kind.name) }
            if (e.kind == AsrException.Kind.NATIVE_ERROR && ++s.consecutiveNativeErrors >= MAX_NATIVE_ERRORS) {
                coordinatorScope.launch { stop(StopReason.AsrFailed(e)) }
            }
            return
        }
        s.consecutiveNativeErrors = 0
        s.metrics.segmentsProcessed.incrementAndGet()
        s.metrics.recordFinalAsr(outcome.result.elapsedMs)
        _state.update { it.copy(votedLanguage = voter.current) }

        when (outcome.reason) {
            RejectReason.NO_SEGMENTS, RejectReason.BAD_TRANSCRIPT -> {
                if (outcome.reason == RejectReason.NO_SEGMENTS) {
                    s.metrics.asrEmptyCount.incrementAndGet()
                } else {
                    s.metrics.asrRejectedCount.incrementAndGet()
                }
                HLog.i(HLog.PIPELINE, "segment #${item.seq} rejected (${if (outcome.reason == RejectReason.NO_SEGMENTS) "empty" else "bad"})")
                if (s.snapshot.trainingCapture) saveTraining(s, item, outcome, AsrStatus.REJECTED)
            }
            RejectReason.OK -> {
                val text = outcome.transcript.trim()
                val effective = DirectionResolver.resolve(direction, outcome.detectedLanguage, voter.current, assets.languages)
                if (outcome.clippedOnset) {
                    s.metrics.clippedOnsetCount.incrementAndGet()
                    HLog.i(HLog.PIPELINE, "segment #${item.seq} has a clipped onset")
                }
                val newestRaw = item.rawIds.maxOrNull() ?: 0
                val card = ExchangeCard(item.seq, null, wallClock(), text, outcome.detectedLanguage, effective, CardTranslation.Pending, item.rawIds)
                _state.update { st ->
                    st.copy(
                        exchanges = (st.exchanges + card).takeLast(LiveState.MAX_CARDS),
                        partial = st.partial?.takeUnless { it.segmentId <= newestRaw },
                        lastAsrError = null,
                    )
                }
                HLog.text(HLog.PIPELINE) { "final #${item.seq} [${outcome.language}]: $text" }
                val audioPath = if (s.snapshot.trainingCapture) saveTraining(s, item, outcome, AsrStatus.ACCEPTED) else null
                launchTranslation(s, item, text, effective, outcome.detectedLanguage, audioPath, t0)
            }
        }
        publishMetrics(s)
    }

    /** Raw (pre-FinalPrep) PCM, like the server's recorder; never breaks the pipeline. */
    private suspend fun saveTraining(s: Session, item: MergedSegment, outcome: FinalAsrOutcome, status: AsrStatus): String? {
        val logprobs = outcome.result.segments.mapNotNull { it.avgLogprob }
        val capture = SegmentCapture(
            sessionId = s.id,
            segmentId = item.seq,
            pcm = item.pcm,
            audioRms = item.rms,
            vadAvgProb = item.vadAvgProb,
            detectedLanguage = outcome.detectedLanguage,
            status = status,
            asrTimeSeconds = outcome.result.elapsedMs / 1000.0,
            asrAvgLogprob = if (logprobs.isEmpty()) null else logprobs.average(),
        )
        val path = try {
            withContext(dispatchers.io) { training.save(capture) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            HLog.w(HLog.PIPELINE, "training save failed: ${e.javaClass.simpleName}: ${e.message}")
            null
        }
        if (path == null && !s.trainingWarned) {
            s.trainingWarned = true
            HLog.w(HLog.PIPELINE, "training audio not saved for segment #${item.seq} (low storage or I/O error)")
        }
        return path
    }

    // ---------------------------------------------------------------------------------------
    // translation job (design 3.7)

    /**
     * Fire-and-forget like the server's `asyncio.create_task`. Driven by the Translator call;
     * cancelled only by stop's 20 s drain timeout or abort, in which case the exchange is still
     * persisted as Failed(TIMEOUT) from a NonCancellable block. No exception but cancellation escapes.
     */
    private fun launchTranslation(
        s: Session,
        item: MergedSegment,
        text: String,
        effective: Direction,
        detectedLanguage: String?,
        audioPath: String?,
        t0: Long,
    ) {
        s.translationScope.launch {
            var written = false
            val hits = idiomScanner.scan(text)
            fun failedRow(raw: String) = NewExchange(
                sessionId = s.id, segmentId = item.seq, direction = effective, rawTranscript = text,
                correctedSource = null, translation = raw, confidence = 0.0, isCorrection = false,
                correction = null, flagged = hits, processingMs = clock() - t0,
                detectedLanguage = detectedLanguage, translationError = true, audioPath = audioPath,
            )
            try {
                val tStart = clock()
                val outcome = translator.translate(text, effective, s.settingsNow.mode, context.recent(), context.topicSummary)
                s.metrics.recordTranslate(clock() - tStart)
                when (outcome) {
                    is TranslationOutcome.Ok -> {
                        val r = outcome.result
                        val flagged: List<FlaggedPhrase> = IdiomMerge.merge(hits, r.flaggedPhrases)
                        s.metrics.idiomPatternDbHits.addAndGet(flagged.count { it.source == "pattern_db" })
                        s.metrics.idiomLlmHits.addAndGet(flagged.count { it.source == "llm" })
                        if (r.confidence < LOW_CONFIDENCE) {
                            s.metrics.lowConfidenceCount.incrementAndGet()
                            HLog.w(HLog.TRANSLATE, "segment #${item.seq} low confidence ${r.confidence}")
                        }
                        if (r.isCorrection) s.metrics.correctionsDetected.incrementAndGet()
                        context.add(ContextExchange(r.corrected, r.translated))
                        val processingMs = clock() - t0
                        written = true
                        val exchangeId = persist(
                            NewExchange(
                                sessionId = s.id, segmentId = item.seq, direction = effective, rawTranscript = text,
                                correctedSource = r.corrected, translation = r.translated, confidence = r.confidence,
                                isCorrection = r.isCorrection, correction = r.correctionDetail, flagged = flagged,
                                processingMs = processingMs, detectedLanguage = detectedLanguage,
                                translationError = false, audioPath = audioPath,
                            ),
                        )
                        updateCard(item.seq) {
                            it.copy(exchangeId = exchangeId, translation = CardTranslation.Done(r, flagged, outcome.model, processingMs))
                        }
                        _state.update { it.copy(lastTranslationError = null) }
                        s.metrics.translationsCompleted.incrementAndGet()
                        s.metrics.recordSpeechEndToTranslation(clock() - (s.startedAtMs + item.endSample / 16))
                        keyHealth.report(null)
                        HLog.text(HLog.TRANSLATE) { "translation #${item.seq}: ${r.translated}" }
                        // Topic update: same lifetime as translations, so stop's drain waits for it.
                        s.translationScope.launch { topicTracker.update(r.corrected, r.translated) }
                    }
                    is TranslationOutcome.Failed -> {
                        written = true
                        val exchangeId = persist(failedRow(outcome.raw))
                        val error = outcome.error
                        updateCard(item.seq) {
                            it.copy(exchangeId = exchangeId, translation = CardTranslation.Failed(error.kind, error.message ?: error.kind.name))
                        }
                        _state.update { it.copy(lastTranslationError = error) }
                        s.metrics.translationErrors.incrementAndGet()
                        keyHealth.report(error.kind)
                        HLog.w(HLog.TRANSLATE, "segment #${item.seq} translation failed: ${error.kind}")
                    }
                }
            } catch (e: CancellationException) {
                if (!written) {
                    withContext(NonCancellable) {
                        val exchangeId = persist(failedRow(text))
                        updateCard(item.seq) {
                            it.copy(exchangeId = exchangeId, translation = CardTranslation.Failed(LlmException.Kind.TIMEOUT, STOPPED_MESSAGE))
                        }
                    }
                }
                throw e
            } catch (e: Exception) {
                HLog.e(HLog.TRANSLATE, "translation job for #${item.seq} failed", e)
            }
        }
    }

    private suspend fun persist(exchange: NewExchange): Long? = try {
        withContext(dispatchers.io) { history.addExchange(exchange) }
    } catch (e: DataException) {
        HLog.e(HLog.PIPELINE, "history write failed for #${exchange.segmentId}: ${e.message}")
        addNotice(SessionNotice.HISTORY_WRITE_FAILED)
        _state.update { it.copy(lastHistoryError = e.message ?: "database error") }
        null
    }

    private fun updateCard(seq: Int, transform: (ExchangeCard) -> ExchangeCard) {
        _state.update { st -> st.copy(exchanges = st.exchanges.map { if (it.seq == seq) transform(it) else it }) }
    }

    private fun addNotice(n: SessionNotice) = _state.update { if (n in it.notices) it else it.copy(notices = it.notices + n) }

    private fun removeNotice(n: SessionNotice) = _state.update { if (n in it.notices) it.copy(notices = it.notices - n) else it }

    // ---------------------------------------------------------------------------------------
    // stop / abort (design 3.8)

    /** Drains (30 s finals, 20 s translations) and ends the session; no-op without one. */
    suspend fun stop(reason: StopReason) {
        val s = session ?: return
        val job = synchronized(s) {
            s.stopping ?: coordinatorScope.async { stopInternal(s, reason, null) }.also { s.stopping = it }
        }
        job.await()
    }

    private fun fail(s: Session, error: SessionError) {
        synchronized(s) {
            if (s.stopping != null) return
            s.stopping = coordinatorScope.async { stopInternal(s, null, error) }
        }
    }

    private suspend fun stopInternal(s: Session, reason: StopReason?, error: SessionError?) {
        val failure: SessionError? = error ?: when (reason) {
            is StopReason.AudioFailed -> SessionError.Audio(
                reason.error.kind,
                reason.error.message ?: "Microphone stopped working",
            )
            is StopReason.AsrFailed -> SessionError.Asr("Transcription keeps failing: ${reason.error.message}")
            else -> null
        }
        HLog.i(HLog.PIPELINE, "session ${s.id} stopping (${reason ?: failure})")
        _state.update { it.copy(phase = SessionPhase.Finishing(s.queue.depth)) }

        // 1. Close the source (unblocks read; the mic indicator goes off) and drain capture.
        runCatching { s.source.close() }
        s.pump?.let { p -> withContext(dispatchers.io) { p.stopAndJoin(1_000) } }
        withTimeoutOrNull(FEEDER_DRAIN_MS) { s.feederDrained.await() }
            ?: HLog.w(HLog.PIPELINE, "feeder did not drain in time")

        // 2-3. Finals (30 s), then translations (20 s); leftovers are dropped / persisted as TIMEOUT.
        if (withTimeoutOrNull(FINALS_DRAIN_MS) { s.depth.first { it == 0 } } == null) {
            val dropped = s.queue.clearWaiting()
            s.metrics.finalsDropped.addAndGet(dropped)
            s.finalWorker?.cancel()
            HLog.w(HLog.PIPELINE, "stop: finals did not drain in 30 s; dropped $dropped waiting")
        }
        val drained = withTimeoutOrNull(TRANSLATIONS_DRAIN_MS) {
            while (true) {
                val active = s.translations.children.filter { it.isActive }.toList()
                if (active.isEmpty()) break
                active.joinAll()
            }
        }
        if (drained == null) {
            HLog.w(HLog.PIPELINE, "stop: translations did not finish in 20 s; cancelling")
            val pending = s.translations.children.toList()
            pending.forEach { it.cancel() }
            withContext(NonCancellable) { pending.joinAll() }
        }

        // 4. End the session row, free the model and the VAD.
        withContext(NonCancellable) {
            try {
                withContext(dispatchers.io) { history.endSession(s.id, context.topicSummary.ifBlank { null }) }
            } catch (e: Exception) {
                HLog.e(HLog.PIPELINE, "endSession failed: ${e.message}")
            }
            recognizers.release()
            s.vad?.let { v -> withContext(dispatchers.segmenter) { runCatching { v.close() } } }
        }
        s.partials.close()
        s.scope.cancel()
        if (s.metrics.framesDropped.get() > 0) addNotice(SessionNotice.AUDIO_DROPPED)
        val line = s.metrics.toJsonLine()
        HLog.i(HLog.METRICS, line)
        _lastMetricsLine.value = line
        session = null
        _state.update {
            it.copy(
                phase = failure?.let { f -> SessionPhase.Failed(f) } ?: SessionPhase.Idle,
                partial = null,
                queueDepth = 0,
                metrics = s.metrics.snapshot(),
            )
        }
        busy.set(false)
    }

    /**
     * Non-suspending best effort (ListeningService.onDestroy without a stop): closes the source,
     * cancels everything and ends the session row on [cleanupScope]. Exchanges already persisted
     * stay; in-flight translations persist themselves as TIMEOUT while being cancelled.
     */
    fun abort() {
        val s = session ?: return
        session = null
        HLog.w(HLog.PIPELINE, "session ${s.id} aborted")
        runCatching { s.source.close() }
        s.partials.close()
        s.scope.cancel()
        recognizers.release()
        val vad = s.vad
        val topic = context.topicSummary.ifBlank { null }
        // Driven to completion on the cleanup scope; VAD closed on its own thread after the last feed.
        cleanupScope.launch {
            vad?.let { v -> withContext(dispatchers.segmenter) { runCatching { v.close() } } }
            try {
                history.endSession(s.id, topic)
            } catch (e: Exception) {
                HLog.e(HLog.PIPELINE, "endSession after abort failed: ${e.message}")
            }
        }
        _state.update { it.copy(phase = SessionPhase.Idle, partial = null, queueDepth = 0) }
        busy.set(false)
    }

    /**
     * The foreground service could not start (startForeground refused): show it like any other
     * start failure. Ignored while a session runs.
     */
    fun reportStartFailure(error: SessionError) {
        if (busy.get()) return
        HLog.w(HLog.SERVICE, "foreground start refused: ${error.message}")
        _state.update { it.copy(phase = SessionPhase.Failed(error)) }
    }

    private fun modelLoadMessage(e: AsrException): String = when (e.kind) {
        AsrException.Kind.MODEL_MISSING -> "Speech model missing - download it in Settings"
        else -> "Speech model failed to load (${e.message}) - delete and re-download it in Settings"
    }

    private companion object {
        const val FRAMES_CAPACITY = 256
        const val MAX_NATIVE_ERRORS = 3
        const val LOW_CONFIDENCE = 0.3
        const val FINALS_DRAIN_MS = 30_000L
        const val TRANSLATIONS_DRAIN_MS = 20_000L
        const val FEEDER_DRAIN_MS = 5_000L
        const val STOPPED_MESSAGE = "Stopped before translation finished"
    }
}
