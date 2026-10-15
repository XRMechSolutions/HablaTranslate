package com.xrmech.habla.app.pipeline

import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.asr.AsrPipeline
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.audio.PartialPrep
import com.xrmech.habla.audio.PartialSnapshot
import com.xrmech.habla.audio.toFloatSamples
import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.LanguageRegistry
import com.xrmech.habla.translate.DirectionResolver
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.model.Direction
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Live partial line for one session (design section 3.6, port of `_on_partial_audio` +
 * `process_partial_audio`). Runs on the coordinator's control dispatcher, so its fields need no
 * locks. Partials never detect and never vote (D026/D035): `AsrPipeline.partial` uses
 * `voter.current ?: sourceLanguage`.
 */
internal class PartialWorker(
    private val profile: AsrProfile,
    private val languages: LanguageRegistry,
    private val voter: LanguageVoter,
    private val quickTranslator: QuickTranslator,
    private val partialsPaused: StateFlow<Boolean>,
    private val compute: CoroutineDispatcher,
    private val clock: () -> Long,
    private val metrics: PipelineMetrics,
    /** The session scope; quick translation jobs are its children. */
    private val scope: CoroutineScope,
    private val settings: () -> AppSettings,
    private val asr: () -> AsrPipeline?,
    private val hasKey: () -> Boolean,
    private val sourceLanguage: (Direction) -> String,
    private val state: MutableStateFlow<LiveState>,
) {
    /**
     * Latest wins (R1 "drop, never queue"): CONFLATED keeps one waiting snapshot while one is
     * being transcribed; older snapshots are overwritten, never transcribed.
     */
    private val slot = Channel<PartialSnapshot>(Channel.CONFLATED)

    var lastFinalizedRawId = 0
        private set
    private var lastPartialText = ""
    private var translationJob: Job? = null
    private var cadenceSegment = -1
    private var cadenceAtMs = 0L

    /** From the segmenter thread; never blocks. */
    fun offer(snapshot: PartialSnapshot) {
        slot.trySend(snapshot)
    }

    /** Server `reset_partial_state` on every raw segment: the partial of [rawId] is now stale. */
    fun onRawSegment(rawId: Int) {
        lastPartialText = ""
        lastFinalizedRawId = maxOf(lastFinalizedRawId, rawId)
        quickTranslator.cancelInFlight()
        translationJob?.cancel()
        translationJob = null
        state.update { it.copy(partial = it.partial?.copy(finalizing = true)) }
    }

    /** Partials switched off: drop the line and any pending translation. */
    fun clearLine() {
        translationJob?.cancel()
        translationJob = null
        state.update { it.copy(partial = null) }
    }

    fun close() {
        slot.close()
    }

    /** Processes snapshots until [close] or cancellation of the session scope. */
    suspend fun run() {
        for (snapshot in slot) process(snapshot)
    }

    private suspend fun process(snapshot: PartialSnapshot) {
        val s = settings()
        if (!s.partialsEnabled || partialsPaused.value) return
        // No wait for the model: a partial is worthless later.
        val pipeline = asr() ?: return
        if (snapshot.segmentId <= lastFinalizedRawId) return

        val samples = withContext(compute) { PartialPrep.process(snapshot.pcm, profile).toFloatSamples() }
        val outcome = pipeline.partial(samples, sourceLanguage(s.direction))
        if (outcome.aborted) {
            metrics.partialsAborted.incrementAndGet()
            return
        }
        val text = outcome.text.trim()
        val language = outcome.language ?: return
        if (text.isEmpty() || text == lastPartialText || snapshot.segmentId <= lastFinalizedRawId) return

        lastPartialText = text
        val direction = DirectionResolver.resolve(s.direction, language, voter.current, languages)
        state.update { st ->
            val previous = st.partial?.takeIf { it.segmentId == snapshot.segmentId }?.translation
            st.copy(partial = PartialLine(snapshot.segmentId, text, previous, language, direction, finalizing = false))
        }
        metrics.partialsShown.incrementAndGet()
        val now = clock()
        if (cadenceSegment == snapshot.segmentId) metrics.recordPartialCadence(now - cadenceAtMs)
        cadenceSegment = snapshot.segmentId
        cadenceAtMs = now
        HLog.text(HLog.PARTIAL) { "partial #${snapshot.segmentId}: $text" }

        if (s.partialTranslation && hasKey()) launchTranslation(snapshot.segmentId, text, direction)
    }

    /**
     * Decoupled from the ASR pass (divergence 3.9.3): the next partial transcribes while this runs.
     * Driven by QuickTranslator (latest wins itself); cancelled by the next partial, by a raw
     * final ([onRawSegment]) or with the session scope; QuickTranslator returns null on any error.
     */
    private fun launchTranslation(segmentId: Int, text: String, direction: Direction) {
        translationJob?.cancel()
        translationJob = scope.launch {
            val translated = quickTranslator.translate(text, direction) ?: return@launch
            state.update { st ->
                val line = st.partial
                if (line != null && line.segmentId == segmentId && line.text == text && !line.finalizing) {
                    st.copy(partial = line.copy(translation = translated))
                } else {
                    st
                }
            }
        }
    }
}
