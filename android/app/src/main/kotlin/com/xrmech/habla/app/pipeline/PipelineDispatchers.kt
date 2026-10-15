package com.xrmech.habla.app.pipeline

import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.asCoroutineDispatcher

/**
 * The coordinator's dispatchers (design section 3.1). Tests pass one StandardTestDispatcher for
 * all four. The thread-priority hooks run first on the capture and segmenter threads; Android
 * passes `Process.setThreadPriority(...)` so this package imports no android.*.
 */
data class PipelineDispatchers(
    /** Single thread: Segmenter and SileroVad are single-thread objects. */
    val segmenter: CoroutineDispatcher,
    /** Serial: owns all mutable session state without locks. */
    val control: CoroutineDispatcher,
    /** FinalPrep / PartialPrep / toFloatSamples. */
    val compute: CoroutineDispatcher,
    /** Repositories, training audio, model load. */
    val io: CoroutineDispatcher,
    val captureThreadSetup: () -> Unit = {},
) {
    companion object {
        @OptIn(ExperimentalCoroutinesApi::class)
        fun production(
            captureThreadSetup: () -> Unit = {},
            segmenterThreadSetup: () -> Unit = {},
        ): PipelineDispatchers {
            // One process-wide thread; it lives as long as the coordinator singleton.
            val segmenter = Executors.newSingleThreadExecutor { r ->
                Thread({
                    segmenterThreadSetup()
                    r.run()
                }, "habla-segmenter").apply { isDaemon = true }
            }.asCoroutineDispatcher()
            return PipelineDispatchers(
                segmenter = segmenter,
                control = Dispatchers.Default.limitedParallelism(1),
                compute = Dispatchers.Default,
                io = Dispatchers.IO,
                captureThreadSetup = captureThreadSetup,
            )
        }
    }
}
