package com.xrmech.habla.app.capture

import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.pipeline.PipelineMetrics
import kotlinx.coroutines.channels.SendChannel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Moves 512-sample frames from an opened [AudioSource] into [frames] on its own thread
 * (`habla-capture`). The thread never waits on anything but `read` (KB latency rule): a full
 * channel drops the frame and counts it rather than blocking, because a blocked read overruns the
 * HAL buffer and loses audio anyway, uncounted.
 *
 * End of stream closes [frames]; an [AudioSourceException] (or any other throwable) closes it
 * with that exception as the cause. [threadSetup] runs first on the capture thread (Android
 * passes `Process.setThreadPriority(THREAD_PRIORITY_URGENT_AUDIO)`); this file imports no android.*.
 */
class CapturePump(
    private val source: AudioSource,
    private val frames: SendChannel<ShortArray>,
    private val metrics: PipelineMetrics,
    private val threadSetup: () -> Unit = {},
    private val nanoClock: () -> Long = System::nanoTime,
) {
    private val _zeroSilenced = MutableStateFlow(false)

    /** True after 62 consecutive all-zero frames (2 s), false again on the first non-zero frame. */
    val zeroSilenced: StateFlow<Boolean> = _zeroSilenced

    @Volatile private var running = false
    private var thread: Thread? = null
    private var zeroRun = 0
    private var lastDropWarnNanos = 0L
    private var warnedOnce = false

    @Synchronized
    fun start() {
        check(thread == null) { "CapturePump already started" }
        require(source.sampleRate == SAMPLE_RATE) {
            "AudioSource sample rate must be $SAMPLE_RATE, got ${source.sampleRate}"
        }
        running = true
        // Driven by source.read until end of stream, a source error, a closed consumer or
        // stopAndJoin(); every exit closes `frames`, which ends the coordinator's feeder.
        thread = Thread(::loop, THREAD_NAME).apply {
            isDaemon = true
            start()
        }
    }

    /** Asks the loop to end and waits up to [timeoutMs]; close the source first so a blocked read returns. */
    fun stopAndJoin(timeoutMs: Long = 1_000) {
        running = false
        val t = synchronized(this) { thread } ?: return
        t.join(timeoutMs)
        if (t.isAlive) HLog.w(HLog.CAPTURE, "capture thread still alive after $timeoutMs ms")
    }

    private fun loop() {
        var failure: Throwable? = null
        try {
            threadSetup()
            while (running) {
                val buf = ShortArray(FRAME)
                val n = source.read(buf)
                when {
                    n > 0 -> if (!offer(if (n == FRAME) buf else buf.copyOf(n))) return
                    n == 0 -> Thread.yield()
                    else -> return
                }
            }
        } catch (e: Throwable) {
            failure = e
            HLog.w(HLog.CAPTURE, "capture ended by ${e.javaClass.simpleName}: ${e.message}")
        } finally {
            frames.close(failure)
        }
    }

    /** Returns false when the consumer has closed the channel (nothing left to feed). */
    private fun offer(frame: ShortArray): Boolean {
        trackZeros(frame)
        val result = frames.trySend(frame)
        if (result.isSuccess) return true
        if (result.isClosed) return false
        val dropped = metrics.framesDropped.incrementAndGet()
        val now = nanoClock()
        if (!warnedOnce || now - lastDropWarnNanos >= DROP_WARN_INTERVAL_NANOS) {
            warnedOnce = true
            lastDropWarnNanos = now
            HLog.w(HLog.CAPTURE, "frames channel full: $dropped frames dropped so far")
        }
        return true
    }

    private fun trackZeros(frame: ShortArray) {
        if (frame.all { it.toInt() == 0 }) {
            zeroRun++
            if (zeroRun >= SILENCED_FRAMES) _zeroSilenced.value = true
        } else {
            zeroRun = 0
            _zeroSilenced.value = false
        }
    }

    companion object {
        const val FRAME = 512
        const val SAMPLE_RATE = 16_000
        const val THREAD_NAME = "habla-capture"

        /** 62 frames x 32 ms = 2 s, the segmenter's calibration length. */
        const val SILENCED_FRAMES = 62
        private const val DROP_WARN_INTERVAL_NANOS = 5_000_000_000L
    }
}
