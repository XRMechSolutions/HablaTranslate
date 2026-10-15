package com.xrmech.habla.app.fakes

import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.app.capture.AudioSource
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * AudioSource that returns scripted steps: a frame of samples, an exception, or end of stream.
 * With [blockWhenEmpty] a read waits (up to 5 s) for the test to push more steps.
 */
class ScriptedAudioSource(
    steps: List<Step> = emptyList(),
    private val blockWhenEmpty: Boolean = false,
    override val sampleRate: Int = 16_000,
) : AudioSource {
    sealed interface Step {
        data class Samples(val samples: ShortArray) : Step
        data class Fail(val error: Throwable) : Step
        data object End : Step
    }

    private val queue = LinkedBlockingQueue(steps)
    override val condition = MutableStateFlow<AudioCondition>(AudioCondition.Normal)

    @Volatile var opened = false
    @Volatile var closed = false
    @Volatile var openError: Throwable? = null

    fun push(step: Step) {
        queue.put(step)
    }

    override fun open() {
        openError?.let { throw it }
        opened = true
    }

    override fun read(buffer: ShortArray): Int {
        if (closed) return -1
        val step = if (blockWhenEmpty) {
            queue.poll(5, TimeUnit.SECONDS) ?: error("ScriptedAudioSource: no step within 5 s")
        } else {
            queue.poll() ?: return -1
        }
        if (closed) return -1
        return when (step) {
            is Step.Samples -> {
                step.samples.copyInto(buffer, 0, 0, minOf(buffer.size, step.samples.size))
                minOf(buffer.size, step.samples.size)
            }
            is Step.Fail -> throw step.error
            Step.End -> -1
        }
    }

    override fun close() {
        closed = true
        queue.put(Step.End)
    }

    companion object {
        fun frame(value: Short, size: Int = 512) = Step.Samples(ShortArray(size) { value })
    }
}
