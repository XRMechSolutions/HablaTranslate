package com.xrmech.habla.app.capture

import com.xrmech.habla.app.fakes.ScriptedAudioSource
import com.xrmech.habla.app.fakes.ScriptedAudioSource.Companion.frame
import com.xrmech.habla.app.fakes.ScriptedAudioSource.Step
import com.xrmech.habla.app.pipeline.PipelineMetrics
import java.time.Duration
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ClosedReceiveChannelException
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout
import org.junit.jupiter.api.assertThrows

@Timeout(10)
class CapturePumpTest {
    private val metrics = PipelineMetrics()

    private suspend fun Channel<ShortArray>.drain(): List<ShortArray> {
        val out = mutableListOf<ShortArray>()
        withTimeout(5_000) { for (f in this@drain) out += f }
        return out
    }

    @Test
    fun `should close the channel when the source reaches end of stream`() = runBlocking {
        val source = ScriptedAudioSource(listOf(frame(1), frame(2), Step.Samples(ShortArray(100) { 3 }), Step.End))
        val frames = Channel<ShortArray>(256)
        CapturePump(source, frames, metrics).start()
        val got = frames.drain()
        assertEquals(listOf(512, 512, 100), got.map { it.size })
        assertEquals(3, got[2][0].toInt())
    }

    @Test
    fun `should hand out a fresh buffer per frame`() = runBlocking {
        val source = ScriptedAudioSource(listOf(frame(1), frame(2), Step.End))
        val frames = Channel<ShortArray>(256)
        CapturePump(source, frames, metrics).start()
        val got = frames.drain()
        assertEquals(1, got[0][0].toInt())
        assertEquals(2, got[1][0].toInt())
    }

    @Test
    fun `should close the channel with the source exception as cause`() = runBlocking {
        val failure = AudioSourceException("dead", AudioSourceException.Kind.DEAD, code = -6)
        val source = ScriptedAudioSource(listOf(frame(1), Step.Fail(failure)))
        val frames = Channel<ShortArray>(256)
        CapturePump(source, frames, metrics).start()
        frames.receive()
        val thrown = assertThrows<AudioSourceException> { runBlocking { withTimeout(5_000) { frames.receive() } } }
        assertSame(failure, thrown)
    }

    @Test
    fun `should count dropped frames when the channel is full`() = runBlocking {
        val source = ScriptedAudioSource(List(5) { frame(1) } + Step.End)
        val frames = Channel<ShortArray>(2)
        val pump = CapturePump(source, frames, metrics)
        pump.start()
        withTimeout(5_000) { while (!frames.isClosedForSend) kotlinx.coroutines.delay(5) }
        assertEquals(3, metrics.framesDropped.get())
        assertEquals(2, frames.drain().size)
    }

    @Test
    fun `should set silenced after 62 zero frames and clear it on a non-zero frame`() = runBlocking {
        val source = ScriptedAudioSource(blockWhenEmpty = true)
        val frames = Channel<ShortArray>(Channel.UNLIMITED)
        val pump = CapturePump(source, frames, metrics)
        pump.start()
        repeat(61) { source.push(frame(0)) }
        repeat(61) { frames.receive() }
        assertFalse(pump.zeroSilenced.value)
        source.push(frame(0))
        frames.receive()
        assertTrue(pump.zeroSilenced.value)
        source.push(frame(7))
        frames.receive()
        assertFalse(pump.zeroSilenced.value)
        source.close()
        pump.stopAndJoin()
    }

    @Test
    fun `should refuse a second start`() {
        val source = ScriptedAudioSource(listOf(Step.End))
        val pump = CapturePump(source, Channel(1), metrics)
        pump.start()
        assertThrows<IllegalStateException> { pump.start() }
        pump.stopAndJoin()
    }

    @Test
    fun `should refuse a source that is not 16 kHz`() {
        val source = ScriptedAudioSource(sampleRate = 8_000)
        assertThrows<IllegalArgumentException> { CapturePump(source, Channel(1), metrics).start() }
    }

    @Test
    fun `should end within the timeout when the source is closed while blocked`() = runBlocking {
        val source = ScriptedAudioSource(blockWhenEmpty = true)
        val frames = Channel<ShortArray>(16)
        val pump = CapturePump(source, frames, metrics)
        pump.start()
        val start = System.nanoTime()
        source.close()
        pump.stopAndJoin(1_000)
        assertTrue(Duration.ofNanos(System.nanoTime() - start) < Duration.ofSeconds(1))
        assertThrows<ClosedReceiveChannelException> { runBlocking { frames.receive() } }
        Unit
    }
}
