package com.xrmech.habla.app.pipeline

import com.xrmech.habla.audio.SpeechSegment
import com.xrmech.habla.shared.MergeProfile
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class SegmentMergerTest {
    private val out = mutableListOf<MergedSegment>()
    private val merger = SegmentMerger(MergeProfile(gapSeconds = 1.2, maxPendingSeconds = 60.0), { out += it })

    private fun seg(id: Int, seconds: Double, endSec: Double, rms: Double = 100.0, prob: Double = 0.5, value: Short = id.toShort()): SpeechSegment {
        val n = (seconds * 16_000).toInt()
        return SpeechSegment(id, ShortArray(n) { value }, n / 16_000.0, rms, prob, ((endSec - seconds) * 16_000).toLong(), (endSec * 16_000).toLong())
    }

    @Test
    fun `should merge a segment ending 0_8 s after the previous one`() {
        merger.onSegment(seg(1, 1.0, endSec = 2.0, rms = 100.0, prob = 0.4))
        merger.onAudio(2.5.samples())
        merger.onSegment(seg(2, 3.0, endSec = 2.8, rms = 300.0, prob = 0.8))
        merger.onAudio(3.9.samples())
        assertTrue(out.isEmpty())
        merger.onAudio(4.0.samples())
        val m = out.single()
        assertEquals(1, m.seq)
        assertEquals(listOf(1, 2), m.rawIds)
        assertEquals(64_000, m.pcm.size)
        assertEquals(1.toShort(), m.pcm[0])
        assertEquals(2.toShort(), m.pcm[16_000])
        assertEquals(250.0, m.rms, 1e-9)
        assertEquals(0.7, m.vadAvgProb, 1e-9)
        assertEquals(4.0, m.durationSec, 1e-9)
    }

    @Test
    fun `should keep segments apart when the second ends 1_3 s later`() {
        merger.onSegment(seg(1, 1.0, endSec = 2.0))
        merger.onAudio(3.2.samples())
        merger.onSegment(seg(2, 1.0, endSec = 3.3))
        merger.onAudio(4.5.samples())
        assertEquals(listOf(listOf(1), listOf(2)), out.map { it.rawIds })
        assertEquals(listOf(1, 2), out.map { it.seq })
    }

    @Test
    fun `should flush at once when 60 s are pending`() {
        merger.onSegment(seg(1, 30.0, endSec = 30.0))
        merger.onSegment(seg(2, 31.0, endSec = 30.5))
        assertEquals(1, out.size)
        assertEquals(61.0, out.single().durationSec, 1e-9)
    }

    @Test
    fun `should ignore empty pcm and do nothing on flush when empty`() {
        merger.onSegment(SpeechSegment(1, ShortArray(0), 0.0, 0.0, 0.0, 0, 100))
        merger.flushNow()
        merger.onAudio(1_000_000)
        assertTrue(out.isEmpty())
    }

    @Test
    fun `should flush pending on flushNow and count seq per merged segment`() {
        merger.onSegment(seg(1, 1.0, endSec = 1.0))
        merger.flushNow()
        merger.onSegment(seg(2, 1.0, endSec = 5.0, value = 9))
        merger.flushNow()
        assertEquals(listOf(1, 2), out.map { it.seq })
        assertArrayEquals(ShortArray(16_000) { 9 }, out[1].pcm)
        assertEquals(80_000L, out[1].endSample)
    }

    private fun Double.samples(): Long = (this * 16_000).toLong()
}

class FinalQueueTest {
    private fun item(seq: Int) = MergedSegment(seq, ShortArray(1), 0.0, 0.0, 0.0, listOf(seq), 0)

    @Test
    fun `should drop the oldest waiting item when a sixth arrives`() {
        val q = FinalQueue()
        q.offer(item(1))
        assertEquals(1, q.take()!!.seq)
        val dropped = (2..7).mapNotNull { q.offer(item(it)) }
        assertEquals(listOf(2), dropped.map { it.seq })
        assertEquals(1, q.finalsDropped)
        assertEquals(6, q.depth)
        assertEquals(6, q.peakDepth)
        q.done()
        assertEquals((3..7).toList(), generateSequence { q.take()?.also { q.done() } }.map { it.seq }.toList())
        assertTrue(q.isIdle)
    }

    @Test
    fun `should return null from take when empty`() {
        val q = FinalQueue()
        assertEquals(null, q.take())
        assertEquals(0, q.depth)
    }
}
