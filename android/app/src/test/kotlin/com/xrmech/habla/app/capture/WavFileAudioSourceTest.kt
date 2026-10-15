package com.xrmech.habla.app.capture

import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class WavFileAudioSourceTest {
    private val clip = File(System.getProperty("habla.sharedDir"), "testdata/clips/mixed_es_en_es.wav")

    private fun wav(channels: Int = 1, rate: Int = 16_000, bits: Int = 16, format: Int = 1, samples: ShortArray, dataSizeOverride: Int? = null): ByteArray {
        val data = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        samples.forEach { data.putShort(it) }
        val b = ByteBuffer.allocate(44 + samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + samples.size * 2).put("WAVE".toByteArray())
        b.put("fmt ".toByteArray()).putInt(16).putShort(format.toShort()).putShort(channels.toShort())
            .putInt(rate).putInt(rate * channels * bits / 8).putShort((channels * bits / 8).toShort()).putShort(bits.toShort())
        b.put("data".toByteArray()).putInt(dataSizeOverride ?: (samples.size * 2)).put(data.array())
        return b.array()
    }

    @Test
    fun `should return every sample of the mixed clip then end of stream`() {
        val source = WavFileAudioSource(clip.inputStream(), paced = false)
        source.open()
        val buf = ShortArray(512)
        var total = 0L
        while (true) {
            val n = source.read(buf)
            if (n < 0) break
            total += n
        }
        assertEquals(665_600L, total)
        assertEquals(-1, source.read(buf))
        source.close()
    }

    @Test
    fun `should decode little endian samples`() {
        val source = WavFileAudioSource(ByteArrayInputStream(wav(samples = shortArrayOf(1, -2, 300))), paced = false)
        source.open()
        val buf = ShortArray(8)
        assertEquals(3, source.read(buf))
        assertEquals(listOf<Short>(1, -2, 300), buf.take(3))
    }

    @Test
    fun `should name the sample rate field for an 8 kHz header`() {
        val source = WavFileAudioSource(ByteArrayInputStream(wav(rate = 8_000, samples = ShortArray(4))), paced = false)
        val e = assertThrows<AudioSourceException> { source.open() }
        assertEquals(AudioSourceException.Kind.IO, e.kind)
        assertTrue(e.message!!.contains("sample rate"))
    }

    @Test
    fun `should name the channels field for a stereo header`() {
        val source = WavFileAudioSource(ByteArrayInputStream(wav(channels = 2, samples = ShortArray(4))), paced = false)
        val e = assertThrows<AudioSourceException> { source.open() }
        assertTrue(e.message!!.contains("channels"))
    }

    @Test
    fun `should reject a non PCM format`() {
        val source = WavFileAudioSource(ByteArrayInputStream(wav(format = 3, samples = ShortArray(4))), paced = false)
        assertTrue(assertThrows<AudioSourceException> { source.open() }.message!!.contains("format"))
    }

    @Test
    fun `should return what exists when the data chunk is truncated`() {
        val bytes = wav(samples = ShortArray(10) { 5 }, dataSizeOverride = 2_000)
        val source = WavFileAudioSource(ByteArrayInputStream(bytes), paced = false)
        source.open()
        val buf = ShortArray(512)
        assertEquals(10, source.read(buf))
        assertEquals(-1, source.read(buf))
    }

    @Test
    fun `should return end of stream after close and allow closing twice`() {
        val source = WavFileAudioSource(ByteArrayInputStream(wav(samples = ShortArray(100))), paced = false)
        source.open()
        source.close()
        source.close()
        assertEquals(-1, source.read(ShortArray(16)))
    }

    @Test
    fun `should release 16000 samples per second when paced`() {
        var now = 0L
        val sleeps = mutableListOf<Long>()
        val source = WavFileAudioSource(
            ByteArrayInputStream(wav(samples = ShortArray(16_000))),
            paced = true,
            clock = { now },
            sleep = { ms -> sleeps += ms; now += ms * 1_000_000 },
        )
        source.open()
        val buf = ShortArray(512)
        var total = 0
        while (true) {
            val n = source.read(buf)
            if (n < 0) break
            total += n
        }
        assertEquals(16_000, total)
        // 16 000 samples must take one second of the fake clock (rounded up per read).
        assertTrue(now in 1_000_000_000L..1_032_000_000L, "elapsed ${now / 1_000_000} ms")
        assertEquals(32L, sleeps.first())
    }
}
