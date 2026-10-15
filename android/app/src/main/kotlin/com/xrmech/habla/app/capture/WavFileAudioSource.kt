package com.xrmech.habla.app.capture

import java.io.BufferedInputStream
import java.io.IOException
import java.io.InputStream
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Reads a RIFF/WAVE file (PCM format 1, mono, 16 000 Hz, 16-bit) as an [AudioSource]. With
 * [paced] the samples leave at 16 000 per second measured by [clock] (nanoseconds); [sleep] waits
 * (milliseconds) and is injectable so tests advance a fake clock instead of sleeping.
 */
class WavFileAudioSource(
    input: InputStream,
    private val paced: Boolean,
    private val clock: () -> Long = System::nanoTime,
    private val sleep: (Long) -> Unit = Thread::sleep,
) : AudioSource {
    override val sampleRate: Int = SAMPLE_RATE

    private val _condition = MutableStateFlow<AudioCondition>(AudioCondition.Normal)
    override val condition: StateFlow<AudioCondition> = _condition

    private val stream = BufferedInputStream(input)
    @Volatile private var closed = false
    private var opened = false
    private var dataRemaining = 0L
    private var released = 0L
    private var startNanos = 0L

    override fun open() {
        check(!opened) { "already opened" }
        opened = true
        try {
            parseHeader()
        } catch (e: IOException) {
            throw AudioSourceException("WAV unreadable: ${e.message}", AudioSourceException.Kind.IO, cause = e)
        }
        startNanos = clock()
    }

    override fun read(buffer: ShortArray): Int {
        if (closed) return -1
        check(opened) { "open() first" }
        if (dataRemaining < 2) return -1
        val want = minOf(buffer.size.toLong(), dataRemaining / 2).toInt()
        val bytes = ByteArray(want * 2)
        val got = try {
            readFully(bytes)
        } catch (e: IOException) {
            if (closed) return -1
            throw AudioSourceException("WAV read failed: ${e.message}", AudioSourceException.Kind.IO, cause = e)
        }
        val samples = got / 2
        if (samples == 0) {
            dataRemaining = 0
            return -1
        }
        // A truncated data chunk ends the stream after what exists.
        dataRemaining = if (got < bytes.size) 0 else dataRemaining - got
        for (i in 0 until samples) {
            buffer[i] = ((bytes[2 * i].toInt() and 0xFF) or (bytes[2 * i + 1].toInt() shl 8)).toShort()
        }
        if (paced) pace(samples)
        if (closed) return -1
        return samples
    }

    override fun close() {
        if (closed) return
        closed = true
        runCatching { stream.close() }
    }

    private fun pace(samples: Int) {
        released += samples
        val dueNanos = startNanos + released * 1_000_000_000L / SAMPLE_RATE
        val waitNanos = dueNanos - clock()
        if (waitNanos > 0) sleep((waitNanos + 999_999) / 1_000_000)
    }

    private fun parseHeader() {
        val riff = readBytes(12)
        if (String(riff, 0, 4, Charsets.US_ASCII) != "RIFF" || String(riff, 8, 4, Charsets.US_ASCII) != "WAVE") {
            throw io("RIFF/WAVE header missing")
        }
        var fmtSeen = false
        while (true) {
            val header = readBytes(8)
            val id = String(header, 0, 4, Charsets.US_ASCII)
            val size = le32(header, 4).toLong() and 0xFFFFFFFFL
            when (id) {
                "fmt " -> {
                    if (size < 16) throw io("fmt chunk too short ($size bytes)")
                    val fmt = readBytes(size.toInt())
                    val format = le16(fmt, 0)
                    val channels = le16(fmt, 2)
                    val rate = le32(fmt, 4)
                    val bits = le16(fmt, 14)
                    if (format != 1) throw io("WAV format must be 1 (PCM), got $format")
                    if (channels != 1) throw io("WAV channels must be 1, got $channels")
                    if (rate != SAMPLE_RATE) throw io("WAV sample rate must be $SAMPLE_RATE, got $rate")
                    if (bits != 16) throw io("WAV bits per sample must be 16, got $bits")
                    if (size % 2 == 1L) skip(1)
                    fmtSeen = true
                }
                "data" -> {
                    if (!fmtSeen) throw io("WAV data chunk before fmt chunk")
                    dataRemaining = size
                    return
                }
                else -> skip(size + (size and 1))
            }
        }
    }

    private fun readFully(bytes: ByteArray): Int {
        var off = 0
        while (off < bytes.size) {
            val n = stream.read(bytes, off, bytes.size - off)
            if (n < 0) break
            off += n
        }
        return off
    }

    private fun readBytes(n: Int): ByteArray {
        val b = ByteArray(n)
        if (readFully(b) < n) throw io("WAV header truncated")
        return b
    }

    private fun skip(n: Long) {
        var left = n
        while (left > 0) {
            val s = stream.skip(left)
            if (s <= 0) {
                if (stream.read() < 0) throw io("WAV header truncated")
                left--
            } else {
                left -= s
            }
        }
    }

    private fun io(msg: String) = AudioSourceException(msg, AudioSourceException.Kind.IO)

    private fun le16(b: ByteArray, o: Int) = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8)

    private fun le32(b: ByteArray, o: Int) = le16(b, o) or (le16(b, o + 2) shl 16)

    private companion object {
        const val SAMPLE_RATE = 16_000
    }
}
