package com.xrmech.habla.data

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Writes 16 kHz mono PCM16 WAVs with the canonical 44-byte RIFF header (the same header Python's
 * `wave` module writes on the server). Samples go out little-endian through one reusable 64 KiB
 * buffer, so a long segment is never copied into a second full-size array.
 */
internal class WavWriter {
    // 64 KiB: 32 768 samples (about 2 s of audio) per write; one allocation per writer.
    private val buffer: ByteBuffer = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN)

    /**
     * Writes [pcm] to [file] and fsyncs it before closing, so a later rename publishes a complete
     * file. Synchronized because the buffer is shared by every call on this writer.
     */
    @Synchronized
    fun write(file: File, pcm: ShortArray) {
        FileOutputStream(file).use { out ->
            buffer.clear()
            putHeader(buffer, pcm.size * BYTES_PER_SAMPLE)
            out.write(buffer.array(), 0, buffer.position())
            var i = 0
            while (i < pcm.size) {
                buffer.clear()
                val n = minOf(pcm.size - i, BUFFER_BYTES / BYTES_PER_SAMPLE)
                for (k in i until i + n) buffer.putShort(pcm[k])
                out.write(buffer.array(), 0, buffer.position())
                i += n
            }
            out.flush()
            out.fd.sync()
        }
    }

    private fun putHeader(b: ByteBuffer, dataBytes: Int) {
        b.put("RIFF".toByteArray(Charsets.US_ASCII))
        b.putInt(36 + dataBytes)
        b.put("WAVE".toByteArray(Charsets.US_ASCII))
        b.put("fmt ".toByteArray(Charsets.US_ASCII))
        b.putInt(16)
        b.putShort(1)
        b.putShort(CHANNELS.toShort())
        b.putInt(SAMPLE_RATE)
        b.putInt(SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE)
        b.putShort((CHANNELS * BYTES_PER_SAMPLE).toShort())
        b.putShort(BITS_PER_SAMPLE.toShort())
        b.put("data".toByteArray(Charsets.US_ASCII))
        b.putInt(dataBytes)
    }

    companion object {
        const val HEADER_BYTES = 44
        const val SAMPLE_RATE = 16_000
        const val CHANNELS = 1
        const val BITS_PER_SAMPLE = 16
        const val BYTES_PER_SAMPLE = 2
        private const val BUFFER_BYTES = 64 * 1024
    }
}
