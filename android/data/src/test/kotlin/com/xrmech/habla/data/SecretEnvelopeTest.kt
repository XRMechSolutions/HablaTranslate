package com.xrmech.habla.data

import javax.crypto.AEADBadTagException
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class SecretEnvelopeTest {
    private val key: SecretKey = newKey()
    private val aad = SecretEnvelope.keyAliasAad
    private val plaintext = "fake-test-key-0001".toByteArray()

    private fun newKey(): SecretKey = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()

    @Test
    fun `round trip`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        assertArrayEquals(plaintext, SecretEnvelope.open(key, blob, aad))
    }

    @Test
    fun `layout is version, iv length, iv, ciphertext and tag`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        assertEquals(1, blob[0].toInt())
        val ivLen = blob[1].toInt()
        assertEquals(12, ivLen)
        assertEquals(2 + ivLen + plaintext.size + 16, blob.size)
    }

    @Test
    fun `two seals use different ivs`() {
        val a = SecretEnvelope.seal(key, plaintext, aad)
        val b = SecretEnvelope.seal(key, plaintext, aad)
        assertEquals(false, a.copyOfRange(2, 14).contentEquals(b.copyOfRange(2, 14)))
    }

    @Test
    fun `aad is the key alias`() {
        assertArrayEquals("habla_api_key_v1".toByteArray(Charsets.UTF_8), aad)
    }

    @Test
    fun `wrong aad fails the tag`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        assertThrows<AEADBadTagException> { SecretEnvelope.open(key, blob, "other".toByteArray()) }
    }

    @Test
    fun `wrong key fails the tag`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        assertThrows<AEADBadTagException> { SecretEnvelope.open(newKey(), blob, aad) }
    }

    @Test
    fun `flipped bit fails the tag`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        blob[blob.size - 3] = (blob[blob.size - 3].toInt() xor 0x01).toByte()
        assertThrows<AEADBadTagException> { SecretEnvelope.open(key, blob, aad) }
    }

    @Test
    fun `truncated blobs are format errors`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        assertThrows<EnvelopeFormatException> { SecretEnvelope.open(key, blob.copyOf(30), aad) }
        assertThrows<EnvelopeFormatException> { SecretEnvelope.open(key, ByteArray(0), aad) }
    }

    @Test
    fun `iv length longer than the blob allows is a format error`() {
        val blob = ByteArray(2 + 12 + 16 + 1).also {
            it[0] = 1
            it[1] = 16
        }
        assertThrows<EnvelopeFormatException> { SecretEnvelope.open(key, blob, aad) }
    }

    @Test
    fun `version 2 is a format error`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        blob[0] = 2
        assertThrows<EnvelopeFormatException> { SecretEnvelope.open(key, blob, aad) }
    }

    @Test
    fun `iv length 11, 17 and a high byte are format errors`() {
        val blob = SecretEnvelope.seal(key, plaintext, aad)
        for (len in listOf(11, 17, 0xF0)) {
            val bad = blob.copyOf().also { it[1] = len.toByte() }
            assertThrows<EnvelopeFormatException> { SecretEnvelope.open(key, bad, aad) }
        }
    }
}
