package com.xrmech.habla.data

import java.security.GeneralSecurityException
import javax.crypto.Cipher
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** A secrets blob that is not a well-formed version-1 envelope. Treated as corruption (D031). */
internal class EnvelopeFormatException(message: String) : GeneralSecurityException(message)

/**
 * Secret blob format version 1: `[0x01][ivLen][iv][ciphertext || 16-byte GCM tag]`, AES-GCM with
 * the caller's AAD (the Keystore alias), so a blob is bound to the key it was sealed for.
 */
internal object SecretEnvelope {
    const val VERSION: Byte = 1

    private const val TRANSFORMATION = "AES/GCM/NoPadding"
    private const val TAG_BITS = 128
    private const val TAG_BYTES = TAG_BITS / 8
    private const val HEADER_BYTES = 2
    private const val MIN_IV = 12
    private const val MAX_IV = 16

    /** UTF-8 of [DataFiles.KEY_ALIAS], the AAD every API-key blob is sealed with. */
    val keyAliasAad: ByteArray get() = DataFiles.KEY_ALIAS.toByteArray(Charsets.UTF_8)

    fun seal(key: SecretKey, plaintext: ByteArray, aad: ByteArray): ByteArray {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        // No caller IV: the Android Keystore rejects one, and SunJCE generates a random IV itself.
        cipher.init(Cipher.ENCRYPT_MODE, key)
        cipher.updateAAD(aad)
        val body = cipher.doFinal(plaintext)
        val iv = cipher.iv
        val blob = ByteArray(HEADER_BYTES + iv.size + body.size)
        blob[0] = VERSION
        blob[1] = iv.size.toByte()
        iv.copyInto(blob, HEADER_BYTES)
        body.copyInto(blob, HEADER_BYTES + iv.size)
        return blob
    }

    /**
     * @throws EnvelopeFormatException when the blob is truncated, has an unknown version or an IV
     *   length outside 12..16.
     * @throws GeneralSecurityException (`AEADBadTagException`) on a wrong key, wrong AAD or a
     *   modified blob.
     */
    fun open(key: SecretKey, blob: ByteArray, aad: ByteArray): ByteArray {
        if (blob.size < HEADER_BYTES + MIN_IV + TAG_BYTES + 1) {
            throw EnvelopeFormatException("envelope too short")
        }
        if (blob[0] != VERSION) throw EnvelopeFormatException("unknown envelope version")
        val ivLen = blob[1].toInt() and 0xFF
        if (ivLen !in MIN_IV..MAX_IV) throw EnvelopeFormatException("bad iv length")
        if (blob.size < HEADER_BYTES + ivLen + TAG_BYTES + 1) {
            throw EnvelopeFormatException("envelope too short for iv")
        }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(TAG_BITS, blob, HEADER_BYTES, ivLen))
        cipher.updateAAD(aad)
        return cipher.doFinal(blob, HEADER_BYTES + ivLen, blob.size - HEADER_BYTES - ivLen)
    }
}
