package com.xrmech.habla.audio

import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.FileAssetSource
import com.xrmech.habla.shared.SharedAssets
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test

class VadModelTest {
    private val profile: AsrProfile = TestProfiles.shipped

    @Test
    fun `bundled model has the pinned size and hash`() {
        val bytes = SileroVad.bundledModelBytes()
        assertEquals(SileroVad.MODEL_SIZE, bytes.size)
        assertEquals(SileroVad.MODEL_SHA256, GoldenFixtures.sha256(bytes))
    }

    @Test
    fun `digital silence is not speech`() {
        SileroVad(SileroVad.bundledModelBytes()).use { vad ->
            val p = vad.speechProbability(FloatArray(512))
            assertTrue(p in 0f..0.05f, "p=$p")
        }
    }

    @Test
    fun `corrupt model bytes throw VadException`() {
        assertFailsWith<VadException> { SileroVad(ByteArray(100)) }
    }

    @Test
    fun `bad constructor arguments throw IllegalArgumentException`() {
        assertFailsWith<IllegalArgumentException> { SileroVad(ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { SileroVad(SileroVad.bundledModelBytes(), threads = 0) }
    }

    @Test
    fun `wrong frame size throws IllegalArgumentException`() {
        SileroVad(SileroVad.bundledModelBytes()).use { vad ->
            assertFailsWith<IllegalArgumentException> { vad.speechProbability(FloatArray(511)) }
        }
        assertFailsWith<IllegalArgumentException> { EnergyVad(profile).speechProbability(FloatArray(513)) }
    }

    @Test
    fun `close is idempotent and later calls throw IllegalStateException`() {
        val vad = SileroVad(SileroVad.bundledModelBytes())
        vad.close()
        vad.close()
        assertFailsWith<IllegalStateException> { vad.speechProbability(FloatArray(512)) }
        assertFailsWith<IllegalStateException> { vad.reset() }
    }

    @Test
    fun `reset restores the first-call output bit-exactly`() {
        val frame = FloatArray(512) { (kotlin.math.sin(it * 0.07) * 0.3).toFloat() }
        SileroVad(SileroVad.bundledModelBytes()).use { vad ->
            val first = vad.speechProbability(frame)
            val second = vad.speechProbability(frame)
            vad.reset()
            val again = vad.speechProbability(frame)
            assertEquals(first.toRawBits(), again.toRawBits())
            assertTrue(first in 0f..1f && second in 0f..1f)
        }
    }

    @Test
    fun `energy vad follows the server formula`() {
        val vad = EnergyVad(profile)
        assertEquals(500.0, vad.energyThreshold)
        assertEquals(0.0, vad.lastEnergy)
        val frame = ShortArray(512) { 700 }.toFloatSamples()
        val p = vad.speechProbability(frame)
        assertEquals(0.7f, p)
        assertEquals(700.0, vad.lastEnergy)
        vad.reset()
        assertEquals(0.0, vad.lastEnergy)
        assertEquals(500.0, vad.energyThreshold)
    }

    @Test
    fun `energy vad rejects invalid thresholds`() {
        val vad = EnergyVad(profile)
        for (bad in listOf(0.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY)) {
            assertFailsWith<IllegalArgumentException>("threshold $bad") { vad.energyThreshold = bad }
        }
        assertEquals(500.0, vad.energyThreshold)
    }

    @Test
    fun `toFloatSamples divides by 32768`() {
        assertContentEquals(floatArrayOf(-1f, 0f, 0.5f), shortArrayOf(-32768, 0, 16384).toFloatSamples())
        assertContentEquals(FloatArray(0), ShortArray(0).toFloatSamples())
    }
}

/** The shipped `shared/asr_profile.json`, loaded through `:shared-assets` like the app does. */
internal object TestProfiles {
    val shipped: AsrProfile by lazy { SharedAssets(FileAssetSource.fromSystemProperty()).asrProfile }
}
