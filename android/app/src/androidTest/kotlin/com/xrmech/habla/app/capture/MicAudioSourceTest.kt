package com.xrmech.habla.app.capture

import android.Manifest
import android.content.Context
import android.media.AudioManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MicAudioSourceTest {
    @get:Rule
    val mic: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO)

    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val audioManager = context.getSystemService(AudioManager::class.java)

    @Test(timeout = 10_000)
    fun readsA512SampleFrameWithinOneSecond() {
        val source = MicAudioSource(context, audioManager)
        source.open()
        try {
            val start = System.nanoTime()
            val n = source.read(ShortArray(512))
            val elapsedMs = (System.nanoTime() - start) / 1_000_000
            assertEquals(512, n)
            assertTrue("read took $elapsedMs ms", elapsedMs < 1_000)
        } finally {
            source.close()
        }
    }

    @Test(timeout = 10_000)
    fun closeEndsABlockedReadWithEndOfStream() {
        val source = MicAudioSource(context, audioManager)
        source.open()
        val result = AtomicInteger(Int.MIN_VALUE)
        val done = CountDownLatch(1)
        Thread {
            // Keep reading until close makes read return -1.
            while (true) {
                val n = source.read(ShortArray(16_000))
                if (n < 0) {
                    result.set(n)
                    break
                }
            }
            done.countDown()
        }.start()
        Thread.sleep(300)
        source.close()
        assertTrue("read did not return after close", done.await(2, TimeUnit.SECONDS))
        assertEquals(-1, result.get())
        source.close()
    }
}
