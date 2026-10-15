package com.xrmech.habla.app.pipeline

import com.xrmech.habla.app.fakes.FakeSpeechRecognizer
import com.xrmech.habla.app.fakes.TestAssets
import com.xrmech.habla.asr.SpeechRecognizer
import com.xrmech.habla.asr.AsrException
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import java.io.File
import java.io.IOException
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import kotlinx.coroutines.async
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

class RecognizerHolderTest {
    @TempDir
    lateinit var dir: File

    private val offline = object : ModelDownloader {
        override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
            throw IOException("offline")
    }

    private fun holder(): RecognizerHolder {
        val assets = TestAssets.shared
        return RecognizerHolder(ModelManager(dir, assets.asrModels, offline), assets, Dispatchers.Unconfined, threads = 1)
    }

    @Test
    fun `should use the profile whisper_cpp threads by default`() {
        val assets = TestAssets.shared
        val h = RecognizerHolder(ModelManager(dir, assets.asrModels, offline), assets, Dispatchers.Unconfined)
        assertEquals(assets.asrProfile.whisperCpp.threads, h.threads)
    }

    @Test
    fun `should throw model missing when the verified file is absent`() = runTest {
        val e = assertThrows<AsrException> { holder().acquire(TestAssets.shared.asrModels.defaultId) }
        assertEquals(AsrException.Kind.MODEL_MISSING, e.kind)
        assertEquals("Speech model missing - download it in Settings", e.message)
    }

    @Test
    fun `should allow release twice with nothing loaded`() {
        val h = holder()
        h.release()
        h.release()
    }

    /** A holder whose model file exists and whose native load waits on [gate]. */
    private class GatedHolder(dir: File, io: kotlinx.coroutines.CoroutineDispatcher) : RecognizerHolder(
        ModelManager(dir, TestAssets.shared.asrModels, object : ModelDownloader {
            override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
                throw IOException("offline")
        }),
        TestAssets.shared,
        io,
        threads = 1,
    ) {
        val gate = CompletableDeferred<Unit>()
        val loaded = FakeSpeechRecognizer()
        private val file = File(dir, "model.bin")

        override fun modelFile(modelId: String): File = file

        /** Like the native load: blocking work that cancellation does not interrupt. */
        override suspend fun load(file: File): SpeechRecognizer {
            withContext(NonCancellable) { gate.await() }
            return loaded
        }
    }

    @Test
    fun `should close a model that finishes loading after release`() = runTest {
        val h = GatedHolder(dir, StandardTestDispatcher(testScheduler))
        val acquiring = async { runCatching { h.acquire("m") } }
        runCurrent()
        h.release() // Stop or a failed start while the model is still loading
        h.gate.complete(Unit)
        val result = acquiring.await()
        assertTrue(result.exceptionOrNull() is CancellationException)
        assertTrue(h.loaded.closed, "a model loaded after release must be closed, not leaked")
    }

    @Test
    fun `should keep a model whose caller was cancelled so release frees it`() = runTest {
        val h = GatedHolder(dir, StandardTestDispatcher(testScheduler))
        val acquiring = async { h.acquire("m") }
        runCurrent()
        acquiring.cancel() // the session scope is cancelled before the load returns
        h.gate.complete(Unit)
        runCurrent()
        assertFalse(h.loaded.closed)
        h.release()
        assertTrue(h.loaded.closed)
    }

    @Test
    fun `should return the loaded model until it is released`() = runTest {
        val h = GatedHolder(dir, StandardTestDispatcher(testScheduler))
        h.gate.complete(Unit)
        val first = h.acquire("m")
        assertSame(first, h.acquire("m"))
        h.release()
        assertTrue(h.loaded.closed)
    }
}
