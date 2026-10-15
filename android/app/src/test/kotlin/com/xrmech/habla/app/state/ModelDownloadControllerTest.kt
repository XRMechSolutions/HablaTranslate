package com.xrmech.habla.app.state

import com.xrmech.habla.app.fakes.TestAssets
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.ModelState
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout
import org.junit.jupiter.api.io.TempDir

@Timeout(20)
class ModelDownloadControllerTest {
    @TempDir
    lateinit var dir: File

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val entry = TestAssets.shared.asrModels.default

    /** Sends 8 KiB of a 200 response, then stalls until cancelled. */
    private inner class StallingDownloader : ModelDownloader {
        val opens = AtomicInteger()
        val streaming = CompletableDeferred<Unit>()

        override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T {
            opens.incrementAndGet()
            var sent = false
            return block(
                DownloadResponse(200, null, entry.sizeBytes, null) { buf ->
                    if (!sent) {
                        sent = true
                        val n = minOf(buf.size, 8_192)
                        buf.fill(1, 0, n)
                        n
                    } else {
                        streaming.complete(Unit)
                        awaitCancellation()
                    }
                },
            )
        }
    }

    @AfterEach
    fun tearDown() {
        scope.cancel()
    }

    @Test
    fun `should ignore a second download while one runs`() = runBlocking {
        val http = StallingDownloader()
        val controller = ModelDownloadController(ModelManager(dir, TestAssets.shared.asrModels, http), scope)
        controller.download(entry.id)
        withTimeout(10_000) { http.streaming.await() }
        controller.download(entry.id)
        delay(200)
        assertEquals(1, http.opens.get())
        assertTrue(controller.running.value)
        controller.cancel()
    }

    @Test
    fun `should cancel the job and keep the partial file`() = runBlocking {
        val http = StallingDownloader()
        val models = ModelManager(dir, TestAssets.shared.asrModels, http)
        val controller = ModelDownloadController(models, scope)
        controller.download(entry.id)
        withTimeout(10_000) { http.streaming.await() }
        controller.cancel()
        withTimeout(10_000) { controller.running.first { !it } }
        val part = File(dir, "models/${entry.file}.part")
        assertTrue(part.isFile, "part file kept for resume")
        assertEquals(8_192L, part.length())
        assertEquals(ModelState.Missing, models.state.value)
    }

    @Test
    fun `should delete the partial file on delete`() = runBlocking {
        val http = StallingDownloader()
        val controller = ModelDownloadController(ModelManager(dir, TestAssets.shared.asrModels, http), scope)
        controller.download(entry.id)
        withTimeout(10_000) { http.streaming.await() }
        controller.delete(entry.id)
        assertTrue(!File(dir, "models/${entry.file}.part").exists())
    }
}
