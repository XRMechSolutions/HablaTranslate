package com.xrmech.habla.asr

import app.cash.turbine.test
import com.xrmech.habla.asr.fakes.FakeModelDownloader
import com.xrmech.habla.asr.fakes.FakeModelDownloader.Reply
import com.xrmech.habla.shared.AsrModelEntry
import com.xrmech.habla.shared.ModelCatalog
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import kotlin.random.Random
import kotlinx.coroutines.async
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir

class ModelManagerTest {
    @TempDir
    lateinit var filesDir: File

    private val content = Random(42).nextBytes(3 * MIB)
    private val entry = entryFor(content)
    private val catalog = ModelCatalog(null, entry.id, listOf(entry))

    private val models get() = File(filesDir, "models")
    private val final get() = File(models, entry.file)
    private val part get() = File(models, entry.file + ".part")
    private val partJson get() = File(models, entry.file + ".part.json")
    private val marker get() = File(models, entry.file + ".verified")

    private fun TestScope.fake() = FakeModelDownloader(content) { testScheduler.currentTime }

    private fun TestScope.manager(
        http: ModelDownloader,
        cat: ModelCatalog = catalog,
        usable: (File) -> Long = { Long.MAX_VALUE },
    ) = ModelManager(filesDir, cat, http, usable, StandardTestDispatcher(testScheduler))

    private fun writePart(bytes: Int, meta: AsrModelEntry = entry) {
        models.mkdirs()
        part.writeBytes(content.copyOfRange(0, bytes))
        partJson.writeText(
            """{"url":"${meta.url}","revision":"${meta.revision}","sha256":"${meta.sha256}","size_bytes":${meta.sizeBytes}}""",
        )
    }

    private fun assertReady(m: ModelManager) {
        assertEquals(ModelState.Ready(final), m.state.value)
        assertArrayEquals(content, final.readBytes())
        assertEquals(final, m.fileFor(entry.id))
        assertFalse(part.exists())
        assertFalse(partJson.exists())
        assertTrue(marker.isFile)
    }

    @Test
    fun `fresh download emits Downloading from 0, then Verifying, then Ready`() = runTest {
        val http = fake()
        val m = manager(http)
        m.state.test {
            assertEquals(ModelState.Missing, awaitItem())
            val job = launch { m.ensure(entry.id) }
            assertEquals(ModelState.Downloading(0, entry.sizeBytes), awaitItem())
            var item = awaitItem()
            var last: ModelState.Downloading? = null
            while (item is ModelState.Downloading) {
                last = item
                item = awaitItem()
            }
            assertEquals(ModelState.Downloading(entry.sizeBytes, entry.sizeBytes), last)
            assertEquals(ModelState.Verifying, item)
            assertEquals(ModelState.Ready(final), awaitItem())
            job.join()
        }
        assertReady(m)
        assertEquals(listOf(0L), http.opens.map { it.rangeStart })
        assertEquals(entry.url, http.opens.single().url)
        assertEquals("${entry.sha256} ${entry.sizeBytes} ${final.lastModified()}", marker.readText().trim())
    }

    @Test
    fun `resume sends the part length as offset`() = runTest {
        writePart(MIB)
        val http = fake()
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(listOf(MIB.toLong()), http.opens.map { it.rangeStart })
        assertReady(m)
    }

    @Test
    fun `part json for another entry restarts from zero`() = runTest {
        writePart(MIB, meta = entry.copy(url = entry.url.replace("rev", "old")))
        val http = fake()
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(listOf(0L), http.opens.map { it.rangeStart })
        assertReady(m)
    }

    @Test
    fun `200 on resume truncates and writes from the start`() = runTest {
        writePart(MIB)
        val http = fake().script(fake().ok200())
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(listOf(MIB.toLong()), http.opens.map { it.rangeStart })
        assertReady(m)
    }

    @Test
    fun `complete part is verified without downloading again`() = runTest {
        writePart(content.size)
        val http = fake()
        val m = manager(http)
        m.ensure(entry.id)
        assertTrue(http.opens.isEmpty())
        assertReady(m)
    }

    @Test
    fun `416 below full length restarts once from zero`() = runTest {
        writePart(MIB)
        val http = fake().script(fake().status(416))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(listOf(MIB.toLong(), 0L), http.opens.map { it.rangeStart })
        assertReady(m)
    }

    @Test
    fun `404 fails at once without retry`() = runTest {
        val http = fake().script(fake().status(404))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(ModelState.Failed("http_404", false), m.state.value)
        assertEquals(1, http.opens.size)
    }

    @Test
    fun `503 four times fails retryable after 2, 8 and 30 second delays`() = runTest {
        val f = fake()
        val http = f.script(f.status(503), f.status(503), f.status(503), f.status(503))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(ModelState.Failed("http_503", true), m.state.value)
        assertEquals(listOf(0L, 2_000L, 10_000L, 40_000L), http.opens.map { it.atMs })
    }

    @Test
    fun `429 honours Retry-After capped at 60 seconds`() = runTest {
        val f = fake()
        val http = f.script(f.status(429, retryAfterSeconds = 120), f.status(429, retryAfterSeconds = 5))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(listOf(0L, 60_000L, 65_000L), http.opens.map { it.atMs })
        assertReady(m)
    }

    @Test
    fun `IOException four times fails as network`() = runTest {
        val http = fake().script(Reply.ConnectFails, Reply.ConnectFails, Reply.ConnectFails, Reply.ConnectFails)
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(ModelState.Failed("network", true), m.state.value)
        assertEquals(4, http.opens.size)
    }

    @Test
    fun `connection drop mid-body resumes from the bytes written`() = runTest {
        val f = fake()
        val http = f.script(f.ok200(failAfter = 2L * MIB))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(listOf(0L, 2L * MIB), http.opens.map { it.rangeStart })
        assertEquals(2_000L, http.opens[1].atMs)
        assertReady(m)
    }

    @Test
    fun `206 with another start drops the part and restarts from zero`() = runTest {
        writePart(MIB)
        val f = fake()
        val http = f.script(f.partial206(from = MIB.toLong(), start = 2L * MIB))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(listOf(MIB.toLong(), 0L), http.opens.map { it.rangeStart })
        assertReady(m)
    }

    @Test
    fun `size mismatch fails before writing`() = runTest {
        val f = fake()
        val http = f.script(f.ok200(contentLength = content.size + 1L))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(ModelState.Failed("size_mismatch", false), m.state.value)
        assertFalse(part.exists())
    }

    @Test
    fun `wrong Content-Range total is a size mismatch`() = runTest {
        writePart(MIB)
        val f = fake()
        val http = f.script(f.partial206(from = MIB.toLong(), total = content.size + 10L))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(ModelState.Failed("size_mismatch", false), m.state.value)
    }

    @Test
    fun `checksum mismatch deletes the part`() = runTest {
        val corrupt = content.copyOf().also { it[1234] = (it[1234] + 1).toByte() }
        val f = fake()
        val http = f.script(f.ok200(body = corrupt))
        val m = manager(http)
        m.ensure(entry.id)
        assertEquals(ModelState.Failed("checksum_mismatch", true), m.state.value)
        assertFalse(part.exists())
        assertFalse(partJson.exists())
        assertNull(m.fileFor(entry.id))
    }

    @Test
    fun `insufficient storage reports the bytes needed`() = runTest {
        val http = fake()
        val m = manager(http, usable = { 10L })
        m.ensure(entry.id)
        assertEquals(ModelState.Failed("insufficient_storage:${entry.sizeBytes + 64L * MIB}", true), m.state.value)
        assertTrue(http.opens.isEmpty())
    }

    @Test
    fun `final file without marker is re-hashed and becomes Ready`() = runTest {
        models.mkdirs()
        final.writeBytes(content)
        val http = fake()
        val m = manager(http)
        assertNull(m.fileFor(entry.id))
        m.ensure(entry.id)
        assertTrue(http.opens.isEmpty())
        assertReady(m)
    }

    @Test
    fun `catalog sha change invalidates the installed file and downloads the new one`() = runTest {
        manager(fake()).ensure(entry.id)
        val newContent = Random(7).nextBytes(3 * MIB)
        val newEntry = entryFor(newContent)
        val http = FakeModelDownloader(newContent) { testScheduler.currentTime }
        val m = manager(http, ModelCatalog(null, newEntry.id, listOf(newEntry)))
        assertNull(m.fileFor(newEntry.id))
        m.ensure(newEntry.id)
        assertEquals(1, http.opens.size)
        assertArrayEquals(newContent, final.readBytes())
        assertEquals(final, m.fileFor(newEntry.id))
    }

    @Test
    fun `prune removes files of models no longer in the catalog`() = runTest {
        models.mkdirs()
        val stale = listOf("ggml-old.bin", "ggml-old.bin.part", "notes.txt").map { File(models, it).apply { writeText("x") } }
        manager(fake()).ensure(entry.id)
        assertTrue(stale.none { it.exists() })
        assertTrue(final.exists() && marker.exists())
    }

    @Test
    fun `concurrent ensure of the same id downloads once`() = runTest {
        val http = fake()
        val m = manager(http)
        val a = async { m.ensure(entry.id) }
        val b = async { m.ensure(entry.id) }
        a.await()
        b.await()
        assertEquals(1, http.opens.size)
        assertReady(m)
    }

    @Test
    fun `delete during a download cancels it, removes all files and leaves Missing`() = runTest {
        val f = fake()
        val http = f.script(f.ok200(stallAfter = MIB.toLong()))
        val m = manager(http)
        val job = launch { m.ensure(entry.id) }
        http.stalled.await()
        assertTrue(part.exists())
        m.delete(entry.id)
        job.join()
        assertFalse(job.isCancelled, "delete must not cancel the caller of ensure")
        assertEquals(ModelState.Missing, m.state.value)
        assertTrue(listOf(final, part, partJson, marker).none { it.exists() })
    }

    @Test
    fun `cancellation keeps the part for resume and sets Missing`() = runTest {
        val f = fake()
        val http = f.script(f.ok200(stallAfter = MIB.toLong()))
        val m = manager(http)
        val job = launch { m.ensure(entry.id) }
        http.stalled.await()
        job.cancelAndJoin()
        assertTrue(job.isCancelled)
        assertEquals(ModelState.Missing, m.state.value)
        assertEquals(MIB.toLong(), part.length())
        assertTrue(partJson.exists())

        m.ensure(entry.id)
        assertEquals(MIB.toLong(), http.opens.last().rangeStart)
        assertReady(m)
    }

    @Test
    fun `refresh reads disk without network`() = runTest {
        val http = fake()
        val m = manager(http)
        m.refresh(entry.id)
        assertEquals(ModelState.Missing, m.state.value)
        m.ensure(entry.id)
        val opened = http.opens.size

        val fresh = manager(http)
        fresh.refresh(entry.id)
        assertEquals(ModelState.Ready(final), fresh.state.value)
        assertEquals(opened, http.opens.size)
    }

    @Test
    fun `unreadable final file fails as io instead of throwing out of ensure`() = runTest {
        // A directory where the model belongs: exists, matches its own length, cannot be opened for hashing.
        models.mkdirs()
        val blocker = File(models, entry.file).apply { mkdir() }
        val odd = entry.copy(sizeBytes = blocker.length())
        val http = fake()
        val m = manager(http, cat = ModelCatalog(null, odd.id, listOf(odd)))
        m.ensure(odd.id)
        assertEquals(ModelState.Failed("io:FileNotFoundException", true), m.state.value)
        assertTrue(http.opens.isEmpty())
    }

    @Test
    fun `ensure does its file work on the io dispatcher, not the caller's thread`() {
        val ioThread = Executors.newSingleThreadExecutor { r -> Thread(r, "test-io") }
        try {
            val seen = CopyOnWriteArrayList<String>()
            val inner = FakeModelDownloader(content) { 0L }
            val http = object : ModelDownloader {
                override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T {
                    seen += Thread.currentThread().name.substringBefore(" @coroutine")
                    return inner.open(url, rangeStart, block)
                }
            }
            val m = ModelManager(filesDir, catalog, http, { Long.MAX_VALUE }, ioThread.asCoroutineDispatcher())
            runBlocking { m.ensure(entry.id) }
            assertEquals(ModelState.Ready(final), m.state.value)
            assertEquals(listOf("test-io"), seen.toList())
        } finally {
            ioThread.shutdown()
        }
    }

    @Test
    fun `unknown id fails with unknown_model`() = runTest {
        val http = fake()
        val m = manager(http)
        m.ensure("nope")
        assertEquals(ModelState.Failed("unknown_model", false), m.state.value)
        assertNull(m.fileFor("nope"))
        assertTrue(http.opens.isEmpty())
    }

    @Test
    fun `fileFor rejects a file whose length or marker does not match`() = runTest {
        val m = manager(fake())
        m.ensure(entry.id)
        assertEquals(final, m.fileFor(entry.id))
        marker.writeText("${"0".repeat(64)} ${entry.sizeBytes} ${final.lastModified()}")
        assertNull(m.fileFor(entry.id))
    }

    @Test
    fun `Content-Range parsing`() {
        assertEquals(1_048_576L to 264_464_607L, ModelManager.parseContentRange("bytes 1048576-264464606/264464607"))
        assertNull(ModelManager.parseContentRange("bytes 0-10/*"))
        assertNull(ModelManager.parseContentRange("bytes 10-5/100"))
        assertNull(ModelManager.parseContentRange(null))
    }

    private companion object {
        const val MIB = 1024 * 1024

        fun entryFor(bytes: ByteArray) = AsrModelEntry(
            id = "ggml-test-q8_0",
            file = "ggml-test-q8_0.bin",
            base = "test",
            quantization = "q8_0",
            revision = "5359861c739e955e79d9a303bcbc70fb988958b1",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/rev/ggml-test-q8_0.bin",
            sizeBytes = bytes.size.toLong(),
            sha256 = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) },
        )
    }
}
