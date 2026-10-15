package com.xrmech.habla.asr

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.request.HttpRequestData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.utils.io.ByteReadChannel
import io.ktor.utils.io.writeFully
import io.ktor.utils.io.writer
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class KtorModelDownloaderTest {

    private val requests = mutableListOf<HttpRequestData>()

    private fun downloader(handler: suspend MockRequestHandleScope.(HttpRequestData) -> io.ktor.client.request.HttpResponseData) =
        KtorModelDownloader(
            MockEngine { request ->
                synchronized(requests) { requests += request }
                handler(request)
            },
        )

    private suspend fun DownloadBody.readAll(): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        val buf = ByteArray(8192)
        while (true) {
            val n = read(buf)
            if (n < 0) break
            out.write(buf, 0, n)
        }
        return out.toByteArray()
    }

    @Test
    fun `no Range header when offset is zero`() = runBlocking {
        downloader { respond(ByteArray(3), HttpStatusCode.OK) }.use { d ->
            d.open("https://example.test/m.bin", 0) { it.status }
        }
        assertNull(requests.single().headers[HttpHeaders.Range])
    }

    @Test
    fun `Range header sent when offset is positive`() = runBlocking {
        downloader { respond(ByteArray(3), HttpStatusCode.PartialContent) }.use { d ->
            d.open("https://example.test/m.bin", 1_048_576) { it.status }
        }
        assertEquals("bytes=1048576-", requests.single().headers[HttpHeaders.Range])
    }

    @Test
    fun `302 is followed with Range preserved and redirect target not exposed`() = runBlocking {
        val body = "abc".toByteArray()
        val status = downloader { request ->
            if (request.url.host == "huggingface.co") {
                respond(
                    ByteArray(0),
                    HttpStatusCode.Found,
                    headersOf(HttpHeaders.Location, "https://cdn.example.test/signed?Expires=1"),
                )
            } else {
                respond(
                    body,
                    HttpStatusCode.PartialContent,
                    headersOf(HttpHeaders.ContentRange, "bytes 10-12/13"),
                )
            }
        }.use { d ->
            d.open("https://huggingface.co/x/resolve/rev/m.bin", 10) { r ->
                assertEquals("bytes 10-12/13", r.contentRange)
                assertTrue(body.contentEquals(r.body.readAll()))
                r.status
            }
        }
        assertEquals(206, status)
        assertEquals(2, requests.size)
        assertEquals("cdn.example.test", requests[1].url.host)
        assertEquals("bytes=10-", requests[0].headers[HttpHeaders.Range])
        assertEquals("bytes=10-", requests[1].headers[HttpHeaders.Range])
    }

    @Test
    fun `body larger than the heap budget is streamed in chunks`() = runBlocking {
        val total = 64L * 1024 * 1024
        val produced = AtomicLong()
        val chunk = ByteArray(64 * 1024) { it.toByte() }
        var consumed = 0L
        var maxAhead = 0L
        downloader {
            val channel = writer(Dispatchers.IO) {
                var sent = 0L
                while (sent < total) {
                    channel.writeFully(chunk)
                    sent += chunk.size
                    produced.set(sent)
                }
            }.channel
            respond(channel, HttpStatusCode.OK, headersOf(HttpHeaders.ContentLength, total.toString()))
        }.use { d ->
            d.open("https://example.test/big.bin", 0) { r ->
                assertEquals(total, r.contentLength)
                val buf = ByteArray(64 * 1024)
                while (true) {
                    val n = r.body.read(buf)
                    if (n < 0) break
                    assertTrue(n <= buf.size)
                    consumed += n
                    maxAhead = maxOf(maxAhead, produced.get() - consumed)
                }
            }
        }
        assertEquals(total, consumed)
        // The producer can only run ahead by the channel buffer; a buffered body would be 64 MiB ahead.
        assertTrue(maxAhead < HEAP_BUDGET_BYTES, "producer ran $maxAhead bytes ahead of the reader")
    }

    @Test
    fun `status and headers are mapped`() = runBlocking {
        val response = downloader {
            respond(
                ByteArray(0),
                HttpStatusCode.TooManyRequests,
                headersOf(
                    HttpHeaders.RetryAfter to listOf("17"),
                    HttpHeaders.ContentLength to listOf("0"),
                ),
            )
        }.use { d -> d.open("https://example.test/m.bin", 0) { Triple(it.status, it.retryAfterSeconds, it.contentRange) } }
        assertEquals(429, response.first)
        assertEquals(17L, response.second)
        assertNull(response.third)
    }

    @Test
    fun `HTTP error statuses do not throw`() = runBlocking {
        val status = downloader { respond("nope", HttpStatusCode.NotFound) }
            .use { d -> d.open("https://example.test/m.bin", 0) { it.status } }
        assertEquals(404, status)
    }

    @Test
    fun `Retry-After only accepts delta seconds`() {
        assertEquals(120L, KtorModelDownloader.parseRetryAfter("120"))
        assertEquals(5L, KtorModelDownloader.parseRetryAfter(" 5 "))
        assertNull(KtorModelDownloader.parseRetryAfter("Wed, 21 Oct 2026 07:28:00 GMT"))
        assertNull(KtorModelDownloader.parseRetryAfter("-3"))
        assertNull(KtorModelDownloader.parseRetryAfter(""))
        assertNull(KtorModelDownloader.parseRetryAfter(null))
    }

    @Test
    fun `User-Agent is set and no Authorization header is sent`() = runBlocking {
        downloader { respond(ByteArray(0), HttpStatusCode.OK) }.use { d ->
            d.open("https://example.test/m.bin", 0) { it.status }
        }
        val headers = requests.single().headers
        assertEquals("HablaAndroid-ModelManager", headers[HttpHeaders.UserAgent])
        assertNull(headers[HttpHeaders.Authorization])
    }

    private companion object {
        const val HEAP_BUDGET_BYTES = 8L * 1024 * 1024
    }
}
