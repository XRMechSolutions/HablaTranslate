package com.xrmech.habla.asr

import io.ktor.client.HttpClient
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.HttpTimeoutConfig
import io.ktor.client.plugins.UserAgent
import io.ktor.client.request.header
import io.ktor.client.request.prepareGet
import io.ktor.client.statement.bodyAsChannel
import io.ktor.http.HttpHeaders
import io.ktor.http.contentLength
import io.ktor.utils.io.readAvailable

/**
 * Production [ModelDownloader] on its own Ktor [HttpClient] (separate from the translation client,
 * whose request timeout and logging suit small JSON calls).
 *
 * - Request timeout is infinite: a 264 MB download must not die while bytes still flow; a stalled
 *   socket is detected by the 30 s socket timeout and surfaces as an IOException (retryable).
 * - Redirects are followed by the client (HF `resolve` -> signed CDN URL) and the target is never
 *   returned, so every attempt starts again from the pinned URL.
 * - No auth header and no body logging, ever.
 *
 * [engine] is a parameter so JVM tests can pass Ktor's MockEngine.
 */
class KtorModelDownloader(
    engine: HttpClientEngine = OkHttp.create(),
) : ModelDownloader, AutoCloseable {

    private val client = HttpClient(engine) {
        expectSuccess = false
        followRedirects = true
        install(HttpTimeout) {
            requestTimeoutMillis = HttpTimeoutConfig.INFINITE_TIMEOUT_MS
            connectTimeoutMillis = CONNECT_TIMEOUT_MS
            socketTimeoutMillis = SOCKET_TIMEOUT_MS
        }
        install(UserAgent) { agent = USER_AGENT }
    }

    override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T {
        require(rangeStart >= 0) { "rangeStart must be >= 0, was $rangeStart" }
        return client.prepareGet(url) {
            if (rangeStart > 0) header(HttpHeaders.Range, "bytes=$rangeStart-")
        }.execute { response ->
            // bodyAsChannel streams; body() would buffer the whole model in memory.
            val channel = response.bodyAsChannel()
            block(
                DownloadResponse(
                    status = response.status.value,
                    contentRange = response.headers[HttpHeaders.ContentRange],
                    contentLength = response.contentLength(),
                    retryAfterSeconds = parseRetryAfter(response.headers[HttpHeaders.RetryAfter]),
                    body = DownloadBody { buffer -> channel.readAvailable(buffer, 0, buffer.size) },
                ),
            )
        }
    }

    override fun close() {
        client.close()
    }

    internal companion object {
        const val USER_AGENT = "HablaAndroid-ModelManager"
        const val CONNECT_TIMEOUT_MS = 15_000L
        const val SOCKET_TIMEOUT_MS = 30_000L

        /** Delta-seconds form only (`Retry-After: 120`); HTTP-date and malformed values give null. */
        fun parseRetryAfter(value: String?): Long? {
            val trimmed = value?.trim() ?: return null
            if (trimmed.isEmpty() || !trimmed.all { it in '0'..'9' }) return null
            return trimmed.toLongOrNull()
        }
    }
}
