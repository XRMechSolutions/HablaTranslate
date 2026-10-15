package com.xrmech.habla.asr

/**
 * Transport seam for [ModelManager]. Production: [KtorModelDownloader]; tests use fakes.
 */
interface ModelDownloader {
    /**
     * GET [url] (redirects followed), with `Range: bytes=<rangeStart>-` when [rangeStart] > 0.
     * [block] runs while the response is open; the body is invalid after it returns.
     *
     * Throws [java.io.IOException] for transport failures (including timeouts); never throws for an
     * HTTP status code.
     */
    suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T
}

/** An open HTTP response. Header values are raw; the redirect target is deliberately not exposed. */
class DownloadResponse(
    val status: Int,
    /** Raw `Content-Range` header, e.g. `bytes 1048576-264464606/264464607`. */
    val contentRange: String?,
    val contentLength: Long?,
    /** `Retry-After` in delta-seconds form only; `null` when absent or an HTTP date. */
    val retryAfterSeconds: Long?,
    val body: DownloadBody,
)

/** Streaming response body. */
fun interface DownloadBody {
    /** Reads up to buffer.size bytes; returns count, or -1 at end of body. Suspends, cancellable. */
    suspend fun read(buffer: ByteArray): Int
}
