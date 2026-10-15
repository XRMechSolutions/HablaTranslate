package com.xrmech.habla.asr.fakes

import com.xrmech.habla.asr.DownloadBody
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.ModelDownloader
import java.io.IOException
import java.util.concurrent.CopyOnWriteArrayList
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.awaitCancellation

/**
 * Scripted [ModelDownloader] serving [content]. Each `open` pops the next [Reply]; with an empty
 * script it answers correctly (200 from 0, 206 from an offset).
 */
class FakeModelDownloader(
    private val content: ByteArray,
    private val clock: () -> Long = { 0L },
) : ModelDownloader {
    data class Open(val url: String, val rangeStart: Long, val atMs: Long)

    sealed interface Reply {
        /** Throws IOException from open() (connect failure). */
        data object ConnectFails : Reply

        data class Respond(
            val status: Int,
            val contentRange: String? = null,
            val contentLength: Long? = null,
            val retryAfterSeconds: Long? = null,
            /** Body bytes; null serves [content] from the requested offset (or 0 for 200). */
            val body: ByteArray? = null,
            /** Throw IOException after this many body bytes. */
            val failAfter: Long? = null,
            /** Suspend until cancelled after this many body bytes. */
            val stallAfter: Long? = null,
        ) : Reply
    }

    val opens = CopyOnWriteArrayList<Open>()
    private val script = ArrayDeque<Reply>()

    /** Completed when a stalling reply has delivered its bytes. */
    val stalled = CompletableDeferred<Unit>()

    fun script(vararg replies: Reply) = apply { script.addAll(replies) }

    fun ok200(body: ByteArray? = null, contentLength: Long? = content.size.toLong(), failAfter: Long? = null, stallAfter: Long? = null) =
        Reply.Respond(200, contentLength = contentLength, body = body, failAfter = failAfter, stallAfter = stallAfter)

    fun partial206(from: Long, total: Long = content.size.toLong(), start: Long = from) = Reply.Respond(
        206,
        contentRange = "bytes $start-${total - 1}/$total",
        contentLength = total - start,
        body = content.copyOfRange(start.toInt(), content.size),
    )

    fun status(code: Int, retryAfterSeconds: Long? = null) =
        Reply.Respond(code, retryAfterSeconds = retryAfterSeconds, body = ByteArray(0), contentLength = 0)

    override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T {
        opens += Open(url, rangeStart, clock())
        val reply = script.removeFirstOrNull() ?: if (rangeStart > 0) partial206(rangeStart) else ok200()
        when (reply) {
            Reply.ConnectFails -> throw IOException("connect failed")
            is Reply.Respond -> {
                val bytes = reply.body ?: content
                var pos = 0
                val body = DownloadBody { buffer ->
                    if (reply.failAfter != null && pos >= reply.failAfter) throw IOException("connection reset")
                    if (reply.stallAfter != null && pos >= reply.stallAfter) {
                        stalled.complete(Unit)
                        awaitCancellation()
                    }
                    if (pos >= bytes.size) return@DownloadBody -1
                    var n = minOf(buffer.size, bytes.size - pos)
                    reply.failAfter?.let { n = minOf(n.toLong(), it - pos).toInt() }
                    reply.stallAfter?.let { n = minOf(n.toLong(), it - pos).toInt() }
                    bytes.copyInto(buffer, 0, pos, pos + n)
                    pos += n
                    n
                }
                return block(DownloadResponse(reply.status, reply.contentRange, reply.contentLength, reply.retryAfterSeconds, body))
            }
        }
    }
}
