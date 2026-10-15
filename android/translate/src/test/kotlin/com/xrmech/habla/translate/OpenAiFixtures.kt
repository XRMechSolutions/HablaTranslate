package com.xrmech.habla.translate

import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockEngineConfig
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.request.HttpRequestData
import io.ktor.client.request.HttpResponseData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.content.TextContent
import io.ktor.http.headersOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope

/** One scripted reply for the mock OpenAI endpoint. */
sealed interface Reply {
    data class Fixture(val name: String, val status: Int = 200, val headers: Map<String, String> = emptyMap(), val delayMs: Long = 0) : Reply
    data class Throw(val error: Throwable, val delayMs: Long = 0) : Reply
}

/** MockEngine-backed OpenAI endpoint that replays fixtures from `src/test/resources/openai/`. */
class FakeOpenAi(private val scope: TestScope, replies: List<Reply> = emptyList()) {
    private val queue = ArrayDeque(replies)
    val requests = mutableListOf<HttpRequestData>()
    /** Virtual time (ms) at which each request reached the engine. */
    val requestTimes = mutableListOf<Long>()
    val logLines = mutableListOf<String>()
    var defaultReply: Reply? = null

    fun enqueue(vararg replies: Reply) { queue.addAll(replies) }

    val client: HttpClient by lazy {
        OpenAiHttp.create(
            MockEngine(
                MockEngineConfig().apply {
                    dispatcher = StandardTestDispatcher(scope.testScheduler)
                    addHandler { request -> handle(request) }
                },
            ),
            debugLogging = true,
            log = TranslateLog { _, m -> synchronized(logLines) { logLines += m } },
        )
    }

    fun bodyOf(index: Int): String = (requests[index].body as TextContent).text

    private suspend fun MockRequestHandleScope.handle(request: HttpRequestData): HttpResponseData {
        requests += request
        requestTimes += scope.testScheduler.currentTime
        val reply = queue.removeFirstOrNull() ?: defaultReply ?: error("no scripted reply for request ${requests.size}")
        return when (reply) {
            is Reply.Throw -> {
                if (reply.delayMs > 0) kotlinx.coroutines.delay(reply.delayMs)
                throw reply.error
            }
            is Reply.Fixture -> {
                if (reply.delayMs > 0) kotlinx.coroutines.delay(reply.delayMs)
                val contentType = if (reply.name.endsWith(".json")) "application/json" else "text/html"
                val headers = headersOf(
                    *(reply.headers.map { it.key to listOf(it.value) } + (HttpHeaders.ContentType to listOf(contentType))).toTypedArray(),
                )
                respond(fixture(reply.name), HttpStatusCode.fromValue(reply.status), headers)
            }
        }
    }

    companion object {
        fun fixture(name: String): String =
            FakeOpenAi::class.java.classLoader.getResourceAsStream("openai/$name")?.use { it.readBytes().toString(Charsets.UTF_8) }
                ?: error("missing fixture openai/$name")
    }
}
