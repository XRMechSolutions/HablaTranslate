package com.xrmech.habla.translate

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockEngineConfig
import io.ktor.client.engine.mock.respond
import io.ktor.client.plugins.HttpRequestTimeoutException
import io.ktor.client.plugins.timeout
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class OpenAiHttpTest {

    private val testKey = "sk-test-0123456789abcdefABCDEF"

    private fun TestScope.mockEngine(handlerDelayMs: Long = 0) = MockEngine(
        MockEngineConfig().apply {
            dispatcher = StandardTestDispatcher(testScheduler)
            addHandler {
                if (handlerDelayMs > 0) delay(handlerDelayMs)
                respond("""{"ok":true}""", HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "application/json"))
            }
        },
    )

    @Test
    fun headersLoggedWithoutAuthorizationValue() = runTest {
        val lines = mutableListOf<Pair<TranslateLog.Level, String>>()
        val client = OpenAiHttp.create(mockEngine(), debugLogging = true, log = TranslateLog { l, m -> lines += l to m })
        client.post("${OpenAiHttp.BASE_URL}/responses") {
            header(HttpHeaders.Authorization, "Bearer $testKey")
            setBody("""{"input":"secret speech"}""")
        }
        val text = lines.joinToString("\n") { it.second }
        assertTrue(lines.isNotEmpty())
        assertTrue("https://api.openai.com/v1/responses" in text, text)
        assertTrue(testKey !in text, text)
        assertTrue(Regex("sk-[A-Za-z0-9_-]{8,}").find(text) == null, text)
        assertTrue("secret speech" !in text, "bodies must never be logged")
        assertTrue(lines.all { it.first == TranslateLog.Level.DEBUG })
    }

    @Test
    fun nothingLoggedWhenDebugLoggingIsOff() = runTest {
        val lines = mutableListOf<String>()
        val client = OpenAiHttp.create(mockEngine(), debugLogging = false, log = TranslateLog { _, m -> lines += m })
        client.post("${OpenAiHttp.BASE_URL}/responses") { header(HttpHeaders.Authorization, "Bearer $testKey") }
        assertEquals(emptyList(), lines)
    }

    @Test
    fun perRequestTimeoutOverrideIsHonoured() = runTest {
        val client = OpenAiHttp.create(mockEngine(handlerDelayMs = 20_000), debugLogging = false)
        assertThrows<HttpRequestTimeoutException> {
            client.post("${OpenAiHttp.BASE_URL}/responses") { timeout { requestTimeoutMillis = 10_000 } }
        }
    }

    @Test
    fun defaultRequestTimeoutAllowsSlowerCalls() = runTest {
        val client = OpenAiHttp.create(mockEngine(handlerDelayMs = 20_000), debugLogging = false)
        val response = client.post("${OpenAiHttp.BASE_URL}/responses")
        assertEquals(200, response.status.value)
    }

    @Test
    fun okHttpEngineBuilds() {
        val engine = OpenAiHttp.okHttpEngine()
        OpenAiHttp.create(engine, debugLogging = false).close()
    }
}
