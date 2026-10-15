package com.xrmech.habla.translate

import com.xrmech.habla.translate.LlmException.Kind
import com.xrmech.habla.translate.model.UsageKind
import io.ktor.http.HttpHeaders
import io.ktor.http.headersOf
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.net.ConnectException
import java.net.UnknownHostException
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

class OpenAiProviderTest {

    private val models = TestAssets.assets.models
    private val key = "sk-proj-TESTkey0123456789abcdefghij"
    private val final = LlmRequest("gpt-5.4-mini", "SYS", "USER", 1024, true, UsageKind.FINAL, 3)

    private fun TestScope.setup(vararg replies: Reply, keyFn: suspend () -> String? = { key }): Pair<FakeOpenAi, OpenAiProvider> {
        val fake = FakeOpenAi(this, replies.toList())
        return fake to OpenAiProvider(fake.client, keyFn, models)
    }

    private suspend fun OpenAiProvider.fail(request: LlmRequest = final): LlmException =
        try {
            complete(request)
            error("expected LlmException")
        } catch (e: LlmException) {
            e
        }

    private fun assertNoSecrets(fake: FakeOpenAi, vararg errors: Throwable) {
        val texts = fake.logLines + errors.map { it.message ?: "" }
        val keyPattern = Regex("sk-[A-Za-z0-9_-]{8,}")
        for (t in texts) {
            assertFalse(key in t, "key leaked: $t")
            assertNull(keyPattern.find(t), "sk- secret leaked: $t")
        }
    }

    @Test
    fun interfacesRequestBodyExamplesByteForByte() = runTest {
        val (fake, provider) = setup(Reply.Fixture("ok_json.json"), Reply.Fixture("ok_text.json"))
        provider.complete(LlmRequest("gpt-6-luna", "SYS", "USER", 1024, true, UsageKind.FINAL, 3))
        provider.complete(LlmRequest("gpt-4o-mini", "", "...", 256, false, UsageKind.QUICK))
        assertEquals(
            """{"model":"gpt-6-luna","input":"USER","max_output_tokens":1024,"store":false,"instructions":"SYS","reasoning":{"effort":"none"},"text":{"verbosity":"low","format":{"type":"json_object"}}}""",
            fake.bodyOf(0),
        )
        assertEquals("""{"model":"gpt-4o-mini","input":"...","max_output_tokens":256,"store":false}""", fake.bodyOf(1))
        assertEquals("https://api.openai.com/v1/responses", fake.requests[0].url.toString())
    }

    @Test
    fun requestBodyPerModelAndKind() {
        for ((id, info) in models.models) {
            for (kind in UsageKind.entries) {
                val json = kind == UsageKind.FINAL
                val body = OpenAiProvider.requestBody(LlmRequest(id, if (json) "S" else "", "U", 100, json, kind), models)
                assertEquals(id, body["model"].toString().trim('"'))
                assertEquals(json, "instructions" in body)
                assertEquals(info.effort != null && info.effort in info.efforts, "reasoning" in body, id)
                assertEquals(info.verbosity || json, "text" in body, id)
            }
        }
        val unknown = OpenAiProvider.requestBody(LlmRequest("gpt-unknown", "", "U", 10, false, UsageKind.QUICK), models)
        assertEquals(setOf("model", "input", "max_output_tokens", "store"), unknown.keys)
    }

    @Test
    fun authorizationHeaderCarriesTheKey() = runTest {
        val (fake, provider) = setup(Reply.Fixture("ok_json.json"))
        provider.complete(final)
        assertEquals("Bearer $key", fake.requests[0].headers[HttpHeaders.Authorization])
        assertNoSecrets(fake)
    }

    @Test
    fun okJsonResponse() = runTest {
        val (_, provider) = setup(Reply.Fixture("ok_json.json"))
        val r = provider.complete(final)
        assertTrue(r.text.startsWith("{\"corrected\": \"Me importa un pepino.\""))
        assertFalse(r.incomplete)
        assertEquals(812, r.usage.inputTokens)
        assertEquals(64, r.usage.outputTokens)
        assertEquals("gpt-5.4-mini", r.usage.model, "usage model is the request id, not the echoed snapshot")
    }

    @Test
    fun firstOutputTextOnlyAndReasoningItemSkipped() = runTest {
        val (_, provider) = setup(Reply.Fixture("ok_text.json"), Reply.Fixture("ok_reasoning_then_message.json"))
        assertEquals("  where is  ", provider.complete(final).text)
        val r = provider.complete(final)
        assertEquals("the answer", r.text)
        assertEquals(200, r.usage.reasoningTokens)
    }

    @Test
    fun cachedAndReasoningUsage() = runTest {
        val (_, provider) = setup(Reply.Fixture("ok_cached_usage.json"))
        val u = provider.complete(final).usage
        assertEquals(2048, u.inputTokens)
        assertEquals(1024, u.cachedInputTokens)
        assertEquals(100, u.outputTokens)
        assertEquals(12, u.reasoningTokens)
    }

    @Test
    fun incompleteResponses() = runTest {
        val (_, provider) = setup(Reply.Fixture("incomplete_max_tokens.json"), Reply.Fixture("incomplete_empty.json"))
        val truncated = provider.complete(final)
        assertTrue(truncated.incomplete)
        assertEquals("{\"corrected\": \"Hola\", \"translated\": \"Hel", truncated.text)
        val empty = provider.complete(final)
        assertTrue(empty.incomplete)
        assertEquals("", empty.text)
        assertEquals(900, empty.usage.inputTokens)
        assertEquals(0, empty.usage.cachedInputTokens)
        assertEquals(0, empty.usage.reasoningTokens)
    }

    @Test
    fun fixtureKindMapping() = runTest {
        data class Case(val fixture: String, val status: Int, val kind: Kind, val retryAfterMs: Long? = null, val headers: Map<String, String> = emptyMap())
        val cases = listOf(
            Case("failed_status.json", 200, Kind.SERVER),
            Case("err_400_effort.json", 400, Kind.BAD_REQUEST),
            Case("err_401.json", 401, Kind.INVALID_KEY),
            Case("err_403_country.json", 403, Kind.INVALID_KEY),
            Case("err_404_model.json", 404, Kind.MODEL_UNAVAILABLE),
            Case("err_429_rate.json", 429, Kind.RATE_LIMITED),
            Case("err_429_rate.json", 429, Kind.RATE_LIMITED, 2_000, mapOf("retry-after" to "2")),
            Case("err_429_rate.json", 429, Kind.RATE_LIMITED, 1_800, mapOf("retry-after" to "2", "retry-after-ms" to "1800")),
            Case("err_429_rate_long.json", 429, Kind.RATE_LIMITED, 60_000, mapOf("retry-after" to "60")),
            Case("err_429_insufficient_quota.json", 429, Kind.QUOTA, null, mapOf("retry-after" to "2")),
            Case("err_429_credit_balance.json", 429, Kind.QUOTA),
            Case("err_429_project_spend.json", 429, Kind.QUOTA),
            Case("err_502_html.html", 429, Kind.RATE_LIMITED),
            Case("err_500.json", 500, Kind.SERVER),
            Case("err_503_overloaded.json", 503, Kind.SERVER, 3_000, mapOf("retry-after" to "3")),
            Case("err_502_html.html", 502, Kind.SERVER),
            // Server parity: Retry-After applies to every 5xx, not only 503.
            Case("err_500.json", 500, Kind.SERVER, 4_000, mapOf("retry-after" to "4")),
            Case("err_502_html.html", 502, Kind.SERVER, 2_500, mapOf("retry-after-ms" to "2500")),
            Case("ok_not_json.txt", 200, Kind.MALFORMED),
            Case("err_400_effort.json", 409, Kind.BAD_REQUEST),
        )
        for (c in cases) {
            val (fake, provider) = setup(Reply.Fixture(c.fixture, c.status, c.headers))
            val e = provider.fail()
            assertEquals(c.kind, e.kind, "${c.fixture} @ ${c.status}")
            assertEquals(c.status, e.httpStatus, c.fixture)
            assertEquals(c.retryAfterMs, e.retryAfterMs, "${c.fixture} retryAfterMs")
            assertNoSecrets(fake, e)
        }
    }

    @Test
    fun errorMessageFormatAndRedaction() = runTest {
        val (fake, provider) = setup(Reply.Fixture("err_401.json", 401), Reply.Fixture("err_502_html.html", 502))
        val e401 = provider.fail()
        assertTrue(e401.message!!.startsWith("OpenAI 401 invalid_api_key: Incorrect API key provided: sk-***"), e401.message)
        val e502 = provider.fail()
        assertEquals("OpenAI 502: Bad Gateway", e502.message)
        assertNoSecrets(fake, e401, e502)
        assertTrue(fake.logLines.any { "502 Bad Gateway" in it }, "first 200 chars of the error body at DEBUG")
    }

    @Test
    fun perKindTimeouts() = runTest {
        val (_, provider) = setup(
            Reply.Fixture("ok_text.json", delayMs = 12_000),
            Reply.Fixture("ok_text.json", delayMs = 12_000),
            Reply.Fixture("ok_text.json", delayMs = 16_000),
            Reply.Fixture("ok_text.json", delayMs = 31_000),
        )
        assertEquals(Kind.TIMEOUT, provider.fail(LlmRequest("gpt-6-luna", "", "x", 256, false, UsageKind.QUICK)).kind)
        provider.complete(LlmRequest("gpt-6-luna", "", "x", 200, false, UsageKind.TOPIC))
        assertEquals(Kind.TIMEOUT, provider.fail(LlmRequest("gpt-6-luna", "", "x", 16, false, UsageKind.TEST)).kind)
        assertEquals(Kind.TIMEOUT, provider.fail(final).kind)
    }

    @Test
    fun networkErrors() = runTest {
        val (fake, provider) = setup(Reply.Throw(UnknownHostException("api.openai.com")), Reply.Throw(ConnectException("Connection refused")))
        val a = provider.fail()
        val b = provider.fail()
        assertEquals(Kind.NETWORK, a.kind)
        assertEquals(Kind.NETWORK, b.kind)
        assertNoSecrets(fake, a, b)
    }

    @Test
    fun noKeyAndInvalidKeySendNothing() = runTest {
        for ((k, kind) in listOf(null to Kind.NO_KEY, "  " to Kind.NO_KEY, "sk-abc\n" to Kind.INVALID_KEY, "sk-abc“def" to Kind.INVALID_KEY)) {
            val (fake, provider) = setup(keyFn = { k })
            val e = provider.fail()
            assertEquals(kind, e.kind, "key=${k?.length}")
            assertEquals(0, fake.requests.size)
            assertFalse("sk-abc" in e.message!!)
        }
    }

    @Test
    fun keyProviderInvokedPerCall() = runTest {
        var current: String? = "sk-first-0000000000"
        var calls = 0
        val (fake, provider) = setup(Reply.Fixture("ok_text.json"), Reply.Fixture("ok_text.json"), keyFn = { calls++; current })
        provider.complete(final)
        current = "sk-second-000000000"
        provider.complete(final)
        current = null
        assertEquals(Kind.NO_KEY, provider.fail().kind)
        assertEquals(3, calls)
        assertEquals("Bearer sk-first-0000000000", fake.requests[0].headers[HttpHeaders.Authorization])
        assertEquals("Bearer sk-second-000000000", fake.requests[1].headers[HttpHeaders.Authorization])
    }

    @Test
    fun testKeyVariants() = runTest {
        val (fake, provider) = setup(
            Reply.Fixture("ok_text.json"),
            Reply.Fixture("ok_text.json"),
            Reply.Fixture("err_401.json", 401),
            Reply.Fixture("err_429_rate.json", 429),
        )
        val withCandidate = assertIs<KeyTestResult.Valid>(provider.testKey("sk-candidate-0000000000"))
        assertEquals(models.quickDefault, withCandidate.usage.model)
        assertEquals("Bearer sk-candidate-0000000000", fake.requests[0].headers[HttpHeaders.Authorization])
        assertEquals(
            """{"model":"gpt-6-luna","input":"Reply with OK.","max_output_tokens":16,"store":false,"reasoning":{"effort":"none"},"text":{"verbosity":"low"}}""",
            fake.bodyOf(0),
        )
        assertIs<KeyTestResult.Valid>(provider.testKey())
        assertEquals("Bearer $key", fake.requests[1].headers[HttpHeaders.Authorization])
        val invalid = assertIs<KeyTestResult.Failed>(provider.testKey("sk-invalid-000000000000"))
        assertEquals(Kind.INVALID_KEY, invalid.error.kind)
        assertEquals(401, invalid.error.httpStatus)
        assertEquals(Kind.RATE_LIMITED, assertIs<KeyTestResult.Failed>(provider.testKey()).error.kind)
        assertNoSecrets(fake, invalid.error)
    }

    @Test
    fun testKeyRethrowsCancellation() = runTest {
        val (_, provider) = setup(Reply.Fixture("ok_text.json", delayMs = 5_000))
        assertThrows<TimeoutCancellationException> { withTimeout(1_000) { provider.testKey() } }
    }

    @Test
    fun retryAfterParsing() {
        val now = 1_760_000_000_000L
        assertEquals(1800L, RetryAfter.parse(headersOf("retry-after-ms" to listOf("1800"), "retry-after" to listOf("5")), now))
        assertEquals(2500L, RetryAfter.parse(headersOf("retry-after" to listOf("2.5")), now))
        assertEquals(null, RetryAfter.parse(headersOf("retry-after" to listOf("-1")), now))
        assertEquals(null, RetryAfter.parse(headersOf("retry-after" to listOf("soon")), now))
        assertEquals(null, RetryAfter.parse(headersOf(), now))
        // 2025-10-09T08:53:20Z is `now`; ten seconds later as an HTTP-date:
        assertEquals(10_000L, RetryAfter.parse(headersOf("retry-after" to listOf("Thu, 9 Oct 2025 08:53:30 GMT")), now))
        assertEquals(null, RetryAfter.parse(headersOf("retry-after" to listOf("Thu, 9 Oct 2025 08:53:10 GMT")), now))
    }
}
