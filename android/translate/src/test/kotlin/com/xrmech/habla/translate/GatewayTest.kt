package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.request.HttpRequestData
import io.ktor.client.request.HttpResponseData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.content.TextContent
import io.ktor.http.headersOf
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Test
import java.io.IOException
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** D051: llm-gateway client, retry/refusal handling and the Translator's Pro route. */
class GatewayTest {
    private val assets = TestAssets.assets

    private val okBody = """{"corrected":"¿Qué onda, güey?","translated":"What's up, dude?",
        "flaggedPhrases":[{"phrase":"qué onda","literal":"what wave","meaning":"what's up","type":"slang","saveWorthy":true}],
        "confidence":0.93,"speakerHint":null,"isCorrection":false,"correctionDetail":null,
        "meta":{"model":"gpt-5.4-mini","attempts":1,"salvaged":false,"passthroughRetried":false,"latencyMs":1430},"futureField":1}"""

    private fun error(code: String, fallback: String, retryAfter: Long? = null, details: String = "{}") =
        """{"error":{"code":"$code","message":"m $code","requestId":"r","retryAfterSeconds":${retryAfter ?: "null"},"fallback":"$fallback","details":$details}}"""

    /** Scripted gateway: each request takes the next reply; requests are recorded. */
    private class Fake(vararg replies: MockRequestHandleScope.(HttpRequestData) -> HttpResponseData) {
        val requests = mutableListOf<HttpRequestData>()
        private val queue = ArrayDeque(replies.toList())
        val engine = MockEngine { req ->
            requests += req
            queue.removeFirst()(this, req)
        }
        fun body(i: Int) = (requests[i].body as TextContent).text
    }

    private fun reply(status: Int, body: String): MockRequestHandleScope.(HttpRequestData) -> HttpResponseData =
        { respond(body, HttpStatusCode.fromValue(status), headersOf(HttpHeaders.ContentType, "application/json")) }

    private val tokens = mutableListOf<Boolean>()
    private fun client(fake: Fake, token: String? = "tok-1", fresh: String = "tok-2") = GatewayClient(
        GatewayClient.http(fake.engine, debugLogging = false), "https://gw.example/", "0.1.0-personal",
    ) { force -> tokens += force; if (token == null) null else if (force) fresh else token }

    private var now = 1_000_000L
    private val sleeps = mutableListOf<Long>()
    private fun gateway(fake: Fake, token: String? = "tok-1") =
        GatewayTranslator(client(fake, token), maxContext = 5, clock = { now }, newRequestId = { "req-1" }, sleep = { sleeps += it })

    private suspend fun GatewayTranslator.run(text: String = "¿Qué onda, güey?", context: List<ContextExchange> = emptyList()) =
        translate(text, Direction.ES_TO_EN, Mode.CONVERSATION, context, "topic")

    @Test
    fun translateSendsTheContractRequestAndMapsTheResult(): Unit = runBlocking {
        val fake = Fake(reply(200, okBody))
        val context = (1..7).map { ContextExchange("s$it", "t$it") }
        val ok = assertIs<GatewayResult.Ok>(gateway(fake).run(context = context))

        val req = fake.requests.single()
        assertEquals("https://gw.example/v1/habla/translate", req.url.toString())
        assertEquals("Bearer tok-1", req.headers[HttpHeaders.Authorization])
        assertEquals("habla", req.headers["X-App-Id"])
        assertEquals("0.1.0-personal", req.headers["X-App-Version"])
        val body = Json.parseToJsonElement(fake.body(0)).jsonObject
        assertEquals("req-1", body["requestId"]!!.jsonPrimitive.content)
        assertEquals("es_to_en", body["direction"]!!.jsonPrimitive.content)
        assertEquals("conversation", body["mode"]!!.jsonPrimitive.content)
        assertEquals("Speaker A", body["speakerLabel"]!!.jsonPrimitive.content)
        val lines = body["context"]!!.jsonArray
        assertEquals(5, lines.size, "max_context_exchanges, newest kept")
        assertEquals("s3", lines.first().jsonObject["source"]!!.jsonPrimitive.content)
        assertEquals("Speaker A", lines.first().jsonObject["speaker"]!!.jsonPrimitive.content)

        assertEquals("gpt-5.4-mini", ok.model)
        assertEquals("What's up, dude?", ok.result.translated)
        assertEquals(0.93, ok.result.confidence)
        val phrase = ok.result.flaggedPhrases.single()
        assertEquals("qué onda", phrase.phrase)
        assertEquals("slang", phrase.type)
        assertEquals("llm", phrase.source)
        assertNull(phrase.spanStart)
    }

    @Test
    fun expiredTokenIsRefreshedOnceAndTheSameBodyResent(): Unit = runBlocking {
        val fake = Fake(reply(401, error("AUTH_EXPIRED", "none")), reply(200, okBody))
        assertIs<GatewayResult.Ok>(gateway(fake).run())
        assertEquals(listOf(false, true), tokens)
        assertEquals("Bearer tok-2", fake.requests[1].headers[HttpHeaders.Authorization])
        assertEquals(fake.body(0), fake.body(1))
    }

    @Test
    fun signedOutMakesNoRequest(): Unit = runBlocking {
        val fake = Fake()
        val r = assertIs<GatewayResult.Unavailable>(gateway(fake, token = null).run())
        assertEquals(GatewayException.SIGNED_OUT, r.error.code)
        assertEquals(LlmException.Kind.NO_KEY, r.error.kind)
        assertTrue(fake.requests.isEmpty())
    }

    @Test
    fun retryFallbackResendsTheSameRequestId(): Unit = runBlocking {
        val fake = Fake(reply(409, error("DUPLICATE_REQUEST", "retry", retryAfter = 1)), reply(200, okBody))
        assertIs<GatewayResult.Ok>(gateway(fake).run())
        assertEquals(2, fake.requests.size)
        assertEquals(fake.body(0), fake.body(1))
        assertEquals(listOf(1000L), sleeps)
    }

    @Test
    fun retriesStopAfterThreeAttempts(): Unit = runBlocking {
        val fake = Fake(*Array(3) { reply(429, error("RATE_LIMITED", "retry", retryAfter = 30)) })
        val r = assertIs<GatewayResult.Unavailable>(gateway(fake).run())
        assertEquals("RATE_LIMITED", r.error.code)
        assertEquals(3, fake.requests.size)
        assertEquals(listOf(3000L, 3000L), sleeps, "Retry-After capped at 3 s inside the 15 s budget")
    }

    @Test
    fun spendCeilingBlocksTheGatewayUntilItResets(): Unit = runBlocking {
        val resets = java.time.Instant.ofEpochMilli(now + 3_600_000).toString()
        val fake = Fake(reply(429, error("SPEND_CEILING_REACHED", "on_device", details = """{"ceilingResetsAt":"$resets"}""")), reply(200, okBody))
        val g = gateway(fake)
        val first = assertIs<GatewayResult.Unavailable>(g.run())
        assertEquals(LlmException.Kind.QUOTA, first.error.kind)
        assertEquals(resets, first.error.resetsAt)
        assertEquals(now + 3_600_000, g.status.value.blockedUntilMs)

        assertIs<GatewayResult.Unavailable>(g.run())
        assertEquals(1, fake.requests.size, "blocked: no request while the allowance is used")

        now += 3_600_001
        assertIs<GatewayResult.Ok>(g.run())
        assertNull(g.status.value.lastError)
    }

    @Test
    fun notEntitledBlocksUntilReset(): Unit = runBlocking {
        val fake = Fake(reply(403, error("NOT_ENTITLED", "on_device")), reply(200, okBody))
        val g = gateway(fake)
        assertIs<GatewayResult.Unavailable>(g.run())
        assertIs<GatewayResult.Unavailable>(g.run())
        assertEquals(1, fake.requests.size)
        g.reset()
        assertIs<GatewayResult.Ok>(g.run())
    }

    @Test
    fun proxyPagesNetworkErrorsAndLongTranscriptsAreUnavailable(): Unit = runBlocking {
        val proxy = Fake(reply(502, "<html>Bad Gateway</html>"))
        assertEquals("HTTP_502", assertIs<GatewayResult.Unavailable>(gateway(proxy).run()).error.code)
        assertEquals(0L, gateway(proxy).status.value.blockedUntilMs)

        val down = Fake({ throw IOException("reset") })
        val e = assertIs<GatewayResult.Unavailable>(gateway(down).run()).error
        assertEquals(GatewayException.NETWORK, e.code)
        assertEquals(LlmException.Kind.NETWORK, e.kind)

        val none = Fake()
        assertEquals("VALIDATION_ERROR", assertIs<GatewayResult.Unavailable>(gateway(none).run("a".repeat(2001))).error.code)
        assertTrue(none.requests.isEmpty())
    }

    @Test
    fun badSuccessBodyIsBadResponse(): Unit = runBlocking {
        val fake = Fake(reply(200, """{"translated":"x"}"""))
        assertEquals(GatewayException.BAD_RESPONSE, assertIs<GatewayResult.Unavailable>(gateway(fake).run()).error.code)
    }

    // ---- Translator routing ---------------------------------------------------------------

    private class Phone(private val r: OnDeviceResult) : OnDeviceTranslator {
        var calls = 0
        override suspend fun translate(text: String, direction: Direction): OnDeviceResult {
            calls++
            return r
        }
    }

    private fun translator(fake: Fake, phone: Phone, token: String? = "tok-1") = Translator(
        FakeProvider(), assets,
        { TranslateSettings("gpt-5.4-mini", "gpt-6-luna", true, finalEngine = TranslateEngine.GATEWAY, previewEngine = TranslateEngine.ON_DEVICE) },
        CostMeter(assets.models), phone, gateway(fake, token),
    )

    @Test
    fun proFinalUsesTheGatewayResultAndNoLocalCost(): Unit = runBlocking {
        val phone = Phone(OnDeviceResult.Ok("local"))
        val costs = CostMeter(assets.models)
        val t = Translator(
            FakeProvider(), assets,
            { TranslateSettings("gpt-5.4-mini", "gpt-6-luna", true, finalEngine = TranslateEngine.GATEWAY) },
            costs, phone, gateway(Fake(reply(200, okBody))),
        )
        val ok = assertIs<TranslationOutcome.Ok>(t.translate("¿Qué onda, güey?", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), ""))
        assertEquals("gpt-5.4-mini", ok.model)
        assertEquals("What's up, dude?", ok.result.translated)
        assertEquals(0, phone.calls)
        assertEquals(0.0, costs.totals.value.sessionCostUsd, "the gateway meters Pro spend, not the phone")
    }

    @Test
    fun refusedProFinalFallsBackToThePhone(): Unit = runBlocking {
        val phone = Phone(OnDeviceResult.Ok("hello"))
        val ok = assertIs<TranslationOutcome.Ok>(
            translator(Fake(reply(403, error("NOT_ENTITLED", "on_device"))), phone).translate("hola", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), ""),
        )
        assertEquals(OnDeviceTranslator.MODEL_ID, ok.model)
        assertEquals("hello", ok.result.translated)
    }

    @Test
    fun withoutAPackTheGatewayReasonIsShownUnlessSignedOut(): Unit = runBlocking {
        val failed = assertIs<TranslationOutcome.Failed>(
            translator(Fake(reply(429, error("SPEND_CEILING_REACHED", "on_device"))), Phone(OnDeviceResult.NotReady))
                .translate("hola", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), ""),
        )
        assertEquals(LlmException.Kind.QUOTA, failed.error.kind)

        val signedOut = assertIs<TranslationOutcome.Failed>(
            translator(Fake(), Phone(OnDeviceResult.NotReady), token = null).translate("hola", Direction.ES_TO_EN, Mode.CONVERSATION, emptyList(), ""),
        )
        assertEquals(Translator.PACK_MISSING, signedOut.error.message)
    }

    @Test
    fun proTopicGoesThroughTheGatewayAndKeepsThePreviousOnFailure(): Unit = runBlocking {
        val fake = Fake(reply(200, """{"summary":"  Ordering food. ","meta":{"model":"gpt-6-luna","attempts":1}}"""), reply(503, error("UPSTREAM_UNAVAILABLE", "on_device")))
        val t = translator(fake, Phone(OnDeviceResult.Ok("x")))
        assertEquals("Ordering food.", t.updateTopicSummary("", "Quiero tacos", "I want tacos"))
        assertEquals("https://gw.example/v1/habla/topic", fake.requests[0].url.toString())
        assertEquals("prev", t.updateTopicSummary("prev", "a", "b"))
    }
}
