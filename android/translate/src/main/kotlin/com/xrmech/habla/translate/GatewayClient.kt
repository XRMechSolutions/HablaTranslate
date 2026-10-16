package com.xrmech.habla.translate

import io.ktor.client.HttpClient
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.network.sockets.ConnectTimeoutException
import io.ktor.client.plugins.HttpRequestTimeoutException
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.logging.LogLevel
import io.ktor.client.plugins.logging.Logger
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.content.TextContent
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import okhttp3.CertificatePinner
import java.io.IOException
import java.net.SocketTimeoutException
import java.net.URI
import java.util.concurrent.TimeUnit
import kotlin.coroutines.cancellation.CancellationException

/** The signed-in user's Firebase ID token for Habla's own project, or null when signed out. */
fun interface IdTokenSource {
    suspend fun idToken(forceRefresh: Boolean): String?
}

/** What the gateway says to do after an error (`fallback` in the error envelope). */
enum class GatewayFallback { ON_DEVICE, RETRY, NONE }

/**
 * A gateway error. [code] is the contract's closed set, plus the client-side codes
 * [SIGNED_OUT], [NETWORK], [TIMEOUT] and [BAD_RESPONSE].
 */
class GatewayException(
    val code: String,
    message: String,
    val httpStatus: Int? = null,
    val fallback: GatewayFallback = GatewayFallback.ON_DEVICE,
    val retryAfterSeconds: Long? = null,
    /** `details.ceilingResetsAt` of SPEND_CEILING_REACHED, ISO-8601. */
    val resetsAt: String? = null,
    cause: Throwable? = null,
) : Exception(message, cause) {
    /** The contract's mapping to the card/banner kinds the app already has. */
    val kind: LlmException.Kind
        get() = when {
            code.startsWith("AUTH_") || code == "REAUTH_REQUIRED" || code == "ACCOUNT_DELETED" -> LlmException.Kind.INVALID_KEY
            code in NO_ACCESS -> LlmException.Kind.NO_KEY
            code == "SPEND_CEILING_REACHED" -> LlmException.Kind.QUOTA
            code == "RATE_LIMITED" || code == "IP_RATE_LIMITED" -> LlmException.Kind.RATE_LIMITED
            code == "UPSTREAM_MALFORMED" || code == BAD_RESPONSE -> LlmException.Kind.MALFORMED
            code == "UPSTREAM_TIMEOUT" || code == TIMEOUT -> LlmException.Kind.TIMEOUT
            code == NETWORK -> LlmException.Kind.NETWORK
            code == "VALIDATION_ERROR" -> LlmException.Kind.BAD_REQUEST
            else -> LlmException.Kind.SERVER
        }

    companion object {
        const val SIGNED_OUT = "SIGNED_OUT"
        const val NETWORK = "NETWORK"
        const val TIMEOUT = "TIMEOUT"
        const val BAD_RESPONSE = "BAD_RESPONSE"
        private val NO_ACCESS = setOf(SIGNED_OUT, "AUTH_REQUIRED", "NOT_ENTITLED", "ACCOUNT_REQUIRED", "EMAIL_NOT_VERIFIED", "ACCOUNT_DISABLED", "TASK_DISABLED")
    }
}

/**
 * One call per method against llm-gateway v1 for app `habla`. Retries across models happen in the
 * gateway. Here the only repeat is one token refresh after AUTH_EXPIRED/AUTH_INVALID, with the same
 * body and requestId. Never logs bodies (user speech) or the token.
 */
class GatewayClient(
    private val http: HttpClient,
    baseUrl: String,
    /** `versionName`, e.g. `0.1.0-personal`; the gateway compares MAJOR.MINOR.PATCH only. */
    private val appVersion: String,
    private val tokens: IdTokenSource,
) {
    private val base = baseUrl.trimEnd('/')

    suspend fun translate(request: GatewayTranslateRequest): GatewayTranslateResponse =
        post("/v1/habla/translate", GatewayTranslateRequest.serializer(), request, GatewayTranslateResponse.serializer())

    suspend fun topic(request: GatewayTopicRequest): GatewayTopicResponse =
        post("/v1/habla/topic", GatewayTopicRequest.serializer(), request, GatewayTopicResponse.serializer())

    suspend fun me(): GatewayMe = authed { token -> http.get("$base/v1/me") { headers(token) } }.decode(GatewayMe.serializer())

    suspend fun config(): GatewayConfig = call { http.get("$base/v1/config") { headers(null) } }.decode(GatewayConfig.serializer())

    private suspend fun <Q, R> post(path: String, qs: KSerializer<Q>, body: Q, rs: KSerializer<R>): R {
        val text = GatewayWire.json.encodeToString(qs, body)
        return authed { token ->
            http.post("$base$path") {
                headers(token)
                setBody(TextContent(text, ContentType.Application.Json))
            }
        }.decode(rs)
    }

    private fun io.ktor.client.request.HttpRequestBuilder.headers(token: String?) {
        if (token != null) header(HttpHeaders.Authorization, "Bearer $token")
        header(APP_ID_HEADER, APP_ID)
        header(APP_VERSION_HEADER, appVersion)
    }

    /** Sends with the cached token; on AUTH_EXPIRED/AUTH_INVALID refreshes once and resends. */
    private suspend fun authed(send: suspend (String) -> HttpResponse): HttpResponse {
        val token = tokens.idToken(forceRefresh = false) ?: throw GatewayException(GatewayException.SIGNED_OUT, "Not signed in")
        val first = call { send(token) }
        if (first.status.value != 401) return first
        val error = errorOf(first)
        if (error.code != "AUTH_EXPIRED" && error.code != "AUTH_INVALID") throw error
        val fresh = tokens.idToken(forceRefresh = true) ?: throw GatewayException(GatewayException.SIGNED_OUT, "Not signed in")
        return call { send(fresh) }
    }

    private suspend fun call(send: suspend () -> HttpResponse): HttpResponse = try {
        send()
    } catch (e: CancellationException) {
        throw e
    } catch (e: HttpRequestTimeoutException) {
        throw GatewayException(GatewayException.TIMEOUT, "Gateway timed out", cause = e)
    } catch (e: ConnectTimeoutException) {
        throw GatewayException(GatewayException.TIMEOUT, "Gateway connect timed out", cause = e)
    } catch (e: SocketTimeoutException) {
        throw GatewayException(GatewayException.TIMEOUT, "Gateway timed out", cause = e)
    } catch (e: IOException) {
        // Includes SSLPeerUnverifiedException from a pin mismatch: never retried, the phone translates.
        throw GatewayException(GatewayException.NETWORK, "Gateway unreachable: ${e.javaClass.simpleName}", cause = e)
    }

    private suspend fun <R> HttpResponse.decode(rs: KSerializer<R>): R {
        if (status.value !in 200..299) throw errorOf(this)
        val text = bodyAsText()
        return try {
            GatewayWire.json.decodeFromString(rs, text)
        } catch (e: SerializationException) {
            throw GatewayException(GatewayException.BAD_RESPONSE, "Gateway reply unreadable", status.value, cause = e)
        } catch (e: IllegalArgumentException) {
            throw GatewayException(GatewayException.BAD_RESPONSE, "Gateway reply unreadable", status.value, cause = e)
        }
    }

    private suspend fun errorOf(response: HttpResponse): GatewayException {
        val status = response.status.value
        val body = try {
            GatewayWire.json.decodeFromString(GatewayErrorEnvelope.serializer(), response.bodyAsText()).error
        } catch (e: SerializationException) {
            null
        } catch (e: IllegalArgumentException) {
            null
        }
        // A proxy page (nginx 502, captive portal) has no envelope: treat as the gateway being down.
        if (body == null) return GatewayException("HTTP_$status", "Gateway HTTP $status", status, if (status == 429 || status >= 500) GatewayFallback.ON_DEVICE else GatewayFallback.NONE)
        val retryAfter = body.retryAfterSeconds ?: response.headers[HttpHeaders.RetryAfter]?.toLongOrNull()
        val resetsAt = (body.details?.get("ceilingResetsAt") as? kotlinx.serialization.json.JsonPrimitive)?.takeIf { it.isString }?.content
        return GatewayException(body.code, body.message.ifEmpty { body.code }, status, fallbackOf(body.fallback), retryAfter, resetsAt)
    }

    companion object {
        const val APP_ID = "habla"
        const val APP_ID_HEADER = "X-App-Id"
        const val APP_VERSION_HEADER = "X-App-Version"

        /** SPKI SHA-256 of ISRG Root X1 and X2, computed 2026-10-16 from letsencrypt.org/certs (contract: pin both). */
        val ISRG_PINS = listOf(
            "sha256/C5+lpZ7tcVwmwQIMcRtPbsQtWLABXhQzejna0wHFr8M=",
            "sha256/diGVwiVYbubAI3RW4hB9xU8e/CH2GnkuvVFZE8zmgzI=",
        )

        /** Client budget above the gateway's 12 s translate deadline, so its own 504 arrives first. */
        const val REQUEST_TIMEOUT_MS = 15_000L
        const val CONNECT_TIMEOUT_MS = 5_000L

        fun fallbackOf(s: String): GatewayFallback = when (s) {
            "on_device" -> GatewayFallback.ON_DEVICE
            "retry" -> GatewayFallback.RETRY
            else -> GatewayFallback.NONE
        }

        /**
         * OkHttp engine pinned to the ISRG roots for an `https` [baseUrl]. A plain-http URL (a local
         * gateway in development) is not pinned.
         */
        fun okHttpEngine(baseUrl: String): HttpClientEngine {
            val uri = URI(baseUrl)
            return OkHttp.create {
                config {
                    connectTimeout(10, TimeUnit.SECONDS)
                    readTimeout(30, TimeUnit.SECONDS)
                    callTimeout(0, TimeUnit.SECONDS)
                    retryOnConnectionFailure(false)
                    if (uri.scheme == "https" && uri.host != null) {
                        certificatePinner(CertificatePinner.Builder().apply { ISRG_PINS.forEach { add(uri.host, it) } }.build())
                    }
                }
            }
        }

        /** Same plugin set as [OpenAiHttp.create]: timeouts, header-only debug logging, Authorization redacted. */
        fun http(engine: HttpClientEngine, debugLogging: Boolean, log: TranslateLog = TranslateLogging.sink): HttpClient {
            val scrubbed = Log(log)
            return HttpClient(engine) {
                expectSuccess = false
                install(HttpTimeout) {
                    connectTimeoutMillis = CONNECT_TIMEOUT_MS
                    socketTimeoutMillis = REQUEST_TIMEOUT_MS
                    requestTimeoutMillis = REQUEST_TIMEOUT_MS
                }
                install(Logging) {
                    logger = object : Logger {
                        override fun log(message: String) = scrubbed.debug(message)
                    }
                    level = if (debugLogging) LogLevel.HEADERS else LogLevel.NONE
                    sanitizeHeader { it.equals(HttpHeaders.Authorization, ignoreCase = true) }
                }
            }
        }
    }
}
