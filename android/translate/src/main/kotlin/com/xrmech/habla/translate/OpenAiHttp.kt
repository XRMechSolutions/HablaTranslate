package com.xrmech.habla.translate

import io.ktor.client.HttpClient
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.logging.LogLevel
import io.ktor.client.plugins.logging.Logger
import io.ktor.client.plugins.logging.Logging
import io.ktor.http.HttpHeaders
import okhttp3.Dispatcher
import java.util.concurrent.TimeUnit

/**
 * Builds the one `HttpClient` used for OpenAI. Production passes [okHttpEngine]; tests pass a
 * `MockEngine`, so tests run the production plugin set.
 *
 * Deliberately absent: `ContentNegotiation` (bodies are built and read as text), `defaultRequest`
 * auth (the key is fetched per attempt), `HttpRequestRetry` (retries live in [ResilientProvider])
 * and certificate pinning (OpenAI rotates certificates; a stale pin would break every install).
 */
object OpenAiHttp {
    const val BASE_URL = "https://api.openai.com/v1"

    const val CONNECT_TIMEOUT_MS = 10_000L
    const val SOCKET_TIMEOUT_MS = 30_000L
    const val REQUEST_TIMEOUT_MS = 30_000L

    /**
     * OkHttp engine whose own timeouts sit above the [HttpTimeout] values, so the plugin is what
     * fires. OkHttp's default of 5 requests per host would queue concurrent finals, partials and
     * topic calls behind each other inside the timeout window, hence 16.
     */
    fun okHttpEngine(): HttpClientEngine = OkHttp.create {
        config {
            connectTimeout(20, TimeUnit.SECONDS)
            readTimeout(60, TimeUnit.SECONDS)
            writeTimeout(60, TimeUnit.SECONDS)
            callTimeout(0, TimeUnit.SECONDS)
            retryOnConnectionFailure(false)
            dispatcher(Dispatcher().apply { maxRequestsPerHost = 16 })
        }
    }

    /**
     * @param debugLogging `BuildConfig.DEBUG` from `:app`: HEADERS when true, nothing otherwise.
     *   Bodies (user speech) are never logged in any build.
     */
    fun create(engine: HttpClientEngine, debugLogging: Boolean, log: TranslateLog = TranslateLogging.sink): HttpClient {
        val scrubbed = Log(log)
        return HttpClient(engine) {
            expectSuccess = false
            install(HttpTimeout) {
                connectTimeoutMillis = CONNECT_TIMEOUT_MS
                socketTimeoutMillis = SOCKET_TIMEOUT_MS
                requestTimeoutMillis = REQUEST_TIMEOUT_MS
            }
            install(Logging) {
                logger = object : Logger {
                    override fun log(message: String) = scrubbed.debug(message)
                }
                level = if (debugLogging) LogLevel.HEADERS else LogLevel.NONE
                // Unconditional: redaction that only exists in one branch is missing where it matters.
                sanitizeHeader { it.equals(HttpHeaders.Authorization, ignoreCase = true) }
            }
        }
    }
}
