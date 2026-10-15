package com.xrmech.habla.translate

import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.supervisorScope
import java.util.concurrent.atomic.AtomicReference

/**
 * Partial (in-progress) translations, port of the server's `_quick_translate`, with latest-wins
 * cancellation: a quick call (~1.3 s) outlasts the partial interval (~1 s), so a newer call
 * cancels the older one instead of letting results land out of order.
 */
class QuickTranslator(
    private val provider: TranslationProvider,
    assets: SharedAssets,
    private val settings: () -> TranslateSettings,
    private val costs: CostMeter,
) {
    private val log = Log()
    private val models = assets.models
    private val prompts = PromptBuilder(assets)
    private val fallback = ModelFallback(models)

    /** The HTTP work of the newest call, or null when idle. */
    private val inFlight = AtomicReference<Job?>(null)

    /**
     * Returns the trimmed translation, or null when partial translation is off, [text] is blank,
     * the call fails, the model returns nothing, or a newer call / [cancelInFlight] superseded it.
     * Throws only `CancellationException`, and only when the caller itself is cancelled.
     */
    suspend fun translate(text: String, direction: Direction): String? {
        val s = settings()
        if (!s.partialTranslation || text.isBlank()) return null
        val effective = if (direction == Direction.AUTO) Direction.ES_TO_EN else direction
        val model = fallback.resolveModel(s.quickModel, models.quickDefault)
        val prompt = try {
            prompts.quick(text, effective)
        } catch (e: IllegalArgumentException) {
            log.error("Quick prompt failed: ${e.message}")
            return null
        }
        val request = LlmRequest(model, "", prompt, QUICK_MAX_TOKENS, json = false, kind = UsageKind.QUICK, maxRetries = 0)

        // supervisorScope: a failed child must surface only through await(), not fail this call.
        return supervisorScope {
            // Lifecycle: created here as a child of this call; driven by the await below;
            // cancelled by a newer translate(), by cancelInFlight(), or with the caller's scope.
            // Errors surface through await() and are handled there.
            val job = async { provider.complete(request) }
            inFlight.getAndSet(job)?.cancel(CancellationException("superseded"))
            try {
                val response = job.await()
                costs.record(response.usage, UsageKind.QUICK)
                PyText.strip(response.text).ifEmpty { null }
            } catch (e: CancellationException) {
                // The caller's own cancellation propagates; a superseded child just yields null.
                currentCoroutineContext().ensureActive()
                null
            } catch (e: LlmException) {
                log.debug("Quick translate failed: ${e.kind}")
                null
            } finally {
                // Only clear our own job; a newer call may already own the slot.
                inFlight.compareAndSet(job, null)
            }
        }
    }

    /** Drops the in-flight partial (called by `:app` when the segment's final is emitted). */
    fun cancelInFlight() {
        inFlight.getAndSet(null)?.cancel(CancellationException("superseded"))
    }

    private companion object {
        const val QUICK_MAX_TOKENS = 256
    }
}
