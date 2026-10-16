package com.xrmech.habla.app.ui.live

import com.xrmech.habla.app.capture.AudioCondition
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.SessionNotice
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.translate.PackState
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.translate.LlmException

enum class Severity { FATAL, ERROR, WARNING, INFO }

/** Declaration order is the tie-break inside one severity (interfaces.md table order). */
enum class BannerKind(val severity: Severity) {
    SESSION_FAILED(Severity.FATAL),
    MIC_DENIED(Severity.ERROR),
    MODEL_MISSING(Severity.ERROR),
    MODEL_FAILED(Severity.ERROR),
    KEY_INVALID(Severity.ERROR),
    NO_QUOTA(Severity.ERROR),
    MODEL_UNAVAILABLE(Severity.ERROR),
    PACK_MISSING(Severity.ERROR),
    ASR_ERROR(Severity.WARNING),
    TRANSLATE_ERROR(Severity.WARNING),
    OFFLINE(Severity.WARNING),
    MIC_SILENCED(Severity.WARNING),
    FALLING_BEHIND(Severity.WARNING),
    AUDIO_DROPPED(Severity.WARNING),
    HISTORY_WRITE_FAILED(Severity.WARNING),
    VAD_FALLBACK(Severity.WARNING),
    THERMAL_PAUSE(Severity.INFO),
    NO_KEY(Severity.INFO),
    MODEL_DOWNLOADING(Severity.INFO),
    LOUD_ENVIRONMENT(Severity.INFO),
}

/** The one action a banner or failed card offers (ui-standards section 4). */
enum class BannerAction(val label: String) {
    GRANT_MIC("Grant"),
    OPEN_APP_SETTINGS("Open settings"),
    DOWNLOAD_MODEL("Download"),
    RETRY_DOWNLOAD("Retry"),
    DELETE_AND_RETRY("Delete & retry"),
    CANCEL_DOWNLOAD("Cancel"),
    OPEN_SETTINGS("Settings"),
    ADD_KEY("Add key"),
    /** Opens Settings > Translation, where the language pack downloads (with the metered prompt). */
    GET_PACK("Get pack"),
}

data class Banner(val kind: BannerKind, val text: String, val action: BannerAction? = null) {
    val severity: Severity get() = kind.severity
}

enum class MicPermission { GRANTED, NOT_ASKED, DENIED, PERMANENTLY_DENIED }

/** Everything the banner area depends on (design section 7). */
data class LiveUiInputs(
    val live: LiveState,
    val modelState: ModelState,
    /** Catalog size of the selected speech model, for the MODEL_MISSING text. */
    val modelSizeBytes: Long,
    val keyState: KeyState,
    val online: Boolean,
    val micPermission: MicPermission,
    /** Selected final model id, named by MODEL_UNAVAILABLE. */
    val finalModel: String,
    /** Finals are translated on the phone (D050): no key needed, the pack is, and offline works. */
    val translateOnDevice: Boolean = false,
    /** Finals use the user's own OpenAI key; false for the phone and for Pro (D051). */
    val keyNeeded: Boolean = true,
    val packState: PackState = PackState.Ready,
)

/** Banners to show plus how many more are hidden behind "+N more". */
data class BannerSet(val shown: List<Banner>, val more: Int) {
    companion object {
        val EMPTY = BannerSet(emptyList(), 0)
    }
}

object BannerReducer {
    const val MAX_SHOWN = 3

    /** Every active banner, deduplicated by kind, ordered fatal > error > warning > info. */
    fun reduce(inputs: LiveUiInputs): List<Banner> {
        val out = LinkedHashMap<BannerKind, Banner>()
        fun add(b: Banner) {
            out.putIfAbsent(b.kind, b)
        }

        val live = inputs.live
        (live.phase as? SessionPhase.Failed)?.let {
            add(Banner(BannerKind.SESSION_FAILED, "${it.error.message} - tap Listen to retry"))
        }
        when (inputs.micPermission) {
            MicPermission.DENIED -> add(Banner(BannerKind.MIC_DENIED, "Microphone permission needed", BannerAction.GRANT_MIC))
            MicPermission.PERMANENTLY_DENIED ->
                add(Banner(BannerKind.MIC_DENIED, "Microphone permission needed", BannerAction.OPEN_APP_SETTINGS))
            else -> Unit
        }
        when (val m = inputs.modelState) {
            ModelState.Missing -> add(
                Banner(
                    BannerKind.MODEL_MISSING,
                    "Speech model not downloaded (${megabytes(inputs.modelSizeBytes)} MB) - needed to transcribe",
                    BannerAction.DOWNLOAD_MODEL,
                ),
            )
            is ModelState.Failed -> add(
                Banner(
                    BannerKind.MODEL_FAILED,
                    "Model download failed: ${ModelFailureText.of(m.reason)}",
                    if (m.retryable) BannerAction.RETRY_DOWNLOAD else BannerAction.DELETE_AND_RETRY,
                ),
            )
            is ModelState.Downloading -> add(
                Banner(
                    BannerKind.MODEL_DOWNLOADING,
                    "Downloading speech model ${megabytes(m.bytes)}/${megabytes(m.total)} MB",
                    BannerAction.CANCEL_DOWNLOAD,
                ),
            )
            else -> Unit
        }
        val tError = live.lastTranslationError
        when (inputs.keyState) {
            KeyState.INVALID -> {
                val status = tError?.takeIf { it.kind == LlmException.Kind.INVALID_KEY }?.httpStatus ?: 401
                add(Banner(BannerKind.KEY_INVALID, "OpenAI rejected the key ($status) - check or replace it", BannerAction.OPEN_SETTINGS))
            }
            KeyState.NO_QUOTA -> add(
                Banner(BannerKind.NO_QUOTA, "OpenAI account has no credit - add billing at platform.openai.com", BannerAction.OPEN_SETTINGS),
            )
            KeyState.ABSENT -> if (inputs.keyNeeded) {
                add(Banner(BannerKind.NO_KEY, "Translation off - add your OpenAI key", BannerAction.ADD_KEY))
            }
            else -> Unit
        }
        if (inputs.translateOnDevice) {
            when (val p = inputs.packState) {
                PackState.Missing -> add(
                    Banner(BannerKind.PACK_MISSING, "Spanish language pack not downloaded (~30 MB) - needed to translate", BannerAction.GET_PACK),
                )
                is PackState.Failed -> add(Banner(BannerKind.PACK_MISSING, "Language pack download failed: ${p.message}", BannerAction.GET_PACK))
                else -> Unit
            }
        }
        if (tError != null) {
            when (tError.kind) {
                LlmException.Kind.MODEL_UNAVAILABLE -> add(
                    Banner(BannerKind.MODEL_UNAVAILABLE, "Model ${inputs.finalModel} not available for this key (404)", BannerAction.OPEN_SETTINGS),
                )
                LlmException.Kind.NO_KEY, LlmException.Kind.INVALID_KEY, LlmException.Kind.QUOTA -> Unit
                else -> add(Banner(BannerKind.TRANSLATE_ERROR, "Translation failed: ${tError.kind} - ${tError.message}"))
            }
        }
        live.lastAsrError?.let { add(Banner(BannerKind.ASR_ERROR, "Transcription failed: $it")) }
        // On-device translation keeps working offline, so there is nothing to warn about then.
        if (!inputs.online && !inputs.translateOnDevice) {
            add(Banner(BannerKind.OFFLINE, "Offline - transcribing only; translations fail until back online"))
        }
        if (live.phase.active && live.audio == AudioCondition.Silenced) {
            add(Banner(BannerKind.MIC_SILENCED, "Microphone in use by another app or muted - paused"))
        }
        val n = live.notices
        if (SessionNotice.FALLING_BEHIND in n) {
            add(Banner(BannerKind.FALLING_BEHIND, "Phone falling behind - skipped ${live.metrics.finalsDropped} segments"))
        }
        if (SessionNotice.AUDIO_DROPPED in n) {
            add(Banner(BannerKind.AUDIO_DROPPED, "Audio dropped (${live.metrics.framesDropped} frames) - phone overloaded"))
        }
        if (SessionNotice.HISTORY_WRITE_FAILED in n) {
            add(Banner(BannerKind.HISTORY_WRITE_FAILED, "Couldn't save to history: ${live.lastHistoryError ?: "database error"}"))
        }
        if (SessionNotice.VAD_FALLBACK in n) {
            add(Banner(BannerKind.VAD_FALLBACK, "Voice detector failed - using energy detection"))
        }
        if (SessionNotice.THERMAL_PAUSE in n) add(Banner(BannerKind.THERMAL_PAUSE, "Phone is hot - live partials paused"))
        if (SessionNotice.LOUD_ENVIRONMENT in n) {
            val rms = live.noiseFloorRms?.toInt() ?: 0
            add(
                Banner(
                    BannerKind.LOUD_ENVIRONMENT,
                    "Loud environment detected (RMS $rms) - speech detection threshold raised automatically",
                ),
            )
        }
        return out.values.sortedWith(compareBy({ it.severity.ordinal }, { it.kind.ordinal }))
    }

    /** At most [MAX_SHOWN] by severity; the rest are counted for "+N more". */
    fun cap(all: List<Banner>): BannerSet = BannerSet(all.take(MAX_SHOWN), maxOf(0, all.size - MAX_SHOWN))

    private fun megabytes(bytes: Long): Long = (bytes + 500_000) / 1_000_000
}

/** ModelManager failure codes (asr ModelManager docs) as user text. */
object ModelFailureText {
    fun of(reason: String): String = when {
        reason == "network" -> "network error"
        reason == "unknown_model" -> "unknown model"
        reason == "size_mismatch" -> "file size mismatch"
        reason == "checksum_mismatch" -> "checksum mismatch"
        reason.startsWith("insufficient_storage:") -> {
            val bytes = reason.substringAfter(':').toLongOrNull()
            if (bytes != null) "not enough storage (${(bytes + 500_000) / 1_000_000} MB needed)" else "not enough storage"
        }
        reason.startsWith("http_") -> "server returned HTTP ${reason.removePrefix("http_")}"
        reason.startsWith("io:") -> "storage error (${reason.removePrefix("io:")})"
        else -> reason
    }
}
