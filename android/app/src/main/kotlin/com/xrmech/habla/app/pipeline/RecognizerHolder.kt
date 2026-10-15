package com.xrmech.habla.app.pipeline

import com.xrmech.habla.asr.AsrException
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.SpeechRecognizer
import com.xrmech.habla.asr.WhisperCppRecognizer
import com.xrmech.habla.shared.SharedAssets
import java.io.File
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

/**
 * Loads the Whisper model once per session (design section 8) and frees it (~500 MB) at stop.
 * `open` so coordinator tests can return a FakeSpeechRecognizer from [acquire].
 *
 * A load that is still running when [release] is called (Stop or a failed start within the 1-3 s
 * load) closes the recognizer it produces instead of keeping it; a load whose caller is cancelled
 * is still registered, so the next [release] frees it. The native model has no finalizer, so
 * either path would otherwise leak it.
 */
open class RecognizerHolder(
    private val models: ModelManager,
    private val assets: SharedAssets,
    private val io: CoroutineDispatcher,
    /** asr usage contract: the profile value (D5 writes the A56-measured count); the recognizer clamps it to the cores. */
    internal val threads: Int = assets.asrProfile.whisperCpp.threads,
) {
    private val lock = Any()
    private var current: SpeechRecognizer? = null

    /** Bumped by [release]; a load that started under an older value discards its result. */
    private var generation = 0

    /**
     * @throws AsrException MODEL_MISSING when the verified file is absent, MODEL_LOAD_FAILED from
     *   the native load.
     * @throws CancellationException when [release] ran while the model was loading.
     */
    open suspend fun acquire(modelId: String): SpeechRecognizer {
        val (loaded, gen) = synchronized(lock) { current to generation }
        loaded?.let { return it }
        val file = modelFile(modelId) ?: throw AsrException(
            "Speech model missing - download it in Settings",
            AsrException.Kind.MODEL_MISSING,
        )
        return withContext(io) {
            val r = load(file)
            // Registered before withContext returns: a caller cancelled meanwhile never receives
            // r, but release() still finds and closes it.
            val kept = synchronized(lock) {
                if (generation == gen) {
                    current = r
                    true
                } else {
                    false
                }
            }
            if (!kept) {
                runCatching { r.close() }
                throw CancellationException("speech model released while loading")
            }
            r
        }
    }

    /** Closes the loaded recognizer and discards any load in progress; idempotent. */
    open fun release() {
        val r = synchronized(lock) {
            generation++
            current.also { current = null }
        } ?: return
        runCatching { r.close() }
    }

    /** The verified model file, or null when it is not downloaded. */
    internal open fun modelFile(modelId: String): File? = models.fileFor(modelId)

    /** Builds and preloads the native recognizer; runs on [io]. */
    internal open suspend fun load(file: File): SpeechRecognizer {
        val r = WhisperCppRecognizer(file, assets.asrProfile, threads)
        try {
            r.preload()
        } catch (e: CancellationException) {
            r.close()
            throw e
        } catch (e: AsrException) {
            r.close()
            throw e
        } catch (e: Exception) {
            r.close()
            throw AsrException("model load failed: ${e.message}", AsrException.Kind.MODEL_LOAD_FAILED, e)
        }
        return r
    }
}
