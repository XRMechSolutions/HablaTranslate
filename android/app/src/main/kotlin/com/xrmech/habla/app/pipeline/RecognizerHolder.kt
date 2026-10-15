package com.xrmech.habla.app.pipeline

import com.xrmech.habla.asr.AsrException
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.SpeechRecognizer
import com.xrmech.habla.asr.WhisperCppRecognizer
import com.xrmech.habla.shared.SharedAssets
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

/**
 * Loads the Whisper model once per session (design section 8) and frees it (~500 MB) at stop.
 * `open` so coordinator tests can return a FakeSpeechRecognizer from [acquire].
 */
open class RecognizerHolder(
    private val models: ModelManager,
    private val assets: SharedAssets,
    private val io: CoroutineDispatcher,
    /** asr usage contract: the profile value (D5 writes the A56-measured count); the recognizer clamps it to the cores. */
    internal val threads: Int = assets.asrProfile.whisperCpp.threads,
) {
    @Volatile private var current: SpeechRecognizer? = null

    /**
     * @throws AsrException MODEL_MISSING when the verified file is absent, MODEL_LOAD_FAILED from
     *   the native load.
     */
    open suspend fun acquire(modelId: String): SpeechRecognizer {
        current?.let { return it }
        val file = models.fileFor(modelId) ?: throw AsrException(
            "Speech model missing - download it in Settings",
            AsrException.Kind.MODEL_MISSING,
        )
        val recognizer = withContext(io) {
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
            r
        }
        current = recognizer
        return recognizer
    }

    /** Closes the loaded recognizer; idempotent. */
    open fun release() {
        val r = current ?: return
        current = null
        runCatching { r.close() }
    }
}
