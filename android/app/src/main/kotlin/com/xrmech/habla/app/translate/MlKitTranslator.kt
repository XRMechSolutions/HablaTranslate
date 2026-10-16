package com.xrmech.habla.app.translate

import com.google.android.gms.tasks.Task
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.translate.OnDeviceResult
import com.xrmech.habla.translate.OnDeviceTranslator
import com.xrmech.habla.translate.model.Direction
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** The Spanish/English pack on the phone (D050). */
sealed interface PackState {
    data object Unknown : PackState
    data object Missing : PackState
    data object Downloading : PackState
    data object Ready : PackState
    data class Failed(val message: String) : PackState
}

/** Seam over the pack management of [MlKitTranslator] (Play services) for Settings tests. */
interface LanguagePack {
    val state: StateFlow<PackState>

    /** Reads the state from disk; no network. */
    suspend fun refresh()

    /** Downloads the pack. Callers ask first on a metered network (as for the speech models). */
    suspend fun download()

    suspend fun delete()
}

/**
 * ML Kit on-device translation for es <-> en (D050). Both clients live for the app's lifetime, so
 * they are never closed. Translation never downloads: only [download] does, after the user asks
 * (about 30 MB; Settings asks before using mobile data, as Google's guide requires).
 */
class MlKitTranslator : OnDeviceTranslator, LanguagePack {
    private val esToEn = Translation.getClient(options(TranslateLanguage.SPANISH, TranslateLanguage.ENGLISH))
    private val enToEs = Translation.getClient(options(TranslateLanguage.ENGLISH, TranslateLanguage.SPANISH))
    private val manager = RemoteModelManager.getInstance()
    private val packs = listOf(TranslateLanguage.SPANISH, TranslateLanguage.ENGLISH).map { TranslateRemoteModel.Builder(it).build() }
    private val spanish = packs.first()

    private val _state = MutableStateFlow<PackState>(PackState.Unknown)
    override val state: StateFlow<PackState> = _state.asStateFlow()

    /** Serializes download, delete and refresh so the state never moves backwards. */
    private val lock = Mutex()

    override suspend fun translate(text: String, direction: Direction): OnDeviceResult {
        if (_state.value == PackState.Unknown) refresh()
        if (_state.value != PackState.Ready) return OnDeviceResult.NotReady
        val client = if (direction == Direction.EN_TO_ES) enToEs else esToEn
        return try {
            OnDeviceResult.Ok(client.translate(text).await())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            HLog.w(HLog.TRANSLATE, "on-device translation failed: ${e.javaClass.simpleName}")
            OnDeviceResult.Failed(e.message ?: e.javaClass.simpleName)
        }
    }

    override suspend fun refresh() = lock.withLock {
        if (_state.value == PackState.Downloading) return@withLock
        _state.value = try {
            if (packs.all { manager.isModelDownloaded(it).await() }) PackState.Ready else PackState.Missing
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            PackState.Failed(e.message ?: e.javaClass.simpleName)
        }
    }

    /** Downloads both packs (English and Spanish); no Wi-Fi condition because Settings already asked. */
    override suspend fun download() = lock.withLock {
        _state.value = PackState.Downloading
        val conditions = DownloadConditions.Builder().build()
        _state.value = try {
            for (pack in packs) manager.download(pack, conditions).await()
            HLog.i(HLog.TRANSLATE, "on-device language pack ready")
            PackState.Ready
        } catch (e: CancellationException) {
            _state.value = PackState.Missing
            throw e
        } catch (e: Exception) {
            HLog.w(HLog.TRANSLATE, "language pack download failed: ${e.javaClass.simpleName}")
            PackState.Failed(e.message ?: "Download failed")
        }
    }

    /** Deletes the Spanish pack (the English one is shared with other ML Kit users of the phone). */
    override suspend fun delete() = lock.withLock {
        _state.value = try {
            manager.deleteDownloadedModel(spanish).await()
            HLog.i(HLog.TRANSLATE, "deleted on-device Spanish pack")
            PackState.Missing
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            PackState.Failed(e.message ?: "Delete failed")
        }
    }

    private companion object {
        fun options(source: String, target: String) =
            TranslatorOptions.Builder().setSourceLanguage(source).setTargetLanguage(target).build()
    }
}

/** Play services Tasks cannot be cancelled; a cancelled caller just stops waiting for the result. */
private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { cont ->
    addOnSuccessListener { if (cont.isActive) cont.resume(it) }
    addOnFailureListener { if (cont.isActive) cont.resumeWithException(it) }
    addOnCanceledListener { cont.cancel() }
}
