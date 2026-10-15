package com.xrmech.habla.app.ui.history

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.pipeline.CardTranslation
import com.xrmech.habla.app.pipeline.ExchangeCard
import com.xrmech.habla.data.ExchangeRow
import com.xrmech.habla.data.ExportSummary
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.SessionRow
import com.xrmech.habla.data.VocabRepository
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.TranslationResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** One stored exchange as a read-only card plus what SessionDetail adds (edit, audio). */
data class DetailItem(val row: ExchangeRow, val card: ExchangeCard) {
    val hasAudio: Boolean get() = row.audioPath != null

    /** Prefill of the Edit transcript dialog. */
    val editableTranscript: String get() = row.correctedTranscript ?: row.rawTranscript
}

data class DetailLocal(
    val exporting: Boolean = false,
    val exportResult: String? = null,
    val exportError: String? = null,
    val editError: String? = null,
    val saved: Set<String> = emptySet(),
)

data class SessionDetailUiState(
    val session: SessionRow? = null,
    val items: List<DetailItem> = emptyList(),
    val loading: Boolean = true,
    val local: DetailLocal = DetailLocal(),
) {
    val canExport: Boolean get() = items.any { it.hasAudio } && !local.exporting

    /** Why Export is disabled, or null when it is available. */
    val exportDisabledReason: String? get() = if (items.none { it.hasAudio }) NO_AUDIO else null

    fun isSaved(item: DetailItem, phrase: FlaggedPhrase) = "${item.row.id}:${phrase.phrase.lowercase()}" in local.saved

    companion object {
        const val NO_AUDIO = "No training audio in this session (turn on Training capture before listening)"
    }
}

sealed interface DetailEffect {
    data class Share(val uri: Uri) : DetailEffect
}

/** One past session (design section 6, SessionDetail). */
class SessionDetailViewModel(
    private val sessionId: Long,
    private val history: HistoryRepository,
    private val vocab: VocabRepository,
    private val exports: ExportPort,
) : ViewModel() {
    private val local = MutableStateFlow(DetailLocal())
    private val _effects = Channel<DetailEffect>(Channel.BUFFERED)
    val effects: Flow<DetailEffect> = _effects.receiveAsFlow()

    private val sessionFlow = combine(flow { emit(history.session(sessionId)) }, history.sessions()) { first, all ->
        all.firstOrNull { it.id == sessionId } ?: first
    }

    val ui: StateFlow<SessionDetailUiState> = combine(sessionFlow, history.exchanges(sessionId), local) { s, rows, loc ->
        SessionDetailUiState(s, rows.map { DetailItem(it, it.toCard()) }, loading = false, local = loc)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SessionDetailUiState())

    /** Suggested SAF file name, or null before the session has loaded. */
    fun suggestedName(): String? = ui.value.session?.let { exports.suggestedName(it.startedAt) }

    /** Saves the trimmed text as the corrected transcript; blank removes the correction. */
    fun editTranscript(exchangeId: Long, text: String) {
        viewModelScope.launch {
            try {
                history.setCorrectedTranscript(exchangeId, text.trim())
                local.update { it.copy(editError = null) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                local.update { it.copy(editError = "Couldn't save the transcript: ${e.message}") }
            }
        }
    }

    /** SAF target picked by the user; a cancelled picker never calls this. */
    fun exportTo(uri: Uri) = runExport { SummaryOnly(exports.exportTo(sessionId, uri)) }

    fun share() {
        val name = suggestedName() ?: return
        runExport {
            val r = exports.exportForShare(sessionId, name)
            _effects.send(DetailEffect.Share(r.uri))
            SummaryOnly(r.summary)
        }
    }

    private class SummaryOnly(val summary: ExportSummary)

    private fun runExport(block: suspend () -> SummaryOnly) {
        val state = ui.value
        if (state.local.exporting || state.items.none { it.hasAudio }) return
        local.update { it.copy(exporting = true, exportError = null, exportResult = null) }
        viewModelScope.launch {
            try {
                val s = block().summary
                local.update { it.copy(exporting = false, exportResult = resultText(s)) }
            } catch (e: CancellationException) {
                local.update { it.copy(exporting = false) }
                throw e
            } catch (e: Exception) {
                HLog.w(HLog.UI, "export failed: ${e.javaClass.simpleName}")
                local.update { it.copy(exporting = false, exportError = e.message ?: e.javaClass.simpleName) }
            }
        }
    }

    fun saveVocab(item: DetailItem, phrase: FlaggedPhrase) {
        val key = "${item.row.id}:${phrase.phrase.lowercase()}"
        if (key in local.value.saved) return
        viewModelScope.launch {
            try {
                vocab.save(phrase, item.row.id, item.row.correctedSource ?: item.row.rawTranscript)
                local.update { it.copy(saved = it.saved + key) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                HLog.w(HLog.UI, "vocab save failed: ${e.message}")
            }
        }
    }

    companion object {
        fun resultText(s: ExportSummary): String {
            val base = "${s.segmentsWritten} segments, ${s.segmentsMissing} missing"
            return if (s.correctedIncluded) "$base (corrected transcripts included)" else base
        }
    }
}

/** Stored exchange as the Live card shape; a failed translation keeps the raw transcript. */
internal fun ExchangeRow.toCard(): ExchangeCard = ExchangeCard(
    seq = segmentId,
    exchangeId = id,
    createdAtMs = timestamp.toEpochMilli(),
    rawTranscript = rawTranscript,
    detectedLanguage = detectedLanguage,
    direction = direction,
    translation = if (translationError) {
        CardTranslation.Failed(LlmException.Kind.SERVER, "")
    } else {
        CardTranslation.Done(
            TranslationResult(
                corrected = correctedSource ?: rawTranscript,
                translated = translation,
                flaggedPhrases = flagged,
                confidence = confidence ?: 0.0,
                isCorrection = isCorrection,
                correctionDetail = correction,
            ),
            flagged,
            model = "",
            processingMs = processingMs ?: 0,
        )
    },
)
