package com.xrmech.habla.app.ui.live

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.pipeline.ExchangeCard
import com.xrmech.habla.app.pipeline.LiveController
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.service.SessionControl
import com.xrmech.habla.app.state.Engines
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.state.ModelDownloadController
import com.xrmech.habla.app.state.NoProAccess
import com.xrmech.habla.app.state.ProAccess
import com.xrmech.habla.app.translate.LanguagePack
import com.xrmech.habla.app.translate.PackState
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.asr.ModelState
import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.data.VocabRepository
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.TranslateEngine
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** data-models section 1 `PipelineStatus` (+ `online`, CCR-6), assembled for the status row. */
data class PipelineStatus(
    val listening: Boolean = false,
    val calibrating: Boolean = false,
    val queueDepth: Int = 0,
    val modelState: ModelState = ModelState.Missing,
    val keyState: KeyState = KeyState.UNKNOWN,
    val lastError: String? = null,
    val online: Boolean = true,
)

/** What the screen knows that the ViewModel cannot read itself (permissions live in the Activity). */
data class LiveLocal(
    val micPermission: MicPermission = MicPermission.NOT_ASKED,
    val notificationsGranted: Boolean = true,
    /** Listen was tapped and the service has not reported a session yet. */
    val startRequested: Boolean = false,
    val keySkipped: Boolean = false,
    /** "<seq>:<phrase>" saved from this screen, so the chip shows "Saved". */
    val saved: Set<String> = emptySet(),
    val vocabError: String? = null,
    val confirmMeteredDownload: Boolean = false,
)

data class SetupRows(
    val micDone: Boolean,
    val notificationsDone: Boolean,
    val modelDone: Boolean,
    val keyDone: Boolean,
) {
    val allDone: Boolean get() = micDone && notificationsDone && modelDone && keyDone
}

data class LiveUiState(
    val live: LiveState = LiveState(),
    val settings: AppSettings? = null,
    val status: PipelineStatus = PipelineStatus(),
    val banners: BannerSet = BannerSet.EMPTY,
    val allBanners: List<Banner> = emptyList(),
    val modelSizeBytes: Long = 0,
    val local: LiveLocal = LiveLocal(),
    val setup: SetupRows? = null,
) {
    val canListen: Boolean get() = !live.phase.active && !local.startRequested
    val canStop: Boolean get() = live.phase.active

    /** "Auto (ES)" once the voter has a language (design section 6, Live). */
    val directionLabel: String
        get() = when (settings?.direction) {
            Direction.ES_TO_EN -> "ES -> EN"
            Direction.EN_TO_ES -> "EN -> ES"
            Direction.AUTO -> live.votedLanguage?.let { "Auto (${it.uppercase()})" } ?: "Auto"
            null -> ""
        }

    fun isSaved(card: ExchangeCard, phrase: FlaggedPhrase): Boolean = LiveViewModel.savedKey(card, phrase) in local.saved
}

sealed interface LiveEffect {
    data object RequestMicPermission : LiveEffect
    data object OpenAppSettings : LiveEffect
    data class ScrollToBanner(val kind: BannerKind) : LiveEffect
}

/** Live screen (design section 6, Live; interfaces.md @ ui LiveViewModel). */
class LiveViewModel(
    private val live: LiveController,
    private val service: SessionControl,
    private val settings: SettingsRepository,
    private val models: ModelManager,
    private val downloads: ModelDownloadController,
    private val keyHealth: KeyHealthTracker,
    private val online: StateFlow<Boolean>,
    private val vocab: VocabRepository,
    private val assets: SharedAssets,
    /** On-device translation pack (D050): drives the pack banner and whether a key is needed. */
    private val pack: LanguagePack,
    /** Pro entitlement (D051): AUTO prefers the gateway, which needs no key. */
    private val pro: ProAccess = NoProAccess,
    /** How long a Listen tap blocks a second one if the service never reports back. */
    private val startTimeoutMs: Long = 10_000,
) : ViewModel() {
    private val local = MutableStateFlow(LiveLocal())
    private val _effects = Channel<LiveEffect>(Channel.BUFFERED)
    val effects: Flow<LiveEffect> = _effects.receiveAsFlow()
    private var startGuard: Job? = null

    private val inputs = combine(
        combine(live.state, models.state, keyHealth.state, online, settings.settings) { l, m, k, o, s -> Inputs(l, m, k, o, s) },
        pack.state,
        pro.active,
    ) { i, p, a -> i.copy(pack = p, proActive = a) }

    val ui: StateFlow<LiveUiState> = combine(inputs, local) { i, loc -> build(i, loc) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), LiveUiState())

    init {
        viewModelScope.launch {
            val id = settings.settings.first().asrModelId
            if (!downloads.running.value) downloads.refresh(id)
        }
        // A session that reached the coordinator (or failed there) releases the Listen guard.
        viewModelScope.launch {
            live.state.collect { s ->
                if (s.phase.active || s.phase is SessionPhase.Failed) clearStartRequest()
            }
        }
    }

    private data class Inputs(
        val live: LiveState,
        val model: ModelState,
        val key: KeyState,
        val online: Boolean,
        val settings: AppSettings,
        val pack: PackState = PackState.Unknown,
        val proActive: Boolean = false,
    )

    private fun build(i: Inputs, loc: LiveLocal): LiveUiState {
        val entry = assets.asrModels.byId(i.settings.asrModelId) ?: assets.asrModels.default
        val final = Engines.resolve(
            i.settings.finalEngine, i.settings.onDevicePreviews,
            hasKey = i.key != KeyState.ABSENT, packReady = i.pack == PackState.Ready, proActive = i.proActive,
        ).first
        val all = BannerReducer.reduce(
            LiveUiInputs(
                i.live, i.model, entry.sizeBytes, i.key, i.online, loc.micPermission, i.settings.finalModel,
                translateOnDevice = final == TranslateEngine.ON_DEVICE,
                keyNeeded = final == TranslateEngine.CLOUD,
                packState = i.pack,
            ),
        )
        val status = PipelineStatus(
            listening = i.live.phase == SessionPhase.Listening,
            calibrating = i.live.phase == SessionPhase.Calibrating,
            queueDepth = i.live.queueDepth,
            modelState = i.model,
            keyState = i.key,
            lastError = (i.live.phase as? SessionPhase.Failed)?.error?.message ?: i.live.lastAsrError,
            online = i.online,
        )
        val setup = if (i.settings.firstRunDone) {
            null
        } else {
            SetupRows(
                micDone = loc.micPermission == MicPermission.GRANTED,
                notificationsDone = loc.notificationsGranted,
                modelDone = i.model is ModelState.Ready,
                keyDone = loc.keySkipped || i.key == KeyState.PRESENT,
            )
        }
        return LiveUiState(i.live, i.settings, status, BannerReducer.cap(all), all, entry.sizeBytes, loc, setup)
    }

    // ---- inputs from the Activity -----------------------------------------------------------

    fun setMicPermission(p: MicPermission) {
        local.update { it.copy(micPermission = p) }
        maybeFinishSetup()
    }

    fun setNotificationsGranted(granted: Boolean) {
        local.update { it.copy(notificationsGranted = granted) }
        maybeFinishSetup()
    }

    // ---- Listen / Stop ----------------------------------------------------------------------

    /** Call from the Listen click handler (FGS while-in-use rule: the activity is resumed). */
    fun onListen() {
        val phase = live.state.value.phase
        if (phase.active || local.value.startRequested) return
        when (local.value.micPermission) {
            MicPermission.GRANTED -> Unit
            MicPermission.PERMANENTLY_DENIED -> {
                _effects.trySend(LiveEffect.OpenAppSettings)
                return
            }
            else -> {
                _effects.trySend(LiveEffect.RequestMicPermission)
                return
            }
        }
        val model = models.state.value
        if (model !is ModelState.Ready) {
            val kind = when (model) {
                is ModelState.Downloading, ModelState.Verifying -> BannerKind.MODEL_DOWNLOADING
                is ModelState.Failed -> BannerKind.MODEL_FAILED
                else -> BannerKind.MODEL_MISSING
            }
            _effects.trySend(LiveEffect.ScrollToBanner(kind))
            return
        }
        local.update { it.copy(startRequested = true) }
        startGuard?.cancel()
        startGuard = viewModelScope.launch {
            delay(startTimeoutMs)
            clearStartRequest()
        }
        HLog.i(HLog.UI, "listen tapped")
        service.start()
    }

    fun onStop() {
        if (!live.state.value.phase.active && !local.value.startRequested) return
        HLog.i(HLog.UI, "stop tapped")
        service.stop()
    }

    private fun clearStartRequest() {
        if (local.value.startRequested) local.update { it.copy(startRequested = false) }
        startGuard?.cancel()
        startGuard = null
    }

    // ---- selectors, vocab, setup, model -----------------------------------------------------

    fun setDirection(d: Direction) {
        viewModelScope.launch { settings.update { it.copy(direction = d) } }
    }

    fun setMode(m: Mode) {
        viewModelScope.launch { settings.update { it.copy(mode = m) } }
    }

    fun saveVocab(card: ExchangeCard, phrase: FlaggedPhrase) {
        val key = savedKey(card, phrase)
        if (key in local.value.saved) return
        val sentence = (card.translation as? com.xrmech.habla.app.pipeline.CardTranslation.Done)?.result?.corrected ?: card.rawTranscript
        viewModelScope.launch {
            try {
                vocab.save(phrase, card.exchangeId, sentence)
                local.update { it.copy(saved = it.saved + key, vocabError = null) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                HLog.w(HLog.UI, "vocab save failed: ${e.message}")
                local.update { it.copy(vocabError = "Couldn't save \"${phrase.phrase}\": ${e.message}") }
            }
        }
    }

    fun skipKey() {
        local.update { it.copy(keySkipped = true) }
        maybeFinishSetup()
    }

    fun dismissSetup() {
        viewModelScope.launch { settings.update { it.copy(firstRunDone = true) } }
    }

    private fun maybeFinishSetup() {
        viewModelScope.launch {
            val s = settings.settings.first()
            if (s.firstRunDone) return@launch
            val rows = build(Inputs(live.state.value, models.state.value, keyHealth.state.value, online.value, s), local.value).setup
            if (rows?.allDone == true) dismissSetup()
        }
    }

    /** Download the selected model; on a metered network the first call only asks. */
    fun downloadModel(confirmed: Boolean = false) {
        if (!confirmed && downloads.isMetered()) {
            local.update { it.copy(confirmMeteredDownload = true) }
            return
        }
        local.update { it.copy(confirmMeteredDownload = false) }
        viewModelScope.launch { downloads.download(settings.settings.first().asrModelId) }
    }

    fun dismissMeteredDownload() = local.update { it.copy(confirmMeteredDownload = false) }

    fun cancelDownload() = downloads.cancel()

    /** MODEL_FAILED for a non-retryable reason (checksum/size): remove the files, then download again. */
    fun deleteAndRetry() {
        if (live.state.value.phase.active) return
        viewModelScope.launch {
            val id = settings.settings.first().asrModelId
            downloads.delete(id)
            downloadModel()
        }
    }

    companion object {
        fun savedKey(card: ExchangeCard, phrase: FlaggedPhrase): String = "${card.seq}:${phrase.phrase.lowercase()}"
    }
}
