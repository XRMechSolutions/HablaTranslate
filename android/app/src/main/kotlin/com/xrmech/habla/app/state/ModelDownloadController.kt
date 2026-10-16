package com.xrmech.habla.app.state

import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.asr.ModelManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * One model download at a time, in the app scope so it survives screen changes (design section 8).
 * Progress and failure live in `ModelManager.state`; Cancel keeps the part file for Range resume.
 */
class ModelDownloadController(
    private val models: ModelManager,
    private val appScope: CoroutineScope,
    /** `ConnectivityManager.isActiveNetworkMetered`; the UI asks before downloading on a metered network. */
    private val meteredCheck: () -> Boolean = { false },
) {
    private var job: Job? = null
    private val _running = MutableStateFlow(false)
    val running: StateFlow<Boolean> = _running.asStateFlow()

    /**
     * The model `ModelManager.state` currently describes (the last one downloaded, refreshed or
     * deleted here), or null before any. With a speech model and a live model (D045) the UI shows
     * `state` only for this id and reads the other from disk.
     */
    private val _stateOwner = MutableStateFlow<String?>(null)
    val stateOwner: StateFlow<String?> = _stateOwner.asStateFlow()

    fun isMetered(): Boolean = meteredCheck()

    /** Single-flight: a call while a download runs is ignored. */
    @Synchronized
    fun download(modelId: String) {
        if (job?.isActive == true) return
        _stateOwner.value = modelId
        _running.value = true
        // Driven by ModelManager.ensure to Ready/Failed; cancelled by cancel()/delete() or the app
        // scope; ensure reports every non-cancel error in ModelManager.state, so nothing is thrown here.
        job = appScope.launch {
            try {
                models.ensure(modelId)
            } finally {
                _running.value = false
            }
        }
    }

    @Synchronized
    fun cancel() {
        job?.cancel()
    }

    /** Cancels a running download and removes every file of [modelId]. */
    suspend fun delete(modelId: String) {
        val running = synchronized(this) { job }
        running?.cancel()
        running?.join()
        _stateOwner.value = modelId
        models.delete(modelId)
        HLog.i(HLog.UI, "deleted speech model $modelId")
    }

    /** Sets ModelManager.state from disk (Ready/Missing) for the selected model, without network. */
    suspend fun refresh(modelId: String) {
        // A running download keeps ownership: refresh leaves its state untouched.
        if (_running.value) return
        _stateOwner.value = modelId
        models.refresh(modelId)
    }
}
