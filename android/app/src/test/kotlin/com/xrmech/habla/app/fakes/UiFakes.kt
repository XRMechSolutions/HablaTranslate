package com.xrmech.habla.app.fakes

import com.xrmech.habla.app.pipeline.LiveController
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.service.SessionControl
import com.xrmech.habla.app.translate.LanguagePack
import com.xrmech.habla.app.translate.PackState
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.ModelDownloader
import java.io.IOException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/** LiveController whose state the test sets directly. */
class FakeLiveController(initial: LiveState = LiveState()) : LiveController {
    override val state = MutableStateFlow(initial)
    override val lastMetricsLine = MutableStateFlow<String?>(null)
}

/** Records start/stop requests instead of starting the foreground service. */
class FakeSessionControl : SessionControl {
    var starts = 0
    var stops = 0

    override fun start() {
        starts++
    }

    override fun stop() {
        stops++
    }
}

/** A downloader that must never be reached (tests that only read disk state). */
object NoNetworkDownloader : ModelDownloader {
    override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
        throw IOException("no network in this test")
}

/** Makes [entry] look downloaded and verified to ModelManager (sparse file of the catalog size + marker). */
fun installVerifiedModel(filesDir: java.io.File, entry: com.xrmech.habla.shared.AsrModelEntry): java.io.File {
    val dir = java.io.File(filesDir, "models").apply { mkdirs() }
    val file = java.io.File(dir, entry.file)
    java.io.RandomAccessFile(file, "rw").use { it.setLength(entry.sizeBytes) }
    java.io.File(dir, entry.file + ".verified").writeText("${entry.sha256} ${entry.sizeBytes} ${file.lastModified()}\n")
    return file
}

/** In-memory language pack (D050): [download] goes straight to Ready; counts calls. */
class FakeLanguagePack(initial: PackState = PackState.Missing) : LanguagePack {
    private val _state = MutableStateFlow(initial)
    override val state: StateFlow<PackState> = _state
    var downloads = 0
        private set
    var deletes = 0
        private set

    override suspend fun refresh() = Unit

    override suspend fun download() {
        downloads++
        _state.value = PackState.Ready
    }

    override suspend fun delete() {
        deletes++
        _state.value = PackState.Missing
    }
}
