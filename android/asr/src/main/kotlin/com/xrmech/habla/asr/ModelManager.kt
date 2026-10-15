package com.xrmech.habla.asr

import com.xrmech.habla.shared.AsrModelEntry
import com.xrmech.habla.shared.ModelCatalog
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.CoroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json

/** Download/verification state of the model most recently passed to ensure/refresh/delete. */
sealed interface ModelState {
    data object Missing : ModelState
    data class Downloading(val bytes: Long, val total: Long) : ModelState
    data object Verifying : ModelState
    data class Ready(val file: File) : ModelState

    /** [reason] is a stable machine code (see [ModelManager]); `:app` maps it to UI text. */
    data class Failed(val reason: String, val retryable: Boolean) : ModelState
}

/**
 * Resumable, SHA-256-verified Whisper model files under `<filesDir>/models/` (design.md section 9).
 * `:app` passes `context.noBackupFilesDir` so hundreds of MB of re-downloadable weights stay out of
 * Auto Backup.
 *
 * Per catalog entry: `<file>` (only ever created by an atomic rename), `<file>.part`,
 * `<file>.part.json` (resume metadata; a part is resumed only when it matches the current entry) and
 * `<file>.verified` (`<sha256> <sizeBytes> <lastModifiedMillis>`, so [fileFor] never re-hashes).
 *
 * Failed reasons: `unknown_model`, `insufficient_storage:<bytesNeeded>`, `network`, `http_<code>`
 * (retryable for 408/429/5xx), `size_mismatch`, `checksum_mismatch`, `io:<ExceptionClass>`.
 *
 * Thread-safe: one [Mutex] serializes ensure/delete; [fileFor] is a stat plus one small read.
 * Main-safe: ensure/delete do their blocking file work (writes, fsync, hashing, deletes) on
 * Dispatchers.IO whatever dispatcher the caller is on.
 */
class ModelManager internal constructor(
    filesDir: File,
    private val catalog: ModelCatalog,
    private val http: ModelDownloader,
    private val usableSpace: (File) -> Long,
    /** Where blocking file work runs; tests pass the test dispatcher so retry delays use virtual time. */
    private val io: CoroutineContext,
) {
    constructor(filesDir: File, catalog: ModelCatalog, http: ModelDownloader) :
        this(filesDir, catalog, http, File::getUsableSpace, Dispatchers.IO)

    private val modelsDir = File(filesDir, MODELS_DIR)
    private val mutex = Mutex()
    private val _state = MutableStateFlow<ModelState>(ModelState.Missing)
    private val active = AtomicReference<Pair<String, Job>?>(null)

    val state: StateFlow<ModelState> = _state.asStateFlow()

    /**
     * Makes [modelId] Ready (download, resume, verify, install) or reports why not in [state]. Returns
     * normally in every non-cancelled case; on cancellation the part is kept for resume, state is
     * Missing and the CancellationException is rethrown.
     */
    suspend fun ensure(modelId: String) {
        val entry = catalog.byId(modelId)
        if (entry == null) {
            _state.value = ModelState.Failed(REASON_UNKNOWN, false)
            return
        }
        mutex.withLock {
            coroutineScope {
                // Child job so delete() can cancel the download without cancelling the caller's scope.
                // Driven by ensureLocked to completion; cancelled by the caller's cancellation or by
                // delete(modelId); errors (only the path assertion) propagate out of coroutineScope.
                val work = launch(io) {
                    try {
                        ensureLocked(entry)
                    } catch (e: CancellationException) {
                        _state.value = ModelState.Missing
                        throw e
                    }
                }
                active.set(modelId to work)
                try {
                    work.join()
                } finally {
                    active.compareAndSet(modelId to work, null)
                }
            }
        }
    }

    /** The verified model file, or null. No hashing: exists + length + marker match only. */
    fun fileFor(modelId: String): File? {
        val entry = catalog.byId(modelId) ?: return null
        val file = File(modelsDir, entry.file)
        return file.takeIf { isVerified(entry, it) }
    }

    /** Sets Ready/Missing from disk without network or hashing; leaves a running download's state. */
    suspend fun refresh(modelId: String) {
        val entry = catalog.byId(modelId)
        if (entry == null) {
            _state.value = ModelState.Failed(REASON_UNKNOWN, false)
            return
        }
        if (active.get()?.first == modelId) return
        _state.value = fileFor(modelId)?.let { ModelState.Ready(it) } ?: ModelState.Missing
    }

    /** Cancels and joins an in-flight download of [modelId], deletes its four files, state Missing. */
    suspend fun delete(modelId: String) {
        active.get()?.takeIf { it.first == modelId }?.second?.let {
            it.cancel()
            it.join()
        }
        mutex.withLock {
            withContext(io) {
                catalog.byId(modelId)?.let { entry -> filesOf(entry).forEach { it.delete() } }
            }
            _state.value = ModelState.Missing
        }
    }

    // ---- ensure, step by step --------------------------------------------------------------

    private suspend fun ensureLocked(entry: AsrModelEntry) {
        val files = filesOf(entry)
        val (final, part, partJson, marker) = files
        if (!modelsDir.isDirectory && !modelsDir.mkdirs()) {
            _state.value = ModelState.Failed("io:IOException", true)
            return
        }

        if (isVerified(entry, final)) {
            _state.value = ModelState.Ready(final)
            return
        }

        // Final file without a valid marker: crash between rename and marker, or the catalog changed.
        if (final.exists()) {
            _state.value = ModelState.Verifying
            val digest = try {
                if (final.length() == entry.sizeBytes) sha256(final) else null
            } catch (e: IOException) {
                _state.value = ModelState.Failed("io:${e.javaClass.simpleName}", true)
                return
            }
            if (digest == entry.sha256) {
                if (!writeMarker(entry, final, marker)) return
                _state.value = ModelState.Ready(final)
                return
            }
            final.delete()
            marker.delete()
        }

        prune()

        var retriesLeft = RETRY_DELAYS_MS.size
        var restartedAfter416 = false
        while (true) {
            val offset = resumeOffset(entry, part, partJson)
            val outcome = if (offset == entry.sizeBytes) {
                Outcome.Complete
            } else {
                val need = entry.sizeBytes - offset + STORAGE_HEADROOM_BYTES
                if (usableSpace(modelsDir) < need) {
                    _state.value = ModelState.Failed("insufficient_storage:$need", true)
                    return
                }
                _state.value = ModelState.Downloading(offset, entry.sizeBytes)
                try {
                    download(entry, offset, part, partJson)
                } catch (e: LocalIoException) {
                    Outcome.Fatal("io:${e.cause.javaClass.simpleName}", retryable = true)
                } catch (e: IOException) {
                    Outcome.Retry(REASON_NETWORK)
                }
            }

            when (outcome) {
                Outcome.Complete -> {
                    install(entry, final, part, partJson, marker)
                    return
                }
                is Outcome.Fatal -> {
                    _state.value = ModelState.Failed(outcome.reason, outcome.retryable)
                    return
                }
                Outcome.RangeNotSatisfiable -> {
                    // offset < size here (offset == size never requests). Restart from 0 once.
                    part.delete()
                    partJson.delete()
                    if (restartedAfter416) {
                        _state.value = ModelState.Failed("http_416", false)
                        return
                    }
                    restartedAfter416 = true
                }
                is Outcome.Retry -> {
                    if (retriesLeft == 0) {
                        _state.value = ModelState.Failed(outcome.reason, true)
                        return
                    }
                    val backoff = RETRY_DELAYS_MS[RETRY_DELAYS_MS.size - retriesLeft]
                    retriesLeft--
                    val wait = outcome.retryAfterSeconds?.let { minOf(it, MAX_RETRY_AFTER_SECONDS) * 1_000 } ?: backoff
                    delay(wait)
                }
            }
        }
    }

    /** Step 5: resume only onto bytes of exactly this catalog entry. */
    private fun resumeOffset(entry: AsrModelEntry, part: File, partJson: File): Long {
        if (part.exists() && readPartMeta(partJson) == PartMeta.of(entry)) {
            val length = part.length()
            if (length <= entry.sizeBytes) return length
        }
        part.delete()
        partJson.delete()
        return 0
    }

    /** Steps 7-8: one HTTP attempt, streaming into the part file. */
    private suspend fun download(entry: AsrModelEntry, offset: Long, part: File, partJson: File): Outcome =
        http.open(entry.url, offset) { response ->
            val status = response.status
            val startAt: Long = when {
                status == 206 -> {
                    val range = parseContentRange(response.contentRange)
                    when {
                        range == null || range.first != offset -> {
                            // A proxy/CDN answered with another range: drop the part, next attempt starts at 0.
                            part.delete()
                            partJson.delete()
                            return@open Outcome.Retry(REASON_NETWORK)
                        }
                        range.second != entry.sizeBytes -> return@open Outcome.Fatal(REASON_SIZE_MISMATCH, false)
                    }
                    offset
                }
                status == 200 -> {
                    val length = response.contentLength
                    if (length != null && length != entry.sizeBytes) return@open Outcome.Fatal(REASON_SIZE_MISMATCH, false)
                    0L // the server ignored Range: start over
                }
                status == 416 -> return@open Outcome.RangeNotSatisfiable
                status == 408 || status == 429 || status in 500..599 ->
                    return@open Outcome.Retry("http_$status", if (status == 429) response.retryAfterSeconds else null)
                else -> return@open Outcome.Fatal("http_$status", false)
            }
            stream(entry, response.body, part, partJson, startAt)
        }

    private suspend fun stream(entry: AsrModelEntry, body: DownloadBody, part: File, partJson: File, startAt: Long): Outcome {
        local { writePartMeta(entry, partJson) }
        val out = local { FileOutputStream(part, startAt > 0) }
        var bytes = startAt
        try {
            val buffer = ByteArray(BUFFER_BYTES)
            var lastBytes = bytes
            var lastNanos = System.nanoTime()
            while (true) {
                currentCoroutineContext().ensureActive()
                val n = body.read(buffer)
                if (n < 0) break
                if (bytes + n > entry.sizeBytes) return Outcome.Fatal(REASON_SIZE_MISMATCH, false)
                local { out.write(buffer, 0, n) }
                bytes += n
                val now = System.nanoTime()
                if (bytes - lastBytes >= PROGRESS_BYTES || now - lastNanos >= PROGRESS_NANOS) {
                    _state.value = ModelState.Downloading(bytes, entry.sizeBytes)
                    lastBytes = bytes
                    lastNanos = now
                }
            }
            local {
                out.flush()
                out.fd.sync()
            }
        } finally {
            try {
                out.close()
            } catch (e: IOException) {
                // Closing after a failed write; the original failure is what matters.
            }
        }
        _state.value = ModelState.Downloading(bytes, entry.sizeBytes)
        return if (bytes == entry.sizeBytes) Outcome.Complete else Outcome.Retry(REASON_NETWORK)
    }

    /** Steps 10-11: hash the whole part, then atomically rename it into place. */
    private suspend fun install(entry: AsrModelEntry, final: File, part: File, partJson: File, marker: File) {
        _state.value = ModelState.Verifying
        val digest = try {
            sha256(part)
        } catch (e: IOException) {
            _state.value = ModelState.Failed("io:${e.javaClass.simpleName}", true)
            return
        }
        if (digest != entry.sha256) {
            part.delete()
            partJson.delete()
            _state.value = ModelState.Failed(REASON_CHECKSUM, true)
            return
        }
        try {
            Files.move(part.toPath(), final.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (e: IOException) {
            _state.value = ModelState.Failed("io:${e.javaClass.simpleName}", true)
            return
        }
        partJson.delete()
        if (!writeMarker(entry, final, marker)) return
        _state.value = ModelState.Ready(final)
    }

    // ---- files -------------------------------------------------------------------------------

    private fun filesOf(entry: AsrModelEntry): List<File> =
        listOf(entry.file, entry.file + PART, entry.file + PART_JSON, entry.file + MARKER).map { safeFile(it) }

    /** Catalog content becomes a file name; never let it leave models/ (shared-assets validates too). */
    private fun safeFile(name: String): File {
        val file = File(modelsDir, name)
        check(file.canonicalFile.parentFile == modelsDir.canonicalFile) { "model file name escapes models/: $name" }
        return file
    }

    /** Step 4: delete everything that is not a current catalog entry's file, part, part.json or marker. */
    private fun prune() {
        val keep = catalog.models.flatMap { listOf(it.file, it.file + PART, it.file + PART_JSON, it.file + MARKER) }.toSet()
        modelsDir.listFiles()?.forEach { f -> if (f.isFile && f.name !in keep) f.delete() }
    }

    private fun isVerified(entry: AsrModelEntry, file: File): Boolean {
        if (!file.isFile || file.length() != entry.sizeBytes) return false
        val marker = File(modelsDir, entry.file + MARKER)
        val text = try {
            marker.readText(Charsets.UTF_8)
        } catch (e: IOException) {
            return false
        }
        return text.trim() == markerLine(entry, file)
    }

    private fun writeMarker(entry: AsrModelEntry, final: File, marker: File): Boolean = try {
        marker.writeText(markerLine(entry, final) + "\n", Charsets.UTF_8)
        true
    } catch (e: IOException) {
        _state.value = ModelState.Failed("io:${e.javaClass.simpleName}", true)
        false
    }

    private fun markerLine(entry: AsrModelEntry, file: File) = "${entry.sha256} ${entry.sizeBytes} ${file.lastModified()}"

    private fun readPartMeta(partJson: File): PartMeta? = try {
        if (partJson.isFile) PART_JSON_CODEC.decodeFromString(PartMeta.serializer(), partJson.readText(Charsets.UTF_8)) else null
    } catch (e: IOException) {
        null
    } catch (e: SerializationException) {
        null
    } catch (e: IllegalArgumentException) {
        null
    }

    private fun writePartMeta(entry: AsrModelEntry, partJson: File) {
        partJson.writeText(PART_JSON_CODEC.encodeToString(PartMeta.serializer(), PartMeta.of(entry)), Charsets.UTF_8)
    }

    private suspend fun sha256(file: File): String = withContext(io) {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(HASH_BUFFER_BYTES)
            while (true) {
                ensureActive()
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        digest.digest().joinToString("") { "%02x".format(it) }
    }

    /** Marks local file failures so they map to `io:<Class>` instead of `network`. */
    private inline fun <T> local(block: () -> T): T = try {
        block()
    } catch (e: IOException) {
        throw LocalIoException(e)
    }

    private class LocalIoException(override val cause: IOException) : Exception(cause)

    private sealed interface Outcome {
        data object Complete : Outcome
        data object RangeNotSatisfiable : Outcome
        data class Retry(val reason: String, val retryAfterSeconds: Long? = null) : Outcome
        data class Fatal(val reason: String, val retryable: Boolean) : Outcome
    }

    @Serializable
    internal data class PartMeta(
        val url: String,
        val revision: String,
        val sha256: String,
        @SerialName("size_bytes") val sizeBytes: Long,
    ) {
        companion object {
            fun of(e: AsrModelEntry) = PartMeta(e.url, e.revision, e.sha256, e.sizeBytes)
        }
    }

    internal companion object {
        const val MODELS_DIR = "models"
        const val PART = ".part"
        const val PART_JSON = ".part.json"
        const val MARKER = ".verified"

        const val REASON_UNKNOWN = "unknown_model"
        const val REASON_NETWORK = "network"
        const val REASON_SIZE_MISMATCH = "size_mismatch"
        const val REASON_CHECKSUM = "checksum_mismatch"

        /** Headroom so the app's database and DataStore writes keep working after the download. */
        const val STORAGE_HEADROOM_BYTES = 64L * 1024 * 1024

        /** Delays before the three automatic retries of a retryable failure. */
        val RETRY_DELAYS_MS = longArrayOf(2_000, 8_000, 30_000)
        const val MAX_RETRY_AFTER_SECONDS = 60L

        const val BUFFER_BYTES = 64 * 1024
        const val HASH_BUFFER_BYTES = 1024 * 1024

        /** Progress at most every 1 MiB or 500 ms (~264 emissions for q8_0), and always at the end. */
        const val PROGRESS_BYTES = 1024L * 1024
        const val PROGRESS_NANOS = 500_000_000L

        private val PART_JSON_CODEC = Json { ignoreUnknownKeys = false }

        /** `bytes <start>-<end>/<total>` -> (start, total); null when malformed or total is `*`. */
        fun parseContentRange(value: String?): Pair<Long, Long>? {
            val m = CONTENT_RANGE.matchEntire(value?.trim() ?: return null) ?: return null
            val start = m.groupValues[1].toLongOrNull() ?: return null
            val end = m.groupValues[2].toLongOrNull() ?: return null
            val total = m.groupValues[3].toLongOrNull() ?: return null
            if (end < start || end >= total) return null
            return start to total
        }

        private val CONTENT_RANGE = Regex("bytes ([0-9]+)-([0-9]+)/([0-9]+)")
    }
}
