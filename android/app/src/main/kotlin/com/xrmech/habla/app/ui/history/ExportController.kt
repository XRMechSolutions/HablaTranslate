package com.xrmech.habla.app.ui.history

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import androidx.core.content.FileProvider
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.data.ExportSummary
import com.xrmech.habla.data.TrainingExporter
import java.io.File
import java.io.IOException
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

/** A finished share export: the FileProvider URI and the counts for the result line. */
data class SharedExport(val uri: Uri, val summary: ExportSummary)

/** What SessionDetailViewModel needs from exporting, so its test can fake it. */
interface ExportPort {
    /** Writes the zip to a SAF document; a failed write deletes the partial document. */
    suspend fun exportTo(sessionId: Long, uri: Uri): ExportSummary

    /** Writes `cacheDir/exports/<name>` and returns its FileProvider URI. */
    suspend fun exportForShare(sessionId: Long, name: String): SharedExport

    fun suggestedName(startedAtUtc: Instant): String
}

/**
 * Training export to a user-chosen document (SAF) or the share sheet (design section 6,
 * SessionDetail). Errors (`IOException`, `ExportException`) propagate to the ViewModel, which shows
 * them verbatim.
 */
class ExportController(
    private val context: Context,
    private val exporter: TrainingExporter,
    private val io: CoroutineDispatcher,
) : ExportPort {

    override suspend fun exportTo(sessionId: Long, uri: Uri): ExportSummary = withContext(io) {
        val resolver = context.contentResolver
        try {
            val out = resolver.openOutputStream(uri, "wt") ?: throw IOException("Couldn't open the chosen file for writing")
            out.use { exporter.export(sessionId, it) }
        } catch (e: Throwable) {
            deletePartial(uri)
            throw e
        }
    }

    override suspend fun exportForShare(sessionId: Long, name: String): SharedExport = withContext(io) {
        val dir = File(context.cacheDir, EXPORT_DIR)
        if (!dir.isDirectory && !dir.mkdirs()) throw IOException("Couldn't create ${dir.path}")
        val file = File(dir, name)
        val summary = try {
            file.outputStream().use { exporter.export(sessionId, it) }
        } catch (e: Throwable) {
            file.delete()
            throw e
        }
        SharedExport(FileProvider.getUriForFile(context, "${context.packageName}.exports", file), summary)
    }

    override fun suggestedName(startedAtUtc: Instant): String = nameFor(startedAtUtc)

    private fun deletePartial(uri: Uri) {
        try {
            DocumentsContract.deleteDocument(context.contentResolver, uri)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            HLog.w(HLog.UI, "partial export not deleted: ${e.javaClass.simpleName}")
        }
    }

    companion object {
        const val EXPORT_DIR = "exports"
        const val MAX_AGE_MS = 24L * 60 * 60 * 1000
        private val NAME_TIME = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(ZoneOffset.UTC)

        /** "habla-20261015T201500Z.zip" */
        fun nameFor(startedAtUtc: Instant): String = "habla-${NAME_TIME.format(startedAtUtc)}.zip"

        /** Share sheet for an exported zip; the chooser shows the system's empty state when no app fits. */
        fun shareChooser(uri: Uri): Intent {
            val send = Intent(Intent.ACTION_SEND)
                .setType("application/zip")
                .putExtra(Intent.EXTRA_STREAM, uri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            return Intent.createChooser(send, "Share training export").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        /** Deletes `cacheDir/exports` files older than 24 h (run at app start). Returns how many. */
        fun cleanCache(cacheDir: File, nowMs: Long = System.currentTimeMillis()): Int {
            val files = File(cacheDir, EXPORT_DIR).listFiles() ?: return 0
            return files.count { it.isFile && nowMs - it.lastModified() > MAX_AGE_MS && it.delete() }
        }
    }
}
