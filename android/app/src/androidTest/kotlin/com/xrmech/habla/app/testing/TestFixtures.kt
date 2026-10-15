package com.xrmech.habla.app.testing

import android.content.Context
import android.os.ParcelFileDescriptor
import androidx.test.platform.app.InstrumentationRegistry
import com.xrmech.habla.app.di.AndroidAssetSource
import com.xrmech.habla.shared.AsrModelEntry
import com.xrmech.habla.shared.ModelCatalog
import com.xrmech.habla.shared.SharedAssets
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream
import java.security.MessageDigest
import kotlinx.serialization.json.Json

/** Emulator test fixtures: the app's real shared assets, the synced clips/goldens and the tiny model. */
object TestFixtures {
    val context: Context get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** The androidTest APK (clips, goldens, models/ are its assets). */
    val testContext: Context get() = InstrumentationRegistry.getInstrumentation().context

    /** Built from the app APK assets through the production AssetSource. */
    val shared: SharedAssets by lazy { SharedAssets(AndroidAssetSource(context.assets)) }

    /** asr's test-only catalog (D037 a): ggml-tiny-q8_0 is never in shared/asr_models.json. */
    val tinyCatalog: ModelCatalog by lazy {
        Json.decodeFromString(ModelCatalog.serializer(), testAsset("models/test_models.json").readBytes().toString(Charsets.UTF_8))
    }

    val tinyEntry: AsrModelEntry get() = tinyCatalog.default

    fun testAsset(path: String): InputStream = testContext.assets.open(path)

    fun clip(name: String): InputStream = testAsset("clips/$name")

    fun golden(name: String): String = testAsset("goldens/$name").use { it.readBytes().toString(Charsets.UTF_8) }

    /**
     * Copies the tiny model from the androidTest assets to `<filesDir>/models/` with the
     * ModelManager marker, so `fileFor` reports it Ready without hashing.
     */
    fun installTinyModel(filesDir: File = context.noBackupFilesDir): File {
        val entry = tinyEntry
        val dir = File(filesDir, "models").apply { mkdirs() }
        val dst = File(dir, entry.file)
        if (dst.length() != entry.sizeBytes || sha256(dst) != entry.sha256) {
            val input = try {
                testAsset("models/${entry.file}")
            } catch (e: FileNotFoundException) {
                // Built with -Phabla.testModelInApk=false: the same pinned file pushed by adb. Apps cannot
                // read /data/local/tmp under SELinux; the shell user can.
                ParcelFileDescriptor.AutoCloseInputStream(
                    InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand("cat $DEVICE_MODEL_DIR/${entry.file}"),
                )
            }
            input.use { i -> dst.outputStream().use { i.copyTo(it, 1 shl 20) } }
            if (dst.length() != entry.sizeBytes || sha256(dst) != entry.sha256) {
                dst.delete()
                throw AssertionError(
                    "tiny model ${entry.file} missing: run gradlew.bat :app:fetchTestWhisperTiny " +
                        "(or adb push it to $DEVICE_MODEL_DIR when the test APK is built without it)",
                )
            }
        }
        File(dir, entry.file + ".verified").writeText("${entry.sha256} ${entry.sizeBytes} ${dst.lastModified()}\n")
        return dst
    }

    const val DEVICE_MODEL_DIR = "/data/local/tmp/habla-models"

    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
