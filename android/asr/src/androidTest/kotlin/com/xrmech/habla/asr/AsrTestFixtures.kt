package com.xrmech.habla.asr

import android.content.Context
import android.os.ParcelFileDescriptor
import androidx.test.platform.app.InstrumentationRegistry
import com.xrmech.habla.shared.AsrModelEntry
import com.xrmech.habla.shared.ModelCatalog
import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.LanguageRegistry
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.text.Normalizer
import java.util.Locale
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Shared helpers for the emulator suites: test model, shared assets, clips and WER. */
object AsrTestFixtures {
    const val DEVICE_MODEL_DIR = "/data/local/tmp/habla-models"
    const val SAMPLE_RATE = 16_000

    val context: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext: Context get() = InstrumentationRegistry.getInstrumentation().context

    private val json = Json { ignoreUnknownKeys = false }

    private fun asset(path: String): String = testContext.assets.open(path).use { it.readBytes().toString(Charsets.UTF_8) }

    // Decoded individually rather than through SharedAssets: SharedAssets currently fails on ART
    // (IdiomRegex pattern rejected by ICU; reported to shared-assets), and this suite needs only these two.
    private val sharedJson = Json { ignoreUnknownKeys = false; explicitNulls = false }
    val profile: AsrProfile by lazy { sharedJson.decodeFromString(AsrProfile.serializer(), asset("shared/asr_profile.json")) }
    val languages: LanguageRegistry by lazy { sharedJson.decodeFromString(LanguageRegistry.serializer(), asset("shared/languages.json")) }

    val testCatalog: ModelCatalog by lazy { json.decodeFromString(ModelCatalog.serializer(), asset("test_models.json")) }

    val tinyEntry: AsrModelEntry get() = testCatalog.default

    /**
     * Copies the adb-pushed test model into `noBackupFilesDir/models/` (apps cannot read
     * /data/local/tmp under SELinux, the shell user can) and checks its sha256.
     */
    fun installTestModel(): File {
        val entry = tinyEntry
        val dir = File(context.noBackupFilesDir, "models").apply { mkdirs() }
        val dst = File(dir, entry.file)
        if (dst.length() == entry.sizeBytes && sha256(dst) == entry.sha256) return dst
        val pfd = InstrumentationRegistry.getInstrumentation().uiAutomation
            .executeShellCommand("cat $DEVICE_MODEL_DIR/${entry.file}")
        ParcelFileDescriptor.AutoCloseInputStream(pfd).use { input ->
            FileOutputStream(dst).use { input.copyTo(it, 1 shl 20) }
        }
        check(dst.length() == entry.sizeBytes && sha256(dst) == entry.sha256) {
            "test model ${entry.file} missing or corrupt on the device (${dst.length()} bytes); " +
                "run android\\gradlew.bat -p android :asr:prepareTestModels"
        }
        return dst
    }

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

    /** 16 kHz mono PCM16 WAV from the androidTest assets as floats in [-1, 1]. */
    fun clip(name: String): FloatArray {
        val bytes = testContext.assets.open("clips/$name").use { it.readBytes() }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        check(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" && String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") { "$name is not WAV" }
        var pos = 12
        var data: FloatArray? = null
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = buf.getInt(pos + 4)
            val body = pos + 8
            when (id) {
                "fmt " -> {
                    check(buf.getShort(body).toInt() == 1 && buf.getShort(body + 2).toInt() == 1 &&
                        buf.getInt(body + 4) == SAMPLE_RATE && buf.getShort(body + 14).toInt() == 16) {
                        "$name must be 16 kHz mono PCM16"
                    }
                }
                "data" -> data = FloatArray(size / 2) { buf.getShort(body + it * 2) / 32768f }
            }
            pos = body + size + (size and 1)
        }
        return checkNotNull(data) { "$name has no data chunk" }
    }

    data class Utterance(val clip: String, val index: Int, val samples: FloatArray, val transcript: String, val language: String)

    /** Cuts utterance [index] of [clip] at its golden segment range (shared/testdata/goldens). */
    fun utterance(clip: String, index: Int): Utterance {
        val golden = Json.parseToJsonElement(asset("goldens/${clip.removeSuffix(".wav")}.json")).jsonObject
        val segments = golden.getValue("segments").jsonArray.associateBy { it.jsonObject.getValue("index").jsonPrimitive.int }
        val utt = golden.getValue("utterances").jsonArray.first { it.jsonObject.getValue("index").jsonPrimitive.int == index }.jsonObject
        val segIdx = utt.getValue("segment_indices").jsonArray.map { it.jsonPrimitive.int }
        val start = segments.getValue(segIdx.first()).jsonObject.getValue("start_sample").jsonPrimitive.int
        val end = segments.getValue(segIdx.last()).jsonObject.getValue("end_sample").jsonPrimitive.int
        return Utterance(
            clip,
            index,
            clip(clip).copyOfRange(start, end),
            utt.getValue("transcript").jsonPrimitive.content,
            utt.getValue("detected_language").jsonPrimitive.content,
        )
    }

    /** design.md section 11: NFC, root lowercase, Unicode punctuation -> space, collapse whitespace. */
    fun normalizeWords(text: String): List<String> {
        val nfc = Normalizer.normalize(text, Normalizer.Form.NFC).lowercase(Locale.ROOT)
        val sb = StringBuilder()
        var i = 0
        while (i < nfc.length) {
            val cp = nfc.codePointAt(i)
            val type = Character.getType(cp)
            val punct = type == Character.CONNECTOR_PUNCTUATION.toInt() || type == Character.DASH_PUNCTUATION.toInt() ||
                type == Character.START_PUNCTUATION.toInt() || type == Character.END_PUNCTUATION.toInt() ||
                type == Character.INITIAL_QUOTE_PUNCTUATION.toInt() || type == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
                type == Character.OTHER_PUNCTUATION.toInt()
            if (punct) sb.append(' ') else sb.appendCodePoint(cp)
            i += Character.charCount(cp)
        }
        return sb.split(Regex("\\s+")).filter { it.isNotEmpty() }
    }

    /** Word Levenshtein distance / reference words. */
    fun wer(reference: String, hypothesis: String): Double {
        val r = normalizeWords(reference)
        val h = normalizeWords(hypothesis)
        if (r.isEmpty()) return if (h.isEmpty()) 0.0 else 1.0
        var prev = IntArray(h.size + 1) { it }
        for (i in 1..r.size) {
            val cur = IntArray(h.size + 1)
            cur[0] = i
            for (j in 1..h.size) {
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (r[i - 1] == h[j - 1]) 0 else 1)
            }
            prev = cur
        }
        return prev[h.size].toDouble() / r.size
    }
}
