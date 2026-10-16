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

    /** The app's model catalog (`shared/asr_models.json`): the small models D5/D6 measure. */
    val appCatalog: ModelCatalog by lazy { sharedJson.decodeFromString(ModelCatalog.serializer(), asset("shared/asr_models.json")) }

    fun installTestModel(): File = installModel(tinyEntry, ":asr:prepareTestModels")

    /**
     * Copies the adb-pushed model [entry] into `noBackupFilesDir/models/` (apps cannot read
     * /data/local/tmp under SELinux, the shell user can) and checks its sha256.
     */
    fun installModel(entry: AsrModelEntry, pushTask: String = ":asr:prepareDeviceBench"): File {
        val dir = File(context.noBackupFilesDir, "models").apply { mkdirs() }
        val dst = File(dir, entry.file)
        if (dst.length() == entry.sizeBytes && sha256(dst) == entry.sha256) return dst
        ParcelFileDescriptor.AutoCloseInputStream(shell("cat $DEVICE_MODEL_DIR/${entry.file}")).use { input ->
            FileOutputStream(dst).use { input.copyTo(it, 1 shl 20) }
        }
        check(dst.length() == entry.sizeBytes && sha256(dst) == entry.sha256) {
            "model ${entry.file} missing or corrupt on the device (${dst.length()} bytes); " +
                "run android\\gradlew.bat -p android $pushTask"
        }
        return dst
    }

    private fun shell(command: String): ParcelFileDescriptor =
        InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand(command)

    /** Bytes of a file only the shell user can read (adb-pushed under /data/local/tmp); empty when absent. */
    fun readDeviceFile(path: String): ByteArray =
        ParcelFileDescriptor.AutoCloseInputStream(shell("cat $path")).use { it.readBytes() }

    /**
     * Writes [text] to [path] as the shell user, so a report survives the test APK being uninstalled
     * after the run and can be fetched with `adb pull`.
     */
    fun writeDeviceFile(path: String, text: String) {
        val dir = path.substringBeforeLast('/')
        ParcelFileDescriptor.AutoCloseInputStream(shell("mkdir -p $dir")).use { it.readBytes() }
        val (out, input) = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommandRw("tee $path")
        ParcelFileDescriptor.AutoCloseOutputStream(input).use { it.write(text.toByteArray(Charsets.UTF_8)) }
        ParcelFileDescriptor.AutoCloseInputStream(out).use { it.readBytes() }
        val written = readDeviceFile(path).size
        check(written == text.toByteArray(Charsets.UTF_8).size) { "report $path: wrote $written bytes" }
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
    fun clip(name: String): FloatArray = decodeWav(name, testContext.assets.open("clips/$name").use { it.readBytes() })

    /** 16 kHz mono PCM16 WAV [bytes] as floats in [-1, 1]; [name] is for messages. */
    fun decodeWav(name: String, bytes: ByteArray): FloatArray {
        check(bytes.size >= 12) { "$name is not WAV (${bytes.size} bytes)" }
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

    const val DEVICE_GOLDEN_DIR = "/data/local/tmp/habla-golden/asr"
    const val DEVICE_BENCH_DIR = "/data/local/tmp/habla-bench"

    /** One golden utterance of the D6 set: the post-FinalPrep WAV the server fed WhisperX, and its outputs. */
    data class GoldenUtterance(
        val clip: String,
        val index: Int,
        val samples: FloatArray,
        val durationSec: Double,
        val serverTranscript: String,
        val serverLanguage: String,
        val serverStatus: String,
        val pcWhisperCpp: String?,
    ) {
        val id: String get() = "$clip#$index"
    }

    /**
     * The golden set pushed by `:asr:prepareDeviceBench` (`HABLA_GOLDEN_DIR`, default
     * shared/testdata/goldens). Fails, never returns empty, when it is missing (design.md section 11).
     */
    val deviceGoldens: List<GoldenUtterance> by lazy {
        val listing = readDeviceFile("$DEVICE_GOLDEN_DIR/index.txt").toString(Charsets.UTF_8)
            .lines().map { it.trim() }.filter { it.endsWith(".json") }
        check(listing.isNotEmpty()) { "golden set not found at $DEVICE_GOLDEN_DIR" }
        listing.flatMap { file ->
            val clip = file.removeSuffix(".json")
            val golden = Json.parseToJsonElement(readDeviceFile("$DEVICE_GOLDEN_DIR/$file").toString(Charsets.UTF_8)).jsonObject
            golden.getValue("utterances").jsonArray.map { e ->
                val u = e.jsonObject
                val wav = checkNotNull(u["final_prep_wav"]?.jsonPrimitive?.content) {
                    "$clip utterance ${u["index"]} has no final_prep_wav: rerun make_goldens.py"
                }
                val bytes = readDeviceFile("$DEVICE_GOLDEN_DIR/$wav")
                val expectedSha = u.getValue("final_prep_sha256").jsonPrimitive.content
                val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
                check(sha == expectedSha) { "$DEVICE_GOLDEN_DIR/$wav sha256 $sha != golden $expectedSha" }
                GoldenUtterance(
                    clip = clip,
                    index = u.getValue("index").jsonPrimitive.int,
                    samples = decodeWav(wav, bytes),
                    durationSec = u.getValue("duration_sec").jsonPrimitive.content.toDouble(),
                    serverTranscript = u.getValue("transcript").jsonPrimitive.content,
                    serverLanguage = u.getValue("detected_language").jsonPrimitive.content,
                    serverStatus = u.getValue("asr_status").jsonPrimitive.content,
                    pcWhisperCpp = u["pc_whisper_cpp"]?.jsonPrimitive?.content,
                )
            }
        }
    }

    /** Word errors and reference words of [hypothesis] against [reference], for word-weighted aggregates. */
    fun wordErrors(reference: String, hypothesis: String): Pair<Int, Int> {
        val r = normalizeWords(reference)
        val h = normalizeWords(hypothesis)
        if (r.isEmpty()) return h.size to 0
        var prev = IntArray(h.size + 1) { it }
        for (i in 1..r.size) {
            val cur = IntArray(h.size + 1)
            cur[0] = i
            for (j in 1..h.size) {
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (r[i - 1] == h[j - 1]) 0 else 1)
            }
            prev = cur
        }
        return prev[h.size] to r.size
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
        val (errors, words) = wordErrors(reference, hypothesis)
        if (words == 0) return if (errors == 0) 0.0 else 1.0
        return errors.toDouble() / words
    }
}
