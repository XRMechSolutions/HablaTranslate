package com.xrmech.habla.asr

import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.time.LocalDate
import java.time.ZoneOffset
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * D6 parity suite on the A56 (design.md section 11): the shipping model and profile on the
 * post-FinalPrep WAVs the server fed WhisperX, compared with the PC whisper.cpp build (engine
 * fidelity, same params, language forced to the golden's) and with WhisperX (server parity, through
 * [AsrPipeline.final] with its own detection).
 *
 * Prerequisite: `android\gradlew.bat -p android :asr:prepareDeviceBench`. Instrumentation args:
 * `habla.model` (default: the catalog default), `habla.threads` (default: profile `whisper_cpp.threads`),
 * `habla.engineWerMax` (engine fidelity gate, default 0.02; a GPU run passes 0 to prove the backend
 * gives the CPU's words). The report goes to
 * `/data/local/tmp/habla-bench/d6-parity-<date>-<cpu|gpu>-<HHmmss>.json`.
 */
@DeviceOnly
@RunWith(AndroidJUnit4::class)
class AsrParityTest {
    private val args get() = InstrumentationRegistry.getArguments()

    // `habla.gpu=true` (experimental, D043): run the parity check on habla_asr_vk.
    @org.junit.Before
    fun selectBackend() {
        com.xrmech.habla.asr.internal.NativeLibraryLoader.preferGpu = args.getString("habla.gpu") == "true"
    }

    @Test
    fun phoneMatchesPcWhisperCppAndServerWhisperX() = runBlocking<Unit> {
        val goldens = AsrTestFixtures.deviceGoldens
        val catalog = AsrTestFixtures.appCatalog
        val entry = args.getString("habla.model")?.let { checkNotNull(catalog.byId(it)) { "unknown model $it" } } ?: catalog.default
        val profile = AsrTestFixtures.profile
        val threads = args.getString("habla.threads")?.toInt() ?: profile.whisperCpp.threads
        val engineWerMax = args.getString("habla.engineWerMax")?.toDouble() ?: 0.02
        val model = AsrTestFixtures.installModel(entry)

        val recognizer = WhisperCppRecognizer(model, profile, threads)
        val rows = mutableListOf<JsonObject>()
        var engineErrors = 0
        var engineWords = 0
        var engineLangSame = 0
        var engineCount = 0
        var serverErrors = 0
        var serverWords = 0
        var serverLangSame = 0
        val clipFailures = mutableListOf<String>()
        val reasons = mutableMapOf<RejectReason, Int>()
        recognizer.use { r ->
            r.preload()
            val info = r.nativeInfo
            if (args.getString("habla.gpu") == "true") {
                // vk loaded and a Vulkan device registered; otherwise this would silently be a CPU run.
                assertTrue("not a GPU run: $info", info != null && info.startsWith("vk | ") && info.contains("[Vulkan"))
            }
            for ((clip, utts) in goldens.groupBy { it.clip }) {
                // One voter per clip, as the server keeps one per session.
                val pipeline = AsrPipeline(r, LanguageVoter(profile), profile, AsrTestFixtures.languages)
                for (u in utts) {
                    val forced = r.transcribe(u.samples, AsrRequest(u.serverLanguage, AsrPass.FINAL))
                    val outcome = pipeline.final(u.samples, sourceLanguage = "es")
                    reasons.merge(outcome.reason, 1, Int::plus)

                    val pc = u.pcWhisperCpp
                    var engineWer: Double? = null
                    if (pc != null) {
                        val (e, w) = AsrTestFixtures.wordErrors(pc, forced.text)
                        engineErrors += e
                        engineWords += w
                        engineCount++
                        if (outcome.detectedLanguage == u.serverLanguage) engineLangSame++
                        engineWer = AsrTestFixtures.wer(pc, forced.text)
                    }
                    val (se, sw) = AsrTestFixtures.wordErrors(u.serverTranscript, outcome.transcript)
                    serverErrors += se
                    serverWords += sw
                    if (outcome.detectedLanguage == u.serverLanguage) serverLangSame++
                    val serverWer = AsrTestFixtures.wer(u.serverTranscript, outcome.transcript)
                    if (sw >= 5 && serverWer > 0.30) clipFailures += "${u.id} WER %.3f".format(serverWer)
                    android.util.Log.i(
                        "HablaAsrParity",
                        "${u.id} dur=${u.durationSec} engineWer=$engineWer serverWer=$serverWer " +
                            "lang=${outcome.detectedLanguage}/${u.serverLanguage} reason=${outcome.reason} ms=${outcome.result.elapsedMs}",
                    )
                    rows += buildJsonObject {
                        put("id", u.id)
                        put("duration_sec", u.durationSec)
                        put("server_language", u.serverLanguage)
                        put("phone_language", outcome.detectedLanguage)
                        put("phone_language_prob", outcome.languageProb?.toDouble())
                        put("reject_reason", outcome.reason.name)
                        put("engine_wer", engineWer)
                        put("server_wer", serverWer)
                        put("server_ref_words", sw)
                        put("transcribe_ms", outcome.result.elapsedMs)
                        // Transcripts are public-domain LibriVox text; kept so a reviewer can see each diff.
                        put("server", u.serverTranscript)
                        put("pc_whisper_cpp", pc)
                        put("phone_forced", forced.text)
                        put("phone_pipeline", outcome.transcript)
                    }
                }
            }

            val n = goldens.size
            val engineWerAgg = if (engineWords == 0) null else engineErrors.toDouble() / engineWords
            val engineLang = if (engineCount == 0) null else engineLangSame.toDouble() / engineCount
            val serverWerAgg = serverErrors.toDouble() / serverWords
            val serverLang = serverLangSame.toDouble() / n
            val gates = buildJsonObject {
                put("engine_wer_max", engineWerMax)
                put("engine_wer_le_max", engineWerAgg != null && engineWerAgg <= engineWerMax)
                put("engine_language_ge_0_98", engineLang != null && engineLang >= 0.98)
                put("server_wer_le_0_10", serverWerAgg <= 0.10)
                put("server_clip_wer_le_0_30", clipFailures.isEmpty())
                put("server_language_ge_0_95", serverLang >= 0.95)
            }
            val report = buildJsonObject {
                put("device", Build.MODEL)
                put("android_sdk", Build.VERSION.SDK_INT)
                put("date", LocalDate.now(ZoneOffset.UTC).toString())
                put("native_info", info)
                put("model_id", entry.id)
                put("threads", threads)
                put("beam_size", profile.whisperCpp.final.beamSize ?: profile.whisper.beamSize)
                put("utterances", n)
                putJsonObject("engine_fidelity") {
                    put("reference", "pc_whisper_cpp (habla/scripts/compare_engines.py, same FINAL params, language forced)")
                    put("aggregate_wer", engineWerAgg)
                    put("reference_words", engineWords)
                    put("language_agreement", engineLang)
                }
                putJsonObject("server_parity") {
                    put("reference", "server WhisperX transcript (make_goldens.py)")
                    put("aggregate_wer", serverWerAgg)
                    put("reference_words", serverWords)
                    put("language_agreement", serverLang)
                    putJsonArray("clips_over_0_30") { clipFailures.forEach { add(it) } }
                }
                putJsonObject("reject_reasons_phone") { reasons.forEach { (k, v) -> put(k.name, v) } }
                put("reject_reasons_server", "all ${goldens.count { it.serverStatus == "ok" }} of $n ok")
                put("gates", gates)
                putJsonArray("rows") { rows.forEach { add(it) } }
            }
            val backend = if (args.getString("habla.gpu") == "true") "gpu" else "cpu"
            val time = java.time.LocalTime.now(ZoneOffset.UTC).toString().take(8).replace(":", "")
            val path = AsrTestFixtures.writeDeviceFile(
                "${AsrTestFixtures.DEVICE_BENCH_DIR}/d6-parity-${LocalDate.now(ZoneOffset.UTC)}-$backend-$time.json",
                PRETTY.encodeToString(JsonObject.serializer(), report),
            )
            android.util.Log.i("HablaAsrParity", "report $path gates=$gates engineWer=$engineWerAgg serverWer=$serverWerAgg")

            assertTrue(
                "engine fidelity WER $engineWerAgg > $engineWerMax (report $path)",
                engineWerAgg != null && engineWerAgg <= engineWerMax,
            )
            assertTrue("engine language agreement $engineLang < 0.98 (report $path)", engineLang != null && engineLang >= 0.98)
            assertTrue("server parity WER $serverWerAgg > 0.10 (report $path)", serverWerAgg <= 0.10)
            assertTrue("clips over 0.30 WER: $clipFailures (report $path)", clipFailures.isEmpty())
            assertTrue("server language agreement $serverLang < 0.95 (report $path)", serverLang >= 0.95)
        }
    }

    private companion object {
        val PRETTY = Json { prettyPrint = true }
    }
}
