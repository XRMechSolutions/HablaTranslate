package com.xrmech.habla.asr

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.xrmech.habla.asr.AsrTestFixtures.GoldenUtterance
import com.xrmech.habla.asr.AsrTestFixtures.SAMPLE_RATE
import com.xrmech.habla.shared.AsrModelEntry
import com.xrmech.habla.shared.AsrProfile
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ceil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.add
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * D5 benchmark on the A56 (design.md section 10). Three independent measurements, each writing its
 * own report under `/data/local/tmp/habla-bench/` (pull with `adb pull`):
 * - [loadAndFinalLatencySweep]: load ms + PSS per model, then final latency per duration bucket over
 *   threads x beam, 3 warm-up calls discarded.
 * - [partialLadder]: partial latency per ladder level on 1..10 s prefixes, and scaled-context WER
 *   against full context (also at pad 128 / min 256).
 * - [sustainedSession]: real-time replay at 60% speech duty through [AsrPipeline] with latest-wins
 *   partials every 1 s, per-5-minute p50/p90, thermal + battery.
 *
 * Prerequisite: `android\gradlew.bat -p android :asr:prepareDeviceBench`. Instrumentation args (all
 * optional): `habla.benchModels` (ids, default all catalog models), `habla.benchThreads` (default
 * 2,3,4; the engine caps threads at the pinned performance cores), `habla.benchBeams` (default
 * 5,3), `habla.model`, `habla.threads`, `habla.beam`, `habla.partialLevel`, `habla.pad`, `habla.min` (sustained/ladder config), `habla.sustainedMinutes`
 * (default 30).
 */
@DeviceOnly
@RunWith(AndroidJUnit4::class)
class AsrBenchmarkTest {
    private val args get() = InstrumentationRegistry.getArguments()

    // `habla.gpu=true` (experimental, D043) loads habla_asr_vk when the APK has it. The loader is
    // process-wide, so CPU and GPU are compared across separate `am instrument` runs.
    // `habla.partialCpu=true` (experimental E6, with habla.gpu): the PARTIAL slot runs on the CPU.
    @org.junit.Before
    fun selectBackend() {
        com.xrmech.habla.asr.internal.NativeLibraryLoader.preferGpu = args.getString("habla.gpu") == "true"
        com.xrmech.habla.asr.internal.NativeLibraryLoader.partialOnCpu = partialOnCpu
    }

    private val partialOnCpu: Boolean get() = args.getString("habla.partialCpu") == "true"
    private val context: Context get() = AsrTestFixtures.context
    private val baseProfile: AsrProfile get() = AsrTestFixtures.profile
    private val catalog get() = AsrTestFixtures.appCatalog

    /** `habla.partialModel` (D045): catalog id of a separate PARTIAL-slot model; null shares the main one. */
    private val partialModelId: String? get() = args.getString("habla.partialModel")

    private fun partialModelFile(): java.io.File? =
        partialModelId?.let { id -> AsrTestFixtures.installModel(checkNotNull(catalog.byId(id)) { "unknown model $id" }) }
    private val candidates get() = AsrTestFixtures.languages.codes
    private val date get() = LocalDate.now(ZoneOffset.UTC).toString()

    /** `gpu`/`cpu` plus a UTC time: CPU and GPU runs of the same day never overwrite each other. */
    private val runTag: String by lazy {
        val backend = if (args.getString("habla.gpu") == "true") "gpu" else "cpu"
        "$backend-" + java.time.LocalTime.now(ZoneOffset.UTC).toString().take(8).replace(":", "")
    }

    private fun intList(key: String, default: List<Int>) =
        args.getString(key)?.split(',')?.map { it.trim().toInt() } ?: default

    private fun configuredEntry(): AsrModelEntry =
        args.getString("habla.model")?.let { checkNotNull(catalog.byId(it)) { "unknown model $it" } } ?: catalog.default

    private fun configuredProfile(): AsrProfile {
        val wc = baseProfile.whisperCpp
        return baseProfile.copy(
            whisperCpp = wc.copy(
                final = wc.final.copy(beamSize = args.getString("habla.beam")?.toInt() ?: wc.final.beamSize),
                partial = wc.partial.copy(
                    level = args.getString("habla.partialLevel") ?: wc.partial.level,
                    audioCtxPadFrames = args.getString("habla.pad")?.toInt() ?: wc.partial.audioCtxPadFrames,
                    audioCtxMinFrames = args.getString("habla.min")?.toInt() ?: wc.partial.audioCtxMinFrames,
                ),
            ),
        )
    }

    private fun profileWith(beam: Int?, level: String? = null, pad: Int? = null, min: Int? = null): AsrProfile {
        val wc = baseProfile.whisperCpp
        return baseProfile.copy(
            whisperCpp = wc.copy(
                final = wc.final.copy(beamSize = beam),
                partial = wc.partial.copy(
                    level = level ?: wc.partial.level,
                    audioCtxPadFrames = pad ?: wc.partial.audioCtxPadFrames,
                    audioCtxMinFrames = min ?: wc.partial.audioCtxMinFrames,
                ),
            ),
        )
    }

    private fun JsonObjectBuilder.putDevice() {
        put("device", Build.MODEL)
        put("android_sdk", Build.VERSION.SDK_INT)
        put("date", date)
        put("started_utc", Instant.now().toString())
        put("charging", isCharging())
        // The native engine pins compute to the faster cores of this set and caps threads at their
        // count (habla_engine.cpp PerfCoreAffinity); instrumentation runs in the foreground cpuset.
        put("cpus_allowed", cpusAllowed())
    }

    private fun cpusAllowed(): String? =
        java.io.File("/proc/self/status").readLines().firstOrNull { it.startsWith("Cpus_allowed_list:") }
            ?.substringAfter(':')?.trim()

    // ---- 1. Load + final latency sweep --------------------------------------------------------------

    @Test
    fun loadAndFinalLatencySweep() = runBlocking<Unit> {
        val goldens = AsrTestFixtures.deviceGoldens
        val modelIds = args.getString("habla.benchModels")?.split(',')?.map { it.trim() }
            ?: catalog.models.filter { it.usableForFinals }.map { it.id }
        val threadList = intList("habla.benchThreads", listOf(2, 3, 4))
        val beams = intList("habla.benchBeams", listOf(5, 3))
        val modelReports = mutableListOf<JsonObject>()
        var anyVariant: String? = null
        for (id in modelIds) {
            val entry = checkNotNull(catalog.byId(id)) { "unknown model $id" }
            val file = AsrTestFixtures.installModel(entry)
            Runtime.getRuntime().gc()
            val pssBefore = pssKb()
            val t0 = SystemClock.elapsedRealtime()
            val loadRecognizer = WhisperCppRecognizer(file, profileWith(null), 4)
            loadRecognizer.preload()
            val loadMs = SystemClock.elapsedRealtime() - t0
            val pssAfter = pssKb()
            anyVariant = loadRecognizer.nativeInfo
            loadRecognizer.close()
            log("load $id ${loadMs}ms pss ${pssBefore / 1024}->${pssAfter / 1024} MB")

            val configs = mutableListOf<JsonObject>()
            for (beam in beams) for (threads in threadList) {
                val r = WhisperCppRecognizer(file, profileWith(beam.takeIf { it != baseProfile.whisper.beamSize }), threads)
                r.use {
                    it.preload()
                    // 3 warm-up calls discarded.
                    goldens.take(3).forEach { u -> it.transcribe(u.samples, AsrRequest(u.serverLanguage, AsrPass.FINAL)) }
                    val samples = goldens.map { u -> u to timeFinal(it, u) }
                    configs += buildJsonObject {
                        put("threads", threads)
                        put("beam", beam)
                        putJsonObject("buckets") {
                            val grouped = samples.groupBy { (u, _) -> bucketOf(u.durationSec) }.toSortedMap()
                            for ((bucket, rows) in grouped) {
                                putJsonObject(bucket) {
                                    put("n", rows.size)
                                    put("detect_p50_ms", pct(rows.map { it.second.detectMs.toDouble() }, 50))
                                    put("transcribe_p50_ms", pct(rows.map { it.second.transcribeMs.toDouble() }, 50))
                                    put("total_p50_ms", pct(rows.map { it.second.totalMs.toDouble() }, 50))
                                    put("total_p90_ms", pct(rows.map { it.second.totalMs.toDouble() }, 90))
                                    put("rtf_p50", pct(rows.map { (u, t) -> t.totalMs / 1000.0 / u.durationSec }, 50))
                                    put("rtf_p90", pct(rows.map { (u, t) -> t.totalMs / 1000.0 / u.durationSec }, 90))
                                }
                            }
                            if ("20-31s" !in grouped) put("20-31s", "no golden utterance in this range")
                        }
                        put("rtf_p90_ge3s", pct(samples.filter { it.first.durationSec >= 3 }.map { (u, t) -> t.totalMs / 1000.0 / u.durationSec }, 90))
                        put("total_p90_ms_lt3s", pct(samples.filter { it.first.durationSec < 3 }.map { it.second.totalMs.toDouble() }, 90))
                    }
                    log("sweep $id beam=$beam threads=$threads ${configs.last()}")
                }
            }
            modelReports += buildJsonObject {
                put("model_id", id)
                put("quantization", entry.quantization)
                put("load_ms", loadMs)
                put("pss_before_mb", pssBefore / 1024)
                put("pss_after_load_mb", pssAfter / 1024)
                put("configs", JsonArray(configs))
            }
        }
        val report = buildJsonObject {
            putDevice()
            put("native_info", anyVariant)
            put("utterances", goldens.size)
            put("audio_sec", goldens.sumOf { it.durationSec })
            put("models", JsonArray(modelReports))
        }
        write("d5-sweep-$date-$runTag.json", report)
        val info = checkNotNull(anyVariant)
        val variant = if (args.getString("habla.gpu") == "true") "vk" else "v86"
        assertTrue(info, info.startsWith("$variant | ") && info.contains("DOTPROD = 1") && info.contains("MATMUL_INT8 = 1"))
        // The vk library alone does not prove GPU execution: ggml must have registered a Vulkan device.
        if (variant == "vk") assertTrue("no Vulkan device: $info", info.contains("[Vulkan"))
    }

    private data class FinalTiming(val detectMs: Long, val transcribeMs: Long, val totalMs: Long)

    private suspend fun timeFinal(r: WhisperCppRecognizer, u: GoldenUtterance): FinalTiming {
        val t0 = SystemClock.elapsedRealtime()
        val guess = r.detectLanguage(u.samples, candidates)
        val t1 = SystemClock.elapsedRealtime()
        r.transcribe(u.samples, AsrRequest(guess.language, AsrPass.FINAL))
        val t2 = SystemClock.elapsedRealtime()
        return FinalTiming(t1 - t0, t2 - t1, t2 - t0)
    }

    private fun bucketOf(sec: Double) = when {
        sec < 3 -> "0-3s"
        sec < 10 -> "3-10s"
        sec < 20 -> "10-20s"
        else -> "20-31s"
    }

    // ---- 2. Partial ladder: latency per level and scaled-context quality ---------------------------

    @Test
    fun partialLadder() = runBlocking<Unit> {
        val goldens = AsrTestFixtures.deviceGoldens
        val entry = configuredEntry()
        val file = AsrTestFixtures.installModel(entry)
        val threads = args.getString("habla.threads")?.toInt() ?: baseProfile.whisperCpp.threads
        // Prefixes of the utterances long enough to give every 1..10 s snapshot.
        val sources = goldens.filter { it.durationSec >= 10.0 }
        check(sources.isNotEmpty()) { "no golden utterance of 10 s or more for the partial ladder" }
        val prefixes = (1..10).flatMap { s -> sources.map { u -> Triple(u, s, u.samples.copyOfRange(0, s * SAMPLE_RATE)) } }

        val texts = mutableMapOf<String, Map<Pair<String, Int>, String>>()
        val levels = mutableListOf<JsonObject>()
        val variants = listOf(
            Triple("parity", null, null),
            Triple("greedy_full_ctx", null, null),
            Triple("greedy_scaled_ctx", null, null),
            Triple("greedy_scaled_ctx", 128, 256),
        )
        var nativeInfo: String? = null
        for ((level, pad, min) in variants) {
            val name = if (pad == null) level else "$level(pad=$pad,min=$min)"
            WhisperCppRecognizer(file, profileWith(null, level, pad, min), threads, partialModelFile()).use { r ->
                r.preload()
                nativeInfo = r.nativeInfo
                repeat(3) { r.transcribe(prefixes[it].third, AsrRequest(prefixes[it].first.serverLanguage, AsrPass.PARTIAL)) }
                val out = mutableMapOf<Pair<String, Int>, String>()
                val bySecond = sortedMapOf<Int, MutableList<Double>>()
                for ((u, s, pcm) in prefixes) {
                    val t0 = SystemClock.elapsedRealtime()
                    val res = r.transcribe(pcm, AsrRequest(u.serverLanguage, AsrPass.PARTIAL))
                    bySecond.getOrPut(s) { mutableListOf() } += (SystemClock.elapsedRealtime() - t0).toDouble()
                    out[u.id to s] = res.text
                }
                texts[name] = out
                val all = bySecond.values.flatten()
                levels += buildJsonObject {
                    put("level", name)
                    put("p50_ms", pct(all, 50))
                    put("p90_ms", pct(all, 90))
                    putJsonObject("p90_ms_by_prefix_s") { bySecond.forEach { (s, v) -> put("$s", pct(v, 90)) } }
                }
                log("ladder $name ${levels.last()}")
            }
        }
        // With a separate partial model every level above ran on it, so its own full-context text is not
        // a quality reference: also run the main model's full-context partials on the same prefixes.
        if (partialModelId != null) {
            WhisperCppRecognizer(file, profileWith(null, "greedy_full_ctx", null, null), threads).use { r ->
                r.preload()
                texts["main_greedy_full_ctx"] = prefixes.associate { (u, s, pcm) ->
                    (u.id to s) to r.transcribe(pcm, AsrRequest(u.serverLanguage, AsrPass.PARTIAL)).text
                }
            }
        }
        fun werAgainst(refName: String, name: String): Double {
            var e = 0
            var w = 0
            for ((k, ref) in texts.getValue(refName)) {
                val (ee, ww) = AsrTestFixtures.wordErrors(ref, texts.getValue(name).getValue(k))
                e += ee
                w += ww
            }
            return if (w == 0) 0.0 else e.toDouble() / w
        }
        fun scaledWer(name: String) = werAgainst("greedy_full_ctx", name)
        val report = buildJsonObject {
            putDevice()
            put("native_info", nativeInfo)
            put("model_id", entry.id)
            put("partial_model_id", partialModelId)
            put("partial_on_cpu", partialOnCpu)
            put("threads", threads)
            putJsonArray("sources") { sources.forEach { add(it.id) } }
            put("levels", JsonArray(levels))
            putJsonObject("scaled_vs_full_ctx_wer") {
                put("pad64_min128", scaledWer("greedy_scaled_ctx"))
                put("pad128_min256", scaledWer("greedy_scaled_ctx(pad=128,min=256)"))
                put("gate", "<= 0.05")
            }
            if (partialModelId != null) {
                putJsonObject("partial_model_vs_main_full_ctx_wer") {
                    put("reference", "${entry.id} greedy_full_ctx partials on the same prefixes")
                    put("greedy_full_ctx", werAgainst("main_greedy_full_ctx", "greedy_full_ctx"))
                    put("greedy_scaled_ctx", werAgainst("main_greedy_full_ctx", "greedy_scaled_ctx"))
                    put("floor", "<= 0.35 (device-acceleration-plan.md section 5, proposed)")
                }
            }
        }
        val partialTag = partialModelId?.let { "-p_$it" }.orEmpty() + if (partialOnCpu) "-pcpu" else ""
        write("d5-ladder$partialTag-$date-$runTag.json", report)
    }

    // ---- 3. Sustained real-time session -------------------------------------------------------------

    @Test
    fun sustainedSession() = runBlocking<Unit> {
        val goldens = AsrTestFixtures.deviceGoldens
        val entry = configuredEntry()
        val file = AsrTestFixtures.installModel(entry)
        val profile = configuredProfile()
        val threads = args.getString("habla.threads")?.toInt() ?: profile.whisperCpp.threads
        val minutes = args.getString("habla.sustainedMinutes")?.toInt() ?: 30
        val durationMs = minutes * 60_000L
        val power = context.getSystemService(PowerManager::class.java)

        val partialLat = mutableListOf<Pair<Long, Long>>() // (t, ms)
        val finalLat = mutableListOf<Triple<Long, Long, Double>>() // (t, total ms, duration s)
        val finalWait = mutableListOf<Long>()
        val queueDepthAtEnqueue = mutableListOf<Int>()
        val thermal = mutableListOf<JsonObject>()
        val submitted = AtomicInteger()
        val started = AtomicInteger()
        val aborted = AtomicInteger()
        val failed = AtomicInteger() // non-ABORTED AsrException: nothing shown
        val filtered = AtomicInteger() // bad transcript or empty: nothing shown
        val displayedAt = mutableListOf<Pair<Int, Long>>() // (segment seq, elapsedRealtime) of shown partials
        val segments = mutableListOf<Pair<Long, Long>>() // seq -> (speech start, speech end), elapsedRealtime
        var nativeInfo: String? = null
        val stats = Mutex()
        val partialAge = mutableListOf<Long>() // capture to result, including time spent waiting
        val maxThermal = AtomicInteger() // written by the monitor coroutine
        val pssPeakKb = AtomicInteger()
        val batteryStart = batteryPct()
        val start = SystemClock.elapsedRealtime()
        fun now() = SystemClock.elapsedRealtime() - start

        WhisperCppRecognizer(file, profile, threads, partialModelFile()).use { r ->
            r.preload()
            nativeInfo = r.nativeInfo
            val pipeline = AsrPipeline(r, LanguageVoter(profile), profile, AsrTestFixtures.languages)
            // AsrPipeline and LanguageVoter are not thread-safe: the app drives them from one serial
            // dispatcher, so the replay does too. Native work still runs on the recognizer's executor,
            // so a waiting partial does not block the final worker.
            val serial = Dispatchers.Default.limitedParallelism(1)
            val partials = Channel<Triple<FloatArray, Long, Int>>(Channel.CONFLATED) // (pcm, captured at, seq)
            val finals = Channel<Pair<GoldenUtterance, Long>>(Channel.UNLIMITED)
            val pendingFinals = AtomicInteger()
            coroutineScope {
                val partialWorker = launch(serial) {
                    for ((pcm, capturedAt, seq) in partials) {
                        started.incrementAndGet()
                        val t0 = SystemClock.elapsedRealtime()
                        val out = pipeline.partial(pcm, "es")
                        val t1 = SystemClock.elapsedRealtime()
                        // Only a non-empty partial is a visible update; failed and filtered ones are
                        // counted separately so a model that fails every call cannot look fast.
                        stats.withLock {
                            when {
                                out.aborted -> aborted.incrementAndGet()
                                out.language == null -> failed.incrementAndGet()
                                out.text.isEmpty() -> filtered.incrementAndGet()
                                else -> {
                                    partialLat += now() to (t1 - t0)
                                    partialAge += t1 - capturedAt
                                    displayedAt += seq to t1
                                }
                            }
                        }
                    }
                }
                val finalWorker = launch(serial) {
                    for ((u, enqueuedAt) in finals) {
                        val t0 = SystemClock.elapsedRealtime()
                        pipeline.final(u.samples, "es")
                        val t1 = SystemClock.elapsedRealtime()
                        pendingFinals.decrementAndGet()
                        stats.withLock {
                            finalWait += t0 - enqueuedAt
                            finalLat += Triple(now(), t1 - enqueuedAt, u.durationSec)
                        }
                    }
                }
                val monitor = launch(Dispatchers.Default) {
                    while (true) {
                        val status = power.currentThermalStatus
                        maxThermal.accumulateAndGet(status, ::maxOf)
                        thermal += buildJsonObject {
                            put("t_s", now() / 1000)
                            put("status", status)
                            // getThermalHeadroom is API 30+ and NaN where unsupported; JSON gets null.
                            val headroom = if (Build.VERSION.SDK_INT >= 30) power.getThermalHeadroom(10) else Float.NaN
                            put("headroom_10s", headroom.takeIf { it.isFinite() }?.toDouble())
                            val pss = pssKb()
                            pssPeakKb.accumulateAndGet(pss, ::maxOf)
                            put("pss_mb", pss / 1024)
                            put("battery_pct", batteryPct())
                        }
                        delay(30_000)
                    }
                }
                // Real-time replay: speech for the utterance's duration with a snapshot every 1 s, then
                // the final at segment end and silence so speech is 60% of wall time.
                var i = 0
                while (now() < durationMs) {
                    val u = goldens[i++ % goldens.size]
                    val segStart = SystemClock.elapsedRealtime()
                    val seq = stats.withLock { segments += segStart to segStart + (u.durationSec * 1000).toLong(); segments.size - 1 }
                    var s = 1
                    while (s <= u.durationSec) {
                        delay(maxOf(0, segStart + s * 1000L - SystemClock.elapsedRealtime()))
                        submitted.incrementAndGet()
                        partials.trySend(Triple(u.samples.copyOfRange(0, s * SAMPLE_RATE), SystemClock.elapsedRealtime(), seq))
                        s++
                    }
                    delay(maxOf(0, segStart + (u.durationSec * 1000).toLong() - SystemClock.elapsedRealtime()))
                    queueDepthAtEnqueue += pendingFinals.getAndIncrement()
                    finals.send(u to SystemClock.elapsedRealtime())
                    delay((u.durationSec * 1000 * 0.4 / 0.6).toLong())
                }
                partials.close()
                finals.close()
                finalWorker.join()
                partialWorker.join()
                monitor.cancelAndJoin()
            }
        }
        val batteryEnd = batteryPct()

        val windows = (0 until ceil(durationMs / 300_000.0).toInt()).map { w ->
            val lo = w * 300_000L
            val hi = lo + 300_000L
            val p = partialLat.filter { it.first in lo until hi }.map { it.second.toDouble() }
            val f = finalLat.filter { it.first in lo until hi }
            // Empty windows report null with n = 0, not 0 ms.
            buildJsonObject {
                put("window_min", "${w * 5}-${w * 5 + 5}")
                put("partials_n", p.size)
                put("partial_p50_ms", pctOrNull(p, 50))
                put("partial_p90_ms", pctOrNull(p, 90))
                put("finals_n", f.size)
                put("final_p50_ms", pctOrNull(f.map { it.second.toDouble() }, 50))
                put("final_p90_ms", pctOrNull(f.map { it.second.toDouble() }, 90))
            }
        }
        // Partials that never started were replaced by a newer snapshot (latest-wins) or were still
        // waiting when the run ended; aborted ones were preempted by a final.
        val dropped = submitted.get() - started.get() + aborted.get()
        val dropRatio = if (submitted.get() == 0) null else dropped.toDouble() / submitted.get()
        val partialP90 = pctOrNull(partialLat.map { it.second.toDouble() }, 90)
        val ageP90 = pctOrNull(partialAge.map { it.toDouble() }, 90)
        val finalsGe3 = finalLat.filter { it.third >= 3 }
        val rtfP90 = pctOrNull(finalsGe3.map { it.second / 1000.0 / it.third }, 90)
        val shortP90 = pctOrNull(finalLat.filter { it.third < 3 }.map { it.second.toDouble() }, 90)
        val finalP99 = pctOrNull(finalLat.map { it.second.toDouble() }, 99)
        val queueP95 = pctOrNull(queueDepthAtEnqueue.map { it.toDouble() }, 95)
        // Longest stretch of speech without a new visible partial, per segment: from speech start through
        // each shown update to speech end (updates after speech end do not count as during speech).
        val shownBySeq = displayedAt.groupBy({ it.first }, { it.second })
        val maxGaps = segments.mapIndexed { seq, (startAt, endAt) ->
            val points = listOf(startAt) + (shownBySeq[seq].orEmpty().filter { it in startAt..endAt }.sorted()) + endAt
            points.zipWithNext { a, b -> (b - a).toDouble() }.maxOrNull() ?: 0.0
        }
        val gapP95 = pctOrNull(maxGaps, 95)
        fun rule(value: Double?, ok: (Double) -> Boolean): Boolean? = value?.let(ok)
        val report = buildJsonObject {
            putDevice()
            put("model_id", entry.id)
            put("partial_model_id", partialModelId)
            put("partial_on_cpu", partialOnCpu)
            put("threads", threads)
            put("beam", profile.whisperCpp.final.beamSize ?: profile.whisper.beamSize)
            put("partial_level", profile.whisperCpp.partial.level)
            put("pad", profile.whisperCpp.partial.audioCtxPadFrames)
            put("min", profile.whisperCpp.partial.audioCtxMinFrames)
            put("minutes", minutes)
            put("speech_duty", 0.6)
            put("partials_submitted", submitted.get())
            put("partials_started", started.get())
            put("partials_aborted_by_final", aborted.get())
            put("partials_failed", failed.get())
            put("partials_filtered", filtered.get())
            put("partial_drop_ratio", dropRatio)
            put("partials_displayed", partialLat.size)
            put("partial_p50_ms", pctOrNull(partialLat.map { it.second.toDouble() }, 50))
            put("partial_p90_ms", partialP90)
            put("partial_age_p90_ms", ageP90)
            put("partial_update_gap_p95_ms", gapP95)
            put("partial_update_gap_max_ms", maxGaps.maxOrNull())
            put("finals", finalLat.size)
            put("final_p50_ms", pctOrNull(finalLat.map { it.second.toDouble() }, 50))
            put("final_p90_ms", pctOrNull(finalLat.map { it.second.toDouble() }, 90))
            put("final_p99_ms", finalP99)
            put("final_rtf_p90_ge3s", rtfP90)
            put("final_total_p90_ms_lt3s", shortP90)
            put("final_wait_max_ms", finalWait.maxOrNull())
            put("final_queue_depth_p95", queueP95)
            put("battery_start_pct", batteryStart)
            put("battery_end_pct", batteryEnd)
            put("max_thermal_status", maxThermal.get())
            put("pss_peak_mb", pssPeakKb.get() / 1024)
            put("native_info", nativeInfo)
            // null = not measurable in this run (no samples), never a pass.
            putJsonObject("decision_rules") {
                putJsonObject("design_v1") { // design.md section 10, the original A56 rules
                    put("partial_p90_le_800ms", rule(partialP90) { it <= 800 })
                    put("partial_drop_le_0_25", rule(dropRatio) { it <= 0.25 })
                    put("final_rtf_p90_ge3s_le_0_5", rule(rtfP90) { it <= 0.5 })
                    put("final_total_p90_lt3s_le_2000ms", rule(shortP90) { it <= 2000 })
                    put("final_queue_depth_p95_le_1", rule(queueP95) { it <= 1 })
                }
                putJsonObject("mobile_proposed") { // device-acceleration-plan.md section 5, until frozen
                    put("final_total_p90_lt3s_le_4000ms", rule(shortP90) { it <= 4000 })
                    put("final_rtf_p90_ge3s_le_1_0", rule(rtfP90) { it <= 1.0 })
                    put("final_p99_le_10000ms", rule(finalP99) { it <= 10_000 })
                    put("final_queue_depth_p95_le_1", rule(queueP95) { it <= 1 })
                    put("partial_age_p90_le_1000ms", rule(ageP90) { it <= 1000 })
                    put("partial_drop_le_0_50", rule(dropRatio) { it <= 0.5 })
                    put("partial_update_gap_p95_le_2500ms", rule(gapP95) { it <= 2500 })
                }
                put("thermal_below_severe", maxThermal.get() < PowerManager.THERMAL_STATUS_SEVERE)
            }
            put("windows", JsonArray(windows))
            put("thermal", JsonArray(thermal))
        }
        val level = profile.whisperCpp.partial.level
        val partialTag = partialModelId?.let { "-p_$it" }.orEmpty() + if (partialOnCpu) "-pcpu" else ""
        write("d5-sustained-${entry.id}$partialTag-t$threads-$level-${minutes}m-$date-$runTag.json", report)
    }

    // ---- helpers -------------------------------------------------------------------------------------

    private fun pct(values: List<Double>, p: Int): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val idx = ceil(p / 100.0 * sorted.size).toInt().coerceIn(1, sorted.size) - 1
        return sorted[idx]
    }

    private fun pctOrNull(values: List<Double>, p: Int): Double? = if (values.isEmpty()) null else pct(values, p)

    private fun pssKb(): Int = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }.totalPss

    private fun batteryPct(): Int =
        context.getSystemService(BatteryManager::class.java).getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)

    private fun isCharging(): Boolean {
        val i: Intent? = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return (i?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
    }

    private fun write(name: String, report: JsonObject) {
        val path = AsrTestFixtures.writeDeviceFile(
            "${AsrTestFixtures.DEVICE_BENCH_DIR}/$name",
            PRETTY.encodeToString(JsonObject.serializer(), report),
        )
        log("report $path")
    }

    private fun log(msg: String) = android.util.Log.i("HablaAsrBench", msg)

    private companion object {
        val PRETTY = Json { prettyPrint = true }
    }
}
