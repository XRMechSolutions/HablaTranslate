package com.xrmech.habla.audio

import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

/**
 * Reads the goldens written by `habla/scripts/make_audio_goldens.py` (test resources `goldens/`)
 * and composes fixtures from their recipes. Speech is read in place from
 * `<habla.sharedDir>/testdata/clips` (D033), never from a copy.
 */
internal object GoldenFixtures {
    const val STALE = "goldens stale: run habla\\.venv\\Scripts\\python.exe habla\\scripts\\make_audio_goldens.py"
    private const val ROOT = "/goldens/"

    val sharedDir: File by lazy {
        val path = System.getProperty("habla.sharedDir")
            ?: error("system property habla.sharedDir is not set; run tests through Gradle")
        File(path).also { check(it.isDirectory) { "habla.sharedDir is not a directory: $path" } }
    }

    val clipsDir: File get() = File(sharedDir, "testdata/clips")

    val manifest: JsonObject by lazy { Json.parseToJsonElement(text("manifest.json")).jsonObject }

    /** File name -> SHA-256 recorded by the generator for every committed golden. */
    val files: Map<String, String> by lazy {
        manifest.obj("files").mapValues { it.value.jsonPrimitive.content }
    }

    fun exists(name: String): Boolean = GoldenFixtures::class.java.getResource(ROOT + name) != null

    fun bytes(name: String): ByteArray =
        GoldenFixtures::class.java.getResourceAsStream(ROOT + name)?.use { it.readBytes() }
            ?: throw FileNotFoundException("test resource goldens/$name missing; $STALE")

    fun text(name: String): String = bytes(name).toString(Charsets.UTF_8)

    private fun gunzip(name: String): ByteBuffer {
        val raw = GZIPInputStream(ByteArrayInputStream(bytes(name))).use { it.readBytes() }
        return ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
    }

    fun shorts(name: String): ShortArray {
        val b = gunzip(name).asShortBuffer()
        return ShortArray(b.remaining()).also { b.get(it) }
    }

    fun floats(name: String): FloatArray {
        val b = gunzip(name).asFloatBuffer()
        return FloatArray(b.remaining()).also { b.get(it) }
    }

    fun doubles(name: String): DoubleArray {
        val b = gunzip(name).asDoubleBuffer()
        return DoubleArray(b.remaining()).also { b.get(it) }
    }

    private val clipCache = HashMap<String, ShortArray>()

    /** PCM16 samples of `shared/testdata/clips/<name>` (16 kHz mono WAV, any chunk layout). */
    fun clip(name: String): ShortArray = clipCache.getOrPut(name) { readWav(File(clipsDir, name)) }

    fun readWav(file: File): ShortArray {
        if (!file.isFile) throw FileNotFoundException("missing clip ${file.absolutePath}")
        val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        require(buf.int == 0x46464952 /* RIFF */) { "${file.name}: not RIFF" }
        buf.int
        require(buf.int == 0x45564157 /* WAVE */) { "${file.name}: not WAVE" }
        var fmtOk = false
        while (buf.remaining() >= 8) {
            val id = buf.int
            val size = buf.int
            val start = buf.position()
            when (id) {
                0x20746d66 /* "fmt " */ -> {
                    val format = buf.short.toInt()
                    val channels = buf.short.toInt()
                    val rate = buf.int
                    buf.int
                    buf.short
                    val bits = buf.short.toInt()
                    require(format == 1 && channels == 1 && rate == 16000 && bits == 16) {
                        "${file.name}: need 16 kHz mono PCM16"
                    }
                    fmtOk = true
                }
                0x61746164 /* "data" */ -> {
                    require(fmtOk) { "${file.name}: data before fmt" }
                    val n = minOf(size, buf.remaining()) / 2
                    val out = ShortArray(n)
                    buf.asShortBuffer().get(out)
                    return out
                }
            }
            buf.position(start + size + (size and 1))
        }
        error("${file.name}: no data chunk")
    }

    private val sourceCache = HashMap<String, ShortArray>()

    /** Committed synthetic fixture (noise, hot clip) by its manifest name. */
    fun source(name: String): ShortArray = sourceCache.getOrPut(name) {
        shorts(manifest.obj("noise").obj(name).str("file"))
    }

    /** Applies the generator's level change: numpy `rint(s * gain)` clipped to int16. */
    fun applyGain(x: ShortArray, gain: Double): ShortArray {
        if (gain == 1.0) return x.copyOf()
        return ShortArray(x.size) { Math.rint(x[it] * gain).coerceIn(-32768.0, 32767.0).toInt().toShort() }
    }

    /** Builds a fixture from its recipe pieces (`clip`, `noise`, `zeros`). */
    fun compose(pieces: JsonArray): ShortArray {
        val parts = pieces.map { compose(it.jsonObject) }
        val out = ShortArray(parts.sumOf { it.size })
        var pos = 0
        for (p in parts) {
            p.copyInto(out, pos)
            pos += p.size
        }
        return out
    }

    fun compose(piece: JsonObject): ShortArray {
        val length = piece.int("length")
        return when (val kind = piece.str("kind")) {
            "zeros" -> ShortArray(length)
            "clip" -> {
                val start = piece.int("start")
                applyGain(clip(piece.str("clip")).copyOfRange(start, start + length), piece.double("gain"))
            }
            "noise" -> {
                val start = piece.int("start")
                source(piece.str("source")).copyOfRange(start, start + length)
            }
            else -> error("unknown piece kind $kind")
        }
    }

    // ---- sessions ------------------------------------------------------------------------

    class ExpectedSegment(
        val id: Int,
        val startSample: Long,
        val endSample: Long,
        val samples: Int,
        val pcmSha256: String,
        val rms: Double,
        val vadAvgProb: Double,
        val flush: Boolean,
    )

    class ExpectedPartial(val segmentId: Int, val samples: Int)

    class ExpectedRun(
        val noiseFloorRms: Double?,
        val splits: Int,
        val segments: List<ExpectedSegment>,
        val partials: List<ExpectedPartial>,
    )

    class Session(
        val name: String,
        val length: Int,
        val pcmSha256: String,
        val pieces: JsonArray,
        val probsFile: String,
        val silero: ExpectedRun,
        val energy: ExpectedRun,
    ) {
        val pcm: ShortArray by lazy { compose(pieces) }
        val probs: FloatArray by lazy { floats(probsFile) }
        override fun toString() = name
    }

    val sessionNames: List<String> by lazy { manifest.obj("sessions").keys.toList() }

    fun session(name: String): Session {
        val s = manifest.obj("sessions").obj(name)
        return Session(
            name = name,
            length = s.int("length"),
            pcmSha256 = s.str("pcmSha256"),
            pieces = s.getValue("pieces").jsonArray,
            probsFile = s.obj("silero").str("probs"),
            silero = run(s.obj("silero")),
            energy = run(s.obj("energy")),
        )
    }

    private fun run(o: JsonObject) = ExpectedRun(
        noiseFloorRms = o["noiseFloorRms"]?.takeIf { it !is JsonNull }?.jsonPrimitive?.double,
        splits = o.int("splits"),
        segments = o.getValue("segments").jsonArray.map {
            val g = it.jsonObject
            ExpectedSegment(
                id = g.int("id"),
                startSample = g.getValue("startSample").jsonPrimitive.long,
                endSample = g.getValue("endSample").jsonPrimitive.long,
                samples = g.int("samples"),
                pcmSha256 = g.str("pcmSha256"),
                rms = g.double("rms"),
                vadAvgProb = g.double("vadAvgProb"),
                flush = g.getValue("flush").jsonPrimitive.boolean,
            )
        },
        partials = o.getValue("partials").jsonArray.map {
            ExpectedPartial(it.jsonObject.int("segmentId"), it.jsonObject.int("samples"))
        },
    )

    // ---- prep fixtures -------------------------------------------------------------------

    class PrepClip(val key: String, val entry: JsonObject) {
        val length: Int get() = entry.int("length")
        val input: ShortArray by lazy {
            val src = entry["source"]
            if (src != null && src !is JsonNull) source(src.jsonPrimitive.content) else compose(entry.obj("input"))
        }
        fun has(field: String) = entry.containsKey(field)
        fun file(field: String) = entry.str(field)
        override fun toString() = key
    }

    val finalPrepClips: List<PrepClip> by lazy {
        manifest.obj("finalPrep").obj("clips").map { (k, v) -> PrepClip(k, v.jsonObject) }
    }

    val partialPrepClips: List<PrepClip> by lazy {
        manifest.obj("partialPrep").obj("clips").map { (k, v) -> PrepClip(k, v.jsonObject) }
    }

    // ---- hashing -------------------------------------------------------------------------

    fun sha256(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }

    /** SHA-256 of samples as little-endian PCM16 bytes (the server hashes its segment bytes). */
    fun sha256(pcm: ShortArray): String {
        val b = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        b.asShortBuffer().put(pcm)
        return sha256(b.array())
    }

    private fun JsonObject.obj(key: String): JsonObject = getValue(key).jsonObject
    private fun JsonObject.str(key: String): String = getValue(key).jsonPrimitive.content
    private fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int
    private fun JsonObject.double(key: String): Double = getValue(key).jsonPrimitive.double
}
