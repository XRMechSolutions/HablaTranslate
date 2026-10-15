package com.xrmech.habla.shared

import java.io.File
import java.text.Normalizer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

class SharedAssetsRealFilesTest {

    private val sharedDir = File(System.getProperty("habla.sharedDir"))
    private val assets = SharedAssets(FileAssetSource.fromSystemProperty())

    @Test
    fun `should load every shared file with the expected values when reading the real directory`() {
        assertEquals(0.35, assets.asrProfile.vad.speechThreshold)
        assertEquals(1.2, assets.asrProfile.merge.gapSeconds)
        assertEquals("ggml-small-q8_0", assets.asrModels.default.id)
        assertEquals(listOf("es", "en"), assets.languages.codes)
        assertTrue(assets.models.info(assets.models.finalDefault) != null)
        assertEquals("greedy_scaled_ctx", assets.asrProfile.whisperCpp.partial.level)
    }

    @Test
    fun `should skip no idiom and load all entries when the shipped patterns are valid`() {
        assertEquals(emptyList(), assets.skippedIdioms)
        val spain = SharedJson.decodeFromString(
            ListSerializer(IdiomPatternDef.serializer()),
            File(sharedDir, "idioms/spain.json").readText(),
        )
        assertEquals(spain.size, assets.idioms.size)
    }

    @Test
    fun `should list every idiom file in index json when files are added to the directory`() {
        val directory = File(sharedDir, "idioms").listFiles()!!
            .map { it.name }
            .filter { it.endsWith(".json") && it != "index.json" }
            .sorted()
        val index = SharedJson.decodeFromString(IdiomIndex.serializer(), File(sharedDir, "idioms/index.json").readText())
        assertEquals(directory, index.files, "index.json files ${index.files} != directory $directory")
    }

    @Test
    fun `should contain only NFC strings when any shipped shared JSON is read`() {
        val files = sharedDir.walkTopDown()
            .onEnter { it.name != "testdata" }
            .filter { it.isFile && it.name.endsWith(".json") }
            .toList()
        assertTrue(files.size >= 7, "found $files")
        for (f in files) {
            val rel = f.relativeTo(sharedDir).invariantSeparatorsPath
            forEachString(Json.parseToJsonElement(f.readText())) { s ->
                assertTrue(Normalizer.isNormalized(s, Normalizer.Form.NFC), "$rel: not NFC: $s")
            }
        }
    }

    private fun forEachString(e: JsonElement, action: (String) -> Unit) {
        when (e) {
            is JsonObject -> e.forEach { (k, v) -> action(k); forEachString(v, action) }
            is JsonArray -> e.forEach { forEachString(it, action) }
            is JsonPrimitive -> if (e.isString) action(e.content)
        }
    }
}
