package com.xrmech.habla.shared

import java.io.FileNotFoundException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** Each case mutates one value of the real files in memory and asserts the exact failure. */
class SharedAssetsValidationTest {

    private val real: Map<String, String> = FileAssetSource.fromSystemProperty().let { src ->
        listOf(
            "asr_profile.json", "languages.json", "prompts.json", "models.json",
            "asr_models.json", "idioms/index.json", "idioms/spain.json",
        ).associateWith { src.readText(it) }
    }

    private fun sourceOf(files: Map<String, String>) =
        AssetSource { p -> files[p] ?: throw FileNotFoundException(p) }

    private fun mutate(path: String, old: String, new: String): Map<String, String> {
        val text = real.getValue(path)
        check(old in text) { "'$old' not found in $path" }
        return real + (path to text.replaceFirst(old, new))
    }

    private fun mutateRegex(path: String, old: Regex, new: String): Map<String, String> {
        val text = real.getValue(path)
        check(old.containsMatchIn(text)) { "$old not found in $path" }
        return real + (path to old.replaceFirst(text, new))
    }

    private fun failure(files: Map<String, String>): SharedAssetException =
        assertFailsWith<SharedAssetException> { SharedAssets(sourceOf(files)) }

    private fun assertMessage(files: Map<String, String>, expectedPrefix: String): SharedAssetException {
        val e = failure(files)
        assertTrue(e.message!!.startsWith(expectedPrefix), "expected prefix <$expectedPrefix> but was <${e.message}>")
        return e
    }

    @Test
    fun `should load when the files are unmodified`() {
        SharedAssets(sourceOf(real))
    }

    @Test
    fun `should reject speech_threshold when it is above 1`() {
        val e = failure(mutate("asr_profile.json", "\"speech_threshold\": 0.35", "\"speech_threshold\": 1.2"))
        assertEquals("shared/asr_profile.json: $.vad.speech_threshold: must be in (0, 1), was 1.2", e.message)
    }

    @Test
    fun `should reject gauss_size when it is even`() {
        assertMessage(
            mutate("asr_profile.json", "\"gauss_size\": 15", "\"gauss_size\": 16"),
            "shared/asr_profile.json: $.final_prep.dynaudnorm.gauss_size: must be odd in 3..301, was 16",
        )
    }

    @Test
    fun `should reject bandpass_hz when low is above high`() {
        assertMessage(
            mutateRegex("asr_profile.json", Regex("\"bandpass_hz\": \\[[^]]*]"), "\"bandpass_hz\": [7500, 80]"),
            "shared/asr_profile.json: $.partial_prep.bandpass_hz: must satisfy 0 < lo < hi < 8000, was [7500, 80]",
        )
    }

    @Test
    fun `should reject beam_size when it exceeds whisper max decoders`() {
        assertMessage(
            mutate("asr_profile.json", "\"beam_size\": 5", "\"beam_size\": 9"),
            "shared/asr_profile.json: $.whisper.beam_size: must be in 1..8, was 9",
        )
    }

    @Test
    fun `should reject default_language when it is not in languages json`() {
        assertMessage(
            mutate("asr_profile.json", "\"default_language\": \"es\"", "\"default_language\": \"fr\""),
            "shared/asr_profile.json: $.whisper.default_language: 'fr' not in languages.json codes [es, en]",
        )
    }

    @Test
    fun `should reject final_default when the model is deprecated`() {
        val e = assertMessage(
            mutate("models.json", "\"final_default\": \"gpt-5.4-mini\"", "\"final_default\": \"gpt-5.4-nano\""),
            "shared/models.json: $.openai.final_default: model 'gpt-5.4-nano' is deprecated",
        )
        assertTrue("shutdown 2027-04-01" in e.message!!)
    }

    @Test
    fun `should reject effort when it is not in efforts`() {
        assertMessage(
            mutate("models.json", "\"effort\": \"none\"", "\"effort\": \"turbo\""),
            "shared/models.json: $.openai.models['gpt-6-luna'].effort: must be one of [none, low, medium, high, xhigh, max], was 'turbo'",
        )
    }

    @Test
    fun `should reject an asr model file when it is path-like`() {
        assertMessage(
            mutate("asr_models.json", "\"file\": \"ggml-small-q8_0.bin\"", "\"file\": \"../../databases/habla.db\""),
            "shared/asr_models.json: $.models[0].file: ",
        )
    }

    @Test
    fun `should reject an asr model revision when it is not pinned`() {
        assertMessage(
            mutate("asr_models.json", "\"revision\": \"5359861c739e955e79d9a303bcbc70fb988958b1\"", "\"revision\": \"main\""),
            "shared/asr_models.json: $.models[0].revision: must be a 40-char lowercase hex commit, was 'main'",
        )
    }

    @Test
    fun `should reject an asr model sha256 when it is uppercase`() {
        val sha = "49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f"
        assertMessage(
            mutate("asr_models.json", sha, sha.uppercase()),
            "shared/asr_models.json: $.models[0].sha256: must be 64 lowercase hex chars",
        )
    }

    @Test
    fun `should reject an asr model size when it is zero`() {
        assertMessage(
            mutate("asr_models.json", "\"size_bytes\": 264464607", "\"size_bytes\": 0"),
            "shared/asr_models.json: $.models[0].size_bytes: must be in 1..4000000000, was 0",
        )
    }

    @Test
    fun `should reject the idiom index when it lists itself`() {
        assertMessage(
            mutate("idioms/index.json", "[\"spain.json\"]", "[\"spain.json\", \"index.json\"]"),
            "shared/idioms/index.json: $.files[1]: 'index.json' is the index itself",
        )
    }

    @Test
    fun `should reject the idiom index when it is unsorted`() {
        assertMessage(
            mutate("idioms/index.json", "[\"spain.json\"]", "[\"spain.json\", \"alpha.json\"]"),
            "shared/idioms/index.json: $.files[1]: 'alpha.json' must sort strictly after 'spain.json'",
        )
    }

    @Test
    fun `should reject the idiom index when a name has uppercase letters`() {
        assertMessage(
            mutate("idioms/index.json", "[\"spain.json\"]", "[\"Spain.json\"]"),
            "shared/idioms/index.json: $.files[0]: 'Spain.json' must match ^[a-z0-9_]+\\.json$",
        )
    }

    @Test
    fun `should reject a template when a placeholder is typo'd`() {
        assertMessage(
            mutate("prompts.json", "{text}", "{txt}"),
            "shared/prompts.json: $.quick_translate: placeholders [source_lang, target_lang, txt] != expected [source_lang, target_lang, text]",
        )
    }

    @Test
    fun `should reject a template when it has a stray brace`() {
        assertMessage(
            mutate("prompts.json", "{text}", "{text} }"),
            "shared/prompts.json: $.quick_translate: Single '}' encountered in format string at index",
        )
    }

    @Test
    fun `should reject a file when it starts with a BOM`() {
        val e = failure(real + ("languages.json" to "﻿" + real.getValue("languages.json")))
        assertEquals("shared/languages.json: UTF-8 BOM not allowed", e.message)
    }

    @Test
    fun `should report file not found with the cause when a file is missing`() {
        val e = failure(real - "asr_models.json")
        assertEquals("shared/asr_models.json: file not found", e.message)
        assertIs<FileNotFoundException>(e.cause)
    }

    @Test
    fun `should report read failed when the source throws another IOException`() {
        val files = real
        val e = assertFailsWith<SharedAssetException> {
            SharedAssets { p -> if (p == "prompts.json") throw java.io.IOException("disk gone") else files.getValue(p) }
        }
        assertEquals("shared/prompts.json: read failed: disk gone", e.message)
    }

    @Test
    fun `should name the key when a file has an unknown key`() {
        val e = assertMessage(mutate("languages.json", "\"default_target\"", "\"foo\": 1, \"default_target\""), "shared/languages.json: ")
        assertTrue("foo" in e.message!!, e.message)
        assertIs<kotlinx.serialization.SerializationException>(e.cause)
    }

    @Test
    fun `should reject an idiom file when an entry has a blank meaning`() {
        assertMessage(
            mutateRegex("idioms/spain.json", Regex("\"meaning\":\"[^\"]*\""), "\"meaning\":\" \""),
            "shared/idioms/spain.json: $[0].meaning: must not be blank",
        )
    }

    @Test
    fun `should skip and record an idiom when its pattern is outside the whitelist`() {
        val files = mutateRegex("idioms/spain.json", Regex("\"pattern\":\"[^\"]*\""), "\"pattern\":\"(?P<x>a)\"")
        val assets = SharedAssets(sourceOf(files))
        assertEquals(1, assets.skippedIdioms.size)
        assertTrue(
            assets.skippedIdioms[0].startsWith("shared/idioms/spain.json[0]: unsupported regex construct '(?P' at index 0"),
            assets.skippedIdioms[0],
        )
        assertEquals(23, assets.idioms.size)
    }

    @Test
    fun `should skip and record an idiom when its pattern has a syntax error`() {
        val files = mutateRegex("idioms/spain.json", Regex("\"pattern\":\"[^\"]*\""), "\"pattern\":\"(unclosed\"")
        val assets = SharedAssets(sourceOf(files))
        assertEquals(1, assets.skippedIdioms.size)
        assertTrue(assets.skippedIdioms[0].startsWith("shared/idioms/spain.json[0]: "), assets.skippedIdioms[0])
        assertEquals(23, assets.idioms.size)
    }

    @Test
    fun `should reject default_target when it is not a language code`() {
        assertMessage(
            mutate("languages.json", "\"default_target\": \"en\"", "\"default_target\": \"fr\""),
            "shared/languages.json: $.default_target: 'fr' not in languages [es, en]",
        )
    }

    @Test
    fun `should reject a decimal number when the field is an Int`() {
        val e = assertMessage(
            mutate("asr_profile.json", "\"silence_duration_ms\": 600", "\"silence_duration_ms\": 600.0"),
            "shared/asr_profile.json: ",
        )
        assertIs<kotlinx.serialization.SerializationException>(e.cause)
    }

    @Test
    fun `should reject NaN when a profile value is not finite`() {
        val e = assertMessage(
            mutate("asr_profile.json", "\"speech_threshold\": 0.35", "\"speech_threshold\": NaN"),
            "shared/asr_profile.json: ",
        )
        assertIs<kotlinx.serialization.SerializationException>(e.cause)
    }

    @Test
    fun `should reject null when a required value is null`() {
        val e = assertMessage(
            mutate("asr_profile.json", "\"speech_threshold\": 0.35", "\"speech_threshold\": null"),
            "shared/asr_profile.json: ",
        )
        assertIs<kotlinx.serialization.SerializationException>(e.cause)
    }

    @Test
    fun `should reject effort when the model lists no efforts`() {
        assertMessage(
            mutate("models.json", "\"efforts\": [],", "\"efforts\": [], \"effort\": \"low\","),
            "shared/models.json: $.openai.models['",
        )
    }
}
