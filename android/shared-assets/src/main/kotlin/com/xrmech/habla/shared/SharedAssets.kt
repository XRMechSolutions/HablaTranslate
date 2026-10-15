package com.xrmech.habla.shared

import java.io.FileNotFoundException
import java.io.IOException
import kotlinx.serialization.DeserializationStrategy
import kotlinx.serialization.builtins.ListSerializer

/**
 * Every `shared/` file, decoded strictly and validated, loaded eagerly in the constructor
 * (blocking I/O: build it off the main thread). Immutable and thread-safe after construction.
 *
 * @throws SharedAssetException for the first problem found; the message starts with `shared/<path>: `.
 */
class SharedAssets(source: AssetSource) {
    val asrProfile: AsrProfile
    val prompts: Prompts
    val languages: LanguageRegistry
    /** The `openai` section of models.json. */
    val models: ModelTable
    /** Index order, then file order; entries whose pattern fails [IdiomRegex.compile] are removed. */
    val idioms: List<IdiomPatternDef>
    val asrModels: ModelCatalog
    /** One `shared/idioms/<file>[<i>]: <message>` per skipped idiom entry; empty for shipped files. */
    val skippedIdioms: List<String>

    init {
        val profile = load(source, ASR_PROFILE, AsrProfile.serializer(), Validators::asrProfile)
        languages = load(source, LANGUAGES, LanguageRegistry.serializer(), Validators::languages)
        Validators.asrProfileLanguage(profile, languages)?.let { throw SharedAssetException("shared/$ASR_PROFILE: $it") }
        asrProfile = profile
        prompts = load(source, PROMPTS, Prompts.serializer(), Validators::prompts)
        models = load(source, MODELS, ModelsFile.serializer()) { Validators.models(it.openai) }.openai
        asrModels = load(source, ASR_MODELS, ModelCatalog.serializer(), Validators::asrModels)

        val index = load(source, IDIOM_INDEX, IdiomIndex.serializer(), Validators::idiomIndex)
        val loaded = ArrayList<IdiomPatternDef>()
        val skipped = ArrayList<String>()
        for (name in index.files) {
            val path = "idioms/$name"
            val entries = load(source, path, ListSerializer(IdiomPatternDef.serializer()), Validators::idioms)
            entries.forEachIndexed { i, entry ->
                // Server parity: a pattern Python cannot compile is skipped, not fatal.
                try {
                    IdiomRegex.compile(entry.pattern)
                    loaded += entry
                } catch (e: IllegalArgumentException) {
                    skipped += "shared/$path[$i]: ${e.message}"
                }
            }
        }
        idioms = loaded.toList()
        skippedIdioms = skipped.toList()
    }

    private companion object {
        const val ASR_PROFILE = "asr_profile.json"
        const val LANGUAGES = "languages.json"
        const val PROMPTS = "prompts.json"
        const val MODELS = "models.json"
        const val ASR_MODELS = "asr_models.json"
        const val IDIOM_INDEX = "idioms/index.json"

        fun <T> load(
            source: AssetSource,
            path: String,
            deserializer: DeserializationStrategy<T>,
            validate: (T) -> String?,
        ): T {
            val text = try {
                source.readText(path)
            } catch (e: FileNotFoundException) {
                throw SharedAssetException("shared/$path: file not found", e)
            } catch (e: IOException) {
                throw SharedAssetException("shared/$path: read failed: ${e.message}", e)
            }
            if (text.startsWith('﻿')) throw SharedAssetException("shared/$path: UTF-8 BOM not allowed")
            val value = try {
                SharedJson.decodeFromString(deserializer, text)
            } catch (e: IllegalArgumentException) {
                // SerializationException extends IllegalArgumentException; both mean malformed content.
                throw SharedAssetException("shared/$path: ${e.message}", e)
            }
            validate(value)?.let { throw SharedAssetException("shared/$path: $it") }
            return value
        }
    }
}
