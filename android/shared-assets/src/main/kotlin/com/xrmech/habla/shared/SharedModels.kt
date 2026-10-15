package com.xrmech.habla.shared

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * The module's single strict JSON instance (design.md section 3). Unknown keys fail so a key added
 * to `shared/` without its Kotlin field breaks the JVM tests in the same change. Semantic rules run
 * in [Validators] after decoding, never in `init {}` blocks.
 */
internal val SharedJson: Json = Json {
    ignoreUnknownKeys = false
    isLenient = false
    allowTrailingComma = false
    allowComments = false
    allowSpecialFloatingPointValues = false
    coerceInputValues = false
    explicitNulls = false
}

// ---- asr_profile.json -----------------------------------------------------------------------

@Serializable
data class AsrProfile(
    val version: Int,
    val audio: AudioProfile,
    val vad: VadProfile,
    val calibration: CalibrationProfile,
    val segmenter: SegmenterProfile,
    /** Coordinator segment merging; semantics owned by `:app`. */
    val merge: MergeProfile,
    @SerialName("partial_prep") val partialPrep: PartialPrepProfile,
    @SerialName("final_prep") val finalPrep: FinalPrepProfile,
    val whisper: WhisperProfile,
    /** Phone-only whisper.cpp settings; the server ignores this section. Validated by `:asr`. */
    @SerialName("whisper_cpp") val whisperCpp: WhisperCppProfile,
    @SerialName("language_vote") val languageVote: LanguageVoteProfile,
    val filters: FilterProfile,
    val context: ContextProfile,
)

@Serializable
data class AudioProfile(
    @SerialName("sample_rate") val sampleRate: Int,
    val channels: Int,
)

@Serializable
data class VadProfile(
    val model: String,
    @SerialName("frame_samples") val frameSamples: Int,
    @SerialName("frame_ms") val frameMs: Int,
    @SerialName("speech_threshold") val speechThreshold: Double,
    @SerialName("energy_fallback_prob_scale") val energyFallbackProbScale: Double,
)

@Serializable
data class CalibrationProfile(
    @SerialName("duration_ms") val durationMs: Int,
    @SerialName("min_energy_threshold") val minEnergyThreshold: Double,
    @SerialName("noise_floor_multiplier") val noiseFloorMultiplier: Double,
    @SerialName("uncalibrated_threshold") val uncalibratedThreshold: Double,
)

@Serializable
data class SegmenterProfile(
    @SerialName("silence_duration_ms") val silenceDurationMs: Int,
    @SerialName("min_speech_ms") val minSpeechMs: Int,
    @SerialName("max_segment_seconds") val maxSegmentSeconds: Double,
    @SerialName("pre_speech_padding_ms") val preSpeechPaddingMs: Int,
    @SerialName("partial_interval_ms") val partialIntervalMs: Int,
)

@Serializable
data class MergeProfile(
    @SerialName("gap_seconds") val gapSeconds: Double,
    @SerialName("max_pending_seconds") val maxPendingSeconds: Double,
)

@Serializable
data class PartialPrepProfile(
    @SerialName("bandpass_hz") val bandpassHz: List<Int>,
    @SerialName("butterworth_order") val butterworthOrder: Int,
    @SerialName("target_rms") val targetRms: Double,
    @SerialName("max_gain") val maxGain: Double,
    @SerialName("min_rms_for_gain") val minRmsForGain: Double,
) {
    /** Size 2 is guaranteed by load-time validation. */
    val lowHz: Int get() = bandpassHz[0]
    val highHz: Int get() = bandpassHz[1]
}

@Serializable
data class FinalPrepProfile(
    @SerialName("highpass_hz") val highpassHz: Int,
    @SerialName("lowpass_hz") val lowpassHz: Int,
    val dynaudnorm: DynaudnormProfile,
)

@Serializable
data class DynaudnormProfile(
    @SerialName("frame_ms") val frameMs: Int,
    @SerialName("gauss_size") val gaussSize: Int,
    val peak: Double,
    @SerialName("max_gain") val maxGain: Double,
)

@Serializable
data class WhisperProfile(
    val model: String,
    @SerialName("beam_size") val beamSize: Int,
    /** Server-only (WhisperX batching); unused on the phone. */
    @SerialName("final_batch_size") val finalBatchSize: Int,
    /** Server-only (WhisperX batching); unused on the phone. */
    @SerialName("partial_batch_size") val partialBatchSize: Int,
    @SerialName("auto_language") val autoLanguage: Boolean,
    @SerialName("default_language") val defaultLanguage: String,
)

@Serializable
data class WhisperCppProfile(
    val threads: Int,
    val final: WhisperCppFinal,
    val partial: WhisperCppPartial,
    /** Null until the D5 benchmark records its provenance block. */
    val measured: WhisperCppMeasured?,
)

@Serializable
data class WhisperCppFinal(
    /** Null means "use `whisper.beam_size`". */
    @SerialName("beam_size") val beamSize: Int?,
    val temperature: Double,
    @SerialName("temperature_inc") val temperatureInc: Double,
    @SerialName("entropy_thold") val entropyThold: Double,
    @SerialName("logprob_thold") val logprobThold: Double,
    @SerialName("no_speech_thold") val noSpeechThold: Double,
)

@Serializable
data class WhisperCppPartial(
    val level: String,
    @SerialName("audio_ctx_pad_frames") val audioCtxPadFrames: Int,
    @SerialName("audio_ctx_min_frames") val audioCtxMinFrames: Int,
)

@Serializable
data class WhisperCppMeasured(
    val device: String,
    val date: String,
    val variant: String,
    @SerialName("model_id") val modelId: String,
    val threads: Int,
    @SerialName("final_p90_ms") val finalP90Ms: Int,
    @SerialName("final_rtf_p90") val finalRtfP90: Double,
    @SerialName("partial_level") val partialLevel: String,
    @SerialName("partial_p90_ms") val partialP90Ms: Int,
    @SerialName("partial_drop_ratio") val partialDropRatio: Double,
    @SerialName("pss_mb") val pssMb: Int,
    @SerialName("battery_pct_30min") val batteryPct30min: Double,
    @SerialName("max_thermal_status") val maxThermalStatus: Int,
    val report: String,
)

@Serializable
data class LanguageVoteProfile(
    val window: Int,
    @SerialName("switch_ratio") val switchRatio: Double,
)

@Serializable
data class FilterProfile(
    @SerialName("min_letters") val minLetters: Int,
    @SerialName("min_letter_ratio") val minLetterRatio: Double,
    @SerialName("clipped_onset_pairs") val clippedOnsetPairs: List<List<String>>,
) {
    /** (clipped, full) pairs; each entry has exactly two strings by load-time validation. */
    val onsetPairs: List<Pair<String, String>> get() = clippedOnsetPairs.map { it[0] to it[1] }
}

@Serializable
data class ContextProfile(
    @SerialName("recent_exchanges") val recentExchanges: Int,
    @SerialName("max_context_exchanges") val maxContextExchanges: Int,
)

// ---- prompts.json ---------------------------------------------------------------------------

/**
 * Prompt texts. TEMPLATE values go through [PromptTemplate.format]; FRAGMENT values are used
 * verbatim as values for other templates and are never formatted.
 */
@Serializable
data class Prompts(
    @SerialName("_comment") val comment: String? = null,
    /** TEMPLATE: source_lang, target_lang, classroom_extra. */
    @SerialName("translator_system") val translatorSystem: String,
    /** FRAGMENT: value for translator_system's classroom_extra in classroom mode. */
    @SerialName("translator_classroom_extra") val translatorClassroomExtra: String,
    /** TEMPLATE: context_block, topic_block, speaker_label, transcript. */
    @SerialName("translator_user") val translatorUser: String,
    /** TEMPLATE: speaker, source, translated. */
    @SerialName("translator_context_line") val translatorContextLine: String,
    /** FRAGMENT: context_block when there is no context yet. */
    @SerialName("translator_no_context") val translatorNoContext: String,
    /** FRAGMENT: topic_block when there is no summary yet. */
    @SerialName("translator_no_topic") val translatorNoTopic: String,
    /** TEMPLATE: source_lang, target_lang, transcript (retry after a passthrough translation). */
    @SerialName("translator_passthrough_retry") val translatorPassthroughRetry: String,
    /** TEMPLATE: source_lang, target_lang, text. */
    @SerialName("quick_translate") val quickTranslate: String,
    /** TEMPLATE: previous_summary, speaker_label, latest_source, latest_translation. */
    @SerialName("topic_summary") val topicSummary: String,
    /** FRAGMENT: previous_summary at the start of a conversation. */
    @SerialName("topic_summary_no_previous") val topicSummaryNoPrevious: String,
    /** TEMPLATE (not used in v1): mode_note. */
    @SerialName("study_summary_system") val studySummarySystem: String,
    /** FRAGMENT (not used in v1): mode_note for classroom sessions. */
    @SerialName("study_summary_classroom_note") val studySummaryClassroomNote: String,
    /** TEMPLATE (not used in v1): exchanges_text, topic_block. */
    @SerialName("study_summary_user") val studySummaryUser: String,
    /** FRAGMENT (not used in v1): topic_block when there is no topic. */
    @SerialName("study_summary_no_topic") val studySummaryNoTopic: String,
)

// ---- languages.json -------------------------------------------------------------------------

@Serializable
data class LanguageRegistry(
    @SerialName("_comment") val comment: String? = null,
    @SerialName("default_target") val defaultTarget: String,
    /** Insertion order of the file is preserved. */
    val languages: Map<String, LanguageInfo>,
) {
    /** Keys in file order, e.g. `["es", "en"]`. */
    val codes: List<String> get() = languages.keys.toList()

    fun name(code: String): String =
        languages[code]?.name ?: throw IllegalArgumentException("unknown language code '$code'")

    fun isSupported(code: String): Boolean = code in languages
}

@Serializable
data class LanguageInfo(val name: String, val flag: String)

// ---- models.json ----------------------------------------------------------------------------

@Serializable
internal data class ModelsFile(
    @SerialName("_comment") val comment: String? = null,
    val openai: ModelTable,
)

@Serializable
data class ModelTable(
    @SerialName("final_default") val finalDefault: String,
    @SerialName("quick_default") val quickDefault: String,
    @SerialName("fallback_order") val fallbackOrder: List<String>,
    val models: Map<String, ModelInfo>,
) {
    fun info(id: String): ModelInfo? = models[id]
}

@Serializable
data class ModelInfo(
    /** USD per 1M input tokens. */
    val input: Double,
    /** USD per 1M cached input tokens. */
    @SerialName("cached_input") val cachedInput: Double,
    /** USD per 1M output tokens. */
    val output: Double,
    /** reasoning.effort values the model accepts; empty means never send one. */
    val efforts: List<String>,
    /** The effort we send; null exactly when [efforts] is empty. */
    val effort: String? = null,
    val verbosity: Boolean,
    @SerialName("structured_outputs") val structuredOutputs: Boolean,
    /** Shutdown note; a deprecated model may not be a default or fallback. */
    val deprecated: String? = null,
)

// ---- asr_models.json ------------------------------------------------------------------------

@Serializable
data class ModelCatalog(
    @SerialName("_comment") val comment: String? = null,
    @SerialName("default_id") val defaultId: String,
    val models: List<AsrModelEntry>,
) {
    fun byId(id: String): AsrModelEntry? = models.firstOrNull { it.id == id }

    /** Exists by load-time validation. */
    val default: AsrModelEntry get() = checkNotNull(byId(defaultId)) { "default_id '$defaultId' not in models" }
}

@Serializable
data class AsrModelEntry(
    val id: String,
    /** Plain file name under `filesDir/models/`; validated against path traversal. */
    val file: String,
    val base: String,
    val quantization: String,
    /** Pinned 40-hex Hugging Face commit. */
    val revision: String,
    val url: String,
    @SerialName("size_bytes") val sizeBytes: Long,
    /** Lowercase 64-hex SHA-256 of the file. */
    val sha256: String,
)

// ---- idioms ---------------------------------------------------------------------------------

@Serializable
internal data class IdiomIndex(
    @SerialName("_comment") val comment: String? = null,
    val files: List<String>,
)

@Serializable
data class IdiomPatternDef(
    /** Python `re` pattern; compile only with [IdiomRegex.compile]. */
    val pattern: String,
    val canonical: String,
    val literal: String = "",
    val meaning: String,
    val region: String = "universal",
    val frequency: String = "common",
)
