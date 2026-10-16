package com.xrmech.habla.shared

/**
 * Semantic validation per shared file (design.md sections 5-7): physical sanity only; consuming
 * modules enforce their own algorithm limits. Each function returns the first violation as
 * `"<json path>: <rule>, was <value>"`, or null when the file is valid.
 */
internal object Validators {

    /** Expected placeholder set per TEMPLATE key (mirrors habla/tests/test_shared_assets.py). */
    val EXPECTED_FIELDS: Map<String, Set<String>> = linkedMapOf(
        "translator_system" to setOf("source_lang", "target_lang", "classroom_extra"),
        "translator_user" to setOf("context_block", "topic_block", "speaker_label", "transcript"),
        "translator_context_line" to setOf("speaker", "source", "translated"),
        "translator_passthrough_retry" to setOf("source_lang", "target_lang", "transcript"),
        "quick_translate" to setOf("source_lang", "target_lang", "text"),
        "topic_summary" to setOf("previous_summary", "speaker_label", "latest_source", "latest_translation"),
        "study_summary_system" to setOf("mode_note"),
        "study_summary_user" to setOf("exchanges_text", "topic_block"),
    )

    /** Nyquist frequency for the 16 kHz audio the profile requires. */
    private const val NYQUIST_HZ = 8000

    /** whisper.cpp WHISPER_MAX_DECODERS; a larger beam fails natively. */
    private const val MAX_BEAM = 8

    /** Largest ggml Whisper model is ~3.1 GB. */
    private const val MAX_MODEL_BYTES = 4_000_000_000L

    private val LANGUAGE_CODE = Regex("^[a-z]{2}$")
    private val MODEL_ID = Regex("^[a-z0-9._-]+$")
    private val MODEL_FILE = Regex("^[A-Za-z0-9._-]+\\.bin$")
    private val REVISION = Regex("^[0-9a-f]{40}$")
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val IDIOM_FILE = Regex("^[a-z0-9_]+\\.json$")
    const val IDIOM_INDEX = "index.json"

    /** Collects only the first failing rule; later checks are still cheap to evaluate. */
    private class FirstViolation {
        var message: String? = null
            private set

        fun rule(ok: Boolean, path: String, rule: String, value: Any?) {
            if (!ok && message == null) message = "$path: $rule, was ${show(value)}"
        }

        fun fail(path: String, text: String) {
            if (message == null) message = "$path: $text"
        }
    }

    private fun show(value: Any?): String = when (value) {
        is String -> "'$value'"
        is List<*> -> value.joinToString(", ", "[", "]") { show(it) }
        else -> value.toString()
    }

    fun asrProfile(p: AsrProfile): String? {
        val v = FirstViolation()
        v.rule(p.version == 1, "$.version", "must be 1", p.version)
        v.rule(p.audio.sampleRate == 16000, "$.audio.sample_rate", "must be 16000", p.audio.sampleRate)
        v.rule(p.audio.channels == 1, "$.audio.channels", "must be 1", p.audio.channels)

        val vad = p.vad
        v.rule(vad.frameSamples == 512, "$.vad.frame_samples", "must be 512", vad.frameSamples)
        val expectedFrameMs = if (p.audio.sampleRate > 0) vad.frameSamples * 1000 / p.audio.sampleRate else -1
        v.rule(vad.frameMs == expectedFrameMs, "$.vad.frame_ms", "must equal frame_samples*1000/sample_rate ($expectedFrameMs)", vad.frameMs)
        v.rule(vad.speechThreshold > 0.0 && vad.speechThreshold < 1.0, "$.vad.speech_threshold", "must be in (0, 1)", vad.speechThreshold)
        v.rule(vad.energyFallbackProbScale > 0.0, "$.vad.energy_fallback_prob_scale", "must be > 0", vad.energyFallbackProbScale)

        val cal = p.calibration
        v.rule(cal.durationMs > 0, "$.calibration.duration_ms", "must be > 0", cal.durationMs)
        v.rule(cal.minEnergyThreshold > 0.0, "$.calibration.min_energy_threshold", "must be > 0", cal.minEnergyThreshold)
        v.rule(cal.noiseFloorMultiplier >= 1.0, "$.calibration.noise_floor_multiplier", "must be >= 1", cal.noiseFloorMultiplier)
        v.rule(cal.uncalibratedThreshold > 0.0, "$.calibration.uncalibrated_threshold", "must be > 0", cal.uncalibratedThreshold)

        val seg = p.segmenter
        v.rule(seg.silenceDurationMs > 0, "$.segmenter.silence_duration_ms", "must be > 0", seg.silenceDurationMs)
        v.rule(seg.minSpeechMs >= 0, "$.segmenter.min_speech_ms", "must be >= 0", seg.minSpeechMs)
        v.rule(seg.maxSegmentSeconds > 0.0, "$.segmenter.max_segment_seconds", "must be > 0", seg.maxSegmentSeconds)
        v.rule(seg.preSpeechPaddingMs >= 0, "$.segmenter.pre_speech_padding_ms", "must be >= 0", seg.preSpeechPaddingMs)
        v.rule(seg.partialIntervalMs > 0, "$.segmenter.partial_interval_ms", "must be > 0", seg.partialIntervalMs)

        val pp = p.partialPrep
        if (pp.bandpassHz.size != 2) {
            v.rule(false, "$.partial_prep.bandpass_hz", "must have exactly 2 entries", pp.bandpassHz)
        } else {
            val (lo, hi) = pp.bandpassHz
            v.rule(lo > 0 && lo < hi && hi < NYQUIST_HZ, "$.partial_prep.bandpass_hz", "must satisfy 0 < lo < hi < $NYQUIST_HZ", pp.bandpassHz)
        }
        v.rule(pp.butterworthOrder >= 1, "$.partial_prep.butterworth_order", "must be >= 1", pp.butterworthOrder)
        v.rule(pp.targetRms > 0.0, "$.partial_prep.target_rms", "must be > 0", pp.targetRms)
        v.rule(pp.maxGain >= 1.0, "$.partial_prep.max_gain", "must be >= 1", pp.maxGain)
        v.rule(pp.minRmsForGain >= 0.0, "$.partial_prep.min_rms_for_gain", "must be >= 0", pp.minRmsForGain)

        val fp = p.finalPrep
        v.rule(fp.highpassHz > 0, "$.final_prep.highpass_hz", "must be > 0", fp.highpassHz)
        v.rule(fp.lowpassHz > fp.highpassHz && fp.lowpassHz < NYQUIST_HZ, "$.final_prep.lowpass_hz", "must satisfy highpass_hz < lowpass_hz < $NYQUIST_HZ", fp.lowpassHz)
        val dn = fp.dynaudnorm
        v.rule(dn.frameMs in 10..8000, "$.final_prep.dynaudnorm.frame_ms", "must be in 10..8000", dn.frameMs)
        v.rule(dn.gaussSize in 3..301 && dn.gaussSize % 2 == 1, "$.final_prep.dynaudnorm.gauss_size", "must be odd in 3..301", dn.gaussSize)
        v.rule(dn.peak > 0.0 && dn.peak <= 1.0, "$.final_prep.dynaudnorm.peak", "must be in (0, 1]", dn.peak)
        v.rule(dn.maxGain >= 1.0 && dn.maxGain <= 100.0, "$.final_prep.dynaudnorm.max_gain", "must be in 1..100", dn.maxGain)

        val w = p.whisper
        v.rule(w.beamSize in 1..MAX_BEAM, "$.whisper.beam_size", "must be in 1..$MAX_BEAM", w.beamSize)
        v.rule(w.finalBatchSize >= 1, "$.whisper.final_batch_size", "must be >= 1", w.finalBatchSize)
        v.rule(w.partialBatchSize >= 1, "$.whisper.partial_batch_size", "must be >= 1", w.partialBatchSize)

        val lv = p.languageVote
        v.rule(lv.window >= 1, "$.language_vote.window", "must be >= 1", lv.window)
        v.rule(lv.switchRatio > 0.0 && lv.switchRatio <= 1.0, "$.language_vote.switch_ratio", "must be in (0, 1]", lv.switchRatio)

        val f = p.filters
        v.rule(f.minLetters >= 0, "$.filters.min_letters", "must be >= 0", f.minLetters)
        v.rule(f.minLetterRatio in 0.0..1.0, "$.filters.min_letter_ratio", "must be in [0, 1]", f.minLetterRatio)
        f.clippedOnsetPairs.forEachIndexed { i, pair ->
            v.rule(pair.size == 2 && pair.all { it.isNotEmpty() }, "$.filters.clipped_onset_pairs[$i]", "must be 2 non-empty strings", pair)
        }

        val c = p.context
        v.rule(c.recentExchanges >= 1, "$.context.recent_exchanges", "must be >= 1", c.recentExchanges)
        v.rule(c.maxContextExchanges >= 1, "$.context.max_context_exchanges", "must be >= 1", c.maxContextExchanges)
        v.rule(c.maxContextExchanges <= c.recentExchanges, "$.context.max_context_exchanges", "must be <= recent_exchanges (${c.recentExchanges})", c.maxContextExchanges)
        return v.message
    }

    /** `whisper.default_language` must be a language code; checked once languages.json is loaded. */
    fun asrProfileLanguage(p: AsrProfile, languages: LanguageRegistry): String? {
        val code = p.whisper.defaultLanguage
        if (languages.isSupported(code)) return null
        return "$.whisper.default_language: '$code' not in languages.json codes ${languages.codes}"
    }

    fun languages(r: LanguageRegistry): String? {
        val v = FirstViolation()
        v.rule(r.languages.size >= 2, "$.languages", "must have at least 2 entries", r.codes)
        for ((code, info) in r.languages) {
            v.rule(LANGUAGE_CODE.matches(code), "$.languages", "codes must match ^[a-z]{2}$", code)
            v.rule(info.name.isNotBlank(), "$.languages.$code.name", "must not be blank", info.name)
            v.rule(info.flag.isNotBlank(), "$.languages.$code.flag", "must not be blank", info.flag)
        }
        if (!r.isSupported(r.defaultTarget)) {
            v.fail("$.default_target", "'${r.defaultTarget}' not in languages ${r.codes}")
        }
        return v.message
    }

    /** Placeholder set equality per TEMPLATE; a scanner syntax error is reported against its key. */
    fun prompts(p: Prompts): String? {
        val templates = mapOf(
            "translator_system" to p.translatorSystem,
            "translator_user" to p.translatorUser,
            "translator_context_line" to p.translatorContextLine,
            "translator_passthrough_retry" to p.translatorPassthroughRetry,
            "quick_translate" to p.quickTranslate,
            "topic_summary" to p.topicSummary,
            "study_summary_system" to p.studySummarySystem,
            "study_summary_user" to p.studySummaryUser,
        )
        for ((key, expected) in EXPECTED_FIELDS) {
            val template = templates.getValue(key)
            val actual = try {
                PromptTemplate.fields(template)
            } catch (e: IllegalArgumentException) {
                return "$.$key: ${e.message}"
            }
            if (actual.toSet() != expected) {
                return "$.$key: placeholders ${actual.sorted()} != expected ${expected.sorted()}"
            }
        }
        return null
    }

    fun models(t: ModelTable): String? {
        val v = FirstViolation()
        fun defaultRule(path: String, id: String) {
            val info = t.models[id]
            when {
                info == null -> v.fail(path, "model '$id' not in models")
                info.deprecated != null -> v.fail(path, "model '$id' is deprecated (${info.deprecated})")
            }
        }
        defaultRule("$.openai.final_default", t.finalDefault)
        defaultRule("$.openai.quick_default", t.quickDefault)
        v.rule(t.fallbackOrder.isNotEmpty(), "$.openai.fallback_order", "must not be empty", t.fallbackOrder)
        t.fallbackOrder.forEachIndexed { i, id -> defaultRule("$.openai.fallback_order[$i]", id) }
        for ((id, m) in t.models) {
            val path = "$.openai.models['$id']"
            v.rule(m.input >= 0.0, "$path.input", "must be >= 0", m.input)
            v.rule(m.cachedInput >= 0.0, "$path.cached_input", "must be >= 0", m.cachedInput)
            v.rule(m.output >= 0.0, "$path.output", "must be >= 0", m.output)
            if (m.efforts.isEmpty()) {
                v.rule(m.effort == null, "$path.effort", "must be absent when efforts is empty", m.effort)
            } else {
                v.rule(m.effort != null && m.effort in m.efforts, "$path.effort", "must be one of ${m.efforts}", m.effort)
            }
        }
        return v.message
    }

    private val MODEL_ROLES = listOf(AsrModelEntry.ROLE_FINAL, AsrModelEntry.ROLE_PARTIAL)

    fun asrModels(c: ModelCatalog): String? {
        val v = FirstViolation()
        val seen = HashSet<String>()
        c.models.forEachIndexed { i, m ->
            val path = "$.models[$i]"
            v.rule(MODEL_ID.matches(m.id), "$path.id", "must match ${MODEL_ID.pattern}", m.id)
            v.rule(seen.add(m.id), "$path.id", "must be unique", m.id)
            v.rule(MODEL_FILE.matches(m.file), "$path.file", "must match ${MODEL_FILE.pattern}", m.file)
            v.rule(m.base.isNotBlank(), "$path.base", "must not be blank", m.base)
            v.rule(m.quantization.isNotBlank(), "$path.quantization", "must not be blank", m.quantization)
            v.rule(REVISION.matches(m.revision), "$path.revision", "must be a 40-char lowercase hex commit", m.revision)
            v.rule(
                m.url.startsWith("https://huggingface.co/") &&
                    m.url.contains("/resolve/${m.revision}/") &&
                    m.url.endsWith("/${m.file}"),
                "$path.url",
                "must start with https://huggingface.co/, contain /resolve/<revision>/ and end with /<file>",
                m.url,
            )
            v.rule(SHA256.matches(m.sha256), "$path.sha256", "must be 64 lowercase hex chars", m.sha256)
            v.rule(m.sizeBytes in 1..MAX_MODEL_BYTES, "$path.size_bytes", "must be in 1..$MAX_MODEL_BYTES", m.sizeBytes)
            v.rule(
                m.roles == null || (m.roles.isNotEmpty() && m.roles.toSet().size == m.roles.size && m.roles.all { it in MODEL_ROLES }),
                "$path.roles",
                "must be a non-empty list of distinct values from $MODEL_ROLES",
                m.roles,
            )
        }
        val default = c.byId(c.defaultId)
        if (default == null) {
            v.fail("$.default_id", "'${c.defaultId}' not in models ${c.models.map { it.id }}")
        } else if (!default.usableForFinals) {
            v.fail("$.default_id", "'${c.defaultId}' must have the '${AsrModelEntry.ROLE_FINAL}' role, has ${default.roles}")
        }
        return v.message
    }

    fun idiomIndex(index: IdiomIndex): String? {
        val v = FirstViolation()
        index.files.forEachIndexed { i, name ->
            val path = "$.files[$i]"
            when {
                !IDIOM_FILE.matches(name) -> v.fail(path, "'$name' must match ${IDIOM_FILE.pattern}")
                name == IDIOM_INDEX -> v.fail(path, "'$name' is the index itself")
                i > 0 && name <= index.files[i - 1] ->
                    v.fail(path, "'$name' must sort strictly after '${index.files[i - 1]}' (sorted, unique)")
            }
        }
        return v.message
    }

    fun idioms(entries: List<IdiomPatternDef>): String? {
        val v = FirstViolation()
        entries.forEachIndexed { i, e ->
            v.rule(e.pattern.isNotBlank(), "$[$i].pattern", "must not be blank", e.pattern)
            v.rule(e.canonical.isNotBlank(), "$[$i].canonical", "must not be blank", e.canonical)
            v.rule(e.meaning.isNotBlank(), "$[$i].meaning", "must not be blank", e.meaning)
        }
        return v.message
    }
}
