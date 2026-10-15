package com.xrmech.habla.translate

import com.xrmech.habla.shared.PromptTemplate
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode

/**
 * Fills the shared prompt templates exactly as the server's `prompts.py` builders do, so the
 * request text is byte-identical (proven by the prompts golden). [PromptTemplate.format] throws
 * `IllegalArgumentException` for a broken template; callers map that to BAD_REQUEST.
 */
internal class PromptBuilder(private val assets: SharedAssets) {

    private val prompts get() = assets.prompts

    /** Server `lang_name`: the registry name, or the code itself when unknown. */
    private fun langName(code: String): String = assets.languages.languages[code]?.name ?: code

    /** Port of `get_translator_system_prompt`. */
    fun system(direction: Direction, mode: Mode): String = PromptTemplate.format(
        prompts.translatorSystem,
        mapOf(
            "source_lang" to langName(direction.sourceCode),
            "target_lang" to langName(direction.targetCode),
            "classroom_extra" to if (mode == Mode.CLASSROOM) prompts.translatorClassroomExtra else "",
        ),
    )

    /** Port of `build_translator_user_prompt` with every speaker labelled [Translator.SPEAKER_LABEL]. */
    fun user(transcript: String, context: List<ContextExchange>, topicSummary: String): String {
        val max = assets.asrProfile.context.maxContextExchanges
        val lines = context.takeLast(max).map {
            PromptTemplate.format(
                prompts.translatorContextLine,
                mapOf("speaker" to Translator.SPEAKER_LABEL, "source" to it.source, "translated" to it.translated),
            )
        }
        val contextBlock = if (lines.isEmpty()) prompts.translatorNoContext else lines.joinToString("\n")
        val topicBlock = topicSummary.ifEmpty { prompts.translatorNoTopic }
        return PromptTemplate.format(
            prompts.translatorUser,
            mapOf(
                "context_block" to contextBlock,
                "topic_block" to topicBlock,
                "speaker_label" to Translator.SPEAKER_LABEL,
                "transcript" to transcript,
            ),
        )
    }

    /** Port of `get_quick_translate_prompt`; the caller has already mapped AUTO. */
    fun quick(text: String, direction: Direction): String = PromptTemplate.format(
        prompts.quickTranslate,
        mapOf(
            "source_lang" to langName(direction.sourceCode),
            "target_lang" to langName(direction.targetCode),
            "text" to text,
        ),
    )

    /** Port of `get_topic_summary_prompt`. */
    fun topic(previous: String, latestSource: String, latestTranslation: String): String = PromptTemplate.format(
        prompts.topicSummary,
        mapOf(
            "previous_summary" to previous.ifEmpty { prompts.topicSummaryNoPrevious },
            "speaker_label" to Translator.SPEAKER_LABEL,
            "latest_source" to latestSource,
            "latest_translation" to latestTranslation,
        ),
    )

    /**
     * The prompt `_retry_passthrough` builds: languages from `parse_direction(direction)`, which
     * maps `auto` to Spanish -> English (the [Direction] codes carry that fallback).
     */
    fun passthroughRetry(transcript: String, direction: Direction): String = PromptTemplate.format(
        prompts.translatorPassthroughRetry,
        mapOf(
            "source_lang" to langName(direction.sourceCode),
            "target_lang" to langName(direction.targetCode),
            "transcript" to transcript,
        ),
    )
}
