"""LLM prompts for Habla translation pipeline.

Templates live in shared/prompts.json so the Android app sends the same
prompts. This module only fills them in.
"""

from server import shared_assets
from server.config import parse_direction, lang_name


def _t(name: str) -> str:
    return shared_assets.prompts()[name]


def get_translator_system_prompt(direction: str, mode: str) -> str:
    """Get the system prompt for the translator LLM."""

    src_code, tgt_code = parse_direction(direction)
    source_lang, target_lang = lang_name(src_code), lang_name(tgt_code)

    classroom_extra = _t("translator_classroom_extra") if mode == "classroom" else ""

    return _t("translator_system").format(
        source_lang=source_lang, target_lang=target_lang, classroom_extra=classroom_extra,
    )


def build_translator_user_prompt(
    transcript: str,
    speaker_label: str,
    context_exchanges: list[dict],
    topic_summary: str = "",
    max_context: int = 5,
) -> str:
    """Build the user message for a translation request."""

    context_lines = []
    for ex in context_exchanges[-max_context:]:
        context_lines.append(_t("translator_context_line").format(
            speaker=ex.get("speaker_label", "Unknown"),
            source=ex.get("corrected", ex.get("source", "")),
            translated=ex.get("translated", ""),
        ))

    context_block = "\n".join(context_lines) if context_lines else _t("translator_no_context")
    topic_block = topic_summary if topic_summary else _t("translator_no_topic")

    return _t("translator_user").format(
        context_block=context_block, topic_block=topic_block,
        speaker_label=speaker_label, transcript=transcript,
    )


def get_quick_translate_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """One-line translation prompt for partial (in-progress) transcripts."""
    return _t("quick_translate").format(source_lang=source_lang, target_lang=target_lang, text=text)


def get_study_summary_prompt(exchanges_text: str, topic_summary: str, mode: str) -> tuple[str, str]:
    """Prompts for the post-session study summary LLM call.

    Returns (system_prompt, user_prompt). The LLM is asked to produce a
    structured JSON study report covering vocab, grammar, corrections, idioms,
    and suggested review topics.
    """
    mode_note = _t("study_summary_classroom_note") if mode == "classroom" else ""
    system = _t("study_summary_system").format(mode_note=mode_note)
    user = _t("study_summary_user").format(
        exchanges_text=exchanges_text,
        topic_block=topic_summary or _t("study_summary_no_topic"),
    )
    return system, user


def get_topic_summary_prompt(
    previous_summary: str,
    latest_source: str,
    latest_translation: str,
    speaker_label: str,
) -> str:
    """Prompt for the topic tracker (if using separate model, or same LLM)."""

    return _t("topic_summary").format(
        previous_summary=previous_summary or _t("topic_summary_no_previous"),
        speaker_label=speaker_label,
        latest_source=latest_source,
        latest_translation=latest_translation,
    )
