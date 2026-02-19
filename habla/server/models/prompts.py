"""LLM prompts for Habla translation pipeline."""


def get_translator_system_prompt(direction: str, mode: str) -> str:
    """Get the system prompt for the translator LLM."""

    if direction == "es_to_en":
        source_lang, target_lang = "Spanish", "English"
    else:
        source_lang, target_lang = "English", "Spanish"

    classroom_extra = ""
    if mode == "classroom":
        classroom_extra = """
Additionally, in classroom mode:
- Be more aggressive about flagging vocabulary worth studying.
- Watch for one speaker correcting another's grammar. If detected, set
  "is_correction" to true and fill in "correction_detail" with the wrong
  form, correct form, and a brief grammatical explanation.
- Flag false friends and common learner mistakes.
"""

    return f"""You are a {source_lang}-to-{target_lang} interpreter for a live conversation.

You receive a speech-to-text transcription that may contain ASR errors,
along with speaker information and recent conversation context.

Tasks:
1. CORRECT: Fix likely ASR errors. Consider homophones, dropped words, and
   context. Output the corrected {source_lang} text.
2. TRANSLATE: Produce natural, idiomatic {target_lang}. NEVER leave {source_lang}
   words untranslated in the output. Handle idioms by meaning, not literally.
3. FLAG PHRASES: Mark notable items worth learning — idioms, slang, false
   friends, regional expressions, interesting grammar constructions.
4. SPEAKER HINT: If you can infer the speaker's role (teacher, student,
   shopkeeper, friend, doctor...) from what they say, suggest it.
{classroom_extra}
IMPORTANT: Respond ONLY with a SINGLE valid JSON object. No explanation outside the JSON.
If you output anything else, the response will be discarded. Do NOT add markdown or code fences.

Output format:
{{
  "corrected": "corrected {source_lang} text",
  "translated": "natural {target_lang} translation",
  "flagged_phrases": [
    {{
      "phrase": "the {source_lang} phrase",
      "literal": "word-for-word {target_lang}",
      "meaning": "actual meaning in {target_lang}",
      "type": "idiom | slang | false_friend | correction | grammar_note",
      "save_worthy": true
    }}
  ],
  "confidence": 0.95,
  "speaker_hint": "role or null",
  "is_correction": false,
  "correction_detail": null
}}"""


def build_translator_user_prompt(
    transcript: str,
    speaker_label: str,
    context_exchanges: list[dict],
    topic_summary: str = "",
) -> str:
    """Build the user message for a translation request."""

    context_lines = []
    for ex in context_exchanges[-5:]:
        spk = ex.get("speaker_label", "Unknown")
        src = ex.get("corrected", ex.get("source", ""))
        tgt = ex.get("translated", "")
        context_lines.append(f"  [{spk}] {src} → {tgt}")

    context_block = "\n".join(context_lines) if context_lines else "(none yet)"
    topic_block = topic_summary if topic_summary else "(no summary yet)"

    return f"""Recent conversation:
{context_block}

Topic context: {topic_block}

Current utterance:
Speaker: {speaker_label}
Transcript: "{transcript}"

Translate now. Respond with a single JSON object only. No extra text."""


def get_topic_summary_prompt(
    previous_summary: str,
    latest_source: str,
    latest_translation: str,
    speaker_label: str,
) -> str:
    """Prompt for the topic tracker (if using separate model, or same LLM)."""

    return f"""Update the conversation summary. Keep it under 100 words.
Track: main topics, key entities, speaking style, domain context.

Previous summary: {previous_summary or "(start of conversation)"}

Latest exchange:
  [{speaker_label}] {latest_source} → {latest_translation}

Updated summary (plain text, no JSON):"""
