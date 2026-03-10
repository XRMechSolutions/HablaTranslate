# Annotation Pipeline

The annotation pipeline enriches raw translations with idiom highlights, speaker attribution, correction detection, and vocabulary flags. It runs after translation and before the result reaches the client.

## Idiom Detection (Dual-Layer)

Two independent systems detect idioms. Pattern DB results take priority during dedup.

### Layer 1: IdiomScanner (Regex Pattern DB)

`services/idiom_scanner.py` (309L) — Pre-compiled regex patterns, <10ms, CPU-only.

**Pattern sources**:
- Static: `data/idioms/spain.json` (26 starter patterns covering universal + Spain-specific idioms)
- Dynamic: `idiom_patterns` DB table (user-contributed)

Both are loaded at startup. The scanner reloads when a new pattern is added via the API.

**Matching**: Single-pass regex scan. Deduplicates by canonical form (lowercased). Returns `IdiomMatch` objects with canonical form, literal translation, meaning, region, frequency, and character span offsets.

**Pattern generation** (for user-saved idioms): The API auto-generates a regex from the phrase. Verbs get conjugation-flexible stems (strip -ar/-er/-ir/-arse/-erse/-irse endings), articles become optional groups, other words are escaped literals. Example: `"tomar el pelo"` → `r"tomar\s+(el\s+)?pelo"`.

### Layer 2: LLM Detection

The translation prompt instructs the LLM to flag idioms, slang, false friends, and grammar notes in `flagged_phrases[]`. These are novel or context-dependent phrases that the regex DB can't catch.

### Merge Strategy (`context_manager._merge_idioms()`)

1. Pattern DB matches first — convert to `FlaggedPhrase` with `source="pattern_db"`
2. LLM-detected phrases — skip any already seen (case-insensitive dedup), set `source="llm"`
3. Metrics: `idiom_pattern_db_hits` and `idiom_llm_hits` tracked separately

When a user saves an LLM-detected idiom, the system generates a regex and inserts it into the pattern DB. Over time, novel LLM detections feed back into the fast path.

## Speaker Tracking

`services/speaker_tracker.py` (78L) — In-memory speaker identity per session.

**Auto-labeling**: Speakers are assigned labels A, B, C... in order of first appearance. Pyannote provides `SPEAKER_00`, `SPEAKER_01`, etc. — the tracker maps these to human-friendly labels.

**Color rotation**: 8-color palette (blue, green, amber, red, violet, pink, cyan, orange) cycles through speakers.

**Role hints**: The LLM suggests speaker roles from context (teacher, student, shopkeeper). These are stored as non-authoritative hints that the user can override.

**Custom names**: Users can rename speakers (e.g., "SPEAKER_00" → "Profesor Garcia") via the UI.

**Persistence**: In-memory by design. Speaker profiles are saved to `data/last_session.json` on shutdown and restored on startup for continuity. Not stored in DB per-session (speaker labels in `exchanges` and `speakers` tables are written at exchange time).

## Correction Detection (Classroom Mode)

In classroom mode, the translation prompt includes additional instructions to detect when one speaker corrects another's grammar. The LLM sets:

- `is_correction: true`
- `correction_detail: {wrong, right, explanation}`

The client renders these as structured correction cards with the wrong form, correct form, and explanation. Corrections are saved to the vocab DB for spaced repetition review.

**Correction tracking**: Per-speaker correction frequency is tracked in metrics. The `GET /api/system/metrics` endpoint reports `corrections.by_speaker` with total exchanges and correction count per speaker.

## Context Manager

`pipeline/context_manager.py` (116L) — Lightweight state holder, not a logic-heavy module.

**Owns**:
- `translator: Translator` — LLM client instance
- `speaker_tracker: SpeakerTracker` — in-memory speaker registry
- `idiom_scanner: IdiomScanner` — regex pattern DB
- `direction: str` — "es_to_en", "en_to_es", or "auto"
- `mode: str` — "conversation" or "classroom"
- `topic_summary: str` — LLM-maintained rolling summary
- `recent_exchanges: deque[dict]` — Last 10 exchanges (context window for LLM)
- `session_id: int | None` — Current DB session

**Key methods**:
- `_merge_idioms()` — Combine pattern DB + LLM detections
- `_update_topic()` — Fire-and-forget topic summary update via translator
- `_save_audio_clip()` — Copy WAV to clips directory if enabled

Does NOT own DB access — that stays in the orchestrator to keep test patches simple.

## Design Rationale

**Why dual-layer idiom detection?** Regex is fast (<10ms) and reliable for known patterns. The LLM catches novel idioms in context but costs ~2-5s. Running both means known idioms are highlighted instantly while new ones are still caught.

**Why pattern DB priority in dedup?** Pattern DB matches include curated metadata (literal translation, region, frequency) that's higher quality than what the LLM generates ad-hoc. When both match the same phrase, the curated version wins.

**Why in-memory speaker tracking?** Speaker identity is ephemeral — labels like "Speaker A" only matter within a session. Persisting to DB mid-session would add write latency for no benefit. The tracker is saved to JSON on shutdown for continuity.

**Why a separate ContextManager?** The orchestrator was growing too large. Extracting context state into its own module makes it independently testable and keeps the orchestrator focused on coordination.

## Planned Direction

- **Grammar rule engine** (7.6): Port Spanish conjugation rules from ConjugationGameKMM. Use rules to validate LLM grammar corrections against known patterns before showing to user. Generate targeted drill suggestions from errors detected in live conversation.
- **Multi-language idioms** (7.9): Make pattern DB language-aware with `idioms/{lang_code}.json` structure. Tag vocab items with source/target language pair.
- **Content import** (7.12): Run idiom scanner on imported text paragraphs, not just live speech.
