# Database

Async SQLite via `aiosqlite` with WAL mode, FTS5 full-text search, and automatic schema migrations. Single file at `data/habla.db`.

## Why SQLite?

- Single-user application — no concurrent write contention
- Zero deployment overhead (no database server to manage)
- WAL mode gives concurrent reads during writes
- FTS5 provides full-text search without an external search engine
- Database file is portable (backup = copy file)
- Async via aiosqlite means no event loop blocking

## Schema

### sessions

| Column | Type | Purpose |
|--------|------|---------|
| `id` | INTEGER PK | Auto-increment |
| `started_at` | DATETIME | Session start (default CURRENT_TIMESTAMP) |
| `ended_at` | DATETIME | Session end (set on close) |
| `mode` | TEXT | "conversation" or "classroom" |
| `direction` | TEXT | "es_to_en" or "en_to_es" |
| `speaker_count` | INTEGER | Number of detected speakers |
| `topic_summary` | TEXT | LLM-maintained topic summary |
| `notes` | TEXT | User-provided session notes |
| `llm_provider` | TEXT | Provider used during session |
| `llm_input_tokens` | INTEGER | Total input tokens (cloud providers) |
| `llm_output_tokens` | INTEGER | Total output tokens |
| `llm_cost_usd` | REAL | Estimated session cost |
| `recording_dir` | TEXT | Path to audio recording directory |
| `study_summary` | TEXT | Post-session LLM-generated study report (JSON string) |

### speakers

| Column | Type | Purpose |
|--------|------|---------|
| `id` | TEXT | Pyannote speaker ID ("SPEAKER_00") |
| `session_id` | INTEGER FK | References sessions |
| `auto_label` | TEXT | Auto-assigned label ("Speaker A") |
| `custom_name` | TEXT | User-given name |
| `role_hint` | TEXT | LLM-suggested role ("instructor") |
| `color` | TEXT | Hex color for UI |
| `utterance_count` | INTEGER | Times this speaker spoke |
| `first_seen_at` | DATETIME | First appearance |

Primary key: `(id, session_id)` — speakers are per-session.

### exchanges

| Column | Type | Purpose |
|--------|------|---------|
| `id` | INTEGER PK | Auto-increment |
| `session_id` | INTEGER FK | References sessions |
| `speaker_id` | TEXT | Speaker attribution |
| `timestamp` | DATETIME | When the exchange occurred |
| `direction` | TEXT | Translation direction for this exchange |
| `raw_transcript` | TEXT | Raw ASR output |
| `corrected_source` | TEXT | LLM-corrected ASR |
| `translation` | TEXT | Final translation |
| `confidence` | REAL | ASR confidence score |
| `is_correction` | BOOLEAN | Teacher correcting student (classroom mode) |
| `correction_json` | TEXT | Correction detail JSON (wrong, right, explanation) |
| `processing_ms` | INTEGER | End-to-end processing time |
| `audio_path` | TEXT | Path to saved WAV clip |
| `bookmarked` | BOOLEAN | User-bookmarked this exchange |
| `bookmark_note` | TEXT | Optional bookmark annotation |
| `detected_language` | TEXT | WhisperX detected language code |

Indexes: `idx_exchanges_session` (session_id), `idx_exchanges_bookmarked` (WHERE bookmarked=1).

### vocab

| Column | Type | Purpose |
|--------|------|---------|
| `id` | INTEGER PK | Auto-increment |
| `exchange_id` | INTEGER FK | Source exchange (optional) |
| `speaker_id` | TEXT | Who said it |
| `term` | TEXT | Phrase to learn |
| `literal` | TEXT | Word-for-word translation (idioms) |
| `meaning` | TEXT | Actual meaning |
| `category` | TEXT | "idiom", "slang", "false_friend", "correction", "grammar_note", "phrase" |
| `source_sentence` | TEXT | Example usage in context |
| `region` | TEXT | "universal", "spain", "mexico", etc. |
| `save_worthy` | BOOLEAN | Worth studying (default true) |
| `created_at` | DATETIME | When saved |
| `ease_factor` | REAL | SM-2 ease factor (default 2.5, range [1.3, 5.0]) |
| `interval_days` | INTEGER | Days until next review (default 1) |
| `next_review` | DATETIME | When to review next |
| `repetitions` | INTEGER | Successful review count |
| `lapse_count` | INTEGER | Times forgotten after learning |
| `times_encountered` | INTEGER | How many times this term appeared |

Indexes: `idx_vocab_next_review`, `idx_vocab_term`.

### vocab_fts (FTS5 virtual table)

Full-text search index over vocab columns: `term`, `meaning`, `literal`, `category`.

Kept in sync via SQLite triggers:
- `ai` (after insert): Insert new row into FTS
- `ad` (after delete): Delete from FTS
- `au` (after update): Delete old + insert new

FTS rebuilt on startup to index any pre-trigger data.

### idiom_patterns

| Column | Type | Purpose |
|--------|------|---------|
| `id` | INTEGER PK | Auto-increment |
| `pattern` | TEXT | Compiled regex pattern |
| `canonical` | TEXT | Normalized form for dedup |
| `literal` | TEXT | Word-for-word translation |
| `meaning` | TEXT | Idiomatic meaning |
| `region` | TEXT | Geographic variant |
| `frequency` | TEXT | "common", "rare", etc. |

### quality_metrics

| Column | Type | Purpose |
|--------|------|---------|
| `id` | INTEGER PK | Auto-increment |
| `session_id` | INTEGER FK | References sessions |
| `segment_id` | INTEGER | Segment number within recording |
| `timestamp` | DATETIME | When recorded |
| `status` | TEXT | "ok", "low_confidence", "asr_rejected", "asr_empty" |
| `confidence` | REAL | WhisperX confidence |
| `audio_rms` | REAL | Audio loudness |
| `duration_seconds` | REAL | Segment duration |
| `speaker_id` | TEXT | Speaker attribution |
| `clipped_onset` | BOOLEAN | VAD clipped speech start |
| `processing_time_ms` | INTEGER | Processing duration |
| `vad_threshold` | REAL | VAD threshold used |
| `model_name` | TEXT | WhisperX model used |

Indexes: `idx_quality_metrics_status`, `idx_quality_metrics_session`.

## Migrations

Schema evolution uses `ALTER TABLE ... ADD COLUMN` wrapped in try/catch (idempotent). No migration framework — each new column is added with a default value and silently ignored if it already exists. This works for SQLite's limited ALTER TABLE support.

## Pydantic Models (`models/schemas.py`, 142L)

| Model | Purpose |
|-------|---------|
| `SpeakerProfile` | Speaker identity (id, label, name, role, color) |
| `FlaggedPhrase` | Vocab item from translation (phrase, meaning, type, source) |
| `CorrectionDetail` | Grammar correction (wrong, right, explanation) |
| `TranslationResult` | LLM output (corrected, translated, confidence, phrases) |
| `Exchange` | Complete utterance record |
| `VocabItem` | Flashcard with SRS metadata |
| WS message types | `WSPartialTranscript`, `WSTranslation`, `WSSpeakersUpdate`, `WSStatus` |

## Design Rationale

**Why SQLite instead of Postgres?** This is a single-user app running on a laptop. SQLite eliminates deployment complexity, runs embedded, and the WAL mode handles the read-during-write pattern (health checks reading while translations write). There's no multi-user write contention to worry about.

**Why no migration framework?** The schema is small (6 tables) and evolves slowly. `ALTER TABLE ADD COLUMN` with try/catch is simpler than maintaining Alembic migration files for a personal project. If the project goes multi-tenant (Phase 10), a proper migration system would be needed.

**Why FTS5 triggers instead of application-level sync?** Triggers guarantee FTS stays in sync even if vocab is modified outside the application (e.g., manual SQLite edits). The trigger approach is also atomic — FTS updates happen in the same transaction as the vocab write.

**Why `status` as an implicit string enum?** SQLite has no enum type. Using a string column with documented valid values ("ok", "low_confidence", "asr_rejected", "asr_empty") is the pragmatic choice. Any new consumer filtering this column must handle all four values.

## Planned Direction

- **Database schema reference doc** (Phase 9.3): Formal documentation of all tables, columns, indexes, relationships, and which code paths write to each column.
- **Multi-language vocab tagging** (7.9): Add source/target language columns to vocab table.
- **Session export** (7.8): Structured export queries joining sessions, exchanges, and vocab.
